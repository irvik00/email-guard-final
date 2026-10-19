"""Synthetic-only tests: data boundary, CV isolation, metrics, explanations, reload."""
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from baseline_common import (DataBoundary,load_inputs,build_pipeline,metrics,threshold_modes,explain,
    scores,select_baseline,digest_json,wilson,sha_file)
from normalization_schema import build_record,make_model_text,canonical_bytes
from normalization_adapters import InputRow
from read_email import parse_email
from train_baselines import execute
from predict_email import predict_path,predict_raw


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='baseline_fixture_',dir=PROJECT/'tests')
        self.root=Path(self.temp.name).resolve()
        self.config=json.loads((PROJECT/'config/day3_baselines.json').read_text(encoding='utf-8'))
        self.config['word']['max_features']=2000;self.config['char']['max_features']=3000
        self.config['smoke_eml']='data/examples/demo.eml'

    def tearDown(self):
        self.assertTrue(self.root.is_relative_to(PROJECT/'tests'));self.temp.cleanup()

    def fixture(self):
        manifest={'model_feature_allowlist':['model_text'],'artifacts':{},'datasets':{'spam_email':{'splits':{}}},
            'group_split_mapping':{},'cv_folds':5,'cv_fold_mapping':{},'cv_fold_class_counts':{}}
        fold_classes={str(i):Counter() for i in range(5)}
        for split,nclass in [('train',20),('validation',10)]:
            records=[]
            for label in [0,1]:
                for i in range(nclass):
                    text=('normal team meeting project schedule research linguistics update' if not label else
                          'win cash special offer prize free money claim immediate')+' '+('trainingtoken' if split=='train' else 'validationonlyword')+f' index{i}'
                    source='spamassassin' if i%2==0 else 'lingspam'
                    subset='spam' if label else 'easy_ham'
                    p=self.root/'data/raw'/f'{source}-{split}-{label}'
                    info={'path':p.relative_to(self.root).as_posix(),'sha256':hashlib.sha256(str(p).encode()).hexdigest()}
                    raw=(f'Subject: Topic\nFrom: sender@example.test\n\n{text}\n').encode()
                    row=InputRow(source,p,f'{info["path"]}::row={i}',i+1,subset if source=='spamassassin' else None,
                        'raw_email' if source=='spamassassin' else 'text_only_preprocessed',original_label=subset if source=='spamassassin' else str(label),
                        raw=raw if source=='spamassassin' else None,fields=None if source=='spamassassin' else {'subject':'Topic','message':text,'label':str(label)})
                    r=build_record(row,info,parse_email(raw) if source=='spamassassin' else None)
                    gid='dup:'+digest_json([r['record_id']]);fold=i%5 if split=='train' else None
                    r.update(duplicate_group_id=gid,quality_status='eligible',quality_flags=[],dataset='spam_email',split=split,protected_test=False,fold_id=fold)
                    records.append(r)
                    manifest['group_split_mapping'][gid]={'role':split,'protected_test':False,'reserved_future':False,
                        'assignments':[{'dataset':'spam_email','split':split,'record_id':r['record_id']}]}
                    if split=='train':
                        manifest['cv_fold_mapping'][gid]=fold;fold_classes[str(fold)].update([str(label)])
            relative=f'splits/spam_email/{split}.jsonl';path=self.root/'data/processed'/relative;path.parent.mkdir(parents=True,exist_ok=True)
            payload=b''.join(canonical_bytes(r)+b'\n' for r in records);path.write_bytes(payload)
            counts=dict(Counter(str(r['spam_label']) for r in records));sources=dict(Counter(r['source'] for r in records))
            manifest['artifacts'][relative]={'sha256':hashlib.sha256(payload).hexdigest(),'rows':len(records),'size_bytes':len(payload)}
            manifest['datasets']['spam_email']['splits'][split]={'path':relative,'status':'ready','records':len(records),'class_counts':counts,'source_counts':sources}
            self.config['expected'][split]={'rows':len(records),'classes':counts}
        manifest['cv_fold_class_counts']={k:dict(v) for k,v in fold_classes.items()}
        closed=self.root/'data/processed/splits/spam_email/test.jsonl';closed.write_bytes(b'closed synthetic sentinel\n')
        manifest['artifacts']['splits/spam_email/test.jsonl']={'sha256':hashlib.sha256(b'closed synthetic sentinel\n').hexdigest(),'rows':1,'size_bytes':26}
        path=self.root/'data/processed/splits/manifest.json';path.write_text(json.dumps(manifest),encoding='utf-8')
        eml=self.root/self.config['smoke_eml'];eml.parent.mkdir(parents=True)
        eml.write_bytes(b'Subject: Meeting\nX-Spam-Status: Yes\nContent-Type: text/plain\n\nnormal team meeting\nX-Spam-Flag: YES\n  continuation\nproject update\n')
        return manifest

    def test_validation_only_word_never_enters_training_vocabulary(self):
        self.fixture()
        with DataBoundary(self.root):data,folds,contract=load_inputs(self.root,self.config)
        model=build_pipeline('word_lr',self.config).fit([r.text for r in data['train']],[r.label for r in data['train']])
        self.assertNotIn('validationonlyword',model['tfidf'].vocabulary_)
        model.predict_proba([r.text for r in data['validation']])
        self.assertNotIn('validationonlyword',model['tfidf'].vocabulary_)
        self.assertEqual(len(folds),40)
        self.assertTrue(contract['test_not_evaluated'])

    def test_closed_paths_denied_before_open_even_when_missing(self):
        self.fixture()
        with DataBoundary(self.root) as guard:
            for relative in ['processed/splits/spam_email/test.jsonl','raw/untroubled/2026-09.7z',
                'processed/splits/russian_auxiliary/train.jsonl','processed/reserves/modern_spam_future_reserve.jsonl','interim/diagnostics.jsonl']:
                with self.assertRaisesRegex(ValueError,'Closed data path denied'):(self.root/'data'/relative).read_bytes()
            with self.assertRaisesRegex(ValueError,'read-only'):(self.root/'data/processed/splits/spam_email/train.jsonl').open('w')
        self.assertFalse(guard.accesses)

    def test_manifest_cannot_redirect_loader_to_test_and_fold_coverage_required(self):
        manifest=self.fixture();path=self.root/'data/processed/splits/manifest.json'
        manifest['datasets']['spam_email']['splits']['train']['path']='splits/spam_email/test.jsonl';path.write_text(json.dumps(manifest),encoding='utf-8')
        with DataBoundary(self.root),self.assertRaisesRegex(ValueError,'Closed/invalid path'):load_inputs(self.root,self.config)
        manifest['datasets']['spam_email']['splits']['train']['path']='splits/spam_email/train.jsonl'
        manifest['cv_fold_mapping'].pop(next(iter(manifest['cv_fold_mapping'])));path.write_text(json.dumps(manifest),encoding='utf-8')
        with DataBoundary(self.root),self.assertRaisesRegex(ValueError,'coverage'):load_inputs(self.root,self.config)

    def test_folds_and_cross_split_groups_are_validated(self):
        manifest=self.fixture()
        with DataBoundary(self.root):data,folds,_=load_inputs(self.root,self.config)
        for f in range(5):
            held={r.group_id for r,x in zip(data['train'],folds) if x==f};learn={r.group_id for r,x in zip(data['train'],folds) if x!=f}
            self.assertFalse(held&learn)
        p=self.root/'data/processed/splits/spam_email/validation.jsonl';rs=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
        rs[0]['duplicate_group_id']=data['train'][0].group_id;p.write_bytes(b''.join(canonical_bytes(r)+b'\n' for r in rs))
        with DataBoundary(self.root),self.assertRaisesRegex(ValueError,'leakage'):load_inputs(self.root,self.config)

    def test_ineligible_or_unknown_input_stops_instead_of_silently_dropping(self):
        self.fixture()
        p=self.root/'data/processed/splits/spam_email/train.jsonl';payload=p.read_bytes()
        rows=[json.loads(line) for line in payload.splitlines()]
        for field,value,reason in [('spam_label',None,'Unknown'),('spam_label',False,'Boolean'),
                                   ('quality_status','quarantine','Ineligible'),('model_text','','Empty')]:
            bad=copy.deepcopy(rows);bad[0][field]=value;p.write_bytes(b''.join(canonical_bytes(r)+b'\n' for r in bad))
            with DataBoundary(self.root),self.assertRaisesRegex(ValueError,reason):load_inputs(self.root,self.config)
        p.write_bytes(payload)

    def test_confusion_undefined_and_wilson_denominators(self):
        m=metrics([0,0,0,1,1],[.1,.8,.2,.9,.3],.5)
        self.assertEqual([m[k] for k in ['TP','FP','TN','FN']],[1,1,2,1]);self.assertAlmostEqual(m['fpr'],1/3)
        self.assertIsNone(metrics([0,0,1],[0,0,0])['precision'])
        self.assertEqual(metrics([0,0,1],[0,0,0])['f1'],0)
        self.assertIsNone(metrics([1,1],[1,1])['fpr'])
        self.assertIsNone(wilson(0,0)['interval'])
        self.assertEqual(wilson(1,2)['denominator'],2)
        p,r,f,_=precision_recall_fscore_support([0,0,1],[0,0,0],average='binary',zero_division=np.nan)
        self.assertTrue(np.isnan(p));self.assertEqual(f,0)

    def test_threshold_grid_direction_conservative_and_no_recall_fallback(self):
        result=threshold_modes([0,0,1,1],[.6,.1,.9,.7],self.config)
        self.assertEqual(result['modes']['conservative']['metrics']['FP'],0)
        self.assertGreater(result['modes']['conservative']['metrics']['recall'],0)
        self.assertIn(.5,[r['threshold'] for r in result['grid']])
        self.assertGreater(max(r['threshold'] for r in result['grid']),.9)
        self.assertEqual(metrics([1],[.5],.5)['TP'],1)
        fallback=threshold_modes([0,0,1],[.9,.8,.1],self.config)
        self.assertTrue(fallback['unmet_fp_scenario']);self.assertEqual(fallback['provisional_threshold'],.5)

    def test_positive_score_direction_and_cv_selection_ties(self):
        class Reversed:
            classes_=np.array([1,0])
            def predict_proba(self,x):return np.array([[.8,.2] for _ in x])
        self.assertEqual(scores(Reversed(),['x'])[0],.8)
        rs={k:{'cv':{'mean':{'f1':.8,'precision':.9}}} for k in self.config['models']}
        self.assertEqual(select_baseline(rs,self.config),'word_lr')
        rs['word_nb']['cv']['mean']['precision']=.95
        self.assertEqual(select_baseline(rs,self.config),'word_nb')

    def test_local_sparse_sum_lr_nb_and_char(self):
        texts=['meeting project regular work']*4+['prize cash free offer']*4;labels=[0]*4+[1]*4
        for name in self.config['models']:
            p=build_pipeline(name,self.config).fit(texts,labels)
            e=explain(p,'meeting free offer',top_n=1)
            self.assertLess(e['absolute_error'],1e-9)
            self.assertTrue(e['top_n_is_partial'])
            self.assertGreater(e['active_features'],len(e['top_features']))
            saved=self.root/(name+'.joblib');joblib.dump(p,saved)
            np.testing.assert_allclose(scores(p,texts),scores(joblib.load(saved),texts),atol=1e-12,rtol=0)

    def test_end_to_end_18_fits_reload_predictions_and_no_fit_predict(self):
        self.fixture()
        with redirect_stdout(io.StringIO()):experiments,thresholds=execute(self.root,self.config)
        self.assertEqual(experiments['fit_contexts_completed'],18)
        self.assertEqual(set(experiments['data_access_log']),{'data/processed/splits/manifest.json','data/processed/splits/spam_email/train.jsonl',
            'data/processed/splits/spam_email/validation.jsonl','data/examples/demo.eml'})
        model_path=self.root/'models/day3/baseline_bundle.joblib'
        before=sha_file(model_path);eml=self.root/self.config['smoke_eml'];original=eml.read_bytes()
        bundle=joblib.load(model_path);p=bundle['pipeline'];expected=make_model_text(**{'subject':parse_email(original)['subject'],'body':parse_email(original)['body_text']})[0]
        observed=[];transform=p['tfidf'].transform
        with patch.object(Pipeline,'fit',side_effect=AssertionError('prediction must not fit')),patch.object(TfidfVectorizer,'fit',side_effect=AssertionError('prediction must not fit')):
            result=predict_path(model_path,eml,self.root)
            with patch.object(p['tfidf'],'transform',side_effect=lambda texts:(observed.extend(texts),transform(texts))[1]):predict_raw(bundle,original)
        self.assertEqual(observed,[expected]);self.assertFalse(result['fit_performed']);self.assertEqual(result['phishing'],'not_available')
        self.assertEqual(sha_file(model_path),before);self.assertEqual(eml.read_bytes(),original)
        self.assertEqual(result['model_text_char_count'],len(expected))
        with self.assertRaises(ValueError):predict_path(model_path,self.root/'data/processed/splits/spam_email/test.jsonl',self.root)
        for name in self.config['models']:
            rows=[json.loads(line) for line in (self.root/f'reports/day3/predictions/{name}_oof.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows),40);self.assertEqual(len({r['record_id'] for r in rows}),40)


if __name__=='__main__':unittest.main()
