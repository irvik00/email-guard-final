"""Day-3 input boundary, sparse estimators, metrics and faithful explanations."""
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from normalization_schema import make_model_text, FILTER_HEADERS, canonical_bytes

ROOT=Path(__file__).resolve().parents[1]
_ACTIVE_GUARDS=[]
_HOOK_INSTALLED=False


def require(condition, reason):
    if not condition: raise ValueError(reason)


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class DataBoundary:
    """Audithook denies all project data except explicit day-3 allowed inputs.

    Hooks cannot be removed; a scoped active stack makes fixture runs isolated.
    No allowlist is derived from caller-supplied closed dataset paths.
    """
    def __init__(self,root,eml=None):
        self.root=Path(root).resolve()
        self.data=self.root/'data'
        self.allowed={self.data/'processed/splits/manifest.json',
                      self.data/'processed/splits/spam_email/train.jsonl',
                      self.data/'processed/splits/spam_email/validation.jsonl'}
        if eml is not None:
            eml=Path(eml).resolve()
            require(not eml.is_relative_to(self.data) or eml.is_relative_to(self.data/'examples'),
                    'Prediction input cannot be a raw/interim/processed dataset')
            require(eml.suffix.lower()=='.eml','Only an explicitly supplied .eml is allowed')
            self.allowed.add(eml)
        self.accesses=Counter()

    def check(self,path,mode='r',flags=0):
        if not isinstance(path,(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(path)).resolve()
        if not path.is_relative_to(self.data):return
        require(path in self.allowed,'Closed data path denied: '+path.relative_to(self.root).as_posix())
        require(not any(c in str(mode) for c in 'wax+') and not (int(flags or 0)&(os.O_WRONLY|os.O_RDWR)),
                'Data inputs are read-only')
        self.accesses[path.relative_to(self.root).as_posix()]+=1

    def __enter__(self):
        global _HOOK_INSTALLED
        if not _HOOK_INSTALLED:
            def hook(event,args):
                if event=='open':
                    for guard in _ACTIVE_GUARDS:guard.check(*args)
            sys.addaudithook(hook);_HOOK_INSTALLED=True
        _ACTIVE_GUARDS.append(self)
        return self

    def __exit__(self,*args):
        require(_ACTIVE_GUARDS[-1] is self,'Unbalanced data boundary')
        _ACTIVE_GUARDS.pop()


@dataclass(frozen=True)
class Example:
    record_id:str
    group_id:str
    source:str
    label:int
    text:str
    parse_status:str
    label_status:str
    fold_id:object=None


def load_inputs(root,config):
    root=Path(root).resolve()
    path=(root/config['manifest']).resolve()
    require(path==root/'data/processed/splits/manifest.json','Only current preparation manifest is allowed')
    manifest_bytes=path.read_bytes()
    manifest=json.loads(manifest_bytes)
    require(manifest['model_feature_allowlist']==['model_text'],'Unexpected feature contract')
    datasets={}; fingerprints={}; ids=set(); groups=set()
    for split in ['train','validation']:
        spec=manifest['datasets']['spam_email']['splits'][split]
        relative=f'splits/spam_email/{split}.jsonl'
        require(spec['path']==relative and spec['status']=='ready','Closed/invalid path in manifest for '+split)
        data_path=root/'data/processed'/relative
        expected=manifest['artifacts'][relative]
        count=size=0; digest=hashlib.sha256(); data=[]
        with data_path.open('rb') as handle:
            for line in handle:
                count+=1;size+=len(line);digest.update(line)
                r=json.loads(line)
                rid,gid=r['record_id'],r['duplicate_group_id']
                require(isinstance(rid,str) and rid and isinstance(gid,str) and gid,'Missing ID/group')
                require(rid not in ids and gid not in groups,'Duplicate ID/group or train/validation leakage')
                require(type(r['spam_label']) is int and r['spam_label'] in {0,1},'Unknown or Boolean spam label')
                require(r['source'] in {'spamassassin','lingspam'} and r['dataset']=='spam_email' and r['split']==split,'Unexpected source/dataset role')
                require(r['quality_status']=='eligible' and r['usable_model_text'] and r['parse_status'] in {'ok','partial'},'Ineligible input '+rid)
                require('needs_review' not in r['quality_flags'] and not r['protected_test'],'Unreviewed/protected input '+rid)
                text=r['model_text']
                require(isinstance(text,str) and text.strip(),'Empty model_text '+rid)
                require(text==make_model_text(r['subject'],r['body_text'])[0],'Builder mismatch '+rid)
                require(hashlib.sha256(text.encode()).hexdigest()==r['model_text_sha256'],'Text hash mismatch '+rid)
                require(not any(line.partition(':')[0].lower() in FILTER_HEADERS for line in text.splitlines()),'Filter headers in features '+rid)
                require(not any(k in r for k in ['headers_raw','html','plain_text','html_text']),'Diagnostics in ML input')
                assignment=manifest['group_split_mapping'][gid]
                require(assignment['role']==split and not assignment['protected_test'] and not assignment.get('reserved_future',False),'Invalid group role '+gid)
                require({'dataset':'spam_email','split':split,'record_id':rid} in assignment['assignments'],'Representative mapping mismatch '+rid)
                ids.add(rid);groups.add(gid)
                data.append(Example(rid,gid,r['source'],r['spam_label'],text,r['parse_status'],r['label_status']['spam'],r.get('fold_id')))
        actual={'sha256':digest.hexdigest(),'size_bytes':size,'rows':count}
        require(actual==expected,'Input differs from artifact manifest: '+split)
        labels=dict(Counter(str(r.label) for r in data));sources=dict(Counter(r.source for r in data))
        require(count==spec['records'] and labels==spec['class_counts'] and sources==spec['source_counts'],'Manifest counts mismatch '+split)
        require(count==config['expected'][split]['rows'] and labels==config['expected'][split]['classes'],'Expected benchmark counts differ: '+split)
        require(set(labels)=={'0','1'},'Both classes required in '+split)
        fingerprints[split]={'path':data_path.relative_to(root).as_posix(),**actual,'classes':labels,'sources':sources}
        datasets[split]=data
    mapping=manifest.get('cv_fold_mapping')
    train=datasets['train']
    if mapping:
        nfolds=manifest['cv_folds']
        require(set(mapping)=={r.group_id for r in train},'CV mapping coverage differs from train')
        folds=np.array([mapping[r.group_id] for r in train])
        require(all(type(mapping[r.group_id]) is int and r.fold_id==mapping[r.group_id] for r in train),'CV fold IDs disagree with records')
        basis='existing preparation manifest cv_fold_mapping; unchanged'
    else:
        nfolds=config['cv']['fallback_folds']
        folds=np.full(len(train),-1,dtype=int)
        for fold,(_,held) in enumerate(StratifiedGroupKFold(nfolds,shuffle=True,random_state=config['seed']).split(
                np.zeros(len(train)),[r.label for r in train],[r.group_id for r in train])):
            folds[held]=fold
        mapping={r.group_id:int(f) for r,f in zip(train,folds)}
        basis='mapping absent: StratifiedGroupKFold shuffle=True seed=42 on train only'
    require(nfolds==5 and set(folds)==set(range(nfolds)),'Expected five nonempty CV folds')
    fold_counts={}
    for fold in range(nfolds):
        held=[r for r,f in zip(train,folds) if f==fold];learn=[r for r,f in zip(train,folds) if f!=fold]
        require({r.label for r in held}=={0,1} and {r.label for r in learn}=={0,1},'CV fold lacks a class')
        require(not {r.group_id for r in held}&{r.group_id for r in learn},'CV group leakage')
        fold_counts[str(fold)]=dict(Counter(str(r.label) for r in held))
    if manifest.get('cv_fold_mapping'):
        require(fold_counts==manifest['cv_fold_class_counts'],'CV class counters disagree with manifest')
    contract={'acceptance':'accepted_for_historical_baseline','manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
        'inputs':fingerprints,'cv_basis':basis,'cv_fold_mapping':mapping,'cv_fold_class_counts':fold_counts,
        'closed_artifacts_from_manifest_only':{k:v for k,v in manifest['artifacts'].items() if k not in {f'splits/spam_email/{s}.jsonl' for s in ['train','validation']}},
        'test_not_evaluated':True,'feature_allowlist':['model_text']}
    return datasets,folds,contract


def build_pipeline(model_id,config):
    require(model_id in {'word_nb','word_lr','char_lr'},'Unknown model ID')
    settings={**config['tfidf'],**config['char' if model_id=='char_lr' else 'word']}
    settings['ngram_range']=tuple(settings['ngram_range'])
    vectorizer=TfidfVectorizer(**settings,dtype=np.float64)
    classifier=MultinomialNB(**config['nb']) if model_id=='word_nb' else LogisticRegression(**config['lr'])
    return Pipeline([('tfidf',vectorizer),('classifier',classifier)])


def scores(pipeline,texts):
    classes=list(pipeline.classes_)
    require(set(classes)=={0,1},'Model must contain both binary classes')
    result=np.asarray(pipeline.predict_proba(texts)[:,classes.index(1)],dtype=float)
    require(np.all(np.isfinite(result)) and np.all((result>=0)&(result<=1)),'Invalid spam scores')
    return result


def metrics(y,score=None,threshold=.5,predictions=None):
    y=np.asarray(y,dtype=int)
    pred=np.asarray(predictions,dtype=int) if predictions is not None else (np.asarray(score)>=threshold).astype(int)
    require(y.shape==pred.shape and set(y)<= {0,1} and set(pred)<= {0,1},'Invalid metric inputs')
    tp=int(np.sum((y==1)&(pred==1)));fp=int(np.sum((y==0)&(pred==1)))
    tn=int(np.sum((y==0)&(pred==0)));fn=int(np.sum((y==1)&(pred==0)))
    div=lambda a,b:a/b if b else None
    return {'precision':div(tp,tp+fp),'recall':div(tp,tp+fn),'f1':div(2*tp,2*tp+fp+fn),
        'accuracy':div(tp+tn,len(y)),'fpr':div(fp,fp+tn),'TP':tp,'FP':fp,'TN':tn,'FN':fn,
        'support':{'0':tn+fp,'1':tp+fn},'n':len(y),
        'zero_division':'undefined -> JSON null (equivalent to sklearn zero_division=np.nan)'}


def wilson(successes,total):
    if not total:return {'numerator':successes,'denominator':total,'level':.95,'interval':None}
    z=NormalDist().inv_cdf(.975);p=successes/total;den=1+z*z/total
    centre=(p+z*z/(2*total))/den
    width=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return {'numerator':successes,'denominator':total,'level':.95,'interval':[max(0,centre-width),min(1,centre+width)]}


def threshold_modes(y,score,config):
    score=np.asarray(score,dtype=float)
    grid=np.unique(np.concatenate([score,[config['threshold']['default'],np.nextafter(np.max(score),math.inf)]]))
    rows=[]
    for t in grid:
        m=metrics(y,score,float(t)); rows.append({'threshold':float(t),'metrics':m})
    def key(row):
        m=row['metrics']
        return (-1 if m['f1'] is None else m['f1'],-m['FP'],-1 if m['recall'] is None else m['recall'],row['threshold'])
    best=max(rows,key=key)
    feasible=[r for r in rows if r['metrics']['fpr'] is not None and r['metrics']['fpr']<=config['threshold']['conservative_fpr_max']]
    conservative=max(feasible,key=key) if feasible else None
    default={'threshold':float(config['threshold']['default']),'metrics':metrics(y,score,config['threshold']['default'])}
    modes={'default':default,'f1':best,'conservative':conservative}
    for row in modes.values():
        if row is not None:
            m=row['metrics'];row['recall_wilson95']=wilson(m['TP'],m['TP']+m['FN']);row['fpr_wilson95']=wilson(m['FP'],m['FP']+m['TN'])
    usable=conservative is not None and (conservative['metrics']['recall'] or 0)>0
    return {'modes':modes,'threshold_default':default['threshold'],'threshold_f1':best['threshold'],
        'threshold_conservative':conservative['threshold'] if conservative else None,
        'provisional_policy':'conservative_validation_fpr_1pct' if usable else 'default_0.5_unmet_fp_scenario',
        'provisional_threshold':conservative['threshold'] if usable else default['threshold'],
        'unmet_fp_scenario':not usable,'grid_size':len(grid),'grid':rows,
        'rule':'score >= threshold; ties: maximum F1, fewer FP, greater recall, larger threshold',
        'validation_tuned_optimistic':True,'scenario_fpr_max':config['threshold']['conservative_fpr_max'],
        'not_business_sla':True,'test_not_evaluated':True}


def select_baseline(results,config):
    tolerance=config['cv']['tie_tolerance']
    best=max(r['cv']['mean']['f1'] for r in results.values())
    contenders=[k for k,r in results.items() if abs(r['cv']['mean']['f1']-best)<=tolerance]
    precision=lambda k:results[k]['cv']['mean']['precision'] if results[k]['cv']['mean']['precision'] is not None else -1
    best_precision=max(precision(k) for k in contenders)
    contenders=[k for k in contenders if abs(precision(k)-best_precision)<=tolerance]
    return min(contenders,key=lambda k:config['cv']['tie_preference'].index(k))


def coefficients(pipeline):
    model=pipeline['classifier'];classes=list(model.classes_)
    require(set(classes)=={0,1},'Explanation requires binary classes')
    if isinstance(model,LogisticRegression):
        direction=1 if classes[1]==1 else -1
        return model.coef_[0]*direction,float(model.intercept_[0])*direction,'logit(spam)'
    require(isinstance(model,MultinomialNB),'Unsupported explanation model')
    spam,ham=classes.index(1),classes.index(0)
    return (model.feature_log_prob_[spam]-model.feature_log_prob_[ham],
            float(model.class_log_prior_[spam]-model.class_log_prior_[ham]),'NB log score ratio spam/ham')


def explain(pipeline,text,top_n=10):
    vector=pipeline['tfidf'].transform([text]).tocsr()
    coef,intercept,kind=coefficients(pipeline)
    contributions=vector.data*coef[vector.indices]
    total=float(np.sum(contributions)+intercept)
    model=pipeline['classifier'];classes=list(model.classes_)
    if isinstance(model,LogisticRegression):
        expected=float(model.decision_function(vector)[0])*(1 if classes[1]==1 else -1)
    else:
        lp=model.predict_log_proba(vector)[0];expected=float(lp[classes.index(1)]-lp[classes.index(0)])
    require(math.isclose(total,expected,rel_tol=1e-9,abs_tol=1e-9),'Local explanation sum does not match model')
    names=pipeline['tfidf'].get_feature_names_out()
    order=sorted(range(len(contributions)),key=lambda i:(-abs(contributions[i]),int(vector.indices[i])))[:top_n]
    active=[{'feature_index':int(vector.indices[i]),'feature':str(names[vector.indices[i]]),
             'tfidf':float(vector.data[i]),'weight':float(coef[vector.indices[i]]),'contribution':float(contributions[i])} for i in order]
    return {'kind':kind,'intercept':intercept,'active_features':int(vector.nnz),'all_contributions_sum':float(np.sum(contributions)),
        'sum_plus_intercept':total,'expected_log_score':expected,'absolute_error':abs(total-expected),
        'top_features':active,'top_contributions_sum':sum(v['contribution'] for v in active),
        'top_n_is_partial':True,'not_probabilities_or_causality':True}


def json_params(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {str(k):json_params(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_params(v) for v in value]
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,type):return value.__module__+'.'+value.__qualname__
    if hasattr(value,'get_params'):return {'class':type(value).__module__+'.'+type(value).__name__,'params':json_params(value.get_params(deep=False))}
    raise TypeError('Unserializable parameter '+type(value).__name__)


def versions():
    return {k:importlib.metadata.version(k) for k in ['scikit-learn','numpy','scipy','joblib','threadpoolctl']}


def code_fingerprints():
    names=['baseline_common.py','train_baselines.py','baseline_reporting.py','predict_email.py','read_email.py','normalization_schema.py']
    return {'src/'+name:sha_file(ROOT/'src'/name) for name in names}
