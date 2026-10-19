"""Day-4 explicit data boundary, fixed cleaner, validation and inference policy."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from baseline_common import ROOT, DataBoundary, require, sha_file, digest_json, load_inputs, scores, metrics
from normalization_schema import make_model_text
from pipeline_common import rows, compare_text, write_json

PROVENANCE=['data/raw/russian_auxiliary/README.md','data/raw/russian_auxiliary/join_datasets.py']


class Day4Boundary(DataBoundary):
    def __init__(self,root=ROOT,eml=None):
        super().__init__(root,eml)
        self.allowed.update(self.root/p for p in PROVENANCE)
        self.allowed.update(self.data/f'processed/splits/russian_auxiliary/{s}.jsonl' for s in ['train','validation'])


def clean_text(record,config):
    """No label/source/ID dependence; no guessing subject from body."""
    subject=record.get('subject');changes=[]
    observable=record.get('feature_availability',{}).get('subject')=='observable'
    if observable and isinstance(subject,str):
        subject,n=re.subn(config['clean_builder']['pattern'],'',subject,count=1,flags=re.I)
        if n:changes.append('remove_leading_numeric_filter_marker')
    text,base_changes=make_model_text(subject,record.get('body_text'))
    return text,changes+base_changes


def ru_mapping(record):
    original=record.get('original_label');label=record.get('spam_label')
    require(type(original) is bool,'RU original label must be native bool; strings/numbers cannot be coerced')
    require(type(label) is int and label in {0,1},'RU mapped label must be integer 0/1')
    require(label==(1 if original is True else 0),'Contradictory RU mapping')
    require(record['label_status']['spam']=='source_intended_unverified','Unexpected RU label status')


def load_day4(root,config,historical_config):
    root=Path(root)
    historical,folds,contract=load_inputs(root,historical_config)
    manifest=json.loads((root/'data/processed/splits/manifest.json').read_bytes())
    sets={};fingerprints={};ids=set();group_roles={};mapping_errors=[]
    kept=['record_id','duplicate_group_id','source','spam_label','original_label','subject','body_text','model_text',
          'label_status','feature_availability','sender','sender_address','sender_domain','links','attachments',
          'unsubscribe','formatting','original_locator','parse_status','quality_flags','fold_id','input_kind']
    for domain in ['spam_email','russian_auxiliary']:
        sets[domain]={}
        for split in ['train','validation']:
            rel=f'splits/{domain}/{split}.jsonl';spec=manifest['datasets'][domain]['splits'][split]
            require(spec['path']==rel and spec['status']=='ready','Unexpected split path or status')
            digest=hashlib.sha256();size=count=0;data=[]
            with (root/'data/processed'/rel).open('rb') as handle:
                for line in handle:
                    digest.update(line);size+=len(line);count+=1;r=json.loads(line)
                    require(r['record_id'] not in ids,'Repeated ID across allowed splits');ids.add(r['record_id'])
                    require(type(r['spam_label']) is int and r['spam_label'] in {0,1},'Invalid spam label')
                    require(r['quality_status']=='eligible' and r['usable_model_text'] and r['parse_status'] in {'ok','partial'},'Ineligible record')
                    require(not r['protected_test'] and 'needs_review' not in r['quality_flags'],'Protected/unreviewed record')
                    require(r['split']==split and r['dataset']==domain,'Record role mismatch')
                    require(r['model_text']==make_model_text(r['subject'],r['body_text'])[0] and r['model_text'].strip(),'Builder/empty text mismatch')
                    require(hashlib.sha256(r['model_text'].encode()).hexdigest()==r['model_text_sha256'],'Record text SHA mismatch')
                    gid=r['duplicate_group_id'];assignment=manifest['group_split_mapping'][gid]
                    require(assignment['role']==split and not assignment['protected_test'] and not assignment.get('reserved_future',False),'Closed group')
                    require({'dataset':domain,'split':split,'record_id':r['record_id']} in assignment['assignments'],'Group assignment mismatch')
                    require(gid not in group_roles or group_roles[gid]==split,'Cross-domain train/validation group leakage')
                    group_roles[gid]=split
                    if domain=='russian_auxiliary':
                        try:ru_mapping(r)
                        except ValueError as error:mapping_errors.append({'record_id':r['record_id'],'reason':str(error)})
                    data.append({k:r.get(k) for k in kept})
            actual={'sha256':digest.hexdigest(),'rows':count,'size_bytes':size}
            require(actual==manifest['artifacts'][rel],'Split SHA/count mismatch '+rel)
            classes=dict(Counter(str(r['spam_label']) for r in data))
            require(count==spec['records'] and classes==spec['class_counts'],'Manifest label count mismatch')
            if domain=='russian_auxiliary':require(count==config['expected_ru'][split]['rows'] and classes==config['expected_ru'][split]['classes'],'RU expected count mismatch')
            sets[domain][split]=data;fingerprints[domain+'/'+split]={'path':'data/processed/'+rel,**actual,'classes':classes}
    # Plain and canonical equality audited across BOTH domains, including short records.
    leaks={};all_train=[r for ds in sets.values() for r in ds['train']];all_val=[r for ds in sets.values() for r in ds['validation']]
    for name,normalizer in [('exact',lambda x:x),('canonical',compare_text)]:
        train_hashes={digest_json(normalizer(r['model_text'])):r for r in all_train};found=[]
        for r in all_val:
            h=digest_json(normalizer(r['model_text']))
            if h in train_hashes:found.append({'train_id':train_hashes[h]['record_id'],'validation_id':r['record_id'],'text_hash':h})
        leaks[name]=found
    contract['day4_inputs']=fingerprints;contract['allowed_text_leakage']=leaks
    contract['ru_status']='blocked_mapping' if mapping_errors else 'source_labels_used_provisionally'
    contract['ru_mapping_errors']=mapping_errors
    contract['provenance']={p:{'sha256':sha_file(root/p),'text':(root/p).read_text(encoding='utf-8')} for p in PROVENANCE}
    return sets,folds,contract


def exclude_collisions(train,held,text_lookup):
    held_hashes={compare_text(text_lookup[r['record_id']]) for r in held}
    removed=[r['record_id'] for r in train if compare_text(text_lookup[r['record_id']]) in held_hashes]
    excluded=set(removed)
    return [r for r in train if r['record_id'] not in excluded],removed


def evaluate(model,data,texts,thresholds):
    """Raw full-domain metrics plus explicit wrapper coverage and analyzer coverage."""
    score=scores(model,texts);vector=model['tfidf'].transform(texts)
    nnz=np.diff(vector.indptr);analyzer=model['tfidf'].build_analyzer();vocab=model['tfidf'].vocabulary_
    coverage=[]
    for text in texts:
        tokens=analyzer(text);coverage.append(sum(t in vocab for t in tokens)/len(tokens) if tokens else 0.)
    y=[r['spam_label'] for r in data]
    result={'n':len(data),'raw_metrics':{str(t):metrics(y,score,t) for t in thresholds},
        'empty_vectors':int(np.sum(nnz==0)),'wrapper_coverage':float(np.mean(nnz>0)),
        'vocabulary_coverage_occurrence_quantiles':dict(zip(['min','p10','p50','p90','max'],np.quantile(coverage,[0,.1,.5,.9,1]).tolist())),
        'coverage_semantics':'analyzer ngram occurrences present in fitted vocabulary / all occurrences; not token/language comprehension',
        'by_source':{}}
    for source in sorted({r['source'] for r in data}):
        mask=np.array([r['source']==source for r in data]);result['by_source'][source]={str(t):metrics(np.array(y)[mask],score[mask],t) for t in thresholds}
    predictions=[{'record_id':r['record_id'],'source':r['source'],'duplicate_group_id':r['duplicate_group_id'],
        'spam_label':r['spam_label'],'spam_score':float(s),'active_features':int(n),'vocabulary_coverage':float(c),
        'wrapper_verdict_at_0.5':('insufficient_features' if n==0 else 'spam' if s>=.5 else 'normal')}
        for r,s,n,c in zip(data,score,nnz,coverage)]
    return result,predictions


def inference(model,text,threshold):
    vector=model['tfidf'].transform([text]);s=float(model['classifier'].predict_proba(vector)[0,list(model.classes_).index(1)])
    return {'verdict':'insufficient_features' if vector.nnz==0 else 'spam' if s>=threshold else 'normal',
        'raw_spam_score':s,'raw_score_reliable':False,'threshold':threshold,'active_features':int(vector.nnz),
        'language':'unknown','language_support':'not established; alphabet is not language identification',
        'phishing':'not_available','category':'not_available','fit_performed':False}


def subset_records(sets,config):
    """Largest-remainder historical quotas; SHA seed ordering independent of scores."""
    hist=sets['spam_email']['validation'];groups={}
    for r in hist:groups.setdefault((r['source'],r['spam_label']),[]).append(r)
    n=config['pilot']['historical_size'];expected={k:len(v)*n/len(hist) for k,v in groups.items()}
    quotas={k:int(v) for k,v in expected.items()}
    for k in sorted(groups,key=lambda k:(-(expected[k]-quotas[k]),k))[:n-sum(quotas.values())]:quotas[k]+=1
    order=lambda r:digest_json([config['seed'],'pilot',r['record_id']])
    result=[r for k in sorted(groups) for r in sorted(groups[k],key=order)[:quotas[k]]]
    for label in [0,1]:
        data=[r for r in sets['russian_auxiliary']['validation'] if r['spam_label']==label]
        result.extend(sorted(data,key=order)[:config['pilot']['ru_per_class']])
    return result,{'historical':{f'{k[0]}:{k[1]}':v for k,v in quotas.items()},'russian_auxiliary':{'0':config['pilot']['ru_per_class'],'1':config['pilot']['ru_per_class']}}
