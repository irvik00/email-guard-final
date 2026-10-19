"""Predeclared SpamAssassin-only text/metadata fits; train-only preprocessing."""
import argparse
import importlib.metadata
import time
from collections import Counter
from extra_common import *

FEATURES=['sender_domain_length','sender_local_digit_ratio','link_count','link_label_mismatch_count',
          'attachment_count','unsubscribe_header_present','unsubscribe_body_keyword','html_present']

def numeric(row):
    import numpy as np
    avail=row.get('feature_availability',{});headers=avail.get('headers')=='observable'
    address=row.get('sender_address');domain=row.get('sender_domain');local=address.rsplit('@',1)[0] if address and '@' in address else None
    links=row.get('links') if avail.get('links')=='observable' else None
    attachments=row.get('attachments') if avail.get('attachments')=='observable' else None
    unsub=row.get('unsubscribe') if avail.get('unsubscribe')=='observable' else None
    formatting=row.get('formatting') if avail.get('formatting')=='observable' else None
    # Unknown label-host comparisons remain missing. An empty observed link list is a genuine count zero.
    mismatches=None if links is None else 0 if not links else sum(l['label_mismatch'] for l in links if type(l.get('label_mismatch')) is bool) if any(type(l.get('label_mismatch')) is bool for l in links) else None
    values=[len(domain) if headers and domain else None,
        sum(c.isdigit() for c in local)/len(local) if headers and local else None,
        len(links) if links is not None else None,mismatches,len(attachments) if attachments is not None else None,
        unsub.get('header_present') if unsub is not None else None,unsub.get('body_hint_found') if unsub is not None else None,
        formatting.get('has_html') if formatting is not None else None]
    return [float(v) if v is not None else np.nan for v in values]

def select_text(rows):return [r['text'] for r in rows]
def select_numeric(rows):
    import numpy as np
    return np.array([r['numeric'] for r in rows],dtype=np.float64)
def missing_matrix(matrix):
    import numpy as np
    return np.isnan(matrix).astype(np.float64)

def load_cohort(split):
    require(split in ['train','validation'],'Closed cohort')
    source=read_json(ROOT/'data/processed/splits/manifest.json');day4=read_json(ROOT/'config/day4_experiments.json')
    clean,fp=load_split('spam_email',split,source,day4);clean={r['record_id']:r for r in clean}
    rows=[]
    with (ROOT/f'data/processed/splits/spam_email/{split}.jsonl').open('rb') as f:
        for line in f:
            original=__import__('json').loads(line)
            if original['source']!='spamassassin':continue
            row=clean[original['record_id']];row['numeric']=numeric(original);rows.append(row)
    require({r['spam_label'] for r in rows}=={0,1},'Cohort lacks one class')
    return rows,fp

def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',action='store_true');args=p.parse_args()
    out,models=paths();dest=out/'D';dest.mkdir(exist_ok=True);models=models/'D';models.mkdir(exist_ok=True)
    boundary=ExtraBoundary('final' if args.validation else 'prepare').install()
    import numpy as np
    import joblib
    from sklearn.pipeline import Pipeline,FeatureUnion
    from sklearn.preprocessing import FunctionTransformer,StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    # Stable qualified function names are essential when loading a saved pipeline in another CLI.
    from extra_metadata import select_text as stable_text,select_numeric as stable_numeric,missing_matrix as stable_missing
    import psutil
    if args.validation:
        manifest=read_json(dest/'candidate_manifest.json')
        require(not (dest/'validation_started.json').exists(),'D validation already evaluated')
        write_json(dest/'validation_started.json',{'frozen_manifest_sha256':sha_file(dest/'candidate_manifest.json'),'started_utc':now()})
        rows,fp=load_cohort('validation');train=read_rows(dest/'private_train_membership.jsonl')
        require(not ({r['duplicate_group_id'] for r in train}&{r['duplicate_group_id'] for r in rows}),'D group leakage')
        results={}
        for name,spec in manifest['models'].items():
            require(sha_file(ROOT/spec['path'])==spec['sha256'],'D weights changed')
            model=joblib.load(ROOT/spec['path']);start=time.perf_counter();scores=model.predict_proba(rows)[:,list(model.classes_).index(1)]
            elapsed=time.perf_counter()-start
            predictions=[{'record_id':r['record_id'],'spam_label':r['spam_label'],'probability':float(v)} for r,v in zip(rows,scores)]
            write_rows(dest/f'{name}_validation_predictions.jsonl',predictions)
            times=[]
            for r in sorted(rows,key=lambda r:digest([42,'D-latency',r['record_id']]))[:64]:
                t=time.perf_counter();model.predict_proba([r]);times.append(time.perf_counter()-t)
            results[name]={'metrics':metrics([r['spam_label'] for r in rows],scores,.5),'coverage':1.,
                'warm_single_record_seconds':times,'wall_seconds':elapsed,'rss_bytes':psutil.Process().memory_info().rss}
        paired={r['record_id']:[r['spam_label']] for r in rows}
        for name in ['text_only','hybrid']:
            for row in read_rows(dest/f'{name}_validation_predictions.jsonl'):paired[row['record_id']].append(int(row['probability']>=.5))
        results['paired_changes']=[{'record_id':rid,'label':y,'text_only':x,'hybrid':h} for rid,(y,x,h) in paired.items() if x!=h]
        results['validation_fingerprint']=fp;results['class_counts']=dict(Counter(r['spam_label'] for r in rows))
        results['accesses']=boundary.report();write_json(dest/'comparison.json',results);print({k:v['metrics'] for k,v in results.items() if k in ['text_only','hybrid']});return
    require(not (dest/'training_started.json').exists(),'Two fixed D fits may run only once')
    rows,fp=load_cohort('train');matrix=select_numeric(rows)
    all_missing=np.isnan(matrix).all(axis=0).tolist()
    # Reject unsupported cohort rather than filling entirely unobserved feature columns with zeros.
    require(not any(all_missing),'A fixed metadata feature has no real train observations; D fit unsupported')
    observation={'features':FEATURES,'train_rows':len(rows),'class_counts':dict(Counter(r['spam_label'] for r in rows)),
        'missing_by_class':{str(label):{feature:int(np.isnan(select_numeric([r for r in rows if r['spam_label']==label])[:,i]).sum()) for i,feature in enumerate(FEATURES)} for label in [0,1]},
        'fingerprint':fp,'cohort_chosen_before_fit':True,'no_auth_brand_history_features':True}
    write_json(dest/'eligibility_before_fit.json',observation)
    write_json(dest/'training_started.json',{'started_utc':now(),'protocol_sha256':sha_file(out/'protocol.json'),
        'code_sha256':sha_file(Path(__file__)),'outer_validation_opened':False})
    def text_pipeline():return Pipeline([('select',FunctionTransformer(stable_text,validate=False)),
        ('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=100000,sublinear_tf=True))])
    numeric_pipeline=Pipeline([('select',FunctionTransformer(stable_numeric,validate=False)),
        ('imputation_and_missing',FeatureUnion([
            ('median',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler(with_mean=False))])),
            ('missing',FunctionTransformer(stable_missing,validate=False))]))])
    pipelines={'text_only':text_pipeline(),'hybrid':FeatureUnion([('text',text_pipeline()),('numeric',numeric_pipeline)])}
    modelspec={}
    for name,features in pipelines.items():
        model=Pipeline([('features',features),('classifier',LogisticRegression(C=1.,class_weight=None,random_state=42,max_iter=2000))])
        start=time.perf_counter();model.fit(rows,[r['spam_label'] for r in rows]);seconds=time.perf_counter()-start
        path=models/f'{name}.joblib';joblib.dump(model,path);loaded=joblib.load(path)
        probe=rows[:32];err=float(np.max(np.abs(model.decision_function(probe)-loaded.decision_function(probe))))
        require(err<=1e-12,'D reload mismatch')
        modelspec[name]={'path':str(path.relative_to(ROOT).as_posix()),'sha256':sha_file(path),'fit_seconds':seconds,
            'iterations':model['classifier'].n_iter_.tolist(),'reload_max_logit_error':err,'reload_rows':len(probe),
            'rss_bytes':psutil.Process().memory_info().rss}
        print('D fitted '+name,flush=True)
    write_rows(dest/'private_train_membership.jsonl',[{k:r[k] for k in ['record_id','duplicate_group_id','spam_label']} for r in rows])
    manifest={'frozen_utc':now(),'status':'completed','models':modelspec,'threshold':.5,'eligible_cohort':observation,
        'versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','joblib']},'not_default':True,'validation_not_opened':True}
    write_json(dest/'candidate_manifest.json',manifest);write_json(models/'candidate_manifest.json',manifest)
    write_json(dest/'train_accesses.json',boundary.report())

if __name__=='__main__':main()
