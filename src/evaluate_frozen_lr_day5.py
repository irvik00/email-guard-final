"""Frozen E0/E2 final validation full-context vs EXACT Laya-visible text. No fit."""
import argparse
import importlib.metadata
import json
import math
import sys
import time

from day5_common import (ROOT, DOMAINS, Boundary, read_json, read_rows, write_json, write_rows,
                         sha_file, require, metrics, probability_metrics)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id');args=parser.parse_args()
    run_id=args.run_id or read_json(ROOT/'reports/day5/current_run.json')['run_id']
    out=ROOT/'reports/day5/runs'/run_id
    boundary=Boundary('verify').install()
    manifest=read_json(out/'candidate_manifest.json')
    require((out/'validation_started.json').exists(),'Final evaluation must be authorized by frozen candidates')
    require(not (out/'lr_validation_comparison.json').exists(),'Frozen LR comparison already exists; no repeated evaluation for tuning')
    validation=read_rows(out/'private_validation_inputs.jsonl');visible=read_rows(out/'private_validation_visible.jsonl')
    require([r['record_id'] for r in validation]==[r['record_id'] for r in visible],'Matched input ID mismatch')
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    from predict_email import load_bundle
    from evaluate_laya_day5 import quantiles
    e0=load_bundle(ROOT/'models/day3/baseline_bundle.joblib',ROOT)
    day4_id=read_json(ROOT/'reports/day4/current_run.json')['run_id']
    modeldir=ROOT/'models/day4/runs'/day4_id;mm=read_json(modeldir/'model_manifest.json')
    require(sha_file(modeldir/'E2_bundle.joblib')==mm['artifacts']['E2_bundle.joblib']['sha256'],'E2 hash mismatch')
    e2=joblib.load(modeldir/'E2_bundle.joblib')
    results={}
    with threadpool_limits(limits=1):
        for name,bundle in [('E0',e0),('E2',e2)]:
            model=bundle['pipeline'];require(list(model.classes_)==[0,1],'LR class order mismatch')
            # Synthetic warmup only; prohibit all estimators' fit methods in this process.
            for stage in model.named_steps.values():
                def denied_fit(*args,**kwargs):raise RuntimeError('Fit forbidden in final frozen-LR evaluation')
                stage.fit=denied_fit
            model.predict_proba(['Synthetic warmup notice.'])
            for context in ['full_context','matched_visible']:
                predictions=[]
                for i,(r,fragment) in enumerate(zip(validation,visible)):
                    text=fragment['visible_fragment'] if context=='matched_visible' else (r['original_model_text'] if name=='E0' else r['text'])
                    start=time.perf_counter();v=model['tfidf'].transform([text]);p=float(model['classifier'].predict_proba(v)[0,1]);elapsed=time.perf_counter()-start
                    predictions.append({'record_id':r['record_id'],'domain':r['domain'],'source':r['source'],'spam_label':r['spam_label'],
                                        'probability':p,'active_features':v.nnz,'seconds':elapsed,'text_sha256':__import__('hashlib').sha256(text.encode()).hexdigest()})
                    if (i+1)%1000==0:print(f'{name}/{context} {i+1}/{len(validation)}',flush=True)
                write_rows(out/f'{name}_{context}_validation_predictions.jsonl',predictions)
                result={}
                for domain in DOMAINS:
                    rows=[r for r in predictions if r['domain']==domain];y=[r['spam_label'] for r in rows];p=[r['probability'] for r in rows]
                    diff=[math.log(max(1e-15,min(1-1e-15,x))/(1-max(1e-15,min(1-1e-15,x)))) for x in p]
                    thresholds=[.5,manifest['config']['evaluation']['threshold_default_e0']] if name=='E0' else [.5]
                    result[domain]={'metrics':{str(t):metrics(y,p,t) for t in thresholds},'probability_metrics':probability_metrics(y,diff),
                                    'coverage':1.,'failures':0,'empty_vectors':sum(r['active_features']==0 for r in rows),
                                    'wrapper_coverage':sum(r['active_features']>0 for r in rows)/len(rows),
                                    'raw_metrics_include_empty_vectors':True,'warm_latency_seconds':quantiles([r['seconds'] for r in rows])}
                results[name+'/'+context]=result
    write_json(out/'lr_validation_comparison.json',{'run_id':run_id,'candidates':results,'fit_performed':False,
               'full_context_builder':{'E0':'unchanged original day3 builder','E2':'clean_builder_v2'},
               'matched_context':'exact same visible_fragment as both Laya candidates',
               'versions':{p:importlib.metadata.version(p) for p in ['numpy','scikit-learn','scipy','joblib']},
               'access_log':boundary.report()})
    print(json.dumps({n:{d:v['metrics'] for d,v in result.items()} for n,result in results.items()},indent=2))


if __name__=='__main__':main()
