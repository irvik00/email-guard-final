"""Versioned day-3 CV/validation experiment; deliberately cannot open held-out data."""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from baseline_common import (ROOT, DataBoundary, require, load_inputs, build_pipeline, scores, metrics,
    threshold_modes, select_baseline, json_params, versions, code_fingerprints, digest_json, sha_file, explain)
from baseline_reporting import render_experiments, render_thresholds, error_analysis, render_status
from pipeline_common import write_json, write_text, JsonlWriter
from predict_email import predict_path


def fit_logged(pipeline,texts,labels,context,cap):
    history=[];started=time.perf_counter()
    while True:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipeline.fit(texts,labels)
        captured=[{'category':w.category.__name__,'message':str(w.message)} for w in caught]
        limit=pipeline['classifier'].get_params().get('max_iter')
        history.append({'max_iter':limit,'warnings':captured})
        for warning in captured:print(f"warning [{context}]: {warning['category']}: {warning['message']}",flush=True)
        if not any(issubclass(w.category,ConvergenceWarning) for w in caught):break
        require(limit is not None and limit<cap,'ConvergenceWarning persists at configured cap: '+context)
        pipeline['classifier'].set_params(max_iter=min(cap,limit*2))
        print(f'convergence retry [{context}]: max_iter={min(cap,limit*2)}; other parameters unchanged',flush=True)
    return {'seconds':time.perf_counter()-started,'attempts':history,'vocabulary_size':len(pipeline['tfidf'].vocabulary_),
            'iterations':json_params(getattr(pipeline['classifier'],'n_iter_',None)),
            'parameters':json_params(pipeline.get_params(deep=True)),
            'rss_bytes_after_fit':psutil.Process().memory_info().rss}


def prediction_rows(path,data,score,threshold,folds=None):
    writer=JsonlWriter(path)
    try:
        for i,(r,s) in enumerate(zip(data,score)):
            writer.write({'record_id':r.record_id,'duplicate_group_id':r.group_id,'source':r.source,
                'spam_label':r.label,'spam_score':float(s),'threshold':float(threshold),'prediction':int(s>=threshold),
                'fold_id':int(folds[i]) if folds is not None else None})
    finally:writer.close()


def source_metrics(data,score,threshold):
    return {source:metrics([r.label for r in data if r.source==source],
                          [s for r,s in zip(data,score) if r.source==source],threshold)
            for source in ['spamassassin','lingspam']}


def save_joblib(path,value):
    temporary=path.with_name(path.name+'.tmp')
    joblib.dump(value,temporary,compress=3)
    os.replace(temporary,path)


def execute(root,config,config_path=None):
    root=Path(root).resolve();started=time.perf_counter()
    # Values below are the declared three variants, not a search space.
    require(config['models']==['word_nb','word_lr','char_lr'],'Exactly the three declared variants are required')
    require(config['seed']==42 and config['cv']['selection_threshold']==.5 and config['threshold']['default']==.5,'Declared seed/default threshold changed')
    config_sha=digest_json(config)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+config_sha[:10]+'-'+uuid.uuid4().hex[:8]
    report_root=root/'reports/day3';model_root=root/'models/day3'
    report_dir=report_root/'runs'/run_id;model_dir=model_root/'runs'/run_id
    report_dir.mkdir(parents=True,exist_ok=False);model_dir.mkdir(parents=True,exist_ok=False)
    state={'run_id':run_id,'status':'running','test_not_evaluated':True,'stages':{},'config_sha256':config_sha}
    write_json(report_dir/'run_state.json',state)
    eml=(root/config['smoke_eml']).resolve()
    try:
        with DataBoundary(root,eml=eml) as boundary, threadpool_limits(limits=config['runtime']['threads']):
            data,folds,contract=load_inputs(root,config)
            code=code_fingerprints();dependency_versions=versions()
            contract.update(run_id=run_id,config=config,config_sha256=config_sha,code_sha256=code,versions=dependency_versions,
                            python_version=sys.version,cpu_only=True,threads=config['runtime']['threads'])
            write_json(report_dir/'contract.json',contract);write_json(report_dir/'config.json',config)
            write_json(report_dir/'acceptance.json',{'decision':'accepted_for_historical_baseline',
                'basis':'DAY3_BASELINES.md and user instruction; only spam_email train/validation',
                'other_data_readiness':'not accepted by this decision; September 2026 security_blocked; needs_review excluded',
                'test_not_evaluated':True,'run_id':run_id})
            state['stages']['inputs']='completed'
            train,valid=data['train'],data['validation'];y=np.array([r.label for r in train]);yv=np.array([r.label for r in valid])
            texts=[r.text for r in train];vtexts=[r.text for r in valid]
            threshold=config['cv']['selection_threshold'];results={};oof_scores={};models={};validation_scores={}
            print(f'inputs accepted: train={len(train)}, validation={len(valid)}, CV folds={len(set(folds))}; held-out data denied',flush=True)
            for model_id in config['models']:
                oof=np.full(len(train),np.nan);seen=np.zeros(len(train),dtype=int);fold_results=[]
                for fold in sorted(set(folds)):
                    learn=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
                    model=build_pipeline(model_id,config)
                    log=fit_logged(model,[texts[i] for i in learn],y[learn],f'{model_id}/CV{fold}',config['runtime']['convergence_max_iter_cap'])
                    oof[held]=scores(model,[texts[i] for i in held]);seen[held]+=1
                    measured=metrics(y[held],oof[held],threshold)
                    fold_results.append({'fold_id':int(fold),'train_records':len(learn),'held_records':len(held),
                        'held_class_counts':dict(Counter(str(int(v)) for v in y[held])),'metrics':measured,'fit':log})
                    print(f"{model_id} CV fold={fold}: F1={measured['f1']:.6f}, features={log['vocabulary_size']}, fit={log['seconds']:.2f}s",flush=True)
                    del model
                require(np.all(seen==1) and np.all(np.isfinite(oof)),'OOF coverage not exactly once')
                mean={k:float(np.mean([f['metrics'][k] for f in fold_results])) if all(f['metrics'][k] is not None for f in fold_results) else None for k in ['precision','recall','f1','accuracy','fpr']}
                std={k:float(np.std([f['metrics'][k] for f in fold_results],ddof=0)) if mean[k] is not None else None for k in mean}
                results[model_id]={'cv':{'folds':fold_results,'mean':mean,'std_population_ddof0':std,
                    'pooled_oof':metrics(y,oof,threshold),'threshold':threshold,'not_independent_test':True}}
                oof_scores[model_id]=oof
                prediction_rows(report_dir/f'predictions/{model_id}_oof.jsonl',train,oof,threshold,folds)
                write_json(report_dir/'cv_checkpoint.json',results)
            selected=select_baseline(results,config)
            write_json(report_dir/'selection_before_validation.json',{'selected':selected,'basis':'highest mean fold binary spam F1 at 0.5; precision and declared preference break numerical ties','cv_results':{k:r['cv']['mean'] for k,r in results.items()},'validation_seen_for_selection':False})
            state['stages']['cv']='completed';print('baseline selected from CV before validation: '+selected,flush=True)
            for model_id in config['models']:
                model=build_pipeline(model_id,config)
                log=fit_logged(model,texts,y,f'{model_id}/full_train',config['runtime']['convergence_max_iter_cap'])
                score=scores(model,vtexts);models[model_id]=model;validation_scores[model_id]=score
                results[model_id]['full_train_fit']=log
                results[model_id]['validation_default']=metrics(yv,score,threshold)
                results[model_id]['validation_by_source']=source_metrics(valid,score,threshold)
                results[model_id]['validation_empty_vectors']=int(np.sum(model['tfidf'].transform(vtexts).getnnz(axis=1)==0))
                prediction_rows(report_dir/f'predictions/{model_id}_validation.jsonl',valid,score,threshold)
                save_joblib(model_dir/f'{model_id}.joblib',model)
                write_json(model_dir/f'{model_id}_feature_names.json',[str(v) for v in model['tfidf'].get_feature_names_out()])
                print(f"{model_id} full train: validation F1={results[model_id]['validation_default']['f1']:.6f}, features={log['vocabulary_size']}, fit={log['seconds']:.2f}s",flush=True)
            dummy=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train),1)),y)
            dummy_metrics=metrics(yv,predictions=dummy.predict(np.zeros((len(valid),1))))
            dummy_by_source={s:metrics([r.label for r in valid if r.source==s],predictions=dummy.predict(np.zeros((sum(r.source==s for r in valid),1)))) for s in ['spamassassin','lingspam']}
            thresholds=threshold_modes(yv,validation_scores[selected],config)
            thresholds.update(run_id=run_id,baseline=selected,chosen_before_validation=True)
            for mode,spec in thresholds['modes'].items():
                if spec is not None:spec['by_source']=source_metrics(valid,validation_scores[selected],spec['threshold'])
            state['stages'].update(validation='completed',thresholds='completed')
            prediction_rows(report_dir/f'predictions/{selected}_validation_provisional.jsonl',valid,validation_scores[selected],thresholds['provisional_threshold'])
            errors,error_json=error_analysis(models,selected,valid,validation_scores[selected],thresholds)
            write_text(report_dir/'error_analysis.md',errors);write_json(report_dir/'error_analysis.json',error_json)
            state['stages']['error_analysis']='completed'
            # Reload three pipelines; the subset is deterministic and never a held-out test.
            reload_indices=sorted(range(len(valid)),key=lambda i:digest_json(valid[i].record_id))[:config['runtime']['reload_sample_size']]
            subset=[vtexts[i] for i in reload_indices];reload_checks={}
            for model_id in config['models']:
                loaded=joblib.load(model_dir/f'{model_id}.joblib')
                before=scores(models[model_id],subset);after=scores(loaded,subset)
                require(np.allclose(before,after,rtol=0,atol=1e-12),'Reload scores changed: '+model_id)
                for text in subset[:3]:explain(loaded,text)
                reload_checks[model_id]={'n':len(subset),'max_absolute_error':float(np.max(np.abs(before-after))),
                    'local_explanations_checked_after_reload':min(3,len(subset)),'passed':True}
            bundle={'bundle_version':1,'run_id':run_id,'model_id':selected,'pipeline':models[selected],'classes':[0,1],
                'thresholds':{k:v for k,v in thresholds.items() if k not in {'grid'}},'versions':dependency_versions,
                'code_sha256':code,'input_fingerprints':contract['inputs'],'config_sha256':config_sha,
                'model_text_builder':{'function':'normalization_schema.make_model_text','input':'parse_email subject/body_text','feature_allowlist':['model_text']},
                'language_scope':'historical English; other languages untested','test_not_evaluated':True}
            save_joblib(model_dir/'baseline_bundle.joblib',bundle)
            loaded=joblib.load(model_dir/'baseline_bundle.joblib')
            bundle_difference=float(np.max(np.abs(scores(loaded['pipeline'],subset)-scores(models[selected],subset))))
            require(bundle_difference<=1e-12,'Bundle reload changed scores')
            write_json(model_dir/'threshold_config.json',{k:v for k,v in thresholds.items() if k!='grid'})
            artifacts={p.name:{'path':p.relative_to(root).as_posix(),'sha256':sha_file(p),'size_bytes':p.stat().st_size}
                for p in model_dir.iterdir() if p.is_file()}
            model_manifest={'status':'completed','run_id':run_id,'selected_baseline':selected,'artifacts':artifacts,
                'classes':[0,1],'threshold_policy':thresholds['provisional_policy'],'threshold':thresholds['provisional_threshold'],
                'input_fingerprints':contract['inputs'],'preparation_manifest_sha256':contract['manifest_sha256'],
                'config':config,'config_sha256':config_sha,'code_sha256':code,'versions':dependency_versions,
                'parameters':{k:r['full_train_fit']['parameters'] for k,r in results.items()},
                'reload_checks':reload_checks,'bundle_reload_max_absolute_error':bundle_difference,
                'model_text_builder':bundle['model_text_builder'],'test_not_evaluated':True,
                'closed_artifacts_from_manifest_only':contract['closed_artifacts_from_manifest_only']}
            write_json(model_dir/'model_manifest.json',model_manifest)
            before_smoke={'eml':sha_file(eml),'model_artifacts':{name:sha_file(model_dir/name) for name in artifacts}}
            smoke=predict_path(model_dir/'baseline_bundle.joblib',eml,root)
            if root==ROOT:
                completed=subprocess.run([sys.executable,'-B',str(ROOT/'src/predict_email.py'),str(eml),
                    '--bundle',str(model_dir/'baseline_bundle.joblib')],capture_output=True,text=True,encoding='utf-8',check=True)
                cli_smoke=json.loads(completed.stdout)
                require(abs(cli_smoke['spam_score']-smoke['spam_score'])<=1e-12,'CLI/in-process score mismatch')
                smoke['cli_subprocess_verified']=True
            else:smoke['cli_subprocess_verified']='synthetic fixture uses the same in-process path; real CLI checked on project run'
            require(before_smoke=={'eml':sha_file(eml),'model_artifacts':{name:sha_file(model_dir/name) for name in artifacts}},'Predict changed input/model files')
            write_json(report_dir/'cli_smoke.json',smoke)
            for split,spec in contract['inputs'].items():require(sha_file(root/spec['path'])==spec['sha256'],'Dataset changed during training: '+split)
            state['stages'].update(save_reload='completed',cli_smoke='completed',data_boundary='completed')
            state.update(status='completed',seconds=time.perf_counter()-started,data_access_log=dict(boundary.accesses))
            tests_path=report_root/'tests_execution.json'
            if tests_path.is_file():state['tests']=json.loads(tests_path.read_text(encoding='utf-8'))
            model_manifest['verification']={'save_reload':'passed','CLI_smoke':smoke,'data_access_log':dict(boundary.accesses)}
            write_json(model_dir/'model_manifest.json',model_manifest)
            experiments={'status':'completed','run_id':run_id,'selected_baseline':selected,'selection':'mean fold binary F1 at 0.5, then precision, then word_lr/char_lr/word_nb',
                'models':results,'dummy_most_frequent':{'validation_default':dummy_metrics,'validation_by_source':dummy_by_source},
                'contract':contract,'validation_ranking':sorted(results,key=lambda k:-results[k]['validation_default']['f1']),
                'selection_changed_after_validation':False,'fit_contexts_completed':len(set(folds))*3+3,
                'reload_checks':reload_checks,'bundle_reload_max_absolute_error':bundle_difference,
                'seconds':state['seconds'],'data_access_log':dict(boundary.accesses),'test_not_evaluated':True}
            write_json(report_dir/'experiments.json',experiments);write_text(report_dir/'experiments.md',render_experiments(experiments))
            write_json(report_dir/'thresholds.json',thresholds);write_text(report_dir/'thresholds.md',render_thresholds(thresholds))
            write_json(report_dir/'run_state.json',state)
            # Archive is immutable; publish latest convenience copies with run metadata last.
            for src,base in [(report_dir,report_root),(model_dir,model_root)]:
                for path in sorted(src.rglob('*')):
                    if not path.is_file() or path.name=='model_manifest.json':continue
                    destination=base/path.relative_to(src);destination.parent.mkdir(parents=True,exist_ok=True)
                    tmp=destination.with_name(destination.name+'.tmp');shutil.copyfile(path,tmp);os.replace(tmp,destination)
            write_json(model_root/'model_manifest.json',model_manifest)
            write_json(report_root/'latest_run.json',{'run_id':run_id,'report_dir':report_dir.relative_to(root).as_posix(),'model_dir':model_dir.relative_to(root).as_posix(),'status':'completed'})
            write_text(root/'reports/DAY3_STATUS.md',render_status(state,experiments,thresholds,smoke))
            print(json.dumps({'status':'completed','run_id':run_id,'selected':selected,'threshold':thresholds['provisional_threshold'],
                'validation':thresholds['modes']['conservative' if not thresholds['unmet_fp_scenario'] else 'default']['metrics'],
                'seconds':round(state['seconds'],2),'test_not_evaluated':True}),flush=True)
            return experiments,thresholds
    except Exception as error:
        state.update(status='blocked',error_type=type(error).__name__,error=str(error)[:500],seconds=time.perf_counter()-started)
        write_json(report_dir/'run_state.json',state)
        write_text(root/'reports/DAY3_STATUS.md',f"# DAY3_STATUS\n\nblocked: {type(error).__name__}: {str(error)[:500]}\n\nRun: {run_id}; stages: {json.dumps(state['stages'])}. Test not evaluated. Partial artifacts retained in the versioned run; no completed latest model published by this run.\n")
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/day3_baselines.json')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    execute(ROOT,config,args.config)


if __name__=='__main__':main()
