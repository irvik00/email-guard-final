"""Run permitted Day-4 classical experiments; never modify Day-3 artifacts."""
import argparse
import copy
import json
import shutil
import sys
import time
import uuid
from collections import Counter
from datetime import datetime,timezone

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from baseline_common import ROOT,require,sha_file,digest_json,build_pipeline,scores,metrics,threshold_modes,json_params,versions,code_fingerprints
from day4_common import Day4Boundary,load_day4,clean_text,exclude_collisions,evaluate,subset_records
from day4_reporting import review_data,dump_rows,render_final
from normalization_reporting import fragment
from pipeline_common import write_json,write_text,rows
from predict_email import load_bundle
from train_baselines import fit_logged,save_joblib


def protected_snapshot(root):
    paths=[p for p in (root/'models/day3').rglob('*') if p.is_file()]
    paths.extend(root/p for p in ['requirements.txt','requirements-lock.txt','config/day3_baselines.json'])
    paths.extend(root/p for p in code_fingerprints())
    return {p.relative_to(root).as_posix():sha_file(p) for p in paths}


def execute(root=ROOT):
    started=time.perf_counter();config=json.loads((root/'config/day4_experiments.json').read_bytes())
    old_config=json.loads((root/config['historical_config']).read_bytes())
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+digest_json(config)[:8]+'-'+uuid.uuid4().hex[:8]
    report=root/'reports/day4/runs'/run_id;models=root/'models/day4/runs'/run_id
    report.mkdir(parents=True);models.mkdir(parents=True)
    write_json(root/'reports/day4/current_run.json',{'run_id':run_id})
    protected=protected_snapshot(root);write_json(report/'preserved_day3_and_environment.json',protected)
    result={'run_id':run_id,'status':'running','config':config,'versions':versions(),'experiments':{},'test_not_evaluated':True,
        'config_sha256':sha_file(root/'config/day4_experiments.json'),
        'code_sha256':{'src/'+p.name:sha_file(p) for p in [root/'src'/n for n in ['day4_common.py','day4_reporting.py','run_day4.py','predict_email_day4.py']]}}
    try:
        with Day4Boundary(root) as boundary,threadpool_limits(1):
            data,folds,contract=load_day4(root,config,old_config);result['contract']=contract
            # Save provenance text privately, keep main JSON concise.
            provenance=contract.pop('provenance');write_json(report/'provenance_review.json',provenance)
            write_json(report/'contract.json',contract);write_json(report/'frozen_config.json',config)
            bundle=load_bundle(root/'models/day3/baseline_bundle.joblib',root);e0=bundle['pipeline']
            require(bundle['thresholds']['provisional_threshold']==config['e0_threshold'],'E0 threshold mismatch')
            hist=data['spam_email'];ru=data['russian_auxiliary'];all_records=[r for ds in data.values() for split in ds.values() for r in split]
            lookup={r['record_id']:clean_text(r,config)[0] for r in all_records}
            changed=[r for r in all_records if lookup[r['record_id']]!=r['model_text']]
            change_counts={domain+'/'+split:dict(Counter(f"{r['source']}:{r['spam_label']}" for r in records if lookup[r['record_id']]!=r['model_text'])) for domain,ds in data.items() for split,records in ds.items()}
            cleaning={'builder':config['clean_builder'],'builder_code_sha256':sha_file(root/'src/day4_common.py'),'status':'no_op' if not changed else 'changed',
                'changed_records':len(changed),'by_source_class_split':change_counts,
                'samples':[{'record_id':r['record_id'],'before':fragment(r['subject']),'after':fragment(lookup[r['record_id']].split('\n\n')[0])} for r in changed[:8]],'fit_exclusions':{}}
            result['experiments']['E0']={'status':'unchanged_saved_day3','validation':{}};predictions={};pipelines={'E0':e0}
            for domain,ds in data.items():
                ev,pred=evaluate(e0,ds['validation'],[r['model_text'] for r in ds['validation']],[.5,config['e0_threshold']])
                result['experiments']['E0']['validation'][domain]=ev;predictions['E0',domain]=pred
                dump_rows(report/f'predictions/E0_{domain}_validation.jsonl',pred)
            saved=list(rows(root/'reports/day3/predictions/char_lr_validation.jsonl'))
            observed=predictions['E0','spam_email'];require(len(saved)==len(observed),'E0 prediction size')
            require(all(a['record_id']==b['record_id'] and abs(a['spam_score']-b['spam_score'])<=1e-12 for a,b in zip(saved,observed)),'E0 score mismatch')
            require(metrics([r['spam_label'] for r in hist['validation']],[p['spam_score'] for p in observed],config['e0_threshold'])['FP']==8,'E0 expected FP mismatch')
            result['e0_verification']={'status':'passed','max_score_error':max(abs(a['spam_score']-b['spam_score']) for a,b in zip(saved,observed)),'fit_performed':False}
            review_data(report,data,contract,observed,config['e0_threshold'],config['seed'])
            selected,quotas=subset_records(data,config)
            subset=[{'record_id':r['record_id'],'source':r['source'],'domain':'russian_auxiliary' if r['source']=='russian_auxiliary' else 'spam_email',
                'spam_label':r['spam_label'],'text_builder':'clean_builder_v2' if changed else 'day3_original',
                'text_sha256':digest_json(lookup[r['record_id']])} for r in selected]
            write_json(report/'pilot_subset_manifest.json',{'run_id':run_id,'seed':42,'quotas':quotas,'selection':'largest-remainder source/class quotas, SHA(seed,pilot,ID); independent of errors/scores','records':subset,'sha256':digest_json(subset),'created_before_laya':True,'independent_test':False})
            # Full E1 fit excludes text collisions with either permitted validation domain.
            all_val=hist['validation']+ru['validation']
            if changed:
                e1={'status':'running','validation':{}};result['experiments']['E1']=e1
                oof=np.full(len(hist['train']),np.nan);fold_results=[]
                for fold in range(5):
                    held=[r for r,f in zip(hist['train'],folds) if f==fold];train=[r for r,f in zip(hist['train'],folds) if f!=fold]
                    train,removed=exclude_collisions(train,held,lookup)
                    cleaning['fit_exclusions']['cv_'+str(fold)]=removed
                    require({r['spam_label'] for r in train}=={0,1},'Purged fold lacks a class')
                    print(f'E1 CV {fold+1}/5 fit rows={len(train)} excluded={len(removed)}',flush=True)
                    model=build_pipeline('char_lr',old_config)
                    fit=fit_logged(model,[lookup[r['record_id']] for r in train],[r['spam_label'] for r in train],f'E1 fold{fold}',32000)
                    s=scores(model,[lookup[r['record_id']] for r in held]);oof[folds==fold]=s
                    fold_results.append({'fold_id':fold,'fit_rows':len(train),'excluded_rows':len(removed),'metrics':metrics([r['spam_label'] for r in held],s,.5),'fit':fit})
                require(np.all(np.isfinite(oof)),'OOF missing')
                e1['cv_summary']={'mean_f1':float(np.mean([f['metrics']['f1'] for f in fold_results])),'std_f1_ddof0':float(np.std([f['metrics']['f1'] for f in fold_results])),'folds':5}
                e1['cv_folds']=fold_results;e1['pooled_oof']=metrics([r['spam_label'] for r in hist['train']],oof,.5)
                dump_rows(report/'predictions/E1_oof.jsonl',[{'record_id':r['record_id'],'duplicate_group_id':r['duplicate_group_id'],'source':r['source'],'spam_label':r['spam_label'],'fold_id':int(f),'spam_score':float(s)} for r,f,s in zip(hist['train'],folds,oof)])
                train,removed=exclude_collisions(hist['train'],all_val,lookup);cleaning['fit_exclusions']['E1_full']=removed
                model=build_pipeline('char_lr',old_config);print(f'E1 full fit rows={len(train)}',flush=True)
                e1['fit']=fit_logged(model,[lookup[r['record_id']] for r in train],[r['spam_label'] for r in train],'E1 full',32000)
                e1['fit_rows']=len(train);e1['status']='completed';pipelines['E1']=model
            else:result['experiments']['E1']={'status':'no_op','reason':'No records changed; identical model not retrained'}
            if contract['ru_status']=='source_labels_used_provisionally':
                e2={'status':'running','validation':{},'label_status':contract['ru_status']};result['experiments']['E2']=e2
                train,removed=exclude_collisions(hist['train']+ru['train'],all_val,lookup);cleaning['fit_exclusions']['E2_full']=removed
                model=build_pipeline('char_lr',old_config);print(f'E2 mixed full fit rows={len(train)} excluded={len(removed)}',flush=True)
                e2['fit']=fit_logged(model,[lookup[r['record_id']] for r in train],[r['spam_label'] for r in train],'E2 full',32000)
                e2['fit_rows']=len(train);e2['fit_classes']=dict(Counter(str(r['spam_label']) for r in train));e2['status']='completed';pipelines['E2']=model
            else:result['experiments']['E2']={'status':'blocked','reason':contract['ru_mapping_errors']}
            result['reload_checks']={};artifacts={}
            for name,model in pipelines.items():
                if name=='E0':continue
                for domain,ds in data.items():
                    texts=[lookup[r['record_id']] for r in ds['validation']]
                    ev,pred=evaluate(model,ds['validation'],texts,[.5,config['e0_threshold']] if name=='E1' else [.5])
                    result['experiments'][name]['validation'][domain]=ev;predictions[name,domain]=pred
                    dump_rows(report/f'predictions/{name}_{domain}_validation.jsonl',pred)
                if name=='E1':
                    tuned=threshold_modes([r['spam_label'] for r in hist['validation']],[p['spam_score'] for p in predictions[name,'spam_email']],old_config)
                    result['experiments'][name]['tuned_threshold_summary']={k:tuned[k] for k in ['provisional_threshold','provisional_policy','unmet_fp_scenario']}
                    result['experiments'][name]['tuned_threshold_summary']['metrics']=tuned['modes']['conservative']['metrics']
                    write_json(report/'E1_tuned_thresholds.json',tuned)
                bundle_new={'pipeline':model,'model_id':name,'run_id':run_id,'builder':config['clean_builder'],'builder_name':'clean_builder_v2' if changed else 'day3_original',
                    'code_sha256':result['code_sha256'],'versions':versions(),'parameters':json_params(model.get_params(deep=True)),
                    'input_fingerprints':contract['day4_inputs'],'config':config,'threshold':config['e0_threshold'] if name=='E1' else .5,
                    'research_candidate':True,'ru_label_status':contract['ru_status'],'not_default':True}
                path=models/f'{name}_bundle.joblib';save_joblib(path,bundle_new)
                artifacts[path.name]={'sha256':sha_file(path),'size_bytes':path.stat().st_size}
                loaded=joblib.load(path)
                require(loaded['builder']==config['clean_builder'] and loaded['code_sha256']==result['code_sha256'],'Reload builder metadata mismatch')
                maxerr=0
                for domain,ds in data.items():
                    texts=[clean_text(r,loaded['config'])[0] for r in ds['validation']]
                    expected=np.array([p['spam_score'] for p in predictions[name,domain]])
                    diff=float(np.max(np.abs(scores(loaded['pipeline'],texts)-expected)));maxerr=max(maxerr,diff)
                require(maxerr<=1e-12,'Reload score mismatch');result['reload_checks'][name]={'passed':True,'max_absolute_error_full_validation':maxerr,'builder_reproduced':True}
            # Common subset metrics BEFORE any decision scores, on each classifier's normal text.
            subset_metrics={}
            for name,model in pipelines.items():
                subset_metrics[name]={}
                for domain in ['spam_email','russian_auxiliary']:
                    subset_data=[r for r in selected if (r['source']=='russian_auxiliary')==(domain=='russian_auxiliary')]
                    texts=[r['model_text'] if name=='E0' else lookup[r['record_id']] for r in subset_data]
                    ev,pred=evaluate(model,subset_data,texts,[.5,config['e0_threshold']] if name=='E0' else [.5])
                    subset_metrics[name][domain]=ev;dump_rows(report/f'predictions/{name}_{domain}_pilot_full_context.jsonl',pred)
            write_json(report/'pilot_classical_full_context.json',subset_metrics)
            write_json(report/'cleaning_audit.json',cleaning)
            write_text(report/'cleaning_audit.md','\n'.join(['# Узкая очистка clean_builder_v2','',f"Статус: {cleaning['status']}; изменено {len(changed)} записей.",
                f"Правило зафиксировано до fit: `{config['clean_builder']['pattern']}` / IGNORECASE; только observable subject; одно начальное совпадение. Body не анализируется как тема.",
                f"SHA builder: {cleaning['builder_code_sha256']}",f"Полные counts source/class/split: {change_counts}",
                f"Исключения fit из-за canonical равенства очищенного текста held/validation: { {k:len(v) for k,v in cleaning['fit_exclusions'].items()} }. ID в JSON; splits/held validation не изменялись.",
                'Слова spam/linguistics/university, даты, Re:, адреса/URL и цитаты не удаляются. Текст и метки не записываются обратно. Правило одинаково для всех source/class, использует только поле и его доступность.',
                '',*['- '+json.dumps(s,ensure_ascii=False) for s in cleaning['samples']]])+'\n')
            # Save candidate manifest and verify E0/environment/input preservation.
            write_json(models/'model_manifest.json',{'run_id':run_id,'status':'completed','artifacts':artifacts,'config':config,'versions':versions(),'code_sha256':result['code_sha256'],'not_default':True,'test_not_evaluated':True})
            for relative,spec in contract['day4_inputs'].items():require(sha_file(root/spec['path'])==spec['sha256'],'Allowed input modified')
            require(protected_snapshot(root)==protected,'Day3 baseline/core environment modified')
            result['preservation_checks']={'day3_models_code_requirements_unchanged':True,'allowed_input_sha_unchanged':True}
            result['data_access_log']=dict(boundary.accesses)
        result['status']='completed';result['runtime_seconds']=time.perf_counter()-started
        render_final(report,result)
        write_json(report/'run_state.json',{'status':'completed','run_id':run_id})
        for p in report.rglob('*'):
            if p.is_file():
                target=root/'reports/day4'/p.relative_to(report);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        write_json(root/'models/day4/current_run.json',{'run_id':run_id,'path':models.relative_to(root).as_posix(),'default':'models/day3/baseline_bundle.joblib'})
        print(json.dumps({'status':'completed','run_id':run_id,'classical_seconds':result['runtime_seconds'],'E1_CV':result['experiments']['E1'].get('cv_summary'),'RU_status':contract['ru_status'],'test_not_evaluated':True}),flush=True)
        return result
    except Exception as error:
        write_json(report/'run_state.json',{'status':'blocked','run_id':run_id,'reason':type(error).__name__+': '+str(error)})
        raise


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    execute()
