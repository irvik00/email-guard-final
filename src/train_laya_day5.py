"""One bounded head-only Laya run, internal calibration and frozen candidates. NO validation."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

from day5_common import (ROOT, DOMAINS, Boundary, read_json, read_rows, write_json, write_rows, sha_file,
                         digest, now, require, fit_temperature, choose_threshold, probability_metrics)


def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    args=parser.parse_args()
    run_id=args.run_id or read_json(ROOT/'reports/day5/current_run.json')['run_id']
    out=ROOT/'reports/day5/runs'/run_id;models=ROOT/'models/day5/runs'/run_id
    require(out.is_dir(),'Prepare contract first')
    require(not (out/'training_started.json').exists(),'A run may train only once. Prepare a new run for explicit reproduction.')
    boundary=Boundary('train').install()
    config=read_json(out/'experiment_contract.json')['config'];data_manifest=read_json(out/'data_manifest.json')
    metadata=read_json(ROOT/'reports/day4/setup/runtime_manifest.json')
    require(all(metadata[k]==config['checkpoint'][k] for k in config['checkpoint']),'Wrong checkpoint')
    source_root=Path(metadata['local_path'])
    require(sha_file(source_root/'model.safetensors')==metadata['files']['multilingual/model.safetensors']['sha256'],'Source weights changed')
    inputs={}
    for part in ['fit','dev','calibration']:
        path=out/f'private_{part}_inputs.jsonl'
        require(sha_file(path)==data_manifest['selection_sha256'][part],'Selected input hash changed')
        inputs[part]=read_rows(path)
        require(all(r['inner_part']==part for r in inputs[part]),'Wrong role in inputs')
    require(not ({r['inner_group_id'] for r in inputs['fit']} & {r['inner_group_id'] for p in ['dev','calibration'] for r in inputs[p]}),'Fit/held groups overlap')
    write_json(out/'training_started.json',{'run_id':run_id,'started_utc':now(),'config_sha256':digest(config),
                'membership_sha256':data_manifest['membership_sha256'],
                'code_sha256':{p:sha_file(ROOT/p) for p in ['src/train_laya_day5.py','src/day5_laya_runtime.py','src/day5_common.py']},
                'validation_opened':False})
    from day5_laya_runtime import (torch,parameter_hash,configure_trainable,encode_rows, batch_tensors,low_forward,
                                  infer_encoded,summarize_prediction_rows,save_stock_checkpoint,synthetic_contract_smoke,
                                  TEMP_MIN,TEMP_MAX,clamp_temperature)
    import laya
    import psutil
    from safetensors.torch import save_file,load_file
    torch.set_num_threads(1);torch.manual_seed(config['seed']);random.seed(config['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True,warn_only=True)
    started=time.perf_counter();models.mkdir(parents=True,exist_ok=False)
    agent=laya.load(str(source_root),device='cuda',fast=False)
    require(agent.device.type=='cuda','CUDA training required; actual SDK load fell back to CPU')
    torch.cuda.reset_peak_memory_stats()
    versions={name:importlib.metadata.version(name) for name in metadata['versions']}
    require(versions==metadata['versions'],'Decision environment versions changed; do not silently upgrade')
    parameters=configure_trainable(agent,config);write_json(out/'named_parameters.json',parameters)
    frozen_before=parameter_hash(agent.model,['encoder.']);action_before=parameter_hash(agent.model,['act_head.'])
    head_before=parameter_hash(agent.model,config['training']['trainable_prefixes'])
    print('Loaded frozen checkpoint; encoding selected fit/dev/calibration only',flush=True)
    encoded={};visible={}
    for part in inputs:
        encoded[part],visible[part]=encode_rows(agent,inputs[part],config,out/f'private_{part}_visible.jsonl')
    smoke=synthetic_contract_smoke(agent,config);write_json(out/'inference_mapping_smoke.json',smoke)
    progress=lambda i,n: print(f'inference {i}/{n}',flush=True)
    base_dev=infer_encoded(agent,encoded['dev'],progress=progress)
    write_rows(out/'base_dev_raw.jsonl',base_dev)
    base_cal=infer_encoded(agent,encoded['calibration'],progress=progress)
    write_rows(out/'base_calibration_raw.jsonl',base_cal)
    initial=summarize_prediction_rows(base_dev)
    write_json(out/'base_internal_dev.json',initial)
    params=[p for p in agent.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    optimizer_ids={id(p) for group in optimizer.param_groups for p in group['params']}
    require(optimizer_ids=={id(p) for p in agent.model.parameters() if p.requires_grad},'Optimizer parameter mismatch')
    require(not any(id(p) in optimizer_ids for p in agent.model.encoder.parameters()),'Optimizer contains encoder')
    order=list(range(len(encoded['fit'])));random.Random(config['seed']).shuffle(order)
    tcfg=config['training'];micro=tcfg['microbatch'];accum=tcfg['accumulation'];effective=micro*accum
    cursor=0;used=[];curve=[];step_logs=[];best=None;oom_retry=0;train_start=time.perf_counter()
    curvefile=(out/'learning_curve.csv').open('w',newline='',encoding='utf-8')
    writer=csv.DictWriter(curvefile,fieldnames=['step','examples_seen','train_loss','dev_loss','historical_F1','historical_FP','ru_F1','ru_FP','mean_domain_F1','elapsed_train_seconds'])
    writer.writeheader()

    def log_curve(step,train_loss,dev):
        avg=sum(dev[d]['metrics']['F1'] for d in DOMAINS)/2
        row={'step':step,'examples_seen':len(used),'train_loss':train_loss,
             'dev_loss':sum(dev[d]['probability_metrics']['NLL'] for d in DOMAINS)/2,
             'historical_F1':dev['spam_email']['metrics']['F1'],'historical_FP':dev['spam_email']['metrics']['FP'],
             'ru_F1':dev['russian_auxiliary']['metrics']['F1'],'ru_FP':dev['russian_auxiliary']['metrics']['FP'],
             'mean_domain_F1':avg,'elapsed_train_seconds':time.perf_counter()-train_start}
        curve.append(row);writer.writerow(row);curvefile.flush();return row

    log_curve(0,None,initial)
    training_status='completed';blocker=None;actual_steps=0;smoke_report=None
    try:
        for step in range(1,tcfg['max_steps']+1):
            if time.perf_counter()-train_start > tcfg['train_seconds_limit']:
                training_status='partial_time_limit';break
            # Warmup from 1/15 to target LR; then linear decay, fixed before training.
            scale=min(step/tcfg['warmup_steps'],max(0.,(tcfg['max_steps']-step+1)/(tcfg['max_steps']-tcfg['warmup_steps'])))
            for group in optimizer.param_groups:group['lr']=tcfg['learning_rate']*scale
            agent.model.train();agent.model.encoder.eval();agent.model.act_head.eval()
            require(not agent.model.encoder.training and agent.model.head.training,'Incorrect frozen/train modes')
            batch_order=[order[(cursor+j)%len(order)] for j in range(effective)]
            retry=True
            while retry:
                retry=False;optimizer.zero_grad(set_to_none=True);losses=[]
                try:
                    for offset in range(0,effective,micro):
                        examples=[encoded['fit'][i] for i in batch_order[offset:offset+micro]]
                        require(all(r['inner_part']=='fit' for r,_ in examples),'Training attempted outside fit')
                        b=batch_tensors(agent,examples);logits,_=low_forward(agent,b)
                        loss=torch.nn.functional.cross_entropy(logits[:,:2].float(),b['label'].to(agent.device))
                        require(bool(torch.isfinite(loss)),'Nonfinite training loss')
                        (loss/accum).backward();losses.append(float(loss.detach().cpu()))
                    require(all(p.grad is None for p in agent.model.encoder.parameters()),'Encoder gradients present')
                    grad=float(torch.nn.utils.clip_grad_norm_(params,tcfg['grad_clip']).item())
                    require(math.isfinite(grad) and grad>0,'Nonfinite/zero decision-head gradients')
                    optimizer.step()
                except torch.cuda.OutOfMemoryError as error:
                    if oom_retry>=tcfg['oom_retries'] or micro==1:raise
                    oom_retry+=1;micro=1;accum=effective;optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();retry=True
                    write_json(out/'oom_retry.json',{'step':step,'error':str(error),'retry_microbatch':micro,'accumulation':accum,'effective_batch':effective})
            cursor+=effective;actual_steps=step
            used.extend(encoded['fit'][i][0]['record_id'] for i in batch_order)
            step_log={'step':step,'loss':sum(losses)/len(losses),'gradient_norm_before_clip':grad,
                      'learning_rate':optimizer.param_groups[0]['lr'],'examples_seen':len(used),
                      'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),
                      'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),
                      'rss_bytes':psutil.Process().memory_info().rss,'elapsed_seconds':time.perf_counter()-train_start}
            step_logs.append(step_log)
            with (out/'training_steps.jsonl').open('ab') as f:f.write(json.dumps(step_log).encode()+b'\n')
            if step==tcfg['smoke_steps']:
                smoke_report={'real_optimizer_steps':step,'examples_seen':len(used),'nonzero_finite_gradient':True,
                              'head_changed':parameter_hash(agent.model,tcfg['trainable_prefixes'])!=head_before,
                              'encoder_unchanged':parameter_hash(agent.model,['encoder.'])==frozen_before,
                              'action_branch_unchanged':parameter_hash(agent.model,['act_head.'])==action_before,
                              'real_device':str(agent.device),'microbatch':micro,'accumulation':accum,**step_log}
                require(smoke_report['head_changed'] and smoke_report['encoder_unchanged'] and smoke_report['action_branch_unchanged'],'Gradient smoke invariants failed')
                write_json(out/'gradient_smoke.json',smoke_report)
                print('Real gradient smoke passed: '+json.dumps(smoke_report),flush=True)
            if step%25==0:print(f"train {step}/{tcfg['max_steps']} loss={step_log['loss']:.5f} grad={grad:.4f} elapsed={step_log['elapsed_seconds']:.1f}s",flush=True)
            if step in tcfg['dev_steps'] or step==tcfg['max_steps']:
                predictions=infer_encoded(agent,encoded['dev'],progress=progress)
                write_rows(out/f'adapted_dev_step_{step}.jsonl',predictions)
                summary=summarize_prediction_rows(predictions);row=log_curve(step,sum(x['loss'] for x in step_logs[-100:])/min(100,len(step_logs)),summary)
                key=(row['mean_domain_F1'],-row['historical_FP'],-step)
                if best is None or key>best['key']:
                    state={k:v.detach().cpu().contiguous() for k,v in agent.model.state_dict().items() if not k.startswith(('encoder.','act_head.'))}
                    save_file(state,str(models/'selected_head.safetensors'))
                    best={'key':key,'step':step,'dev_predictions':predictions,'dev_summary':summary}
                print('DEV '+json.dumps(row),flush=True)
    except Exception as error:
        training_status='partial_technical_error' if actual_steps else 'blocked'
        blocker={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        write_json(out/'training_blocker.json',blocker)
        print('Training interrupted: '+str(error),flush=True)
    finally:curvefile.close()
    if actual_steps and best is None:
        agent.model.eval();predictions=infer_encoded(agent,encoded['dev'],progress=progress)
        summary=summarize_prediction_rows(predictions)
        save_file({k:v.detach().cpu().contiguous() for k,v in agent.model.state_dict().items() if not k.startswith(('encoder.','act_head.'))},str(models/'selected_head.safetensors'))
        best={'key':(sum(summary[d]['metrics']['F1'] for d in DOMAINS)/2,-summary['spam_email']['metrics']['FP'],-actual_steps),
              'step':actual_steps,'dev_predictions':predictions,'dev_summary':summary}
        write_rows(out/f'adapted_dev_step_{actual_steps}.jsonl',predictions)
    training={'run_id':run_id,'status':training_status,'blocker':blocker,'actual_optimizer_steps':actual_steps,
              'selected_step':best['step'] if best else None,'fit_pool':len(inputs['fit']),
              'examples_presented':len(used),'unique_records_used':len(set(used)),'fraction_of_selected_fit_epoch':len(used)/len(inputs['fit']),
              'actual_used_counts':dict(Counter(r['domain']+':'+str(r['spam_label']) for r in inputs['fit'] if r['record_id'] in set(used))),
              'trainable_parameters':sum(p.numel() for p in params),'total_parameters':sum(p.numel() for p in agent.model.parameters()),
              'optimizer_parameter_set_checked':True,'real_device':str(agent.device),'gpu':torch.cuda.get_device_name(),
              'microbatch':micro,'accumulation':accum,'effective_batch':effective,'oom_retries':oom_retry,
              'encoder_sha256_before':frozen_before,'encoder_sha256_after':parameter_hash(agent.model,['encoder.']),
              'head_sha256_before':head_before,'head_sha256_final_trajectory':parameter_hash(agent.model,tcfg['trainable_prefixes']),
              'act_head_sha256_before':action_before,'act_head_sha256_after':parameter_hash(agent.model,['act_head.']),
              'training_seconds':time.perf_counter()-train_start,'total_process_seconds_so_far':time.perf_counter()-started,
              'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),
              'rss_bytes':psutil.Process().memory_info().rss,'versions':versions,'gradient_smoke':smoke_report,
              'learning_curve':curve,'selected_dev':best['dev_summary'] if best else None,
              'not_RLCD':True,'validation_opened':False}
    write_rows(out/'actual_training_membership.jsonl',[{'ordinal':i,'record_id':rid} for i,rid in enumerate(used)])
    require(training['encoder_sha256_before']==training['encoder_sha256_after'],'Encoder changed after training')
    write_json(out/'training.json',training)
    # Calibration data are opened for logits only after selecting adapted weights.
    candidates={};calibration={};reload_checks={}
    raw_by_candidate={'base':(base_dev,base_cal)}
    if best:
        agent.model.load_state_dict(load_file(str(models/'selected_head.safetensors')),strict=False)
        adapted_cal=infer_encoded(agent,encoded['calibration'],progress=progress)
        write_rows(out/'adapted_calibration_raw.jsonl',adapted_cal)
        raw_by_candidate['adapted']=(best['dev_predictions'],adapted_cal)
        training['head_sha256_selected']=parameter_hash(agent.model,tcfg['trainable_prefixes'])
        require(training['head_sha256_selected']!=head_before,'Selected trained head unchanged')
    require([TEMP_MIN,TEMP_MAX]==config['calibration']['sdk_range'],'Unexpected installed SDK temperature limits')
    for name,(dev,cal) in raw_by_candidate.items():
        labels=[r['spam_label'] for r in cal];diff=[r['logit_difference'] for r in cal]
        fitted=fit_temperature(labels,diff,TEMP_MIN,TEMP_MAX);temperature=fitted['applied_temperature']
        require(clamp_temperature(temperature)==temperature,'SDK temperature would be silently clamped')
        threshold=choose_threshold(dev,temperature)
        calibration[name]={'temperature_fit':fitted,'threshold_selection':threshold,
                           'calibration_fitting_sample':{'raw':probability_metrics(labels,diff,1.),'calibrated':probability_metrics(labels,diff,temperature)},
                           'internal_dev_after_temperature_fixed':summarize_prediction_rows(dev,temperature,threshold['threshold']),
                           'warning':'calibration metrics on its own fitting set are not independent evidence'}
        destination=models/(name+'_calibrated')
        artifacts=save_stock_checkpoint(agent,destination,source_root,temperature,copy_original_weights=(name=='base'))
        candidates[name]={'checkpoint_path':str(destination.relative_to(ROOT).as_posix()),'artifacts':artifacts,
                          'temperature':temperature,'threshold':threshold['threshold'],'unmet_fp_scenario':threshold['unmet_fp_scenario'],
                          'selected_training_step':best['step'] if name=='adapted' else 0,
                          'head_changed':name=='adapted','encoder_reference':config['checkpoint'],
                          'questions':config['questions'],'input_contract':config['input'],'class_mapping':{'false':0,'true':1,'yes':'spam'},
                          'not_default':True}
        reloaded=laya.load(str(destination),device='cuda',fast=False)
        require(reloaded.temperature[2]==temperature and not reloaded.temperature_by_options,'Temperature/config roundtrip mismatch')
        require(parameter_hash(reloaded.model,['encoder.'])==frozen_before,'Reload encoder changed')
        reference=dev[:12];reencoded=encoded['dev'][:12]
        new=infer_encoded(reloaded,reencoded,temperature)
        max_diff=max(abs(a['logit_difference']-b['logit_difference']) for a,b in zip(reference,new))
        require(max_diff<=1e-5,'Reload logit mismatch')
        check=synthetic_contract_smoke(reloaded,config)
        reload_checks[name]={'raw_logit_difference_max_error':max_diff,'tolerance':1e-5,'samples':len(new),
                             'temperature_effective':reloaded.temperature[2],'temperature_by_options':reloaded.temperature_by_options,
                             'encoder_unchanged':True,'predictions_match':all((r['probability']>=threshold['threshold'])==
                                 (__import__('day5_common').sigmoid(a['logit_difference']/temperature)>=threshold['threshold']) for a,r in zip(reference,new)),
                             'sdk_mapping_smoke':check,'fit_performed_on_reload':False}
        del reloaded;torch.cuda.empty_cache()
    write_json(out/'reload_checks.json',reload_checks)
    write_json(out/'training.json',training)
    write_json(out/'calibration.json',{'run_id':run_id,'candidates':calibration,'calibration_size':len(inputs['calibration']),
                                     'sdk_temperature_limits':[TEMP_MIN,TEMP_MAX],'validation_not_opened':True})
    manifest={'run_id':run_id,'frozen_utc':now(),'status':training_status,'config':config,'candidates':candidates,
              'membership_sha256':data_manifest['membership_sha256'],'training_input_sha256':data_manifest['selection_sha256'],
              'training_steps':actual_steps,'selected_step':best['step'] if best else None,'versions':versions,
              'source_weights_sha256':metadata['files']['multilingual/model.safetensors']['sha256'],
              'trainable_parameters_file':str((out/'named_parameters.json').relative_to(ROOT).as_posix()),
              'training_manifest_sha256':sha_file(out/'training.json'),'calibration_manifest_sha256':sha_file(out/'calibration.json'),
              'validation_not_opened':True,'default_model':'models/day3/baseline_bundle.joblib'}
    write_json(models/'candidate_manifest.json',manifest)
    write_json(out/'candidate_manifest.json',manifest)
    write_json(out/'train_accesses.json',boundary.report())
    write_json(ROOT/'reports/day5/current_run.json',{'run_id':run_id,'report_path':str(out.relative_to(ROOT).as_posix()),'stage':'candidates_frozen',
                                                   'candidate_manifest_sha256':sha_file(models/'candidate_manifest.json')})
    print('Candidates frozen before any external validation: '+json.dumps({n:{'T':c['temperature'],'threshold':c['threshold']} for n,c in candidates.items()}),flush=True)


if __name__=='__main__':main()
