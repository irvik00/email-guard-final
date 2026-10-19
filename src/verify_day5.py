"""Independent day-5 audit: stored membership/logits/metrics/weights; no fit, no closed files."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from day5_common import (ROOT,DOMAINS,Boundary,read_json,read_rows,write_json,sha_file,require,
                         digest,compare_text,metrics,sigmoid,probability_metrics)


def archive_training_runtime(out):
    expected=read_json(out/'training_started.json')['code_sha256']['src/day5_laya_runtime.py']
    text=(ROOT/'src/day5_laya_runtime.py').read_text(encoding='utf-8')
    current=hashlib.sha256(text.encode()).hexdigest()
    if current!=expected:
        # Preserve the exact executed training source; final evaluation later added error coverage.
        old='''def infer_encoded(agent, encoded, temperature=1., progress=None):
    agent.model.eval();rows=[]
    with torch.no_grad():
        for i,(record,item) in enumerate(encoded):
            b=batch_tensors(agent,[(record,item)])
            start=time.perf_counter()
            logits,act=agent._forward(b)
            if agent.device.type=='cuda':torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            diff=float(logits[0,1]-logits[0,0]);p=sigmoid(diff/temperature)
            rows.append({k:record[k] for k in ['record_id','domain','source','spam_label','duplicate_group_id']}
                        | {'logits_false_true':[float(x) for x in logits[0,:2]],'logit_difference':diff,
                           'raw_probability':sigmoid(diff),'probability':p,'sdk_rounded_noul':round(p,4),
                           'backend_seconds':seconds,'status':'ok'})
            if progress and (i+1)%250==0:progress(i+1,len(encoded))
    return rows


'''
        a=text.index('def infer_encoded(');b=text.index('def summarize_prediction_rows(',a)
        text=text[:a]+old+text[b:]
    require(hashlib.sha256(text.encode()).hexdigest()==expected,'Cannot recover exact training source snapshot')
    path=out/'code_snapshots/day5_laya_runtime_training.py.txt';path.parent.mkdir(exist_ok=True)
    path.write_text(text,encoding='utf-8',newline='\n')
    write_json(out/'code_history.json',{'training_runtime_sha256':expected,'current_runtime_sha256':current,
               'snapshot':str(path.relative_to(ROOT).as_posix()),'change':'After training: add final-inference error coverage and real_device metadata; training forward/optimizer/input math unchanged.' if current!=expected else 'unchanged'})


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    boundary=Boundary('verify').install()
    run_id=read_json(ROOT/'reports/day5/current_run.json')['run_id'];out=ROOT/'reports/day5/runs'/run_id
    started=time.perf_counter();training=read_json(out/'training.json');data=read_json(out/'data_manifest.json')
    candidates=read_json(out/'candidate_manifest.json');models=ROOT/'models/day5/runs'/run_id
    require(sha_file(models/'candidate_manifest.json')==sha_file(out/'candidate_manifest.json'),'Candidate copies mismatch')
    freeze=read_json(out/'validation_started.json')
    require(freeze['candidate_manifest_sha256']==sha_file(models/'candidate_manifest.json'),'Manifest not preserved after validation')
    archive_training_runtime(out)
    before=read_json(out/'preservation_before.json')
    for rel,spec in before.items():require(sha_file(ROOT/rel)==spec['sha256'],'Day3/4/diagnostic artifact changed: '+rel)
    membership=read_rows(out/'membership.jsonl');require(sha_file(out/'membership.jsonl')==data['membership_sha256'],'Membership changed')
    require(len(membership)==33906,'Incomplete full train membership')
    group_roles={};text_roles={}
    for r in membership:
        for key in [('global',r['duplicate_group_id']),('inner',r['inner_group_id'])]:
            require(key not in group_roles or group_roles[key]==r['inner_part'],'Inner group leakage')
            group_roles[key]=r['inner_part']
        for key in [('clean',r['clean_text_sha256']),('original',r['original_text_sha256'])]:
            require(key not in text_roles or text_roles[key]==r['inner_part'],'Inner exact leakage')
            text_roles[key]=r['inner_part']
    fit=read_rows(out/'private_fit_inputs.jsonl');fit_ids={r['record_id'] for r in fit}
    used=read_rows(out/'actual_training_membership.jsonl')
    require(all(r['record_id'] in fit_ids for r in used),'Fit outside selected fit')
    require(len(used)==training['examples_presented'] and len({r['record_id'] for r in used})==training['unique_records_used'],'Used-record counts mismatch')
    for p,h in data['selection_sha256'].items():require(sha_file(out/f'private_{p}_inputs.jsonl')==h,'Selected input changed')
    steps=read_rows(out/'training_steps.jsonl')
    require([r['step'] for r in steps]==list(range(1,training['actual_optimizer_steps']+1)),'Training step gaps')
    require(all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm_before_clip']) and r['gradient_norm_before_clip']>0 for r in steps),'Invalid real gradients/loss')
    require(training['encoder_sha256_before']==training['encoder_sha256_after'],'Encoder changed')
    require(training['head_sha256_before']!=training['head_sha256_selected'],'No head adaptation')
    require(training['act_head_sha256_before']==training['act_head_sha256_after'],'Action branch changed')
    # Actual visible exact/canonical collisions across inner selected roles.
    owners={};inner_collisions=[]
    for part in ['fit','dev','calibration']:
        for r in read_rows(out/f'private_{part}_visible.jsonl'):
            require(hashlib.sha256(r['visible_fragment'].encode()).hexdigest()==r['visible_sha256'],'Visible hash mismatch')
            require(r['sequence_tokens']<=1024 and r['token_stats']['visible_tokens']<=700,'Input cap violated')
            for kind,text in [('exact',r['visible_fragment']),('canonical',compare_text(r['visible_fragment']))]:
                key=(kind,digest(text))
                if key in owners and owners[key][0]!=part:inner_collisions.append([owners[key],[part,r['record_id']]])
                owners[key]=(part,r['record_id'])
    require(not inner_collisions,'Visible inner leakage')
    final=read_json(out/'laya_validation_comparison.json');cal=read_json(out/'calibration.json')
    invariance={};score_checks={}
    for name,candidate in candidates['candidates'].items():
        predictions=read_rows(out/f'{name}_validation_predictions.jsonl')
        for domain in DOMAINS:
            rows=[r for r in predictions if r['domain']==domain and r['status']=='ok'];y=[r['spam_label'] for r in rows]
            for r in rows:
                delta=r['logits_false_true'][1]-r['logits_false_true'][0]
                require(abs(delta-r['logit_difference'])<1e-5,'Saved logit difference mismatch')
                require(abs(sigmoid(r['logit_difference'])-r['raw_probability'])<1e-12,'Raw mapping mismatch')
            modes=final['candidates'][name]['by_domain'][domain]['modes']
            for mode,t,th in [('raw_0.5',1.,.5),('calibrated_0.5',candidate['temperature'],.5),
                              ('calibrated_dev_threshold',candidate['temperature'],candidate['threshold'])]:
                expected=metrics(y,[sigmoid(r['logit_difference']/t) for r in rows],th)
                require(expected==modes[mode]['metrics'],'Final confusion mismatch')
                calculated=probability_metrics(y,[r['logit_difference'] for r in rows],t)
                require(abs(calculated['NLL']-modes[mode]['probability_metrics']['NLL'])<1e-12,'NLL mismatch')
                require(sum(b['count'] for b in calculated['ECE_bins'])==len(rows),'ECE count mismatch')
            invariance[name+'/'+domain]=modes['raw_0.5']['metrics']==modes['calibrated_0.5']['metrics']
            require(invariance[name+'/'+domain],'Temperature changed half-threshold predictions')
        cfg=read_json(ROOT/candidate['checkpoint_path']/'rl_agent_config.json')
        require(cfg['temperature'][2]==candidate['temperature'] and not cfg['temperature_by_options'],'Persisted T not applicable')
        score_checks[name]=len(predictions)
    # Compare ALL source/adapted tensor values independently of the training process.
    import torch
    from safetensors import safe_open
    source=ROOT/'.cache/day4/laya_weights/multilingual/model.safetensors'
    weight_audit={}
    for name,candidate in candidates['candidates'].items():
        target=ROOT/candidate['checkpoint_path']/'model.safetensors'
        require(sha_file(target)==candidate['artifacts']['weights_sha256'],'Saved weight file changed')
        frozen_checked=changed_head=total=0
        with safe_open(str(source),framework='pt',device='cpu') as original,safe_open(str(target),framework='pt',device='cpu') as saved:
            require(set(original.keys())==set(saved.keys()),'Architecture/state keys changed')
            for key in original.keys():
                a=original.get_tensor(key).float();b=saved.get_tensor(key).float()
                require(a.shape==b.shape,'Tensor shape changed')
                equal=torch.equal(a,b);total+=1
                if key.startswith(('encoder.','act_head.')):
                    require(equal,'Frozen tensor changed: '+key);frozen_checked+=1
                elif not equal:
                    require(key.startswith(tuple(candidates['config']['training']['trainable_prefixes'])),'Unexpected changed tensor: '+key)
                    changed_head+=1
                del a,b
        require((changed_head>0)==(name=='adapted'),'Adaptation status does not match weights')
        weight_audit[name]={'all_tensor_keys_shapes_checked':total,'unchanged_frozen_tensors':frozen_checked,'changed_trainable_tensors':changed_head}
    logs={}
    for filename in ['prepare_accesses.json','train_accesses.json','final_accesses.json']:
        access=read_json(out/filename);require(not access['denied'],'Unexpected prohibited access attempt')
        for key in access['accesses']:
            if key.startswith('data/'):
                require(key.rsplit(':',1)[0] in access['allowed_data_paths'],'Data access escaped allowlist')
        logs[filename]={'phase':access['phase'],'data_opens':{k:v for k,v in access['accesses'].items() if k.startswith('data/')}}
    lr=read_json(out/'lr_validation_comparison.json');require(not lr['fit_performed'],'LR refit performed')
    fragments=read_rows(out/'private_validation_visible.jsonl');hashes={r['record_id']:r['visible_sha256'] for r in fragments}
    for name in ['E0','E2']:
        matched=read_rows(out/f'{name}_matched_visible_validation_predictions.jsonl')
        require(all(r['text_sha256']==hashes[r['record_id']] for r in matched),'LR matched text differs from Laya')
        for context in ['full_context','matched_visible']:
            predictions=read_rows(out/f'{name}_{context}_validation_predictions.jsonl')
            for domain in DOMAINS:
                rows=[r for r in predictions if r['domain']==domain]
                result=lr['candidates'][name+'/'+context][domain]
                for threshold,stored in result['metrics'].items():
                    require(metrics([r['spam_label'] for r in rows],[r['probability'] for r in rows],float(threshold))==stored,'Frozen LR confusion mismatch')
    old_e0=read_rows(ROOT/'reports/day3/predictions/char_lr_validation.jsonl')
    new_e0=[r for r in read_rows(out/'E0_full_context_validation_predictions.jsonl') if r['domain']=='spam_email']
    require(len(old_e0)==len(new_e0)==1275,'Original E0 prediction coverage mismatch')
    require(all(a['record_id']==b['record_id'] and abs(a['spam_score']-b['probability'])<=1e-12 for a,b in zip(old_e0,new_e0)),'E0 differs from day3 saved predictions')
    result={'run_id':run_id,'status':'passed','protected_files_identical':len(before),
            'membership_rows':len(membership),'optimizer_steps_checked':len(steps),'actual_fit_records':len(set(r['record_id'] for r in used)),
            'inner_visible_collisions':len(inner_collisions),'temperature_half_prediction_invariance':invariance,
            'raw_logit_mapping_checked':score_checks,'weight_tensor_audit':weight_audit,
            'access_audit':logs,'no_closed_data_opened':True,'no_fit_performed':True,
            'all_frozen_lr_confusions_recomputed':True,'E0_full_context_matches_day3_saved_predictions':True,
            'audit_seconds':time.perf_counter()-started}
    write_json(out/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
