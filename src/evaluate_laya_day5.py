"""Single final existing-validation pass for fully frozen day-5 Laya candidates."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

from day5_common import (ROOT, DOMAINS, Boundary, read_json, read_rows, write_json, write_rows,
                         sha_file, load_split, digest, compare_text, require, sigmoid, metrics, probability_metrics, now)


def quantiles(values):
    vals=sorted(values)
    def at(p):
        j=(len(vals)-1)*p;a=int(j);b=min(a+1,len(vals)-1)
        return vals[a]+(vals[b]-vals[a])*(j-a)
    return {'n':len(vals),'p50':at(.5),'p95':at(.95),'max':vals[-1]} if vals else {'n':0}


def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id');args=parser.parse_args()
    current=read_json(ROOT/'reports/day5/current_run.json');run_id=args.run_id or current['run_id']
    out=ROOT/'reports/day5/runs'/run_id;models=ROOT/'models/day5/runs'/run_id
    path=models/'candidate_manifest.json';manifest=read_json(path)
    require(sha_file(path)==sha_file(out/'candidate_manifest.json'),'Frozen manifest copies differ')
    require(manifest['validation_not_opened'] and manifest['candidates'],'No candidates frozen before validation')
    require(not (out/'validation_started.json').exists(),'Final evaluation already started. Do not repeat validation for tuning.')
    frozen_hash=sha_file(path)
    write_json(out/'validation_started.json',{'started_utc':now(),'candidate_manifest_sha256':frozen_hash,
                'question':manifest['config']['questions'],'temperature_and_threshold_frozen':True,
                'code_sha256':{p:sha_file(ROOT/p) for p in ['src/evaluate_laya_day5.py','src/day5_laya_runtime.py','src/day5_common.py']}})
    boundary=Boundary('final').install()
    source=read_json(ROOT/'data/processed/splits/manifest.json');day4=read_json(ROOT/'config/day4_experiments.json')
    train=[];validation=[];fingerprints={}
    for domain in DOMAINS:
        tr,trfp=load_split(domain,'train',source,day4);val,valfp=load_split(domain,'validation',source,day4)
        train.extend(tr);validation.extend(val);fingerprints[domain]={'train':trfp,'validation':valfp}
    train_keys={};leaks=[]
    for r in train:
        for key in [('group',r['duplicate_group_id']),('clean_exact',digest(r['text'])),('clean_canonical',digest(compare_text(r['text']))),
                    ('original_exact',digest(r['original_model_text'])),('original_canonical',digest(compare_text(r['original_model_text'])))]:
            train_keys[key]=r['record_id']
    for r in validation:
        for key in [('group',r['duplicate_group_id']),('clean_exact',digest(r['text'])),('clean_canonical',digest(compare_text(r['text']))),
                    ('original_exact',digest(r['original_model_text'])),('original_canonical',digest(compare_text(r['original_model_text'])))]:
            if key in train_keys:leaks.append({'type':key[0],'train':train_keys[key],'validation':r['record_id']})
    require(not leaks,'Allowed train/validation leakage after fixed cleaning')
    write_json(out/'final_data_audit.json',{'input_fingerprints':fingerprints,'global_group_exact_canonical_leaks':leaks,
                                         'candidate_frozen_before_validation':True})
    write_rows(out/'private_validation_inputs.jsonl',validation)
    del train,train_keys
    from day5_laya_runtime import torch,encode_rows,infer_encoded,synthetic_contract_smoke,parameter_hash
    import laya
    import psutil
    torch.set_num_threads(1)
    config=manifest['config'];results={};cached=None;visible=None
    for name,candidate in manifest['candidates'].items():
        ckpt=ROOT/candidate['checkpoint_path']
        require(sha_file(ckpt/'model.safetensors')==candidate['artifacts']['weights_sha256'],'Candidate weight hash changed')
        require(sha_file(ckpt/'rl_agent_config.json')==candidate['artifacts']['config_sha256'],'Candidate temperature/config changed')
        start=time.perf_counter();agent=laya.load(str(ckpt),device='cuda',fast=False)
        if agent.device.type=='cuda':torch.cuda.synchronize()
        cold=time.perf_counter()-start
        require(agent.temperature[2]==candidate['temperature'] and not agent.temperature_by_options,'Candidate temperature not effective')
        if cached is None:
            prep=time.perf_counter();cached,visible=encode_rows(agent,validation,config,out/'private_validation_visible.jsonl')
            preparation=time.perf_counter()-prep
            # Audit actual visible text vs selected fit/dev/calibration, without altering rules after validation.
            prior={}
            for part in ['fit','dev','calibration']:
                for r in read_rows(out/f'private_{part}_visible.jsonl'):
                    prior[digest(compare_text(r['visible_fragment']))]=(part,r['record_id'])
            collisions=[{'validation_id':r['record_id'],'prior_part':prior[digest(compare_text(r['visible_fragment']))][0],
                         'prior_id':prior[digest(compare_text(r['visible_fragment']))][1]}
                        for r in visible if digest(compare_text(r['visible_fragment'])) in prior]
            write_json(out/'validation_visible_audit.json',{'collisions':collisions,'count':len(collisions),
                        'policy':'report only; do not delete or change frozen candidates after validation'})
        else:preparation=0.
        warm=synthetic_contract_smoke(agent,config)
        if agent.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        print('Final validation '+name+' (fixed candidates; no selection)',flush=True)
        start=time.perf_counter()
        predictions=infer_encoded(agent,cached,progress=lambda i,n:print(f'{name}: {i}/{n}',flush=True),allow_failures=True)
        elapsed=time.perf_counter()-start
        write_rows(out/f'{name}_validation_predictions.jsonl',predictions)
        result={'by_domain':{},'cold_model_load_seconds':cold,'preparation_seconds':preparation,
                'inference_wall_seconds':elapsed,'real_device':str(agent.device),
                'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                'rss_bytes_after_inference':psutil.Process().memory_info().rss,'synthetic_smoke':warm}
        for domain in DOMAINS:
            allrows=[r for r in predictions if r['domain']==domain];ok=[r for r in allrows if r['status']=='ok']
            y=[r['spam_label'] for r in ok];diff=[r['logit_difference'] for r in ok]
            trunc=sum(r['token_stats']['truncated'] for r in visible if r['domain']==domain)
            modes={}
            for mode,t,threshold in [('raw_0.5',1.,.5),('calibrated_0.5',candidate['temperature'],.5),
                                      ('calibrated_dev_threshold',candidate['temperature'],candidate['threshold'])]:
                probs=[sigmoid(x/t) for x in diff]
                modes[mode]={'metrics':metrics(y,probs,threshold),'probability_metrics':probability_metrics(y,diff,t)} if ok else {'metrics':None}
            result['by_domain'][domain]={'n':len(allrows),'successful':len(ok),'coverage':len(ok)/len(allrows),
                'failures':[{'record_id':r['record_id'],'error':r['error']} for r in allrows if r['status']!='ok'],
                'metrics_conditional_on_success':len(ok)!=len(allrows),'modes':modes,'truncated':trunc,
                'truncated_fraction':trunc/len(allrows),'warm_latency_seconds':quantiles([r['backend_seconds'] for r in ok])}
        results[name]=result
        write_json(out/'laya_validation_comparison.json',{'run_id':run_id,'candidates':results,'validation_is_previously_used':True})
        del agent;gc.collect();torch.cuda.empty_cache()
    require(sha_file(path)==frozen_hash,'Frozen candidate manifest changed during validation')
    write_json(out/'final_accesses.json',boundary.report())
    write_json(ROOT/'reports/day5/current_run.json',{'run_id':run_id,'report_path':str(out.relative_to(ROOT).as_posix()),
               'stage':'laya_validation_completed','candidate_manifest_sha256':frozen_hash})
    print(json.dumps({n:{d:r['modes']['calibrated_dev_threshold']['metrics'] for d,r in v['by_domain'].items()} for n,v in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
