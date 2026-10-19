"""One outer-validation pass after all new A/B/C manifests are frozen."""
import argparse
import gc
import shutil
import time
from extra_common import *

def main():
    out,models=paths();require(not (out/'validation_started.json').exists(),'Comparison already started; never repeat for tuning')
    manifests={}
    for branch in ['A','B','C']:
        path=out/branch/'candidate_manifest.json';m=read_json(path)
        require(m['validation_not_opened'],'Candidate not frozen before validation')
        require(sha_file(path)==sha_file(models/branch/'candidate_manifest.json'),'Candidate manifest copies differ')
        manifests[branch]=sha_file(path)
    if (out/'D/candidate_manifest.json').exists():manifests['D']=sha_file(out/'D/candidate_manifest.json')
    write_json(out/'validation_started.json',{'started_utc':now(),'manifests_sha256':manifests,
        'protocol_sha256':sha_file(out/'protocol.json'),'code_sha256':sha_file(Path(__file__)),'no_selection_after_validation':True})
    boundary=ExtraBoundary('final').install()
    source=read_json(ROOT/'data/processed/splits/manifest.json');day4=read_json(ROOT/'config/day4_experiments.json')
    tr=[];val=[];fingerprints={}
    for domain in DOMAINS:
        train,tfp=load_split(domain,'train',source,day4);valid,vfp=load_split(domain,'validation',source,day4)
        tr.extend(train);val.extend(valid);fingerprints[domain]={'train':tfp,'validation':vfp}
    keys=set()
    for r in tr:
        keys.update([('group',r['duplicate_group_id']),('canonical',digest(compare_text(r['text'])))])
    require(not any(('group',r['duplicate_group_id']) in keys or ('canonical',digest(compare_text(r['text']))) in keys for r in val),'Outer group/text leakage')
    old=read_rows(DAY5/'private_validation_inputs.jsonl')
    require([{k:r[k] for k in ['record_id','text','spam_label','duplicate_group_id']} for r in old]==
            [{k:r[k] for k in ['record_id','text','spam_label','duplicate_group_id']} for r in val],'Day5 comparison input differs')
    write_rows(out/'C/private_validation_inputs.jsonl',val)
    write_json(out/'outer_data_audit.json',{'fingerprints':fingerprints,'group_canonical_leaks':0,'Day5_input_identity':True,'rows':len(val)})
    del tr,old,keys
    from extra_laya_runtime import torch,encode_rows,infer_encoded
    from evaluate_laya_day5 import quantiles
    import laya
    import psutil
    torch.set_num_threads(1);results={};cached=None
    for branch in ['A','B']:
        manifest=read_json(out/branch/'candidate_manifest.json');candidate=manifest['candidates']['adapted']
        path=ROOT/candidate['checkpoint_path']
        require(sha_file(path/'model.safetensors')==candidate['artifacts']['weights_sha256'],'New Laya weights changed')
        began=time.perf_counter();agent=laya.load(str(path),device='cuda',fast=False);cold=time.perf_counter()-began
        if cached is None:
            cached,visible=encode_rows(agent,val,manifest['config'],out/'private_validation_laya_visible.jsonl')
            require([r['visible_sha256'] for r in visible]==[r['visible_sha256'] for r in read_rows(DAY5/'private_validation_visible.jsonl')],
                    'Laya inputs differ from Day5')
        # Synthetic warmup; no data or thresholds selected here.
        agent.predict('Scheduled notification confirmed.',manifest['config']['questions'],max_len=1024,head_max_len=256)
        torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
        predictions=infer_encoded(agent,cached,temperature=candidate['temperature'],allow_failures=True,
            progress=lambda i,n:print(f'{branch} validation {i}/{n}',flush=True))
        write_rows(out/branch/'validation_predictions.jsonl',predictions)
        results[branch]={'cold_load_seconds':cold,'inference_seconds':time.perf_counter()-began,
            'peak_vram_bytes':torch.cuda.max_memory_allocated(),'rss_bytes':psutil.Process().memory_info().rss,
            'warm_latency_seconds':quantiles([r['backend_seconds'] for r in predictions if r['status']=='ok']),
            'truncation_by_domain':{d:sum(v['token_stats']['truncated'] for v in visible if v['domain']==d) for d in DOMAINS}}
        del agent;gc.collect();torch.cuda.empty_cache()
    require(all(sha_file(out/b/'candidate_manifest.json')==h for b,h in manifests.items()),'Candidate changed after validation')
    write_json(out/'new_laya_resources.json',results);write_json(out/'outer_accesses.json',boundary.report())

if __name__=='__main__':main()
