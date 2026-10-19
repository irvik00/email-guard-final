"""Real end-to-end frozen E5+sklearn latency on64 inner-dev records; no fit or outer evaluation."""
import os
import sys
import time
from extra_common import *

def main():
    out,_=paths();boundary=ExtraBoundary('train').install()
    os.environ.update(HF_HOME=str(ROOT/'.cache/extra/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',USE_TF='0',TOKENIZERS_PARALLELISM='false')
    # Reuse the already installed primary sklearn stack read-only. Nothing is installed or
    # updated in either environment; assert the exact versions used by the fitted classifier.
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    import importlib.metadata
    import torch
    import joblib
    from transformers import AutoTokenizer,AutoModel
    manifest=read_json(out/'C/candidate_manifest.json')
    versions={n:importlib.metadata.version(n) for n in manifest['versions']}
    require(versions==manifest['versions'],'Read-only sklearn dependency versions differ')
    require(sha_file(ROOT/manifest['classifier_path'])==manifest['classifier_sha256'],'C classifier changed')
    classifier=joblib.load(ROOT/manifest['classifier_path']);require(list(classifier.classes_)==[0,1],'C mapping changed')
    source=manifest['source']
    for name,spec in source['files'].items():require(sha_file(Path(source['local_path'])/name)==spec['sha256'],'E5 source changed')
    torch.set_num_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/manifest['tokenizer_path'],local_files_only=True,trust_remote_code=False)
    encoder=AutoModel.from_pretrained(source['local_path'],local_files_only=True,trust_remote_code=False).to('cuda').eval()
    for p in encoder.parameters():p.requires_grad_(False)
    def predict(text):
        tokens=tokenizer('query: '+text,truncation=True,max_length=512,padding=True,return_tensors='pt').to('cuda')
        with torch.no_grad():
            hidden=encoder(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1)
            vector=torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy()
        logit=float(classifier.decision_function(vector)[0]);probability=sigmoid(logit/manifest['temperature'])
        return logit,probability,probability>=manifest['threshold']
    data=sorted(read_rows(out/'C/private_dev_inputs.jsonl'),key=lambda r:digest([42,'latency',r['record_id']]))[:64]
    old={r['record_id']:r for r in read_rows(out/'C/dev_raw.jsonl')};predict('Synthetic warm-up notification.')
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();results=[];borderline=[]
    for r in data:
        began=time.perf_counter();logit,p,decision=predict(r['text']);torch.cuda.synchronize();seconds=time.perf_counter()-began
        error=abs(logit-old[r['record_id']]['logit_difference']);require(error<=1e-4,'E5 full prediction differs beyond float32 tolerance')
        prior=sigmoid(old[r['record_id']]['logit_difference']/manifest['temperature'])
        if decision!=(prior>=manifest['threshold']):
            require(abs(prior-manifest['threshold'])<=1e-5,'Non-boundary class change on E5 reload')
            borderline.append(r['record_id'])
        results.append({'record_id':r['record_id'],'domain':r['domain'],'seconds':seconds,'raw_logit_abs_error':error})
    write_json(out/'C/warm_end_to_end.json',{'status':'completed','rows':results,'samples':64,'outer_validation_opened':False,
        'includes':'tokenization+CUDA encoder+masked mean+L2+CPU copy+actual sklearn LR+T+threshold',
        'primary_dependencies_reused_read_only':True,'versions':versions,'fit_performed':False,
        'near_threshold_float32_changes':borderline,'max_logit_error':max(r['raw_logit_abs_error'] for r in results),
        'peak_vram_bytes':torch.cuda.max_memory_allocated(),'accesses':boundary.report()})
    print('Real E5+sklearn end-to-end warm latency measured64 inner-dev records; boundary changes',len(borderline))

if __name__=='__main__':main()
