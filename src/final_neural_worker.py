"""Saved Laya/C inference only, without access to raw splits, archive or ground truth."""
import argparse
import importlib.metadata
import time
from final_eval_common import *

def neutral_prediction(r,c,status='ok',**kwargs):
    return {'record_id':r['record_id'],'model_id':c['name'],'status':status,'decision':None,'score':None,
        'threshold':c['threshold'],'temperature':c['temperature'],'score_semantics':c['score_semantics'],**kwargs}

def check_candidate(c):
    for p,h in c['files'].items():require(sha_file(ROOT/p)==h,'Saved component file changed: '+p)
    for name,v in c['versions'].items():require(importlib.metadata.version(name)==v,'Version changed: '+name)

def laya(out,c,data):
    from day5_laya_runtime import torch,encode_rows,infer_encoded
    from decision_protocol import validate_noul
    import laya as sdk
    torch.set_num_threads(1);check_candidate(c);require(torch.cuda.is_available(),'CUDA unavailable')
    started=time.perf_counter();agent=sdk.load(str(ROOT/c['checkpoint_path']),device='cuda',fast=False)
    agent.model.eval()
    for p in agent.model.parameters():p.requires_grad_(False)
    require(abs(agent.temperature[2]-c['temperature'])<1e-12 and not agent.temperature_by_options,'Laya calibration differs')
    resources={'load_seconds':time.perf_counter()-started,'device':str(agent.device),'dtype':str(agent.dtype),
        'amp_enabled':agent.amp_enabled,'batch_size':1};output=[];smoke=[]
    torch.cuda.reset_peak_memory_stats();config={'questions':c['questions'],'input':c['input_contract']}
    for i,r in enumerate(data):
        if not r['text'].strip():output.append(neutral_prediction(r,c,'insufficient_features',reason='empty_model_text'));continue
        try:
            encoded,visible=encode_rows(agent,[r],config)
            raw=infer_encoded(agent,encoded,c['temperature'])[0];v=visible[0]
            require(raw['status']=='ok','Laya forward failed')
            # Only the first three verified SDK decoder comparisons, no need to duplicate
            # the expensive actual forward for every test message.
            if len(smoke)<3:
                response=agent.predict(v['visible_fragment'],c['questions'],max_len=1024,head_max_len=256)
                error=abs(validate_noul(response)-raw['probability']);require(error<=.000051,'SDK mapping differs')
                smoke.append({'record_id':r['record_id'],'raw_response':response,'probability':raw['probability'],'error':error})
            output.append(neutral_prediction(r,c,score=raw['probability'],decision=int(raw['probability']>=c['threshold']),
                logit_difference=raw['logit_difference'],logits_false_true=raw['logits_false_true'],
                token_stats=v['token_stats'],sequence_tokens=v['sequence_tokens'],
                actual_input_ids_sha256=digest(encoded[0][1]['ids']),
                visible_sha256=v['visible_sha256'],backend_seconds=raw['backend_seconds'],real_device=raw['real_device']))
        except Exception as e:output.append(neutral_prediction(r,c,'error',error_type=type(e).__name__,private_error=str(e)))
        if (i+1)%250==0:
            write_json(out/'worker_progress.json',{'model':c['name'],'processed':i+1,'total':len(data),'utc':now()});print(c['name'],i+1,'/',len(data),flush=True)
    resources.update(peak_vram_bytes=torch.cuda.max_memory_allocated(),model_training=agent.model.training,
        frozen_parameters=all(not p.requires_grad for p in agent.model.parameters()),sdk_decoder_checks=smoke)
    return output,resources

def e5(out,c,data):
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    import torch
    import joblib
    from threadpoolctl import threadpool_limits,threadpool_info
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(1);check_candidate(c);require(torch.cuda.is_available(),'CUDA unavailable')
    m=c['saved_manifest'];began=time.perf_counter()
    clf=joblib.load(ROOT/m['classifier_path']);require(list(clf.classes_)==[0,1],'C class direction differs')
    tok=AutoTokenizer.from_pretrained(ROOT/m['tokenizer_path'],local_files_only=True,trust_remote_code=False)
    enc=AutoModel.from_pretrained(m['encoder_reference'],local_files_only=True,trust_remote_code=False).to('cuda').eval()
    for p in enc.parameters():p.requires_grad_(False)
    resources={'load_seconds':time.perf_counter()-began,'device':'CUDA encoder+CPU sklearn','dtype':'float32','batch_size':16}
    output=[];torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(limits=1):
        resources['threadpools']=threadpool_info()
        for start in range(0,len(data),16):
            chunk=data[start:start+16];ok=[r for r in chunk if r['text'].strip()]
            for r in chunk:
                if not r['text'].strip():output.append(neutral_prediction(r,c,'insufficient_features',reason='empty_model_text'))
            if not ok:continue
            try:
                began=time.perf_counter();tokens=tok(['query: '+r['text'] for r in ok],truncation=True,max_length=512,padding=True,return_tensors='pt')
                actual=[ids[mask.bool()].tolist() for ids,mask in zip(tokens['input_ids'],tokens['attention_mask'])]
                tokens=tokens.to('cuda')
                with torch.no_grad():
                    hidden=enc(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1)
                    pooled=(hidden*mask).sum(1)/mask.sum(1)
                    vectors=torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy()
                logits=clf.decision_function(vectors);elapsed=time.perf_counter()-began
                for r,z,ids in zip(ok,logits,actual):
                    full=tok('query: '+r['text'],truncation=False)['input_ids'];diff=float(z);p=sigmoid(diff/c['temperature'])
                    output.append(neutral_prediction(r,c,score=p,decision=int(p>=c['threshold']),logit_difference=diff,
                        actual_input_ids_sha256=digest(ids),sequence_tokens=len(ids),
                        token_stats={'truncated':len(full)>len(ids),'original_tokens':len(full),'visible_tokens':len(ids),
                            'tokens_include_prefix_and_specials':True},batch_forward_seconds=elapsed))
            except Exception as e:
                output.extend(neutral_prediction(r,c,'error',error_type=type(e).__name__,private_error=str(e)) for r in ok)
            if start%512==0:
                write_json(out/'worker_progress.json',{'model':c['name'],'processed':min(start+16,len(data)),'total':len(data),'utc':now()})
                print(c['name'],min(start+16,len(data)),'/',len(data),flush=True)
    resources.update(peak_vram_bytes=torch.cuda.max_memory_allocated(),model_training=enc.training,frozen_parameters=True)
    return output,resources

def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--dataset',required=True);p.add_argument('--model',required=True);a=p.parse_args()
    out=a.out.resolve();require(out.is_relative_to(BASE/'runs'),'Outside final run')
    boundary=FinalBoundary(out).install() # remains validation phase: inference cannot open ANY raw test
    from personal_eval_common import setup_offline
    setup_offline();protocol=read_json(out/'FINAL_EVALUATION_PROTOCOL.json');lock=read_json(out/'protocol_lock.json')
    require(sha_file(out/'FINAL_EVALUATION_PROTOCOL.json')==lock['sha256'],'Protocol changed')
    require(a.dataset in TEST_SETS and a.model in COMPONENTS[1:],'Unknown worker mode')
    c=next(x for x in protocol['components'] if x['name']==a.model)
    data=read_rows(out/f'private_inputs_{a.dataset}.jsonl');spec=read_json(out/'test_data_manifest.json')['datasets'][a.dataset]
    require(sha_file(out/f'private_inputs_{a.dataset}.jsonl')==spec['inference_inputs_sha256'],'Inputs changed')
    require(all(r['spam_label']==0 and r['source']=='evaluation_text' and not any(k in r for k in ['local_path','original_locator','member_path','ground_truth']) for r in data),'Real labels/paths in model adapter')
    destination=out/f'component_{a.dataset}_{a.model}.jsonl';require(not destination.exists(),'Do not silently repeat completed test inference')
    rows,resources=(laya if a.model=='Laya_day5' else e5)(out,c,data)
    require(len(rows)==len(data) and {r['record_id'] for r in rows}=={r['record_id'] for r in data},'Worker prediction count/IDs differ')
    write_rows(destination,rows);write_json(out/f'resources_{a.dataset}_{a.model}.json',{'status':'completed','resources':resources,
        'successful':sum(component_ok(r) for r in rows),'records':len(rows),'no_ground_truth_in_forward':True,
        'boundary':boundary.report(),'versions':{n:importlib.metadata.version(n) for n in c['versions']}})
    print(a.dataset,a.model,'completed',sum(component_ok(r) for r in rows),'/',len(rows),flush=True)

if __name__=='__main__':main()
