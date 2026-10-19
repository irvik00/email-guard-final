"""One saved candidate per offline process. No dataset access, no training."""
import argparse
import importlib.metadata
import time
from personal_eval_common import *

def validate_files(candidate):
    for rel,spec in candidate['files'].items():
        require(sha_file(ROOT/rel)==spec,'Frozen candidate file differs: '+rel)
    for n,v in candidate['versions'].items():require(importlib.metadata.version(n)==v,'Version differs: '+n)

def laya_predictions(out,candidate,records):
    from day5_laya_runtime import torch,encode_rows,infer_encoded
    from decision_protocol import token_prefix,validate_noul
    from laya.common import build_sequence,QTYPES,render_options
    import laya
    torch.set_num_threads(1)
    validate_files(candidate);require(torch.cuda.is_available(),'CUDA unavailable; no automatic hardware substitution')
    began=time.perf_counter();agent=laya.load(str(ROOT/candidate['checkpoint_path']),device='cuda',fast=False)
    agent.model.eval()
    for p in agent.model.parameters():p.requires_grad_(False)
    require(not agent.temperature_by_options,'Unexpected per-options calibration')
    require(abs(agent.temperature[QTYPES['noul']]-candidate['temperature'])<1e-12,'Stored temperature not effective')
    load_seconds=time.perf_counter()-began;output=[]
    questions=candidate['questions'];internal=agent._to_internal(questions['spam']);opts=render_options(internal)
    require(opts[0].startswith('false:') and opts[1].startswith('true:'),'Wrong option mapping')
    limit=candidate['input_contract'];empty,_=build_sequence(agent.tok,'',internal,limit['context_limit'],limit['head_max_len'])
    cap=min(limit['state_token_cap'],limit['context_limit']-len(empty));require(cap==700,'Unexpected state room change')
    torch.cuda.reset_peak_memory_stats()
    for r in records:
        started=time.perf_counter()
        if not r['text'].strip():
            output.append(prediction_row(r,candidate,'insufficient_features',insufficient_features=True));continue
        try:
            if candidate['name']!='Laya_question_A':
                encoded,visible=encode_rows(agent,[r],{'questions':questions,'input':limit})
                raw=infer_encoded(agent,encoded,candidate['temperature'])[0];v=visible[0]
                require(raw['status']=='ok','Laya inference did not succeed')
                p=raw['probability'];raw_response=agent.predict(v['visible_fragment'],questions,max_len=limit['context_limit'],head_max_len=limit['head_max_len'])
                require(abs(validate_noul(raw_response)-p)<=.000051,'SDK raw response differs from option logits')
                tokens=v['token_stats'];seq,marks=encoded[0][1]['ids'],encoded[0][1]['markers']
                diff=raw['logit_difference'];logits=raw['logits_false_true'];literal=v['mask_literal_replacements']
            else:
                # Long saved Day4 question + criteria, kept separate from the training question.
                sanitized=r['text'].replace(agent.tok.mask_token,' ');literal=r['text'].count(agent.tok.mask_token)
                fragment,tokens=token_prefix(agent.tok,sanitized,cap)
                seq,marks=build_sequence(agent.tok,fragment,internal,limit['context_limit'],limit['head_max_len'])
                actual=agent.tok(fragment,add_special_tokens=False)['input_ids']
                require(len(seq)==len(empty)+len(actual),'Unexpected internal truncation')
                require(sum(t==agent.tok.mask_token_id for t in seq)==2,'Literal mask reaches service positions')
                captured={};forward=agent._forward
                def capture(batch):
                    zs,acts=forward(batch);captured['logits']=zs[0,:2].tolist();return zs,acts
                agent._forward=capture
                try:raw_response=agent.predict(fragment,questions,max_len=limit['context_limit'],head_max_len=limit['head_max_len'])
                finally:agent._forward=forward
                logits=captured['logits'];diff=logits[1]-logits[0];p=sigmoid(diff/candidate['temperature'])
                require(abs(validate_noul(raw_response)-p)<=.000051,'Long question mapping mismatch')
            torch.cuda.synchronize()
            output.append(prediction_row(r,candidate,score=p,decision=int(p>=candidate['threshold']),raw_response=raw_response,
                raw_logits_false_true=logits,logit_difference=diff,token_stats=tokens,sequence_tokens=len(seq),
                actual_input_ids_sha256=digest(seq),mask_literal_replacements=literal,
                backend_total_seconds=time.perf_counter()-started,real_device=str(agent.device)))
        except Exception as error:
            output.append(prediction_row(r,candidate,'error',error_type=type(error).__name__,private_error=str(error)))
    resources={'load_seconds':load_seconds,'device':str(agent.device),'dtype':str(agent.dtype),
        'amp_enabled':agent.amp_enabled,'model_training':agent.model.training,
        'peak_vram_bytes':torch.cuda.max_memory_allocated(),'classes':'false0 true1, yes->spam',
        'frozen_parameters':all(not p.requires_grad for p in agent.model.parameters())}
    return output,resources

def e5_predictions(out,candidate,records):
    # Exact installed primary sklearn stack is reused read-only, as in the saved C experiment.
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    import torch
    import joblib
    from threadpoolctl import threadpool_limits,threadpool_info
    from transformers import AutoModel,AutoTokenizer
    validate_files(candidate);torch.set_num_threads(1)
    require(torch.cuda.is_available(),'CUDA unavailable; no automatic hardware substitution')
    began=time.perf_counter();manifest=candidate['saved_manifest']
    classifier=joblib.load(ROOT/manifest['classifier_path']);require(list(classifier.classes_)==[0,1],'C class mapping differs')
    tok=AutoTokenizer.from_pretrained(ROOT/manifest['tokenizer_path'],local_files_only=True,trust_remote_code=False)
    encoder=AutoModel.from_pretrained(manifest['encoder_reference'],local_files_only=True,trust_remote_code=False).to('cuda').eval()
    for p in encoder.parameters():p.requires_grad_(False)
    load=time.perf_counter()-began;output=[];torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(limits=1):
        pools=threadpool_info()
        for r in records:
            started=time.perf_counter()
            if not r['text'].strip():output.append(prediction_row(r,candidate,'insufficient_features',insufficient_features=True));continue
            try:
                full=tok('query: '+r['text'],return_offsets_mapping=True,truncation=False)
                capped=tok('query: '+r['text'],return_offsets_mapping=True,truncation=True,max_length=512)
                ids=capped['input_ids'];end=max(b for a,b in capped['offset_mapping']);truncated=len(full['input_ids'])>len(ids)
                tokens=tok('query: '+r['text'],truncation=True,max_length=512,padding=True,return_tensors='pt').to('cuda')
                require(tokens['input_ids'][0].tolist()==ids,'C actual IDs differ from diagnostics')
                with torch.no_grad():
                    hidden=encoder(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1)
                    pooled=(hidden*mask).sum(1)/mask.sum(1);vector=torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy()
                logit=float(classifier.decision_function(vector)[0]);p=sigmoid(logit/candidate['temperature'])
                torch.cuda.synchronize()
                output.append(prediction_row(r,candidate,score=p,decision=int(p>=candidate['threshold']),logit_difference=logit,
                    token_stats={'original_tokens':len(full['input_ids']),'visible_tokens':len(ids),'truncated':truncated,
                        'visible_chars':max(0,end-7) if truncated else len(r['text']),'original_chars':len(r['text']),
                        'tokens_include_prefix_and_specials':True},sequence_tokens=len(ids),actual_input_ids_sha256=digest(ids),
                    backend_total_seconds=time.perf_counter()-started,real_device='cuda encoder + CPU sklearn'))
            except Exception as error:
                output.append(prediction_row(r,candidate,'error',error_type=type(error).__name__,private_error=str(error)))
    return output,{'load_seconds':load,'device':'cuda','dtype':'float32','model_training':encoder.training,
        'peak_vram_bytes':torch.cuda.max_memory_allocated(),'threadpools':pools,'frozen_parameters':True}

def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);parser.add_argument('--model',required=True);args=parser.parse_args()
    out=args.out.resolve();require(out.is_relative_to(BASE/'runs'),'Outside private evaluation runs')
    boundary=PersonalBoundary(out).install();setup_offline()
    manifest=read_json(out/'manifest.json');candidate=next(c for c in manifest['candidates'] if c['name']==args.model)
    require(sha_file(out/'private_inputs.jsonl')==manifest['private_inputs_sha256'],'Inputs changed after freeze')
    records=read_rows(out/'private_inputs.jsonl')
    output,resources=(e5_predictions if candidate['kind']=='e5' else laya_predictions)(out,candidate,records)
    write_rows(out/f'predictions_{args.model}.jsonl',output)
    write_json(out/f'resources_{args.model}.json',{'status':'completed','records':len(output),'successful':sum(r['status']=='ok' for r in output),
        'resources':resources,'boundary':boundary.report(),'fit_performed':False,
        'versions':{n:importlib.metadata.version(n) for n in candidate['versions']}})
    print(args.model,'completed',sum(r['status']=='ok' for r in output),'/',len(output),flush=True)

if __name__=='__main__':main()
