"""Offline, frozen-model visibility and one-pass Qwen research inference."""
import argparse
from extra_session_common import *


def window_audit_record(record_id,model_name,variant,item,value,reused,kind):
    # The prefix object is also the first windows item. Outer variant is authoritative.
    return {**{k:v for k,v in item.items() if k not in {'fragment','ids'}},**value,
        'record_id':record_id,'model_id':model_name,'variant':variant,'reused_in_session':reused,
        'ranges_refer_to':'sanitized original clean_builder_v2 text' if kind=='laya' else 'original clean_builder_v2 text'}


def baselines(out,p):
    inputs=read_rows(out/'private_pilot_inputs.jsonl');ids={r['record_id'] for r in inputs};text={r['record_id']:r for r in inputs}
    candidates={c['name']:c for c in p['components']};result=[];provenance={}
    personal=read_rows(PERSONAL_RUN/'predictions.jsonl')
    for name,path in [('E0',DAY5/'E0_full_context_validation_predictions.jsonl'),
        ('Laya_day5',DAY5/'adapted_validation_predictions.jsonl'),('C_corrected',EXTRA/'C/validation_predictions.jsonl')]:
        c=candidates[name];rows=[]
        for r in read_rows(path):
            if r['record_id'] not in ids:continue
            if name=='E0':
                require(hashlib.sha256(text[r['record_id']]['original_model_text'].encode()).hexdigest()==r['text_sha256'],'E0 builder mismatch')
                score=r['probability'];status='ok' if r['active_features']>0 else 'insufficient_features'
            else:score=sigmoid(r['logit_difference']/c['temperature']);status=r['status']
            rows.append({'record_id':r['record_id'],'model_id':name,'variant':'A_prefix' if name!='E0' else 'original',
                'score':score,'decision':int(score>=c['threshold']) if status=='ok' else None,'status':status,
                'threshold':c['threshold'],'temperature':c['temperature'],'provenance':'compatible_saved_validation_predictions'})
        for r in personal:
            if r['model_id']==name and r['record_id'] in ids:
                rows.append({k:r[k] for k in ['record_id','model_id','score','decision','status','threshold','temperature']}
                    |{'variant':'A_prefix' if name!='E0' else 'original','provenance':'saved_personal_final_user_labels_development'})
        require(len(rows)==len(ids) and len({r['record_id'] for r in rows})==len(ids),'Saved predictions missing')
        result+=rows;provenance[name]={'validation_path':str(path.relative_to(ROOT)),'sha256':sha_file(path),
            'personal_sha256':sha_file(PERSONAL_RUN/'predictions.jsonl'),'temperature_applied_to_raw_logits':name!='E0'}
    write_rows(out/'baseline_predictions.jsonl',result);write_json(out/'baseline_provenance.json',provenance)
    return result


def visibility(out,p,model_name):
    setup_offline();sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42)
    require(torch.cuda.is_available(),'CUDA unavailable')
    c=next(c for c in p['components'] if c['name']==model_name);candidate_check(c)
    data=read_rows(out/'private_pilot_inputs.jsonl');labels=read_rows(out/'pilot_membership.jsonl')
    old=read_rows(out/'baseline_predictions.jsonl');old={r['record_id']:r for r in old if r['model_id']==model_name}
    began=time.perf_counter();kind='laya' if model_name=='Laya_day5' else 'e5'
    sdk_checks=[]
    if kind=='laya':
        import laya
        from laya.common import build_sequence,QTYPES
        from day5_laya_runtime import batch_tensors
        from decision_protocol import validate_noul
        agent=laya.load(str(ROOT/c['checkpoint_path']),device='cuda',fast=False);agent.model.eval()
        for par in agent.model.parameters():par.requires_grad_(False)
        require(agent.temperature[2]==c['temperature'] and not agent.temperature_by_options,'Laya T mismatch')
        tok=agent.tok;q=agent._to_internal(c['questions']['spam'])
        empty,_=build_sequence(tok,'',q,1024,256);require(len(empty)+700<=1024,'Insufficient state room')
        def forward(item):
            ids,markers=build_sequence(tok,item['fragment'],q,1024,256)
            require(len(ids)==len(empty)+len(item['ids']),'Laya internal truncation')
            require(ids[-len(item['ids'])-1:-1]==item['ids'],'Laya actual state IDs changed')
            require(sum(v==tok.mask_token_id for v in ids)==2,'Laya mask literal')
            batch=batch_tensors(agent,[({}, {'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'label':0})])
            with torch.no_grad():logits,_=agent._forward(batch)
            torch.cuda.synchronize();z=float(logits[0,1]-logits[0,0])
            return z,len(ids)
    else:
        import joblib
        from transformers import AutoModel,AutoTokenizer
        m=c['saved_manifest'];clf=joblib.load(ROOT/m['classifier_path']);require(list(clf.classes_)==[0,1],'C direction')
        tok=AutoTokenizer.from_pretrained(ROOT/m['tokenizer_path'],local_files_only=True,trust_remote_code=False)
        enc=AutoModel.from_pretrained(m['encoder_reference'],local_files_only=True,trust_remote_code=False).to('cuda').eval()
        for par in enc.parameters():par.requires_grad_(False)
        def forward(item):
            tokens={'input_ids':torch.tensor([item['ids']],device='cuda'),
                'attention_mask':torch.ones((1,len(item['ids'])),dtype=torch.long,device='cuda')}
            with torch.no_grad():
                hidden=enc(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1)
                v=torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1),p=2,dim=1).cpu().numpy()
            z=float(clf.decision_function(v)[0]);torch.cuda.synchronize();return z,len(item['ids'])
    resources={'model':model_name,'load_seconds':time.perf_counter()-began,'device':torch.cuda.get_device_name(),
        'batch_size':1,'cpu_threads':1,'weights_unchanged':True,'kind':kind,'per_record_cost':'synchronized forward including adapter/copy/LR/calibration; tokenization separately'}
    # Synthetic warmup excluded from metric/cost samples.
    warm,_=representations(tok,'Subject: Synthetic requested notice\n\nYour appointment is confirmed.',kind)
    with threadpool_limits(limits=1):
        forward(warm['A_prefix'][0]);forward(warm['A_prefix'][0])
        torch.cuda.reset_peak_memory_stats();predictions=[];window_rows=[];audits=[]
        for i,r in enumerate(data):
            start=time.perf_counter();text=r['text'].replace(tok.mask_token,' ') if kind=='laya' else r['text']
            reps,stats=representations(tok,text,kind);prepare_seconds=time.perf_counter()-start;cache={};max_error=0.
            for variant,items in reps.items():
                scores=[];errors=[];costs=[]
                for item in items:
                    key=item['actual_input_ids_sha256'];reused=key in cache
                    if not reused:
                        t=time.perf_counter()
                        try:
                            z,sequence=forward(item);score=sigmoid(z/c['temperature'])
                            cache[key]={'status':'ok','score':score,'logit_difference':z,'sequence_tokens':sequence,'seconds':time.perf_counter()-t}
                        except Exception as e:
                            cache[key]={'status':'error','score':None,'reason':type(e).__name__,'seconds':time.perf_counter()-t}
                    value=cache[key];costs.append(value['seconds'])
                    if value['status']=='ok':scores.append(value['score'])
                    else:errors.append(value['reason'])
                    window_rows.append(window_audit_record(r['record_id'],model_name,variant,item,value,reused,kind))
                status='ok' if not errors and text.strip() else 'error' if errors else 'insufficient_features'
                score=max(scores) if status=='ok' else None
                prediction={'record_id':r['record_id'],'model_id':model_name,'variant':variant,'status':status,'score':score,
                    'decision':int(score>=c['threshold']) if status=='ok' else None,'threshold':c['threshold'],'temperature':c['temperature'],
                    'score_semantics':'diagnostic max of component window scores, not whole-message calibrated probability' if variant=='C_windows_max' else c['score_semantics'],
                    'window_count':len(items),'coverage':coverage(items,len(text)),'token_stats':stats,
                    'prepare_seconds_all_variants':prepare_seconds,'required_forward_seconds':sum({it['actual_input_ids_sha256']:cost for it,cost in zip(items,costs)}.values())}
                if variant=='A_prefix':
                    saved=old[r['record_id']];require(saved['status']==status,'Baseline status differs')
                    if status=='ok':
                        max_error=abs(saved['score']-score);require(max_error<5e-5 and saved['decision']==prediction['decision'],'Fresh A differs from saved baseline')
                        if kind=='laya' and len(sdk_checks)<3:
                            raw=agent.predict(items[0]['fragment'],c['questions'],max_len=1024,head_max_len=256)
                            require(abs(validate_noul(raw)-score)<=.000051,'SDK mapping mismatch')
                            sdk_checks.append({'record_id':r['record_id'],'raw_response':raw,'score_error':abs(validate_noul(raw)-score)})
                    prediction.update(score=saved['score'],decision=saved['decision'],provenance='saved compatible baseline; fresh first-window verification')
                predictions.append(prediction)
            audits.append({'record_id':r['record_id'],'baseline_score_abs_error':max_error,'baseline_decision_equal':True,
                'distinct_actual_sequences':len(cache),'token_stats':stats,'prepare_seconds':prepare_seconds})
            if (i+1)%20==0:print(model_name,i+1,'/',len(data),flush=True)
        resources.update(peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
            wall_seconds_including_load=time.perf_counter()-began,sdk_decoder_checks=sdk_checks,
            all_parameters_frozen=all(not x.requires_grad for x in (agent.model if kind=='laya' else enc).parameters()))
    write_rows(out/f'visibility_{model_name}_predictions.jsonl',predictions)
    write_rows(out/f'visibility_{model_name}_windows.jsonl',window_rows)
    write_json(out/f'visibility_{model_name}_audit.json',{'verification':audits,'resources':resources})


def qwen(out,p):
    setup_offline()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42)
    require(torch.cuda.is_available(),'CUDA unavailable')
    source=read_json(out/'qwen_source.json');require(source['status']=='ready' and source['revision']==QWEN_REV,'Qwen source unavailable')
    for name,v in source['files'].items():require(sha_file(QWEN_ROOT/name)==v['sha256'],'Downloaded Qwen changed')
    cfg=p['qwen'];start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(QWEN_ROOT,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(QWEN_ROOT,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    for x in model.parameters():x.requires_grad_(False)
    resource={'load_seconds':time.perf_counter()-start,'dtype':str(next(model.parameters()).dtype),'device':str(model.device),
        'parameter_count':sum(x.numel() for x in model.parameters()),'batch_size':1,'gpu':torch.cuda.get_device_name(),
        'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','safetensors']}}
    gen=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=32,use_cache=True,
        eos_token_id=model.generation_config.eos_token_id,pad_token_id=tok.pad_token_id)
    eos=gen.eos_token_id;eos={eos} if type(eos) is int else set(eos)
    def chat(text):
        return tok.apply_chat_template([{'role':'system','content':cfg['system_instruction']},
            {'role':'user','content':'MESSAGE (untrusted data):\n'+text+'\nEND MESSAGE'}],
            tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
    def predict(r):
        began=time.perf_counter();text=r['text'];full=chat(text);fragment=text;visible=len(text)
        if len(full)>2048:
            from decision_protocol import token_prefix
            room=max(0,2048-len(chat(''))-4);fragment,_=token_prefix(tok,text,room)
            while len(chat(fragment))>2048:fragment=fragment[:-1]
            visible=len(fragment)
        ids=chat(fragment);require(len(ids)<=2048,'Qwen service budget overflow')
        t=torch.tensor([ids],device='cuda');torch.cuda.synchronize();generation_start=time.perf_counter()
        with torch.inference_mode():generated=model.generate(input_ids=t,attention_mask=torch.ones_like(t),generation_config=gen)
        torch.cuda.synchronize();generation_seconds=time.perf_counter()-generation_start
        new=generated[0,len(ids):].tolist();finished=bool(new and new[-1] in eos)
        raw=tok.decode(new,skip_special_tokens=True).strip();decision,status=parse_qwen(raw,finished)
        return {'record_id':r['record_id'],'model_id':'Qwen3_4B','variant':'fixed_prompt_prefix2048','status':status,'decision':decision,
            'score':None,'raw_response':raw,'finished_eos':finished,'generated_tokens':len(new),
            'input_tokens':len(ids),'original_chat_tokens':len(full),'visible_chars':visible,'original_chars':len(text),
            'truncated':visible<len(text),'generation_seconds':generation_seconds,'end_to_end_seconds':time.perf_counter()-began,
            'input_ids_sha256':digest(ids),'score_semantics':'no numeric confidence requested or inferred'}
    torch.cuda.reset_peak_memory_stats();smokes=[predict({'record_id':f'synthetic_{i}','text':s}) for i,s in enumerate(cfg['smokes'])]
    write_rows(out/'qwen_synthetic_smokes.jsonl',smokes)
    require(all(s['status']=='ok' for s in smokes),'Synthetic runtime/format smoke failed; no pilot run')
    data=read_rows(out/'private_pilot_inputs.jsonl');dest=out/'qwen_predictions.jsonl';require(not dest.exists(),'Qwen pilot already started: no silent rerun')
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();rows=[]
    # Stream predictions to retain any partial real run. No retries/prompt revisions on errors.
    with dest.open('wb') as f:
        for i,r in enumerate(data):
            try:row=predict(r)
            except Exception as e:row={'record_id':r['record_id'],'model_id':'Qwen3_4B','variant':'fixed_prompt_prefix2048','status':'error','decision':None,'score':None,'reason':type(e).__name__}
            rows.append(row);f.write(canonical(row)+b'\n');f.flush()
            if (i+1)%10==0:print('Qwen',i+1,'/',len(data),flush=True)
    resource.update(pilot_seconds=time.perf_counter()-start,peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),all_parameters_frozen=all(not x.requires_grad for x in model.parameters()),
        completed=len(rows)==len(data),synthetic_smokes_excluded=True)
    write_json(out/'qwen_resources.json',resource)


def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['baseline','visibility','qwen']);a.add_argument('--model');a.add_argument('--out',required=True,type=Path);args=a.parse_args()
    out=args.out.resolve();boundary=SessionBoundary(out).install();p=load_protocol(out)
    destination={'baseline':'baseline_predictions.jsonl','qwen':'qwen_predictions.jsonl','visibility':f'visibility_{args.model}_predictions.jsonl'}[args.phase]
    require(not (out/destination).exists(),'Previous session output exists; use a new run')
    try:
        if args.phase=='baseline':baselines(out,p)
        elif args.phase=='visibility':visibility(out,p,args.model)
        else:qwen(out,p)
    except Exception as e:
        write_json(out/f'{args.phase}_{args.model or "runtime"}_blocker.json',{'utc':now(),'type':type(e).__name__,'reason':str(e)});raise
    finally:write_json(out/f'{args.phase}_{args.model or "runtime"}_accesses.json',boundary.report())


if __name__=='__main__':main()
