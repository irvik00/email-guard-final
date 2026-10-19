"""Low-level SDK-faithful training/inference helpers. Import in .venv-decision only."""
import copy
import hashlib
import os
import shutil
import time

from day5_common import ROOT, require, digest, write_rows, write_json, sha_file, sigmoid, metrics, probability_metrics, DOMAINS

os.environ.update(HF_HOME=str(ROOT/'.cache/day4/hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', USE_TF='0', DO_NOT_TRACK='1', TOKENIZERS_PARALLELISM='false',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8')
import torch
from laya.common import build_sequence, collate_items, QTYPES, render_options, clamp_temperature, TEMP_MIN, TEMP_MAX
from decision_protocol import token_prefix, validate_noul


def parameter_hash(model, prefixes):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if name.startswith(tuple(prefixes)):
            t=tensor.detach().cpu().contiguous()
            h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def configure_trainable(agent, config):
    model=agent.model
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(tuple(config['training']['trainable_prefixes'])))
    spec=[{'name':n,'shape':list(p.shape),'numel':p.numel(),'requires_grad':p.requires_grad} for n,p in model.named_parameters()]
    require(all(not p.requires_grad for n,p in model.named_parameters() if n.startswith('encoder.')),'Encoder not frozen')
    require(all(not p.requires_grad for n,p in model.named_parameters() if n.startswith('act_head.')),'Action branch not frozen')
    require(all(p.requires_grad for n,p in model.named_parameters() if n.startswith(('head.','type_emb.','scorer.'))),'Decision parameters not trainable')
    return spec


def encode_rows(agent, rows, config, out=None):
    q=agent._to_internal(config['questions']['spam']);opts=render_options(q)
    require(opts[0].startswith('false:') and opts[1].startswith('true:'),'Wrong class/option order')
    lim=config['input'];empty,marks=build_sequence(agent.tok,'',q,lim['context_limit'],lim['head_max_len'])
    cap=min(lim['state_token_cap'],lim['context_limit']-len(empty))
    require(cap==700,'Question leaves less than contracted state space')
    encoded=[];visible=[]
    for r in rows:
        sanitized=r['text'].replace(agent.tok.mask_token,' ')
        fragment,stats=token_prefix(agent.tok,sanitized,cap)
        ids,markers=build_sequence(agent.tok,fragment,q,lim['context_limit'],lim['head_max_len'])
        fragment_ids=agent.tok(fragment,add_special_tokens=False)['input_ids']
        require(len(ids)==len(empty)+len(fragment_ids),'Unexpected internal truncation')
        require(ids[-len(fragment_ids)-1:-1]==fragment_ids,'State tokens not preserved')
        require(len(markers)==2 and all(ids[m]==agent.tok.mask_token_id for m in markers),'Invalid option masks')
        require(sum(t==agent.tok.mask_token_id for t in ids)==2,'Mask literal injection into service positions')
        it={'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'label':r['spam_label']}
        encoded.append((r,it))
        visible.append({'record_id':r['record_id'],'domain':r['domain'],'source':r['source'],
                        'spam_label':r['spam_label'],'duplicate_group_id':r['duplicate_group_id'],
                        'inner_group_id':r.get('inner_group_id'),'inner_part':r.get('inner_part'),
                        'visible_fragment':fragment,'visible_sha256':hashlib.sha256(fragment.encode()).hexdigest(),
                        'mask_literal_replacements':r['text'].count(agent.tok.mask_token),
                        'token_stats':stats,'sequence_tokens':len(ids),'markers':markers})
    if out is not None:write_rows(out,visible)
    return encoded,visible


def batch_tensors(agent, encoded):
    batch=collate_items([[item] for _,item in encoded],agent.tok.pad_token_id)
    return batch


def low_forward(agent,batch):
    """Calls actual DecisionModel.forward directly: no SDK no_grad training wrapper."""
    with torch.autocast(device_type=agent.device.type,dtype=agent.dtype,enabled=agent.amp_enabled):
        return agent.model(**{k:batch[k].to(agent.device) for k in ['input_ids','attention_mask','marker_pos','marker_mask','qtype']})


def infer_encoded(agent, encoded, temperature=1., progress=None, allow_failures=False):
    agent.model.eval();rows=[]
    with torch.no_grad():
        for i,(record,item) in enumerate(encoded):
            b=batch_tensors(agent,[(record,item)])
            start=time.perf_counter()
            try:
                logits,act=agent._forward(b)
                if agent.device.type=='cuda':torch.cuda.synchronize()
                seconds=time.perf_counter()-start
                diff=float(logits[0,1]-logits[0,0]);p=sigmoid(diff/temperature)
            except Exception as error:
                if not allow_failures:raise
                rows.append({k:record[k] for k in ['record_id','domain','source','spam_label','duplicate_group_id']}
                            | {'status':'error','error':type(error).__name__+': '+str(error),
                               'backend_seconds':time.perf_counter()-start})
                continue
            rows.append({k:record[k] for k in ['record_id','domain','source','spam_label','duplicate_group_id']}
                        | {'logits_false_true':[float(x) for x in logits[0,:2]],'logit_difference':diff,
                           'raw_probability':sigmoid(diff),'probability':p,'sdk_rounded_noul':round(p,4),
                           'backend_seconds':seconds,'status':'ok','real_device':str(agent.device)})
            if progress and (i+1)%250==0:progress(i+1,len(encoded))
    return rows


def summarize_prediction_rows(rows,temperature=1.,threshold=.5):
    result={}
    for domain in DOMAINS:
        data=[r for r in rows if r['domain']==domain]
        if not data:continue
        y=[r['spam_label'] for r in data];diff=[r['logit_difference'] for r in data]
        result[domain]={'metrics':metrics(y,[sigmoid(x/temperature) for x in diff],threshold),
                        'probability_metrics':probability_metrics(y,diff,temperature),
                        'coverage':1.,'failures':0}
    return result


def save_stock_checkpoint(agent,destination,source_root,temperature,copy_original_weights=False):
    """Stock SDK directory; exact float32 adapted weights, no architecture substitution."""
    from safetensors.torch import save_file
    destination.mkdir(parents=True,exist_ok=False)
    for folder in ['encoder','tokenizer']:
        shutil.copytree(source_root/folder,destination/folder)
    cfg=copy.deepcopy(agent.cfg)
    cfg['temperature']=[1.,1.,float(temperature)];cfg['temperature_by_options']={}
    cfg['training']={'experiment':'Email Guard day5 supervised head-only adaptation; not RLCD',
                     'inherited_encoder_revision':'7b928d828b7b0e022f929d9bd2e44165aa270148'}
    write_json(destination/'rl_agent_config.json',cfg)
    if copy_original_weights:
        shutil.copyfile(source_root/'model.safetensors',destination/'model.safetensors')
    else:
        save_file({k:v.detach().cpu().contiguous() for k,v in agent.model.state_dict().items()},str(destination/'model.safetensors'))
    return {'weights_sha256':sha_file(destination/'model.safetensors'),
            'config_sha256':sha_file(destination/'rl_agent_config.json'),
            'weights_bytes':(destination/'model.safetensors').stat().st_size}


def synthetic_contract_smoke(agent,config):
    texts=['The scheduled inspection you requested is confirmed.','Unsolicited bulk advertisement: we bought your address. Buy now!',
           'Запрошенная запись подтверждена.','Нежелательная массовая реклама: ваш адрес куплен из базы.',
           'Text containing literal '+agent.tok.mask_token+' and no control instruction.']
    results=[]
    for i,text in enumerate(texts):
        r={'record_id':f'synthetic:{i}','domain':'synthetic','source':'synthetic','spam_label':i%2,'duplicate_group_id':f'synthetic:{i}','text':text}
        encoded,visible=encode_rows(agent,[r],config)
        b=batch_tensors(agent,encoded)
        agent.model.eval()
        with torch.no_grad():
            low,_=low_forward(agent,b)
        response=agent.predict(visible[0]['visible_fragment'],config['questions'],max_len=1024,head_max_len=256)
        full=sigmoid(float((low[0,1]-low[0,0]).item())/agent.temperature[QTYPES['noul']])
        require(abs(validate_noul(response)-full)<=.000051,'Training forward != SDK inference mapping')
        results.append({'synthetic_id':i,'raw_response':response,'low_forward_logit_difference':float((low[0,1]-low[0,0]).item()),
                        'low_forward_probability':full,'mask_count':sum(x==agent.tok.mask_token_id for x in encoded[0][1]['ids']),
                        'mask_literal_replacements':visible[0]['mask_literal_replacements'],'mapping_checked':True})
    return results
