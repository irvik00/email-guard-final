"""Reuse stock Day-5 SDK path with explicit partial-freeze and variable input caps."""
from day5_laya_runtime import *
from day5_laya_runtime import encode_rows as day5_encode, save_stock_checkpoint as day5_save

def configure_trainable(agent, config):
    partial='encoder_learning_rate' in config['training']
    prefixes=['head.','type_emb.','scorer.']
    if partial:
        # Inspect actual ModuleList and resolve its canonical named_modules path.
        layers=agent.model.encoder.layers
        require(isinstance(layers,torch.nn.ModuleList) and len(layers)>=2,'Unsupported installed encoder blocks')
        names=dict(agent.model.named_modules())
        selected=[next(n for n,m in names.items() if m is block) for block in list(layers)[-2:]]
        prefixes += [n+'.' for n in selected]
        config['training']['resolved_encoder_blocks']=selected
    config['training']['trainable_prefixes']=prefixes
    for name,p in agent.model.named_parameters():p.requires_grad_(name.startswith(tuple(prefixes)))
    require(not any(p.requires_grad for p in agent.model.act_head.parameters()),'Action branch unfrozen')
    return [{'name':n,'shape':list(p.shape),'numel':p.numel(),'requires_grad':p.requires_grad} for n,p in agent.model.named_parameters()]

def frozen_hash(agent):
    names=[n for n,p in agent.model.named_parameters() if not p.requires_grad]
    # Exact names are delimited by a sentinel in the hashing adapter, not ambiguous prefix matching.
    import hashlib
    h=hashlib.sha256()
    for n in names:
        t=agent.model.state_dict()[n].detach().cpu().contiguous()
        h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def encode_rows(agent,rows,config,out=None):
    cap=config['input']['state_token_cap']
    if cap==700:return day5_encode(agent,rows,config,out)
    # Same SDK construction and actual offset-token prefix, with a separately declared cap.
    q=agent._to_internal(config['questions']['spam']);lim=config['input']
    empty,_=build_sequence(agent.tok,'',q,lim['context_limit'],lim['head_max_len'])
    require(len(empty)+cap<=lim['context_limit'],'Question exceeds context contract')
    encoded=[];visible=[]
    import hashlib
    for r in rows:
        fragment,stats=token_prefix(agent.tok,r['text'].replace(agent.tok.mask_token,' '),cap)
        ids,marks=build_sequence(agent.tok,fragment,q,lim['context_limit'],lim['head_max_len'])
        actual=agent.tok(fragment,add_special_tokens=False)['input_ids']
        require(len(ids)==len(empty)+len(actual) and ids[-len(actual)-1:-1]==actual,'Silent SDK truncation')
        require(sum(x==agent.tok.mask_token_id for x in ids)==2,'Mask injection')
        encoded.append((r,{'ids':ids,'markers':marks,'qtype':QTYPES['noul'],'label':r['spam_label']}))
        visible.append({'record_id':r['record_id'],'domain':r['domain'],'spam_label':r['spam_label'],
            'visible_fragment':fragment,'visible_sha256':hashlib.sha256(fragment.encode()).hexdigest(),
            'token_stats':stats,'sequence_tokens':len(ids)})
    if out:write_rows(out,visible)
    return encoded,visible

def save_stock_checkpoint(agent,destination,source_root,temperature,copy_original_weights=False):
    artifacts=day5_save(agent,destination,source_root,temperature,copy_original_weights)
    cfg=__import__('day5_common').read_json(destination/'rl_agent_config.json')
    cfg['training']={'experiment':'Email Guard EXTRA supervised adaptation; see parent candidate_manifest',
                     'initial_encoder_revision':'7b928d828b7b0e022f929d9bd2e44165aa270148'}
    write_json(destination/'rl_agent_config.json',cfg)
    artifacts['config_sha256']=sha_file(destination/'rl_agent_config.json')
    return artifacts
