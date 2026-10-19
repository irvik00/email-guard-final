"""Predeclared inner-dev inference controls, never selects weights/T/threshold."""
import argparse
import copy
import gc
from extra_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--e5-matched',action='store_true');a=p.parse_args()
    out,models=paths();boundary=ExtraBoundary('train').install()
    from extra_laya_runtime import torch,encode_rows,infer_encoded,summarize_prediction_rows
    import laya
    torch.set_num_threads(1)
    dest=out/('e5_matched_control.json' if a.e5_matched else 'context_control.json')
    require(not dest.exists(),'Control already evaluated')
    manifest=read_json(out/'A/candidate_manifest.json');candidate=manifest['candidates']['adapted']
    cfg=manifest['config'];rows=read_rows(out/'A/private_dev_inputs.jsonl')
    agent=laya.load(str(ROOT/candidate['checkpoint_path']),device='cuda',fast=False)
    short=read_rows(out/f"A/adapted_dev_step_{manifest['selected_step']}.jsonl")
    if a.e5_matched:
        visible=read_rows(out/'C/private_dev_visible.jsonl');byid={r['record_id']:r for r in visible}
        for row in rows:row['text']=byid[row['record_id']]['visible_fragment']
        cfg=copy.deepcopy(cfg);cfg['input'].update(state_token_cap=1500,context_limit=2048)
        key='e5_exact_fragment_Laya_cap1500_context2048'
    else:
        require(agent.model.encoder.config.max_position_embeddings>=2048,'Encoder length unsupported')
        cfg=copy.deepcopy(cfg);cfg['input'].update(state_token_cap=1500,context_limit=2048)
        key='cap1500_context2048'
    encoded,visible_laya=encode_rows(agent,rows,cfg,out/('private_e5_matched_laya_visible.jsonl' if a.e5_matched else 'private_long_dev_visible.jsonl'))
    if a.e5_matched:
        require(all(v['visible_fragment']==byid[v['record_id']]['visible_fragment'] for v in visible_laya),
                'E5-visible fragment exceeds Laya cap; no claim of exact matching allowed')
    pred=infer_encoded(agent,encoded,candidate['temperature'])
    write_rows(out/(key+'_predictions.jsonl'),pred)
    short_visible={r['record_id']:r for r in read_rows(out/'A/private_dev_visible.jsonl')}
    longids={rid for rid,r in short_visible.items() if r['token_stats']['truncated']}
    paired=[]
    old={r['record_id']:r for r in short}
    for r in pred:
        before=sigmoid(old[r['record_id']]['logit_difference']/candidate['temperature'])>=candidate['threshold']
        after=r['probability']>=candidate['threshold']
        if before!=after:paired.append({'record_id':r['record_id'],'label':r['spam_label'],'before':int(before),'after':int(after)})
    result={'status':'completed','control':key,'weights_T_threshold_unchanged':True,'inner_dev_only':True,
        'all_dev':summarize_prediction_rows(pred,candidate['temperature'],candidate['threshold']),
        'input_contract':cfg['input'],
        'long_subset_n':len(longids),'long_subset':summarize_prediction_rows([r for r in pred if r['record_id'] in longids],candidate['temperature'],candidate['threshold']),
        'short_long_subset':summarize_prediction_rows([r for r in short if r['record_id'] in longids],candidate['temperature'],candidate['threshold']),
        'paired_class_changes':paired,'extra_Laya_truncations':sum(v['token_stats']['truncated'] for v in visible_laya),
        'warning':'inference-only distribution change, no new fit and no validation tuning','accesses':boundary.report()}
    write_json(dest,result);print({'control':key,'long_subset':len(longids),'paired_changes':len(paired)})

if __name__=='__main__':main()
