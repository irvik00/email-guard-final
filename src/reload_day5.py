"""Independent stock reload: both raw option logits, unrounded scores, thresholds and SDK T."""
import importlib.metadata
import json
import sys
import gc

from day5_common import ROOT,DOMAINS,Boundary,read_json,read_rows,write_json,require,digest,sigmoid


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    boundary=Boundary('verify').install()
    run_id=read_json(ROOT/'reports/day5/current_run.json')['run_id'];out=ROOT/'reports/day5/runs'/run_id
    manifest=read_json(out/'candidate_manifest.json');config=manifest['config'];training=read_json(out/'training.json')
    dev=read_rows(out/'private_dev_inputs.jsonl')
    selected=[]
    for domain in DOMAINS:
        for label in [0,1]:
            bucket=[r for r in dev if r['domain']==domain and r['spam_label']==label]
            selected.extend(sorted(bucket,key=lambda r:digest([42,'reload-pairs',r['record_id']]))[:4])
    require(len(selected)==16 and all(r['inner_part']=='dev' for r in selected),'Reload must use only fixed internal dev')
    from day5_laya_runtime import torch,encode_rows,infer_encoded,parameter_hash
    from decision_protocol import validate_noul
    import laya
    torch.set_num_threads(1);checks={}
    for name,candidate in manifest['candidates'].items():
        filename='base_dev_raw.jsonl' if name=='base' else f"adapted_dev_step_{manifest['selected_step']}.jsonl"
        expected={r['record_id']:r for r in read_rows(out/filename)}
        agent=laya.load(str(ROOT/candidate['checkpoint_path']),device='cuda',fast=False)
        require(not agent.model.training and agent.temperature[2]==candidate['temperature'] and not agent.temperature_by_options,'Reload config mismatch')
        require(candidate['questions']==config['questions'] and candidate['input_contract']==config['input'],'Question/builder/input contract mismatch')
        expected_head=training['head_sha256_before'] if name=='base' else training['head_sha256_selected']
        require(parameter_hash(agent.model,config['training']['trainable_prefixes'])==expected_head,'Reloaded head not exact saved selection')
        require(parameter_hash(agent.model,['encoder.'])==training['encoder_sha256_before'],'Reloaded encoder changed')
        encoded,visible=encode_rows(agent,selected,config)
        actual=infer_encoded(agent,encoded,candidate['temperature'])
        maximum=score_error=0.;sdk=[]
        for old,new,v in zip([expected[r['record_id']] for r in selected],actual,visible):
            maximum=max(maximum,max(abs(a-b) for a,b in zip(old['logits_false_true'],new['logits_false_true'])))
            wanted=sigmoid(old['logit_difference']/candidate['temperature'])
            score_error=max(score_error,abs(wanted-new['probability']))
            require((wanted>=candidate['threshold'])==(new['probability']>=candidate['threshold']),'Reload decision mismatch')
            response=agent.predict(v['visible_fragment'],config['questions'],max_len=1024,head_max_len=256)
            require(abs(validate_noul(response)-new['probability'])<=.000051,'Reload SDK did not apply persisted T to score')
            sdk.append({'record_id':new['record_id'],'raw_logits_false_true':new['logits_false_true'],
                        'probability':new['probability'],'sdk_raw_response':response})
        require(maximum<=1e-5 and score_error<=1e-5,'Reload pair/score tolerance exceeded')
        checks[name]={'samples':len(actual),'sample_source':'4internal-dev records per domain/class by fixed hash, never validation',
                      'max_raw_option_logit_error':maximum,'max_unrounded_score_error':score_error,'tolerance':1e-5,
                      'threshold_decisions_match':True,'question_input_mapping_persisted':True,
                      'encoder_and_selected_head_hashes_match':True,'temperature_effective':agent.temperature[2],
                      'temperature_by_options':agent.temperature_by_options,'sdk_vs_unrounded_max_rounding_tolerance':.000051,
                      'fit_performed':False,'records':sdk}
        del agent;gc.collect();torch.cuda.empty_cache()
    write_json(out/'independent_reload.json',{'run_id':run_id,'status':'passed','candidates':checks,'access_log':boundary.report()})
    print(json.dumps({name:{k:v for k,v in c.items() if k!='records'} for name,c in checks.items()},indent=2))


if __name__=='__main__':main()
