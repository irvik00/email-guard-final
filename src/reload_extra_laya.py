"""Independent checkpoint reload on fixed internal-dev IDs, both option logits and SDK outputs."""
from extra_common import *

def main():
    out,models=paths();boundary=ExtraBoundary('train').install()
    from extra_laya_runtime import torch,encode_rows,infer_encoded,parameter_hash
    from decision_protocol import validate_noul
    import laya
    torch.set_num_threads(1);results={}
    for branch in ['A','B']:
        manifest=read_json(out/branch/'candidate_manifest.json');candidate=manifest['candidates']['adapted']
        rows=read_rows(out/branch/'private_dev_inputs.jsonl')
        selected=[]
        for domain in DOMAINS:
            for label in [0,1]:
                selected+=sorted([r for r in rows if r['domain']==domain and r['spam_label']==label],key=lambda r:digest([42,'extra-reload',r['record_id']]))[:4]
        old={r['record_id']:r for r in read_rows(out/branch/f"adapted_dev_step_{manifest['selected_step']}.jsonl")}
        agent=laya.load(str(ROOT/candidate['checkpoint_path']),device='cuda',fast=False)
        encoded,visible=encode_rows(agent,selected,manifest['config']);pred=infer_encoded(agent,encoded,candidate['temperature'])
        errors=[]
        for row,v in zip(pred,visible):
            expected=old[row['record_id']]
            for x,y in zip(row['logits_false_true'],expected['logits_false_true']):errors.append(abs(x-y))
            response=agent.predict(v['visible_fragment'],manifest['config']['questions'],max_len=1024,head_max_len=256)
            require(abs(validate_noul(response)-row['probability'])<=.000051,'Reload SDK probability mapping differs')
            require((row['probability']>=candidate['threshold'])==(sigmoid(expected['logit_difference']/candidate['temperature'])>=candidate['threshold']),'Reload class decision changed')
        require(max(errors)<=1e-5,'Raw option logits changed on reload')
        results[branch]={'samples':len(pred),'both_option_logits_max_error':max(errors),'threshold_classes_identical':True,
            'temperature':agent.temperature[2],'question':manifest['config']['questions'],'fit_on_reload':False,
            'weight_sha256':sha_file(ROOT/candidate['checkpoint_path']/'model.safetensors')}
        del agent;torch.cuda.empty_cache()
    write_json(out/'independent_reload.json',{'status':'passed','candidates':results,'accesses':boundary.report()})
    print(results)

if __name__=='__main__':main()
