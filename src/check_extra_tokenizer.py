"""Reload the saved C tokenizer copy; compare actual capped IDs with frozen dev artifacts."""
import os
from extra_common import *

def main():
    out,_=paths();boundary=ExtraBoundary('train').install()
    os.environ.update(HF_HOME=str(ROOT/'.cache/extra/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',HF_HUB_DISABLE_TELEMETRY='1')
    from transformers import AutoTokenizer
    manifest=read_json(out/'C/candidate_manifest.json');folder=ROOT/manifest['tokenizer_path']
    files={}
    for name,spec in manifest['source']['files'].items():
        if name in ['README.md','model.safetensors','pytorch_model.bin']:continue
        require(sha_file(folder/name)==spec['sha256'],'Saved tokenizer/config copy changed: '+name);files[name]=spec['sha256']
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    data=read_rows(out/'C/private_dev_inputs.jsonl');visible={r['record_id']:r for r in read_rows(out/'C/private_dev_visible.jsonl')}
    selected=[]
    for domain in DOMAINS:
        for label in [0,1]:selected+=sorted([r for r in data if r['domain']==domain and r['spam_label']==label],key=lambda r:digest([42,'C-reload',r['record_id']]))[:4]
    for row in selected:
        ids=tokenizer('query: '+row['text'],truncation=True,max_length=512)['input_ids']
        require(len(ids)<=512 and digest(ids)==visible[row['record_id']]['token_stats']['actual_input_ids_sha256'],'Saved tokenizer reload differs')
    write_json(out/'C/tokenizer_reload.json',{'status':'passed','samples':len(selected),'all_capped_ID_hashes_identical':True,
        'files_sha256':files,'fit_performed':False,'accesses':boundary.report()})
    print('Saved C tokenizer reloaded;16 capped-ID hashes identical')

if __name__=='__main__':main()
