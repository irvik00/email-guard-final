"""Compute only22 previously reviewed personal embeddings, once for all three LRs."""
import argparse
from domain_weighting_common import *

def main(out):
    boundary=WeightBoundary(out).install();p=protocol(out);setup_offline()
    require((out/'models_frozen.json').exists(),'No personal evaluation before both model contracts frozen')
    import torch
    import numpy as np
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    m=p['W0_manifest'];source=m['source']
    for name,spec in source['files'].items():
        if name!='README.md':require(sha_file(Path(source['local_path'])/name)==spec['sha256'],'Encoder source changed')
    tok=AutoTokenizer.from_pretrained(ROOT/m['tokenizer_path'],local_files_only=True,trust_remote_code=False)
    enc=AutoModel.from_pretrained(m['encoder_reference'],local_files_only=True,trust_remote_code=False).to('cuda').eval()
    for par in enc.parameters():par.requires_grad_(False)
    rows=read_rows(PREVIOUS/'private_pilot_inputs.jsonl');ids=set(p['pilot_ids']['personal_development']);rows=[r for r in rows if r['record_id'] in ids]
    require({r['record_id'] for r in rows}=={f'mail_{i:03d}' for i in range(1,23)},'Unreviewed personal ID')
    vectors=[];capped=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for off in range(0,len(rows),16):
        chunk=rows[off:off+16];batch=tok(['query: '+r['text'] for r in chunk],truncation=True,max_length=512,padding=True,return_tensors='pt')
        for r,seq,mask in zip(chunk,batch['input_ids'],batch['attention_mask']):capped.append({'record_id':r['record_id'],'actual_ids_sha256':digest(seq[mask.bool()].tolist())})
        batch=batch.to('cuda')
        with torch.no_grad():
            hidden=enc(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
            vectors.append(torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1),p=2,dim=1).cpu().numpy())
    matrix=np.vstack(vectors);require(matrix.shape==(22,384) and np.isfinite(matrix).all(),'Personal embedding invariant')
    np.save(out/'private_personal_embeddings.npy',matrix,allow_pickle=False)
    write_json(out/'personal_embedding_manifest.json',{'ids':[r['record_id'] for r in rows],'sha256':sha_file(out/'private_personal_embeddings.npy'),
        'source_input_sha256':sha_file(PREVIOUS/'private_pilot_inputs.jsonl'),'prefix':'query: ','max_length':512,
        'pooling':'attention-mask mean then L2','revision':source['revision'],'actual_capped_IDs':capped,
        'seconds':time.perf_counter()-start,'encoder_frozen':all(not x.requires_grad for x in enc.parameters()),
        'peak_vram_bytes':torch.cuda.max_memory_allocated(),'accesses':boundary.report()})
    print('22 reviewed personal embeddings ready; no raw email reads',flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');sys.stderr.reconfigure(encoding='utf8')
    a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);args=a.parse_args();main(args.out.resolve())
