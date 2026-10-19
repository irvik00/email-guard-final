"""Frozen E5 feature extraction on the exact allowed Day-5 parts (local inference)."""
import argparse
import os
import time
import hashlib
from extra_common import *

os.environ.update(HF_HOME=str(ROOT/'.cache/extra/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',USE_TF='0',TOKENIZERS_PARALLELISM='false')

def visible_text(tokenizer,text):
    prefix='query: ';full=tokenizer(prefix+text,return_offsets_mapping=True,truncation=False)
    capped=tokenizer(prefix+text,return_offsets_mapping=True,truncation=True,max_length=512)
    offsets=capped['offset_mapping'];end=max((b for a,b in offsets),default=len(prefix))
    fragment=text[:max(0,end-len(prefix))]
    # Short inputs retain all whitespace, while capped inputs preserve precisely the actual visible tokens.
    if len(full['input_ids'])<=512:fragment=text
    # SentencePiece segmentation can depend on the word suffix beyond a truncation boundary.
    # Inference always uses the original capped IDs. The matched Laya control receives the exact
    # character span covered by those offsets, without pretending standalone re-tokenization is invariant.
    same=tokenizer(prefix+fragment,truncation=True,max_length=512)['input_ids']==capped['input_ids']
    return fragment,{'original_tokens_with_prefix_and_special':len(full['input_ids']),
                     'sequence_tokens':len(capped['input_ids']),'truncated':len(full['input_ids'])>512,
                     'visible_chars':len(fragment),'original_chars':len(text),'standalone_retokenization_exact':same,
                     'actual_input_ids_sha256':digest(capped['input_ids'])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',action='store_true');p.add_argument('--reload',action='store_true');a=p.parse_args()
    out,models=paths();boundary=ExtraBoundary('train').install()
    import torch
    import numpy as np
    from transformers import AutoTokenizer,AutoModel
    import psutil
    source=read_json(out/'e5_source.json');root=Path(source['local_path'])
    for n,s in source['files'].items():require(sha_file(root/n)==s['sha256'],'E5 source changed')
    torch.set_num_threads(1);torch.manual_seed(42);start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    encoder=AutoModel.from_pretrained(root,local_files_only=True,trust_remote_code=False).to('cuda').eval()
    for parameter in encoder.parameters():parameter.requires_grad_(False)
    dimension=encoder.config.hidden_size
    parts=['validation'] if a.validation else ['fit','dev','calibration']
    results={};torch.cuda.reset_peak_memory_stats()
    def embed(texts):
        batch=tokenizer(['query: '+text for text in texts],max_length=512,truncation=True,padding=True,return_tensors='pt').to('cuda')
        with torch.no_grad():
            hidden=encoder(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1)
            return torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy()
    if a.reload:
        old=read_json(out/'C/embeddings.json');rows=read_rows(out/'C/private_dev_inputs.jsonl')
        matrix=np.load(ROOT/old['parts']['dev']['matrix_path'],allow_pickle=False)
        indices=[]
        for domain in DOMAINS:
            for label in [0,1]:
                indices+=sorted([i for i,r in enumerate(rows) if r['domain']==domain and r['spam_label']==label],
                                key=lambda i:digest([42,'C-reload',rows[i]['record_id']]))[:4]
        fresh=np.vstack([embed([rows[i]['text']]) for i in indices])
        error=float(np.max(np.abs(fresh-matrix[indices])))
        require(error<=1e-5,'Freshly reloaded E5 embeddings changed')
        np.save(out/'C/reload_probe_embeddings.npy',fresh,allow_pickle=False)
        write_json(out/'C/independent_encoder_reload.json',{'status':'passed','samples':len(indices),
            'dev_indices':indices,'max_embedding_abs_error':error,'tolerance':1e-5,'fit_on_reload':False,
            'weights_hashes_checked':True,'accesses':boundary.report()})
        print({'C_encoder_reload':'passed','max_error':error});return
    for part in parts:
        rows=read_rows(out/f'C/private_{part}_inputs.jsonl')
        if part!='validation':require(all(r['inner_part']==part for r in rows),'Wrong selected role')
        visible=[]
        for row in rows:
            fragment,stats=visible_text(tokenizer,row['text'])
            visible.append({'record_id':row['record_id'],'domain':row['domain'],'spam_label':row['spam_label'],
                'visible_fragment':fragment,'text_hash':hashlib.sha256(row['text'].encode()).hexdigest(),
                'visible_sha256':hashlib.sha256(fragment.encode()).hexdigest(),'token_stats':stats})
        cachekey=digest({'text_hashes':[r['text_hash'] for r in visible],'model_revision':source['revision'],
                        'prefix':'query: ','tokenization':'max512,padding_batch16,masked_mean,L2,float32',
                        'tokenizer_sha256':source['files']['tokenizer.json']['sha256']})
        cache=ROOT/'.cache/extra/embeddings'/f'{cachekey}.npy';cache.parent.mkdir(parents=True,exist_ok=True)
        began=time.perf_counter();hit=cache.exists()
        if hit:matrix=np.load(cache,allow_pickle=False)
        else:
            matrix=np.empty((len(rows),dimension),dtype=np.float32)
            for offset in range(0,len(rows),16):
                # Use original text + the actual max512 tokenizer. Visible prefix is for audit/matched control.
                matrix[offset:offset+16]=embed([r['text'] for r in rows[offset:offset+16]])
                if offset%1000==0:print(f'E5 {part} {offset}/{len(rows)}',flush=True)
            np.save(cache,matrix,allow_pickle=False)
        require(matrix.shape==(len(rows),dimension) and bool(np.isfinite(matrix).all()),'Embedding shape/finite invariant')
        require(bool(np.allclose(np.linalg.norm(matrix,axis=1),1.,atol=1e-5)),'E5 normalization failure')
        write_rows(out/f'C/private_{part}_visible.jsonl',visible)
        results[part]={'matrix_path':str(cache.relative_to(ROOT).as_posix()),'sha256':sha_file(cache),
            'input_sha256':sha_file(out/f'C/private_{part}_inputs.jsonl'),'visible_sha256':sha_file(out/f'C/private_{part}_visible.jsonl'),
            'cache_key':cachekey,'cache_hit':hit,'rows':len(rows),'embedding_seconds':time.perf_counter()-began,
            'truncated_by_domain':{d:sum(r['token_stats']['truncated'] for r in visible if r['domain']==d) for d in DOMAINS}}
    latency=[]
    if not a.validation:
        # New inference-only measurement, fixed SHA-ordered 64 inner-dev records; never fit.
        dev=read_rows(out/'C/private_dev_inputs.jsonl')
        bench=sorted(dev,key=lambda r:digest([42,'latency',r['record_id']]))[:64]
        embed(['Warm-up synthetic text.']);torch.cuda.synchronize()
        for r in bench:
            t=time.perf_counter();embed([r['text']]);torch.cuda.synchronize();latency.append(time.perf_counter()-t)
    versions={n:__import__('importlib.metadata',fromlist=['version']).version(n) for n in ['torch','transformers','tokenizers','numpy','safetensors']}
    report={'model':source,'dimension':dimension,'encoder_parameters':sum(p.numel() for p in encoder.parameters()),
        'encoder_frozen':all(not p.requires_grad for p in encoder.parameters()),'eval_mode':not encoder.training,
        'encoder_weights_sha256_after':sha_file(root/('model.safetensors' if (root/'model.safetensors').exists() else 'pytorch_model.bin')),
        'versions':versions,'parts':results,'warm_single_record_seconds':latency,'device':'cuda',
        'wall_seconds':time.perf_counter()-start,'peak_vram_bytes':torch.cuda.max_memory_allocated(),
        'rss_bytes':psutil.Process().memory_info().rss,'accesses':boundary.report()}
    write_json(out/('C/validation_embeddings.json' if a.validation else 'C/embeddings.json'),report)

if __name__=='__main__':main()
