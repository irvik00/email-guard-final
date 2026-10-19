"""One larger-fit frozen-E5 LR, using the original inner partition and calibration rule."""
import argparse
import warnings
from extra_session_common import *


def text_keys(r):
    return {('clean_exact',digest(r['text'])),('clean_canonical',digest(compare_text(r['text']))),
            ('original_exact',digest(r['original_model_text'])),('original_canonical',digest(compare_text(r['original_model_text'])))}


def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);args=a.parse_args();out=args.out.resolve()
    boundary=SessionBoundary(out,train=True).install();p=load_protocol(out);setup_offline()
    require(not (out/'C_larger_fit').exists(),'One fit per run; existing outputs retained')
    dest=out/'C_larger_fit';dest.mkdir();started=time.perf_counter()
    write_json(dest/'started.json',{'utc':now(),'protocol_sha256':sha_file(out/'EXTRA_SESSION_PROTOCOL.json')})
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    import torch
    import numpy as np
    import joblib
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42)
    c=next(c for c in p['components'] if c['name']=='C_corrected');candidate_check(c);m=c['saved_manifest']
    require(sha_file(DAY5/'membership.jsonl')==p['larger_fit']['membership_sha256'],'Original inner membership changed')
    membership={r['record_id']:r for r in read_rows(DAY5/'membership.jsonl')}
    split=read_json(ROOT/'data/processed/splits/manifest.json');day4=read_json(ROOT/'config/day4_experiments.json')
    all_rows=[];fingerprints={}
    for domain in DOMAINS:
        rows,fp=load_split(domain,'train',split,day4);all_rows+=rows;fingerprints[domain]=fp
    require({r['record_id'] for r in all_rows}==set(membership),'Full original train IDs changed')
    for r in all_rows:
        old=membership[r['record_id']]
        require(old['clean_text_sha256']==digest(r['text']) and old['original_text_sha256']==digest(r['original_model_text']),'Inner text changed')
        require(all(old[k]==r[k] for k in ['domain','source','spam_label','duplicate_group_id']),'Inner metadata changed')
        r.update(inner_part=old['inner_part'],inner_group_id=old['inner_group_id'],conflicting_group=old['conflicting_group'])
    original={part:read_rows(EXTRA/f'C/private_{part}_inputs.jsonl') for part in ['fit','dev','calibration']}
    for part,rows in original.items():
        require(all(membership[r['record_id']]['inner_part']==part for r in rows),'Original selected role mismatch')
    excluded=set(m['calibration_excluded_ids']);cal=[r for r in original['calibration'] if r['record_id'] not in excluded]
    require(len(original['fit'])==6000 and len(original['dev'])==600 and len(cal)==999,'Wrong fixed selected sets')
    baseline_ids={r['record_id'] for r in original['fit']}
    byid={r['record_id']:r for r in all_rows};selected=[]
    for domain in DOMAINS:
        eligible=[r for r in all_rows if r['inner_part']=='fit' and r['domain']==domain and not r['conflicting_group']]
        n=min(p['larger_fit']['target_per_class'][domain],*(sum(r['spam_label']==y for r in eligible) for y in [0,1]))
        for label in [0,1]:
            selected+=sorted([r for r in eligible if r['spam_label']==label],key=lambda r:digest([42,'fit-subset',r['record_id']]))[:n]
    require(baseline_ids<={r['record_id'] for r in selected},'Larger fit does not contain original fit6000')
    require(all(r['inner_part']=='fit' for r in selected),'Non-fit in larger fit')
    audit_parts({part:[r for r in all_rows if r['inner_part']==part] for part in ['fit','dev','calibration']})
    tok=AutoTokenizer.from_pretrained(ROOT/m['tokenizer_path'],local_files_only=True,trust_remote_code=False)
    def capped(r):return digest(tok('query: '+r['text'],truncation=True,max_length=512)['input_ids'])
    reserved=[r for r in all_rows if r['inner_part']!='fit'];reserved_keys=defaultdict(list);reserved_ids=defaultdict(list)
    for i,r in enumerate(reserved):
        for key in text_keys(r):reserved_keys[key].append(r['record_id'])
        reserved_ids[capped(r)].append(r['record_id'])
        if (i+1)%2000==0:print('C reserved-token audit',i+1,'/',len(reserved),flush=True)
    retained=[];collisions=[];known_baseline=[];fit_hashes={};selected_pre=len(selected)
    for i,r in enumerate(selected):
        h=capped(r);fit_hashes[r['record_id']]=h
        text_hits=set(x for key in text_keys(r) for x in reserved_keys.get(key,[]));hits=set(reserved_ids.get(h,[]))|text_hits
        if hits:
            entry={'fit_id':r['record_id'],'fit_label':r['spam_label'],'reserved_ids':sorted(hits),
                'reserved_roles':{i:membership[i]['inner_part'] for i in hits},'actual_input_ids_sha256':h,
                'full_text_matches':sorted(text_hits),'label_conflict':any(byid[i]['spam_label']!=r['spam_label'] for i in hits)}
            if r['record_id'] not in baseline_ids:collisions.append(entry);continue
            known_baseline.append(entry)
        retained.append(r)
        if (i+1)%2000==0:print('C fit-token audit',i+1,'/',len(selected),flush=True)
    require(len(retained)>6000,'No larger eligible fit')
    chosen_reserved={r['record_id']:r for r in original['dev']+cal}
    require(all(not (set(x['reserved_ids'])&set(chosen_reserved)) for x in known_baseline),'Baseline collision with actual corrected dev/cal')
    parts={'fit':retained,'dev':original['dev'],'calibration':cal};selected_hashes={}
    for part,rows in parts.items():
        selected_hashes[part]=defaultdict(list)
        for r in rows:selected_hashes[part][fit_hashes.get(r['record_id']) or capped(r)].append(r)
    cross=[]
    for a,b in [('fit','dev'),('fit','calibration'),('dev','calibration')]:
        cross += [{'parts':[a,b],'ids':[x['record_id'] for x in selected_hashes[a][h]+selected_hashes[b][h]]} for h in set(selected_hashes[a])&set(selected_hashes[b])]
    require(not cross,'Actual capped-ID leakage after exclusion')
    conflicts=[]
    for h,rows in selected_hashes['fit'].items():
        if len({r['spam_label'] for r in rows})>1:conflicts.append({'input_ids_sha256':h,'ids':[r['record_id'] for r in rows],'labels':[r['spam_label'] for r in rows]})
    write_rows(dest/'fit_membership.jsonl',[{k:r[k] for k in ['record_id','domain','source','spam_label','duplicate_group_id','inner_part','inner_group_id']}
        |{'baseline_fit':r['record_id'] in baseline_ids,'actual_input_ids_sha256':fit_hashes[r['record_id']]} for r in retained])
    selection={'source_membership_sha256':sha_file(DAY5/'membership.jsonl'),'full_inner_counts':{part:sum(r['inner_part']==part for r in all_rows) for part in ['fit','dev','calibration']},
        'selected_before_capped_exclusions':selected_pre,'selected_after':len(retained),'baseline_preserved':6000,
        'new_fit_exclusions':collisions,'known_baseline_vs_unused_reserved':known_baseline,'selected_cross_capped_collisions':cross,
        'fit_capped_label_conflicts':conflicts,'counts':counts(retained),'dev_count':600,'calibration_count':999,
        'input_fingerprints':fingerprints,'frozen_before_fit_utc':now(),'all_original_held_rows_checked':len(reserved)}
    write_json(dest/'selection_and_collisions.json',selection)
    # Provenance-checked cache reuse; only new fit rows need encoder inference.
    spec=read_json(EXTRA/'C/embeddings.json');matrices={}
    for part,rows in original.items():
        entry=spec['parts'][part]
        require(sha_file(EXTRA/f'C/private_{part}_inputs.jsonl')==entry['input_sha256'],'Cached input mismatch')
        require(sha_file(ROOT/entry['matrix_path'])==entry['sha256'],'Embedding cache changed')
        require(spec['model']['revision']==m['source']['revision'] and spec['model']['files']['tokenizer.json']['sha256']==m['source']['files']['tokenizer.json']['sha256'],'Cache encoder/tokenizer incompatible')
        visible=read_rows(EXTRA/f'C/private_{part}_visible.jsonl')
        require(len(visible)==len(rows) and all(r['record_id']==v['record_id'] and capped(r)==v['token_stats']['actual_input_ids_sha256'] for r,v in zip(rows,visible)),'Cached capped token IDs mismatch')
        matrices[part]=np.load(ROOT/entry['matrix_path'],allow_pickle=False)
    enc=AutoModel.from_pretrained(m['encoder_reference'],local_files_only=True,trust_remote_code=False).to('cuda').eval()
    for par in enc.parameters():par.requires_grad_(False)
    def embed(rows):
        batch=tok(['query: '+r['text'] for r in rows],truncation=True,max_length=512,padding=True,return_tensors='pt').to('cuda')
        with torch.no_grad():
            hidden=enc(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
            return torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1),p=2,dim=1).cpu().numpy()
    baseindex={r['record_id']:i for i,r in enumerate(original['fit'])};matrix=np.empty((len(retained),384),dtype=np.float32)
    newidx=[]
    for i,r in enumerate(retained):
        if r['record_id'] in baseindex:matrix[i]=matrices['fit'][baseindex[r['record_id']]]
        else:newidx.append(i)
    torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    for off in range(0,len(newidx),16):
        positions=newidx[off:off+16];matrix[positions]=embed([retained[i] for i in positions])
        if off%1000==0:print('C new embeddings',off,'/',len(newidx),flush=True)
    embedding_seconds=time.perf_counter()-t
    require(np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix,axis=1),1.,atol=1e-5),'Embedding invariant')
    np.save(dest/'fit_embeddings.npy',matrix,allow_pickle=False)
    calindices=[i for i,r in enumerate(original['calibration']) if r['record_id'] not in excluded]
    with threadpool_limits(limits=1):
        clf=LogisticRegression(C=1.,class_weight=None,random_state=42,max_iter=2000)
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');clf.fit(matrix,[r['spam_label'] for r in retained])
        fit_seconds=time.perf_counter()-t;require(list(clf.classes_)==[0,1],'LR direction')
        raw={}
        for part,rows,vectors in [('dev',original['dev'],matrices['dev']),('calibration',cal,matrices['calibration'][calindices])]:
            logits=clf.decision_function(vectors)
            raw[part]=[{k:r[k] for k in ['record_id','domain','source','spam_label']}|{'logit_difference':float(z)} for r,z in zip(rows,logits)]
            write_rows(dest/f'{part}_raw.jsonl',raw[part])
        fitted=fit_temperature([r['spam_label'] for r in raw['calibration']],[r['logit_difference'] for r in raw['calibration']],.5,5.)
        T=fitted['applied_temperature'];threshold=choose_threshold(raw['dev'],T)
        joblib.dump(clf,dest/'classifier.joblib');loaded=joblib.load(dest/'classifier.joblib')
        error=float(np.max(np.abs(loaded.decision_function(matrices['dev'])-clf.decision_function(matrices['dev']))));require(error<=1e-12,'Reload changed logits')
        probe=sorted(range(len(original['dev'])),key=lambda i:digest([42,'larger-fit-reload',original['dev'][i]['record_id']]))[:8]
        fresh=embed([original['dev'][i] for i in probe]);embedding_error=float(np.max(np.abs(fresh-matrices['dev'][probe])))
        require(embedding_error<=1e-5,'Reloaded encoder/cache mismatch')
        manifest={'frozen_utc':now(),'experiment':'C_larger_inner_fit','not_default':True,'fit_rows':len(retained),'new_embeddings':len(newidx),
            'source':m['source'],'encoder_reference':m['encoder_reference'],'tokenizer_path':m['tokenizer_path'],'builder':m['builder'],
            'prefix':'query: ','max_tokens':512,'pooling':m['pooling'],'encoder_frozen':True,'temperature':T,'threshold':threshold['threshold'],
            'calibration':fitted,'threshold_selection':threshold,'fit_parameters':clf.get_params(),'fit_iterations':clf.n_iter_.tolist(),
            'fit_seconds':fit_seconds,'new_embeddings_seconds':embedding_seconds,'warnings':[str(w.message) for w in caught],
            'classifier_path':str((dest/'classifier.joblib').relative_to(ROOT)),'classifier_sha256':sha_file(dest/'classifier.joblib'),
            'fit_membership_sha256':sha_file(dest/'fit_membership.jsonl'),'fit_embeddings_sha256':sha_file(dest/'fit_embeddings.npy'),
            'reload':{'dev_rows':600,'max_logit_error':error,'fresh_encoder_probe_rows':8,'max_embedding_error':embedding_error},
            'cached_fit_reused':6000,'dev_rows':600,'calibration_rows':999,
            'versions':{n:importlib.metadata.version(n) for n in ['scikit-learn','numpy','scipy','joblib','torch','transformers']}}
        write_json(dest/'candidate_manifest.json',manifest)
        # Only now read the frozen pilot. No pilot values influence fit/calibration/threshold.
        pilot=read_rows(out/'private_pilot_inputs.jsonl');vmeta=read_json(EXTRA/'C/validation_embeddings.json')['parts']['validation']
        vrows=read_rows(EXTRA/'C/private_validation_inputs.jsonl');vindex={r['record_id']:i for i,r in enumerate(vrows)}
        require(sha_file(ROOT/vmeta['matrix_path'])==vmeta['sha256'],'Validation cache changed')
        require(sha_file(EXTRA/'C/private_validation_inputs.jsonl')==vmeta['input_sha256'],'Validation cache input changed')
        vmat=np.load(ROOT/vmeta['matrix_path'],allow_pickle=False);pvec=np.empty((len(pilot),384),dtype=np.float32);missing=[]
        for i,r in enumerate(pilot):
            if r['record_id'] in vindex:
                j=vindex[r['record_id']];require(r['text']==vrows[j]['text'],'Pilot validation cached text mismatch');pvec[i]=vmat[j]
            else:missing.append(i)
        for off in range(0,len(missing),16):
            positions=missing[off:off+16];pvec[positions]=embed([pilot[i] for i in positions])
        logits=loaded.decision_function(pvec)
        predictions=[{'record_id':r['record_id'],'model_id':'C_larger_fit','variant':'original_prefix512',
            'score':sigmoid(float(z)/T),'decision':int(sigmoid(float(z)/T)>=threshold['threshold']),
            'logit_difference':float(z),'temperature':T,'threshold':threshold['threshold'],'status':'ok'} for r,z in zip(pilot,logits)]
        write_rows(out/'C_larger_fit_predictions.jsonl',predictions)
    write_json(dest/'resources.json',{'wall_seconds':time.perf_counter()-started,'new_embeddings_seconds':embedding_seconds,
        'fit_seconds':fit_seconds,'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),
        'baseline_6000_embeddings_reused':True,'pilot_validation_256_embeddings_reused':True,'pilot_personal_embeddings_computed':len(missing)})
    write_json(dest/'accesses.json',boundary.report());print('C_LARGER_FIT COMPLETE',len(retained),T,threshold['threshold'],flush=True)


if __name__=='__main__':main()
