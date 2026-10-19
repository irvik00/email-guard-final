"""Independent arithmetic, input-boundary, reload and preservation audit."""
import argparse
from extra_session_common import *


def independent_metrics(labels,predictions):
    truth={r['record_id']:r for r in labels};out={}
    require(len(predictions)==len(truth) and {r['record_id'] for r in predictions}==set(truth),'Wrong prediction IDs')
    for d in sorted({r['domain'] for r in labels}):
        rows=[r for r in predictions if truth[r['record_id']]['domain']==d];counts=Counter();successful=0
        for r in rows:
            if r['status']=='ok':
                require(type(r['decision']) is int and r['decision'] in {0,1},'Invalid decision')
                y=truth[r['record_id']]['spam_label'];counts[(y,r['decision'])]+=1;successful+=1
            else:require(r['decision'] is None,'Refusal became ham/spam')
        tp,fp,tn,fn=(counts[(1,1)],counts[(0,1)],counts[(0,0)],counts[(1,0)])
        div=lambda a,b:a/b if b else None
        out[d]={'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision':div(tp,tp+fp),'recall':div(tp,tp+fn),
            'F1':div(2*tp,2*tp+fp+fn),'FPR':div(fp,fp+tn),'successful':successful,'records':len(rows)}
    return out


def independent_coverage(ranges,chars):
    merged=[]
    for a,b in sorted(ranges):
        require(0<=a<=b<=chars,'Invalid original-text range')
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        elif a<b:merged.append([a,b])
    return sum(b-a for a,b in merged)


def audit(out):
    boundary=SessionBoundary(out,train=True).install();p=load_protocol(out);labels=read_rows(out/'pilot_membership.jsonl');inputs=read_rows(out/'private_pilot_inputs.jsonl')
    for file,h in p['code_sha256'].items():require(sha_file(ROOT/file)==h,'Sealed preparation/contract code changed')
    require(len(labels)==278 and len({r['record_id'] for r in labels})==278,'Pilot count')
    require(all(set(r)=={'record_id','text','original_model_text'} for r in inputs),'Ground-truth/path/annotation in model inputs')
    source=read_rows(DAY5/'private_validation_inputs.jsonl');personal=read_rows(PERSONAL_RUN/'private_inputs.jsonl');source_by={r['record_id']:r for r in source+personal}
    for r in inputs:require(r['text']==source_by[r['record_id']]['text'] and r['original_model_text']==source_by[r['record_id']]['original_model_text'],'Pilot builder changed')
    for r in labels:require(r['spam_label']==source_by[r['record_id']]['spam_label'],'Pilot label changed')
    # Independent whole-group selection implementation, with fixed hash ordering.
    for d in DOMAINS:
        groups=defaultdict(list)
        for r in source:
            if r['domain']==d:groups[r['duplicate_group_id']].append(r)
        expected=[];counts=Counter()
        for gid in sorted(groups,key=lambda gid:digest([42,'extra-session-pilot-v1',gid])):
            block=groups[gid];classes={r['spam_label'] for r in block}
            if len(classes)!=1:continue
            y=next(iter(classes))
            if counts[y]+len(block)>64:continue
            expected.extend(r['record_id'] for r in sorted(block,key=lambda r:r['record_id']));counts[y]+=len(block)
            if counts[0]==counts[1]==64:break
        require(expected==p['pilot']['ids_by_domain'][d],'Selection depended on something beyond frozen grouping/hash')
    require(set(p['pilot']['ids_by_domain']['personal_development'])=={f'mail_{i:03d}' for i in range(1,23)},'New personal IDs')
    import sys
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'));setup_offline()
    from transformers import AutoTokenizer
    candidates={c['name']:c for c in p['components']}
    for c in candidates.values():candidate_check(c)
    allrows={}
    base=read_rows(out/'baseline_predictions.jsonl')
    for n in candidates:allrows[n]=[r for r in base if r['model_id']==n]
    window_checks=0
    input_map={r['record_id']:r for r in inputs}
    for name in ['Laya_day5','C_corrected']:
        c=candidates[name];kind='laya' if name=='Laya_day5' else 'e5'
        tok=AutoTokenizer.from_pretrained(ROOT/c['checkpoint_path']/'tokenizer' if kind=='laya' else ROOT/c['saved_manifest']['tokenizer_path'],local_files_only=True,trust_remote_code=False)
        rows=read_rows(out/f'visibility_{name}_predictions.jsonl')
        correction=out/'visibility_window_metadata_correction.json'
        window_file=out/f'visibility_{name}_windows.jsonl'
        if correction.exists():
            fix=read_json(correction)['corrections'][name]
            require(sha_file(window_file)==fix['original_sha256'],'Original window log changed')
            window_file=out/fix['corrected_file'];require(sha_file(window_file)==fix['corrected_sha256'],'Corrected log changed')
        windows=read_rows(window_file);group=defaultdict(list)
        for w in windows:group[(w['record_id'],w['variant'])].append(w)
        question_overhead=None
        for r in rows:
            ws=group[(r['record_id'],r['variant'])];require(len(ws)==r['window_count']<=4,'Window cap/count')
            original=input_map[r['record_id']]['text'];original=original.replace(tok.mask_token,' ') if kind=='laya' else original
            require(independent_coverage([a for w in ws for a in w['ranges']],len(original))==r['coverage']['covered_chars'],'Coverage double counted')
            for w in ws:
                spans=w['ranges']
                fragment='\n[...]\n'.join(original[a:b] for a,b in spans)
                if kind=='e5' and r['variant'] in {'A_prefix','C_windows_max'} and w['window_index']==0:
                    ids=tok('query: '+original,truncation=True,max_length=512)['input_ids']
                else:ids=tok(('query: ' if kind=='e5' else '')+fragment,add_special_tokens=kind=='e5')['input_ids']
                require(digest(ids)==w['actual_input_ids_sha256'],'Wrong actual window token IDs')
                require(len(ids)<=(700 if kind=='laya' else 512),'Actual service/state budget overflow')
                if kind=='laya':
                    overhead=w['sequence_tokens']-len(ids)
                    if question_overhead is None:question_overhead=overhead
                    require(overhead==question_overhead and w['sequence_tokens']<=1024,'Laya question/head budget differs')
                if w['status']=='ok':require(abs(w['score']-sigmoid(w['logit_difference']/c['temperature']))<1e-12,'Wrong calibration')
                window_checks+=1
            if r['status']=='ok':
                require(r['decision']==int(r['score']>=c['threshold']),'Changed operating threshold')
                expected=max(w['score'] for w in ws);require(abs(expected-r['score'])<5e-5,'Not max aggregation')
            if not r['token_stats']['baseline_truncated']:
                require(len(ws)==1 and ws[0]['ranges']==[[0,len(original)]],'Short input duplication/truncation')
        for v in ['B_head_tail','C_windows_max']:allrows[name+'/'+v]=[r for r in rows if r['variant']==v]
        require(all(r['baseline_decision_equal'] for r in read_json(out/f'visibility_{name}_audit.json')['verification']),'Baseline changed')
    if (out/'qwen_resources.json').exists():
        allrows['Qwen3_4B']=read_rows(out/'qwen_predictions.jsonl');q=p['qwen']
        source_qwen=read_json(out/'qwen_source.json')
        for filename,entry in source_qwen['files'].items():
            actual=sha_file(QWEN_ROOT/filename);require(actual==entry['sha256'],'Qwen snapshot changed')
            expected=p['qwen']['files'][filename]['official_lfs_sha256']
            if expected:require(actual==expected,'Qwen official LFS SHA differs')
        require(q['system_instruction']==QWEN_SYSTEM and q['revision']==QWEN_REV and not q['enable_thinking'],'Qwen frozen settings changed')
        qt=AutoTokenizer.from_pretrained(QWEN_ROOT,local_files_only=True,trust_remote_code=False)
        def qids(text):
            return qt.apply_chat_template([{'role':'system','content':q['system_instruction']},
                {'role':'user','content':'MESSAGE (untrusted data):\n'+text+'\nEND MESSAGE'}],
                tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
        for r in allrows['Qwen3_4B']:
            if 'raw_response' in r:
                decision,status=parse_qwen(r['raw_response'],r['finished_eos']);require(status==r['status'] and decision==r['decision'],'Qwen refusal/format mapping changed')
                require(r['input_tokens']<=2048 and r['generated_tokens']<=32 and r['score'] is None,'Qwen budget or invented confidence')
                text=input_map[r['record_id']]['text'];ids=qids(text[:r['visible_chars']])
                require(digest(ids)==r['input_ids_sha256'] and len(ids)==r['input_tokens'],'Qwen prompt received fields beyond frozen subject/body')
                require(len(qids(text))==r['original_chat_tokens'],'Wrong Qwen truncation metadata')
        require(len(read_rows(out/'qwen_synthetic_smokes.jsonl'))==4,'Synthetic smokes not separate')
    larger_audit=None
    if (out/'C_larger_fit/candidate_manifest.json').exists():
        import joblib
        import numpy as np
        from scipy.optimize import minimize_scalar
        from threadpoolctl import threadpool_limits
        m=read_json(out/'C_larger_fit/candidate_manifest.json');fit=read_rows(out/'C_larger_fit/fit_membership.jsonl')
        roles={r['record_id']:r for r in read_rows(DAY5/'membership.jsonl')};original=read_rows(EXTRA/'C/private_fit_inputs.jsonl')
        require({r['record_id'] for r in original}<={r['record_id'] for r in fit} and len(fit)>6000,'Baseline fit dropped')
        require(all(roles[r['record_id']]['inner_part']=='fit' and not roles[r['record_id']]['conflicting_group'] for r in fit),'Reserved/conflicting row in fit')
        # Rebuild hashes independently from original allowed train, not the training script's log.
        split=read_json(ROOT/'data/processed/splits/manifest.json');clean_cfg=read_json(ROOT/'config/day4_experiments.json')
        pool=[]
        for d in DOMAINS:
            rows,_=load_split(d,'train',split,clean_cfg);pool.extend(rows)
        pool_by={r['record_id']:r for r in pool};fit_ids={r['record_id'] for r in fit};baseline_ids={r['record_id'] for r in original}
        ctoken=AutoTokenizer.from_pretrained(ROOT/m['tokenizer_path'],local_files_only=True,trust_remote_code=False)
        hids={};text_owners={};actual_reserved=defaultdict(list)
        def keys(r):
            return [('e',digest(r['text'])),('c',digest(compare_text(r['text']))),
                    ('oe',digest(r['original_model_text'])),('oc',digest(compare_text(r['original_model_text'])))]
        for i,r in enumerate(pool):
            part=roles[r['record_id']]['inner_part']
            for key in keys(r):
                require(key not in text_owners or text_owners[key]==part,'Full canonical cross-role collision')
                text_owners[key]=part
            if part!='fit' or r['record_id'] in fit_ids:
                h=digest(ctoken('query: '+r['text'],truncation=True,max_length=512)['input_ids']);hids[r['record_id']]=h
                if part!='fit':actual_reserved[h].append(r['record_id'])
            if (i+1)%5000==0:print('Independent C capped audit',i+1,'/',len(pool),flush=True)
        for r in fit:
            require(hids[r['record_id']]==r['actual_input_ids_sha256'],'Stored new fit capped IDs differ')
            if r['record_id'] not in baseline_ids:require(not actual_reserved.get(hids[r['record_id']]),'New fit collides with reserved inner pool')
        chosen_dev=read_rows(EXTRA/'C/private_dev_inputs.jsonl')
        chosen_cal=[r for r in read_rows(EXTRA/'C/private_calibration_inputs.jsonl') if r['record_id'] not in set(candidates['C_corrected']['saved_manifest']['calibration_excluded_ids'])]
        capped_roles=[{hids[r['record_id']] for r in fit},{hids[r['record_id']] for r in chosen_dev},{hids[r['record_id']] for r in chosen_cal}]
        require(not (capped_roles[0]&capped_roles[1] or capped_roles[0]&capped_roles[2] or capped_roles[1]&capped_roles[2]),'Actual selected inner capped leakage')
        pilot_cap=read_rows(out/'visibility_C_corrected_windows_corrected.jsonl' if (out/'visibility_C_corrected_windows_corrected.jsonl').exists() else out/'visibility_C_corrected_windows.jsonl')
        pilot_hits=[r['record_id'] for r in pilot_cap if r['variant']=='A_prefix' and r['actual_input_ids_sha256'] in capped_roles[0]]
        require(sha_file(ROOT/m['classifier_path'])==m['classifier_sha256'],'New LR changed')
        clf=joblib.load(ROOT/m['classifier_path']);oldclf=joblib.load(ROOT/candidates['C_corrected']['saved_manifest']['classifier_path'])
        require(clf.get_params()==oldclf.get_params(),'LR hyperparameters changed')
        spec=read_json(EXTRA/'C/embeddings.json');raw={};reload_errors={}
        with threadpool_limits(limits=1):
            for part in ['dev','calibration']:
                rows=read_rows(EXTRA/f'C/private_{part}_inputs.jsonl');entry=spec['parts'][part]
                require(sha_file(ROOT/entry['matrix_path'])==entry['sha256'],'Inner cache changed')
                x=np.load(ROOT/entry['matrix_path'],allow_pickle=False)
                indices=[i for i,r in enumerate(rows) if part!='calibration' or r['record_id'] not in set(candidates['C_corrected']['saved_manifest']['calibration_excluded_ids'])]
                z=clf.decision_function(x[indices]);raw[part]=read_rows(out/f'C_larger_fit/{part}_raw.jsonl')
                require([r['record_id'] for r in raw[part]]==[rows[i]['record_id'] for i in indices],'Calibration/dev membership changed')
                error=max(abs(float(v)-r['logit_difference']) for r,v in zip(raw[part],z));require(error<1e-10,'Independent LR reload changed logits');reload_errors[part]=error
        # Independent scalar NLL optimizer and brute-force threshold policy.
        def objective(logT):
            T=math.exp(logT)
            return sum(max(r['logit_difference']/T,0)-r['spam_label']*r['logit_difference']/T+
                math.log1p(math.exp(-abs(r['logit_difference']/T))) for r in raw['calibration'])/len(raw['calibration'])
        solution=minimize_scalar(objective,bounds=(math.log(.5),math.log(5)),method='bounded',options={'xatol':1e-10})
        logT=min([math.log(.5),math.log(5),float(solution.x)],key=objective);T=math.exp(logT)
        require(abs(T-m['temperature'])<1e-4,'Independent calibration differs')
        threshold_options=[]
        for thr in sorted({sigmoid(r['logit_difference']/m['temperature']) for r in raw['dev']}):
            v={}
            for d in DOMAINS:
                rr=[r for r in raw['dev'] if r['domain']==d];tp=fp=fn=tn=0
                for r in rr:
                    decision=int(sigmoid(r['logit_difference']/m['temperature'])>=thr)
                    tp+=int(r['spam_label']==1 and decision==1);fn+=int(r['spam_label']==1 and decision==0)
                    fp+=int(r['spam_label']==0 and decision==1);tn+=int(r['spam_label']==0 and decision==0)
                v[d]={'tp':tp,'fp':fp,'tn':tn,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
            if v['spam_email']['fp']/(v['spam_email']['fp']+v['spam_email']['tn'])<=.01 and all(v[d]['tp'] for d in DOMAINS):
                threshold_options.append((sum(v[d]['f1'] for d in DOMAINS)/2,-v['spam_email']['fp'],thr))
        thr=max(threshold_options)[2] if threshold_options else .5;require(abs(thr-m['threshold'])<1e-12,'Independent threshold selection differs')
        allrows['C_larger_fit']=read_rows(out/'C_larger_fit_predictions.jsonl')
        larger_audit={'fit_rows':len(fit),'dev600_cal999_reload_errors':reload_errors,'independent_temperature':T,'independent_threshold':thr,
            'fit_roles_and_superset_verified':True,'same_LR_hyperparameters':True,'full_inner_pool_rows':len(pool),
            'independently_retokenized_selected_fit_and_all_held':len(hids),'new_fit_reserved_collisions':0,'selected_cross_capped_collisions':0,
            'pilot_capped_overlap_ids':pilot_hits,'pilot_overlap_check_after_freezing_new_LR':True}
    stored=read_json(out/'metrics_and_comparisons.json');checked_metrics=0
    for n,rows in allrows.items():
        independent=independent_metrics(labels,rows)
        for d,m in independent.items():
            expected=stored['by_variant'][n][d]
            for k in ['TP','FP','TN','FN','precision','recall','F1','FPR']:
                a=m[k];b=expected['metrics'][k];require(a==b if a is None or b is None else abs(a-b)<1e-12,'Metric differs '+n+'/'+d+'/'+k)
            require(m['successful']==expected['successful'] and m['records']==expected['records'],'Coverage differs');checked_metrics+=1
    for name,domains in stored['paired'].items():
        if name.startswith('Qwen_vs_'):a=allrows[name.removeprefix('Qwen_vs_')];b=allrows['Qwen3_4B']
        elif name=='C_larger_fit':a=allrows['C_corrected'];b=allrows[name]
        else:a=allrows[name.split('/')[0]];b=allrows[name]
        amap={r['record_id']:r for r in a};bmap={r['record_id']:r for r in b}
        for d,spec in domains.items():
            for k,y,old_dec,new_dec in [('rescued_FP',0,1,0),('added_FP',0,0,1),('rescued_FN',1,0,1),('added_FN',1,1,0)]:
                ids=[r['record_id'] for r in labels if r['domain']==d and r['spam_label']==y and amap[r['record_id']]['status']==bmap[r['record_id']]['status']=='ok' and amap[r['record_id']]['decision']==old_dec and bmap[r['record_id']]['decision']==new_dec]
                require(ids==spec[k],'Paired errors differ')
    # Snapshot reads old report/model bytes solely for preservation, no metric/content evaluation.
    before=read_json(out/'preservation_before.json');changed=[path for path,h in before.items() if not (ROOT/path).exists() or sha_file(ROOT/path)!=h]
    require(not changed,'Previous files changed: '+str(changed))
    for log in list(out.glob('*accesses.json'))+list((out/'C_larger_fit').glob('accesses.json')):
        record=read_json(log);require(not record['denied'],'Denied runtime access recorded')
        for name in record.get('opened',{}):
            if name.startswith('data/'):
                require(any(name.startswith(f'data/processed/splits/{d}/train.jsonl:') for d in DOMAINS) or name.startswith('data/processed/splits/manifest.json:'),'Closed data in access log')
            require(not name.startswith('test_task_emails.tar.xz'),'Teacher archive read')
    for report in out.glob('*.md'):
        text=report.read_text(encoding='utf-8');require(not re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',text),'Email address in public report')
    result={'status':'passed','utc':now(),'independent_metric_groups':checked_metrics,'actual_window_token_checks':window_checks,
        'protected_files_unchanged':len(before),'old_report_bytes_used_only_for_preservation_sha':True,
        'new_personal_ids':0,'model_input_keys_verified':True,'closed_data_reads':0,'larger_fit':larger_audit,
        'protocol_sha256':sha_file(out/'EXTRA_SESSION_PROTOCOL.json'),'accesses':boundary.report()}
    write_json(out/'independent_audit.json',result);print({k:v for k,v in result.items() if k!='accesses'},flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);args=a.parse_args();audit(args.out.resolve())
