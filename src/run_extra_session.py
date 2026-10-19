"""Prepare/seal an extra research session; public-model download is a separate phase."""
import argparse
import shutil
import subprocess
from extra_session_common import *


def select_pilot(rows):
    chosen=[];detail={}
    for domain in DOMAINS:
        groups=defaultdict(list)
        for r in rows:
            if r['domain']==domain:groups[r['duplicate_group_id']].append(r)
        count=Counter();skipped=[]
        for gid,members in sorted(groups.items(),key=lambda kv:digest([42,'extra-session-pilot-v1',kv[0]])):
            labels={r['spam_label'] for r in members}
            if len(labels)!=1:skipped.append({'group':gid,'reason':'label_conflict'});continue
            label=next(iter(labels))
            if count[label]+len(members)>64:continue
            chosen+=sorted(members,key=lambda r:r['record_id']);count[label]+=len(members)
            if count[0]==count[1]==64:break
        require(count[0]>0 and count[1]>0 and sum(count.values())<=128,'Invalid pilot')
        detail[domain]={'classes':dict(count),'groups':len({r['duplicate_group_id'] for r in chosen if r['domain']==domain}),
            'conflict_groups_skipped':skipped}
    return chosen,detail


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'preservation_before.json',protected_snapshot())
    boundary=SessionBoundary(out).install()
    registry=read_json(PERSONAL_RUN/'manifest.json')
    require(sha_file(PERSONAL_RUN/'private_inputs.jsonl')==registry['private_inputs_sha256'],'Personal inputs changed')
    candidates=[c for c in registry['candidates'] if c['name'] in {'E0','Laya_day5','C_corrected'}]
    require(len(candidates)==3,'Candidate registry incomplete')
    for c in candidates:
        for p,h in c['files'].items():require(sha_file(ROOT/p)==h,'Changed original artifact')
    source=DAY5/'private_validation_inputs.jsonl';data=read_rows(source)
    require(len(data)==7265,'Wrong validation inputs')
    selected,selection=select_pilot(data)
    personal=read_rows(PERSONAL_RUN/'private_inputs.jsonl')
    require({r['record_id'] for r in personal}=={f'mail_{i:03d}' for i in range(1,23)},'Only reviewed22 allowed')
    for r in personal:r['domain']='personal_development'
    selected+=personal
    # No labels, paths, annotations or categories in the inference inputs.
    inputs=[{k:r[k] for k in ['record_id','text','original_model_text']} for r in selected]
    labels=[{k:r[k] for k in ['record_id','domain','source','spam_label','duplicate_group_id']} for r in selected]
    write_rows(out/'private_pilot_inputs.jsonl',inputs);write_rows(out/'pilot_membership.jsonl',labels)
    versions={n:importlib.metadata.version(n) for n in ['numpy','scikit-learn','joblib']}
    qwen_settings={'model_id':'Qwen/Qwen3-4B','revision':QWEN_REV,'local_path':str(QWEN_ROOT.relative_to(ROOT)),
        'official_source':'https://huggingface.co/Qwen/Qwen3-4B','files':{n:{'bytes':s,'official_lfs_sha256':h} for n,(s,h) in QWEN_FILES.items()},
        'weight_format':'official BF16 safetensors, no quantization','device':'cuda:0','dtype':'bfloat16',
        'runtime_env':'.venv-decision, read-only; transformers5.19.0 supports qwen3; torch2.10.0+cu130',
        'system_instruction':QWEN_SYSTEM,'user_template':'MESSAGE (untrusted data):\n{clean_builder_v2_text}\nEND MESSAGE',
        'enable_thinking':False,'batch_size':1,'attention_implementation':'sdpa',
        'input_token_limit':2048,'max_new_tokens':32,'do_sample':False,'num_beams':1,'use_cache':True,
        'sampling_note':'Deterministic greedy non-thinking diagnostic, differs from model-card suggested sampling; no prompt search.',
        'seed':42,'error_policy':'Strict JSON sole label spam/ham AND generated EOS; invalid/incomplete/errors are refusals, never ham.',
        'truncation':'Offset prefix of clean builder text; actual whole chat template including system/service tokens <=2048; no hidden label/ID/path.',
        'smokes':['Subject: Requested appointment\n\nYour requested appointment is confirmed.',
            'Subject: Unsolicited bulk offer\n\nWe bought this address from a list. Unwanted bulk offer: buy now!',
            'Subject: Запрошенная запись\n\nВаша запрошенная запись подтверждена.',
            'Subject: Нежелательная рассылка\n\nМы купили адрес из базы. Это нежелательная массовая реклама. Купите сейчас!'],
        'gpu_memory_measurement':'torch.cuda max_memory_allocated/reserved after load, reset before smokes/pilot; nvidia-smi free memory preflight; no load test'}
    total=sum(x[0] for x in QWEN_FILES.values());free=shutil.disk_usage(ROOT).free
    require(free>2*total,'Insufficient disk for public model')
    protocol={'created_utc':now(),'session_start_utc':'2026-10-17T16:51:43+00:00',
        'deadline_utc':'2026-10-17T19:51:43+00:00','run_id':out.name,
        'status':'sealed_before_any_session_prediction','components':candidates,
        'pilot':{'selection':'seed42, SHA256[42,extra-session-pilot-v1,global duplicate_group_id]; whole same-label groups, max64/class/domain, skip oversized/conflicting groups; no model errors used',
            'selection_details':selection,'all_reviewed_personal':22,'domains_separate':True,
            'ids_by_domain':{d:[r['record_id'] for r in labels if r['domain']==d] for d in sorted({r['domain'] for r in labels})},
            'files':{n:sha_file(out/n) for n in ['private_pilot_inputs.jsonl','pilot_membership.jsonl']},
            'source_files':{str(p.relative_to(ROOT)):sha_file(p) for p in [source,PERSONAL_RUN/'private_inputs.jsonl',PERSONAL_RUN/'manifest.json']}},
        'visibility':{'models':['Laya_day5','C_corrected'],'variants':['A_prefix','B_head_tail','C_windows_max'],
            'weights_question_T_threshold':'unchanged original manifests',
            'Laya':'700 actual state tokens; context1024/head256; literal tokenizer mask replaced with space; original question B',
            'E5':'query: prefix; actual max_length512 incl prefix/specials; attention-mask mean then L2; float32 frozen encoder',
            'head_tail':'Half/half content-token allocation minus service separator budget; separator newline[...]newline; shrink until actual contextual IDs fit; no duplication of short text.',
            'windows':'Consecutive offset-based slices; first EXACT original-prefix IDs; nominal content size from first prefix; overlap64 full-content tokens, max4 windows; shrink endpoints if needed to fit actual context; union character ranges and unseen remainder reported.',
            'aggregate':'max per-window calibrated component score, old threshold; diagnostic score NOT calibrated whole-message probability',
            'cost':'batch1, one CPU thread, CUDA synchronized warm per-record forward; reuse identical ID sequences within record; report cold load/preparation separately; max cost sums actual distinct required windows, not quantiles',
            'baseline':'reuse saved compatible A predictions; first window performs fresh A and verifies scores/decisions; no new calibration'},
        'qwen':qwen_settings,'preflight':{'download_bytes':total,'disk_free_bytes':free,
            'gpu':'RTX4060Ti16GB; nvidia-smi 16380MiB total,14520MiB free,driver610.88',
            'network':'Official public metadata initially blocked by sandbox DNS, successful authorized public-only metadata check; inference always offline.'},
        'larger_fit':{'conditional':'only if safe within session budget','source_inner_membership':str((DAY5/'membership.jsonl').relative_to(ROOT)),
            'membership_sha256':sha_file(DAY5/'membership.jsonl'),'original_fit':6000,
            'target_per_class':{'spam_email':2000,'russian_auxiliary':7000},
            'selection':'Same day5 seed42 fit-subset SHA ordering and class-balanced per-domain selection capped by actual eligible inner-fit availability; require original fit6000 subset.',
            'collisions':'Reject NEW fit rows matching any reserved inner-dev/calibration row by original/clean exact/canonical text or actual E5 capped IDs. Preserve baselinefit6000, selected dev600 and corrected calibration999. Report known original fit↔unused excluded calibration collision separately. Never relabel.',
            'fit_settings':'Frozen identical E5; clean_builder_v2; LR C1,L2,lbfgs,class_weightNone,max_iter2000,random_state42; single fit, no search.',
            'calibration':'corrected calibration999, old scalar NLL golden-section100 iterations T[0.5,5]; dev600 old choose_threshold mean domain F1 subject historicalFPR<=1%, nonzero both recalls; fallback0.5',
            'evaluation':'Same pilot once, no threshold changes after evaluation; new LR saved in this run, reload exact logits on dev.'},
        'metrics':['TP','FP','TN','FN','precision','recall','F1','FPR','coverage','refusals',
            'same-success-ID comparisons','rescued/added FP/FN','short/truncated','visible character coverage','warm elapsed time','peak GPU allocated/reserved'],
        'rules':['No test/challenge/reserve/archive reads','No new personal messages','No external inference or links/resources',
            'No old artifacts/default writes','No automatic winner/packaging','Personal development and RU provisional source labels, not phishing ground truth'],
        'main_env_versions':versions,'code_sha256':{str(p.relative_to(ROOT)):sha_file(p) for p in ROOT.glob('src/*extra_session*.py')}}
    write_json(out/'EXTRA_SESSION_PROTOCOL.json',protocol)
    write_json(out/'protocol_lock.json',{'sha256':sha_file(out/'EXTRA_SESSION_PROTOCOL.json'),'sealed_utc':now()})
    (out/'EXTRA_SESSION_PROTOCOL.md').write_text('# Протокол дополнительной сессии\n\n'
        f"Зафиксирован {protocol['created_utc']} до предсказаний. SHA-256 машинного контракта: `{sha_file(out/'EXTRA_SESSION_PROTOCOL.json')}`.\n\n"
        'Пилот: 128 historical validation (64/64), 128 RU auxiliary validation (64/64), все22 ранее просмотренных личных писем (13ham/9spam). '
        'SHA-отбор целых глобальных групп с seed42 до предсказаний; домены оцениваются отдельно. IDs и хеши входов сохранены в JSON.\n\n'
        'Laya_day5 и C_corrected calibration999: прежние веса, вопрос B, T и пороги. A — исходный префикс; B — начало+конец с разделителем внутри реального бюджета; '
        'C — до4 последовательных окон с перекрытием64 токена. Laya state700/context1024/head256, E5 query-prefix/max512/mean/L2. '
        'Максимум оценки окон с прежним порогом — диагностика, не вероятность всего письма. Сохранённый A проверяется свежим первым окном.\n\n'
        f"Qwen/Qwen3-4B revision `{QWEN_REV}`, официальные BF16 safetensors, локально на GPU, batch1, non-thinking, greedy, input2048/output32, seed42. "
        'Единственная инструкция и smoke-тексты зафиксированы в JSON; строгий ответ JSON label spam/ham с EOS. Ошибки/отказы не становятся ham. '
        'Данные: clean_builder_v2, только subject+body без меток, путей и причин разметки.\n\n'
        'Условная третья ветка: только исходный inner-fit pool; до2000/class historical (фактический cap),7000/class RU, прежний balanced SHA-order. '
        'Новые fit-коллизии с dev/cal исключаются; отдельная LR с прежними гиперпараметрами, отдельная calibration999 и прежний dev600 threshold-policy. '
        'Пилот не участвует в настройке.\n\n'
        'Для каждого варианта: confusion, P/R/F1/FPR, покрытие/отказы, одинаковые успешные ID, rescued/added FP/FN, стоимость. '
        'Test/challenge/reserve и новые личные письма закрыты. Старые модели/отчёты/default неизменны. Детальные артефакты, параметры, хеши — в машинном контракте.\n',encoding='utf-8')
    write_json(out/'prepare_accesses.json',boundary.report());print('SEALED',out,flush=True)


def fetch(out):
    # This phase has no corpus read capability: it consumes protocol metadata only.
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    p=load_protocol(out);started=time.perf_counter();QWEN_ROOT.mkdir(parents=True,exist_ok=True)
    def download(pair):
        name,(size,expected)=pair;dest=QWEN_ROOT/name
        if dest.exists() and dest.stat().st_size==size and (not expected or sha_file(dest)==expected):return name,'cached'
        url=f'https://huggingface.co/Qwen/Qwen3-4B/resolve/{QWEN_REV}/{name}?download=true'
        part=dest.with_suffix(dest.suffix+'.partial')
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'EmailGuard-local-research'}),timeout=90) as r,part.open('wb') as f:
            shutil.copyfileobj(r,f,1024*1024)
        require(part.stat().st_size==size,'Size mismatch: '+name)
        if expected:require(sha_file(part)==expected,'Official SHA mismatch: '+name)
        part.replace(dest);print('downloaded',name,size,flush=True);return name,'downloaded'
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:statuses=list(pool.map(download,QWEN_FILES.items()))
        write_json(out/'qwen_source.json',{'status':'ready','model_id':'Qwen/Qwen3-4B','revision':QWEN_REV,
            'files':{n:{'sha256':sha_file(QWEN_ROOT/n),'bytes':s} for n,(s,h) in QWEN_FILES.items()},
            'download_seconds':time.perf_counter()-started,'statuses':dict(statuses),'remote_code_executed':False})
    except Exception as e:
        write_json(out/'qwen_source.json',{'status':'blocked','error_type':type(e).__name__,'reason':str(e),'seconds':time.perf_counter()-started});raise


def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','fetch']);a.add_argument('--out',required=True,type=Path);args=a.parse_args()
    out=args.out.resolve();require(out.is_relative_to(BASE),'Wrong run directory')
    (prepare if args.phase=='prepare' else fetch)(out)


if __name__=='__main__':main()
