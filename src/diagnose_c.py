"""Append-only C diagnostics. No fitting of encoder/LR and no outer data access."""
import argparse
import contextlib
import importlib.metadata
import shutil
import subprocess
import time
from day5_common import *

ORIGINAL = ROOT/'reports/extra_experiments/runs/20261012T095633Z-7d01f7b8/C'
STAGES = ('tokenization', 'host_to_cuda', 'encoder', 'pooling', 'cuda_to_cpu',
          'sklearn_predict', 'calibration', 'serialization')

class DiagnosticBoundary(Boundary):
    def check(self, event, args):
        super().check(event, args)
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        mode = str(args[1] or ''); flags = int(args[2] or 0) if len(args)>2 else 0
        protected = any(p.is_relative_to(ROOT/k) for k in
            ['reports/day3','reports/day4','reports/day5','reports/extra_experiments',
             'reports/laya_diagnostics','models/day3','models/day4','models/day5','models/extra_experiments'])
        if protected and (any(c in mode for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR)):
            raise PermissionError('Original artifact is immutable: '+str(p))

def cross_collisions(parts, key):
    owners = defaultdict(list)
    for part, rows in parts.items():
        for r in rows: owners[key(r)].append((part,r))
    return [v for v in owners.values() if len({p for p,r in v})>1]

def corrected_calibration(rows, excluded):
    require(excluded <= {r['record_id'] for r in rows}, 'Unknown exclusion')
    return [r for r in rows if r['record_id'] not in excluded]

def fixed_sample(rows):
    return sorted(rows,key=lambda r:digest([42,'latency',r['record_id']]))[:64]

def snapshot():
    files = set()
    for folder in ['reports','models','src','tests','config','docs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not any(
                p.is_relative_to(ROOT/k) for k in ['reports/c_diagnostics','models/c_diagnostics']):
                if p != ROOT/'reports/C_DIAGNOSTICS.md': files.add(p)
    files.update(ROOT.glob('requirements*.txt'))
    files.update(ROOT/e/'pyvenv.cfg' for e in ['.venv','.venv-decision'])
    return {p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted(files) if p.exists()}

def runtime():
    os.environ.update(HF_HOME=str(ROOT/'.cache/extra/hf'),HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',USE_TF='0',TOKENIZERS_PARALLELISM='false')
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'))

def analyze(out):
    runtime(); boundary=DiagnosticBoundary('train').install()
    import numpy as np
    import joblib
    from transformers import AutoTokenizer
    manifest=read_json(ORIGINAL/'candidate_manifest.json')
    spec=read_json(ORIGINAL/'embeddings.json')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/manifest['tokenizer_path'],local_files_only=True,trust_remote_code=False)
    inputs={}; visible={}; matrix={}
    for part in ['fit','dev','calibration']:
        path=ORIGINAL/f'private_{part}_inputs.jsonl'
        require(sha_file(path)==manifest['input_sha256'][part], 'Input changed')
        inputs[part]=read_rows(path); visible[part]=read_rows(ORIGINAL/f'private_{part}_visible.jsonl')
        require([r['record_id'] for r in inputs[part]]==[r['record_id'] for r in visible[part]], 'Visible alignment')
        entry=spec['parts'][part]; require(sha_file(ROOT/entry['matrix_path'])==entry['sha256'],'Cache changed')
        matrix[part]=np.load(ROOT/entry['matrix_path'],mmap_mode='r',allow_pickle=False)
        # Recompute actual model inputs for every selected inner row, not standalone cut fragments.
        for r,v in zip(inputs[part],visible[part]):
            ids=tokenizer('query: '+r['text'],truncation=True,max_length=512)['input_ids']
            require(digest(ids)==v['token_stats']['actual_input_ids_sha256'], 'Actual token hash mismatch')
    canonical_groups=cross_collisions(visible,lambda r:digest(compare_text(r['visible_fragment'])))
    token_groups=cross_collisions(visible,lambda r:r['token_stats']['actual_input_ids_sha256'])
    require(len(canonical_groups)==1 and len(token_groups)==1,'Expected single collision; revise protocol if changed')
    group=token_groups[0]; require({p for p,r in group}=={'fit','calibration'}, 'Unexpected roles')
    require({r['record_id'] for p,r in group}=={r['record_id'] for p,r in canonical_groups[0]},'Different collision sets')
    records=[]; vectors=[]; full=[]
    for part,v in group:
        i=next(i for i,r in enumerate(inputs[part]) if r['record_id']==v['record_id']); r=inputs[part][i]
        vectors.append(matrix[part][i]); full.append(r['text'])
        records.append({k:r[k] for k in ['record_id','source','domain','spam_label','duplicate_group_id','inner_group_id','original_locator']} |
            {'part':part,'row_index_zero_based':i,'text_sha256':digest(r['text']),
             'canonical_full_sha256':digest(compare_text(r['text'])),'visible_sha256':v['visible_sha256'],
             'token_stats':v['token_stats']})
    excluded={r['record_id'] for p,r in group if p=='calibration'}
    cal=read_rows(ORIGINAL/'calibration_raw.jsonl'); dev=read_rows(ORIGINAL/'dev_raw.jsonl')
    clean=corrected_calibration(cal,excluded); require(len(clean)==999,'Unexpected calibration size')
    filtered={p:[r for r in rows if not(p=='calibration' and r['record_id'] in excluded)] for p,rows in visible.items()}
    require(not cross_collisions(filtered,lambda r:digest(compare_text(r['visible_fragment']))),'Canonical overlap remains')
    require(not cross_collisions(filtered,lambda r:r['token_stats']['actual_input_ids_sha256']),'Token overlap remains')
    fitted=fit_temperature([r['spam_label'] for r in clean],[r['logit_difference'] for r in clean],.5,5.)
    # These are the already frozen C bounds, not any SDK constraint.
    fitted['fixed_experimental_bounds']=fitted.pop('sdk_limits')
    fitted['optimum_outside_fixed_range']=fitted.pop('optimum_outside_sdk_range')
    temperature=fitted['applied_temperature']; operating=choose_threshold(dev,temperature)
    variant=read_json(ORIGINAL/'candidate_manifest.json')
    variant.update(experiment='C_calibration_disjoint_diagnostic',frozen_utc=now(),
        original_manifest_sha256=sha_file(ORIGINAL/'candidate_manifest.json'),temperature=temperature,
        threshold=operating['threshold'],calibration=fitted,threshold_selection=operating,
        calibration_rows=999,calibration_excluded_ids=sorted(excluded),calibration_membership_sha256=digest([r['record_id'] for r in clean]),
        encoder_or_classifier_refitted=False,outer_validation_evaluated=False,
        original_membership_sha256=variant.pop('membership_sha256'),
        input_sha256={p:sha_file(ORIGINAL/f'private_{p}_inputs.jsonl') for p in ['fit','dev']},
        calibration_logits_sha256=sha_file(ORIGINAL/'calibration_raw.jsonl'),
        calibration_rule='Original calibration order minus excluded IDs; no replacement',not_default=True)
    dest=ROOT/'models/c_diagnostics'/out.name; dest.mkdir(parents=True)
    shutil.copyfile(ROOT/manifest['classifier_path'],dest/'classifier.joblib')
    require(sha_file(dest/'classifier.joblib')==manifest['classifier_sha256'],'Copied classifier changed')
    variant['classifier_path']=(dest/'classifier.joblib').relative_to(ROOT).as_posix()
    write_rows(out/'corrected_calibration_logits.jsonl',clean)
    write_json(dest/'candidate_manifest.json',variant); write_json(out/'corrected_variant.json',variant)
    prior=joblib.load(ROOT/manifest['classifier_path']); reloaded=joblib.load(dest/'classifier.joblib')
    logits=reloaded.decision_function(matrix['dev']); old=prior.decision_function(matrix['dev'])
    require(np.array_equal(logits,old),'Copied classifier reload differs')
    require(np.max(np.abs(logits-np.array([r['logit_difference'] for r in dev])))<=1e-4,'Stored logits differ')
    report={'status':'passed','records':records,'cross_inner_canonical_visible_groups':len(canonical_groups),
        'cross_inner_actual_token_groups':len(token_groups),'all_inner_token_hashes_recomputed':7600,
        'actual_input_identical':True,'label_conflict':len({r['spam_label'] for r in records})>1,
        'same_original_group':records[0]['duplicate_group_id']==records[1]['duplicate_group_id'],
        'full_text_equal':full[0]==full[1],'full_canonical_equal':compare_text(full[0])==compare_text(full[1]),
        'embedding_max_abs_difference':float(np.max(np.abs(vectors[0]-vectors[1]))),
        'meaning':'Distinct full records collapse to the same capped E5 input; fit/calibration dependence, not conflicting labels',
        'corrected_cross_inner_groups':0,'excluded_ids':sorted(excluded),'counts_before':counts(cal),'counts_after':counts(clean),
        'original_temperature':manifest['temperature'],'original_threshold':manifest['threshold'],
        'corrected_temperature':temperature,'corrected_threshold':operating['threshold'],
        'original_calibration':manifest['calibration'],'corrected_calibration':fitted,
        'dev_decisions_changed':sum((sigmoid(r['logit_difference']/temperature)>=operating['threshold']) !=
            (sigmoid(r['logit_difference']/manifest['temperature'])>=manifest['threshold']) for r in dev),
        'dev_metrics':operating['by_domain'],'reload_max_logit_difference':float(np.max(np.abs(logits-old))),
        'no_outer_evaluation':True,'accesses':boundary.report()}
    write_json(out/'collision_analysis.json',report)
    print('Collision audited; actual tokens identical; calibration999 saved; T',temperature,'threshold',operating['threshold'],flush=True)

def quantiles(rows,key):
    import numpy as np
    a=np.asarray([r[key]*1000 for r in rows])
    return {'n':len(a),'p50_ms':float(np.quantile(a,.5)),'p95_ms':float(np.quantile(a,.95)),
            'mean_ms':float(a.mean()),'min_ms':float(a.min()),'max_ms':float(a.max())}

def profile(out,mode):
    runtime(); boundary=DiagnosticBoundary('train').install()
    import torch
    import joblib
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_info,threadpool_limits
    from transformers import AutoTokenizer,AutoModel
    manifest=read_json(ORIGINAL/'candidate_manifest.json'); source=manifest['source']
    require({n:importlib.metadata.version(n) for n in manifest['versions']}==manifest['versions'],'Version mismatch')
    require(sha_file(ROOT/manifest['classifier_path'])==manifest['classifier_sha256'],'Classifier changed')
    for name,spec in source['files'].items():require(sha_file(Path(source['local_path'])/name)==spec['sha256'],'Source changed')
    classifier=joblib.load(ROOT/manifest['classifier_path']); torch.set_num_threads(1)
    if mode=='one': torch.set_num_interop_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/manifest['tokenizer_path'],local_files_only=True,trust_remote_code=False)
    encoder=AutoModel.from_pretrained(source['local_path'],local_files_only=True,trust_remote_code=False).to('cuda').eval()
    for p in encoder.parameters():p.requires_grad_(False)
    data=fixed_sample(read_rows(ORIGINAL/'private_dev_inputs.jsonl'))
    require([r['record_id'] for r in data]==[r['record_id'] for r in read_json(ORIGINAL/'warm_end_to_end.json')['rows']],'Profiling membership changed')
    previous={r['record_id']:r for r in read_rows(ORIGINAL/'dev_raw.jsonl')}
    before=threadpool_info(); pools=threadpool_limits(limits=1) if mode=='one' else contextlib.nullcontext()
    def predict(r,instrument=False):
        timings={}; total=time.perf_counter(); last=total
        def mark(name,sync=False):
            nonlocal last
            if sync:torch.cuda.synchronize()
            end=time.perf_counter(); timings[name]=end-last; last=end
        tokens=tokenizer('query: '+r['text'],truncation=True,max_length=512,padding=True,return_tensors='pt')
        if instrument:mark('tokenization')
        tokens=tokens.to('cuda')
        if instrument:mark('host_to_cuda',True)
        with torch.no_grad():
            hidden=encoder(**tokens).last_hidden_state
            if instrument:mark('encoder',True)
            mask=tokens['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1)
            pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
            if instrument:mark('pooling',True)
            vector=pooled.cpu().numpy()
            if instrument:mark('cuda_to_cpu',True)
        logit=float(classifier.decision_function(vector)[0])
        if instrument:mark('sklearn_predict')
        prob=sigmoid(logit/manifest['temperature']); decision=prob>=manifest['threshold']
        if instrument:mark('calibration')
        payload=canonical({'record_id':r['record_id'],'logit':logit,'probability':prob,'spam':decision})
        if instrument:mark('serialization')
        torch.cuda.synchronize(); elapsed=time.perf_counter()-total
        error=abs(logit-previous[r['record_id']]['logit_difference']) if r['record_id'] in previous else 0.
        require(error<=1e-4,'Numerical regression')
        return {'record_id':r['record_id'],'domain':r['domain'],'seconds':elapsed,'stage_seconds':timings,
                'logit':logit,'probability':prob,'spam':decision,'raw_logit_abs_error':error,'serialized_bytes':len(payload)}
    began=time.perf_counter()
    with pools:
        active=threadpool_info()
        # Identical five synthetic warm-ups in each independent process; no optimization on examples.
        for _ in range(5): predict({'text':'Synthetic warm-up notification.','record_id':'synthetic','domain':'synthetic'})
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        whole=[]; stages=[]
        # Separate whole-call pass avoids attributing instrumentation sync overhead to normal inference.
        for r in data: whole.append(predict(r,False))
        for r in data: stages.append(predict(r,True))
    versions={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','scikit-learn','scipy','joblib','threadpoolctl']}
    summaries={domain:{'end_to_end':quantiles([r for r in whole if domain=='all' or r['domain']==domain],'seconds'),
        'instrumented_whole':quantiles([r for r in stages if domain=='all' or r['domain']==domain],'seconds'),
        'stages':{k:quantiles([{'s':r['stage_seconds'][k]} for r in stages if domain=='all' or r['domain']==domain],'s') for k in STAGES}}
        for domain in ['all',*DOMAINS]}
    write_json(out/f'profile_{mode}.json',{'mode':mode,'status':'completed','started_utc':now(),'wall_seconds':time.perf_counter()-began,
        'whole_rows':whole,'instrumented_rows':stages,'summaries':summaries,'five_synthetic_warmups':True,
        'threadpool_before':before,'threadpool_active':active,'torch_threads':torch.get_num_threads(),
        'torch_interop_threads':torch.get_num_interop_threads(),'environment':{n:os.environ.get(n) for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']},
        'versions':versions,'device':torch.cuda.get_device_name(),'peak_vram_bytes':torch.cuda.max_memory_allocated(),
        'process_rss_bytes':psutil.Process().memory_info().rss,'membership_sha256':digest([r['record_id'] for r in data]),
        'classifier_sha256':manifest['classifier_sha256'],'encoder_sha256':source['files']['model.safetensors']['sha256'],
        'serialization':'UTF8 canonical JSON bytes in memory, not disk/network IO','accesses':boundary.report()})
    print(mode,'whole-call',summaries['all']['end_to_end'],flush=True)

def audit(out):
    boundary=DiagnosticBoundary('train').install()
    import numpy as np
    import joblib
    original=read_json(ORIGINAL/'candidate_manifest.json'); variant=read_json(out/'corrected_variant.json')
    cal=read_rows(ORIGINAL/'calibration_raw.jsonl'); clean=read_rows(out/'corrected_calibration_logits.jsonl')
    require(clean==corrected_calibration(cal,set(variant['calibration_excluded_ids'])),'Wrong exclusions')
    fitted=fit_temperature([r['spam_label'] for r in clean],[r['logit_difference'] for r in clean],.5,5.)
    require(fitted['applied_temperature']==variant['temperature'],'Independent T mismatch')
    dev=read_rows(ORIGINAL/'dev_raw.jsonl'); rule=choose_threshold(dev,variant['temperature'])
    require(rule['threshold']==variant['threshold'],'Independent threshold mismatch')
    require(sha_file(ROOT/variant['classifier_path'])==original['classifier_sha256'],'Independent weights mismatch')
    loaded=joblib.load(ROOT/variant['classifier_path'])
    matrix=read_json(ORIGINAL/'embeddings.json')['parts']['dev']
    logits=loaded.decision_function(np.load(ROOT/matrix['matrix_path'],allow_pickle=False))
    require(max(abs(float(z)-r['logit_difference']) for z,r in zip(logits,dev))<=1e-4,'Independent reload mismatch')
    profiles={m:read_json(out/f'profile_{m}.json') for m in ['default','one']}
    ids=[r['record_id'] for r in fixed_sample(read_rows(ORIGINAL/'private_dev_inputs.jsonl'))]
    for p in profiles.values():
        require([r['record_id'] for r in p['whole_rows']]==ids,'Independent IDs mismatch')
        require([r['record_id'] for r in p['instrumented_rows']]==ids,'Stage IDs mismatch')
        for r in p['instrumented_rows']:
            require(set(r['stage_seconds'])==set(STAGES),'Missing stage')
            require(sum(r['stage_seconds'].values())<=r['seconds'],'Stage accounting invalid')
        require(not p['accesses']['accesses'] and not p['accesses']['denied'],'Unexpected original data access')
    require(all(t['num_threads']==1 for t in profiles['one']['threadpool_active']),'Control thread limits not applied')
    changes=sum(a['spam']!=b['spam'] for a,b in zip(profiles['default']['whole_rows'],profiles['one']['whole_rows']))
    require(changes==0,'CPU control changed decisions')
    before=read_json(out/'preservation_before.json'); after={n:sha_file(ROOT/n) for n in before}
    require(before==after,'Original artifact changed')
    write_json(out/'independent_audit.json',{'status':'passed','protected_files':len(before),'changed_files':[],
        'corrected_calibration_rows':len(clean),'reload_rows':len(dev),'profile_ids_identical':True,
        'control_decision_changes':changes,'closed_data_opened':False,'accesses':boundary.report()})
    print('Independent audit passed;',len(before),'original files unchanged',flush=True)

def report(out):
    c=read_json(out/'collision_analysis.json'); profiles={m:read_json(out/f'profile_{m}.json') for m in ['default','one']}
    audit_result=read_json(out/'independent_audit.json')
    lines=['# Завершающая диагностика C', '',f'Диагностический запуск `{out.name}`. Исходный C: `20261012T095633Z-7d01f7b8`. E0 остаётся рабочим вариантом.',
        '', '## Совпадение между inner-частями', '',
        'Полная повторная проверка токеновых SHA всех 7600 выбранных inner-записей подтвердила ровно одну cross-inner группу по видимому нормализованному префиксу и ровно одну по фактическим capped input IDs.',
        '', '| Часть | ID | Метка | Исходный локатор |', '|---|---|---|---|']
    for r in c['records']:lines.append(f"| {r['part']} | `{r['record_id']}` | {r['spam_label']} (spam) | `{json.dumps(r['original_locator'],ensure_ascii=False)}` |")
    lines += ['', 'Это fit↔calibration, без dev. Конфликта меток нет. Полные тексты различны, исходные duplicate_group_id и inner_group_id различны. После ограничения 512 токенами E5 получает одинаковые последовательности; это зависимость калибровки от fit, а не пересечение полных писем или ошибка направления меток.',
        f"Максимальная абсолютная разность сохранённых embedding: {c['embedding_max_abs_difference']:.9g}. Подробные длины, флаги усечения, хеши и группы: `collision_analysis.json`. Сырые письма не публикуются.",
        '', '## Отдельный исправленный вариант', '',
        'Из calibration удалена одна зависимая запись, без замены: 1000→999. Fit6000 и dev600 неизменны; encoder и LR не обучались повторно. Повторены только исходная scalar temperature calibration на calibration999 и исходное правило operating threshold на dev600, с прежними границами T=[0.5,5]. Пересечений по canonical visible prefix и фактическим input IDs между этими тремя частями осталось 0.',
        f"T: {c['original_temperature']:.12g}→{c['corrected_temperature']:.12g}; threshold: {c['original_threshold']:.16g}→{c['corrected_threshold']:.16g}. Unconstrained optimum T: {c['original_calibration']['unconstrained_temperature']:.12g}→{c['corrected_calibration']['unconstrained_temperature']:.12g}. Значение вне диапазона приведено только как диагностика, диапазон не расширялся.",
        f"Изменённых решений на dev: {c['dev_decisions_changed']}/600. Reload копии LR: max |Δlogit|={c['reload_max_logit_difference']:.9g}. Отдельный manifest и копия LR: `models/c_diagnostics/{out.name}/`; encoder/tokenizer остаются проверяемыми ссылками на исходные неизменные артефакты.",
        '', '| Inner-dev | TP / FP / TN / FN | F1 | FPR |','|---|---|---:|---:|']
    for d,m in c['dev_metrics'].items():lines.append(f"| {d} | {m['TP']} / {m['FP']} / {m['TN']} / {m['FN']} | {m['F1']:.6f} | {m['FPR']:.6f} |")
    lines += ['', 'Это показатели части выбора порога, не независимая оценка обобщения. Outer validation повторно не читалась и не оценивалась. Закрытые test/challenge/reserve не открывались.',
        '', '## Изолированное профилирование', '',
        'Два последовательных отдельных процесса, один GPU, одинаковые фиксированные 64 inner-dev ID и порядок исходного C: 38 historical и 26 RU. Batch=1, float32, eval/no_grad, неизменные исходные веса, исходные T/threshold. В каждом процессе 5 одинаковых синтетических прогревов; затем один проход полного вызова и отдельный инструментированный проход по тем же 64 записям.',
        'Полный wall-clock вызов измеряется непосредственно: tokenization→H2D→encoder→masked mean/L2→D2H→sklearn decision_function→T/sigmoid/threshold→UTF-8 JSON serialization. После вызова CUDA синхронизирована. Для компонентного прохода CUDA синхронизируется на границах этапов; поэтому его wall time отдельно и он не заменяет обычный end-to-end. Компонентные квантили не суммируются. Сериализация — байты в памяти, без файлового/сетевого I/O.',
        '', '| Режим | Область | n | End-to-end p50, ms | p95, ms | Instrumented whole p50, ms | p95, ms |', '|---|---|---:|---:|---:|---:|---:|']
    for mode,p in profiles.items():
        for d,s in p['summaries'].items():
            a=s['end_to_end']; b=s['instrumented_whole']
            lines.append(f"| {mode} | {d} | {a['n']} | {a['p50_ms']:.4f} | {a['p95_ms']:.4f} | {b['p50_ms']:.4f} | {b['p95_ms']:.4f} |")
    lines += ['', '| Этап (64 примера) | default p50, ms | p95, ms | one p50, ms | p95, ms |', '|---|---:|---:|---:|---:|']
    for k in STAGES:
        a=profiles['default']['summaries']['all']['stages'][k]; b=profiles['one']['summaries']['all']['stages'][k]
        lines.append(f"| {k} | {a['p50_ms']:.4f} | {a['p95_ms']:.4f} | {b['p50_ms']:.4f} | {b['p95_ms']:.4f} |")
    lines += ['', '## Потоки, воспроизводимость и ограничения', '',
        '`default` воспроизводит прежний профиль: torch intra-op=1, остальные CPU pools/environment унаследованы. `one` — отдельный контроль: OMP/MKL/OpenBLAS/NumExpr env=1 до импортов, torch intra/inter-op=1, threadpoolctl limits=1. Это контроль всей CPU-политики, не доказательство вклада одной конкретной библиотеки. Полные фактические threadpool_info, версии, сырые времена всех записей, численные ошибки и память сохранены в profile_default.json/profile_one.json.', '']
    for mode,p in profiles.items():
        pools=[{'api':x['internal_api'],'threads':x['num_threads'],'prefix':x['prefix']} for x in p['threadpool_active']]
        lines.append(f"- `{mode}`: torch intra/inter={p['torch_threads']}/{p['torch_interop_threads']}; pools={json.dumps(pools)}; GPU={p['device']}; peak VRAM={p['peak_vram_bytes']/1024**2:.1f} MiB; max |Δlogit|={max(r['raw_logit_abs_error'] for r in p['whole_rows']):.9g}.")
    a=profiles['default']['summaries']['all']['end_to_end']; b=profiles['one']['summaries']['all']['end_to_end']
    lines += ['', f"Отношение default/one по медиане полного вызова: {a['p50_ms']/b['p50_ms']:.3f}×; по p95: {a['p95_ms']/b['p95_ms']:.3f}×. Решения между режимами одинаковы на 64/64.",
        'Два процесса запускались последовательно; один прогон на режим не отделяет эффекты нагрева GPU, фоновой нагрузки ОС и порядка запусков. Нет доказательства универсальной производительности, throughput, холодного старта или скорости на иных длинах/пакетах. Загрузка модели/диска и сетевой ответ не входят в steady-state latency. Исходный необычный профиль остаётся фактом своего запуска; новые результаты не исправляют его задним числом.',
        f"Независимый аудит отдельным процессом: {audit_result['status']}; SHA-256 {audit_result['protected_files']} исходных файлов неизменны. Проверены отбор, исключение, T/threshold, настоящий joblib reload на 600 dev векторах, лимиты потоков, одинаковые ID и решения, отсутствие обращений к закрытым данным.",
        '', '## Команды', '', 'Из корня проекта, PowerShell:', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_c_diagnostics.py',
        '.\\.venv\\Scripts\\python.exe src/diagnose_c.py run',
        f'.\\.venv\\Scripts\\python.exe src/diagnose_c.py audit --out reports/c_diagnostics/runs/{out.name}',
        '```', '', 'Команда run создаёт новый архив, заранее фиксирует протокол, самостоятельно запускает анализ и два профиля через существующую .venv-decision, затем независимый аудит через .venv. Старые архивы не перезаписываются. Новый C_DIAGNOSTICS.md содержит последний диагностический запуск.',
        f'Архив протокола, хешей, анализа, измерений и аудита: `reports/c_diagnostics/runs/{out.name}/`.']
    old=read_json(ORIGINAL/'warm_end_to_end.json')
    lines += ['', '## Сопоставление с прежним профилем и технический журнал', '',
        'Прежний исходный профиль имел один прогрев и не включал сериализацию; новый имеет пять прогревов и включает JSON. Это отдельная диагностика, старые цифры не заменяются.']
    for d in DOMAINS:
        q=quantiles([r for r in old['rows'] if r['domain']==d],'seconds')
        lines.append(f"- Исходный C, {d}: p50={q['p50_ms']:.4f} ms, p95={q['p95_ms']:.4f} ms.")
    lines += ['Если сотни миллисекунд не воспроизводятся уже в default, снижение относительно старого результата нельзя приписать ограничению CPU-потоков. Причина прежней аномалии остаётся неизвестной; контроль потоков характеризует только нынешние запуски.',
        'Предварительный архив 20261012T141009Z-cfce28e9 сохранён: analyze и default реально завершились, но управляющий процесс остановился на UnicodeEncodeError при выводе progress bar в cp1251. В новом скрипте stdout/stderr явно настроены на UTF-8. Ни модели, ни данные из-за этого не менялись. Итоговые таблицы выше относятся к полному повторному запуску; выбор лучшего времени между архивами не выполнялся.']
    text='\n'.join(lines)+'\n'; (out/'C_DIAGNOSTICS.md').write_text(text,encoding='utf-8')
    (ROOT/'reports/C_DIAGNOSTICS.md').write_text(text,encoding='utf-8')

def run():
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+digest(now())[:8]
    out=ROOT/'reports/c_diagnostics/runs'/run_id; out.mkdir(parents=True)
    write_json(out/'protocol.json',{'frozen_utc':now(),'original_C':str(ORIGINAL.relative_to(ROOT)),
        'collision_rule':'Recompute all inner actual token IDs; remove dependent calibration row only; abort on unexpected roles/count',
        'weights':'No encoder or LR fit; copied classifier SHA identical','calibration':'Original fixed T [0.5,5], same golden search and dev threshold rule',
        'profile':'Fixed original64, five synthetic warmups, separate whole/instrumented passes, batch1 original weights/T/threshold',
        'threads':'default original torch intra1 inherited pools; one env CPU1+intra/inter1+threadpoolctl1',
        'evaluation':'Inner only, no outer validation or closed data','quantiles':'NumPy linear p50/p95 direct whole wall; never sum stage quantiles',
        'code_sha256':sha_file(Path(__file__))})
    shutil.copyfile(Path(__file__),out/'executed_source.py.txt')
    write_json(out/'preservation_before.json',snapshot())
    for stage,python in [('analyze','.venv-decision'),('default','.venv-decision'),('one','.venv-decision'),('audit','.venv')]:
        env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'; env['PYTHONIOENCODING']='utf-8'
        if stage=='one':env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        print('Running',stage,'in',out.name,flush=True)
        with (out/f'{stage}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([str(ROOT/python/'Scripts/python.exe'),str(Path(__file__)),stage,'--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
        require(result.returncode==0,'Diagnostic stage failed; see '+str(out/f'{stage}.log'))
        print((out/f'{stage}.log').read_text(encoding='utf-8')[-1600:],flush=True)
    report(out); print('Completed',out.relative_to(ROOT),flush=True)

if __name__=='__main__':
    # Windows hosts may default to cp1251; HF progress bars contain Unicode.
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','analyze','default','one','audit']);p.add_argument('--out',type=Path);args=p.parse_args()
    if args.stage=='run':run()
    else:
        require(args.out is not None,'--out required');out=args.out.resolve()
        require(out.is_relative_to(ROOT/'reports/c_diagnostics/runs'),'Output outside diagnostic archive')
        if args.stage=='analyze':analyze(out)
        elif args.stage=='audit':audit(out)
        else:profile(out,args.stage)
