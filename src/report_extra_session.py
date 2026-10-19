"""Local reports from actual new-session artifacts; no closed evaluation reads."""
import argparse
from extra_session_common import *

NAMES={'spam_email':'Historical validation pilot','russian_auxiliary':'RU auxiliary validation pilot','personal_development':'Личные письма — development'}


def fmt(x):return 'N/A' if x is None else f'{x:.4f}'


def quantile(values,p):
    values=sorted(values)
    if not values:return None
    x=(len(values)-1)*p;a=int(x);b=min(a+1,len(values)-1)
    return values[a]+(values[b]-values[a])*(x-a)


def metric_tables(labels,variants):
    result=[]
    for domain in NAMES:
        notes=[]
        result+=['\n## '+NAMES[domain]+'\n',
            '| Вариант | Решений / всего | TP | FP | TN | FN | Precision | Recall | F1 | FPR |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,rows in variants.items():
            s=summary(labels,rows)[domain];m=s['metrics']
            result.append('| '+name+' | '+f"{s['successful']}/{s['records']}"+' | '+' | '.join(str(m[k]) for k in ['TP','FP','TN','FN'])+' | '+
                ' | '.join(fmt(m[k]) for k in ['precision','recall','F1','FPR'])+' |')
            if s['refusals']:notes.append('\nОтказы '+name+': `'+json.dumps(s['refusals'],ensure_ascii=False)+'`. Метрики условны на успешном покрытии.\n')
        result.extend(notes)
    return '\n'.join(result)


def change_tables(labels,old,new):
    details=paired(labels,old,new);lines=['\n| Набор | Общих успешных ID | Исправлено FP | Добавлено FP | Исправлено FN | Добавлено FN |',
        '|---|---:|---:|---:|---:|---:|']
    for d,v in details.items():
        lines.append('| '+NAMES[d]+' | '+str(v['same_success_ids'])+' | '+' | '.join(str(len(v[k])) for k in ['rescued_FP','added_FP','rescued_FN','added_FN'])+' |')
        if v['excluded_noncommon']:lines.append(f"\n{NAMES[d]}: {v['excluded_noncommon']} ID вне общего успешного покрытия; paired old/new metrics в JSON.\n")
    lines+=['\n| Набор, только общие успешные ID | До TP/FP/TN/FN; F1 | После TP/FP/TN/FN; F1 |',
        '|---|---|---|']
    for d,v in details.items():
        a=v['old_metrics'];b=v['new_metrics']
        lines.append('| '+NAMES[d]+' | '+'/'.join(str(a[k]) for k in ['TP','FP','TN','FN'])+'; '+fmt(a['F1'])+' | '+
            '/'.join(str(b[k]) for k in ['TP','FP','TN','FN'])+'; '+fmt(b['F1'])+' |')
    return '\n'.join(lines)


def report(out):
    boundary=SessionBoundary(out).install();p=load_protocol(out);labels=read_rows(out/'pilot_membership.jsonl')
    base=read_rows(out/'baseline_predictions.jsonl');baseline={n:[r for r in base if r['model_id']==n] for n in ['E0','Laya_day5','C_corrected']}
    variants=dict(baseline);visibility={};comparisons={};resources={}
    for model in ['Laya_day5','C_corrected']:
        rows=read_rows(out/f'visibility_{model}_predictions.jsonl');visibility[model]={v:[r for r in rows if r['variant']==v] for v in ['A_prefix','B_head_tail','C_windows_max']}
        for v in ['B_head_tail','C_windows_max']:
            variants[model+'/'+v]=visibility[model][v]
            comparisons[model+'/'+v]=paired(labels,baseline[model],visibility[model][v])
        resources[model]=read_json(out/f'visibility_{model}_audit.json')['resources']
    text=['# Видимость текста: фактический эксперимент\n',f"Run `{out.name}`. Контракт запечатан {p['created_utc']}; SHA `{sha_file(out/'EXTRA_SESSION_PROTOCOL.json')}`.\n",
        'Прежние веса, T, thresholds и вопрос B не изменены. A — сохранённый baseline, воспроизведённый первым окном на всех278 ID. '
        'B и C меняют inference-контракт и являются исследовательскими вариантами. Максимум C — диагностическая композиция, не откалиброванная вероятность всего письма.\n',
        'Отбор: 128 целых групп historical (64ham/64spam),128 целых групп RU (64/64),22 личных (13/9). '
        'Это class-balanced пилоты, precision/F1 зависят от такого состава; они не заменяют полный validation. '
        'RU source-level labels provisional; личные письма уже использовались в разработке. Наборы не объединяются.\n']
    text.append(metric_tables(labels,{model+'/'+v:rows for model,parts in visibility.items() for v,rows in parts.items()}))
    for model,parts in visibility.items():
        for v in ['B_head_tail','C_windows_max']:
            text+=['\n## '+model+' / '+v+' относительно A\n',change_tables(labels,parts['A_prefix'],parts[v])]
    text+=['\n## Короткие и усечённые входы\n','| Модель | Набор | Коротких / усечённых A | B: ошибок до→после, короткие | B: ошибок до→после, усечённые | C: ошибок до→после, короткие | C: ошибок до→после, усечённые |',
        '|---|---|---:|---:|---:|---:|---:|']
    short_long={}
    for model,parts in visibility.items():
        maps={v:{r['record_id']:r for r in rows} for v,rows in parts.items()}
        short_long[model]={}
        for d in NAMES:
            records=[r for r in labels if r['domain']==d];sets={flag:[r for r in records if maps['A_prefix'][r['record_id']]['token_stats']['baseline_truncated']==flag] for flag in [False,True]}
            cells=[];short_long[model][d]={}
            for v in ['B_head_tail','C_windows_max']:
                for flag in [False,True]:
                    subset=sets[flag];a=sum(maps['A_prefix'][r['record_id']]['decision']!=r['spam_label'] for r in subset);b=sum(maps[v][r['record_id']]['decision']!=r['spam_label'] for r in subset)
                    cells.append(f'{a}→{b} (n={len(subset)})');short_long[model][d][v+('/truncated' if flag else '/short')]={'n':len(subset),'errors_before':a,'errors_after':b}
            text.append('| '+model+' | '+NAMES[d]+' | '+f'{len(sets[False])}/{len(sets[True])}'+' | '+' | '.join(cells)+' |')
    text+=['\n## Видимый объём и стоимость\n','Диапазоны — Unicode-символы clean_builder_v2; Laya предварительно заменяет буквальный mask-токен пробелом. '
        'Покрытие — объединение диапазонов исходного текста, без повторного счёта перекрытий и добавленного разделителя. '
        'Это доля символов, а не гарантия семантического покрытия. Служебные токены входят в лимиты.\n',
        '| Модель / вариант | Набор | Полностью видно | Средняя доля символов | Среднее число окон | Forward p50/p95, ms | Всего forward, s |',
        '|---|---|---:|---:|---:|---:|---:|']
    coverage_cost={}
    for model,parts in visibility.items():
        coverage_cost[model]={}
        domain={r['record_id']:r['domain'] for r in labels}
        for v,rows in parts.items():
            coverage_cost[model][v]={}
            for d in NAMES:
                rr=[r for r in rows if domain[r['record_id']]==d];seconds=[r['required_forward_seconds'] for r in rr]
                c={'fully_covered':sum(r['coverage']['unseen_chars']==0 for r in rr),'n':len(rr),
                    'mean_fraction':sum(r['coverage']['fraction'] for r in rr)/len(rr),'mean_windows':sum(r['window_count'] for r in rr)/len(rr),
                    'forward_seconds_sum':sum(seconds),'forward_p50_seconds':quantile(seconds,.5),'forward_p95_seconds':quantile(seconds,.95),
                    'shared_prepare_seconds_sum':sum(r['prepare_seconds_all_variants'] for r in rr),
                    'unseen_chars_total':sum(r['coverage']['unseen_chars'] for r in rr)}
                coverage_cost[model][v][d]=c
                text.append('| '+model+'/'+v+' | '+NAMES[d]+' | '+f"{c['fully_covered']}/{len(rr)}"+' | '+fmt(c['mean_fraction'])+' | '+fmt(c['mean_windows'])+' | '+
                    f"{1000*c['forward_p50_seconds']:.2f}/{1000*c['forward_p95_seconds']:.2f}"+' | '+f"{sum(seconds):.3f}"+' |')
    for model,res in resources.items():
        audit=read_json(out/f'visibility_{model}_audit.json')['verification']
        text.append(f"\n{model}: load={res['load_seconds']:.3f}s, весь процесс с load={res['wall_seconds_including_load']:.3f}s; "
            f"peak allocated/reserved={res['peak_vram_allocated_bytes']/2**20:.1f}/{res['peak_vram_reserved_bytes']/2**20:.1f}MiB; "
            f"max |savedA−freshA score|={max(x['baseline_score_abs_error'] for x in audit):.9g}, решений изменено0/278.\n")
    text.append('\nBatch1, CPU pools1; CUDA синхронизирована. Перед измерением2 синтетических прогрева; одинаковая actual-ID последовательность внутри записи повторно не вычисляется. '
        'Показанные времена — измеренные затраты необходимых уникальных forward данной композиции, с adapter/CUDA→CPU/LR/calibration; '
        'подготовка всех представлений измерена совместно и сохранена отдельно в JSON. Это не полная end-to-end latency каждого варианта и не throughput. '
        'Первые3 SDK-проверки Laya дают дополнительный forward, не включённый в таблицу. Квантили компонент не складывались. '
        'Предупреждения tokenizer о длине относились только к офлайн подсчёту offsets полного текста; в encoder подавались проверенные ограниченные последовательности.\n')
    text.append('\n## Вывод\nC исправил2 historical FN без новых FP на этом пилоте; оба новых представления дали одинаковые решения. '
        'Laya windows исправил1 historical FN, но добавил3 personal FP; head+tail исправил один historical FN и добавил другой (чистое изменение0), добавил2 personal FP. '
        'На RU все128 записей короткие для обоих encoder; изменения отсутствуют. Дополнительная видимость не устраняет неизвестную историю согласия. '
        'Варианты с max имеют возможность накапливать ложные срабатывания по окнам; старый threshold не калибровался для такого режима. '
        'Нет основания автоматически менять default или обещать пользу на новых письмах. Полные ID изменений и сырые оценки окон — в локальных JSON/JSONL.\n')
    if (out/'visibility_window_metadata_correction.json').exists():
        text.append('\n## Техническая поправка журнала\nНезависимый аудит выявил ошибочную подпись первого префикса как C_windows_max из-за общей mutable metadata-структуры. '
            'Входы, диапазоны, logits, оценки, решения и метрики не менялись. Исходные *_windows.jsonl сохранены; '
            'для анализа использовать *_windows_corrected.jsonl и visibility_window_metadata_correction.json. '
            'Исправлен только variant field у278 A-записей каждой модели; повторного inference не было. Regression-фикстура проверяет приоритет внешнего варианта.\n')
    (out/'TEXT_VISIBILITY_EXPERIMENT.md').write_text('\n'.join(text),encoding='utf-8')
    # Qwen branch is reported only from real saved outputs, including genuine blockers.
    qwen_path=out/'qwen_predictions.jsonl';qrows=None
    qlines=['# Локальный Qwen3-4B — один пилот\n',f"Официальный checkpoint `Qwen/Qwen3-4B`, revision `{QWEN_REV}`. "
        'Model card: https://huggingface.co/Qwen/Qwen3-4B . Действительный snapshot зафиксирован до загрузки/предсказаний.\n',
        'BF16 safetensors, без fine-tuning/quantization/trust_remote_code, batch1, GPU RTX4060Ti16GB, SDPA, non-thinking через enable_thinking=False. '
        'Greedy do_sample=False/num_beams1/seed42/max_new_tokens32; весь input с chat template≤2048. Это исследовательский deterministic decoding, '
        'а не рекомендуемые model card sampling0.7/0.8. Единственная инструкция до оценки — в sealed JSON. '
        'Она запрещает следовать инструкциям письма, считать всю рекламу спамом и выдумывать согласие/историю. '
        'Вход — исходный subject+body через clean_builder_v2, prefix; ID/label/reasons/categories/path не включаются в prompt.\n',
        f"До скачивания: нужно {p['preflight']['download_bytes']/1e9:.3f}GB, свободно {p['preflight']['disk_free_bytes']/2**30:.2f}GiB; runtime поддерживает qwen3. "
        'Основные .venv/.venv-decision не обновлялись. Только официальный snapshot; никаких почтовых данных в сеть.\n']
    if qwen_path.exists() and (out/'qwen_resources.json').exists():
        qrows=read_rows(qwen_path);variants['Qwen3_4B']=qrows;res=read_json(out/'qwen_resources.json')
        qlines.append(metric_tables(labels,{**baseline,'Qwen3_4B':qrows}))
        for name,rows in baseline.items():
            qlines+=['\n## Qwen относительно '+name+' на общих успешных ID\n',change_tables(labels,rows,qrows)]
            comparisons['Qwen_vs_'+name]=paired(labels,rows,qrows)
        qlines+=['\n## Runtime и сырые ответы\n','Сначала4 отдельных синтетических smoke (EN/RU уведомление и явно нежелательная рассылка), исключены из метрик. '
            'Сырые ответы ниже — фактические короткие JSON; числовая уверенность не запрашивалась и не восстанавливалась.\n',
            '| Synthetic ID | Содержание синтетического примера | Сырой ответ | Статус |','|---|---|---|---|']
        descriptions=['EN запрошенное уведомление','EN явно нежелательная массовая рассылка','RU запрошенное уведомление','RU явно нежелательная массовая рассылка']
        for r,desc in zip(read_rows(out/'qwen_synthetic_smokes.jsonl'),descriptions):qlines.append(f"| {r['record_id']} | {desc} | `{r['raw_response']}` | {r['status']} |")
        qlines.append('\nСинтетическое RU запрошенное уведомление модель назвала spam. Это реальная семантическая ошибка при корректном формате; '
            'smoke проверял runtime/формат, не разрешал подбор prompt. Исходный ответ сохранён, формулировка после него не менялась.\n')
        times=[r['end_to_end_seconds'] for r in qrows if r['status']=='ok'];gen=[r['generation_seconds'] for r in qrows if r['status']=='ok']
        qlines.append(f"\nLoad={res['load_seconds']:.3f}s; pilot wall={res['pilot_seconds']:.3f}s. Batch1 warm end-to-end p50/p95="
            f"{quantile(times,.5):.3f}/{quantile(times,.95):.3f}s; generation p50/p95={quantile(gen,.5):.3f}/{quantile(gen,.95):.3f}s. "
            f"GPU peak allocated/reserved={res['peak_vram_allocated_bytes']/2**30:.3f}/{res['peak_vram_reserved_bytes']/2**30:.3f}GiB. "
            'torch.cuda peaks сброшены после load/smokes перед pilot; CUDA синхронизирована. End-to-end включает подготовку chat/токенизацию/H2D/generate/декодирование/JSON parsing, '
            'без disk write и cold load. Это одиночный локальный прогон, не нагрузочный тест.\n')
        qlines+=['\n| Набор | Усечено input | Полностью видимых | Средняя доля символов |','|---|---:|---:|---:|']
        domains={r['record_id']:r['domain'] for r in labels}
        for d in NAMES:
            rr=[r for r in qrows if domains[r['record_id']]==d and 'original_chars' in r]
            qlines.append('| '+NAMES[d]+' | '+str(sum(r['truncated'] for r in rr))+' | '+f"{sum(not r['truncated'] for r in rr)}/{len(rr)}"+' | '+fmt(sum(r['visible_chars']/r['original_chars'] for r in rr)/len(rr))+' |')
        qlines+=['\n## Личная development-диагностика\n','| ID | Финальная метка | E0 | Laya_day5 | C_corrected | Qwen |','|---|---:|---:|---:|---:|---:|']
        maps={n:{r['record_id']:r for r in rows} for n,rows in {**baseline,'Qwen':qrows}.items()}
        for r in labels:
            if r['domain']=='personal_development':qlines.append('| '+r['record_id']+' | '+str(r['spam_label'])+' | '+' | '.join(str(maps[n][r['record_id']]['decision']) for n in ['E0','Laya_day5','C_corrected','Qwen'])+' |')
        qlines.append('\nЗдесь0=подтверждённый пользовательский ham,1=spam; личные примеры описаны только ID и исходом. '
            'Это ошибки относительно финального пользовательского эталона, не доказанные намерения отправителя. '
            'Qwen и Laya используют разные tokenizer/context budgets; наблюдаемая разница не изолирует только архитектуру. '
            'Результаты не подтверждают phishing или категорийную классификацию, не восстанавливают скрытое согласие и не определяют нового победителя. '
            'Сырые ответы пилота сохранены локально; личное содержимое в публичные отчёты не включено. Prompt не менялся после smoke/ошибок.\n')
        qlines.append('\nИнтеграционный tokenizer-only smoke до первого Qwen forward обнаружил изменившийся runtime default apply_chat_template(return_dict=True). '
            'Добавлено явное return_dict=False для списка IDs; это не смена prompt/model/decoding и не повтор пилота. '
            'В non-thinking template есть пустой think-prefill; он входит в input budget, модель не генерирует рассуждение. Подробности: qwen_integration_preflight.json.\n')
        qlines.append('\nДо готовности source manifest одна попытка preflight завершилась FileNotFoundError, до model forward и до pilot predictions. '
            'Она сохранена отдельно в qwen_attempt0_preflight_*.json. После завершения официальной загрузки выполнен ровно один smoke/pilot pass. '
            'Runtime предупреждал, что sampling temperature/top_p/top_k игнорируются при do_sample=False; использовался заранее зафиксированный greedy decoding.\n')
    else:
        blockers=[read_json(x) for x in out.glob('qwen*blocker.json')]
        source=read_json(out/'qwen_source.json') if (out/'qwen_source.json').exists() else {'status':'pending'}
        qlines.append('\nРеальный статус: '+json.dumps({'source_status':source.get('status'),'blockers':blockers},ensure_ascii=False)+'. Метрики отсутствуют; планы не считаются экспериментом.\n')
    (out/'QWEN_PILOT.md').write_text('\n'.join(qlines),encoding='utf-8')
    if (out/'C_larger_fit/candidate_manifest.json').exists() and (out/'C_larger_fit_predictions.jsonl').exists():
        bigger=read_rows(out/'C_larger_fit_predictions.jsonl');variants['C_larger_fit']=bigger
        manifest=read_json(out/'C_larger_fit/candidate_manifest.json');selection=read_json(out/'C_larger_fit/selection_and_collisions.json');res=read_json(out/'C_larger_fit/resources.json')
        comparisons['C_larger_fit']=paired(labels,baseline['C_corrected'],bigger)
        clines=['# C: увеличение только исходного inner-fit pool\n',
            f"Исходный inner-fit={selection['full_inner_counts']['fit']}; dev/calibration-пулы сохранены. Тексты {selection['all_original_held_rows_checked']} всех held inner-строк использовались только для проверки коллизий. "
            f"Baseline6000 входит в новый fit{manifest['fit_rows']}; новых embedding={manifest['new_embeddings']}.\n",
            'Отбор повторяет старый SHA fit-subset seed42, balanced по классам внутри домена, nominal historical≤2000/class и RU≤7000/class. '
            'Historical ограничен фактическим числом spam в inner-fit; параметры encoder/preprocessing/LR не менялись. '
            'Меняется объём fit и состав дополнительных примеров, а не encoder/представление текста. '
            'Historical2032/RU14000: доля RU выросла с66.7% до87.3%, поскольку historical spam почти исчерпан. '
            'Поэтому это проверка объявленного расширения pool, а не чистый причинный эффект числа строк при неизменной смеси доменов.\n',
            f"После отбора {selection['selected_before_capped_exclusions']}→{selection['selected_after']}; исключено новых fit-коллизий с reserved dev/cal={len(selection['new_fit_exclusions'])}. "
            f"Выбранные fit/dev600/cal999: cross-capped collisions0; intra-fit capped label-conflicts={len(selection['fit_capped_label_conflicts'])}. "
            f"Старых fit совпадений с невыбранными reserved inner-dev/calibration={len(selection['known_baseline_vs_unused_reserved'])}; исходная зависимая cal-запись остаётся исключённой calibration999. "
            'Полные original/clean exact/canonical, ID/global/inner groups между ролями проверены; метки не менялись. Подробные исключения/конфликты — в selection_and_collisions.json.\n',
            f"Повторно использован проверенный кеш6000fit/dev600/cal1000 (минус прежнее исключение), pinned E5 revision/tokenizer/actual capped-ID SHA проверены. "
            f"LR единожды fit, {manifest['fit_iterations']} iterations, {manifest['fit_seconds']:.3f}s; новые embeddings {manifest['new_embeddings_seconds']:.3f}s. "
            f"Новая NLL calibration999: T={manifest['temperature']:.12g}; dev600 threshold={manifest['threshold']:.16g}, прежний метод и ограничения. "
            'Новый T рассчитан из новых logits, старое значение не переносилось автоматически. Пилот прочитан только после сохранения candidate manifest.\n',
            metric_tables(labels,{'C_corrected':baseline['C_corrected'],'C_larger_fit':bigger}),change_tables(labels,baseline['C_corrected'],bigger),
            f"\nReload:600 dev max|Δlogit|={manifest['reload']['max_logit_error']}; fresh encoder8 probes max|Δembedding|={manifest['reload']['max_embedding_error']:.9g}. "
            f"Весь этап wall={res['wall_seconds']:.3f}s; peak allocated/reserved={res['peak_vram_allocated_bytes']/2**20:.1f}/{res['peak_vram_reserved_bytes']/2**20:.1f}MiB. "
            'Validation embeddings256 повторно использованы по ID и точному тексту,22 personal вычислены локально. Это не отдельный latency benchmark новой LR.\n',
            'Новая LR и manifest находятся в C_larger_fit/ данного run. Это отдельный исследовательский кандидат. '
            'Balanced RU-heavy train/pilot, provisional source labels, многократно используемый validation и маленький личный development ограничивают перенос. '
            'Выбранный по этим наблюдениям кандидат требует новых, ещё не просмотренных писем. Default не менялся.\n']
        (out/'C_LARGER_FIT.md').write_text('\n'.join(clines),encoding='utf-8')
    all_metrics={n:summary(labels,rows) for n,rows in variants.items()}
    write_json(out/'metrics_and_comparisons.json',{'by_variant':all_metrics,'paired':comparisons,'short_truncated':short_long,'coverage_cost':coverage_cost,
        'domains_not_combined':True,'no_new_threshold_tuning_on_pilot':True})
    write_json(out/'confusion_matrices.json',{n:{d:{k:x['metrics'][k] for k in ['TP','FP','TN','FN']}|{'successful':x['successful'],'records':x['records']} for d,x in domains.items()} for n,domains in all_metrics.items()})
    slines=['# Итог дополнительной исследовательской сессии\n',f"Run `{out.name}`; протокол `{sha_file(out/'EXTRA_SESSION_PROTOCOL.json')}`. "
        'Фактические новые эксперименты; прежние модели/default/финальные отчёты сохранены. Test/challenge/reserve/teacher/new personal не открывались.\n',
        metric_tables(labels,variants),
        '\n## Ответы для решения о следующих шагах\n',
        '1. Видимость помогла C на2 historical FN, Laya windows на1 historical FN. На личных Laya B/C добавили2/3 FP, C не изменился. '
        'Это ограниченная полезность на pilot, без универсального улучшения.\n']
    if qrows:
        personal=all_metrics['Qwen3_4B']['personal_development'];lay=all_metrics['Laya_day5']['personal_development']
        delta=comparisons['Qwen_vs_Laya_day5']['personal_development']
        slines.append(f"2. Qwen personal: FP/FN={personal['metrics']['FP']}/{personal['metrics']['FN']} против Laya {lay['metrics']['FP']}/{lay['metrics']['FN']}; "
            f"исправлено FP={len(delta['rescued_FP'])},FN={len(delta['rescued_FN'])}; добавлено FP={len(delta['added_FP'])},FN={len(delta['added_FN'])}. "
            'Доменные результаты и точные ID в QWEN_PILOT.md/JSON; нет оснований считать скрытую историю согласия наблюдаемой.\n')
    else:slines.append('2. Qwen не завершён; реальная причина/статус в QWEN_PILOT.md. Mock-результатов нет.\n')
    if 'C_larger_fit' in all_metrics:
        slines.append('3. Расширенный C реально обучен, отдельно откалиброван и сохранён/reload. Изменения относительно C_corrected:\n')
        slines.append(change_tables(labels,baseline['C_corrected'],variants['C_larger_fit']))
        slines.append('\nНа RU FP1→0,FN5→2; на личных FP5→3,FN3→3 (один FN исправлен, другой добавлен); historical FN4→7. '
            'Улучшение зависит от домена. Из-за доступного inner-fit доля RU66.7%→87.3%; эффект размера и новой смеси доменов здесь не разделён.\n')
    else:slines.append('3. Более крупный fit C не завершён; старый C_corrected сохранён.\n')
    slines+=['\n4. Для будущей проверки на новых русскоязычных письмах имеет смысл отдельно зафиксировать C_larger_fit с исходным prefix512, '
        'с C_corrected/E0 как контролями. Он уменьшил personal FP, но добавил historical FN; это кандидат для проверки, не новый winner. '
        'Смешивать его с head+tail без отдельного протокола нельзя: такая комбинация не измерялась. '
        'Любой новый выбор по этому development-пилоту требует будущей проверки на новых письмах с заранее фиксированным протоколом. '
        'Старые уже оценённые тесты не дают независимого подтверждения выбора после этой сессии. Нового победителя/default не назначено.\n',
        '5. Для демо полезны отображение ограниченного видимого диапазона/усечения, явных отказов и режима ручного решения; '
        'новые представления и Qwen — только помеченные исследовательские опции с ценой обработки. '
        'Не добавлять «phishing detected» без соответствующего эталона; согласие/историю оставлять unknown. Упаковка не запускалась.\n',
        '\n## Воспроизведение\n','Команды в REPRODUCTION.md этого каталога. Повтор полного эксперимента — в новом out; существующие predictions не перезаписываются. '
        'report/audit разрешено повторять в этом out, они не вызывают модель и не открывают закрытые данные.\n']
    (out/'EXTRA_SESSION_SUMMARY.md').write_text('\n'.join(slines),encoding='utf-8')
    audit=read_json(out/'independent_audit.json') if (out/'independent_audit.json').exists() else None
    if audit and audit.get('larger_fit'):
        extra='\n## Завершающие проверки\n\n'
        extra+=f"Независимый аудит passed: {audit['independent_metric_groups']} metric groups, {audit['actual_window_token_checks']} token/window checks, "
        extra+=f"{audit['protected_files_unchanged']} прежних файлов с неизменными SHA-256, закрытых data reads0. "
        extra+=f"C: независимая повторная токенизация {audit['larger_fit']['independently_retokenized_selected_fit_and_all_held']} selected-fit/all-held строк; "
        extra+=f"selected cross-capped пересечений0, новых fit↔reserved совпадений0; fit↔pilot capped совпадений={len(audit['larger_fit']['pilot_capped_overlap_ids'])}. "
        extra+='Два baseline-fit совпадения с невыбранным calibration пулом сохранены/описаны, конфликтов меток нет; active dev600/cal999 не затронуты. '
        extra+='Загрузка новой LR, независимые T/threshold и dev/cal logits проверены. 11 unit/regression checks passed; модели не повторяли pilot. '
        extra+='Ошибки журнала окон и Qwen preflight описаны отдельно, исходные артефакты сохранены. Нерешённых технических блокеров нет.\n'
        with (out/'EXTRA_SESSION_SUMMARY.md').open('a',encoding='utf-8') as f:f.write(extra)
        with (out/'C_LARGER_FIT.md').open('a',encoding='utf-8') as f:f.write(extra)
    notes=['# Материал для презентации дополнительной сессии\n',
        '## Слайд1 — честный контракт\n\n278 заранее выбранных ID:128 historical validation,128 RU auxiliary validation,22 ранее просмотренных личных писем. '
        'Test/challenge/reserve закрыты. Старый E0/default сохранён. Одно фиксированное сравнение каждого варианта; без prompt/threshold поиска по пилоту.\n',
        '## Слайд2 — больше текста не гарантирует лучшее решение\n\nC head+tail/windows:2 historical FN исправлены, personal без изменений. '
        'Laya windows:1 historical FN исправлен,3 personal FP добавлены. Max окон не является вероятностью всего письма. '
        'Покрытие и цена окон показаны в TEXT_VISIBILITY_EXPERIMENT.md.\n']
    if qrows:
        qr=read_json(out/'qwen_resources.json');qtimes=[r['end_to_end_seconds'] for r in qrows if r['status']=='ok']
        notes.append('## Слайд3 — проверили настоящий локальный Qwen\n\nОфициальный Qwen3-4B BF16, non-thinking, без обучения, один prompt и один pilot pass. '+
            '; '.join(NAMES[d]+': FP/FN='+str(all_metrics['Qwen3_4B'][d]['metrics']['FP'])+'/'+str(all_metrics['Qwen3_4B'][d]['metrics']['FN']) for d in NAMES)+'. '
            'Qwen не устранил проблему согласия/ожидаемости; не выбирать другой prompt задним числом. '+
            f"Измерено p50={quantile(qtimes,.5):.3f}s,p95={quantile(qtimes,.95):.3f}s/end-to-end; peak allocated={qr['peak_vram_allocated_bytes']/2**30:.3f}GiB. Это не нагрузочный тест.\n")
    else:notes.append('## Слайд3 — Qwen\n\nЗавершённого пилота нет; реальные blocker/status см. QWEN_PILOT.md. Метрики не выдумываются.\n')
    if 'C_larger_fit' in all_metrics:
        notes.append('## Слайд4 — расширение fit C имеет цену\n\n6000→16032, encoder и LR hyperparameters сохранены, отдельные calibration999/dev600. '+
            '; '.join(NAMES[d]+': F1 '+fmt(all_metrics['C_corrected'][d]['metrics']['F1'])+'→'+fmt(all_metrics['C_larger_fit'][d]['metrics']['F1']) for d in NAMES)+'. '
            'RU доля66.7%→87.3%: это не изоляция одного лишь размера fit. Новый кандидат требует свежих писем.\n')
    else:notes.append('## Слайд4 — расширение fit\n\nВетка не завершена, исходный C_corrected сохранён.\n')
    notes.append('## Слайд5 — что добавить в демо\n\nПоказывать видимый диапазон/усечение, отказ и ручное решение; E0 остаётся default. '
        'Исследовательские переключатели отдельно от рабочего решения. C_larger_fit — кандидат для будущей RU проверки. '
        'Qwen не включать в автоматический default по этому пилоту. Не заявлять phishing, неизвестную историю подписки или production readiness. '
        'Личные письма не публиковать; на слайдах только агрегаты. Упаковка не выполнялась.\n')
    (out/'PRESENTATION_NOTES_EXTRA_SESSION.md').write_text('\n'.join(notes),encoding='utf-8')
    write_json(out/'report_accesses.json',boundary.report())
    print('Reports saved',out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);args=a.parse_args();report(args.out.resolve())
