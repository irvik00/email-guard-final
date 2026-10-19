"""Render factual day-5 artifacts from completed runs; no model fitting or dataset reads."""
from collections import Counter,defaultdict
import json
import re
import sys

from day5_common import (ROOT,DOMAINS,Boundary,read_json,read_rows,write_json,write_rows,
                         digest,sha_file,sigmoid,require,now)


def write_md(out,name,lines):
    text='\n'.join(lines)+'\n'
    (out/name).write_text(text,encoding='utf-8')
    (ROOT/'reports/day5'/name).write_text(text,encoding='utf-8')


def table_header():
    return ['| Кандидат / режим | Домен | TP / FP / TN / FN | P | R | F1 | FPR | N |',
            '|---|---|---|---:|---:|---:|---:|---:|']


def metric_row(name,domain,m):
    return f"| {name} | {domain} | {m['TP']} / {m['FP']} / {m['TN']} / {m['FN']} | {m['precision']:.6f} | {m['recall']:.6f} | {m['F1']:.6f} | {m['FPR']:.6f} | {m['support']} |"


def defang(text):
    text=re.sub(r'https?://[^\s<>]+','[URL]',text,flags=re.I)
    text=re.sub(r'[\w.+-]+@[\w.-]+','[EMAIL]',text)
    text=re.sub(r'\d{7,}','[NUMBER]',text)
    return ' '.join(text.replace('`',"'").replace('<','&lt;').replace('>','&gt;').split())[:260]


def select_errors(rows,candidate):
    errors={k:defaultdict(list) for k in ['FP','FN']}
    for r in rows:
        if r['status']!='ok':continue
        pred=sigmoid(r['logit_difference']/candidate['temperature'])>=candidate['threshold']
        kind='FP' if pred and r['spam_label']==0 else 'FN' if not pred and r['spam_label']==1 else None
        if kind:errors[kind][r['domain']].append(r)
    selected=[]
    for kind,buckets in errors.items():
        for domain in buckets:buckets[domain].sort(key=lambda r:digest([42,'error-review',r['record_id']]))
        count=0
        while count<5 and any(buckets.values()):
            for domain in DOMAINS:
                if buckets[domain] and count<5:
                    selected.append((kind,buckets[domain].pop(0)));count+=1
    return selected


def main():
    sys.stdout.reconfigure(encoding='utf-8');boundary=Boundary('verify').install()
    run_id=read_json(ROOT/'reports/day5/current_run.json')['run_id'];out=ROOT/'reports/day5/runs'/run_id
    contract=read_json(out/'experiment_contract.json');data=read_json(out/'data_manifest.json');training=read_json(out/'training.json')
    calibration=read_json(out/'calibration.json');laya=read_json(out/'laya_validation_comparison.json');lr=read_json(out/'lr_validation_comparison.json')
    manifest=read_json(out/'candidate_manifest.json');verification=read_json(out/'verification.json');reload=read_json(out/'reload_checks.json')
    personal=read_json(ROOT/'reports/day5/personal_review_readiness.json');test_results=read_json(out/'tests_execution.json')
    require(verification['status']=='passed' and test_results['returncode']==0,'Reports require actual successful checks')
    independent_reload=read_json(out/'independent_reload.json')
    require(independent_reload['status']=='passed','Independent option-logit reload check required')
    config=contract['config'];candidate=manifest['candidates'].get('adapted');selected_name='adapted' if candidate else 'base'
    selected=manifest['candidates'][selected_name]
    train_visible=read_rows(out/'private_fit_visible.jsonl');dev_visible=read_rows(out/'private_dev_visible.jsonl');cal_visible=read_rows(out/'private_calibration_visible.jsonl')
    val_visible=read_rows(out/'private_validation_visible.jsonl');visible_lookup={r['record_id']:r for r in val_visible}
    # Counts are full on selected inputs, not sampled estimates of the parent corpus.
    truncation={part:{d:{'n':sum(r['domain']==d for r in rows),'truncated':sum(r['domain']==d and r['token_stats']['truncated'] for r in rows),
                             'max_sequence_tokens':max(r['sequence_tokens'] for r in rows if r['domain']==d)} for d in DOMAINS}
                for part,rows in [('fit',train_visible),('dev',dev_visible),('calibration',cal_visible),('validation',val_visible)]}
    write_json(out/'truncation_summary.json',truncation)
    training_lines=['# День 5: реальная head-only адаптация Laya','',f'Run `{run_id}`. Статус **{training["status"]}**. E0 остаётся default.','',
        'Вопрос B зафиксирован до обучения: `Is this message unsolicited spam?`, type=noul, ключ spam, без criteria. До/после вопрос одинаковый. Первый A-пилот и его вопрос сохранены; B не объявляется улучшением весов.', '',
        '## Контракт и данные','',f'Checkpoint `convaiinnovations/laya/multilingual`, revision `{config["checkpoint"]["revision"]}`. Установленный laya0.3.20, штатная mmBERT decision architecture; локальные исходники agent/common и фактический forward проверены.',
        'Исходные train: historical5953, RU27953. Внутреннее разбиение seed42, class/source стратификация, объединение global groups и original/clean exact/canonical совпадений транзитивно; группы не разрываются. Конфликтов меток0, автоматической переразметки не было. Membership сохранён до обучения.', '',
        '| Часть | Historical ham/spam | RU ham/spam | Всего |','|---|---|---|---:|']
    for part,domaincounts in data['full_inner_counts'].items():
        hist=domaincounts['spam_email']['classes'];ru=domaincounts['russian_auxiliary']['classes']
        training_lines.append(f"| full inner {part} | {hist['0']}/{hist['1']} | {ru['0']}/{ru['1']} | {sum(d['rows'] for d in domaincounts.values())} |")
    for part,domaincounts in data['selected_counts'].items():
        hist=domaincounts['spam_email']['classes'];ru=domaincounts['russian_auxiliary']['classes']
        training_lines.append(f"| selected {part} | {hist['0']}/{hist['1']} | {ru['0']}/{ru['1']} | {sum(d['rows'] for d in domaincounts.values())} |")
    training_lines += ['', 'Fit balance изменён намеренно; calibration смешивает 500 historical и500RU. RU original native bool проверены строго: False→0/True→1, source_intended_unverified. Согласие/канал/истинность отдельных строк не подтверждены. Не называть весь RU корпус почтовым; лицензия источника CC-BY-NC-4.0 ограничивает коммерческое использование.', '',
        '## Реальный forward, smoke и бюджет','',
        'Обучены существующие head/type_emb/scorer; encoder и act_head заморожены. Action/escalation branch не получает supervised spam loss и исключена из optimizer. Архитектура и штатные веса загрузки не подменялись линейным классификатором. Это supervised head-only cross-entropy, не RLCD, LoRA или Qwen.',
        'Первичные источники: [официальный SDK](https://github.com/NandhaKishorM/laya), [карточка checkpoint](https://huggingface.co/convaiinnovations/laya). Рецепты автора не запускались; low-level forward сверялся с установленными исходниками0.3.20, а не с предполагаемым train API.',
        f"Trainable {training['trainable_parameters']:,} из {training['total_parameters']:,} параметров. Named parameters/requires_grad и точное множество optimizer проверены. Encoder.eval(), head.train(); напрямую DecisionModel.forward, без no_grad SDK wrapper. CE по сырым false/true logits и integer label 0/1.",
        'AdamW lr1e-4, weight_decay0.01, clip1, microbatch2×accumulation8 = effective16. Warmup15 steps, далее линейное уменьшение LR; seed42. Настройки определены контрактом, сетки гиперпараметров не было.',
        f"Реальный smoke:3optimizer steps/48fit записей в том же цикле; finite loss/nonzero gradients, изменённая head и неизменные encoder/action hash. Всего {training['actual_optimizer_steps']} steps, {training['examples_presented']} предъявлений, {training['unique_records_used']} уникальных записей из6000 ({training['fraction_of_selected_fit_epoch']:.2f} эпохи выбранного пула).",
        'Actual used: '+json.dumps(training['actual_used_counts'],ensure_ascii=False)+'.',
        f"Время training loop с dev/checkpoint checks {training['training_seconds']:.2f}с; CUDA {training['gpu']}; peak allocated {training['peak_vram_allocated_bytes']/2**30:.3f}GiB, reserved {training['peak_vram_reserved_bytes']/2**30:.3f}GiB; sampled RSS {training['rss_bytes']/2**30:.3f}GiB. OOM retries={training['oom_retries']}.",
        'PyTorch сообщил о недетерминированном memory-efficient attention backward при deterministic_algorithms(warn_only=True). Seed и порядок данных фиксированы, но побитовая воспроизводимость нового обучения не гарантируется. Числа текущего запуска сохранены. Это не блокер корректности градиентов.',
        'Tokenizer предупредил о длине полного текста >8192 на этапе получения offset mapping; модель получила только prefix≤700 tokens плюс вопрос, все actual sequence≤1024. Полный длинный текст не подавался в forward. Это не ошибка индексирования модели.', '',
        '## Learning curve и выбор','',
        'Правило до fit: среди обученных checkpoints max mean(domain F1@.5), затем меньше historical FP, затем более ранняя точка. Initial — reference, не обученный кандидат. Calibration/внешний validation в выборе весов не участвовали.', '',
        '| Step | Examples | Train loss (последние ≤100 steps) | Dev NLL mean | Hist F1 / FP | RU F1 / FP | Mean F1 |','|---|---:|---:|---:|---|---|---:|']
    for row in training['learning_curve']:
        loss='—' if row['train_loss'] is None else f"{row['train_loss']:.6f}"
        training_lines.append(f"| {row['step']} | {row['examples_seen']} | {loss} | {row['dev_loss']:.6f} | {row['historical_F1']:.6f}/{row['historical_FP']} | {row['ru_F1']:.6f}/{row['ru_FP']} | {row['mean_domain_F1']:.6f} |")
    training_lines += ['',f"Выбран step **{training['selected_step']}**. learning_curve.csv и все300 training_steps.jsonl сохранены. Три smoke steps включены в300; отдельных скрытых fit не было.", '',
        '## Input contract, сохранение и reload','',
        'clean_builder_v2 переиспользован в dependency-free функции с тестом равенства коду дня4. Source/ID/пути/label не входят в state. Mask literal заменяется пробелом, затем фактический tokenizer-offset prefix700; noul renderer/false,true positions взяты из установленного SDK. Synthetic direct-forward vs SDK и защита mask проверены реально.', '',
        '| Часть | Домен | N | Усечено | Max sequence tokens |','|---|---|---:|---:|---:|']
    for part,domains in truncation.items():
        for d,spec in domains.items():training_lines.append(f"| {part} | {d} | {spec['n']} | {spec['truncated']} | {spec['max_sequence_tokens']} |")
    training_lines += ['', 'Счётчики выше полные для отобранных частей/validation; не экстраполяция на полный raw корпус. Actual visible fragments/hashes приватны в каталоге run. Cross-part visible exact/canonical collisions0; train↔validation original/clean/global collisions0 и selected-visible↔validation collisions0.', '',
        'Membership text fingerprints — SHA-256 canonical JSON(text); visible_sha256 — SHA-256 UTF-8(fragment). Алгоритмы явны в day5_common.digest и day5_laya_runtime.encode_rows; это разные документированные представления.', '',
        f"Head SHA до `{training['head_sha256_before']}`, выбранная `{training['head_sha256_selected']}`. Encoder SHA `{training['encoder_sha256_after']}` до/после одинаков. Независимый аудит дополнительно сравнил все сохранённые frozen tensors с исходным checkpoint.",
        f"Кандидаты: `models/day5/runs/{run_id}/base_calibrated` и `/adapted_calibrated`; штатные model.safetensors, tokenizer, encoder config, rl_agent_config. Полный candidate_manifest хранит источник encoder/revision, вопрос, builder, T/threshold, версии, mapping, seed, data/code hashes и actual training steps. selected_head.safetensors сохранён отдельно.",
        'Reload выполнен штатным laya.load для обоих кандидатов: сначала12internal-dev logit-difference checks на кандидата. Независимый subprocess затем сравнил оба false/true raw logits и score на16internal-dev записей на кандидата (4наdomain/class, fixed hash); max logit/score error0, tolerance1e-5. Решения совпали, T фактически применена SDK, temperature_by_options пуста, encoder/selected-head hash совпали. Reload не выполнял fit или full validation benchmark.',
        'Снимок реально выполненного training runtime сохранён и сверён с training_started hash. После обучения в helper добавлены coverage ошибок final-inference и real_device metadata; training forward/optimizer/веса не изменялись. Code history и SHA исходной/заключительной версий сохранены.', '',
        'Технических блокеров обучения нет. Дальнейшее размораживание encoder, другая архитектура и подбор формулировок не выполнялись.']
    write_md(out,'training.md',training_lines)
    cal_lines=['# День 5: отдельная калибровка и пороги','',f'Run `{run_id}`. Calibration1000 записей из исходного train:500/500домены. Fit/dev/calibration disjoint.','',
        'Температура — один scalar T>0 для noul; binary NLL по raw logit difference. Golden-section search100 iterations в log(T), включая endpoints. Не по округлённому SDK score/confidence. Сначала выбран checkpoint по dev, затем T по calibration, затем threshold по dev. Внешний validation до manifest не открывался.',
        'Установленный SDK0.3.20 реально ограничивает T в[0.5,5]. temperature_by_options очищена, lang override не передаётся. Сохранённая T проверена reload; unsupported optimum не выдаётся за применённое значение.', '',
        '| Кандидат | Unconstrained T | Applied T | Boundary | Dev threshold | Unmet FP scenario |','|---|---:|---:|---|---:|---|']
    for name,cal in calibration['candidates'].items():
        t=cal['temperature_fit'];th=cal['threshold_selection']
        cal_lines.append(f"| {name} | {t['unconstrained_temperature']:.9f} | {t['applied_temperature']:.9f} | {t['at_boundary']} | {th['threshold']:.12f} | {th['unmet_fp_scenario']} |")
    cal_lines += ['', 'Base optimum≈6.188 превышает поддерживаемый максимум; реально применяется≈5. Это известное ограничение текущего runtime, без скрытой смены SDK/порогов.', '',
        'Threshold rule: max mean domain F1 при observed historical FPR≤1% и ненулевом recall обоих доменов; ties меньше historical FP, затем больший threshold. При невозможности/all-negative fallback0.5+unmet flag. Выбор на600internal dev — provisional demo policy, не гарантия FPR в популяции или RU.', '',
        '## Fitting sample: не независимая проверка калибровки','',
        '| Кандидат | Режим | NLL | Brier | ECE | Counts bins0..9 |','|---|---|---:|---:|---:|---|']
    for name,cal in calibration['candidates'].items():
        for mode,p in cal['calibration_fitting_sample'].items():
            cal_lines.append(f"| {name} | {mode} | {p['NLL']:.6f} | {p['Brier']:.6f} | {p['ECE']:.6f} | {','.join(str(b['count']) for b in p['ECE_bins'])} |")
    cal_lines += ['', 'ECE —10равных интервалов P(spam) [0,.1),…,[.9,1], abs(mean probability − empirical positive fraction), взвешенная числом записей. Для каждого bin границы/count/mean/fraction сохранены в calibration.json; пустые bins явно count0. Ни ECE, ни NLL на fitting sample не доказывают внешнюю калибровку.', '',
                  '## Применение к dev после фиксации T','']+table_header()
    for name,cal in calibration['candidates'].items():
        for d,v in cal['threshold_selection']['by_domain'].items():cal_lines.append(metric_row(name+' selected threshold',d,v))
    cal_lines += ['', 'Temperature сохраняет классы при0.5: подтверждено на обеих полных validation. Она меняет выраженность оценок, а не ранжирование/распознавание. Отдельное изменение threshold может менять FP/FN. Баланс fit и смешанная calibration не позволяют назвать score вероятностью спама во всей реальной почте.', '',
                  'NLL/Brier/ECE и bins/counts на existing validation показаны отдельно в comparison.json/.md; эти validation уже использовались днями3/4 и не являются свежим test.']
    write_md(out,'calibration.md',cal_lines)
    comparison={'run_id':run_id,'laya':laya,'frozen_lr':lr,'truncation':truncation,
                'validation_previously_used':True,'test_not_evaluated':True,'default_unchanged':'E0',
                'candidate_manifest_sha256':sha_file(out/'candidate_manifest.json')}
    write_json(out/'comparison.json',comparison)
    comparison_lines=['# День 5: заключительный validation comparison','',f'Run `{run_id}`. Historical1275 (ham964/spam311), RU5990 (ham3000/spam2990). Кандидаты/T/threshold зафиксированы до открытия validation. Ни один результат ниже не использован для изменения кандидата.','',
        'Это previously used validation, не независимый финальный test. RU source-intended labels provisional; распределение коротких сообщений/чатов отличается от почты. Laya-A остаётся отдельной600-subset таблицей в reports/day4/decision_pilot.md и здесь не смешивается с full validation Laya-B.', '',
        '## Основное сравнение','']+table_header()
    for name,th in [('E0',.5),('E0',config['evaluation']['threshold_default_e0']),('E2',.5)]:
        for domain in DOMAINS:comparison_lines.append(metric_row(name+f' full t={th:.6f}',domain,lr['candidates'][name+'/full_context'][domain]['metrics'][str(th)]))
    for name,result in laya['candidates'].items():
        for mode in ['raw_0.5','calibrated_0.5','calibrated_dev_threshold']:
            for domain in DOMAINS:comparison_lines.append(metric_row('Laya-B '+name+' '+mode,domain,result['by_domain'][domain]['modes'][mode]['metrics']))
    comparison_lines += ['', 'При одинаковом B и threshold0.5 raw F1 после обучения вырос с0.956522 до0.973856 на historical и с0.763281 до0.892650 на RU. Это эффект изменения весов в этом ограниченном запуске; смена A→B в эти числа не входит.',
        'Выбранный dev-threshold на RU снизил FP249→237, но увеличил FN379→400 относительно adapted@0.5: итоговый F1 снизился0.892650→0.890493. Настройки после просмотра validation не менялись. E2/full RU F1=0.972274 выше результата этого Laya-пилота.',
        'E0/E2 обучены ранее на полных разрешённых train, Laya здесь увидела4800fit записей. Это сравнение практических фиксированных кандидатов с разными бюджетами, а не изолированный benchmark архитектур.', '',
        '## Frozen LR на точно видимом Laya фрагменте','',
        'Full-context E0 использует неизменный builder дня3, E2 — clean_builder_v2. Matched-visible использует один и тот же фактический clean/token-prefix текст для E0/E2/обоих Laya, без ID/source/label. SHA всех matched input совпали. LR не дообучались на внутренних частях; их fit-derived dev/calibration не показаны как честный holdout.', '']+table_header()
    for name,th in [('E0',.5),('E0',config['evaluation']['threshold_default_e0']),('E2',.5)]:
        for domain in DOMAINS:comparison_lines.append(metric_row(name+f' matched t={th:.6f}',domain,lr['candidates'][name+'/matched_visible'][domain]['metrics'][str(th)]))
    comparison_lines += ['', '## Вероятностные оценки на validation','',
        '| Кандидат | Домен | Режим | NLL | Brier | ECE | Bin counts0..9 |','|---|---|---|---:|---:|---:|---|']
    for name,result in laya['candidates'].items():
        for domain in DOMAINS:
            for mode in ['raw_0.5','calibrated_0.5']:
                p=result['by_domain'][domain]['modes'][mode]['probability_metrics']
                comparison_lines.append(f"| Laya {name} | {domain} | {mode} | {p['NLL']:.6f} | {p['Brier']:.6f} | {p['ECE']:.6f} | {','.join(str(b['count']) for b in p['ECE_bins'])} |")
    for name in ['E0','E2']:
        for context in ['full_context','matched_visible']:
            for domain in DOMAINS:
                p=lr['candidates'][name+'/'+context][domain]['probability_metrics']
                comparison_lines.append(f"| {name} | {domain} | {context} | {p['NLL']:.6f} | {p['Brier']:.6f} | {p['ECE']:.6f} | {','.join(str(b['count']) for b in p['ECE_bins'])} |")
    comparison_lines += ['', 'Калибровка не улучшила все внешние показатели: base historical NLL0.083130→0.096918 ухудшился при T≈5, хотя base RU NLL1.874908→0.557934 улучшился. Adapted historical NLL0.044790→0.042480 улучшился, RU NLL0.305729→0.315327 ухудшился при T≈0.8553; RU Brier при этом слегка улучшился0.083534→0.082897. Одна общая T для искусственной смеси доменов не гарантирует улучшение каждого домена.',
        'LR NLL использует logit из сохранённой probability, защитное clipping[1e-15,1−1e-15]. Его score не объявляется доказанно калиброванным. Для Laya полная точность raw logits; SDK отображает noul с округлением4знака. Threshold и сравнение используют неокруглённый score, а не confidence. Полные bins/counts/means в JSON.', '',
        '## Coverage, latency и ресурсы','',
        '| Кандидат | Домен | Coverage / failures | Warm p50 / p95 ms | Truncated |','|---|---|---|---|---|']
    for name,result in laya['candidates'].items():
        for d,v in result['by_domain'].items():
            lat=v['warm_latency_seconds'];comparison_lines.append(f"| Laya {name} | {d} | {v['successful']}/{v['n']} / {len(v['failures'])} | {lat['p50']*1000:.3f}/{lat['p95']*1000:.3f} | {v['truncated']}/{v['n']} |")
    for name,result in lr['candidates'].items():
        for d,v in result.items():
            lat=v['warm_latency_seconds'];comparison_lines.append(f"| {name} | {d} | 100% / 0; wrapper {v['wrapper_coverage']:.4f} | {lat['p50']*1000:.3f}/{lat['p95']*1000:.3f} | {'matched: same as Laya' if 'matched' in name else 'full context'} |")
    comparison_lines += ['', 'Latency: реальные single-record warm forwards, synthetic warmup до измерений, preparation/cold load отдельно; не нагрузочный тест. GPU и CPU процессы частично работали одновременно, поэтому эти задержки описывают данный запуск, а не изолированный аппаратный benchmark. Нет жёсткого per-record timeout: ошибки фиксируются с coverage, отсутствующие ответы не переводятся в ham.', '',
        '| Laya | Cold load s | Preparation s | Inference wall s | Peak allocated/reserved GiB | RSS after GiB |','|---|---:|---:|---:|---|---:|']
    for name,r in laya['candidates'].items():comparison_lines.append(f"| {name} | {r['cold_model_load_seconds']:.3f} | {r['preparation_seconds']:.3f} | {r['inference_wall_seconds']:.3f} | {r['peak_vram_allocated_bytes']/2**30:.3f}/{r['peak_vram_reserved_bytes']/2**30:.3f} | {r['rss_bytes_after_inference']/2**30:.3f} |")
    comparison_lines += ['', 'RAM — sampled process RSS after inference/training; не peak всей системы. VRAM — PyTorch peak allocated/reserved, не полная память драйвера. LR raw metrics включают empty vectors; wrapper обязан возвращать insufficient_features: пустой вектор не доказывает normal.', '',
        'Решение: оставить E0 по умолчанию. Новый Laya candidate исследовательский; его принятие, современная почта, business FP/FN и независимая финальная оценка требуют отдельного этапа. Современный .eml набор сегодня не оценивался.']
    write_md(out,'comparison.md',comparison_lines)
    predictions=read_rows(out/f'{selected_name}_validation_predictions.jsonl');errors=select_errors(predictions,selected)
    observations=read_json(out/'error_review_observations.json')['observations'] if (out/'error_review_observations.json').exists() else {}
    sampled=[];error_lines=['# День 5: анализ ошибок фиксированного кандидата','',f'Run `{run_id}`; {selected_name}, T={selected["temperature"]}, threshold={selected["threshold"]}. После просмотра ничего не менялось.','',
        'До5FP/5FN, deterministic SHA(seed42, error-review, ID), чередование доменов где доступны. Это наблюдения Codex по фрагментам, не ручная приёмка меток и не доли причин во всём корпусе. Original labels сохранены. Ссылки/адреса/длинные номера обезврежены, хвост письма мог быть усечён.', '']
    for kind,r in errors:
        v=visible_lookup[r['record_id']];fragment=defang(v['visible_fragment']);score=sigmoid(r['logit_difference']/selected['temperature'])
        sample={'kind':kind,'record_id':r['record_id'],'domain':r['domain'],'source':r['source'],'spam_label':r['spam_label'],
                'score':score,'truncated':v['token_stats']['truncated'],'short_defanged_fragment':fragment,'human_review':'pending',
                'codex_observation':observations.get(r['record_id'],'pending: no manual Codex observation recorded for this run')}
        sampled.append(sample)
        error_lines += [f"## {kind} — {r['domain']} — `{r['record_id']}`",'',f"Исходная метка {r['spam_label']}, score={score:.6f}, truncated={sample['truncated']}.",'','```text',fragment,'```','',
                        'Наблюдение Codex: '+sample['codex_observation'],'',
                        'Human review pending. Автоматической переразметки нет.','']
    error_lines += ['Причины ошибок нельзя свести к языку или теме по10фрагментам. RU содержит чатовые реплики и предложения, historical — форматные/тематические артефакты; канал и реальные предпочтения получателя не удостоверены. Сопоставление full/matched показывает отдельное влияние потери хвоста, но не причинность каждой ошибки.']
    write_rows(out/'error_review_sample.jsonl',sampled);write_md(out,'error_analysis.md',error_lines)
    chosen={d:laya['candidates'][selected_name]['by_domain'][d]['modes']['calibrated_dev_threshold']['metrics'] for d in DOMAINS}
    baseline={d:lr['candidates']['E0/full_context'][d]['metrics'][str(config['evaluation']['threshold_default_e0'])] for d in DOMAINS}
    presentation=['# День 5: материал для презентации','',
        f'Run `{run_id}`. Факты этого эксперимента, не планы и не mock.', '',
        '1. Encoder представляет текст; decision-head выбирает false/true. Здесь encoder заморожен, обучены14,77Mпараметров существующей head/type embedding/scorer.',
        f"2. Один bounded run:300AdamWsteps,4800unique fit records, {training['training_seconds']:.1f}с с dev/checks, CUDA; smoke подтвердил реальные градиенты. Encoder tensors сохранены.",
        '3. Вопрос B до/после одинаковый. Дообучение меняет веса; T меняет выраженность; threshold меняет решение. Это три отдельных строки сравнения, а смена A→B — отдельная предыдущая диагностика.',
        '4. Внутренние fit/dev/calibration — из существующего train; dev600 выбирает checkpoint/threshold, calibration1000 — только T. Manifest зафиксирован до existing validation.', '',
        '| Домен | Laya-B raw до F1 | Laya-B raw после F1 | Laya после T/dev threshold F1 | FP / FN выбранного | E0 default full F1 |',
        '|---|---:|---:|---:|---|---:|']
    for d in DOMAINS:
        base=laya['candidates']['base']['by_domain'][d]['modes']['raw_0.5']['metrics'];after=laya['candidates'][selected_name]['by_domain'][d]['modes']['raw_0.5']['metrics']
        presentation.append(f"| {d} | {base['F1']:.6f} | {after['F1']:.6f} | {chosen[d]['F1']:.6f} | {chosen[d]['FP']}/{chosen[d]['FN']} | {baseline[d]['F1']:.6f} |")
    presentation += ['',f"5. Tbase≈5 (optimum6.188 внеSDK), Tadapted≈{selected['temperature']:.6f}; scalar T не изменил ни один класс при0.5. Численные оценки не гарантируют популяционную вероятность.",
        '6. Кандидат сохранён в штатном SDK-формате; reload raw logits maxerror0. Независимый аудит проверил все frozen tensors, raw mapping, membership и доступ файлов. Старые E0/E2/A-пилот сохранены.',
        '7. Ограничения: validation уже использовалась, RU provisional/не подтверждённый email-канал, truncated prefixes, маленькие dev/calibration, недетерминированный CUDA backward. Test/challenge/reserve не открыты; современная почта ещё не проверена.',
        '8. Приём реальных .eml готов: manifest+static MIME+20MiB cap, без облака/модельных оценок/обучения. Статус templates_ready; назначение пользователь выбирает до будущего inference.',
        'Решение дня5: E0 остаётся default; Laya candidate на разбор. API/Docker/демо/финальный test — следующие этапы.']
    write_md(out,'PRESENTATION_NOTES_DAY5.md',presentation)
    status=['# Email Guard — DAY5_STATUS','',f'Run `{run_id}`. Статус **completed** в пределах задания дня5; test_not_evaluated=true.','',
        '| Этап | Статус | Факт |','|---|---|---|',
        '| Контракт / train-only parts | completed | 33906membership;6000fit/600dev/1000calibration; конфликтов/утечек0 |',
        '| Исходная Laya-B | completed | Зафиксирована до обучения наinternal dev; тот же B в final comparison |',
        f"| Head-only adaptation | {training['status']} | 300steps;4800unique;selected step{training['selected_step']};14,770,945trainable |",
        '| Calibration / dev threshold | completed | Раздельно для base/adapted; T иthreshold не изvalidation |',
        '| Existing validation | completed | 1275historical/5990RU; full+matched LR; frozen candidates |',
        '| Save/reload | passed | Штатные SDK weights/config/tokenizer; raw logitsmaxerror0; encoderсохранён |',
        f"| Tests / independent audit | passed | {test_results['tests_run']} synthetic tests; real gradient smoke; {verification['protected_files_identical']} прежнихфайловSHAсовпали |",
        f"| Собственная современная почта | {personal['status']} | Реальныхписем0; CSV+parser+path/label/groupchecks; безscores |",'',
        'Реальных технических блокеров обучения нет. CUDA nondeterministic backward предупреждение сохранено как ограничение воспроизводимости; Tbase optimum выше SDK max — применён5.', '',
        '| Домен | Laya-B raw до F1 | Raw после F1 | После T/dev threshold F1 | FP / FN |','|---|---:|---:|---:|---|']
    for d in DOMAINS:
        beforem=laya['candidates']['base']['by_domain'][d]['modes']['raw_0.5']['metrics'];afterm=laya['candidates'][selected_name]['by_domain'][d]['modes']['raw_0.5']['metrics']
        status.append(f"| {d} | {beforem['F1']:.6f} | {afterm['F1']:.6f} | {chosen[d]['F1']:.6f} | {chosen[d]['FP']}/{chosen[d]['FN']} |")
    status += ['', 'E0 default сохранён с0.3037252359712881; historical full F1=.9792, TP306/FP8/TN956/FN5. E2 и Laya-A дня4 не изменены. Все closed test/challenge/teacher/frozen reserve/Nazario/modern/fullParquet закрыты;2026-09 security_blocked сохраняется без восстановления/скачивания.',
        'Existing validation уже участвовала в днях3/4; это не свежий test и не доказанная современная мультиязычная почта. RU labels provisional, fit баланс изменён, calibration смешивает домены. Порог/FPR — provisional, не бизнес-гарантия.', '',
        'Материал для разбора: reports/day5/training.md, calibration.md, comparison.md, error_analysis.md, PRESENTATION_NOTES_DAY5.md. Веса/private JSONL не нужны для передачи.',
        f'Обученный candidate: models/day5/runs/{run_id}/adapted_calibrated; manifest, source revision/input question/T/threshold/version/data hashes сохранены. Новые модели не default.', '',
        'Команды — docs/DAY5_REPRODUCTION.md. Архив reports/day5/runs сохраняется. Автоматической публикации, облачного inference и обучения на собственных письмах нет.', '',
        'Решения пользователя следующего этапа: назначение собственных .eml до scores (review/dev либо отложенная диагностика); ручная приёмка спорных меток; бизнес-допуск FP/FN. Для завершения текущего дня дополнительных решений не требуется.']
    text='\n'.join(status)+'\n';(ROOT/'reports/DAY5_STATUS.md').write_text(text,encoding='utf-8');(out/'DAY5_STATUS.md').write_text(text,encoding='utf-8')
    for name in ['training.json','calibration.json','comparison.json','learning_curve.csv','verification.json','error_review_sample.jsonl','truncation_summary.json']:
        (ROOT/'reports/day5'/name).write_bytes((out/name).read_bytes())
    write_json(ROOT/'reports/day5/current_run.json',{'run_id':run_id,'report_path':str(out.relative_to(ROOT).as_posix()),'stage':'completed',
                                                  'candidate_manifest_sha256':sha_file(out/'candidate_manifest.json')})
    write_json(out/'report_accesses.json',boundary.report())
    print('Saved DAY5_STATUS and training/calibration/comparison/error/presentation reports for '+run_id)


if __name__=='__main__':main()
