"""Private inert validation reports; no test dataset access or model selection."""
import json
from collections import Counter

import numpy as np

from baseline_common import explain, coefficients, digest_json
from normalization_reporting import fragment


LIMITATIONS=('Исторический английский text-only baseline. Ling-Spam: mapping_inferred; source-level метки не заменяют ручную верификацию. '
    'Источники имеют тематические и форматные артефакты; source не входит в признаки, но текст может его выдавать. '
    'Отложенный spam_email/test и все дополнительные корпуса не читались и не оценивались. '
    'Сентябрь2026 security_blocked; needs_review Untroubled исключён из экспериментов. '
    'Мультиязычность, современная почта, устойчивость к обходам, phishing и категории не проверены. '
    'Spam_score — модельная оценка, её калибровка не доказана. CV/OOF использованы для выбора, validation — для порога; '
    'это не независимый финальный тест и не подтверждение требования курса F1>=0.86. '
    'Нужны свежий проверенный ham, phishing-negative, тематическая разметка и доступный тест преподавателя.')


def number(x):return 'undefined' if x is None else f'{x:.6f}'


def metric_table(rows):
    lines=['| Вариант | Precision | Recall | F1 | Accuracy | FPR | TP | FP | TN | FN | Support ham/spam |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,m in rows:
        lines.append(f"| {name} | "+' | '.join(number(m[k]) for k in ['precision','recall','f1','accuracy','fpr'])+
            ' | '+' | '.join(str(m[k]) for k in ['TP','FP','TN','FN'])+f" | {m['support']['0']}/{m['support']['1']} |")
    return lines


def render_experiments(report):
    c=report['contract']
    lines=['# День 3: три исторических baseline', '',f"Run: `{report['run_id']}`; статус **{report['status']}**. Принято только accepted_for_historical_baseline.", '',
        f"Входы: train={c['inputs']['train']['rows']} ({json.dumps(c['inputs']['train']['classes'])}), validation={c['inputs']['validation']['rows']} ({json.dumps(c['inputs']['validation']['classes'])}). SHA/пути/config/code/версии — experiments.json и contract.json данного run.", '',
        '## Кросс-валидация', '',f"{c['cv_basis']}; 5 общих фолдов. Оба класса в каждом. Порог сравнения 0.5; positive=spam=1. Для каждого fit Pipeline получает только обучающую часть: словарь/IDF/классификатор обучаются внутри фолда. Каждая train-запись имеет ровно одну OOF-оценку от не видевшей её модели. Никакого fit_transform на полном train перед CV.", '',
        '| Модель | Mean F1 | Std F1 (ddof=0) | Mean precision | Mean recall | Pooled OOF F1 |', '|---|---:|---:|---:|---:|---:|']
    for k,r in report['models'].items():
        cv=r['cv'];lines.append(f"| {k} | {number(cv['mean']['f1'])} | {number(cv['std_population_ddof0']['f1'])} | {number(cv['mean']['precision'])} | {number(cv['mean']['recall'])} | {number(cv['pooled_oof']['f1'])} |")
    lines += ['',f"Baseline **{report['selected_baseline']}** выбран до validation: {report['selection']}. Численное равенство: допуск 1e-12. Undefined precision в tie-break уступает определённой; не подменяется идеальной оценкой.", '',
        'Mean/std — по фолдам, pooled OOF — единая confusion на всех OOF-предсказаниях. Это разные агрегаты и оценки для выбора, не независимый test.', '',
        '## Validation при фиксированном 0.5', '']
    lines+=metric_table([(k,r['validation_default']) for k,r in report['models'].items()]+[('Dummy most_frequent',report['dummy_most_frequent']['validation_default'])])
    lines+=['',f"Порядок на validation: {', '.join(report['validation_ranking'])}. Выбранный baseline по этой таблице не менялся. Dummy обучен только выбору наиболее частого класса, без текстовых признаков/CV-поиска.", '',
        '## Validation по источникам при 0.5', '']
    lines+=metric_table([(k+'/'+source,m) for k,r in report['models'].items() for source,m in r['validation_by_source'].items()]+
        [('Dummy/'+s,m) for s,m in report['dummy_most_frequent']['validation_by_source'].items()])
    lines+=['','Нулевой знаменатель даёт JSON null / undefined (семантика sklearn zero_division=np.nan). F1=2TP/(2TP+FP+FN); FPR=FP/(FP+TN). Confusion и support приведены явно.', '',
        '## Настройки, время и сохранение', '',
        'TF-IDF: lowercase=True, sublinear_tf=True, norm=l2, stop_words=None, float64; word (1,2), min_df=2, max_features=80000; char_wb (3,5), min_df=3, max_features=150000. NB alpha=1.0. LR C=1.0, liblinear, L2 через l1_ratio=0, class_weight=None, max_iter=2000, random_state=42, tol=1e-4. Lowercase — преобразование эксперимента, оригиналы не переписаны.', '',
        'scikit-learn1.9.1: penalty имеет служебное значение deprecated в get_params, L2 задана l1_ratio=0; n_jobs не задаётся. Полный фактический deep get_params каждого fit сохранён в JSON, включая возможное увеличение max_iter при ConvergenceWarning.', '',
        '| Модель | Фолд | Признаков | Fit, с | Предупреждений/попыток |', '|---|---|---:|---:|---|']
    for k,r in report['models'].items():
        for f in r['cv']['folds']+[{'fold_id':'full train','fit':r['full_train_fit']}]:
            fit=f['fit'];n=sum(len(a['warnings']) for a in fit['attempts'])
            lines.append(f"| {k} | {f['fold_id']} | {fit['vocabulary_size']} | {fit['seconds']:.3f} | {n}/{len(fit['attempts'])} |")
    lines+=['',f"Реально завершено {report['fit_contexts_completed']} fit-контекстов (15 CV + 3 full train). Последовательно, CPU, BLAS threads={c['threads']}; корпусные sparse-матрицы не превращались в dense. Общее время {report['seconds']:.2f} с.", '',
        f"Версии: `{json.dumps(c['versions'])}`. Reload всех трёх Pipeline: `{json.dumps(report['reload_checks'])}`; bundle max score error={report['bundle_reload_max_absolute_error']:.3g}.", '',
        f"Доступ к проектным данным во время runner: `{json.dumps(report['data_access_log'])}`. Хеши закрытых артефактов скопированы исключительно из manifest, файлы не открывались. Старый общий audit дня2 не запускался.", '',
        'OOF/validation predictions содержат только ID/группу/source/label/score/порог/fold, без текстов. Модели и отчёты версионированы по run_id; latest-копии ссылаются на сохранённый run.', '',LIMITATIONS,'']
    return '\n'.join(lines)


def render_thresholds(report):
    lines=['# День 3: пороги выбранного baseline', '',f"Run `{report['run_id']}`; baseline **{report['baseline']}**, выбран по CV до validation.", '',
        f"Правило: {report['rule']}. Сетка {report['grid_size']} значений: уникальные validation scores, 0.5 и nextafter(max_score,+inf), допускающий отсутствие положительных предсказаний. Порог выбирался только у этого baseline.", '']
    lines+=metric_table([(name+f" t={row['threshold']:.17g}",row['metrics']) for name,row in report['modes'].items() if row is not None])
    lines+=['',f"Provisional policy: **{report['provisional_policy']}**, threshold={report['provisional_threshold']:.17g}; unmet_fp_scenario={report['unmet_fp_scenario']}.", '',
        'Режим conservative максимизирует validation F1 при наблюдаемом FPR<=1%. Это предварительный инженерный сценарий, не SLA и не гарантия будущих FP. Если recall=0/все отрицательные, режим не признаётся рабочим; fallback — 0.5 с unmet_fp_scenario. Режим максимального validation F1 исследовательский.', '',
        '## 95% Wilson intervals для долей', '', '| Режим | Recall TP/(TP+FN) | Wilson95 | FPR FP/(FP+TN) | Wilson95 |', '|---|---|---|---|---|']
    for name,row in report['modes'].items():
        if row is None:continue
        a,b=row['recall_wilson95'],row['fpr_wilson95']
        fmt=lambda r:'undefined' if r['interval'] is None else '['+', '.join(f'{v:.6f}' for v in r['interval'])+']'
        lines.append(f"| {name} | {a['numerator']}/{a['denominator']} | {fmt(a)} | {b['numerator']}/{b['denominator']} | {fmt(b)} |")
    lines+=['','Wilson описывает наблюдаемую долю при предпосылках биномиальной модели; это не CI для F1 и не защита от дрейфа, ошибок меток или зависимости кампаний. Validation использована для настройки порога, поэтому оценки после выбора оптимистичны; интервалы не учитывают этот выбор.', '',
        '## Источники в трёх режимах','']
    lines+=metric_table([(name+'/'+s,m) for name,row in report['modes'].items() if row is not None for s,m in row['by_source'].items()])
    lines+=['',LIMITATIONS,'']
    return '\n'.join(lines)


def feature_display(value,char=False):
    return fragment(repr(value) if char else value)


def error_analysis(models,selected,data,score,thresholds):
    model=models[selected];threshold=thresholds['provisional_threshold']
    categories={k:[] for k in ['FP','FN','TP','TN']}
    for i,(r,s) in enumerate(zip(data,score)):
        prediction=int(s>=threshold)
        kind='TP' if r.label==1 and prediction else 'FN' if r.label==1 else 'FP' if prediction else 'TN'
        categories[kind].append(i)
    chosen=[]
    for kind in ['FP','FN','TP','TN']:
        ordering=(lambda i:(-score[i],data[i].record_id)) if kind=='FP' else (lambda i:(score[i],data[i].record_id)) if kind=='FN' else (lambda i:digest_json(data[i].record_id))
        chosen.extend((kind,i) for i in sorted(categories[kind],key=ordering)[:5 if kind in {'FP','FN'} else 2])
    global_features={}
    for model_id,pipeline in models.items():
        if model_id not in {'word_lr','char_lr'}:continue
        coef,intercept,kind=coefficients(pipeline);names=pipeline['tfidf'].get_feature_names_out()
        positive=sorted(np.flatnonzero(coef>0),key=lambda j:(-coef[j],int(j)))[:15]
        negative=sorted(np.flatnonzero(coef<0),key=lambda j:(coef[j],int(j)))[:15]
        convert=lambda indexes:[{'feature_index':int(j),'feature_display':feature_display(str(names[j]),model_id=='char_lr'),'weight':float(coef[j])} for j in indexes]
        global_features[model_id]={'intercept':intercept,'direction':'positive increases logit spam=1; classes checked','positive':convert(positive),'negative':convert(negative)}
    examples=[]
    for kind,i in chosen:
        r=data[i];explanation=explain(model,r.text)
        for feature in explanation['top_features']:
            feature['feature_display']=feature_display(feature.pop('feature'),selected=='char_lr')
        analyzer=model['tfidf'].build_analyzer();tokens=analyzer(r.text);vocab=model['tfidf'].vocabulary_
        coverage=sum(t in vocab for t in tokens)/len(tokens) if tokens else None
        examples.append({'kind':kind,'record_id':r.record_id,'duplicate_group_id':r.group_id,'source':r.source,'spam_label':r.label,
            'spam_score':float(score[i]),'threshold':threshold,'prediction':int(score[i]>=threshold),'fragment':fragment(r.text),
            'parse_status':r.parse_status,'label_status':r.label_status,'text_chars':len(r.text),
            'analyzed_token_occurrence_coverage':coverage,'explanation':explanation,
            'observation':'score/порог и знак суммы вкладов измерены; коэффициенты показывают связи с обучающим корпусом, не доказанную причину ошибки. '+
                ('Низкое покрытие словарём (<20%) наблюдается; недостаток знакомых признаков — гипотеза.' if coverage is not None and coverage<.2 else 'Покрытие словарём приведено явно; само по себе оно не объясняет метку.')+
                (' Исходный parse_status=partial: возможен форматный шум, без автоматической коррекции.' if r.parse_status=='partial' else '')})
    result={'baseline':selected,'threshold':threshold,'data_scope':'validation only','counts':{k:len(v) for k,v in categories.items()},
        'selection':'up to 5 most-confident FP, 5 most-confident FN, 2 deterministic TP and TN; no rule tuning','examples':examples,
        'global_lr_features':global_features,'labels_or_model_changed_from_review':False}
    lines=['# День 3: приватный анализ validation', '',f"Baseline {selected}; provisional threshold={threshold:.17g}. Только validation: `{json.dumps(result['counts'])}`. {result['selection']}.", '',
        'Фрагменты до160 символов, HTML/Markdown экранированы, контакты/URL скрыты только здесь. Обучающие тексты не менялись. Примеры не использовались для переразметки или повторного подбора модели. Нулевые количества ошибок не дополняются выдуманными примерами.', '',
        '## Глобальные LR-признаки', '',
        'Знак проверен по classes_: плюс увеличивает logit spam=1. Это ассоциации корпуса, не причинность. Char_wb показаны с границами через repr/экранирование: это фрагменты, а не слова. Полный словарь хранится приватно вместе с каждой моделью.', '']
    for model_id,g in global_features.items():
        lines += [f"### {model_id}; intercept={g['intercept']:.9g}", '', '| Направление | Признак (обезврежен) | Вес |', '|---|---|---:|']
        for side in ['positive','negative']:
            for f in g[side]:lines.append(f"| {side} | {f['feature_display']} | {f['weight']:.9g} |")
        lines.append('')
    lines+=['## Локальные примеры и реальные вклады', '']
    for e in examples:
        x=e['explanation']
        lines += [f"### {e['kind']} — `{e['record_id']}`", '',f"Источник {e['source']}; группа `{e['duplicate_group_id']}`; label={e['spam_label']}, prediction={e['prediction']}, spam_score={e['spam_score']:.9g}, threshold={threshold:.9g}. label_status={e['label_status']}; parse={e['parse_status']}.",
            f"Фрагмент: {e['fragment']}", '',f"Длина={e['text_chars']}, покрытие вхождений анализатора словарём={number(e['analyzed_token_occurrence_coverage'])}, активных признаков={x['active_features']}. {e['observation']}", '',
            f"{x['kind']}: сумма ВСЕХ вкладов={x['all_contributions_sum']:.12g}; intercept={x['intercept']:.12g}; вместе={x['sum_plus_intercept']:.12g}; модель={x['expected_log_score']:.12g}; ошибка={x['absolute_error']:.3g}.", '',
            '| Присутствующий признак | TF-IDF x | Вес w | Вклад x*w |', '|---|---:|---:|---:|']
        for f in x['top_features']:lines.append(f"| {f['feature_display']} | {f['tfidf']:.9g} | {f['weight']:.9g} | {f['contribution']:.9g} |")
        lines+=['',f"Top-N сумма={x['top_contributions_sum']:.12g} — неполная часть общей суммы. Это вклады в logit/log-score ratio, не вероятности. Intercept показан отдельно; равенство проверено до/после сохранения модели на выбранной validation-подвыборке.", '']
    lines+=['## Наблюдения и гипотезы для дня 4', '',
        'Наблюдения выше ограничены выбранными validation-примерами; полный source-разрез находится в experiments.md/thresholds.md. Тематические подсказки Ling-Spam и артефакты SpamAssassin могут влиять на коэффициенты. Низкое покрытие/короткий текст/partial parsing — измеряемые признаки, но без отдельного контролируемого сравнения не доказанные причины FP/FN.', '',
        'Следующий день: вручную проверить спорные validation-метки и контекст; проверить устойчивость к тематическим/форматным артефактам на разрешённых новых данных; уточнить стоимость FP/FN и сценарий порога; отдельно проверить калибровку. Изменения оформлять новым экспериментом, не подстраивать правила по закрытому test. Ничего из этого сегодня к модели не применялось.', '',LIMITATIONS,'']
    return '\n'.join(lines),result


def render_status(state,experiments,thresholds,smoke):
    metrics=thresholds['modes']['conservative' if not thresholds['unmet_fp_scenario'] else 'default']['metrics']
    lines=['# Email Guard — DAY3_STATUS', '',f"Статус: **completed** для исторического baseline дня3. Run `{state['run_id']}`. test_not_evaluated=true. Это не приёмка всех дополнительных данных дня2.", '',
        '| Этап | Статус | Факт |','|---|---|---|',
        '| Входы/контракт | completed | accepted_for_historical_baseline; train5953/validation1275; SHA и группы сверены |',
        '| CV | completed | 15 fit, три Pipeline, общие 5 фолдов manifest; fit TF-IDF внутри каждого фолда |',
        '| Полный train/validation | completed | ещё 3 fit; Dummy most_frequent; source-разрезы |',
        f"| Baseline | completed | {experiments['selected_baseline']}; выбор по CV до validation |",
        f"| Порог | completed | {thresholds['provisional_policy']}, {thresholds['provisional_threshold']:.17g}; FP={metrics['FP']}, FN={metrics['FN']} |",
        '| Анализ ошибок | completed | validation only; реальные локальные суммы и глобальные LR-веса |',
        '| Сохранение/reload | completed | три Pipeline + bundle, score tolerance1e-12 |',
        f"| CLI smoke | completed | учебный service_notice.eml; subprocess={smoke['cli_subprocess_verified']}; без fit/изменения входа и моделей |",
        '| Закрытые наборы | test_not_evaluated | audithook запрет чтения; хеши только из manifest |','',
        f"Общее время {state['seconds']:.2f} с. Версии `{json.dumps(experiments['contract']['versions'])}`. Результаты/модели версионированы в reports/day3/runs и models/day3/runs, latest-копии имеют run_id и manifest.", '']
    if 'tests' in state:lines+=['Тесты: `'+json.dumps(state['tests'],ensure_ascii=False)+'`.','']
    lines += [LIMITATIONS, '', 'Реальных блокеров исторического baseline не осталось. Дополнительные задачи: финальное решение до открытия test; бизнес-сценарий FP/FN; внешняя проверка/новые источники/метки. security_blocked 2026-09 сохраняется без скачивания/восстановления.', '',
        '## Воспроизведение', '', '```powershell', '.\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v',
        '.\\.venv\\Scripts\\python.exe -B .\\src\\train_baselines.py',
        '.\\.venv\\Scripts\\python.exe -B .\\src\\predict_email.py .\\data\\examples\\service_notice.eml', '```', '',
        'Каждый запуск создаёт новый run_id, прежние результаты сохраняются. Общий audit/подготовку дня2 не запускать: они читают отложенные данные.', '',
        'Для разбора прислать только reports/DAY3_STATUS.md и reports/day3/experiments.md, thresholds.md, error_analysis.md. Большие JSONL/модели/данные не нужны.', '']
    return '\n'.join(lines)
