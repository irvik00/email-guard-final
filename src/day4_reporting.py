"""Private Day-4 reviews and concise evidence-based Markdown reports."""
import re
from collections import Counter

import numpy as np

from baseline_common import digest_json
from normalization_reporting import fragment
from russian_inventory import safe_fragment
from pipeline_common import write_json,write_text,JsonlWriter


def private_fragment(text,limit=1200):
    text=re.sub(r'(?<!\w)\+?\d[\d ()\t.-]{6,}\d(?!\w)','[номер]',text or '')
    return safe_fragment(text,limit)


def dump_rows(path,data):
    writer=JsonlWriter(path)
    try:
        for row in data:writer.write(row)
    finally:metadata=writer.close()
    return metadata


def field_state(record,key):
    availability=record.get('feature_availability') or {}
    group={'sender':'headers','sender_address':'headers','sender_domain':'headers'}.get(key,key)
    value=record.get(key)
    if availability.get(group)=='unavailable':return 'unavailable_null' if value is None else 'unavailable_nonnull'
    if value is None:return 'observable_missing' if availability.get(group)=='observable' else 'unknown_null'
    if value=='' or value==[]:return 'observable_empty'
    return 'observed_nonempty'


def review_data(path,sets,contract,e0_predictions,threshold,seed):
    reviews=[];field_stats={};data_stats={}
    for domain,ds in sets.items():
        for split,data in ds.items():
            key=domain+'/'+split
            lengths=[len(r['model_text']) for r in data]
            scripts=Counter()
            for r in data:
                t=r['model_text'];cyr=any('\u0400'<=c<='\u04ff' for c in t);lat=bool(re.search('[A-Za-z]',t));cjk=any('\u4e00'<=c<='\u9fff' for c in t)
                scripts.update(['cyrillic_and_latin' if cyr and lat else 'cyrillic' if cyr else 'latin' if lat else 'other_or_no_letters'])
                if cjk:scripts['CJK_present']+=1
            data_stats[key]={'n':len(data),'classes':dict(Counter(str(r['spam_label']) for r in data)),
                'original_label_type_counts':dict(Counter(type(r['original_label']).__name__ for r in data)),
                'original_label_counts':dict(Counter(str(r['original_label']) for r in data)),
                'label_status_counts':dict(Counter(r['label_status']['spam'] for r in data)),
                'length_chars_quantiles':dict(zip(['min','p10','p50','p90','max'],np.quantile(lengths,[0,.1,.5,.9,1]).tolist())),
                'empty_texts':sum(not r['model_text'].strip() for r in data),'script_record_counts':dict(scripts),
                'groups':len({r['duplicate_group_id'] for r in data})}
            for source in sorted({r['source'] for r in data}):
                subset=[r for r in data if r['source']==source];field_stats[key+'/'+source]={}
                for field in ['subject','body_text','sender','sender_address','sender_domain','links','attachments','unsubscribe','formatting']:
                    field_stats[key+'/'+source][field]=dict(Counter(field_state(r,field) for r in subset))
                # Static text URLs do not reconstruct MIME href/metadata; no requests.
                field_stats[key+'/'+source]['text_url_presence_derived']={'nonempty':sum(bool(re.search(r'https?://|www\.',r['model_text'],re.I)) for r in subset),'n':len(subset),'not_original_html_links':True}
    hist=sets['spam_email']['validation']
    for r,p in zip(hist,e0_predictions):
        pred=int(p['spam_score']>=threshold);truth=r['spam_label']
        if pred==truth:continue
        reviews.append({'kind':'FP' if pred else 'FN','record_id':r['record_id'],'source':r['source'],
            'original_label':r['original_label'],'spam_label':truth,'spam_score':p['spam_score'],'threshold':threshold,
            'subject_context':private_fragment(r['subject'],240),'body_context':private_fragment(r['body_text']),
            'context_truncated':len(r['body_text'] or '')>1200,'status':'pending_human_review',
            'observation':'Model disagrees with source label; recipient consent and true class are not established.',
            'human_decision':None,'human_reason':None,'human_consent_evidence':None,'reviewer':None})
    ru=sets['russian_auxiliary']['train']
    for label in [0,1]:
        pool=sorted((r for r in ru if r['spam_label']==label),key=lambda r:(len(r['model_text']),r['record_id']))
        for length_bin,indices in enumerate(np.array_split(np.arange(len(pool)),3)):
            candidates=[pool[int(i)] for i in indices]
            selected=sorted(candidates,key=lambda r:digest_json([seed,'ru_review',r['record_id']]))[:10]
            for r in selected:
                t=r['model_text'];reviews.append({'kind':'RU_train_review','record_id':r['record_id'],'source':r['source'],
                    'original_label':r['original_label'],'spam_label':label,'length_chars':len(t),'length_rank_tertile':length_bin,
                    'context':private_fragment(t,900),'context_truncated':len(t)>900,'status':'pending_human_review',
                    'observed_channel':'not reliably identifiable from individual text; train sample only',
                    'human_decision':None,'human_reason':None,'human_channel':None,'reviewer':None})
    dump_rows(path/'private_review.jsonl',reviews)
    write_json(path/'data_review.json',{'counts_full_allowed_splits':data_stats,'fields_full_allowed_splits':field_stats,
        'leakage_full_allowed_splits':contract['allowed_text_leakage'],'ru_status':contract['ru_status'],
        'ru_mapping_errors':contract['ru_mapping_errors'],'review_sample_counts':dict(Counter(r['kind'] for r in reviews))})
    lines=['# День 4: разрешённые данные и review','',f"RU status: **{contract['ru_status']}**. Самопроверка Codex не является ручной верификацией.",'',
        '| Набор | N | ham/spam | Длины символов min/p50/p90/max | Пустые | Группы |','|---|---:|---|---|---:|---:|']
    for key,s in data_stats.items():
        q=s['length_chars_quantiles'];lines.append(f"| {key} | {s['n']} | {s['classes']} | {[q[k] for k in ['min','p50','p90','max']]} | {s['empty_texts']} | {s['groups']} |")
    lines.extend(['','Все счётчики таблицы полные по четырём разрешённым splits; test/parquet не читались.',
        f"Группы между train/validation не пересекаются. Совпадения model_text: exact={len(contract['allowed_text_leakage']['exact'])}, canonical={len(contract['allowed_text_leakage']['canonical'])}; проверены перекрёстно оба корпуса.",'',
        'README называет RuSpam, русский text-classification, CC-BY-NC-4.0; это ограничивает коммерческое использование. join_datasets.py прочитан как текст, не запущен. В ветке __label__ham/spam код явно назначает 0/1, затем bool(0/1); в ветке чужого Parquet применяется astype(bool), поэтому прежние строковые кодировки неизвестны. В разрешённых splits проверено нативное original_label bool и точное False→0/True→1. Явного противоречия mapping нет; это поддержка намерения автора, не доказательство каждой метки.',
        'Исходные объединённые train_data.txt/labeled_dataset_from_hf.txt и типы их предшественников не проверялись. Полный Parquet закрыт на этот день. В splits только text/label, канал и отдельная provenance строки недоступны. Искусственный баланс, группировка и отбор дня2 меняют исходное распределение.',
        '',f"Приватный private_review.jsonl содержит все {sum(r['kind']=='FP' for r in reviews)} FP и {sum(r['kind']=='FN' for r in reviews)} FN E0, а также {sum(r['kind']=='RU_train_review' for r in reviews)} RU train примеров: SHA(seed42,ID), по10 из трёх тертилей длины каждого класса. Есть отдельные пустые поля решения человека; метки не менялись.",
        'Наблюдения по RU-выборке следует читать как наблюдения по отдельным текстам. Короткие реплики, объявления и разговорная форма совместимы с чатами, но не подтверждают почтовый канал всего корпуса. Это auxiliary corpus, не verified Russian email.',
        '', 'Письменности (полный подсчёт, не распознавание языков):', *[f'- {key}: {s["script_record_counts"]}.' for key,s in data_stats.items()]])
    write_text(path/'data_review.md','\n'.join(lines)+'\n')
    lines=['# Доступность полей — только разрешённые splits','',
        'unavailable_null = формат не предоставляет поле; observable_missing = поле допустимо форматом, но значение отсутствует; observable_empty = извлечение не нашло элементов/пустое поле. unknown_null остаётся неизвестным. observed_nonempty не подтверждает полноту/подлинность. Нули вместо неизвестных метаданных не добавлялись. Все подсчёты ниже полные.','']
    for key,fields in field_stats.items():
        lines.extend([f'## {key}','', '| Поле | Состояния/число |','|---|---|',*[f'| {field} | {value} |' for field,value in fields.items()],''])
    lines.append('Отправители/домены/имена вложений не использованы как признаки. links — метаданные статического парсера; в text-only они null. Производное наличие URL по строке — отдельное наблюдение, не реконструкция исходного HTML. unsubscribe/formatting доступны только при исходной структуре; [] и null различаются.')
    write_text(path/'feature_availability.md','\n'.join(lines)+'\n')
    return reviews,data_stats


def metric_table(result):
    lines=['| Модель | Область | Порог | P | R | F1 | FP/FN | FPR | Отказы empty/N |','|---|---|---:|---:|---:|---:|---|---:|---|']
    for name,model in result['experiments'].items():
        for domain,s in model.get('validation',{}).items():
            for threshold,m in s['raw_metrics'].items():
                fmt=lambda v:'undefined' if v is None else f'{v:.6f}'
                lines.append(f"| {name} | {domain} | {float(threshold):.6f} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {m['FP']}/{m['FN']} | {fmt(m['fpr'])} | {s['empty_vectors']}/{s['n']} |")
    return lines


def render_final(path,result):
    write_json(path/'experiments.json',result)
    lines=['# День 4: E0/E1/E2','',f"Run `{result['run_id']}`. E0 остаётся моделью по умолчанию; baseline дня3 неизменён.",'',*metric_table(result),'',
        'Метрики raw посчитаны на всех записях каждой области отдельно. Отказ обёртки не исключает запись из raw confusion; coverage и распределения покрытия словаря находятся в JSON. FPR=FP/(FP+TN), positive spam=1; undefined→null. RU source labels provisional; область short text/chat отличается от email. Нет объединённого универсального F1.',
        '',f"E1 CV: {result['experiments'].get('E1',{}).get('cv_summary','no_op/blocked')}",
        f"E1 validation-only tuned threshold: {result['experiments'].get('E1',{}).get('tuned_threshold_summary','not applicable')}",
        'E1 при старом пороге — сравнение с одинаковым числом порога, не утверждение одинаковой калибровки. Порог tuned-on-validation оптимистичен и не заменяет E0 автоматически.',
        'E2 меняет одновременно язык, канал, баланс классов, размер train и словарь; эффект нельзя приписать только языку. E2 — исследовательский candidate, окончательного operating point нет. Параметры char_lr дня3 неизменны, vocabulary/IDF fit только на разрешённом train.',
        '', f"Reload: {result.get('reload_checks')}. Время классических этапов {result.get('runtime_seconds',0):.2f} с.",
        f"Закрытые данные не читались; data access log: {result.get('data_access_log')}.",
        f"Сохранность E0/основного окружения/входов: {result.get('preservation_checks')}.",
        '', 'Числа train-исключений из-за совпадений после очистки, ID и точные правила — cleaning_audit.md/.json. Нейтральные результаты/ухудшения сохранены. Анализ ошибок не приводил к дополнительному подбору regex/C/порога.']
    write_text(path/'experiments.md','\n'.join(lines)+'\n')
    questions=['# Вопросы, требующие решения человека','',
        '1. Как трактовать 8 FP и5 FN E0 в private_review.jsonl: верна ли исходная метка, было ли согласие на рассылку? Заполнить human_decision/reason/consent_evidence; без этого метки остаются исходными.',
        '2. Допускаете ли RU с source-intended метками и CC-BY-NC-4.0 для дальнейших некоммерческих исследований? Для основной русской почты нужны подтверждённые письма/канал и проверенная разметка; текущий набор не заменяет их.',
        '3. Какой допустим баланс ложных срабатываний/пропусков в будущем применении? Текущий E0-порог принят только для демо без перемещения/удаления; общего FPR<=1% недостаточно для гарантии по источникам.',
        '4. Оставить ли узкую очистку служебных subject-маркеров в следующем кандидате после просмотра cleaning_audit и метрик? E0 по умолчанию сохранён.',
        '5. Какую одну цель утвердить для дня5: улучшение почтовой модели с новыми проверенными данными или LoRA/decision-head feasibility? Перед финальным test потребуется отдельная фиксация модели/правил/порога; сегодня test закрыт.']
    write_text(path/'review_questions.md','\n'.join(questions)+'\n')
    write_text(path/'PRESENTATION_NOTES_DAY4.md','\n'.join(['# Фактически выполнено в день4','',*metric_table(result),'',
        'E0 не переобучался; E1 — узкая очистка темы и одинаковые5 train-фолдов; E2 — разрешённое смешение train с provisional RU. Raw метрики и отказы разделены. Test/challenges/резерв не оценены.',
        'Laya — готовая локальная модель, не собственное обучение Jev; фактический статус/результаты находятся в decision_pilot.md. Успешный synthetic smoke не доказывает защиту от prompt injection или качество категорий.',
        'Точные входы/config/code/версии, сохранённые модели, reload и команды — эксперименты и DAY4_STATUS.'])+'\n')
