"""Append aggregate interpretation/audits; no inference, data or protocol changes."""
import argparse
import shutil
from final_eval_common import *

def append_block(path,text):
    marker='<!-- final-interpretation -->';original=path.read_text(encoding='utf-8')
    if marker in original:original=original[:original.index(marker)].rstrip()+'\n'
    path.write_text(original+'\n'+marker+'\n'+text+'\n',encoding='utf-8')

def audit_and_interpret(out):
    guard=FinalBoundary(out).install();protocol=read_json(out/'FINAL_EVALUATION_PROTOCOL.json');lock=read_json(out/'protocol_lock.json')
    require(sha_file(out/'FINAL_EVALUATION_PROTOCOL.json')==lock['sha256'],'Protocol mutated')
    rows=read_rows(out/'final_test_predictions.jsonl');results=read_json(out/'final_results.json');overlaps=read_json(out/'overlap_audit.json')
    validation=read_json(out/'validation_results.json');metadata=read_json(out/'test_data_manifest.json')['datasets']
    bykey={(r['dataset'],r['record_id'],r['model_id']):r for r in rows};truths={d:read_rows(out/f'private_truth_{d}.jsonl') for d in TEST_SETS}
    # Independently compute arithmetic without evaluate()/confusion().
    checked=0;refusals={}
    for dataset in TEST_SETS:
        for subset,ids in [('full',{r['record_id'] for r in truths[dataset]}),('nonoverlap',set(overlaps[dataset]['nonoverlap_ids']))]:
            for name in MODEL_NAMES:
                chosen=[r for r in rows if r['dataset']==dataset and r['model_id']==name and r['record_id'] in ids]
                available=[r for r in chosen if r['status']=='ok' and type(r['decision']) is int]
                tp=sum(r['spam_label']==1 and r['decision']==1 for r in available)
                fp=sum(r['spam_label']==0 and r['decision']==1 for r in available)
                tn=sum(r['spam_label']==0 and r['decision']==0 for r in available)
                fn=sum(r['spam_label']==1 and r['decision']==0 for r in available)
                s=results[dataset][subset]['metrics'][name];matrix=s.get('confusion',s.get('automatic_confusion'))
                require(matrix=={'TP':tp,'FP':fp,'TN':tn,'FN':fn} and s['automatic']==len(available),'Independent counts differ')
                if name!='E0_Laya_review':
                    for key,value in {'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
                        'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'FPR':fp/(fp+tn) if fp+tn else None}.items():
                        require(s[key]==value,'Independent ratio differs')
                checked+=1
        refusals[dataset]={}
        for name in COMPONENTS:
            missing=[r for r in rows if r['dataset']==dataset and r['model_id']==name and not component_ok(r)]
            refusals[dataset][name]={'count':len(missing),'true_classes':dict(Counter(str(r['spam_label']) for r in missing)),
                'statuses':dict(Counter(r['status'] for r in missing))}
    # Actual identical feature strings should produce the same results despite different
    # archive/file IDs. C batching permits normal floating-point differences.
    pairs=defaultdict(list)
    for dataset,truth in truths.items():
        for r in truth:
            for kind,h in r['keys']:
                if kind in {'original_exact','clean_exact'}:pairs[(kind,h)].append((dataset,r['record_id']))
    consistency={n:{'pairs':0,'max_abs_score_difference':0.,'different_decisions':0} for n in COMPONENTS}
    for (kind,h),group in pairs.items():
        if len(group)<2:continue
        names=['E0'] if kind=='original_exact' else ['Laya_day5','C_corrected']
        reference=group[0]
        for other in group[1:]:
            for name in names:
                a=bykey[(reference[0],reference[1],name)];b=bykey[(other[0],other[1],name)]
                if not(component_ok(a) and component_ok(b)):continue
                d=abs(a['score']-b['score']);c=consistency[name];c['pairs']+=1;c['max_abs_score_difference']=max(c['max_abs_score_difference'],d)
                c['different_decisions']+=int(a['decision']!=b['decision'])
                require(d<=1e-4 if name=='C_corrected' else d<=1e-12,'Identical-input scores differ beyond declared tolerance')
    teacher=overlaps['teacher_test'];cross_ids={x['record_id'] for x in teacher['cross_test_matches']}
    non=set(teacher['nonoverlap_ids']);cross_non=len(cross_ids&non)
    unique_non={h for r in truths['teacher_test'] if r['record_id'] in non for kind,h in r['keys'] if kind=='clean_canonical'}
    all_teacher_empty=[r for r in truths['teacher_test'] if r['text_characters']==0]
    # Save initial automatically rendered reports; only these new aggregate reports change.
    names=['FINAL_TEST_RESULTS.md','ENSEMBLE_VALIDATION.md','PRESENTATION_NOTES_FINAL_EVALUATION.md','PACKAGING_RECOMMENDATION.md']
    original=out/'generated_initial_reports';original.mkdir(exist_ok=True)
    for name in names:
        if not(original/name).exists():shutil.copyfile(out/name,original/name)
    v=['## Интерпретация обмена ошибок и покрытия', '',
        'Исторический validation: OR спасает4 FN E0 без добавления FP (TP310/FP8/TN956/FN1); AND убирает5 FP, добавляя12 FN (294/3/961/17). Majority не улучшает историческую F1 E0. Эти результаты сохранены независимо от последующих tests.',
        'RU validation: E0 отвечает3418/5990, так что OR/AND/majority имеют то же ограничение покрытия; Laya/C отвечают5990/5990. Их F1 вычислены на разных множествах успешных решений и не сравниваются как одинаковая full-dataset задача. Paired изменения ошибок в JSON рассчитаны только на записях, где оба варианта дали решение.',
        'Личные development-письма: OR и AND ровно совпадают с Laya и E0 по бинарным решениям соответственно; majority даёт иной компромисс7TP/6FP/7TN/2FN. Это не независимый финальный тест, метки не исправлялись по моделям.']
    append_block(out/'ENSEMBLE_VALIDATION.md','\n'.join(v))
    extra=['## Ключевые ограничения teacher test и отказов', '',
        f"Из1600 teacher-писем 988 ({988/1600:.2%}) имеют полный exact/canonical/raw match с development:828 с historical train,160 с historical validation. Actual selected Laya/C fit затронут у380 teacher-записей. Поэтому новый архив не означает новый независимый корпус;160 совпадений с validation ранее уже оценивались моделями как validation.",
        f"Дополнительный nonoverlap включает612 писем (ham134/spam478), из них {cross_non} совпадают с отдельно оцениваемым historical test. Всего уникальных teacher ID с cross-test match={len(cross_ids)}. Эти оценки нельзя объединять как независимые выборки.",
        f"В полном teacher52 canonical duplicate groups (48 пар и4 тройки), конфликтов меток в них0. В nonoverlap — {len(unique_non)} уникальных непустых canonical content и {len(all_teacher_empty)} отдельных записей без текста. Число независимых примеров не установлено и меньше612 по одним только точным повторениям; ни одна запись не удалялась из основного результата.",
        'На полном teacher E0 FPR=1.5%, на612 без train/validation совпадений —9/134=6.7164%; у OR соответственно2.8333% и17/134=12.6866%. Более высокая F1 OR не означает меньшего риска ложных срабатываний. Непересекающийся разрез имеет существенно иную долю spam и не заменяет основную таблицу.',
        '', '| Набор | Компонент | Без решения | Истинные классы без решения | Статусы |','|---|---|---:|---|---|']
    for dataset,models in refusals.items():
        for name,r in models.items():extra.append(f"| {dataset} | {name} | {r['count']} | {r['true_classes']} | {r['statuses']} |")
    extra+=['', 'Teacher: два письма с нулевым model_text имеют folder label spam. Все компоненты возвращают insufficient_features; это2 отказа, не2 ham и не скрытые FN. Бинарные показатели условны на1598/1600, nonoverlap —610/612. Review отправляет эти2 случая на ручную проверку отдельно от128 разногласий.',
        'Historical: E0/зависимые бинарные ансамбли имеют1 отказ на истинном spam; Laya/C дают ответы на1276/1276. RU auxiliary:2580 отказов E0 из-за отсутствующих sparse features, поэтому строгие композиции отвечают3410/5990. Полная F1 C=0.936297 при5990/5990 не является напрямую сопоставимой с majority F1=0.942489 на3410/5990.',
        '', '## Повторный технический запуск и независимые проверки', '',
        'Первый запуск остановился до model test inference на ошибочной проверке protected_test=True. Исходный protocol/код/validation и technical_failure.json сохранены. Исправлена только интерпретация schema role=test; все настройки компонентов/ансамблей неизменны. Teacher впервые прочитан после seal исправленного протокола. Никакого accuracy-driven повторного обучения/подбора/переоценки test не было.',
        f"Отдельно пересчитаны {checked} наборов confusion/ratios собственным арифметическим аудитом. Проверена одинаковость scores на совпадающих exact входах с различными dataset/file IDs: {__import__('json').dumps(consistency,ensure_ascii=False)}.",
        'Проверка exact входов — эмпирическое подтверждение отсутствия влияния пути/имени на проверенных повторениях, дополнительно к fixture и изоляции GT. C допускает обычные float32 batch-различия <=1e-4; decisions сравниваются отдельно. Это не новое голосование или изменение thresholds.']
    append_block(out/'FINAL_TEST_RESULTS.md','\n'.join(extra))
    package=['## Практический итог по измеренным ограничениям', '',
        'E0 готов как воспроизводимый локальный исследовательский baseline для упаковки с заявленным историческим English-сценарием и обязательным отказом на недостаточных признаках. Это техническая готовность артефакта, не production-приёмка и не универсальный антиспам: FPR на unseen teacher ham6.72%, на RU auxiliary2580 отказов, личные письма ранее показали контекстные ошибки.',
        'OR можно сохранить как заранее заявленный дополнительный режим с более высокой полнотой и более высоким FPR; он не назначается default по teacher. На historical test F1 OR0.980952 немного ниже E00.982400, хотя на validation OR был лучше. AND на historical test имеет0 наблюдаемых FP ценой16FN и1отказа; это не гарантия нулевого FPR на будущей почте.',
        'Review на historical test даёт1249 автоматических/27 проверок, на teacher1470/130; на RU auxiliary2053/3937. Для русской вспомогательной задачи C имеет полное покрытие и более сильную F1, но source labels provisional и перенос на реальную русскую почту не подтверждён. Независимость teacher ограничена988 train/validation копиями и162 cross-test копиями.',
        'Упаковка и публикация не выполнялись. Нужны отдельные продуктовые решения о допустимых FP/FN и нагрузке ручной проверки; по текущим tests новую функцию стоимости или winner-selection не выводили.']
    append_block(out/'PACKAGING_RECOMMENDATION.md','\n'.join(package))
    presentation=['## Акценты для защиты', '',
        '- OR помог на historical validation:4 меньше FN без роста FP; на historical test добавил3FP ради2 исправленных FN и оказался немного хуже E0 по F1.',
        '- Teacher: полный1600 и вторичный612 показаны раздельно. 988 train/validation совпадений;162 совпадения с historical test;52 внутренних canonical duplicate groups. Большие метрики teacher не объявляются независимым подтверждением выбранной модели.',
        '- Teacher OR:30 меньше FN,8 больше FP относительно E0. Выбор зависит от заранее не заданной стоимости ошибок; default E0 сохранён.',
        '- Пустые teacher-письма:2 отказа на spam, не скрытые ham. RU: E0 и зависимые ансамбли имеют57% покрытия; selective F1 не сравнивается с full coverage C/Laya как одинаковая задача.',
        '-15 fixture tests, SHA/model/protocol/metric audit; исходный остановленный запуск сохранён и техническое исправление обозначено. Модели не переобучались, thresholds не подбирались; упаковка не выполнялась.']
    append_block(out/'PRESENTATION_NOTES_FINAL_EVALUATION.md','\n'.join(presentation))
    # Check shared reports for concrete email addresses/URLs present in inference text.
    texts='\n'.join(r['original_model_text'] for d in TEST_SETS for r in read_rows(out/f'private_inputs_{d}.jsonl'))
    tokens=set(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',texts))|set(re.findall(r'https?://[^\s<>\"]+',texts))
    leaks={}
    for name in names+['FINAL_EVALUATION_PROTOCOL.md']:
        report=(out/name).read_text(encoding='utf-8');hits={t for t in tokens if len(t)>8 and t in report}
        leaks[name]=len(hits);require(not hits,'Email/URL leak into aggregate report')
    # Finally rehash the untouched archive after inference, using the same sealed protocol.
    guard.seal();require(guard.lock==lock['sha256'],'Seal changed')
    archive_sha=sha_file(TEACHER);require(archive_sha==metadata['teacher_test']['archive_sha256'],'Archive changed after inference')
    write_json(out/'supplementary_audit.json',{'status':'passed','independent_metric_groups':checked,'identical_input_consistency':consistency,
        'refusals_by_true_class':refusals,'teacher_cross_test_unique_records':len(cross_ids),'nonoverlap_cross_test_records':cross_non,
        'teacher_archive_sha256_after_inference':archive_sha,'archive_unchanged':True,'source_email_or_url_hits':leaks,
        'no_model_inference_or_setting_change':True,'writer_sha256':sha_file(Path(__file__)),
        'protocol_sha256_unchanged':lock['sha256'],'report_sha256':{n:sha_file(out/n) for n in names}})
    print('Supplementary audit passed;',checked,'metric groups; exact-input consistency',consistency,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    out=a.out.resolve();require(out.is_relative_to(BASE/'runs'),'Outside final runs');audit_and_interpret(out)
