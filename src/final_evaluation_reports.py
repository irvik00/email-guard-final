"""Aggregate-only reports for the sealed final experiment."""
from final_eval_common import *

def fmt(x):return 'N/A' if x is None else f'{x:.6f}'

def binary_table(results):
    lines=['| Вариант | Решений / n | TP | FP | TN | FN | Precision | Recall | F1 | FPR |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in BINARY:
        r=results[name];c=r['confusion']
        lines.append(f"| {name} | {r['automatic']}/{r['records']} | {c['TP']} | {c['FP']} | {c['TN']} | {c['FN']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['F1'])} | {fmt(r['FPR'])} |")
    return '\n'.join(lines)

def review_text(r):
    c=r['automatic_confusion']
    return f"E0_Laya_review: автоматически {r['automatic']}/{r['records']} ({fmt(r['automatic_fraction'])}); needs_review={r['needs_review']}, истинных ham={r['review_by_true_class'].get('0',0)}, spam={r['review_by_true_class'].get('1',0)}. Причины={r['review_reasons']}. Среди автоматических: TP/FP/TN/FN={c['TP']}/{c['FP']}/{c['TN']}/{c['FN']}, ошибок={r['automatic_errors']}. Это селективный режим; F1 на всём наборе не вычисляется, результат ручной проверки неизвестен."

def changes_table(changes):
    lines=['| Вариант против E0 | Исправлено FN | Добавлено FN | Убрано FP | Добавлено FP |','|---|---:|---:|---:|---:|']
    for name,d in changes.items():
        c=d['counts'];lines.append(f"| {name} | {c.get('rescued_E0_FN',0)} | {c.get('added_FN',0)} | {c.get('removed_E0_FP',0)} | {c.get('added_FP',0)} |")
    return '\n'.join(lines)

def validation_markdown(data):
    lines=['# Ансамбли: validation и личная development-диагностика', '',
        'Этап выполнен до доступа к отложенным test и архиву преподавателя. Использованы совместимые сохранённые predictions, не новые fits. C — именно calibration999; неизменные encoder/LR logits старого C преобразованы его исправленными сохранёнными T/threshold. Laya — adapted day5, точный короткий вопрос B.',
        'Шкалы scores не усредняются. Бинарные голоса только после индивидуальных preprocessing/T/threshold. OR/AND требуют E0 и Laya; majority требует все3. Любой отказ нужного компонента даёт component_unavailable; review отправляет такой случай на проверку отдельно от разногласия. Empty sparse vector — insufficient_features, не скрытый ham.',
        'Разные корпуса не объединяются в одну метрику. Исторический и RU validation ранее использовались в разработке; RU labels provisional, почтовый характер не доказан. Личные письма — просмотренные development data.']
    for name,s in data.items():
        lines+=['',f'## {name}', '',binary_table(s['metrics']), '',review_text(s['metrics']['E0_Laya_review']),
            '',changes_table(s['changes_vs_E0'])]
    lines+=['', 'Проверка личных ожидаемых значений', '',
        'OR=8/7/6/1; AND=4/2/11/5; majority=7/6/7/2 (TP/FP/TN/FN). Review:13 автоматических,9 на проверку; среди автоматических4/2/6/1. Все ожидания проверены assert по реальным сохранённым predictions; ответы по ID не подгонялись.',
        'OR может спасать пропуски ценой FP, AND убирать FP ценой пропусков; таблицы показывают фактический обмен ошибок. Review не устраняет ошибки согласия и не гарантирует результат человеческой проверки. При неполном покрытии бинарные метрики условны на успешные ответы; причины/ID в JSON.',
        'Численная FP/FN cost function ранее не задана. Основной финальный baseline — E0 по инструкции пользователя; варианты ансамблей, Laya и C остаются заранее заявленными дополнительными сравнениями независимо от результата validation.']
    return '\n'.join(lines)+'\n'

def protocol_markdown(p):
    lines=['# FINAL_EVALUATION_PROTOCOL', '',f"Run `{p['run_id']}`; зафиксирован UTC `{p['frozen_utc']}` до inference на тестах и первого чтения teacher archive. Первоначальный протокол был сохранён до первого открытия любого test.",
        '', '**Основной вариант: E0.** Численная стоимость FP/FN не задана, новая функция стоимости не придумывается. Дополнительные бинарные сравнения: Laya_day5, C_corrected, E0_OR_Laya, E0_AND_Laya, Majority_E0_Laya_C. E0_Laya_review — отдельный режим помощника, не бинарный классификатор на всех письмах.',
        '', '| Компонент | Builder | T | Threshold |','|---|---|---:|---:|']
    for c in p['components']:lines.append(f"| {c['name']} | {c['builder']} | {c['temperature']:.16g} | {c['threshold']:.16g} |")
    lines+=['', 'Точный вопрос Laya:', '', '```json',__import__('json').dumps(next(c for c in p['components'] if c['name']=='Laya_day5')['questions'],ensure_ascii=False,indent=2),'```',
        '', 'E0 использует исходный subject+body builder дня3. Laya: clean_builder_v2, cap700/context1024/head256, сохранённые веса adapted300. C: исправленный calibration999 manifest, неизменный E5 query-prefix/max512/masked mean+L2 и LR. Хеши всех model/config/tokenizer/manifest файлов, версии и полный preprocessing-контракт в FINAL_EVALUATION_PROTOCOL.json. Ни одна модель не переобучается.',
        '', '## Композиции и отказы', '',
        'OR: хотя бы один spam. AND: оба spam. Majority: минимум2 из3. Review: согласие→общая метка, разногласие→needs_review. Scores не складываются и не усредняются.',
        p['ensemble_errors'],p['component_errors'],
        'E0 при пустом тексте/нулевом sparse vector: insufficient_features, score может сохраняться как диагностический, decision=null. Это существующая политика отказа обёртки, применяется одинаково на validation и test.',
        '', '## Данные и порядок доступа', '',
        '1. Сначала совместимые сохранённые historical/RU validation и личные predictions; проверка ожидаемых ансамблей.',
        '2. Этот markdown и машинный протокол + SHA lock. До этого test_task_emails.tar.xz проверен только stat (имя/размер/mtime), без хеширования/просмотра содержимого.',
        '3. После seal: historical spam_email/test, teacher archive, дополнительный russian_auxiliary/test. Остальные challenge/reserve/modern/Nazario и заблокированный Defender источник остаются закрыты.',
        'Teacher ожидается1600 (ham600/spam1000), метки из ham/spam директории, фактические counts и SHA архива проверяются после seal и сохраняются в test_data_manifest.json. Имя/путь/папка/эталон не передаются в текст. Legacy adapter label всегда константа0; forward получает только tokens/masks/qtype, не labels.',
        'Архив не извлекается на диск: только потоковое чтение обычных .eml после проверки ВСЕХ членов. Абсолютные/drive/backslash/.. пути, дубли путей, symlinks/hardlinks/devices/special entries запрещены. Максимум20MiB/member и1GiB total. HTML статический; ссылки/изображения/вложения не открываются и не исполняются.',
        '', '## Пересечения и отчётность', '',
        'Сравнить rawSHA (где есть), точный и NFC/casefold/whitespace-канонический полный original/clean model_text, глобальные группы со всеми historical/RU train и validation. Отдельно отметить actual fit/dev/calibration membership текущих компонентов и более широкий pool.',
        'Основной teacher результат — полный набор1600. Вторичный результат — письма без перечисленных пересечений с train/validation; не подменять им основной. Показать дубли/конфликты внутри tests и cross-test matches без удаления или смены меток. Near/semantic/truncated-prefix независимость не гарантируется.',
        'Каждый тест показывается отдельно. RU auxiliary — отдельная дополнительная оценка source_intended_unverified, не подтверждённый почтовый тест. Бинарные метрики: TP/FP/TN/FN, precision, recall, F1, FPR, coverage; нулевые знаменатели N/A. Review: автоматическая доля, истинные классы/причины проверки, ошибки среди автоматических; без full-set F1.',
        '', '## Технический повтор', '',
        f"Первоначальный запуск `{p['technical_retry']['initial_run']}`, protocol SHA `{p['technical_retry']['initial_protocol_sha256']}`, остановился на первой исторической test-записи: ошибочная проверка protected_test=True. В схеме этот флаг относится к специальной защите, обычный test имеет role=test при protected_test=False. Исправлена только проверка role/назначения/отсутствия reserve. Исторические model predictions ещё не выполнялись, teacher archive не открывался; исходный запуск сохранён. Веса, вопросы, preprocessing, T, thresholds, голосование и основной E0 не менялись.",
        '', '## Запрет изменений после открытия', '',
        'Не выбирать пороги, вопросы, веса или исключения по тестам. Не менять рабочий default. Факт технического дефекта требует сохранённого исходного запуска и явного повторного attempt, без маскировки. Победитель, выбранный по teacher, не получает независимого подтверждения от того же teacher. Spam не является подтверждённым phishing. Упаковка/публикация не выполняются.']
    return '\n'.join(lines)+'\n'

def write_final_reports(out,protocol,validation,results,metadata,overlaps):
    lines=['# Финальная оценка фиксированных моделей и ансамблей', '',
        f"Run `{out.name}`. Основной E0 зафиксирован до открытия tests. Protocol SHA: `{sha_file(out/'FINAL_EVALUATION_PROTOCOL.json')}`. Дополнительные сравнения заданы заранее; настройки после тестов не менялись.",
        'Результаты разных наборов не объединяются. Bинарные metrics при отказах относятся только к успешным решениям с явным coverage. Review — отдельная селективная система без full-set F1. Рабочий default сохранён.',
        f"Технический повтор после остановленного `{protocol['technical_retry']['initial_run']}`: исправлена неверная интерпретация protected_test у обычного historical test. Первый запуск не дошёл до test inference и открытия teacher. Первоначальный protocol/код/validation сохранены; настройки моделей/ансамблей не менялись."]
    for dataset in TEST_SETS:
        s=results[dataset];meta=metadata[dataset];o=overlaps[dataset]
        lines+=['',f'## {dataset}: полный набор', '',f"n={meta['rows']}, classes={meta['classes']}.",
            binary_table(s['full']['metrics']), '',review_text(s['full']['metrics']['E0_Laya_review']),
            '',changes_table(s['full']['changes_vs_E0']), '',
            f"Пересечений с train/validation: {o['overlapping_records']} записей; с конфликтом исходных меток: {o['label_conflict_records']}. Canonical duplicate groups внутри набора: {len(o['canonical_duplicate_groups'])}. Cross-test exact/canonical/raw matches: {len(o['cross_test_matches'])} пар ключей (могут повторять одну пару записей)."]
        if o['overlapping_records']:
            lines+=['',f"### {dataset}: дополнительная непересекающаяся часть", '',
                f"n={s['nonoverlap']['records']}; это дополнительный разрез, основной выше не заменён.",binary_table(s['nonoverlap']['metrics']),
                '',review_text(s['nonoverlap']['metrics']['E0_Laya_review'])]
        else:lines+=['', 'Непересекающаяся часть совпадает с полным набором по проверенным ключам.']
        if dataset=='teacher_test':
            lines+=['',f"Teacher archive SHA256=`{meta['archive_sha256']}`; архив неизменен. Проверено членов={meta['archive_safety']['safe_members_checked']}; filesystem extraction не выполнялась; parse_errors={meta['parse_errors']}, предупреждения parser у {meta['warnings_records']} писем.",
                'Эталон — директория ham/spam; labels/paths изолированы от model text. Локаторы и полный overlap audit находятся только в приватных JSONL/JSON.']
        if dataset=='russian_auxiliary_test':lines+=['', 'Ограничение: RU auxiliary source_intended_unverified, искусственная квота классов, может включать сообщения/чаты. Не подтверждает качество на русской электронной почте всех пользователей.']
    lines+=['', '## Ограничения независимости', '',
        'Train/validation pool шире actual selected fit6000 Laya/C; actual роли сохранены у match records. Пересечения не удалялись из полного результата; конфликтующие source labels не исправлялись догадками. Полный exact/canonical audit не доказывает отсутствия общих шаблонов, семантических/near/visible-prefix копий.',
        'Teacher folder labels и исторические source-level labels принимаются как предоставленный эталон, не ручная верификация phishing. Validation уже использовался для разработки и порогов. Личные письма остаются development и не входят в тестовые totals.',
        'Выбор любого нового победителя по результатам teacher требует новой независимой проверки. Основной E0 был выбран до teacher по правилу протокола, не по максимальной итоговой F1. Ни одного общего показателя accuracy по разным тестам не вычислялось.',
        '', '## Воспроизведение', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -p test_final_evaluation.py',
        '.\\.venv\\Scripts\\python.exe -B src/run_final_evaluation.py run',
        f'.\\.venv\\Scripts\\python.exe -B src/run_final_evaluation.py audit --out reports/final_evaluation/runs/{out.name}',
        '```', '', 'run создаёт новый архив и новый заранее сохранённый протокол; повторное открытие уже оценённых tests не будет первой независимой оценкой. Для аудита сохранённого запуска inference не требуется.']
    (out/'FINAL_TEST_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    presentation=['# Финальная оценка: материал для презентации', '',
        'Прозрачные ансамбли фиксированных E0/Laya_day5/C_corrected: OR, AND, majority и отдельный review. Scores не смешиваются. Сначала проверены validation и личная development-диагностика, затем сохранён протокол и только потом открыты tests.',
        'Основной E0 задан до тестов: численная FP/FN стоимость ранее не определена. Остальные варианты — дополнительные сравнения. Не было обучения, новых порогов, новых вопросов или весов голосования.', '']
    for dataset in TEST_SETS:
        presentation+=[f'## {dataset}', '',binary_table(results[dataset]['full']['metrics']), '',review_text(results[dataset]['full']['metrics']['E0_Laya_review']),
            '',f"Полных train/validation пересечений: {overlaps[dataset]['overlapping_records']}; вторичный nonoverlap n={results[dataset]['nonoverlap']['records']}.", '']
    presentation+=['Tests показаны отдельно, единой общей accuracy нет. RU auxiliary имеет provisional source labels и не доказанный email-формат. Teacher labels получены из ham/spam, путь не является признаком. Пересечения/дубли и ограничения независимости открыто показаны.',
        'Review снижает долю автоматических решений, но оставляет ошибки согласия и требует человеческого решения спорных случаев. Это не F1 классификатора на всём наборе. Spam не подтверждает phishing.',
        'Исходные модели/отчёты и default E0 сохранены; личные письма не публикуются. Упаковка и публикация пока не выполнялись. Этот файл содержит только агрегаты, без писем, адресов, кодов и персональных ссылок.']
    (out/'PRESENTATION_NOTES_FINAL_EVALUATION.md').write_text('\n'.join(presentation)+'\n',encoding='utf-8')
    recommendation=['# Рекомендация по упаковке', '',
        'Технический кандидат для первой локальной упаковки — сохранённый E0 с его исходным builder, фиксированным operating threshold и явными отказами. Это основной вариант заранее сохранённого протокола; default автоматически не изменён. Режим E0_Laya_review можно предлагать отдельно как помощника с обязательным ручным решением разногласий/ошибок компонентов.',
        '', '## Что реально измерено', '']
    for d in TEST_SETS:
        r=results[d]['full']['metrics']['E0'];c=r['confusion']
        recommendation.append(f"- E0, {d}: TP/FP/TN/FN={c['TP']}/{c['FP']}/{c['TN']}/{c['FN']}; F1={fmt(r['F1'])}; FPR={fmt(r['FPR'])}; coverage={r['automatic']}/{r['records']}.")
    recommendation+=['', 'Полные таблицы всех дополнительных сравнений и изменения FP/FN — в FINAL_TEST_RESULTS.md; validation отдельно в ENSEMBLE_VALIDATION.md. Ансамбль с лучшей наблюдаемой F1 на teacher не становится автоматически выбранным и независимо подтверждённым финальным продуктом.',
        '', '## Условия и ограничения', '',
        'Готовность измерена для локального inference сохранённых артефактов и корректного голосования; это не проверка контейнера/API, многопользовательской нагрузки, production latency, доступности GPU или безопасности сервиса. Публикация и упаковка в задаче не выполнялись.',
        'Для упаковки зафиксировать версию preprocessing/артефактов и контракт отказов, конфиденциальность загрузок/логов, лимиты MIME/архивов, отсутствие внешних сетевых обращений. Laya/C требуют соответствующего сохранённого runtime и больших локальных весов; нейросети в эксперименте запускались последовательно.',
        'Качество ограничено конкретными наборами и source-level эталонами. Нет подтверждённой phishing-разметки, универсальной поддержки языков, достоверной истории согласия или репрезентативной почты всех пользователей. Возможные overlap/duplicate/semantic dependencies ограничивают независимость.',
        'Если после этих тестов будет выбрана иная модель/композиция, её дальнейшую независимую проверку нужно проводить на новых данных. Настройка стоимости FP/FN — будущее продуктовое решение; её не восстанавливали из удобных тестовых результатов.']
    (out/'PACKAGING_RECOMMENDATION.md').write_text('\n'.join(recommendation)+'\n',encoding='utf-8')
