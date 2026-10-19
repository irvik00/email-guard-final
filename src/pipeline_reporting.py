"""Human-readable Russian reports and bounded, inert private review snippets."""
import json
from collections import Counter, defaultdict
from normalization_reporting import fragment
from pipeline_common import stable_order


LIMITATIONS = ("Модели не обучались; метрик precision/recall/F1 нет. Исторический spam_email не измеряет качество современной почты и независимость от источника. "
    "Ling-Spam: mapping_inferred; русский: source_intended_unverified, канал неизвестен, подвыборка искусственно 50/50 до дедупликации. "
    "Языки unknown: измерены лишь признаки письменностей. Date не аутентифицирован. Заголовки/URL и вложения не проверены на подлинность или безопасность. "
    "modern_spam_test за сентябрь 2026 имеет статус security_blocked: Microsoft Defender заблокировал и удалил архив по сообщению пользователя; скачивание/восстановление не выполняются. "
    "Сентябрь 2025 сохранён отдельно как modern_spam_2025_period_challenge и не заменяет 2026. Одноклассовые проверки 2025 и phishing_positive_test позволяют в будущем оценить пропуски/recall положительного класса, но не FP или полноценную F1. "
    "Modern train candidates ограничены январём–августом 2025 по collection timestamp UTC; октябрь–декабрь 2025 и январь 2026 заморожены в отдельном резерве, включая запрет train/validation-копий их групп. "
    "Компонента из 1000 сообщений помечена needs_review: одинаковое plain-тело с техническим шумом при различающемся HTML; письма не перекодированы и метки не изменены. "
    "Временной challenge по заявленному периоду не изолирует чистый временной дрейф. Нужны свежий разрешённый обезличенный ham, проверенные phishing-negative, тест преподавателя и тематическая разметка.")


def quality_markdown(report, normalized):
    lines=["# День 2, шаг 3: качество и дубликаты", "", f"Статус: **{report['status']}**. Все пять источников сравнивались совместно; оригиналы сохранены в data/interim. Нормализация: {normalized['reuse']['mode']}. Русский Parquet повторно просканирован: {normalized['reuse']['russian_parquet_rescanned']}.", "",
           "| Источник | Нормализовано | Пригодно до конфликтов | Quarantine | Исключено по качеству | Сохранено в records |", "|---|---:|---:|---:|---:|---:|"]
    for source,s in report["sources"].items():
        lines.append(f"| {source} | {s['normalized_records']:,} | {s['quality_eligible_before_conflicts']:,} | {s['quarantine_records']:,} | {s['quality_excluded']:,} | {s['processed_records']:,} |")
    norm_exclusions=sum(sum(s["excluded_by_reason"].values()) for s in normalized["sources"].values())
    lines += ["", f"Баланс нормализации: {sum(s['input_records'] for s in normalized['sources'].values()):,} входных элементов/строк = {report['normalized_records']:,} нормализованных + {norm_exclusions:,} исключений разбора + русские пригодные вне reservoir. Полный русский проход и счётчики взяты из проверенного результата шага 2, если cache переиспользован.", "",
              f"Баланс шага 3: {report['normalized_records']:,} = {report['processed_records']:,} сохранённых + {report['quarantine_records']:,} quarantine + {report['quality_excluded_records']:,} непригодных по качеству. Дубли не уничтожаются: records содержит все пригодные неконфликтные оригиналы; для benchmark выбран один представитель группы.", "",
              "## Совпадения", "", f"Всего групп со singleton: {report['group_count']:,}; групп с >1 оригиналом: {report['duplicate_group_count']:,}; оригиналов в них: {report['records_in_duplicate_groups']:,}; межисточниковых групп: {report['cross_source_group_count']:,}; групп Nazario 2024/2025: {report['nazario_2024_2025_group_count']:,}.", "",
              f"Подтверждённых связей exact_original: {report['exact_original_links']:,}; normalized_model_text: {report['normalized_model_text_links']:,}. Связи exact/normalized записаны звёздами к первому оригиналу; это не число всех возможных попарных комбинаций.", "",
              "Exact сравнивает реальные raw-байты либо канонические исходные табличные поля с отдельными пространствами ключей. Для сравнения model_text применяется NFC → casefold → NFC → LF → сведение Unicode-пробелов; учебный model_text не переписывается. URL, адреса, числа, пунктуация и смешанные алфавиты остаются. Пустые тексты не образуют ключ группы.", ""]
    near=report["near"]
    lines += [f"Near: MinHash {near['num_perm']} permutations, seed={near['seed']}, схема {near['scheme']}, LSH threshold={near['lsh_threshold']} ({near['lsh_bands']} bands × {near['lsh_rows_per_band']} rows). После кандидатов: точные множества символьных 5-грамм, Jaccard ≥ {near['jaccard_min']}, отношение длин ≥ {near['length_ratio_min']}, минимум {near['near_min_chars']} символов сравнения.", "",
              f"Индексировано уникальных длинных текстов {near['unique_long_texts_indexed']:,}; пар-кандидатов {near['candidate_pairs']:,}; уже связаны подтверждёнными путями и не проверялись повторно {near['connected_candidate_pairs_skipped']:,}; прошли длину {near['length_ratio_pass_pairs']:,}; точных Jaccard-проверок {near['exact_jaccard_comparisons']:,}; подтверждено новых near-связей {near['confirmed_pairs']:,}; время {near['seconds']:.2f} с. Число всех попарно близких пар не вычислялось.", "",
              "LSH вероятностный и может пропустить пары. Короткие сообщения получают только exact/normalized сравнение. Общий URL/домен, отправитель или типовая фраза не являются самостоятельным правилом объединения. Связная компонента может соединять разные крайние сообщения цепочкой; одинаковость каждой пары не заявляется. Полный состав, provenance, причины и метрики связей находятся в duplicate_groups.jsonl.", "",
              f"Конфликтных групп: {report['quarantine_groups']:,}; с конфликтом уже в exact/normalized компоненте: {report['exact_or_normalized_conflict_groups']:,}; только near_label_conflict: {report['near_only_conflict_groups']:,}. Никакого голосования или переноса меток между оригиналами. Spam=1 и phishing=1 не конфликтуют, null не становится 0.", "",
              "## Качество и доступность", ""]
    for source,s in report["sources"].items():
        lines += [f"### {source}", "", f"Метки spam: `{json.dumps(s['spam_labels'])}`; phishing: `{json.dumps(s['phishing_labels'])}`. Длины model_text: `{json.dumps(s['length_chars'])}`.", "",
                  f"Флаги (полные подсчёты, могут пересекаться): `{json.dumps(s['flags'],ensure_ascii=False)}`.", "",
                  f"Недоступные группы признаков: `{json.dumps(s['unavailable_features'])}`; null-поля: `{json.dumps(s['missing_fields'])}`.", "",
                  f"Периоды: `{json.dumps(s['collection_periods'])}`. Письменности, не языки: `{json.dumps(s['script_indicators'])}`. Язык: unknown.", ""]
    lines += ["Пустая тема, короткий текст, HTML и предупреждение сохраняются флагами; сами по себе не удаляют запись. Удаляются лишь непригодный model_text/failed и карантин конфликтов. Метки source_provided тоже являются метками источника, не независимой ручной верификацией.", "",
              "## Крупнейшие компоненты", "", "| Группа | Оригиналов | Источники | Quarantine |", "|---|---:|---|---|"]
    for g in report["largest_groups"]:
        lines.append(f"| `{g['duplicate_group_id']}` | {g['size']:,} | `{json.dumps(g['sources'])}` | {g['quarantine']} |")
    lines += ["", f"Время шага 3: {report['timing_seconds']['total']:.2f} с. Фрагменты для приёмки — data_review_sample.md. Контакты/ссылки обезврежены только в этом отчёте.", "", LIMITATIONS, ""]
    return "\n".join(lines)


def split_markdown(report, manifest):
    lines=["# День 2, шаг 4: назначения и разбиение", "", f"Статус: **{report['status']}**. Seed={report['seed']}; целевые доли train/validation/test = 70/15/15. Детерминированный отбор представителя: пригодность → ok перед partial → доступные raw headers → стабильный record_id, только внутри источников/известной целевой оси назначения.", "",
           "| Набор | Split | Записей/групп | Классы | Источники | Статус |", "|---|---|---:|---|---|---|"]
    for dataset,spec in manifest["datasets"].items():
        for split,s in spec["splits"].items():
            lines.append(f"| {dataset} | {split} | {s['records']:,} | `{json.dumps(s['class_counts'])}` | `{json.dumps(s['source_counts'])}` | {s['status']} |")
    lines += ["", "## Резервирование и глобальный приоритет", "",
              f"Защищены Nazario из phishing-2025.txt и modern_spam_2025_period_challenge за {', '.join(report['modern_period_challenge_periods'])}. Collection timestamp имени в UTC, не Date письма. Train разрешён только за {', '.join(report['modern_train_periods'])}.", "",
              f"modern_spam_test: **{report['modern_test_status']}**, ожидаемый период {', '.join(report['modern_test_periods'])}, архив `{report['modern_test_archive']}`. Причина: {report['modern_test_block_reason']}. Основание — сообщение пользователя, к карантину Defender обращения не было. Сентябрь 2025 не является заменой.", "",
              f"Резерв: {report['modern_reserve']['records']:,} оригиналов / {report['modern_reserve']['groups']:,} групп, периоды `{json.dumps(report['modern_reserve']['period_counts'])}`, путь `{report['modern_reserve']['path']}`. Сохраняются все пригодные оригиналы будущих периодов, а не только представители. Никаких train/validation-копий этих групп; резерв не используется для обучения или выбора настроек.", "",
              f"Защищённых групп {report['protected_groups']:,}; пересекающихся между двумя challenges {len(report['cross_challenge_overlap_groups']):,}. Они могут оставаться в обеих проверках с cross_challenge_overlap; оценки нельзя усреднять как независимые. Совпадений групп резерва с защищённым test: {sum(s['reserved_future'] and s['protected_test'] for s in manifest['group_split_mapping'].values()):,}; такие оригиналы остаются в reserve и test, но никогда в train/validation.", "",
              f"Подавлено назначений по глобальному test > validation > train: {report['suppressed_assignments']:,}, по наборам `{json.dumps(report['suppressed_by_dataset'])}`. Nazario-2024 кандидатов исключено из-за test-групп: {report['nazario_2024_excluded_by_test_group']:,}. Полный индекс подавленных назначений — exclusions.jsonl, не удаление оригиналов.", "",
              "Сначала тестовые группы резервируются целиком. Остальные benchmark-группы стратифицированы по источнику/целевой метке представителя, отсортированы по SHA-256(seed, purpose, group), доли округляются методом наибольших остатков. После этого назначения согласованы глобально; пропорции не исправляются нарушением приоритета. Совпавший русский test запрещает английскую train-копию и наоборот.", "",
              "## До/после глобального согласования", "", "| Набор | Split | Предложено | Осталось |", "|---|---|---:|---:|"]
    for dataset,spec in report["before_global_priority"].items():
        for split,s in spec["splits"].items():
            lines.append(f"| {dataset} | {split} | {s['records']:,} | {report['datasets'][dataset]['splits'][split]['records']:,} |")
    lines += ["", "Это доли после выбора одного представителя группы, а не после oversampling. Удаление повторов/карантина меняет исходный баланс; фактические классы и источники показаны. Для spam_email source сохранён для отдельных метрик SpamAssassin/Ling-Spam; source не входит в model_text, но текстовые артефакты источников возможны.", "",
              f"Наличие обоих классов: `{json.dumps(report['class_presence_checks'])}`. CV: {report['cv']['folds']} folds для {report['cv']['records']:,} train-групп spam_email; validation/test не участвуют. Обучение CV не запускалось.", "",
              "## Audit и ограничения", "", f"Независимый аудит: `{json.dumps(manifest.get('audit',{}),ensure_ascii=False)}`.", "",
              "Manifest хранит точные пути, SHA-256/объёмы/строки, конфигурацию, версии зависимостей/кода, group split mapping и CV mapping. Пустой/недоступный benchmark помечается явно и не публикуется как успешный пустой набор.", "", LIMITATIONS, "",
              "Pending: " + ", ".join(report["pending"]) + ". Шаг 5 остаётся ручной приёмкой.", ""]
    return "\n".join(lines)


def review_markdown(quality, split_result, seed):
    metadata, groups = quality.metadata, quality.groups
    assignments = split_result["assignments"]
    protected_records = {a["record_id"]:a["dataset"] for a in assignments if a["protected"]}
    protected_groups = {a["group_id"] for a in assignments if a["protected"]}
    reserved_groups = {gid for gid,s in split_result["group_mapping"].items() if s["reserved_future"]}
    protected_shown = Counter()
    shown = set()
    lines=["# День 2: приватная выборка для ручной приёмки", "",
           "Короткие фрагменты до 160 символов; оригиналы не выгружаются целиком. HTML/Markdown экранированы, URL и контакты обезврежены только в отчёте. Это проверка формата/происхождения, не изменение правил по защищённому test. Выборка детерминирована seed=42; метки не подтверждены человеком этим скриптом.", ""]

    def add(rid, title=None):
        gid = quality.by_record[rid]
        if gid in reserved_groups:
            return False
        if rid in shown:
            return False
        if gid in protected_groups:
            dataset = protected_records.get(rid)
            if dataset is None or protected_shown[dataset] >= 3:
                return False
            protected_shown[dataset] += 1
        shown.add(rid)
        r=quality.record(rid)
        original="\n\n".join(v for v in [r["subject"],r["body_text"]] if v)
        lines.extend([f"- {'['+title+'] ' if title else ''}Локатор: `{r['original_locator']}`; группа `{gid}`.  ",
            f"  spam={r['spam_label']}, phishing={r['phishing_label']}, original_label={json.dumps(r['original_label'],ensure_ascii=False)}; label_status=`{json.dumps(r['label_status'])}`; parse={r['parse_status']}.  ",
            f"  Оригинал: {fragment(original)}  ", f"  model_text: {fragment(r['model_text'])}  ",
            f"  Доступность: `{json.dumps(r['feature_availability'])}`.  ",
            f"  Quality: `{json.dumps(metadata[rid]['flags'])}`; преобразования: `{json.dumps(r['transformations'])}`; quarantine={groups[gid]['quarantine']}.", ""])
        return True

    candidates=defaultdict(list)
    for rid,m in metadata.items():
        if quality.by_record[rid] in protected_groups | reserved_groups:
            continue
        axis="phishing" if m["source"]=="nazario" else "spam"
        if m[axis] is not None:
            candidates[(m["source"],m[axis])].append(rid)
    for (source,label),rids in sorted(candidates.items()):
        limit=10 if source=="lingspam" else 5
        lines += [f"## {source}, исходный целевой класс {label}", ""]
        for rid in sorted(rids,key=lambda r:stable_order(seed,"source-review",r))[:limit]:
            add(rid)
    lines += ["## Защищённые проверки: только формат (до 3 на challenge)", ""]
    for a in sorted((a for a in assignments if a["protected"]),key=lambda a:stable_order(seed,"protected-format",a["record_id"])):
        if protected_shown[a["dataset"]] < 3:
            add(a["record_id"],a["dataset"])
    lines += ["## Near-пары (до 3; не защищённый test)", ""]
    pairs=0
    for gid,g in sorted(groups.items(),key=lambda kv:stable_order(seed,"near-review",kv[0])):
        if gid in protected_groups | reserved_groups:
            continue
        for edge in g["links"]:
            if edge["reason"]=="near_model_text_5gram_jaccard" and pairs<3:
                lines += [f"Jaccard={edge['jaccard']:.6f}, length_ratio={edge['length_ratio']:.6f}; это связь компоненты, не доказательство тождественности всех её пар.", ""]
                add(edge["a"],"near A"); add(edge["b"],"near B"); pairs+=1
        if pairs>=3:
            break
    if not pairs:
        lines += ["Подходящих near-пар вне защищённых групп нет.", ""]
    lines += ["## Конфликты (до 3 групп)", ""]
    conflicts=0
    for gid,g in sorted(groups.items(),key=lambda kv:stable_order(seed,"conflict-review",kv[0])):
        if g["quarantine"] and gid not in protected_groups | reserved_groups and conflicts<3:
            lines += [f"Группа `{gid}`, причина {g['quarantine_reason']}, оси `{g['conflict_axes']}`.", ""]
            by_label={}
            for rid in g["member_ids"]:
                key=(metadata[rid]["spam"],metadata[rid]["phishing"])
                by_label.setdefault(key,rid)
            for rid in list(by_label.values())[:3]:
                add(rid,"conflict")
            conflicts+=1
    if not conflicts:
        lines += ["Конфликтов не найдено в подтверждённых компонентах.", ""]
    lines += ["## Возможные проблемы декодирования (до 3)", ""]
    decode=[rid for rid,m in metadata.items() if "possible_decoding_problem" in m["flags"] and quality.by_record[rid] not in protected_groups | reserved_groups]
    for rid in sorted(decode,key=lambda r:stable_order(seed,"decoding-review",r))[:3]:
        add(rid,"decoding")
    if not decode:
        lines += ["Таких флагов нет.", ""]
    lines += [f"Всего уникальных показанных оригиналов: {len(shown)}. Наблюдения по этим фрагментам не обобщаются на весь корпус; полные подсчёты находятся в отчётах.", ""]
    return "\n".join(lines)


def status_markdown(manifest, normalized, quality, split, total_seconds):
    complete=manifest["status"]=="complete"
    return "\n".join(["# Email Guard — DAY2_STATUS", "", "Заключительные исправления после ручного просмотра выполнены реальным запуском. Исправленные отчёты ожидают приёмки; сентябрь 2026 остаётся security_blocked.", "",
        "| Условие | Статус | Результат |", "|---|---|---|",
        f"| Шаг 2: нормализация | {'done' if all(s['status']=='complete' for s in normalized['sources'].values()) else 'partial'} | {quality['normalized_records']:,} записей, пять источников; schema 2; {normalized['reuse']['mode']} |",
        f"| Шаг 3: качество/группы | done | {quality['group_count']:,} групп, {quality['quarantine_records']:,} quarantine, {quality['processed_records']:,} пригодных оригиналов |",
        f"| Шаг 4: назначения/разбиение | {'done' if complete else 'partial'} | test > validation > train согласован глобально |",
        "| modern_spam_test 2026-09 | security_blocked | Defender заблокировал и удалил архив по сообщению пользователя; не скачивался/не восстанавливался |",
        "| modern_spam_2025_period_challenge | done | отдельная защищённая одноклассовая проверка, не замена 2026 |",
        f"| Будущий резерв | frozen | {split['modern_reserve']['records']:,} оригиналов, train/validation-копии запрещены |",
        "| Компонента из 1000 писем | needs_review | MIME-аудит 10 оригиналов: общий plain-шум; ошибка декодера не подтверждена; метки/тела сохранены |",
        f"| Независимый аудит | done | {manifest['audit']['artifacts_checked']} файлов, утечек групп: {manifest['audit']['global_group_leakage']} |",
        "| Near-поиск | done | вероятностный LSH + точный Jaccard; неполная полнота поиска явно оговорена |",
        "| Обучение/fit преобразований | не запускались | ни моделей, ни TF-IDF/embeddings |",
        "| Шаг 5: ручная приёмка | pending | DAY2_STATUS.md, data_quality_report.md, split_report.md, data_review_sample.md |", "",
        f"Время pipeline: {total_seconds:.2f} с. Объём опубликованных processed JSONL: {sum(a['size_bytes'] for a in manifest['artifacts'].values()):,} байт. Временная SQLite содержит только производные данные и удаляется после публикации.", "",
        "## Настройки и ограничения", "", f"modern_spam_test: {', '.join(split['modern_test_periods'])}, {split['modern_test_status']}; {split['modern_test_period_basis']}. modern_spam_2025_period_challenge: {', '.join(split['modern_period_challenge_periods'])}.", "",
        LIMITATIONS, "", "Pending: " + ", ".join(split["pending"]) + ".", "",
        "## Повторение", "", "```powershell", ".\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v",
        ".\\.venv\\Scripts\\python.exe -B .\\src\\data_pipeline.py --trial",
        ".\\.venv\\Scripts\\python.exe -B .\\src\\data_pipeline.py",
        ".\\.venv\\Scripts\\python.exe -B .\\src\\pipeline_audit.py", "```", "",
        "Pipeline проверяет SHA исходных файлов и cache-контракт; корректные результаты не пересчитывает. --force-normalize принудительно пересоздаёт нормализацию; --force-process повторяет качество/разбиение. Пробный режим пишет только data/interim/trial, data/processed/trial и reports/trial.", ""])
