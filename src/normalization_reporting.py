"""Compact statistics and inert review fragments; no full emails in reports/logs."""
import hashlib
import json
import re
from collections import Counter, defaultdict

from russian_inventory import safe_fragment


def new_stats(source):
    return {"source": source, "status": "complete", "input_records": 0, "input_email_candidates": 0,
            "written": 0, "parse_status_counts": {"ok": 0, "partial": 0, "failed": 0},
            "excluded_by_reason": Counter(), "original_label_counts": Counter(),
            "written_label_counts": {axis: {"0": 0, "1": 0, "null": 0} for axis in ["spam", "phishing", "category"]},
            "empty_subject": 0, "empty_model_text": 0, "records_with_warnings": 0,
            "warning_counts": Counter(), "transformation_counts": Counter(), "removed_filter_header_lines": 0,
            "possible_filter_marker": 0, "possible_filter_marker_locators": [],
            "filter_headers_in_diagnostics": Counter(), "subset_written_counts": Counter()}


def count_written(stats, record):
    stats["written"] += 1
    stats["parse_status_counts"][record["parse_status"]] += 1
    for axis in ["spam", "phishing", "category"]:
        value = record[axis + "_label"]
        stats["written_label_counts"][axis]["null" if value is None else str(value)] += 1
    stats["empty_subject"] += record["subject"] is None or not record["subject"].strip()
    stats["empty_model_text"] += not record["usable_model_text"]
    stats["records_with_warnings"] += bool(record["parse_warnings"])
    stats["warning_counts"].update(record["parse_warnings"])
    stats["possible_filter_marker"] += record["possible_filter_marker"]
    if record["possible_filter_marker"] and len(stats["possible_filter_marker_locators"]) < 20:
        stats["possible_filter_marker_locators"].append(record["original_locator"])
    stats["subset_written_counts"][record["subset"] or "not_applicable"] += 1
    for change in record["transformations"]:
        stats["transformation_counts"][change["operation"]] += 1
        stats["removed_filter_header_lines"] += len(change.get("line_numbers", []))


def fragment(text):
    if text is not None:
        text = re.sub(r"(?<!\w)\+?\d[\d ()\t.-]{6,}\d(?!\w)", "[телефон/номер]", text)
    return safe_fragment(text, limit=160)


class ReviewSamples:
    def __init__(self, seed):
        self.seed = seed
        self.groups = defaultdict(list)

    def add(self, record):
        label = record["phishing_label"] if record["source"] == "nazario" else record["spam_label"]
        if label is None:
            return
        group = (record["source"], label)
        limit = 10 if record["source"] == "lingspam" else 5
        priority = hashlib.sha256(f"{self.seed}:{record['record_id']}".encode()).hexdigest()
        items = self.groups[group]
        if len(items) == limit and priority >= items[-1][0]:
            return
        original = "\n\n".join(v for v in [record["subject"], record["body_text"]] if v)
        item = {"record_id": record["record_id"], "original_locator": record["original_locator"],
                "spam_label": record["spam_label"], "phishing_label": record["phishing_label"],
                "original_label": record["original_label"], "label_status": record["label_status"],
                "original_fragment": fragment(original), "model_fragment": fragment(record["model_text"]),
                "feature_availability": record["feature_availability"], "parse_status": record["parse_status"],
                "transformations": record["transformations"], "possible_filter_marker": record["possible_filter_marker"]}
        items.append((priority, item))
        items.sort(key=lambda pair: pair[0])
        del items[limit:]

    def render(self):
        lines = ["# Шаг 2: фрагменты для ручной проверки", "",
                 f"Только диагностическая выборка, не подтверждение меток. По 5 записей на источник/известный класс, Ling-Spam — по 10. Отбор по наименьшему SHA-256(seed:record_id), seed={self.seed}. Фрагменты до 160 символов, контакты/ссылки обезврежены, Markdown/HTML экранированы только здесь. Поля JSONL не изменены ради отчёта; инструкции в текстах не выполнялись.", ""]
        for (source, label), items in sorted(self.groups.items()):
            axis = "phishing" if source == "nazario" else "spam"
            lines += [f"## {source}: {axis}={label} ({len(items)} примеров)", ""]
            for _, item in items:
                lines += [f"- Локатор: `{item['original_locator']}`; ID: `{item['record_id']}`.  ",
                          f"  Метки: spam={item['spam_label']}, phishing={item['phishing_label']}; исходная={json.dumps(item['original_label'], ensure_ascii=False)}; статус={item['label_status'][axis]}; parse={item['parse_status']}.  ",
                          f"  Оригинал: {item['original_fragment']}  ",
                          f"  model_text: {item['model_fragment']}  ",
                          f"  Доступность: `{json.dumps(item['feature_availability'], ensure_ascii=False, sort_keys=True)}`.  ",
                          f"  Преобразования: `{json.dumps(item['transformations'], ensure_ascii=False)}`; possible_filter_marker={item['possible_filter_marker']}.", ""]
        return "\n".join(lines)


def render_report(report):
    lines = ["# Email Guard — нормализация, день 2, шаг 2", "", "Дата выполнения: 11 октября 2026 (Москва). Только единая структура и извлечение; без обучения, дедупликации, тематической разметки и разделения.", "",
             f"Статус: **{report['status']}**. Режим: **{'пробный' if report['config']['trial'] else 'полный'}**. Исходные файлы не переписаны. Текущие SHA-256 вычислены потоково один раз на файл и сопоставлены с инвентаризацией; после прохода сверены размеры и mtime. Отчёты инвентаризации не менялись.", "",
             "## Счётчики", "", "| Источник | Прочитано | Записано | ok | partial | failed | Исключено |", "|---|---:|---:|---:|---:|---:|---:|"]
    for source, stats in report["sources"].items():
        counts = stats["parse_status_counts"]
        lines.append(f"| {source} ({stats['status']}) | {stats['input_records']:,} | {stats['written']:,} | {counts['ok']:,} | {counts['partial']:,} | {counts['failed']:,} | {sum(stats['excluded_by_reason'].values()):,} |")
    lines += ["", "Прочитано: для SpamAssassin — обычные/служебные элементы, кроме каталогов; для Nazario — mbox-элементы; CSV — логические строки; Parquet — реально просканированные строки. Русские пригодные, но не выбранные строки не являются ошибками/дубликатами и не записываются миллионами в excluded_records.", "",
              "## Метки и преобразования", "", "| Источник | spam 0 / 1 / null (записано) | phishing 0 / 1 / null | Пустая/отсутствующая тема | Пустой model_text |", "|---|---|---|---:|---:|"]
    for source, stats in report["sources"].items():
        def labels(axis):
            values = stats["written_label_counts"][axis]
            return " / ".join(str(values[k]) for k in ["0", "1", "null"])
        lines.append(f"| {source} | {labels('spam')} | {labels('phishing')} | {stats['empty_subject']} | {stats['empty_model_text']} |")
    lines += ["", "SpamAssassin: метки папок, source_provided. Ling-Spam: рабочие строковые 0→ham/1→spam, mapping_inferred, разрешено docs/DAY2_STEP2.md; авторский словарь не найден. Nazario: phishing=1 по источнику, spam=null; служебный элемент исключается только при трёх совместных маркерах. Русский: исходный bool сохраняется, False→0/True→1, source_intended_unverified. Возможное старое astype(bool) не доказано и по словам не исправлялось.", "",
              "**null** — разметка/поле неизвестны или недоступны; **0** — известный отрицательный класс этой оси. Category везде null. Отрицательный spam не означает отрицательный phishing; реклама и предложения работы не являются автоматически спамом.", "",
              "model_text: отдельно очищенные доступные subject и body_text, в этом порядке, разделитель два LF. Только LF и внешние пробелы плюс узкое удаление явных строк заголовков прежнего фильтра и их непустых продолжений. Оригинальные поля сохранены. HTML-альтернатива не дублирует plain. Нет перевода, лемматизации, удаления пунктуации, casefold или замены похожих букв.", ""]
    for source, stats in report["sources"].items():
        lines += [f"- {source}: исключения `{json.dumps(stats['excluded_by_reason'], ensure_ascii=False)}`; преобразования `{json.dumps(stats['transformation_counts'])}`; удалённых строк фильтра {stats['removed_filter_header_lines']}; possible_filter_marker {stats['possible_filter_marker']}; записей с предупреждениями {stats['records_with_warnings']}."]
    lines += ["", "Типы предупреждений и число их вхождений (одна запись может иметь несколько):", ""]
    for source, stats in report["sources"].items():
        if stats["warning_counts"]:
            lines.append(f"- {source}: `{json.dumps(stats['warning_counts'], ensure_ascii=False)}`.")
    lines += ["", "Локаторы possible_filter_marker (не более 20 на источник):", ""]
    found_markers = False
    for stats in report["sources"].values():
        for locator in stats["possible_filter_marker_locators"]:
            found_markers = True
            lines.append(f"- `{locator}`")
    if not found_markers:
        lines.append("Такие темы не найдены в обработанных записях.")
    lines += ["", "possible_filter_marker лишь отмечает начальные [SPAM]/[JUNK]/[BULK] или ***SPAM***; происхождение не установлено, префиксы автоматически не удаляются. Обычные предложения со словом spam сохраняются. Заголовки X-Spam и прочие ответы фильтра хранятся только в diagnostics и не включены в будущий список признаков.", "",
              "## Русский вспомогательный источник", ""]
    russian = report["sources"].get("russian_auxiliary", {}).get("russian_scan")
    if russian:
        lines += [f"Метаданные: {russian['metadata_rows']:,}; просканировано {russian['scanned_rows']:,}; полный проход: {russian['scan_complete']}. Исходные значения: `{json.dumps(russian['native_label_counts'])}`; null-текстов {russian['null_text']}, пробельных/пустых {russian['blank_text_nonnull']}.", "",
                  f"Пригодно: `{json.dumps(russian['eligible_per_class'])}`; выбрано: `{json.dumps(russian['selected_per_class'])}`; недобор до квоты: `{json.dumps(russian['shortfall_per_class'])}`; пригодных вне выборки: {russian['eligible_not_selected']:,}.", "",
                  f"Reservoir Algorithm R отдельно по классу; seed={russian['sampling']['seed']}, RNG False={russian['sampling']['rng_seed_false']}, True={russian['sampling']['rng_seed_true']}; квота {russian['sampling']['max_per_class']:,}. Чтение text/label пакетами по 16384, порядок исходных строк, выход сортируется по номеру. Точные выбранные номера сохранены в JSON-отчёте и локаторах JSONL. Весь файл не загружался в pandas/список.", "",
                  "Баланс выборки 50/50 создан нами; он не оценивает долю спама в исходном файле или почте. Данные остаются auxiliary/text_only, не смешаны с английскими. Канал сообщений неизвестен; проверка на русском email этим набором не заменяется.", ""]
    lines += ["## Сверка с инвентаризацией", ""]
    for source, check in report["inventory_comparison"].items():
        lines.append(f"- {source}: `{json.dumps(check, ensure_ascii=False)}`")
    lines += ["", "В SpamAssassin служебные cmds увеличивают счётчик входных элементов, но не число email; в Nazario один элемент — метаданные ящика. Пустые темы и предупреждения не удаляют записи: partial сохраняется. Пробный проход намеренно ограничен и не сравнивается с полным числом записей.", "",
              "## Свежие источники", ""]
    if report["pending_adapter"]:
        for item in report["pending_adapter"]:
            lines.append(f"- `{item['path']}`, {item['size_bytes']:,} байт — pending_adapter. Не распакован и не обработан.")
    elif "untroubled" in report["sources"]:
        modern = report["sources"]["untroubled"]
        lines.append(f"Untroubled 2025.7z: статус {modern['status']}, нормализовано {modern['written']:,}; spam=1/source_provided, phishing=null. Полный список членов и объёмов — untroubled_archive_members.json. Неадаптированных обнаруженных архивов нет.")
    else:
        lines.append("Свежие архивы SPAM Archive 2025/2026 не обнаружены в data/raw.")
    lines += ["", "## Файлы, схема и хеши", "", "Схема и ограничения: docs/DATA_SCHEMA.md. Конфигурация: normalization_config.json рядом с отчётом. Ошибки: normalization_errors.jsonl; список исключений — excluded_records.jsonl. [Фрагменты для приёмки](normalization_review_sample.md). Большие JSONL и приватная диагностика не предназначены для загрузки в чат/Git.", "",
              "source_file_sha256 — байты контейнера целиком; source_version — его SHA-256, не дата сбора. raw_sha256 для raw/mbox — исходные байты записи; для CSV/Parquet null. record_content_sha256 для таблиц — канонический JSON исходных полей (UTF-8, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False); для raw совпадает с raw_sha256. model_text_sha256 в отчёте не подменяет эти хеши. Хеши выходных файлов и кода — в JSON-отчёте.", "",
              "Mbox хешируется после From-строки до следующего разделителя/EOF, включая промежуточные пустые строки, без пересериализации. ID учитывает источник, относительный путь, отпечаток файла, локатор и позицию: одинаковые тексты разных записей сохраняют разные ID.", "",
              "## Проверки и ограничения", ""]
    verified = report.get("verification", {})
    lines += [f"Проверка созданных JSONL: **{verified.get('status', 'unknown')}**; уникальных ID {verified.get('unique_record_ids', 0):,}, связанных диагностических записей {verified.get('linked_diagnostics', 0):,}. Ошибок записей: {report['record_error_count']}; ошибок источников: {len(report['source_errors'])}.", "",
              "| Выходной файл | Строк | Байт |", "|---|---:|---:|"]
    for name, artifact in verified.get("artifacts", {}).items():
        lines.append(f"| {name} | {artifact['rows']:,} | {artifact['size_bytes']:,} |")
    lines += ["", "Полные SHA-256 выходных файлов сохранены в normalization_report.json; --verify-only проверяет их, а также хеш журнала ошибок.", "",
              "Обязательные проверки: типы и метки, схема каждой выходной записи, уникальность ID без удаления одинаковых текстов, связь всей диагностики с raw/mbox записями, соответствие model_text правилам, валидность SHA-256 и файлов после записи. Прогон тестов — команда ниже; результаты фактического прогона сообщает итог задачи.", "",
              "Границы парсера прежние: статический HTML, неполное извлечение URL/MIME, отсутствие проверки SPF/DKIM/DMARC и безопасности вложений/ссылок. Заголовки отправителя не доказывают подлинность, List-Unsubscribe — не доказательство ham. Дубликаты/конфликты — следующий шаг. Язык отдельных записей не подтверждён автоматически; права на источники и ограничения лицензий остаются прежними.", "",
              "## Воспроизведение", "", "Из корня проекта:", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v",
              ".\\.venv\\Scripts\\python.exe -B .\\src\\prepare_data.py --trial",
              ".\\.venv\\Scripts\\python.exe -B .\\src\\prepare_data.py",
              ".\\.venv\\Scripts\\python.exe -B .\\src\\prepare_data.py --verify-only", "```", "",
              "Пробные выходы: data/interim/trial и reports/trial. Полные: data/interim и reports. Параметры CLI: --raw-dir, --output-dir, --reports-dir, --seed, --russian-max-per-class, --max-record-bytes, --trial-limit. Зависимость pyarrow уже зафиксирована в requirements.txt. Повторный запуск заменяет только результаты шага 2, исходники и инвентаризацию не меняет.", ""]
    return "\n".join(lines)
