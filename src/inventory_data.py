"""Read-only inventory of the downloaded Email Guard datasets.

No extraction, network access, attachment execution, training or corpus creation.
The external documentation notes below were checked separately on 2026-10-10;
reruns reproduce local measurements, not a fresh check of those websites.
"""
import argparse
import csv
import hashlib
import json
import mailbox
import re
import sys
import tarfile
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesHeaderParser
from pathlib import Path

sys.dont_write_bytecode = True
from read_email import parse_email

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
REPORTS = ROOT / "reports"
SEED = 42
LIMIT = 20
MAX_SAMPLE_BYTES = 20 * 1024 * 1024
SA_LABELS = {"easy_ham": "ham", "easy_ham_2": "ham", "hard_ham": "ham",
             "spam": "spam", "spam_2": "spam"}
MESSAGE_NAME = re.compile(r"\d+\.[0-9a-f]{32}\Z")
HEADER_NAMES = ["Subject", "From", "To", "Cc", "Date", "Message-ID", "Reply-To",
                "Received", "Content-Type", "MIME-Version", "List-Unsubscribe",
                "List-Unsubscribe-Post", "Authentication-Results", "DKIM-Signature",
                "X-Spam-Flag", "X-Spam-Status"]
EXTERNAL = {
    "checked_on": "2026-10-10",
    "spamassassin": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/readme.html",
        "facts": "ham/spam are folder labels; sender retains text copyright; some addresses obfuscated; older versions overlap newer ones.",
        "published_counts": {"easy_ham": 2500, "easy_ham_2": 1400, "hard_ham": 250, "spam": 500, "spam_2": 1397},
    },
    "lingspam": {
        "url": "https://www.kaggle.com/mandygu/lingspam-dataset/activity",
        "facts": "Publisher card: 2893 messages, 2412 legitimate, 481 spam, headers removed, license Unknown. No explicit numeric-label mapping found.",
    },
    "nazario": {
        "url": "https://monkey.org/~jose/phishing/README.txt",
        "license_url": "https://monkey.org/~jose/phishing/LICENSE.txt",
        "facts": "Author: hand-classified personal inbox, errors possible, later files not anonymized; CC BY 4.0 with attribution.",
    },
}


def rel(path):
    return path.relative_to(ROOT).as_posix()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest():
    return [{"path": rel(p), "size_bytes": p.stat().st_size, "sha256": sha_file(p)}
            for p in sorted(RAW.rglob("*")) if p.is_file()]


def rank(record):
    return hashlib.sha256(f"{SEED}:{record['locator']}".encode()).hexdigest()


def choose(records):
    """Up to 20 per source/class, round-robin across files, hash order within file."""
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups[record["class"]][record["file"]].append(record)
    selected = {}
    for label, files in sorted(groups.items()):
        queues = [sorted(rs, key=rank) for _, rs in sorted(files.items())]
        sample = []
        for index in range(LIMIT):
            for queue in queues:
                if index < len(queue) and len(sample) < LIMIT:
                    sample.append(queue[index])
        selected[label] = sample
    return selected


def email_summary(raw):
    """Only presence, sizes, names and diagnostics; never content or addresses."""
    headers = BytesHeaderParser(policy=policy.default).parsebytes(raw)
    parsed = parse_email(raw)
    fields = {name: parsed[name] is not None and bool(parsed[name]) for name in
              ["subject", "sender", "sender_address", "sender_domain", "recipient",
               "plain_text", "html", "html_text", "body_text", "headers_raw"]}
    fields.update({"links_detected": bool(parsed["links"]),
                   "attachments_detected": bool(parsed["attachments"]),
                   "list_unsubscribe": parsed["unsubscribe"]["header_present"],
                   "list_unsubscribe_post": parsed["unsubscribe"]["list_unsubscribe_post"] is not None,
                   "unsubscribe_body_hint": parsed["unsubscribe"]["body_hint_found"],
                   "formatting_body_char_count": parsed["formatting"]["body_char_count"] is not None,
                   "formatting_uppercase_letter_ratio": parsed["formatting"]["uppercase_letter_ratio"] is not None,
                   "parse_warnings": bool(parsed["parse_warnings"])})
    return {"status": "parsed", "fields_present": fields,
            "headers_present": {h: h in headers for h in HEADER_NAMES},
            "all_header_names": sorted(set(h.lower() for h in headers.keys())),
            "body_source": parsed["body_source"],
            "body_char_count": parsed["formatting"]["body_char_count"],
            "link_count": len(parsed["links"]), "attachment_count": len(parsed["attachments"]),
            "attachment_mime_types": sorted(set(a["content_type"] for a in parsed["attachments"])),
            "warnings": parsed["parse_warnings"]}


def summarize_samples(selected, loader, kind="email"):
    result = {}
    for label, records in selected.items():
        items = []
        for record in records:
            item = dict(record)
            if record.get("size_bytes", 0) > MAX_SAMPLE_BYTES:
                item.update(status="skipped_size_limit")
                items.append(item)
                continue
            raw = loader(record)
            item["sample_sha256"] = hashlib.sha256(raw).hexdigest()
            try:
                if kind == "email":
                    item.update(email_summary(raw))
                else:
                    row = json.loads(raw)
                    body = row["message"]
                    item.update(status="inspected_text_only", fields_present={
                        "subject": bool(row["subject"].strip()), "message": bool(body.strip()),
                        "label": bool(row["label"].strip())},
                        subject_char_count=len(row["subject"]), message_char_count=len(body),
                        message_line_count=len(body.splitlines()),
                        spaced_punctuation=bool(re.search(r"\w\s+[-,:;.]\s+\w", body)),
                        header_like_residue=bool(re.search(r"(?im)^\s*(content\s*-\s*length|from|to|subject)\s*:", body)),
                        linguistics_terms=bool(re.search(r"\b(linguist\w*|syntax|phonolog\w*|linguistics)\b", body, re.I)),
                        commercial_terms=bool(re.search(r"\b(money|dollar\w*|credit|profit\w*|income|million\w*)\b", body, re.I)))
            except Exception as error:
                # Inventory boundary: retain locator and exception type, never message text.
                item.update(status="error", error_type=type(error).__name__)
            items.append(item)
        successes = [i for i in items if "fields_present" in i]
        result[label] = {"selected_count": len(items), "successful_count": len(successes),
                         "by_file": dict(Counter(r["file"] for r in records)),
                         "field_present_counts": dict(Counter(k for i in successes for k, v in i["fields_present"].items() if v)),
                         "header_present_counts": dict(Counter(k for i in successes for k, v in i.get("headers_present", {}).items() if v)),
                         "sample_only": True, "records": items}
    return result


def inspect_spamassassin():
    files, records = [], []
    for path in sorted((RAW / "spamassassin").glob("*")):
        if not path.is_file() or not tarfile.is_tarfile(path):
            continue
        with tarfile.open(path, "r:*") as archive:
            members = []
            for member in archive.getmembers():
                folder = member.name.split("/")[0]
                is_record = member.isfile() and folder in SA_LABELS and bool(MESSAGE_NAME.fullmatch(member.name.rsplit("/", 1)[-1]))
                members.append({"name": member.name, "size_bytes": member.size,
                                "type": "file" if member.isfile() else "directory" if member.isdir() else "other",
                                "role": "email" if is_record else "service_or_other"})
                if is_record:
                    records.append({"file": rel(path), "member": member.name,
                                    "locator": f"{rel(path)}::{member.name}",
                                    "class": SA_LABELS[folder], "subset": folder, "size_bytes": member.size})
            files.append({"path": rel(path), "format": "tar.bz2", "member_count": len(members),
                          "uncompressed_regular_bytes": sum(m["size_bytes"] for m in members if m["type"] == "file"),
                          "email_count": sum(m["role"] == "email" for m in members),
                          "members": members})

    def load(record):
        with tarfile.open(ROOT / record["file"], "r:*") as archive:
            with archive.extractfile(record["member"]) as handle:
                return handle.read()

    return {"source": "SpamAssassin", "format": "individual_email_in_tar_bz2",
            "whole_corpus": {"record_count": len(records), "class_counts": dict(Counter(r["class"] for r in records)),
                             "subset_counts": dict(Counter(r["subset"] for r in records)),
                             "count_method": "Regular members with number.MD5-style name in known class folders; cmds and directories excluded. Full MIME validity only sampled."},
            "label_provenance": {"rule": SA_LABELS, "status": "documented_by_project_and_publisher", "spam_values": {"ham": 0, "spam": 1},
                                 "phishing_label": None, "category_label": None},
            "local_documentation": {"readme": [], "license": [], "description": [], "note": "Only class directories, email files and cmds inside archives."},
            "files": files, "samples": summarize_samples(choose(records), load)}


def inspect_lingspam():
    path = RAW / "lingspam/messages.csv"
    records, rows, nonempty = [], {}, Counter()
    malformed = []
    with path.open(encoding="utf-8-sig", errors="strict", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if columns != ["subject", "message", "label"]:
            raise ValueError("Unexpected CSV schema; inspect before assigning fields")
        for index, row in enumerate(reader, 1):
            if None in row or any(v is None for v in row.values()):
                malformed.append(index)
                continue
            rows[index] = row
            nonempty.update(k for k, v in row.items() if v.strip())
            records.append({"file": rel(path), "row_number": index, "csv_physical_end_line": reader.line_num,
                            "locator": f"{rel(path)}::data_row={index}", "class": row["label"]})

    def load(record):
        return json.dumps(rows[record["row_number"]], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    return {"source": "Ling-Spam", "format": "CSV_text_only_with_preprocessing_indicators", "columns": columns,
            "whole_corpus": {"record_count": len(records), "class_counts": dict(Counter(r["class"] for r in records)),
                             "nonempty_fields": dict(nonempty), "malformed_row_numbers": malformed,
                             "count_method": "csv.DictReader logical data rows, not physical lines; UTF-8 strict."},
            "label_provenance": {"raw_field": "label", "status": "numeric_semantics_inferred_not_explicitly_documented",
                                 "candidate_mapping": {"0": "ham", "1": "spam"},
                                 "evidence": "Raw counts 0=2412, 1=481 match publisher legitimate/spam totals respectively. No explicit codebook found; sample word indicators are corroboration, not truth labels.",
                                 "confirmed_spam_label": None, "phishing_label": None, "category_label": None},
            "feature_availability": {"available_columns": columns,
                                     "unknown_original_features": ["From", "To", "Date", "Message-ID", "Received", "raw_headers", "MIME", "original_html", "original_links", "attachments", "unsubscribe_headers", "original_formatting"],
                                     "note": "CSV text is not serialized RFC email. Header-like strings or URL fragments in message do not restore original metadata. Subject is the CSV field, not a verified raw header."},
            "local_documentation": {"readme": [], "license": [], "description": [], "note": "Only messages.csv downloaded; publisher card license Unknown."},
            "files": [{"path": rel(path), "format": "CSV", "encoding": "UTF-8 (strict read, optional BOM accepted)"}],
            "samples": summarize_samples(choose(records), load, "csv")}


def inspect_nazario():
    files, records = [], []
    for path in sorted((RAW / "nazario").glob("*")):
        if not path.is_file():
            continue
        starts, header_starts = [], []
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.startswith(b"From "):
                    starts.append(offset)
                    header_starts.append(handle.tell())
        box = mailbox.mbox(path, create=False)
        # mailbox opens rb+ by default; replace with rb before any indexing/reading.
        box._file.close()
        box._file = path.open("rb")
        try:
            mailbox_count = len(box)
            if mailbox_count != len(starts):
                raise ValueError("mbox boundary counts disagree")
            internal, missing = [], []
            for i, start in enumerate(starts):
                stop = starts[i + 1] if i + 1 < len(starts) else path.stat().st_size
                with box.get_file(i) as message_file:
                    headers = BytesHeaderParser(policy=policy.default).parse(message_file)
                flags = {"folder_internal_subject": "FOLDER INTERNAL DATA" in str(headers.get("Subject", "")).upper(),
                         "internal_sender": "MAIL SYSTEM INTERNAL DATA" in str(headers.get("From", "")).upper(),
                         "x_imap_header": "X-IMAP" in headers}
                metadata = all(flags.values())
                if not headers:
                    missing.append(i + 1)
                record = {"file": rel(path), "mbox_record_number": i + 1,
                          "separator_byte_offset": start, "message_byte_start": header_starts[i], "message_byte_stop_exclusive": stop,
                          "size_bytes": stop - header_starts[i], "locator": f"{rel(path)}::mbox_record={i+1}@byte={start}",
                          "class": "internal_metadata_unknown" if metadata else "phishing",
                          "internal_metadata_indicators": flags if any(flags.values()) else {}}
                records.append(record)
                if metadata:
                    internal.append(record["locator"])
            files.append({"path": rel(path), "format": "mbox", "physical_mbox_records": mailbox_count,
                          "email_count": mailbox_count - len(internal), "internal_metadata_locators": internal,
                          "no_header_record_numbers": missing, "preamble_bytes": starts[0] if starts else path.stat().st_size})
        finally:
            box.close()

    def load(record):
        with (ROOT / record["file"]).open("rb") as handle:
            handle.seek(record["message_byte_start"])
            return handle.read(record["size_bytes"])

    classes = dict(Counter(r["class"] for r in records))
    return {"source": "Nazario", "format": "mbox", "whole_corpus": {
                "container_record_count": len(records), "email_count": classes.get("phishing", 0), "class_counts": classes,
                "count_method": "All line-start From delimiters, cross-checked with mailbox.mbox; header scan of every record. Three matching metadata indicators exclude one internal record from email count. Phishing class inherited from source, not independently re-annotated."},
            "label_provenance": {"status": "source_level_hand_classification_claim", "rule": {"phishing": 1, "internal_metadata_unknown": None},
                                 "spam_label": None, "category_label": None, "negative_phishing_examples": 0},
            "local_documentation": {"readme": [], "license": [], "description": [], "note": "Only two mbox files downloaded; external author README/license checked separately."},
            "files": files, "samples": summarize_samples(choose(records), load)}


def render_markdown(report):
    lines = ["# Инвентаризация Email Guard — день 2", "", "Дата проверки: 10 октября 2026. Только инвентаризация локальных данных.", "",
             "## Метод и границы", "",
             "Проверены README_READER.md, docs/DATASETS.md, docs/DAY1_STATUS.md, docs/CODEX_PROCESS_DATA_PROMPT.md и src/read_email.py. AGENTS.md в проекте и родительских каталогах не найден. Более широкое задание подготовки из docs ограничено текущим запросом пользователя.", "",
             "Поиск данных ограничен data/raw. Исходные файлы открывались для чтения; SHA-256 и размеры до/после совпали. tar.bz2 прочитаны без распаковки. Ссылки из писем не открывались; вложения, cmds, HTML/JS и инструкции в данных не выполнялись. HTML разобран статически существующим parse_email. Общий корпус, разбиение и модели не создавались.", "",
             "Полные подсчёты ниже относятся ко всем файлам/членам архивов/CSV-строкам/mbox-элементам. MIME и наличие полей проверены только на выборке: до 20 записей каждого класса каждого источника, включая отдельную служебную запись Nazario. Выборка: seed=42, SHA-256 от `42:<локатор>`, сортировка внутри каждого файла и циклический отбор между файлами. Это диагностическая, не статистически репрезентативная выборка.", "",
             "CSV-локатор — номер логической строки данных с 1 (заголовок не считается); physical_end_line — номер последней физической строки. mbox-локатор — номер элемента с 1 и смещение From-разделителя в байтах. Хеш mbox-образца: исходные байты после строки From до следующего разделителя/EOF, включая промежуточные пустые строки; без пересериализации и нормализации. Хеш CSV-образца — канонический JSON полей UTF-8, не исходный байтовый фрагмент CSV. Хеши файлов относятся к исходникам целиком.", "",
             "## Файлы на диске (полный перечень)", "", "| Файл | Байт | MiB |", "|---|---:|---:|"]
    for file in report["raw_files"]:
        lines.append(f"| `{file['path']}` | {file['size_bytes']:,} | {file['size_bytes']/1048576:.3f} |")
    lines += ["", f"Всего файлов: {len(report['raw_files'])}; байт: {sum(f['size_bytes'] for f in report['raw_files']):,}. SHA-256 каждого файла и полный перечень членов архивов с размерами — в JSON.", ""]
    for key, source in report["sources"].items():
        lines += [f"## {source['source']}", "", f"Формат: `{source['format']}`.", "", "Полный подсчёт:", "", "```json", json.dumps(source["whole_corpus"], ensure_ascii=False, indent=2), "```", ""]
        if key == "spamassassin":
            lines += ["Отдельные письма имеют имена `номер.хеш`, без `.eml`. Корень каждого архива соответствует его подкорпусу. В каждом архиве также есть каталог и файл `cmds` (текст команд переименования, не письмо). Других служебных файлов, ссылок или вложенных архивов не найдено.", "", "| Архив | Членов | Писем | Распакованный размер обычных файлов, байт |", "|---|---:|---:|---:|"]
            for file in source["files"]:
                lines.append(f"| `{Path(file['path']).name}` | {file['member_count']} | {file['email_count']} | {file['uncompressed_regular_bytes']:,} |")
            lines += ["", "Метки наследуются от папок: easy_ham/easy_ham_2/hard_ham → spam_label=0; spam/spam_2 → spam_label=1. phishing_label и category_label неизвестны. Это разметка источника, не повторная ручная проверка каждого письма. [Описание Apache](https://spamassassin.apache.org/old/publiccorpus/readme.html).", "",
                      "В скачанном spam_2 **1396**, в онлайн-README — **1397**; локально всего **6046**, а не 6047. Причина расхождения не установлена, полнота загрузки не подтверждена контрольной суммой издателя.", ""]
        elif key == "lingspam":
            lines += ["Колонки: subject, message, label. 2893 логических строки; тема непуста у 2831, отсутствует/пуста у 62; message и label непусты у всех. Это полный подсчёт CSV, не выборка.", "",
                      "Числовые значения подтверждены: `0=2412`, `1=481`. [Карточка издателя](https://www.kaggle.com/mandygu/lingspam-dataset/activity) заявляет 2412 legitimate и 481 spam, удалённые заголовки и лицензию Unknown. Соответствие **0→ham, 1→spam выведено по совпадению численностей**, явный словарь кодировки не найден. До его подтверждения это кандидатное соответствие; подтверждённая spam_label остаётся null. phishing_label/category_label также null.", "",
                      "CSV даёт тему и обработанный текст, а не оригинальное RFC/MIME-письмо. В выборке проверены строки, пунктуация с пробелами и остаточные строки вида content - length. Точный вариант обработки (bare/lemm/stop/lemm_stop) не установлен. Из этих данных нельзя достоверно восстановить From/To/Date, raw headers, HTML, исходные ссылки, вложения, отписку или форматирование. Их отсутствие в CSV означает неизвестность в оригинале.", ""]
        else:
            lines += ["Расширение .txt не определяет формат: оба файла содержат mbox с множеством писем. 2024: 403 элемента = 402 письма + 1 служебная запись. 2025: 481 элемент = 481 письмо. Итого **884 элемента, 883 письма**.", "",
                      "Служебная запись: `data/raw/nazario/phishing-2024.txt::mbox_record=1@byte=0`. Одновременно найдены маркеры FOLDER INTERNAL DATA в Subject, Mail System Internal Data в From и X-IMAP. Она остаётся в исходнике, учтена отдельно с неизвестными метками и не считается фишингом.", "",
                      "Для остальных писем phishing_label=1 наследуется от фишингового источника; отрицательных примеров нет, spam_label/category_label=null. [Авторский README](https://monkey.org/~jose/phishing/README.txt) описывает ручную разметку собственной почты и возможные ошибки. По его описанию поздние файлы не обезличены; [лицензия CC BY 4.0](https://monkey.org/~jose/phishing/LICENSE.txt).", ""]
        lines += ["Локальных README, описания источника и LICENSE в скачанной папке/архивах нет. docs/DATASETS.md — описание проекта, а не комплект документации издателя.", "", "### Наличие полей — только выборка", "", "Числа означают количество непустых/обнаруженных значений среди успешно разобранных образцов. Ноль означает «не обнаружено этой проверкой», не отсутствие во всём корпусе. Поля From/To не удостоверяют подлинность; извлечение ссылок/вложений ограничено текущим парсером.", ""]
        if key == "lingspam":
            fields = ["subject", "message", "label"]
        else:
            fields = ["subject", "sender", "sender_address", "sender_domain", "recipient", "headers_raw", "plain_text", "html", "html_text", "body_text", "links_detected", "attachments_detected", "list_unsubscribe", "list_unsubscribe_post", "unsubscribe_body_hint", "formatting_body_char_count", "formatting_uppercase_letter_ratio", "parse_warnings"]
        labels = list(source["samples"])
        lines += ["| Поле/признак | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
        for field in fields:
            lines.append("| " + field + " | " + " | ".join(f"{source['samples'][c]['field_present_counts'].get(field,0)}/{source['samples'][c]['successful_count']}" for c in labels) + " |")
        if key != "lingspam":
            lines += ["", "Заголовки исходного письма — также только выборка:", "",
                      "| Заголовок | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
            for field in HEADER_NAMES:
                lines.append("| " + field + " | " + " | ".join(f"{source['samples'][c]['header_present_counts'].get(field,0)}/{source['samples'][c]['successful_count']}" for c in labels) + " |")
        lines += ["", "Локаторы выборки (значения полей и полные письма не публикуются):", ""]
        for label, sample in source["samples"].items():
            lines += [f"**{label}**: выбрано {sample['selected_count']}, успешно {sample['successful_count']}; распределение по файлам: `{json.dumps(sample['by_file'], ensure_ascii=False)}`.", ""]
            if key == "lingspam":
                indicators = {k: sum(bool(i.get(k)) for i in sample["records"]) for k in ["spaced_punctuation", "header_like_residue", "linguistics_terms", "commercial_terms"]}
                lines += [f"Индикаторы текста только в выборке: `{json.dumps(indicators)}`. Ключевые слова не доказывают класс.", ""]
            for item in sample["records"]:
                lines.append(f"- `{item['locator']}` — {item['status']}")
            lines.append("")
        lines += ["Подробные флаги полей, имена заголовков, длины, типы вложений, предупреждения и хеши каждого образца — в JSON. Значения заголовков/URL/имена вложений не сохранены в отчёт.", ""]
    for title, key in [("Подтверждённые факты", "confirmed_facts"), ("Неизвестное и неоднозначное", "unknowns"), ("Потенциальные дубликаты версий корпуса", "potential_duplicate_versions"), ("Ограничения для задач", "limitations")]:
        lines += [f"## {title}", ""]
        lines += [f"- {item}" for item in report[key]]
        lines.append("")
    lines += ["## Пригодность", "", "| Источник | Антиспам | Фишинг | Тематические категории |", "|---|---|---|---|",
              "| SpamAssassin, 6046 писем | Да: ham 4150, spam 1896 | Нет достоверной разметки этой оси | Нет разметки |",
              "| Ling-Spam, 2893 строки | Текстовый кандидат; сначала подтвердить числовую кодировку меток | Метки неизвестны | Нет разметки; лингвистическая рассылка не заменяет категории |",
              "| Nazario, 883 письма + 1 служебный элемент | Независимые spam-метки неизвестны | Положительные примеры; для бинарной задачи нужны проверенные отрицательные | Нет разметки |", "",
              "## Повторный запуск", "", "Из корня проекта, Python 3.12, без новых зависимостей:", "", "```powershell", ".\\.venv\\Scripts\\python.exe .\\src\\inventory_data.py", "```", "",
              "Допустим системный `python .\\src\\inventory_data.py`. Скрипт работает без сети; перезаписывает только два отчёта. Онлайн-сведения зафиксированы в скрипте как заметки проверки от 10 октября 2026, при повторном запуске сайты не перепроверяются. Для локальных данных результат детерминирован при неизменных исходниках, скрипте и парсере. Ограничение одного образца — 20 MiB; пропуски/ошибки видны по статусам.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Read-only dataset inventory")
    parser.add_argument("--russian-only", action="store_true",
                        help="Update Russian Parquet; reuse existing inventory of other sources")
    args = parser.parse_args()
    if args.russian_only:
        from russian_inventory import augment_inventory
        report = json.loads((REPORTS / "data_inventory.json").read_text(encoding="utf-8"))
        markdown = (REPORTS / "data_inventory.md").read_text(encoding="utf-8")
        report, markdown, review = augment_inventory(report, markdown, ROOT)
        write_reports(report, markdown, review)
        print(json.dumps({"status": "inventory_complete", "updated_source": "russian_auxiliary",
                          "whole_corpus": report["sources"]["russian_auxiliary"]["whole_corpus"]}, ensure_ascii=False, indent=2))
        return
    before = manifest() if RAW.exists() else []
    if not before:
        report = {"status": "data_raw_absent_or_empty", "raw_files": [], "sources": {}}
        markdown = "# Инвентаризация Email Guard\n\ndata/raw отсутствует или не содержит файлов. Поиск за пределами data/raw не выполнялся.\n"
    else:
        sources = {}
        if (RAW / "spamassassin").is_dir():
            sources["spamassassin"] = inspect_spamassassin()
        if (RAW / "lingspam/messages.csv").is_file():
            sources["lingspam"] = inspect_lingspam()
        if (RAW / "nazario").is_dir():
            sources["nazario"] = inspect_nazario()
        if manifest() != before:
            raise RuntimeError("Source manifest changed during inventory; reports not written")
        covered = {f["path"] for s in sources.values() for f in s["files"]}
        report = {
            "schema_version": 1, "status": "inventory_complete", "checked_on": "2026-10-10",
            "scope": "inventory_only", "raw_files": before, "sources_unchanged_sha256_verified": True,
            "uninspected_files": [f["path"] for f in before if f["path"] not in covered],
            "sampling": {"seed": SEED, "maximum_per_source_class": LIMIT, "max_sample_bytes": MAX_SAMPLE_BYTES,
                         "selection": "SHA256(seed:locator) order per file; round robin sorted files per source/class",
                         "sample_results_are_not_corpus_field_counts": True},
            "implementation_sha256": {rel(p): sha_file(p) for p in [Path(__file__), ROOT / "src/read_email.py"]},
            "external_documentation": EXTERNAL, "sources": sources,
            "confirmed_facts": [
                "8 файлов в трёх папках data/raw; дополнительных источников, README и LICENSE локально нет.",
                "SpamAssassin: 6046 записей-писем по структуре архивов; ham=4150, spam=1896; 5 cmds исключены из числа писем.",
                "Ling-Spam: 2893 CSV-строки, label 0=2412 и 1=481, subject/message/label; 62 пустые темы.",
                "Nazario: 884 mbox-элемента; 883 письма с положительной фишинговой меткой источника и 1 служебный элемент с неизвестными метками.",
                "Исходные SHA-256 и размеры не изменились. Поля проверены в ограниченных выборках, локаторы и методы сохранены."],
            "unknowns": [
                "Точное происхождение скачанных файлов и совпадение с контрольными суммами издателя не установлены; ссылки в docs описывают ожидаемые источники.",
                "Ling-Spam: 0→ham/1→spam — вывод из совпадающих численностей, не найденный явный словарь; точный вариант обработки и лицензия неизвестны.",
                "Причина расхождения spam_2: 1396 локально против 1397 в README Apache не установлена.",
                "Достоверность каждой исходной метки не перепроверена; язык каждого письма, весь диапазон дат и полнота MIME по корпусам не аудированы.",
                "Phishing для антиспам-корпусов и spam для Nazario неизвестны; неизвестное не превращено в 0."],
            "potential_duplicate_versions": [
                "Ling-Spam bare/lemm/stop/lemm_stop могут быть обработками одних сообщений. Локально только messages.csv; принадлежность конкретному варианту неизвестна, четыре независимых корпуса не заявлены.",
                "Старые датированные версии SpamAssassin пересекаются с новыми по описанию Apache. Здесь по одному архиву каждого подкорпуса; easy_ham_2/spam_2 не объявлены дублями easy_ham/spam только из-за суффикса.",
                "Nazario 2024/2025 — два контейнера; годы и разные имена не гарантируют уникальность писем. Совпадения внутри/между источниками и близкие дубликаты в этой инвентаризации не искались и не удалялись."],
            "limitations": [
                "SpamAssassin старый и с частично изменёнными адресами; полный MIME сохранён в исходниках, но реальные поля варьируют. Права на тексты остаются у отправителей по README Apache.",
                "Ling-Spam тематически узок, обработанный CSV теряет оригинальные метаданные; распознавание рассылки может подменить антиспам. Лицензия на карточке Unknown.",
                "Nazario — почта одного человека и только положительный класс; возможны ошибки разметки. Для бинарной фишинговой задачи нужны независимо проверенные отрицательные, включая обычный спам.",
                "Не оценивать фишинг по одному лишь различию Nazario и других источников: модель может узнавать источник.",
                "Для рекламы/работы/финансов/служебных уведомлений/другого нет готовых меток ни у одного скачанного источника; spam и phishing — отдельные оси.",
                "Русский email-корпус и независимый набор преподавателя в data/raw отсутствуют. Энрон, русский Parquet и Spambase не скачаны; отсутствие необязательных источников не блокирует инвентаризацию.",
                "Выборка до 20 на класс описывает только наблюдавшиеся записи. Ноль найденных вложений/HTML/ссылок не доказывает их отсутствия в корпусе; текущий парсер извлекает лишь часть типов ссылок.",
                "В выборке Nazario X-Spam-Status есть у 20/20 фишинговых писем: это ответ почтового фильтра, возможная утечка целевого признака и маркер источника при будущем обучении, а не независимая spam-разметка. На данном этапе он только учтён, ничего не удалено.",
                "Подлинность отправителей, SPF/DKIM/DMARC, безопасность URL и содержимого вложений, работа отписки не проверялись. Поздние Nazario содержат персональные сведения по описанию автора."],
        }
        markdown = render_markdown(report)
    review = None
    if (RAW / "russian_auxiliary/processed_combined.parquet").is_file():
        from russian_inventory import augment_inventory
        report, markdown, review = augment_inventory(report, markdown, ROOT)
    write_reports(report, markdown, review)
    print(json.dumps({"status": report["status"], "reports": ["reports/data_inventory.json", "reports/data_inventory.md"],
                      "sources": {k: v["whole_corpus"] for k, v in report["sources"].items()}}, ensure_ascii=False, indent=2))


def write_reports(report, markdown, review=None):
    REPORTS.mkdir(exist_ok=True)
    outputs = [("data_inventory.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
               ("data_inventory.md", markdown)]
    if review is not None:
        outputs.append(("russian_review_sample.md", review))
    staged = []
    for name, content in outputs:
        target = REPORTS / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        staged.append((temporary, target))
    for temporary, target in staged:
        temporary.replace(target)


if __name__ == "__main__":
    main()
