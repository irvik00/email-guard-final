"""Bounded-memory Parquet inventory; downloaded scripts are never imported/run."""
import bisect
import copy
import hashlib
import html
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BATCH_SIZE = 16384
PARTS = 5
PER_CLASS = 10
SEED = 42
SECTION = "## Русский вспомогательный набор"
END_SECTION = "<!-- end-russian-inventory -->"
PREFIX = "Russian auxiliary: "
REVIEWED_SHA256 = "b874d9ac10146eca333018963781d4adbb66dcd77a256dde789703c697885e52"


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(value):
    # JSON preserves native bool versus string "false"/"0"; never bool(value).
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def segments(parquet, batch_size, parts):
    """Return batches sliced at equal row-range boundaries; no whole-file table."""
    total = parquet.metadata.num_rows
    boundaries = [total * i // parts for i in range(parts + 1)]
    start = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["label", "text"], use_threads=False):
        offset = 0
        while offset < batch.num_rows:
            part = min(parts - 1, bisect.bisect_right(boundaries, start + offset) - 1)
            length = min(batch.num_rows - offset, boundaries[part + 1] - (start + offset))
            yield part, start + offset, batch.slice(offset, length)
            offset += length
        start += batch.num_rows


def safe_fragment(text, limit=160):
    """Inert short text; remove active links and escape Markdown/HTML syntax."""
    if text is None:
        return "[NULL]"
    text = " ".join(text.split())
    # Also neutralize bare domains/email addresses and Telegram handles.
    text = re.sub(r"(?i)(?:[a-z][a-z0-9+.-]*://|www\.)\S+|\S+@\S+|\b(?:[\w-]+\.)+[a-zа-я]{2,}(?:/\S*)?|(?<!\w)@[\w]+", "[ссылка/контакт]", text)
    text = text[:limit] + ("…" if len(text) > limit else "")
    text = "".join(c for c in text if c.isprintable())
    text = html.escape(text)
    return re.sub(r"([\\`*_|\[\]{}])", r"\\\1", text) or "[пусто]"


def sample_features(text):
    if text is None:
        return {"char_count": None, "word_count": None, "observation": "null_text"}
    header_lines = re.findall(r"(?im)^(?:from|to|subject|date|received|message-id|mime-version):", text)
    # These are structural indicators, not claims of authentic email/chat origin.
    return {"char_count": len(text), "word_count": len(text.split()),
            "short_at_most_280_chars": len(text) <= 280,
            "cyrillic_present": bool(re.search(r"[А-Яа-яЁё]", text)),
            "email_header_block_indicator": len(header_lines) >= 2,
            "dialogue_line_indicator": bool(re.search(r"(?m)^(?:\d{1,2}:\d{2}|[—-]\s)", text)),
            "observation": "text_only; origin_not_proven"}


def inspect_parquet(path, batch_size=BATCH_SIZE, parts=PARTS, per_class=PER_CLASS):
    """Two projected batch scans. Python objects are only counts and <=10/class."""
    before = {"size_bytes": path.stat().st_size, "sha256": sha_file(path)}
    counts, part_counts = Counter(), defaultdict(Counter)
    null_text = empty_text = whitespace_text = rows_read = 0
    with pq.ParquetFile(path) as parquet:
        schema = parquet.schema_arrow
        if "label" not in schema.names or "text" not in schema.names:
            raise ValueError("Expected text and label columns; inspect unknown schema explicitly")
        if not (pa.types.is_string(schema.field("text").type) or pa.types.is_large_string(schema.field("text").type)):
            raise ValueError("text must be an Arrow string column")
        label_type = schema.field("label").type
        if not (pa.types.is_boolean(label_type) or pa.types.is_integer(label_type) or pa.types.is_string(label_type)):
            raise ValueError("Unsupported label type; do not guess coercion")
        metadata_rows = parquet.metadata.num_rows
        row_groups = [{"index": i, "row_count": parquet.metadata.row_group(i).num_rows} for i in range(parquet.num_row_groups)]
        for part, _, batch in segments(parquet, batch_size, parts):
            rows_read += batch.num_rows
            labels, texts = batch.column("label"), batch.column("text")
            # to_pylist is applied ONLY to the distinct values/counts of one batch.
            for item in pc.value_counts(labels).to_pylist():
                token = key(item["values"])
                counts[token] += item["counts"]
                part_counts[part][token] += item["counts"]
            null_text += texts.null_count
            empty_text += pc.sum(pc.cast(pc.equal(texts, ""), pa.int64())).as_py() or 0
            stripped_empty = pc.equal(pc.utf8_trim_whitespace(texts), "")
            whitespace_text += pc.sum(pc.cast(stripped_empty, pa.int64())).as_py() or 0
        if rows_read != metadata_rows or sum(counts.values()) != metadata_rows:
            raise ValueError("Metadata/streaming counts disagree")

        # Allocate fairly across non-empty fifths, then choose occurrence ranks.
        targets = {}
        for token in sorted(counts):
            quotas = Counter()
            for _ in range(min(per_class, counts[token])):
                eligible = [p for p in range(parts) if quotas[p] < part_counts[p][token]]
                selected = min(eligible, key=lambda p: (quotas[p], p))
                quotas[selected] += 1
            for part, quota in sorted(quotas.items()):
                seed = int(hashlib.sha256(f"{SEED}:{token}:{part}".encode()).hexdigest(), 16)
                targets[part, token] = sorted(random.Random(seed).sample(range(part_counts[part][token]), quota))

        samples, seen = defaultdict(list), Counter()
        for part, start, batch in segments(parquet, batch_size, parts):
            labels, texts = batch.column("label"), batch.column("text")
            for item in pc.value_counts(labels).to_pylist():
                value, count = item["values"], item["counts"]
                token = key(value)
                ranks = targets.get((part, token), [])
                begin, end = seen[part, token], seen[part, token] + count
                chosen = ranks[bisect.bisect_left(ranks, begin):bisect.bisect_left(ranks, end)]
                if chosen:
                    mask = pc.is_null(labels) if value is None else pc.equal(labels, pa.scalar(value, type=label_type))
                    positions = pc.indices_nonzero(mask)
                    for occurrence in chosen:
                        position = positions[occurrence - begin].as_py()
                        text = texts[position].as_py()
                        record = {"row_number": start + position + 1, "part": part + 1,
                                  "original_label": value, "original_label_arrow_type": str(label_type),
                                  "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None,
                                  "fragment": safe_fragment(text), **sample_features(text)}
                        samples[token].append(record)
                seen[part, token] = end
    after = {"size_bytes": path.stat().st_size, "sha256": sha_file(path)}
    if before != after:
        raise RuntimeError("Parquet changed during read-only inventory")
    sample_groups = {}
    for token in sorted(counts):
        records = sorted(samples[token], key=lambda r: r["row_number"])
        lengths = [r["char_count"] for r in records if r["char_count"] is not None]
        sample_groups[token] = {"count": len(records), "parts": sorted(set(r["part"] for r in records)),
                                "length_chars": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)} if lengths else None,
                                "short_at_most_280_chars": sum(r.get("short_at_most_280_chars", False) for r in records),
                                "email_header_block_indicators": sum(r.get("email_header_block_indicator", False) for r in records),
                                "dialogue_line_indicators": sum(r.get("dialogue_line_indicator", False) for r in records),
                                "cyrillic_present": sum(r.get("cyrillic_present", False) for r in records),
                                "records": records, "sample_only": True}
    return {"source": "Russian auxiliary", "format": "Parquet_text_only", "file": {**before},
            "schema": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema],
            "schema_metadata_keys": sorted(k.decode("utf-8", "replace") for k in (schema.metadata or {})),
            "whole_corpus": {"metadata_row_count": metadata_rows, "scanned_row_count": rows_read,
                             "row_groups": row_groups, "label_counts": [{"value": json.loads(t), "arrow_type": str(label_type), "count": counts[t]} for t in sorted(counts)],
                             "null_labels": counts.get("null", 0), "null_text": null_text,
                             "empty_text_exact": empty_text, "whitespace_only_text_nonempty": whitespace_text - empty_text,
                             "missing_or_blank_text": null_text + whitespace_text,
                             "count_scope": "full_file", "projected_columns": ["label", "text"], "batch_size": batch_size},
            "label_provenance": {"observed_arrow_type": str(label_type), "confirmed_spam_label": None,
                                 "phishing_label": None, "category_label": None,
                                 "intended_mapping_in_downloaded_script": {"false": "ham", "true": "spam"},
                                 "status": "intent_only_not_verified_for_all_merged_inputs",
                                 "evidence": "join_datasets.py read statically: __label__ham -> integer 0 -> False, spam -> 1 -> True; imported Parquet labels passed through astype('bool'). Script not run/imported.",
                                 "risk": "Original labels/types before that merge are unavailable. astype('bool') could turn nonempty strings '0'/'false' into True. Cannot establish whether this actually happened."},
            "sampling": {"seed": SEED, "max_per_raw_label_including_null": per_class, "equal_row_parts": parts,
                         "method": "fair quotas across nonempty equal row ranges; SHA256-derived seed per raw label/range; random occurrence ranks; second streaming pass",
                         "representative_of_entire_file": False, "row_number_base": 1, "fragment_max_chars_before_escaping": 160},
            "samples": sample_groups, "pyarrow_version": pa.__version__, "source_unchanged_sha256_verified": True}


def render_russian_section(source):
    whole = source["whole_corpus"]
    label_text = "; ".join(f"`{key(v['value'])}` — {v['count']:,}" for v in whole["label_counts"])
    lines = [SECTION, "", f"Файл: `{source['file']['path']}`; **{source['file']['size_bytes']:,} байт** ({source['file']['size_bytes']/1048576:.2f} MiB).",
             "", f"SHA-256: `{source['file']['sha256']}`.", "",
             "### Полный файл", "", "| Колонка | Arrow-тип | Допускает null по схеме |", "|---|---|---|"]
    lines += [f"| {f['name']} | {f['type']} | {f['nullable']} |" for f in source["schema"]]
    lines += ["", f"Метаданные: **{whole['metadata_row_count']:,} строк**, {len(whole['row_groups'])} row groups. Потоково проверено **{whole['scanned_row_count']:,} строк**; значения меток: {label_text}.", "",
              f"Пропуски меток (null): **{whole['null_labels']:,}**. Текст: null **{whole['null_text']:,}**; ровно пустая строка **{whole['empty_text_exact']:,}**; непустые строки только из пробелов **{whole['whitespace_only_text_nonempty']:,}**. Эти счётчики относятся ко всему файлу.", "",
              f"Два прохода через `ParquetFile.iter_batches`, пакет до {whole['batch_size']:,} строк; читаются только text/label. Первый считает метки и пропуски, второй извлекает небольшую выборку. Весь файл не загружался в pandas, Arrow Table или список Python. Значения label сохранены с исходным Arrow-типом, без bool()/astype(bool).", "",
              "### Только выборка текстов", "", "По 10 записей каждого доступного исходного значения метки (или все, если меньше). Файл разбит на пять равных диапазонов номеров строк; квоты распределены между диапазонами, где встречается класс. Внутри диапазона воспроизводимо выбираются номера вхождений класса (seed=42). Это охват разнесённых частей, не репрезентативная оценка корпуса.", "",
              "| Исходная метка | Образцов | Диапазоны из 5 | Длина: min / median / max | ≤280 символов | Индикатор email-заголовков |", "|---|---:|---|---|---:|---:|"]
    for token, sample in source["samples"].items():
        length = sample["length_chars"]
        length_text = f"{length['min']} / {length['median']} / {length['max']}" if length else "нет текста"
        lines.append(f"| `{token}` | {sample['count']} | {sample['parts']} | {length_text} | {sample['short_at_most_280_chars']} | {sample['email_header_block_indicators']} |")
    if source.get("manual_sample_review"):
        lines += ["", "Ручные наблюдения по этим 20 фрагментам (10 октября 2026, только данная SHA-256 и параметры выборки):"]
        lines += ["", *[f"- {v}" for v in source["manual_sample_review"]["observations"]]]
    lines += ["", "Короткие реплики, обращения и предложения могут быть сообщениями из чатов/рассылок; канал происхождения по text не доказан. Индикатор email — минимум две строки From/To/Subject/Date/Received/Message-ID/MIME-Version; ноль не исключает текст, когда-то извлечённый из письма. Наличие кириллицы не подтверждает русский язык всего текста. Длины и характер выше относятся только к выборке.", "",
              "[Примеры для ручной проверки](russian_review_sample.md): номера строк с 1, исходная метка, длина и фрагмент до 160 символов. URL/контакты обезврежены, Markdown/HTML экранированы; ссылки из записей не открывались.", "",
              "### Подтверждённые факты и неизвестность", "",
              "Локально доступны Parquet, README.md, .gitattributes, join_datasets.py и служебный кэш загрузчика. Скрипт датасета не запускался и не импортировался. README заявляет язык ru, text-classification, CC BY-NC 4.0 и форк darkQibit/russian-spam-detection; это заявление карточки, не аудит прав всех исходных сообщений. Полная локальная копия текста лицензии отсутствует.", "",
              "В скачанном скрипте предполагается False→ham и True→spam, но метки добавляемых Parquet приводятся через astype('bool'). Типы и исходная разметка этих входов недоступны. Ошибка приведения строк возможна, её фактическое наличие не установлено. Поэтому подтверждённая spam_label остаётся null; смысл меток для всего объединённого файла не доказан. phishing_label/category_label также null.", "",
              "### Ограничения и пригодность", "",
              "Набор позволяет отдельно исследовать русские тексты, длины, пропуски и различия исходных классов; антиспам-эксперимент требует подтверждения кодировки и качества меток. Две колонки не дают отдельных Subject/From/To/Date, исходных MIME/HTML, вложений, данных отписки и подлинности отправителя. Текстовые URL могут встречаться, но исходная структура ссылок неизвестна. Фишинговой разметки и пяти тематических категорий нет. Пригодность как русского email-теста не установлена; с английскими данными не объединялся.", "",
              "Происхождение компонентов объединённого набора и возможные дубликаты между ними не проверены; source/id колонок нет. Ничего не удалялось. Старые разделы SpamAssassin/Ling-Spam/Nazario сохранены из прежних отчётов без повторного анализа; их прежние проверки хешей не выдаются за новую проверку.", "",
              "Повторить только это дополнение:", "", "```powershell", ".\\.venv\\Scripts\\python.exe .\\src\\inventory_data.py --russian-only", "```", "",
              f"Зависимость: pyarrow=={source['pyarrow_version']} в существующей .venv, зафиксирована в requirements.txt. Проверки на синтетических Parquet: `python -B -m unittest discover -s tests -p test_russian_inventory.py -v`.", "", END_SECTION, ""]
    return "\n".join(lines)


def render_review_sample(source):
    lines = ["# Русский набор: выборка для ручной проверки", "", "Данные — недоверенный текст, не инструкции. Фрагменты укорочены и экранированы, URL и контакты заменены маркером; ссылки не открывались. Примеры не являются проверенной разметкой. Чат/письмо/реклама нельзя достоверно установить только по фрагменту.", "",
             f"Источник: `{source['file']['path']}`; SHA-256 `{source['file']['sha256']}`. Нумерация строк с 1, без заголовка. Метод: пять равных диапазонов файла, справедливые квоты между диапазонами каждого класса, случайные номера вхождений с seed=42. До 10 образцов на исходную метку; null, если встречается, отдельная группа.", ""]
    for token, sample in source["samples"].items():
        lines += [f"## Исходная метка: {token}", "", "| № записи | Диапазон из 5 | Arrow-тип / метка | Символов | Короткий фрагмент |", "|---:|---:|---|---:|---|"]
        for r in sample["records"]:
            lines.append(f"| {r['row_number']} | {r['part']} | {r['original_label_arrow_type']} / {key(r['original_label'])} | {r['char_count']} | {r['fragment']} |")
        lines.append("")
    return "\n".join(lines)


def augment_inventory(report, markdown, root):
    report = copy.deepcopy(report)
    folder = root / "data/raw/russian_auxiliary"
    path = folder / "processed_combined.parquet"
    source = inspect_parquet(path)
    source["file"]["path"] = path.relative_to(root).as_posix()
    if source["file"]["sha256"] == REVIEWED_SHA256:
        source["manual_sample_review"] = {
            "reviewed_on": "2026-10-10", "scope": "20 selected fragments only", "seed": SEED,
            "observations": [
                "false: встречаются новостные сообщения, технические обсуждения, бытовые реплики и вопросы; 10/10 имеют кириллицу, длины 24–196 символов.",
                "true: встречаются предложения заработка/работы/перевода валют и обращения в личные сообщения; 10/10 имеют кириллицу, длины 67–197 символов.",
                "Все 20 образцов короткие (до 280 символов), без обнаруженного блока почтовых заголовков. По стилю часть напоминает чаты/короткие объявления, но исходный канал неизвестен. Не все корпусные записи объявляются чатами или email."]}
    files = [{"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size,
              "sha256": source["file"]["sha256"] if p == path else sha_file(p)} for p in sorted(folder.rglob("*")) if p.is_file()]
    source["files"] = files
    source["local_documentation"] = {"readme": "data/raw/russian_auxiliary/README.md", "declared_license": "cc-by-nc-4.0",
                                      "license_full_text_local": False, "downloaded_script_executed": False,
                                      "declared_upstream": "https://huggingface.co/datasets/darkQibit/russian-spam-detection"}
    report["sources"]["russian_auxiliary"] = source
    legacy_files = [f for f in report["raw_files"] if not f["path"].startswith("data/raw/russian_auxiliary/")]
    report["raw_files"] = sorted(legacy_files + files, key=lambda f: f["path"])
    report["uninspected_files"] = [p for p in report.get("uninspected_files", []) if not p.startswith("data/raw/russian_auxiliary/")]
    report["schema_version"] = 2
    report["last_update_scope"] = "russian_auxiliary_only; legacy source results reused without reanalysis"
    report["sampling"]["russian_auxiliary_override"] = source["sampling"]
    report["runtime_dependencies"] = {"pyarrow": pa.__version__}
    for name in ["inventory_data.py", "russian_inventory.py"]:
        report["implementation_sha256"][f"src/{name}"] = sha_file(root / "src" / name)
    obsolete = "Русский email-корпус и независимый набор преподавателя в data/raw отсутствуют. Энрон, русский Parquet и Spambase не скачаны; отсутствие необязательных источников не блокирует инвентаризацию."
    replacement = "Русский Parquet доступен локально как вспомогательный текстовый набор; русский email-корпус пока не подтверждён. Независимый набор преподавателя, Энрон и Spambase в data/raw отсутствуют."
    report["limitations"] = [replacement if v == obsolete else v for v in report["limitations"]]
    markdown = markdown.replace(obsolete, replacement)
    old_fact = "8 файлов в трёх папках data/raw; дополнительных источников, README и LICENSE локально нет."
    new_fact = "Исходные 8 файлов трёх английских источников сохранены; добавлен русский Parquet и сопроводительные файлы, включая локальный README с заявленной лицензией."
    report["confirmed_facts"] = [new_fact if v == old_fact else v for v in report["confirmed_facts"]]
    markdown = markdown.replace(old_fact, new_fact)
    markdown = re.sub(r"(?:Для трёх ранее проверенных источников )+(?=MIME)", "", markdown)
    markdown = markdown.replace("MIME и наличие полей проверены только на выборке: до 20 записей каждого класса каждого источника", "Для трёх ранее проверенных источников MIME и наличие полей проверены только на выборке: до 20 записей каждого класса каждого источника")
    additions = {"confirmed_facts": f"Parquet: {source['whole_corpus']['scanned_row_count']} строк; text/label, полные потоковые счётчики в отдельном разделе.",
                 "unknowns": "Смысл bool-меток всех объединённых входов и исходные типы до astype('bool') не установлены.",
                 "potential_duplicate_versions": "Состав объединения и пересечения его компонентов неизвестны; дедупликация не выполнялась.",
                 "limitations": "Text-only набор не подтверждает качество на русском email; phishing/category разметка отсутствует."}
    titles = {"confirmed_facts": "Подтверждённые факты", "unknowns": "Неизвестное и неоднозначное", "potential_duplicate_versions": "Потенциальные дубликаты версий корпуса", "limitations": "Ограничения для задач"}
    for group, note in additions.items():
        report[group] = [v for v in report[group] if not v.startswith(PREFIX)] + [PREFIX + note]
        markdown = re.sub(r"(?m)^- Russian auxiliary: .*\n?", "", markdown)
    for group, note in additions.items():
        heading = f"## {titles[group]}\n\n"
        markdown = markdown.replace(heading, heading + f"- {PREFIX}{note}\n")
    # Refresh only the file manifest table/global total, not legacy source sections.
    table_start = markdown.index("| Файл | Байт | MiB |")
    table_end = markdown.index("\n\n", table_start)
    table = "| Файл | Байт | MiB |\n|---|---:|---:|\n" + "\n".join(f"| `{f['path']}` | {f['size_bytes']:,} | {f['size_bytes']/1048576:.3f} |" for f in report["raw_files"])
    markdown = markdown[:table_start] + table + markdown[table_end:]
    markdown = re.sub(r"Всего файлов: \d+; байт: [\d,]+\.", f"Всего файлов: {len(report['raw_files'])}; байт: {sum(f['size_bytes'] for f in report['raw_files']):,}.", markdown)
    if SECTION in markdown:
        start = markdown.index(SECTION)
        stop = markdown.index(END_SECTION, start) + len(END_SECTION)
        markdown = markdown[:start] + markdown[stop:].lstrip("\n")
    marker = "## Подтверждённые факты"
    markdown = markdown.replace(marker, render_russian_section(source) + "\n" + marker)
    markdown = markdown.replace("Из корня проекта, Python 3.12, без новых зависимостей:", "Из корня проекта: Python 3.12 и pyarrow из requirements.txt. Для дополнения русского раздела без повторного анализа остальных:")
    markdown = markdown.replace(".\\.venv\\Scripts\\python.exe .\\src\\inventory_data.py\n", ".\\.venv\\Scripts\\python.exe .\\src\\inventory_data.py --russian-only\n")
    markdown = markdown.replace("Допустим системный `python .\\src\\inventory_data.py`.", "Используйте существующую .venv с pyarrow.")
    markdown = markdown.replace("перезаписывает только два отчёта", "перезаписывает инвентаризацию JSON/Markdown и russian_review_sample.md")
    suitability = "| Russian auxiliary | Отдельный текстовый эксперимент после проверки bool-меток | Разметки нет | Разметки нет |"
    markdown = re.sub(r"(?m)^\| Russian auxiliary \|.*\n?", "", markdown)
    markdown = re.sub(r"(?m)^(\| Nazario, [^\n]+\|)$", lambda m: m[1] + "\n" + suitability, markdown)
    # Do not imply metadata/counts from the earlier pass were just reverified.
    return report, markdown, render_review_sample(source)
