"""Read-only adapters for the four known formats; no archive extraction."""
import csv
import random
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from email import policy
from email.parser import BytesHeaderParser
from pathlib import Path

from inventory_data import SA_LABELS, MESSAGE_NAME

SA_FILES = ["20030228_easy_ham.tar.bz2", "20030228_easy_ham_2.tar.bz2",
            "20030228_hard_ham.tar.bz2", "20030228_spam.tar.bz2", "20050311_spam_2.tar.bz2"]
SOURCE_FILES = {
    "spamassassin": ["spamassassin/" + p for p in SA_FILES],
    "lingspam": ["lingspam/messages.csv"],
    "nazario": ["nazario/phishing-2024.txt", "nazario/phishing-2025.txt"],
    "russian_auxiliary": ["russian_auxiliary/processed_combined.parquet"],
}


@dataclass
class InputRow:
    source: str
    path: Path
    locator: str
    position: int
    subset: str | None
    input_kind: str
    original_label: object = None
    raw: bytes | None = None
    fields: dict | None = None
    size_bytes: int | None = None
    excluded_reason: str | None = None
    bounds: dict | None = None


def project_path(path, root):
    return path.resolve().relative_to(root.resolve()).as_posix()


def iter_spamassassin(paths, root, max_bytes, trial_limit=None):
    for path in paths:
        accepted = 0
        with tarfile.open(path, "r:*") as archive:
            for position, member in enumerate(archive, 1):
                if member.isdir():
                    continue
                name_parts = member.name.split("/")
                is_email = (member.isfile() and len(name_parts) == 2 and name_parts[0] in SA_LABELS
                            and bool(MESSAGE_NAME.fullmatch(name_parts[1])))
                row = InputRow("spamassassin", path, f"{project_path(path, root)}::{member.name}", position,
                               name_parts[0] if is_email else None, "raw_email", size_bytes=member.size)
                if not is_email:
                    row.excluded_reason = "service_file" if member.isfile() else "non_regular_archive_member"
                    yield row
                    continue
                if trial_limit is not None and accepted >= trial_limit:
                    continue
                accepted += 1
                row.original_label = row.subset
                if member.size > max_bytes:
                    row.excluded_reason = "raw_record_too_large"
                else:
                    with archive.extractfile(member) as handle:
                        row.raw = handle.read(max_bytes + 1)
                    if len(row.raw) != member.size:
                        raise ValueError("Truncated tar member")
                yield row


def iter_lingspam(paths, root, trial_limit=None):
    for path in paths:
        with path.open(encoding="utf-8-sig", errors="strict", newline="") as handle:
            # csv defaults to 128 KiB/field; permit long legitimate text fields.
            # This changes only a parser limit, not how much data we load at once.
            csv.field_size_limit(max(csv.field_size_limit(), path.stat().st_size))
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != ["subject", "message", "label"]:
                raise ValueError("Ling-Spam schema differs from inventory")
            for position, fields in enumerate(reader, 1):
                if trial_limit is not None and position > trial_limit:
                    break
                yield InputRow("lingspam", path, f"{project_path(path, root)}::data_row={position}", position,
                               None, "text_only_preprocessed", original_label=fields.get("label"), fields=fields)


def mbox_bounds(path):
    """Identical delimiter convention to inventory; retain exact on-disk bytes."""
    starts, message_starts = [], []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith(b"From "):
                starts.append(offset)
                message_starts.append(handle.tell())
    if not starts or starts[0] != 0:
        raise ValueError("Expected mbox with From delimiter at byte zero")
    file_size = path.stat().st_size
    return [(start, message_starts[i], starts[i + 1] if i + 1 < len(starts) else file_size)
            for i, start in enumerate(starts)]


def is_internal_metadata(raw):
    headers = BytesHeaderParser(policy=policy.default).parsebytes(raw)
    return ("FOLDER INTERNAL DATA" in str(headers.get("Subject", "")).upper()
            and "MAIL SYSTEM INTERNAL DATA" in str(headers.get("From", "")).upper()
            and "X-IMAP" in headers)


def iter_nazario(paths, root, max_bytes, trial_limit=None):
    for path in paths:
        bounds = mbox_bounds(path)
        with path.open("rb") as handle:
            for position, (separator, start, stop) in enumerate(bounds, 1):
                if trial_limit is not None and position > trial_limit:
                    break
                row = InputRow("nazario", path, f"{project_path(path, root)}::mbox_record={position}@byte={separator}",
                               position, path.stem, "mbox_email", size_bytes=stop - start,
                               bounds={"separator_byte_offset": separator, "message_byte_start": start, "message_byte_stop_exclusive": stop})
                if row.size_bytes > max_bytes:
                    row.excluded_reason = "raw_record_too_large"
                else:
                    handle.seek(start)
                    row.raw = handle.read(stop - start)
                    if len(row.raw) != stop - start:
                        raise ValueError("Truncated mbox record")
                yield row


def sample_russian(path, root, per_class, seed, on_invalid, trial_scan_limit=None, progress=None):
    """Algorithm R per native bool class. Keep at most 2*per_class rows in memory."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    reservoirs = {False: [], True: []}
    generators = {False: random.Random(seed), True: random.Random(seed + 1)}
    eligible = Counter({False: 0, True: 0})
    counts = Counter({False: 0, True: 0})
    scanned = null_labels = null_text = blank_text = 0
    file_name = project_path(path, root)
    with pq.ParquetFile(path) as parquet:
        schema = parquet.schema_arrow
        if schema.names != ["text", "label"] or not pa.types.is_boolean(schema.field("label").type) or not pa.types.is_string(schema.field("text").type):
            raise ValueError("Russian Parquet schema differs: expected text:string, label:bool")
        metadata_rows = parquet.metadata.num_rows
        for batch in parquet.iter_batches(batch_size=16384, columns=["text", "label"], use_threads=False):
            if trial_scan_limit is not None:
                remaining = trial_scan_limit - scanned
                if remaining <= 0:
                    break
                batch = batch.slice(0, min(batch.num_rows, remaining))
            texts, labels = batch.column("text"), batch.column("label")
            # These Python lists are bounded by one 16384-row batch, not the file.
            label_values = labels.to_pylist()
            valid_text = pc.fill_null(pc.not_equal(pc.utf8_trim_whitespace(texts), ""), False).to_pylist()
            null_text += texts.null_count
            blank_text += batch.num_rows - sum(valid_text) - texts.null_count
            for offset, (label, good_text) in enumerate(zip(label_values, valid_text)):
                position = scanned + offset + 1
                if label is None:
                    null_labels += 1
                elif type(label) is bool:
                    counts[label] += 1
                if type(label) is not bool or not good_text:
                    row = InputRow("russian_auxiliary", path, f"{file_name}::row={position}", position, None,
                                   "text_only", original_label=label,
                                   excluded_reason="missing_or_invalid_label" if type(label) is not bool else "missing_or_blank_text")
                    on_invalid(row)
                    continue
                eligible[label] += 1
                reservoir = reservoirs[label]
                index = len(reservoir) if len(reservoir) < per_class else generators[label].randrange(eligible[label])
                if index < per_class:
                    text = texts[offset].as_py()
                    row = InputRow("russian_auxiliary", path, f"{file_name}::row={position}", position, None,
                                   "text_only", original_label=label, fields={"text": text, "label": label})
                    if index == len(reservoir):
                        reservoir.append(row)
                    else:
                        reservoir[index] = row
            scanned += batch.num_rows
            if progress and scanned // 1000000 != (scanned - batch.num_rows) // 1000000:
                progress(scanned)
    if trial_scan_limit is None and scanned != metadata_rows:
        raise ValueError("Parquet row count mismatch")
    selected = sorted(reservoirs[False] + reservoirs[True], key=lambda r: r.position)
    metrics = {"metadata_rows": metadata_rows, "scanned_rows": scanned, "scan_complete": scanned == metadata_rows,
               "native_label_counts": {"false": counts[False], "true": counts[True], "null": null_labels},
               "null_text": null_text, "blank_text_nonnull": blank_text,
               "eligible_per_class": {"false": eligible[False], "true": eligible[True]},
               "selected_per_class": {"false": len(reservoirs[False]), "true": len(reservoirs[True])},
               "shortfall_per_class": {"false": max(0, per_class - eligible[False]), "true": max(0, per_class - eligible[True])},
               "eligible_not_selected": sum(eligible.values()) - len(selected),
               "sampling": {"algorithm": "reservoir Algorithm R separately per native bool class", "seed": seed,
                            "rng_seed_false": seed, "rng_seed_true": seed + 1, "max_per_class": per_class,
                            "batch_size": 16384, "read_order": "Parquet physical row order", "output_order": "ascending original row number",
                            "selected_original_row_numbers": [r.position for r in selected],
                            "balanced_sample_is_not_source_prevalence": True}}
    return selected, metrics
