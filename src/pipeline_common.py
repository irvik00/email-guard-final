"""Small shared IO/provenance helpers; no data-dependent fitting or network IO."""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from inventory_data import sha_file
from normalization_schema import canonical_bytes, sha_bytes, validate_record

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"spamassassin": "spamassassin.jsonl", "lingspam": "lingspam.jsonl", "nazario": "nazario.jsonl",
         "russian_auxiliary": "russian_auxiliary_sample.jsonl", "untroubled": "untroubled.jsonl"}


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("wb")
        self.digest = hashlib.sha256()
        self.count = self.size = 0

    def write(self, record):
        line = canonical_bytes(record) + b"\n"
        self.handle.write(line)
        self.digest.update(line)
        self.count += 1
        self.size += len(line)

    def close(self):
        self.handle.close()
        return {"sha256": self.digest.hexdigest(), "rows": self.count, "size_bytes": self.size}


def compare_text(text):
    import unicodedata
    text = unicodedata.normalize("NFC", text).casefold()
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def stable_order(seed, purpose, value):
    return sha_bytes(canonical_bytes([seed, purpose, value]))


def enrich(record, date_value=None):
    """Schema-1 parsed cache -> schema 2, preserving IDs, original/model texts and hashes."""
    record["schema_version"] = 2
    record["source_path"] = record["original_locator"].split("::", 1)[0]
    record.setdefault("collection_period", None)
    record.setdefault("collection_period_provenance", "unknown")
    if record["source"] == "nazario":
        match = re.search(r"phishing-(\d{4})\.txt$", record["source_path"])
        if match:
            record["collection_period"] = match[1]
            record["collection_period_provenance"] = "source_filename_declared_year; not authenticated message date"
    record["declared_date"] = date_value
    record["declared_date_iso"] = None
    record["date_provenance"] = "email_Date_header_unverified" if date_value is not None else "unavailable_or_missing"
    if date_value:
        try:
            record["declared_date_iso"] = parsedate_to_datetime(date_value).isoformat()
        except (ValueError, TypeError, OverflowError):
            record["date_provenance"] = "email_Date_header_unverified_unparseable"
    record.setdefault("date_disagreements", [])
    validate_record(record)
    return record


def set_member_period(record, member):
    parts = member.split("/")
    directory_period = "/".join(parts[:2]).replace("/", "-") if len(parts) >= 3 else None
    record["archive_directory_period"] = directory_period
    record["archive_declared_period"] = Path(record["source_path"]).stem
    stamp = re.match(r"^(\d{9,11})\.", parts[-1])
    try:
        moment = datetime.fromtimestamp(int(stamp[1]), timezone.utc) if stamp else None
    except (ValueError, OverflowError, OSError):
        moment = None
    record["collection_timestamp"] = moment.isoformat() if moment else None
    record["collection_period"] = moment.strftime("%Y-%m") if moment else None
    record["collection_period_provenance"] = "archive_member_Unix_timestamp_UTC_unverified_collection_time" if moment else "unknown"
    if moment and directory_period and record["collection_period"] != directory_period:
        record["date_disagreements"].append("timestamp_vs_archive_directory_month")
    if moment and moment.strftime("%Y") != record["archive_declared_period"]:
        record["date_disagreements"].append("timestamp_outside_archive_declared_year")
    if moment and record["declared_date_iso"]:
        header_date = datetime.fromisoformat(record["declared_date_iso"])
        if header_date.tzinfo is None:
            record["date_disagreements"].append("Date_header_timezone_unknown")
        elif header_date.astimezone(timezone.utc).strftime("%Y-%m") != record["collection_period"]:
            record["date_disagreements"].append("unverified_Date_header_vs_collection_month")
    return record


def fingerprint_code():
    names = ["read_email.py", "inventory_data.py", "russian_inventory.py", "normalization_schema.py",
             "normalization_adapters.py", "normalization_reporting.py", "prepare_data.py", "untroubled_adapter.py",
             "pipeline_normalization.py", "pipeline_common.py", "data_quality.py", "split_data.py", "pipeline_audit.py",
             "pipeline_reporting.py", "data_pipeline.py"]
    return {"src/" + n: sha_file(ROOT / "src" / n) for n in names if (ROOT / "src" / n).is_file()}


def publish_tree(stage, destination, manifest_name=None):
    """Replace only our generated files; never recursively delete user directories."""
    stage, destination = Path(stage), Path(destination)
    paths = sorted(p for p in stage.rglob("*") if p.is_file() and (not manifest_name or p.relative_to(stage).as_posix() != manifest_name))
    for path in paths:
        target = destination / path.relative_to(stage)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)
    if manifest_name:
        target = destination / manifest_name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage / manifest_name, target)
