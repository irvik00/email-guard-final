"""Record construction, narrow model-text cleaning and dependency-free validation."""
import hashlib
import json
import re

SCHEMA_VERSION = 2
FILTER_HEADERS = {
    "x-spam-status", "x-spam-flag", "x-spam-level", "x-spam-score", "x-spam-report",
    "x-spam-tests", "x-spam-checker-version", "x-spam-prev-status", "x-spamd-result",
    "x-rspamd-score", "x-rspamd-action", "x-bogosity",
}
FILTER_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):[^\n]*$")
POSSIBLE_MARKER = re.compile(r"^\s*(?:\*{3,}\s*SPAM\s*\*{3,}|\[(?:SPAM|JUNK|BULK)(?:\?|:[^\]]*)?\])", re.I)
STRUCTURAL = ["sender", "sender_address", "sender_domain", "recipient", "links",
              "attachments", "unsubscribe", "formatting"]
AXES = ["spam", "phishing", "category"]
STATUSES = {"source_provided", "mapping_inferred", "source_intended_unverified", "unlabeled"}


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def make_model_text(subject, body):
    """Keep original fields; remove only explicit allowlisted filter header lines."""
    parts, changes = [], []
    for field, original in [("subject", subject), ("body_text", body)]:
        if original is None:
            continue
        text = original.replace("\r\n", "\n").replace("\r", "\n")
        if text != original:
            changes.append({"operation": "normalize_line_endings", "field": field})
        kept, removed, names = [], [], set()
        in_filter = False
        for number, line in enumerate(text.split("\n"), 1):
            match = FILTER_LINE.fullmatch(line)
            if match and match[1].lower() in FILTER_HEADERS:
                in_filter = True
                removed.append(number)
                names.add(match[1].lower())
            elif in_filter and line.startswith((" ", "\t")) and line.strip():
                removed.append(number)
            else:
                in_filter = False
                kept.append(line)
        if removed:
            changes.append({"operation": "remove_filter_header_lines", "field": field,
                            "line_numbers": removed, "header_names": sorted(names)})
        text = "\n".join(kept)
        stripped = text.strip()
        if stripped != text:
            changes.append({"operation": "trim_outer_whitespace", "field": field})
        if stripped:
            parts.append(stripped)
    return "\n\n".join(parts), changes


def labels_for(source, original_label, subset):
    spam = phishing = None
    status = dict.fromkeys(AXES, "unlabeled")
    provenance = {"spam": None, "phishing": None, "category": None}
    if source == "spamassassin":
        if subset in {"easy_ham", "easy_ham_2", "hard_ham"}:
            spam = 0
        elif subset in {"spam", "spam_2"}:
            spam = 1
        else:
            raise ValueError("Unknown SpamAssassin subset")
        status["spam"] = "source_provided"
        provenance["spam"] = {"basis": "archive class folder", "original_folder": subset,
                              "rule": "easy_ham/easy_ham_2/hard_ham=0; spam/spam_2=1"}
    elif source == "lingspam":
        if type(original_label) is not str or original_label not in {"0", "1"}:
            raise ValueError("Unrecognized Ling-Spam label")
        spam = {"0": 0, "1": 1}[original_label]
        status["spam"] = "mapping_inferred"
        provenance["spam"] = {"basis": "DAY2_STEP2 working mapping; publisher counts and reviewed sample",
                              "rule": "CSV string 0=ham, 1=spam", "author_codebook_found": False}
    elif source == "nazario":
        phishing = 1
        status["phishing"] = "source_provided"
        provenance["phishing"] = {"basis": "author hand-classified phishing mailbox; internal metadata excluded",
                                  "rule": "source-level positive phishing; no independent spam label"}
    elif source == "untroubled":
        if original_label != "spam":
            raise ValueError("Untroubled original label must be source-level spam")
        spam = 1
        status["spam"] = "source_provided"
        provenance["spam"] = {"basis": "Untroubled SPAM Archive source-level positive label",
                              "rule": "spam archive=1; phishing remains unknown",
                              "source_url": "https://untroubled.org/spam/"}
    elif source == "russian_auxiliary":
        if type(original_label) is not bool:
            raise ValueError("Russian label must be native bool")
        spam = 1 if original_label is True else 0
        status["spam"] = "source_intended_unverified"
        provenance["spam"] = {"basis": "DAY2_STEP2 accepted author intent for auxiliary experiment",
                              "rule": "native False=ham(0), True=spam(1)",
                              "limitation": "Possible historical astype(bool) problem not established; original merged input types unknown"}
    else:
        raise ValueError("Unknown source")
    return {"spam_label": spam, "phishing_label": phishing, "category_label": None,
            "label_status": status, "label_provenance": provenance}


def identity(source, file_path, file_hash, locator, position):
    return source + ":" + sha_bytes(canonical_bytes([source, file_path, file_hash, locator, position]))


def base_record(row, file_info):
    return {"schema_version": SCHEMA_VERSION,
            "record_id": identity(row.source, file_info["path"], file_info["sha256"], row.locator, row.position),
            "source": row.source, "source_version": "sha256:" + file_info["sha256"],
            "source_file_sha256": file_info["sha256"], "original_locator": row.locator,
            "source_path": file_info["path"], "collection_period": None,
            "original_position": row.position, "subset": row.subset,
            "input_kind": row.input_kind, "domain_role": "auxiliary" if row.source == "russian_auxiliary" else "email",
            "original_label": row.original_label}


def build_record(row, file_info, parsed=None):
    record = base_record(row, file_info)
    record.update(labels_for(row.source, row.original_label, row.subset))
    if parsed is not None:
        record.update({k: parsed[k] for k in ["subject", "body_text", "body_source", *STRUCTURAL]})
        record["feature_availability"] = {k: "observable" for k in
                                          ["subject", "body_text", "headers", "mime", "html", "links", "attachments", "unsubscribe", "formatting"]}
        record["parse_warnings"] = parsed["parse_warnings"]
        record["raw_sha256"] = sha_bytes(row.raw)
        record["record_content_sha256"] = record["raw_sha256"]
    else:
        record.update(dict.fromkeys(STRUCTURAL))
        record.update(subject=row.fields.get("subject"), body_text=row.fields["message"] if row.source == "lingspam" else row.fields["text"], body_source="dataset_text")
        record["feature_availability"] = {k: "unavailable" for k in
                                          ["subject", "body_text", "headers", "mime", "html", "links", "attachments", "unsubscribe", "formatting"]}
        record["feature_availability"]["body_text"] = "observable"
        if row.source == "lingspam":
            record["feature_availability"]["subject"] = "observable"
        record["raw_sha256"] = None
        record["record_content_sha256"] = sha_bytes(canonical_bytes(row.fields))
        record["parse_warnings"] = []
    record["model_text"], record["transformations"] = make_model_text(record["subject"], record["body_text"])
    record["model_text_sha256"] = sha_bytes(record["model_text"].encode("utf-8"))
    record["possible_filter_marker"] = bool(POSSIBLE_MARKER.search(record["subject"] or ""))
    record["usable_model_text"] = bool(record["model_text"])
    record["parse_status"] = "partial" if record["parse_warnings"] or not record["usable_model_text"] else "ok"
    validate_record(record)
    return record


def validate_record(record):
    required = {"record_id", "source", "source_version", "source_file_sha256", "original_locator", "subset", "input_kind",
                "domain_role", "original_label", "spam_label", "phishing_label", "category_label", "label_status", "label_provenance",
                "subject", "body_text", "body_source", *STRUCTURAL, "feature_availability", "model_text", "raw_sha256",
                "record_content_sha256", "model_text_sha256", "parse_status", "parse_warnings", "usable_model_text", "transformations", "possible_filter_marker"}
    if missing := required - record.keys():
        raise ValueError(f"Missing schema fields: {sorted(missing)}")
    if record.get("schema_version") == 2:
        if type(record.get("source_path")) is not str or "collection_period" not in record:
            raise ValueError("Schema 2 requires relative source_path and collection_period")
        if record["collection_period"] is not None and type(record["collection_period"]) is not str:
            raise TypeError("collection_period")
    for field in ["record_id", "source", "source_version", "original_locator", "model_text"]:
        if type(record[field]) is not str:
            raise TypeError(field)
    for field in ["subject", "body_text", "body_source", "sender", "sender_address", "sender_domain", "recipient", "subset"]:
        if record[field] is not None and type(record[field]) is not str:
            raise TypeError(field)
    for field in ["spam_label", "phishing_label", "category_label"]:
        value = record[field]
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError(field)
    if record["category_label"] is not None:
        raise ValueError("No category labels assigned in step 2")
    for axis in AXES:
        if record["label_status"].get(axis) not in STATUSES:
            raise ValueError("Unknown label status")
        if (record[axis + "_label"] is None) != (record["label_status"][axis] == "unlabeled"):
            raise ValueError("Unknown label/status mismatch")
    if record["input_kind"] not in {"raw_email", "mbox_email", "text_only", "text_only_preprocessed"}:
        raise ValueError("Unknown input kind")
    if record["domain_role"] not in {"email", "auxiliary"}:
        raise ValueError("Unknown domain role")
    if record["parse_status"] not in {"ok", "partial", "failed"} or type(record["usable_model_text"]) is not bool:
        raise ValueError("Parse status")
    if record["parse_status"] == "failed" and record["usable_model_text"]:
        raise ValueError("Failed record cannot be usable")
    if record["usable_model_text"] != bool(record["model_text"]):
        raise ValueError("Text usability mismatch")
    for field in ["source_file_sha256", "record_content_sha256", "raw_sha256", "model_text_sha256"]:
        value = record[field]
        if value is None and field == "raw_sha256":
            continue
        if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Invalid SHA-256")
    for field in ["links", "attachments"]:
        if record[field] is not None and type(record[field]) is not list:
            raise TypeError(field)
    for field in ["unsubscribe", "formatting"]:
        if record[field] is not None and type(record[field]) is not dict:
            raise TypeError(field)
    if type(record["parse_warnings"]) is not list or any(type(w) is not str for w in record["parse_warnings"]):
        raise TypeError("Warnings must be strings")
    if any(v not in {"observable", "unavailable"} for v in record["feature_availability"].values()):
        raise ValueError("Availability")
    if record["input_kind"].startswith("text_only"):
        if any(record[k] is not None for k in STRUCTURAL) or record["raw_sha256"] is not None:
            raise ValueError("Synthetic text-only metadata")
        if record["feature_availability"]["headers"] != "unavailable":
            raise ValueError("Text-only headers unavailable")
    elif record["raw_sha256"] is None:
        raise ValueError("Raw email requires original-byte hash")
    # Cross-check labels against the explicit adapter rule; bool is not accepted as int.
    if any(record[k] != v for k, v in labels_for(record["source"], record["original_label"], record["subset"]).items()):
        raise ValueError("Source label rule mismatch")
    if record["model_text"] != make_model_text(record["subject"], record["body_text"])[0]:
        raise ValueError("Unexpected model text")
    if record["model_text_sha256"] != sha_bytes(record["model_text"].encode("utf-8")):
        raise ValueError("Model text hash mismatch")
