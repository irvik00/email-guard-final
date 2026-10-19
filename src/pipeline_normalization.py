"""Reuse verified parsed data; migrate schema and add the modern source safely."""
import json
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path

from inventory_data import sha_file
from normalization_adapters import SOURCE_FILES
from normalization_reporting import ReviewSamples, new_stats, count_written, render_report
from normalization_schema import build_record, base_record, sha_bytes, canonical_bytes, validate_record, FILTER_HEADERS
from prepare_data import Config, run as legacy_normalize, verify_outputs
from pipeline_common import NAMES, JsonlWriter, enrich, set_member_period, rows, write_json, write_text, publish_tree, ROOT
from read_email import parse_email
from untroubled_adapter import process_archive

LEGACY_SCHEMA_HASH = "9da69554d6d3e5908e2b11f3f0f889bca1cede58e7215aed3e589599ffd85fd4"
LEGACY_PREPARE_HASH = "9efbcb62a4b5355e2b2108cd04ad8071013d80f644e82061948f9056e6c5c557"
NORMALIZATION_CODE = ["read_email.py", "inventory_data.py", "russian_inventory.py", "normalization_schema.py",
                      "normalization_adapters.py", "normalization_reporting.py", "prepare_data.py",
                      "pipeline_normalization.py", "pipeline_common.py", "untroubled_adapter.py"]


def source_fingerprints(root, config):
    raw = root / config["raw_dir"]
    paths = [raw / relative for names in SOURCE_FILES.values() for relative in names]
    paths.append(raw / config["archive"]["path"])
    result, states = [], {}
    for path in paths:
        if path.is_file():
            stat = path.stat()
            result.append({"path": path.relative_to(root).as_posix(), "size_bytes": stat.st_size, "sha256": sha_file(path)})
            states[path] = (stat.st_size, stat.st_mtime_ns)
    return result, states


def cache_key(config, fingerprints):
    # Splitting/near settings do not invalidate parsing or Russian sampling.
    settings = {k: config[k] for k in ["schema_version", "seed", "normalization", "archive", "trial"]}
    code = {"src/" + n: sha_file(ROOT / "src" / n) for n in NORMALIZATION_CODE}
    versions={name:importlib.metadata.version(name) for name in ["pyarrow","py7zr"]}
    return sha_bytes(canonical_bytes({"settings": settings, "inputs": fingerprints, "code": code,
                                     "dependency_versions":versions,"python_version":sys.version})), code


def verify_normalized(directory):
    ids, raw_ids, diagnostics = set(), set(), set()
    artifacts, counts = {}, {}
    for source, name in NAMES.items():
        path, count = Path(directory) / name, 0
        if not path.is_file():
            raise ValueError("Missing normalized source output: " + name)
        for record in rows(path):
            validate_record(record)
            if record["schema_version"] != 2 or record["source"] != source or record["record_id"] in ids:
                raise ValueError("Invalid normalized schema/source/ID")
            if any(k in record for k in ["headers_raw", "html", "plain_text", "html_text"]):
                raise ValueError("Diagnostics leaked into main records")
            ids.add(record["record_id"])
            if record["raw_sha256"]:
                raw_ids.add(record["record_id"])
            else:
                fields = ({"subject": record["subject"], "message": record["body_text"], "label": record["original_label"]}
                          if source == "lingspam" else {"text": record["body_text"], "label": record["original_label"]})
                if sha_bytes(canonical_bytes(fields)) != record["record_content_sha256"]:
                    raise ValueError("Tabular content hash mismatch")
            count += 1
        counts[source] = count
        artifacts[name] = {"sha256": sha_file(path), "size_bytes": path.stat().st_size, "rows": count}
    for name in ["diagnostics.jsonl", "excluded_records.jsonl"]:
        path, count = Path(directory) / name, 0
        for record in rows(path):
            if name == "diagnostics.jsonl":
                if record["record_id"] in diagnostics:
                    raise ValueError("Repeated diagnostics ID")
                diagnostics.add(record["record_id"])
            elif record["record_id"] in ids or not record.get("reason"):
                raise ValueError("Invalid normalization exclusion")
            count += 1
        artifacts[name] = {"sha256": sha_file(path), "size_bytes": path.stat().st_size, "rows": count}
    if raw_ids != diagnostics:
        raise ValueError("Normalized diagnostics linkage mismatch")
    return {"status": "passed", "written_counts": counts, "unique_record_ids": len(ids), "linked_diagnostics": len(diagnostics), "artifacts": artifacts}


def ensure_normalized(root, config, fingerprints, source_states, force=False):
    interim, reports = root / config["interim_dir"], root / config["reports_dir"]
    interim.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    key, code = cache_key(config, fingerprints)
    report_path = reports / "normalization_report.json"
    old = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    if not force and old and old.get("pipeline_cache_key") == key and old.get("status") == "complete":
        verified = verify_normalized(interim)
        if verified != old["verification"] or sha_file(reports / "normalization_errors.jsonl") != old["normalization_errors_sha256"]:
            raise ValueError("Normalized cache differs from its manifest; use --force-normalize to rebuild")
        print("normalization: verified schema-2 cache reused", flush=True)
        return old, {"mode": "verified_schema2_cache", "russian_parquet_rescanned": False}

    reusable = not force and old and old.get("schema_version") == 1 and old.get("status") == "complete" and not old.get("source_errors")
    if reusable:
        current_by_path = {x["path"]: x["sha256"] for x in fingerprints}
        reusable = all(current_by_path.get(x["path"]) == x["sha256"] for x in old["source_files"])
        reusable = reusable and all(old.get("config", {}).get(k) == v for k, v in
                                    {"seed": config["seed"], "russian_max_per_class": config["normalization"]["russian_max_per_class"],
                                     "max_record_bytes": config["normalization"]["max_record_bytes"], "trial": config["trial"],
                                     "trial_limit": config["normalization"]["trial_limit"]}.items())
        reusable = reusable and all(digest == code.get(name) or (name == "src/normalization_schema.py" and digest == LEGACY_SCHEMA_HASH)
                                    or (name == "src/prepare_data.py" and digest == LEGACY_PREPARE_HASH)
                                    or name == "src/normalization_reporting.py" for name, digest in old.get("code_sha256", {}).items())
        # Reporting-only changes do not affect parsed records. Verify all bytes before migration.
        if reusable and verify_outputs(interim) != old["verification"]:
            raise ValueError("Legacy cache output hash mismatch")
    with tempfile.TemporaryDirectory(prefix=".normalization-v2-", suffix=".tmp", dir=interim) as work_name:
        work = Path(work_name)
        if not work.resolve().is_relative_to(interim.resolve()):
            raise ValueError("Unsafe normalization temporary workspace")
        if reusable:
            base, base_reports, base_report = interim, reports, old
            mode = "verified_schema1_cache_migrated; texts/IDs/original hashes preserved"
            print("normalization: verified previous four-source outputs; no Parquet rescan", flush=True)
        else:
            base, base_reports = work / "base", work / "base_reports"
            base_report = legacy_normalize(Config(project_root=root, raw_dir=root / config["raw_dir"], output_dir=base,
                reports_dir=base_reports, seed=config["seed"], russian_max_per_class=config["normalization"]["russian_max_per_class"],
                max_record_bytes=config["normalization"]["max_record_bytes"], trial=config["trial"], trial_limit=config["normalization"]["trial_limit"]),
                precomputed_files={(root/x["path"]).resolve():x["sha256"] for x in fingerprints})
            if base_report["source_errors"]:
                raise ValueError("Base normalization source failed; inspect staged reports")
            mode = "base_normalization_rebuilt"
        stage = work / "ready"
        stage.mkdir()
        stats = {s: new_stats(s) for s in NAMES}
        writers = {name: JsonlWriter(stage / name) for name in [*NAMES.values(), "diagnostics.jsonl", "excluded_records.jsonl", "normalization_errors.jsonl"]}
        dates = {}
        review = ReviewSamples(config["seed"])
        archive_inventory = None
        record_errors = base_report["record_error_count"]
        source_errors = []
        try:
            for diagnostic in rows(base / "diagnostics.jsonl"):
                dates[diagnostic["record_id"]] = next((h["value"] for h in diagnostic["headers_raw"] if h["name"].lower() == "date"), None)
                writers["diagnostics.jsonl"].write(diagnostic)
            for source, name in NAMES.items():
                if source == "untroubled":
                    continue
                for record in rows(base / name):
                    enrich(record, dates.get(record["record_id"]))
                    writers[name].write(record)
                    count_written(stats[source], record)
                    review.add(record)
                previous = base_report["sources"][source]
                for k in ["input_records", "input_email_candidates", "original_label_counts", "excluded_by_reason", "filter_headers_in_diagnostics", "missing_files", "status", "russian_scan"]:
                    if k in previous:
                        stats[source][k] = previous[k]
                stats[source]["parse_status_counts"]["failed"] = previous["parse_status_counts"]["failed"]
            for name in ["excluded_records.jsonl", "normalization_errors.jsonl"]:
                source_path = base / name if name == "excluded_records.jsonl" else base_reports / name
                for record in rows(source_path):
                    writers[name].write(record)
            del dates
            modern_path = root / config["raw_dir"] / config["archive"]["path"]
            current = stats["untroubled"]
            current["missing_files"] = []
            file_info = next((x for x in fingerprints if x["path"] == modern_path.relative_to(root).as_posix()), None)

            def handle(row, member):
                nonlocal record_errors
                current["input_records"] += 1
                current["original_label_counts"][json.dumps(row.original_label)] += 1
                current["input_email_candidates"] += row.excluded_reason != "service_file"
                reason, failure = row.excluded_reason, None
                parsed = record = None
                if reason is None:
                    try:
                        parsed = parse_email(row.raw)
                        record = build_record(row, file_info, parsed)
                        date = next((h["value"] for h in parsed["headers_raw"] if h["name"].lower() == "date"), None)
                        enrich(record, date)
                        set_member_period(record, member)
                    except (MemoryError, OSError):
                        raise
                    except Exception as error:
                        reason, failure = "record_error", type(error).__name__
                if reason:
                    item = base_record(row, file_info)
                    item.update(reason=reason, parse_status="failed" if failure else "excluded", usable_model_text=False,
                                raw_sha256=sha_bytes(row.raw) if row.raw is not None else None,
                                size_bytes=row.size_bytes, spam_label=None, phishing_label=None, category_label=None)
                    writers["excluded_records.jsonl"].write(item)
                    current["excluded_by_reason"][reason] += 1
                    if failure:
                        record_errors += 1
                        current["parse_status_counts"]["failed"] += 1
                        writers["normalization_errors.jsonl"].write({"record_id": item["record_id"], "original_locator": row.locator,
                            "source": "untroubled", "stage": "parse_or_build", "error_type": failure, "parse_status": "failed"})
                    return
                writers[NAMES["untroubled"]].write(record)
                filters = [h for h in parsed["headers_raw"] if h["name"].lower() in FILTER_HEADERS]
                current["filter_headers_in_diagnostics"].update(h["name"].lower() for h in filters)
                writers["diagnostics.jsonl"].write({"record_id": record["record_id"], "source": "untroubled", "original_locator": row.locator,
                    "raw_sha256": record["raw_sha256"], **{k: parsed[k] for k in ["headers_raw", "plain_text", "html", "html_text", "sender_domain_info", "parse_warnings"]},
                    "filter_result_headers": filters})
                count_written(current, record)
                review.add(record)
                if current["written"] % 2000 == 0:
                    print(f"untroubled: normalized {current['written']:,}", flush=True)

            if modern_path.is_file():
                print("untroubled: inspect then single solid-archive pass", flush=True)
                checkpoint = {name: (writer.handle.tell(), writer.count, writer.size, writer.digest.copy()) for name,writer in writers.items()}
                errors_before_archive = record_errors
                try:
                    archive_inventory = process_archive(modern_path, root, work / "spools", config, handle,
                                                         config["normalization"]["trial_limit"] if config["trial"] else None)
                except (MemoryError, OSError):
                    raise
                except Exception as error:
                    # A corrupt/unsupported container is source-blocking, not empty success.
                    # Roll back every partially emitted modern record/diagnostic, keep others.
                    for name,writer in writers.items():
                        offset,count,size,digest=checkpoint[name]
                        writer.handle.seek(offset);writer.handle.truncate()
                        writer.count,writer.size,writer.digest=count,size,digest
                    record_errors=errors_before_archive
                    current=stats["untroubled"]=new_stats("untroubled")
                    current.update(status="blocked",missing_files=[],source_error_type=type(error).__name__)
                    failure={"source":"untroubled","stage":"archive_container","original_locator":file_info["path"],"error_type":type(error).__name__}
                    source_errors.append(failure)
                    writers["normalization_errors.jsonl"].write(failure)
                    for group in list(review.groups):
                        if group[0]=="untroubled":del review.groups[group]
            else:
                current["status"] = "missing"
                current["missing_files"] = [modern_path.relative_to(root).as_posix()]
            for writer in writers.values():
                writer.close()
            for path, initial in source_states.items():
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != initial:
                    raise RuntimeError("Source changed during normalization")
            verification = verify_normalized(stage)
            report = {"schema_version": 2, "step": "day2_pipeline_normalization", "status": "partial_source_blocked" if source_errors else "partial_source_missing" if any(s["status"]=="missing" for s in stats.values()) else "complete" if not record_errors else "complete_with_record_errors",
                "date_moscow": "2026-10-11", "config": {"trial": config["trial"], "seed": config["seed"], **config["normalization"]},
                "pipeline_cache_key": key, "code_sha256": code, "sources": stats, "source_files": fingerprints,
                "dependency_versions":{name:importlib.metadata.version(name) for name in ["pyarrow","py7zr"]},"python_version":sys.version,
                "source_size_mtime_unchanged": True, "record_error_count": record_errors, "source_errors": source_errors, "pending_adapter": [],
                "verification": verification, "normalization_errors_sha256": sha_file(stage / "normalization_errors.jsonl"),
                "inventory_comparison": base_report.get("inventory_comparison", {}), "reuse": {"mode": mode, "russian_parquet_rescanned": not reusable},
                "model_feature_allowlist": ["model_text"], "source_headers_are_diagnostics_only": True}
            report["inventory_comparison"]["untroubled"] = {"status": "new_source_not_in_original_inventory", "actual_input": current["input_records"]}
            publish_tree(stage, interim)
            os.replace(interim / "normalization_errors.jsonl", reports / "normalization_errors.jsonl")
            if archive_inventory:
                write_json(reports / "untroubled_archive_members.json", archive_inventory)
            write_json(reports / "normalization_config.json", report["config"])
            write_text(reports / "normalization_review_sample.md", review.render())
            text = render_report(report) + "\n## Расширение pipeline\n\n" + mode + ". Для schema 2 актуальная команда — src/data_pipeline.py; старый prepare_data.py готовит только исходные четыре источника.\nUntroubled: spam=1/source_provided, phishing=null. Полные члены/типы/размеры и API — untroubled_archive_members.json. Период записи взят из Unix timestamp имени в UTC, Date остаётся неподтверждённым; расхождения учитывает отчёт качества. Архив 2026-09 локально отсутствует; modern_spam_test принят из сентября 2025 по явной конфигурации.\n"
            write_text(reports / "normalization_report.md", text)
            write_json(report_path, report)
            return report, report["reuse"]
        finally:
            for writer in writers.values():
                if not writer.handle.closed:
                    writer.handle.close()
