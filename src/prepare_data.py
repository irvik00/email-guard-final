"""Email Guard day 2 step 2: separate, reproducible normalized JSONL sources."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

sys.dont_write_bytecode = True
from inventory_data import sha_file
from normalization_adapters import SOURCE_FILES, iter_spamassassin, iter_lingspam, iter_nazario, is_internal_metadata, project_path, sample_russian
from normalization_reporting import ReviewSamples, count_written, new_stats, render_report
from normalization_schema import FILTER_HEADERS, base_record, build_record, canonical_bytes, sha_bytes, validate_record
from read_email import parse_email

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = {"spamassassin": "spamassassin.jsonl", "lingspam": "lingspam.jsonl",
                "nazario": "nazario.jsonl", "russian_auxiliary": "russian_auxiliary_sample.jsonl"}
DATA_FILES = [*OUTPUT_NAMES.values(), "diagnostics.jsonl", "excluded_records.jsonl"]


@dataclass
class Config:
    project_root: Path = ROOT
    raw_dir: Path = ROOT / "data/raw"
    output_dir: Path = ROOT / "data/interim"
    reports_dir: Path = ROOT / "reports"
    seed: int = 42
    russian_max_per_class: int = 20000
    max_record_bytes: int = 20 * 1024 * 1024
    trial: bool = False
    trial_limit: int = 20


def json_line(record):
    return canonical_bytes(record) + b"\n"


class StagedOutputs:
    def __init__(self, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=".normalization-", suffix=".tmp", dir=output_dir))
        self.handles = {name: (self.directory / name).open("wb") for name in DATA_FILES + ["normalization_errors.jsonl"]}

    def write(self, name, record):
        self.handles[name].write(json_line(record))

    def close(self):
        for handle in self.handles.values():
            if not handle.closed:
                handle.flush()
                handle.close()

    def publish(self, output_dir, reports_dir):
        # Each replace is atomic. The report with hashes is published last.
        for name in DATA_FILES:
            os.replace(self.directory / name, output_dir / name)
        os.replace(self.directory / "normalization_errors.jsonl", reports_dir / "normalization_errors.jsonl")
        self.directory.rmdir()


def verify_outputs(directory):
    all_ids, raw_ids, diagnostic_ids = set(), set(), set()
    counts, artifacts = {}, {}
    for source, name in OUTPUT_NAMES.items():
        digest, count = hashlib.sha256(), 0
        with (directory / name).open("rb") as handle:
            for line in handle:
                digest.update(line)
                record = json.loads(line)
                validate_record(record)
                if record["source"] != source or record["record_id"] in all_ids:
                    raise ValueError("Wrong source or duplicate record ID")
                if any(k in record for k in ["headers_raw", "html", "plain_text", "html_text"]):
                    raise ValueError("Diagnostics leaked into main record")
                all_ids.add(record["record_id"])
                if record["input_kind"] in {"raw_email", "mbox_email"}:
                    raw_ids.add(record["record_id"])
                else:
                    fields = ({"subject": record["subject"], "message": record["body_text"], "label": record["original_label"]}
                              if source == "lingspam" else {"text": record["body_text"], "label": record["original_label"]})
                    if record["record_content_sha256"] != sha_bytes(canonical_bytes(fields)):
                        raise ValueError("Text record content hash mismatch")
                count += 1
        counts[source] = count
        artifacts[name] = {"sha256": digest.hexdigest(), "size_bytes": (directory / name).stat().st_size, "rows": count}
    for name in ["diagnostics.jsonl", "excluded_records.jsonl"]:
        digest, count = hashlib.sha256(), 0
        with (directory / name).open("rb") as handle:
            for line in handle:
                digest.update(line)
                record = json.loads(line)
                if name == "diagnostics.jsonl":
                    if record["record_id"] in diagnostic_ids:
                        raise ValueError("Repeated diagnostics ID")
                    diagnostic_ids.add(record["record_id"])
                elif record["record_id"] in all_ids or not record.get("reason"):
                    raise ValueError("Excluded record also included or missing reason")
                count += 1
        artifacts[name] = {"sha256": digest.hexdigest(), "size_bytes": (directory / name).stat().st_size, "rows": count}
    if diagnostic_ids != raw_ids:
        raise ValueError("Diagnostics do not match raw/mbox records")
    return {"status": "passed", "written_counts": counts, "unique_record_ids": len(all_ids),
            "linked_diagnostics": len(diagnostic_ids), "artifacts": artifacts}


def inventory_comparison(inventory, stats, file_info, trial):
    result = {}
    for source, item in stats.items():
        previous = inventory.get("sources", {}).get(source)
        if not previous:
            result[source] = {"status": "no_prior_inventory"}
            continue
        whole = previous["whole_corpus"]
        expected = whole.get("record_count", whole.get("container_record_count", whole.get("scanned_row_count")))
        actual = item["input_email_candidates"] if source == "spamassassin" else item["input_records"]
        check = {"expected_input": expected, "actual_input": actual,
                 "status": "trial_limited" if trial else "matched" if expected == actual else "different",
                 "reason": "trial limits per file/source" if trial else "email candidates exclude service files; sampling does not change scanned row count"}
        if item.get("missing_files"):
            check["reason"] = "missing source file(s) listed in source report"
        if item["status"] == "failed":
            check["reason"] = "source/container failure, not a complete pass"
        if source == "russian_auxiliary" and item.get("russian_scan"):
            original = {"null" if v["value"] is None else "true" if v["value"] is True else "false": v["count"] for v in whole["label_counts"]}
            current = {k: v for k, v in item["russian_scan"]["native_label_counts"].items() if v or k in original}
            check["label_counts_match"] = original == current if not trial else None
        result[source] = check
    return result


def run(config, precomputed_files=None):
    root, raw, output, reports = [p.resolve() for p in [config.project_root, config.raw_dir, config.output_dir, config.reports_dir]]
    if not raw.is_relative_to(root) or output.is_relative_to(raw) or reports.is_relative_to(raw) or raw.is_relative_to(output) or raw.is_relative_to(reports):
        raise ValueError("Paths must keep output/reports separate from raw; raw must be within project root")
    reports.mkdir(parents=True, exist_ok=True)
    inventory_path = root / "reports/data_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}
    expected_hashes = {f["path"]: f["sha256"] for f in inventory.get("raw_files", [])}
    files, info, stat_before = {}, {}, {}
    stats = {s: new_stats(s) for s in SOURCE_FILES}
    for source, names in SOURCE_FILES.items():
        files[source] = [raw / name for name in names if (raw / name).is_file()]
        stats[source]["missing_files"] = [project_path(raw / name, root) for name in names if not (raw / name).is_file()]
        if stats[source]["missing_files"]:
            stats[source]["status"] = "partial_available" if files[source] else "missing"
        for path in files[source]:
            metadata = path.stat()
            stat_before[path] = (metadata.st_size, metadata.st_mtime_ns)
            relative = project_path(path, root)
            digest = precomputed_files[path.resolve()] if precomputed_files is not None and path.resolve() in precomputed_files else sha_file(path)
            info[path] = {"path": relative, "size_bytes": metadata.st_size, "sha256": digest,
                          "matches_inventory_sha256": digest == expected_hashes[relative] if relative in expected_hashes else None}
    used_paths = set(info)
    pending = [{"path": project_path(p, root), "size_bytes": p.stat().st_size, "status": "pending_adapter"}
               for p in sorted(raw.rglob("*")) if p.is_file() and p not in used_paths
               and re_archive(p)]
    staged = StagedOutputs(output)
    review = ReviewSamples(config.seed)
    errors, source_errors = 0, []
    seen_ids = set()

    def excluded(row, reason, failed=False, stage=None, error_type=None):
        nonlocal errors
        item = base_record(row, info[row.path])
        item.update(reason=reason, parse_status="failed" if failed else "excluded", usable_model_text=False,
                    raw_sha256=sha_bytes(row.raw) if row.raw is not None else None, size_bytes=row.size_bytes,
                    spam_label=None, phishing_label=None, category_label=None)
        if row.bounds:
            item["raw_byte_bounds"] = row.bounds
        staged.write("excluded_records.jsonl", item)
        stats[row.source]["excluded_by_reason"][reason] += 1
        if failed:
            stats[row.source]["parse_status_counts"]["failed"] += 1
            errors += 1
            staged.write("normalization_errors.jsonl", {"record_id": item["record_id"], "source": row.source,
                         "original_locator": row.locator, "stage": stage, "error_type": error_type, "parse_status": "failed"})

    def handle(row):
        current = stats[row.source]
        current["input_records"] += 1
        if row.source == "spamassassin" and row.subset is not None:
            current["input_email_candidates"] += 1
        current["original_label_counts"][json.dumps(row.original_label, ensure_ascii=False)] += 1
        if row.excluded_reason:
            excluded(row, row.excluded_reason)
            return
        stage = "input_validation"
        try:
            if row.fields is not None and (set(row.fields) != ({"subject", "message", "label"} if row.source == "lingspam" else {"text", "label"})
                                           or any(type(v) is not str for k, v in row.fields.items() if k != "label")):
                raise ValueError("Invalid text row fields")
            if row.source == "nazario" and is_internal_metadata(row.raw):
                excluded(row, "mbox_internal_metadata")
                return
            parsed = None
            if row.raw is not None:
                stage = "parse_email"
                parsed = parse_email(row.raw)
            stage = "schema_and_model_text"
            record = build_record(row, info[row.path], parsed)
            if record["record_id"] in seen_ids:
                raise ValueError("Record ID collision")
        except (MemoryError, OSError):
            raise
        except Exception as error:
            excluded(row, "record_error", failed=True, stage=stage, error_type=type(error).__name__)
            return
        # Write failures are run failures, outside the per-record error boundary.
        staged.write(OUTPUT_NAMES[row.source], record)
        seen_ids.add(record["record_id"])
        if parsed is not None:
            filters = [h for h in parsed["headers_raw"] if h["name"].lower() in FILTER_HEADERS]
            current["filter_headers_in_diagnostics"].update(h["name"].lower() for h in filters)
            diagnostic = {"record_id": record["record_id"], "source": row.source, "original_locator": row.locator,
                          "raw_sha256": record["raw_sha256"], **{k: parsed[k] for k in ["headers_raw", "plain_text", "html", "html_text", "sender_domain_info"]},
                          "filter_result_headers": filters, "parse_warnings": parsed["parse_warnings"]}
            if row.bounds:
                diagnostic["raw_byte_bounds"] = row.bounds
            staged.write("diagnostics.jsonl", diagnostic)
        count_written(current, record)
        review.add(record)

    trial_limit = config.trial_limit if config.trial else None
    try:
        for source, paths in files.items():
            if not paths:
                continue
            print(f"{source}: processing {len(paths)} file(s)", flush=True)
            try:
                if source == "spamassassin":
                    rows = iter_spamassassin(paths, root, config.max_record_bytes, trial_limit)
                elif source == "lingspam":
                    rows = iter_lingspam(paths, root, trial_limit)
                elif source == "nazario":
                    rows = iter_nazario(paths, root, config.max_record_bytes, trial_limit)
                else:
                    quota = min(config.russian_max_per_class, config.trial_limit) if config.trial else config.russian_max_per_class
                    def invalid(row):
                        excluded(row, row.excluded_reason)
                    rows, scan = sample_russian(paths[0], root, quota, config.seed, invalid,
                                                trial_scan_limit=50000 if config.trial else None,
                                                progress=lambda n: print(f"russian_auxiliary: scanned {n:,} rows", flush=True))
                    stats[source]["russian_scan"] = scan
                for row in rows:
                    handle(row)
                if source == "russian_auxiliary":
                    stats[source]["input_records"] = scan["scanned_rows"]
                    stats[source]["original_label_counts"] = scan["native_label_counts"]
            except (MemoryError, OSError):
                raise
            except Exception as error:
                stats[source]["status"] = "failed"
                failure = {"source": source, "stage": "container_or_source", "error_type": type(error).__name__}
                source_errors.append(failure)
                staged.write("normalization_errors.jsonl", failure)
            print(f"{source}: written {stats[source]['written']:,}; status {stats[source]['status']}", flush=True)
        for path, initial in stat_before.items():
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != initial:
                raise RuntimeError("Source changed during run")
        staged.close()
        verification = verify_outputs(staged.directory)
        for source, current in stats.items():
            if verification["written_counts"][source] != current["written"]:
                raise ValueError("Written count mismatch")
        comparison = inventory_comparison(inventory, stats, info, config.trial)
        config_json = {k: project_path(v, root) if isinstance(v, Path) and k != "project_root" else "." if k == "project_root" else v for k, v in asdict(config).items()}
        code_names = ["prepare_data.py", "normalization_adapters.py", "normalization_schema.py", "normalization_reporting.py", "read_email.py", "inventory_data.py", "russian_inventory.py"]
        code_hashes = {"src/" + name: sha_file(root / "src" / name) for name in code_names if (root / "src" / name).is_file()}
        report = {"schema_version": 1, "step": "day2_step2_only", "date_moscow": "2026-10-11",
                  "status": "failed_source" if source_errors else "complete_with_record_errors" if errors else "complete",
                  "config": config_json, "sources": stats, "source_files": list(info.values()),
                  "source_size_mtime_unchanged": True, "inventory_comparison": comparison,
                  "pending_adapter": pending, "record_error_count": errors, "source_errors": source_errors,
                  "verification": verification, "code_sha256": code_hashes,
                  "model_feature_allowlist": ["model_text"], "source_headers_are_diagnostics_only": True}
        error_file = staged.directory / "normalization_errors.jsonl"
        report["normalization_errors_sha256"] = sha_file(error_file)
        contents = {"normalization_config.json": json.dumps(config_json, ensure_ascii=False, indent=2) + "\n",
                    "normalization_review_sample.md": review.render(), "normalization_report.md": render_report(report),
                    "normalization_report.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n"}
        if source_errors:
            report["unpublished_staging_directory"] = project_path(staged.directory, root)
            contents["normalization_report.json"] = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            # Keep successfully processed staging data reviewable, but do not replace
            # an earlier complete data run with a failed source/container run.
            os.replace(error_file, reports / "normalization_errors.jsonl")
        else:
            staged.publish(output, reports)
        for name, content in contents.items():
            temporary = reports / (name + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, reports / name)
        return report
    finally:
        staged.close()


def re_archive(path):
    lower = path.name.lower()
    return lower.endswith((".7z", ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".rar"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--russian-max-per-class", type=int, default=20000)
    parser.add_argument("--max-record-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--trial-limit", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if min(args.russian_max_per_class, args.max_record_bytes, args.trial_limit) <= 0:
        parser.error("Limits must be positive")
    output = args.output_dir / "trial" if args.trial else args.output_dir
    reports = args.reports_dir / "trial" if args.trial else args.reports_dir
    if args.verify_only:
        verification = verify_outputs(output)
        previous = json.loads((reports / "normalization_report.json").read_text(encoding="utf-8"))
        if verification != previous["verification"]:
            raise ValueError("Outputs differ from normalization report")
        if sha_file(reports / "normalization_errors.jsonl") != previous["normalization_errors_sha256"]:
            raise ValueError("Error log differs from report")
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return
    report = run(Config(raw_dir=args.raw_dir, output_dir=output, reports_dir=reports, seed=args.seed,
                        russian_max_per_class=args.russian_max_per_class, max_record_bytes=args.max_record_bytes,
                        trial=args.trial, trial_limit=args.trial_limit))
    print(json.dumps({"status": report["status"], "counts": {s: v["written"] for s, v in report["sources"].items()},
                      "pending_adapter": report["pending_adapter"]}, ensure_ascii=False, indent=2))
    if report["source_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
