"""One sequential day-2 pipeline: normalization -> quality/groups -> roles/splits -> audit."""
import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
from data_quality import build_quality, provenance
from split_data import build_splits, ROLES
from pipeline_normalization import source_fingerprints, ensure_normalized, cache_key, verify_normalized
from pipeline_common import ROOT, NAMES, JsonlWriter, write_json, write_text, fingerprint_code, publish_tree
from normalization_schema import canonical_bytes, sha_bytes
from inventory_data import sha_file
from normalization_reporting import render_report
from pipeline_audit import audit
from pipeline_reporting import quality_markdown, split_markdown, review_markdown, status_markdown


def validate_config(root, config):
    if config["seed"] != 42 or config["schema_version"] != 2:
        raise ValueError("This declared pipeline requires seed=42/schema=2")
    paths = {k:(root/config[k]).resolve() for k in ["raw_dir","interim_dir","processed_dir","reports_dir"]}
    if any(not p.is_relative_to(root.resolve()) for p in paths.values()):
        raise ValueError("All pipeline paths must remain within project root")
    raw = paths["raw_dir"]
    if any(p.is_relative_to(raw) or raw.is_relative_to(p) for k,p in paths.items() if k!="raw_dir"):
        raise ValueError("Outputs cannot overlap data/raw")
    if len(set(paths.values())) != 4:
        raise ValueError("Pipeline output roots must be distinct")
    n = config["duplicates"]
    if n["near_min_chars"] < 120 or n["shingle_chars"] != 5 or n["jaccard_min"] < .95 or n["length_ratio_min"] < .90 or not 0 < n["lsh_threshold"] < n["jaccard_min"]:
        raise ValueError("Invalid duplicate-search thresholds")
    if set(config["splits"]["fractions"]) != {"train","validation","test"} or abs(sum(config["splits"]["fractions"].values())-1)>1e-12:
        raise ValueError("Invalid split fractions")
    if any(v<=0 for v in config["normalization"].values()):
        raise ValueError("Normalization limits must be positive")
    s=config["splits"]
    period_sets=[set(s[k]) for k in ["modern_test_periods","modern_period_challenge_periods","modern_train_periods","modern_reserve_periods"]]
    if any(a & b for i,a in enumerate(period_sets) for b in period_sets[i+1:]):
        raise ValueError("Modern period roles must be disjoint")
    if s["modern_test_status"] != "security_blocked" or s["modern_test_periods"] != ["2026-09"]:
        raise ValueError("September 2026 must remain security_blocked; do not substitute or restore archive")
    if not period_sets[2] <= {f"2025-{m:02d}" for m in range(1,9)}:
        raise ValueError("Modern train must remain January-August 2025")
    archive_path=(paths["raw_dir"]/config["archive"]["path"]).resolve()
    if not archive_path.is_relative_to(paths["raw_dir"]):
        raise ValueError("Archive path must remain within data/raw")


def configured(config_path, trial=False):
    config=json.loads(Path(config_path).read_text(encoding="utf-8"))
    config["trial"]=trial
    if trial:
        for key in ["interim_dir","processed_dir","reports_dir"]:
            config[key] = (Path(config[key])/"trial").as_posix()
    return config


def emitted_record(quality, rid):
    record=quality.record(rid)
    meta=quality.metadata[rid]
    record.update(duplicate_group_id=quality.by_record[rid],quality_status="eligible",quality_flags=meta["flags"],
                  language="unknown",normalization_record_id=rid)
    return record


def refresh_reporting_only(root, config, fingerprints):
    """Known presentation-only migration; never bless a changed parser/adapter/cleaner."""
    reports=root/config["reports_dir"]
    path=reports/"normalization_report.json"
    if not path.is_file():return False
    old=json.loads(path.read_text(encoding="utf-8"))
    key,code=cache_key(config,fingerprints)
    if old.get("schema_version")!=2 or old.get("status")!="complete":return False
    changed={k for k in set(code)|set(old.get("code_sha256",{})) if code.get(k)!=old.get("code_sha256",{}).get(k)}
    if changed!={"src/normalization_reporting.py"} or old["code_sha256"]["src/normalization_reporting.py"]!="044cb4097f0f97cc09c444c0e160c0d1639b6d16f210126523d88939e0c813e5":return False
    tree=ast.parse((ROOT/"src/normalization_reporting.py").read_text(encoding="utf-8"))
    tree.body=[node for node in tree.body if not isinstance(node,ast.FunctionDef) or node.name!="render_report"]
    if hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()!="34ecbce9cb55b87f34f2af3abd18b1448cc64ec0b12776cf6d6bdffcba5a240f":return False
    if old.get("source_files")!=fingerprints or old.get("config")!={"trial":config["trial"],"seed":config["seed"],**config["normalization"]}:return False
    if old.get("python_version")!=sys.version or old.get("dependency_versions")!={name:importlib.metadata.version(name) for name in ["pyarrow","py7zr"]}:return False
    if verify_normalized(root/config["interim_dir"])!=old["verification"] or sha_file(reports/"normalization_errors.jsonl")!=old["normalization_errors_sha256"]:
        raise ValueError("Cannot refresh presentation of a corrupt normalized cache")
    previous_text=(reports/"normalization_report.md").read_text(encoding="utf-8")
    marker="\n## Расширение pipeline\n"
    extension=marker+previous_text.split(marker,1)[1] if marker in previous_text else ""
    old.setdefault("data_generation_code_sha256",old["code_sha256"])
    old.update(code_sha256=code,pipeline_cache_key=key,
               presentation_refresh={"changed_function":"normalization_reporting.render_report only",
                                     "non_renderer_AST_sha256":"34ecbce9cb55b87f34f2af3abd18b1448cc64ec0b12776cf6d6bdffcba5a240f",
                                     "normalized_data_unchanged_verified":True})
    write_text(reports/"normalization_report.md",render_report(old)+extension)
    write_json(path,old)
    print("normalization: verified presentation-only refresh; data generation hashes retained",flush=True)
    return True


def export_files(stage, quality, splits):
    writers={name:JsonlWriter(stage/name) for name in ["records.jsonl","duplicate_groups.jsonl","quarantine.jsonl","exclusions.jsonl"]}
    datasets=copy.deepcopy(splits["report"]["datasets"])
    for dataset,spec in datasets.items():
        for split,values in spec["splits"].items():
            path = f"splits/{dataset}/{split}.jsonl" if dataset in {"spam_email","russian_auxiliary"} else f"splits/{dataset}.jsonl"
            values["path"] = path if values["records"] else None
            values["status"] = "ready" if values["records"] else "no_eligible_records_or_period_not_sampled"
            if dataset == "modern_spam_test":
                values.update(path=None,status="security_blocked")
            if values["records"]:
                writers[path]=JsonlWriter(stage/path)
    reserve=splits["report"]["modern_reserve"]
    if reserve["path"]:
        writers[reserve["path"]]=JsonlWriter(stage/reserve["path"])
    try:
        for gid,group in sorted(quality.groups.items()):
            writers["duplicate_groups.jsonl"].write(group)
        for rid,meta in sorted(quality.metadata.items()):
            gid=quality.by_record[rid]
            if not meta["eligible"]:
                writers["exclusions.jsonl"].write({**provenance(meta),"duplicate_group_id":gid,"exclusion_scope":"quality",
                    "reason":"unusable_text_or_failed", "quality_flags":meta["flags"]})
            elif quality.groups[gid]["quarantine"]:
                writers["quarantine.jsonl"].write({**provenance(meta),"duplicate_group_id":gid,
                    "reason":quality.groups[gid]["quarantine_reason"],"conflict_axes":quality.groups[gid]["conflict_axes"]})
            else:
                writers["records.jsonl"].write(emitted_record(quality,rid))
        for item in splits["suppressed"]:
            writers["exclusions.jsonl"].write({**item,"exclusion_scope":"assignment","original_locator":quality.metadata[item["record_id"]]["original_locator"]})
        for item in splits["assignments"]:
            record=emitted_record(quality,item["record_id"])
            record.update(dataset=item["dataset"],split=item["split"],protected_test=item["protected"],
                          cross_challenge_overlap=item["cross_challenge_overlap"],
                          fold_id=splits["cv_fold_mapping"].get(item["group_id"]) if item["dataset"]=="spam_email" and item["split"]=="train" else None)
            writers[datasets[item["dataset"]]["splits"][item["split"]]["path"]].write(record)
        for rid in splits["reserve_record_ids"]:
            record=emitted_record(quality,rid)
            record.update(dataset="modern_spam_future_reserve",split="reserve",usage_status="frozen_do_not_use_for_training_or_setting_selection")
            writers[reserve["path"]].write(record)
        artifacts={name:writer.close() for name,writer in writers.items()}
        return artifacts,datasets
    finally:
        for writer in writers.values():
            if not writer.handle.closed:
                writer.handle.close()


def run_pipeline(root, config, force_normalize=False, force_process=False):
    started=time.perf_counter()
    root=Path(root).resolve()
    validate_config(root,config)
    reports,processed,interim=[root/config[k] for k in ["reports_dir","processed_dir","interim_dir"]]
    reports.mkdir(parents=True,exist_ok=True)
    processed.mkdir(parents=True,exist_ok=True)
    print("pipeline: fingerprint raw inputs once",flush=True)
    fingerprints,source_states=source_fingerprints(root,config)
    if not force_normalize:
        refresh_reporting_only(root,config,fingerprints)
    normalized,reuse=ensure_normalized(root,config,fingerprints,source_states,force_normalize)
    normalized={**normalized,"reuse":reuse}
    code=fingerprint_code()
    code["src/audit_decoding_group.py"]=sha_file(ROOT/"src/audit_decoding_group.py")
    versions={name:importlib.metadata.version(name) for name in ["pyarrow","py7zr","datasketch","numpy","scipy"]}
    process_key=sha_bytes(canonical_bytes({"config":config,"code":code,"versions":versions,"normalized":normalized["verification"]["artifacts"]}))
    manifest_path=processed/"splits/manifest.json"
    previous=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if not force_process and manifest_path.is_file():
        previous=json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("process_cache_key")==process_key:
            verified=audit(processed,interim,previous)
            print("pipeline: processed cache verified by independent audit; reused",flush=True)
            return previous,verified
    print("pipeline: quality and joint duplicate search",flush=True)
    with tempfile.TemporaryDirectory(prefix=".pipeline-",suffix=".tmp",dir=processed) as work_name:
        work=Path(work_name)
        # Temp cleanup only targets this newly created, verified workspace subtree.
        if not work.resolve().is_relative_to(processed.resolve()):
            raise ValueError("Unsafe temporary workspace")
        quality=build_quality(interim,work,config)
        try:
            splits=build_splits(quality,config)
            stage=work/"ready"
            stage.mkdir()
            print("pipeline: export separate datasets and mappings",flush=True)
            artifacts,datasets=export_files(stage,quality,splits)
            statuses=[s["status"] for s in normalized["sources"].values()]
            partial=splits["report"]["status"]!="complete" or any(s!="complete" for s in statuses) or normalized["record_error_count"]>0
            partial = partial or any(v["path"] is None for d in datasets.values() for v in d["splits"].values())
            manifest={"manifest_version":1,"pipeline_version":config["pipeline_version"],"schema_version":2,"status":"partial" if partial else "complete",
                "seed":config["seed"],"parameters":config,"process_cache_key":process_key,"raw_inputs":fingerprints,
                "normalized_inputs":normalized["verification"]["artifacts"],"source_status":{k:v["status"] for k,v in normalized["sources"].items()},
                "code_sha256":code,"dependency_versions":versions,"artifacts":artifacts,"datasets":datasets,
                "group_split_mapping":splits["group_mapping"],"cv_folds":splits["report"]["cv"]["folds"],"cv_fold_mapping":splits["cv_fold_mapping"],
                "cv_fold_class_counts":splits["report"]["cv"]["fold_class_counts"],
                "pending":splits["report"]["pending"],"modern_period_assumption":config["splits"]["modern_test_period_basis"],
                "security_blocked_inputs":[{"path":config["splits"]["modern_test_archive"],"status":"security_blocked","evidence":"user report; no Defender quarantine accessed", "reason":config["splits"]["modern_test_block_reason"]}],
                "reserves":{"modern_spam_future_reserve":splits["report"]["modern_reserve"]},
                "needs_review_groups":quality.report["needs_review_groups"],
                "review_evidence":{name:{"path":(reports/name).relative_to(root).as_posix(),"sha256":sha_file(reports/name)} for name in ["decoding_and_group_audit.md","decoding_and_group_audit.json"] if (reports/name).is_file()},
                "model_feature_allowlist":["model_text"],"no_models_or_fitted_transformations":True,
                "normalization_reuse":reuse}
            print("pipeline: independent staged audit",flush=True)
            verified=audit(stage,interim,manifest)
            manifest["audit"]=verified
            write_json(stage/"splits/manifest.json",manifest)
            quality_report={**quality.report,"normalization_exclusions":normalized["verification"]["artifacts"]["excluded_records.jsonl"]["rows"],
                            "code_sha256":code,"parameters":config["duplicates"],"audit":verified}
            split_report={**splits["report"],"audit":verified,"manifest_path":(processed/"splits/manifest.json").relative_to(root).as_posix()}
            review=review_markdown(quality,splits,config["seed"])
            # Remove only obsolete, hash-verified outputs registered by the old manifest.
            # In particular the old September-2025 file must not masquerade as modern_spam_test.
            stale=[]
            for name,spec in (previous or {}).get("artifacts",{}).items():
                if name in artifacts: continue
                path=processed/name
                if not path.resolve().is_relative_to(processed.resolve()):raise ValueError("Unsafe obsolete output path")
                if path.is_file():
                    if sha_file(path)!=spec["sha256"]:raise ValueError("Obsolete output changed outside pipeline: "+name)
                    stale.append(path)
            for path in stale:path.unlink()
            # Publish final manifest only after every staged consistency check passed.
            publish_tree(stage,processed,manifest_name="splits/manifest.json")
            elapsed=time.perf_counter()-started
            write_json(reports/"data_quality_report.json",quality_report)
            write_text(reports/"data_quality_report.md",quality_markdown(quality_report,normalized))
            write_json(reports/"split_report.json",split_report)
            write_text(reports/"split_report.md",split_markdown(split_report,manifest))
            write_text(reports/"data_review_sample.md",review)
            write_json(reports/"pipeline_config.json",config)
            write_text(reports/"DAY2_STATUS.md",status_markdown(manifest,normalized,quality_report,split_report,elapsed))
            write_json(reports/"pipeline_run.json",{"status":manifest["status"],"seconds":elapsed,"processed_bytes":sum(a["size_bytes"] for a in artifacts.values()),
                "normalization_reuse":reuse,"audit":verified,"quality_timing":quality.report["timing_seconds"]})
            print(json.dumps({"status":manifest["status"],"seconds":round(elapsed,2),"normalized":quality.report["normalized_records"],
                "groups":quality.report["group_count"],"processed":quality.report["processed_records"],"quarantine":quality.report["quarantine_records"],
                "audit":verified["status"]},ensure_ascii=False),flush=True)
            return manifest,verified
        finally:
            quality.db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/day2_pipeline.json")
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--trial",action="store_true")
    parser.add_argument("--force-normalize",action="store_true")
    parser.add_argument("--force-process",action="store_true")
    args=parser.parse_args()
    config=configured(args.config,args.trial)
    try:
        run_pipeline(args.project_root,config,args.force_normalize,args.force_process)
    except (MemoryError,OSError):
        reports=args.project_root/config["reports_dir"]
        write_text(reports/"DAY2_STATUS.md","# DAY2_STATUS\n\nblocked: system/memory/write failure; final manifest was not declared successful. Inspect terminal and rerun/audit.\n")
        raise
    except Exception as error:
        reports=args.project_root/config["reports_dir"]
        write_text(reports/"DAY2_STATUS.md",f"# DAY2_STATUS\n\nblocked: {type(error).__name__}: {str(error)[:300]}. No successful final manifest for this run.\n")
        raise


if __name__=="__main__":
    main()
