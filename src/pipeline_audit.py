"""Independent streaming audit of published JSONL/manifests, no training."""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from normalization_schema import validate_record, FILTER_HEADERS, identity
from pipeline_common import ROOT, NAMES, rows


def require(condition, message):
    if not condition:
        raise ValueError("Audit: " + message)


def independent_comparison(text):
    return " ".join(unicodedata.normalize("NFC", unicodedata.normalize("NFC", text).casefold()).split())


def audit(processed, interim, manifest=None):
    processed, interim = Path(processed), Path(interim)
    manifest = manifest or json.loads((processed / "splits/manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] in {"complete", "partial"}, "invalid manifest status")
    hashes = {}
    for relative, expected in manifest["artifacts"].items():
        path = processed / relative
        require(path.resolve().is_relative_to(processed.resolve()), "artifact path traversal")
        require(path.is_file(), "missing artifact " + relative)
        digest, count, size = hashlib.sha256(), 0, 0
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                size += len(line)
                json.loads(line)
                count += 1
        require({"sha256": digest.hexdigest(), "rows": count, "size_bytes": size} == expected, "artifact mismatch " + relative)
        hashes[relative] = digest.hexdigest()
    require(manifest["schema_version"] == 2 and manifest["seed"] == 42, "schema/seed")
    settings=manifest["parameters"]["splits"]
    require(settings["modern_test_status"] == "security_blocked" and settings["modern_test_periods"] == ["2026-09"], "blocked 2026 test replaced")
    require(manifest["datasets"]["modern_spam_test"]["splits"]["test"]["status"] == "security_blocked", "missing blocked status")
    require(manifest["datasets"]["modern_spam_test"]["splits"]["test"]["path"] is None and not (processed/"splits/modern_spam_test.jsonl").exists(), "stale or substituted modern test")
    require(any(i["status"] == "security_blocked" and i["path"] == settings["modern_test_archive"] for i in manifest["security_blocked_inputs"]), "blocked input provenance missing")
    groups, membership, sample_near_ids = {}, {}, set()
    sampled_near_edges = []
    for group in rows(processed / "duplicate_groups.jsonl"):
        gid = group["duplicate_group_id"]
        require(gid not in groups, "repeated group ID")
        require(group["size"] == len(group["members"]), "group size mismatch")
        member_ids = sorted(m["record_id"] for m in group["members"])
        encoded = json.dumps(member_ids, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        require(gid == "dup:" + hashlib.sha256(encoded).hexdigest(), "unstable group ID")
        axes = [axis for axis in ["spam", "phishing"] if {m[axis] for m in group["members"] if m[axis] is not None} == {0,1}]
        require(axes == group["conflict_axes"] and bool(axes) == group["quarantine"], "axis conflict/quarantine mismatch")
        for rid in member_ids:
            require(rid not in membership, "original occurs in two groups")
            membership[rid] = gid
        for edge in group["links"]:
            require(edge["a"] in member_ids and edge["b"] in member_ids, "link outside component")
            if edge["reason"] == "near_model_text_5gram_jaccard":
                require(edge["jaccard"] >= manifest["parameters"]["duplicates"]["jaccard_min"], "near Jaccard threshold")
                require(edge["length_ratio"] >= manifest["parameters"]["duplicates"]["length_ratio_min"], "near length threshold")
                if len(sampled_near_edges) < 20:
                    sampled_near_edges.append(edge)
                    sample_near_ids.update([edge["a"],edge["b"]])
        adjacency = defaultdict(set)
        for edge in group["links"]:
            adjacency[edge["a"]].add(edge["b"]);adjacency[edge["b"]].add(edge["a"])
        reached,pending=set(),[member_ids[0]]
        while pending:
            rid=pending.pop()
            if rid not in reached:
                reached.add(rid);pending.extend(adjacency[rid]-reached)
        require(reached == set(member_ids), "group is not a connected component of recorded links")
        groups[gid] = group
    originals, sample_texts = {}, {}
    for source, name in NAMES.items():
        for record in rows(interim / name):
            validate_record(record)
            rid = record["record_id"]
            require(rid not in originals and rid in membership, "normalized membership/ID mismatch")
            require(record["source"] == source, "normalized source mismatch")
            require(rid == identity(source, record["source_path"], record["source_file_sha256"], record["original_locator"], record["original_position"]), "record ID provenance mismatch")
            comparison = independent_comparison(record["model_text"])
            originals[rid] = {"source": source, "spam": record["spam_label"], "phishing": record["phishing_label"],
                "model_hash": record["model_text_sha256"], "original_hash": record["record_content_sha256"],
                "raw": record["raw_sha256"] is not None, "compare_hash": hashlib.sha256(comparison.encode()).hexdigest(),
                "period": record.get("collection_period"), "source_path": record["source_path"], "length": len(comparison),
                "eligible": bool(comparison) and record["parse_status"] != "failed" and record["usable_model_text"]}
            if rid in sample_near_ids:
                sample_texts[rid] = comparison
    require(set(originals) == set(membership), "group mapping does not cover normalized originals")
    for group in groups.values():
        for edge in group["links"]:
            a,b = originals[edge["a"]], originals[edge["b"]]
            if edge["reason"] == "exact_original":
                require(a["raw"] == b["raw"] and a["original_hash"] == b["original_hash"], "incorrect exact edge")
            elif edge["reason"] == "normalized_model_text":
                require(a["length"] > 0 and a["compare_hash"] == b["compare_hash"], "incorrect normalized edge")
            else:
                require(edge["reason"] == "near_model_text_5gram_jaccard", "unknown duplicate edge")
                require(min(a["length"], b["length"]) >= 120, "near grouping of short messages")
    for edge in sampled_near_edges:
        a,b = sample_texts[edge["a"]], sample_texts[edge["b"]]
        aa,bb = {a[i:i+5] for i in range(len(a)-4)}, {b[i:i+5] for i in range(len(b)-4)}
        actual = len(aa & bb) / len(aa | bb)
        ratio = min(len(a),len(b))/max(len(a),len(b))
        require(abs(actual-edge["jaccard"]) < 1e-12 and abs(ratio-edge["length_ratio"]) < 1e-12, "sampled near metrics do not recompute")
    quarantine = {r["record_id"] for r in rows(processed / "quarantine.jsonl")}
    expected_quarantine = {rid for rid,gid in membership.items() if groups[gid]["quarantine"] and originals[rid]["eligible"]}
    require(quarantine == expected_quarantine, "quarantine coverage mismatch")
    excluded = {r["record_id"] for r in rows(processed / "exclusions.jsonl") if r["exclusion_scope"] == "quality"}
    require(excluded == {rid for rid,m in originals.items() if not m["eligible"]}, "quality exclusions mismatch")
    retained = set()
    for record in rows(processed / "records.jsonl"):
        validate_record(record)
        rid = record["record_id"]
        require(rid in originals and rid not in retained and rid not in quarantine and rid not in excluded, "invalid retained record")
        require(record["duplicate_group_id"] == membership[rid], "processed group mismatch")
        require(record["model_text_sha256"] == originals[rid]["model_hash"], "training model text changed during comparisons")
        require(record["spam_label"] == originals[rid]["spam"] and record["phishing_label"] == originals[rid]["phishing"], "label transfer or modification")
        retained.add(rid)
    require(retained | quarantine | excluded == set(originals), "normalized accounting does not balance")
    reserve_spec=manifest["reserves"]["modern_spam_future_reserve"]
    reserved_originals={rid for rid,m in originals.items() if m["source"] == "untroubled" and m["period"] in settings["modern_reserve_periods"] and rid in retained}
    reserved_groups={membership[rid] for rid in reserved_originals}
    reserve_seen=set(); reserve_periods=Counter()
    if reserve_spec["path"]:
        require(reserve_spec["path"] in manifest["artifacts"], "reserve missing artifact")
        for record in rows(processed/reserve_spec["path"]):
            rid=record["record_id"]
            require(rid in reserved_originals and rid not in reserve_seen, "unexpected or repeated reserve original")
            require(record["split"] == "reserve" and record["dataset"] == "modern_spam_future_reserve" and record["usage_status"] == "frozen_do_not_use_for_training_or_setting_selection", "reserve usage role")
            require(record["model_text_sha256"] == originals[rid]["model_hash"] and record["duplicate_group_id"] == membership[rid], "reserve text/group changed")
            reserve_seen.add(rid); reserve_periods.update([record["collection_period"]])
    require(reserve_seen == reserved_originals, "future originals not fully reserved")
    require(reserve_spec["records"] == len(reserve_seen) and reserve_spec["groups"] == len(reserved_groups) and reserve_spec["period_counts"] == dict(reserve_periods), "reserve counters mismatch")
    observed, global_roles, observed_assignments = {}, defaultdict(set), defaultdict(list)
    challenge_members = defaultdict(set)
    for dataset, spec in manifest["datasets"].items():
        observed[dataset] = {}
        seen_groups = set()
        for split, expected in spec["splits"].items():
            values, classes, sources = 0, Counter(), Counter()
            if expected["path"] is None:
                require(expected["records"] == 0 and expected["status"] != "ready", "missing result labelled success")
                observed[dataset][split] = {"records":0,"groups":0,"class_counts":{},"source_counts":{}}
                continue
            for record in rows(processed / expected["path"]):
                validate_record(record)
                rid, gid = record["record_id"], record["duplicate_group_id"]
                require(rid in retained and membership[rid] == gid and gid not in seen_groups, "excluded/non-representative/repeated benchmark group")
                require(record["dataset"] == dataset and record["split"] == split, "dataset role mismatch")
                require(record["model_text_sha256"] == originals[rid]["model_hash"], "split text mutation")
                require(record["spam_label"] == originals[rid]["spam"] and record["phishing_label"] == originals[rid]["phishing"], "split label mutation")
                require(not any(line.partition(":")[0].lower() in FILTER_HEADERS for line in record["model_text"].splitlines()), "filter result leaked into model text")
                require(not any(k in record for k in ["headers_raw", "html", "plain_text", "html_text"]), "diagnostics in ML split")
                source = record["source"]
                if dataset == "spam_email":
                    require(source in {"spamassassin", "lingspam"}, "spam_email source")
                elif dataset == "russian_auxiliary":
                    require(source == "russian_auxiliary" and record["domain_role"] == "auxiliary", "auxiliary source/role")
                elif dataset.startswith("modern_spam"):
                    require(source == "untroubled" and record["spam_label"] == 1, "modern source/label")
                    require(dataset != "modern_spam_test", "security-blocked modern test has records")
                    period_key="modern_train_periods" if dataset == "modern_spam_train_candidates" else "modern_period_challenge_periods"
                    require(record.get("collection_period") in settings[period_key], "modern period outside permitted role")
                    require(record.get("collection_period_provenance") == "archive_member_Unix_timestamp_UTC_unverified_collection_time", "modern timestamp provenance")
                    from datetime import datetime, timezone
                    require(datetime.fromisoformat(record["collection_timestamp"]).astimezone(timezone.utc).strftime("%Y-%m") == record["collection_period"], "modern timestamp period mismatch")
                elif dataset.startswith("phishing"):
                    require(source == "nazario" and record["phishing_label"] == 1 and record["spam_label"] is None, "phishing source/axis")
                    require(record["source_path"].endswith("phishing-2025.txt" if dataset.endswith("test") else "phishing-2024.txt"), "phishing period")
                if dataset.endswith(("test", "challenge")) and dataset not in {"spam_email", "russian_auxiliary"}:
                    require(record["protected_test"] and split == "test", "challenge protection lost")
                    challenge_members[gid].add(dataset)
                fold = record.get("fold_id")
                if dataset == "spam_email" and split == "train" and manifest["cv_folds"]:
                    require(type(fold) is int and 0 <= fold < manifest["cv_folds"] and manifest["cv_fold_mapping"].get(gid) == fold, "CV fold mapping")
                else:
                    require(fold is None, "non-train record participates in CV")
                axis = "phishing_label" if dataset.startswith("phishing") else "spam_label"
                require(type(record[axis]) is int and record[axis] in {0,1}, "unknown target treated as class")
                classes.update([str(record[axis])]); sources.update([source]); values += 1
                seen_groups.add(gid); global_roles[gid].add(split)
                observed_assignments[gid].append({"dataset":dataset,"split":split,"record_id":rid})
            actual = {"records": values, "groups": values, "class_counts":dict(classes), "source_counts":dict(sources)}
            require(all(actual[k] == expected[k] for k in actual), "class/source counts differ from manifest " + dataset + "/" + split)
            if dataset in {"spam_email", "russian_auxiliary"} and manifest["status"] == "complete":
                require({"0","1"} <= set(classes), "missing benchmark class")
            observed[dataset][split] = actual
    require(all(len(roles) == 1 for roles in global_roles.values()), "global train/validation/test group leakage")
    require(set(manifest["group_split_mapping"]) == set(groups), "incomplete final group role mapping")
    for gid, spec in manifest["group_split_mapping"].items():
        require(spec["reserved_future"] == (gid in reserved_groups), "future group reservation lost")
        if gid in reserved_groups:
            require(not (global_roles.get(gid,set()) & {"train","validation"}), "future reserve copy used for training/setting selection")
            require(spec["role"] in {"reserve","test"}, "future group role")
        require(sorted(spec["assignments"], key=lambda x:(x["dataset"],x["split"],x["record_id"])) == sorted(observed_assignments[gid], key=lambda x:(x["dataset"],x["split"],x["record_id"])), "group assignment mapping mismatch")
        if global_roles.get(gid):
            require(global_roles[gid] == {spec["role"]}, "final group role mismatch")
        if spec["protected_test"]:
            require(not global_roles.get(gid) or global_roles[gid] == {"test"}, "protected group in training/validation")
        require(spec["cross_challenge_overlap"] == (len(challenge_members[gid]) > 1), "cross-challenge overlap flag mismatch")
    train_groups = {gid for gid, a in observed_assignments.items() if any(x["dataset"] == "spam_email" and x["split"] == "train" for x in a)}
    require(set(manifest["cv_fold_mapping"]) <= train_groups, "CV uses validation/test")
    fold_classes=defaultdict(Counter)
    for gid,fold in manifest["cv_fold_mapping"].items():
        rid=next(a["record_id"] for a in observed_assignments[gid] if a["dataset"]=="spam_email" and a["split"]=="train")
        fold_classes[str(fold)].update([str(originals[rid]["spam"])])
    require({f:dict(c) for f,c in fold_classes.items()} == manifest.get("cv_fold_class_counts",{}), "CV class counts mismatch")
    require(all({"0","1"} <= set(fold_classes[str(f)]) for f in range(manifest["cv_folds"])), "CV fold missing class")
    return {"status":"passed", "normalized_records":len(originals), "processed_records":len(retained), "quarantine_records":len(quarantine),
            "quality_excluded_records":len(excluded), "group_count":len(groups), "artifacts_checked":len(hashes),
            "reserved_originals":len(reserve_seen), "reserved_groups":len(reserved_groups), "modern_spam_test_status":"security_blocked",
            "sampled_near_pairs_recomputed":len(sampled_near_edges), "global_group_leakage":0, "datasets":observed}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--interim-dir", type=Path, default=ROOT/"data/interim")
    args=parser.parse_args()
    print(json.dumps(audit(args.processed_dir,args.interim_dir),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
