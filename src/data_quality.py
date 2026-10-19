"""Joint exact/normalized/near grouping with disk-backed texts and explicit quarantine."""
import json
import sqlite3
import time
from collections import Counter, defaultdict, OrderedDict

from datasketch import MinHash, MinHashLSH, LeanMinHash
from normalization_schema import canonical_bytes, sha_bytes
from pipeline_common import NAMES, rows, compare_text, JsonlWriter


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, value):
        self.parent.setdefault(value, value)

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            previous = value
            value = self.parent[value]
            self.parent[previous] = root
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def quality_flags(record, near_min=120):
    text = record["model_text"]
    flags = []
    if not text.strip():
        flags.append("blank_model_text")
    if not (record["subject"] or "").strip():
        flags.append("subject_unavailable" if record["feature_availability"]["subject"] == "unavailable" else "empty_or_missing_subject")
    if len(text) < near_min:
        flags.append("short_text_no_near_search")
    if record["parse_warnings"]:
        flags.append("parser_warning")
    if not (record["body_text"] or "").strip():
        flags.append("empty_or_unavailable_body")
    if record["formatting"] is not None and record["formatting"].get("has_html"):
        flags.append("html_present")
    if "\ufffd" in text or any("кодиров" in w.lower() or "байты" in w.lower() for w in record["parse_warnings"]):
        flags.append("possible_decoding_problem")
    if record["possible_filter_marker"]:
        flags.append("possible_filter_marker")
    if record["input_kind"].startswith("text_only"):
        flags.append("structural_metadata_unavailable")
    if record["parse_status"] == "failed":
        flags.append("parse_failed")
    if record.get("date_provenance") == "email_Date_header_unverified_unparseable":
        flags.append("unparseable_declared_Date_header")
    flags.extend(record.get("date_disagreements", []))
    return flags


def conflict_axes(members):
    return [axis for axis in ["spam", "phishing"] if {m[axis] for m in members if m[axis] is not None} == {0, 1}]


def provenance(meta):
    return {k: meta[k] for k in ["record_id", "source", "source_path", "source_file_sha256", "original_locator", "subset",
                                "spam", "phishing", "original_label", "label_status", "collection_period", "input_kind", "parse_status"]}


class QualityResult:
    def __init__(self, db, metadata, groups, by_record, report):
        self.db, self.metadata, self.groups, self.by_record, self.report = db, metadata, groups, by_record, report

    def record(self, record_id):
        value = self.db.execute("SELECT payload FROM records WHERE id=?", (record_id,)).fetchone()
        if value is None:
            raise KeyError(record_id)
        return json.loads(value[0])


def build_quality(interim, work, config):
    start = time.perf_counter()
    db = sqlite3.connect(work / "quality_cache.sqlite")
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE records(id TEXT PRIMARY KEY, comparison TEXT NOT NULL, payload TEXT NOT NULL)")
    metadata, uf, edges = {}, UnionFind(), []
    exact, normalized = {}, {}
    stats = {}
    params = config["duplicates"]
    for source, name in NAMES.items():
        current = {"normalized_records": 0, "quality_eligible_before_conflicts": 0, "flags": Counter(),
                   "parse_status": Counter(), "spam_labels": Counter(), "phishing_labels": Counter(),
                   "missing_fields": Counter(), "unavailable_features": Counter(), "lengths": [],
                   "script_indicators": Counter(), "collection_periods": Counter(), "language": "unknown; scripts measured, no language classifier"}
        stats[source] = current
        for record in rows(interim / name):
            rid = record["record_id"]
            if rid in metadata:
                raise ValueError("Repeated normalized ID")
            text = compare_text(record["model_text"])
            flags = quality_flags(record, params["near_min_chars"])
            eligible = bool(text) and record["parse_status"] != "failed" and record["usable_model_text"]
            meta = {"record_id": rid, "source": source, "source_path": record["source_path"], "source_file_sha256": record["source_file_sha256"],
                    "original_locator": record["original_locator"], "subset": record["subset"], "original_label": record["original_label"],
                    "label_status": record["label_status"], "spam": record["spam_label"], "phishing": record["phishing_label"],
                    "collection_period": record.get("collection_period"), "parse_status": record["parse_status"],
                    "input_kind": record["input_kind"], "eligible": eligible, "flags": flags,
                    "length": len(text), "original_model_length": len(record["model_text"]), "warnings": record["parse_warnings"],
                    "has_headers": record["feature_availability"]["headers"] == "observable"}
            metadata[rid] = meta
            uf.add(rid)
            db.execute("INSERT INTO records VALUES (?,?,?)", (rid, text, canonical_bytes(record).decode("utf-8")))
            current["normalized_records"] += 1
            current["quality_eligible_before_conflicts"] += eligible
            current["flags"].update(flags)
            current["parse_status"].update([record["parse_status"]])
            for axis in ["spam", "phishing"]:
                value = record[axis + "_label"]
                current[axis + "_labels"].update(["null" if value is None else str(value)])
            current["lengths"].append(len(record["model_text"]))
            current["missing_fields"].update(k for k in ["subject", "body_text", "sender", "sender_address", "sender_domain", "recipient", "collection_period"] if record.get(k) is None)
            current["unavailable_features"].update(k for k,v in record["feature_availability"].items() if v == "unavailable")
            current["collection_periods"].update([record.get("collection_period") or "unknown"])
            model = record["model_text"]
            current["script_indicators"].update(k for k, present in [("cyrillic_present", any("\u0400" <= c <= "\u04ff" for c in model)),
                ("latin_ascii_present", any("a" <= c <= "z" or "A" <= c <= "Z" for c in model)),
                ("greek_present", any("\u0370" <= c <= "\u03ff" for c in model))] if present)
            # Empty texts never become duplicate keys, including the original-content key.
            if not text:
                continue
            exact_key = ("raw:" if record["raw_sha256"] else "fields:") + record["record_content_sha256"]
            normalized_key = sha_bytes(text.encode("utf-8"))
            for kind, key, index in [("exact_original", exact_key, exact), ("normalized_model_text", normalized_key, normalized)]:
                previous = index.get(key)
                if previous is None:
                    index[key] = rid
                else:
                    uf.union(previous, rid)
                    edges.append({"a": previous, "b": rid, "reason": kind, "key_sha256": key.split(":")[-1]})
        db.commit()
        print(f"quality: indexed {source}: {current['normalized_records']:,}", flush=True)
    ingest_seconds = time.perf_counter() - start
    anchors = sorted(normalized.values())
    near_start = time.perf_counter()
    index = MinHashLSH(threshold=params["lsh_threshold"], num_perm=params["num_perm"])
    cache = OrderedDict()

    def shingles(rid):
        if rid in cache:
            cache.move_to_end(rid)
            return cache[rid]
        text = db.execute("SELECT comparison FROM records WHERE id=?", (rid,)).fetchone()[0]
        n = params["shingle_chars"]
        value = {text[i:i+n] for i in range(len(text) - n + 1)}
        cache[rid] = value
        if len(cache) > 128:
            cache.popitem(last=False)
        return value

    searched = candidates = length_pass = comparisons = confirmed = connected_skipped = 0
    for rid in anchors:
        if metadata[rid]["length"] < params["near_min_chars"]:
            continue
        grams = shingles(rid)
        signature = MinHash(num_perm=params["num_perm"], seed=config["seed"], scheme="legacy", gpu_mode="disable")
        # Bounded permutation arrays even for very long MIME-derived text.
        batch = []
        for gram in sorted(grams):
            batch.append(gram.encode("utf-8"))
            if len(batch) == 1024:
                signature.update_batch(batch)
                batch.clear()
        if batch:
            signature.update_batch(batch)
        for other in sorted(index.query(signature)):
            candidates += 1
            if uf.find(rid) == uf.find(other):
                connected_skipped += 1
                continue  # Recorded confirmed paths already connect these originals.
            ratio = min(metadata[rid]["length"], metadata[other]["length"]) / max(metadata[rid]["length"], metadata[other]["length"])
            if ratio < params["length_ratio_min"]:
                continue
            length_pass += 1
            other_grams = shingles(other)
            if min(len(grams), len(other_grams)) / max(len(grams), len(other_grams)) < params["jaccard_min"]:
                continue
            comparisons += 1
            intersection = len(grams & other_grams)
            union = len(grams) + len(other_grams) - intersection
            jaccard = intersection / union
            if jaccard >= params["jaccard_min"]:
                confirmed += 1
                uf.union(rid, other)
                edges.append({"a": other, "b": rid, "reason": "near_model_text_5gram_jaccard", "jaccard": jaccard,
                              "length_ratio": ratio, "intersection_5grams": intersection, "union_5grams": union})
        index.insert(rid, LeanMinHash(signature))
        searched += 1
        if searched % 2000 == 0:
            print(f"near: {searched:,} unique long texts; {candidates:,} candidates; {confirmed:,} confirmed pairs", flush=True)
    cache.clear()
    near_seconds = time.perf_counter() - near_start
    components = defaultdict(list)
    for rid in sorted(metadata):
        components[uf.find(rid)].append(rid)
    by_record, groups = {}, {}
    for member_ids in components.values():
        gid = "dup:" + sha_bytes(canonical_bytes(sorted(member_ids)))
        members = [metadata[rid] for rid in member_ids]
        axes = conflict_axes(members)
        groups[gid] = {"duplicate_group_id": gid, "member_ids": member_ids, "members": [provenance(m) for m in members],
                       "size": len(member_ids), "conflict_axes": axes, "quarantine": bool(axes), "links": []}
        for rid in member_ids:
            by_record[rid] = gid
    for edge in edges:
        groups[by_record[edge["a"]]]["links"].append(edge)
    for gid, reason in config.get("quality_review", {}).get("groups", {}).items():
        if gid in groups:
            groups[gid].update(review_status="needs_review", review_reason=reason)
            for rid in groups[gid]["member_ids"]:
                metadata[rid]["flags"].append("needs_review")
                stats[metadata[rid]["source"]]["flags"].update(["needs_review"])
    exact_conflict_groups = near_conflict_groups = 0
    # Separately identify conflicts already supported by exact/normalized equality.
    equality = UnionFind()
    for rid in metadata:
        equality.add(rid)
    for edge in edges:
        if edge["reason"] != "near_model_text_5gram_jaccard":
            equality.union(edge["a"], edge["b"])
    equality_members = defaultdict(list)
    for rid in metadata:
        equality_members[equality.find(rid)].append(metadata[rid])
    equality_bad = {root for root, members in equality_members.items() if conflict_axes(members)}
    for group in groups.values():
        if group["quarantine"]:
            exact_conflict = any(equality.find(rid) in equality_bad for rid in group["member_ids"])
            group["quarantine_reason"] = "exact_or_normalized_label_conflict" if exact_conflict else "near_label_conflict"
            exact_conflict_groups += exact_conflict
            near_conflict_groups += not exact_conflict
        else:
            group["quarantine_reason"] = None
    lengths_summary = {}
    for source, current in stats.items():
        lengths = sorted(current.pop("lengths"))
        current["length_chars"] = {k: lengths[min(len(lengths)-1, int((len(lengths)-1)*q))] if lengths else None
                                   for k,q in [("min",0), ("p25",.25), ("median",.5), ("p75",.75), ("p95",.95), ("max",1)]}
        current["quality_excluded"] = sum(not m["eligible"] for m in metadata.values() if m["source"] == source)
        current["quarantine_records"] = sum(m["eligible"] and groups[by_record[rid]]["quarantine"] for rid,m in metadata.items() if m["source"] == source)
        current["processed_records"] = current["normalized_records"] - current["quality_excluded"] - current["quarantine_records"]
    duplicate_groups = [g for g in groups.values() if g["size"] > 1]
    cross_source = [g for g in duplicate_groups if len({m["source"] for m in g["members"]}) > 1]
    nazario_cross_year = [g for g in duplicate_groups if {m["subset"] for m in g["members"] if m["source"] == "nazario"} >= {"phishing-2024", "phishing-2025"}]
    report = {"status": "complete", "sources": stats, "normalized_records": len(metadata), "group_count": len(groups),
              "needs_review_groups": [gid for gid,g in groups.items() if g.get("review_status") == "needs_review"],
              "duplicate_group_count": len(duplicate_groups), "records_in_duplicate_groups": sum(g["size"] for g in duplicate_groups),
              "cross_source_group_count": len(cross_source), "nazario_2024_2025_group_count": len(nazario_cross_year),
              "exact_original_links": sum(e["reason"] == "exact_original" for e in edges),
              "normalized_model_text_links": sum(e["reason"] == "normalized_model_text" for e in edges),
              "quarantine_groups": exact_conflict_groups + near_conflict_groups,
              "exact_or_normalized_conflict_groups": exact_conflict_groups, "near_only_conflict_groups": near_conflict_groups,
              "quarantine_records": sum(s["quarantine_records"] for s in stats.values()),
              "quality_excluded_records": sum(s["quality_excluded"] for s in stats.values()),
              "processed_records": sum(s["processed_records"] for s in stats.values()),
              "largest_groups": [{"duplicate_group_id": g["duplicate_group_id"], "size": g["size"], "quarantine": g["quarantine"],
                                  "sources": dict(Counter(m["source"] for m in g["members"]))} for g in sorted(groups.values(), key=lambda g:(-g["size"],g["duplicate_group_id"]))[:10]],
              "near": {"status": "complete", "method": "MinHash/LSH candidates then exact Python string 5gram sets", **params,
                       "seed": config["seed"], "scheme": "datasketch legacy", "lsh_bands": index.b, "lsh_rows_per_band": index.r,
                       "unique_long_texts_indexed": searched, "candidate_pairs": candidates, "length_ratio_pass_pairs": length_pass,
                       "connected_candidate_pairs_skipped": connected_skipped,
                       "exact_jaccard_comparisons": comparisons, "confirmed_pairs": confirmed,
                       "seconds": near_seconds, "limitations": "Probabilistic candidate retrieval may miss near pairs; connected components need not have all-pairs similarity >= threshold. Short messages only exact/normalized. No URL/domain/sender-only grouping."},
              "timing_seconds": {"ingest_exact": ingest_seconds, "near": near_seconds, "total": time.perf_counter()-start}}
    return QualityResult(db, metadata, groups, by_record, report)
