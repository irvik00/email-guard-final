"""Representative selection, declared benchmark roles, and global group priorities."""
from collections import Counter, defaultdict
from pipeline_common import stable_order

PRIORITY = {"train": 1, "validation": 2, "reserve": 2.5, "test": 3}
ROLES = {"spam_email": "historical anti-spam benchmark; source metrics required",
         "russian_auxiliary": "separate text-only auxiliary experiment; artificial sampled balance",
         "modern_spam_train_candidates": "positive spam candidates for a separate future experiment",
         "modern_spam_test": "requested September 2026 positive-only challenge; security_blocked, no replacement",
         "modern_spam_2025_period_challenge": "protected September 2025 positive-only period challenge; not the 2026 test",
         "phishing_train_candidates": "positive phishing candidates; vetted negatives still missing",
         "phishing_positive_test": "protected single-class Nazario 2025 phishing challenge"}


def choose_rep(members, predicate, axis):
    suitable = [m for m in members if predicate(m) and m[axis] is not None and m["eligible"]]
    if not suitable:
        return None
    return min(suitable, key=lambda m: (m["parse_status"] != "ok", not m["has_headers"], m["record_id"]))["record_id"]


def allocate_counts(n, fractions):
    counts = {k: int(n*v) for k,v in fractions.items()}
    remainder = n - sum(counts.values())
    for key in sorted(fractions, key=lambda k: (-(n*fractions[k]-counts[k]), -PRIORITY[k]))[:remainder]:
        counts[key] += 1
    return counts


def assign_benchmark(candidates, metadata, fractions, seed, purpose):
    strata = defaultdict(list)
    for gid, rid in candidates.items():
        meta = metadata[rid]
        strata[(meta["source"], meta["spam"])].append((gid,rid))
    result = []
    for stratum, items in sorted(strata.items()):
        ordered = sorted(items, key=lambda x: (stable_order(seed, purpose + repr(stratum), x[0]), x[0]))
        counts = allocate_counts(len(ordered), fractions)
        cursor = 0
        for split in ["test", "validation", "train"]:
            for gid, rid in ordered[cursor:cursor+counts[split]]:
                result.append({"dataset": purpose, "split": split, "group_id": gid, "record_id": rid, "protected": False})
            cursor += counts[split]
    return result


def assignment_counts(assignments, metadata):
    result = {}
    for dataset in ROLES:
        entries = [a for a in assignments if a["dataset"] == dataset]
        splits = ["train", "validation", "test"] if dataset in {"spam_email", "russian_auxiliary"} else ["test" if dataset.endswith(("test", "challenge")) else "train"]
        result[dataset] = {"role": ROLES[dataset], "splits": {}}
        for split in splits:
            selected = [metadata[a["record_id"]] for a in entries if a["split"] == split]
            axis = "phishing" if dataset.startswith("phishing") else "spam"
            result[dataset]["splits"][split] = {"records": len(selected), "groups": len(selected),
                "class_counts": dict(Counter("null" if m[axis] is None else str(m[axis]) for m in selected)),
                "source_counts": dict(Counter(m["source"] for m in selected))}
    return result


def build_splits(quality, config):
    metadata, groups, by_record = quality.metadata, quality.groups, quality.by_record
    proposals, protected, representative_candidates = [], set(), {}
    modern_periods = set(config["splits"]["modern_test_periods"])
    challenge_periods = set(config["splits"]["modern_period_challenge_periods"])
    train_periods = set(config["splits"]["modern_train_periods"])
    reserve_periods = set(config["splits"]["modern_reserve_periods"])
    reserve_ids = sorted(rid for rid,m in metadata.items() if m["source"] == "untroubled" and m["collection_period"] in reserve_periods and m["eligible"] and not groups[by_record[rid]]["quarantine"])
    reserved = {by_record[rid] for rid in reserve_ids}
    for gid, group in sorted(groups.items()):
        if group["quarantine"]:
            continue
        members = [metadata[rid] for rid in group["member_ids"]]
        for dataset, predicate, axis in [
            ("phishing_positive_test", lambda m: m["source"] == "nazario" and m["source_path"].endswith(config["splits"]["protected_nazario_file"]), "phishing"),
            ("modern_spam_test", lambda m: config["splits"]["modern_test_status"] != "security_blocked" and m["source"] == "untroubled" and m["collection_period"] in modern_periods, "spam"),
            ("modern_spam_2025_period_challenge", lambda m: m["source"] == "untroubled" and m["collection_period"] in challenge_periods, "spam")]:
            rid = choose_rep(members, predicate, axis)
            if rid:
                proposals.append({"dataset": dataset, "split": "test", "group_id": gid, "record_id": rid, "protected": True})
                protected.add(gid)
    for dataset, predicate, axis in [
        ("spam_email", lambda m: m["source"] in {"spamassassin", "lingspam"}, "spam"),
        ("russian_auxiliary", lambda m: m["source"] == "russian_auxiliary", "spam"),
        ("modern_spam_train_candidates", lambda m: m["source"] == "untroubled" and m["collection_period"] in train_periods, "spam"),
        ("phishing_train_candidates", lambda m: m["source"] == "nazario" and m["source_path"].endswith("phishing-2024.txt"), "phishing")]:
        candidates = {}
        for gid, group in sorted(groups.items()):
            if group["quarantine"]:
                continue
            rid = choose_rep([metadata[r] for r in group["member_ids"]], predicate, axis)
            if rid:
                candidates[gid] = rid
        representative_candidates[dataset] = len(candidates)
        if dataset in {"spam_email", "russian_auxiliary"}:
            # Protected groups are reserved before random/stratified assignment.
            remaining = {gid:rid for gid,rid in candidates.items() if gid not in protected | reserved}
            proposals.extend(assign_benchmark(remaining, metadata, config["splits"]["fractions"], config["seed"], dataset))
            for gid in sorted(set(candidates) & (protected | reserved)):
                proposals.append({"dataset": dataset, "split": "train", "group_id": gid, "record_id": candidates[gid], "protected": False, "pre_reserved": True})
        else:
            proposals.extend({"dataset": dataset, "split": "train", "group_id": gid, "record_id": rid, "protected": False} for gid,rid in sorted(candidates.items()))
    # Entire future groups are frozen, including earlier/cross-source copies.
    # A protected test may also occur in the reserve; no train/validation may.
    global_roles = {gid:"reserve" for gid in reserved}
    for proposal in proposals:
        gid, split = proposal["group_id"], proposal["split"]
        if PRIORITY[split] > PRIORITY.get(global_roles.get(gid), 0):
            global_roles[gid] = split
    final, suppressed = [], []
    for proposal in sorted(proposals, key=lambda a:(a["dataset"],a["split"],a["group_id"],a["record_id"])):
        if PRIORITY[proposal["split"]] < PRIORITY[global_roles[proposal["group_id"]]]:
            suppressed.append({**proposal, "reason": "protected_test_group" if proposal["group_id"] in protected else "frozen_future_reserve_group" if proposal["group_id"] in reserved else "global_test_validation_train_priority", "retained_group_role": global_roles[proposal["group_id"]]})
        else:
            final.append(proposal)
    challenges = defaultdict(set)
    for proposal in final:
        if proposal["protected"]:
            challenges[proposal["group_id"]].add(proposal["dataset"])
    cross_challenge = sorted(gid for gid, names in challenges.items() if len(names) > 1)
    for proposal in final:
        proposal["cross_challenge_overlap"] = proposal["group_id"] in cross_challenge
    train = [a for a in final if a["dataset"] == "spam_email" and a["split"] == "train"]
    class_counts = Counter(metadata[a["record_id"]]["spam"] for a in train)
    folds = min(config["splits"]["cv_folds"], min((class_counts.get(label,0) for label in [0,1]), default=0))
    cv = {}
    fold_class_counts = {}
    if folds >= 2:
        strata = defaultdict(list)
        for a in train:
            meta = metadata[a["record_id"]]
            strata[(meta["spam"], meta["source"])].append(a["group_id"])
        class_offsets = Counter()
        for key, gids in sorted(strata.items()):
            for i, gid in enumerate(sorted(gids, key=lambda x: stable_order(config["seed"],"cv"+repr(key),x))):
                cv[gid] = (class_offsets[key[0]] + i) % folds
            class_offsets[key[0]] += len(gids)
        fold_class_counts = {str(fold): dict(Counter(str(metadata[a["record_id"]]["spam"]) for a in train if cv[a["group_id"]] == fold)) for fold in range(folds)}
    else:
        folds = 0
    group_mapping = {gid: {"role": global_roles.get(gid, "unassigned"), "protected_test": gid in protected, "reserved_future": gid in reserved,
                           "cross_challenge_overlap": gid in cross_challenge, "assignments": []} for gid in sorted(groups)}
    for a in final:
        group_mapping[a["group_id"]]["assignments"].append({k:a[k] for k in ["dataset", "split", "record_id"]})
    before, after = assignment_counts(proposals, metadata), assignment_counts(final, metadata)
    class_checks = {}
    for benchmark in ["spam_email", "russian_auxiliary"]:
        class_checks[benchmark] = {split: {"both_classes": {"0", "1"} <= set(values["class_counts"]),
                                                 "explanation": "group-aware representative counts, no oversampling"}
                                  for split,values in after[benchmark]["splits"].items()}
    report = {"status": ("partial_security_blocked" if config["splits"]["modern_test_status"] == "security_blocked" else "complete") if all(c["both_classes"] for b in class_checks.values() for c in b.values()) else "partial_insufficient_classes",
              "seed": config["seed"], "fractions_requested": config["splits"]["fractions"], "modern_test_periods": sorted(modern_periods),
              "modern_test_period_basis": config["splits"]["modern_test_period_basis"], "protected_groups": len(protected),
              "modern_test_status": config["splits"]["modern_test_status"],
              "modern_test_archive": config["splits"]["modern_test_archive"],
              "modern_test_block_reason": config["splits"]["modern_test_block_reason"],
              "modern_period_challenge_periods": sorted(challenge_periods), "modern_train_periods": sorted(train_periods),
              "modern_reserve": {"dataset":"modern_spam_future_reserve", "role":"frozen; never training or setting selection", "records":len(reserve_ids), "groups":len(reserved), "period_counts":dict(Counter(metadata[rid]["collection_period"] for rid in reserve_ids)), "path":"reserves/modern_spam_future_reserve.jsonl" if reserve_ids else None},
              "cross_challenge_overlap_groups": cross_challenge, "representative_candidates": representative_candidates,
              "before_global_priority": before, "datasets": after, "suppressed_assignments": len(suppressed),
              "suppressed_by_dataset": dict(Counter(a["dataset"] for a in suppressed)),
              "nazario_2024_excluded_by_test_group": sum(metadata[a["record_id"]]["source"] == "nazario" and a["dataset"] == "phishing_train_candidates" for a in suppressed),
              "class_presence_checks": class_checks, "cv": {"folds": folds, "records": len(cv), "fold_class_counts": fold_class_counts,
              "scope": "spam_email/train only", "reason": "five folds if each class has >=5 independent groups; else reduced"},
              "pending": ["security_blocked_modern_spam_test_2026_09", "pending_teacher_test", "pending_category_labels", "pending_fresh_verified_ham", "pending_verified_phishing_negatives", "pending_group_decoding_review", "pending_manual_acceptance_of_corrections"]}
    return {"assignments": final, "suppressed": suppressed, "group_mapping": group_mapping, "cv_fold_mapping": cv, "reserve_record_ids":reserve_ids, "report": report}
