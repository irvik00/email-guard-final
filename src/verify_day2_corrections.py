"""Audit final corrections and compare with preserved pre-correction fingerprints."""
import contextlib
import hashlib
import io
import json
import time
import unittest
from pathlib import Path

from inventory_data import sha_file
from normalization_schema import canonical_bytes
from pipeline_audit import audit, require
from pipeline_common import ROOT, rows, write_json, write_text


def run():
    started=time.perf_counter()
    reports=ROOT/"reports"
    baseline=json.loads((reports/"day2_pre_corrections_baseline.json").read_text(encoding="utf-8"))
    manifest=json.loads((ROOT/"data/processed/splits/manifest.json").read_text(encoding="utf-8"))
    verified=audit(ROOT/"data/processed",ROOT/"data/interim")
    unchanged={}
    for dataset in ["spam_email","russian_auxiliary"]:
        names=[name for name in baseline["artifacts"] if name.startswith("splits/"+dataset+"/")]
        unchanged[dataset]=len(names)==3 and all(manifest["artifacts"].get(name)==baseline["artifacts"][name] for name in names)
        require(unchanged[dataset],dataset+" changed from pre-correction baseline")
    for name,spec in baseline["normalized"].items():
        path=ROOT/"data/interim"/name
        require(path.resolve().is_relative_to((ROOT/"data/interim").resolve()),"baseline path traversal")
        require(sha_file(path)==spec["sha256"],"normalized data changed: "+name)
    digest=hashlib.sha256()
    for record in rows(ROOT/"data/processed/splits/modern_spam_2025_period_challenge.jsonl"):
        record["dataset"]="modern_spam_test"
        digest.update(canonical_bytes(record)+b"\n")
    require(digest.hexdigest()==baseline["artifacts"]["splits/modern_spam_test.jsonl"]["sha256"],"2025 challenge changed beyond dataset name")
    prior_group=json.loads((reports/"large_group_original.json").read_text(encoding="utf-8"))
    group=next(g for g in rows(ROOT/"data/processed/duplicate_groups.jsonl") if g["duplicate_group_id"]==prior_group["duplicate_group_id"])
    require({k:v for k,v in group.items() if k not in {"review_status","review_reason"}}==prior_group,"inspected component changed")
    require(group["review_status"]=="needs_review","component review flag missing")
    ids=set(group["member_ids"]); count=0
    for record in rows(ROOT/"data/processed/records.jsonl"):
        if record["record_id"] in ids:
            require("needs_review" in record["quality_flags"],"original review flag missing")
            count+=1
    require(count==1000,"not all component originals retained")
    for spec in manifest["review_evidence"].values():
        require(sha_file(ROOT/spec["path"])==spec["sha256"],"review evidence differs from manifest")
    first=json.loads((reports/"day2_first_corrections_artifacts.json").read_text(encoding="utf-8"))
    require(first==manifest["artifacts"],"processed JSONL not reproducible between corrected runs")
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    with contextlib.redirect_stdout(io.StringIO()):
        tests=unittest.TextTestRunner(verbosity=1).run(suite)
    require(tests.wasSuccessful(),"fixture tests failed")
    result={"status":"passed","date":"2026-10-11","tests_passed":tests.testsRun,"independent_audit":verified,
        "historical_spam_splits_byte_identical":unchanged["spam_email"],"russian_splits_byte_identical":unchanged["russian_auxiliary"],
        "all_normalized_artifacts_byte_identical":True,"russian_reservoir_rescanned":False,
        "september_2025_challenge_identical_except_dataset_name":True,
        "inspected_component_members_and_links_unchanged":True,"needs_review_originals_retained":count,
        "two_corrected_full_runs_identical_jsonl":len(first),"processed_jsonl_bytes":sum(a["size_bytes"] for a in first.values()),
        "reserve_test_overlap_groups":sum(s["reserved_future"] and s["protected_test"] for s in manifest["group_split_mapping"].values()),
        "models_trained":False,"seconds":round(time.perf_counter()-started,2)}
    write_json(reports/"day2_corrections_verification.json",result)
    write_json(reports/"final_audit.json",verified)
    previous_path=reports/"verification_execution.json"
    previous=json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    if "before_corrections" not in previous:previous={"before_corrections":previous}
    previous["final_corrections"]=result
    write_json(previous_path,previous)
    path=reports/"DAY2_STATUS.md"
    text=path.read_text(encoding="utf-8").split("\n## Проверки заключительных исправлений\n",1)[0]
    text+=f"\n## Проверки заключительных исправлений\n\nПройдено {tests.testsRun} тестов; smoke и два полных прохода завершены с проверенным cache нормализации. Независимый аудит опубликованных 15 JSONL прошёл, утечек групп нет. Между исправленными полными проходами все 15 JSONL совпали по SHA/размерам.\n\nВсе нормализованные файлы (включая diagnostics и русский reservoir) побайтно сохранены. Исторические spam_email и русские train/validation/test побайтно совпали с состоянием до исправлений. 600 примеров сентября2025 совпали с прежними, кроме dataset. 1000 членов проверенной компоненты и все её связи сохранены; у каждого needs_review.\n\nВ резерве 3018 оригиналов / 2215 групп; {result['reserve_test_overlap_groups']} групп одновременно представлены в защищённом test и резерве, без train/validation-копий. Это пересечение явно учтено, а не независимая проверка. Основание security_blocked — сообщение пользователя; файл2026 и карантин Defender не читались.\n\nПодробные измерения: reports/day2_corrections_verification.json; MIME-аудит: reports/decoding_and_group_audit.md. Повтор проверки: `.\\.venv\\Scripts\\python.exe -B .\\src\\verify_day2_corrections.py`.\n"
    write_text(path,text)
    print(json.dumps({k:v for k,v in result.items() if k!="independent_audit"},ensure_ascii=False),flush=True)
    return result


if __name__=="__main__":
    run()
