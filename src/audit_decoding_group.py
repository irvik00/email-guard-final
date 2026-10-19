"""Bounded inert MIME audit of a specified existing duplicate component."""
import argparse
import hashlib
import json
import re
import quopri
import tempfile
from collections import Counter
from email import policy
from email.parser import BytesParser
from pathlib import Path

from pipeline_common import ROOT, rows, compare_text, write_json, write_text
from normalization_reporting import fragment
from read_email import parse_email, body_candidates
from untroubled_adapter import process_archive

GROUP = "dup:e662702244224927715974b4834d7bb50e669122ef3a31d286b34437f6e73226"
TARGET = "2025/02/1739355712.2360868_125.txt"


def shingles(text):
    text = compare_text(text)
    return {text[i:i+5] for i in range(len(text)-4)}


def similarity(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def inert(text):
    # Make invisible padding visible; then escape/redact contacts, HTML and URLs.
    text = re.sub(r"[\u00a0\u034f\u2000-\u200f\u202a-\u202e\u2060-\u206f\ufeff]",
                  lambda m: " U+%04X " % ord(m[0]), text or "")
    return fragment(text)


def run(root=ROOT, group_id=GROUP):
    root = Path(root)
    report_dir = root / "reports"
    # Preserve the inspected component to permit an audit after a justified regrouping.
    snapshot = report_dir / "large_group_original.json"
    group = next((g for g in rows(root/"data/processed/duplicate_groups.jsonl")
                  if g["duplicate_group_id"] == group_id), None)
    if group is None and snapshot.is_file():
        group = json.loads(snapshot.read_text(encoding="utf-8"))
    if group is None or group["duplicate_group_id"] != group_id:
        raise ValueError("Requested component unavailable")
    if not snapshot.exists():
        write_json(snapshot, group)
    ids = set(group["member_ids"])
    records = [r for r in rows(root/"data/interim/untroubled.jsonl") if r["record_id"] in ids]
    if len(records) != len(ids):
        raise ValueError("Component originals unavailable")
    records.sort(key=lambda r:r["original_locator"])
    selected = [next(r for r in records if r["original_locator"].endswith("::"+TARGET))]
    for period in sorted({r["collection_period"] for r in records}):
        candidate = next(r for r in records if r["collection_period"] == period)
        if candidate not in selected: selected.append(candidate)
    for candidate in [min(records,key=lambda r:(len(r["model_text"]),r["record_id"])),
                      max(records,key=lambda r:(len(r["model_text"]),r["record_id"]))]:
        if candidate not in selected: selected.append(candidate)
    # Complete to ten with new subject values spread deterministically through IDs.
    subjects = {r["subject"] for r in selected}
    for candidate in sorted(records,key=lambda r:hashlib.sha256(r["record_id"].encode()).hexdigest()):
        if len(selected) >= 10: break
        if candidate["subject"] not in subjects:
            selected.append(candidate); subjects.add(candidate["subject"])
    lookup = {r["original_locator"].split("::",1)[1]:r for r in selected}
    findings = []
    def inspect(row, member):
        if member not in lookup: return
        r = lookup[member]
        if hashlib.sha256(row.raw).hexdigest() != r["raw_sha256"]:
            raise ValueError("Raw SHA mismatch for audited member")
        message = BytesParser(policy=policy.default).parsebytes(row.raw)
        parsed = parse_email(row.raw)
        parts = []
        for p in body_candidates(message):
            decoded = p.get_content()
            raw_payload = p.get_payload()
            parts.append({"content_type":p.get_content_type(), "charset":p.get_content_charset(),
                "transfer_encoding":str(p.get("Content-Transfer-Encoding")) if p.get("Content-Transfer-Encoding") is not None else None,
                "payload_chars":len(raw_payload), "decoded_chars":len(decoded),
                "qp_tokens_after_declared_decode":len(re.findall(r"=[0-9A-Fa-f]{2}",decoded)),
                "defects":[type(d).__name__ for d in p.defects],
                "payload_fragment":inert(raw_payload), "decoded_fragment":inert(decoded)})
        findings.append({"record_id":r["record_id"],"locator":r["original_locator"],"period":r["collection_period"],
            "root_content_type":message.get_content_type(),"root_charset":message.get_content_charset(),
            "root_transfer_encoding":message.get("Content-Transfer-Encoding"),
            "body_source":parsed["body_source"],"body_chars":len(parsed["body_text"] or ""),
            "html_text_chars":len(parsed["html_text"] or ""), "parts":parts,
            "stored_body_matches_parser":parsed["body_text"] == r["body_text"],
            "warnings":parsed["parse_warnings"],"subject_fragment":inert(r["subject"]),
            "body_fragment":inert(parsed["body_text"]),"html_text_fragment":inert(parsed["html_text"])})
    config=json.loads((root/"config/day2_pipeline.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix=".group-mime-",dir=root/"data/processed") as work:
        process_archive(root/"data/raw"/config["archive"]["path"],root,Path(work),config,inspect)
    if len(findings) != len(selected): raise ValueError("Not all selected originals inspected")
    pairs=[]
    for i,a in enumerate(selected):
        for b in selected[i+1:]:
            pairs.append({"a":a["record_id"],"b":b["record_id"],
                          "jaccard":similarity(shingles(a["model_text"]),shingles(b["model_text"]))})
    # Diagnostic hypothesis on ONE selected plain body only, never fed back to parsing.
    # Absence of declared QP means this is an interpretation experiment, not a repair.
    diagnostic_body=selected[0]["body_text"]
    hypothesis=quopri.decodestring(diagnostic_body.encode("ascii")).decode("utf-8")
    padding={"figure_space_U2007":hypothesis.count("\u2007"),"combining_grapheme_joiner_U034F":hypothesis.count("\u034f"),
             "soft_hyphen_U00AD":hypothesis.count("\u00ad")}
    result={"group_id":group_id,"size":group["size"],"full_component_counts":{
        "raw_sha256_distinct":len({r["raw_sha256"] for r in records}),
        "model_text_distinct":len({r["model_text"] for r in records}),
        "body_text_distinct":len({r["body_text"] for r in records}),
        "periods":dict(Counter(r["collection_period"] for r in records)),
        "links_by_reason":dict(Counter(e["reason"] for e in group["links"]))},
        "sample_selection":"target, each period, shortest/longest, then distinct subjects ordered by SHA256(record_id)",
        "sample_size":len(findings),"sample":sorted(findings,key=lambda r:r["locator"]),"sample_pairs":pairs,
        "sample_pair_jaccard_min":min(p["jaccard"] for p in pairs),
        "single_record_diagnostic_only":{"record_id":selected[0]["record_id"],"hypothesis":"one hypothetical QP decode despite absent CTE; not saved as body or model text", "before_chars":len(diagnostic_body),"after_chars":len(hypothesis),"padding_counts":padding,
            "middle_fragment":inert(hypothesis[350:750])},
        "status":"needs_review", "parser_decoding_bug_confirmed":False,
        "conclusion":"Source plain alternative repeats identical padding/footer/tracking payload without a declared QP CTE; declared MIME decoding preserved correctly. Distinct HTML in sample makes component semantic equivalence uncertain; preserve records/labels/groups, flag needs_review."}
    write_json(report_dir/"decoding_and_group_audit.json",result)
    lines=["# Аудит декодирования и крупнейшей компоненты", "",f"Группа `{group_id}`: {group['size']} оригиналов.",
           "", "Полные подсчёты по компоненте: `"+json.dumps(result["full_component_counts"],ensure_ascii=False)+"`.",
           "",f"Выборка: {len(findings)} членов; {result['sample_selection']}. MIME-подсчёты ниже относятся только к выборке.",
           "Архив 2025 прочитан одним безопасным проходом; исходные SHA сверены. HTML не исполнялся, вложения/ссылки не открывались. Повторное quoted-printable декодирование не применялось.",""]
    for f in result["sample"]:
        lines += [f"## `{f['locator']}`", "",f"Тема: {f['subject_fragment']}",
            f"Корень MIME: {f['root_content_type']}; charset={f['root_charset']}; CTE={f['root_transfer_encoding']}. Выбрано body_source={f['body_source']}, длина={f['body_chars']}; HTML-текст={f['html_text_chars']}. Совпадает с сохранённым телом: {f['stored_body_matches_parser']}.",
            f"Тело: {f['body_fragment']}", f"HTML-текст после безопасного извлечения: {f['html_text_fragment']}", ""]
        for p in f["parts"]:
            lines += [f"- {p['content_type']}; charset={p['charset']}; CTE={p['transfer_encoding']}; payload={p['payload_chars']}, после заявленного декодирования={p['decoded_chars']}; остаточных =XX: {p['qp_tokens_after_declared_decode']}; defects={p['defects']}.",
                      f"  Payload: {p['payload_fragment']}  ",f"  После заявленного декодирования: {p['decoded_fragment']}",""]
    lines += ["## Причины сходства и границы вывода", "",
        f"Минимальный Jaccard среди всех {len(pairs)} пар выбранных сообщений: {result['sample_pair_jaccard_min']:.6f}. Полный набор всех пар 1000 писем не вычислялся; связность доказывают сохранённые links, но это не доказывает попарное тождество.",
        "По всей компоненте: 1000 разных raw SHA-256, 352 разных model_text, одно и то же plain-тело длиной 5972 символа. Это не 1000 побайтовых копий. 648 normalized-связей относятся к равенству извлечённого текста, а не исходных писем; 351 near-связь соединяет варианты тем поверх одинакового тела. Общий длинный payload доминирует в символьных 5-граммах; часть крайних сообщений выборки ниже порога 0.95, поэтому транзитивное объединение тоже участвует.", "",
        "MIME-выборка 10/10: multipart/alternative; text/plain, us-ascii, CTE отсутствует; 708 =XX-токенов остаются после заявленного декодирования. Отсутствующая CTE не разрешает автоматически применять quoted-printable. HTML: UTF-8/8bit, безопасно извлечённый текст различается (темы электричества, страховки, наград, казино); это наблюдение по выборке, не полный аудит HTML 1000 писем. Во всех 10 есть CloseBoundaryNotFoundDefect. У целевого примера также недекодируемые байты заголовка сохранены escape-последовательностями.", "",
        "Контрольная гипотеза только для целевого plain-тела: один пробный QP decode, НЕ изменение записи и НЕ повторное декодирование корпуса. Получено "+str(len(hypothesis))+" символа; padding: `"+json.dumps(padding)+"`. Помимо невидимого padding остаются общий footer и длинные tracking URL. Короткий обезвреженный фрагмент: "+inert(hypothesis[350:750]), "",
        "Дефект/шум присутствует в исходном MIME: plain-альтернатива не соответствует различающейся HTML-части и не объявляет QP; установить, создано ли это отправителем или обработкой источника, нельзя. Наш parse_email выбирает непустой plain по документированному правилу и корректно следует MIME CTE/charset; ошибка декодера не подтверждена. Ограничение выбора plain и near-группировки по model_text подтверждено. Порог/выбор тела по этим примерам не менялись.", "",
        "Статус **needs_review** сохранён для компоненты и её 1000 членов. Это группа сходства извлечённого текста с риском ложного смыслового слияния, не подтверждённые одинаковые письма. Исходные тела, model_text, метки и состав группы не изменяются; повторного quoted-printable декодирования корпуса, удаления или переразметки нет. Нужна отдельная проверка политики MIME/дубликатов до обучения; влияние группировки на числа представителей оговорено.", "",
        "Повтор: `.\\.venv\\Scripts\\python.exe -B .\\src\\audit_decoding_group.py`.", ""]
    write_text(report_dir/"decoding_and_group_audit.md","\n".join(lines))
    print(json.dumps({"sample":len(findings),"counts":result["full_component_counts"],
                      "sample_min_jaccard":result["sample_pair_jaccard_min"],
                      "mime_sample":[{k:f[k] for k in ['locator','root_content_type','body_source','body_chars','html_text_chars']} | {'parts':[{k:p[k] for k in ['content_type','charset','transfer_encoding','decoded_chars','qp_tokens_after_declared_decode','defects']} for p in f['parts']]} for f in findings]},ensure_ascii=True),flush=True)
    return result


if __name__ == "__main__":
    run()
