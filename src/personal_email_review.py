"""Validate an explicit local personal-email manifest and parse .eml statically; NO inference."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import sys

from day5_common import ROOT, Boundary, read_json, load_split, compare_text, digest, require, write_json
from read_email import parse_email
from normalization_schema import make_model_text

FIELDS=['local_path','spam_label','category','consent','label_reason','approximate_date','language_self_reported','review_status']
MAX_BYTES=20*1024*1024


def manifest_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        require(reader.fieldnames==FIELDS,'CSV header must match documented eight fields exactly')
        return list(reader)


def resolve_entries(rows,root=ROOT):
    root=Path(root).resolve();base=root/'data/personal_review';entries=[];seen=set()
    for number,row in enumerate(rows,2):
        require(None not in row and all(isinstance(v,str) for v in row.values()),f'Malformed CSV row {number}')
        path=(root/row['local_path']).resolve()
        require(path.is_relative_to(base) and path.suffix.lower()=='.eml',f'Row {number}: local .eml must stay inside data/personal_review')
        require(path not in seen,f'Row {number}: repeated file path');seen.add(path)
        require(row['spam_label'] in {'0','1','unknown'},f'Row {number}: label must be literal 0/1/unknown')
        require(row['consent'] in {'yes','no','unknown'},f'Row {number}: invalid consent')
        require(row['review_status'] in {'pending','reviewed','needs_review'},f'Row {number}: invalid review_status')
        require(row['spam_label']=='unknown' or bool(row['label_reason'].strip()),f'Row {number}: known label requires label_reason')
        entries.append((path,row))
    return entries


def parse_static(path):
    require(path.stat().st_size<=MAX_BYTES,'Email exceeds same 20 MiB input limit as baseline')
    raw=path.read_bytes();require(len(raw)<=MAX_BYTES,'Email grew beyond size limit')
    parsed=parse_email(raw)
    text,_=make_model_text(parsed.get('subject'),parsed.get('body_text'))
    # Never return full text/addresses/links/attachment filenames into general reports.
    return {'raw_sha256':hashlib.sha256(raw).hexdigest(),'model_text_hash':digest(text),
            'canonical_text_hash':digest(compare_text(text)), 'size_bytes':len(raw),
            'model_text_characters':len(text),'parse_warning_count':len(parsed.get('parse_warnings',[])),
            'attachment_count':len(parsed.get('attachments',[])), 'usable_text':bool(text.strip())}


def inspect_entries(entries,train_hashes=None):
    metadata=[];groups=defaultdict(list)
    for i,(path,row) in enumerate(entries):
        item=parse_static(path)
        item.update(index=i+1,spam_label=row['spam_label'],review_status=row['review_status'],consent=row['consent'])
        metadata.append(item);groups[item['canonical_text_hash']].append(item)
    conflicts=[{'record_indices':[r['index'] for r in group],'labels':sorted({r['spam_label'] for r in group if r['spam_label']!='unknown'})}
               for group in groups.values() if len({r['spam_label'] for r in group if r['spam_label']!='unknown'})>1]
    overlaps=[r['index'] for r in metadata if train_hashes and r['canonical_text_hash'] in train_hashes]
    return {'status':'templates_ready' if not entries else 'parsed_pending_purpose_and_human_review',
            'records':len(entries),'labels':dict(Counter(r['spam_label'] for r in metadata)),
            'consent':dict(Counter(r['consent'] for r in metadata)),
            'canonical_groups':len(groups),'duplicate_groups':sum(len(g)>1 for g in groups.values()),
            'conflicting_label_groups':conflicts,'allowed_train_overlap_record_indices':overlaps,
            'unknown_label_is_not_ham':True,'model_inference_performed':False,'training_performed':False,
            'no_cloud':True,'content_in_report':False,'metadata_only':metadata}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'docs/personal_email_manifest_template.csv')
    parser.add_argument('--purpose',choices=['unassigned','review_dev','held_out_diagnostic'],default='unassigned')
    args=parser.parse_args()
    # The manifest is explicit, not a directory scan. For data/ manifests use a single resolved allowlist entry.
    manifest=args.manifest.resolve()
    require(not manifest.is_relative_to(ROOT/'data') or manifest.is_relative_to(ROOT/'data/personal_review'),'Manifest outside personal directory denied')
    # Resolve paths without opening any email; then install final exact allowlist.
    records=manifest_rows(manifest);entries=resolve_entries(records)
    boundary=Boundary('personal',personal_paths=([manifest] if manifest.is_relative_to(ROOT/'data') else [])+[p for p,_ in entries]).install()
    hashes=set()
    if entries:
        prepared=read_json(ROOT/'data/processed/splits/manifest.json');cfg=read_json(ROOT/'config/day4_experiments.json')
        for domain in ['spam_email','russian_auxiliary']:
            rows,_=load_split(domain,'train',prepared,cfg)
            hashes.update(digest(compare_text(r['text'])) for r in rows)
    result=inspect_entries(entries,hashes);result['purpose']=args.purpose
    result['purpose_note']='No scores today, including when a purpose is supplied; purpose choice belongs to user before future evaluation.'
    result['access_log']=boundary.report()
    write_json(ROOT/'reports/day5/personal_review_readiness.json',result)
    print(__import__('json').dumps({k:v for k,v in result.items() if k not in {'metadata_only','access_log'}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
