"""Freeze day-5 contract/membership from existing train ONLY."""
from collections import Counter
from datetime import datetime, timezone
import json
import sys

from day5_common import (ROOT, DOMAINS, Boundary, read_json, write_json, write_rows, sha_file,
                         digest, group_and_partition, proportional_sample, counts, preserved_files,
                         load_split, audit_parts, require, now)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    boundary = Boundary('prepare').install()
    config = read_json(ROOT/'config/day5_laya.json')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+digest(config)[:10]
    out = ROOT/'reports/day5/runs'/run_id
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'preservation_before.json',preserved_files())
    # Contract file exists before any data-dependent selection or model invocation.
    contract={'run_id':run_id,'created_utc':now(),'config':config,'validation_not_opened':True,
              'technical_decisions':[
                  'Frozen encoder and frozen action/escalation branch; train existing transformer decision head/type embedding/scorer only.',
                  'Smoke is first 3 steps of same optimizer trajectory; counts towards maximum 300.',
                  'Checkpoint selection excludes untrained initial reference; select among actual trained checkpoints.',
                  'Transitive union of global groups and original/clean exact/canonical equality before source/class split.',
                  'Conflicting-label components retained in membership, excluded from actual fit/dev/calibration subsets without relabeling.',
                  'Final Laya inference single record for honest per-record warm latency; frozen LR full-context and matched text measured separately.',
                  'SDK rounded scores are logged; NLL/Brier/calibration/threshold use full precision raw-logit softmax. Rounding discrepancy reported.'
              ],
              'code_sha256':{str(p.relative_to(ROOT).as_posix()):sha_file(p) for p in [ROOT/'src/prepare_day5.py',ROOT/'src/day5_common.py',ROOT/'src/normalization_schema.py']},
              'official_sources':['https://github.com/NandhaKishorM/laya','https://huggingface.co/convaiinnovations/laya','https://github.com/NandhaKishorM/laya/tree/main/notebooks'],
              'primary_source_note':'Installed laya0.3.20 agent/common are executable architecture reference; do not execute downloaded notebooks or adopt author training/calibration splits.'}
    write_json(out/'experiment_contract.json',contract)
    manifest=read_json(ROOT/'data/processed/splits/manifest.json')
    day4=read_json(ROOT/'config/day4_experiments.json')
    rows=[];fingerprints={}
    for domain in DOMAINS:
        data,fp=load_split(domain,'train',manifest,day4)
        rows.extend(data);fingerprints[domain]=fp
    require(len({r['record_id'] for r in rows})==len(rows),'Cross-domain ID collision')
    parts,conflicts=group_and_partition(rows,config['seed'])
    conflict_ids={gid for c in conflicts for gid in c['ids']}
    selected={'fit':[],'dev':[],'calibration':[]}
    for domain in DOMAINS:
        eligible=[r for r in parts['fit'] if r['domain']==domain and r['record_id'] not in conflict_ids]
        pair=min(config['selection']['fit_per_class'][domain],*(sum(r['spam_label']==label for r in eligible) for label in [0,1]))
        for label in [0,1]:
            pool=[r for r in eligible if r['spam_label']==label]
            selected['fit'].extend(sorted(pool,key=lambda r:digest([config['seed'],'fit-subset',r['record_id']]))[:pair])
        for part,n in [('dev',config['selection']['dev_per_domain']),('calibration',config['selection']['calibration_per_domain'])]:
            pool=[r for r in parts[part] if r['domain']==domain and r['record_id'] not in conflict_ids]
            selected[part].extend(proportional_sample(pool,n,part+'-subset',config['seed']))
    audit=audit_parts(parts)
    selected_ids={part:{r['record_id'] for r in data} for part,data in selected.items()}
    membership=[{k:r[k] for k in ['record_id','duplicate_group_id','inner_group_id','inner_part','domain','source','spam_label','label_status']}
                | {'selected':r['record_id'] in selected_ids[r['inner_part']], 'conflicting_group':r['record_id'] in conflict_ids,
                   'clean_text_sha256':digest(r['text']),'original_text_sha256':digest(r['original_model_text'])} for r in rows]
    write_rows(out/'membership.jsonl',membership)
    for part,data in selected.items():
        write_rows(out/f'private_{part}_inputs.jsonl',data)
    data_manifest={'run_id':run_id,'created_utc':now(),'input_fingerprints':fingerprints,
                   'split_manifest_sha256':sha_file(ROOT/'data/processed/splits/manifest.json'),
                   'full_inner_counts':{part:counts(data) for part,data in parts.items()},
                   'selected_counts':{part:counts(data) for part,data in selected.items()},
                   'conflicting_groups':conflicts,'audit':audit,
                   'selection_sha256':{part:sha_file(out/f'private_{part}_inputs.jsonl') for part in selected},
                   'membership_sha256':sha_file(out/'membership.jsonl'),
                   'cleaning_changed_train_rows':sum(bool(r['cleaning_changes']) for r in rows),
                   'not_a_new_global_split':True,'ru_status':'source_labels_used_provisionally'}
    write_json(out/'data_manifest.json',data_manifest)
    write_json(out/'prepare_accesses.json',boundary.report())
    write_json(ROOT/'reports/day5/current_run.json',{'run_id':run_id,'report_path':str(out.relative_to(ROOT).as_posix()),'stage':'prepared'})
    print(json.dumps({'run_id':run_id,'full_inner_counts':data_manifest['full_inner_counts'],
                      'selected_counts':data_manifest['selected_counts'],'conflicting_groups':len(conflicts),
                      'validation_not_opened':True},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
