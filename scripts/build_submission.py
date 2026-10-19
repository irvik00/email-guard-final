"""Explicit allowlist packaging: CPU code/E0 archive plus separate verified Laya files."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.contract import contract,sha_file
from scripts.prepare_model_artifacts import prepare

DOCS=['README.md','MODEL_CARD.md','MODEL_ARTIFACTS.md','PRESENTATION_HANDOFF.md','DEFENSE_QA.md','RELEASE_STATUS.md']
DIRECT=['Dockerfile','.dockerignore','compose.yaml','.env.example','requirements-app.in','requirements-app.lock.txt',
        'requirements-app-laya.lock.txt','requirements-docker.lock.txt','requirements.txt','requirements-lock.txt',
        'requirements-decision.txt','requirements-decision-lock.txt']
SCRIPTS=['prepare_model_artifacts.py','prepare_docker_dependencies.py','smoke_release.py','build_submission.py',
         'export_final_tables.py','build_app_release.py','verify_app.py','audit_app.py','audit_submission.py']
REPRO=['DAY3_BASELINES.md','DAY4_EXPERIMENTS.md','DAY5_REPRODUCTION.md','EXTRA_EXPERIMENTS_REPRODUCTION.md']


def members():
    paths=[ROOT/p for p in DOCS+DIRECT]
    paths += [p for folder in ('app','src','config') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT/'scripts'/p for p in SCRIPTS]
    paths += [ROOT/'tests'/p for p in ('test_app.py','test_release.py','test_read_email.py','test_baselines.py','test_day5.py')]
    paths += [ROOT/'docs'/p for p in REPRO]
    paths += [p for p in (ROOT/'deployment').rglob('*') if p.is_file()]
    paths += [p for p in (ROOT/'examples/app').glob('*') if p.is_file()]
    paths += [ROOT/'vendor/laya-0.3.20-py3-none-any.whl']
    paths += [ROOT/'reports/release'/p for p in ('FINAL_TABLES.md','final_metrics.json','final_metrics.csv','synthetic_reference.json','runtime_probe.json')]
    paths += [ROOT/'reports/release'/p for p in ('unit_app.txt','unit_release.txt','initial_relocation_failures.json')]
    paths += [p for p in (ROOT/'reports/release/smoke_runs').glob('*.json') if p.is_file()]
    paths += [ROOT/'reports/app/UI_review_synthetic.png']
    paths += [ROOT/'reports/final_evaluation/runs/20261017T132144Z-d663811e/FINAL_EVALUATION_PROTOCOL.json']
    return sorted(set(paths))


def stage(dest):
    dest=dest.resolve()
    if not dest.is_relative_to(ROOT/'release'):raise ValueError('Submission must stay within workspace release/')
    dest.mkdir(parents=True,exist_ok=True)
    for src in members():
        if not src.is_file():raise FileNotFoundError(src.relative_to(ROOT))
        target=dest/src.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha_file(target)==sha_file(src):continue
        shutil.copyfile(src,target)
    prepare(ROOT,dest/'artifacts',['E0'])
    # Separately supplied weights never enter the code/CPU archive.
    laya=ROOT/'release/email_guard_laya_artifacts'
    prepare(ROOT,laya,['Laya_day5'])
    expected={p.relative_to(ROOT).as_posix() for p in members()}
    expected.update('artifacts/'+rel for rel in contract()['E0']['files'])
    return expected,laya


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage-only',action='store_true');a=p.parse_args()
    dest=ROOT/'release/email_guard_submission';expected,laya=stage(dest)
    if a.stage_only:
        print(json.dumps({'status':'staged','files':len(expected),'directory':str(dest)}));return
    # Remove only caches created by our relocation tests, after checking targets.
    caches=[dest/'.cache']+list(dest.rglob('__pycache__'))
    for cache in caches:
        if cache.exists():
            resolved=cache.resolve()
            if not resolved.is_relative_to(dest.resolve()):raise ValueError('Unsafe cache cleanup')
            shutil.rmtree(resolved)
    actual={p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
    extras=actual-expected-{'SUBMISSION_MANIFEST.json','SHA256SUMS.txt'}
    if extras:raise ValueError('Unexpected files in submission; inspect before packaging: '+repr(sorted(extras)))
    # Never archive runtime caches/extra files created during relocation tests.
    file_specs={rel:{'sha256':sha_file(dest/rel),'bytes':(dest/rel).stat().st_size} for rel in sorted(expected)}
    manifest={'version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'files':file_specs,
        'private_or_raw_data_included':False,'research_environments_included':False,'large_laya_weights_included':False,
        'cpu_model_included':'E0, unchanged saved bundle','docker_execution_verified':False,
        'large_weights_separate':'../email_guard_laya_artifacts','schema':'relative path, byte size, SHA-256; manifest itself excluded from its own hashes'}
    (dest/'SUBMISSION_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (dest/'SHA256SUMS.txt').write_text('\n'.join(s['sha256']+'  '+rel for rel,s in file_specs.items())+'\n',encoding='utf-8')
    archive=ROOT/'release/email_guard_submission.zip'
    # Preserve earlier archive rather than overwriting a previous release.
    if archive.exists():
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.rename(archive.with_name('email_guard_submission_'+stamp+'.zip'))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in sorted(expected|{'SUBMISSION_MANIFEST.json','SHA256SUMS.txt'}):
            z.write(dest/rel,'email_guard_submission/'+rel)
    laya_files={p.relative_to(laya).as_posix():{'sha256':sha_file(p),'bytes':p.stat().st_size}
                for p in laya.rglob('*') if p.is_file() and p.name!='MODEL_FILES.json'}
    (laya/'MODEL_FILES.json').write_text(json.dumps({'component':'Laya_day5','files':laya_files},indent=2)+'\n',encoding='utf-8')
    summary={'submission_zip':{'path':archive.relative_to(ROOT).as_posix(),'sha256':sha_file(archive),'bytes':archive.stat().st_size},
             'submission_files':len(expected)+2,'laya_separate_files':len(laya_files),'laya_separate_bytes':sum(v['bytes'] for v in laya_files.values()),
             'docker_built_or_run':False}
    (ROOT/'release/archives.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
