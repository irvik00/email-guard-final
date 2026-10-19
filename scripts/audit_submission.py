"""Independent archive/content/hash/aggregate audit, without email inference or extraction."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.contract import contract,sha_file


def main():
    archive=ROOT/'release/email_guard_submission.zip'
    count=0;total=0
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();prefix='email_guard_submission/'
        assert len(names)==len(set(names))
        manifest=json.loads(z.read(prefix+'SUBMISSION_MANIFEST.json'))
        expected=set(manifest['files'])|{'SUBMISSION_MANIFEST.json','SHA256SUMS.txt'}
        assert {n.removeprefix(prefix) for n in names}==expected
        for name in names:
            p=PurePosixPath(name)
            assert name.startswith(prefix) and not p.is_absolute() and '..' not in p.parts and '\\' not in name
            rel=name.removeprefix(prefix)
            assert not any(part in {'.cache','.venv','.venv-app','.venv-decision','data','personal_review','wheelhouse-windows'} for part in p.parts)
            assert p.suffix not in {'.safetensors','.parquet','.7z','.xz','.jsonl'}
            assert '.env' not in p.parts
            if p.suffix=='.eml':assert rel.startswith('examples/app/demo_')
            if p.suffix=='.joblib':assert rel=='artifacts/models/day3/baseline_bundle.joblib'
            if rel in manifest['files']:
                h=hashlib.sha256();size=0
                with z.open(name) as f:
                    for b in iter(lambda:f.read(1048576),b''):h.update(b);size+=len(b)
                spec=manifest['files'][rel]
                assert h.hexdigest()==spec['sha256'] and size==spec['bytes'],rel
                assert sha_file(ROOT/'release/email_guard_submission'/rel)==spec['sha256'],rel
                count+=1;total+=size
        release=json.loads(z.read(prefix+'app/release_manifest.json'))
        assert set(release['components'])=={'E0','Laya_day5'} and release['default']=='baseline'
        for rel,expected_hash in contract()['E0']['files'].items():
            assert hashlib.sha256(z.read(prefix+'artifacts/'+rel)).hexdigest()==expected_hash
        for section in ('shared_preprocessing_sha256','application_files_sha256','verification_files_sha256'):
            for rel,expected_hash in release[section].items():
                if rel in manifest['files']:assert manifest['files'][rel]['sha256']==expected_hash,rel
        docker=z.read(prefix+'Dockerfile').decode();compose=z.read(prefix+'compose.yaml').decode();ignore=z.read(prefix+'.dockerignore').decode()
        assert 'COPY . .' not in docker and 'COPY ./ ' not in docker
        assert '--require-hashes' in docker and '--no-index' in docker
        assert '127.0.0.1:${EMAIL_GUARD_PORT:-8000}:8000' in compose
        assert 'EMAIL_GUARD_DISABLE_LAYA: "1"' in compose and 'read_only: true' in compose
        assert '**' in ignore and '!models' not in ignore and '!data' not in ignore
        metrics=json.loads(z.read(prefix+'reports/release/final_metrics.json'))
        assert len(metrics['binary'])==42 and len(metrics['review'])==7
        for r in metrics['binary']:
            tp,fp,tn,fn=(r[k] for k in ('TP','FP','TN','FN'))
            assert tp+fp+tn+fn==r['successful']<=r['n']
            assert abs(r['coverage']-r['successful']/r['n'])<1e-12
            for key,value in [('precision',tp/(tp+fp)),('recall',tp/(tp+fn)),('F1',2*tp/(2*tp+fp+fn)),('FPR',fp/(fp+tn))]:
                assert abs(r[key]-value)<0.000001,key
        assert metrics['overlap']['teacher_full']==1600 and metrics['overlap']['teacher_nonoverlap']==612
        assert metrics['overlap']['teacher_train_validation_overlap']==988
    laya=ROOT/'release/email_guard_laya_artifacts'
    for rel,h in contract()['Laya_day5']['files'].items():assert sha_file(laya/rel)==h
    before=json.loads((ROOT/'reports/release/preservation_before.json').read_bytes())
    allowed={'app/contract.py','app/inference.py','app/worker_client.py','app/laya_worker.py','app/run.py','app/README.md','app/release_manifest.json'}
    changed=[rel for rel,h in before['sha256'].items() if sha_file(ROOT/rel)!=h]
    assert set(changed)<=allowed,changed
    assert all([(ROOT/rel).stat().st_size,(ROOT/rel).stat().st_mtime_ns]==stat for rel,stat in before['old_models_stat'].items())
    result={'status':'passed','zip_files':count+2,'zip_uncompressed_content_bytes':total,'archive_sha256':sha_file(archive),
        'large_laya_files_checked':6,'metrics_rows_checked':42,'selective_review_rows':7,
        'old_snapshot_files':len(before['sha256']),'authorized_app_changes':changed,
        'old_model_files_stat_unchanged':len(before['old_models_stat']),
        'raw_data_or_private_email_in_archive':False,'neural_weights_in_cpu_archive':False,
        'docker_check_scope':'Static files only; no Docker Engine available',
        'benchmark_or_training_performed':False}
    path=ROOT/'reports/release/submission_audit.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
