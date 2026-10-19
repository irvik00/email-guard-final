"""Generate app-only release inventory from trusted final protocol; never read datasets."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.contract import contract, sha_file, PROTOCOL, PROTOCOL_SHA256


def main():
    specs = contract()
    shared = ['src/read_email.py', 'src/normalization_schema.py', 'src/day5_common.py',
              'src/day5_laya_runtime.py', 'src/decision_protocol.py', 'src/final_eval_common.py',
              'src/predict_email.py', 'src/baseline_common.py', 'config/day4_experiments.json']
    for spec in specs.values():
        for rel, expected in spec['files'].items():
            if sha_file(ROOT / rel) != expected:
                raise ValueError('Frozen artifact differs: ' + rel)
    for env, dest in [('.venv-app', 'requirements-app.lock.txt'), ('.venv-decision', 'requirements-app-laya.lock.txt')]:
        freeze = subprocess.check_output([str(ROOT / env / 'Scripts/python.exe'), '-m', 'pip', 'freeze'], text=True)
        if env == '.venv-decision':
            # Portable version instead of the old private local-wheel file URL.
            freeze = '\n'.join('laya==0.3.20' if line.startswith('laya @ ') else line for line in freeze.splitlines()) + '\n'
        (ROOT / dest).write_text(freeze, encoding='utf-8')
    app_files = [p for p in (ROOT/'app').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'release_manifest.json']
    app_files += [ROOT/'requirements-app.in', ROOT/'requirements-app.lock.txt', ROOT/'requirements-app-laya.lock.txt', Path(__file__).resolve()]
    app_files += [ROOT/'Dockerfile', ROOT/'.dockerignore', ROOT/'compose.yaml', ROOT/'.env.example', ROOT/'requirements-docker.lock.txt']
    verification_files = [ROOT/'tests/test_app.py', ROOT/'scripts/verify_app.py', ROOT/'scripts/audit_app.py']
    verification_files += [ROOT/'tests/test_release.py', ROOT/'scripts/smoke_release.py', ROOT/'scripts/prepare_model_artifacts.py']
    verification_files += [p for p in (ROOT/'examples/app').glob('*') if p.is_file()]
    sdk_wheel = ROOT/'vendor/laya-0.3.20-py3-none-any.whl'
    manifest = {'version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_final_protocol': {'path': PROTOCOL, 'sha256': PROTOCOL_SHA256},
        'default': 'baseline', 'modes': {'baseline': {'components': ['E0'], 'rule': 'fixed threshold; zero active features → refusal'},
            'recall': {'components': ['E0', 'Laya_day5'], 'rule': 'OR; either refusal/error → no decision'},
            'review': {'components': ['E0', 'Laya_day5'], 'rule': 'agreement → result; disagreement/refusal/error → needs_review'}},
        'components': specs, 'shared_preprocessing_sha256': {rel: sha_file(ROOT/rel) for rel in shared},
        'application_files_sha256': {p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted(app_files)},
        'verification_files_sha256': {p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted(verification_files)},
        'trusted_sdk_wheel': {'path':sdk_wheel.relative_to(ROOT).as_posix(),'sha256':sha_file(sdk_wheel)},
        'runtime': {'web_e0': {'windows':'.venv-app/Scripts/python.exe','posix':'.venv-app/bin/python'},
                    'neural': {'windows':'.venv-decision/Scripts/python.exe','posix':'.venv-decision/bin/python'},
                    'operator_configuration':{'EMAIL_GUARD_ARTIFACT_ROOT':'artifacts mount root; relative IDs and hashes fixed',
                        'EMAIL_GUARD_LAYA_PYTHON':'trusted local runtime interpreter only; never accepted via API',
                        'EMAIL_GUARD_HOST':'127.0.0.1 locally;0.0.0.0 inside CPU container with loopback publishing'},
                    'docker_verified':False,'docker_gpu_profile':False,
                    'neural_runtime_policy': 'Existing trusted local CUDA environment used read-only; separate app dependency freeze; no research dependencies modified',
                    'python': sys.version.split()[0], 'device': 'cuda', 'cpu_threads': 1, 'uvicorn_workers': 1,
                    'parallel_analysis': 2, 'parallel_neural': 1, 'neural_timeout_seconds': 60,
                    'neural_startup_timeout_seconds': 180, 'neural_queue_wait_seconds': 1,
                    'automatic_restart': False, 'cpu_fallback': False},
        'uploads': {'max_bytes': 10*1024*1024, 'stream_checked': True, 'persistence': False,
                    'filename_in_model_or_path': False, 'raw_email_html_rendered': False, 'external_fetch': False},
        'category': {'method': 'not_available', 'value': None}, 'phishing_classifier': False,
        'training_performed': False, 'held_out_inference_performed': False,
        'closed_records_parsed_for_app_verification': False,
        'preservation_hashing': 'Old report/code/config/doc bytes hashed without record decoding; model sizes/mtimes plus deployed weight SHA256 checked',
        'ensemble_probability': None, 'scores_averaged': False}
    (ROOT/'app/release_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'components': list(specs), 'app_files':len(app_files), 'manifest':'app/release_manifest.json'}))


if __name__ == '__main__':
    main()
