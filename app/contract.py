"""Trusted, fixed inference contract inherited from the final evaluation."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
PROTOCOL = 'reports/final_evaluation/runs/20261017T132144Z-d663811e/FINAL_EVALUATION_PROTOCOL.json'
PROTOCOL_SHA256 = '9bc7e77f7b9c06f64e9660957c0498dc90928450b4322387ade42c532a523b33'
MAX_BYTES = 10 * 1024 * 1024
MODES = ('baseline', 'recall', 'review')


def artifact_root():
    """Operator-only mount location; artifact identities remain fixed by hashes."""
    default = ROOT / 'artifacts' if (ROOT / 'artifacts').is_dir() else ROOT
    configured = Path(os.environ.get('EMAIL_GUARD_ARTIFACT_ROOT', str(default))).expanduser()
    return (ROOT / configured).resolve() if not configured.is_absolute() else configured.resolve()


def artifact_path(relative):
    base = artifact_root()
    path = (base / relative).resolve()
    if not path.is_relative_to(base / 'models'):
        raise ValueError('artifact_path_outside_models')
    return path


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def contract():
    path = ROOT / PROTOCOL
    if sha_file(path) != PROTOCOL_SHA256:
        raise ValueError('protocol_integrity_error')
    source = json.loads(path.read_bytes())
    return {c['name']: c for c in source['components'] if c['name'] in ('E0', 'Laya_day5')}


def verify_component(name):
    spec = contract()[name]
    for rel, expected in spec['files'].items():
        path = artifact_path(rel)
        if sha_file(path) != expected:
            raise ValueError('artifact_integrity_error')
    release = json.loads((ROOT / 'app/release_manifest.json').read_bytes())
    for rel, expected in release['shared_preprocessing_sha256'].items():
        if sha_file(ROOT / rel) != expected:
            raise ValueError('preprocessing_integrity_error')
    return spec


def unavailable(name, reason='component_unavailable', status='unavailable'):
    return {'model_id': name, 'status': status, 'decision': None,
            'score': None, 'reason': reason}


def compose(mode, components):
    if mode not in MODES:
        raise ValueError('unknown_mode')
    if mode == 'baseline':
        part = components['E0']
        return {'status': part['status'], 'decision': part['decision'],
                'reason': part.get('reason', 'component_decision'),
                'unavailable_components': [] if part['status'] == 'ok' else ['E0']}
    # Exactly the previously tested handling, including refusal on either input.
    from final_eval_common import vote
    return vote('E0_OR_Laya' if mode == 'recall' else 'E0_Laya_review', components)
