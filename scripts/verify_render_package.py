"""Verify exact deployment-copy bytes and unchanged saved E0 identity; stdlib only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'SUBMISSION_MANIFEST.json').read_bytes())
    for rel, record in manifest['files'].items():
        path = (ROOT / rel).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(), rel
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], rel
    release = json.loads((ROOT / 'app/release_manifest.json').read_bytes())
    for rel, sha in release['components']['E0']['files'].items():
        assert digest(ROOT / 'artifacts' / rel) == sha, rel
    for rel, sha in release['shared_preprocessing_sha256'].items():
        assert digest(ROOT / rel) == sha, rel
    assert release['components']['E0']['threshold'] == 0.3037252359712881
    sums = {}
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        sha, rel = line.split('  ', 1)
        assert rel not in sums, rel
        sums[rel] = sha
        assert digest(ROOT / rel) == sha, rel
    assert set(sums) == set(manifest['files']) | {'SUBMISSION_MANIFEST.json'}
    print(json.dumps({'status': 'passed', 'files': len(manifest['files']),
                      'E0': 'original bundle, preprocessing and threshold',
                      'scope': 'File integrity only; does not prove Docker build/public deployment'}))


if __name__ == '__main__':
    main()
