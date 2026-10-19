"""Run the publisher's literal Quickstart inputs, using the frozen multilingual checkpoint."""
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import re
import sys
import time

from diagnose_laya import ROOT, boundary, dump, protected_snapshot, sha


def extract_quickstart(readme_text):
    quickstart = readme_text.split('## Quickstart', 1)[1]
    block = re.search(r'^```python[ \t]*\n(.*?)^```', quickstart, re.S | re.M).group(1)
    literals = {}
    for node in ast.parse(block).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'state', 'questions'}:
                    literals[target.id] = ast.literal_eval(node.value)
    return literals


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.addaudithook(boundary)
    before = protected_snapshot()
    readme = ROOT / '.cache/day4/laya_weights/README.md'
    # Parse literals ONLY. Never execute publisher code or downloaded scripts.
    literals = extract_quickstart(readme.read_text(encoding='utf-8'))
    assert set(literals) == {'state', 'questions'}
    assert set(literals['questions']) == {'department', 'urgency', 'churn_risk'}
    metadata = json.loads((ROOT / 'reports/day4/setup/runtime_manifest.json').read_bytes())
    import torch
    import laya
    torch.set_num_threads(1)
    agent = laya.load(metadata['local_path'], device='cuda', fast=False)
    assert not agent.model.training
    start = time.perf_counter()
    raw = agent.predict(literals['state'], literals['questions'], max_len=1024, head_max_len=256)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    after = protected_snapshot()
    assert before == after
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Exact public README Quickstart state/questions, direct frozen multilingual agent; not English Router replication',
              'source': 'https://huggingface.co/convaiinnovations/laya/blob/' + metadata['revision'] + '/README.md',
              'readme_sha256': sha(readme),
              'state_sha256': __import__('hashlib').sha256(literals['state'].encode()).hexdigest(),
              'questions_sha256': __import__('hashlib').sha256(json.dumps(literals['questions'], sort_keys=True).encode()).hexdigest(),
              'checkpoint_revision': metadata['revision'], 'subfolder': metadata['subfolder'],
              'laya_version': importlib.metadata.version('laya'), 'real_device': str(agent.device),
              'raw_response': raw, 'inference_seconds': seconds,
              'protected_unchanged': True, 'protected_file_count': len(before),
              'datasets_opened': 0, 'network_connections': 0}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'reports/laya_diagnostics' / ('official-' + stamp)
    out.mkdir(exist_ok=False)
    dump(out / 'preservation_before.json', before)
    dump(out / 'preservation_after.json', after)
    dump(out / 'results.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('Saved ' + str(out))


if __name__ == '__main__':
    main()
