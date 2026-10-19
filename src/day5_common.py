"""Day-5 stdlib contracts, strict phase boundary, grouping and numerical metrics."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata

from normalization_schema import make_model_text

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ('spam_email', 'russian_auxiliary')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        for row in rows:
            f.write(canonical(row) + b'\n')


def read_rows(path):
    with Path(path).open('rb') as f:
        return [json.loads(line) for line in f]


def now():
    return datetime.now(timezone.utc).isoformat()


class Boundary:
    """Explicit train/final/personal phase allowlist; no network or data writes."""
    def __init__(self, phase, personal_paths=()):
        require(phase in {'prepare', 'train', 'final', 'personal', 'verify'}, 'Unknown phase')
        self.phase = phase
        self.allowed = {ROOT / 'data/processed/splits/manifest.json'}
        if phase in {'prepare', 'final', 'personal'}:
            self.allowed.update(ROOT / f'data/processed/splits/{domain}/train.jsonl' for domain in DOMAINS)
        if phase == 'final':
            self.allowed.update(ROOT / f'data/processed/splits/{domain}/validation.jsonl' for domain in DOMAINS)
        if phase == 'personal':
            for path in personal_paths:
                path = Path(path).resolve()
                require(path.is_relative_to(ROOT / 'data/personal_review'), 'Personal path must be inside data/personal_review')
                self.allowed.add(path)
        self.accesses = Counter()
        self.denied = []

    def check(self, event, args):
        if event in {'socket.connect', 'socket.bind'}:
            self.denied.append(event)
            raise PermissionError('No network/cloud inference in day5 runner')
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = str(args[1] or '')
        flags = int(args[2] or 0) if len(args) > 2 else 0
        if path.is_relative_to(ROOT / 'data'):
            if path not in self.allowed or any(c in mode for c in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR):
                self.denied.append(path.relative_to(ROOT).as_posix())
                raise PermissionError('Closed/read-only data: ' + path.relative_to(ROOT).as_posix())
        if path.is_relative_to(ROOT / 'data') or path.is_relative_to(ROOT / 'reports/day5') or path.is_relative_to(ROOT / 'models/day5'):
            self.accesses[str(path.relative_to(ROOT).as_posix()) + ':' + mode] += 1

    def install(self):
        sys.addaudithook(self.check)
        return self

    def report(self):
        return {'phase': self.phase, 'allowed_data_paths': sorted(str(p.relative_to(ROOT).as_posix()) for p in self.allowed),
                'accesses': dict(self.accesses), 'denied': self.denied, 'no_network': True}


def clean_text(record, day4_config):
    """Same observable-subject cleaner as day4_common.clean_text, dependency-free."""
    subject = record.get('subject')
    changes = []
    if record.get('feature_availability', {}).get('subject') == 'observable' and isinstance(subject, str):
        subject, n = re.subn(day4_config['clean_builder']['pattern'], '', subject, count=1, flags=re.I)
        if n:
            changes.append('remove_leading_numeric_filter_marker')
    text, more = make_model_text(subject, record.get('body_text'))
    return text, changes + more


def compare_text(text):
    text = unicodedata.normalize('NFC', text).casefold()
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    return re.sub(r'\s+', ' ', text).strip()


def load_split(domain, split, manifest, day4_config):
    require(domain in DOMAINS and split in {'train', 'validation'}, 'Closed split')
    rel = f'splits/{domain}/{split}.jsonl'
    path = ROOT / 'data/processed' / rel
    spec = manifest['datasets'][domain]['splits'][split]
    require(spec['path'] == rel and spec['status'] == 'ready', 'Manifest path/status mismatch')
    h = hashlib.sha256()
    count = size = 0
    rows = []
    for line in path.open('rb'):
        h.update(line); size += len(line); count += 1
        r = json.loads(line)
        require(r['dataset'] == domain and r['split'] == split, 'Role mismatch')
        require(type(r['spam_label']) is int and r['spam_label'] in {0, 1}, 'Invalid label')
        require(r['quality_status'] == 'eligible' and r['usable_model_text'] and not r['protected_test']
                and 'needs_review' not in r['quality_flags'], 'Closed/ineligible record')
        require(r['parse_status'] in {'ok', 'partial'}, 'Invalid parse status')
        require(r['model_text'] == make_model_text(r['subject'], r['body_text'])[0], 'Original builder mismatch')
        require(hashlib.sha256(r['model_text'].encode()).hexdigest() == r['model_text_sha256'], 'Text hash mismatch')
        assignment = manifest['group_split_mapping'][r['duplicate_group_id']]
        require(assignment['role'] == split and not assignment['protected_test'] and not assignment.get('reserved_future', False), 'Protected group')
        require({'dataset': domain, 'split': split, 'record_id': r['record_id']} in assignment['assignments'], 'Group membership mismatch')
        if domain == 'russian_auxiliary':
            require(type(r['original_label']) is bool and r['spam_label'] == int(r['original_label']), 'RU mapping contradiction; do not reverse labels')
            require(r['label_status']['spam'] == 'source_intended_unverified', 'RU provenance changed')
        text, changes = clean_text(r, day4_config)
        require(bool(text.strip()), 'Empty cleaned text')
        rows.append({'record_id': r['record_id'], 'duplicate_group_id': r['duplicate_group_id'],
                     'domain': domain, 'source': r['source'], 'spam_label': r['spam_label'],
                     'label_status': r['label_status']['spam'], 'text': text, 'original_model_text': r['model_text'],
                     'cleaning_changes': changes, 'original_locator': r['original_locator']})
    actual = {'sha256': h.hexdigest(), 'rows': count, 'size_bytes': size}
    require(actual == manifest['artifacts'][rel], 'Input hash/count mismatch')
    require(dict(Counter(str(r['spam_label']) for r in rows)) == spec['class_counts'] and count == spec['records'], 'Class count mismatch')
    require(len({r['record_id'] for r in rows}) == len(rows), 'Repeated IDs')
    return rows, {'path': path.relative_to(ROOT).as_posix(), **actual}


class DSU:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def group_and_partition(rows, seed=42):
    dsu = DSU(len(rows))
    seen = {}
    for i, r in enumerate(rows):
        for key in [('group', r['duplicate_group_id']), ('exact', digest(r['text'])),
                    ('canonical', digest(compare_text(r['text']))),
                    ('original_exact', digest(r['original_model_text'])),
                    ('original_canonical', digest(compare_text(r['original_model_text'])))]:
            if key in seen: dsu.union(i, seen[key])
            else: seen[key] = i
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[dsu.find(i)].append(r)
    strata = defaultdict(list)
    conflicts = []
    for group in groups.values():
        gid = 'inner:' + digest(sorted(r['record_id'] for r in group))
        for r in group: r['inner_group_id'] = gid
        if len({r['spam_label'] for r in group}) > 1:
            conflicts.append({'group': gid, 'ids': [r['record_id'] for r in group], 'labels': [r['spam_label'] for r in group]})
        # Signature stratifies multi-source groups without breaking them apart.
        signature = tuple(sorted({(r['domain'], r['source'], r['spam_label']) for r in group}))
        strata[signature].append(group)
    parts = {'fit': [], 'dev': [], 'calibration': []}
    for signature, blocks in sorted(strata.items()):
        blocks.sort(key=lambda g: digest([seed, 'inner-partition', g[0]['inner_group_id']]))
        n = sum(len(g) for g in blocks)
        targets = {'fit': n * .7, 'dev': n * .15, 'calibration': n * .15}
        counts = dict.fromkeys(parts, 0)
        for block in blocks:
            part = max(parts, key=lambda p: (targets[p] - counts[p], -list(parts).index(p)))
            for r in block: r['inner_part'] = part
            parts[part].extend(block); counts[part] += len(block)
    audit_parts(parts)
    return parts, conflicts


def audit_parts(parts):
    owners = {}
    for part, rows in parts.items():
        for r in rows:
            for key in [('id', r['record_id']), ('gid', r['duplicate_group_id']), ('inner', r['inner_group_id']),
                        ('exact', digest(r['text'])), ('canonical', digest(compare_text(r['text']))),
                        ('original_exact', digest(r['original_model_text'])), ('original_canonical', digest(compare_text(r['original_model_text'])))]:
                require(key not in owners or owners[key] == part, 'Inner leakage: ' + key[0])
                owners[key] = part
    return {'passed': True, 'roles': {p: len(v) for p, v in parts.items()}, 'checked': ['ID', 'global group', 'clean exact/canonical', 'original exact/canonical']}


def proportional_sample(rows, n, purpose, seed=42):
    buckets = defaultdict(list)
    for r in rows: buckets[(r['source'], r['spam_label'])].append(r)
    n = min(n, len(rows)); exact = {k: len(v) * n / len(rows) for k, v in buckets.items()}
    quotas = {k: int(v) for k, v in exact.items()}
    for k in sorted(buckets, key=lambda k: (-(exact[k] - quotas[k]), k))[:n-sum(quotas.values())]: quotas[k] += 1
    return [r for k in sorted(buckets) for r in sorted(buckets[k], key=lambda r: digest([seed, purpose, r['record_id']]))[:quotas[k]]]


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def metrics(labels, probs, threshold=.5):
    require(len(labels) == len(probs) and len(labels) > 0, 'Empty/misaligned metric data')
    tp = fp = tn = fn = 0
    for y, p in zip(labels, probs):
        require(type(y) is int and y in {0, 1} and math.isfinite(p), 'Invalid metric input')
        if y == 1 and p >= threshold: tp += 1
        elif y == 0 and p >= threshold: fp += 1
        elif y == 0: tn += 1
        else: fn += 1
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'precision': precision, 'recall': recall,
            'F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., 'FPR': fp/(fp+tn) if fp+tn else None,
            'support': len(labels), 'ham_support': tn+fp, 'spam_support': tp+fn, 'threshold': threshold}


def probability_metrics(labels, logits, temperature=1., bins=10):
    probs = [sigmoid(x/temperature) for x in logits]
    n = len(labels)
    nll = sum(max(x/temperature, 0) - y*x/temperature + math.log1p(math.exp(-abs(x/temperature))) for y,x in zip(labels,logits))/n
    brier = sum((p-y)**2 for y,p in zip(labels,probs))/n
    details = []
    for i in range(bins):
        idx = [j for j,p in enumerate(probs) if min(int(p*bins), bins-1) == i]
        details.append({'bin': i, 'lower': i/bins, 'upper': (i+1)/bins, 'count': len(idx),
                        'mean_probability': sum(probs[j] for j in idx)/len(idx) if idx else None,
                        'positive_fraction': sum(labels[j] for j in idx)/len(idx) if idx else None})
    ece = sum(b['count']/n * abs(b['mean_probability']-b['positive_fraction']) for b in details if b['count'])
    return {'NLL': nll, 'Brier': brier, 'ECE': ece, 'ECE_bins': details, 'n': n, 'temperature': temperature,
            'ECE_definition': '10 equal-width bins of P(spam); abs(mean probability - fraction positive), count-weighted'}


def golden_minimize(function, lo, hi, iterations=100):
    phi = (math.sqrt(5)-1)/2
    a,b = lo,hi; c=b-phi*(b-a); d=a+phi*(b-a)
    for _ in range(iterations):
        if function(c) <= function(d): b,d=d,c; c=b-phi*(b-a)
        else: a,c=c,d; d=a+phi*(b-a)
    return min([lo, hi, (a+b)/2], key=function)


def fit_temperature(labels, logits, lo=.5, hi=5.):
    # Stable NLL; no fitting on dev or external validation.
    def objective(log_t):
        t=math.exp(log_t)
        return sum(max(x/t,0)-y*x/t+math.log1p(math.exp(-abs(x/t))) for y,x in zip(labels,logits))/len(labels)
    unrestricted = math.exp(golden_minimize(objective, math.log(.01), math.log(1000.)))
    bounded = math.exp(golden_minimize(objective, math.log(lo), math.log(hi)))
    bounded = min(hi,max(lo,bounded))
    return {'unconstrained_temperature': unrestricted, 'applied_temperature': bounded,
            'sdk_limits': [lo,hi], 'at_boundary': abs(bounded-lo)<1e-6 or abs(bounded-hi)<1e-6,
            'optimum_outside_sdk_range': unrestricted < lo or unrestricted > hi,
            'method': '100 golden-section iterations log(T), endpoints included; binary NLL calibration only'}


def choose_threshold(rows, temperature):
    # Sweep score ties together, O(n log n); never silently choose all-negative.
    groups = defaultdict(list)
    totals = {d: Counter(r['spam_label'] for r in rows if r['domain']==d) for d in DOMAINS}
    for r in rows: groups[sigmoid(r['logit_difference']/temperature)].append(r)
    tp = dict.fromkeys(DOMAINS,0); fp=dict.fromkeys(DOMAINS,0); best=None
    for threshold in sorted(groups,reverse=True):
        for r in groups[threshold]:
            if r['spam_label']: tp[r['domain']] += 1
            else: fp[r['domain']] += 1
        hist_fpr=fp['spam_email']/totals['spam_email'][0]
        if hist_fpr > .01 or any(tp[d] == 0 for d in DOMAINS): continue
        mean=sum(2*tp[d]/(2*tp[d]+fp[d]+totals[d][1]-tp[d]) for d in DOMAINS)/2
        key=(mean,-fp['spam_email'],threshold)
        if best is None or key > best[0]: best=(key,threshold)
    threshold=best[1] if best else .5
    return {'threshold': threshold, 'unmet_fp_scenario': best is None,
            'selection_data': 'internal dev only, after calibration fixed',
            'by_domain': {d: metrics([r['spam_label'] for r in rows if r['domain']==d],
                                    [sigmoid(r['logit_difference']/temperature) for r in rows if r['domain']==d],threshold) for d in DOMAINS}}


def preserved_files():
    paths=set()
    for folder in ['reports/day3','reports/day4','models/day3','models/day4','reports/laya_diagnostics']:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    for pattern in ['reports/DAY3_STATUS.md','reports/DAY4_STATUS.md','reports/laya_diagnostics.md', 'config/day3*.json','config/day4*.json',
                    'src/*day4*.py','src/baseline_common.py','src/train_baselines.py','src/predict_email.py','src/read_email.py',
                    'src/normalization_schema.py','src/decision_protocol.py','src/run_decision_pilot.py','requirements*.txt',
                    '.venv/pyvenv.cfg','.venv-decision/pyvenv.cfg']:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    return {p.relative_to(ROOT).as_posix(): {'sha256':sha_file(p),'size_bytes':p.stat().st_size} for p in sorted(paths)}


def counts(rows):
    return {d: {'rows':sum(r['domain']==d for r in rows),
                'classes':dict(Counter(str(r['spam_label']) for r in rows if r['domain']==d)),
                'source_classes':dict(Counter(r['source']+':'+str(r['spam_label']) for r in rows if r['domain']==d))} for d in DOMAINS}
