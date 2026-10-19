"""In-memory MIME parsing and exact fixed component adapters."""
import math
import threading
from .contract import ROOT, MAX_BYTES, contract, verify_component, unavailable, compose, artifact_root, artifact_path
from normalization_schema import make_model_text
from day5_common import clean_text
from read_email import parse_email
from .categories import suggest_category


class InputError(ValueError):
    pass


def parse_upload(raw):
    if len(raw) > MAX_BYTES:
        raise InputError('file_too_large')
    if not raw or not raw.strip():
        raise InputError('empty_file')
    try:
        parsed = parse_email(raw)
    except (ValueError, TypeError, UnicodeError, RecursionError, IndexError):
        raise InputError('invalid_eml') from None
    # MIME parser is deliberately tolerant. Binary garbage with no mail headers
    # is not an email; recoverable MIME defects remain visible warnings.
    if not parsed['headers_raw'] and (b'\x00' in raw or not parsed['body_text'].strip()):
        raise InputError('invalid_eml')
    return parsed


def build_texts(parsed):
    import json
    cfg = json.loads((ROOT / 'config/day4_experiments.json').read_bytes())
    original, original_changes = make_model_text(parsed['subject'], parsed['body_text'])
    clean, clean_changes = clean_text({'subject': parsed['subject'], 'body_text': parsed['body_text'],
                                      'feature_availability': {'subject': 'observable'}}, cfg)
    return original, clean, {'E0': original_changes, 'Laya_day5': clean_changes}


class E0:
    def __init__(self):
        from predict_email import load_bundle
        from threadpoolctl import threadpool_limits
        self.spec = verify_component('E0')
        self.pools = threadpool_limits(limits=1)
        self.bundle = load_bundle(artifact_path(self.spec['bundle_path']), artifact_root())
        self.lock = threading.Lock()

    def predict(self, text):
        with self.lock:
            pipeline = self.bundle['pipeline']
            vector = pipeline['tfidf'].transform([text])
            score = float(pipeline['classifier'].predict_proba(vector)[0, list(pipeline.classes_).index(1)])
        if not math.isfinite(score):
            return unavailable('E0', 'nonfinite_score', 'error')
        ok = vector.nnz > 0
        return {'model_id': 'E0', 'status': 'ok' if ok else 'insufficient_features',
                'decision': int(score >= self.spec['threshold']) if ok else None,
                'reason': 'threshold' if ok else 'zero_active_features',
                'score': score, 'threshold': self.spec['threshold'], 'temperature': 1.,
                'active_features': int(vector.nnz), 'model_text_chars': len(text),
                'truncated': False, 'builder': self.spec['builder'],
                'score_semantics': self.spec['score_semantics']}


def safe_metadata(parsed):
    def short(value, limit=1000):
        return str(value or '')[:limit]
    links = [{k: short(link.get(k), 500) if k in ('url', 'label', 'domain', 'scheme') else link.get(k)
              for k in ('url', 'label', 'domain', 'scheme', 'label_mismatch', 'scheme_high_risk', 'scheme_unusual')}
             for link in parsed['links'][:50]]
    attachments = [{k: short(a.get(k), 300) for k in ('filename', 'content_type', 'disposition')}
                   for a in parsed['attachments'][:50]]
    unsub = parsed['unsubscribe']
    return {'subject': short(parsed['subject']), 'sender': short(parsed['sender']),
            'provenance': 'unverified_email_headers',
            'display_truncated': len(str(parsed['subject'] or '')) > 1000 or len(str(parsed['sender'] or '')) > 1000,
            'body_source': parsed['body_source'], 'body_chars': len(parsed['body_text']),
            'links': links, 'link_count': len(parsed['links']),
            'attachments': attachments, 'attachment_count': len(parsed['attachments']),
            'unsubscribe': {'header_present': unsub['header_present'], 'body_hint_found': unsub['body_hint_found'],
                            'one_click_declared': unsub['one_click_declared'], 'working': None},
            'parser_warnings': parsed['parse_warnings'],
            'category': suggest_category(parsed['subject'], parsed['body_text']),
            'phishing': 'not_evaluated',
            'observations_are_model_explanations': False}


class Engine:
    def __init__(self, enable_laya=True):
        self.e0 = None
        self.e0_reason = 'not_initialized'
        try:
            self.e0 = E0()
            self.e0_reason = None
        except Exception as error:
            self.e0_reason = type(error).__name__  # never leak text/paths through errors
        from .worker_client import LayaWorker
        self.laya = LayaWorker(enabled=enable_laya)
        self.laya.start()

    def readiness(self):
        e0 = {'status': 'ready' if self.e0 is not None else 'unavailable', 'reason': self.e0_reason}
        laya = self.laya.readiness()
        both = e0['status'] == laya['status'] == 'ready'
        return {'components': {'E0': e0, 'Laya_day5': laya},
                'modes': {'baseline': e0['status'] == 'ready', 'recall': both, 'review': both},
                'default': 'baseline'}

    def predict_texts(self, original, clean, mode):
        try:
            first = self.e0.predict(original) if self.e0 else unavailable('E0', self.e0_reason)
        except Exception:
            first = unavailable('E0', 'inference_error', 'error')
        parts = {'E0': first}
        if mode != 'baseline':
            parts['Laya_day5'] = self.laya.predict(clean)
        # Thresholds also remain visible for failed / unavailable components.
        for name, part in parts.items():
            spec = contract()[name]
            part.setdefault('threshold', spec['threshold'])
            part.setdefault('temperature', spec['temperature'])
            part.setdefault('builder', spec['builder'])
        result = compose(mode, parts)
        failure = any(p['status'] in ('error', 'unavailable', 'busy', 'initializing') for p in parts.values())
        return result | {'mode': mode, 'components': parts, 'infrastructure_failure': failure,
                         'ensemble_score': None,
                         'rule': {'baseline': 'E0 operating threshold', 'recall': 'E0 OR Laya; both must succeed',
                                  'review': 'agreement → decision; disagreement/refusal → needs_review'}[mode]}

    def analyze(self, raw, mode):
        parsed = parse_upload(raw)
        original, clean, changes = build_texts(parsed)
        result = self.predict_texts(original, clean, mode)
        for name, part in result['components'].items():
            part['builder_transformations'] = changes[name]
        return result | {'email': safe_metadata(parsed)}

    def close(self):
        self.laya.close()
