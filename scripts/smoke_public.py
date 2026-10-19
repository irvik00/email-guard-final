"""Check a deployed CPU demo using ONLY six bundled fictional emails; stdlib only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Exact public https://NAME.onrender.com URL')
    parser.add_argument('--label', default='public')
    parser.add_argument('--allow-local', action='store_true', help='HTTP loopback only, for local verification')
    args = parser.parse_args()
    base = args.url.rstrip('/')
    p = urlsplit(base)
    local = args.allow_local and p.scheme == 'http' and p.hostname in ('127.0.0.1', 'localhost')
    if (p.scheme != 'https' and not local) or not p.hostname or p.path or p.query or p.fragment or p.username or p.password:
        parser.error('Use one HTTPS origin without path/credentials, e.g. https://NAME.onrender.com')
    opener = build_opener(NoRedirect())
    checks = []

    def check(name, ok):
        checks.append({'name': name, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(name)

    def call(path, raw=None, origin=None, content_type='message/rfc822'):
        headers = {}
        if raw is not None:
            headers = {'Content-Type': content_type, 'Origin': origin or base}
        req = Request(base + path, data=raw, headers=headers, method='POST' if raw is not None else 'GET')
        try:
            response = opener.open(req, timeout=90)
        except HTTPError as exc:
            response = exc
        with response:
            return response.code, json.loads(response.read())

    status, health = call('/health')
    check('health_no_upload_persistence', status == 200 and health.get('upload_persistence') is False)
    status, ready = call('/ready')
    check('baseline_ready', status == 200 and ready['modes']['baseline'])
    check('CPU_profile_Laya_modes_unavailable', not ready['modes']['recall'] and not ready['modes']['review'])
    expected = json.loads((ROOT / 'reports/release/synthetic_reference.json').read_bytes())
    score_deltas = []
    for path in sorted((ROOT / 'examples/app').glob('*.eml')):
        status, result = call('/api/analyze?mode=baseline', path.read_bytes())
        want = expected[path.name]['baseline']
        check(path.name + '_decision', status == 200 and result['status'] == want['status'] and result['decision'] == want['decision'])
        component, old = result['components']['E0'], want['components']['E0']
        delta = abs(component['score'] - old['score']) if component.get('score') is not None and old.get('score') is not None else 0
        score_deltas.append(delta)
        check(path.name + '_E0_score', delta <= 1e-10 and component['status'] == old['status'] and component['decision'] == old['decision'])
    for name, raw, mime, origin, want in [
        ('empty', b'', 'message/rfc822', None, 400),
        ('unsupported_media_type', b'{}', 'application/json', None, 415),
        ('foreign_origin_forbidden', b'Subject: Synthetic\r\n\r\nDemo', 'message/rfc822', 'https://untrusted.example.invalid', 403),
    ]:
        status, result = call('/api/analyze', raw, origin, mime)
        check(name, status == want and result.get('decision') is None)
    for mode in ('recall', 'review'):
        status, _ = call('/ready?mode=' + mode)
        check(mode + '_readiness_503', status == 503)
    label = ''.join(c for c in args.label if c.isalnum() or c in '-_')[:40] or 'public'
    now = datetime.now(timezone.utc)
    report = {'status': 'passed', 'utc': now.isoformat(), 'url': base, 'checks': checks,
              'max_E0_score_delta': max(score_deltas), 'emails': 'Only six bundled fictional examples',
              'scope': 'HTTP functionality and equivalence to saved outputs; not a new model-quality benchmark',
              'runtime': 'HTTP target only; this script does not itself prove how the target is hosted'}
    folder = ROOT / 'reports/deployment'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / ('public-smoke-' + now.strftime('%Y%m%dT%H%M%S%fZ') + '-' + label + '.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'checks': len(checks), 'report': str(output),
                      'max_E0_score_delta': max(score_deltas)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
