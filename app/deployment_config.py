"""Explicit public origin from operator/Render configuration; local by default."""
import os
import re
from urllib.parse import urlsplit


def deployment_settings(environ=None):
    env = os.environ if environ is None else environ
    hosts = ['127.0.0.1', 'localhost', '[::1]', 'testserver']
    url = env.get('EMAIL_GUARD_PUBLIC_URL') or env.get('RENDER_EXTERNAL_URL', '')
    if not url:
        return hosts, frozenset()
    parsed = urlsplit(url)
    host = (parsed.hostname or '').lower()
    if (parsed.scheme != 'https' or not host or parsed.username or parsed.password
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment
            or not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?', host)
            or '..' in host):
        raise ValueError('Public URL must be one explicit HTTPS origin without path or credentials')
    port = parsed.port
    origin = 'https://' + host + (':' + str(port) if port and port != 443 else '')
    if host not in hosts:
        hosts.append(host)
    return hosts, frozenset([origin])
