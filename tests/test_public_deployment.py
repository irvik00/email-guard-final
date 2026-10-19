"""Deployment boundary tests; synthetic/stub API responses only."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.deployment_config import deployment_settings
from app.web import create_app
from fastapi.testclient import TestClient


class PublicDeploymentTests(unittest.TestCase):
    def test_default_remains_local(self):
        hosts, origins = deployment_settings({})
        self.assertNotIn('*', hosts)
        self.assertEqual(origins, frozenset())
        self.assertNotIn('email-guard-demo.onrender.com', hosts)

    def test_exact_render_origin(self):
        hosts, origins = deployment_settings({'RENDER_EXTERNAL_URL': 'https://email-guard-demo.onrender.com'})
        self.assertIn('email-guard-demo.onrender.com', hosts)
        self.assertEqual(origins, frozenset(['https://email-guard-demo.onrender.com']))

    def test_invalid_operator_url_fails(self):
        for url in ['https://*.onrender.com', 'http://public.example', 'https://x.example/path',
                    'https://user:secret@x.example', 'https://x.example?query=1', 'https://x.example:invalid']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                deployment_settings({'EMAIL_GUARD_PUBLIC_URL': url})

    def test_public_proxy_origin_and_untrusted_requests(self):
        engine = Mock()
        engine.analyze.return_value = {'status': 'ok', 'decision': 0, 'infrastructure_failure': False}
        with patch.dict(os.environ, {'RENDER_EXTERNAL_URL': 'https://email-guard-demo.onrender.com',
                                     'EMAIL_GUARD_PUBLIC_URL': ''}):
            application = create_app(engine=engine, enable_laya=False)
        with TestClient(application, base_url='http://email-guard-demo.onrender.com') as client:
            headers = {'Origin': 'https://email-guard-demo.onrender.com', 'Content-Type': 'message/rfc822'}
            self.assertEqual(client.post('/api/analyze', content=b'Subject: Demo\r\n\r\nDemo', headers=headers).status_code, 200)
            engine.analyze.reset_mock()
            headers['Origin'] = 'https://attacker.example'
            self.assertEqual(client.post('/api/analyze', content=b'private', headers=headers).status_code, 403)
            engine.analyze.assert_not_called()
            self.assertEqual(client.get('/health', headers={'Host': 'attacker.example'}).status_code, 400)


if __name__ == '__main__':
    unittest.main()
