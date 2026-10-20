"""Archived public results must match the saved aggregates, not fabricated live metrics."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.web import create_app
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1]


class EvaluationPageTests(unittest.TestCase):
    def test_exact_saved_metrics_and_provenance(self):
        source=ROOT/'reports/release/final_metrics.json'
        aggregate=json.loads(source.read_bytes())
        shown=json.loads((ROOT/'app/static/evaluation.json').read_bytes())
        self.assertEqual(shown['source_metrics_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
        for row in shown['binary']:
            self.assertEqual(row,next(x for x in aggregate['binary'] if (x['dataset'],x['model'])==(row['dataset'],row['model'])))
        self.assertEqual(len(shown['binary']),6)
        self.assertEqual(shown['overlap']['train_validation_matches'],988)

    def test_page_and_json_do_not_invoke_models(self):
        engine=Mock()
        with TestClient(create_app(engine=engine,enable_laya=False)) as client:
            page=client.get('/evaluation');data=client.get('/evaluation-data.json')
            self.assertEqual(page.status_code,200);self.assertEqual(data.status_code,200)
            self.assertIn('0.969294',page.text)
            self.assertIn('0.939891',page.text)
            self.assertIn('988',page.text)
            self.assertIn('не запускает повторный анализ',page.text)
            self.assertEqual(client.get('/').status_code,200)
            self.assertIn('/evaluation',client.get('/').text)
            engine.analyze.assert_not_called()


if __name__=='__main__':unittest.main()
