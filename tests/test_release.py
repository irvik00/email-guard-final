"""Real relocation of the unchanged E0 bundle and operator-only path configuration."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.contract import artifact_path, artifact_root
from app.inference import E0, Engine, build_texts, parse_upload
from scripts.prepare_model_artifacts import prepare


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'.cache/app-tmp').mkdir(parents=True,exist_ok=True)

    def test_relocated_e0_without_laya_preserves_scores_and_refusals(self):
        raw=(ROOT/'examples/app/demo_02.eml').read_bytes()
        original,clean,changes=build_texts(parse_upload(raw))
        source=ROOT if (ROOT/'models/day3/baseline_bundle.joblib').is_file() else ROOT/'artifacts'
        with patch.dict(os.environ,{'EMAIL_GUARD_ARTIFACT_ROOT':str(source)}):
            old=E0().predict(original)
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache/app-tmp') as td:
            target=Path(td)/'portable mount with spaces'
            prepare(source,target,['E0'])
            with patch.dict(os.environ,{'EMAIL_GUARD_ARTIFACT_ROOT':str(target)}):
                new=E0().predict(original)
                self.assertEqual(new['score'],old['score']);self.assertEqual(new['decision'],old['decision'])
                engine=Engine(enable_laya=False)
                try:
                    self.assertTrue(engine.readiness()['modes']['baseline'])
                    self.assertFalse(engine.readiness()['modes']['recall'])
                    self.assertIsNone(engine.analyze(raw,'recall')['decision'])
                    self.assertEqual(engine.analyze(raw,'review')['status'],'needs_review')
                    self.assertEqual(engine.e0.predict('')['status'],'insufficient_features')
                finally:engine.close()

    def test_relative_operator_artifact_root_and_escape(self):
        with patch.dict(os.environ,{'EMAIL_GUARD_ARTIFACT_ROOT':'artifacts'}):
            self.assertEqual(artifact_root(),ROOT/'artifacts')
            with self.assertRaises(ValueError):artifact_path('../outside.joblib')

    def test_unknown_destination_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache/app-tmp') as td:
            p=Path(td)/'models/day3/baseline_bundle.joblib';p.parent.mkdir(parents=True);p.write_bytes(b'unknown')
            source=ROOT if (ROOT/'models/day3/baseline_bundle.joblib').is_file() else ROOT/'artifacts'
            with self.assertRaises(ValueError):prepare(source,Path(td),['E0'])
            self.assertEqual(p.read_bytes(),b'unknown')


if __name__=='__main__':unittest.main()
