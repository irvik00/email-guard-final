"""Topic rules are separate from the frozen spam-model contract."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.categories import suggest_category
from app.inference import Engine, safe_metadata, parse_upload


class CategoryTests(unittest.TestCase):
    def test_five_requested_examples(self):
        for subject, body, want in [
            ('Скидка 30% на коллекцию', 'Товары со скидкой.', 'advertising'),
            ('Вакансия: Frontend-разработчик', 'Мы ознакомились с вашим резюме.', 'job_offer'),
            ('Выписка по карте за июнь', 'Ваша выписка готова.', 'financial'),
            ('Подтверждение смены пароля', 'Ваш пароль изменён.', 'service'),
            ('Приглашение на день рождения', 'Приходи в субботу!', 'other')]:
            with self.subTest(subject=subject):
                result = suggest_category(subject, body)
                self.assertEqual(result['id'], want)
                self.assertFalse(result['trained'])
                self.assertFalse(result['affects_spam_decision'])
                self.assertIsNone(result['confidence'])

    def test_english_ambiguity_empty_and_length_limit(self):
        self.assertEqual(suggest_category('Job offer', 'Interview invitation')['id'], 'job_offer')
        self.assertIsNone(suggest_category('', '')['id'])
        self.assertEqual(suggest_category('Password changed', 'Bank account payment')['id'], 'service')
        self.assertIn('Финансовые сообщения', suggest_category('Password changed', 'Bank account payment')['alternatives'])
        self.assertTrue(suggest_category('', 'x' * 50001)['body_scan_truncated'])
        self.assertFalse(suggest_category('', 'x' * 50000)['body_scan_truncated'])

    def test_live_engine_decisions_remain_saved_synthetic_values(self):
        import json
        root = Path(__file__).resolve().parents[1]
        expected = json.loads((root / 'reports/release/synthetic_reference.json').read_bytes())
        engine = Engine(enable_laya=False)
        try:
            for path in sorted((root / 'examples/app').glob('*.eml')):
                r = engine.analyze(path.read_bytes(), 'baseline')
                old = expected[path.name]['baseline']
                self.assertEqual((r['status'], r['decision']), (old['status'], old['decision']))
                self.assertAlmostEqual(r['components']['E0']['score'], old['components']['E0']['score'], places=10)
                self.assertEqual(r['email']['category']['method'], 'keyword_rules_v1')
        finally:
            engine.close()


if __name__ == '__main__':
    unittest.main()
