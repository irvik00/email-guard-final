"""Проверяем смысл разбора писем, а не оценку будущей ML-модели."""
import sys
import unittest
import json
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from read_email import parse_email, read_email, read_headers, domain_metadata, link_metadata, EmailHTMLParser


class EmailReaderTests(unittest.TestCase):
    def parse(self, message):
        return parse_email(message.as_bytes())

    def test_original_example(self):
        path = Path(__file__).resolve().parents[1] / "data/examples/service_notice.eml"
        headers = read_headers(path)
        self.assertEqual(headers["subject"], "Подтверждение смены пароля")
        self.assertIsNone(headers["list_unsubscribe"])
        result = read_email(path)
        self.assertEqual(result["sender_domain"], "service.example")
        self.assertIn("пароль", result["body_text"].lower())

    def test_missing_is_different_from_empty_header(self):
        absent = parse_email(b"Subject: example\r\n\r\nText")
        empty = parse_email(b"From:\r\nSubject: example\r\n\r\nText")
        self.assertIsNone(absent["sender"])
        self.assertEqual(empty["sender"], "")

    def test_alternatives_are_separate_and_html_links_survive(self):
        message = EmailMessage()
        message.set_content("Текстовая версия.")
        message.add_alternative('<p>HTML версия. <a href="https://target.example">Кнопка</a></p>', subtype="html")
        result = self.parse(message)
        self.assertEqual(result["body_text"], "Текстовая версия.")
        self.assertEqual(result["body_source"], "plain")
        self.assertEqual(result["links"][0]["url"], "https://target.example")
        self.assertEqual(result["links"][0]["label"], "Кнопка")

    def test_empty_plain_falls_back_to_html(self):
        message = EmailMessage()
        message.set_content("")
        message.add_alternative("<p>Только HTML.</p>", subtype="html")
        result = self.parse(message)
        self.assertEqual(result["body_source"], "html")
        self.assertEqual(result["body_text"], "Только HTML.")

    def test_html_skips_script_style_and_decodes_entities(self):
        message = EmailMessage()
        message.set_content('<head><title>Title</title></head><style>BAD STYLE</style>'
                            '<script>BAD SCRIPT</script><p>А &amp; Б</p><p>В</p>', subtype="html")
        result = self.parse(message)
        self.assertEqual(result["body_text"], "А & Б\n\nВ")

    def test_text_attachment_never_becomes_body(self):
        message = EmailMessage()
        message.set_content("Основное письмо.")
        message.add_attachment("Не текст письма: https://attachment.example", filename="заметки.txt")
        result = self.parse(message)
        self.assertEqual(result["body_text"], "Основное письмо.")
        self.assertEqual(result["attachments"][0]["filename"], "заметки.txt")
        self.assertEqual(result["links"], [])

    def test_attached_email_is_not_merged(self):
        nested = EmailMessage()
        nested.set_content("Чужое письмо https://nested.example")
        message = EmailMessage()
        message.set_content("Основное письмо.")
        message.add_attachment(nested, filename="forwarded.eml")
        result = self.parse(message)
        self.assertEqual(len(result["attachments"]), 1)
        self.assertNotIn("Чужое", result["body_text"])
        self.assertEqual(result["links"], [])

    def test_inline_named_text_is_an_attachment(self):
        message = EmailMessage()
        message.make_mixed()
        attachment = EmailMessage()
        attachment.set_content("Вложение.")
        attachment.add_header("Content-Disposition", "inline", filename="notes.txt")
        message.attach(attachment)
        body = EmailMessage()
        body.set_content("Настоящее тело.")
        message.attach(body)
        result = self.parse(message)
        self.assertEqual(result["body_text"], "Настоящее тело.")
        self.assertEqual(result["attachments"][0]["filename"], "notes.txt")

    def test_unknown_charset_does_not_silently_pass(self):
        result = parse_email(b'Content-Type: text/plain; charset="nonexistent"\r\n\r\nHello')
        self.assertEqual(result["body_text"], "Hello")
        self.assertTrue(result["parse_warnings"])

    def test_cp1251_text(self):
        message = EmailMessage()
        message.set_content("Письмо в старой кодировке.", charset="windows-1251")
        result = self.parse(message)
        self.assertEqual(result["body_text"], "Письмо в старой кодировке.")
        self.assertEqual(result["parse_warnings"], [])

    def test_related_start_selects_body(self):
        message = EmailMessage()
        message.make_related()
        message.set_param("start", "<body>")
        resource = EmailMessage()
        resource.set_content("Ресурс, не тело.")
        resource["Content-ID"] = "<resource>"
        body = EmailMessage()
        body.set_content("<p>Правильное тело.</p>", subtype="html")
        body["Content-ID"] = "<body>"
        message.attach(resource)
        message.attach(body)
        result = self.parse(message)
        self.assertEqual(result["body_text"], "Правильное тело.")

    def test_attachment_only_has_no_body(self):
        message = EmailMessage()
        message.add_attachment(b"example", maintype="application", subtype="octet-stream", filename="file.bin")
        result = self.parse(message)
        self.assertIsNone(result["body_text"])
        self.assertIsNone(result["formatting"]["body_char_count"])

    def test_unsubscribe_is_declared_but_not_verified(self):
        message = EmailMessage()
        message["List-Unsubscribe"] = "<https://news.example/unsubscribe>"
        message["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
        message.set_content('<a href="https://news.example/unsubscribe">Отписаться</a>', subtype="html")
        unsub = self.parse(message)["unsubscribe"]
        self.assertTrue(unsub["header_present"])
        self.assertTrue(unsub["one_click_declared"])
        self.assertEqual(unsub["html_candidate_links"], ["https://news.example/unsubscribe"])
        self.assertIsNone(unsub["working"])

    def test_duplicate_subject_warns(self):
        result = parse_email(b"Subject: first\r\nSubject: second\r\n\r\nBody")
        self.assertEqual(result["subject"], "first")
        self.assertTrue(any("Subject" in warning for warning in result["parse_warnings"]))

    def test_bad_url_is_preserved_with_unknown_domain(self):
        message = EmailMessage()
        message.set_content('<a href="https://[broken">Ссылка</a>', subtype="html")
        link = self.parse(message)["links"][0]
        self.assertEqual(link["url"], "https://[broken")
        self.assertIsNone(link["domain"])

    def test_formatting_uses_letters_not_punctuation(self):
        message = EmailMessage()
        message.set_content("Аа 123!!!")
        formatting = self.parse(message)["formatting"]
        self.assertEqual(formatting["uppercase_letter_ratio"], 0.5)
        self.assertTrue(formatting["has_repeated_exclamations"])


class LinkAndHeaderTests(unittest.TestCase):
    def test_self_closing_tags_use_inherited_handler(self):
        parser = EmailHTMLParser()
        parser.feed('<p>A<br/>B</p><a href="https://example.org"/>C')
        text, links = parser.result()
        self.assertIn("A", text)
        self.assertIn("B", text)
        self.assertEqual(links[0]["label"], "")
        self.assertNotIn("C", links[0]["label"])

    def test_domain_normalization_preserves_source_flags(self):
        unicode_domain = domain_metadata("ПРИМЕР.РФ.")
        ace_domain = domain_metadata("XN--E1AFMKFD.XN--P1AI")
        self.assertEqual(unicode_domain["domain"], "xn--e1afmkfd.xn--p1ai")
        self.assertEqual(ace_domain["domain"], unicode_domain["domain"])
        self.assertTrue(unicode_domain["source_has_non_ascii"])
        self.assertFalse(unicode_domain["source_has_punycode"])
        self.assertTrue(ace_domain["source_has_punycode"])

    def test_invalid_domain_is_not_silently_corrected(self):
        for domain in ("example..org", "example.org..", "bad_domain.org", "xn--.org"):
            with self.subTest(domain=domain):
                result = domain_metadata(domain)
                self.assertIsNone(result["domain"])
                self.assertIsNotNone(result["normalization_error"])

    def test_ip_address_is_not_idna_encoded(self):
        self.assertEqual(domain_metadata("2001:DB8::1")["domain"], "2001:db8::1")

    def test_label_mismatch_has_three_states(self):
        def link(label, url="https://bank.example/login"):
            return link_metadata({"label": label, "url": url, "source": "html"})
        self.assertFalse(link("https://BANK.example.")["label_mismatch"])
        self.assertFalse(link("bank.example")["label_mismatch"])
        self.assertTrue(link("https://bank.example", "https://other.example")["label_mismatch"])
        self.assertIsNone(link("Открыть личный кабинет")["label_mismatch"])
        self.assertIsNone(link("Подробнее.")["label_mismatch"])

    def test_equivalent_unicode_and_ace_domains_do_not_mismatch(self):
        result = link_metadata({"label": "https://пример.рф", "url": "https://xn--e1afmkfd.xn--p1ai", "source": "html"})
        self.assertFalse(result["label_mismatch"])

    def test_risky_unusual_common_and_relative_schemes(self):
        for url, risky, unusual in [
            ("javascript:void(0)", True, False), ("data:text/plain,Hi", True, False),
            ("file:///C:/sample.txt", True, False), ("ftp://files.example", False, True),
            ("tel:+12345", False, False), ("mailto:a@example.org", False, False),
            ("#section", False, False), ("/settings", False, False),
            ("//example.org/path", False, False),
        ]:
            with self.subTest(url=url):
                result = link_metadata({"label": None, "url": url, "source": "html"})
                self.assertEqual(result["scheme_high_risk"], risky)
                self.assertEqual(result["scheme_unusual"], unusual)

    def test_raw_headers_preserve_duplicates_and_order(self):
        result = parse_email(b"Received: first\r\nReceived: second\r\nSubject: test\r\n\r\nBody")
        self.assertEqual(result["headers_raw"][:2], [
            {"name": "Received", "value": "first"}, {"name": "Received", "value": "second"}])

    def test_invalid_header_bytes_can_be_saved_as_utf8_json(self):
        result = parse_email(b"X-Test: \xff\r\nSubject: test\r\n\r\nBody")
        json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.assertTrue(result["parse_warnings"])


class SavedFixtureTests(unittest.TestCase):
    def parse_fixture(self, name):
        return read_email(Path(__file__).resolve().parents[1] / "data/fixtures" / name)

    def test_plain(self):
        result = self.parse_fixture("plain.eml")
        self.assertEqual(result["body_source"], "plain")
        self.assertIn("meeting", result["body_text"])

    def test_html(self):
        result = self.parse_fixture("html.eml")
        self.assertEqual(result["body_source"], "html")
        self.assertEqual(result["body_text"], "Hello\n\nworld")

    def test_alternative(self):
        result = self.parse_fixture("alternative.eml")
        self.assertEqual(result["body_text"], "Plain version")
        self.assertEqual(result["links"][0]["url"], "https://example.org/hidden-in-html")

    def test_related_image(self):
        result = self.parse_fixture("related_image.eml")
        self.assertEqual(result["body_text"], "Body with image")
        self.assertEqual(result["attachments"][0]["filename"], "logo.png")

    def test_attached_email(self):
        result = self.parse_fixture("nested_email.eml")
        self.assertEqual(result["body_text"], "Outer body")
        self.assertNotIn("Inner secret", result["body_text"])
        self.assertEqual(result["attachments"][0]["content_type"], "message/rfc822")

    def test_broken_charset(self):
        result = self.parse_fixture("broken_charset.eml")
        self.assertEqual(result["body_text"], "Hello fallback")
        self.assertTrue(result["parse_warnings"])

    def test_suspicious_links(self):
        result = self.parse_fixture("suspicious_links.eml")
        html_links = [link for link in result["links"] if link["source"] == "html"]
        self.assertTrue(html_links[0]["label_mismatch"])
        self.assertTrue(html_links[1]["scheme_high_risk"])
        self.assertFalse(html_links[2]["scheme_high_risk"])


if __name__ == "__main__":
    unittest.main()
