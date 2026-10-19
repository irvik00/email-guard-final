"""Чтение .eml для Email Guard. Только стандартная библиотека Python 3.12."""
import argparse
import json
import ipaddress
import re
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
UNSUBSCRIBE_PATTERN = re.compile(r"unsubscribe|отпис\w*|отказ\w*\s+от\s+рассыл", re.IGNORECASE)
HIGH_RISK_SCHEMES = {"javascript", "data", "file"}
COMMON_SCHEMES = {"http", "https", "mailto", "tel", "cid"}


def header(message: EmailMessage, name: str) -> str | None:
    value = message.get(name)
    return str(value) if value is not None else None


def read_headers(path: Path) -> dict:
    """Сохраняем интерфейс твоей первой функции."""
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    return {key: header(message, name) for key, name in {
        "subject": "Subject", "sender": "From", "recipient": "To",
        "list_unsubscribe": "List-Unsubscribe",
    }.items()}


class EmailHTMLParser(HTMLParser):
    """Извлекает текст и href без браузера и выполнения JavaScript."""
    BLOCKS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "table"}
    SKIP = {"script", "style", "head", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_chunks: list[str] = []
        self.links: list[dict] = []
        self.skip_stack: list[str] = []
        self.anchor: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple]) -> None:
        if tag in self.SKIP:
            self.skip_stack.append(tag)
        if self.skip_stack:
            return
        if tag in self.BLOCKS:
            self.text_chunks.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            self.anchor = {"url": href.strip(), "label": "", "source": "html"} if href else None
            if self.anchor is not None:
                self.links.append(self.anchor)

    def handle_endtag(self, tag: str) -> None:
        if self.skip_stack:
            if tag == self.skip_stack[-1]:
                self.skip_stack.pop()
            return
        if tag == "a":
            self.anchor = None
        if tag in self.BLOCKS:
            self.text_chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_stack:
            return
        self.text_chunks.append(data)
        if self.anchor is not None:
            self.anchor["label"] += data

    def result(self) -> tuple[str, list[dict]]:
        text = "\n".join(" ".join(line.split()) for line in "".join(self.text_chunks).splitlines())
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        for link in self.links:
            link["label"] = " ".join(link["label"].split())
        return text, self.links


def body_candidates(part: EmailMessage):
    """Не включаем вложения, включая приложенные письма, в тело."""
    if part.get_content_disposition() == "attachment" or part.get_filename() is not None:
        return
    if part.get_content_maintype() == "message":
        return
    if part.is_multipart():
        children = list(part.iter_parts())
        if part.get_content_type() == "multipart/related" and children:
            # В related тело — корневая часть; остальное обычно картинки/ресурсы.
            start = part.get_param("start")
            children = [next((child for child in children if start and
                             header(child, "Content-ID") == start), children[0])]
        for child in children:
            yield from body_candidates(child)
    elif part.get_content_type() in {"text/plain", "text/html"}:
        yield part


def decode_text(part: EmailMessage | None, warnings: list[str]) -> str | None:
    if part is None:
        return None
    try:
        text = part.get_content()
    except (LookupError, UnicodeError):
        warnings.append("Не удалось использовать заявленную кодировку; применён UTF-8 с заменой ошибок.")
        text = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")
    if "\ufffd" in text:
        warnings.append("В тексте есть символы замены: возможна ошибка кодировки.")
    return text.strip()


def attachments(part: EmailMessage) -> list[dict]:
    filename = part.get_filename()
    disposition = part.get_content_disposition()
    if filename is not None or disposition == "attachment" or part.get_content_maintype() == "message":
        return [{"filename": filename, "content_type": part.get_content_type(),
                 "disposition": disposition}]
    result = []
    if part.is_multipart():
        for child in part.iter_parts():
            result.extend(attachments(child))
    return result


def domain_metadata(host: str | None) -> dict:
    """Сравнимый домен плюс исходное значение. Встроенный IDNA, не IDNA 2008."""
    metadata = {
        "domain_original": host, "domain": None,
        "source_has_non_ascii": any(ord(c) > 127 for c in host) if host is not None else None,
        "source_has_punycode": any(label.lower().startswith("xn--") for label in host.split("."))
                               if host is not None else None,
        "normalization_error": None,
    }
    if not host:
        return metadata
    candidate = host.removesuffix(".")
    try:
        metadata["domain"] = str(ipaddress.ip_address(candidate))
        return metadata
    except ValueError:
        pass  # Это имя, а не IP-адрес.
    try:
        if not candidate or any(not label for label in candidate.split(".")):
            raise ValueError("Пустая метка домена")
        normalized = candidate.encode("idna").decode("ascii").lower()
        labels = normalized.split(".")
        if len(normalized) > 253 or any(
            len(label) > 63 or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
            for label in labels
        ):
            raise ValueError("Недопустимый формат домена")
        # Проверяем, что уже переданная ACE-метка действительно декодируется.
        for label in labels:
            if label.startswith("xn--"):
                label.encode("ascii").decode("idna")
        metadata["domain"] = normalized
    except (UnicodeError, ValueError) as error:
        metadata["normalization_error"] = type(error).__name__
    return metadata


def label_domain(label: str | None) -> str | None:
    """Распознаём только подпись, целиком состоящую из URL или имени хоста."""
    if not label:
        return None
    value = label.strip()
    if any(char.isspace() for char in value):
        return None
    try:
        if value.lower().startswith(("https://", "http://")) or value.startswith("//"):
            parsed = urlsplit(value)
        else:
            # Обычный текст вроде «Открыть» не считается адресом.
            if "." not in value or any(char in value for char in "@?#"):
                return None
            parsed = urlsplit("//" + value)
            if not parsed.hostname or "." not in parsed.hostname.removesuffix("."):
                return None
        return domain_metadata(parsed.hostname)["domain"]
    except ValueError:
        return None


def link_metadata(link: dict) -> dict:
    parse_error = None
    try:
        parsed = urlsplit(link["url"])
        scheme, host = parsed.scheme.lower(), parsed.hostname
    except ValueError as error:
        match = re.match(r"^([a-z][a-z0-9+.-]*):", link["url"], re.IGNORECASE)
        scheme = match.group(1).lower() if match else None
        host = None
        parse_error = type(error).__name__
    metadata = domain_metadata(host)
    shown_domain = label_domain(link["label"])
    mismatch = (shown_domain != metadata["domain"]
                if shown_domain is not None and metadata["domain"] is not None else None)
    return {
        **link, **metadata, "scheme": scheme, "url_parse_error": parse_error,
        "label_domain": shown_domain, "label_mismatch": mismatch,
        "scheme_high_risk": scheme in HIGH_RISK_SCHEMES if scheme is not None else None,
        "scheme_unusual": bool(scheme) and scheme not in COMMON_SCHEMES | HIGH_RISK_SCHEMES
                          if scheme is not None else None,
        "has_explicit_scheme": bool(scheme) if scheme is not None else None,
    }


def parse_email(raw_bytes: bytes) -> dict:
    """Общий вход для будущих .eml-загрузок в API."""
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    warnings = []
    for part in message.walk():
        warnings.extend(type(defect).__name__ for defect in part.defects)
    for name in ("Subject", "From", "To", "List-Unsubscribe", "List-Unsubscribe-Post"):
        if len(message.get_all(name, [])) > 1:
            warnings.append(f"Заголовок {name} повторяется; показано первое значение.")
    candidates = list(body_candidates(message))
    plain_part = next((p for p in candidates if p.get_content_type() == "text/plain"), None)
    html_part = next((p for p in candidates if p.get_content_type() == "text/html"), None)
    plain_text = decode_text(plain_part, warnings)
    html = decode_text(html_part, warnings)
    html_text, html_links = None, []
    if html is not None:
        parser = EmailHTMLParser()
        parser.feed(html)
        parser.close()
        html_text, html_links = parser.result()
    # Пустой plain не должен скрывать непустую HTML-версию.
    if plain_text:
        body_text, body_source = plain_text, "plain"
    elif html_text is not None:
        body_text, body_source = html_text, "html"
    elif plain_text is not None:
        body_text, body_source = plain_text, "plain"
    else:
        body_text, body_source = None, None
    links = []
    seen = set()
    text_links = []
    for text, source in ((plain_text, "plain"), (html_text, "html_text")):
        for match in URL_PATTERN.finditer(text or ""):
            # Это приближённое извлечение URL из свободного текста.
            url = match.group().rstrip(".,;!?)\u00bb")
            text_links.append({"url": url, "label": None, "source": source})
    for link in html_links + text_links:
        key = (link["url"], link["label"], link["source"])
        if key not in seen:
            seen.add(key)
            links.append(link_metadata(link))
    sender = header(message, "From")
    sender_address = parseaddr(sender)[1] if sender is not None else None
    sender_host = (sender_address.rsplit("@", 1)[-1]
                   if sender_address and "@" in sender_address else None)
    sender_domain_info = domain_metadata(sender_host)
    headers_raw = []
    for name, value in message.raw_items():
        safe_name = name.encode("utf-8", errors="backslashreplace").decode("utf-8")
        safe_value = value.encode("utf-8", errors="backslashreplace").decode("utf-8")
        if safe_name != name or safe_value != value:
            warnings.append("Недекодируемые байты заголовка представлены escape-последовательностями в headers_raw.")
        headers_raw.append({"name": safe_name, "value": safe_value})
    unsub = header(message, "List-Unsubscribe")
    unsub_post = header(message, "List-Unsubscribe-Post")
    unsub_links = [link["url"] for link in html_links if
                   UNSUBSCRIBE_PATTERN.search((link["label"] or "") + " " + link["url"])]
    combined = "\n".join(filter(None, (plain_text, html_text)))
    letters = [char for char in (body_text or "") if char.isalpha()]
    return {
        "subject": header(message, "Subject"), "sender": sender,
        "sender_address": sender_address,
        "sender_domain": sender_domain_info["domain"],
        "sender_domain_info": sender_domain_info,
        "recipient": header(message, "To"),
        # raw_items сохраняет порядок и повторы, но это не побайтовый оригинал.
        "headers_raw": headers_raw,
        "plain_text": plain_text, "html": html, "html_text": html_text,
        "body_text": body_text, "body_source": body_source,
        "links": links, "attachments": attachments(message),
        "unsubscribe": {
            "list_unsubscribe": unsub, "list_unsubscribe_post": unsub_post,
            "header_present": unsub is not None,
            "body_hint_found": bool(UNSUBSCRIBE_PATTERN.search(combined)),
            "html_candidate_links": list(dict.fromkeys(unsub_links)),
            "one_click_declared": unsub_post is not None and
                                  unsub_post.strip().lower() == "list-unsubscribe=one-click",
            "working": None,
        },
        "formatting": {
            "has_html": html is not None,
            "body_char_count": len(body_text) if body_text is not None else None,
            "uppercase_letter_ratio": sum(c.isupper() for c in letters) / len(letters)
                                      if letters else None,
            "has_repeated_exclamations": bool(re.search(r"!{3,}", body_text))
                                        if body_text is not None else None,
        },
        "parse_warnings": list(dict.fromkeys(warnings)),
    }


def read_email(path: Path) -> dict:
    return parse_email(path.read_bytes())


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Разобрать письмо .eml в JSON без сетевых запросов.")
    parser.add_argument("path", nargs="?", type=Path,
                        default=project_root / "data" / "examples" / "service_notice.eml")
    parser.add_argument("--output", type=Path, help="Дополнительно сохранить JSON в файл.")
    args = parser.parse_args()
    try:
        result = read_email(args.path)
    except OSError as error:
        parser.exit(1, f"Не удалось прочитать файл: {error}\n")
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        except OSError as error:
            parser.exit(1, f"Не удалось сохранить JSON: {error}\n")


if __name__ == "__main__":
    main()
