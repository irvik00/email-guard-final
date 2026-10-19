"""Optional RU/EN topic hints. Handwritten rules, not a trained classifier."""
import re

# Rules describe topics, not consent, legitimacy or the spam decision.
RULES = [
    ('service', 'Служебные уведомления', [
        ('пароль / password', r'\b(?:парол\w*|passwords?)\b'),
        ('подтверждение / verification', r'\b(?:подтвержден\w*|верификац\w*|verification)\b'),
        ('код входа / sign-in code', r'\b(?:код\s+(?:входа|подтверждения)|sign[ -]?in\s+code|security\s+alert)\b'),
        ('уведомление системы', r'\b(?:служебн\w*\s+уведомлен\w*|system\s+notification)\b'),
    ]),
    ('job_offer', 'Предложения работы', [
        ('вакансия / vacancy', r'\b(?:ваканси\w*|vacanc\w*|job\s+offer|job\s+opening)\b'),
        ('собеседование / interview', r'\b(?:собеседован\w*|interviews?)\b'),
        ('резюме / resume', r'\b(?:резюме|resume|recruit\w*)\b'),
        ('предложение позиции', r'\b(?:предлож\w*\s+(?:вам\s+)?позици\w*|we\s+are\s+hiring)\b'),
    ]),
    ('financial', 'Финансовые сообщения', [
        ('выписка / statement', r'\b(?:выписк\w*|bank\s+statement|card\s+statement)\b'),
        ('платёж / payment', r'\b(?:платеж\w*|оплат\w*|payments?|invoice\w*|transaction\w*)\b'),
        ('карта / кредит', r'\b(?:карт[аыуе]|кредит\w*|credit\s+card|loan\w*)\b'),
        ('банковский счёт', r'\b(?:банковск\w*\s+счет\w*|bank\s+account|баланс\w*|balance)\b'),
    ]),
    ('advertising', 'Реклама', [
        ('скидка / discount', r'\b(?:скид\w*|discount\w*)\b'),
        ('распродажа / sale', r'\b(?:распродаж\w*|sale|clearance)\b'),
        ('промокод / promo', r'\b(?:промокод\w*|promo\s*code|coupon\w*)\b'),
        ('выгодное предложение', r'\b(?:специальн\w*\s+предложен\w*|special\s+offer|limited\s+offer)\b'),
    ]),
]
COMPILED = [(key, label, [(term, re.compile(pattern, re.I)) for term, pattern in rules])
            for key, label, rules in RULES]


def suggest_category(subject, body):
    original_body_length = len(str(body or ''))
    subject = str(subject or '')[:5000].casefold().replace('ё', 'е')
    body = str(body or '')[:50000].casefold().replace('ё', 'е')
    base = {'method': 'keyword_rules_v1', 'trained': False, 'confidence': None,
            'affects_spam_decision': False, 'validated_quality': False,
            'body_scan_truncated': original_body_length > 50000}
    if not subject.strip() and not body.strip():
        return base | {'id': None, 'value': None, 'matched_rules': [], 'alternatives': []}
    candidates = []
    for key, label, rules in COMPILED:
        matches = [(term, 2 if pattern.search(subject) else 1)
                   for term, pattern in rules if pattern.search(subject) or pattern.search(body)]
        if matches:
            candidates.append((sum(weight for _, weight in matches), key, label, [term for term, _ in matches]))
    # Python's stable sort preserves the documented rule priority for ties.
    candidates.sort(key=lambda c: c[0], reverse=True)
    if not candidates:
        return base | {'id': 'other', 'value': 'Другое', 'matched_rules': [], 'alternatives': []}
    _, key, label, terms = candidates[0]
    return base | {'id': key, 'value': label, 'matched_rules': terms,
                   'alternatives': [label for _, _, label, _ in candidates[1:3]]}
