'use strict';
const $ = id => document.getElementById(id);
const notes = {
  baseline: 'E0 анализирует тему и текст письма. Оценка сравнивается с сохранённым порогом. Если активных признаков нет, вердикт не выдаётся.',
  recall: 'Спам, если хотя бы один компонент сказал «спам». Нужны успешные ответы E0 и Laya. Такой режим может чаще ошибаться на обычных письмах.',
  review: 'При согласии E0 и Laya — общий результат. При разногласии или отказе — ручная проверка. Отсутствие модели не заменяется ответом другой.'
};
const statusText = {ok:'ответ получен', insufficient_features:'недостаточно признаков', unavailable:'недоступен', initializing:'загружается', error:'ошибка', busy:'занят'};
let readiness = null;
function node(tag, text, parent, css) {
  const element = document.createElement(tag);
  element.textContent = text;
  if (css) element.className = css;
  parent.appendChild(element);
  return element;
}
function modeNote() { $('mode-note').textContent = notes[$('mode').value]; }
$('mode').addEventListener('change', modeNote);
modeNote();
$('email').addEventListener('change', () => {
  const file = $('email').files[0];
  $('file-title').textContent = file ? file.name : 'Выберите письмо .eml';
  $('file-meta').textContent = file ? (file.size / 1024).toFixed(1) + ' KiB · готово к отправке' : 'Оригинал письма, до 10 MiB';
});
async function health() {
  try {
    const response = await fetch('/ready');
    readiness = await response.json();
    if (!readiness.modes) throw new Error('No readiness');
    for (const option of $('mode').options) option.disabled = !readiness.modes[option.value];
    const neural = readiness.modes.recall && readiness.modes.review;
    $('runtime-badge').textContent = neural ? 'E0 + Laya' : 'CPU · E0';
    $('runtime-badge').className = 'badge ' + (readiness.modes.baseline ? 'ready-badge' : 'neutral');
    $('health').textContent = readiness.modes.baseline
      ? (neural ? '● E0 и Laya готовы · доступны все три режима' : '● E0 готова · режимы с Laya на этом сервере недоступны')
      : 'Модель E0 пока недоступна. Дождитесь готовности сервера.';
  } catch (_) {
    $('health').textContent = 'Нет связи с приложением. Если сервер спит, дождитесь его запуска.';
    $('runtime-badge').textContent = 'Нет связи';
    $('runtime-badge').className = 'badge neutral';
  }
}
health(); setInterval(health, 10000);

function render(result) {
  const box = $('result');
  box.replaceChildren(); box.className = 'panel result-panel';
  const head = node('div', '', box, 'result-heading');
  node('span', '02 / РЕЗУЛЬТАТ', head, 'eyebrow');
  node('span', result.mode || 'Проверка файла', head, 'badge lilac-badge');
  if (result.error) {
    const warning = node('div', '', box, 'error-panel');
    node('h2', 'Не удалось выполнить анализ', warning);
    node('p', result.message || 'Вердикт отсутствует. Попробуйте ещё раз.', warning);
    return;
  }
  const hasDecision = result.status === 'ok';
  const verdict = node('div', '', box, 'verdict ' + (hasDecision ? (result.decision === 1 ? 'spam-verdict' : 'ham-verdict') : 'review-verdict'));
  node('span', hasDecision ? 'РЕШЕНИЕ МОДЕЛИ' : 'АВТОМАТИЧЕСКОГО РЕШЕНИЯ НЕТ', verdict, 'eyebrow');
  node('h2', result.status === 'needs_review' ? 'Нужна проверка' : hasDecision ? (result.decision === 1 ? 'Похоже на спам' : 'Обычное письмо') : 'Вердикт отсутствует', verdict);
  let explanation = hasDecision ? (result.decision === 1 ? 'Оценка пересекла порог выбранного режима. Проверьте контекст письма перед любым действием.' : 'Выбранный режим отнёс письмо к обычным. Это не подтверждение подлинности и не гарантия безопасности.') : 'Данных или доступных компонентов недостаточно. Неизвестность не считается обычным письмом.';
  if (result.reason === 'disagreement') explanation = 'E0 и Laya разошлись в решениях. Письмо требует ручного просмотра.';
  if (result.infrastructure_failure) explanation = 'Необходимый компонент недоступен. Автоматического переключения на другую модель нет.';
  node('p', explanation, verdict);

  const email = result.email;
  if (email) {
    node('p', email.subject || 'Без темы', box, 'meta-title');
    node('p', 'От: ' + (email.sender || 'не указан') + ' · подлинность не проверена', box, 'meta-sender');
    const topic = email.category || {};
    const topicBox = node('div', '', box, 'topic-card');
    node('span', 'ПРЕДПОЛАГАЕМАЯ КАТЕГОРИЯ', topicBox, 'eyebrow');
    node('h3', topic.value || 'Недостаточно данных', topicBox);
    node('p', topic.method === 'keyword_rules_v1' ? 'По ключевым словам RU/EN · правила, не обученная модель. На вердикт спама не влияет.' : 'Классификатор категорий не подключён.', topicBox);
    if (topic.matched_rules && topic.matched_rules.length) node('p', 'Сработали правила: ' + topic.matched_rules.join(', ') + '.', topicBox);
    if (topic.alternatives && topic.alternatives.length) node('p', 'Также есть признаки тем: ' + topic.alternatives.join(', ') + '.', topicBox);
    if (topic.body_scan_truncated) node('p', 'Для подсказки темы просмотрен только начальный фрагмент длинного письма.', topicBox);
    const stats = node('div', '', box, 'stat-grid');
    for (const [count, label] of [[email.body_chars, 'символов текста'], [email.link_count, 'ссылок'], [email.attachment_count, 'вложений']]) {
      const tile = node('div', '', stats, 'stat-tile');
      node('strong', String(count), tile); node('span', label, tile);
    }
  }

  node('h3', 'Оценки компонентов', box, 'detail-label');
  for (const [name, component] of Object.entries(result.components || {})) {
    const card = node('div', '', box, 'component-card');
    const heading = node('div', '', card, 'component-header');
    node('span', name === 'E0' ? 'E0 · TF-IDF + Logistic Regression' : name, heading);
    node('span', statusText[component.status] || component.status, heading, 'badge neutral');
    const finite = typeof component.score === 'number' && Number.isFinite(component.score);
    if (finite) {
      node('div', component.score.toFixed(6), card, 'score-number');
      const bar = node('progress', '', card, 'score-bar');
      bar.max = 1; bar.value = Math.max(0, Math.min(1, component.score));
      bar.setAttribute('aria-label', 'Оценка компонента ' + name);
      node('p', 'Порог: ' + component.threshold + ' · T: ' + component.temperature, card);
      node('p', 'Оценка модели, не гарантия вероятности спама для этого письма.', card);
    } else node('p', 'Оценка недоступна. Вердикт не подставляется.', card);
    if (component.status === 'insufficient_features') node('p', 'Нет активных TF-IDF-признаков. Показанная оценка диагностическая; решения по ней нет.', card);
    if (component.truncated && component.token_stats) node('p', 'Laya видит первые ' + component.token_stats.visible_tokens + ' из ' + component.token_stats.original_tokens + ' токенов. Остальная часть не участвовала в решении.', card);
    if (component.mask_literal_replacements) node('p', 'Служебный mask-токен заменён пробелом по сохранённому протоколу.', card);
  }

  if (email) {
    node('h3', 'Что проверить вручную', box, 'detail-label');
    const signals = node('ul', '', box, 'signals');
    let flags = 0;
    for (const link of email.links) {
      if (link.label_mismatch) {node('li', 'Домен в подписи ссылки отличается от адреса перехода.', signals, 'warning'); flags++;}
      if (link.scheme_high_risk || link.scheme_unusual) {node('li', 'Необычная схема ссылки: ' + link.scheme + '. Она не исполнялась.', signals, 'warning'); flags++;}
      if (flags >= 6) break;
    }
    const unsub = email.unsubscribe;
    node('li', unsub.header_present || unsub.body_hint_found ? 'Найдено указание на отписку. Работоспособность не проверялась.' : 'Указание на отписку не найдено. Само по себе это не доказывает спам.', signals, 'info');
    if (email.attachment_count) node('li', 'Есть вложения. Прочитаны только названия и MIME-типы; содержимое не проверялось.', signals, 'info');
    for (const warning of email.parser_warnings) node('li', 'Предупреждение разбора: ' + warning, signals, 'warning');
    const details = node('details', '', box, 'details-block');
    node('summary', 'Ссылки, вложения и технические детали', details);
    node('p', 'Наблюдения парсера описывают письмо и не объясняют решение ML-модели. Ссылки не открывались; HTML не исполнялся.', details);
    const list = node('ul', '', details);
    for (const link of email.links) node('li', (link.label ? 'Подпись: ' + link.label + ' · ' : '') + 'Адрес: ' + link.url, list);
    if (email.link_count > email.links.length) node('li', 'Показаны первые 50 ссылок.', list);
    for (const attachment of email.attachments) node('li', 'Вложение: ' + (attachment.filename || 'без имени') + ' · ' + attachment.content_type, list);
    if (email.attachment_count > email.attachments.length) node('li', 'Показаны первые 50 вложений.', list);
    node('p', 'Источник тела: ' + email.body_source + '. Подготовка модели: ' + Object.values(result.components).map(c => c.builder).join(', ') + '.', details);
    if (email.display_truncated) node('p', 'Длинные сведения сокращены для отображения.', details);
  }
  const rule = node('details', '', box, 'details-block');
  node('summary', 'Как принято решение', rule); node('p', notes[result.mode], rule);
  node('p', 'Шкалы компонентов различаются. Единой вероятности ансамбля нет. Подлинность отправителя и фишинг этим анализом не подтверждаются.', rule);
  if (window.matchMedia('(max-width: 650px)').matches) box.scrollIntoView({behavior:'smooth', block:'start'});
}

$('form').addEventListener('submit', async event => {
  event.preventDefault(); const file = $('email').files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.eml')) {render({error:'extension', message:'Выберите файл .eml.'}); return;}
  if (file.size > 10 * 1024 * 1024) {render({error:'size', message:'Размер письма превышает 10 MiB.'}); return;}
  $('submit').disabled = true; $('submit').textContent = 'Анализируем письмо…';
  try {
    const response = await fetch('/api/analyze?mode=' + encodeURIComponent($('mode').value), {method:'POST', headers:{'Content-Type':'message/rfc822'}, body:file});
    render(await response.json());
  } catch (_) {render({error:'network', message:'Нет ответа приложения. Вердикт отсутствует.'});}
  finally {$('submit').disabled = false; $('submit').textContent = 'Проанализировать письмо →'; health();}
});
