'use strict';
const $ = id => document.getElementById(id);
const notes = {
  baseline: 'E0 с исходным порогом. При недостатке признаков вердикта нет.',
  recall: 'Спам, если хотя бы один компонент сказал «спам». Требуются успешные ответы обоих компонентов. Этот режим может чаще ошибаться на обычных письмах.',
  review: 'При согласии — общий результат; при разногласии или отказе — ручная проверка. Автоматического переключения на baseline нет.'
};
const statusText = {ok:'ответ получен', insufficient_features:'недостаточно признаков', unavailable:'недоступен', initializing:'загружается', error:'ошибка', busy:'занят'};
let readiness = null;
function node(tag, text, parent, css) {
  const n = document.createElement(tag); n.textContent = text;
  if (css) n.className = css;
  parent.appendChild(n); return n;
}
function modeNote() { $('mode-note').textContent = notes[$('mode').value]; }
$('mode').addEventListener('change', modeNote); modeNote();
async function health() {
  try {
    readiness = await (await fetch('/ready')).json();
    for (const option of $('mode').options) option.disabled = !readiness.modes[option.value];
    $('health').textContent = ['baseline','recall','review'].map(m => m + ': ' + (readiness.modes[m] ? 'готов' : 'недоступен / загружается')).join(' · ');
  } catch (_) { $('health').textContent = 'Нет связи с приложением.'; }
}
health(); setInterval(health, 10000);
function render(r) {
  const box=$('result'); box.replaceChildren(); box.hidden=false;
  if (r.error) { node('h2', r.message || 'Обработка не выполнена', box); return; }
  let title = r.status==='needs_review' ? 'Нужна проверка' : r.status==='ok' ? (r.decision===1 ? 'Спам' : 'Обычное') : 'Вердикт отсутствует';
  node('h2', title, box, r.status==='ok' && r.decision===1 ? 'spam' : '');
  node('p', 'Режим: ' + r.mode, box);
  if (r.reason==='disagreement') node('p', 'Компоненты разошлись в решениях. Требуется ручной просмотр.', box);
  if (r.reason==='component_unavailable' || r.infrastructure_failure) node('p', 'Необходимый компонент не ответил успешно. Решение не заменено ответом другой модели.', box);
  const e=r.email;
  if (e) {
    node('h3','Сведения из письма — подлинность не проверена',box);
    node('p','Тема: ' + (e.subject || 'не указана'),box);
    node('p','Отправитель (From): ' + (e.sender || 'не указан'),box);
    node('p','Категория: не определена. Проверенный классификатор категорий не подключён.',box);
    node('h3','Наблюдения парсера',box);
    node('p','Наблюдения описывают содержимое письма и не объясняют решение ML-модели.',box,'muted');
    const list=node('ul','',box);
    node('li','Текст: ' + e.body_chars + ' символов; источник тела: ' + e.body_source + '.',list);
    if (!e.body_chars) node('li','Тело письма пусто: информации для анализа может быть недостаточно.',list);
    node('li','Ссылок: ' + e.link_count + '. Они не открывались и не являются кликабельными.',list);
    for (const l of e.links) {
      node('li',(l.label ? 'Подпись: '+l.label+' · ' : '') + 'Адрес: ' + l.url, list);
      if (l.label_mismatch) node('li','Подозрительный признак: домен в подписи ссылки отличается от домена адреса перехода.',list);
      if (l.scheme_high_risk || l.scheme_unusual) node('li','Подозрительный признак: схема ссылки «' + l.scheme + '». Ссылка не исполнялась.',list);
    }
    if(e.link_count>e.links.length) node('li','Остальные ссылки скрыты из-за лимита отображения (50).',list);
    node('li','Вложений: '+e.attachment_count+'. Содержимое не открывалось.',list);
    for(const a of e.attachments) node('li','Вложение: '+(a.filename||'без имени')+' · '+a.content_type,list);
    if(e.attachment_count>e.attachments.length) node('li','Остальные вложения скрыты из-за лимита отображения (50).',list);
    node('li',e.unsubscribe.header_present || e.unsubscribe.body_hint_found ? 'Есть указание на отписку. Работоспособность не проверена; это не доказательство легитимности.' : 'Указание на отписку не найдено. Это само по себе не доказывает спам.',list);
    for(const w of e.parser_warnings) node('li','Предупреждение разбора: '+w,list);
    if(e.display_truncated) node('li','Длинные сведения в интерфейсе сокращены.',list);
  }
  const details=node('details','',box); node('summary','Оценки компонентов и правило решения',details);
  node('p',notes[r.mode],details);
  for (const [name,c] of Object.entries(r.components)) {
    node('h3',name,details);
    node('p','Статус: '+(statusText[c.status]||c.status)+(c.reason?' · '+c.reason:''),details);
    node('p','Оценка: '+(c.score===null ? 'нет' : Number(c.score).toFixed(6))+' · порог: '+c.threshold+' · T: '+c.temperature,details);
    node('p','Решение: '+(c.decision===null?'отсутствует':c.decision===1?'спам':'обычное')+' · подготовка: '+c.builder,details);
    if(c.status==='insufficient_features') node('p','Недостаточно признаков. Диагностическая оценка не считается вердиктом.',details);
    if(c.truncated) node('p','Вход Laya усечён: видимы первые '+c.token_stats.visible_tokens+' из '+c.token_stats.original_tokens+' токенов. Остальная часть не участвовала в решении.',details);
    if(c.mask_literal_replacements) node('p','Служебный mask-токен в тексте заменён пробелом по сохранённому протоколу.',details);
  }
  node('p','Шкалы компонентов различаются. Единой вероятности ансамбля нет.',details);
}
$('form').addEventListener('submit',async event=>{
  event.preventDefault(); const f=$('email').files[0];
  if(!f) return;
  if(!f.name.toLowerCase().endsWith('.eml')) { render({error:'extension',message:'Выберите файл .eml.'}); return; }
  if(f.size>10*1024*1024) { render({error:'size',message:'Размер письма превышает 10 MiB.'}); return; }
  $('submit').disabled=true; $('submit').textContent='Анализируем…';
  try {
    const response=await fetch('/api/analyze?mode='+encodeURIComponent($('mode').value), {method:'POST',headers:{'Content-Type':'message/rfc822'},body:f});
    render(await response.json());
  } catch(_) { render({error:'network',message:'Нет ответа приложения. Вердикт отсутствует.'}); }
  finally { $('submit').disabled=false; $('submit').textContent='Анализировать'; health(); }
});
