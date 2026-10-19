"""Post-inference, aggregate-only interpretation. Never changes labels/models/settings."""
import argparse
import shutil
from personal_eval_common import *

def marked_append(path,text):
    start='<!-- derived-findings -->';end='<!-- /derived-findings -->'
    original=path.read_text(encoding='utf-8')
    if start in original:original=original[:original.index(start)].rstrip()+'\n'
    path.write_text(original+'\n'+start+'\n'+text+'\n'+end+'\n',encoding='utf-8')

def render(out):
    boundary=PersonalBoundary(out).install()
    manifest=read_json(out/'manifest.json');result=read_json(out/'comparison.json');rows=read_rows(out/'predictions.jsonl')
    comp=result['comparisons'];main=result['main_models'];members={r['record_id']:r for r in manifest['records']}
    modelrows={name:{r['record_id']:r for r in rows if r['model_id']==name} for name in [c['name'] for c in manifest['candidates']]}
    original=out/'generated_initial_reports';original.mkdir(exist_ok=True)
    names=['PERSONAL_EMAIL_EVALUATION.md','LAYA_PERSONAL_DIAGNOSTICS.md','FINAL_MODEL_RECOMMENDATION.md','PRESENTATION_NOTES_PERSONAL_EMAILS.md']
    for name in names:
        if not (original/name).exists():shutil.copyfile(out/name,original/name)
    def counts_table(part='confirmed'):
        table=['| Модель | Покрытие | TP / FP / TN / FN | F1 |','|---|---:|---|---:|']
        for name in main:
            v=comp[name][part];s=v['metrics'];f='N/A' if s['F1'] is None else f"{s['F1']:.4f}"
            table.append(f"| {name} | {v['successful']}/{v['records']} | {s['TP']} / {s['FP']} / {s['TN']} / {s['FN']} | {f} |")
        return '\n'.join(table)
    # No changed membership/labels: these are explanatory notes after inference, not tuning.
    qualitative={'created_utc':now(),'status':'post_inference_explanatory_only','no_new_inference_or_selection':True,
        'original_predeclared_groups_unchanged':True,
        'money_prize_group_note':'Frozen reason-keyword group includes mail004 because its reason references prize messages; this is not a directly readable money claim in damaged body.',
        'mixed_scripts_note':'Frozen script-count groups do not identify languages and can miss short Cyrillic insertions in long foreign templates.',
        'all_main_models_FN':[rid for rid in members if members[rid]['spam_label']==1 and
            all(modelrows[name][rid]['status']=='ok' and modelrows[name][rid]['decision']==0 for name in main)],
        'all_main_models_FP':[rid for rid in members if members[rid]['spam_label']==0 and
            all(modelrows[name][rid]['status']=='ok' and modelrows[name][rid]['decision']==1 for name in main)]}
    scores={}
    for name in ['Laya_day5','Laya_extra_A','Laya_extra_B']:
        ref=modelrows['Laya_base_B_calibrated'];new=modelrows[name]
        ok=[rid for rid in ref if ref[rid]['status']=='ok' and new[rid]['status']=='ok']
        scores[name]={'n':len(ok),'decisions_changed_from_calibrated_base':[rid for rid in ok if ref[rid]['decision']!=new[rid]['decision']],
            'mean_abs_score_change':sum(abs(ref[rid]['score']-new[rid]['score']) for rid in ok)/len(ok) if ok else None,
            'max_abs_score_change':max((abs(ref[rid]['score']-new[rid]['score']) for rid in ok),default=None),
            'note':'Same decisions do not imply identical weights or scores; T/threshold differ by saved candidate'}
    qualitative['adapted_laya_vs_calibrated_base']=scores;write_json(out/'qualitative_analysis.json',qualitative)
    campaigns=read_json(out/'campaign_evidence.json')
    notes=['## Содержательный разбор после предсказаний', '',
        'Это объяснение уже выполненного фиксированного эксперимента. Метки, manifest, части набора, вопросы, веса, T и пороги не изменены; новых прогонов/настроек нет.',
        '', '### Ошибки в заданных группах', '', '| Модель | Реклама с пользовательской меткой ham: FP/n | Обычные служебные: FP/n | Группа reason о деньгах/призах: FN/n | Mixed scripts: FP/FN/n |','|---|---:|---:|---:|---|']
    for name in main:
        a=comp[name]['user_ham_advertising'];b=comp[name]['ordinary_service_notifications'];c=comp[name]['money_or_prize_claims'];d=comp[name]['mixed_scripts']
        notes.append(f"| {name} | {a['metrics']['FP']}/{a['records']} | {b['metrics']['FP']}/{b['records']} | {c['metrics']['FN']}/{c['records']} | {d['metrics']['FP']}/{d['metrics']['FN']}/{d['records']} |")
    notes+=['', 'Ложные срабатывания на mail017/019 — ошибки относительно окончательных пользовательских меток0, не повод переоткрывать разметку. Согласованная рекламная рассылка может выглядеть text-only модели как массовая реклама: фактическую историю согласия она не получает. Это ограничение доступных признаков, а не доказанная причина конкретного внутреннего решения модели.',
        f"Общий пропуск всех8 основных вариантов: {', '.join(qualitative['all_main_models_FN']) or 'нет'}. Общее ложное срабатывание: {', '.join(qualitative['all_main_models_FP']) or 'нет'}.",
        'mail013 содержит контекстно зависимое обращение: бинарная метка подтверждена пользователем, но контекст прежней переписки/ожидаемости не входит в текстовые модели. mail014 — подтверждённое обычное финансовое уведомление; MIME parser отмечает CloseBoundaryNotFoundDefect и недекодируемые байты заголовка. Парсинг не падал; нельзя приписать все FP этому дефекту без отдельного контролируемого исследования. Для сравнения применён неизменный существующий parser, PDF не открывался.',
        'mail004 имеет предупреждение о символах замены. Его final spam_label=1 остаётся подтверждённым. Frozen группа reason о деньгах/призах включает это письмо по упоминанию других призовых писем в объяснении; повреждённый текст не доказывает читаемое обещание денег. Членство группы не исправлялось по предсказаниям.',
        'Mixed scripts — только формальная характеристика письма. Доля кириллицы может быть ниже10% в длинном иностранном автоответе с короткой русской вставкой; такие письма могут попасть в Latin-dominant/other группу. Нельзя объявить Latin-dominant современной английской выборкой или оценивать языковую поддержку по этой эвристике.',
        '', '### Связанность и усечение', '']
    for g in campaigns['shared_link_host_groups']:notes.append('- Общий host SHA '+g['host_sha256'][:12]+'…: '+', '.join(g['ids'])+'.')
    notes+=['', 'Совпадение внешнего хоста/сокращателя или unsubscribe-поставщика само по себе не удостоверяет одну кампанию. Exact/canonical дубликатов0; near-text пар при заранее заданном Jaccard>=0.55 — '+str(len(campaigns['similar_text_pairs']))+'. Связанность важна для интерпретации малых счётчиков ошибок; число независимых атак не установлено.',
        '', '| Модель | Усечено /22 | Максимальная последовательность |','|---|---:|---:|']
    for name in main:
        entries=list(modelrows[name].values());mx=max((r.get('sequence_tokens',0) for r in entries),default=0)
        notes.append(f"| {name} | {comp[name]['truncated']['records']} | {mx if mx else 'нет token cap'} |")
    notes+=['', 'Предупреждение tokenizer о полном числе токенов сверх model_max_length относится к предварительному подсчёту/offset mapping. На encoder реально передавались только проверенные capped последовательности: Laya в пределах context1024, C512. Это не runtime indexing error; фактические input-ID SHA и размеры записаны для каждого письма. Усечение не скрывалось и не менялось ради правильного ответа.',
        '', '### Проверки', '', '9 тестов CSV/путей/меток/метрик/границ прошли. Реальные модели загружены из сохранённых артефактов, все9 вариантов дали22/22 результатов. Независимый audit проверил SHA прежних файлов, настройки, final labels, score→решение и SDK noul→yes→spam. Личные исходники и старые модели/отчёты неизменны.']
    marked_append(out/'PERSONAL_EMAIL_EVALUATION.md','\n'.join(notes))
    diag=['## Адаптированные Laya и исходная калибровка', '',
        'Все варианты используют точный сохранённый короткий обучающий вопрос B. Сравнение scores ниже выполняется на неизменных inference-результатах, без новой калибровки.',
        '', '| ID | Метка | base calibrated | day5 | extra A | extra B |','|---|---:|---:|---:|---:|']
    for rid in members:
        ss=[]
        for name in ['Laya_base_B_calibrated','Laya_day5','Laya_extra_A','Laya_extra_B']:
            r=modelrows[name][rid];ss.append('N/A' if r['score'] is None else f"{r['score']:.6f} ({r['decision']})")
        diag.append(f"| {rid} | {members[rid]['spam_label']} | "+' | '.join(ss)+' |')
    diag+=['', 'В этом наборе day5 и extra A дают те же бинарные решения, что исходные веса с сохранённой калибровкой дня5. Это не доказывает отсутствия обучения: SHA весов различны, scores меняются, численные различия ниже/в qualitative_analysis.json. Дополнительный B добавил FP относительно калиброванной исходной Laya на mail008 и mail015.']
    for name,v in scores.items():
        diag.append(f"- {name}: mean |Δscore|={v['mean_abs_score_change']:.6f}, max={v['max_abs_score_change']:.6f}; изменённые решения: {', '.join(v['decisions_changed_from_calibrated_base']) or 'нет'}.")
    changes=result['original_question_changed_ids']
    diag+=['',f"A→B меняет {len(changes)} решения: {', '.join(changes)}. A помечает все22 письма spam; B исправляет два ham (mail002,mail009), но пропускает spam mail013. Это диагностический результат, не выбор нового вопроса.",
        'У исходной B11FP/1FN. Сохранённые ранее T5 и threshold0.771568222982013 уменьшают FP до7 при том же1FN. Новая калибровка на личных письмах не выполнялась; улучшение нельзя приписать дообучению, потому что это сравнение одинаковых исходных весов с разными ранее сохранёнными operating settings.',
        'Наблюдаемая высокая чувствительность к вопросу и FP на подтверждённых обычных письмах ограничивают пригодность исходной Laya для самостоятельного антиспама. Ни SDK confidence, ни имя бренда, ни высокая P(yes) не являются подтверждением phishing.']
    marked_append(out/'LAYA_PERSONAL_DIAGNOSTICS.md','\n'.join(diag))
    recommendation=['# Итоговая рекомендация для демо', '',
        'Сохранить E0 как основной демонстрационный вариант и явно показывать исследовательские альтернативы C_corrected и Laya_day5. E0 уже является рабочим default, переключения не выполнялись. Основание для такого демо — минимум наблюдаемых FP (2/13 ham) при сохранении простого проверенного локального пайплайна; цена —5 пропусков из9 spam. Это ограниченный выбор демонстрационного поведения, не утверждение универсальной победы.',
        '',counts_table(), '',
        'У Laya_day5/base calibrated/extra A выше полнота (8/9), но7/13 обычных писем ошибочно помечены spam. C_corrected занимает другой компромисс:6/9 обнаружений,5FP. E2 на личном наборе не улучшил E0: те же5FN при5FP вместо2, поэтому только прежние хорошие RU auxiliary-метрики не оправдывают смену default. Более длительное/частично размороженное обучение Laya не дало на этих22 письмах улучшения бинарных решений относительно калиброванных исходных весов.',
        'Показывать score, сохранённый порог, ограничения и возможность ручного решения; не обещать автоматическую надёжную сортировку реальной почты по этим результатам. Выбор иного баланса FP/FN — отдельное продуктовое решение, а не разрешение подбирать новый порог по личным письмам.',
        '', '## Что подтверждено и чего нет', '',
        'English: ранее сохранённые historical validation-результаты, не новый репрезентативный современный English-тест. Russian: ранее RU auxiliary и реальные пользовательские письма с кириллицей; auxiliary может содержать чаты/короткие сообщения. Mixed: реальные письма со смешанными скриптами и иностранными шаблонами с русскими вставками показывают конкретные ошибки; формальная script-эвристика не является языковой разметкой. В этом наборе нет независимой полноценной оценки по языкам.',
        'Личный набор уже просмотрен и используется для разработки. Один пользователь,22 письма, возможные связанные рассылки/хосты, одна запись2019 года, несколько усечений и MIME-дефектов не дают оснований объявлять модель универсально лучшей. Категории вспомогательные; phishing ground truth, достоверной аутентификации отправителей и истории согласия у моделей нет.',
        '', '## Обучение и упаковка', '',
        'Новое обучение не выполнялось. Наблюдения дают основания сначала изучить современные шаблоны, видимость коротких вставок и доступный контекст согласия. Само увеличение числа шагов не устранило FP на этих письмах. Следующая адаптация потребует более разнообразных пользовательских train/dev-примеров с отдельной проверкой по новым кампаниям; эти22 записи уже не независимый test.',
        'Упаковка и публикация в задаче не выполнялись. До них остаются заранее выбранное демонстрационное поведение/отказы, ограничения для русского и смешанного текста, приватность журналов и отдельный зафиксированный финальный протокол. Закрытые test/challenge/reserve/тест преподавателя не открывались. В каталоге нет .git: .gitignore настроен на личные данные/reports/models, но состояние tracked-файлов проверить невозможно.',
        'Бинарные метки окончательны; вопросов о mail017/019 и анализа альтернатив нет. Для выполненного сравнения дополнительных решений пользователя не требуется. Смена рабочего default и переход к публикации остаются за пределами этой задачи.',
        '', '## Воспроизведение интерпретации', '', 'После основного run:', '', '```powershell',
        f'.\\.venv\\Scripts\\python.exe -B src/report_personal_findings.py --out reports/personal_email_evaluation/runs/{out.name}',
        '```', '', 'Команда читает только сохранённые manifest/comparison/predictions, не запускает модели и не изменяет настройки. Исходные автоматически сформированные версии новых отчётов сохранены в generated_initial_reports/.']
    (out/'FINAL_MODEL_RECOMMENDATION.md').write_text('\n'.join(recommendation)+'\n',encoding='utf-8')
    presentation=['# Материал для презентации: личные письма', '',
        '22 реальных письма одного пользователя; окончательные бинарные метки:13 ham и9 spam. Проверка пользователем17.10.2026; спорных0. 21 письмо с современной датой заголовка, одно2019 года. Категории вспомогательные, phishing-разметки нет.',
        'До inference зафиксированы SHA разметки/писем/артефактов, точные вопросы, preprocessing, calibration и operating thresholds. Восемь основных вариантов плюс отдельный длинный вопрос Laya; покрытие каждого22/22. Обучение и подбор порогов не выполнялись.',
        '',counts_table(), '',
        'Исходная Laya: длинный вопрос пометил все письма spam; короткий изменил3 решения (2 исправленных ham и1 пропущенный spam). Ранее сохранённая калибровка исходных весов уменьшила FP с11 до7. Day5 и длительное head-only обучение сохранили те же бинарные решения, partial encoder —9FP. Это реальные запуски, не mock или демонстрация API-обёртки.',
        'Рекомендация для демо: оставить E0 default и показывать C/Laya как сравниваемые исследовательские варианты. E0 имеет меньше FP, но пропускает5/9 spam; Laya ловит8/9 при7FP. Различие показывает компромисс, а не универсальную победу.',
        'Главное ограничение: text-only модель не знает фактического согласия на рассылку; пользовательские объяснения не добавлялись во вход. Современная многоязычная почта, связанные шаблоны, усечение и MIME-дефекты остаются источниками ошибок. Нужны дополнительные независимо собранные примеры; текущий development-набор не финальный test.',
        'Приватность: локальные процессы, никаких внешних inference API, переходов по ссылкам или исполнения вложений. E0 и предыдущие модели/отчёты сохранены; закрытые наборы не открывались. Этот файл содержит только агрегаты, без адресов, имён, кодов, персональных ссылок и текстов писем.']
    (out/'PRESENTATION_NOTES_PERSONAL_EMAILS.md').write_text('\n'.join(presentation)+'\n',encoding='utf-8')
    write_json(out/'interpretation_manifest.json',{'created_utc':now(),'no_inference':True,'no_label_or_setting_changes':True,
        'writer_sha256':sha_file(Path(__file__)),'inputs':{n:sha_file(out/n) for n in ['manifest.json','comparison.json','predictions.jsonl','campaign_evidence.json']},
        'outputs':{n:sha_file(out/n) for n in names},'boundary':boundary.report()})
    print('Interpretation complete, no inference or setting changes',flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    out=a.out.resolve();require(out.is_relative_to(BASE/'runs'),'Outside personal runs');render(out)
