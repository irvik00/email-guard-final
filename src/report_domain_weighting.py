"""Reports/tables only; no fitting or raw email reads."""
import argparse
import csv
from domain_weighting_common import *

NAMES={'pilot_historical':'Historical pilot128','pilot_ru':'RU auxiliary pilot128','personal_development':'Личный development22',
       'full_historical_validation':'Полный historical validation1275','full_ru_validation':'Полный RU auxiliary validation5990'}
def f(v):return 'N/A' if v is None else f'{v:.6f}'


def main(out):
    boundary=WeightBoundary(out).install();p=protocol(out);data=read_json(out/'metrics_and_paired_changes.json');stats=read_json(out/'weight_statistics.json');models=read_json(out/'models_frozen.json')['models']
    lines=['# Контролируемые веса доменов: фактические результаты\n',f"Run `{out.name}`, sealed protocol SHA `{sha_file(out/'DOMAIN_WEIGHTING_PROTOCOL.json')}`.\n",
        'Обучены ровно две новые LR:W1 иW2. W0 — прежний C_larger_fit, без повторного fit. '
        'Все три используют одни16032 ID в одинаковом порядке и одну проверенную матрицу embeddings. '
        'Encoder/tokenizer/revision/query-prefix/max512/mean/L2/preprocessing/LR parameters неизменны. '
        'Только sample_weight меняет вклад доменов. Calibration999 и dev600 прежние; для каждой новой LR собственные T и operating threshold, без outer/personal настройки.\n',
        '## Фактические количества и веса\n','Метки0=ham,1=spam. W0 показан с математически эквивалентными единичными весами; sample_weight при его первоначальном fit не передавался. '
        'Class_weight=None у всех, дополнительного class rebalancing нет.\n',
        '| Вариант | Группировка | Группа | Строк | Сумма весов | Средний вес |','|---|---|---|---:|---:|---:|']
    weight_csv=[]
    for name,table in stats.items():
        for kind,values in [('total',{'all':table['total']}),('domain',table['domain']),('class',table['class']),('domain_class',table['domain_class'])]:
            for group,v in values.items():
                lines.append(f"| {name} | {kind} | {group} | {v['rows']} | {v['sum']:.9f} | {v['mean']:.12f} |")
                weight_csv.append({'variant':name,'grouping':kind,'group':group,**v})
    lines+=['\nFloat64 заданные веса нормированы до N_total16032, среднего1; вес=N_total×target_share/N_domain. '
        'Внутри установленного sklearn1.9.1 при float32 features выполняется штатное преобразование весов кfloat32. '
        'Его округление измерено ниже; оно не корректировалось дополнительным множителем или сменой dtype embeddings. '
        'C и общий целевой масштаб весов сохранены. Это вычислительная погрешность, а не новая доля/регуляризация.\n',
        '| Вариант | Native dtype | Native sum | Высокоточная сумма native values | Относительная ошибка native sum |','|---|---|---:|---:|---:|']
    for name in ['W1','W2']:
        for t in models[name]['native_weight_validation']:lines.append(f"| {name} | {t['dtype']} | {t['native_sum']:.9f} | {t['high_precision_sum']:.9f} | {t['native_sum']/16032-1:.12g} |")
    lines+=['\n## Зафиксированные operating settings\n','| Вариант | T | Threshold | Эквивалентный raw-logit cutoff | Fit iterations | Reload max Δlogit |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in models.items():
        cutoff=m['temperature']*math.log(m['threshold']/(1-m['threshold']))
        lines.append(f"| {name} | {m['temperature']:.12g} | {m['threshold']:.16g} | {cutoff:.12g} | {m['fit_iterations']} | {m['reload']['max_logit_error']} |")
    lines.append('\nCalibration — прежний unweighted binary NLL на999, T[.5,5],100 golden-section итераций поlogT с endpoints. '
        'Threshold — точный прежний max среднего F1 двух доменов на dev600, historicalFPR≤1%, recall>0 обоих; tie: меньше historicalFP, затем больший threshold; fallback.5. '
        'Raw-режим во всех моделях имеет общий cutoff0, температура его границу не меняет.\n')
    metric_csv=[];paired_csv=[];confusion={}
    for key in SETS:
        lines+=['\n## '+NAMES[key]+'\n','Это отдельный набор; pilot входит в соответствующий полный validation. Строки этих таблиц нельзя суммировать в общую оценку.\n']
        for mode,title in [('raw','A — сырые LR-решения, logit≥0'),('operating','B — собственные T и operating threshold')]:
            lines+=['\n### '+title+'\n','| Вариант | Успешно/всего | TP | FP | TN | FN | Precision | Recall | F1 | FPR | Отказы |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
            for name in ['W0','W1','W2']:
                s=data['metrics'][key][name][mode];m=s['metrics']
                lines.append('| '+name+' | '+f"{s['successful']}/{s['records']}"+' | '+' | '.join(str(m[k]) for k in ['TP','FP','TN','FN'])+' | '+
                    ' | '.join(f(m[k]) for k in ['precision','recall','F1','FPR'])+' | '+(json.dumps(s['refusals'],ensure_ascii=False) if s['refusals'] else '0')+' |')
                metric_csv.append({'evaluation_set':key,'variant':name,'mode':mode,'records':s['records'],'successful':s['successful'],'coverage':s['coverage'],
                    **{k:m[k] for k in ['TP','FP','TN','FN','precision','recall','F1','FPR']},'refusals':json.dumps(s['refusals'])})
                confusion[key+'/'+name+'/'+mode]={k:m[k] for k in ['TP','FP','TN','FN']}|{'records':s['records'],'successful':s['successful']}
            lines+=['\n| Относительно W0 | Общих успешных ID | Исправлено FP | Добавлено FP | Исправлено FN | Добавлено FN |','|---|---:|---:|---:|---:|---:|']
            for name in ['W1','W2']:
                s=data['paired'][key][name][mode];lines.append('| '+name+' | '+str(s['common_successful'])+' | '+' | '.join(str(len(s[k])) for k in ['rescued_FP','added_FP','rescued_FN','added_FN'])+' |')
                paired_csv.append({'evaluation_set':key,'variant':name,'mode':mode,'common_successful':s['common_successful'],'noncommon':s['noncommon'],
                    **{k:len(s[k]) for k in ['rescued_FP','added_FP','rescued_FN','added_FN']},'ID_details':json.dumps({k:s[k] for k in ['rescued_FP','added_FP','rescued_FN','added_FN']},ensure_ascii=False)})
        if key=='personal_development':
            lines+=['\nЛичные изменения operating относительно W0 — только ID; это исходы относительно окончательной пользовательской разметки.\n']
            for name in ['W1','W2']:
                s=data['paired'][key][name]['operating'];lines.append(name+': '+ '; '.join(k+'='+(', '.join(s[k]) or 'нет') for k in ['rescued_FP','added_FP','rescued_FN','added_FN'])+'.\n')
    lines+=['\n## Дополнительный ориентир C_corrected fit6000\n','Только сохранённые совместимые operating predictions, без новых запусков encoder. '
        'Другие объём и состав fit, поэтому это не причинное сравнение весов. Основная интерпретация — W0/W1/W2 на одинаковом fit16032.\n',
        '| Набор | C6000 TP/FP/TN/FN | C6000 F1 | W0 F1 | W1 F1 | W2 F1 |','|---|---|---:|---:|---:|---:|']
    for key in SETS:
        ref=data['reference_C6000_operating'][key]['metrics'];lines.append('| '+NAMES[key]+' | '+'/'.join(str(ref[k]) for k in ['TP','FP','TN','FN'])+' | '+f(ref['F1'])+' | '+
            ' | '.join(f(data['metrics'][key][n]['operating']['metrics']['F1']) for n in ['W0','W1','W2'])+' |')
    lines+=['\n## Сопоставление raw и operating\n','| Набор | Вариант | Raw F1 | Operating F1 | Raw FP/FN | Operating FP/FN |','|---|---|---:|---:|---:|---:|']
    for key in ['full_historical_validation','full_ru_validation','personal_development']:
        for name in ['W0','W1','W2']:
            a=data['metrics'][key][name]['raw']['metrics'];b=data['metrics'][key][name]['operating']['metrics']
            lines.append(f"| {NAMES[key]} | {name} | {f(a['F1'])} | {f(b['F1'])} | {a['FP']}/{a['FN']} | {b['FP']}/{b['FN']} |")
    if (out/'interpretation.md').exists():lines.append('\n'+(out/'interpretation.md').read_text(encoding='utf8'))
    lines+=['\n## Проверки, границы и ограничения\n',
        f"Fit IDs/order SHA `{p['fit_ids_order_sha256']}`. Все ID/global groups fit↔dev/cal/validation раздельны; selected inner capped совпадений0. "
        f"Full validation capped совпадений с fit: {len(p['overlap_checks']['full_validation_capped_overlap_ids'])}. При наличии они сохраняются в полном результате и списке protocol, не удаляются. "
        'Кеши проверены по SHA/revision/tokenizer/input order/contract; повторного26k tokenization-аудита и чтения raw train не было. '
        'Два ранее известных baseline-fit совпадения с невыбранным calibration пулом остаются описаны в предыдущем аудите; активные999/600 не затронуты.\n',
        'Все logits конечны, LR convergence проверена, joblib reload настоящий. Метрики при возможных отказах условны на успешном покрытии, отказы не являются ham. '
        'Encoder пересчитан только для22 уже просмотренных personal, один общий cache для W0/W1/W2; Qwen/Laya не запускались. '
        'Полные validation повторно использовались для исследований, личные — development, RU source labels provisional. '
        'Выбор после этих наблюдений требует новых данных; прежние финальные test-оценки принадлежат прежним моделям. '
        'Согласие/история/подтверждённый phishing не возникают из текста или метки spam. Default и упаковка не менялись.\n']
    if (out/'independent_audit.json').exists():
        audit=read_json(out/'independent_audit.json');lines.append(f"\nНезависимый аудит: {audit['status']}; {audit['metric_groups']} core metric groups, {audit['paired_groups']} paired groups, reload новых моделей и независимая calibration/threshold проверены. "
            f"Сохранность {audit['targeted_unchanged_files']} контрольных входов/default/config подтверждена SHA. Полный прежний аудит не повторялся; raw/closed reads0.\n")
    lines+=['\n## Команды\n','Из корня проекта, новый датированный out создаётся автоматически:\n',
        '```powershell\n.\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -p test_domain_weighting.py\n.\\.venv\\Scripts\\python.exe -B src/run_domain_weighting.py run\n```\n',
        'Проверка существующего run без обучения:\n',
        '```powershell\n'+f"$out = 'reports/domain_weighting/runs/{out.name}'\n"+
        '.\\.venv\\Scripts\\python.exe -B src/audit_domain_weighting.py --out $out\n.\\.venv\\Scripts\\python.exe -B src/report_domain_weighting.py --out $out\n```\n',
        'Модели W1/W2 и manifests находятся в models/ данного run; W0 остаётся ссылкой на прежний файл. '
        'Исходные predictions/cal/dev logits, веса и ID-изменения — локальные производные, не для публикации личной почты.\n']
    (out/'DOMAIN_WEIGHTING_RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    for filename,rows in [('metrics.csv',metric_csv),('paired_changes.csv',paired_csv),('weight_statistics.csv',weight_csv)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'confusion_matrices.json',confusion)
    notes=['# Материал для презентации: веса доменов\n','## Контроль\n\nОдинаковые fit16032, embeddings, encoder и LR-гиперпараметры. W0 сохранён, обучены только W1/W2. '
        'W0 фактическая доля historical12.67%,RU87.33%; W1 целевой вклад1/3:2/3, W2 1/2:1/2. Общий целевой вес16032, среднее1.\n',
        '## Результаты на полном validation\n\n| Режим | Набор | W0 F1 | W1 F1 | W2 F1 |\n|---|---|---:|---:|---:|']
    for mode in ['raw','operating']:
        for key in ['full_historical_validation','full_ru_validation']:
            notes.append('| '+mode+' | '+NAMES[key]+' | '+' | '.join(f(data['metrics'][key][n][mode]['metrics']['F1']) for n in ['W0','W1','W2'])+' |')
    notes.append('\n## Личные письма и ограничения\n\n'+ '; '.join(n+': FP/FN='+str(data['metrics']['personal_development'][n]['operating']['metrics']['FP'])+'/'+str(data['metrics']['personal_development'][n]['operating']['metrics']['FN']) for n in ['W0','W1','W2'])+
        '. Это22 development-письма, не независимый тест. Сырые границы сравниваются отдельно от operating settings. '
        'Нет нового default, нет упаковки, нет повторной финальной test-оценки. Кандидат по этим данным требует новых писем.\n')
    if (out/'interpretation.md').exists():notes.append((out/'interpretation.md').read_text(encoding='utf8'))
    (out/'PRESENTATION_NOTES_DOMAIN_WEIGHTING.md').write_text('\n'.join(notes),encoding='utf8')
    write_json(out/'report_accesses.json',boundary.report());print('Reports/CSV/confusion matrices saved',out,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);args=a.parse_args();main(args.out.resolve())
