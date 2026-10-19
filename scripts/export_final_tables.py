"""Export only public aggregate markdown tables; never open saved email inputs/predictions."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/final_evaluation/runs/20261017T132144Z-d663811e'


def main():
    out=ROOT/'reports/release';out.mkdir(exist_ok=True)
    rows=[];reviews=[];provenance={}
    for filename in ('FINAL_TEST_RESULTS.md','ENSEMBLE_VALIDATION.md'):
        path=RUN/filename;raw=path.read_bytes();text=raw.decode('utf-8')
        provenance[filename]=hashlib.sha256(raw).hexdigest()
        dataset=None;reading=False
        for line in text.splitlines():
            if line.startswith('## '):
                title=line[3:];dataset=title.split(':')[0].strip();reading=False
                if dataset not in ('historical_test','teacher_test','russian_auxiliary_test','spam_email_validation','russian_auxiliary_validation','personal_development'):dataset=None
            elif line.startswith('### teacher_test:'):
                dataset='teacher_nonoverlap_train_validation';reading=False
            if not dataset:continue
            if line.startswith('| Вариант | Решений / n |'):reading=True;continue
            if reading and line.startswith('|---'):continue
            if reading and line.startswith('| '):
                v=[s.strip() for s in line.strip('|').split('|')]
                if len(v)==10:
                    successful,n=map(int,v[1].split('/'))
                    rows.append({'dataset':dataset,'model':v[0],'n':n,'successful':successful,'coverage':successful/n,
                                 **dict(zip(('TP','FP','TN','FN'),map(int,v[2:6]))),
                                 **dict(zip(('precision','recall','F1','FPR'),map(float,v[6:11])))})
                    continue
            if reading and not line.startswith('|'):reading=False
            if line.startswith('E0_Laya_review: автоматически '):
                m=re.search(r'автоматически (\d+)/(\d+).*needs_review=(\d+), истинных ham=(\d+), spam=(\d+).*TP/FP/TN/FN=(\d+)/(\d+)/(\d+)/(\d+), ошибок=(\d+)',line)
                if m:
                    v=list(map(int,m.groups()))
                    reviews.append(dict(zip(('automatic','n','needs_review','review_ham','review_spam','TP','FP','TN','FN','automatic_errors'),v))|{'dataset':dataset,'automatic_fraction':v[0]/v[1],'full_set_F1':None})
    assert len(rows)==42 and len(reviews)==7,(len(rows),len(reviews))
    result={'source_run':'20261017T132144Z-d663811e','source_report_sha256':provenance,'binary':rows,'review':reviews,
        'semantics':'Binary metrics conditional on successful coverage. Review confusion selective only, no full-set F1.',
        'overlap':{'teacher_full':1600,'teacher_train_validation_overlap':988,'teacher_nonoverlap':612,
                   'teacher_overlap_train':828,'teacher_overlap_validation':160,'teacher_cross_historical_test_records':162},
        'recomputed_inference':False,'contains_email_text_or_personal_identifiers':False}
    (out/'final_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'final_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    parts=['# Итоговые агрегаты сохранённой оценки','',result['semantics'],'','Чтение только публичных финальных Markdown-отчётов. Новый inference не выполнялся.','']
    for dataset in dict.fromkeys(r['dataset'] for r in rows):
        parts += ['## '+dataset,'','| Вариант | Покрытие | TP/FP/TN/FN | Precision | Recall | F1 | FPR |','|---|---:|---|---:|---:|---:|---:|']
        for r in rows:
            if r['dataset']==dataset:
                parts.append(f"| {r['model']} | {r['successful']}/{r['n']} | {r['TP']}/{r['FP']}/{r['TN']}/{r['FN']} | {r['precision']:.6f} | {r['recall']:.6f} | {r['F1']:.6f} | {r['FPR']:.6f} |")
        review=next(r for r in reviews if r['dataset']==dataset)
        parts+=['',f"Review: автоматически {review['automatic']}/{review['n']}; проверка {review['needs_review']} (ham {review['review_ham']}, spam {review['review_spam']}); среди auto TP/FP/TN/FN={review['TP']}/{review['FP']}/{review['TN']}/{review['FN']}. F1 всего набора не определена.",'']
    (out/'FINAL_TABLES.md').write_text('\n'.join(parts)+'\n',encoding='utf-8')
    print(json.dumps({'binary_aggregate_rows':len(rows),'review_aggregate_rows':len(reviews),'no_record_inputs_read':True}))


if __name__=='__main__':main()
