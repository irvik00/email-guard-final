"""Independent arithmetic and publication-text audit of saved personal predictions."""
import argparse
from personal_eval_common import *

def independent_metrics(rows,labels):
    ok=[r for r in rows if r['status']=='ok']
    counts={k:0 for k in ['TP','FP','TN','FN']}
    for r in ok:
        y=labels[r['record_id']];decision=r['decision']
        counts[('TP' if decision==1 else 'FN') if y==1 else ('FP' if decision==1 else 'TN')]+=1
    a,b,c,d=[counts[k] for k in ['TP','FP','TN','FN']]
    counts.update(precision=a/(a+b) if a+b else None,recall=a/(a+d) if a+d else None,
        F1=2*a/(2*a+b+d) if 2*a+b+d else None,FPR=b/(b+c) if b+c else None,support=len(ok))
    return counts

def main(out):
    boundary=PersonalBoundary(out).install();m=read_json(out/'manifest.json');summary=read_json(out/'comparison.json')
    predictions=read_rows(out/'predictions.jsonl');inputs=read_rows(out/'private_inputs.jsonl')
    labels={r['record_id']:r['spam_label'] for r in m['records']};checked=0
    for candidate in m['candidates']:
        rows=[r for r in predictions if r['model_id']==candidate['name']]
        for part,expected in summary['comparisons'][candidate['name']].items():
            ids=expected['ids'] if part=='truncated' else m['subsets'][part]
            selected=[r for r in rows if r['record_id'] in set(ids)]
            actual=independent_metrics(selected,labels)
            require(all(actual[k]==expected['metrics'][k] for k in actual),'Independent confusion/metric mismatch')
            checked+=1
    source=ROOT/m['annotation_original']['path'];boundary.allowed.add(source)
    rawrows=final_annotations(source)
    with (out/'PERSONAL_EMAIL_ANNOTATION.normalized.csv').open(encoding='utf-8-sig',newline='') as f:normalized=list(csv.DictReader(f))
    require(len(rawrows)==len(normalized)==len(labels),'Final user CSV count differs')
    for raw,norm in zip(rawrows,normalized):
        rid=Path(raw['local_path']).stem
        require(int(raw['spam_label'])==labels[rid]==int(norm['spam_label']),'Final user label mismatch')
        for key in ['category','label_reason','approximate_date','date_scope','label_source','label_reviewed_at']:
            require(raw[key]==norm[key],'Non-path user annotation mutated')
    texts='\n'.join(r['original_model_text'] for r in inputs)
    email_tokens=set(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',texts))
    url_tokens=set(re.findall(r'https?://[^\s<>\"]+',texts))
    reports=['PERSONAL_EMAIL_EVALUATION.md','LAYA_PERSONAL_DIAGNOSTICS.md','FINAL_MODEL_RECOMMENDATION.md','PRESENTATION_NOTES_PERSONAL_EMAILS.md']
    leak_counts={}
    for name in reports:
        text=(out/name).read_text(encoding='utf-8')
        hits={t for t in email_tokens|url_tokens if len(t)>8 and t in text}
        leak_counts[name]=len(hits);require(not hits,'Source email/URL appears in shareable report')
    interp=read_json(out/'interpretation_manifest.json')
    for name,h in interp['outputs'].items():require(sha_file(out/name)==h,'Interpreted report changed')
    result={'status':'passed','independent_metric_groups_checked':checked,'original_user_labels_and_auxiliary_fields_preserved':True,
        'source_email_or_url_hits_in_reports':leak_counts,'privacy_check_limit':'Lexical addresses/URLs from model text only; no claim of complete anonymization beyond aggregate-only report construction',
        'records':len(labels),'candidate_predictions':len(predictions),'interpretation_hashes_match':True,'accesses':boundary.report()}
    write_json(out/'output_audit.json',result);print('Output audit passed:',checked,'metric groups;',len(predictions),'predictions; no source email/URL in reports',flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    out=a.out.resolve();require(out.is_relative_to(BASE/'runs'),'Outside personal runs');main(out)
