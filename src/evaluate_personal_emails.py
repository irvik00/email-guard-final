"""Freeze final user labels, evaluate saved candidates locally, audit and report."""
import argparse
import importlib.metadata
import subprocess
import shutil
from personal_eval_common import *

DAY5='20261012T083529Z-0950e3dc34'
EXTRA='20261012T095633Z-7d01f7b8'
CORRECTED_C='20261012T141331Z-43c7c5dc'

def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()

def candidate_registry():
    import joblib
    from predict_email import load_bundle
    from baseline_common import versions
    candidates=[]
    path=ROOT/'models/day3/baseline_bundle.joblib';b=load_bundle(path)
    candidates.append({'name':'E0','kind':'sklearn','is_main':True,'bundle_path':rel(path),
        'files':{rel(path):sha_file(path),rel(path.parent/'model_manifest.json'):sha_file(path.parent/'model_manifest.json')},
        'threshold':b['thresholds']['provisional_threshold'],'temperature':1.,'builder':'day3_original',
        'score_semantics':'Saved char-LR predict_proba class1, no new calibration, not a personal spam guarantee',
        'versions':b['versions'],'threshold_source':'Day3 saved provisional operating threshold'})
    path=ROOT/'models/day4/runs/20261011T140555Z-60a6fd4a-6407bea2/E2_bundle.joblib'
    m=read_json(path.parent/'model_manifest.json');require(sha_file(path)==m['artifacts'][path.name]['sha256'],'E2 hash differs')
    b=joblib.load(path);require(b['versions']==versions(),'E2 versions differ')
    require(b['builder_name']=='clean_builder_v2','E2 builder differs')
    for n,h in b['code_sha256'].items():require(sha_file(ROOT/n)==h,'E2 code differs')
    candidates.append({'name':'E2','kind':'sklearn','is_main':True,'bundle_path':rel(path),
        'files':{rel(path):sha_file(path),rel(path.parent/'model_manifest.json'):sha_file(path.parent/'model_manifest.json')},
        'threshold':b['threshold'],'temperature':1.,'builder':b['builder_name'],'versions':b['versions'],
        'score_semantics':'Saved char-LR predict_proba class1, no new calibration, not a personal spam guarantee',
        'threshold_source':'Day4 E2 saved operating threshold'})
    runtime=read_json(ROOT/'reports/day4/setup/runtime_manifest.json');day5=read_json(ROOT/f'models/day5/runs/{DAY5}/candidate_manifest.json')
    long=read_json(ROOT/'config/day4_experiments.json')['pilot']['questions'];short=day5['config']['questions']
    checkpoint=Path(runtime['local_path'])
    original_files={rel(ROOT/'.cache/day4/laya_weights'/p):v['sha256'] for p,v in runtime['files'].items() if p.startswith('multilingual/')}
    # Original short-B SDK diagnostic used source T=1 and a pre-existing fixed half threshold.
    cfg=read_json(checkpoint/'rl_agent_config.json');require(cfg['temperature'][2]==1. and not cfg['temperature_by_options'],'Original calibration differs')
    for name,question,main in [('Laya_original_B',short,True),('Laya_question_A',long,False)]:
        candidates.append({'name':name,'kind':'laya','is_main':main,'checkpoint_path':rel(checkpoint),
            'files':original_files,'threshold':.5,'temperature':1.,'questions':question,
            'input_contract':day5['config']['input'],'builder':'clean_builder_v2','versions':runtime['versions'],
            'score_semantics':'Unrounded softmax true-option probability from original Laya noul logits / stored T=1; SDK noul is rounded4',
            'threshold_source':'Existing original synthetic diagnostics fixed0.5, not newly selected on personal emails',
            'question_source':'config/day4_experiments.json pilot' if not main else f'models/day5/runs/{DAY5}/candidate_manifest.json config.questions'})
    manifests=[('Laya_base_B_calibrated',ROOT/f'models/day5/runs/{DAY5}/candidate_manifest.json','base'),
        ('Laya_day5',ROOT/f'models/day5/runs/{DAY5}/candidate_manifest.json','adapted'),
        ('Laya_extra_A',ROOT/f'models/extra_experiments/{EXTRA}/A/candidate_manifest.json','adapted'),
        ('Laya_extra_B',ROOT/f'models/extra_experiments/{EXTRA}/B/candidate_manifest.json','adapted')]
    for name,path,key in manifests:
        m=read_json(path);c=m['candidates'][key];ckpt=ROOT/c['checkpoint_path']
        files={rel(p):sha_file(p) for p in ckpt.rglob('*') if p.is_file()}
        require(files[rel(ckpt/'model.safetensors')]==c['artifacts']['weights_sha256'],'Laya weights differ')
        require(files[rel(ckpt/'rl_agent_config.json')]==c['artifacts']['config_sha256'],'Laya config differs')
        files[rel(path)]=sha_file(path)
        candidates.append({'name':name,'kind':'laya','is_main':True,'checkpoint_path':rel(ckpt),'files':files,
            'threshold':c['threshold'],'temperature':c['temperature'],'questions':c['questions'],
            'input_contract':c['input_contract'],'builder':c['input_contract']['builder'],'versions':m['versions'],
            'score_semantics':'Unrounded softmax true-option probability from Laya noul logits / saved calibration T; SDK noul is rounded4',
            'threshold_source':rel(path)+' candidates.'+key,'selected_training_step':c['selected_training_step']})
    path=ROOT/f'models/c_diagnostics/{CORRECTED_C}/candidate_manifest.json';m=read_json(path)
    require(m['experiment']=='C_calibration_disjoint_diagnostic' and m['calibration_rows']==999,'Corrected C required, no fallback')
    require(not m['encoder_or_classifier_refitted'],'C correction must preserve weights')
    files={rel(path):sha_file(path),m['classifier_path']:m['classifier_sha256']}
    files.update({rel(Path(m['source']['local_path'])/n):s['sha256'] for n,s in m['source']['files'].items()})
    files.update({rel(p):sha_file(p) for p in (ROOT/m['tokenizer_path']).rglob('*') if p.is_file()})
    v=dict(m['versions']);v.update(read_json(ROOT/f'reports/extra_experiments/runs/{EXTRA}/C/embeddings.json')['versions'])
    candidates.append({'name':'C_corrected','kind':'e5','is_main':True,'files':files,'threshold':m['threshold'],
        'temperature':m['temperature'],'builder':'clean_builder_v2','saved_manifest':m,'versions':v,
        'score_semantics':'sigmoid(E5-frozen normalized embedding LR decision_function / saved T), corrected calibration999',
        'threshold_source':rel(path),'input_contract':{'prefix':'query: ','max_tokens':512,'pooling':m['pooling']}})
    return candidates

def prepare(out,annotation):
    from read_email import parse_email
    from normalization_schema import make_model_text
    rows=final_annotations(annotation);entries=[resolve_personal_path(r) for r in rows]
    boundary=PersonalBoundary(out,[annotation]+[p for p,_ in entries]).install()
    cfg=read_json(ROOT/'config/day4_experiments.json');records=[];normalized=[];all_parsed=[];members=[]
    for row,(path,moved) in zip(rows,entries):
        sha=sha_file(path);raw=path.read_bytes();require(len(raw)<=20*1024*1024,'Personal email exceeds input size limit')
        try:parsed=parse_email(raw);error=None
        except Exception as e:parsed={'headers_raw':[]};error=type(e).__name__
        dates=date_metadata(parsed,row);original,changes=make_model_text(parsed.get('subject'),parsed.get('body_text'))
        clean,clean_changes=clean_model_text(parsed,cfg);script=script_metadata(clean)
        normalized.append(normalize_annotation(row,path,sha,dates));rid=path.stem
        record={'record_id':rid,'domain':'personal','source':'personal_user','duplicate_group_id':'personal:'+digest(compare_text(clean)),
            'spam_label':int(row['spam_label']),'label_status':'human_confirmed_final','label_source':row['label_source'],
            'label_reviewed_at':row['label_reviewed_at'],'text':clean,'original_model_text':original,
            'original_locator':rel(path),'cleaning_changes':clean_changes,'parse_error':error}
        records.append(record);all_parsed.append(parsed)
        members.append({'record_id':rid,'local_path':rel(path),'original_csv_path':row['local_path'],'path_corrected':moved,
            'raw_sha256':sha,'size_bytes':len(raw),'spam_label':int(row['spam_label']),'human_spam_label':int(row['spam_label']),
            'label_status':'human_confirmed_final','category':row['category'],'category_status':'auxiliary_only',
            'phishing_label':'unknown','label_source':row['label_source'],'label_reviewed_at':row['label_reviewed_at'],
            'dates':dates,'script':script,'parse_error':error,'parse_warnings':parsed.get('parse_warnings',[]),
            'body_source':parsed.get('body_source'),'subject_present':parsed.get('subject') is not None,
            'body_present':parsed.get('body_text') is not None,'attachment_count':len(parsed.get('attachments',[])),
            'original_text_characters':len(original),'clean_text_characters':len(clean),
            'original_text_sha256':digest(original),'clean_text_sha256':digest(clean),'cleaning_changes':clean_changes})
    normalized_path=out/'PERSONAL_EMAIL_ANNOTATION.normalized.csv';annotation_csv(normalized_path,normalized)
    with normalized_path.open(encoding='utf-8-sig',newline='') as f:
        reread=csv.DictReader(f);require(reread.fieldnames==NORMALIZED_FIELDS and list(reread)==normalized,'CSV roundtrip differs')
    write_rows(out/'private_inputs.jsonl',records)
    campaigns=campaign_metadata(records,all_parsed);write_json(out/'campaign_evidence.json',campaigns)
    main_ids=[r['record_id'] for r in members];modern=[r['record_id'] for r in members if r['dates']['modern']]
    subsets={'confirmed':main_ids,'modern':modern,'disputed':[],
        'old_or_unknown_date':[r['record_id'] for r in members if not r['dates']['modern']],
        'user_ham_advertising':[r['record_id'] for r in members if r['spam_label']==0 and r['category']=='реклама'],
        'ordinary_service_notifications':[r['record_id'] for r in members if r['spam_label']==0 and r['category']=='служебные уведомления'],
        'money_or_prize_claims':[Path(r['local_path']).stem for r in rows if re.search('обещан|перевод|приз|бонус|сюрприз|подароч',r['label_reason'],re.I)],
        'mixed_scripts':[r['record_id'] for r in members if r['script']['group']=='mixed_scripts']}
    for group in ['cyrillic_dominant','latin_dominant_language_unknown','other_or_unknown']:
        subsets[group]=[r['record_id'] for r in members if r['script']['group']==group]
    candidates=candidate_registry()
    for c in candidates:
        for name,h in c['files'].items():require(sha_file(ROOT/name)==h,'Candidate artifact changed before freeze')
    ignored=(ROOT/'.gitignore').read_text(encoding='utf-8')
    privacy={'ignore_personal':'/data/personal_review/' in ignored,'ignore_reports':'/reports/' in ignored,
        'ignore_models':'/models/' in ignored,'git_repository_present':(ROOT/'.git').exists()}
    require(all(privacy[k] for k in ['ignore_personal','ignore_reports','ignore_models']),'Private outputs must be gitignored')
    if privacy['git_repository_present']:
        proc=subprocess.run(['git','-C',str(ROOT),'status','--short'],capture_output=True,text=True,encoding='utf-8')
        privacy['git_status_exit_code']=proc.returncode;(out/'git_status.private.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    previous=[rel(p) for p in BASE.rglob('predictions.jsonl') if not p.is_relative_to(out)]
    manifest={'run_id':out.name,'frozen_utc':now(),'client_review_date':'2026-10-17','phase':'personal_review_dev_exploratory',
        'annotation_original':{'path':rel(annotation),'sha256':sha_file(annotation),'columns':SOURCE_FIELDS,'rows':len(rows)},
        'annotation_normalized':{'path':rel(normalized_path),'sha256':sha_file(normalized_path),'columns':NORMALIZED_FIELDS,'rows':len(rows),
            'policy':'Correct only stale file paths; spam_label copied literally; 24-column explicit provenance/unknown schema; UTF8 BOM DictWriter'},
        'user_instruction':'Final spam_label, label_source and label_reviewed_at authoritative; no disputed binary labels, no mail017/019 alternatives',
        'overridden_attachment_context':['Old24-column malformed-rows claim does not apply to current8-column CSV',
            'Old dispute/exclusion/sensitivity instructions superseded by direct final user labels'],
        'records':members,'subsets':subsets,'candidates':candidates,'private_inputs_sha256':sha_file(out/'private_inputs.jsonl'),
        'parser_sha256':sha_file(ROOT/'src/read_email.py'),'builder_sha256':sha_file(ROOT/'src/normalization_schema.py'),
        'cleaner_config_sha256':sha_file(ROOT/'config/day4_experiments.json'),
        'rules':{'modern':'Header Date calendar date2025-01-01 through2026-10-17; untrusted headers only, user approximate date reported separately',
            'labels':'All final confirmed labels, including mail017/019; no sensitivity analysis and no relabeling',
            'metrics':'Counts first; conditional successful scores+coverage; undefined ratios N/A; all22 retained',
            'questions':'Exact saved A and B only; original weights, T1, fixed0.5 for diagnostic; adapted questions unchanged',
            'models':'Saved manifest builder/T/threshold; no fit, no recalibration or threshold search; E0 unchanged',
            'privacy':'Offline network-denying audit hook, static MIME only; no sender/links/attachments appended to model text',
            'refusal':'Empty text or sparse vector reported insufficient_features, decision missing; raw score may remain diagnostic',
            'campaigns':'No deduplication; weak hints, individual messages are not independent attack families'},
        'prior_personal_model_prediction_files':previous,'already_content_reviewed':True,'independent_final_test':False,
        'privacy':privacy,'new_training':False,'closed_test_opened':False,'jev_integration':'No separate integration found; direct_laya backend only',
        'code_sha256':{n:sha_file(ROOT/n) for n in ['src/personal_eval_common.py','src/evaluate_personal_emails.py','src/personal_neural_worker.py']}}
    write_json(out/'manifest.json',manifest);write_json(out/'prepare_accesses.json',boundary.report())
    return manifest,boundary

def sparse_predictions(out,candidate,records):
    import joblib
    from threadpoolctl import threadpool_limits
    from baseline_common import versions
    require(candidate['versions']==versions(),'Sparse stack versions differ')
    for name,h in candidate['files'].items():require(sha_file(ROOT/name)==h,'Sparse candidate hash differs')
    bundle=joblib.load(ROOT/candidate['bundle_path']);pipeline=bundle['pipeline'];require(list(pipeline.classes_)==[0,1],'Sparse class mapping')
    output=[]
    with threadpool_limits(limits=1):
        for r in records:
            text=r['original_model_text'] if candidate['builder']=='day3_original' else r['text']
            try:
                v=pipeline['tfidf'].transform([text]);score=float(pipeline['classifier'].predict_proba(v)[0,list(pipeline.classes_).index(1)])
                insufficient=not text.strip() or v.nnz==0
                output.append(prediction_row(r,candidate,'insufficient_features' if insufficient else 'ok',score=score,
                    decision=None if insufficient else int(score>=candidate['threshold']),insufficient_features=insufficient,
                    active_features=int(v.nnz),model_text_characters=len(text),token_stats={'truncated':False,'token_limit':'none, sparse ngrams'},
                    raw_intercept_decision=int(score>=candidate['threshold']) if insufficient else None))
            except Exception as e:output.append(prediction_row(r,candidate,'error',error_type=type(e).__name__,private_error=str(e)))
    write_rows(out/f"predictions_{candidate['name']}.jsonl",output)
    return output

def audit(out):
    boundary=PersonalBoundary(out).install();m=read_json(out/'manifest.json');inputs=read_rows(out/'private_inputs.jsonl')
    require(sha_file(out/'private_inputs.jsonl')==m['private_inputs_sha256'],'Private input changed')
    with (out/'PERSONAL_EMAIL_ANNOTATION.normalized.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);require(reader.fieldnames==NORMALIZED_FIELDS,'CSV schema differs');csvrows=list(reader)
    require(len(csvrows)==len(m['records']),'Annotation count differs')
    rows=read_rows(out/'predictions.jsonl');checks={};byid={r['record_id']:r for r in inputs}
    for c in m['candidates']:
        pred=[r for r in rows if r['model_id']==c['name']]
        require({r['record_id'] for r in pred}==set(byid) and len(pred)==len(byid),'Missing/duplicated model predictions')
        for r in pred:
            require(r['spam_label']==byid[r['record_id']]['spam_label'],'Final label changed')
            require(r['threshold']==c['threshold'] and r['temperature']==c.get('temperature',1.),'Operating settings changed')
            if r['status']=='ok':
                require(math.isfinite(r['score']) and 0<=r['score']<=1,'Invalid successful score')
                require(type(r['decision']) is int and r['decision']==int(r['score']>=c['threshold']),'Verdict differs')
                if 'logit_difference' in r:require(abs(r['score']-sigmoid(r['logit_difference']/c['temperature']))<1e-12,'Calibration mapping differs')
                if 'raw_response' in r:
                    from decision_protocol import validate_noul
                    require(abs(validate_noul(r['raw_response'])-r['score'])<=.000051,'Actual SDK raw response differs')
            else:require(r['decision'] is None,'Failure converted to ham/spam')
        for name,h in c['files'].items():require(sha_file(ROOT/name)==h,'Model artifact mutated')
        checks[c['name']]={'rows':len(pred),'successful':sum(r['status']=='ok' for r in pred)}
    before=read_json(out/'preservation_before.json');changed=[p for p,h in before.items() if sha_file(ROOT/p)!=h]
    require(not changed,'Prior model/report/code changed')
    # Separate exact source allowlist, never original processed train/validation/test.
    rawboundary=PersonalBoundary(out,[ROOT/m['annotation_original']['path']]+[ROOT/r['local_path'] for r in m['records']]).install()
    boundary.allowed=rawboundary.allowed
    require(sha_file(ROOT/m['annotation_original']['path'])==m['annotation_original']['sha256'],'Original user CSV changed')
    for r,n in zip(m['records'],csvrows):
        require(sha_file(ROOT/r['local_path'])==r['raw_sha256'],'Original personal email changed')
        require(int(n['spam_label'])==r['spam_label'] and n['human_spam_label']==n['spam_label'],'Normalized label differs')
        require(n['phishing_label']=='unknown' and n['review_status']=='reviewed','Axes/status corrupted')
    require(m['subsets']['disputed']==[],'Obsolete dispute returned')
    require(not any('sensitivity' in k for k in m['subsets']),'Unexpected alternative label analysis')
    write_json(out/'independent_audit.json',{'status':'passed','models':checks,'protected_files':len(before),'changed_files':changed,
        'original_personal_files_unchanged':True,'labels_final_confirmed':True,'closed_data_opened':False,
        'manifest_sha256':sha_file(out/'manifest.json'),'accesses':boundary.report()})
    print('Independent audit passed;',len(before),'protected files preserved',flush=True)

def summary(out):
    m=read_json(out/'manifest.json');records=read_rows(out/'private_inputs.jsonl');rows=read_rows(out/'predictions.jsonl')
    comparisons={}
    for c in m['candidates']:
        data=[r for r in rows if r['model_id']==c['name']]
        comparisons[c['name']]={part:model_summary(records,data,ids) for part,ids in m['subsets'].items()}
        trunc=[r['record_id'] for r in data if r.get('token_stats',{}).get('truncated')]
        comparisons[c['name']]['truncated']=model_summary(records,data,trunc)
        comparisons[c['name']]['truncated']['ids']=trunc
    disagreements=[]
    main=[c['name'] for c in m['candidates'] if c['is_main']]
    for record in records:
        r={p['model_id']:p for p in rows if p['record_id']==record['record_id'] and p['model_id'] in main}
        decisions={name:p['decision'] for name,p in r.items()}
        if len(set(decisions.values()))>1:disagreements.append({'record_id':record['record_id'],'label':record['spam_label'],'decisions':decisions})
    a={r['record_id']:r for r in rows if r['model_id']=='Laya_question_A'}
    b={r['record_id']:r for r in rows if r['model_id']=='Laya_original_B'}
    question_changes=[rid for rid in a if a[rid]['status']=='ok' and b[rid]['status']=='ok' and a[rid]['decision']!=b[rid]['decision']]
    result={'comparisons':comparisons,'main_models':main,'disagreements':disagreements,'original_question_changed_ids':question_changes,
        'original_questions_successfully_compared':sum(a[rid]['status']=='ok' and b[rid]['status']=='ok' for rid in a),
        'is_development_diagnostic':True,'no_alternative_labels':True}
    write_json(out/'comparison.json',result);return m,result,rows

def reports(out):
    m,res,rows=summary(out);comp=res['comparisons'];members=m['records'];main=res['main_models']
    fmt=lambda x:'N/A' if x is None else f'{x:.4f}'
    def table(part):
        lines=['| Модель | Покрытие | TP | FP | TN | FN | Precision | Recall | F1 | FPR |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name in main:
            v=comp[name][part];s=v['metrics']
            lines.append(f"| {name} | {v['successful']}/{v['records']} | {s['TP']} | {s['FP']} | {s['TN']} | {s['FN']} | {fmt(s['precision'])} | {fmt(s['recall'])} | {fmt(s['F1'])} | {fmt(s['FPR'])} |")
        return lines
    counts=Counter(r['spam_label'] for r in members)
    intro=[f'# Личные письма: {out.name}', '', 'Локальная исследовательская диагностика разработки, не независимый финальный тест. Содержимое уже просмотрено при ассистентском черновике и пользовательской разметке. Метки берутся из окончательного CSV пользователя; spam=1. Категории вспомогательные, phishing у всех unknown.',
        f"Подтверждено {len(members)} записей: ham={counts[0]}, spam={counts[1]}; спорных0. Современный поднабор по Date={len(m['subsets']['modern'])}; старые/неизвестные={len(m['subsets']['old_or_unknown_date'])}. mail017/019 включены с окончательными метками0, альтернативные метки не анализировались.",
        'Исходный CSV:8 столбцов, все22 строки структурно полны. Отдельный24-column CSV расширяет схему происхождением/unknown и исправляет устаревший /emails/ путь. Исходный файл и .eml сохранены; DictWriter, UTF8 BOM, roundtrip проверен. Проверка label_reviewed_at — ровно дата из CSV, время не выдумывалось.',
        'Согласие отражено только при явном упоминании в финальном label_reason, ожидаемость служебного сообщения отдельно; метка ham сама по себе не создаёт consent=yes. Историю согласия spam-писем не угадываем. Категории и признаки подражания брендам не превращаются в phishing-разметку.',
        '', '## Основная оценка', '']
    intro+=table('confirmed')+['', '## Современные письма', '']+table('modern')
    intro += ['', 'Date и Received — неподтверждённые заголовки, не достоверный collection timestamp. mail016: Date='+str(next(r for r in members if r['record_id']=='mail_016')['dates']['date_header_iso'])+'; Received также2019, поэтому в современную оценку не включено. Во всей подтверждённой оценке оно сохранено.',
        '', '## Ошибки и отказы', '', '| Модель | FP IDs | FN IDs | Нет решения |','|---|---|---|---|']
    for name in main:
        v=comp[name]['confirmed'];intro.append(f"| {name} | {', '.join(v['FP_ids']) or '—'} | {', '.join(v['FN_ids']) or '—'} | {', '.join(v['unscored_ids']) or '—'} |")
    intro += ['', 'Отказ/ошибка/insufficient_features имеют decision=null, не ham. Покрытие и список ID публикуются; метрики рассчитаны только на успешных решениях при неполном покрытии. Неопределённые отношения — N/A, причины в comparison.json. Порог каждого кандидата взят из сохранённого протокола, по личным письмам не выбирался.',
        '', '## Состав, даты и parser', '', '| ID | Подтверждённая метка | Категория (aux) | Date | Скрипты (не язык) | Символов clean | MIME warnings |','|---|---:|---|---|---|---:|---:|']
    for r in members:intro.append(f"| {r['record_id']} | {r['spam_label']} | {r['category']} | {r['dates']['date_header_iso'] or 'unknown'} | {r['script']['group']} | {r['clean_text_characters']} | {len(r['parse_warnings'])} |")
    intro += ['', 'Все перечисленные файлы проверены на наличие, уникальность путей и SHA-256. campaign_evidence.json содержит canonical duplicates, char5gram similarity>=0.55, группы одинакового sender-domain/shared-link-host (сами домены только SHA). Общие tracker/unsubscribe/поставщики — слабые признаки инфраструктуры, не доказанные кампании. Повторяющиеся шаблоны не считаются независимыми видами атак; дубли не удалялись.',
        '', '## Предварительно заданные диагностические группы', '', '| Группа | n | ID |','|---|---:|---|']
    for key in ['user_ham_advertising','ordinary_service_notifications','money_or_prize_claims','mixed_scripts','old_or_unknown_date']:
        intro.append(f"| {key} | {len(m['subsets'][key])} | {', '.join(m['subsets'][key]) or '—'} |")
    intro += ['', 'Группы по категориям/reason вспомогательные и пересекаются. Их метрики и усечённые ID для каждого кандидата сохранены в comparison.json. Подписка/ожидаемость часто не видны text-only модели: label_reason, consent, категория, даты и пользовательские ответы никогда не добавлялись к её входу.',
        '', '## Расхождения', '', '| ID | Метка | Решения моделей (1 spam,0 ham,null отказ) |','|---|---:|---|']
    for d in res['disagreements']:intro.append(f"| {d['record_id']} | {d['label']} | {json.dumps(d['decisions'],ensure_ascii=False)} |")
    intro += ['', '## Настройки и сохранность', '', '| Кандидат | Builder | T | Operating threshold |','|---|---|---:|---:|']
    for c in m['candidates']:intro.append(f"| {c['name']} | {c['builder']} | {c.get('temperature',1.):.12g} | {c['threshold']:.16g} |")
    intro += ['', 'Исходная Laya_original_B использует SDK T1 и ранее фиксированный порог0.5. Laya_base_B_calibrated — отдельный вариант тех же исходных весов с калибровкой дня5 T5 и порогом0.771568222982013. Они не смешиваются. C_corrected — calibration999 из завершённой диагностики C, без подмены исходным C.',
        'Вход: E0 — исходный subject+body builder дня3; E2/Laya/C — clean_builder_v2. Sender, дополнительные ссылки, содержимое вложений и человеческие объяснения не добавлялись. HTML разбирался статически существующим parser, script/style и вложения не исполнялись. Neural candidates запускались последовательно в отдельных offline процессах, eval/no_grad, GPU освобождалась при завершении процесса.',
        'Предыдущие отчёты/модели проверены только по SHA-256; исходные processed splits не загружались. Test/challenge/reserve/тест преподавателя закрыты, обучение/подбор порога/публикация не выполнялись. E0 не заменён.',
        'Личные письма, исходная разметка и отчёты исключены .gitignore. '+('Git status сохранён локально.' if m['privacy']['git_repository_present'] else 'В каталоге нет .git; проверить tracked-файлы невозможно, git init не выполнялся.'),
        '', '## Воспроизведение', '', 'Из корня проекта:', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -p test_personal_evaluation.py',
        '.\\.venv\\Scripts\\python.exe -B src/evaluate_personal_emails.py run',
        f'.\\.venv\\Scripts\\python.exe -B src/evaluate_personal_emails.py audit --out reports/personal_email_evaluation/runs/{out.name}',
        '```', '', 'run создаёт новый датированный архив, не перезаписывает прежние запуски. Используются существующие .venv и .venv-decision, загрузки/установки/переобучение отсутствуют. Private CSV и private_inputs содержат персональные данные и остаются локальными.']
    (out/'PERSONAL_EMAIL_EVALUATION.md').write_text('\n'.join(intro)+'\n',encoding='utf-8')
    bymodel={name:{r['record_id']:r for r in rows if r['model_id']==name} for name in ['Laya_question_A','Laya_original_B']}
    lines=['# Laya: диагностика на личных письмах', '', 'Точные сохранённые вопросы A/B, исходный multilingual checkpoint, одинаковый clean_builder_v2 и cap700/context1024/head256, исходные SDK T1 и ранее фиксированный threshold0.5. Это отдельная диагностика вопросов, без настройки дообученных вариантов.',
        '', '```json',json.dumps({c['name']:c['questions'] for c in m['candidates'] if c['name'] in bymodel},ensure_ascii=False,indent=2),'```',
        '',f"Успешно сопоставлено {res['original_questions_successfully_compared']} пар; изменилось решение у {len(res['original_question_changed_ids'])}: {', '.join(res['original_question_changed_ids']) or 'ни у одного письма'}.",
        'Scores ниже — не confidence, а unrounded P(true=yes=spam) по option logits. Реальные сырые SDK ответы, logits, SHA фактических токенов и проверка mapping сохранены в predictions_*.jsonl. Новые вопросы не подбирались.',
        '', '| ID | Метка | A score / решение | B score / решение | Исходных state tokens | Видимых A/B | Sequence A/B | Усечение A/B |','|---|---:|---|---|---:|---|---|---|']
    for r in members:
        rid=r['record_id'];a=bymodel['Laya_question_A'][rid];b=bymodel['Laya_original_B'][rid]
        ta=a.get('token_stats',{});tb=b.get('token_stats',{})
        lines.append(f"| {rid} | {r['spam_label']} | {fmt(a['score'])} / {a['decision']} | {fmt(b['score'])} / {b['decision']} | {ta.get('original_tokens','N/A')} | {ta.get('visible_tokens','N/A')}/{tb.get('visible_tokens','N/A')} | {a.get('sequence_tokens','N/A')}/{b.get('sequence_tokens','N/A')} | {ta.get('truncated','N/A')}/{tb.get('truncated','N/A')} |")
    lines+=['', 'Для исходных вопросов:', '']
    for name in bymodel:
        v=comp[name]['confirmed'];s=v['metrics'];lines.append(f"- {name}: TP/FP/TN/FN={s['TP']}/{s['FP']}/{s['TN']}/{s['FN']}, coverage={v['successful']}/{v['records']}.")
    lines+=['', '## Усечение по вариантам', '']
    for c in m['candidates']:
        v=comp[c['name']]['truncated'];lines.append(f"- {c['name']}: {v['records']} — {', '.join(v['ids']) or 'нет'}.")
    lines+=['', 'Нет отдельной интеграции Jev в исследованном src/config; эти запуски — Laya через локальный SDK, не Jev. Длинный вопрос не даёт модели фактической истории согласия. Изменение score или решения на22 просмотренных письмах не доказывает превосходство формулировки вообще.']
    (out/'LAYA_PERSONAL_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    recommendation=['# Рекомендация для итогового демо', '', 'Рабочий E0 сохранён без изменения. Для демонстрации многоязычной альтернативы разумно рассмотреть E2: существующий char-LR обучен на historical+RU auxiliary, сохраняет простой локальный inference и подтверждённый reload. Личный набор используется как диагностика ограничений, не как достаточное основание объявить универсального победителя.',
        '', 'Фактическое сравнение на личном наборе:', '']+table('confirmed')
    recommendation+=['', 'Исходный E0 проверялся ранее прежде всего на исторической английской почте. E2, адаптированные Laya и C ранее оценивались на English historical и RU auxiliary раздельно; RU auxiliary не доказанный корпус электронной почты. Настоящий набор даёт22 подтверждённых пользовательских решения, включая смешанные скрипты, согласованные рассылки и служебные письма. Scripts не доказывают язык: Latin-only не означает English; отдельной репрезентативной современной EN-оценки здесь нет.',
        'Выбор кандидата/порога по этим письмам превращает их в development data. Разница в1 письмо, несколько одной кампании и unknown phishing не обосновывают перенос на всех пользователей. Все кандидаты text-only: история согласия, контактов и отношений с отправителем отсутствует на входе.',
        'Новое обучение не запускалось. Возможные основания для будущей адаптации — ошибки на согласованных рассылках, усечение и перенос на современные смешанные тексты; сначала нужен отдельный набор с независимыми кампаниями, источниками и контекстом согласия. Подгонять текущие модели по22 письмам не следует.',
        'До упаковки остаются выбор демонстрационного кандидата и интерфейса отказа, правила обращения с пользовательским согласием, проверка приватности логов и отдельная заранее зафиксированная финальная оценка. Закрытый test сейчас не открывался. Публикация/API/контейнеры в этой задаче не выполнялись.',
        'Пользовательских вопросов о бинарных метках не осталось. Решение о смене default и допустимом компромиссе FP/FN остаётся отдельным продуктовым выбором; никаких настроек по личным меткам автоматически не менялось.']
    (out/'FINAL_MODEL_RECOMMENDATION.md').write_text('\n'.join(recommendation)+'\n',encoding='utf-8')
    presentation=['# Материал для презентации: личные письма', '', f'Локальный запуск {out.name}; {len(members)} писем, ham{counts[0]}/spam{counts[1]}, бинарные метки окончательно подтверждены пользователем17.10.2026. Спорных0, категории вспомогательные, phishing ground truth нет.',
        'Датированный manifest зафиксирован до предсказаний: SHA разметки и писем, части набора, точные вопросы, веса, T/threshold и код. Ни одного обучения или выбора порога на личных письмах. Ссылки/вложения не открывались, внешние API не использовались.', '']+table('confirmed')
    presentation+=['',f"Modern subset: {len(m['subsets']['modern'])}; одно письмо2019 рассматривается отдельно. Исходные вопросы Laya изменили решение в {len(res['original_question_changed_ids'])}/{res['original_questions_successfully_compared']} успешно сопоставленных парах.",
        'Показать ограничения: мало писем одного пользователя, возможные связанные шаблоны/кампании, уже просмотренный development-набор, различия token caps, неизвестный контекст согласия для модели. Не утверждать универсальную победу по разнице в одно письмо.',
        'Практический вывод: сохранён E0; сравнение включает E2, исходные/адаптированные Laya и C с исправленной калибровкой. Новая модель не подменяет существующую по результату удобного набора. Следующий шаг — выбор демонстрационного поведения и приватной обработки, затем отдельная фиксированная проверка.',
        'Этот файл содержит только агрегаты; без адресов, кодов, персональных ссылок, имён и текстов писем. Подробные CSV/предсказания предназначены только для локального использования.']
    (out/'PRESENTATION_NOTES_PERSONAL_EMAILS.md').write_text('\n'.join(presentation)+'\n',encoding='utf-8')

def run(annotation):
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+digest(now())[:8]
    out=BASE/'runs'/run_id;out.mkdir(parents=True,exist_ok=False);print('Preparing private run',run_id,flush=True)
    m,boundary=prepare(out,annotation)
    write_json(out/'preservation_before.json',protected_snapshot())
    for path in ['src/personal_eval_common.py','src/evaluate_personal_emails.py','src/personal_neural_worker.py']:
        shutil.copyfile(ROOT/path,out/(Path(path).name+'.txt'))
    write_json(out/'inference_started.json',{'started_utc':now(),'manifest_sha256':sha_file(out/'manifest.json'),'no_fit':True})
    records=read_rows(out/'private_inputs.jsonl');allrows=[]
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    for c in m['candidates']:
        name=c['name'];print('Evaluating',name,flush=True)
        try:
            if c['kind']=='sklearn':pred=sparse_predictions(out,c,records)
            else:
                with (out/f'worker_{name}.log').open('w',encoding='utf-8') as log:
                    p=subprocess.run([str(ROOT/'.venv-decision/Scripts/python.exe'),'-B',str(ROOT/'src/personal_neural_worker.py'),
                        '--out',str(out),'--model',name],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                require(p.returncode==0,'Worker exited nonzero; see private worker log')
                pred=read_rows(out/f'predictions_{name}.jsonl')
            require(len(pred)==len(records),'Prediction count differs')
        except Exception as e:
            pred=[prediction_row(r,c,'unavailable',error_type=type(e).__name__,private_error=str(e)) for r in records]
            write_rows(out/f'predictions_{name}.jsonl',pred)
        allrows.extend(pred);print(name,'coverage',sum(r['status']=='ok' for r in pred),'/',len(pred),flush=True)
    write_rows(out/'predictions.jsonl',allrows)
    fields=['record_id','model_id','spam_label','label_status','label_source','label_reviewed_at','score','score_semantics',
        'threshold','temperature','decision','status','insufficient_features','active_features','sequence_tokens','truncated','error_type']
    with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:(r.get('token_stats',{}).get('truncated') if k=='truncated' else r.get(k,'')) for k in fields} for r in allrows)
    reports(out)
    with (out/'independent_audit.log').open('w',encoding='utf-8') as log:
        p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-B',str(Path(__file__)),'audit','--out',str(out)],
            env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    require(p.returncode==0,'Independent audit failed; see log')
    write_json(out/'run_result.json',{'status':'completed','run_id':run_id,'manifest_sha256':sha_file(out/'manifest.json'),
        'fit_performed':False,'model_errors':[c['name'] for c in m['candidates'] if any(r['status']!='ok' for r in allrows if r['model_id']==c['name'])],
        'independent_audit':'passed','parent_boundary':boundary.report()})
    print('Completed',rel(out),flush=True)
    res=read_json(out/'comparison.json')
    print(json.dumps({k:v['confirmed']['metrics'] for k,v in res['comparisons'].items()},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','audit']);p.add_argument('--out',type=Path)
    p.add_argument('--annotation',type=Path,default=PERSONAL/'PERSONAL_EMAIL_ANNOTATION.csv');a=p.parse_args()
    if a.stage=='run':
        source=a.annotation.resolve();require(source.is_relative_to(PERSONAL),'Annotation outside personal root');run(source)
    else:
        require(a.out is not None,'--out required');out=a.out.resolve();require(out.is_relative_to(BASE/'runs'),'Audit outside personal runs');audit(out)
