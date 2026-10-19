"""Three-stage final evaluation: cached development, sealed contract, held-out inference."""
import argparse
import shutil
import subprocess
import time
from final_eval_common import *

DAY5='20261012T083529Z-0950e3dc34'
EXTRA='20261012T095633Z-7d01f7b8'
INITIAL_STOPPED='20261017T132052Z-1e046e6a'

def ordinary_test_assignment(assignment,domain,record_id):
    # protected_test marks challenge protection, not every ordinary held-out split.
    return (assignment['role']=='test' and not assignment.get('reserved_future',False) and
        {'dataset':domain,'split':'test','record_id':record_id} in assignment['assignments'])

def components():
    original=read_json(PERSONAL_RUN/'manifest.json');registry={c['name']:c for c in original['candidates']}
    chosen=[registry[n] for n in COMPONENTS]
    for c in chosen:
        for p,h in c['files'].items():require(sha_file(ROOT/p)==h,'Component artifacts differ')
    c=chosen[2]['saved_manifest'];require(c['calibration_rows']==999 and c['experiment']=='C_calibration_disjoint_diagnostic','Corrected C required')
    return chosen

def cached_validation(out,chosen,prepared):
    files={'E0':ROOT/f'reports/day5/runs/{DAY5}/E0_full_context_validation_predictions.jsonl',
        'Laya_day5':ROOT/f'reports/day5/runs/{DAY5}/adapted_validation_predictions.jsonl',
        'C_corrected':ROOT/f'reports/extra_experiments/runs/{EXTRA}/C/validation_predictions.jsonl'}
    data={n:read_rows(p) for n,p in files.items()};candidates={c['name']:c for c in chosen}
    # Reapply corrected calibration only to saved uncalibrated C logits from the same
    # frozen encoder/classifier. No need to repeat inference or fit any component.
    old=read_json(ROOT/f'reports/extra_experiments/runs/{EXTRA}/C/candidate_manifest.json')
    require(old['classifier_sha256']==candidates['C_corrected']['saved_manifest']['classifier_sha256'],'C cached LR differs')
    require(old['source']==candidates['C_corrected']['saved_manifest']['source'],'C cached encoder differs')
    require(read_json(ROOT/f'models/day5/runs/{DAY5}/candidate_manifest.json')['candidates']['adapted']['checkpoint_path']==candidates['Laya_day5']['checkpoint_path'],'Laya cache checkpoint differs')
    results={};allrows=[]
    for domain in DOMAINS:
        partition={n:[r for r in rows if r['domain']==domain] for n,rows in data.items()}
        ids={n:{r['record_id'] for r in rows} for n,rows in partition.items()}
        require(all(ids[n]==ids['E0'] and len(ids[n])==len(partition[n]) for n in COMPONENTS),'Cached validation ID alignment differs')
        truth=[{'record_id':r['record_id'],'spam_label':r['spam_label'],'label_status':'validation_source_labels'} for r in partition['E0']]
        expected=prepared['datasets'][domain]['splits']['validation']
        require(len(truth)==expected['records'] and dict(Counter(str(r['spam_label']) for r in truth))==expected['class_counts'],'Validation counts differ')
        labels={r['record_id']:r['spam_label'] for r in truth};pred={}
        for name,rows in partition.items():
            candidate=candidates[name];pred[name]={}
            for r in rows:
                require(r['spam_label']==labels[r['record_id']],'Cached validation label differs')
                p=float(r['probability']) if name=='E0' else sigmoid(r['logit_difference']/candidate['temperature'])
                ok=r.get('status','ok')=='ok' and not(name=='E0' and r.get('active_features',1)==0)
                pred[name][r['record_id']]={'model_id':name,'status':'ok' if ok else 'insufficient_features' if name=='E0' else 'error',
                    'score':p,'decision':int(p>=candidate['threshold']) if ok else None,'threshold':candidate['threshold'],
                    'temperature':candidate['temperature'],'prediction_origin':'saved_compatible_validation',
                    **({'logit_difference':r['logit_difference']} if 'logit_difference' in r else {})}
        dataset=domain+'_validation';rows=combine(dataset,truth,pred);allrows.extend(rows)
        results[dataset]={'metrics':summary(rows),'changes_vs_E0':paired_changes(rows),'source_validation_previously_used':True}
    personal=[r for r in read_rows(PERSONAL_RUN/'predictions.jsonl') if r['model_id'] in COMPONENTS]
    oldmanifest=read_json(PERSONAL_RUN/'manifest.json');truth=[{'record_id':r['record_id'],'spam_label':r['spam_label'],'label_status':'human_confirmed_final'} for r in oldmanifest['records']]
    pred={n:{r['record_id']:r for r in personal if r['model_id']==n} for n in COMPONENTS}
    for name in COMPONENTS:
        candidate=candidates[name]
        require(set(pred[name])=={r['record_id'] for r in truth},'Personal prediction IDs differ')
        require(all(r['threshold']==candidate['threshold'] and r['temperature']==candidate['temperature'] for r in pred[name].values()),'Personal artifact settings differ')
    rows=combine('personal_development',truth,pred);allrows.extend(rows);s=summary(rows)
    expectations={'E0_OR_Laya':{'TP':8,'FP':7,'TN':6,'FN':1},'E0_AND_Laya':{'TP':4,'FP':2,'TN':11,'FN':5},
        'Majority_E0_Laya_C':{'TP':7,'FP':6,'TN':7,'FN':2}}
    for name,counts in expectations.items():require(s[name]['confusion']==counts,'Personal ensemble check differs; do not tune answers')
    require(s['E0_Laya_review']['automatic']==13 and s['E0_Laya_review']['needs_review']==9 and
        s['E0_Laya_review']['automatic_confusion']=={'TP':4,'FP':2,'TN':6,'FN':1},'Personal review check differs')
    results['personal_development']={'metrics':s,'changes_vs_E0':paired_changes(rows),'independent_test':False,'expected_checks':'passed'}
    write_rows(out/'ensemble_validation_predictions.jsonl',allrows);save_csv(out/'ensemble_validation_predictions.csv',allrows)
    write_json(out/'validation_results.json',results)
    sources={n:{'path':str(p.relative_to(ROOT).as_posix()),'sha256':sha_file(p)} for n,p in files.items()}
    sources['personal']={'path':str((PERSONAL_RUN/'predictions.jsonl').relative_to(ROOT).as_posix()),'sha256':sha_file(PERSONAL_RUN/'predictions.jsonl')}
    write_json(out/'validation_sources.json',sources);return results

def freeze(out,chosen,prepared,guard):
    stat=TEACHER.stat()
    protocol={'version':1,'run_id':out.name,'frozen_utc':now(),'main_variant':'E0','cost_function_FP_FN':None,
        'technical_retry':{'initial_run':INITIAL_STOPPED,
            'initial_protocol_sha256':sha_file(BASE/'runs'/INITIAL_STOPPED/'FINAL_EVALUATION_PROTOCOL.json'),
            'reason':'Incorrect assertion protected_test=True for ordinary held-out records; actual role=test and membership authorize test, protection flag refers to challenges',
            'initial_historical_test_access':'first record decoded, then stopped; metadata inspected to diagnose schema; no model test inference',
            'initial_teacher_archive_access':False,'component_or_ensemble_settings_changed':False},
        'main_variant_reason':'No previously fixed numerical FP/FN cost; E0 main by user instruction, not selected on teacher test',
        'additional_binary_comparisons':list(BINARY[1:]),'separate_assistant_mode':'E0_Laya_review',
        'components':chosen,'composition_rules':{'E0_OR_Laya':'binary OR of E0 and Laya decisions',
            'E0_AND_Laya':'binary AND of E0 and Laya decisions','Majority_E0_Laya_C':'at least2 of3 spam votes',
            'E0_Laya_review':'same decisions -> automatic; disagreement -> needs_review'},
        'component_errors':'error/insufficient_features/unavailable/missing never converted to ham; decision null',
        'ensemble_errors':'Strict complete required components; OR/AND require both, majority all3. Missing -> component_unavailable/null. Review -> needs_review/component_unavailable, counted separately from disagreement.',
        'score_policy':'No averaging, score conversion, new threshold or weights; ensemble only binary decisions',
        'binary_metrics':['TP','FP','TN','FN','precision','recall','F1','FPR','coverage'],
        'review_metrics':['automatic fraction','review counts by true class and reason','automatic confusion/errors; no whole-dataset F1'],
        'undefined_ratios':'null/N/A with reason; incomplete-coverage metrics conditional on successful decisions',
        'datasets':{'historical_test':{'source':'data/processed/splits/spam_email/test.jsonl','expected':prepared['artifacts']['splits/spam_email/test.jsonl'],
            'classes':prepared['datasets']['spam_email']['splits']['test']['class_counts']},
            'teacher_test':{'source':str(TEACHER.relative_to(ROOT)),'expected_rows':1600,'classes':{'0':600,'1':1000},
                'size_bytes_stat_only':stat.st_size,'mtime_ns_stat_only':stat.st_mtime_ns,'archive_sha256':'record only after sealed protocol, before parsing',
                'label_rule':'exact case-insensitive ham/spam directory; file/path/GT never passed as model text'},
            'russian_auxiliary_test':{'source':'data/processed/splits/russian_auxiliary/test.jsonl','expected':prepared['artifacts']['splits/russian_auxiliary/test.jsonl'],
                'classes':prepared['datasets']['russian_auxiliary']['splits']['test']['class_counts'],
                'limits':'source_intended_unverified; artificially sampled classes, may be chat/short messages rather than email; separate supplemental evaluation'}},
        'evaluation_order':list(TEST_SETS),'inference':{'Laya':'CUDA eval/no_grad, batch1, exact saved B/cap700/context1024/head256; SDK decoder cross-check first3',
            'C':'CUDA frozen float32 E5 query-prefix/max512/masked mean+L2, batch16, CPU saved LR/T/threshold',
            'E0':'saved original subject+body builder, no new feature fitting; empty sparse vector -> insufficient_features',
            'model_input':'only model text; constant adapter label0 is not GT and not passed to model forward; paths/folders excluded'},
        'overlap_rules':{'reference':'all historical/RU train and validation pools; actual selected fit/calibration IDs distinguished in audit',
            'keys':['raw SHA256 when available','original_model_text exact UTF8 SHA256','NFC+casefold+whitespace canonical original',
                    'clean_builder_v2 exact','canonical clean_builder_v2','global group where available'],
            'teacher_primary':'all1600, no hidden deletions/relabeling',
            'secondary':'all records without any full exact/canonical/raw/global-group match to train or validation; show only separately',
            'within_test':'report duplicates/conflicting labels, no deduplication',
            'cross_test':'report exact/canonical overlaps separately; do not combine metrics',
            'limits':'No exhaustive semantic/near/visible-prefix independence guarantee; no changes after seeing overlap'},
        'archive_safety':{'extract_to_filesystem':False,'reject':'absolute/Windows-drive/backslash/.. paths, duplicate paths, links/devices/special entries',
            'max_member_bytes':20*1024**2,'max_total_regular_bytes':1024**3,'static_MIME_only':True,'no_external_resources':True},
        'closed_after_protocol':'all other challenges/reserves/modern/Nazario/security_blocked archive remain closed',
        'default_model_unchanged':True,'no_training_or_tuning':True,'packaging_or_publication':False,
        'preparation_manifest_sha256':sha_file(ROOT/'data/processed/splits/manifest.json'),
        'stage1_results_sha256':sha_file(out/'validation_results.json'),
        'code_sha256':{p:sha_file(ROOT/p) for p in ['src/final_eval_common.py','src/run_final_evaluation.py','src/final_neural_worker.py','src/final_evaluation_reports.py',
            'src/read_email.py','src/normalization_schema.py','src/day5_laya_runtime.py','src/decision_protocol.py']}}
    write_json(out/'FINAL_EVALUATION_PROTOCOL.json',protocol)
    from final_evaluation_reports import protocol_markdown
    (out/'FINAL_EVALUATION_PROTOCOL.md').write_text(protocol_markdown(protocol),encoding='utf-8')
    guard.seal();write_json(out/'protocol_lock.json',{'sha256':guard.lock,'sealed_utc':now(),'teacher_not_opened_before_seal':True})
    return protocol

def verified_split(domain,split,prepared):
    path=ROOT/f'data/processed/splits/{domain}/{split}.jsonl';expected=prepared['artifacts'][f'splits/{domain}/{split}.jsonl']
    h=hashlib.sha256();n=size=0
    with path.open('rb') as f:
        for line in f:
            h.update(line);size+=len(line);n+=1;r=__import__('json').loads(line)
            require(r['dataset']==domain and r['split']==split and type(r['spam_label']) is int and r['spam_label'] in {0,1},'Split/label differs')
            require(r['quality_status']=='eligible' and r['usable_model_text'] and 'needs_review' not in r['quality_flags'],'Ineligible saved split record')
            require(hashlib.sha256(r['model_text'].encode()).hexdigest()==r['model_text_sha256'],'Stored text SHA differs')
            yield r
    require({'sha256':h.hexdigest(),'rows':n,'size_bytes':size}==expected,'Processed split hash/count differs')

def reference_index(out,prepared):
    cfg=read_json(ROOT/'config/day4_experiments.json');index=defaultdict(list);groups=defaultdict(list);counts={}
    selection={part:{r['record_id'] for r in read_rows(ROOT/f'reports/day5/runs/{DAY5}/private_{part}_inputs.jsonl')} for part in ['fit','dev','calibration']}
    c=read_json(ROOT/'models/c_diagnostics/20261012T141331Z-43c7c5dc/candidate_manifest.json');c_excluded=set(c['calibration_excluded_ids'])
    for domain in DOMAINS:
        for split in ['train','validation']:
            count=0
            for r in verified_split(domain,split,prepared):
                count+=1;clean=clean_text(r,cfg)[0];roles=[]
                if domain=='spam_email' and split=='train':roles.append('E0_fit')
                if r['record_id'] in selection['fit']:roles+=['Laya_fit','C_fit']
                if r['record_id'] in selection['dev']:roles+=['Laya_dev','C_dev']
                if r['record_id'] in selection['calibration']:
                    roles.append('Laya_calibration')
                    if r['record_id'] not in c_excluded:roles.append('C_calibration999')
                ref={'record_id':r['record_id'],'dataset':domain,'split':split,'label':r['spam_label'],'actual_component_roles':roles}
                for key in text_keys(r['model_text'],clean,r.get('raw_sha256')):index[key].append(ref)
                groups[r['duplicate_group_id']].append(ref)
            counts[domain+'/'+split]=count
    write_json(out/'overlap_reference_manifest.json',{'counts':counts,'keys':len(index),'group_keys':len(groups),
        'full_pool_is_broader_than_selected_component_fit':True,'no_test_reference_content_read_yet':True})
    return index,groups

def prepare_processed_test(out,dataset,domain,prepared):
    cfg=read_json(ROOT/'config/day4_experiments.json');inputs=[];truth=[]
    for r in verified_split(domain,'test',prepared):
        assignment=prepared['group_split_mapping'][r['duplicate_group_id']]
        require(ordinary_test_assignment(assignment,domain,r['record_id']),'Test group role/assignment differs')
        clean=clean_text(r,cfg)[0];rid=r['record_id'];inputs.append(neutral_input(rid,dataset,r['model_text'],clean))
        truth.append({'record_id':rid,'spam_label':r['spam_label'],'label_status':r['label_status']['spam'],'source':r['source'],
            'duplicate_group_id':r['duplicate_group_id'],'keys':text_keys(r['model_text'],clean,r.get('raw_sha256')),'parse_status':r['parse_status']})
    expected=prepared['datasets'][domain]['splits']['test'];require(dict(Counter(str(r['spam_label']) for r in truth))==expected['class_counts'],'Test classes differ')
    write_rows(out/f'private_inputs_{dataset}.jsonl',inputs);write_rows(out/f'private_truth_{dataset}.jsonl',truth)
    return truth,{'rows':len(truth),'classes':dict(Counter(str(r['spam_label']) for r in truth)),
        'inference_inputs_sha256':sha_file(out/f'private_inputs_{dataset}.jsonl'),'truth_sha256':sha_file(out/f'private_truth_{dataset}.jsonl'),
        'source_expected_hash':prepared['artifacts'][f'splits/{domain}/test.jsonl'],'ground_truth_not_in_model_payload':True}

def prepare_teacher(out,protocol):
    from read_email import parse_email
    p=protocol['datasets']['teacher_test'];stat=TEACHER.stat()
    require(stat.st_size==p['size_bytes_stat_only'] and stat.st_mtime_ns==p['mtime_ns_stat_only'],'Teacher archive changed since protocol metadata')
    before=sha_file(TEACHER);cfg=read_json(ROOT/'config/day4_experiments.json');inputs=[];truth=[]
    with tarfile.open(TEACHER,'r:xz') as archive:
        files,safety=validate_tar_members(archive.getmembers())
        counts=dict(Counter(str(label) for member,label in files))
        require(len(files)==1600 and counts=={'0':600,'1':1000},'Actual teacher class counts differ from user contract')
        for i,(member,label) in enumerate(sorted(files,key=lambda x:x[0].name),1):
            f=archive.extractfile(member);require(f is not None,'Cannot stream regular member');raw=f.read(20*1024**2+1)
            require(len(raw)==member.size and len(raw)<=20*1024**2,'Member read size differs')
            try:parsed=parse_email(raw);parse_error=None
            except Exception as e:parsed={};parse_error=type(e).__name__
            original,clean=model_payload(parsed.get('subject'),parsed.get('body_text'),cfg);rid=f'teacher:{i:04d}'
            inputs.append(neutral_input(rid,'teacher_test',original,clean))
            truth.append({'record_id':rid,'spam_label':label,'label_status':'teacher_folder_provided','source':'teacher',
                'member_locator':member.name,'raw_sha256':hashlib.sha256(raw).hexdigest(),
                'keys':text_keys(original,clean,hashlib.sha256(raw).hexdigest()),'parse_error':parse_error,
                'parse_warnings':parsed.get('parse_warnings',[]),'attachment_count':len(parsed.get('attachments',[])),
                'body_source':parsed.get('body_source'),'text_characters':len(clean)})
    require(before==sha_file(TEACHER),'Teacher archive mutated')
    write_rows(out/'private_inputs_teacher_test.jsonl',inputs);write_rows(out/'private_truth_teacher_test.jsonl',truth)
    return truth,{'rows':len(truth),'classes':counts,'archive_sha256':before,'archive_unchanged':True,'archive_safety':safety,
        'parse_errors':sum(r['parse_error'] is not None for r in truth),'warnings_records':sum(bool(r['parse_warnings']) for r in truth),
        'inference_inputs_sha256':sha_file(out/'private_inputs_teacher_test.jsonl'),'truth_sha256':sha_file(out/'private_truth_teacher_test.jsonl'),
        'ground_truth_not_in_model_payload':True}

def overlap_audit(out,datasets,index,groups):
    output={};earlier_tests=defaultdict(list)
    for dataset in TEST_SETS:
        truth=datasets[dataset];matches=[];overlap_ids=[];conflicting=set();within=defaultdict(list);cross=[]
        for r in truth:
            allrefs={};types=set()
            for kind,h in r['keys']:
                within[(kind,h)].append(r)
                for ref in index.get((kind,h),[]):allrefs[(ref['dataset'],ref['split'],ref['record_id'])]=ref;types.add(kind)
                for old in earlier_tests.get((kind,h),[]):cross.append({'record_id':r['record_id'],'prior_dataset':old[0],'prior_id':old[1],'type':kind})
            for ref in groups.get(r.get('duplicate_group_id'),[]):allrefs[(ref['dataset'],ref['split'],ref['record_id'])]=ref;types.add('global_group')
            if allrefs:
                overlap_ids.append(r['record_id']);refs=list(allrefs.values())
                conflict=any(ref['label']!=r['spam_label'] for ref in refs)
                if conflict:conflicting.add(r['record_id'])
                matches.append({'record_id':r['record_id'],'label':r['spam_label'],'match_types':sorted(types),'reference_records':refs,'any_label_conflict':conflict})
        # Within-corpus duplicates use canonical clean text only; no removal.
        dup=[{'ids':[r['record_id'] for r in v],'labels':sorted({r['spam_label'] for r in v})} for (kind,h),v in within.items() if kind=='clean_canonical' and len(v)>1]
        output[dataset]={'overlapping_records':len(overlap_ids),'overlap_ids':overlap_ids,'matching_records':matches,
            'nonoverlap_ids':[r['record_id'] for r in truth if r['record_id'] not in set(overlap_ids)],
            'label_conflict_records':len(conflicting),'canonical_duplicate_groups':dup,'cross_test_matches':cross,
            'limits':'Whole exact/canonical/raw/global checks only; no semantic/near/prefix independence guarantee'}
        for r in truth:
            for key in r['keys']:earlier_tests[key].append((dataset,r['record_id']))
    write_json(out/'overlap_audit.json',output);return output

def sparse(out,dataset,c,inputs):
    import joblib
    from baseline_common import versions
    from threadpoolctl import threadpool_limits
    require(versions()==c['versions'],'E0 versions differ')
    for p,h in c['files'].items():require(sha_file(ROOT/p)==h,'E0 artifact differs')
    b=joblib.load(ROOT/c['bundle_path']);pipeline=b['pipeline'];require(list(pipeline.classes_)==[0,1],'E0 classes differ')
    result=[]
    with threadpool_limits(limits=1):
        for r in inputs:
            try:
                v=pipeline['tfidf'].transform([r['original_model_text']]);score=float(pipeline['classifier'].predict_proba(v)[0,1])
                ok=bool(r['original_model_text'].strip()) and v.nnz>0
                result.append({'record_id':r['record_id'],'model_id':'E0','status':'ok' if ok else 'insufficient_features',
                    'decision':int(score>=c['threshold']) if ok else None,'score':score,'threshold':c['threshold'],'temperature':1.,'active_features':int(v.nnz)})
            except Exception as e:result.append({'record_id':r['record_id'],'model_id':'E0','status':'error','decision':None,'score':None,
                'error_type':type(e).__name__,'private_error':str(e),'threshold':c['threshold'],'temperature':1.})
    write_rows(out/f'component_{dataset}_E0.jsonl',result);return result

def independent_audit(out):
    guard=FinalBoundary(out).install();protocol=read_json(out/'FINAL_EVALUATION_PROTOCOL.json');lock=read_json(out/'protocol_lock.json')
    require(sha_file(out/'FINAL_EVALUATION_PROTOCOL.json')==lock['sha256'],'Protocol changed')
    rows=read_rows(out/'final_test_predictions.jsonl');summaries=read_json(out/'final_results.json')
    indexed=defaultdict(dict)
    for r in rows:indexed[(r['dataset'],r['record_id'])][r['model_id']]=r
    for dataset in TEST_SETS:
        truth=read_rows(out/f'private_truth_{dataset}.jsonl');labels={r['record_id']:r['spam_label'] for r in truth}
        for name in MODEL_NAMES:
            p=[r for r in rows if r['dataset']==dataset and r['model_id']==name]
            require(len(p)==len(labels) and {r['record_id'] for r in p}==set(labels),'Final predictions missing/duplicated')
            require(all(r['spam_label']==labels[r['record_id']] for r in p),'Final labels differ')
            require(evaluate(p,name=='E0_Laya_review')==summaries[dataset]['full']['metrics'][name],'Stored metrics differ')
        for rid in labels:
            part=indexed[(dataset,rid)]
            for name in ENSEMBLES:
                expected=vote(name,part);actual=part[name]
                require(all(actual[k]==expected[k] for k in expected),'Final vote differs')
        inputs=read_rows(out/f'private_inputs_{dataset}.jsonl');require(all(r['spam_label']==0 and not any(k in r for k in ['member_locator','original_locator','local_path']) for r in inputs),'Label/path in model inputs')
    for c in protocol['components']:
        for p,h in c['files'].items():require(sha_file(ROOT/p)==h,'Component changed')
    before=read_json(out/'preservation_before.json');changed=[p for p,h in before.items() if sha_file(ROOT/p)!=h];require(not changed,'Old artifact changed')
    accesses=read_json(out/'final_accesses.json')
    require(all(a['protocol_sha256']==lock['sha256'] and a['first_open_utc']>protocol['frozen_utc'] for a in accesses['first_test_accesses']),'Test opened before seal')
    require(len(accesses['first_test_accesses'])==3,'Unexpected held-out access scope')
    write_json(out/'independent_audit.json',{'status':'passed','protected_files':len(before),'changed_files':changed,
        'protocol_before_all_test_accesses':True,'tests_read':accesses['first_test_accesses'],'component_artifacts_unchanged':True,
        'voting_and_metrics_checked':True,'no_path_or_GT_in_text_input':True})
    print('Independent final audit passed;',len(before),'old files preserved',flush=True)

def run():
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+digest(now())[:8];out=BASE/'runs'/run_id;out.mkdir(parents=True)
    guard=FinalBoundary(out).install();print('Stage1 validation only:',run_id,flush=True)
    prepared=read_json(ROOT/'data/processed/splits/manifest.json');chosen=components()
    validation=cached_validation(out,chosen,prepared)
    from final_evaluation_reports import validation_markdown,write_final_reports
    (out/'ENSEMBLE_VALIDATION.md').write_text(validation_markdown(validation),encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    with (out/'tests.log').open('w',encoding='utf-8') as log:
        test=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-B','-m','unittest','discover','-s','tests','-p','test_final_evaluation.py'],env=env,stdout=log,stderr=subprocess.STDOUT)
    require(test.returncode==0,'Final-evaluation fixture tests failed before test access')
    write_json(out/'preservation_before.json',old_artifacts_snapshot())
    protocol=freeze(out,chosen,prepared,guard);print('Stage2 protocol sealed:',guard.lock,flush=True)
    for name in protocol['code_sha256']:
        shutil.copyfile(ROOT/name,out/(Path(name).name+'.txt'))
    print('Stage3 opening only declared final tests',flush=True)
    index,groups=reference_index(out,prepared);datasets={};metadata={}
    datasets['historical_test'],metadata['historical_test']=prepare_processed_test(out,'historical_test','spam_email',prepared)
    datasets['teacher_test'],metadata['teacher_test']=prepare_teacher(out,protocol)
    datasets['russian_auxiliary_test'],metadata['russian_auxiliary_test']=prepare_processed_test(out,'russian_auxiliary_test','russian_auxiliary',prepared)
    write_json(out/'test_data_manifest.json',{'datasets':metadata,'protocol_sha256':guard.lock,'prepared_after_seal_utc':now()})
    overlaps=overlap_audit(out,datasets,index,groups);del index,groups
    allrows=[];results={}
    for dataset in TEST_SETS:
        inputs=read_rows(out/f'private_inputs_{dataset}.jsonl');pred={}
        for c in chosen:
            name=c['name'];print('Inference',dataset,name,flush=True)
            if name=='E0':predictions=sparse(out,dataset,c,inputs)
            else:
                path=out/f'component_{dataset}_{name}.jsonl'
                try:
                    with (out/f'worker_{dataset}_{name}.log').open('w',encoding='utf-8') as log:
                        p=subprocess.run([str(ROOT/'.venv-decision/Scripts/python.exe'),'-B',str(ROOT/'src/final_neural_worker.py'),
                            '--out',str(out),'--dataset',dataset,'--model',name],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
                    require(p.returncode==0,'Worker failed; original log retained')
                    predictions=read_rows(path)
                except Exception as e:
                    predictions=[{'record_id':r['record_id'],'model_id':name,'status':'unavailable','decision':None,'score':None,
                        'threshold':c['threshold'],'temperature':c['temperature'],'error_type':type(e).__name__,'private_error':str(e)} for r in inputs]
                    write_rows(out/f'failure_{dataset}_{name}.jsonl',predictions)
            pred[name]={r['record_id']:r for r in predictions};print(dataset,name,'coverage',sum(component_ok(r) for r in predictions),'/',len(inputs),flush=True)
        combined=combine(dataset,datasets[dataset],pred);allrows.extend(combined)
        non=set(overlaps[dataset]['nonoverlap_ids']);secondary=[r for r in combined if r['record_id'] in non]
        results[dataset]={'full':{'metrics':summary(combined),'changes_vs_E0':paired_changes(combined)},
            'nonoverlap':{'records':len(non),'metrics':summary(secondary)},'overlapping_records':overlaps[dataset]['overlapping_records']}
        write_rows(out/'final_test_predictions.jsonl',allrows);write_json(out/'final_results.json',results)
    save_csv(out/'final_test_predictions.csv',allrows)
    write_json(out/'confusion_matrices.json',{d:{n:(r.get('confusion') or r.get('automatic_confusion')) for n,r in s['full']['metrics'].items()} for d,s in results.items()})
    write_json(out/'final_accesses.json',guard.report());write_final_reports(out,protocol,validation,results,metadata,overlaps)
    with (out/'independent_audit.log').open('w',encoding='utf-8') as log:
        p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-B',str(Path(__file__)),'audit','--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    require(p.returncode==0,'Independent final audit failed; see preserved log')
    write_json(out/'run_result.json',{'status':'completed','run_id':run_id,'protocol_sha256':guard.lock,
        'teacher_archive_unchanged':metadata['teacher_test']['archive_unchanged'],'default_model_changed':False,
        'test_inference_attempts':1,'no_training_or_tuning':True,'independent_audit':'passed'})
    print('Final evaluation completed:',out.relative_to(ROOT),flush=True)
    print(__import__('json').dumps({d:{n:r for n,r in s['full']['metrics'].items()} for d,s in results.items()},indent=2),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8');sys.dont_write_bytecode=True
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','audit']);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.stage=='run':run()
    else:
        require(a.out is not None,'--out required');out=a.out.resolve();require(out.is_relative_to(BASE/'runs'),'Outside final runs');independent_audit(out)
