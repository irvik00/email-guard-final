"""Two controlled domain-weight LR fits; W0 is a saved, immutable baseline."""
import argparse
import subprocess
import warnings
from domain_weighting_common import *


def prepare(out):
    import joblib
    out.mkdir(parents=True,exist_ok=False);boundary=WeightBoundary(out).install()
    m=read_json(W0_DIR/'candidate_manifest.json');oldseal=read_json(PREVIOUS/'artifact_manifest_final.json')
    oldseal['files']={Path(k).as_posix():h for k,h in oldseal['files'].items()}
    needed=['candidate_manifest.json','fit_membership.jsonl','fit_embeddings.npy','classifier.joblib','selection_and_collisions.json']
    for n in needed:require(sha_file(W0_DIR/n)==oldseal['files']['C_larger_fit/'+n],'Previous sealed C artifact changed')
    audit=read_json(PREVIOUS/'independent_audit.json');require(audit['status']=='passed' and audit['larger_fit']['fit_rows']==16032,'Unverified W0')
    require(sha_file(PREVIOUS/'independent_audit.json')==oldseal['files']['independent_audit.json'],'Previous audit changed')
    fit=read_rows(W0_DIR/'fit_membership.jsonl');require(len(fit)==16032 and len({r['record_id'] for r in fit})==16032,'Fit count/duplicates')
    counts=Counter(r['domain'] for r in fit);require(dict(counts)=={'spam_email':2032,'russian_auxiliary':14000},'Unexpected fit counts')
    clf=joblib.load(ROOT/m['classifier_path']);require(clf.get_params()==m['fit_parameters'],'W0 parameters changed')
    require(m['prefix']=='query: ' and m['max_tokens']==512 and m['pooling']=='attention-mask mean then L2' and m['encoder_frozen'],'Wrong input contract')
    original_p=read_json(PREVIOUS/'EXTRA_SESSION_PROTOCOL.json');corrected=next(c for c in original_p['components'] if c['name']=='C_corrected')['saved_manifest']
    require(sha_file(PREVIOUS/'EXTRA_SESSION_PROTOCOL.json')==read_json(PREVIOUS/'protocol_lock.json')['sha256'],'Previous protocol lock changed')
    require(sha_file(DAY5/'membership.jsonl')==read_json(W0_DIR/'selection_and_collisions.json')['source_membership_sha256'],'Original inner membership changed')
    spec=read_json(EXTRA/'embeddings.json');vspec=read_json(EXTRA/'validation_embeddings.json')
    require(spec['model']==m['source']==vspec['model'],'Incompatible encoder revision/tokenizer metadata')
    inner={};hashes={};files=[W0_DIR/n for n in needed]+[PREVIOUS/n for n in ['EXTRA_SESSION_PROTOCOL.json','protocol_lock.json','independent_audit.json','artifact_manifest_final.json','pilot_membership.jsonl','private_pilot_inputs.jsonl','C_larger_fit_predictions.jsonl','C_LARGER_FIT.md']]
    files += [EXTRA/'embeddings.json',EXTRA/'validation_embeddings.json',DAY5/'membership.jsonl',DAY5/'data_manifest.json',ROOT/'src/extra_session_larger_c.py',ROOT/'src/day5_common.py',ROOT/'src/normalization_schema.py',ROOT/'config/day4_experiments.json']
    all_roles={r['record_id']:r for r in read_rows(DAY5/'membership.jsonl')}
    require(all(all_roles[r['record_id']]['inner_part']=='fit' and not all_roles[r['record_id']]['conflicting_group'] for r in fit),'Reserved row in fit')
    fitids={r['record_id'] for r in fit};fitgroups={r['duplicate_group_id'] for r in fit};fitcaps={r['actual_input_ids_sha256'] for r in fit}
    for part in ['dev','calibration']:
        entry=spec['parts'][part];inp=EXTRA/f'private_{part}_inputs.jsonl';visible=EXTRA/f'private_{part}_visible.jsonl'
        require(sha_file(inp)==entry['input_sha256'] and sha_file(visible)==entry['visible_sha256'],'Inner input/visible cache changed')
        rows=read_rows(inp);vis=read_rows(visible);ids=[i for i,r in enumerate(rows) if part!='calibration' or r['record_id'] not in corrected['calibration_excluded_ids']]
        inner[part]=[rows[i] for i in ids];require(not fitids&{r['record_id'] for r in inner[part]} and not fitgroups&{r['duplicate_group_id'] for r in inner[part]},'Inner ID/group overlap')
        require(not fitcaps&{vis[i]['token_stats']['actual_input_ids_sha256'] for i in ids},'Selected inner capped overlap')
        files += [inp,visible,ROOT/entry['matrix_path']]
    ventry=vspec['parts']['validation'];vpath=EXTRA/'private_validation_inputs.jsonl';vvisible=EXTRA/'private_validation_visible.jsonl'
    require(sha_file(vpath)==ventry['input_sha256'] and sha_file(vvisible)==ventry['visible_sha256'],'Validation cache input changed')
    val=read_rows(vpath);vvis=read_rows(vvisible);require(len(val)==len(vvis)==7265,'Validation count mismatch')
    require(not fitids&{r['record_id'] for r in val} and not fitgroups&{r['duplicate_group_id'] for r in val},'Validation ID/group in fit')
    fit_exact={all_roles[r['record_id']]['clean_text_sha256'] for r in fit}
    require(not fit_exact&{digest(r['text']) for r in val},'Clean exact fit/validation collision')
    cap_overlap=[v['record_id'] for v in vvis if v['token_stats']['actual_input_ids_sha256'] in fitcaps]
    files += [vpath,vvisible,ROOT/ventry['matrix_path']]
    files += [EXTRA/'validation_predictions.jsonl',PREVIOUS/'baseline_predictions.jsonl']
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model']:
        path=ROOT/m['tokenizer_path']/name
        if path.exists():
            require(sha_file(path)==m['source']['files'][name]['sha256'],'Saved tokenizer differs from verified source')
            files.append(path)
    for path in files:hashes[path.relative_to(ROOT).as_posix()]=sha_file(path)
    for part in ['dev','calibration']:require(hashes[(ROOT/spec['parts'][part]['matrix_path']).relative_to(ROOT).as_posix()]==spec['parts'][part]['sha256'],'Inner matrix changed')
    require(hashes[(ROOT/ventry['matrix_path']).relative_to(ROOT).as_posix()]==ventry['sha256'],'Validation matrix changed')
    labels=read_rows(PREVIOUS/'pilot_membership.jsonl');pilot_ids={}
    for domain in ['spam_email','russian_auxiliary','personal_development']:pilot_ids[domain]=[r['record_id'] for r in labels if r['domain']==domain]
    require([len(pilot_ids[d]) for d in pilot_ids]==[128,128,22],'Pilot changed')
    for n,v in m['versions'].items():
        if n not in {'torch','transformers'}:require(importlib.metadata.version(n)==v,'Main runtime changed')
    protected={p.relative_to(ROOT).as_posix():sha_file(p) for p in [ROOT/'models/day3/baseline_bundle.joblib',ROOT/'models/day3/model_manifest.json',ROOT/'.gitignore',ROOT/'.venv/pyvenv.cfg',ROOT/'.venv-decision/pyvenv.cfg']}
    for f in (ROOT/'config').glob('*.json'):protected[f.relative_to(ROOT).as_posix()]=sha_file(f)
    write_json(out/'preservation_before.json',protected)
    p={'created_utc':now(),'run_id':out.name,'status':'sealed_before_weighted_fit','W0_manifest':m,
        'input_hashes':hashes,'fit_ids_order_sha256':digest([r['record_id'] for r in fit]),'fit_counts':dict(counts),
        'lr_parameters':m['fit_parameters'],'variants':{'W0':'existing saved LR, no retraining/sample_weight','W1':SHARES['W1'],'W2':SHARES['W2']},
        'weight_formula':'N_total*target_domain_share/N_domain; one float64 weight per original row; no physical duplicates',
        'normalization':'float64 math.fsum close to N_total at1e-8 absolute; mean1. Native sklearn float32 validation rounding logged without ad-hoc rescaling; relative tolerance1e-6.',
        'class_weight':m['fit_parameters']['class_weight'],'class_weight_interaction':'None: only sample_weight; both classes balanced in each domain, weighted class totals remain N/2',
        'reference_C6000':corrected,'reference_policy':'Saved compatible C_corrected operating predictions only, supplementary reference; not part of causal W0/W1/W2 comparison',
        'calibration_excluded_ids':corrected['calibration_excluded_ids'],'calibration_rows':999,'dev_rows':600,
        'calibration_rule':'Original unweighted scalar NLL,100 golden-section logT iterations,T[.5,5], endpoints included; no sample weighting in calibration/dev',
        'threshold_rule':'Exact day5_common.choose_threshold: max mean domain F1, historicalFPR<=.01, positive recall both, ties fewer historicalFP then higher threshold, fallback.5',
        'pilot_ids':pilot_ids,'evaluation_order':['fixed pilot first','full historical and RU validation second'],
        'evaluation_modes':{'raw':'logit>=0, no temperature needed','operating':'sigmoid(logit/ownT)>=ownThreshold'},
        'metrics':['TP','FP','TN','FN','precision','recall','F1','FPR','coverage','refusals','rescued/added FP/FN vs W0 on common successful IDs'],
        'full_validation_counts':dict(Counter(r['domain'] for r in val)),
        'overlap_checks':{'fit_dev_cal_validation_ID_group_disjoint':True,'selected_inner_capped_overlap':0,'full_validation_capped_overlap_ids':cap_overlap,
            'canonical_audit':'Reuse previous verified audit by SHA; no repeated26k tokenization, no raw train read'},
        'rules':['Same fit16032/order/embeddings; two new fits only; no regularization/grid search',
            'Old artifacts/default read-only; no tests/teacher/reserve/new personal/raw data; no Qwen/Laya runs',
            'Personal22 and validation only after model manifests frozen; no label changes or ID rules','No overall merged score, no default/winner/packaging',
            'Any newly observed capped validation dependency remains in full evaluation, flagged; fit membership never changed'],
        'main_versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','joblib']},
        'code_hashes':{f.relative_to(ROOT).as_posix():sha_file(f) for f in [*ROOT.glob('src/*domain_weighting*.py'),ROOT/'tests/test_domain_weighting.py']}}
    write_json(out/'DOMAIN_WEIGHTING_PROTOCOL.json',p);write_json(out/'protocol_lock.json',{'sha256':sha_file(out/'DOMAIN_WEIGHTING_PROTOCOL.json'),'utc':now()})
    (out/'DOMAIN_WEIGHTING_PROTOCOL.md').write_text('# Протокол весов доменов\n\n'+f"Зафиксирован {p['created_utc']} до обучения. SHA `{sha_file(out/'DOMAIN_WEIGHTING_PROTOCOL.json')}`.\n\n"
        'Ровно прежние16032 ID в прежнем порядке: historical2032,RU14000. Тот же E5 revision,query-prefix,max512,masked mean/L2 и проверенная матрицаfloat32. '
        'W0 — сохранённая LR; обучаются только W1 (вклад1/3 historical,2/3 RU) и W2 (1/2,1/2). '
        'Вес строки=N_total×доля/N_domain, математическая сумма=N_total, среднее1. Float64 веса; штатное округление sklearn кfloat32 регистрируется отдельно. '
        'Все LR-параметры сохранены, включая C1,lbfgs,seed42,class_weight=None,max_iter2000,tol1e-4. Физических копий строк нет.\n\n'
        'Каждой новой LR — отдельные logits/calibration999/T[.5,5]/dev600 threshold по точному прежнему правилу. '
        'Cal/dev не взвешиваются. До их фиксации pilot/full validation не оцениваются. '
        'Сначала те же128historical+128RU+22personal pilot; затем полные1275historical и5990RU validation, отдельно. '
        'Для всех W0/W1/W2: raw logit>=0 и собственные T/operating threshold; confusion/P/R/F1/FPR/coverage/refusals и парные изменения против W0.\n\n'
        'Закрытые tests/teacher/challenge/reserve/RU test и новые personal недоступны. Старые модели/default/отчёты не изменяются. '
        'Никаких дополнительных долей после результатов. Повторно используются проверенные manifests/кеши, прежний дорогостоящий аудит не повторяется. '
        'Гипотеза о весах проверяется на одинаковом fit; личные данные — development, RU labels provisional. Не выбирать новый default по этим наблюдениям.\n',encoding='utf8')
    write_json(out/'prepare_accesses.json',boundary.report());print('PROTOCOL SEALED',out,flush=True)


def train(out):
    boundary=WeightBoundary(out).install();p=protocol(out)
    import numpy as np
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    import sklearn.linear_model._logistic as logistic
    rows=read_rows(W0_DIR/'fit_membership.jsonl');require(digest([r['record_id'] for r in rows])==p['fit_ids_order_sha256'],'Fit order changed')
    x=np.load(W0_DIR/'fit_embeddings.npy',allow_pickle=False);require(x.shape==(16032,384) and x.dtype==np.float32 and np.isfinite(x).all(),'Fit matrix invariant')
    inner,mat=inner_parts(p);models={'W0':p['W0_manifest']};stats={'W0':weight_table(rows,sample_weights(rows))}
    for name in ['W1','W2']:
        dest=out/'models'/name;require(not dest.exists(),'Do not retrain existing variant');dest.mkdir(parents=True)
        w=sample_weights(rows,p['variants'][name]);np.save(dest/'sample_weight.npy',w,allow_pickle=False);stats[name]=weight_table(rows,w)
        trace=[];original=logistic._check_sample_weight
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            if result is not None:trace.append({'dtype':str(result.dtype),'native_sum':float(np.sum(result)),'high_precision_sum':math.fsum(result),
                'length':len(result),'actual_weight_hash':digest(result.tolist())})
            return result
        clf=LogisticRegression(**p['lr_parameters']);start=time.perf_counter();logistic._check_sample_weight=capture
        try:
            with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');clf.fit(x,[r['spam_label'] for r in rows],sample_weight=w)
        finally:logistic._check_sample_weight=original
        require(trace and all(abs(t['native_sum']/len(rows)-1)<1e-6 for t in trace),'Native weight scaling unexpected')
        require(not any(issubclass(a.category,ConvergenceWarning) for a in caught) and max(clf.n_iter_)<clf.max_iter,'LR did not converge')
        require(clf.get_params()==p['lr_parameters'] and list(clf.classes_)==[0,1],'LR parameters/direction changed')
        raw={}
        for part in ['calibration','dev']:
            z=clf.decision_function(mat[part]);require(np.isfinite(z).all(),'Nonfinite logits')
            raw[part]=[{k:r[k] for k in ['record_id','domain','spam_label']}|{'logit_difference':float(v)} for r,v in zip(inner[part],z)]
            write_rows(dest/f'{part}_raw.jsonl',raw[part])
        calibration=fit_temperature([r['spam_label'] for r in raw['calibration']],[r['logit_difference'] for r in raw['calibration']],.5,5.)
        T=calibration['applied_temperature'];operating=choose_threshold(raw['dev'],T)
        joblib.dump(clf,dest/'classifier.joblib');reloaded=joblib.load(dest/'classifier.joblib')
        a=clf.decision_function(mat['dev']);b=reloaded.decision_function(mat['dev']);error=float(np.max(np.abs(a-b)))
        require(error<=1e-12 and np.array_equal(a>=0,b>=0),'Raw reload changed')
        require(all((sigmoid(float(u)/T)>=operating['threshold'])==(sigmoid(float(v)/T)>=operating['threshold']) for u,v in zip(a,b)),'Operating reload changed')
        m={'variant':name,'frozen_utc':now(),'fit_rows':len(rows),'fit_ids_order_sha256':p['fit_ids_order_sha256'],
            'fit_embeddings_sha256':p['W0_manifest']['fit_embeddings_sha256'],'fit_parameters':clf.get_params(),
            'classifier_path':(dest/'classifier.joblib').relative_to(ROOT).as_posix(),'classifier_sha256':sha_file(dest/'classifier.joblib'),
            'sample_weight_sha256':sha_file(dest/'sample_weight.npy'),'weight_stats':stats[name],'native_weight_validation':trace,
            'fit_iterations':clf.n_iter_.tolist(),'seconds_including_calibration_reload':time.perf_counter()-start,
            'warnings':[str(c.message) for c in caught],'temperature':T,'threshold':operating['threshold'],
            'operating_logit_cutoff':T*math.log(operating['threshold']/(1-operating['threshold'])),
            'calibration':calibration,'threshold_selection':operating,'reload':{'dev_rows':600,'max_logit_error':error,'both_decisions_identical':True},
            'encoder_trained':False,'builder':p['W0_manifest']['builder'],'prefix':'query: ','max_tokens':512,'pooling':p['W0_manifest']['pooling']}
        write_json(dest/'candidate_manifest.json',m);models[name]=m;print(name,'frozen',clf.n_iter_.tolist(),'T',T,'threshold',m['threshold'],flush=True)
    write_json(out/'weight_statistics.json',stats);write_json(out/'models_frozen.json',{'utc':now(),'protocol_sha256':sha_file(out/'DOMAIN_WEIGHTING_PROTOCOL.json'),'models':models,'new_fit_calls':2,'W0_refitted':False})
    write_json(out/'train_accesses.json',boundary.report())


def evaluate(out):
    boundary=WeightBoundary(out).install();p=protocol(out)
    import numpy as np
    import joblib
    from threadpoolctl import threadpool_limits
    require(not (out/'evaluation_started.json').exists(),'Existing evaluation retained; no silent repeat')
    freeze=read_json(out/'models_frozen.json');write_json(out/'evaluation_started.json',{'utc':now(),'models_sha256':sha_file(out/'models_frozen.json'),'order':p['evaluation_order']})
    meta=read_json(EXTRA/'validation_embeddings.json')['parts']['validation'];vrows=read_rows(EXTRA/'private_validation_inputs.jsonl');vmatrix=np.load(ROOT/meta['matrix_path'],allow_pickle=False)
    vidx={r['record_id']:i for i,r in enumerate(vrows)};personal=read_json(out/'personal_embedding_manifest.json');plabels={r['record_id']:r for r in read_rows(PREVIOUS/'pilot_membership.jsonl')}
    require(sha_file(out/'private_personal_embeddings.npy')==personal['sha256'],'Personal cache changed')
    datasets={};full_ids={}
    for d,key in [('spam_email','pilot_historical'),('russian_auxiliary','pilot_ru')]:
        indices=[vidx[i] for i in p['pilot_ids'][d]];datasets[key]=([vrows[i] for i in indices],vmatrix[indices])
    datasets['personal_development']=([plabels[i] for i in personal['ids']],np.load(out/'private_personal_embeddings.npy',allow_pickle=False))
    for d,key in [('spam_email','full_historical_validation'),('russian_auxiliary','full_ru_validation')]:
        indices=[i for i,r in enumerate(vrows) if r['domain']==d];datasets[key]=([vrows[i] for i in indices],vmatrix[indices])
    old={r['record_id']:r for r in read_rows(PREVIOUS/'C_larger_fit_predictions.jsonl')};metrics={};changes={};previous_errors=[];reference={}
    ref_full={r['record_id']:r for r in read_rows(EXTRA/'validation_predictions.jsonl')}
    ref_pilot={r['record_id']:r for r in read_rows(PREVIOUS/'baseline_predictions.jsonl') if r['model_id']=='C_corrected'}
    with threadpool_limits(limits=1):
        for key in SETS:
            rows,matrix=datasets[key];pred={}
            for name,m in freeze['models'].items():
                require(sha_file(ROOT/m['classifier_path'])==m['classifier_sha256'],'Classifier changed')
                clf=joblib.load(ROOT/m['classifier_path']);pred[name]=predict_rows(clf,rows,matrix,m['temperature'],m['threshold'])
                if name=='W0' and key in SETS[:3]:
                    for r in pred[name]:
                        prior=old[r['record_id']];require(r['operating_decision']==prior['decision'] and abs(r['logit_difference']-prior['logit_difference'])<1e-4,'W0 pilot incompatibility')
                        previous_errors.append(abs(r['logit_difference']-prior['logit_difference']))
                write_rows(out/f'predictions_{key}_{name}.jsonl',pred[name])
            metrics[key]={name:{mode:model_metrics(pr,mode) for mode in ['raw','operating']} for name,pr in pred.items()}
            changes[key]={name:{mode:paired_errors(pred['W0'],pred[name],mode) for mode in ['raw','operating']} for name in ['W1','W2']}
            ref=[];refm=p['reference_C6000']
            for r in rows:
                if key in SETS[:3]:
                    source=ref_pilot[r['record_id']];ref.append({'record_id':r['record_id'],'spam_label':r['spam_label'],'status':source['status'],'operating_decision':source['decision']})
                else:
                    source=ref_full[r['record_id']];require(source['spam_label']==r['spam_label'],'Reference label mismatch')
                    ref.append({'record_id':r['record_id'],'spam_label':r['spam_label'],'status':source['status'],
                        'operating_decision':int(sigmoid(source['logit_difference']/refm['temperature'])>=refm['threshold']) if source['status']=='ok' else None})
            reference[key]=model_metrics(ref,'operating')
            print('evaluated',key,len(rows),'W0/W1/W2 raw+operating',flush=True)
    write_json(out/'metrics_and_paired_changes.json',{'metrics':metrics,'paired':changes,'sets_separate':True,
        'reference_C6000_operating':reference,'W0_pilot_verification':{'rows':len(previous_errors),'max_logit_error':max(previous_errors),'decisions_identical':True}})
    write_json(out/'evaluate_accesses.json',boundary.report())


def main():
    sys.stdout.reconfigure(encoding='utf8');sys.stderr.reconfigure(encoding='utf8')
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['run','prepare','train','evaluate']);a.add_argument('--out',type=Path);args=a.parse_args()
    out=(args.out or BASE/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-W012')).resolve();require(out.is_relative_to(BASE),'Wrong output')
    if args.phase=='run':
        prepare(out);train(out)
        for executable,script in [(ROOT/'.venv-decision/Scripts/python.exe','domain_weighting_personal.py')]:
            r=subprocess.run([str(executable),'-B',str(ROOT/'src'/script),'--out',str(out)],capture_output=True,text=True,encoding='utf8',timeout=600)
            (out/(script+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf8');require(r.returncode==0,'Personal encoder blocked; see process log')
        evaluate(out)
        for script in ['report_domain_weighting.py','audit_domain_weighting.py','report_domain_weighting.py']:
            r=subprocess.run([sys.executable,'-B',str(ROOT/'src'/script),'--out',str(out)],capture_output=True,text=True,encoding='utf8',timeout=600)
            (out/(script+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf8');print(r.stdout,flush=True);require(r.returncode==0,script+' failed; see log')
        print('COMPLETED',out,flush=True)
    else:{'prepare':prepare,'train':train,'evaluate':evaluate}[args.phase](out)

if __name__=='__main__':main()
