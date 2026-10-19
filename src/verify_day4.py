"""Independent no-fit audit of permitted Day4 outputs and final evidence handoff."""
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from baseline_common import ROOT,require,sha_file,digest_json,scores,metrics,threshold_modes,json_params,versions
from day4_common import Day4Boundary,load_day4,clean_text,subset_records
from day4_reporting import dump_rows
from pipeline_common import rows,write_json,write_text
from predict_email import load_bundle
from run_day4 import protected_snapshot


def main():
    run=json.loads((ROOT/'reports/day4/current_run.json').read_bytes())['run_id'];report=ROOT/'reports/day4/runs'/run
    result=json.loads((report/'experiments.json').read_bytes());config=result['config'];old=json.loads((ROOT/config['historical_config']).read_bytes())
    require(result['status']=='completed','Classical run incomplete')
    for path,h in result['code_sha256'].items():require(sha_file(ROOT/path)==h,'Experiment code changed: '+path)
    preserved=json.loads((report/'preserved_day3_and_environment.json').read_bytes())
    require(protected_snapshot(ROOT)==preserved,'Day3/core environment preservation failed')
    checks={'run_id':run,'status':'running','fit_performed':False,'test_not_evaluated':True,'models':{},'day3_unchanged':True,'allowed_input_hashes':'passed'}
    with Day4Boundary(ROOT) as boundary,threadpool_limits(1):
        data,folds,contract=load_day4(ROOT,config,old)
        require(contract['day4_inputs']==result['contract']['day4_inputs'],'Allowed input manifest changed')
        e0=load_bundle(ROOT/'models/day3/baseline_bundle.joblib')['pipeline'];models={'E0':e0}
        manifest=json.loads((ROOT/'models/day4/runs'/run/'model_manifest.json').read_bytes())
        for filename,spec in manifest['artifacts'].items():
            path=ROOT/'models/day4/runs'/run/filename;require(sha_file(path)==spec['sha256'] and path.stat().st_size==spec['size_bytes'],'Candidate artifact mismatch')
            bundle=joblib.load(path);require(bundle['versions']==versions() and bundle['builder']==config['clean_builder'],'Candidate metadata mismatch')
            require(json_params(bundle['pipeline'].get_params(deep=True))==bundle['parameters'],'Saved hyperparameters differ')
            models[bundle['model_id']]=bundle['pipeline']
        for name,model in models.items():
            checks['models'][name]={}
            for domain,ds in data.items():
                validation=ds['validation'];text=[r['model_text'] if name=='E0' else clean_text(r,config)[0] for r in validation]
                saved=list(rows(report/f'predictions/{name}_{domain}_validation.jsonl'))
                require([r['record_id'] for r in validation]==[r['record_id'] for r in saved],'Prediction ID/order mismatch')
                require(all(r['spam_label']==p['spam_label'] and r['source']==p['source'] for r,p in zip(validation,saved)),'Prediction label/source mismatch')
                score=scores(model,text);expected=np.array([r['spam_score'] for r in saved]);error=float(np.max(np.abs(score-expected)))
                require(error<=1e-12,'Reload score difference')
                ev=result['experiments'][name]['validation'][domain]
                for t,m in ev['raw_metrics'].items():require(metrics([r['spam_label'] for r in validation],score,float(t))==m,'Domain metrics mismatch')
                for source,ms in ev['by_source'].items():
                    mask=np.array([r['source']==source for r in validation])
                    for t,m in ms.items():require(metrics(np.array([r['spam_label'] for r in validation])[mask],score[mask],float(t))==m,'Source metric mismatch')
                active=np.diff(model['tfidf'].transform(text).indptr)
                require(np.array_equal(active,[p['active_features'] for p in saved]) and int(sum(active==0))==ev['empty_vectors'],'Wrapper coverage mismatch')
                checks['models'][name][domain]={'n':len(saved),'max_score_difference':error,'confusion_source_coverage':'passed'}
        if 'E1' in models:
            oof=list(rows(report/'predictions/E1_oof.jsonl'));train=data['spam_email']['train']
            require(len(oof)==len(train) and len({r['record_id'] for r in oof})==len(train),'OOF exact-once mismatch')
            require(all(p['record_id']==r['record_id'] and p['fold_id']==int(f) and p['duplicate_group_id']==r['duplicate_group_id'] for p,r,f in zip(oof,train,folds)),'OOF fold/group metadata mismatch')
            for fold,spec in enumerate(result['experiments']['E1']['cv_folds']):
                held=[p for p in oof if p['fold_id']==fold]
                require(metrics([p['spam_label'] for p in held],[p['spam_score'] for p in held],.5)==spec['metrics'],'OOF metric mismatch')
            saved_t=json.loads((report/'E1_tuned_thresholds.json').read_bytes())
            pred=list(rows(report/'predictions/E1_spam_email_validation.jsonl'))
            computed=threshold_modes([p['spam_label'] for p in pred],[p['spam_score'] for p in pred],old)
            require(computed==saved_t,'E1 threshold grid/Wilson mismatch')
            checks['OOF_and_thresholds']='passed'
        fields={}
        for domain,ds in data.items():
            for split,records in ds.items():
                for source in sorted({r['source'] for r in records}):
                    subset=[r for r in records if r['source']==source];available=[r for r in subset if r['attachments'] is not None]
                    attachments=[a for r in available for a in r['attachments']]
                    links=[l for r in subset if r['links'] is not None for l in r['links']]
                    fields[domain+'/'+split+'/'+source]={'records':len(subset),'attachment_metadata_available_records':len(available),
                        'attachment_entries':len(attachments),'nonempty_attachment_filenames':sum(bool(a.get('filename')) for a in attachments),
                        'missing_attachment_filename_entries':sum(not a.get('filename') for a in attachments),
                        'link_entries':len(links),'link_domain_observed_entries':sum(bool(l.get('domain')) for l in links),
                        'link_metadata_unavailable_records':sum(r['links'] is None for r in subset),
                        'unsubscribe_metadata_unavailable_records':sum(r['unsubscribe'] is None for r in subset)}
        write_json(report/'feature_availability_details.json',fields)
        # Annotate observed contexts separately from the untouched human decision fields.
        review=list(rows(report/'private_review.jsonl'))
        require(Counter(r['kind'] for r in review)=={'FP':8,'FN':5,'RU_train_review':60},'Review coverage differs')
        observations={
            'd7abdcd3e316':'Торговые предложения напитков; согласие и корректность ham неизвестны.',
            '6daf1c599e23':'Приглашение на семинар; в тексте заявляется связь с регистрацией, но это не проверено.',
            '4f03f70750d1':'Предложение страхования поездки; возможна путаница разрешённой рекламы со spam.',
            '0e51b627f0eb':'Регистрация/проживание на научном workshop; легитимность и согласие не устанавливаются автоматически.',
            'a6147eaa3144':'Гороскоп/рассылка; согласие неизвестно.',
            '1859ed67cf92':'Напоминание подписок; единственный удалённый (SPAM? score) subject-маркер. Правило не устранило FP.',
            'ccf4dea74674':'Игровой дайджест; рассылка не равна нежелательному сообщению без контекста согласия.',
            'e450787008a4':'Предложение проката автомобилей; нет оснований автоматически менять ham-метку.',
            'b04f58dfca73':'Нестандартное разделение букв в An Information; форматная гипотеза, не доказанная причина FN.',
            'da96a567cad9':'CJK, предложение торговой сети; покрытие историческим словарём мало, язык точно не определялся.',
            'a8537d93626d':'CJK, повторяющиеся предложения материалов; языковая область отличается.',
            'bb5a6df54043':'Приглашение на sci-fi convention с пересылкой; source spam не подтверждён человеком.',
            '0d5c048c191f':'Французский фрагмент опроса, виден U+FFFD; источник ошибки декодирования не устанавливался.'}
        for r in review:
            if r['kind'] in {'FP','FN'}:r['codex_observation']=observations[r['record_id'][-12:]]
            else:r['codex_observation']='В контексте есть разговорная реплика/объявление/продвижение; исходная метка не подтверждена человеком. Канал по одной строке неизвестен.'
            require(r['human_decision'] is None and r['reviewer'] is None,'Human review falsely populated')
        dump_rows(report/'private_review.jsonl',review)
        require(Counter((r['spam_label'],r['length_rank_tertile']) for r in review if r['kind']=='RU_train_review')=={(label,bin):10 for label in [0,1] for bin in range(3)},'RU sample length/class quotas')
        selected,_=subset_records(data,config);subset=json.loads((report/'pilot_subset_manifest.json').read_bytes())
        require([r['record_id'] for r in selected]==[r['record_id'] for r in subset['records']],'Subset deterministic selection mismatch')
        pilot=json.loads((report/'decision_pilot.json').read_bytes());checks['decision_status']=pilot['status']
        if pilot['status']=='completed':
            for path,h in pilot['runtime_code_sha256'].items():require(sha_file(ROOT/'src'/path)==h,'Decision runtime code differs')
            predictions=list(rows(report/'decision_predictions.jsonl'));contexts=list(rows(report/'private_decision_visible_fragments.jsonl'))
            require(len(predictions)==len(contexts)==600 and all(p['status']=='ok' for p in predictions),'Decision coverage mismatch')
            require([p['record_id'] for p in predictions]==[r['record_id'] for r in selected],'Decision subset order')
            actual={c['record_id']:c for c in contexts}
            for p,r in zip(predictions,selected):
                require(p['spam_label']==r['spam_label'] and p['source']==r['source'],'Decision source/label mismatch')
                c=actual[r['record_id']];require(hashlib.sha256(c['text'].encode()).hexdigest()==p['actual_fragment_sha256']==c['text_sha256'],'Visible context SHA mismatch')
                if p['mask_literal_replacements']==0:require(clean_text(r,config)[0].startswith(c['text']),'Visible fragment is not original prefix')
                require(p['token_stats']['visible_tokens']<=pilot['runtime']['state_cap'],'Context limit violated')
            for domain in ['spam_email','russian_auxiliary']:
                ps=[p for p in predictions if p['domain']==domain]
                require(metrics([p['spam_label'] for p in ps],[p['spam_score'] for p in ps],.5)==pilot['metrics']['Laya'][domain]['metrics']['0.5'],'Decision confusion mismatch')
                for name,model in models.items():
                    saved=list(rows(report/f'predictions/{name}_{domain}_pilot_same_fragment.jsonl'))
                    texts=[actual[p['record_id']]['text'] for p in saved];score=scores(model,texts)
                    require(np.allclose(score,[p['spam_score'] for p in saved],atol=1e-12,rtol=0),'Matched-context LR score mismatch')
            checks['decision_subset_context_metrics']='passed'
            checks['Laya_score_quantiles']={domain:np.quantile([p['spam_score'] for p in predictions if p['domain']==domain],[0,.1,.5,.9,1]).tolist() for domain in ['spam_email','russian_auxiliary']}
        checks['data_access_log']=dict(boundary.accesses)
    # Real subprocess CLI smoke without modifying the supplied example or model.
    example=ROOT/'data/examples/service_notice.eml';before=sha_file(example)
    completed=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-B',str(ROOT/'src/predict_email_day4.py'),str(example)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True)
    smoke=json.loads(completed.stdout);require(smoke['verdict']=='insufficient_features' and smoke['active_features']==0,'Day4 CLI did not refuse empty vector')
    require(sha_file(example)==before and protected_snapshot(ROOT)==preserved,'Predict changed E0/example')
    write_json(report/'CLI_smoke.json',smoke);checks['CLI_empty_vector_refusal']='passed';checks['status']='passed'
    write_json(report/'verification.json',checks)
    # Additional factual observations from the reviewed train sample, not corpus-wide estimates.
    review_note='''\n## Наблюдения Codex по зафиксированной выборке (не ручная приёмка)\n\nПросмотрены короткие контексты всех60 RU train примеров. Среди ham видны бытовые/технические реплики, игровой диалог, описание вакансии и новостная формулировка; среди spam — предложения подработки, продвижение каналов, просьба о переводе, сексуальное предложение и текст почти из emoji. Есть латинские/греческие подмены букв и фрагмент с украинскими формами. Это наблюдения по выборке, не доли во всём корпусе и не подтверждение согласия. Вакансия встречается и в ham, поэтому тему нельзя автоматически считать spam. Отдельная истинность меток и provenance каждой строки неизвестны.\n\nВсе8 FP/5 FN E0 получили отдельное поле codex_observation; human_decision/reviewer остались null. Обсуждения workshop/игровой дайджест и коммерческие рассылки нельзя автоматически переразметить. Французский U+FFFD и низкое покрытие CJK сохранены как ограничения.\n'''
    data_text=(report/'data_review.md').read_text(encoding='utf-8').split('\n## Наблюдения Codex')[0]+review_note
    write_text(report/'data_review.md',data_text)
    detailed='\n## Имена вложений и домены: полный подсчёт\n\nОтсутствующая структура остаётся unavailable; missing_filename означает существующую запись вложения без имени, а не отсутствие вложения.\n\n'+'\n'.join('- '+k+': '+json.dumps(v,ensure_ascii=False) for k,v in fields.items())+'\n'
    ft=(report/'feature_availability.md').read_text(encoding='utf-8').split('\n## Имена вложений')[0]+detailed;write_text(report/'feature_availability.md',ft)
    tests_log=(ROOT/'reports/day4/setup/all_tests.log').read_text(encoding='utf-8',errors='replace')
    require('Ran 85 tests' in tests_log and tests_log.rstrip().endswith('OK'),'All fixture test execution not confirmed')
    write_json(report/'tests_execution.json',{'status':'passed','tests':85,'new_day4_tests':10,'scope':'synthetic fixtures; local HTTP timeout test; real closed files not opened','log_sha256':sha_file(ROOT/'reports/day4/setup/all_tests.log')})
    # Final negative observations; never tune the prompt/model after reading these.
    synthetic=list(rows(report/'decision_synthetic_predictions.jsonl'))
    synthetic_ham=[r for r in synthetic if r['expected_hypothesis']==0 and r['status']=='ok']
    synthetic_ham_fp=sum(r['spam_score']>=.5 for r in synthetic_ham)
    decision_notes=['','## Независимый разбор зафиксированного пилота','',
        f'Все {synthetic_ham_fp}/{len(synthetic_ham)} synthetic примеров с expected ham дали positive. Это гипотезы авторских фикстур, не оценки реальной популяции. Injection-пары до/после также дали1.0; на фоне общего positive-bias это не подтверждает устойчивость. Длинные парные хвосты были усечены и не видны модели.',
        f"Квантили Laya scores min/p10/p50/p90/max: {checks.get('Laya_score_quantiles')}. Вопрос/критерии/порог по этим наблюдениям не менялись.",
        'Подтверждение того же публичного checkpoint: [официальный каталог Unsloth](https://github.com/unslothai/unsloth/blob/548fe076fa0b6b65417585879da93ae41491aa38/studio/backend/core/systemone/catalog.py) задаёт convaiinnovations/laya, subfolder multilingual; [runtime](https://github.com/unslothai/unsloth/blob/548fe076fa0b6b65417585879da93ae41491aa38/studio/backend/core/systemone/laya_runtime.py) вызывает _load_laya с этим subfolder. Прочитанные исходники сохранены в setup; здесь использован прямой SDK издателя, а не Studio/закрытый Jev.','']
    dt=(report/'decision_pilot.md').read_text(encoding='utf-8').split('\n## Независимый разбор')[0]+'\n'.join(decision_notes)
    write_text(report/'decision_pilot.md',dt)
    cleaning_note='\n## Наблюдаемый эффект единственного изменения\n\nTrain не изменился; fit E1 повторил исходные параметры. Для изменённого ham score E0=.4633603423, E1=.4656879560, E2=.5399290749. E1 не устранил FP при принятом старом пороге и слегка повысил score; это не улучшение. Ни regex, ни состав train не расширялись после этого наблюдения.\n'
    ct=(report/'cleaning_audit.md').read_text(encoding='utf-8').split('\n## Наблюдаемый эффект')[0]+cleaning_note
    write_text(report/'cleaning_audit.md',ct)
    decision_summary=pilot.get('metrics',{}).get('Laya',{})
    status='completed' if pilot['status']=='completed' else 'partial'
    lines=['# Email Guard — DAY4_STATUS','',f"Статус **{status}** в пределах разрешённых исследовательских этапов. Run `{run}`. test_not_evaluated=true.",'',
        '| Этап | Статус | Факт |','|---|---|---|',
        '| Контракт/данные | completed | 4 splits:5953/1275 historical,27953/5990 RU; SHA,группы,exact/canonical тексты проверены; утечек0 |',
        '| Review | completed_pending_human | все8FP/5FN E0,60RU train; отдельные human-поля null; source_labels_used_provisionally |',
        '| E0 | preserved | сохранён по умолчанию с порогом0.3037252359712881; без fit/перезаписи |',
        '| E1 | completed | 5CV+1full fit;1validation ham marker,0train изменений; CV/метрики не улучшились |',
        '| E2 | completed_research | 1fit на33906train; RU F1@.5=.972274; historical F1@.5=.941374 против E0 .946309 |',
        f"| Laya | {pilot['status']} | 600validation,12пар/24synthetic; CUDA RTX4060Ti; прямой SDK offline; не Jev |",
        '| Тесты/reload/аудит/CLI | passed | 85fixture тестов; full-validation reload diff0; independent audit; empty-vector refusal |','',
        'E1 исходные гиперпараметры char_lr;0исключений fit после проверки новых совпадений. E2 не заменяет почтовую модель: одновременно изменены канал/язык/баланс/размер/словарь. RU не называется проверенным почтовым корпусом. Исторический тест, RU test, challenges, резерв, modern кандидаты, Nazario и общий Parquet не читались;2026-09 security_blocked сохраняется.',
        '', f"Laya только common validation subset: { {d: {'F1':s['metrics']['0.5']['f1'],'FP':s['metrics']['0.5']['FP'],'FN':s['metrics']['0.5']['FN'],'coverage':s['coverage']} for d,s in decision_summary.items()} }.",
        'С фиксированным noul-вопросом Laya склонна к spam: RU149/150ham ошибочно positive. Вопрос не подбирался после benchmark, калибровка отсутствует. Это отрицательный feasibility-result, не runtime-блокер. Усечено70/300 historical,0/300RU; matched-context LR сравнение сохранено.',
        f"Warm client latency: {pilot.get('latency_warm_sequential_seconds')}; RAM — sampled worker RSS, VRAM — peak PyTorch allocated/reserved, не вся память драйвера. Cold load отдельно. Ошибок/таймаутов0 при полном coverage; нагрузочный тест не выполнялся.",
        '', 'Day4 inference: по умолчанию E0 и прежний порог; active_features=0→verdict=insufficient_features. Raw score только диагностический, язык unknown не означает доказанную неподдержку. Phishing/category=not_available. Никаких действий с почтой. Исходный CLI дня3 сохранён.',
        'Версии численных библиотек основной .venv совпали с E0; requirements и baseline модели/код проверены SHA. Отдельная .venv-decision содержит torch2.10.0+cu130/laya0.3.20. Публичная revision Laya и Apache2.0 зафиксированы runtime_manifest. README RU: CC-BY-NC-4.0. Ничего не публиковалось и тексты не уходили в облачный inference.',
        '', 'Начальные устранённые ошибки окружения/API записаны в setup: ensurepip sandbox, loopback sandbox, load(compile=...) mismatch; успешные запуски приведены отдельно. Текущих технических блокеров нет. Ручная приёмка меток/данных и выбор цели дня5 остаются решениями человека. Qwen training/веса сегодня не запускались/не скачивались.',
        '', '## Повтор из корня проекта','', '```powershell',
        '.\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v',
        '.\\.venv\\Scripts\\python.exe -B .\\src\\run_day4.py',
        '.\\.venv\\Scripts\\python.exe -B .\\src\\run_decision_pilot.py',
        '.\\.venv\\Scripts\\python.exe -B .\\src\\verify_day4.py',
        '.\\.venv\\Scripts\\python.exe -B .\\src\\predict_email_day4.py .\\data\\examples\\service_notice.eml','```','',
        'Установка изолированного runtime и версии — docs/DAY4_REPRODUCTION.md. Новый classical-run получает новыйrun_id; E0 не меняется. Общую подготовку/аудит дня2 не запускать. Для просмотра передать только этот статус, experiments.md,data_review.md,decision_pilot.md,review_questions.md; модели/корпуса/private_review не нужны.']
    write_text(ROOT/'reports/DAY4_STATUS.md','\n'.join(lines)+'\n');write_text(report/'DAY4_STATUS.md','\n'.join(lines)+'\n')
    for filename in ['verification.json','data_review.md','private_review.jsonl','feature_availability.md','feature_availability_details.json','CLI_smoke.json','tests_execution.json','decision_pilot.md','cleaning_audit.md']:
        (ROOT/'reports/day4'/filename).write_bytes((report/filename).read_bytes())
    for path in (ROOT/'reports/day4/setup').iterdir():
        if path.is_file():
            target=report/'setup'/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    pn=(report/'PRESENTATION_NOTES_DAY4.md').read_text(encoding='utf-8').split('\n## Decision: фактический результат')[0]
    pn+='\n## Decision: фактический результат\n\nLaya CUDA:600/600ответов, errors/timeouts0; historical subset F1=.764398 FP45/FN0, RU subset F1=.668151 FP149/FN0. Warm p50=.01784с,p95=.02463с, cold model load10.00с. Условие fixed noul+threshold.5 не менялось после benchmark. Это отрицательный результат пригодности, не обучение Jev.\n\nСравнение LR на тех же видимых префиксах и полных subset сохранено;70historical текстов усечены. Все10обычных synthetic ham дали positive; injection-пары с1.0до/после не доказывают защиту. Test остаётся закрыт. Default E0 сохранён.\n'
    write_text(report/'PRESENTATION_NOTES_DAY4.md',pn);write_text(ROOT/'reports/day4/PRESENTATION_NOTES_DAY4.md',pn)
    print(json.dumps(checks,ensure_ascii=False),flush=True)
    return checks


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
