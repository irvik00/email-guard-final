"""Independent saved-model/OOF/validation audit with the day-3 data boundary."""
import json
import math
from pathlib import Path

import joblib
import numpy as np

from baseline_common import (ROOT,DataBoundary,load_inputs,sha_file,require,code_fingerprints,
    scores,metrics,threshold_modes,select_baseline,explain,json_params)
from pipeline_common import write_json,write_text,rows


def review_notes(report_root,model_manifest,experiments,thresholds):
    """Additional analysis of existing validation evidence, never another model fit."""
    evidence=json.loads((report_root/'error_analysis.json').read_text(encoding='utf-8'))
    examples=evidence['examples'];fp=[e for e in examples if e['kind']=='FP'];fn=[e for e in examples if e['kind']=='FN']
    cjk=[e for e in fn if any('\u4e00'<=c<='\u9fff' for c in e['fragment'])]
    damaged=[e for e in fn if '\ufffd' in e['fragment']]
    marker=[e for e in fp if '(SPAM?' in e['fragment']]
    default=thresholds['modes']['default']['metrics'];provisional=thresholds['modes']['conservative' if not thresholds['unmet_fp_scenario'] else 'default']
    after=provisional['metrics'];upper=provisional['fpr_wilson95']['interval'][1]
    lines=['## Дополнительный разбор зафиксированных результатов', '',
        f"Run `{model_manifest['run_id']}`. Только уже сохранённые validation-фрагменты и измерения; модель, признаки, метки и порог после просмотра не менялись.", '',
        f"Порог {thresholds['provisional_threshold']:.9g} ниже исходного0.5: FP {default['FP']} → {after['FP']}, FN {default['FN']} → {after['FN']}. Название conservative означает ограничение FPR, а не обязательное увеличение порога. Максимальный F1 и constrained F1 в этом запуске совпали.", '',
        f"Общий FPR={after['FP']}/{after['support']['0']}={after['fpr']:.3%}; Wilson95 upper={upper:.3%}. Наблюдаемые <=1% не доказывают истинный FPR<=1%.", '',
        'По источникам provisional: '+ '; '.join(f"{s}: FP={m['FP']}/{m['support']['0']}, FPR={m['fpr']:.3%}" for s,m in provisional['by_source'].items())+
        '. Ограничение задано для общего validation; оно не является отдельным ограничением каждого источника.', '',
        f"Показано {len(fp)} из {evidence['counts']['FP']} FP. В выбранных ham-фрагментах встречаются семинар, торговые предложения, гороскоп, страхование и напоминание подписки. Это совместимо с гипотезой путаницы легитимных рассылок с маркетинговым спамом; право на рассылку/истинность метки по фрагменту не проверяется.", '',
        'Глобальные char-веса: фрагменты you/your/clic повышают оценку spam; lingu/ngui/uist, цитирование и2002 понижают её. Тематика лингвистического корпуса и историческая дата — возможные подсказки источника. Частичное совпадение признака в отдельной ошибке не доказывает её причину.', '']
    for e in cjk:
        lines += [f"- FN `{e['record_id']}`: CJK в коротком фрагменте, покрытие вхождений char-анализатора словарём={e['analyzed_token_occurrence_coverage']:.3%}, score={e['spam_score']:.6f}. В одном случае покрытие очень низкое; это наблюдение, не оценка качества для конкретного языка."]
    for e in damaged:
        lines += [f"- FN `{e['record_id']}`: виден U+FFFD в фрагменте на французском; возможен шум декодирования/языковой области. Корпус не перекодировался и причина ошибки не доказана."]
    for e in marker:
        lines += [f"- FP `{e['record_id']}`: в исходном model_text виден маркер темы `(SPAM? 08.00)`. Это возможный артефакт прежнего фильтра, а не добавленные runner заголовки. Узкий builder дня2 удаляет перечисленные filter-header строки, а не произвольные маркеры тем. Benchmark сегодня не менялся; отдельная версия очистки и проверка таких артефактов — гипотеза для дня4."]
    smoke=json.loads((report_root/'cli_smoke.json').read_text(encoding='utf-8'))
    lines += ['',f"CLI smoke на неизменённом учебном .eml: active_features={smoke['active_features']}, score={smoke['spam_score']:.9g}, verdict={smoke['spam_verdict']}. В примере присутствует кириллица и получен empty_vector; оценка задаётся intercept/prior, это не подтверждение поддержки русского. Предупреждения сохранены, входной пример под желаемый ответ не переделывался.", '',
        'До следующего эксперимента уточнить допустимость FP на легитимных рассылках, влияние меток/корпусной тематики/маркеров и языковой области. Возможные изменения оформлять новым вариантом; к закрытому test сегодня не обращаться.', '']
    notes='\n'.join(lines)
    run_dir=report_root/'runs'/model_manifest['run_id']
    for directory in [report_root,run_dir]:
        write_text(directory/'validation_review_notes.md',notes)
        original=(directory/'error_analysis.md').read_text(encoding='utf-8').split('\n## Дополнительный разбор зафиксированных результатов\n',1)[0]
        write_text(directory/'error_analysis.md',original+'\n'+notes)
    return {'scope':'stored validation evidence only','models_or_thresholds_changed':False,
        'CJK_FN_examples':len(cjk),'replacement_character_FN_examples':len(damaged),'possible_old_filter_subject_marker_FP_examples':len(marker),
        'notes_sha256':sha_file(report_root/'validation_review_notes.md')}


def run():
    report_root=ROOT/'reports/day3';model_root=ROOT/'models/day3'
    manifest=json.loads((model_root/'model_manifest.json').read_text(encoding='utf-8'))
    experiments=json.loads((report_root/'experiments.json').read_text(encoding='utf-8'))
    thresholds=json.loads((report_root/'thresholds.json').read_text(encoding='utf-8'))
    require(manifest['status']=='completed' and manifest['run_id']==experiments['run_id']==thresholds['run_id'],'Latest run mismatch')
    require(manifest['code_sha256']==code_fingerprints(),'Experiment code differs from the saved run')
    tests=json.loads((report_root/'tests_execution.json').read_text(encoding='utf-8'))
    require(tests['status']=='passed' and tests['code_sha256']==manifest['code_sha256'],'Tests are missing or stale')
    with DataBoundary(ROOT) as boundary:
        data,folds,contract=load_inputs(ROOT,manifest['config'])
        require(contract['inputs']==manifest['input_fingerprints'] and contract['manifest_sha256']==manifest['preparation_manifest_sha256'],'Experiment input contract differs')
        require(contract['cv_fold_mapping']==experiments['contract']['cv_fold_mapping'],'CV mapping changed')
        before={}
        for name,spec in manifest['artifacts'].items():
            for path in [model_root/name,ROOT/spec['path']]:
                require(path.resolve().is_relative_to(model_root.resolve()),'Model path traversal')
                require(sha_file(path)==spec['sha256'] and path.stat().st_size==spec['size_bytes'],'Model artifact changed '+name)
                before[str(path)]=spec['sha256']
        observed={};y=np.array([r.label for r in data['train']]);yv=np.array([r.label for r in data['validation']])
        texts=[r.text for r in data['validation']]
        for name,spec in experiments['models'].items():
            # These are our own manifest-verified local Pipeline artifacts.
            model=joblib.load(model_root/f'{name}.joblib')
            require(json_params(model.get_params(deep=True))==manifest['parameters'][name],'Saved parameters mismatch')
            require(list(model.classes_)==[0,1],'Class direction changed')
            features=json.loads((model_root/f'{name}_feature_names.json').read_text(encoding='utf-8'))
            require(features==list(model['tfidf'].get_feature_names_out()),'Feature names/vocabulary mismatch')
            rows_oof=list(rows(report_root/f'predictions/{name}_oof.jsonl'))
            require(len(rows_oof)==len(data['train']),'OOF count mismatch')
            for record,expected,fold in zip(rows_oof,data['train'],folds):
                require(record['record_id']==expected.record_id and record['duplicate_group_id']==expected.group_id and record['source']==expected.source and record['spam_label']==expected.label and record['fold_id']==int(fold),'OOF metadata/fold mismatch')
            oof=np.array([r['spam_score'] for r in rows_oof]);require(np.all(np.isfinite(oof)) and np.all((oof>=0)&(oof<=1)),'Invalid OOF scores')
            require(metrics(y,oof,.5)==spec['cv']['pooled_oof'],'Pooled OOF metrics mismatch')
            for f in spec['cv']['folds']:
                mask=folds==f['fold_id'];require(metrics(y[mask],oof[mask],.5)==f['metrics'],'Fold metrics mismatch')
            val_rows=list(rows(report_root/f'predictions/{name}_validation.jsonl'))
            require(len(val_rows)==len(data['validation']),'Validation prediction count')
            for record,expected in zip(val_rows,data['validation']):
                require(record['record_id']==expected.record_id and record['source']==expected.source and record['spam_label']==expected.label and record['duplicate_group_id']==expected.group_id,'Validation prediction metadata')
            saved=np.array([r['spam_score'] for r in val_rows]);loaded=scores(model,texts)
            require(np.allclose(saved,loaded,atol=1e-12,rtol=0),'Full validation reload score mismatch')
            require(metrics(yv,saved,.5)==spec['validation_default'],'Validation metrics mismatch')
            for source,m in spec['validation_by_source'].items():
                mask=np.array([r.source==source for r in data['validation']]);require(metrics(yv[mask],saved[mask],.5)==m,'Source metrics mismatch')
            for text in texts[:3]:explain(model,text)
            observed[name]={'validation_max_reload_error':float(np.max(np.abs(saved-loaded))),'OOF_rows':len(oof),'validation_rows':len(saved),'features':len(features)}
        require(select_baseline(experiments['models'],manifest['config'])==manifest['selected_baseline'],'Baseline selection changed')
        selected=manifest['selected_baseline']
        s=np.array([r['spam_score'] for r in rows(report_root/f'predictions/{selected}_validation.jsonl')])
        recomputed=threshold_modes(yv,s,manifest['config'])
        for key in ['threshold_default','threshold_f1','threshold_conservative','provisional_threshold','provisional_policy','grid_size','unmet_fp_scenario']:
            require(recomputed[key]==thresholds[key],'Threshold selection mismatch '+key)
        for mode,record in recomputed['modes'].items():
            for key,value in record.items():require(value==thresholds['modes'][mode][key],'Threshold metrics/Wilson mismatch')
        bundle=joblib.load(model_root/'baseline_bundle.joblib')
        require(bundle['model_id']==selected and bundle['thresholds']['provisional_threshold']==thresholds['provisional_threshold'],'Bundle policy mismatch')
        require(np.allclose(scores(bundle['pipeline'],texts),s,atol=1e-12,rtol=0),'Bundle validation mismatch')
        for path,expected in before.items():require(sha_file(path)==expected,'Verification modified a model artifact')
        result={'status':'passed','run_id':manifest['run_id'],'tests_passed':tests['tests_run'],'models':observed,
            'OOF_mapping_and_metrics':'passed','threshold_grid_and_Wilson':'passed','bundle_reload':'passed',
            'artifact_hashes_before_after_unchanged':True,'data_access_log':dict(boundary.accesses),'test_not_evaluated':True,'fit_performed':False}
    result['validation_review']=review_notes(report_root,manifest,experiments,thresholds)
    write_json(report_root/'verification.json',result)
    write_json(report_root/'runs'/manifest['run_id']/'verification.json',result)
    status_path=ROOT/'reports/DAY3_STATUS.md'
    status=status_path.read_text(encoding='utf-8').split('\n## Независимая проверка сохранённых результатов\n',1)[0]
    status+='\n## Независимая проверка сохранённых результатов\n\nverify_baselines.py: passed. SHA latest/архивных моделей, словари, параметры, OOF/fold/source метаданные и confusion, validation-оценки всех трёх загруженных Pipeline, выбранный baseline, сетка порогов и Wilson сверены. Модели/входы не менялись, fit не выполнялся, закрытые данные не читались. Дополнительные наблюдения по зафиксированным ошибкам добавлены в error_analysis.md.\n\nПовтор: `.\\.venv\\Scripts\\python.exe -B .\\src\\verify_baselines.py`. Результат reports/day3/verification.json.\n'
    write_text(status_path,status)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':run()
