"""Exactly one E5-embedding LR fit, scalar calibration and inner-dev operating point."""
import argparse
import shutil
import time
import warnings
import importlib.metadata
from extra_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',action='store_true');p.add_argument('--reload',action='store_true');args=p.parse_args()
    out,models=paths();boundary=ExtraBoundary('train').install()
    import numpy as np
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import psutil
    models=models/'C';models.mkdir(parents=True,exist_ok=True)
    if args.reload:
        manifest=read_json(out/'C/candidate_manifest.json');probe=read_json(out/'C/independent_encoder_reload.json')
        classifier=joblib.load(models/'classifier.joblib')
        fresh=np.load(out/'C/reload_probe_embeddings.npy',allow_pickle=False)
        before=read_rows(out/'C/dev_raw.jsonl')
        old=np.array([before[i]['logit_difference'] for i in probe['dev_indices']])
        new=classifier.decision_function(fresh);error=float(np.max(np.abs(new-old)))
        require(error<=1e-4,'C full encoder/LR reload changed logits')
        require(bool(np.all((new/manifest['temperature']>=math.log(manifest['threshold']/(1-manifest['threshold'])))==
                            (old/manifest['temperature']>=math.log(manifest['threshold']/(1-manifest['threshold']))))),'C reload changed operating decisions')
        write_json(out/'C/independent_full_reload.json',{'status':'passed','samples':len(old),'max_logit_abs_error':error,
            'threshold_decisions_identical':True,'fit_on_reload':False,'accesses':boundary.report()});print('C full reload passed');return
    if args.validation:
        manifest=read_json(out/'C/candidate_manifest.json');spec=read_json(out/'C/validation_embeddings.json')
        require(sha_file(models/'classifier.joblib')==manifest['classifier_sha256'],'C classifier changed')
        classifier=joblib.load(models/'classifier.joblib')
        rows=read_rows(out/'C/private_validation_inputs.jsonl');matrix=np.load(ROOT/spec['parts']['validation']['matrix_path'],allow_pickle=False)
        logits=classifier.decision_function(matrix).tolist()
        write_rows(out/'C/validation_predictions.jsonl',[{k:r[k] for k in ['record_id','domain','source','spam_label','duplicate_group_id']}|
            {'logit_difference':float(z),'status':'ok','real_device':'CUDA frozen encoder + CPU LR'} for r,z in zip(rows,logits)])
        write_json(out/'C/validation_accesses.json',boundary.report());return
    require(not (out/'C/training_started.json').exists(),'One C fit per run')
    write_json(out/'C/training_started.json',{'started_utc':now(),'protocol_sha256':sha_file(out/'protocol.json'),
        'code_sha256':{n:sha_file(ROOT/n) for n in ['src/fit_extra_e5.py','src/extra_e5_embeddings.py']},'validation_not_opened':True})
    spec=read_json(out/'C/embeddings.json');inputs={};matrices={}
    for part in ['fit','dev','calibration']:
        inputs[part]=read_rows(out/f'C/private_{part}_inputs.jsonl');entry=spec['parts'][part]
        require(entry['input_sha256']==sha_file(out/f'C/private_{part}_inputs.jsonl'),'C membership changed')
        require(entry['sha256']==sha_file(ROOT/entry['matrix_path']),'Embedding cache changed')
        matrices[part]=np.load(ROOT/entry['matrix_path'],allow_pickle=False)
    classifier=LogisticRegression(C=1.,class_weight=None,random_state=42,max_iter=2000)
    began=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');classifier.fit(matrices['fit'],[r['spam_label'] for r in inputs['fit']])
    require(list(classifier.classes_)==[0,1],'LR class direction')
    fit_seconds=time.perf_counter()-began
    raw={}
    for part in ['dev','calibration']:
        logits=classifier.decision_function(matrices[part])
        raw[part]=[{k:r[k] for k in ['record_id','domain','source','spam_label','duplicate_group_id']}|
                    {'logit_difference':float(z),'status':'ok'} for r,z in zip(inputs[part],logits)]
        write_rows(out/f'C/{part}_raw.jsonl',raw[part])
    fitted=fit_temperature([r['spam_label'] for r in raw['calibration']],[r['logit_difference'] for r in raw['calibration']],.5,5.)
    temperature=fitted['applied_temperature'];threshold=choose_threshold(raw['dev'],temperature)
    joblib.dump(classifier,models/'classifier.joblib')
    loaded=joblib.load(models/'classifier.joblib')
    err=float(np.max(np.abs(loaded.decision_function(matrices['dev'])-classifier.decision_function(matrices['dev']))))
    require(err<=1e-12,'C classifier reload mismatch')
    source=spec['model'];shutil.copytree(Path(source['local_path']),models/'tokenizer',
        ignore=shutil.ignore_patterns('model.safetensors','pytorch_model.bin','README.md'))
    manifest={'frozen_utc':now(),'experiment':'C','status':'completed','source':source,'dimension':spec['dimension'],
        'encoder_reference':source['local_path'],'encoder_frozen':True,'tokenizer_path':str((models/'tokenizer').relative_to(ROOT).as_posix()),
        'classifier_path':str((models/'classifier.joblib').relative_to(ROOT).as_posix()),'classifier_sha256':sha_file(models/'classifier.joblib'),
        'builder':'clean_builder_v2','prefix':'query: ','max_tokens':512,'pooling':'attention-mask mean then L2',
        'temperature':temperature,'threshold':threshold['threshold'],'temperature_range':[.5,5.],
        'calibration':fitted,'threshold_selection':threshold,'fit_rows':6000,'fit_iterations':classifier.n_iter_.tolist(),
        'fit_seconds':fit_seconds,'rss_bytes':psutil.Process().memory_info().rss,'warnings':[str(w.message) for w in caught],
        'membership_sha256':read_json(out/'C/data_manifest.json')['membership_sha256'],
        'input_sha256':{p:sha_file(out/f'C/private_{p}_inputs.jsonl') for p in inputs},
        'versions':{n:importlib.metadata.version(n) for n in ['scikit-learn','numpy','scipy','joblib']},
        'reload':{'max_logit_error':err,'rows':600,'fit_on_reload':False},'not_default':True,'validation_not_opened':True}
    write_json(models/'candidate_manifest.json',manifest);write_json(out/'C/candidate_manifest.json',manifest)
    write_json(out/'C/train_accesses.json',boundary.report())
    print({'C':'frozen','T':temperature,'threshold':threshold['threshold'],'iterations':classifier.n_iter_.tolist()})

if __name__=='__main__':main()
