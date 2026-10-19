"""Extra experiments: immutable Day-5 membership and strictly local phase boundaries."""
import copy
import os
import sys
from pathlib import Path
from day5_common import *

DAY5 = ROOT/'reports/day5/runs/20261012T083529Z-0950e3dc34'
BASE = ROOT/'reports/extra_experiments'

def paths(run_id=None, branch=None):
    run_id = run_id or read_json(BASE/'current_run.json')['run_id']
    out = BASE/'runs'/run_id
    model = ROOT/'models/extra_experiments'/run_id
    return (out/branch, model/branch) if branch else (out, model)

class ExtraBoundary(Boundary):
    def check(self, event, args):
        super().check(event, args)
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            mode = str(args[1] or '')
            flags = int(args[2] or 0) if len(args)>2 else 0
            protected = any(p.is_relative_to(ROOT/k) for k in ['reports/day3','reports/day4','reports/day5',
                'models/day3','models/day4','models/day5','reports/laya_diagnostics'])
            if protected and (any(c in mode for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR)):
                raise PermissionError('Prior experiment is immutable: '+str(p))
            if p.is_relative_to(BASE) or p.is_relative_to(ROOT/'models/extra_experiments'):
                self.accesses[str(p.relative_to(ROOT).as_posix())+':'+mode] += 1

def protected_snapshot():
    files = set()
    for folder in ['reports/day3','reports/day4','reports/day5','reports/laya_diagnostics',
                   'models/day3','models/day4','models/day5','src','config']:
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(ROOT.glob('requirements*.txt'))
    files.update(ROOT.glob('reports/DAY[345]_STATUS.md'))
    files.add(ROOT/'reports/laya_diagnostics.md')
    for env in ['.venv','.venv-decision']:
        files.add(ROOT/env/'pyvenv.cfg')
    return {p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted(files) if p.exists()}

def prepare():
    import shutil
    from datetime import datetime, timezone
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+digest(now())[:8]
    out,models=paths(run_id);out.mkdir(parents=True);models.mkdir(parents=True)
    config=read_json(ROOT/'config/day5_laya.json')
    config['training'].update(max_steps=1000,warmup_steps=50,dev_steps=[100,300,600,1000],
        train_seconds_limit=3600,order='epoch e: shuffle seed42+e; first epoch identical to day5; wrap batches across epochs')
    contract={'run_id':run_id,'frozen_utc':now(),'day5_run':DAY5.name,'seed':42,
        'A':copy.deepcopy(config),'B':copy.deepcopy(config),
        'C':{'model_id':'intfloat/multilingual-e5-small','prefix':'query: ','max_length':512,
            'pooling':'attention-mask mean, L2 normalize','lr':{'C':1.,'penalty':'l2','class_weight':None,'random_state':42,'max_iter':2000},
            'temperature_range':[.5,5.],'threshold_rule':config['threshold']},
        'context':{'candidate':'A selected weights','cap':1500,'context':2048,'selection':False},
        'D':{'cohort':'SpamAssassin only, existing train/validation, both labels',
            'text':'clean_builder_v2 char TF-IDF (3,5), min_df2,max_df1,sublinear_tf,L2,max_features100000',
            'classifier':'LR C1 L2 class_weightNone seed42 max_iter2000 threshold0.5',
            'numeric':['sender_domain_length','sender_local_digit_ratio','link_count','link_label_mismatch_count',
                       'attachment_count','unsubscribe_header_present','unsubscribe_body_keyword','html_present'],
            'missing':'NaN + explicit indicators; train-only median imputation/scaling; never source/path/filter/auth/brand/history'},
        'validation':'single pass after A/B/C manifests frozen; reuse Day5 E0/E2/Laya predictions with identity checks',
        'closed':'all test/challenge/teacher/reserve/Nazario/modern/fullParquet',
        'default':'E0 unchanged','no_synthetic_augmentation':True,'budget_total_seconds':14400}
    contract['B']['training']['encoder_learning_rate']=1e-5
    contract['B']['training']['unfreeze']='two final actual encoder blocks discovered from installed model; all other encoder tensors and act_head frozen'
    write_json(out/'protocol.json',contract)
    dm=read_json(DAY5/'data_manifest.json')
    for branch in ['A','B','C']:
        dest=out/branch;dest.mkdir()
        write_json(dest/'experiment_contract.json',{'config':contract.get(branch),'frozen_utc':now()})
        write_json(dest/'data_manifest.json',dm)
        for part in ['fit','dev','calibration']:
            src=DAY5/f'private_{part}_inputs.jsonl'
            require(sha_file(src)==dm['selection_sha256'][part],'Day5 selected inputs changed')
            shutil.copyfile(src,dest/src.name)
    write_json(out/'preservation_before.json',protected_snapshot())
    write_json(BASE/'current_run.json',{'run_id':run_id,'stage':'protocol_frozen'})
    print(run_id)

if __name__=='__main__':prepare()
