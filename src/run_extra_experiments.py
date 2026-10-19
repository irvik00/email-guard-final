"""Sequential real extra experiments. New run by default; resume skips completed stages."""
import argparse
import os
import shutil
import subprocess
import traceback
from datetime import datetime,timezone
from extra_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');args=p.parse_args()
    if not args.resume:prepare()
    out,models=paths();env=os.environ.copy();tmp=ROOT/'.cache/extra/tmp';tmp.mkdir(parents=True,exist_ok=True)
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',TEMP=str(tmp),TMP=str(tmp))
    statuses=[]
    def run(name,venv,script,arguments,artifact,limit=5400):
        if (out/artifact).exists():
            statuses.append({'stage':name,'status':'reused_completed','artifact':artifact});return True
        argv=[str(ROOT/venv/'Scripts/python.exe'),'-B',str(ROOT/'src'/script)]+arguments
        log=out/(name+'_driver.log');start=now()
        if venv=='.venv-decision':
            frozen=datetime.fromisoformat(read_json(out/'protocol.json')['frozen_utc'])
            remaining=14400-(datetime.now(timezone.utc)-frozen).total_seconds()
            if remaining<=0:
                statuses.append({'stage':name,'status':'partial_time_budget','reason':'4-hour GPU experiment budget exhausted; CPU stages continue'})
                write_json(out/'driver_status.json',{'stages':statuses});return False
            limit=min(limit,remaining)
        try:
            with log.open('w',encoding='utf-8') as stream:
                process=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=limit)
            require(process.returncode==0 and (out/artifact).exists(),f'{name}: exit{process.returncode}, expected artifact absent')
            statuses.append({'stage':name,'status':'completed','started_utc':start,'finished_utc':now(),'command':argv})
            okay=True
        except Exception as error:
            statuses.append({'stage':name,'status':'blocked','started_utc':start,'finished_utc':now(),
                'error':type(error).__name__+': '+str(error),'command':argv,'log':str(log.relative_to(ROOT))})
            okay=False
        write_json(out/'driver_status.json',{'stages':statuses});return okay
    # Exactly the requested order, GPUs sequential. A and B each guard against a second fit.
    a=run('A','.venv-decision','train_extra_laya.py',['--branch','A'],'A/candidate_manifest.json',4200)
    if a:
        if not (out/'A/training_source.py.txt').exists():
            shutil.copyfile(ROOT/'src/train_extra_laya.py',out/'A/training_source.py.txt')
        run('A_context','.venv-decision','extra_context.py',[],'context_control.json',900)
    b=run('B','.venv-decision','train_extra_laya.py',['--branch','B'],'B/candidate_manifest.json',4200)
    if b:shutil.copyfile(ROOT/'src/train_extra_laya.py',out/'B/training_source.py.txt')
    setup=run('C_download','.venv','setup_extra_public.py',['e5'],'e5_source.json',1800)
    c=False
    if setup:
        emb=run('C_embeddings','.venv-decision','extra_e5_embeddings.py',[],'C/embeddings.json',3600)
        if emb:c=run('C_fit','.venv','fit_extra_e5.py',[],'C/candidate_manifest.json',900)
    if a and c:run('C_matched','.venv-decision','extra_context.py',['--e5-matched'],'e5_matched_control.json',900)
    psl=run('D_psl','.venv','setup_extra_public.py',['psl'],'psl_ready.json',180)
    # PSL setup uses a versioned config snapshot, so emit a local ready marker explicitly.
    if (ROOT/'config/registry/psl_manifest.json').exists():
        write_json(out/'psl_ready.json',read_json(ROOT/'config/registry/psl_manifest.json'));psl=True
    if psl:run('D_demo','.venv','demo_sender_evidence.py',[],'sender_demo_summary.json',120)
    d=run('D_fit','.venv','extra_metadata.py',[],'D/candidate_manifest.json',1800)
    if a and b and c:
        done=run('outer_Laya','.venv-decision','evaluate_extra.py',[],'new_laya_resources.json',1800)
        if done:
            ev=run('outer_E5_embed','.venv-decision','extra_e5_embeddings.py',['--validation'],'C/validation_embeddings.json',1800)
            if ev:run('outer_E5_LR','.venv','fit_extra_e5.py',['--validation'],'C/validation_predictions.jsonl',120)
            if d:run('outer_D','.venv','extra_metadata.py',['--validation'],'D/comparison.json',600)
    else:
        write_json(out/'comparison_blocker.json',{'reason':'At least one A/B/C manifest unavailable; other training/evidence branches were attempted',
            'candidates_available':{'A':a,'B':b,'C':c},'no_mock_metrics':True})
    if a and b:run('reload_AB','.venv-decision','reload_extra_laya.py',[],'independent_reload.json',600)
    if c:
        er=run('reload_C_encoder','.venv-decision','extra_e5_embeddings.py',['--reload'],'C/independent_encoder_reload.json',600)
        if er:run('reload_C_full','.venv','fit_extra_e5.py',['--reload'],'C/independent_full_reload.json',120)
        run('reload_C_tokenizer','.venv-decision','check_extra_tokenizer.py',[],'C/tokenizer_reload.json',180)
        run('C_readout_profile','.venv','profile_extra_lr.py',[],'C/warm_lr_readout.json',120)
        run('C_end_to_end_profile','.venv-decision','profile_extra_e5.py',[],'C/warm_end_to_end.json',600)
    run('tests','.venv','check_extra_tests.py',[],'tests_execution.json',300)
    if a and b and c and d:
        audited=run('audit','.venv-decision','verify_extra.py',[],'verification.json',600)
        if audited:
            run('reports','.venv','report_extra.py',[],'EXTRA_EXPERIMENTS_STATUS.md',300)
            run('finalize','.venv','finalize_extra.py',[],'artifact_checks.json',300)
    if not (out/'EXTRA_EXPERIMENTS_STATUS.md').exists():
        report='# EXTRA_EXPERIMENTS_STATUS: partial\n\nRun '+out.name+'. E0 remains default.\n\n'
        for stage in statuses:report+='- '+stage['stage']+': '+stage['status']+'; '+stage.get('error','')+'\n'
        report+='\nNo mock metrics. Completed checkpoints/logs preserved. Closed tests not opened.\n'
        (ROOT/'reports/EXTRA_EXPERIMENTS_STATUS.md').write_text(report,encoding='utf-8')
    print('Run:',out.name)

if __name__=='__main__':main()
