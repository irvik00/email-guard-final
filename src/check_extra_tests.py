"""Focused new invariants plus existing Day-5/diagnostics regression fixtures."""
import os
import re
import subprocess
from extra_common import *

def main():
    out,_=paths();tmp=ROOT/'.cache/extra/test_tmp';tmp.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(TEMP=str(tmp),TMP=str(tmp),PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    runs=[]
    for venv,pattern in [('.venv','test_extra_experiments.py'),('.venv-decision','test_extra_laya.py'),
                         ('.venv','test_day5.py'),('.venv','test_laya_diagnostics.py')]:
        command=[str(ROOT/venv/'Scripts/python.exe'),'-B','-m','unittest','discover','-s','tests','-p',pattern,'-v']
        began=now();process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        log=process.stdout+process.stderr;(out/f'tests_{pattern}.log').write_text(log,encoding='utf-8')
        count=re.search(r'Ran (\d+) test',log)
        runs.append({'command':command,'started_utc':began,'finished_utc':now(),'returncode':process.returncode,
            'tests':int(count.group(1)) if count else None,'source_sha256':sha_file(ROOT/'tests'/pattern)})
    write_json(out/'tests_execution.json',{'status':'passed' if all(r['returncode']==0 for r in runs) else 'failed',
        'runs':runs,'total_tests':sum(r['tests'] or 0 for r in runs)})
    require(all(r['returncode']==0 for r in runs),'Focused tests failed; see per-suite logs')
    print('Passed',sum(r['tests'] for r in runs),'focused/regression tests')

if __name__=='__main__':main()
