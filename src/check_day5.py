"""Run actual focused synthetic tests and persist execution evidence; no training."""
import io
import sys
import time
import unittest

from day5_common import ROOT,read_json,write_json,sha_file,now


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    out=ROOT/'reports/day5/runs'/read_json(ROOT/'reports/day5/current_run.json')['run_id']
    suite=unittest.TestSuite()
    for pattern in ['test_day5.py','test_laya_diagnostics.py']:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
    stream=io.StringIO();started=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    log=stream.getvalue();(out/'tests_execution.log').write_text(log,encoding='utf-8')
    payload={'executed_utc':now(),'returncode':0 if result.wasSuccessful() else 1,'tests_run':result.testsRun,
             'failures':len(result.failures),'errors':len(result.errors),'seconds':time.perf_counter()-started,
             'scope':'synthetic fixtures only; no corpus test or day2 audit; real gradient/reload checks separately recorded',
             'test_code_sha256':{p:sha_file(ROOT/p) for p in ['tests/test_day5.py','tests/test_laya_diagnostics.py']}}
    write_json(out/'tests_execution.json',payload);print(log);print(payload)
    raise SystemExit(payload['returncode'])


if __name__=='__main__':main()
