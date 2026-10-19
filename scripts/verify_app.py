"""App verification using synthetic .eml and the already reviewed 22 development inputs."""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.contract import MAX_BYTES, compose, sha_file

DEV = ROOT/'reports/personal_email_evaluation/runs/20261017T120741Z-a38c25e1'


def boundary():
    allowed = {DEV/'private_inputs.jsonl', DEV/'predictions.jsonl'}
    opened = set()
    def check(event,args):
        if event != 'open' or not isinstance(args[0],(str,bytes,Path)):
            return
        path=Path(args[0]).resolve()
        if path.is_relative_to(ROOT/'data'):
            raise PermissionError('No raw/split data allowed during app verification')
        if path.is_relative_to(ROOT/'reports'):
            rel=path.relative_to(ROOT).as_posix()
            if not (path in allowed or path.is_relative_to(ROOT/'reports/app') or
                    rel=='reports/final_evaluation/runs/20261017T132144Z-d663811e/FINAL_EVALUATION_PROTOCOL.json'):
                raise PermissionError('Unapproved report content')
            opened.add(rel)
    sys.addaudithook(check)
    return opened


def new_out(label):
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+label
    out=ROOT/'reports/app/verification_runs'/run
    out.mkdir(parents=True,exist_ok=False)
    return out


def write(out,name,value):
    (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def wait_ready(engine):
    start=time.monotonic()
    while engine.laya.state=='initializing' and time.monotonic()-start<185:
        time.sleep(.5)
    if not engine.readiness()['modes']['recall']:
        raise RuntimeError('Neural startup failed: '+json.dumps(engine.readiness()))


def components(out):
    from app.inference import Engine
    inputs=[json.loads(s) for s in (DEV/'private_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    old={(r['record_id'],r['model_id']):r for r in map(json.loads,(DEV/'predictions.jsonl').read_text(encoding='utf-8').splitlines()) if r['model_id'] in ('E0','Laya_day5')}
    engine=Engine(); comparison=[]; modes={m:[] for m in ('baseline','recall','review')}
    try:
        wait_ready(engine)
        for row in inputs:
            result=engine.predict_texts(row['original_model_text'],row['text'],'review')
            for name,current in result['components'].items():
                reference=old[(row['record_id'],name)]
                delta=abs(current['score']-reference['score']) if current['score'] is not None and reference.get('score') is not None else None
                ok=current['status']==reference['status'] and current['decision']==reference['decision']
                if not ok or (delta is not None and delta>(1e-12 if name=='E0' else 0.000051)):
                    raise AssertionError('Development equivalence differs: '+row['record_id']+' '+name)
                comparison.append({'record_id':row['record_id'],'component':name,'status':current['status'],
                                   'decision_equal':ok,'score_abs_delta':delta,'truncated':current.get('truncated',False)})
            for mode in modes:
                vote=compose(mode,result['components']);d=vote['decision']; y=row['spam_label']
                modes[mode].append({'record_id':row['record_id'],'decision':d,'status':vote['status'],
                    'confusion': None if d is None else ('TP' if d else 'FN') if y else ('FP' if d else 'TN')})
        counts={mode:{k:sum(r['confusion']==k for r in rows) for k in ('TP','FP','TN','FN')}|
                     {'automatic':sum(r['status']=='ok' for r in rows),'needs_review':sum(r['status']=='needs_review' for r in rows)}
                for mode,rows in modes.items()}
        expected={'recall':{'TP':8,'FP':7,'TN':6,'FN':1,'automatic':22,'needs_review':0},
                  'review':{'TP':4,'FP':2,'TN':6,'FN':1,'automatic':13,'needs_review':9}}
        for mode in expected:
            if counts[mode]!=expected[mode]:raise AssertionError('Saved ensemble mismatch: '+mode)
        write(out,'development_equivalence.json',{'records':len(inputs),'component_comparisons':comparison,
            'modes':counts,'privacy':'No email text or headers saved; IDs and verification statistics only',
            'purpose':'implementation equivalence on old development inputs; no new quality estimate or tuning',
            'readiness':engine.readiness()})
    finally:
        engine.close()


def http_checks(out,base,baseline_only):
    import httpx
    checks=[]; demonstrations=[]
    def record(name,response,expected):
        ok=response.status_code==expected
        checks.append({'name':name,'http_status':response.status_code,'expected':expected,'passed':ok})
        if not ok:raise AssertionError(name+' '+str(response.status_code))
    with httpx.Client(base_url=base,trust_env=False,timeout=70) as client:
        record('health',client.get('/health'),200)
        record('baseline_ready',client.get('/ready'),200)
        if not baseline_only:
            start=time.monotonic()
            while not client.get('/ready').json()['modes']['recall'] and time.monotonic()-start<185:
                time.sleep(1)
        readiness=client.get('/ready').json()
        record('mode_readiness',client.get('/ready?mode=recall'),503 if baseline_only else 200)
        html=client.get('/');record('russian_page',html,200)
        if '<html lang="ru">' not in html.text:raise AssertionError('Russian UI absent')
        record('offline_api_docs',client.get('/api-docs'),200)
        record('openapi',client.get('/openapi.json'),200)
        modes=['baseline','recall','review']
        for path in sorted((ROOT/'examples/app').glob('*.eml')):
            raw=path.read_bytes()
            for mode in modes:
                response=client.post('/api/analyze',params={'mode':mode},content=raw,headers={'Content-Type':'message/rfc822'})
                record(path.name+'_'+mode,response,503 if baseline_only and mode!='baseline' else 200)
                answer=response.json()
                if baseline_only and mode!='baseline' and answer['decision'] is not None:
                    raise AssertionError('Missing neural component produced binary result')
                demonstrations.append({'synthetic_file':path.relative_to(ROOT).as_posix(),'mode':mode,'response':answer})
        for code,raw,expected in [('empty',b'',400),('binary_garbage',b'\x00\x00broken',400),
                                  ('oversize',b'x'*(MAX_BYTES+1),413)]:
            record(code,client.post('/api/analyze',content=raw,headers={'Content-Type':'message/rfc822'}),expected)
        def chunks():
            for _ in range(11):yield b'x'*(1024*1024)
        record('chunked_oversize',client.post('/api/analyze',content=chunks(),headers={'Content-Type':'message/rfc822'}),413)
        record('unknown_mode',client.post('/api/analyze?mode=C',content=b'body',headers={'Content-Type':'message/rfc822'}),422)
        record('arbitrary_model_rejected',client.post('/api/analyze?model_path=bad.joblib',content=b'body',headers={'Content-Type':'message/rfc822'}),422)
        record('cross_origin',client.post('/api/analyze',content=b'body',headers={'Content-Type':'message/rfc822','Origin':'https://evil.example'}),403)
        empty=client.post('/api/analyze?mode=baseline',content=b'From: test@example.invalid\r\n\r\n',headers={'Content-Type':'message/rfc822'})
        record('insufficient_features',empty,200)
        if empty.json()['status']!='insufficient_features' or empty.json()['decision'] is not None:
            raise AssertionError('Insufficient features incorrectly classified')
        write(out,'http_verification.json',{'base_url':base,'baseline_only':baseline_only,'checks':checks,
              'readiness':readiness,'synthetic_predictions':demonstrations,'actual_model_outputs':True})


def preservation(out):
    before=json.loads((ROOT/'reports/app/preservation_before.json').read_bytes())
    changed=[rel for rel,sha in before['sha256'].items() if not (ROOT/rel).is_file() or sha_file(ROOT/rel)!=sha]
    models_changed=[rel for rel,stat in before['model_stat'].items() if not (ROOT/rel).is_file() or
        {'bytes':(ROOT/rel).stat().st_size,'mtime_ns':(ROOT/rel).stat().st_mtime_ns}!=stat]
    # Direct verification of every deployed weight/config against final hashes.
    from app.contract import verify_component
    for name in ('E0','Laya_day5'):verify_component(name)
    result={'old_hashed_files':len(before['sha256']),'old_model_stat_files':len(before['model_stat']),
        'changed_old_hashed_files':changed,'changed_old_model_files':models_changed,
        'deployed_artifacts_sha256':'match frozen final protocol',
        'model_stat_limit':'All old model file sizes/mtimes checked; full SHA256 checked for deployed artifacts, not every unused research weight'}
    write(out,'preservation.json',result)
    if changed or models_changed:raise AssertionError('Old research artifacts changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['components','http','preservation'])
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args()
    # Preservation reads hashes of existing reports, not record contents.
    opened=boundary() if args.phase!='preservation' else set()
    out=new_out(args.phase)
    try:
        if args.phase=='components':components(out)
        elif args.phase=='http':http_checks(out,args.base_url,args.baseline_only)
        else:preservation(out)
        write(out,'status.json',{'status':'passed','phase':args.phase,'allowed_report_access':sorted(opened),
                               'closed_raw_or_split_files_accessed':False})
        print(json.dumps({'status':'passed','phase':args.phase,'report':str(out.relative_to(ROOT))}))
    except Exception as error:
        write(out,'status.json',{'status':'failed','phase':args.phase,'error_type':type(error).__name__})
        raise


if __name__=='__main__':main()
