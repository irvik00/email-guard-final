"""Functional HTTP smoke and comparison to saved synthetic answers, never a ML benchmark."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import httpx

ROOT=Path(__file__).resolve().parents[1]


def call(url,path,raw=None):
    # Handles an early 413 while a large request is still being sent on Windows.
    with httpx.Client(base_url=url,trust_env=False,timeout=70) as client:
        response=client.request('POST' if raw is not None else 'GET',path,content=raw,
            headers={'Content-Type':'message/rfc822'} if raw is not None else {})
        return response.status_code,response.json()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--label',default='local')
    p.add_argument('--with-laya',action='store_true');a=p.parse_args()
    if not a.url.startswith('http://127.0.0.1:'):raise ValueError('Loopback URL required')
    expected=json.loads((ROOT/'reports/release/synthetic_reference.json').read_bytes())
    checks=[];predictions=[]
    def check(name,ok):
        checks.append({'name':name,'passed':bool(ok)})
        if not ok:raise AssertionError(name)
    start=time.monotonic()
    while time.monotonic()-start<185:
        try:
            status,ready=call(a.url,'/ready?mode='+('recall' if a.with_laya else 'baseline'))
            if status==200:break
        except OSError:pass
        time.sleep(1)
    else:raise RuntimeError('Readiness did not become ready')
    check('readiness',status==200)
    for path in sorted((ROOT/'examples/app').glob('*.eml')):
        for mode in ('baseline','recall','review'):
            status,r=call(a.url,'/api/analyze?mode='+mode,path.read_bytes())
            want=expected[path.name][mode]
            if not a.with_laya and mode!='baseline':
                check(path.name+'_'+mode+'_unavailable',status==503 and r['decision'] is None and r['status']==('needs_review' if mode=='review' else 'component_unavailable'))
            else:
                check(path.name+'_'+mode+'_decision',status==200 and r['status']==want['status'] and r['decision']==want['decision'])
                for name,c in r['components'].items():
                    old=want['components'][name]
                    delta=abs(c['score']-old['score']) if c.get('score') is not None and old.get('score') is not None else 0
                    check(path.name+'_'+mode+'_'+name+'_score',c['status']==old['status'] and c['decision']==old['decision'] and delta<=(1e-10 if name=='E0' else .000051))
            predictions.append({'file':path.name,'mode':mode,'http_status':status,'status':r['status'],'decision':r['decision'],
                'components':{n:{k:c.get(k) for k in ('score','threshold','temperature','status','decision','truncated')} for n,c in r['components'].items()}})
    for name,raw,want in [('empty',b'',400),('corrupt',b'\x00\x01broken',400),('oversize',b'x'*(10*1024*1024+1),413)]:
        status,r=call(a.url,'/api/analyze',raw);check(name,status==want and r['decision'] is None)
    status,r=call(a.url,'/api/analyze',b'From: demo@example.invalid\r\n\r\n')
    check('insufficient_features',status==200 and r['status']=='insufficient_features' and r['decision'] is None)
    chunks=(b'x'*(1024*1024) for _ in range(11))
    with httpx.Client(base_url=a.url,trust_env=False,timeout=30) as client:
        response=client.post('/api/analyze',content=chunks,headers={'Content-Type':'message/rfc822'})
        body=response.json()
    check('chunked_oversize',response.status_code==413 and body['decision'] is None)
    folder=ROOT/'reports/release/smoke_runs';folder.mkdir(parents=True,exist_ok=True)
    label=''.join(c for c in a.label if c.isalnum() or c in '-_')[:50]
    path=folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+label+'.json')
    path.write_text(json.dumps({'status':'passed','label':label,'url':a.url,'with_laya':a.with_laya,
        'checks':checks,'predictions':predictions,'comparison':'saved synthetic app outputs; no fitting or benchmark',
        'execution_environment':'HTTP target only; label is not evidence of Docker execution'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','checks':len(checks),'report':str(path.relative_to(ROOT))}))


if __name__=='__main__':main()
