"""Independent audit of app manifest and saved synthetic HTTP responses; no inference/datasets."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    release=json.loads((ROOT/'app/release_manifest.json').read_bytes())
    source=release['source_final_protocol']
    assert sha(ROOT/source['path'])==source['sha256']
    protocol=json.loads((ROOT/source['path']).read_bytes())
    fixed={x['name']:x for x in protocol['components'] if x['name'] in ('E0','Laya_day5')}
    assert release['components']==fixed and set(release['modes'])=={'baseline','recall','review'}
    checked=0
    for section in [release['shared_preprocessing_sha256'],release['application_files_sha256'],release['verification_files_sha256']]+[s['files'] for s in fixed.values()]:
        for rel,expected in section.items():
            assert sha(ROOT/rel)==expected,rel
            checked+=1
    assert sha(ROOT/release['trusted_sdk_wheel']['path'])==release['trusted_sdk_wheel']['sha256']
    assert release['default']=='baseline'
    latest={}
    for status in sorted((ROOT/'reports/app/verification_runs').glob('*/status.json')):
        s=json.loads(status.read_bytes())
        if s['status']=='passed' and s['phase']=='http':
            report=json.loads((status.parent/'http_verification.json').read_bytes())
            latest[report['baseline_only']]=(status.parent,report)
    assert set(latest)=={False,True}
    predictions=0;http_checks=0
    for baseline_only,(folder,report) in latest.items():
        assert all(c['passed'] and c['http_status']==c['expected'] for c in report['checks'])
        http_checks+=len(report['checks'])
        for item in report['synthetic_predictions']:
            predictions+=1; r=item['response']; parts=r['components'];mode=item['mode']
            assert r['ensemble_score'] is None
            ok=[]
            for name,c in parts.items():
                assert name in fixed
                assert c['threshold']==fixed[name]['threshold'] and c['temperature']==fixed[name]['temperature']
                valid=c['status']=='ok' and type(c['decision']) is int and c['decision'] in (0,1)
                ok.append(valid)
                if valid:
                    assert math.isfinite(c['score']) and 0<=c['score']<=1
                    assert c['decision']==int(c['score']>=c['threshold'])
                else:assert c['decision'] is None
                if name=='Laya_day5' and valid:
                    assert c['token_stats']['visible_tokens']<=700 and c['sequence_tokens']<=1024
                    z=c['logit_difference']/c['temperature']
                    p=1/(1+math.exp(-z))
                    assert abs(p-c['score'])<1e-12
            if mode=='baseline':
                assert set(parts)=={'E0'} and r['decision']==parts['E0']['decision']
            else:
                assert set(parts)=={'E0','Laya_day5'}
                if all(ok):
                    a,b=parts['E0']['decision'],parts['Laya_day5']['decision']
                    expected=int(a or b) if mode=='recall' else a if a==b else None
                    assert r['decision']==expected
                    if mode=='review' and a!=b:assert r['status']=='needs_review'
                else:
                    assert r['decision'] is None
                    assert r['status']==('needs_review' if mode=='review' else 'component_unavailable')
            assert not set(r['email']) & {'body_text','html','plain_text','text','local_path','filename'}
            assert r['email']['category']['value'] is None
            assert r['email']['observations_are_model_explanations'] is False
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result={'status':'passed','utc':now,'hashed_release_files_checked':checked,
            'http_checks_independently_checked':http_checks,'synthetic_mode_outputs_checked':predictions,
            'component_thresholds_temperature_scores_and_votes':'consistent',
            'raw_or_closed_dataset_accessed':False,'new_training':False,
            'limitations':'Static manifest and saved response audit; not a load test, container test or new quality evaluation'}
    path=ROOT/'reports/app'/('independent_audit_'+now+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
