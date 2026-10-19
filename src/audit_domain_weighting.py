"""Targeted independent audit. No refitting, encoder rerun or raw-dataset scan."""
import argparse
from domain_weighting_common import *

def main(out):
    boundary=WeightBoundary(out).install();p=protocol(out)
    import numpy as np
    import joblib
    from scipy.optimize import minimize_scalar
    from threadpoolctl import threadpool_limits
    fit=read_rows(W0_DIR/'fit_membership.jsonl');require(digest([r['record_id'] for r in fit])==p['fit_ids_order_sha256'],'Fit order changed')
    freeze=read_json(out/'models_frozen.json');require(freeze['new_fit_calls']==2 and not freeze['W0_refitted'],'Unexpected model fits')
    inner,matrices=inner_parts(p);reload={};calchecks={};weightchecks={}
    with threadpool_limits(limits=1):
        for name in ['W1','W2']:
            dest=out/'models'/name;m=freeze['models'][name];clf=joblib.load(ROOT/m['classifier_path'])
            require(clf.get_params()==p['lr_parameters'] and np.isfinite(clf.coef_).all() and np.isfinite(clf.intercept_).all(),'Model parameters/finite check')
            require(sha_file(ROOT/m['classifier_path'])==m['classifier_sha256'],'New classifier hash changed')
            w=np.load(dest/'sample_weight.npy',allow_pickle=False);counts=Counter(r['domain'] for r in fit)
            expected=np.array([len(fit)*p['variants'][name][r['domain']]/counts[r['domain']] for r in fit],dtype=np.float64)
            require(np.array_equal(w,expected) and abs(math.fsum(w)-len(fit))<1e-8,'Weight formula/order/sum mismatch')
            native=w.astype(np.float32);require(all(t['actual_weight_hash']==digest(native.tolist()) for t in m['native_weight_validation']),'Actual sklearn weights differ')
            weightchecks[name]={'sum64':math.fsum(w),'native_sum':float(np.sum(native)),'mean64':float(math.fsum(w)/len(w))}
            raw={}
            for part in ['calibration','dev']:
                rows=read_rows(dest/f'{part}_raw.jsonl');z=clf.decision_function(matrices[part]);require([r['record_id'] for r in rows]==[r['record_id'] for r in inner[part]],'Inner membership changed')
                require(all(r['spam_label']==s['spam_label'] for r,s in zip(rows,inner[part])),'Inner label changed')
                error=float(np.max(np.abs(z-np.array([r['logit_difference'] for r in rows]))));require(error<=1e-10,'Independent reload changed logits');reload[name+'/'+part]=error;raw[part]=rows
            def nll(logT):
                T=math.exp(logT)
                return math.fsum(max(r['logit_difference']/T,0)-r['spam_label']*r['logit_difference']/T+math.log1p(math.exp(-abs(r['logit_difference']/T))) for r in raw['calibration'])/999
            opt=minimize_scalar(nll,bounds=(math.log(.5),math.log(5)),method='bounded',options={'xatol':1e-10})
            T=math.exp(min([math.log(.5),math.log(5),float(opt.x)],key=nll));require(abs(T-m['temperature'])<1e-4,'Independent T differs')
            options=[]
            for threshold in {sigmoid(r['logit_difference']/m['temperature']) for r in raw['dev']}:
                stats={}
                for d in DOMAINS:
                    tp=fp=tn=fn=0
                    for r in raw['dev']:
                        if r['domain']!=d:continue
                        dec=sigmoid(r['logit_difference']/m['temperature'])>=threshold;y=r['spam_label']
                        tp+=int(y==1 and dec);fp+=int(y==0 and dec);tn+=int(y==0 and not dec);fn+=int(y==1 and not dec)
                    stats[d]=(tp,fp,tn,fn)
                a=stats['spam_email'];b=stats['russian_auxiliary']
                if a[1]/(a[1]+a[2])<=.01 and a[0]>0 and b[0]>0:
                    mean=(2*a[0]/(2*a[0]+a[1]+a[3])+2*b[0]/(2*b[0]+b[1]+b[3]))/2;options.append((mean,-a[1],threshold))
            threshold=max(options)[2] if options else .5;require(abs(threshold-m['threshold'])<1e-12,'Independent threshold differs')
            calchecks[name]={'independent_T':T,'independent_threshold':threshold}
        data=read_json(out/'metrics_and_paired_changes.json');groups=0;pairs=0
        for key in SETS:
            allpred={}
            for name in ['W0','W1','W2']:
                rows=read_rows(out/f'predictions_{key}_{name}.jsonl');allpred[name]=rows;m=freeze['models'][name]
                require(len({r['record_id'] for r in rows})==len(rows),'Repeated prediction ID')
                for r in rows:
                    if r['status']=='ok':
                        require(math.isfinite(r['logit_difference']) and r['raw_decision']==int(r['logit_difference']>=0),'Wrong raw boundary')
                        require(r['temperature']==m['temperature'] and r['threshold']==m['threshold'],'Wrong operating settings')
                        require(abs(r['score']-sigmoid(r['logit_difference']/m['temperature']))<1e-12 and r['operating_decision']==int(r['score']>=m['threshold']),'Wrong calibrated decision')
                    else:require(r.get('raw_decision') is None and r.get('operating_decision') is None,'Error became ham')
                for mode in ['raw','operating']:
                    counts=Counter((r['spam_label'],r[mode+'_decision']) for r in rows if r['status']=='ok');tp,fp,tn,fn=[counts[k] for k in [(1,1),(0,1),(0,0),(1,0)]]
                    saved=data['metrics'][key][name][mode];require([saved['metrics'][k] for k in ['TP','FP','TN','FN']]==[tp,fp,tn,fn],'Confusion differs')
                    div=lambda a,b:a/b if b else None
                    values={'precision':div(tp,tp+fp),'recall':div(tp,tp+fn),'F1':div(2*tp,2*tp+fp+fn),'FPR':div(fp,fp+tn)}
                    require(all((v==saved['metrics'][k]) if v is None else abs(v-saved['metrics'][k])<1e-12 for k,v in values.items()),'Metric arithmetic differs')
                    require(saved['successful']==tp+fp+tn+fn and saved['records']==len(rows),'Coverage mismatch');groups+=1
            for name in ['W1','W2']:
                a={r['record_id']:r for r in allpred['W0']};b={r['record_id']:r for r in allpred[name]};require(set(a)==set(b),'Different comparison IDs')
                for mode in ['raw','operating']:
                    for k,y,d0,d1 in [('rescued_FP',0,1,0),('added_FP',0,0,1),('rescued_FN',1,0,1),('added_FN',1,1,0)]:
                        ids=[i for i in a if a[i]['status']==b[i]['status']=='ok' and a[i]['spam_label']==y and a[i][mode+'_decision']==d0 and b[i][mode+'_decision']==d1]
                        require(ids==data['paired'][key][name][mode][k],'Paired change differs')
                    pairs+=1
    protected=read_json(out/'preservation_before.json');require(all(sha_file(ROOT/path)==h for path,h in protected.items()),'Default/config artifact changed')
    for access in out.glob('*accesses.json'):
        record=read_json(access);require(not record['denied'],'Unexpected denied runtime access')
        require(not any(path.startswith('data/') or path.startswith('test_task_emails.tar.xz') or path.startswith('reports/final_evaluation/') for path in record['opened']),'Closed data read')
    for report in out.glob('*.md'):require(not re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',report.read_text(encoding='utf8')),'Address leaked to report')
    result={'status':'passed','utc':now(),'metric_groups':groups,'paired_groups':pairs,'reload':reload,'calibration_and_threshold':calchecks,
        'weights':weightchecks,'targeted_unchanged_files':len(set(protected)|set(p['input_hashes'])),
        'fit_ids_order_unchanged':True,'new_fit_calls':2,'W0_refitted':False,'raw_closed_data_reads':0,
        'previous_verified_audit_reused':True,'no_expensive_retokenization':True,'accesses':boundary.report()}
    write_json(out/'independent_audit.json',result);print({k:v for k,v in result.items() if k!='accesses'},flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);args=a.parse_args();main(args.out.resolve())
