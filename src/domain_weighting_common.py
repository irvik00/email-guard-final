"""Frozen-domain-weight experiment contracts. No raw dataset or network access."""
from day5_common import *
from personal_eval_common import evaluated_metrics,setup_offline
import importlib.metadata
import time

BASE=ROOT/'reports/domain_weighting/runs'
PREVIOUS=ROOT/'reports/extra_session/runs/20261017T165143Z-visibility-qwen'
W0_DIR=PREVIOUS/'C_larger_fit'
EXTRA=ROOT/'reports/extra_experiments/runs/20261012T095633Z-7d01f7b8/C'
DAY5=ROOT/'reports/day5/runs/20261012T083529Z-0950e3dc34'
SHARES={'W1':{'spam_email':1/3,'russian_auxiliary':2/3},'W2':{'spam_email':.5,'russian_auxiliary':.5}}
SETS=['pilot_historical','pilot_ru','personal_development','full_historical_validation','full_ru_validation']


class WeightBoundary:
    def __init__(self,out):self.out=Path(out).resolve();self.opened=Counter();self.denied=[]
    def check(self,event,args):
        if event in {'socket.connect','socket.bind'}:
            self.denied.append(event);raise PermissionError('Local frozen experiment: no network')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=str(args[1] or '');flags=int(args[2] or 0)
        writing=any(c in mode for c in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR))
        if path.is_relative_to(ROOT/'data') or path==ROOT/'test_task_emails.tar.xz' or path.is_relative_to(ROOT/'reports/final_evaluation'):
            self.denied.append(str(path));raise PermissionError('All raw/split data and closed evaluation are unavailable')
        if writing and path.is_relative_to(ROOT) and not path.is_relative_to(self.out):
            self.denied.append(str(path));raise PermissionError('Previous project artifacts are read-only')
        if path.is_relative_to(ROOT) and '__pycache__' not in path.parts:self.opened[path.relative_to(ROOT).as_posix()+':'+mode]+=1
    def install(self):sys.addaudithook(self.check);return self
    def report(self):return {'opened':dict(self.opened),'denied':self.denied,'raw_and_closed_data_reads':0,'network':False}


def protocol(out):
    out=Path(out).resolve();require(out.is_relative_to(BASE),'Wrong run root')
    lock=read_json(out/'protocol_lock.json');require(sha_file(out/'DOMAIN_WEIGHTING_PROTOCOL.json')==lock['sha256'],'Protocol changed')
    p=read_json(out/'DOMAIN_WEIGHTING_PROTOCOL.json')
    for f,h in p['input_hashes'].items():require(sha_file(ROOT/f)==h,'Input artifact changed: '+f)
    for f,h in p['code_hashes'].items():require(sha_file(ROOT/f)==h,'Frozen implementation changed: '+f)
    return p


def sample_weights(rows,shares=None):
    """One weight per original row, in exact original order, never oversampling."""
    import numpy as np
    n=len(rows);require(n>0,'Empty fit');counts=Counter(r['domain'] for r in rows)
    if shares is not None:
        require(set(shares)==set(counts) and all(math.isfinite(v) and v>0 for v in shares.values()),'Invalid shares')
        require(abs(math.fsum(shares.values())-1)<1e-12,'Shares must sum to1')
    w=np.array([1. if shares is None else n*shares[r['domain']]/counts[r['domain']] for r in rows],dtype=np.float64)
    require(np.isfinite(w).all() and (w>0).all() and abs(math.fsum(w)-n)<1e-8,'Weight normalization')
    return w


def weight_table(rows,w):
    def stat(indices):
        values=[float(w[i]) for i in indices];return {'rows':len(values),'sum':math.fsum(values),'mean':math.fsum(values)/len(values)}
    return {'total':stat(range(len(rows))),
        'domain':{d:stat([i for i,r in enumerate(rows) if r['domain']==d]) for d in sorted({r['domain'] for r in rows})},
        'class':{str(y):stat([i for i,r in enumerate(rows) if r['spam_label']==y]) for y in [0,1]},
        'domain_class':{d+':'+str(y):stat([i for i,r in enumerate(rows) if r['domain']==d and r['spam_label']==y]) for d in sorted({r['domain'] for r in rows}) for y in [0,1]}}


def inner_parts(p):
    spec=read_json(EXTRA/'embeddings.json');excluded=set(p['calibration_excluded_ids']);rows={};matrices={}
    import numpy as np
    for part in ['dev','calibration']:
        data=read_rows(EXTRA/f'private_{part}_inputs.jsonl');entry=spec['parts'][part]
        require(sha_file(EXTRA/f'private_{part}_inputs.jsonl')==entry['input_sha256'],'Inner cache input mismatch')
        require(sha_file(ROOT/entry['matrix_path'])==entry['sha256'],'Inner matrix mismatch')
        indices=[i for i,r in enumerate(data) if part!='calibration' or r['record_id'] not in excluded]
        rows[part]=[data[i] for i in indices];matrices[part]=np.load(ROOT/entry['matrix_path'],allow_pickle=False)[indices]
    require(len(rows['dev'])==600 and len(rows['calibration'])==999,'Changed dev/calibration')
    return rows,matrices


def predict_rows(model,rows,matrix,T,threshold):
    import numpy as np
    z=model.decision_function(matrix);require(len(z)==len(rows) and np.isfinite(z).all(),'Nonfinite/missing logits')
    return [{'record_id':r['record_id'],'domain':r['domain'],'spam_label':r['spam_label'],'status':'ok',
        'logit_difference':float(v),'score':sigmoid(float(v)/T),'raw_decision':int(v>=0),
        'operating_decision':int(sigmoid(float(v)/T)>=threshold),'temperature':T,'threshold':threshold} for r,v in zip(rows,z)]


def model_metrics(rows,mode):
    ok=[r for r in rows if r['status']=='ok' and type(r.get(mode+'_decision')) is int]
    return {'records':len(rows),'successful':len(ok),'coverage':len(ok)/len(rows),
        'refusals':dict(Counter(r['status'] for r in rows if r not in ok)),
        'metrics':evaluated_metrics([r['spam_label'] for r in ok],[r[mode+'_decision'] for r in ok])}


def paired_errors(old,new,mode):
    a={r['record_id']:r for r in old};b={r['record_id']:r for r in new}
    require(set(a)==set(b) and len(a)==len(old)==len(new),'Paired IDs differ')
    ids=[i for i in a if a[i]['status']==b[i]['status']=='ok' and type(a[i].get(mode+'_decision')) is int and type(b[i].get(mode+'_decision')) is int]
    result={'common_successful':len(ids),'noncommon':len(old)-len(ids)}
    for k,y,d0,d1 in [('rescued_FP',0,1,0),('added_FP',0,0,1),('rescued_FN',1,0,1),('added_FN',1,1,0)]:
        result[k]=[i for i in ids if a[i]['spam_label']==y and a[i][mode+'_decision']==d0 and b[i][mode+'_decision']==d1]
    result['old_common']=evaluated_metrics([a[i]['spam_label'] for i in ids],[a[i][mode+'_decision'] for i in ids])
    result['new_common']=evaluated_metrics([b[i]['spam_label'] for i in ids],[b[i][mode+'_decision'] for i in ids])
    return result
