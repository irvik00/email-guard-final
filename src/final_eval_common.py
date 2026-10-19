"""Frozen binary compositions, split-safe evaluation, safe streamed tar reading."""
import csv
import hashlib
import os
from pathlib import PurePosixPath, PureWindowsPath
import tarfile
from day5_common import *

BASE=ROOT/'reports/final_evaluation'
TEACHER=ROOT/'test_task_emails.tar.xz'
PERSONAL_RUN=ROOT/'reports/personal_email_evaluation/runs/20261017T120741Z-a38c25e1'
COMPONENTS=('E0','Laya_day5','C_corrected')
ENSEMBLES=('E0_OR_Laya','E0_AND_Laya','Majority_E0_Laya_C','E0_Laya_review')
BINARY=COMPONENTS+ENSEMBLES[:3]
MODEL_NAMES=BINARY+(ENSEMBLES[3],)
TEST_SETS=('historical_test','teacher_test','russian_auxiliary_test')

class FinalBoundary:
    def __init__(self,out):
        self.out=Path(out).resolve();self.phase='validation';self.accesses=Counter();self.first_test_accesses=[]
        self.allowed={ROOT/'data/processed/splits/manifest.json'};self.lock=None;self.denied=[]
        self.tests={TEACHER,ROOT/'data/processed/splits/spam_email/test.jsonl',ROOT/'data/processed/splits/russian_auxiliary/test.jsonl'}
    def seal(self):
        path=self.out/'FINAL_EVALUATION_PROTOCOL.json'
        require(path.is_file(),'Protocol must exist before seal')
        self.lock=sha_file(path);self.phase='final'
        self.allowed.update(ROOT/f'data/processed/splits/{d}/{s}.jsonl' for d in DOMAINS for s in ['train','validation','test'])
    def check(self,event,args):
        if event in {'socket.connect','socket.bind'}:raise PermissionError('No network in final evaluation')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode=str(args[1] or '');flags=int(args[2] or 0)
        writing=any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR))
        if p in self.tests:
            if self.phase!='final' or not self.lock or writing:
                self.denied.append(str(p));raise PermissionError('Held-out content requires sealed final protocol')
            require(sha_file(self.out/'FINAL_EVALUATION_PROTOCOL.json')==self.lock,'Final protocol mutated')
            if not any(a['path']==p.relative_to(ROOT).as_posix() for a in self.first_test_accesses):
                self.first_test_accesses.append({'path':p.relative_to(ROOT).as_posix(),'first_open_utc':now(),'protocol_sha256':self.lock})
        if p.is_relative_to(ROOT/'data'):
            if writing or p not in self.allowed:
                self.denied.append(str(p));raise PermissionError('Non-allowlisted data remains closed')
            self.accesses[p.relative_to(ROOT).as_posix()+':'+mode]+=1
        if writing and (p.is_relative_to(ROOT/'models') or
                        (p.is_relative_to(ROOT/'reports') and not p.is_relative_to(self.out))):
            raise PermissionError('Previous models/reports are immutable')
        if writing and self.lock and p in {self.out/'FINAL_EVALUATION_PROTOCOL.json',self.out/'FINAL_EVALUATION_PROTOCOL.md'}:
            raise PermissionError('Final protocol is immutable')
    def install(self):sys.addaudithook(self.check);return self
    def report(self):return {'phase':self.phase,'accesses':dict(self.accesses),'denied':self.denied,
        'first_test_accesses':self.first_test_accesses,'protocol_sha256':self.lock,'no_network':True}

def component_ok(row):
    return row is not None and row.get('status')=='ok' and type(row.get('decision')) is int and row['decision'] in {0,1}

def vote(name,components):
    require(name in ENSEMBLES,'Unknown ensemble')
    needed=COMPONENTS if name=='Majority_E0_Laya_C' else COMPONENTS[:2]
    missing=[n for n in needed if not component_ok(components.get(n))]
    if missing:
        return {'status':'needs_review' if name=='E0_Laya_review' else 'component_unavailable',
            'decision':None,'reason':'component_unavailable','unavailable_components':missing}
    values=[components[n]['decision'] for n in needed]
    if name=='E0_Laya_review':
        return {'status':'ok','decision':values[0],'reason':'agreement','unavailable_components':[]} if values[0]==values[1] else {
            'status':'needs_review','decision':None,'reason':'disagreement','unavailable_components':[]}
    result=int(any(values)) if name=='E0_OR_Laya' else int(all(values)) if name=='E0_AND_Laya' else int(sum(values)>=2)
    return {'status':'ok','decision':result,'reason':'fixed_binary_composition','unavailable_components':[]}

def confusion(rows):
    c=dict.fromkeys(['TP','FP','TN','FN'],0)
    for r in rows:
        y,d=r['spam_label'],r['decision'];require(type(y) is int and y in {0,1} and type(d) is int and d in {0,1},'Nonbinary metric input')
        c[('TP' if d else 'FN') if y else ('FP' if d else 'TN')]+=1
    return c

def evaluate(rows,review=False):
    ok=[r for r in rows if component_ok(r)];counts=confusion(ok);n=len(rows)
    a,b,c,d=[counts[k] for k in ['TP','FP','TN','FN']]
    common={'records':n,'automatic':len(ok),'coverage':len(ok)/n if n else None,
        'class_counts':dict(Counter(str(r['spam_label']) for r in rows)),
        'status_counts':dict(Counter(r['status'] for r in rows))}
    if review:
        pending=[r for r in rows if not component_ok(r)]
        return common|{'automatic_fraction':len(ok)/n if n else None,'needs_review':len(pending),
            'review_by_true_class':dict(Counter(str(r['spam_label']) for r in pending)),
            'review_reasons':dict(Counter(r.get('reason','unknown') for r in pending)),
            'automatic_confusion':counts,'automatic_errors':b+d,
            'note':'No full-dataset F1: manual review outcomes unknown; auto confusion is selective only'}
    undefined={}
    def ratio(k,x,y):
        if y:return x/y
        undefined[k]='zero denominator among successful decisions';return None
    return common|{'confusion':counts,'precision':ratio('precision',a,a+b),'recall':ratio('recall',a,a+d),
        'F1':ratio('F1',2*a,2*a+b+d),'FPR':ratio('FPR',b,b+c),'undefined_reasons':undefined,
        'metrics_conditional_on_coverage':len(ok)!=n,
        'unavailable_ids':[r['record_id'] for r in rows if not component_ok(r)]}

def combine(dataset,truth,components):
    output=[]
    for t in truth:
        rid=t['record_id'];parts={name:components[name].get(rid) for name in COMPONENTS}
        for name in COMPONENTS:
            item=parts[name] or {'status':'error','decision':None,'error_type':'missing_prediction'}
            item=dict(item);item.pop('spam_label',None);item.pop('domain',None)
            output.append(item|{'dataset':dataset,'record_id':rid,'model_id':name,'spam_label':t['spam_label'],
                'label_status':t.get('label_status','source_provided')})
        for name in ENSEMBLES:
            output.append(vote(name,parts)|{'dataset':dataset,'record_id':rid,'model_id':name,'spam_label':t['spam_label'],
                'label_status':t.get('label_status','source_provided'),'component_decisions':{
                    k:(v['decision'] if component_ok(v) else None) for k,v in parts.items()},'score':None})
    return output

def summary(rows):
    return {name:evaluate([r for r in rows if r['model_id']==name],name=='E0_Laya_review') for name in MODEL_NAMES}

def paired_changes(rows):
    baseline={r['record_id']:r for r in rows if r['model_id']=='E0'};result={}
    for name in BINARY[1:]:
        c=Counter();ids=defaultdict(list)
        for r in rows:
            if r['model_id']!=name:continue
            old=baseline[r['record_id']]
            if not(component_ok(r) and component_ok(old)):continue
            y=r['spam_label'];a=old['decision'];b=r['decision']
            kind='rescued_E0_FN' if y==1 and a==0 and b==1 else 'added_FN' if y==1 and a==1 and b==0 else \
                 'removed_E0_FP' if y==0 and a==1 and b==0 else 'added_FP' if y==0 and a==0 and b==1 else None
            if kind:c[kind]+=1;ids[kind].append(r['record_id'])
        result[name]={'counts':dict(c),'ids':dict(ids)}
    return result

def validate_tar_members(members,max_file_bytes=20*1024**2,max_total_bytes=1024**3):
    files=[];seen=set();total=0;skipped=[]
    for member in members:
        name=member.name;p=PurePosixPath(name);w=PureWindowsPath(name)
        require(not p.is_absolute() and not w.is_absolute() and not w.drive and '\\' not in name and '\x00' not in name and '..' not in p.parts,'Unsafe tar path')
        require(member.isdir() or member.isfile(),'Special/link/device tar entry forbidden')
        normalized=p.as_posix().removeprefix('./').rstrip('/')
        require(normalized not in seen,'Repeated tar path');seen.add(normalized)
        if member.isdir():continue
        require(0<=member.size<=max_file_bytes,'Archive member size limit');total+=member.size
        require(total<=max_total_bytes,'Archive total size limit')
        if p.suffix.casefold()!='.eml':skipped.append({'path_sha256':digest(name),'size_bytes':member.size});continue
        classes=[part.casefold() for part in p.parts if part.casefold() in {'ham','spam'}]
        require(len(classes)==1,'Expected exactly one ham/spam class directory')
        files.append((member,{'ham':0,'spam':1}[classes[0]]))
    return files,{'regular_member_bytes':total,'non_eml_regular_files':skipped,'safe_members_checked':len(members),
        'extraction':'none: regular members streamed in memory; no archive filesystem extraction'}

def model_payload(subject,body,config):
    from normalization_schema import make_model_text
    original=make_model_text(subject,body)[0]
    clean=clean_text({'subject':subject,'body_text':body,'feature_availability':{'subject':'observable'}},config)[0]
    return original,clean

def neutral_input(rid,dataset,original,clean):
    # Ground truth and local/member paths are deliberately absent. The legacy adapter's
    # label slot is a constant zero; actual forward receives only IDs/masks/qtype.
    return {'record_id':rid,'domain':dataset,'source':'evaluation_text','duplicate_group_id':'visible:'+digest(compare_text(clean)),
        'text':clean,'original_model_text':original,'spam_label':0,'label_source':'not_supplied_to_model','label_reviewed_at':''}

def text_keys(original,clean,raw_sha=None):
    keys=[]
    for name,text in [('original',original),('clean',clean)]:
        if not text.strip():continue
        keys.extend([(name+'_exact',hashlib.sha256(text.encode()).hexdigest()),
                     (name+'_canonical',hashlib.sha256(compare_text(text).encode()).hexdigest())])
    if raw_sha:keys.append(('raw',raw_sha))
    return keys

def save_csv(path,rows):
    fields=['dataset','record_id','model_id','spam_label','label_status','score','threshold','temperature','decision','status','reason','error_type']
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in rows)

def old_artifacts_snapshot():
    paths=set()
    for folder in ['models','reports','src','config','tests','docs']:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(BASE))
    paths.update(ROOT.glob('requirements*.txt'));paths.add(ROOT/'.gitignore')
    return {p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted(paths)}
