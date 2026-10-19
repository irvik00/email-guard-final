"""Strict final user labels and local-only, read-only personal-email evaluation."""
import csv
import io
from email.utils import parsedate_to_datetime
from itertools import combinations
from day5_common import *

PERSONAL = ROOT/'data/personal_review'
BASE = ROOT/'reports/personal_email_evaluation'
SOURCE_FIELDS = ['local_path','spam_label','category','label_reason','approximate_date','date_scope',
                 'label_source','label_reviewed_at']
NORMALIZED_FIELDS = SOURCE_FIELDS + ['record_id','original_local_path','raw_sha256','human_spam_label',
    'review_status','initial_label_source','binary_label_source','category_status','phishing_label','consent',
    'expectedness','consent_evidence','date_header_iso','received_date_min_iso','received_date_max_iso','language_self_reported']

class PersonalBoundary:
    """No network; exact personal allowlist; immutable previous artifacts and raw inputs."""
    def __init__(self, out, personal=()):
        self.out=Path(out).resolve(); self.allowed={Path(p).resolve() for p in personal}
        self.accesses=Counter(); self.denied=[]
    def check(self,event,args):
        if event in {'socket.connect','socket.bind'}:
            self.denied.append(event); raise PermissionError('Personal email cannot use network')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve(); mode=str(args[1] or ''); flags=int(args[2] or 0)
        writing=any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR))
        if p.is_relative_to(ROOT/'data'):
            if writing or p not in self.allowed:
                self.denied.append(p.relative_to(ROOT).as_posix());raise PermissionError('Non-allowlisted/read-only data')
            self.accesses[p.relative_to(ROOT).as_posix()+':'+mode]+=1
        if writing and (p.is_relative_to(ROOT/'models') or
                        (p.is_relative_to(ROOT/'reports') and not p.is_relative_to(self.out))):
            raise PermissionError('Previous model/report is immutable')
    def install(self):sys.addaudithook(self.check);return self
    def report(self):return {'no_network':True,'allowed_personal_files':[str(p.relative_to(ROOT)) for p in sorted(self.allowed)],
        'data_accesses':dict(self.accesses),'denied':self.denied}

def final_annotations(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);require(reader.fieldnames==SOURCE_FIELDS,'Expected current final eight-column user CSV')
        rows=list(reader)
    require(rows,'Empty final annotation')
    seen=set()
    for n,r in enumerate(rows,2):
        require(None not in r and all(isinstance(v,str) for v in r.values()),f'Malformed CSV row {n}')
        require(r['spam_label'] in {'0','1'},f'Final label must be literal 0 or 1, row {n}')
        require(r['label_source']=='human_reviewed_assistant_draft',f'Unconfirmed provenance, row {n}')
        require(bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}',r['label_reviewed_at'])),f'Review date missing, row {n}')
        datetime.strptime(r['label_reviewed_at'],'%Y-%m-%d')
        require(bool(r['label_reason'].strip()),f'Final label reason missing, row {n}')
        p=Path(r['local_path']);require(p.name not in seen,'Repeated email ID');seen.add(p.name)
        require(bool(re.fullmatch(r'mail_\d{3}\.eml',p.name)),'Unexpected personal filename')
        resolved=(ROOT/p).resolve();require(resolved.is_relative_to(PERSONAL),'Path escapes personal root')
    return rows

def resolve_personal_path(row,root=ROOT):
    root=Path(root).resolve(); base=root/'data/personal_review'
    specified=(root/row['local_path']).resolve()
    require(specified.is_relative_to(base),'Personal path escapes root')
    require(bool(re.fullmatch(r'mail_\d{3}\.eml',specified.name)),'Invalid filename')
    if specified.is_file():return specified,False
    # Known stale /emails/ prefix: bounded single-file fallback, no recursive/global scan.
    require(specified.parent==base/'emails','Unexpected relocation; do not search automatically')
    fallback=base/specified.name
    require(fallback.is_file(),'Explicit personal file missing: '+specified.name)
    return fallback,True

def date_metadata(parsed,annotation,cutoff='2025-01-01',as_of='2026-10-17'):
    dates=[];received=[];errors=[];received_chronological=[]
    for h in parsed.get('headers_raw',[]):
        kind=h['name'].lower()
        if kind not in {'date','received'}:continue
        value=h['value'].rpartition(';')[2] if kind=='received' else h['value']
        try:
            dt=parsedate_to_datetime(value);value=dt.isoformat()
            (dates if kind=='date' else received).append(value)
            if kind=='received' and dt.tzinfo is not None:received_chronological.append((dt.timestamp(),value))
        except (ValueError,TypeError,OverflowError):errors.append(kind)
    date=dates[0] if dates else None
    dated=date[:10] if date else None
    modern=bool(dated and cutoff<=dated<=as_of)
    # A recent user claim is separate evidence; do not promote a known old header.
    scope='modern_header_claim' if modern else 'old_header' if dated and dated<cutoff else 'future_header' if dated and dated>as_of else 'unknown_header'
    timed=sorted(received_chronological)
    return {'date_header_iso':date,'received_date_min_iso':timed[0][1] if timed else None,
        'received_date_max_iso':timed[-1][1] if timed else None,'received_dates_iso':received,
        'header_date_errors':errors,'header_date_count':len(dates),'date_subset':scope,
        'modern':modern,'user_approximate_date':annotation['approximate_date'],'user_date_scope':annotation['date_scope'],
        'date_authenticity':'unverified headers/user claim, not trusted server arrival'}

def script_metadata(text):
    # Descriptive script counts, NOT a language detector. URLs/addresses are omitted here only.
    text=re.sub(r'https?://\S+|[\w.+-]+@[\w.-]+',' ',text)
    c=Counter()
    for x in text:
        if not x.isalpha():continue
        name=unicodedata.name(x,'')
        c['cyrillic' if 'CYRILLIC' in name else 'latin' if 'LATIN' in name else
          'japanese_cjk' if any(k in name for k in ['HIRAGANA','KATAKANA','CJK']) else 'other']+=1
    total=sum(c.values()); substantial=[k for k,v in c.items() if v>=20 and v/max(total,1)>=.1]
    group='mixed_scripts' if len(substantial)>=2 else 'cyrillic_dominant' if substantial==['cyrillic'] else 'latin_dominant_language_unknown' if substantial==['latin'] else 'other_or_unknown'
    return {'counts':dict(c),'group':group,'language':'unknown','heuristic':'At least20 letters and10% per substantial script, URLs/emails omitted; scripts do not identify language'}

def normalize_annotation(row,path,sha,dates):
    r=dict(row);r['original_local_path']=r['local_path'];r['local_path']=path.relative_to(ROOT).as_posix()
    rid=path.stem;reason=row['label_reason'].casefold();ham=row['spam_label']=='0'
    explicit_consent=ham and 'согласи' in reason
    # Confirmation of spam_label does not imply advertising consent or phishing/category truth.
    r.update(record_id=rid,raw_sha256=sha,human_spam_label=row['spam_label'],review_status='reviewed',
        initial_label_source='assistant_draft',binary_label_source='user_final_csv',
        category_status='auxiliary_only',phishing_label='unknown',consent='yes' if explicit_consent else 'unknown',
        expectedness='user_confirmed_expected' if ham and ('ожида' in reason or 'запрош' in reason) else 'unknown',
        consent_evidence='explicit_final_csv_reason' if explicit_consent else 'not_specified',language_self_reported='unknown',
        **{k:dates[k] or '' for k in ['date_header_iso','received_date_min_iso','received_date_max_iso']})
    require(set(r)==set(NORMALIZED_FIELDS),'Normalized CSV schema mismatch')
    return r

def annotation_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=NORMALIZED_FIELDS,quoting=csv.QUOTE_MINIMAL,extrasaction='raise')
        writer.writeheader();writer.writerows(rows)

def evaluated_metrics(labels,decisions):
    require(len(labels)==len(decisions),'Mismatched metric arrays')
    tp=fp=tn=fn=0
    for y,d in zip(labels,decisions):
        require(type(y) is int and y in {0,1} and type(d) is int and d in {0,1},'Non-binary metric input')
        tp+=int(y==1 and d==1);fp+=int(y==0 and d==1);tn+=int(y==0 and d==0);fn+=int(y==1 and d==0)
    undefined={}
    def divide(name,a,b):
        if b:return a/b
        undefined[name]='zero denominator on successful predictions';return None
    return {'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision':divide('precision',tp,tp+fp),
        'recall':divide('recall',tp,tp+fn),'F1':divide('F1',2*tp,2*tp+fp+fn),
        'FPR':divide('FPR',fp,fp+tn),'support':len(labels),'undefined_reasons':undefined}

def model_summary(records,predictions,ids):
    ids=set(ids);rows=[r for r in predictions if r['record_id'] in ids]
    require(len(rows)==len(ids) and len({r['record_id'] for r in rows})==len(ids),'Missing/repeated prediction rows')
    labels={r['record_id']:r['spam_label'] for r in records};ok=[r for r in rows if r['status']=='ok' and type(r['decision']) is int]
    m=evaluated_metrics([labels[r['record_id']] for r in ok],[r['decision'] for r in ok])
    return {'records':len(ids),'successful':len(ok),'coverage':len(ok)/len(ids) if ids else None,
        'unscored_ids':[r['record_id'] for r in rows if r not in ok],'conditional_on_success':len(ok)!=len(ids),
        'metrics':m,'FP_ids':[r['record_id'] for r in ok if labels[r['record_id']]==0 and r['decision']==1],
        'FN_ids':[r['record_id'] for r in ok if labels[r['record_id']]==1 and r['decision']==0]}

def campaign_metadata(records,parsed):
    groups=defaultdict(list); senders=defaultdict(list); near=[]; grams={}
    for r,p in zip(records,parsed):
        rid=r['record_id'];send=p.get('sender_domain')
        if send:senders[digest(send.casefold())].append(rid)
        for host in {x.get('hostname') or x.get('domain') for x in p.get('links',[]) if x.get('hostname') or x.get('domain')}:
            groups[digest(host.casefold())].append(rid)
        t=compare_text(re.sub(r'https?://\S+|\d+',' ',r['text']))
        grams[rid]={t[i:i+5] for i in range(max(0,len(t)-4))}
    for a,b in combinations(records,2):
        ga,gb=grams[a['record_id']],grams[b['record_id']]
        score=len(ga&gb)/len(ga|gb) if ga|gb else 0
        if score>=.55:near.append({'ids':[a['record_id'],b['record_id']],'char5gram_jaccard':score})
    exact=defaultdict(list)
    for r in records:exact[digest(compare_text(r['text']))].append(r['record_id'])
    return {'canonical_text_duplicates':[v for v in exact.values() if len(v)>1],
        'similar_text_pairs':near,'same_sender_domain_groups':[{'domain_sha256':k,'ids':v} for k,v in senders.items() if len(v)>1],
        'shared_link_host_groups':[{'host_sha256':k,'ids':v} for k,v in groups.items() if len(v)>1],
        'caution':'Weak campaign/infrastructure hints, not verified attacks; shared trackers/unsubscribe domains are not proof. No deduplication or label changes.'}

def protected_snapshot():
    files=set()
    for folder in ['reports','models','src','config','docs','tests']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(BASE):files.add(p)
    files.update(ROOT.glob('requirements*.txt'));files.add(ROOT/'.gitignore')
    files.update(ROOT/e/'pyvenv.cfg' for e in ['.venv','.venv-decision'])
    return {p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted(files) if p.exists()}

def clean_model_text(parsed,cfg):
    # Frozen Day5 cleaner is dependency-free and mathematically identical to Day4.
    from day5_common import clean_text
    return clean_text({'subject':parsed.get('subject'),'body_text':parsed.get('body_text'),
                       'feature_availability':{'subject':'observable'}},cfg)

def setup_offline():
    os.environ.update(HF_HOME=str(ROOT/'.cache/day4/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',USE_TF='0',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def prediction_row(record,candidate,status='ok',**kwargs):
    return {'record_id':record['record_id'],'model_id':candidate['name'],'spam_label':record['spam_label'],
        'label_status':'human_confirmed_final','label_source':record['label_source'],
        'label_reviewed_at':record['label_reviewed_at'],'threshold':candidate['threshold'],
        'temperature':candidate.get('temperature',1.),'score_semantics':candidate['score_semantics'],
        'status':status,'decision':None,'score':None,'insufficient_features':False,
        'missing_sender_history_features':'not_used_by_saved_text_only_model',**kwargs}
