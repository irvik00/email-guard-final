"""Offline observations, with provenance and unknowns. Never returns a phishing verdict."""
import csv
import ipaddress
import json
import re
import unicodedata
from datetime import datetime,timezone
from email.utils import parseaddr,getaddresses
from pathlib import Path
from urllib.parse import urlsplit
from read_email import parse_email,domain_metadata,label_domain

ROOT=Path(__file__).resolve().parents[1]

def normalize_domain(value):
    return domain_metadata(value)['domain']

def domain_within(host,base):
    host=normalize_domain(host);base=normalize_domain(base)
    return host==base or host.endswith('.'+base) if host and base else None

class PublicSuffix:
    """PSL prevailing-rule algorithm including wildcards, exceptions and private rules."""
    def __init__(self,path):
        self.exact=set();self.wild=set();self.exceptions=set()
        for raw in Path(path).read_text(encoding='utf-8').splitlines():
            rule=raw.strip()
            if not rule or rule.startswith('//'):continue
            target=self.exceptions if rule.startswith('!') else self.wild if rule.startswith('*.') else self.exact
            target.add(normalize_domain(rule.removeprefix('!').removeprefix('*.')))
    def organizational(self,domain):
        domain=normalize_domain(domain)
        if not domain:return None
        try:ipaddress.ip_address(domain);return None
        except ValueError:pass
        labels=domain.split('.');suffix=1  # PSL default '*' rule, explicitly includes reserved .example fixtures.
        for i in range(len(labels)):
            tail='.'.join(labels[i:]);size=len(labels)-i
            if tail in self.exceptions:suffix=size-1;break
            if tail in self.exact:suffix=max(suffix,size)
            if i>0 and tail in self.wild:suffix=max(suffix,size+1)
        return '.'.join(labels[-suffix-1:]) if len(labels)>suffix else None

def scripts(value):
    found=set()
    for c in value:
        if c.isalpha():
            name=unicodedata.name(c,'')
            found.update(s for s in ['LATIN','CYRILLIC','GREEK'] if s in name)
    return sorted(found)

def unicode_host(domain):
    try:return domain.encode('ascii').decode('idna') if domain else None
    except UnicodeError:return domain

def skeleton(text):
    # Limited diagnostic, never identity verification. Documented non-exhaustive confusables.
    return unicodedata.normalize('NFKC',text).casefold().translate(str.maketrans('аесорухіјΑΒΕΖΗΙΚΜΝΟΡΤΥΧ','aecopyxijabezhikmnoptyx'))

def timestamp(value):
    if not value:return None
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:raise ValueError('Timezone required for trusted local history')
    return dt.astimezone(timezone.utc)

def load_history(path):
    if path is None:return None
    path=Path(path)
    if path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    rows=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows,list):raise ValueError('History JSON must be an event list')
    return rows

def normalize_address(value):
    address=parseaddr(value or '')[1]
    if '@' not in address:return None
    local,domain=address.rsplit('@',1);host=normalize_domain(domain)
    # Local part is case-sensitive; no heuristic merging of distinct mailboxes.
    return local+'@'+host if host else None

def history_evidence(events,user_id,sender,cutoff):
    unknown={'state':'unknown','seen_before':None,'count':None,'previous_reply':None,
             'opened_observed_count':None,'user_spam_feedback':None,'whitelisted':False}
    if events is None:return unknown|{'reason':'history_not_provided'}
    if not user_id or not normalize_address(sender) or not cutoff:return unknown|{'reason':'trusted_cutoff_user_or_sender_missing'}
    limit=timestamp(cutoff);before=[];excluded=0
    for event in events:
        if event.get('user_id')!=user_id or normalize_address(event.get('sender'))!=normalize_address(sender):continue
        received=timestamp(event.get('trusted_received_at'))
        if received is None or received>=limit:excluded+=1;continue
        before.append(event)
    def state_before(key):
        observed=[]
        for event in before:
            t=timestamp(event.get(key))
            observed.append(t<limit and t>=timestamp(event['trusted_received_at']) if t is not None else None)
        return observed
    replies=state_before('replied_at');opens=state_before('opened_at')
    # A label may have been entered after receipt. Without its trusted event time it cannot be
    # used as a pre-cutoff signal, even when the message itself was received earlier.
    feedback=[e['user_label'] for e in before if e.get('user_label') in ['spam','ham']
              and timestamp(e.get('user_label_at')) is not None
              and timestamp(e['trusted_received_at'])<=timestamp(e['user_label_at'])<limit]
    return {'state':'observed_local_history','trusted_cutoff':limit.isoformat(),'seen_before':bool(before),'count':len(before),
        'scope':'supplied_local_history_only; account completeness unknown',
        'previous_reply':True if any(x is True for x in replies) else None if any(x is None for x in replies) else False,
        'opened_observed_count':sum(x is True for x in opens) if before and all(x is not None for x in opens) else None,
        'opened_missing_count':sum(x is None for x in opens),'user_spam_feedback':feedback or None,
        'feedback_missing_trusted_time':sum(e.get('user_label') in ['spam','ham'] and not e.get('user_label_at') for e in before),
        'excluded_missing_or_future_received':excluded,'whitelisted':False,
        'warning':'Local interaction is evidence only; opening/familiarity does not authenticate sender'}

def extract_evidence(raw,registry,psl,history=None,user_id=None,trusted_received_at=None):
    parsed=parse_email(raw);headers={}
    for item in parsed['headers_raw']:headers.setdefault(item['name'].lower(),[]).append(item['value'])
    display,address=parseaddr(parsed['sender'] or '')
    sender=parsed['sender_domain'];decoded=unicode_host(sender)
    def header_domains(name):
        if name not in headers:return {'domains':None,'state':'observable_missing','source':name}
        values=[normalize_domain(a.rsplit('@',1)[1]) for _,a in getaddresses(headers[name]) if '@' in a]
        return {'domains':values,'state':'observed','source':name}
    claimed=[];claimtext=unicodedata.normalize('NFKC',display+' '+(parsed['subject'] or '')).casefold()
    for brand in registry['brands']:
        if any(re.search(r'(?<!\w)'+re.escape(alias.casefold())+r'(?!\w)',claimtext) for alias in brand['aliases']):
            claimed.append(brand)
    relations=[]
    for brand in claimed:
        official=brand['official_web_domains'];mail=brand['confirmed_mail_domains']
        website=any(domain_within(sender,d) for d in official) if sender else None
        mailmatch=sender in mail if sender and mail else None
        sk=skeleton(decoded or '')
        similar=[d for d in official if skeleton(unicode_host(d) or '')==sk and sender!=d]
        links=[]
        for link in parsed['links']:
            host=link['domain'];shown=link['label_domain']
            links.append({'actual_domain':host,'label_domain':shown,'label_hostname_mismatch':link['label_mismatch'],
                'actual_org_domain':psl.organizational(host),'within_claimed_brand_web_domain':any(domain_within(host,d) for d in official) if host else None,
                'source':'static MIME HTML/text URL; never fetched'})
        relations.append({'brand_id':brand['brand_id'],'claim_source':'From display name / Subject alias match; unverified claim',
            'sender_within_web_domain':website,'sender_confirmed_mail_domain':mailmatch,
            'mail_relation':'confirmed_public_contact_domain_only' if mailmatch else 'unknown',
            'lookalike_candidates':similar,'links':links,'phishing_label':None,
            'warning':'Absent registry relation is unknown/candidate mismatch; partners and mailing providers possible'})
    body=parsed['body_text'];text=body or ''
    contacts=[]
    for kind,pattern in [('email',r'[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}'),('phone',r'(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)')]:
        for m in re.finditer(pattern,text):
            if kind=='phone' and not 7<=sum(c.isdigit() for c in m.group())<=15:continue
            contacts.append({'kind':kind,'value':m.group(),'context':text[max(0,m.start()-35):m.end()+35],
                'source':'static decoded body regex; approximate, not identity verification'})
    auth=[]
    for value in headers.get('authentication-results',[]):
        auth.append({'raw_claim':value,'authserv_id_claim':value.split(';',1)[0].strip(),
            'reported':re.findall(r'\b(spf|dkim|dmarc)\s*=\s*([a-zA-Z]+)',value,re.I),
            'trusted':None,'trust_state':'unknown_untrusted_upload','cryptographically_verified':False})
    letters=[c for c in text if not c.isspace()]
    unusual=sorted(set(c for c in text if unicodedata.category(c) in {'Cf','Cc'} and c not in '\n\t\r'))
    return {'schema_version':1,'observations':{'display_name':display or None,'sender_address':address or None,
        'sender_domain':sender,'sender_organizational_domain':psl.organizational(sender),
        'sender_domain_scripts':scripts(decoded) if decoded else None,
        'sender_domain_mixed_scripts':any(len(scripts(label))>1 for label in decoded.split('.')) if decoded else None,
        'reply_to':header_domains('reply-to'),'return_path':header_domains('return-path'),
        'sender_brand_relations':[{'brand_id':b['brand_id'],
            'within_public_web_domain':any(domain_within(sender,d) for d in b['official_web_domains']) if sender else None,
            'confirmed_public_contact_mail_domain':sender in b['confirmed_mail_domains'] if sender and b['confirmed_mail_domains'] else None,
            'authenticates_message':False} for b in registry['brands']],
        'brand_claims':relations,'contacts':contacts if body is not None else None,
        'digit_ratio':sum(c.isdigit() for c in letters)/len(letters) if letters else None,
        'unusual_codepoints':[f'U+{ord(c):04X}' for c in unusual] if body is not None else None,
        'unsubscribe':parsed['unsubscribe'],'attachments':parsed['attachments'],'formatting':parsed['formatting'],
        'authentication_results':auth if auth else None,'dkim_signature_present':'dkim-signature' in headers,
        'dkim_signature_verified':None,'auth_source':'raw MIME headers; no receiver trust boundary'},
        'history':history_evidence(history,user_id,address,trusted_received_at),
        'phishing_label':None,'spam_score_adjustment':None,'links_opened':False,
        'registry_version':registry['version'],'sources':{'sender':'parsed From','body':'static MIME parser',
            'authentication':'RFC8601; uploaded headers can be forged','history':'optional local file, trusted time only'}}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('eml',type=Path);p.add_argument('--history',type=Path)
    p.add_argument('--user-id');p.add_argument('--trusted-received-at');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    path=a.eml.resolve()
    if path.suffix.lower()!='.eml':raise ValueError('Explicit .eml required')
    if path.is_relative_to(ROOT/'data') and not path.is_relative_to(ROOT/'data/personal_review'):
        raise ValueError('Corpus data are closed for the personal evidence CLI')
    if a.eml.stat().st_size>20*1024*1024:raise ValueError('Maximum .eml size is 20 MiB')
    registry=json.loads((ROOT/'config/registry/official_domains.json').read_text(encoding='utf-8'))
    result=extract_evidence(a.eml.read_bytes(),registry,PublicSuffix(ROOT/'config/registry/public_suffix_list.dat'),
                            load_history(a.history),a.user_id,a.trusted_received_at)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
