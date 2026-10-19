"""Fictitious .example fixtures only; these demos are not a quality benchmark."""
from extra_common import *
from sender_evidence import extract_evidence,PublicSuffix

def main():
    out,_=paths();dest=out/'sender_demo';dest.mkdir(exist_ok=True)
    psl=PublicSuffix(ROOT/'config/registry/public_suffix_list.dat')
    registry={'version':'fictitious-demo-only','brands':[{'brand_id':'demo-bank','aliases':['Demo Bank'],
        'official_web_domains':['bank.example'],'confirmed_mail_domains':[]}]}
    messages={
        'ordinary_notice':('Demo Bank <notice@bank.example>','Requested appointment',
            '<p>The appointment you requested is confirmed.</p><a href="https://bank.example/account">https://bank.example</a>'),
        'brand_mismatch':('Demo Bank <notice@bаnk.example>','Demo Bank account notice',
            '<p>Contact +7 (999) 123-45-67 or helper@public-mail.example.</p><a href="https://bank.example.evil.example/account">https://bank.example</a>'),
        'advertisement':('Local Store <offers@store.example>','Advertisement',
            '<p>Weekly sale. To unsubscribe contact help@store.example.</p>')}
    events=[{'user_id':'synthetic-user','sender':'notice@bank.example','trusted_received_at':'2025-01-01T00:00:00Z',
        'replied_at':'2025-01-02T00:00:00Z','opened_at':None,'user_label':'ham','user_label_at':'2025-01-02T00:00:00Z'},
        {'user_id':'synthetic-user','sender':'notice@bank.example','trusted_received_at':'2025-04-01T00:00:00Z',
         'user_label':'spam','user_label_at':'2025-04-02T00:00:00Z'}]
    write_json(dest/'fictitious_history.json',events)
    summaries={}
    for key,(sender,subject,body) in messages.items():
        raw=(f'From: {sender}\r\nSubject: {subject}\r\nReply-To: help@support.example\r\n'
             'Authentication-Results: trusted-looking.example; spf=pass; dkim=pass; dmarc=pass\r\n'
             'Content-Type: text/html; charset=utf-8\r\n\r\n'+body).encode()
        (dest/f'{key}.eml').write_bytes(raw)
        result=extract_evidence(raw,registry,psl,events if key=='ordinary_notice' else None,
            'synthetic-user','2025-02-01T00:00:00Z')
        write_json(dest/f'{key}_evidence.json',result)
        observations=result['observations']
        summaries[key]={'fictitious':True,'sender_domain':observations['sender_domain'],
            'mixed_scripts':observations['sender_domain_mixed_scripts'],'brand_claims':observations['brand_claims'],
            'contact_count':len(observations['contacts']) if observations['contacts'] is not None else None,
            'authentication_trusted':observations['authentication_results'][0]['trusted'],
            'history':result['history'],'phishing_label':result['phishing_label'],'spam_score_adjustment':None}
    write_json(out/'sender_demo_summary.json',{'status':'module_ready','quality_metrics':None,'cases':summaries,
        'real_registry_sha256':sha_file(ROOT/'config/registry/official_domains.json'),
        'psl_sha256':sha_file(ROOT/'config/registry/public_suffix_list.dat')})
    print('3 explicitly fictitious sender/history demos completed; no real quality metrics')

if __name__=='__main__':main()
