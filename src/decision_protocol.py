"""Pure local decision contract and deterministic tokenizer-offset prefix limiting."""
import json
import math
from urllib.parse import urlparse
from urllib.request import Request,build_opener,ProxyHandler,HTTPRedirectHandler


def validate_noul(response):
    answer=response.get('answers',{}).get('spam',{})
    value=answer.get('noul')
    if answer.get('type')!='noul' or type(value) not in {int,float} or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('Invalid noul answer: expected finite yes-probability in[0,1]; confidence is not spam_score')
    return float(value)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('Local endpoint redirect forbidden')


def local_request(url,text,questions,timeout=30):
    parsed=urlparse(url)
    if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.username or parsed.password or parsed.path!='/v1/systemone':
        raise ValueError('Only literal http://127.0.0.1:port/v1/systemone allowed')
    request=Request(url,json.dumps({'model':'laya-multilingual','state':text,'questions':questions}).encode(),{'Content-Type':'application/json'})
    # Never route corpus text through environment proxy or external redirect.
    with build_opener(ProxyHandler({}),NoRedirect()).open(request,timeout=timeout) as response:
        body=response.read(1024*1024+1)
    if len(body)>1024*1024:raise ValueError('Oversized local answer')
    result=json.loads(body);validate_noul(result);return result


def token_prefix(tokenizer,text,cap):
    if type(cap) is not int or cap<0:raise ValueError('Invalid token cap')
    result=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=result['input_ids'];offsets=result['offset_mapping']
    if len(ids)<=cap:return text,{'original_tokens':len(ids),'visible_tokens':len(ids),'truncated':False,'visible_chars':len(text),'original_chars':len(text)}
    end=offsets[cap-1][1] if cap else 0
    prefix=text[:end]
    # Tokenization at a new end boundary can change; explicitly check real tokens.
    while prefix and len(tokenizer(prefix,add_special_tokens=False)['input_ids'])>cap:
        end-=1;prefix=text[:end]
    visible=len(tokenizer(prefix,add_special_tokens=False)['input_ids'])
    return prefix,{'original_tokens':len(ids),'visible_tokens':visible,'truncated':True,'visible_chars':len(prefix),'original_chars':len(text)}
