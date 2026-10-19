"""Download only explicitly named public E5 checkpoint and PSL; never reads email data."""
import argparse
import json
import urllib.request
from extra_common import *

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'EmailGuard-local-research'}),timeout=90) as response:
        return response.read()

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['e5','psl']);p.add_argument('--revision');a=p.parse_args()
    out,_=paths()
    if a.kind=='e5':
        repo='intfloat/multilingual-e5-small'
        pin=ROOT/'config/extra_e5_reference.json'
        revision=a.revision or (read_json(pin)['revision'] if pin.exists() else None)
        cached_manifest=ROOT/'.cache/extra/e5'/str(revision)/'source_manifest.json'
        if revision and cached_manifest.exists():
            cached=read_json(cached_manifest)
            require(cached['revision']==revision and cached['license']=='mit','Cached E5 reference mismatch')
            for name,spec in cached['files'].items():
                require(sha_file(Path(cached['local_path'])/name)==spec['sha256'],'Cached public file changed')
            write_json(out/'e5_source.json',cached|{'reused_verified_local_cache':True})
            print('Using verified pinned local E5 checkpoint '+revision);return
        url='https://huggingface.co/api/models/'+repo+('/revision/'+revision if revision else '')
        info=json.loads(get(url))
        revision=info['sha'];names={f['rfilename'] for f in info['siblings']}
        selected=['config.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json','sentencepiece.bpe.model','README.md']
        weights='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
        selected += [weights];dest=ROOT/'.cache/extra/e5'/revision;dest.mkdir(parents=True,exist_ok=True)
        records={}
        for name in selected:
            if name not in names:continue
            path=dest/name
            if not path.exists():
                url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
                print('Downloading public '+name,flush=True)
                # Streaming weights; no private input is ever attached to a request.
                with urllib.request.urlopen(url,timeout=120) as src,path.open('wb') as target:
                    while chunk:=src.read(1024*1024):target.write(chunk)
            records[name]={'sha256':sha_file(path),'bytes':path.stat().st_size}
        manifest={'model_id':repo,'revision':revision,'license':info.get('cardData',{}).get('license'),
            'local_path':str(dest),'source':'https://huggingface.co/'+repo,'downloaded_utc':now(),'files':records,
            'remote_code_executed':False}
        require(manifest['license']=='mit','Unexpected E5 license; inspect before use')
        write_json(out/'e5_source.json',manifest)
        write_json(dest/'source_manifest.json',manifest)
        if not pin.exists():write_json(pin,{'model_id':repo,'revision':revision,'license':manifest['license']})
        print(revision)
    else:
        url='https://publicsuffix.org/list/public_suffix_list.dat';dest=ROOT/'config/registry/public_suffix_list.dat'
        if dest.exists():
            require(sha_file(dest)==read_json(ROOT/'config/registry/psl_manifest.json')['sha256'],'Existing PSL snapshot changed')
            write_json(out/'psl_ready.json',read_json(ROOT/'config/registry/psl_manifest.json'))
            print('Using immutable local PSL snapshot');return
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(get(url))
        write_json(ROOT/'config/registry/psl_manifest.json',{'source_url':url,'checked_at':now(),'sha256':sha_file(dest),
            'license':'MPL-2.0 as declared in downloaded snapshot','include_private_suffixes':True})
        write_json(out/'psl_ready.json',read_json(ROOT/'config/registry/psl_manifest.json'))

if __name__=='__main__':main()
