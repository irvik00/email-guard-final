"""Anonymous public-weight download only; no corpus input or inference calls."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
os.environ.update(HF_HOME=str(ROOT/'.cache/day4/hf'),HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_XET='1',USE_TF='0',DO_NOT_TRACK='1')


def main():
    from huggingface_hub import snapshot_download
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',default='7b928d828b7b0e022f929d9bd2e44165aa270148')
    args=parser.parse_args()
    start=time.perf_counter();repo='convaiinnovations/laya'
    with urlopen('https://huggingface.co/api/models/'+repo+'/revision/'+args.revision,timeout=60) as response:info=json.load(response)
    revision=info['sha'];license=info.get('cardData',{}).get('license')
    if license!='apache-2.0':raise ValueError('Unexpected model license: '+str(license))
    destination=ROOT/'.cache/day4/laya_weights'
    snapshot_download(repo,revision=revision,token=False,local_dir=destination,
        allow_patterns=['README.md','LICENSE','multilingual/rl_agent_config.json','multilingual/model.safetensors','multilingual/tokenizer/*','multilingual/encoder/config.json'])
    metadata={'model_id':repo,'subfolder':'multilingual','revision':revision,'license':license,'local_path':str(destination/'multilingual'),
        'download_seconds':time.perf_counter()-start,'source':'https://huggingface.co/'+repo+'/tree/'+revision+'/multilingual',
        'versions':{n:importlib.metadata.version(n) for n in ['laya','torch','transformers','tokenizers','huggingface_hub','safetensors']},'files':{}}
    for path in destination.rglob('*'):
        if path.is_file() and '.cache' not in path.relative_to(destination).parts:
            h=hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
            metadata['files'][path.relative_to(destination).as_posix()]={'sha256':h.hexdigest(),'size_bytes':path.stat().st_size}
    upstream_revision='548fe076fa0b6b65417585879da93ae41491aa38'
    upstream={}
    for name in ['catalog.py','laya_runtime.py']:
        upstream_path='studio/backend/core/systemone/'+name
        url='https://raw.githubusercontent.com/unslothai/unsloth/'+upstream_revision+'/'+upstream_path
        with urlopen(url,timeout=45) as response:source=response.read()
        (ROOT/'reports/day4/setup'/('unsloth_'+name+'.txt')).write_bytes(source)
        upstream[name]={'url':url,'sha256':hashlib.sha256(source).hexdigest()}
    metadata['official_unsloth_checkpoint_evidence']={'revision':upstream_revision,'files':upstream,
        'observation':'Official catalog names convaiinnovations/laya, multilingual; runtime uses _load_laya with that subfolder. Same public checkpoint family, direct publisher SDK instead of Studio.'}
    output=ROOT/'reports/day4/setup/runtime_manifest.json';output.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k!='files'}))


if __name__=='__main__':main()
