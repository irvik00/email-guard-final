"""Hash the prepared Linux amd64 wheels and resolve the official Python image digest."""
import argparse
from datetime import datetime,timezone
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--resolve-base',action='store_true');a=p.parse_args()
    folder=ROOT/'deployment/wheelhouse-linux';wheels={};packages={}
    for path in sorted(folder.glob('*.whl')):
        with zipfile.ZipFile(path) as z:
            member=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
            meta=BytesParser().parsebytes(z.read(member))
        name=meta['Name'].lower().replace('_','-');version=meta['Version']
        if name in packages:raise ValueError('Multiple wheels for '+name)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        packages[name]=(version,digest)
        wheels[path.name]={'sha256':digest,'bytes':path.stat().st_size,'name':name,'version':version}
    expected={}
    for line in (ROOT/'requirements-app.lock.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        name,version=line.split('==');expected[name.lower().replace('_','-')]=version
    assert {n:v for n,(v,h) in packages.items()}==expected,'Wheel set does not match locked app environment'
    (ROOT/'requirements-docker.lock.txt').write_text('\n'.join(f'{name}=={version} --hash=sha256:{digest}' for name,(version,digest) in sorted(packages.items()))+'\n',encoding='utf-8')
    result={'platform':'linux/amd64','python':'3.12','wheels':wheels,'total_bytes':sum(w['bytes'] for w in wheels.values()),
            'note':'Cross-platform pip download and metadata/hash checks only; Linux installation and Docker build not executed here'}
    (ROOT/'deployment/wheelhouse_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if a.resolve_base:
        tag='3.12.4-slim-bookworm'
        with urllib.request.urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull',timeout=30) as r:
            token=json.load(r)['token']
        accept='application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json'
        req=urllib.request.Request('https://registry-1.docker.io/v2/library/python/manifests/'+tag,headers={'Authorization':'Bearer '+token,'Accept':accept})
        with urllib.request.urlopen(req,timeout=30) as r:
            body=r.read();digest=r.headers['Docker-Content-Digest'];index=json.loads(body)
        if hashlib.sha256(body).hexdigest()!=digest.removeprefix('sha256:'):raise ValueError('Registry digest mismatch')
        platform=next(x for x in index['manifests'] if x['platform']['os']=='linux' and x['platform']['architecture']=='amd64')
        base={'repository':'docker.io/library/python','tag':tag,'index_digest':digest,'linux_amd64_manifest_digest':platform['digest'],
              'reference':'python:'+tag+'@'+digest,'resolved_utc':datetime.now(timezone.utc).isoformat(),
              'source':'Official Docker Registry API; manifest metadata only, no Docker image build or execution'}
        (ROOT/'deployment/base_image.json').write_text(json.dumps(base,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'linux_wheels':len(wheels),'bytes':result['total_bytes'],'base_resolved':a.resolve_base}))


if __name__=='__main__':main()
