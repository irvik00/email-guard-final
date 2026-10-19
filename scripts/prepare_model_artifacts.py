"""Copy only frozen, hash-verified E0/Laya files into a relocatable model mount."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.contract import contract, sha_file


def prepare(source,destination,names):
    source=Path(source).resolve();destination=Path(destination).resolve();count=0;total=0
    for name in names:
        for rel,expected in contract()[name]['files'].items():
            src=(source/rel).resolve();dst=(destination/rel).resolve()
            if not src.is_relative_to(source/'models') or not dst.is_relative_to(destination/'models'):
                raise ValueError('Unsafe model path')
            if sha_file(src)!=expected:raise ValueError('Source artifact differs: '+rel)
            if dst.exists():
                if sha_file(dst)!=expected:raise ValueError('Destination differs; refusing overwrite: '+rel)
            else:
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,dst)
                if sha_file(dst)!=expected:raise ValueError('Copied artifact differs: '+rel)
            count+=1;total+=dst.stat().st_size
    return {'components':names,'files_verified':count,'bytes':total,'downloads':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT)
    p.add_argument('--destination',type=Path,default=ROOT/'artifacts')
    p.add_argument('--with-laya',action='store_true')
    p.add_argument('--only-laya',action='store_true')
    a=p.parse_args()
    names=['Laya_day5'] if a.only_laya else ['E0','Laya_day5'] if a.with_laya else ['E0']
    print(json.dumps(prepare(a.source,a.destination,names)))


if __name__=='__main__':main()
