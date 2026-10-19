"""Day-4 refusal wrapper; default remains the unchanged Day-3 model/threshold."""
import argparse
import json
import sys
from pathlib import Path

import joblib

from baseline_common import ROOT,require,sha_file,versions
from day4_common import Day4Boundary,clean_text,inference
from normalization_schema import make_model_text
from predict_email import load_bundle
from read_email import parse_email


def predict(email,bundle_path=None):
    email=Path(email).resolve()
    with Day4Boundary(ROOT,eml=email):
        require(email.stat().st_size<=20*1024*1024,'Email exceeds20 MiB')
        raw=email.read_bytes();parsed=parse_email(raw)
        if bundle_path is None:
            bundle=load_bundle(ROOT/'models/day3/baseline_bundle.joblib');text=make_model_text(parsed['subject'],parsed['body_text'])[0]
            threshold=bundle['thresholds']['provisional_threshold'];name='E0';builder='day3_original'
        else:
            path=Path(bundle_path).resolve();require(path.is_relative_to(ROOT/'models/day4/runs'),'Only own Day4 candidate artifacts allowed')
            manifest=json.loads((path.parent/'model_manifest.json').read_bytes())
            require(manifest['status']=='completed' and sha_file(path)==manifest['artifacts'][path.name]['sha256'],'Candidate SHA/status mismatch')
            bundle=joblib.load(path);require(bundle['versions']==versions(),'Candidate versions differ')
            require(bundle['code_sha256']['src/day4_common.py']==sha_file(ROOT/'src/day4_common.py'),'Candidate builder code differs')
            text=clean_text({'subject':parsed['subject'],'body_text':parsed['body_text'],'feature_availability':{'subject':'observable'}},bundle['config'])[0]
            threshold=bundle['threshold'];name=bundle['model_id'];builder=bundle['builder_name']
        result=inference(bundle['pipeline'],text,threshold)
        result.update(model_id=name,builder=builder,run_id=bundle['run_id'],parser_warnings=parsed['parse_warnings'],demo_only_no_mail_actions=True)
        return result


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('email',type=Path);parser.add_argument('--bundle',type=Path)
    args=parser.parse_args();print(json.dumps(predict(args.email,args.bundle),ensure_ascii=False,indent=2))
