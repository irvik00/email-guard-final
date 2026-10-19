"""Apply only a locally generated historical baseline bundle to one explicit .eml."""
import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

from baseline_common import ROOT, DataBoundary, require, sha_file, versions
from normalization_schema import make_model_text
from read_email import parse_email


def load_bundle(path,root=ROOT):
    root=Path(root).resolve();path=Path(path).resolve();trusted=root/'models/day3'
    require(path.is_relative_to(trusted) and path.name=='baseline_bundle.joblib','Only this project\'s own baseline bundle is accepted')
    manifest_path=path.parent/'model_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['status']=='completed','Model run not completed')
    require(sha_file(path)==manifest['artifacts']['baseline_bundle.joblib']['sha256'],'Bundle hash differs from model manifest')
    # Joblib is loaded only after path/hash checks against our own generated manifest.
    bundle=joblib.load(path)
    require(bundle['run_id']==manifest['run_id'] and list(bundle['pipeline'].classes_)==[0,1],'Bundle run/classes mismatch')
    require(bundle['versions']==versions(),'Installed numerical/model versions differ from saved bundle')
    for name in ['src/read_email.py','src/normalization_schema.py']:
        require(bundle['code_sha256'][name]==sha_file(ROOT/name),'Parser/model-text builder changed since training')
    return bundle


def predict_raw(bundle,raw,threshold=None):
    require(len(raw)<=20*1024*1024,'Email exceeds 20 MiB input limit')
    parsed=parse_email(raw)
    text,changes=make_model_text(parsed['subject'],parsed['body_text'])
    pipeline=bundle['pipeline']
    vector=pipeline['tfidf'].transform([text])
    classes=list(pipeline.classes_)
    score=float(pipeline['classifier'].predict_proba(vector)[0,classes.index(1)])
    operating=bundle['thresholds']['provisional_threshold'] if threshold is None else float(threshold)
    require(math.isfinite(operating),'Threshold must be finite')
    diagnostics=['historical_English_text_only_baseline; language support not established','spam_score is not a calibrated guarantee']
    if not text.strip():diagnostics.append('empty_model_text; intercept/prior prediction retained')
    if vector.nnz==0:diagnostics.append('empty_vector; intercept/prior prediction retained')
    if any('\u0400'<=c<='\u04ff' for c in text):diagnostics.append('Cyrillic_present; Russian support not evaluated')
    if not any('a'<=c<='z' or 'A'<=c<='Z' for c in text):diagnostics.append('no_ASCII_Latin_letters; language unknown/out_of_scope')
    return {'model_id':bundle['model_id'],'run_id':bundle['run_id'],'spam_score':score,
        'threshold':operating,'threshold_policy':bundle['thresholds']['provisional_policy'] if threshold is None else 'explicit_override',
        'spam_verdict':int(score>=operating),'phishing':'not_available','category':'not_available',
        'active_features':int(vector.nnz),'model_text_char_count':len(text),'builder_transformations':changes,
        'parser_warnings':parsed['parse_warnings'],'limitations':diagnostics,'fit_performed':False}


def predict_path(bundle_path,email_path,root=ROOT,threshold=None):
    email_path=Path(email_path).resolve()
    with DataBoundary(root,eml=email_path) as boundary:
        bundle=load_bundle(bundle_path,root)
        require(email_path.stat().st_size<=20*1024*1024,'Email exceeds 20 MiB input limit')
        result=predict_raw(bundle,email_path.read_bytes(),threshold)
        result['data_access_log']=dict(boundary.accesses)
        return result


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('email',type=Path)
    parser.add_argument('--bundle',type=Path,default=ROOT/'models/day3/baseline_bundle.joblib')
    parser.add_argument('--threshold',type=float)
    args=parser.parse_args()
    try: result=predict_path(args.bundle,args.email,threshold=args.threshold)
    except (ValueError,OSError,KeyError) as error:parser.exit(1,f'Prediction blocked: {error}\n')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
