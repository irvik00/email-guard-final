"""Measure actual frozen sklearn readout on cached inner-dev E5 vectors; never fits."""
import time
from extra_common import *

def main():
    out,_=paths();boundary=ExtraBoundary('train').install()
    import numpy as np
    import joblib
    manifest=read_json(out/'C/candidate_manifest.json');spec=read_json(out/'C/embeddings.json')
    path=ROOT/manifest['classifier_path'];require(sha_file(path)==manifest['classifier_sha256'],'LR changed')
    classifier=joblib.load(path)
    data=read_rows(out/'C/private_dev_inputs.jsonl');before=read_rows(out/'C/dev_raw.jsonl')
    matrix=np.load(ROOT/spec['parts']['dev']['matrix_path'],allow_pickle=False)
    indices=sorted(range(len(data)),key=lambda i:digest([42,'latency',data[i]['record_id']]))[:64]
    classifier.decision_function(matrix[:1]);rows=[];errors=[]
    for i in indices:
        start=time.perf_counter();logit=float(classifier.decision_function(matrix[i:i+1])[0])
        probability=sigmoid(logit/manifest['temperature']);decision=probability>=manifest['threshold']
        seconds=time.perf_counter()-start;errors.append(abs(logit-before[i]['logit_difference']))
        rows.append({'record_id':data[i]['record_id'],'domain':data[i]['domain'],'seconds':seconds})
    require(max(errors)<=1e-5,'Frozen LR probe changed beyond float32 single/bulk tolerance')
    write_json(out/'C/warm_lr_readout.json',{'status':'completed','rows':rows,'max_logit_error':max(errors),'float32_tolerance':1e-5,
        'same_64_ids_as_encoder_benchmark':True,'not_end_to_end':True,'fit_performed':False,'accesses':boundary.report()})
    print('Real frozen LR readout latency recorded on64 internal dev records')

if __name__=='__main__':main()
