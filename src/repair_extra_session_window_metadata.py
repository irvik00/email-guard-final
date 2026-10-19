"""Mechanical metadata-only correction; preserve all original outputs and predictions."""
import argparse
from extra_session_common import *

def main(out):
    boundary=SessionBoundary(out).install();load_protocol(out);corrections={}
    for model in ['Laya_day5','C_corrected']:
        source=out/f'visibility_{model}_windows.jsonl';dest=out/f'visibility_{model}_windows_corrected.jsonl'
        require(not dest.exists(),'Metadata correction already exists')
        rows=read_rows(source);pred=read_rows(out/f'visibility_{model}_predictions.jsonl');index=0;result=[];changed=0
        for r in pred:
            for widx in range(r['window_count']):
                original=rows[index];require(original['record_id']==r['record_id'] and original['window_index']==widx,'Unexpected original journal order')
                require(original['variant'] in {r['variant'],'C_windows_max'},'Unexpected metadata defect')
                fixed=dict(original);fixed['variant']=r['variant'];changed+=int(original['variant']!=fixed['variant']);result.append(fixed);index+=1
        require(index==len(rows),'Unused rows');write_rows(dest,result)
        corrections[model]={'original_file':source.name,'original_sha256':sha_file(source),
            'corrected_file':dest.name,'corrected_sha256':sha_file(dest),'rows':len(rows),'changed_variant_fields':changed,
            'predictions_sha256_unchanged':sha_file(out/f'visibility_{model}_predictions.jsonl')}
    write_json(out/'visibility_window_metadata_correction.json',{'utc':now(),
        'reason':'Mutable prefix dictionary shared by A and first C window; metadata expansion overwrote outer variant label. Only journal variant field affected; no input, score, threshold, decision or metric changed.',
        'rerun_inference':False,'original_files_retained':True,'corrections':corrections,'accesses':boundary.report()})
    print('Window metadata corrected without model inference',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);args=a.parse_args();main(args.out.resolve())
