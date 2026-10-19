"""Offline direct Laya worker in .venv-decision; JSON-lines on local stdio only."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.update(HF_HOME=str(ROOT/'.cache/day4/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',USE_TF='0',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false')


def boundary(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(ROOT/'data'):raise ValueError('Decision worker cannot open any dataset file')
    if event in {'socket.connect','socket.bind'}:raise ValueError('Offline decision worker cannot use network sockets')


def emit(value):
    sys.__stdout__.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');sys.__stdout__.flush()


def main():
    sys.stdin.reconfigure(encoding='utf-8');sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    sys.addaudithook(boundary)
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import psutil
        import laya
        from laya.common import build_sequence
        from decision_protocol import token_prefix,validate_noul
        torch.set_num_threads(1)
        metadata=json.loads((ROOT/'reports/day4/setup/runtime_manifest.json').read_bytes())
        config=json.loads((ROOT/'config/day4_experiments.json').read_bytes())['pilot']
        device='cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cuda':
            t=torch.ones((10,10),device=device);answer=(t@t)[0,0].item();torch.cuda.synchronize()
            if answer!=10.:raise ValueError('CUDA tensor smoke failed')
            torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();agent=laya.load(metadata['local_path'],device=device,fast=False)
        if agent.device.type=='cuda':torch.cuda.synchronize()
        cold=time.perf_counter()-started
        question=config['questions']['spam'];internal=agent._to_internal(question)
        empty_sequence,_=build_sequence(agent.tok,'',internal,config['context_limit'],config['head_max_len'])
        # Reserved head includes actual instruction/options; enforce no implicit state truncation.
        room=config['context_limit']-len(empty_sequence)
        cap=min(config['state_token_cap'],room)
        if cap<=0:raise ValueError('No state context remaining')
        emit({'event':'ready','device':str(agent.device),'gpu_name':torch.cuda.get_device_name() if agent.device.type=='cuda' else None,
            'torch':torch.__version__,'cuda_tensor_smoke':10. if device=='cuda' else None,'cold_model_load_seconds':cold,
            'rss_bytes_after_load':psutil.Process().memory_info().rss,'config':agent.cfg,
            'tokenizer_class':type(agent.tok).__name__,'tokenizer_model_max_length':agent.tok.model_max_length,
            'context_limit':config['context_limit'],'actual_state_room':room,'state_cap':cap,
            'revision':metadata['revision'],'weights_size_bytes':metadata['files']['multilingual/model.safetensors']['size_bytes'],
            'versions':metadata['versions']})
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            start=time.perf_counter()
            try:
                original=request['text'];sanitized=original.replace(agent.tok.mask_token,' ')
                fragment,stats=token_prefix(agent.tok,sanitized,cap)
                seq,_=build_sequence(agent.tok,fragment,internal,config['context_limit'],config['head_max_len'])
                require_tokens=len(agent.tok(fragment,add_special_tokens=False)['input_ids'])
                if len(seq)!=len(empty_sequence)+require_tokens:raise ValueError('Unexpected internal state truncation')
                prep_seconds=time.perf_counter()-start
                infer_start=time.perf_counter();response=agent.predict(fragment,config['questions'],max_len=config['context_limit'],head_max_len=config['head_max_len'])
                if agent.device.type=='cuda':torch.cuda.synchronize()
                infer_seconds=time.perf_counter()-infer_start;score=validate_noul(response)
                emit({'event':'prediction','record_id':request['record_id'],'status':'ok','spam_score':score,
                    'token_stats':stats,'mask_literal_replacements':original.count(agent.tok.mask_token),
                    'actual_fragment':fragment,'actual_fragment_sha256':hashlib.sha256(fragment.encode()).hexdigest(),
                    'preparation_seconds':prep_seconds,'backend_inference_seconds':infer_seconds,'backend_total_seconds':time.perf_counter()-start,
                    'rss_bytes':psutil.Process().memory_info().rss,'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,'real_device':str(agent.device)})
            except Exception as error:
                emit({'event':'prediction','record_id':request.get('record_id'),'status':'error','error':type(error).__name__+': '+str(error),'backend_total_seconds':time.perf_counter()-start})


if __name__=='__main__':
    try:main()
    except Exception as error:
        emit({'event':'startup_error','error':type(error).__name__+': '+str(error)})
        raise
