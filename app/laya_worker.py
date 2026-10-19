"""Private stdin/stdout worker; never accepts model paths or network requests."""
import contextlib
import json
import math
import sys
from .contract import ROOT, verify_component, unavailable, artifact_path


def send(value):
    sys.__stdout__.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    sys.__stdout__.flush()


def main():
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import importlib.metadata
            spec = verify_component('Laya_day5')
            for name, expected in spec['versions'].items():
                if importlib.metadata.version(name) != expected:
                    raise ValueError('runtime_version_mismatch')
            from day5_laya_runtime import encode_rows, infer_encoded
            import torch
            import laya
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError('cuda_unavailable')
            agent = laya.load(str(artifact_path(spec['checkpoint_path'])), device='cuda', fast=False)
            agent.model.eval()
            for param in agent.model.parameters():
                param.requires_grad_(False)
            config = {'questions': spec['questions'], 'input': spec['input_contract']}
        send({'event': 'ready', 'device': str(agent.device)})
    except Exception as error:
        send({'event': 'unavailable', 'reason': type(error).__name__})
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            text = request['text']
            if not isinstance(text, str) or len(text) > 12 * 1024 * 1024:
                raise ValueError('invalid_text')
            if not text.strip():
                send(unavailable('Laya_day5', 'empty_model_text', 'insufficient_features'))
                continue
            # Legacy metadata slots are constant placeholders, not email labels.
            row = {'record_id': 'upload', 'domain': 'application', 'source': 'in_memory', 'spam_label': 0,
                   'duplicate_group_id': 'unused', 'text': text}
            with contextlib.redirect_stdout(sys.stderr):
                encoded, visible = encode_rows(agent, [row], config)
                prediction = infer_encoded(agent, encoded, temperature=spec['temperature'])[0]
            score = prediction['probability']
            if not math.isfinite(score):
                raise ValueError('nonfinite_score')
            send({'model_id': 'Laya_day5', 'status': 'ok', 'reason': 'threshold',
                  'decision': int(score >= spec['threshold']), 'score': score,
                  'threshold': spec['threshold'], 'temperature': spec['temperature'],
                  'builder': spec['builder'], 'score_semantics': spec['score_semantics'],
                  'logit_difference': prediction['logit_difference'],
                  'model_text_chars': len(text), 'truncated': visible[0]['token_stats']['truncated'],
                  'token_stats': visible[0]['token_stats'], 'sequence_tokens': visible[0]['sequence_tokens'],
                  'mask_literal_replacements': visible[0]['mask_literal_replacements']})
        except Exception:
            send(unavailable('Laya_day5', 'inference_error', 'error'))


if __name__ == '__main__':
    main()
