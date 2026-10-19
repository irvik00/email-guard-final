"""Bounded, offline synthetic Laya diagnostics; never reads project datasets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HOME=str(ROOT / '.cache/day4/hf'), HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  USE_TF='0', DO_NOT_TRACK='1', TOKENIZERS_PARALLELISM='false')


def boundary(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(ROOT / 'data'):
            raise PermissionError('Diagnostics cannot open ANY project dataset')
    if event in {'socket.connect', 'socket.bind'}:
        raise PermissionError('Diagnostics cannot use network sockets')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def protected_snapshot():
    paths = set()
    for folder in ('reports/day4', 'models/day3', 'models/day4'):
        paths.update(p for p in (ROOT / folder).rglob('*') if p.is_file())
    for pattern in ('src/*day4*.py', 'src/decision_protocol.py', 'src/run_decision_pilot.py',
                    'src/prepare_decision_runtime.py', 'config/day4*.json', 'requirements*.txt',
                    'reports/DAY4_STATUS.md', '.venv/pyvenv.cfg', '.venv-decision/pyvenv.cfg'):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    return {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(paths)}


def make_plan(original_questions):
    messages = [
        ('en_notice', 'en', False,
         'Subject: Requested service appointment\nYou booked a boiler inspection for Tuesday at 10:00. This is the confirmation you requested. No purchase or action is required.'),
        ('en_unsolicited', 'en', True,
         'Subject: Unrequested bulk promotion\nWe bought your address from a mailing list and are sending this unsolicited advertisement to thousands of people. Buy our miracle wealth course today! You never subscribed.'),
        ('ru_notice', 'ru', False,
         'Тема: Подтверждение вашей записи\nВы записались на проверку счётчика во вторник в 10:00. Это запрошенное вами подтверждение. Покупать ничего не требуется.'),
        ('ru_unsolicited', 'ru', True,
         'Тема: Незапрошенная массовая реклама\nМы купили ваш адрес из базы и рассылаем это нежелательное рекламное письмо тысячам людей. Купите чудо-курс мгновенного богатства! Вы на эту рассылку не подписывались.'),
    ]
    minimal = {'spam': {'type': 'noul', 'instructions': 'Is this message unsolicited spam?'}}
    fact = {'fact': {'type': 'noul', 'instructions': 'Does the text say the code is 731?'}}
    rows = []
    for variant, questions in [('A_original_day4_question', original_questions), ('B_separate_minimal_question', minimal)]:
        for ident, lang, expected, text in messages:
            rows.append(dict(id=f'{variant}/{ident}', variant=variant, lang=lang,
                             expected_yes=expected, state=text, questions=questions))
    for lang, yes, no in [('en', 'The code is 731.', 'The code is 284.'),
                          ('ru', 'Код: 731.', 'Код: 284.')]:
        for expected, text in [(True, yes), (False, no)]:
            rows.append(dict(id=f'C_explicit_fact/{lang}_{expected}', variant='C_explicit_fact',
                             lang=lang, expected_yes=expected, state=text, questions=fact))
    for expected, text in [(True, 'The code is 731.'), (False, 'The code is 284.')]:
        rows.append(dict(id=f'D_state_dict/en_{expected}', variant='D_state_dict', lang='en',
                         expected_yes=expected, state={'body': text}, questions=fact))
    official_question = {'churn_risk': {'type': 'noul', 'instructions': 'Does the user threaten to cancel or leave?'}}
    for expected, text in [(True, 'Refund the duplicate charge, or we will cancel our plan.'),
                           (False, 'Thank you for the refund. We will keep our plan.')]:
        rows.append(dict(id=f'E_official_example_question/en_{expected}', variant='E_official_example_question',
                         lang='en', expected_yes=expected, state=text, questions=official_question))
    return {'policy': 'Fixed before inference; no datasets, fitting, calibration or selection by validation.',
            'official_example': 'Publisher SDK question, new synthetic states, direct multilingual loading; NOT default English Router replication.',
            'cases': rows}


def main():
    started = time.perf_counter()
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    sys.addaudithook(boundary)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'reports/laya_diagnostics' / run_id
    out.mkdir(parents=True, exist_ok=False)
    before = protected_snapshot()
    dump(out / 'preservation_before.json', before)
    metadata = json.loads((ROOT / 'reports/day4/setup/runtime_manifest.json').read_bytes())
    config = json.loads((ROOT / 'config/day4_experiments.json').read_bytes())['pilot']
    plan = make_plan(config['questions'])
    dump(out / 'fixed_plan.json', plan)
    checkpoint_checks = {}
    for rel, recorded in metadata['files'].items():
        path = ROOT / '.cache/day4/laya_weights' / rel
        checkpoint_checks[rel] = {'sha256': sha(path), 'size_bytes': path.stat().st_size,
                                 'matches_day4_manifest': sha(path) == recorded['sha256']}
    assert checkpoint_checks['multilingual/model.safetensors']['matches_day4_manifest']
    assert checkpoint_checks['multilingual/rl_agent_config.json']['matches_day4_manifest']

    import torch
    import laya
    from laya.common import QTYPES, build_sequence, render_options, serialize_state
    from decision_protocol import validate_noul
    torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_started = time.perf_counter()
    agent = laya.load(metadata['local_path'], device=device, fast=False)
    if agent.device.type == 'cuda':
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    assert not agent.model.training
    source_checks = {}
    for name, obj in [('to_internal', agent._to_internal), ('decode_answers', agent._decode_answers),
                      ('render_options', render_options), ('build_sequence', build_sequence),
                      ('serialize_state', serialize_state)]:
        source = inspect.getsource(obj)
        (out / f'sdk_{name}.py.txt').write_text(source, encoding='utf-8')
        source_checks[name] = {'installed_path': inspect.getsourcefile(obj),
                              'source_sha256': hashlib.sha256(source.encode()).hexdigest()}
    installed = Path(inspect.getsourcefile(agent.__class__)).parent
    for filename in ['agent.py', 'common.py']:
        reviewed = ROOT / '.cache/day4/runtime_review/laya' / filename
        source_checks[filename] = {'installed_sha256': sha(installed / filename),
                                   'matches_reviewed_day4_sdk': sha(installed / filename) == sha(reviewed)}

    # Capture the actual tensor delivered to the forward pass and its unrounded logits.
    # Return the original results unchanged: no monkey-patched model math/decoding.
    original_forward = agent._forward
    captured = {}

    def observed_forward(batch):
        logits, acts = original_forward(batch)
        captured.update(input_ids=batch['input_ids'].cpu().tolist(),
                        marker_pos=batch['marker_pos'].cpu().tolist(),
                        logits=logits.tolist(), act_probabilities=acts.tolist())
        return logits, acts

    agent._forward = observed_forward
    rows = []
    with (out / 'raw_responses.jsonl').open('w', encoding='utf-8') as log:
        for case in plan['cases']:
            captured.clear()
            qid, qdef = next(iter(case['questions'].items()))
            internal = agent._to_internal(qdef)
            options = render_options(internal)
            assert options[0].startswith('false:') and options[1].startswith('true:')
            state_ids = agent.tok(serialize_state(case['state']), add_special_tokens=False)['input_ids']
            seq, markers = build_sequence(agent.tok, case['state'], internal, 1024, 256)
            empty, _ = build_sequence(agent.tok, '', internal, 1024, 256)
            assert len(seq) == len(empty) + len(state_ids)
            assert seq[-len(state_ids)-1:-1] == state_ids  # Includes every state token.
            full_instruction = agent.tok('noul question: ' + qdef['instructions'], add_special_tokens=False)['input_ids']
            assert seq[1:markers[0]-1] == full_instruction  # No instruction truncation.
            call_started = time.perf_counter()
            raw = agent.predict(case['state'], case['questions'], max_len=1024, head_max_len=256)
            if agent.device.type == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter() - call_started
            assert captured['input_ids'][0] == seq
            assert captured['marker_pos'][0] == markers
            # Independently reproduce the SDK's true-option softmax, with the same stored temperature.
            # No lang argument/options override in these calls or the original worker.
            assert not agent.temperature_by_options
            z = torch.tensor(captured['logits'][0][:2], dtype=torch.float64) / agent.temperature[QTYPES['noul']]
            p_true = float(torch.softmax(z, dim=-1)[1])
            score = raw['answers'][qid]['noul']
            assert abs(score - p_true) <= 0.000051
            if qid == 'spam':
                assert validate_noul(raw) == score
            row = dict(case, raw_response=raw, score_yes=score, positive_at_fixed_half=score >= 0.5,
                       expected_agreement_at_half=(score >= 0.5) == case['expected_yes'],
                       render_options=options, actual_forward=captured.copy(),
                       decoded_actual_sequence=agent.tok.decode(seq), state_tokens=len(state_ids),
                       instruction_tokens=len(full_instruction), no_truncation=True,
                       independently_recomputed_p_true=p_true, seconds=seconds)
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            log.flush()
            rows.append(row)
            print(f"{case['id']}: expected={case['expected_yes']} raw_noul={score}", flush=True)
    after = protected_snapshot()
    dump(out / 'preservation_after.json', after)
    unchanged = before == after
    assert unchanged, 'Protected Day3/Day4 files changed'
    result = {'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.perf_counter() - started, 'model_load_seconds': load_seconds,
              'checkpoint': {k: metadata[k] for k in ('model_id', 'subfolder', 'revision', 'local_path')},
              'checkpoint_file_checks': checkpoint_checks, 'sdk_source_checks': source_checks,
              'versions': {name: importlib.metadata.version(name) for name in metadata['versions']},
              'real_device': str(agent.device), 'amp_enabled': agent.amp_enabled,
              'dtype': str(agent.dtype), 'model_training': agent.model.training,
              'temperature': list(agent.temperature), 'temperature_by_options': agent.temperature_by_options,
              'encoder_config': agent.model.encoder.config.to_dict(),
              'fixed_plan_sha256': sha(out / 'fixed_plan.json'), 'protected_file_count': len(before),
              'protected_unchanged': unchanged, 'all_forward_state_and_question_checks_passed': True,
              'all_mapping_checks_passed': True, 'datasets_opened': 0, 'network_connections': 0,
              'cases': rows}
    dump(out / 'results.json', result)
    dump(ROOT / 'reports/laya_diagnostics_latest.json', {'run_id': run_id, 'results': str(out / 'results.json')})
    print(f"Saved {out}; {len(rows)} calls; preserved {len(before)} files; elapsed {result['elapsed_seconds']:.1f}s")


if __name__ == '__main__':
    main()
