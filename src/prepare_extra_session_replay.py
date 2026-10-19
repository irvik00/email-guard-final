"""Seal a fresh replay of the same frozen pilot/contracts; never overwrite an old run."""
import argparse
from datetime import timedelta
from extra_session_common import *

a=argparse.ArgumentParser();a.add_argument('--source',required=True,type=Path);a.add_argument('--out',required=True,type=Path);args=a.parse_args()
source=args.source.resolve();out=args.out.resolve();require(out.is_relative_to(BASE) and source.is_relative_to(BASE) and out!=source,'Wrong run roots')
p=load_protocol(source);out.mkdir(parents=True,exist_ok=False);write_json(out/'preservation_before.json',protected_snapshot())
boundary=SessionBoundary(out).install()
for name in ['private_pilot_inputs.jsonl','pilot_membership.jsonl']:
    (out/name).write_bytes((source/name).read_bytes())
started=datetime.now(timezone.utc)
p.update(run_id=out.name,created_utc=started.isoformat(),session_start_utc=started.isoformat(),
    deadline_utc=(started+timedelta(hours=3)).isoformat(),status='sealed_replay_before_any_prediction',
    replay_of={'source_run':source.name,'protocol_sha256':sha_file(source/'EXTRA_SESSION_PROTOCOL.json'),'same_pilot_IDs_and_parameters':True})
p['code_sha256']={str(f.relative_to(ROOT)):sha_file(f) for f in ROOT.glob('src/*extra_session*.py')}
write_json(out/'EXTRA_SESSION_PROTOCOL.json',p);write_json(out/'protocol_lock.json',{'sha256':sha_file(out/'EXTRA_SESSION_PROTOCOL.json'),'sealed_utc':now()})
(out/'EXTRA_SESSION_PROTOCOL.md').write_text('# Повтор фиксированного дополнительного пилота\n\n'
    f"Зафиксирован {p['created_utc']} до inference; source run `{source.name}`, SHA `{sha_file(out/'EXTRA_SESSION_PROTOCOL.json')}`.\n\n"
    'Те же278 ID, модели, текстовые представления, вопрос/T/threshold и Qwen prompt/generation settings. '
    'Полный контракт в EXTRA_SESSION_PROTOCOL.json. Новых tests или personal нет. Deadline3ч от нового старта; '
    'повтор development-данных не даёт новой независимой оценки.\n',encoding='utf-8')
write_json(out/'prepare_accesses.json',boundary.report());print('SEALED REPLAY',out,flush=True)
