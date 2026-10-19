"""Final report regeneration, code archive and artifact integrity; no inference or fit."""
import ast
import shutil
from extra_common import *

def main():
    out,models=paths();boundary=ExtraBoundary('verify').install()
    verify=read_json(out/'verification.json');require(verify['status']=='passed','Independent audit missing')
    require(read_json(out/'tests_execution.json')['status']=='passed','Actual tests failed')
    for name in ['independent_encoder_reload','independent_full_reload','tokenizer_reload']:
        require(read_json(out/f'C/{name}.json')['status']=='passed','C component reload missing')
    stages=read_json(out/'driver_status.json')
    had_audit_stop=any(s.get('stage')=='audit' and s.get('status')=='blocked' for s in stages['stages'])
    stages['post_driver_resolutions']={'audit':'passed; initial stop classified a newly authored A/B source as prior artifact, now both executed SHA revisions archived' if had_audit_stop else 'passed',
        'verification_sha256':sha_file(out/'verification.json'),'timestamp_utc':now(),
        'outer_validation_repeated':False,'training_repeated':False}
    write_json(out/'driver_status.json',stages)
    technical={'status':'resolved','unresolved_technical_blockers':[],
        'implementation_history_observed_in_run':'20261012T095633Z-7d01f7b8',
        'events':[
            {'issue':'Restricted environment DNS getaddrinfo failed for public E5 download','resolution':'Narrow public download with network permission, no email reads or outgoing private text; pinned verified cache now supports offline reproduction'},
            {'issue':'Mid-word SentencePiece cut-fragment retokenization differs','resolution':'Initial preprocessing failed before LR fit; inference retains original capped512 IDs, exact visible char span from original offsets; source/error archived'},
            {'issue':'Two E5-visible dev spans exceed Laya700, max1362','resolution':'Initial assert failed before control inference; exact-span supplemental control fixed at1500/2048, original main A contract/T/threshold unchanged'},
            {'issue':'Preservation snapshot included new current-run train_extra_laya.py before B extension','resolution':'Preserved initial hash; byte-identical A source and B source both archived/verified, all prior model/report files unchanged'},
            {'issue':'Single-vector vs bulk sklearn float32 dot rounding','resolution':'Measured max1.9073486328125e-6; component latency check uses explicit1e-5 tolerance, no model/threshold changes'},
            {'issue':'Report renderer closing bracket syntax error','resolution':'Corrected before final report generation; all Python sources AST parsed; no model or score changes'}],
        'data_limitations':['one E5 fit/calibration prefix collision needs_review, outer prefix collisions0',
            'C constrained T0.5 (unconstrained0.4654); historical outer FPR1.2448% at dev-policy',
            'sampled training RSS, not whole-process OS peak; latency is a local single-run measurement, not a load test',
            'reused validation/provisional RU channel and labels/one seed/pretraining overlap unknown']}
    write_json(out/'technical_notes.json',technical)
    sources=set((ROOT/'src').glob('*extra*.py'))|{ROOT/'src/sender_evidence.py',ROOT/'src/demo_sender_evidence.py'}
    sources.update(ROOT/'src'/name for name in ['day5_common.py','day5_laya_runtime.py','decision_protocol.py','read_email.py','normalization_schema.py'])
    sources.update((ROOT/'tests').glob('test_extra*.py'))
    source_hashes={};archive=out/'code_snapshots';archive.mkdir(exist_ok=True)
    for path in sorted(sources):
        ast.parse(path.read_text(encoding='utf-8'))
        rel=path.relative_to(ROOT).as_posix();target=archive/(rel.replace('/','__')+'.txt')
        shutil.copyfile(path,target);source_hashes[rel]=sha_file(path)
    write_json(out/'code_manifest.json',{'created_utc':now(),'source_sha256':source_hashes,
        'executed_A_training_source_sha256':sha_file(out/'A/training_source.py.txt'),
        'executed_B_training_source_sha256':sha_file(out/'B/training_source.py.txt'),
        'C_successful_training_code_sha256':read_json(out/'C/training_started.json')['code_sha256'],
        'D_training_code_sha256':read_json(out/'D/training_started.json')['code_sha256'],
        'protocol_sha256':sha_file(out/'protocol.json'),'config_e5_reference_sha256':sha_file(ROOT/'config/extra_e5_reference.json'),
        'registry_sha256':sha_file(ROOT/'config/registry/official_domains.json'),'psl_sha256':sha_file(ROOT/'config/registry/public_suffix_list.dat')})
    for rel,h in read_json(out/'C/training_started.json')['code_sha256'].items():
        require(sha_file(ROOT/rel)==h,'Successful C executed source changed')
    require(sha_file(ROOT/'src/extra_metadata.py')==read_json(out/'D/training_started.json')['code_sha256'],'D executed source changed')
    from report_extra import main as render
    render()
    names=['comparison.md','training.md','context.md','sender_features.md','metadata_experiment.md','PRESENTATION_NOTES_EXTRA.md','comparison.json','verification.json']
    hashes={}
    for name in names:
        require(sha_file(out/name)==sha_file(BASE/name),'Latest/archive reports differ')
        hashes[name]=sha_file(out/name)
    require(sha_file(out/'EXTRA_EXPERIMENTS_STATUS.md')==sha_file(ROOT/'reports/EXTRA_EXPERIMENTS_STATUS.md'),'Status copies differ')
    hashes['EXTRA_EXPERIMENTS_STATUS.md']=sha_file(out/'EXTRA_EXPERIMENTS_STATUS.md')
    frozen=read_json(out/'validation_started.json')['manifests_sha256']
    require(all(sha_file(out/b/'candidate_manifest.json')==h==sha_file(models/b/'candidate_manifest.json') for b,h in frozen.items()),'Frozen model/report manifest changed')
    write_json(out/'artifact_checks.json',{'status':'passed','reports_sha256':hashes,'code_ast_parsed':len(sources),
        'all_latest_archive_pairs_identical':True,'test_count':read_json(out/'tests_execution.json')['total_tests'],
        'manifests_still_frozen':True,
        'accesses':boundary.report()})
    print('Final artifact checks passed; no training/inference repeated')

if __name__=='__main__':main()
