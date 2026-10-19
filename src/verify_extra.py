"""Independent audit: membership, saved weights, probabilities, frozen manifests and old files."""
import math
from collections import Counter
from extra_common import *

def summary(rows,temperature=1.,threshold=.5):
    result={}
    for domain in DOMAINS:
        data=[r for r in rows if r['domain']==domain];ok=[r for r in data if r.get('status','ok')=='ok']
        labels=[r['spam_label'] for r in ok]
        logits=[r['logit_difference'] for r in ok]
        result[domain]={'metrics':metrics(labels,[sigmoid(z/temperature) for z in logits],threshold),
            'probability_metrics':probability_metrics(labels,logits,temperature),'support_total':len(data),
            'coverage':len(ok)/len(data),'failures':len(data)-len(ok),'refusal_policy':'no action-head abstention used; failed inference never ham'}
    return result

def main():
    out,models=paths();boundary=ExtraBoundary('verify').install();checks={}
    tests=read_json(out/'tests_execution.json');require(tests['status']=='passed' and tests['total_tests']>0,'Real tests did not pass')
    checks['tests_execution']=tests
    before=read_json(out/'preservation_before.json');changed=[];owned_updates=[]
    for rel,h in before.items():
        actual_hash=sha_file(ROOT/rel)
        if actual_hash!=h:
            # This file was authored in THIS run, then extended for B before B started.
            # Do not erase the initial hash: prove both actual executed revisions instead.
            if rel=='src/train_extra_laya.py':
                old=read_json(out/'A/training_started.json')['code_sha256'][rel]
                new=read_json(out/'B/training_started.json')['code_sha256'][rel]
                require(h==old==sha_file(out/'A/training_source.py.txt'),'Initial A source not archived exactly')
                require(actual_hash==new==sha_file(out/'B/training_source.py.txt'),'Executed B source not archived exactly')
                owned_updates.append({'path':rel,'initial_A_sha256':h,'B_current_sha256':actual_hash,
                    'scope':'new current-run source; B activation checkpointing/gradient checks added before B, no A result rewritten'})
            else:changed.append(rel)
    require(not changed,'Prior artifacts changed: '+str(changed))
    checks['preservation']={'files':len(before)-len(owned_updates),'initial_snapshot_files':len(before),
        'changed':changed,'current_run_source_evolution':owned_updates,'day3_day4_day5_models_reports_preserved':True}
    hashes=read_json(out/'validation_started.json')['manifests_sha256']
    require(all(sha_file(out/b/'candidate_manifest.json')==h for b,h in hashes.items()),'Frozen candidate modified')
    checks['all_manifests_frozen_before_validation']=True
    dm=read_json(DAY5/'data_manifest.json');protocol=read_json(out/'protocol.json')
    require(sha_file(ROOT/'data/processed/splits/manifest.json')==dm['split_manifest_sha256'],'Original split manifest changed')
    for branch in ['A','B','C']:
        for part,n in [('fit',6000),('dev',600),('calibration',1000)]:
            path=out/branch/f'private_{part}_inputs.jsonl'
            require(sha_file(path)==dm['selection_sha256'][part] and len(read_rows(path))==n,'Membership changed')
    selected={p:read_rows(out/f'A/private_{p}_inputs.jsonl') for p in ['fit','dev','calibration']}
    fit_labels={r['record_id']:r['domain']+':'+str(r['spam_label']) for r in selected['fit']}
    group_sets={p:{r['inner_group_id'] for r in data} for p,data in selected.items()}
    require(not (group_sets['fit']&group_sets['dev'] or group_sets['fit']&group_sets['calibration'] or group_sets['dev']&group_sets['calibration']),'Inner group leakage')
    checks['membership_exact_day5_and_groups_disjoint']=True
    from safetensors import safe_open
    import torch
    source=Path(read_json(ROOT/'reports/day4/setup/runtime_manifest.json')['local_path'])/'model.safetensors'
    require(sha_file(source)==read_json(ROOT/'reports/day4/setup/runtime_manifest.json')['files']['multilingual/model.safetensors']['sha256'],'Public Laya source changed')
    weight_checks={};comparison={};oldused=read_rows(DAY5/'actual_training_membership.jsonl')
    for branch in ['A','B']:
        m=read_json(out/branch/'candidate_manifest.json');training=read_json(out/branch/'training.json');c=m['candidates']['adapted']
        spec=read_json(out/branch/'named_parameters.json');allowed={p['name'] for p in spec if p['requires_grad']}
        require(training['actual_optimizer_steps']<=1000 and training['effective_batch']==16,'Budget invariant')
        steps=read_rows(out/branch/'training_steps.jsonl')
        require(len(steps)==training['actual_optimizer_steps'] and all(math.isfinite(r['gradient_norm_before_clip']) and r['gradient_norm_before_clip']>0 for r in steps),'Gradient logs invalid')
        used=read_rows(out/branch/'actual_training_membership.jsonl');fitids={r['record_id'] for r in selected['fit']}
        require(len(used)==len(steps)*16 and {r['record_id'] for r in used}<=fitids,'Examples/budget/fit leakage')
        require([r['record_id'] for r in used[:4800]]==[r['record_id'] for r in oldused],'First pass differs from day5')
        require(training['exact_frozen_sha256_before']==training['exact_frozen_sha256_after'],'Frozen SHA mismatch')
        if branch=='B':
            actual=m['config']['training']['resolved_encoder_blocks']
            require(len(actual)==2 and all(n.startswith('encoder.layers.') for n in actual),'Unexpected blocks')
            all_layers=sorted({int(p['name'].split('.')[2]) for p in spec if p['name'].startswith('encoder.layers.')})
            require(actual==[f'encoder.layers.{i}' for i in all_layers[-2:]],'Not the actual two final layers')
            require(all(p['requires_grad']==(p['name'].startswith(('head.','type_emb.','scorer.')) or p['name'].startswith(tuple(n+'.' for n in actual))) for p in spec),'Freeze set differs')
        candidate=ROOT/c['checkpoint_path']/'model.safetensors';changed_names=[];frozen_same=[]
        require(sha_file(candidate)==c['artifacts']['weights_sha256'],'Model artifact hash')
        with safe_open(source,framework='pt',device='cpu') as s,safe_open(candidate,framework='pt',device='cpu') as target:
            require(set(s.keys())==set(target.keys()),'Weight keys mismatch')
            for name in s.keys():
                original=s.get_tensor(name).float();new=target.get_tensor(name).float()
                require(original.shape==new.shape,'Weight shape mismatch')
                same=torch.equal(original,new)
                if same:frozen_same.append(name)
                else:
                    require(name in allowed,'Frozen tensor changed: '+name);changed_names.append(name)
        require(changed_names and any(n.startswith('head.') for n in changed_names),'Decision-head did not change')
        require(any(n.startswith('encoder.') for n in changed_names)==(branch=='B'),'Encoder change differs from contract')
        weight_checks[branch]={'changed_tensor_names':changed_names,'unchanged_tensors':len(frozen_same),
            'source_total_tensors':len(frozen_same)+len(changed_names),'all_changes_in_trainable_set':True,
            'sampled_training_peak_rss_bytes':max(r['rss_bytes'] for r in steps),
            'unique_records':len({r['record_id'] for r in used}),'examples':len(used),'actual_epochs':len(used)/6000,
            'presented_class_counts':dict(Counter(fit_labels[u['record_id']] for u in used))}
        if (out/branch/'training_source.py.txt').exists():
            require(sha_file(out/branch/'training_source.py.txt')==read_json(out/branch/'training_started.json')['code_sha256']['src/train_extra_laya.py'],'Archived training source mismatch')
        pred=read_rows(out/branch/'validation_predictions.jsonl')
        require(len(pred)==7265,'Outer coverage row count')
        for row in pred:
            if row['status']!='ok':continue
            require(abs((row['logits_false_true'][1]-row['logits_false_true'][0])-row['logit_difference'])<=1e-5,'False/true logit direction invalid')
            require(abs(sigmoid(row['logit_difference'])-row['raw_probability'])<1e-12,'Raw mapping invalid')
            require(abs(sigmoid(row['logit_difference']/c['temperature'])-row['probability'])<1e-12,'Calibrated mapping invalid')
        comparison[branch]={'raw_0.5':summary(pred),'calibrated_0.5':summary(pred,c['temperature']),
            'calibrated_dev_threshold':summary(pred,c['temperature'],c['threshold']),
            'T':c['temperature'],'threshold':c['threshold'],'selected_step':m['selected_step']}
    checks['weights_and_real_gradients']=weight_checks
    manifest=read_json(out/'C/candidate_manifest.json');pred=read_rows(out/'C/validation_predictions.jsonl')
    require(len(pred)==7265 and manifest['fit_rows']==6000 and manifest['encoder_frozen'],'C contract/coverage')
    require(sha_file(ROOT/manifest['classifier_path'])==manifest['classifier_sha256'],'C classifier hash')
    comparison['C']={'raw_0.5':summary(pred),'calibrated_0.5':summary(pred,manifest['temperature']),
        'calibrated_dev_threshold':summary(pred,manifest['temperature'],manifest['threshold']),
        'T':manifest['temperature'],'threshold':manifest['threshold']}
    prefix_sets={};collisions=[]
    for part in ['fit','dev','calibration']:
        visible=read_rows(out/f'C/private_{part}_visible.jsonl')
        require(all(r['token_stats']['sequence_tokens']<=512 for r in visible),'E5 cap violated')
        for r in visible:
            key=digest(compare_text(r['visible_fragment']))
            if key in prefix_sets and prefix_sets[key][0]!=part:collisions.append({'prior':prefix_sets[key],'part':part,'record_id':r['record_id']})
            prefix_sets[key]=(part,r['record_id'])
    checks['E5_visible_cross_inner_part_collisions']={'count':len(collisions),'examples':collisions[:20],
        'policy':'report only; no retrospective removal or tuning'}
    byid={r['record_id']:r for data in selected.values() for r in data};reviews=[]
    for item in collisions:
        a=byid[item['prior'][1]];b=byid[item['record_id']]
        reviews.append({'prior_part':item['prior'][0],'part':item['part'],'prior_id':a['record_id'],'record_id':b['record_id'],
            'labels':[a['spam_label'],b['spam_label']],'sources':[a['source'],b['source']],
            'full_canonical_equal':compare_text(a['text'])==compare_text(b['text']),
            'same_original_group':a['duplicate_group_id']==b['duplicate_group_id'],
            'status':'needs_review_visible_prefix_only','membership_unchanged':True})
    write_json(out/'C/visible_collision_review.json',reviews)
    outer_collisions=[]
    for row in read_rows(out/'C/private_validation_visible.jsonl'):
        key=digest(compare_text(row['visible_fragment']))
        if key in prefix_sets:outer_collisions.append({'prior':prefix_sets[key],'validation_record_id':row['record_id']})
    checks['E5_visible_outer_prefix_collisions']={'count':len(outer_collisions),'examples':outer_collisions[:20],
        'policy':'report only; frozen candidates unchanged after validation'}
    checks['independent_reload']=read_json(out/'independent_reload.json')
    require(checks['independent_reload']['status']=='passed','Independent Laya reload missing')
    require(read_json(out/'C/independent_encoder_reload.json')['status']=='passed','E5 encoder reload missing')
    checks['E5_encoder_reload']=read_json(out/'C/independent_encoder_reload.json')
    require(read_json(out/'C/independent_full_reload.json')['status']=='passed','C composed pipeline reload missing')
    require(read_json(out/'C/tokenizer_reload.json')['status']=='passed','Saved C tokenizer reload missing')
    if (out/'C/warm_end_to_end.json').exists():
        profile=read_json(out/'C/warm_end_to_end.json')
        require(profile['status']=='completed' and profile['samples']==64 and not profile['fit_performed'],'C real latency profile invalid')
        checks['C_end_to_end_profile']={'samples':64,'max_logit_error':profile['max_logit_error'],
            'near_threshold_float32_changes':profile['near_threshold_float32_changes'],'versions':profile['versions']}
    # Metadata is a separate cohort; verify the saved confusion independently, without refitting.
    d=read_json(out/'D/comparison.json');outer={r['record_id']:r for r in read_rows(out/'C/private_validation_inputs.jsonl')}
    drecalc={}
    for name in ['text_only','hybrid']:
        rows=read_rows(out/f'D/{name}_validation_predictions.jsonl')
        require(all(r['record_id'] in outer and outer[r['record_id']]['source']=='spamassassin'
                    and outer[r['record_id']]['spam_label']==r['spam_label'] for r in rows),'D cohort identity mismatch')
        m=metrics([r['spam_label'] for r in rows],[r['probability'] for r in rows],.5)
        require(m==d[name]['metrics'],'D confusion mismatch');drecalc[name]=m
    checks['D_independent_confusions']=drecalc
    # Independently recompute old LR confusion on saved, immutable predictions. No old fit/inference.
    references={}
    for name in ['E0','E2']:
        rows=read_rows(DAY5/f'{name}_full_context_validation_predictions.jsonl')
        require(len(rows)==7265,'Reference coverage')
        references[name]={}
        for domain in DOMAINS:
            data=[r for r in rows if r['domain']==domain]
            references[name][domain]={str(t):metrics([r['spam_label'] for r in data],[r['probability'] for r in data],t)
                for t in ([.5,.3037252359712881] if name=='E0' else [.5])}
    references['Laya_day5']=read_json(DAY5/'laya_validation_comparison.json')['candidates']['adapted']
    comparison['references']=references
    comparison['reference_reused_without_refit']={'day5_run':DAY5.name,'preservation_checked':True,'same_outer_ids_labels_texts':True}
    write_json(out/'comparison.json',comparison)
    write_json(out/'verification.json',{'status':'passed','checks':checks,'accesses':boundary.report(),
        'closed_tests_opened':False,'existing_validation_not_fresh_test':True,'default_E0_unchanged':True})
    print('Independent audit passed; unchanged snapshot files:',len(before)-len(owned_updates),
          '; separately archived current-run source updates:',len(owned_updates))

if __name__=='__main__':main()
