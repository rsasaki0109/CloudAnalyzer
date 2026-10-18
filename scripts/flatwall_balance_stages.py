"""Independent fixed-state correspondence, matrix and timing accounting checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np

TIMES = ['baseline_ns','balanced_ns','search_ns','count_ns','accumulate_ns','destroy_ns','staged_ns']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in saved.files}


def oracle(root):
    spec = importlib.util.spec_from_file_location('independent_fixed_state_search',root/'scripts/flatwall_balance_dynamics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state(root, origin):
    path = root/origin['original_npz']
    if sha(path) != origin['original_npz_sha256']:
        raise ValueError('original fixed state hash differs')
    data = arrays(path)
    if origin['kind'] == 'final':
        metadata = read(root/origin['metadata'])
        source = data['source_sensor'] if metadata['alignment_eligible'] else np.empty((0,3))
        if origin['alignment_eligible'] != metadata['alignment_eligible']:
            raise ValueError('original alignment eligibility differs')
        pose = data['pose']
        threshold, kernel = metadata['threshold_sq'], metadata['kernel']
        expected_h, expected_b = data['normal_matrix'], data['right_hand_side']
    elif origin['kind'] == 'step' and origin['step'] in [0,499]:
        source, pose = data['source_at_prediction'], data['before'][origin['step']]
        threshold, kernel = float(data['threshold_sq']), float(data['kernel'])
        expected_h, expected_b = data['normal_matrix'][origin['step']], data['right_hand_side'][origin['step']]
    else:
        raise ValueError('unpromised state kind')
    cloud, center = data['map_points_world'], data['center_world']
    payload = np.concatenate([pose.ravel(),center,[threshold,kernel,1.,20.],source.ravel(),cloud.ravel()]).astype('<f8').tobytes()
    if hashlib.sha256(payload).hexdigest() != origin['binary_sha256']:
        raise ValueError('binary input is not the original fixed state')
    return dict(source=source,pose=pose,cloud=cloud,center=center,threshold=threshold,kernel=kernel,
        expected_h=expected_h,expected_b=expected_b,original=data)


def expected(root, origins):
    independent = oracle(root)
    expected_states = []
    for origin in origins:
        item = state(root,origin)
        world = independent.transformed(item['source'],item['pose'])
        search = independent.Search(item['cloud'])
        ids, target_ids, squared = search.matches(world,item['threshold'],exhaustive=True)
        targets = item['cloud'][target_ids]
        h,b,multiplicity,weights,_,_ = independent.system(world,ids,targets,item['center'],item['kernel'])
        np.testing.assert_allclose(h,item['expected_h'],rtol=5e-12,atol=2e-8)
        np.testing.assert_allclose(b,item['expected_b'],rtol=5e-12,atol=2e-8)
        old = item['original']
        if origin['kind'] == 'final':
            np.testing.assert_array_equal(ids,old['matched_source_indices'])
            np.testing.assert_array_equal(targets,old['target_world'])
            np.testing.assert_array_equal(multiplicity,old['target_multiplicity'])
            np.testing.assert_allclose(weights,old['weights'],rtol=5e-12,atol=2e-12)
        elif origin['step'] == 499:
            start,end = old['tail_offsets'][-2:]
            np.testing.assert_array_equal(ids,old['tail_source_indices'][start:end])
            np.testing.assert_array_equal(targets,item['cloud'][old['tail_target_indices'][start:end]])
            np.testing.assert_array_equal(multiplicity,old['tail_multiplicity'][start:end])
        arm, residual = world[ids]-item['center'],world[ids]-targets
        j = np.zeros((len(ids),3,6))
        j[:,:,:3] = np.eye(3)
        x,y,z = arm.T
        j[:,0,4],j[:,0,5],j[:,1,3],j[:,1,5],j[:,2,3],j[:,2,4] = z,-y,-z,x,y,-x
        robust = item['kernel']**2/(item['kernel']+squared)**2
        baseline_h = np.einsum('nri,n,nrj->ij',j,robust,j)
        baseline_b = -np.einsum('nri,n,nr->i',j,robust,residual)
        expected_states.append(dict(name=origin['name'],source_points=len(item['source']),map_points=len(item['cloud']),
            source_indices=ids,targets=targets,distance_sq=squared,multiplicity=multiplicity,
            balanced_h=h,balanced_b=b,baseline_h=baseline_h,baseline_b=baseline_b))
    return expected_states


def check_run(run, expected_states, threads):
    if (run['schema'],run['threads'],run['repetitions']) != ('cloudanalyzer.odometry_stage_profile.v1',threads,32):
        raise ValueError('native run identity differs')
    if len(run['validation']) != 64 or len(run['samples']) != 64*6:
        raise ValueError('incomplete native state/sample inventory')
    maximum = 0.
    for observed,wanted in zip(run['validation'],expected_states):
        for name in ['name','source_points','map_points']:
            if observed[name] != wanted[name]:
                raise ValueError('fixed state identity differs')
        for name in ['source_indices','targets','multiplicity']:
            value = np.asarray(observed[name]).reshape(-1,3) if name=='targets' else observed[name]
            np.testing.assert_array_equal(value,wanted[name])
        np.testing.assert_allclose(observed['distance_sq'],wanted['distance_sq'],rtol=5e-12,atol=2e-12)
        for name in ['baseline_h','baseline_b','balanced_h','balanced_b']:
            np.testing.assert_allclose(observed[name],wanted[name],rtol=5e-12,atol=2e-8)
        errors = []
        for suffix in ['h','b']:
            native = np.asarray(observed['balanced_'+suffix])
            decomposed = np.asarray(observed['decomposed_'+suffix])
            np.testing.assert_allclose(decomposed,native,rtol=5e-12,atol=5e-9)
            errors.append(float(np.max(np.abs(native-decomposed))))
        if observed['maximum_decomposition_abs_error'] != max(errors):
            raise ValueError('native decomposition error differs')
        maximum = max(maximum,max(errors))
    for index,sample in enumerate(run['samples']):
        round_index,state_index = divmod(index,64)
        order = [(round_index+state_index+i)%3 for i in range(3)]
        if (sample['round'],sample['state_index'],sample['warmup'],sample['repetitions'],sample['order']) != (
                round_index,state_index,round_index==0,32,order):
            raise ValueError('sample order, warmup or repetition count differs')
        if any(type(sample[key]) is not int or sample[key] <= 0 for key in TIMES):
            raise ValueError('invalid native elapsed time')
        if sum(sample[key] for key in TIMES[2:6]) > sample['staged_ns']:
            raise ValueError('stage times exceed their outer window')
        for name in ['maximum_baseline_abs_error','maximum_balanced_abs_error','maximum_staged_abs_error']:
            if not np.isfinite(sample[name]) or not 0 <= sample[name] <= 1e-6:
                raise ValueError('native batch last-result self-check differs')
    return dict(states_checked=64,ordered_pair_rows_checked=sum(len(s['source_indices']) for s in expected_states),
        samples_checked=64*6,maximum_decomposition_abs_error=maximum)


def summarize(runs, origins):
    if [r['threads'] for r in runs] != [1,4,4,1]:
        raise ValueError('counterbalanced process order differs')
    groups = dict(all=list(range(64)),eligible=[],legacy_final=[],fine_final=[],capped_steps=[],ineligible=[])
    per_state = []
    for i,origin in enumerate(origins):
        group = 'ineligible' if not origin['alignment_eligible'] else 'capped_steps' if origin['kind']=='step' else 'fine_final' if origin['case'].endswith('0.1875') else 'legacy_final'
        groups[group].append(i)
        if origin['alignment_eligible']:
            groups['eligible'].append(i)
        item = dict(name=origin['name'],group=group,threads={})
        for threads in [1,4]:
            samples = [s for r in runs if r['threads']==threads for s in r['samples'] if s['state_index']==i and not s['warmup']]
            if len(samples) != 10:
                raise ValueError('require all ten measured blocks per state/thread')
            item['threads'][str(threads)] = {key:statistics.median(s[key]/32/1000 for s in samples) for key in TIMES}
        per_state.append(item)
    results = {}
    for name,ids in groups.items():
        result = dict(states=len(ids),threads={})
        for threads in [1,4]:
            samples = [s for r in runs if r['threads']==threads for s in r['samples'] if s['state_index'] in ids and not s['warmup']]
            sums = {key:sum(per_state[i]['threads'][str(threads)][key] for i in ids) for key in TIMES[:2]}
            outer = sum(s['staged_ns'] for s in samples)
            components = {key:sum(s[key] for s in samples)/outer for key in TIMES[2:6]}
            components['clock_and_boundary_ns'] = 1-sum(components.values())
            result['threads'][str(threads)] = dict(
                sum_state_median_baseline_us=sums['baseline_ns'],sum_state_median_balanced_us=sums['balanced_ns'],
                balanced_over_baseline=sums['balanced_ns']/sums['baseline_ns'],
                staged_over_whole_balanced_mean_ratio=outer/sum(s['balanced_ns'] for s in samples),
                pooled_staged_time_fraction=components,measured_blocks=len(samples))
        for policy in ['baseline','balanced']:
            result[policy+'_one_over_four_threads'] = result['threads']['1']['sum_state_median_'+policy+'_us']/result['threads']['4']['sum_state_median_'+policy+'_us']
        results[name] = result
    return dict(schema='cloudanalyzer.flatwall_balance_stages_summary.v1',states=64,processes=4,
        validation_states=256,measured_blocks=1280,warmup_blocks=256,evaluations_per_method_per_block=32,
        aggregate='Equal-state sum of medians, ten measured blocks per state/thread. Stage fractions pool every measured stage window, including clock/boundary remainder.',
        production_arithmetic_changed=False,defaults_changed=False,whole_registration_speedup_established=False,
        groups=results,per_state_median_evaluation_us=per_state)
