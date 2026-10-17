#!/usr/bin/env python3
"""Independently check real reciprocal samples and all frame accounting."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ['raw-legacy', 'gyro-legacy', 'raw-0.1875', 'gyro-0.1875']
POLICIES = ['repeat-motion', 'last-pose']
POLICY = dict(workers=4, reverse_tie_break='lowest original source index', broad_phase_padding_m=1e-10, weak_ratio=1e-3)


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def rows(path): return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, atol=2e-8):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=2e-12, atol=atol):
        raise ValueError('numeric accounting differs')


def geometry(h, weight, radius):
    if weight <= 0 or radius is None or radius <= 0: return None
    d = np.diag([1., 1., 1., 1/radius, 1/radius, 1/radius])
    h = np.asarray(h)
    values = np.linalg.eigvalsh(d@((h+h.T)/2)@d/weight)
    if values[-1] <= 0: return None
    return dict(eigenvalues=values.tolist(), min_max_ratio=float(values[0]/values[-1]),
                weak_directions_below_1e_3=int(sum(values/values[-1] < 1e-3)))


def check_geometry(actual, expected):
    if expected is None:
        if actual is not None: raise ValueError('unsupported geometry reported')
    else:
        if actual is None or set(actual) != set(expected): raise ValueError('missing geometry')
        close(actual['eigenvalues'], expected['eigenvalues'], 2e-11)
        close(actual['min_max_ratio'], expected['min_max_ratio'], 2e-11)
        if actual['weak_directions_below_1e_3'] != expected['weak_directions_below_1e_3']: raise ValueError('weak count differs')


def summarize(frames):
    matched = [r for r in frames if r['matches']]
    return dict(scans=len(frames), matches=sum(r['matches'] for r in frames), unique_target_queries=sum(r['unique_targets'] for r in frames),
        saved_reverse_queries=sum(r['saved_reverse_queries'] for r in frames), reciprocal_matches=sum(r['reciprocal_matches'] for r in frames),
        reverse_to_unmatched_source=sum(r['reverse_to_unmatched_source'] for r in frames),
        median_reciprocal_fraction=float(np.median([r['reciprocal_matches']/r['matches'] for r in matched])) if matched else None,
        median_saved_query_fraction=float(np.median([r['saved_reverse_queries']/r['matches'] for r in matched])) if matched else None,
        median_retained_weight_fraction=float(np.median([r['reciprocal']['weight_sum']/r['original']['weight_sum'] for r in matched])) if matched else None,
        supported_original_systems=sum(r['original']['geometry'] is not None for r in frames),
        supported_reciprocal_systems=sum(r['reciprocal']['geometry'] is not None for r in frames),
        weak_reciprocal_systems=sum(r['reciprocal']['geometry'] is not None and r['reciprocal']['geometry']['weak_directions_below_1e_3'] > 0 for r in frames),
        no_reciprocal_system_scan_indices=[r['scan_index'] for r in frames if r['reciprocal']['geometry'] is None],
        maximum_matrix_oracle_error=max(r['matrix_oracle_error'] for r in frames), maximum_rhs_oracle_error=max(r['rhs_oracle_error'] for r in frames))


def check_sample(path, frame, original):
    with np.load(path, allow_pickle=False) as saved: data = {k: saved[k].copy() for k in saved.files}
    source, ids, target, weights, center = [data[k] for k in ['source_world', 'matched_source_indices', 'target_world', 'weights', 'center_world']]
    m = frame['matches']
    if (source.shape != (frame['source_points'], 3) or target.shape != (m, 3) or weights.shape != (m,) or ids.shape != (m,)
            or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(source)) or np.any(np.diff(ids.astype(np.int64)) <= 0)):
        raise ValueError('sample dimensions or indices differ')
    close(source, source)
    close(target, target)
    close(center, frame['center_world'], 0.)
    groups = sorted(set(map(tuple, target.tolist())))
    lookup = {key: i for i, key in enumerate(groups)}
    group = np.array([lookup[tuple(p)] for p in target], dtype=np.int64)
    multiplicity = Counter(map(tuple, target.tolist()))
    counts = np.array([multiplicity[tuple(p)] for p in target], dtype=np.int64)
    reverse = np.empty(m, dtype=np.int64)
    squared = np.empty(m)
    for start in range(0, m, 64):
        x = source[None, :, 0]-target[start:start+64, None, 0]
        y = source[None, :, 1]-target[start:start+64, None, 1]
        z = source[None, :, 2]-target[start:start+64, None, 2]
        distance = x*x+y*y+z*z
        chosen = np.argmin(distance, axis=1)
        reverse[start:start+64] = chosen
        squared[start:start+64] = distance[np.arange(len(chosen)), chosen]
    mask = reverse == ids
    for name, expected in [('target_group', group), ('target_multiplicity', counts), ('reverse_source_indices', reverse), ('reciprocal', mask)]:
        if not np.array_equal(data[name], expected): raise ValueError('sample reverse/group accounting differs')
    close(data['reverse_distance_sq'], squared, 2e-12)
    if (len(groups) != frame['unique_targets'] or m-len(groups) != frame['saved_reverse_queries']
            or int(mask.sum()) != frame['reciprocal_matches'] or int(sum(counts > 1)) != frame['reused_target_rows']
            or int(np.max(counts, initial=0)) != frame['maximum_target_multiplicity']
            or int(sum(~np.isin(reverse, ids))) != frame['reverse_to_unmatched_source']): raise ValueError('sample coverage differs')
    q = source[ids]
    close(weights, original['kernel']**2/(original['kernel']+np.sum((q-target)**2, axis=1))**2, 2e-12)
    for name, keep in [('original', np.ones(m, dtype=bool)), ('reciprocal', mask)]:
        a, residual, w = q[keep]-center, q[keep]-target[keep], weights[keep]
        j = np.zeros((len(a), 3, 6))
        for i, (x, y, z) in enumerate(a):
            j[i] = np.column_stack([np.eye(3), [[0., z, -y], [-z, 0., x], [y, -x, 0.]]])
        h = np.einsum('nri,n,nrj->ij', j, w, j)
        b = -np.einsum('nri,n,nr->i', j, w, residual)
        for key, expected in [('normal_matrix', h), ('right_hand_side', b)]:
            close(data[name+'_'+key], expected)
            close(frame[name][key], expected)
        close(frame[name]['weight_sum'], w.sum())
        radius = float(np.sqrt(np.sum(w*np.sum(a*a, axis=1))/w.sum())) if w.sum() > 0 else None
        if radius is None:
            if frame[name]['radius_m'] is not None: raise ValueError('empty radius differs')
        else: close(frame[name]['radius_m'], radius)
        close(frame[name]['weighted_cost'], np.sum(w*np.sum(residual*residual, axis=1)))
    return m


def verify(folder=None):
    folder = Path(folder) if folder else Path(__file__).resolve().parent
    root = folder.parents[2]
    manifest = read(folder/'manifest.json')
    for base, mapping in [(folder, manifest['files_sha256']), (root, manifest['dependencies']), (root, manifest['checkout_sources_sha256'])]:
        for name, digest in mapping.items():
            if sha(base/name) != digest: raise ValueError('file hash differs: '+name)
    protocol, freeze, receipt = [read(folder/name) for name in ['protocol.json', 'controller-freeze.json', 'receipt.json']]
    if (protocol['schema'] != 'cloudanalyzer.flatwall_correspondence_consistency.v1' or protocol['policy'] != POLICY
            or protocol['total_frames'] != 2560 or protocol['state_feedback'] is not False or protocol['reference_poses_used'] is not False
            or protocol['parameter_search'] is not False): raise ValueError('protocol scope differs')
    if (sha(folder/'protocol.json') != freeze['protocol_sha256'] or receipt['protocol_sha256'] != freeze['protocol_sha256']
            or receipt['controller_freeze_sha256'] != sha(folder/'controller-freeze.json')): raise ValueError('protocol/receipt binding differs')
    for name, digest in manifest['checkout_sources_sha256'].items():
        if freeze['source_files_sha256'].get(name) != digest: raise ValueError('frozen source binding differs')
    for name, digest in receipt['output_files_sha256'].items():
        if (folder/name).is_file() and sha(folder/name) != digest: raise ValueError('output closure differs')
    summary, catalog = read(folder/'summary.json'), read(folder/'snapshot-catalog.json')
    cases = [f'{r}/{p}/{a}' for r in ['flatwall_01', 'flatwall_02'] for p in POLICIES for a in ARMS]
    if [c['name'] for c in protocol['cases']] != cases or set(summary['cases']) != set(cases): raise ValueError('case accounting differs')
    lookup = {(r['case'], r['scan_index']): r for r in catalog}
    if len(lookup) != len(catalog) or len(catalog) != 2560: raise ValueError('snapshot inventory differs')
    total, public_queries, private_queries, samples = 0, 0, 0, 0
    for case, definition in zip(cases, protocol['cases']):
        recording, policy, arm = case.split('/')
        count = 185 if recording == 'flatwall_01' else 135
        if definition['scan_indices'] != list(range(1, count+1)): raise ValueError('frozen cohort differs')
        if recording == 'flatwall_01' and policy == 'repeat-motion':
            old, relative = root/'benchmarks/slam/flatwall-frontend-constraints', arm
        else:
            old = root/('benchmarks/slam/flatwall-prediction-policy' if recording == 'flatwall_01' else 'benchmarks/slam/flatwall02-prediction')
            relative = policy+'/'+arm
        originals = rows(old/relative/'frames.jsonl')
        old_catalog = {r['scan_index']: r for r in read(old/'snapshot-catalog.json') if r['arm'] == arm and (recording == 'flatwall_01' or r['policy'] == policy)}
        frames = rows(folder/case/'frames.jsonl')
        if [r['scan_index'] for r in frames] != list(range(1, count+1)) or len(originals) != count: raise ValueError('frame identities differ')
        for ordinal, (frame, original) in enumerate(zip(frames, originals)):
            index, m = frame['scan_index'], frame['matches']
            if (frame['case'] != case or frame['timestamp_ns'] != original['reference_ns'] or m != original['matches']
                    or frame['source_points'] != original['source_points']): raise ValueError('original frame accounting differs')
            digest = next(v['sha256'] for k, v in old_catalog[index]['files'].items() if k.endswith('.npz'))
            if frame['original_snapshot_sha256'] != digest: raise ValueError('original snapshot binding differs')
            close(frame['center_world'], np.array(original['prediction_pose'])[:3, 3], 0.)
            if (frame['unique_targets']+frame['saved_reverse_queries'] != m or not 0 <= frame['reciprocal_matches'] <= frame['unique_targets'] <= m
                    or not 0 <= frame['reverse_to_unmatched_source'] <= m-frame['reciprocal_matches']
                    or not 0 <= frame['reused_target_rows'] <= m or not 0 <= frame['maximum_target_multiplicity'] <= m):
                raise ValueError('reverse query accounting differs')
            for key in ['normal_matrix', 'right_hand_side', 'weight_sum', 'radius_m']:
                if original[key] is None:
                    if frame['original'][key] is not None: raise ValueError('original support differs')
                else: close(frame['original'][key], original[key])
            for name in ['original', 'reciprocal']:
                system = frame[name]
                h, b = np.array(system['normal_matrix']), np.array(system['right_hand_side'])
                if h.shape != (6, 6) or b.shape != (6,) or system['weight_sum'] < 0: raise ValueError('invalid system dimensions/support')
                close(h, h.T); close(b, b)
                if np.linalg.eigvalsh(h)[0] < -2e-8: raise ValueError('nonpositive system')
                check_geometry(system['geometry'], geometry(h, system['weight_sum'], system['radius_m']))
            if (not 0 <= frame['reciprocal']['weight_sum'] <= frame['original']['weight_sum']+2e-8
                    or not 0 <= frame['reciprocal']['weighted_cost'] <= frame['original']['weighted_cost']+2e-8
                    or np.linalg.eigvalsh(np.array(frame['original']['normal_matrix'])-frame['reciprocal']['normal_matrix'])[0] < -2e-8):
                raise ValueError('subset accounting differs')
            if not 0 <= frame['matrix_oracle_error'] <= 2e-8 or not 0 <= frame['rhs_oracle_error'] <= 2e-8: raise ValueError('arithmetic oracle differs')
            item = lookup[(case, index)]
            chosen = ordinal in {1, count//2, count-1}
            if item['public_sample'] is not chosen or item['file'] != f'{case}/scan-{index:03d}.npz': raise ValueError('sample selection differs')
            if chosen:
                path = folder/item['file']
                if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']: raise ValueError('sample hash differs')
                public_queries += check_sample(path, frame, original)
                samples += 1
            private_queries += m
        if summary['cases'][case] != summarize(frames): raise ValueError('case summary differs')
        total += len(frames)
    if summary['frames'] != total or total != 2560 or samples != 48 or summary['exhaustive_reverse_queries'] != private_queries: raise ValueError('total accounting differs')
    for name in ['original_poses_and_forward_pairs_unchanged', 'all_source_points_in_reverse_search', 'all_reverse_indices_independently_checked']:
        if summary[name] is not True: raise ValueError('audit scope differs')
    for name in ['forward_nearest_search_replayed', 'state_feedback', 'reference_poses_used', 'outlier_labels_certified', 'trajectory_accuracy_claim', 'speed_claim']:
        if summary[name] is not False: raise ValueError('unsupported claim')
    return dict(cases=16, frames=total, public_samples=samples, public_exhaustive_reverse_queries=public_queries,
        private_exhaustive_reverse_queries_recorded=private_queries, all_frame_spectra_and_accounting_checked=True,
        all_private_raw_systems_publicly_recomputed=False, state_feedback=False)


if __name__ == '__main__': print(json.dumps(verify(), indent=2))
