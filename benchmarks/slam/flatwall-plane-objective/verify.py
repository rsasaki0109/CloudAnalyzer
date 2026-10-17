#!/usr/bin/env python3
"""Verify all published systems and selected real map/normal arithmetic independently."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ['raw-legacy', 'gyro-legacy', 'raw-0.1875', 'gyro-0.1875']
POLICIES = ['repeat-motion', 'last-pose']
NORMAL_POLICY = dict(neighbors=12, radius_m=1., minimum_neighbors=6,
    minimum_middle_variance_m2=1e-8, minimum_middle_largest_ratio=.1,
    maximum_smallest_middle_ratio=.1, broad_phase_padding_m=1e-10)


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path): return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def close(a, b, atol=2e-8):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=2e-12, atol=atol):
        raise ValueError('numeric accounting differs')


def geometry(h, weight, radius):
    if weight <= 0 or radius is None or radius <= 0: return None
    scale = np.diag([1., 1., 1., 1/radius, 1/radius, 1/radius])
    h = np.asarray(h)
    values = np.linalg.eigvalsh(scale@((h+h.T)/2)@scale/weight)
    if values[-1] <= 0: return None
    return dict(eigenvalues=values.tolist(), min_max_ratio=float(values[0]/values[-1]),
                weak_directions_below_1e_3=int(sum(values/values[-1] < 1e-3)))


def check_geometry(actual, expected):
    if expected is None:
        if actual is not None: raise ValueError('unsupported geometry reported')
    else:
        if actual is None or set(actual) != set(expected): raise ValueError('missing geometry fields')
        close(actual['eigenvalues'], expected['eigenvalues'], 2e-11)
        close(actual['min_max_ratio'], expected['min_max_ratio'], 2e-11)
        if actual['weak_directions_below_1e_3'] != expected['weak_directions_below_1e_3']:
            raise ValueError('weak direction count differs')


def summarize(frames):
    supported = [r for r in frames if r['plane_geometry'] is not None]
    counts = Counter()
    for row in frames: counts.update(row['normal_status_counts'])
    return dict(scans=len(frames), original_matches=sum(r['matches'] for r in frames),
        accepted_normal_matches=sum(r['accepted_normal_matches'] for r in frames), normal_status_counts=dict(counts),
        supported_systems=len(supported), no_plane_system_scan_indices=[r['scan_index'] for r in frames if r['plane_geometry'] is None],
        plane_weak_systems=sum(r['plane_geometry']['weak_directions_below_1e_3'] > 0 for r in supported),
        same_subset_point_weak_systems=sum(r['point_geometry']['weak_directions_below_1e_3'] > 0 for r in supported),
        plane_more_weak_directions=sum(r['plane_geometry']['weak_directions_below_1e_3'] > r['point_geometry']['weak_directions_below_1e_3'] for r in supported),
        median_normal_fraction=float(np.median([r['accepted_normal_matches']/r['matches'] for r in frames if r['matches']])) if any(r['matches'] for r in frames) else None,
        maximum_plane_matrix_oracle_error=max(r['plane_matrix_oracle_error'] for r in frames),
        maximum_plane_rhs_oracle_error=max(r['plane_rhs_oracle_error'] for r in frames))


def sample(path, frame, original, selected_queries):
    with np.load(path, allow_pickle=False) as saved: data = {k: saved[k].copy() for k in saved.files}
    cloud, neighbors = data['map_points'], data['neighbors']
    q, p, w, center = (data[k] for k in ['source_world', 'target_world', 'weights', 'center_world'])
    if q.shape != (frame['matches'], 3) or p.shape != q.shape or cloud.shape != (frame['map_points'], 3):
        raise ValueError('sample point dimensions differ')
    if neighbors.shape != (len(q), 12) or neighbors.dtype.kind != 'i' or np.any(neighbors < -1) or np.any(neighbors >= len(cloud)):
        raise ValueError('invalid neighbor indices')
    close(cloud[np.lexsort((cloud[:, 2], cloud[:, 1], cloud[:, 0]))], cloud, 0.)
    membership = {point.tobytes() for point in cloud}
    if any(point.tobytes() not in membership for point in p): raise ValueError('sample target absent from map')
    for query in selected_queries:
        squared = np.array([float((point-p[query])@(point-p[query])) for point in cloud])
        eligible = np.flatnonzero(squared <= 1.)
        ordered = eligible[np.lexsort((eligible, squared[eligible]))[:12]]
        expected = np.full(12, -1, dtype=np.int64)
        expected[:len(ordered)] = ordered
        if not np.array_equal(neighbors[query], expected): raise ValueError('exhaustive neighborhood differs')
    covariances, values, normals, counts, labels, centroids = [], [], [], [], [], []
    for indices in neighbors:
        ids = indices[indices >= 0]
        if len(set(ids.tolist())) != len(ids) or not np.array_equal(indices[:len(ids)], ids):
            raise ValueError('duplicate or nontrailing neighbor padding')
        points = cloud[ids]
        centroid = points.mean(axis=0) if len(points) else np.zeros(3)
        delta = points-centroid
        covariance = delta.T@delta/max(len(points), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        normal *= np.sign(normal[np.argmax(abs(normal))])
        label = ('insufficient_neighbors' if len(points) < 6 else
                 'line_or_point' if eigenvalues[1] <= 1e-8 or eigenvalues[1] < .1*eigenvalues[2] else
                 'not_planar' if eigenvalues[0] > .1*eigenvalues[1] else 'accepted')
        covariances.append(covariance); values.append(eigenvalues); normals.append(normal)
        counts.append(len(ids)); labels.append(label); centroids.append(centroid)
    covariances = np.array(covariances).reshape(-1, 3, 3)
    normals, values, centroids = [np.array(x).reshape(-1, 3) for x in [normals, values, centroids]]
    mask = np.array(labels) == 'accepted'
    for name, expected in [('covariance', covariances), ('eigenvalues', values), ('centroids', centroids), ('neighbor_count', counts)]:
        close(data[name], expected, 2e-10)
    if data['labels'].tolist() != labels or not np.array_equal(data['accepted'], mask): raise ValueError('normal acceptance differs')
    close(data['normals'][mask], normals[mask], 2e-10)
    if dict(Counter(labels)) != frame['normal_status_counts'] or int(mask.sum()) != frame['accepted_normal_matches']:
        raise ValueError('normal coverage differs')
    close(center, frame['center_world'], 0.)
    close(center, np.array(original['prediction_pose'])[:3, 3], 0.)
    close(w, original['kernel']**2/(original['kernel']+np.sum((q-p)**2, axis=1))**2, 2e-12)
    q, p, n, w = q[mask], p[mask], normals[mask], w[mask]
    arm, residual = q-center, q-p
    plane_j = np.column_stack([n, np.cross(arm, n)])
    plane_r = np.einsum('ni,ni->n', n, residual)
    point_j = np.zeros((len(q), 3, 6))
    for i, (x, y, z) in enumerate(arm):
        point_j[i] = np.column_stack([np.eye(3), [[0., z, -y], [-z, 0., x], [y, -x, 0.]]])
    systems = dict(plane_normal_matrix=np.einsum('ni,n,nj->ij', plane_j, w, plane_j),
        plane_right_hand_side=-np.einsum('ni,n,n->i', plane_j, w, plane_r),
        point_normal_matrix=np.einsum('nri,n,nrj->ij', point_j, w, point_j),
        point_right_hand_side=-np.einsum('nri,n,nr->i', point_j, w, residual))
    close(data['plane_jacobian'], plane_j)
    close(data['plane_residual'], plane_r)
    for name, expected in systems.items():
        close(data[name], expected)
        close(frame[name], expected)
    close(frame['weight_sum'], w.sum())
    radius = float(np.sqrt(np.sum(w*np.sum(arm**2, axis=1))/w.sum())) if w.sum() > 0 else None
    if radius is None:
        if frame['radius_m'] is not None: raise ValueError('empty radius differs')
    else: close(frame['radius_m'], radius)
    close(frame['plane_cost'], np.sum(w*plane_r**2))
    close(frame['point_cost'], np.sum(w*np.sum(residual**2, axis=1)))


def verify(folder=None):
    folder = Path(folder) if folder else Path(__file__).resolve().parent
    root = folder.parents[2]
    manifest = read(folder/'manifest.json')
    for base, mapping in [(folder, manifest['files_sha256']), (root, manifest['dependencies']), (root, manifest['checkout_sources_sha256'])]:
        for name, expected in mapping.items():
            if sha(base/name) != expected: raise ValueError('file hash differs: '+name)
    protocol, freeze = read(folder/'protocol.json'), read(folder/'controller-freeze.json')
    if sha(folder/'protocol.json') != freeze['protocol_sha256']: raise ValueError('protocol binding differs')
    for name, digest in manifest['checkout_sources_sha256'].items():
        if freeze['source_files_sha256'].get(name) != digest: raise ValueError('frozen source binding differs')
    if (protocol['normal_policy'] != NORMAL_POLICY or protocol['workers'] != 4 or protocol['state_feedback'] is not False
            or protocol['reference_poses_used'] is not False or protocol['parameter_search'] is not False or protocol['total_frames'] != 2560):
        raise ValueError('frozen diagnostic policy differs')
    summary, catalog = read(folder/'summary.json'), read(folder/'snapshot-catalog.json')
    receipt, oracle = read(folder/'receipt.json'), read(folder/'neighborhood-oracle.json')
    if receipt['protocol_sha256'] != sha(folder/'protocol.json') or receipt['controller_freeze_sha256'] != sha(folder/'controller-freeze.json'):
        raise ValueError('receipt binding differs')
    for name, digest in receipt['output_files_sha256'].items():
        if (folder/name).is_file() and sha(folder/name) != digest: raise ValueError('output closure differs')
    expected_cases = [f'{recording}/{policy}/{arm}' for recording in ['flatwall_01', 'flatwall_02'] for policy in POLICIES for arm in ARMS]
    if [case['name'] for case in protocol['cases']] != expected_cases or set(summary['cases']) != set(expected_cases):
        raise ValueError('case accounting differs')
    lookup = {(r['case'], r['scan_index']): r for r in catalog}
    oracle_lookup = {(r['case'], r['scan_index']): r for r in oracle['samples']}
    if len(lookup) != len(catalog) or len(catalog) != 2560 or len(oracle_lookup) != len(oracle['samples']) or len(oracle_lookup) != 48:
        raise ValueError('snapshot or sample accounting differs')
    total, sample_count, queries = 0, 0, 0
    for case in expected_cases:
        recording, policy, arm = case.split('/')
        count = 185 if recording == 'flatwall_01' else 135
        if recording == 'flatwall_01' and policy == 'repeat-motion':
            old = root/'benchmarks/slam/flatwall-frontend-constraints'
            original_catalog = [r for r in read(old/'snapshot-catalog.json') if r['arm'] == arm]
            original_frames = rows(old/arm/'frames.jsonl')
        else:
            old = root/('benchmarks/slam/flatwall-prediction-policy' if recording == 'flatwall_01' else 'benchmarks/slam/flatwall02-prediction')
            original_catalog = [r for r in read(old/'snapshot-catalog.json') if r['arm'] == arm and (recording == 'flatwall_01' or r['policy'] == policy)]
            original_frames = rows(old/policy/arm/'frames.jsonl')
        old_catalog = {r['scan_index']: r for r in original_catalog}
        frames = rows(folder/case/'frames.jsonl')
        if [r['scan_index'] for r in frames] != list(range(1, count+1)) or len(original_frames) != count:
            raise ValueError('original frame identities differ')
        for ordinal, (frame, original) in enumerate(zip(frames, original_frames)):
            index = frame['scan_index']
            if frame['case'] != case or frame['timestamp_ns'] != original['reference_ns'] or frame['matches'] != original['matches']:
                raise ValueError('original match metadata differs')
            old_npz = next(v['sha256'] for k, v in old_catalog[index]['files'].items() if k.endswith('.npz'))
            if old_npz != frame['original_snapshot_sha256']: raise ValueError('original snapshot binding differs')
            close(frame['center_world'], np.array(original['prediction_pose'])[:3, 3], 0.)
            statuses = frame['normal_status_counts']
            if (set(statuses)-{'accepted', 'insufficient_neighbors', 'line_or_point', 'not_planar'} or any(not isinstance(n, int) or n < 0 for n in statuses.values())
                    or sum(statuses.values()) != frame['matches'] or statuses.get('accepted', 0) != frame['accepted_normal_matches']):
                raise ValueError('normal coverage accounting differs')
            if not 0 <= frame['weight_sum'] <= frame['accepted_normal_matches']+1e-9: raise ValueError('normal support differs')
            for kind in ['plane', 'point']:
                h, b = np.array(frame[kind+'_normal_matrix']), np.array(frame[kind+'_right_hand_side'])
                if h.shape != (6, 6) or b.shape != (6,): raise ValueError('system dimensions differ')
                close(h, h.T)
                close(b, b)
                if np.linalg.eigvalsh(h)[0] < -2e-8: raise ValueError('system is not positive semidefinite')
                check_geometry(frame[kind+'_geometry'], geometry(h, frame['weight_sum'], frame['radius_m']))
            if frame['plane_cost'] < 0 or frame['plane_cost'] > frame['point_cost']+2e-8: raise ValueError('projected cost differs')
            if not 0 <= frame['plane_matrix_oracle_error'] <= 2e-8 or not 0 <= frame['plane_rhs_oracle_error'] <= 2e-8:
                raise ValueError('all-frame arithmetic oracle exceeded tolerance')
            item = lookup[(case, index)]
            chosen = ordinal in {1, count//2, count-1}
            if item['public_sample'] is not chosen or item['file'] != f'{case}/scan-{index:03d}.npz': raise ValueError('sample selection differs')
            if chosen:
                path = folder/item['file']
                if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']: raise ValueError('sample bytes differ')
                selected = sorted({0, frame['matches']//2, frame['matches']-1}) if frame['matches'] else []
                record = oracle_lookup[(case, index)]
                if (record['selected_match_rows'] != selected or record['exhaustive_neighborhoods_checked'] != len(selected)
                        or record['no_matches'] != (frame['matches'] == 0)): raise ValueError('oracle selection differs')
                sample(path, frame, original, selected)
                sample_count += 1
                queries += len(selected)
        if summary['cases'][case] != summarize(frames): raise ValueError('case summary differs')
        total += len(frames)
    if total != summary['frames'] or queries != oracle['queries_checked']: raise ValueError('total accounting differs')
    for name in ['state_feedback', 'reference_poses_used', 'map_nearest_match_search_independently_replayed', 'physical_observability_certified', 'speed_claim']:
        if summary[name] is not False: raise ValueError('unsupported claim: '+name)
    for name in ['original_poses_and_matches_unchanged', 'map_insertion_and_membership_replayed']:
        if summary[name] is not True: raise ValueError('audit scope differs')
    return dict(cases=16, frames=total, public_map_samples=sample_count, exhaustive_sample_queries=queries,
        normal_covariance_and_system_samples=sample_count, all_frame_spectra_checked=total,
        full_private_neighborhood_search_independently_replayed=False, state_feedback=False, physical_observability_certified=False)


if __name__ == '__main__': print(json.dumps(verify(), indent=2))
