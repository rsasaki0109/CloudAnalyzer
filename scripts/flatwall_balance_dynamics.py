"""Independent ordered cell search, native-step arithmetic and tail diagnostics."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

OFFSETS = [(0, 0, 0)] + [v for v in itertools.product((-1, 0, 1), repeat=3) if any(v)]


class Search:
    """KD broad phase with exhaustive ordered 27-cell fallback for ties/bounds."""
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.isfinite(self.points).all() or not len(self.points):
            raise ValueError('finite nonempty (M,3) map required')
        self.keys = np.floor(self.points).astype(np.int64)
        self.cells = {}
        for i, key in enumerate(self.keys):
            self.cells.setdefault(tuple(key), []).append(i)
        self.tree = cKDTree(self.points)

    def exhaustive(self, point):
        key = np.floor(point).astype(np.int64)
        ids = [i for offset in OFFSETS for i in self.cells.get(tuple(key+offset), [])]
        if not ids:
            return -1
        delta = self.points[ids]-point
        squared = delta[:, 0]**2+delta[:, 1]**2+delta[:, 2]**2
        return ids[int(np.argmin(squared))]

    def matches(self, points, threshold_sq, *, exhaustive=False):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not np.isfinite(threshold_sq) or threshold_sq < 0:
            raise ValueError('invalid query points or threshold')
        if exhaustive or len(self.points) < 2:
            target = np.array([self.exhaustive(p) for p in points], dtype=np.int64)
        else:
            distance, ids = self.tree.query(points, k=2, workers=1, eps=0.)
            target = ids[:, 0].astype(np.int64)
            # A strict global NN inside the cell neighborhood is also its local NN.
            # Recompute all ambiguous/tied or outside-neighborhood queries in native order.
            inside = np.max(np.abs(self.keys[target]-np.floor(points).astype(np.int64)), axis=1) <= 1
            fallback = ~inside | (distance[:, 1]-distance[:, 0] <= 1e-10)
            for i in np.flatnonzero(fallback):
                target[i] = self.exhaustive(points[i])
        valid = target >= 0
        indices = np.flatnonzero(valid)
        delta = self.points[target[valid]]-points[valid]
        squared = delta[:, 0]**2+delta[:, 1]**2+delta[:, 2]**2
        keep = squared <= threshold_sq
        return indices[keep], target[valid][keep], squared[keep]


def system(world, source_ids, target, center, kernel):
    selected = world[source_ids]
    _, inverse, counts = np.unique(target, axis=0, return_inverse=True, return_counts=True)
    multiplicity = counts[inverse.reshape(-1)]
    residual = selected-target
    squared = residual[:, 0]**2+residual[:, 1]**2+residual[:, 2]**2
    weights = kernel**2/(kernel+squared)**2/multiplicity
    arm = selected-center
    j = np.zeros((len(selected), 3, 6))
    j[:, :, :3] = np.eye(3)
    x, y, z = arm.T
    j[:, 0, 4], j[:, 0, 5] = z, -y
    j[:, 1, 3], j[:, 1, 5] = -z, x
    j[:, 2, 3], j[:, 2, 4] = y, -x
    h = np.einsum('nri,n,nrj->ij', j, weights, j)
    b = -np.einsum('nri,n,nr->i', j, weights, residual)
    return h, b, multiplicity, weights, float(np.dot(weights, squared)), len(counts)


def transformed(points, pose):
    # Same scalar association as the native transform, avoiding a BLAS/FMA choice.
    return np.column_stack([pose[i, 0]*points[:, 0]+pose[i, 1]*points[:, 1]+pose[i, 2]*points[:, 2]+pose[i, 3]
                            for i in range(3)])


def centered_distance(a, b, center):
    delta = (a[:3, :3]-b[:3, :3])@b[:3, :3].T
    translation = (a[:3, 3]-b[:3, 3])-delta@(b[:3, 3]-center)
    skew = np.array([delta[2, 1]-delta[1, 2], delta[0, 2]-delta[2, 0], delta[1, 0]-delta[0, 1]])/2
    return float(np.linalg.norm(translation)), float(np.arctan2(np.linalg.norm(skew), 1+np.trace(delta)/2))


def tail_summary(states, center):
    if len(states) != 24:
        raise ValueError('require all 24 predeclared tail states')
    recurrence = []
    for lag in range(1, 9):
        pair_equal, count_equal, distances, differences = 0, 0, [], []
        for a, b in zip(states[lag:], states[:-lag]):
            same = np.array_equal(a['source_indices'], b['source_indices']) and np.array_equal(a['target_indices'], b['target_indices'])
            pair_equal += int(same)
            if same:
                count_equal += int(np.array_equal(a['multiplicity'], b['multiplicity']))
                differences.append(float(np.max(np.abs(a['weights']-b['weights']), initial=0.)))
            distances.append(centered_distance(a['before'], b['before'], center))
        recurrence.append(dict(lag=lag, comparisons=24-lag, identical_ordered_pair_sets=pair_equal,
            identical_multiplicities=count_equal, maximum_translation_m=max(d[0] for d in distances),
            maximum_rotation_rad=max(d[1] for d in distances),
            maximum_weight_change_when_pairs_equal=max(differences) if differences else None))
    changes = []
    for previous, current in zip(states, states[1:]):
        a = dict(zip(previous['source_indices'], zip(previous['target_indices'], previous['multiplicity'], previous['weights'])))
        b = dict(zip(current['source_indices'], zip(current['target_indices'], current['multiplicity'], current['weights'])))
        same = [(a[i], b[i]) for i in a.keys() & b.keys() if a[i][0] == b[i][0]]
        changes.append(dict(changed_target_or_support=sum(a.get(i, (-1,))[0] != b.get(i, (-1,))[0] for i in a.keys() | b.keys()),
            unchanged_target_count_changed=sum(x[1] != y[1] for x, y in same),
            maximum_unchanged_target_weight_change=max((abs(x[2]-y[2]) for x, y in same), default=0.)))
    shifts = [np.linalg.norm(s['step'][:3]) for s in states]
    turns = [np.linalg.norm(s['step'][3:]) for s in states]
    return dict(tail_states=24, step_translation_m_range=[float(min(shifts)), float(max(shifts))],
        step_rotation_rad_range=[float(min(turns)), float(max(turns))], lags=recurrence,
        consecutive_changes=changes,
        stopping_tolerances=dict(translation_m=1e-4, rotation_rad=1e-5),
        stopping_policy_changed=False)


def verify_trace(arrays, *, exhaustive_final=True):
    source, points, center = [arrays[k] for k in ['source_at_prediction', 'map_points_world', 'center_world']]
    kernel, threshold = [float(arrays[k]) for k in ['kernel', 'threshold_sq']]
    before, after, matrices, rhs = [arrays[k] for k in ['before', 'after', 'normal_matrix', 'right_hand_side']]
    if before.shape != (500, 4, 4) or after.shape != before.shape or matrices.shape != (500, 6, 6) or rhs.shape != (500, 6):
        raise ValueError('require all 500 complete native steps')
    np.testing.assert_array_equal(before[0], np.eye(4))
    np.testing.assert_array_equal(before[1:], after[:-1])
    search, tail = Search(points), []
    maximum_h, maximum_b = 0., 0.
    for i in range(500):
        world = transformed(source, before[i])
        ids, target_ids, _ = search.matches(world, threshold)
        h, b, multiplicity, weights, cost, unique = system(world, ids, points[target_ids], center, kernel)
        np.testing.assert_allclose(matrices[i], h, rtol=2e-12, atol=2e-8)
        np.testing.assert_allclose(rhs[i], b, rtol=2e-12, atol=2e-8)
        np.testing.assert_allclose(arrays['weighted_cost'][i], cost, rtol=2e-12, atol=2e-8)
        if len(ids) != arrays['matches'][i] or unique != arrays['unique_targets'][i]:
            raise ValueError('trace support differs')
        maximum_h = max(maximum_h, float(np.max(np.abs(matrices[i]-h))))
        maximum_b = max(maximum_b, float(np.max(np.abs(rhs[i]-b))))
        step = np.linalg.solve(matrices[i], rhs[i])
        update = np.eye(4)
        update[:3, :3] = Rotation.from_rotvec(step[3:]).as_matrix()
        update[:3, 3] = center-update[:3, :3]@center+step[:3]
        np.testing.assert_allclose(after[i], update@before[i], rtol=0, atol=2e-10)
        if np.linalg.norm(step[:3]) < 1e-4-2e-12 and np.linalg.norm(step[3:]) < 1e-5-2e-12:
            raise ValueError('native capped trace contains a converged step')
        if i >= 476:
            start, end = arrays['tail_offsets'][i-476:i-474]
            for name, actual in [('source_indices', ids), ('target_indices', target_ids), ('multiplicity', multiplicity)]:
                np.testing.assert_array_equal(arrays['tail_'+name][start:end], actual)
            np.testing.assert_allclose(arrays['tail_weights'][start:end], weights, rtol=2e-12, atol=2e-12)
            tail.append(dict(before=before[i], source_indices=ids, target_indices=target_ids,
                multiplicity=multiplicity, weights=weights, step=step))
    final = arrays['final_source_world']
    np.testing.assert_allclose(final, transformed(source, after[-1]), rtol=0, atol=2e-8)
    ids, target_ids, _ = search.matches(final, threshold, exhaustive=exhaustive_final)
    np.testing.assert_array_equal(arrays['final_source_indices'], ids)
    np.testing.assert_array_equal(arrays['final_target_world'], points[target_ids])
    _, _, _, weights, _, _ = system(final, ids, points[target_ids], center, kernel)
    np.testing.assert_allclose(arrays['final_weights'], weights, rtol=2e-12, atol=2e-12)
    return dict(native_steps_checked=500, final_pairs_checked=len(ids), maximum_matrix_abs_error=maximum_h,
        maximum_rhs_abs_error=maximum_b, tail=tail_summary(tail, center))
