"""Deterministic reverse-neighbor diagnostics on captured forward pairs."""
import numpy as np
from scipy.spatial import cKDTree

POLICY = dict(workers=4, reverse_tie_break='lowest original source index',
              broad_phase_padding_m=1e-10, weak_ratio=1e-3)


def points(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError('expected finite (N,3) points')
    return result


def diagnose(source_world, matched_source_indices, target_world, *, workers=4):
    """Search every source, including unmatched points; deduplicate target queries."""
    source, target = points(source_world), points(target_world)
    indices = np.asarray(matched_source_indices)
    if (workers not in {1, 4} or indices.shape != (len(target),) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= len(source)) or np.any(np.diff(indices.astype(np.int64)) <= 0)):
        raise ValueError('invalid captured source indices or worker policy')
    unique, inverse, counts = np.unique(target, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    reverse = np.empty(len(unique), dtype=np.int64)
    distance_sq = np.empty(len(unique))
    if len(unique):
        tree = cKDTree(source)
        nearest, _ = tree.query(unique, k=1, eps=0., workers=workers)
        if not np.isfinite(nearest).all(): raise ValueError('nonfinite reverse distance')
        candidates = tree.query_ball_point(unique, nearest+POLICY['broad_phase_padding_m'], workers=workers, return_sorted=True)
        for i, hits in enumerate(candidates):
            ids = np.asarray(hits, dtype=np.int64)
            delta = source[ids]-unique[i]
            squared = np.sum(delta*delta, axis=1)
            best = np.lexsort((ids, squared))[0]
            reverse[i], distance_sq[i] = ids[best], squared[best]
    reverse = reverse[inverse]
    return dict(target_group=inverse, target_multiplicity=counts[inverse],
                reverse_source_indices=reverse, reverse_distance_sq=distance_sq[inverse],
                reciprocal=reverse == indices, unique_targets=len(unique),
                saved_reverse_queries=len(target)-len(unique))


def point_system(source_world, target_world, weights, center):
    q, p = points(source_world), points(target_world)
    w, c = np.asarray(weights, dtype=float), np.asarray(center, dtype=float)
    if (q.shape != p.shape or w.shape != (len(q),) or c.shape != (3,)
            or not np.isfinite(c).all() or not np.isfinite(w).all() or np.any(w < 0)):
        raise ValueError('invalid fixed-match system inputs')
    a, residual = q-c, q-p
    j = np.zeros((len(q), 3, 6))
    j[:, :, :3] = np.eye(3)
    x, y, z = a.T
    j[:, 0, 4], j[:, 0, 5] = z, -y
    j[:, 1, 3], j[:, 1, 5] = -z, x
    j[:, 2, 3], j[:, 2, 4] = y, -x
    h = np.einsum('nri,n,nrj->ij', j, w, j)
    b = -np.einsum('nri,n,nr->i', j, w, residual)
    support = float(w.sum())
    radius = float(np.sqrt(np.sum(w*np.sum(a*a, axis=1))/support)) if support > 0 else None
    geometry = None
    if radius is not None and radius > 0:
        d = np.array([1., 1., 1., 1/radius, 1/radius, 1/radius])
        values = np.linalg.eigvalsh((h+h.T)*.5*d[:, None]*d[None, :]/support)
        if values[-1] > 0:
            geometry = dict(eigenvalues=values.tolist(), min_max_ratio=float(values[0]/values[-1]),
                weak_directions_below_1e_3=int(sum(values/values[-1] < POLICY['weak_ratio'])))
    return dict(normal_matrix=h, right_hand_side=b, weight_sum=support, radius_m=radius,
                weighted_cost=float(np.sum(w*np.sum(residual*residual, axis=1))), geometry=geometry)
