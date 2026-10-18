"""Independent target-balanced final-pose arithmetic and frozen real trial helpers."""
import numpy as np


def describe(snapshot):
    source = snapshot['source_world']
    indices = snapshot['matched_source_indices']
    target = snapshot['target_world']
    q = source[indices]
    _, inverse, counts = np.unique(target, axis=0, return_inverse=True, return_counts=True)
    multiplicity = counts[inverse.reshape(-1)]
    residual, arm = q-target, q-snapshot['center_world']
    robust = snapshot['kernel']**2/(snapshot['kernel']+np.sum(residual*residual, axis=1))**2
    weights = robust/multiplicity if snapshot['target_balance'] else robust
    jacobian = np.zeros((len(q), 3, 6))
    for i, (x, y, z) in enumerate(arm):
        jacobian[i] = np.column_stack([np.eye(3), [[0., z, -y], [-z, 0., x], [y, -x, 0.]]])
    h = np.einsum('nri,n,nrj->ij', jacobian, weights, jacobian)
    b = -np.einsum('nri,n,nr->i', jacobian, weights, residual)
    np.testing.assert_array_equal(snapshot['target_multiplicity'], multiplicity)
    for actual, expected in [(snapshot['robust_weights'], robust), (snapshot['weights'], weights)]:
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    for actual, expected in [(snapshot['normal_matrix'], h), (snapshot['right_hand_side'], b)]:
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-8)
    support = float(weights.sum())
    radius = float(np.sqrt(np.sum(weights*np.sum(arm*arm, axis=1))/support)) if support > 0 else None
    np.testing.assert_allclose(snapshot['weight_sum'], support, rtol=2e-12, atol=2e-12)
    if radius is None:
        if snapshot['weighted_rms_radius_m'] is not None: raise ValueError('empty radius changed')
    else: np.testing.assert_allclose(snapshot['weighted_rms_radius_m'], radius, rtol=2e-12, atol=2e-12)
    geometry = None
    if radius is not None and radius > 0:
        d = np.diag([1., 1., 1., 1/radius, 1/radius, 1/radius])
        values = np.linalg.eigvalsh(d@((h+h.T)/2)@d/support)
        geometry = dict(eigenvalues=values.tolist(), min_max_ratio=float(values[0]/values[-1]),
                        weak_directions_below_1e_3=int(sum(values/values[-1] < 1e-3)))
    return dict(source_points=len(source), matches=len(target), unique_targets=len(counts),
        target_balance=snapshot['target_balance'], weight_sum=support, robust_weight_sum=float(robust.sum()), radius_m=radius,
        normal_matrix=h.tolist(), right_hand_side=b.tolist(), geometry=geometry,
        weighted_cost=float(np.sum(weights*np.sum(residual*residual, axis=1))),
        matrix_oracle_error=float(np.max(np.abs(h-snapshot['normal_matrix']), initial=0.)),
        rhs_oracle_error=float(np.max(np.abs(b-snapshot['right_hand_side']), initial=0.)))
