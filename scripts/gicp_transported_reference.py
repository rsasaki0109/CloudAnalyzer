"""Independent unit-weight transported-shape residual optimizer for audits."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import audit_flatwall_gicp_linearization as factor
import audit_flatwall_gicp_registration as frozen

SCALES = [1., .5, .25, .125, .0625, .03125]
ARMIJO = 1e-4


def canonical(shapes):
    other = np.swapaxes(shapes, -1, -2)
    return np.where(shapes == other, shapes, shapes/2.+other/2.)


def metric_gradient(source, target, source_shapes, pose, combined):
    """Additional rotation gradient of .5 r.T C(R)^-1 r at fixed IDs."""
    residual = source @ pose[:3, :3].T+pose[:3, 3]-target
    e = np.linalg.solve(combined, residual[..., None])[..., 0]
    rotated = pose[:3, :3] @ canonical(source_shapes) @ pose[:3, :3].T
    return np.cross(e, (rotated @ e[..., None])[..., 0]).sum(axis=0)


def score(source, target, cs, ct, pose, limit, tree=None):
    world = source @ pose[:3, :3].T+pose[:3, 3]
    if not np.isfinite(world).all():
        raise FloatingPointError("nonfinite transformed source")
    if len(target) and len(source):
        if tree is None: tree = cKDTree(target)
        _, hits = tree.query(world, k=1, eps=0., workers=1)
        d2 = np.sum((world-target[hits])**2, axis=1)
        if not np.isfinite(d2).all(): raise FloatingPointError("nonfinite nearest squared distance")
        si = np.flatnonzero(d2 < limit**2).astype(np.uint64)
        ti, distances = hits[si].astype(np.uint64), d2[si]
    else:
        si = ti = np.empty(0, dtype=np.uint64)
        distances = np.empty(0)
    c = ct[ti]+pose[:3, :3] @ cs[si] @ pose[:3, :3].T
    residual = world[si]-target[ti]
    e = np.linalg.solve(c, residual[..., None])[..., 0]
    cost = .5*float(np.einsum("ni,ni->", residual, e))
    if not np.isfinite(cost): raise FloatingPointError("nonfinite transported cost")
    return dict(source_indices=si, target_indices=ti, squared_distances=distances,
                weights=np.ones(len(si)), combined_shapes=c, cost=cost)


def acceptable(baseline_ids, candidate_ids, baseline_cost, candidate_cost, scale, directional):
    return (np.array_equal(baseline_ids, candidate_ids) and candidate_cost < baseline_cost
            and candidate_cost <= baseline_cost+ARMIJO*scale*directional)


def reference(source, target, source_shapes, target_shapes, initial_pose, center, *,
              max_correspondence_distance=2., max_iterations=30, translation_tolerance=1e-4,
              rotation_tolerance=1e-5, robust_scale=None, cycle_guard=None):
    if robust_scale is not None:
        raise ValueError("transported objective requires unit weights (robust_scale=None)")
    cs, ct = canonical(source_shapes), canonical(target_shapes)
    pose, history, accepted = initial_pose.copy(), [], 0
    termination, failure_reason, cycle = "max_iterations", None, None
    tree = cKDTree(target) if len(target) else None
    for _ in range(max_iterations):
        try:
            matches = score(source, target, cs, ct, pose, max_correspondence_distance, tree)
            si, ti = matches["source_indices"], matches["target_indices"]
            linear, radius = factor.reference(source[si], target[ti], cs[si], ct[ti], pose, center, matches["weights"])
            metric = metric_gradient(source[si], target[ti], cs[si], pose, linear["combined_shapes"])
            gradient = -linear["right_hand_side"].copy()
            gradient[3:] += metric
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            termination = "nonfinite"
            break
        row = dict(pose_before=pose.copy(), pose_after=pose.copy(), source_indices=si, target_indices=ti,
            squared_distances=matches["squared_distances"], weights=matches["weights"], combined_shapes=linear["combined_shapes"],
            normal_matrix=linear["normal_matrix"], right_hand_side=linear["right_hand_side"],
            metric_gradient_rotation=metric, objective_gradient=gradient, trial_support_equal=[],
            cost_before=linear["cost"], cost_after=None, step=None, trial_scales=np.empty(0), trial_costs=[], accepted_scale=None)
        history.append(row)
        if len(si) < 3:
            termination = "insufficient_correspondences"
            break
        corrected = dict(linear, right_hand_side=-gradient)
        delta = frozen.solve(corrected, radius)
        if delta is None:
            termination = "singular_system"
            break
        if not np.isfinite(delta).all():
            termination = "nonfinite"
            break
        row["step"] = delta
        if np.hypot.reduce(delta[:3]) <= translation_tolerance and np.hypot.reduce(delta[3:]) <= rotation_tolerance:
            termination = "converged_step"
            break
        if cycle_guard is not None:
            import audit_flatwall_gicp_cycles as cycles
            cycle = cycles.cycle_at(history, center, convergence_translation=translation_tolerance,
                                    convergence_rotation=rotation_tolerance, **cycle_guard)
            if cycle is not None:
                termination, failure_reason = "cycle_detected", "repeated nonstationary GICP outer states"
                break
        directional = float(gradient @ delta)
        if not np.isfinite(directional):
            termination = "nonfinite"
            break
        if directional >= 0.:
            termination = "no_descent"
            break
        finite = False
        for scale in SCALES:
            candidate = frozen.retract(pose, center, delta, scale)
            try:
                trial = score(source, target, cs, ct, candidate, max_correspondence_distance, tree)
                cost = trial["cost"]
                same = np.array_equal(si, trial["source_indices"])
                finite = True
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                cost, same = None, False
            row["trial_scales"] = np.append(row["trial_scales"], scale)
            row["trial_costs"].append(cost)
            row["trial_support_equal"].append(bool(same))
            if cost is not None and acceptable(si, trial["source_indices"], linear["cost"], cost, scale, directional):
                pose, accepted = candidate, accepted+1
                row.update(pose_after=pose.copy(), cost_after=cost, accepted_scale=scale)
                break
        else:
            termination = "no_descent" if finite else "nonfinite"
            break
    try:
        final = score(source, target, cs, ct, pose, max_correspondence_distance, tree)
        final_cost = final["cost"]
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        termination, final_cost = "nonfinite", None
        final = dict(source_indices=np.empty(0, dtype=np.uint64), target_indices=np.empty(0, dtype=np.uint64),
                     squared_distances=np.empty(0), weights=np.empty(0))
    result = dict(objective="transported", pose=pose, termination=termination, converged=termination == "converged_step",
        iterations=len(history), accepted_steps=accepted, final_cost=final_cost,
        matched_source_indices=final["source_indices"], matched_target_indices=final["target_indices"],
        squared_distances=final["squared_distances"], weights=final["weights"], failure_reason=failure_reason, history=history)
    if cycle is not None: result["cycle"] = cycle
    return result
