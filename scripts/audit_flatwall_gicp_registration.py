#!/usr/bin/env python3
"""Independent iterative GICP checks on fixed maps from a real recording."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import audit_flatwall_gicp_linearization as factor
import audit_flatwall_odometry_linearization as previous

CASES = factor.CASES
ROWS = [0, 1, 2, 9, 17, 33, 49, 60, 63]
SCALES = [1., .5, .25, .125, .0625, .03125]


def nearest(source, target, pose, limit, robust_scale=None):
    """Exhaustive Euclidean matching, minimum input index on exact ties."""
    world = source @ pose[:3, :3].T + pose[:3, 3]
    sources, targets, distances = [], [], []
    if len(target):
        for start in range(0, len(world), 64):
            residual = world[start:start+64, None, :] - target[None, :, :]
            squared = residual[..., 0]**2 + residual[..., 1]**2 + residual[..., 2]**2
            index = squared.argmin(axis=1)
            d2 = squared[np.arange(len(index)), index]
            good = np.flatnonzero(d2 < limit**2)
            sources.extend(start + good)
            targets.extend(index[good])
            distances.extend(d2[good])
    si, ti = np.asarray(sources, dtype=np.uint64), np.asarray(targets, dtype=np.uint64)
    distances = np.asarray(distances)
    weights = np.ones(len(si)) if robust_scale is None else (robust_scale**2/(robust_scale**2+distances))**2
    return si, ti, distances, weights


def retract(pose, center, delta, scale):
    q = Rotation.from_rotvec(scale*delta[3:]).as_matrix()
    trial = np.eye(4)
    trial[:3, :3] = q @ pose[:3, :3]
    trial[:3, 3] = q @ (pose[:3, 3]-center) + center + scale*delta[:3]
    return trial


def frozen_cost(source, target, shapes, pose, weights):
    residual = source @ pose[:3, :3].T + pose[:3, 3] - target
    mahal = np.einsum("ni,ni->n", residual, np.linalg.solve(shapes, residual[..., None])[..., 0])
    return .5*float(weights @ mahal), mahal


def solve(system, radius):
    if radius is None or radius <= 0:
        return None
    d = np.array([1., 1., 1., 1./radius, 1./radius, 1./radius])
    h = system["normal_matrix"]
    a = .5*(h+h.T)/system["weight_sum"] * d[:, None] * d[None, :]
    b = system["right_hand_side"]/system["weight_sum"]*d
    try:
        lower = np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        return None
    if np.any(np.diag(lower)**2 <= 1e-12*np.diag(a).max()):
        return None
    return d*np.linalg.solve(a, b)


def reference(source, target, source_shapes, target_shapes, initial_pose, center,
              *, max_correspondence_distance=2., max_iterations=30,
              translation_tolerance=1e-4, rotation_tolerance=1e-5, robust_scale=None):
    pose, history, accepted = initial_pose.copy(), [], 0
    termination = "max_iterations"
    for _ in range(max_iterations):
        si, ti, d2, weights = nearest(source, target, pose, max_correspondence_distance, robust_scale)
        linear, radius = factor.reference(source[si], target[ti], source_shapes[si], target_shapes[ti], pose, center, weights)
        row = dict(pose_before=pose.copy(), pose_after=pose.copy(), source_indices=si, target_indices=ti,
                   squared_distances=d2, weights=weights, combined_shapes=linear["combined_shapes"],
                   normal_matrix=linear["normal_matrix"], right_hand_side=linear["right_hand_side"],
                   cost_before=linear["cost"], cost_after=None, step=None,
                   trial_scales=np.empty(0), trial_costs=[], accepted_scale=None)
        history.append(row)
        if np.count_nonzero(weights) < 3:
            termination = "insufficient_correspondences"
            break
        delta = solve(linear, radius)
        if delta is None:
            termination = "singular_system"
            break
        row["step"] = delta
        if np.linalg.norm(delta[:3]) <= translation_tolerance and np.linalg.norm(delta[3:]) <= rotation_tolerance:
            termination = "converged_step"
            break
        for scale in SCALES:
            trial = retract(pose, center, delta, scale)
            cost, _ = frozen_cost(source[si], target[ti], linear["combined_shapes"], trial, weights)
            row["trial_scales"] = np.append(row["trial_scales"], scale)
            row["trial_costs"].append(cost)
            if cost < linear["cost"]:
                pose, accepted = trial, accepted+1
                row.update(pose_after=trial.copy(), cost_after=cost, accepted_scale=scale)
                break
        else:
            termination = "no_descent"
            break
    si, ti, d2, weights = nearest(source, target, pose, max_correspondence_distance, robust_scale)
    final, _ = factor.reference(source[si], target[ti], source_shapes[si], target_shapes[ti], pose, center, weights)
    return dict(pose=pose, termination=termination, converged=termination == "converged_step",
                iterations=len(history), accepted_steps=accepted, final_cost=final["cost"],
                matched_source_indices=si, matched_target_indices=ti, squared_distances=d2, weights=weights,
                failure_reason=None, history=history)


def equal_bits(a, b):
    """Recursive exact serial/parallel comparison, including signed zero."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_bits(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_bits(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
    elif isinstance(a, float):
        assert np.float64(a).view(np.uint64) == np.float64(b).view(np.uint64)
    else:
        assert a == b


def compare(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            compare(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected):
            compare(x, y)
    elif isinstance(actual, np.ndarray):
        if actual.dtype.kind in "iu":
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, atol=2e-7 if actual.shape == (4, 4) else 2e-6, rtol=1e-9)
    elif isinstance(actual, float):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-9)
    else:
        assert actual == expected, (actual, expected)


def save_result(path, inputs, result):
    arrays = dict(inputs)
    metadata = {k: v for k, v in result.items() if k != "history" and not isinstance(v, np.ndarray)}
    arrays.update({k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    metadata["history"] = []
    for i, row in enumerate(result["history"]):
        metadata["history"].append({k: v for k, v in row.items() if not isinstance(v, np.ndarray)})
        arrays.update({f"iteration_{i:02d}_{k}": v for k, v in row.items() if isinstance(v, np.ndarray)})
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)


def audit(input_directory, output, slice_protocol, build_inputs):
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    import audit_flatwall_raw_point_time as fields
    from importlib.metadata import version

    if output.exists() or output.is_relative_to(input_directory):
        raise ValueError("use a fresh output outside frozen evidence")
    if os.environ.get("RAYON_NUM_THREADS") != "4" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("use a fresh four-Rayon, one-OpenBLAS process")
    inputs = {p.relative_to(input_directory).as_posix(): fields.artifact(p)
              for p in sorted(input_directory.rglob("*")) if p.is_file()}
    controller, binary = fields.artifact(Path(__file__)), fields.artifact(Path(native.__file__))
    protocol = dict(schema="cloudanalyzer.flatwall_gicp_registration.v1", slice=factor.read(slice_protocol),
                    inputs=inputs, controller=controller, native=binary, cases=CASES, rows=ROWS,
                    build_inputs=fields.artifact(build_inputs), slice_protocol=fields.artifact(slice_protocol),
                    reference_dependencies={p.name: fields.artifact(p) for p in [Path(factor.__file__), Path(previous.__file__)]},
                    versions={name: version(name) for name in ["numpy", "scipy"]})
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output/"protocol.json", protocol)
    rows, summaries = [], []
    params = dict(max_correspondence_distance=2., max_iterations=30,
                  translation_tolerance=1e-4, rotation_tolerance=1e-5, robust_scale=None)
    for case in CASES:
        destination = output/case
        destination.mkdir()
        replay, poses, case_rows = previous.MapReplay(), [], []
        for i in range(64):
            with np.load(input_directory/case/f"frame-{i:02d}.npz", allow_pickle=False) as file:
                data = {k: file[k].copy() for k in file.files}
            if i in ROWS:
                initial = np.eye(4) if not poses else poses[-1].copy()
                if len(poses) >= 2:
                    initial = poses[-1] @ np.linalg.inv(poses[-2]) @ poses[-1]
                center = initial[:3, 3].copy()
                source = data["source_sensor"]
                target = np.asarray([p for cell in replay.cells.values() for p in cell]).reshape(-1, 3)
                source_shapes, good_source = factor.shapes(source, source)
                target_shapes, good_target = factor.shapes(target, target)
                s, t, cs, ct = source[good_source], target[good_target], source_shapes[good_source], target_shapes[good_target]
                actual = core.gicp_register(s, t, cs, ct, initial, center, capture_history=True, **params)
                parallel = core.gicp_register(s, t, cs, ct, initial, center, capture_history=True, parallel=True, **params)
                equal_bits(actual, parallel)
                expected = reference(s, t, cs, ct, initial, center, **params)
                compare(actual, expected)
                cost_errors, shape_errors, system_errors = [], [], []
                for step, ref in zip(actual["history"], expected["history"]):
                    si, ti = step["source_indices"], step["target_indices"]
                    for scale, cost in zip(step["trial_scales"], step["trial_costs"]):
                        candidate = retract(step["pose_before"], center, step["step"], scale)
                        frozen = core.gicp_frozen_cost(s[si], t[ti], step["combined_shapes"], candidate, weights=step["weights"])
                        independent, mahal = frozen_cost(s[si], t[ti], step["combined_shapes"], candidate, step["weights"])
                        np.testing.assert_allclose(frozen["cost"], independent, atol=2e-6, rtol=1e-9)
                        np.testing.assert_allclose(frozen["mahalanobis_squared"], mahal, atol=2e-6, rtol=1e-9)
                        np.testing.assert_allclose(cost, independent, atol=2e-6, rtol=1e-9)
                        cost_errors.append(abs(cost-independent))
                    shape_errors.append(float(np.max(np.abs(step["combined_shapes"]-ref["combined_shapes"]), initial=0)))
                    system_errors.append(float(np.max(np.abs(step["normal_matrix"]-ref["normal_matrix"]), initial=0)))
                    np.testing.assert_allclose(step["combined_shapes"], ref["combined_shapes"], atol=2e-12, rtol=1e-12)
                save_result(destination/f"frame-{i:02d}.npz", dict(source=s, target=t, source_shapes=cs,
                            target_shapes=ct, initial_pose=initial, center_world=center,
                            retained_source_indices=np.flatnonzero(good_source), retained_target_indices=np.flatnonzero(good_target)), actual)
                item = dict(case=case, row=i, source_points=len(s), target_points=len(t),
                            unsupported_source_points=int((~good_source).sum()), unsupported_target_points=int((~good_target).sum()),
                            termination=actual["termination"], iterations=actual["iterations"], accepted_steps=actual["accepted_steps"],
                            final_matches=len(actual["weights"]), final_cost=actual["final_cost"],
                            native_reference_pose_max_error=float(np.max(np.abs(actual["pose"]-expected["pose"]))),
                            max_trial_cost_error=max(cost_errors, default=0.), max_shape_error=max(shape_errors, default=0.),
                            max_system_error=max(system_errors, default=0.), serial_parallel_bit_equal=True,
                            trial_count=sum(len(step["trial_scales"]) for step in actual["history"]),
                            accepted_scales=dict(Counter(str(step["accepted_scale"]) for step in actual["history"] if step["accepted_scale"] is not None)),
                            iterations_summary=[dict(matches=len(step["weights"]), cost_before=step["cost_before"],
                                cost_after=step["cost_after"], accepted_scale=step["accepted_scale"],
                                trial_scales=step["trial_scales"].tolist(), trial_costs=step["trial_costs"]) for step in actual["history"]])
                case_rows.append(item)
                print(json.dumps({k:v for k,v in item.items() if k != "iterations_summary"}), flush=True)
            replay.insert(data["inserted_world"], data["pose"][:3, 3])
            poses.append(data["pose"])
        fields.save(destination/"frames.json", case_rows)
        rows.extend(case_rows)
        summaries.append(dict(case=case, registrations=len(case_rows), statuses=dict(Counter(r["termination"] for r in case_rows)),
                              iterations=sum(r["iterations"] for r in case_rows), accepted_steps=sum(r["accepted_steps"] for r in case_rows),
                              trial_count=sum(r["trial_count"] for r in case_rows),
                              max_pose_error=max(r["native_reference_pose_max_error"] for r in case_rows),
                              max_trial_cost_error=max(r["max_trial_cost_error"] for r in case_rows),
                              max_system_error=max(r["max_system_error"] for r in case_rows),
                              artifacts={p.name: fields.artifact(p) for p in sorted(destination.iterdir())}))
    fields.save(output/"frames.json", rows)
    fields.save(output/"summary.json", summaries)
    for name, expected in inputs.items():
        assert fields.artifact(input_directory/name) == expected, name
    assert controller == fields.artifact(Path(__file__))
    assert binary == fields.artifact(Path(native.__file__))
    receipt = dict(schema=protocol["schema"], protocol=fields.artifact(output/"protocol.json"),
                   controller=controller, native=binary, verified_registrations=len(rows),
                   verified_outer_iterations=sum(r["iterations"] for r in rows),
                   verified_trial_costs=sum(r["trial_count"] for r in rows), serial_parallel_bit_equal=True,
                   frozen_input_artifacts_unchanged=len(inputs), no_trajectory_or_frontend_adoption=True,
                   outputs={p.name: fields.artifact(p) for p in sorted(output.iterdir()) if p.is_file()})
    fields.save(output/"receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["input", "out", "slice-protocol", "build-inputs"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input.resolve(), args.out.resolve(), args.slice_protocol.resolve(), args.build_inputs.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ["outputs", "protocol"]}))


if __name__ == "__main__":
    main()
