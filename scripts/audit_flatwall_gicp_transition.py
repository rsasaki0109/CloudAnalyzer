#!/usr/bin/env python3
"""Separate GICP motion, source-metric rotation and assignment cost changes.

Fixed-support counterfactual costs on captured real poses; no optimizer or
acceptance rule is implemented here. Assignments include target world shapes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np


def _geometry(points, shapes, name):
    points, shapes = np.asarray(points, dtype=np.float64), np.asarray(shapes, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or shapes.shape != (len(points), 3, 3):
        raise ValueError(f"{name} needs (N,3) points and (N,3,3) shapes")
    if not np.isfinite(points).all() or not np.isfinite(shapes).all():
        raise ValueError(f"{name} points and shapes must be finite")
    scale = np.max(np.abs(shapes), axis=(1, 2), initial=0.)
    transpose = shapes.swapaxes(1, 2)
    if np.any(np.abs(shapes-transpose) > 1e-12*scale[:, None, None]):
        raise ValueError(f"{name} shapes must be symmetric")
    canonical = np.where(shapes == transpose, shapes, shapes/2.+transpose/2.)
    try:
        np.linalg.cholesky(canonical)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} shapes must be strictly positive definite") from error
    return points, canonical


def _pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose needs finite shape (4,4)")
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-12, rtol=0.)
            or not np.allclose(rotation@rotation.T, np.eye(3), atol=1e-10, rtol=0.)
            or abs(np.linalg.det(rotation)-1.) > 1e-10):
        raise ValueError("pose must be proper rigid and homogeneous")
    return pose


def _pairs(source_ids, target_ids, source_count, target_count):
    source_ids, target_ids = np.asarray(source_ids), np.asarray(target_ids)
    if (source_ids.ndim != 1 or target_ids.shape != source_ids.shape
            or source_ids.dtype.kind not in "iu" or target_ids.dtype.kind not in "iu"):
        raise ValueError("correspondence IDs need equal-length integer vectors")
    if (np.any(source_ids < 0) or np.any(source_ids >= source_count)
            or np.any(target_ids < 0) or np.any(target_ids >= target_count)
            or len(np.unique(source_ids)) != len(source_ids)):
        raise ValueError("correspondence IDs must be valid and source IDs unique")
    return dict(zip(source_ids.tolist(), target_ids.tolist()))


def _cost(source, target, cs, ct, source_ids, target_ids, metric_pose, residual_pose):
    rotation = metric_pose[:3, :3]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            combined = ct[target_ids]+rotation@cs[source_ids]@rotation.T
            residual = source[source_ids]@residual_pose[:3, :3].T+residual_pose[:3, 3]-target[target_ids]
            mahal = np.einsum("ni,ni->n", residual, np.linalg.solve(combined, residual[..., None])[..., 0])
            cost = .5*float(mahal.sum())
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("nonfinite or invalid transition cost") from error
    if not np.isfinite(combined).all() or not np.isfinite(mahal).all() or not np.isfinite(cost):
        raise ValueError("nonfinite transition cost")
    return cost, mahal, combined


def transition_costs(source, target, source_shapes, target_shapes, pose_before, pose_after,
                     source_before, target_before, source_after, target_after):
    """Return owned cost diagnostics, with cube axes (assignment, metric, pose).

    Both states must have unit weights. The caller checks that captured poses,
    IDs and weights belong to the same transition. Common source IDs are kept
    in original source order; added/removed IDs and costs are separate. A target
    assignment selects both its world point and its world shape.
    """
    source, cs = _geometry(source, source_shapes, "source")
    target, ct = _geometry(target, target_shapes, "target")
    poses = [_pose(pose_before), _pose(pose_after)]
    old = _pairs(source_before, target_before, len(source), len(target))
    new = _pairs(source_after, target_after, len(source), len(target))
    common = np.asarray(sorted(old.keys() & new.keys()), dtype=np.uint64)
    added = np.asarray(sorted(new.keys()-old.keys()), dtype=np.uint64)
    removed = np.asarray(sorted(old.keys()-new.keys()), dtype=np.uint64)
    ids = [np.asarray([pairs[int(i)] for i in common], dtype=np.uint64) for pairs in [old, new]]
    cube, mahal, combined = np.empty((2, 2, 2)), np.empty((2, 2, 2, len(common))), np.empty((2, 2, len(common), 3, 3))
    for assignment in range(2):
        for metric in range(2):
            for pose in range(2):
                cube[assignment, metric, pose], mahal[assignment, metric, pose], combined[assignment, metric] = _cost(
                    source, target, cs, ct, common, ids[assignment], poses[metric], poses[pose])

    def subset_cost(source_ids, pairs, metric, pose):
        target_ids = np.asarray([pairs[int(i)] for i in source_ids], dtype=np.uint64)
        return _cost(source, target, cs, ct, source_ids, target_ids, poses[metric], poses[pose])[0]

    old_ids, new_ids = [np.asarray(sorted(pairs), dtype=np.uint64) for pairs in [old, new]]
    before = subset_cost(old_ids, old, 0, 0)
    frozen_after = subset_cost(old_ids, old, 0, 1)
    refreshed_after = subset_cost(new_ids, new, 1, 1)
    added_cost, removed_cost = subset_cost(added, new, 1, 1), subset_cost(removed, old, 0, 0)
    motion = float(cube[0, 0, 1]-cube[0, 0, 0])
    metric = float(cube[0, 1, 1]-cube[0, 0, 1])
    assignment = float(cube[1, 1, 1]-cube[0, 1, 1])
    assignment_first = float(cube[1, 0, 1]-cube[0, 0, 1])
    metric_second = float(cube[1, 1, 1]-cube[1, 0, 1])
    return dict(common_source_indices=common, common_target_indices=np.asarray(ids),
                added_source_indices=added, removed_source_indices=removed, cube=cube,
                common_mahalanobis_squared=mahal, common_combined_shapes=combined,
                baseline_cost=before, frozen_after_cost=frozen_after, refreshed_after_cost=refreshed_after,
                motion_delta=motion, source_metric_delta=metric, assignment_delta=assignment,
                assignment_first_delta=assignment_first, source_metric_second_delta=metric_second,
                assignment_metric_interaction=assignment-assignment_first,
                added_cost=added_cost, removed_cost=removed_cost,
                common_delta=float(cube[1, 1, 1]-cube[0, 0, 0]),
                full_delta=refreshed_after-before,
                accounting_error=(refreshed_after-before)-(motion+metric+assignment+added_cost-removed_cost))


def _native_check(core, inputs, poses, center, result, tolerance):
    source, target, cs, ct = inputs
    si = result["common_source_indices"]
    max_shape, max_cost = 0., 0.
    for assignment in range(2):
        ti = result["common_target_indices"][assignment]
        for metric in range(2):
            linear = core.gicp_linearization(source[si], target[ti], cs[si], ct[ti], poses[metric], center)
            native_shapes = linear["combined_shapes"]
            independent = result["common_combined_shapes"][assignment, metric]
            np.testing.assert_allclose(independent, native_shapes, atol=tolerance["combined_shape_atol"], rtol=tolerance["combined_shape_rtol"])
            max_shape = max(max_shape, float(np.max(np.abs(independent-native_shapes), initial=0.)))
            for pose in range(2):
                native = core.gicp_frozen_cost(source[si], target[ti], native_shapes, poses[pose])
                expected = result["cube"][assignment, metric, pose]
                np.testing.assert_allclose(native["cost"], expected, atol=tolerance["cost_atol"], rtol=tolerance["cost_rtol"])
                np.testing.assert_allclose(native["mahalanobis_squared"], result["common_mahalanobis_squared"][assignment, metric, pose], atol=tolerance["cost_atol"], rtol=tolerance["cost_rtol"])
                max_cost = max(max_cost, abs(native["cost"]-expected))
    return max_shape, max_cost


def audit(inputs, output, slice_protocol, reused_native):
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    import audit_flatwall_gicp_linearization as factor
    import audit_flatwall_gicp_registration as registration
    import audit_flatwall_gicp_shapes as previous
    import audit_flatwall_raw_point_time as fields

    if output.exists() or output.is_relative_to(inputs):
        raise ValueError("use a fresh output outside original evidence")
    if os.environ.get("RAYON_NUM_THREADS") != "4" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("use a fresh four-Rayon, one-OpenBLAS process")
    frozen = {p.relative_to(inputs).as_posix(): fields.artifact(p) for p in sorted(inputs.rglob("*")) if p.is_file()}
    controller, binary = fields.artifact(Path(__file__)), fields.artifact(Path(native.__file__))
    binding = factor.read(reused_native)
    assert binary["sha256"] == binding["native_sha256"]
    for name, expected in binding["package_sha256"].items():
        assert fields.artifact(Path(native.__file__).parent/name)["sha256"] == expected, name
    plan = factor.read(slice_protocol)
    tolerance = plan["numeric"]
    output.mkdir(parents=True)
    fields.save(output/"protocol.json", dict(schema="cloudanalyzer.flatwall_gicp_transition.v1", slice=plan,
        input_artifacts=frozen, controller=controller, native=binary, reused_native=fields.artifact(reused_native),
        slice_protocol=fields.artifact(slice_protocol), reference_dependencies={p.name: fields.artifact(p) for p in [Path(factor.__file__), Path(registration.__file__), Path(previous.__file__), Path(fields.__file__)]},
        versions={name: version(name) for name in ["numpy", "scipy", "rosbags"]},
        cube_axis_order=["assignment", "source_metric_pose", "residual_pose"]))
    frames, transitions, limited_tails, summaries = [], [], [], []
    for path in sorted(inputs.rglob("frame-*.npz")):
        relative = path.relative_to(inputs)
        family, case = relative.parts[:2]
        row_number = int(path.stem[6:])
        with np.load(path, allow_pickle=False) as arrays:
            data = {key: arrays[key].copy() for key in arrays.files}
        captured = previous.load_registration(data)
        geometry = [data[name] for name in ["source", "target", "source_shapes", "target_shapes"]]
        rows, saved = [], {}
        for index, item in enumerate(captured["history"]):
            if item["accepted_scale"] is None:
                continue
            next_item = captured["history"][index+1] if index+1 < len(captured["history"]) else None
            if next_item is None:
                next_si, next_ti, next_weights = [captured[name] for name in ["matched_source_indices", "matched_target_indices", "weights"]]
                next_cost = captured["final_cost"]
                previous.equal_bits(item["pose_after"], captured["pose"])
            else:
                next_si, next_ti, next_weights = [next_item[name] for name in ["source_indices", "target_indices", "weights"]]
                next_cost = next_item["cost_before"]
                previous.equal_bits(item["pose_after"], next_item["pose_before"])
            assert np.array_equal(item["weights"], np.ones(len(item["weights"])))
            assert np.array_equal(next_weights, np.ones(len(next_weights)))
            poses = [item["pose_before"], item["pose_after"]]
            result = transition_costs(*geometry, *poses, item["source_indices"], item["target_indices"], next_si, next_ti)
            positions = {int(source_id): position for position, source_id in enumerate(item["source_indices"])}
            common_positions = np.asarray([positions[int(i)] for i in result["common_source_indices"]], dtype=np.int64)
            np.testing.assert_allclose(result["common_combined_shapes"][0, 0], item["combined_shapes"][common_positions],
                                       atol=tolerance["combined_shape_atol"], rtol=tolerance["combined_shape_rtol"])
            for name, expected in [("baseline_cost", item["cost_before"]), ("frozen_after_cost", item["cost_after"]), ("refreshed_after_cost", next_cost)]:
                np.testing.assert_allclose(result[name], expected, atol=tolerance["cost_atol"], rtol=tolerance["cost_rtol"])
            np.testing.assert_allclose(result["accounting_error"], 0., atol=tolerance["accounting_atol"], rtol=0.)
            np.testing.assert_allclose(result["common_delta"], result["motion_delta"]+result["assignment_first_delta"]+result["source_metric_second_delta"], atol=tolerance["accounting_atol"], rtol=tolerance["accounting_rtol"])
            shape_error, cost_error = _native_check(core, geometry, poses, data["center_world"], result, tolerance)
            item_summary = dict(family=family, case=case, row=row_number, iteration=index+1, termination=captured["termination"],
                common_sources=len(result["common_source_indices"]), added_sources=len(result["added_source_indices"]), removed_sources=len(result["removed_source_indices"]),
                changed_targets=int(np.count_nonzero(result["common_target_indices"][0] != result["common_target_indices"][1])),
                cube=result["cube"].reshape(-1).tolist(), max_native_shape_error=shape_error, max_native_cost_error=cost_error,
                **{key: value for key, value in result.items() if not isinstance(value, np.ndarray)})
            rows.append(item_summary)
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    saved[f"transition_{index:02d}_{key}"] = value
        destination = output/relative.parent
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination/path.name, **saved)
        frame = dict(family=family, case=case, row=row_number, termination=captured["termination"],
            accepted_transitions=len(rows), refreshed_increases=sum(r["full_delta"] > 0. for r in rows),
            common_increases=sum(r["common_delta"] > 0. for r in rows),
            support_changed_transitions=sum(bool(r["added_sources"] or r["removed_sources"]) for r in rows),
            max_native_cost_error=max((r["max_native_cost_error"] for r in rows), default=0.),
            max_native_shape_error=max((r["max_native_shape_error"] for r in rows), default=0.))
        frames.append(frame)
        transitions.extend(rows)
        if captured["termination"] == "max_iterations":
            limited_tails.extend(rows[-10:])
        print(json.dumps(frame), flush=True)
    for family, case in sorted({(r["family"], r["case"]) for r in frames}):
        selected = [r for r in frames if (r["family"], r["case"]) == (family, case)]
        rows = [r for r in transitions if (r["family"], r["case"]) == (family, case)]
        summaries.append(dict(family=family, case=case, frames=len(selected), statuses=dict(Counter(r["termination"] for r in selected)),
            accepted_transitions=len(rows), refreshed_increases=sum(r["full_delta"] > 0. for r in rows),
            common_increases=sum(r["common_delta"] > 0. for r in rows),
            support_changed_transitions=sum(bool(r["added_sources"] or r["removed_sources"]) for r in rows),
            max_native_cost_error=max((r["max_native_cost_error"] for r in rows), default=0.),
            max_native_shape_error=max((r["max_native_shape_error"] for r in rows), default=0.),
            max_accounting_error=max((abs(r["accounting_error"]) for r in rows), default=0.)))
    assert len(frames) == 72 and len(transitions) == 686 and len(limited_tails) == 100
    for name, expected in frozen.items():
        assert fields.artifact(inputs/name) == expected, name
    assert fields.artifact(Path(__file__)) == controller and fields.artifact(Path(native.__file__)) == binary
    for name, value in [("frames.json", frames), ("transitions.json", transitions), ("limited-tails.json", limited_tails), ("summary.json", summaries)]:
        fields.save(output/name, value)
    fields.save(output/"receipt.json", dict(schema="cloudanalyzer.flatwall_gicp_transition.v1", controller=controller, native=binary,
        histories=len(frames), accepted_transitions=len(transitions), native_cube_cost_checks=8*len(transitions),
        native_combined_shape_checks=4*len(transitions), limited_tail_transitions=len(limited_tails),
        original_input_artifacts_unchanged=len(frozen), original_poses_and_statuses_unchanged=True,
        outputs={name: fields.artifact(output/name) for name in ["protocol.json", "frames.json", "transitions.json", "limited-tails.json", "summary.json"]},
        scope="Fixed-support costs at captured poses with explicit support accounting; no new optimizer, accepted pose, acceptance rule, physical accuracy or speed result."))
    print(json.dumps(summaries), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--slice-protocol", required=True, type=Path)
    parser.add_argument("--reused-native", required=True, type=Path)
    args = parser.parse_args()
    audit(*(p.resolve() for p in [args.input, args.out, args.slice_protocol, args.reused_native]))


if __name__ == "__main__":
    main()
