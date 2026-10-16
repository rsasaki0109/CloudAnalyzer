#!/usr/bin/env python3
"""Fixed-match real-recording GICP factor audit, without state updates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import audit_flatwall_odometry_linearization as previous

CASES = ["raw-legacy", "raw-fine", "rotation-legacy", "rotation-fine"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def shapes(points, queries):
    """Declared normalized PCA shape model; unsupported rows remain excluded."""
    result = np.zeros((len(queries), 3, 3))
    valid = np.zeros(len(queries), dtype=bool)
    if len(points) < 5 or not len(queries):
        return result, valid
    _, indices = cKDTree(points).query(queries, k=min(20, len(points)), workers=1)
    for row, neighbors in enumerate(indices):
        local = points[neighbors]
        local = local - local.mean(axis=0)
        values, vectors = np.linalg.eigh(local.T @ local / len(local))
        if not np.isfinite(values).all() or values[-1] <= 0 or values[1] <= 1e-12 * values[-1]:
            continue
        normal = vectors[:, 0]
        result[row] = np.eye(3) + (.001 - 1.) * np.outer(normal, normal)
        valid[row] = True
    return result, valid


def reference(source, target, source_shapes, target_shapes, pose, center, weights):
    world = source @ pose[:3, :3].T + pose[:3, 3]
    arm, residual = world-center, world-target
    combined = target_shapes + pose[:3, :3] @ source_shapes @ pose[:3, :3].T
    jacobian = np.zeros((len(source), 3, 6))
    jacobian[:, :, :3] = np.eye(3)
    x, y, z = arm.T
    jacobian[:, 0, 4], jacobian[:, 0, 5] = z, -y
    jacobian[:, 1, 3], jacobian[:, 1, 5] = -z, x
    jacobian[:, 2, 3], jacobian[:, 2, 4] = y, -x
    weighted_residual = np.linalg.solve(combined, residual[..., None])[..., 0]
    weighted_jacobian = np.linalg.solve(combined, jacobian)
    mahalanobis = np.einsum("ni,ni->n", residual, weighted_residual)
    h = np.einsum("nri,n,nrj->ij", jacobian, weights, weighted_jacobian)
    b = -np.einsum("nri,n,nr->i", jacobian, weights, weighted_residual)
    support = float(weights.sum())
    radius = float(np.sqrt(np.sum(weights*np.sum(arm*arm, axis=1))/support)) if support > 0 else None
    return {"normal_matrix": h, "right_hand_side": b, "source_world": world,
            "combined_shapes": combined, "mahalanobis_squared": mahalanobis,
            "cost": .5*float(weights @ mahalanobis), "weight_sum": support,
            "positive_weight_points": int(np.count_nonzero(weights))}, radius


def equal_bits(a, b):
    for name, value in a.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value.view(np.uint64), b[name].view(np.uint64))
        elif isinstance(value, float):
            assert np.float64(value).view(np.uint64) == np.float64(b[name]).view(np.uint64), name
        else:
            assert value == b[name], name


def audit(input_directory, output, slice_protocol, build_inputs):
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    import audit_flatwall_raw_point_time as fields
    from importlib.metadata import version

    if output.exists() or output.is_relative_to(input_directory):
        raise ValueError("use a fresh output outside frozen evidence")
    if os.environ.get("RAYON_NUM_THREADS") != "4" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("use a fresh four-Rayon, one-OpenBLAS process")
    frozen = {path.relative_to(input_directory).as_posix(): fields.artifact(path)
              for path in input_directory.rglob("*") if path.is_file()}
    controller = fields.artifact(Path(__file__))
    native_artifact = fields.artifact(Path(native.__file__))
    protocol = {"schema": "cloudanalyzer.flatwall_gicp_linearization.v1", "slice": read(slice_protocol),
                "cases": CASES, "frame_count": 64, "inputs": {"captured": frozen,
                "controller": controller, "native": native_artifact,
                "map_replay_controller": fields.artifact(Path(previous.__file__)),
                "slice_protocol": fields.artifact(slice_protocol),
                "build_inputs": fields.artifact(build_inputs)},
                "versions": {name: version(name) for name in ["numpy", "scipy"]},
                "scope": "Declared normalized surface shapes on fixed actual matches only; no trajectory, solver, covariance calibration, physical observability/accuracy, timing, GPU or state adoption."}
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output / "protocol.json", protocol)
    all_frames, summaries = [], []
    for case in CASES:
        destination = output / case
        destination.mkdir()
        metadata = read(input_directory / case / "snapshot-metadata.json")
        replay = previous.MapReplay()
        rows = []
        for row, meta in enumerate(metadata):
            with np.load(input_directory / case / f"frame-{row:02d}.npz", allow_pickle=False) as arrays:
                snapshot = {name: arrays[name].copy() for name in arrays.files}
            indices = snapshot["matched_source_indices"]
            source = snapshot["source_sensor"][indices]
            target = snapshot["target_world"]
            target_map = np.asarray([p for cell in replay.cells.values() for p in cell]).reshape(-1, 3)
            source_shape, good_source = shapes(snapshot["source_sensor"], source)
            target_shape, good_target = shapes(target_map, target)
            good = good_source & good_target
            source, target = source[good], target[good]
            source_shape, target_shape = source_shape[good], target_shape[good]
            weights = snapshot["weights"][good]
            pose, center = snapshot["pose"], snapshot["center_world"]
            actual = core.gicp_linearization(source, target, source_shape, target_shape, pose, center, weights=weights)
            parallel = core.gicp_linearization(source, target, source_shape, target_shape, pose, center, weights=weights, parallel=True)
            equal_bits(actual, parallel)
            expected, radius = reference(source, target, source_shape, target_shape, pose, center, weights)
            np.testing.assert_array_equal(actual["source_world"].view(np.uint64), snapshot["source_world"][indices[good]].view(np.uint64))
            for field in ["normal_matrix", "right_hand_side", "cost", "weight_sum"]:
                np.testing.assert_allclose(actual[field], expected[field], atol=2e-6, rtol=1e-10)
            np.testing.assert_allclose(actual["source_world"], expected["source_world"], atol=2e-8, rtol=0.)
            np.testing.assert_allclose(actual["combined_shapes"], expected["combined_shapes"], atol=2e-12, rtol=1e-12)
            np.testing.assert_allclose(actual["mahalanobis_squared"], expected["mahalanobis_squared"], atol=2e-9, rtol=1e-10)
            identity_shapes = np.tile(np.eye(3)*.5, (len(source), 1, 1))
            isotropic = core.gicp_linearization(source, target, identity_shapes, identity_shapes, pose, center, weights=weights)
            isotropic_parallel = core.gicp_linearization(source, target, identity_shapes, identity_shapes, pose, center, weights=weights, parallel=True)
            equal_bits(isotropic, isotropic_parallel)
            # An exact identity matrix in world axes gives the independent P2P control.
            p2p_shape = np.tile(np.eye(3), (len(source), 1, 1))
            p2p, _ = reference(source, target, np.zeros_like(p2p_shape), p2p_shape, pose, center, weights)
            for field in ["normal_matrix", "right_hand_side", "cost"]:
                np.testing.assert_allclose(isotropic[field], p2p[field], atol=2e-6, rtol=1e-10)
            arrays = {"source": source, "target": target, "source_shapes": source_shape,
                      "target_shapes": target_shape, "weights": weights, "pose": pose, "center_world": center,
                      "retained_match_rows": np.flatnonzero(good),
                      "gicp_normal_matrix": actual["normal_matrix"], "gicp_rhs": actual["right_hand_side"],
                      "gicp_combined_shapes": actual["combined_shapes"], "gicp_mahalanobis_squared": actual["mahalanobis_squared"],
                      "p2p_normal_matrix": p2p["normal_matrix"], "p2p_rhs": p2p["right_hand_side"]}
            np.savez_compressed(destination / f"frame-{row:02d}.npz", **arrays)
            item = {"case": case, "row": row, "original_matches": len(indices), "retained_pairs": len(source),
                    "unsupported_source_pairs": int(np.count_nonzero(~good_source)),
                    "unsupported_target_pairs": int(np.count_nonzero(~good_target)),
                    "weight_sum": expected["weight_sum"], "radius_m": radius,
                    "gicp_geometry": previous.spectrum(actual["normal_matrix"], expected["weight_sum"], radius),
                    "p2p_geometry": previous.spectrum(p2p["normal_matrix"], expected["weight_sum"], radius),
                    "normal_matrix_max_error": float(np.max(np.abs(actual["normal_matrix"]-expected["normal_matrix"]))),
                    "rhs_max_error": float(np.max(np.abs(actual["right_hand_side"]-expected["right_hand_side"]))),
                    "cost_abs_error": abs(actual["cost"]-expected["cost"]),
                    "mahalanobis_max_error": float(np.max(np.abs(actual["mahalanobis_squared"]-expected["mahalanobis_squared"]), initial=0.)),
                    "identity_control_normal_matrix_max_error": float(np.max(np.abs(isotropic["normal_matrix"]-p2p["normal_matrix"])))}
            rows.append(item)
            replay.insert(snapshot["inserted_world"], snapshot["pose"][:3, 3])
        fields.save(destination / "frames.json", rows)
        supported = [r for r in rows if r["gicp_geometry"] is not None]
        summary = {"case": case, "frames": len(rows), "supported_frames": len(supported),
                   "original_pairs": sum(r["original_matches"] for r in rows), "retained_pairs": sum(r["retained_pairs"] for r in rows),
                   "serial_parallel_bit_equal": True, "identity_controls_match_p2p": True,
                   "geometry": {model: {"weak_frames": sum(r[model+"_geometry"]["weak_directions_below_1e-3"] > 0 for r in supported),
                       "min_ratio": min(r[model+"_geometry"]["min_max_ratio"] for r in supported),
                       "median_ratio": float(np.median([r[model+"_geometry"]["min_max_ratio"] for r in supported]))}
                       for model in ["gicp", "p2p"]},
                   "max_normal_matrix_error": max(r["normal_matrix_max_error"] for r in rows),
                   "max_rhs_error": max(r["rhs_max_error"] for r in rows),
                   "max_cost_error": max(r["cost_abs_error"] for r in rows),
                   "artifacts": {p.name: fields.artifact(p) for p in sorted(destination.iterdir())}}
        summaries.append(summary)
        all_frames.extend(rows)
        print(json.dumps({k:v for k,v in summary.items() if k != "artifacts"}), flush=True)
    fields.save(output / "frames.json", all_frames)
    fields.save(output / "summary.json", summaries)
    assert controller == fields.artifact(Path(__file__))
    assert native_artifact == fields.artifact(Path(native.__file__))
    for name, expected in frozen.items():
        assert fields.artifact(input_directory / name) == expected, name
    receipt = {"schema": protocol["schema"], "protocol": fields.artifact(output / "protocol.json"),
               "controller": controller, "native": native_artifact, "verified_frames": len(all_frames),
               "serial_parallel_bit_equal": True, "identity_controls_match_p2p": True,
               "frozen_input_artifacts_unchanged": len(frozen), "no_states_adopted": True,
               "outputs": {p.name: fields.artifact(p) for p in sorted(output.iterdir()) if p.is_file()}}
    fields.save(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["input", "out", "slice-protocol", "build-inputs"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input.resolve(), args.out.resolve(), args.slice_protocol.resolve(), args.build_inputs.resolve())
    print(json.dumps({"verified_frames": result["verified_frames"]}))


if __name__ == "__main__":
    main()
