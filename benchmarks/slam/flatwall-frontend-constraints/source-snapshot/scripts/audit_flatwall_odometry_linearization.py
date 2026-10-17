#!/usr/bin/env python3
"""Independent final-pose fixed-match point-to-point audit; no timing or GT."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

OFFSETS = [(0, 0, 0)] + [p for p in itertools.product((-1, 0, 1), repeat=3) if any(p)]
LIDAR = dict(min_range=1.5, max_range=80., map_voxel=1., map_points=20,
             deskew=False, diagnostics=True)
CASES = [("raw-legacy", False, None, "omitted"),
         ("raw-fine", False, .1875, "explicit-0.1875"),
         ("rotation-legacy", True, None, "legacy"),
         ("rotation-fine", True, .1875, "explicit-0.1875")]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def selected(points, voxel):
    """First input point per floor cell, without native voxel helpers."""
    seen, output = set(), []
    for point in points:
        key = tuple(math.floor(float(v) / voxel) for v in point)
        if key not in seen:
            seen.add(key)
            output.append(point)
    return np.asarray(output, dtype=np.float64).reshape(-1, 3)


def filtered(points):
    output = []
    for x, y, z in points:
        length = math.sqrt(x*x + y*y + z*z)
        if LIDAR["min_range"] <= length <= LIDAR["max_range"]:
            output.append((x, y, z))
    return np.asarray(output, dtype=np.float64).reshape(-1, 3)


def transformed(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


class MapReplay:
    """Exhaustive 27-cell search; no native cell-bound pruning or hashmap order."""
    def __init__(self, voxel=1., capacity=20, max_range=80.):
        self.voxel, self.capacity, self.max_range = voxel, capacity, max_range
        self.cells = {}

    def key(self, point):
        return tuple(math.floor(float(v) / self.voxel) for v in point)

    def matches(self, points, threshold_sq):
        indices, targets, distances = [], [], []
        for index, point in enumerate(points):
            key, best, target = self.key(point), math.inf, None
            for offset in OFFSETS:
                for candidate in self.cells.get(tuple(a+b for a, b in zip(key, offset)), ()):
                    dx, dy, dz = point - candidate
                    distance = dx*dx + dy*dy + dz*dz
                    if distance < best:
                        best, target = distance, candidate
            if target is not None and best <= threshold_sq:
                indices.append(index)
                targets.append(target)
                distances.append(best)
        return (np.asarray(indices, dtype=np.int64),
                np.asarray(targets, dtype=np.float64).reshape(-1, 3),
                np.asarray(distances, dtype=np.float64))

    def insert(self, points, center):
        for point in points:
            cell = self.cells.setdefault(self.key(point), [])
            if len(cell) < self.capacity:
                cell.append(point.copy())
        kept = {}
        for key, points in self.cells.items():
            dx, dy, dz = points[0] - center
            if math.sqrt(dx*dx + dy*dy + dz*dz) <= self.max_range:
                kept[key] = points
        self.cells = kept


def reconstruct(snapshot):
    q = snapshot["source_world"][snapshot["matched_source_indices"]]
    arm = q - snapshot["center_world"]
    residual = q - snapshot["target_world"]
    jacobian = np.zeros((len(q), 3, 6))
    jacobian[:, :, :3] = np.eye(3)
    x, y, z = arm.T
    jacobian[:, 0, 4], jacobian[:, 0, 5] = z, -y
    jacobian[:, 1, 3], jacobian[:, 1, 5] = -z, x
    jacobian[:, 2, 3], jacobian[:, 2, 4] = y, -x
    distance_sq = np.sum(residual * residual, axis=1)
    weights = snapshot["kernel"] ** 2 / (snapshot["kernel"] + distance_sq) ** 2
    h = np.einsum("nri,n,nrj->ij", jacobian, weights, jacobian)
    b = -np.einsum("nri,n,nr->i", jacobian, weights, residual)
    weight_sum = float(weights.sum())
    radius = (float(np.sqrt(np.sum(weights * np.sum(arm * arm, axis=1)) / weight_sum))
              if weight_sum > 0 else None)
    return weights, h, b, weight_sum, radius


def spectrum(h, weight_sum, radius):
    if not np.isfinite(h).all() or not (weight_sum > 0) or radius is None or not (radius > 0):
        return None
    scale = np.array([1., 1., 1., 1./radius, 1./radius, 1./radius])
    normalized = h * scale[:, None] * scale[None, :] / weight_sum
    values = np.linalg.eigvalsh((normalized + normalized.T) / 2.)
    # Preserve signed eigenvalues; no clipping, inversion, damping or state feedback.
    return {"eigenvalues": values.tolist(), "min_max_ratio": float(values[0] / values[-1]),
            "weak_directions_below_1e-3": int(np.count_nonzero(values / values[-1] < 1e-3))}


def audit(bag, output, raw_previous, rotation_previous, slice_protocol):
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    import audit_flatwall_raw_point_time as fields
    from importlib.metadata import version

    if os.environ.get("RAYON_NUM_THREADS") != "4" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("use a fresh process with four Rayon and one OpenBLAS threads")
    if output.exists():
        raise ValueError("use a fresh output directory")
    inputs = {"recording": fields.artifact(bag), "controller": fields.artifact(Path(__file__)),
              "helpers": fields.artifact(Path(fields.__file__)), "native": fields.artifact(Path(native.__file__)),
              "slice_protocol": fields.artifact(slice_protocol),
              "numpy_version": version("numpy"), "previous": {}}
    assert inputs["recording"]["sha256"] == fields.BAG_SHA256
    old = {}
    for name, rotation, _, folder in CASES:
        directory = (rotation_previous if rotation else raw_previous) / folder
        inputs["previous"][name] = {filename: fields.artifact(directory / filename)
                                    for filename in ["startup-poses.npz", "lidar-diagnostics.json", "receipt.json"]}
        with np.load(directory / "startup-poses.npz", allow_pickle=False) as arrays:
            old[name] = {key: arrays[key].copy() for key in arrays.files}
        old[name]["reports"] = read(directory / "lidar-diagnostics.json")
    protocol = {"schema": "cloudanalyzer.flatwall_odometry_linearization.v1", "inputs": inputs,
                "slice": read(slice_protocol), "cases": [c[0] for c in CASES], "lidar": LIDAR,
                "scan_indices": list(range(1, 65)), "weak_ratio": 1e-3,
                "scope": "Conditional fixed-match uncalibrated point-to-point geometry only; no physical observability, covariance, GICP, accuracy, causal, timing or GPU claim."}
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output / "protocol.json", protocol)
    reader = core.BagReader(str(bag))
    series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
    imu_times = series.timestamps_ns()
    trajectory = series.rotation_trajectory(int(imu_times[0]), int(imu_times[-1]), gyro_bias=np.zeros(3))
    scans = []
    for index, scan in enumerate(reader.scans("/livox/points", raw_time_field="t")):
        if index > 64:
            break
        if index:
            stamps = scan.point_times_ns(unit="ns", origin="header")
            scans.append((scan.positions(), stamps, int(stamps.max())))
    assert len(scans) == 64
    all_rows, cases = [], []
    for name, rotation, voxel, _ in CASES:
        destination = output / name
        destination.mkdir()
        odometries = {enabled: core.LidarOdometry(**LIDAR, registration_voxel=voxel,
                                                 capture_linearization=enabled) for enabled in [False, True]}
        replay = MapReplay()
        rows, poses, metadata = [], [], []
        for row, (raw_points, stamps, reference) in enumerate(scans):
            assert reference == int(old[name]["timestamps_ns"][row])
            points = (trajectory.deskew(raw_points, stamps, reference, imu_from_lidar=np.eye(4))
                      if rotation else raw_points)
            for enabled, odometry in odometries.items():
                pose = odometry.register(points)
                np.testing.assert_array_equal(pose.view(np.uint64), old[name]["lidar_poses"][row].view(np.uint64))
                assert odometry.last_report() == old[name]["reports"][row]
                if not enabled:
                    assert odometry.last_linearization() is None
            poses.append(pose)
            snapshot = odometries[True].last_linearization()
            assert snapshot["finite"]
            assert (snapshot["map_voxel"], snapshot["map_points"], snapshot["max_range"]) == (1., 20, 80.)
            np.testing.assert_allclose(snapshot["threshold_sq"], odometries[True].last_report()["threshold_m"]**2, rtol=2e-15)
            np.testing.assert_allclose(snapshot["kernel"], odometries[True].last_report()["threshold_m"] / 9., rtol=2e-15)
            if row == 0:
                expected_center = np.zeros(3)
            elif row == 1:
                expected_center = old[name]["lidar_poses"][row-1, :3, 3]
            else:
                previous, last = old[name]["lidar_poses"][row-2:row]
                motion = np.eye(4)
                motion[:3, :3] = previous[:3, :3].T @ last[:3, :3]
                motion[:3, 3] = previous[:3, :3].T @ (last[:3, 3] - previous[:3, 3])
                expected_center = (last @ motion)[:3, 3]
            np.testing.assert_allclose(snapshot["center_world"], expected_center, atol=2e-8, rtol=0.)
            np.testing.assert_array_equal(snapshot["pose"].view(np.uint64), pose.view(np.uint64))
            near = filtered(points)
            frame = selected(near, .5)
            source = selected(frame, 1.5) if voxel is None else selected(near, voxel)
            np.testing.assert_array_equal(snapshot["source_sensor"].view(np.uint64), source.view(np.uint64))
            world_error = float(np.max(np.abs(transformed(source, pose) - snapshot["source_world"]), initial=0.))
            insertion_error = float(np.max(np.abs(transformed(frame, pose) - snapshot["inserted_world"]), initial=0.))
            assert max(world_error, insertion_error) <= 2e-8
            assert snapshot["alignment_eligible"] == bool(replay.cells and len(source) >= 10)
            indices, targets, d2 = (replay.matches(snapshot["source_world"], snapshot["threshold_sq"])
                                    if snapshot["alignment_eligible"] else
                                    (np.empty(0, dtype=np.int64), np.empty((0, 3)), np.empty(0)))
            np.testing.assert_array_equal(snapshot["matched_source_indices"], indices)
            np.testing.assert_array_equal(snapshot["target_world"].view(np.uint64), targets.view(np.uint64))
            np.testing.assert_array_equal(snapshot["squared_distances"].view(np.uint64), d2.view(np.uint64))
            weights, h, b, weight_sum, radius = reconstruct(snapshot)
            np.testing.assert_allclose(snapshot["weights"], weights, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(snapshot["normal_matrix"], h, atol=2e-8, rtol=2e-12)
            np.testing.assert_allclose(snapshot["right_hand_side"], b, atol=2e-8, rtol=2e-12)
            np.testing.assert_allclose(snapshot["weight_sum"], weight_sum, atol=2e-12, rtol=2e-12)
            if radius is None:
                assert snapshot["weighted_rms_radius_m"] is None
            else:
                np.testing.assert_allclose(snapshot["weighted_rms_radius_m"], radius, atol=2e-12, rtol=2e-12)
            report = odometries[True].last_report()
            if snapshot["alignment_eligible"]:
                assert report["correspondences"]["matched_points"] == len(indices)
            arrays = {key: value for key, value in snapshot.items() if isinstance(value, np.ndarray)}
            np.savez_compressed(destination / f"frame-{row:02d}.npz", **arrays)
            metadata.append({key: value for key, value in snapshot.items() if key not in arrays})
            item = {"case": name, "row": row, "scan_index": row+1, "reference_ns": reference,
                    "source_points": len(source), "matches": len(indices), "weight_sum": weight_sum,
                    "radius_m": radius, "geometry": spectrum(h, weight_sum, radius),
                    "normal_matrix_max_error": float(np.max(np.abs(snapshot["normal_matrix"] - h))),
                    "rhs_max_error": float(np.max(np.abs(snapshot["right_hand_side"] - b))),
                    "source_world_max_error_m": world_error, "insertion_max_error_m": insertion_error}
            rows.append(item)
            replay.insert(snapshot["inserted_world"], pose[:3, 3])
        fields.save(destination / "snapshot-metadata.json", metadata)
        np.savez_compressed(destination / "startup-poses.npz", timestamps_ns=old[name]["timestamps_ns"], lidar_poses=np.asarray(poses))
        fields.save(destination / "frames.json", rows)
        supported = [r["geometry"] for r in rows if r["geometry"] is not None]
        summary = {"case": name, "frames": 64, "capture_modes_exact_old_pose_bits_and_reports": True,
                   "supported_frames": len(supported),
                   "weak_frame_count": sum(g["weak_directions_below_1e-3"] > 0 for g in supported),
                   "min_ratio": min(g["min_max_ratio"] for g in supported),
                   "median_ratio": float(np.median([g["min_max_ratio"] for g in supported])),
                   "max_normal_matrix_error": max(r["normal_matrix_max_error"] for r in rows),
                   "max_rhs_error": max(r["rhs_max_error"] for r in rows),
                   "artifacts": {p.name: fields.artifact(p) for p in sorted(destination.iterdir())}}
        cases.append(summary)
        all_rows.extend(rows)
        print(json.dumps({key: value for key, value in summary.items() if key != "artifacts"}), flush=True)
    fields.save(output / "frames.json", all_rows)
    fields.save(output / "summary.json", cases)
    assert inputs["recording"] == fields.artifact(bag)
    assert inputs["controller"] == fields.artifact(Path(__file__))
    assert inputs["native"] == fields.artifact(Path(native.__file__))
    receipt = {"schema": protocol["schema"], "inputs": inputs, "cases": [c["case"] for c in cases],
               "verified_frames": len(all_rows), "pose_sequences_compared": 8,
               "protocol": fields.artifact(output / "protocol.json"),
               "outputs": {p.name: fields.artifact(p) for p in sorted(output.iterdir()) if p.is_file()}}
    fields.save(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bag", "out", "raw-previous", "rotation-previous", "slice-protocol"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bag.resolve(), args.out.resolve(), args.raw_previous.resolve(),
                   args.rotation_previous.resolve(), args.slice_protocol.resolve())
    print(json.dumps({"verified_frames": result["verified_frames"]}))


if __name__ == "__main__":
    main()
