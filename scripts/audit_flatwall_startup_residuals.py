#!/usr/bin/env python3
"""Frozen-pose startup residual decomposition and reference/clock sensitivity."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib import import_module
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path

import numpy as np

PARAMS = dict(
    gyro_bias=(0.0,) * 3,
    accel_bias=(0.0,) * 3,
    gravity_magnitude=9.80665,
    max_gravity_error=1.0,
    max_rotation_rms=0.05,
    max_velocity_rms=0.2,
    max_gyro_bias_change=0.2,
    max_iterations=15,
    max_gap_s=0.05,
    min_singular_ratio=1e-6,
)
STRIDES = [1, 3, 7]
OFFSETS_NS = list(range(-50_000_000, 50_000_001, 5_000_000))
SOURCES = [
    ("raw-legacy", "raw", "omitted"),
    ("raw-fine", "raw", "explicit-0.1875"),
    ("rotation-legacy", "rotation", "legacy"),
    ("rotation-fine", "rotation", "explicit-0.1875"),
]


def shifted_references(stamps, offset_ns):
    stamps = np.asarray(stamps)
    if stamps.dtype != np.dtype("int64") or stamps.ndim != 1 or len(stamps) < 3:
        raise ValueError("need at least three int64 references")
    limit = np.iinfo(np.int64)
    if any(
        not 0 < int(b) - int(a) <= limit.max for a, b in zip(stamps[:-1], stamps[1:])
    ):
        raise ValueError("references must strictly increase without interval overflow")
    shifted = [int(t) + int(offset_ns) for t in stamps]
    if any(t < limit.min or t > limit.max for t in shifted):
        raise ValueError("reference clock offset overflows int64")
    return np.asarray(shifted, dtype=np.int64)


def reference_indices(count, stride):
    if stride <= 0 or count < 3 or (count - 1) % stride or (count - 1) // stride < 2:
        raise ValueError(
            "reference stride must preserve both endpoints and at least three poses"
        )
    return np.arange(0, count, stride, dtype=np.int64)


def decompose(position, dt, world_dp, world_dv):
    """Separate pose and integrated-IMU terms through the same gravity projection.

    This is an algebraic decomposition, not a causal sensor fault diagnosis.
    """
    position = np.asarray(position, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    world_dp = np.asarray(world_dp, dtype=np.float64)
    world_dv = np.asarray(world_dv, dtype=np.float64)
    if position.ndim != 2 or position.shape[1:] != (3,) or len(position) < 3:
        raise ValueError("need at least three positions")
    if (
        dt.shape != (len(position) - 1,)
        or world_dp.shape != (len(dt), 3)
        or world_dv.shape != world_dp.shape
    ):
        raise ValueError("position/interval/integral shapes differ")
    if not all(
        np.isfinite(a).all() for a in [position, dt, world_dp, world_dv]
    ) or np.any(dt <= 0):
        raise ValueError("decomposition needs finite arrays and positive durations")
    delta = np.diff(position - position[0], axis=0)
    average = (delta - world_dp) / dt[:, None]
    y = np.diff(average, axis=0) - world_dv[:-1]
    alpha = 0.5 * (dt[:-1] + dt[1:])
    weight = np.sum(alpha**2)
    gravity = np.sum(alpha[:, None] * y, axis=0) / weight
    residual = y - alpha[:, None] * gravity
    pose_term = np.diff(delta / dt[:, None], axis=0)
    imu_term = -np.diff(world_dp / dt[:, None], axis=0) - world_dv[:-1]

    def projected(term):
        return term - alpha[:, None] * (np.sum(alpha[:, None] * term, axis=0) / weight)

    pose_residual, imu_residual = projected(pose_term), projected(imu_term)
    np.testing.assert_allclose(
        residual, pose_residual + imu_residual, rtol=1e-11, atol=2e-12
    )
    energy = float(np.sum(residual**2))
    pose_energy, imu_energy = (
        float(np.sum(pose_residual**2)),
        float(np.sum(imu_residual**2)),
    )
    cross_energy = float(2 * np.sum(pose_residual * imu_residual))
    np.testing.assert_allclose(
        energy, pose_energy + imu_energy + cross_energy, rtol=1e-10, atol=2e-12
    )
    return dict(
        gravity=gravity,
        average=average,
        alpha=alpha,
        y=y,
        residual=residual,
        pose_residual=pose_residual,
        imu_residual=imu_residual,
        energy=dict(total=energy, pose=pose_energy, imu=imu_energy, cross=cross_energy),
        max_sum_error=float(np.max(np.abs(residual - pose_residual - imu_residual))),
    )


def distribution(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    energy = norms**2
    order = np.argsort(-energy, kind="stable")
    total = float(np.sum(energy))
    return dict(
        rms=float(np.sqrt(np.mean(energy))),
        norm_quantiles={
            name: float(np.quantile(norms, q))
            for name, q in [("p50", 0.5), ("p90", 0.9), ("p99", 0.99)]
        },
        maximum=float(norms[order[0]]),
        maximum_row=int(order[0]),
        top1_energy_share=float(energy[order[:1]].sum() / total) if total else 0.0,
        top5_energy_share=float(energy[order[:5]].sum() / total) if total else 0.0,
        top_rows=[int(i) for i in order[:5]],
    )


def analyse(times, gyro, accel, stamps, poses):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    segments = []
    for start, end in zip(stamps[:-1], stamps[1:]):
        if start < times[0] or end > times[-1]:
            raise ValueError("uncovered reference interval")
        bounds = np.r_[start, times[(times > start) & (times < end)], end]
        held = np.searchsorted(times, bounds[:-1], side="right") - 1
        segments.append(
            (np.diff(bounds).astype(np.float64) * 1e-9, gyro[held], accel[held])
        )

    def integrated(bias, positions=False):
        result = []
        for durations, angular, force in segments:
            r, v, p = np.eye(3), np.zeros(3), np.zeros(3)
            for dt, w, a in zip(durations, angular, force):
                if positions:
                    world_a = r @ (a - PARAMS["accel_bias"])
                    p += v * dt + 0.5 * world_a * dt**2
                    v += world_a * dt
                r = r @ Rotation.from_rotvec((w - bias) * dt).as_matrix()
            result.append((r, v, p))
        return result

    def residual(bias):
        return np.concatenate(
            [
                Rotation.from_matrix(
                    delta[0].T @ poses[i, :3, :3].T @ poses[i + 1, :3, :3]
                ).as_rotvec()
                for i, delta in enumerate(integrated(bias))
            ]
        )

    fit = least_squares(
        residual, PARAMS["gyro_bias"], xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=100
    )
    deltas = integrated(fit.x, positions=True)
    dt = (
        np.array(
            [int(b) - int(a) for a, b in zip(stamps[:-1], stamps[1:])], dtype=np.float64
        )
        * 1e-9
    )
    world_dv = np.array([poses[i, :3, :3] @ delta[1] for i, delta in enumerate(deltas)])
    world_dp = np.array([poses[i, :3, :3] @ delta[2] for i, delta in enumerate(deltas)])
    decomposition = decompose(poses[:, :3, 3], dt, world_dp, world_dv)
    gravity, velocity_residual = decomposition["gravity"], decomposition["residual"]
    velocity = decomposition["average"] - 0.5 * dt[:, None] * gravity
    velocity = np.vstack([velocity, velocity[-1] + gravity * dt[-1] + world_dv[-1]])
    rotation_residual = fit.fun.reshape(-1, 3)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    diagnostics = dict(
        gravity_error=abs(float(np.linalg.norm(gravity)) - PARAMS["gravity_magnitude"]),
        rotation_rms=float(np.sqrt(np.mean(np.sum(rotation_residual**2, axis=1)))),
        velocity_rms=float(np.sqrt(np.mean(np.sum(velocity_residual**2, axis=1)))),
        gyro_singular_ratio=float(singular[-1] / singular[0]),
        max_gyro_bias_change=float(np.max(np.abs(fit.x - PARAMS["gyro_bias"]))),
        optimizer_success=bool(fit.success),
    )
    failed = [
        key
        for key in ["gravity_error", "rotation_rms", "velocity_rms", "gyro_bias_change"]
        if diagnostics["max_gyro_bias_change" if key == "gyro_bias_change" else key]
        > PARAMS["max_" + key]
    ]
    if diagnostics["gyro_singular_ratio"] < PARAMS["min_singular_ratio"]:
        failed.append("gyro_rank")
    if not fit.success:
        failed.append("optimizer")
    return dict(
        accepted=not failed,
        failed_limits=failed,
        diagnostics=diagnostics,
        gravity=gravity.tolist(),
        bias=np.r_[fit.x, PARAMS["accel_bias"]].tolist(),
        velocities=velocity.tolist(),
        rotation_residuals=rotation_residual.tolist(),
        velocity_residuals=velocity_residual.tolist(),
        pose_projected_residuals=decomposition["pose_residual"].tolist(),
        imu_projected_residuals=decomposition["imu_residual"].tolist(),
        decomposition_energy=decomposition["energy"],
        decomposition_max_sum_error=decomposition["max_sum_error"],
        pose_term_rms=float(
            np.sqrt(np.mean(np.sum(decomposition["pose_residual"] ** 2, axis=1)))
        ),
        imu_term_rms=float(
            np.sqrt(np.mean(np.sum(decomposition["imu_residual"] ** 2, axis=1)))
        ),
        rotation_distribution=distribution(rotation_residual),
        velocity_distribution=distribution(velocity_residual),
    )


@lru_cache(maxsize=1)
def native_series(bag):
    import cloudanalyzer_core as core

    return core.BagReader(str(bag)).imu_series("/livox/imu", acceleration_scale=9.80665)


def evaluate(task):
    source, stride, offset, indices, stamps, poses, times, gyro, accel, bag = task
    result = analyse(times, gyro, accel, stamps, poses)
    # Each process owns one immutable topic snapshot across all its cases.
    series = native_series(bag)
    try:
        native = series.initialize_lidar(
            stamps, poses, imu_from_lidar=np.eye(4), **PARAMS
        )
    except ValueError as error:
        assert not result["accepted"], str(error)
        if "IMU initialization rejected:" in str(error):
            assert all(label in str(error) for label in result["failed_limits"])
        else:
            assert set(result["failed_limits"]) & {
                "gyro_rank",
                "gyro_bias_change",
                "optimizer",
            }, str(error)
        result["native"] = dict(accepted=False, error=str(error))
    else:
        assert result["accepted"]
        np.testing.assert_array_equal(native["timestamps_ns"], stamps)
        errors = {}
        for key in ["gravity", "bias", "velocities"]:
            np.testing.assert_allclose(native[key], result[key], rtol=0, atol=2e-5)
            errors[key] = float(np.max(np.abs(native[key] - result[key])))
        result["native"] = dict(accepted=True, max_absolute_fit_differences=errors)
    result.update(
        source=source,
        stride=stride,
        offset_ns=offset,
        reference_count=len(stamps),
        saved_reference_indices=indices.tolist(),
        imu_reference_times_ns=stamps.tolist(),
    )
    return result


def compact(result):
    return {
        key: result[key]
        for key in [
            "source",
            "stride",
            "offset_ns",
            "reference_count",
            "accepted",
            "failed_limits",
            "diagnostics",
            "native",
            "decomposition_energy",
            "decomposition_max_sum_error",
            "pose_term_rms",
            "imu_term_rms",
            "rotation_distribution",
            "velocity_distribution",
        ]
    }


def audit(bag, output, raw, rotation, workers):
    import cloudanalyzer_core._core as native
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    import audit_flatwall_raw_point_time as fields

    bag, output, raw, rotation = (p.resolve() for p in [bag, output, raw, rotation])
    if output.exists() or any(output.is_relative_to(p) for p in [raw, rotation]):
        raise ValueError("use a fresh output outside previous evidence")
    if (
        os.environ.get("RAYON_NUM_THREADS") != "4"
        or os.environ.get("OPENBLAS_NUM_THREADS") != "1"
    ):
        raise ValueError("use fresh RAYON_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=1")
    parents = dict(raw=raw, rotation=rotation)
    files = {
        name: {
            filename: parents[parent] / directory / filename
            for filename in [
                "startup-poses.npz",
                "independent.json",
                "receipt.json",
                "lidar-diagnostics.json",
            ]
        }
        for name, parent, directory in SOURCES
    }

    def inputs():
        return dict(
            recording=fields.artifact(bag),
            native=fields.artifact(Path(native.__file__)),
            controller=fields.artifact(Path(__file__)),
            helper=fields.artifact(Path(fields.__file__)),
            numpy_version=version("numpy"),
            scipy_version=version("scipy"),
            rosbags_version=version("rosbags"),
            independent_reader=fields.artifact(Path(inspect.getfile(AnyReader))),
            optimizer_implementation=fields.artifact(
                Path(inspect.getfile(inspect.unwrap(least_squares)))
            ),
            rotation_implementation=fields.artifact(
                Path(import_module(type(Rotation.identity()).__module__).__file__)
            ),
            standard_types=fields.artifact(
                Path(import_module("rosbags.typesys.stores.ros1_noetic").__file__)
            ),
            previous={
                name: {key: fields.artifact(path) for key, path in group.items()}
                for name, group in files.items()
            },
        )

    frozen = inputs()
    assert frozen["recording"]["sha256"] == fields.BAG_SHA256
    protocol = dict(
        schema="cloudanalyzer.flatwall_startup_residuals.v1",
        inputs=frozen,
        parameters=PARAMS,
        acceleration_scale=9.80665,
        imu_from_lidar=np.eye(4).tolist(),
        reference_strides=STRIDES,
        clock_offsets_ns=OFFSETS_NS,
        workers=workers,
        clock_sign="IMU timestamp = saved LiDAR reference timestamp + offset_ns",
        clock_scope="Nonzero offsets only on raw poses with stride 1; zero-offset gyro-compensated poses are not reused for a joint calibration.",
        reference_policy="Indices 0,stride,...,63; both endpoints and the same total span are preserved.",
        decomposition="A=diff(diff(p)/dt), B=-diff(R*delta_p/dt)-R*delta_v[:-1]; project each through P=I-alpha*alpha.T/(alpha.T*alpha). P*(A+B)=P*A+P*B, with signed cross energy.",
        tolerance=dict(baseline_values=1e-10, native_estimates=2e-5),
        scope="Exploratory equation/clock/reference sensitivity, no smoothing, deletion, reweighting, changed limits, state adoption, physical calibration/uncertainty/causal diagnosis, ground-truth pose parsing, timing or GPU result. Coarse-batch acceptance does not validate omitted references.",
    )
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output / "protocol.json", protocol)
    types = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic == "/livox/imu"]
        assert len(connections) == 1 and connections[0].msgtype == "sensor_msgs/msg/Imu"
        messages = [
            types.deserialize_ros1(payload, c.msgtype)
            for c, _, payload in reader.messages(connections=connections)
        ]
    times = np.array(
        [
            int(m.header.stamp.sec) * 10**9 + int(m.header.stamp.nanosec)
            for m in messages
        ],
        dtype=np.int64,
    )
    gyro = np.array(
        [
            [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]
            for m in messages
        ]
    )
    accel = (
        np.array(
            [
                [
                    m.linear_acceleration.x,
                    m.linear_acceleration.y,
                    m.linear_acceleration.z,
                ]
                for m in messages
            ]
        )
        * 9.80665
    )
    assert (
        np.all(np.diff(times) > 0)
        and np.diff(times).max() <= 50_000_000
        and len({m.header.frame_id for m in messages}) == 1
    )
    snapshots, tasks, common_references = {}, [], None
    for name, _, _ in SOURCES:
        with np.load(files[name]["startup-poses.npz"], allow_pickle=False) as arrays:
            stamps, poses = arrays["timestamps_ns"].copy(), arrays["lidar_poses"].copy()
        assert stamps.shape == (64,) and poses.shape == (64, 4, 4)
        if common_references is None:
            common_references = stamps.copy()
        else:
            np.testing.assert_array_equal(stamps, common_references)
        snapshots[name] = stamps, poses
        settings = [(stride, 0) for stride in STRIDES]
        if name.startswith("raw-"):
            settings += [(1, offset) for offset in OFFSETS_NS if offset]
        for stride, offset in settings:
            indices = reference_indices(len(stamps), stride)
            shifted = shifted_references(stamps[indices], offset)
            tasks.append(
                (
                    name,
                    stride,
                    offset,
                    indices,
                    shifted,
                    poses[indices],
                    times,
                    gyro,
                    accel,
                    bag,
                )
            )
    assert len(tasks) == 52
    first = snapshots[SOURCES[0][0]][0]
    mask = (times >= int(first[0]) + OFFSETS_NS[0]) & (
        times <= int(first[-1]) + OFFSETS_NS[-1]
    )
    dt_ns = np.diff(times[mask])
    sensor = dict(
        prefix_imu_samples=int(mask.sum()),
        frame_id=messages[0].header.frame_id,
        sample_interval_ns=dict(
            min=int(dt_ns.min()), median=float(np.median(dt_ns)), max=int(dt_ns.max())
        ),
        gyro_rad_s=dict(
            mean=gyro[mask].mean(axis=0).tolist(), std=gyro[mask].std(axis=0).tolist()
        ),
        scaled_acceleration_norm_m_s2=dict(
            mean=float(np.linalg.norm(accel[mask], axis=1).mean()),
            std=float(np.linalg.norm(accel[mask], axis=1).std()),
        ),
        scope="Descriptive measurements on the fixed expanded prefix, without stationary or sensor-noise calibration.",
    )
    fields.save(output / "sensor.json", sensor)
    results, baselines, artifacts = [], [], {}
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        evaluated = (
            pool.map(evaluate, tasks, chunksize=1) if pool else map(evaluate, tasks)
        )
        for result in evaluated:
            name, stride, offset = (
                result["source"],
                result["stride"],
                result["offset_ns"],
            )
            if stride == 1 and offset == 0:
                old = json.loads(files[name]["independent.json"].read_text())
                assert (
                    old["accepted"] == result["accepted"]
                    and old["failed_limits"] == result["failed_limits"]
                )
                for key in old["diagnostics"]:
                    np.testing.assert_allclose(
                        result["diagnostics"][key],
                        old["diagnostics"][key],
                        rtol=0,
                        atol=1e-10,
                    )
                for key in ["gravity", "bias", "velocities"]:
                    np.testing.assert_allclose(
                        result[key], old[key], rtol=0, atol=1e-10
                    )
                baselines.append(compact(result))
            target = output / name / f"stride-{stride}-offset-{offset}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            fields.save(target, result)
            artifacts[target.relative_to(output).as_posix()] = fields.artifact(target)
            results.append(compact(result))
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    fields.save(output / "baseline-summary.json", baselines)
    fields.save(output / "sensitivity.json", results)
    assert inputs() == frozen
    for name in ["sensor.json", "baseline-summary.json", "sensitivity.json"]:
        artifacts[name] = fields.artifact(output / name)
    receipt = dict(
        schema=protocol["schema"],
        protocol=fields.artifact(output / "protocol.json"),
        inputs=frozen,
        cases=len(results),
        baseline_values_match_previous=True,
        all_native_acceptance_matches=True,
        maximum_decomposition_sum_error=max(
            r["decomposition_max_sum_error"] for r in results
        ),
        no_state_adopted=True,
        no_reference_poses_used=True,
        outputs=artifacts,
    )
    fields.save(output / "receipt.json", receipt)
    return dict(
        cases=len(results),
        baselines=baselines,
        sensor=sensor,
        accepted_exploratory_cases=[
            {
                key: r[key]
                for key in ["source", "stride", "offset_ns", "reference_count"]
            }
            for r in results
            if r["accepted"]
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bag", "out", "raw", "rotation"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=[1, 4], default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.bag, args.out, args.raw, args.rotation, args.workers), indent=2
        )
    )


if __name__ == "__main__":
    main()
