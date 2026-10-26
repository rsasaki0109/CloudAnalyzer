"""Import an immutable external SLAM map/trajectory into the HD drafting workflow."""
from __future__ import annotations

import json
import hashlib
import io
from pathlib import Path
import platform
import shutil
from typing import Any

import numpy as np

from ca import mapping_job as jobs
from ca.core.slam_run import load_frame
from ca.posegraph_fix import read_trajectory

SCHEMA = "cloudanalyzer.mapping_slam_import.v1"
MIN_FREE_BYTES = 10 * 1024**3
MAX_MAP_BYTES = 512 * 1024**2
MAX_TRAJECTORY_BYTES = 16 * 1024**2


def _provenance(value: dict[str, Any]) -> dict[str, Any]:
    strings = {"backend", "version", "source", "license", "coordinate_frame",
               "length_unit", "up_axis", "pose_convention", "time_unit", "time_basis"}
    if not isinstance(value, dict) or set(value) != strings | {"configuration"}:
        raise ValueError("provenance needs backend, version, source, license, coordinate_frame, "
                         "length_unit, up_axis, pose_convention, time_unit, time_basis and configuration")
    if any(not isinstance(value[k], str) or not 1 <= len(value[k].strip()) <= 2048 for k in strings):
        raise ValueError("provenance declarations need bounded, nonempty strings")
    if (value["length_unit"], value["up_axis"], value["pose_convention"], value["time_unit"]) != (
        "metres", "z", "world_from_sensor", "seconds"
    ):
        raise ValueError("supply a z-up metre-frame map and world_from_sensor TUM poses in seconds")
    if not isinstance(value["configuration"], dict):
        raise ValueError("configuration must be an explicit JSON object")
    encoded = json.dumps(value, allow_nan=False)
    if len(encoded.encode("utf-8")) > 65536:
        raise ValueError("provenance exceeds 64 KiB")
    return dict(json.loads(encoded))


def _space(root: Path, output_budget: int) -> dict[str, Any]:
    existing = root.parent
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < MIN_FREE_BYTES + output_budget:
        raise ValueError("insufficient disk space: import must retain 10 GiB free plus its output budget")
    return {"free_bytes_before": free, "minimum_free_bytes": MIN_FREE_BYTES,
            "estimated_import_output_bytes": output_budget, "point_map_copied": False}


def inspect_tum_trajectory(path: str | Path) -> dict[str, Any]:
    """Validate a bounded TUM snapshot, regardless of the producer's extension."""
    trajectory = Path(path)
    if not trajectory.is_file() or not 0 < trajectory.stat().st_size <= MAX_TRAJECTORY_BYTES:
        raise ValueError("TUM trajectory is missing, empty or exceeds its size limit")
    with trajectory.open('rb') as stream:
        raw = stream.read(MAX_TRAJECTORY_BYTES + 1)
    if not raw or len(raw) > MAX_TRAJECTORY_BYTES:
        raise ValueError("TUM trajectory is empty or exceeds its size limit")
    rows = np.loadtxt(io.BytesIO(raw), ndmin=2)
    if rows.ndim != 2 or rows.shape[1] != 8 or len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError("TUM needs at least two finite timestamped poses with eight columns")
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(rows[:, 4:], axis=1)
        increasing = np.all(np.diff(rows[:, 0]) > 0)
        path_length = float(np.linalg.norm(np.diff(rows[:, 1:4], axis=0), axis=1).sum())
    if not increasing or not np.allclose(norms, 1., atol=1e-5, rtol=0):
        raise ValueError("TUM timestamps must increase strictly and quaternions must be unit length")
    if not np.isfinite(path_length):
        raise ValueError("TUM path length must be finite")
    return {"schema": "cloudanalyzer.tum_preflight.v1", "pose_rows": len(rows),
            "first_timestamp": float(rows[0, 0]), "last_timestamp": float(rows[-1, 0]),
            "path_length_m": path_length, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "accuracy_verified": False}


def inspect_slam_inputs(map_path: str, trajectory_path: str, provenance: dict[str, Any]) -> dict[str, Any]:
    """Read-only preflight shared by import and external SLAM producers.

    Validates representation, not common-frame truth or estimator accuracy.
    Does not create a job or require the native mapping backend.
    """
    declared = _provenance(provenance)
    cloud, trajectory = Path(map_path).resolve(), Path(trajectory_path).resolve()
    if cloud.suffix.lower() != ".ply" or trajectory.suffix.lower() != ".tum":
        raise ValueError("external mapping inputs must be a .ply map and a .tum trajectory")
    for path, limit in [(cloud, MAX_MAP_BYTES), (trajectory, MAX_TRAJECTORY_BYTES)]:
        if not path.is_file() or not 0 < path.stat().st_size <= limit:
            raise ValueError(f"input is missing, empty or exceeds its import size limit: {path}")
    trajectory_check = inspect_tum_trajectory(trajectory)
    # Use the existing scan reader; retain the original PLY, including attributes.
    points = load_frame(cloud)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("map must contain finite XYZ points")
    count = len(points)
    bounds = {"minimum": points.min(axis=0).tolist(), "maximum": points.max(axis=0).tolist()}
    del points
    return {"schema": "cloudanalyzer.slam_preflight.v1", "producer": declared,
            "map_points": count, "map_bounds": bounds,
            **{key: trajectory_check[key] for key in ("pose_rows", "first_timestamp", "last_timestamp", "path_length_m")}, "frame_alignment_verified": False,
            "accuracy_verified": False}


def import_mapping_job(
    map_path: str, trajectory_path: str, out_dir: str, provenance: dict[str, Any],
    max_attempts: int = 6, minimum_retained_fraction: float = .9,
) -> dict[str, Any]:
    """Reference an existing PLY and TUM without running SLAM or copying the map.

    Producer/frame/time/configuration declarations are retained, not independently
    certified. The map and poses must already share a metre frame; no alignment,
    scaling or clock correction is performed. PLY is limited to 512 MiB and TUM
    to 16 MiB for this initial adapter. Inputs remain immutable dependencies.
    Only small metadata, KITTI poses and a vertex-only g2o snapshot are written.
    The snapshot contains no external SLAM constraints or uncertainty estimates.
    Raw-recording/fusion retries are unavailable for this frozen-map workflow.
    """
    declared = _provenance(provenance)
    if type(max_attempts) is not int or not 2 <= max_attempts <= 8:
        raise ValueError("mapping imports need 2..8 shared HD attempts")
    if not np.isfinite(minimum_retained_fraction) or not 0 <= minimum_retained_fraction <= 1:
        raise ValueError("minimum_retained_fraction must be between 0 and 1")
    root = Path(out_dir).resolve()
    if root.exists():
        raise FileExistsError(f"import output directory already exists: {root}")
    cloud, trajectory = Path(map_path).resolve(), Path(trajectory_path).resolve()
    if cloud.suffix.lower() != ".ply" or trajectory.suffix.lower() != ".tum":
        raise ValueError("external mapping inputs must be a .ply map and a .tum trajectory")
    for path, limit in [(cloud, MAX_MAP_BYTES), (trajectory, MAX_TRAJECTORY_BYTES)]:
        if not path.is_file() or not 0 < path.stat().st_size <= limit:
            raise ValueError(f"input is missing, empty or exceeds its import size limit: {path}")
    storage = _space(root, 1024**2)
    inputs = {"map": jobs._artifact(cloud), "source_trajectory": jobs._artifact(trajectory)}
    adapter = declared["configuration"].get("cloudanalyzer_glim_export")
    if adapter is not None:
        if not isinstance(adapter, dict) or adapter.get("schema") != "cloudanalyzer.glim_export.v1":
            raise ValueError("invalid GLIM adapter provenance")
        originals = adapter.get("inputs")
        if not isinstance(originals, dict):
            raise ValueError("missing GLIM export identities")
        for original, current in [("map", "map"), ("trajectory", "source_trajectory")]:
            ref = originals.get(original)
            if not isinstance(ref, dict) or any(ref.get(key) != inputs[current][key] for key in ("sha256", "bytes")):
                raise ValueError("GLIM export changed after preparation")
    if adapter is not None and adapter.get('execution') is not None:
        execution = adapter['execution']
        if (not isinstance(execution, dict) or execution.get('schema') != 'cloudanalyzer.glim_execution_link.v1'
                or execution.get('trajectory_sha256') != inputs['source_trajectory']['sha256']
                or not isinstance(execution.get('report'), dict)):
            raise ValueError('invalid GLIM execution link')
        report = execution['report']
        if set(report) != {'path', 'sha256', 'bytes'} or not isinstance(report['path'], str):
            raise ValueError('invalid GLIM execution report identity')
        jobs._verify(report)
        inputs['glim_execution_report'] = report
    preflight = inspect_slam_inputs(str(cloud), str(trajectory), declared)
    rows = np.loadtxt(trajectory, ndmin=2)
    poses, _ = read_trajectory(trajectory)
    norms = np.linalg.norm(rows[:, 4:], axis=1)
    path_length, count = preflight["path_length_m"], preflight["map_points"]
    for artifact in inputs.values():
        jobs._verify(artifact)
    native = jobs._native()
    storage = _space(root, 1024**2 + len(rows) * 768)
    root.mkdir(parents=True, exist_ok=False)
    with jobs._locked(root):
        source = {"schema": SCHEMA, "inputs": inputs, "producer_declaration": declared,
                  "preflight": preflight, "alignment_applied": False, "coordinates_rescaled": False,
                  "timestamps_rebased": False, "graph_role": "pose_snapshot_without_constraints"}
        jobs._save(root / "slam-import.json", source)
        job: dict[str, Any] = {
            "schema": jobs.SCHEMA, "job_dir": str(root), "status": "pointcloud_running",
            "source": jobs._artifact(root / "slam-import.json"),
            "runtime": {"python": platform.python_version(), "native": native},
            "pointcloud_options": {"origin": "external_slam"},
            "max_attempts": max_attempts, "minimum_retained_fraction": minimum_retained_fraction,
            "pointcloud": None, "attempts": [], "selected": None, "storage": storage,
            "scope": "Imported frozen point map and HD road drafts. Producer/frame/time declarations "
                     "and source support do not establish independent accuracy, road semantics or georeferencing.",
        }
        jobs._save(root / "job.json", job)
        try:
            np.savetxt(root / "trajectory.txt", poses[:, :3].reshape(-1, 12), fmt="%.17g")
            quaternions = rows[:, 4:] / norms[:, None]
            with (root / "poses.g2o").open("x", encoding="utf-8") as stream:
                for index, (row, quaternion) in enumerate(zip(rows, quaternions)):
                    values = [*row[1:4], *quaternion]
                    stream.write(f"VERTEX_SE3:QUAT {index} " + " ".join(f"{v:.17g}" for v in values) + "\n")
            for artifact in inputs.values():
                jobs._verify(artifact)
            job["pointcloud"] = {
                "files": {**inputs, "slam_import": job["source"], "trajectory": jobs._artifact(root / "trajectory.txt"),
                          "graph": jobs._artifact(root / "poses.g2o")},
                "origin": "external_slam", "frames": len(rows), "map_points": count,
                "frame_count_meaning": "imported_pose_rows", "input_recording_coverage": "not_verified",
                "path_length_m": path_length,
                "coordinate_frame": declared["coordinate_frame"],
                "quality_status": "imported_unverified", "reports": {"import": str(root / "slam-import.json")},
                "graph_role": "pose_snapshot_without_constraints",
            }
            job["storage"]["free_bytes_after"] = shutil.disk_usage(root).free
            job["status"] = "pointcloud_ready"
        except BaseException as error:
            job.update(status="pointcloud_failed", error=str(error), error_type=type(error).__name__)
            jobs._save(root / "job.json", job)
            if not isinstance(error, Exception):
                raise
        jobs._save(root / "job.json", job)
    return jobs.inspect_mapping_job(str(root))
