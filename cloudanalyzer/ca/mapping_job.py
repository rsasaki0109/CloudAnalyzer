"""Persistent mapping runs controlled by an agent through small, explicit tools.

The caller chooses hypotheses and reads evidence between attempts. This module
executes and records those choices; it does not contain a language model or infer
traffic rules from a source-support score.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ca._rust import core
from ca.posegraph_fix import fix_session, odometry
from ca.vector_map import build_vector_map

SCHEMA = "cloudanalyzer.mapping_job.v1"


def _artifact(path: str | Path) -> dict[str, Any]:
    file = Path(path).resolve()
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(file), "sha256": digest.hexdigest(), "bytes": file.stat().st_size}


def _verify(artifact: dict[str, Any]) -> None:
    if _artifact(artifact["path"]) != artifact:
        raise ValueError(f"recorded input or artifact changed: {artifact['path']}")


def _save(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    lock = root / ".mapping-lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError("mapping job is busy; inspect job.json before retrying") from error
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def _native() -> dict[str, Any]:
    module = core()
    if module is None or not hasattr(module, "audit_vector_map_quality"):
        raise RuntimeError('mapping jobs need an updated Rust core: pip install "cloudanalyzer[fast]"')
    extension = getattr(module, "_core", module)
    file = extension.__file__
    if not file:
        raise RuntimeError("native core has no binary path for provenance")
    return {"version": getattr(module, "__version__", "unknown"),
            "extension": _artifact(file)}


def _load(root: Path) -> dict[str, Any]:
    job: dict[str, Any] = json.loads((root / "job.json").read_text(encoding="utf-8"))
    if job.get("schema") != SCHEMA:
        raise ValueError("unsupported mapping job schema")
    return job


def _native_panic(error: BaseException) -> bool:
    # PyO3 panics inherit BaseException, unlike ordinary native ValueErrors.
    return type(error).__module__ == "pyo3_runtime" and type(error).__name__ == "PanicException"


def _inputs(job: dict[str, Any]) -> None:
    if job["pointcloud"] is None:
        raise ValueError("generate the point-cloud map before an HD candidate")
    _verify(job["source"])
    if _native() != job["runtime"]["native"]:
        raise ValueError("native core changed; use a new mapping job")
    for artifact in job["pointcloud"]["files"].values():
        _verify(artifact)


def inspect_mapping_job(job_dir: str) -> dict[str, Any]:
    """Read a mapping job's artifact paths, attempt evidence and remaining budget.

    No point clouds are loaded. Running/failed stages remain visible. Read each
    report before choosing another candidate or selecting a generated draft.
    """
    job = _load(Path(job_dir).resolve())
    for attempt in job["attempts"]:
        if attempt["status"] == "audited_draft":
            attempt["extent"] = _extent(attempt["extraction"], job.get("minimum_retained_fraction", 0.9))
    remaining = job["max_attempts"] - len(job["attempts"])
    job["remaining_attempts"] = remaining
    ready = job.get("pointcloud") is not None and job["status"] not in {"pointcloud_failed", "pointcloud_running"}
    job["next_actions"] = (["generate_mapping_candidate"] if ready and remaining else []) + (
        ["select_mapping_candidate"] if any(a["status"] == "audited_draft" and a["extent"]["passes_requested_extent"] for a in job["attempts"]) else []
    )
    return job


def start_mapping_job(
    source: str,
    out_dir: str,
    keyframe_spacing: float = 1.0,
    remove_dynamic: bool = True,
    max_attempts: int = 4,
    pointcloud_topic: str | None = None,
    imu_topic: str | None = None,
    minimum_retained_fraction: float = 0.9,
) -> dict[str, Any]:
    """Generate a point-cloud map and corrected trajectory from a raw recording.

    Accepts MCAP, ROS1 bag or rosbag2 SQLite recordings. Uses LiDAR odometry,
    loop closure, the recording's IMU gravity and optional dynamic removal.
    A NEW output directory holds the job and stage reports, including failures.
    Call generate_mapping_candidate next with explicit road assumptions and a reason.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 8:
        raise ValueError("max_attempts must be an integer between 1 and 8")
    if not 0 <= keyframe_spacing < float("inf"):
        raise ValueError("keyframe_spacing must be finite and nonnegative")
    if not 0 <= minimum_retained_fraction <= 1:
        raise ValueError("minimum_retained_fraction must be between 0 and 1")
    source_path = Path(source).resolve()
    if source_path.suffix.lower() not in {".mcap", ".bag", ".db3"} or not source_path.is_file():
        raise ValueError("source must be an existing MCAP, ROS1 bag or rosbag2 SQLite file")
    native = _native()
    root = Path(out_dir).resolve()
    job: dict[str, Any] = {
        "schema": SCHEMA, "job_dir": str(root), "status": "pointcloud_running",
        "source": _artifact(source_path), "runtime": {"python": platform.python_version(), "native": native},
        "pointcloud_options": {"keyframe_spacing": keyframe_spacing, "remove_dynamic": remove_dynamic,
                               "pointcloud_topic": pointcloud_topic, "imu_topic": imu_topic},
        "max_attempts": max_attempts, "minimum_retained_fraction": minimum_retained_fraction,
        "pointcloud": None, "attempts": [], "selected": None,
        "scope": "Generated point-cloud map and HD road drafts; source checks are not independent survey truth. Lane identity, traffic rules, equipment and georeferencing are unresolved unless supplied separately.",
    }
    root.mkdir(parents=True, exist_ok=False)
    with _locked(root):
        _save(root / "job.json", job)
        try:
            initial = odometry(str(source_path), str(root / "odometry"),
                               pointcloud_topic=pointcloud_topic, imu_topic=imu_topic)
            _save(root / "odometry-report.json", initial)
            corrected = fix_session(initial["scans"], str(root / "pointcloud"),
                                    poses=initial["trajectory"], gravity=initial["gravity"],
                                    keyframe_spacing=keyframe_spacing, remove_dynamic=remove_dynamic)
            _save(root / "pointcloud-report.json", corrected)
            _verify(job["source"])
            if corrected["map_points"] <= 0:
                raise ValueError("corrected point-cloud map is empty")
            files = corrected["outputs"]
            job["pointcloud"] = {
                "files": {"map": _artifact(files["map"]), "trajectory": _artifact(files["kitti"]),
                          "graph": _artifact(files["g2o"])},
                "frames": initial["frames"], "map_points": corrected["map_points"],
                "path_length_m": initial["path_length_m"],
                "reports": {"odometry": str(root / "odometry-report.json"),
                            "correction": str(root / "pointcloud-report.json")},
                "quality_status": "generated_unverified", "coordinate_frame": "local_slam_metres",
            }
            job["status"] = "pointcloud_ready"
        except BaseException as error:
            job["status"] = "pointcloud_failed"
            job["error"] = str(error)
            job["error_type"] = type(error).__name__
            if not isinstance(error, Exception) and not _native_panic(error):
                _save(root / "job.json", job)
                raise
        _save(root / "job.json", job)
    return inspect_mapping_job(str(root))


def _quality_summary(audit: dict[str, Any]) -> dict[str, Any]:
    q = audit["quality"]
    return {"lanes_checked": len(q["lanes"]), "needs_review": q["low_support_lanes"],
            "omitted": q["omitted_lanes"], "malformed": q["malformed_lanes"],
            "limited": q["limited"], "sampled_points": q["sampled_points"],
            "supported_samples": sum(l[c]["supported"] for l in q["lanes"] for c in ("center", "left", "right")),
            "validation_errors": [i for i in audit["validation"]["issues"] if i["severity"] == "error"]}


def _extent(extraction: dict[str, Any], minimum: float) -> dict[str, Any]:
    total, generated = extraction["trajectory_length"], extraction["generated_length"]
    if not math.isfinite(total) or not math.isfinite(generated) or total <= 0 or generated < 0:
        raise ValueError("generation report needs finite, nonnegative extent and positive trajectory length")
    fraction = generated / total
    return {"trajectory_length_m": total, "generated_length_m": generated,
            "retained_fraction": fraction, "minimum_retained_fraction": minimum,
            "passes_requested_extent": fraction >= minimum}


def generate_mapping_candidate(job_dir: str, road_options: dict[str, Any], reason: str) -> dict[str, Any]:
    """Generate and audit one HD road hypothesis against the job's frozen point cloud.

    Supply a decision reason and explicit forward_lanes, backward_lanes,
    left_hand_traffic, lane_width and speed_limit. Other build_vector_map fitting
    options are accepted; changing input maps or coordinate metadata is excluded.
    Each bounded attempt keeps its settings, native failures, output hashes and
    source/OSM audits. Failed trials do not replace earlier drafts or their selection.
    """
    required = {"forward_lanes", "backward_lanes", "left_hand_traffic", "lane_width", "speed_limit"}
    excluded = {"cloud", "trajectory", "out_dir", "reference_map", "existing_map", "projection", "origin_lat", "origin_lon"}
    allowed = set(inspect.signature(build_vector_map).parameters) - excluded
    if not reason.strip() or not required <= road_options.keys() or road_options.keys() - allowed:
        raise ValueError("supply a reason, explicit lane/traffic/width/speed assumptions, and supported fitting options")
    json.dumps(road_options, allow_nan=False)
    root = Path(job_dir).resolve()
    with _locked(root):
        job = _load(root)
        if job["pointcloud"] is None:
            raise ValueError("generate the point-cloud map before an HD candidate")
        if len(job["attempts"]) >= job["max_attempts"]:
            raise ValueError("mapping attempt budget exhausted; inspect the retained candidates")
        _inputs(job)
        job.setdefault("minimum_retained_fraction", 0.9)
        attempt: dict[str, Any] = {"id": len(job["attempts"]) + 1, "status": "running",
                                   "reason": reason.strip(), "road_options": road_options}
        job["attempts"].append(attempt)
        _save(root / "job.json", job)
        target = root / f"candidate-{attempt['id']:02d}"
        try:
            files = job["pointcloud"]["files"]
            report = build_vector_map(files["map"]["path"], files["trajectory"]["path"], str(target), **road_options)
            attempt["files"] = {key: _artifact(path) for key, path in report["files"].items()}
            module = core()
            assert module is not None
            audit = json.loads(module.audit_vector_map_quality(files["map"]["path"], report["files"]["editable_map"]))
            reopened = json.loads(module.audit_vector_map_quality(files["map"]["path"], report["files"]["map"]))
            _save(root / f"candidate-{attempt['id']:02d}-quality.json", {"editable": audit, "reopened_osm": reopened})
            _inputs(job)
            attempt["quality_report"] = _artifact(root / f"candidate-{attempt['id']:02d}-quality.json")
            attempt["quality"] = _quality_summary(audit)
            attempt["reopened_quality"] = _quality_summary(reopened)
            attempt["extraction"] = report["extraction"]
            attempt["extent"] = _extent(report["extraction"], job["minimum_retained_fraction"])
            attempt["export_issues"] = report["autoware_issues"]
            attempt["status"] = "audited_draft"
            job["status"] = "candidates_ready" if job["selected"] is None else "selected_draft"
        except BaseException as error:
            attempt["status"] = "failed"
            attempt["error"] = str(error)
            attempt["error_type"] = type(error).__name__
            if not isinstance(error, Exception) and not _native_panic(error):
                _save(root / "job.json", job)
                raise
        _save(root / "job.json", job)
    return inspect_mapping_job(str(root))


def select_mapping_candidate(job_dir: str, candidate_id: int, reason: str) -> dict[str, Any]:
    """Select an audited HD draft explicitly and record the agent's justification.

    Requires the job's retained-extent goal, nonempty lanes, complete audits and zero structural/import/export errors,
    and unchanged source/artifacts. Low source support stays visible and does not
    become a pass. Selecting is a local draft decision, not deployment certification.
    Read all candidates' road assumptions and retained extent before choosing.
    """
    if not reason.strip():
        raise ValueError("a selection needs a reason")
    root = Path(job_dir).resolve()
    with _locked(root):
        job = _load(root)
        _inputs(job)
        attempt = next((a for a in job["attempts"] if a["id"] == candidate_id), None)
        if attempt is None or attempt["status"] != "audited_draft":
            raise ValueError("select an audited draft candidate")
        for artifact in [*attempt["files"].values(), attempt["quality_report"]]:
            _verify(artifact)
        extent = _extent(attempt["extraction"], job.get("minimum_retained_fraction", 0.9))
        if not extent["passes_requested_extent"]:
            raise ValueError("candidate does not meet the job's retained-extent goal; inspect deferred road length")
        for q in [attempt["quality"], attempt["reopened_quality"]]:
            if not q["lanes_checked"] or q["omitted"] or q["malformed"] or q["limited"] or q["validation_errors"]:
                raise ValueError("candidate needs complete source audits and nonempty, structurally valid roads")
        if any(i["severity"] == "error" for i in attempt["export_issues"]):
            raise ValueError("resolve export errors before selecting")
        job["selected"] = {"candidate_id": candidate_id, "reason": reason.strip(),
                           "source_quality_passed": not attempt["quality"]["needs_review"] and not attempt["reopened_quality"]["needs_review"],
                           "deployment_ready": False, "files": attempt["files"],
                           "assumptions": attempt["road_options"], "extent": extent}
        job["status"] = "selected_draft"
        _save(root / "job.json", job)
    return inspect_mapping_job(str(root))
