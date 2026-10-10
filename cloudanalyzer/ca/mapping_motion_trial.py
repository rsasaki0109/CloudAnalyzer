"""Explicit alternative motion processing in a new, retained mapping job."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ca import mapping_job as jobs, mapping_frames as frames
from ca.core.bag_ingest import imu_ups, materialize_pointcloud_bag
from ca.posegraph_fix import fix_session, read_trajectory
from ca.mapping_trajectory import MAX_BYTES, MAX_POSES

SCHEMA = "cloudanalyzer.mapping_motion_trial.v1"
MAX_RETURNS = 5_000_000


def _verify(inputs: dict[str, Any]) -> None:
    for artifact in inputs.values():
        jobs._verify(artifact)


def trial_mapping_motion(
    job_dir: str, out_dir: str, find_loops: bool, use_gravity: bool, reason: str, max_attempts: int = 4,
) -> dict[str, Any]:
    """Generate ONE alternative point map without adopting it or changing the baseline.

    Choose explicit loop-search/gravity booleans and a reason; use a NEW external
    output directory and a new 1..8 HD attempt budget. Decode the hashed original
    recording afresh. Keep original odometry, reference-graph node IDs, thinning
    and dynamic policy; reconstruct only the odometry edge chain, then apply the
    chosen correction policy. Expanded fusion frames are not inherited. No
    reference/ground truth is supplied to generation. Failed/partial output is
    retained and never silently repeated. Old HD geometry/audits cannot be reused
    with moved poses: the child has no HD candidates or selection. Evaluate the
    child with the same reference/protocol, inspect local regressions, and draft
    and audit new HD geometry explicitly before any adoption decision. This is
    an exploratory whole-motion trial, not a local motion patch or accuracy gate.
    """
    if type(find_loops) is not bool or type(use_gravity) is not bool:
        raise ValueError("find_loops and use_gravity must be explicit booleans")
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 4096:
        raise ValueError("supply a bounded nonempty trial reason")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 8:
        raise ValueError("max_attempts must be an explicit new 1..8 HD attempt budget")
    baseline, root = Path(job_dir).resolve(), Path(out_dir).resolve()
    if root.exists():
        raise FileExistsError(f"motion trial directory already exists; retain and inspect it: {root}")
    if root.is_relative_to(baseline) or any((p / "job.json").is_file() for p in (root.parent, *root.parent.parents)):
        raise ValueError("save the motion trial outside existing mapping jobs")
    snapshot = jobs._artifact(baseline / "job.json")
    parent = jobs._load(baseline)
    jobs._inputs(parent)
    motion = parent["pointcloud"].get("source_motion")
    if not motion:
        raise ValueError("motion trial requires hashed original source_motion")
    inputs = {"baseline_job": snapshot, "source": parent["source"],
              **{f"baseline_{k}": v for k, v in parent["pointcloud"]["files"].items()},
              **{f"source_motion_{k}": v for k, v in motion.items()},
              "native": parent["runtime"]["native"]["extension"]}
    _verify(inputs)
    for key in ("source_motion_trajectory", "baseline_graph", "baseline_trajectory"):
        if inputs[key]["bytes"] > MAX_BYTES:
            raise ValueError("motion/graph input exceeds the 16 MiB limit")
    poses, stamps = read_trajectory(Path(motion["trajectory"]["path"]))
    if (stamps is None or not 3 <= len(poses) <= MAX_POSES or not np.isfinite(poses).all()
            or not np.isfinite(stamps).all() or not np.all(np.diff(stamps) > 0)):
        raise ValueError("original motion needs 3..4096 finite timestamped poses")
    options = parent["pointcloud_options"]
    for key, default in (("scan_voxel_m", .4), ("map_voxel_m", .2)):
        value = options.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("baseline needs positive finite thinning settings")
    if type(options["remove_dynamic"]) is not bool:
        raise ValueError("baseline needs an explicit dynamic-removal policy")
    policy = {"find_loops": find_loops, "use_gravity": use_gravity}
    trial: dict[str, Any] = {"schema": SCHEMA, "status": "running", "inputs": inputs,
        "policy": policy, "reason": reason.strip(), "outputs": {},
        "protocol": {"initial_motion": "hashed_original_odometry", "node_policy": "exact_baseline_reference_graph_ids",
                     "expanded_fusion_frames_inherited": False, "reference_used_for_generation": False,
                     "odometry_sigma_t_m": .05, "odometry_sigma_r_deg": .25,
                     "gravity_sigma_floor_deg": .1, "gravity_mount": "identity_then_existing_drive_calibration_policy",
                     "maximum_raw_frames": MAX_POSES, "maximum_raw_returns": MAX_RETURNS,
                     "motion_trial_executions": 1, "old_hd_geometry_or_audits_inherited": False,
                     "baseline_adopted": False},
        "scope": "Exploratory point-map/motion candidate. Evaluate on the same reference/protocol; uncertainty is uncalibrated. No existing HD map is validly transferred to moved poses, no local-motion retention or adoption is established."}
    child: dict[str, Any] = {"schema": jobs.SCHEMA, "job_dir": str(root), "status": "pointcloud_running",
        "source": parent["source"], "runtime": parent["runtime"], "pointcloud_options": dict(options),
        "max_attempts": max_attempts, "minimum_retained_fraction": parent.get("minimum_retained_fraction", .9),
        "pointcloud": None, "attempts": [], "selected": None, "scope": trial["scope"],
        "retry_inputs": inputs, "motion_trial": {"baseline_job_dir": str(baseline), "policy": policy, "reason": reason.strip()}}
    root.mkdir(parents=True, exist_ok=False)
    with jobs._locked(root):
        jobs._save(root / "trial.json", trial)
        jobs._save(root / "job.json", child)
        try:
            paths, fresh_stamps = materialize_pointcloud_bag(parent["source"]["path"], root / "scans",
                topic=options.get("pointcloud_topic"), kitti_bin=True, max_frames=MAX_POSES + 1)
            _verify(inputs)
            if len(paths) != len(poses) or len(paths) > MAX_POSES:
                raise ValueError("fresh recording does not cover the exact original motion frames")
            sizes = [p.stat().st_size for p in paths]
            if any(n == 0 or n % 16 for n in sizes) or sum(sizes) // 16 > MAX_RETURNS:
                raise ValueError("fresh scans need at most five million valid KITTI records")
            for path in paths:
                if not np.isfinite(np.fromfile(path, dtype=np.float32)).all():
                    raise ValueError("fresh scans contain nonfinite records")
            manifest = {"source": parent["source"], "pointcloud_topic": options.get("pointcloud_topic"),
                        "frames": [jobs._artifact(p) for p in paths], "timestamps_s": list(fresh_stamps)}
            poses, _, ids, baseline_poses = frames._motion(parent, manifest)
            if len(ids) < 3:
                raise ValueError("motion trial needs at least three retained graph nodes")
            jobs._save(root / "scans.json", manifest)
            generated = {"scans_manifest": jobs._artifact(root / "scans.json"),
                         **{f"scan_{i}": artifact for i, artifact in enumerate(manifest["frames"])}}
            # Rebuild original odometry edges only; do not retain previous loop edges.
            module = jobs.core()
            assert module is not None
            graph = module.PoseGraph.from_poses(np.ascontiguousarray(poses[ids]), .05, .25, ids)
            initial = root / "original-keyframes.g2o"
            initial.write_text(graph.to_g2o())
            generated["initial_graph"] = jobs._artifact(initial)
            gravity = None
            if use_gravity:
                ups = imu_ups(parent["source"]["path"], fresh_stamps, topic=options.get("imu_topic"))
                if not set(ids) <= set(ups):
                    raise ValueError("requested IMU gravity does not cover every retained node")
                for vector in ups.values():
                    if np.asarray(vector).shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                        raise ValueError("requested IMU gravity contains invalid directions")
                gravity = root / "gravity.txt"
                gravity.write_text("".join(f"{i} {' '.join(f'{v:.9f}' for v in up)}\n" for i, up in sorted(ups.items())))
                generated["gravity"] = jobs._artifact(gravity)
            trial["generated_inputs"] = generated
            trial["retained_original_frame_ids"] = ids
            trial["excluded_original_frame_ids"] = [i for i in range(len(poses)) if i not in set(ids)]
            jobs._save(root / "trial.json", trial)
            _verify(inputs)
            _verify(generated)
            correction = fix_session(str(root / "scans"), str(root / "pointcloud"), poses=str(initial),
                voxel=options.get("scan_voxel_m", .4), map_voxel=options.get("map_voxel_m", .2),
                loops=find_loops, gravity=None if gravity is None else str(gravity),
                remove_dynamic=options["remove_dynamic"])
            jobs._save(root / "pointcloud-report.json", correction)
            outputs = {"map": jobs._artifact(correction["outputs"]["map"]),
                       "graph": jobs._artifact(correction["outputs"]["g2o"]),
                       "trajectory": jobs._artifact(correction["outputs"]["kitti"])}
            corrected_graph = module.PoseGraph.from_g2o(Path(outputs["graph"]["path"]).read_text())
            corrected, _ = read_trajectory(Path(outputs["trajectory"]["path"]))
            if (list(corrected_graph.node_ids) != ids or corrected.shape != baseline_poses.shape
                    or not np.isfinite(corrected).all() or not np.allclose(corrected, corrected_graph.poses(), atol=1e-10, rtol=0)
                    or correction["nodes"] != len(ids) or correction["scans"] != len(ids) or correction["map_points"] <= 0):
                raise ValueError("motion candidate must preserve exact retained nodes and consistent nonempty outputs")
            _verify(inputs)
            _verify(generated)
            _verify(outputs)
            trial["outputs"] = outputs
            trial["correction_report"] = jobs._artifact(root / "pointcloud-report.json")
            shifts = np.linalg.norm(corrected[:, :3, 3] - baseline_poses[:, :3, 3], axis=1)
            trial["pose_change_from_baseline"] = {"mean_translation_m": float(shifts.mean()), "max_translation_m": float(shifts.max())}
            trial["status"] = "ready_unverified"
            child["pointcloud"] = {"files": outputs, "frames": len(paths), "map_points": correction["map_points"],
                "path_length_m": parent["pointcloud"]["path_length_m"], "source_motion": motion,
                "quality_status": "generated_unverified", "coordinate_frame": "local_slam_metres",
                "reports": {"correction": str(root / "pointcloud-report.json")}}
            child["status"] = "pointcloud_ready"
        except BaseException as error:
            trial.update(status="failed", error=str(error), error_type=type(error).__name__)
            child.update(status="pointcloud_failed", error=str(error), error_type=type(error).__name__)
            if not isinstance(error, Exception) and not jobs._native_panic(error):
                jobs._save(root / "trial.json", trial)
                jobs._save(root / "job.json", child)
                raise
        jobs._save(root / "trial.json", trial)
        child["motion_trial"]["report"] = jobs._artifact(root / "trial.json")
        jobs._save(root / "job.json", child)
    return {"schema": SCHEMA, "status": trial["status"], "job_dir": str(root), "policy": policy,
            "report": jobs._artifact(root / "trial.json"), "job": jobs._artifact(root / "job.json"),
            "pointcloud": child["pointcloud"], "error": trial.get("error"),
            "retained_original_frame_ids": trial.get("retained_original_frame_ids"),
            "scope": trial["scope"],
            "next_actions": ["evaluate_mapping_trajectory", "inspect_mapping_trajectory_comparison", "inspect_mapping_job"] if trial["status"] == "ready_unverified" else ["inspect_mapping_job"]}
