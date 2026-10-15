"""Inspected non-keyframe fusion with bounded, two-sided pose-consistency checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from ca import mapping_job as jobs, mapping_retry as retries
from ca.posegraph_fix import fix_session, match_scans, read_trajectory

PROTOCOL: dict[str, Any] = {"model": "interpolated_odometry_correction_with_two_sided_icp_check",
    "maximum_raw_frames": 4096, "maximum_unused_frames": 256, "maximum_selected_frames": 64,
    "maximum_bracket_seconds": 4., "maximum_bracket_travel_m": 3., "maximum_scan_points": 6000,
    "maximum_total_raw_returns": 5_000_000, "neighbor_distance_m": .75, "minimum_overlap_fraction": .65,
    "icp_overlap": .6, "icp_max_iterations": 30, "maximum_icp_rms_m": .5,
    "maximum_pose_translation_m": .25, "maximum_pose_rotation_deg": 1.5,
    "minimum_gap_returns": 3, "gap_radius_m": .75,
    "pose_update_applied": False, "independent_accuracy_established": False}


def _motion(job: dict[str, Any], manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    motion = job["pointcloud"].get("source_motion")
    if not motion:
        raise ValueError("unused-frame trials require a new job with hashed original odometry")
    retries._verify_inputs(motion)
    poses, timestamps = read_trajectory(Path(motion["trajectory"]["path"]))
    if timestamps is None or len(poses) != len(manifest["frames"]) or not np.isfinite(poses).all():
        raise ValueError("original motion must provide one finite timestamped pose per decoded frame")
    stamps = np.asarray(manifest["timestamps_s"])
    if (timestamps.shape != stamps.shape or not np.isfinite(stamps).all() or not np.all(np.diff(stamps) > 0)
        or not np.allclose(timestamps, stamps, atol=1e-6, rtol=0)):
        raise ValueError("original odometry and fresh recording timestamps do not match")
    module = jobs.core()
    assert module is not None
    graph = module.PoseGraph.from_g2o(Path(job["pointcloud"]["files"]["graph"]["path"]).read_text())
    ids = list(graph.node_ids)
    paths = [Path(a["path"]) for a in manifest["frames"]]
    if len(ids) < 2 or any(type(i) is not int or i < 0 or i >= len(poses) for i in ids) or ids != sorted(set(ids)):
        raise ValueError("corrected graph must retain ordered original frame IDs")
    if match_scans(paths, list(range(len(poses)))) != list(range(len(poses))):
        raise ValueError("decoded frame names do not match original pose rows")
    corrected = np.asarray(graph.poses())
    reference = np.loadtxt(job["pointcloud"]["files"]["trajectory"]["path"]).reshape(-1, 3, 4)
    if corrected.shape != (len(ids), 4, 4) or not np.allclose(corrected[:, :3], reference, atol=1e-10, rtol=0):
        raise ValueError("corrected graph changed the frozen trajectory")
    return poses, stamps, ids, corrected


def _pose(poses: np.ndarray, stamps: np.ndarray, ids: list[int], corrected: np.ndarray, frame: int) -> tuple[np.ndarray | None, dict[str, Any], list[str]]:
    index = int(np.searchsorted(ids, frame))
    if index == 0 or index == len(ids):
        return None, {}, ["no_two_sided_corrected_bracket"]
    lo, hi = ids[index - 1], ids[index]
    seconds = float(stamps[hi] - stamps[lo])
    travel = float(np.linalg.norm(corrected[index, :3, 3] - corrected[index - 1, :3, 3]))
    bracket = {"before_frame_id": lo, "after_frame_id": hi, "seconds": seconds, "travel_m": travel}
    holds = []
    if seconds > PROTOCOL["maximum_bracket_seconds"]:
        holds.append("long_time_bracket")
    if travel > PROTOCOL["maximum_bracket_travel_m"]:
        holds.append("long_motion_bracket")
    if holds:
        return None, bracket, holds
    corrections = corrected[index - 1:index + 1] @ np.linalg.inv(poses[[lo, hi]])
    fraction = (stamps[frame] - stamps[lo]) / seconds
    correction = np.eye(4)
    correction[:3, :3] = Slerp([stamps[lo], stamps[hi]], Rotation.from_matrix(corrections[:, :3, :3]))([stamps[frame]]).as_matrix()[0]
    correction[:3, 3] = (1 - fraction) * corrections[0, :3, 3] + fraction * corrections[1, :3, 3]
    return correction @ poses[frame], bracket, []


def _sample(points: np.ndarray) -> np.ndarray:
    stride = max(1, int(np.ceil(len(points) / PROTOCOL["maximum_scan_points"])))
    return np.ascontiguousarray(points[::stride], dtype=np.float64)


def _align(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return np.asarray(points @ pose[:3, :3].T + pose[:3, 3])


def _check(moving: np.ndarray, target: np.ndarray, origin: np.ndarray) -> dict[str, Any]:
    """Validate the inspected pose; ICP's alternative transform is never applied."""
    source, reference = _sample(moving - origin), _sample(target - origin)
    if len(source) < 30 or len(reference) < 30:
        return {"passes": False, "holds": ["too_few_registration_returns"]}
    module = jobs.core()
    assert module is not None
    tree = cKDTree(reference)
    before = tree.query(source)[0]
    try:
        result = module.icp(source, reference, max_iterations=PROTOCOL["icp_max_iterations"], overlap=PROTOCOL["icp_overlap"], match_centroids=False, point_to_plane=True)
    except (ValueError, RuntimeError) as error:
        return {"passes": False, "holds": ["registration_failed"], "error": str(error), "error_type": type(error).__name__}
    transform = np.asarray(result["transformation"])
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0., 0., 0., 1.], atol=1e-6, rtol=0)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or abs(np.linalg.det(transform[:3, :3]) - 1.) > 1e-6):
        return {"passes": False, "holds": ["invalid_registration_transform"]}
    angle = float(np.degrees(Rotation.from_matrix(transform[:3, :3]).magnitude()))
    translation = float(np.linalg.norm(transform[:3, 3]))
    after = tree.query(_align(source, transform))[0]
    overlap = [float(np.mean(d <= PROTOCOL["neighbor_distance_m"])) for d in (before, after)]
    metrics = {"converged": bool(result["converged"]), "rms_initial_m": float(result["rms_initial"]),
        "rms_final_m": float(result["rms_final"]), "pose_translation_at_sensor_m": translation,
        "pose_rotation_deg": angle, "initial_overlap_fraction": overlap[0], "registered_overlap_fraction": overlap[1],
        "moving_sample_points": len(source), "reference_sample_points": len(reference)}
    holds = []
    if not all(np.isfinite(v) for k, v in metrics.items() if k != "converged"):
        return {"passes": False, "holds": ["invalid_registration_metrics"]}
    if not metrics["converged"]:
        holds.append("registration_not_converged")
    if min(overlap) < PROTOCOL["minimum_overlap_fraction"]:
        holds.append("insufficient_two_sided_overlap")
    if metrics["rms_final_m"] > PROTOCOL["maximum_icp_rms_m"] or metrics["rms_final_m"] > metrics["rms_initial_m"] + 1e-6:
        holds.append("registration_residual")
    if translation > PROTOCOL["maximum_pose_translation_m"] or angle > PROTOCOL["maximum_pose_rotation_deg"]:
        holds.append("interpolated_pose_disagrees_with_scan_registration")
    return {**metrics, "holds": holds, "passes": not holds}


def inspect(root: Path, cid: int, gap_evidence: dict[str, Any], offset: int) -> dict[str, Any]:
    if type(offset) is not int or offset < 0:
        raise ValueError("unused-frame offset must be nonnegative")
    with jobs._locked(root):
        job = jobs._load(root); jobs._inputs(job)
        parent = retries._parent(job, cid)
        jobs._verify(gap_evidence)
        gaps = json.loads(Path(gap_evidence["path"]).read_text())
        if gaps["candidate_id"] != cid:
            raise ValueError("inspect unused frames for the same gap baseline")
        retries._verify_inputs(gaps["inputs"])
        jobs._verify(gaps["raw_manifest"])
        manifest = json.loads(Path(gaps["raw_manifest"]["path"]).read_text())
        inputs = {**gaps["inputs"], "gap_evidence": gap_evidence, "raw_manifest": gaps["raw_manifest"],
                  **{f"source_motion_{k}": v for k, v in job["pointcloud"].get("source_motion", {}).items()}}
        path = root / f"unused-frames-{cid:02d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["inputs"] != inputs or saved["protocol"] != PROTOCOL:
                raise ValueError("unused-frame inspection inputs or protocol changed")
        else:
            paths = retries._frames(manifest)
            poses, stamps, ids, corrected = _motion(job, manifest)
            unused = [i for i in range(len(paths)) if i not in ids]
            if len(unused) > PROTOCOL["maximum_unused_frames"]:
                raise ValueError("unused-frame inspection is limited to 256 excluded frames")
            module = jobs.core(); assert module is not None
            clouds = [np.asarray(module.read(str(p))["positions"]) for p in paths]
            if sum(map(len, clouds)) > PROTOCOL["maximum_total_raw_returns"] or any(not np.isfinite(p).all() for p in clouds):
                raise ValueError("unused-frame inspection needs at most five million finite raw returns")
            rows = []
            for frame in unused:
                pose, bracket, holds = _pose(poses, stamps, ids, corrected, frame)
                row: dict[str, Any] = {"frame_id": frame, "raw_frame": manifest["frames"][frame], "timestamp_s": float(stamps[frame]),
                    "raw_returns": len(clouds[frame]), "bracket": bracket, "holds": holds, "gap_observations": [], "registration": {}}
                if pose is not None:
                    points = _align(clouds[frame], pose); tree = cKDTree(points[:, :2])
                    observations = []
                    for gap in gaps["gaps"]:
                        counts = [{"station_m": p["station_m"], "returns": len(tree.query_ball_point(p["trajectory"][:2], PROTOCOL["gap_radius_m"]))}
                                  for p in gap["profiles"] if gap["from_m"] <= p["station_m"] <= gap["to_m"]]
                        supported = [p for p in counts if p["returns"] >= PROTOCOL["minimum_gap_returns"]]
                        if supported:
                            observations.append({"gap_id": gap["id"], "from_m": gap["from_m"], "to_m": gap["to_m"], "profiles": supported})
                    row.update(pose_hypothesis=pose.tolist(), gap_observations=observations)
                    if not observations:
                        holds.append("no_observed_missing_interval_returns")
                    else:
                        for name, idx in (("before", ids.index(bracket["before_frame_id"])), ("after", ids.index(bracket["after_frame_id"]))):
                            target = _align(clouds[ids[idx]], corrected[idx])
                            check = _check(points, target, pose[:3, 3]); row["registration"][name] = check
                            holds.extend(f"{name}:{h}" for h in check["holds"])
                row["eligible"] = not holds
                rows.append(row)
            retries._frames(manifest); retries._verify_inputs(inputs); jobs._inputs(job)
            saved = {"schema": "cloudanalyzer.unused_frames.v1", "candidate_id": cid, "inputs": inputs,
                "raw_manifest": gaps["raw_manifest"], "protocol": PROTOCOL, "frames": rows,
                "original_retained_frames": ids, "pointcloud_options": gaps["pointcloud_options"], "source_extent": parent["extent"],
                "note": "Two-sided registration checks geometric consistency, not independent pose accuracy or road identity. Interpolated correction is a pose hypothesis; ICP corrections are not applied. Gap counts are raw repeated returns, not proof of coherent ground or recovered road. Original retained poses and station denominator remain fixed. No frames are automatically adopted."}
            jobs._save(path, saved)
        retries._verify_inputs(saved["inputs"])
        page = saved["frames"][offset:offset + 8]
        bounded = [{**r, "gap_observations": r["gap_observations"][:8], "gap_observations_total": len(r["gap_observations"])} for r in page]
        return {"file": jobs._artifact(path), "candidate_id": cid, "frames_total": len(saved["frames"]),
                "eligible_total": sum(r["eligible"] for r in saved["frames"]), "frames": bounded,
                "next_offset": offset + 8 if offset + 8 < len(saved["frames"]) else None,
                "protocol": saved["protocol"], "source_extent": saved["source_extent"], "note": saved["note"]}


def validate(evidence: dict[str, Any], ids: Any) -> dict[str, Any]:
    jobs._verify(evidence); saved = json.loads(Path(evidence["path"]).read_text())
    retries._verify_inputs(saved["inputs"])
    jobs._verify(saved["raw_manifest"])
    retries._frames(json.loads(Path(saved["raw_manifest"]["path"]).read_text()))
    if saved["protocol"] != PROTOCOL:
        raise ValueError("unused-frame consistency protocol changed")
    eligible = {r["frame_id"] for r in saved["frames"] if r["eligible"]}
    if not isinstance(ids, list) or not 1 <= len(ids) <= PROTOCOL["maximum_selected_frames"] or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids) or not set(ids) <= eligible:
        raise ValueError("choose 1..64 distinct inspected eligible unused frame IDs")
    return cast(dict[str, Any], saved)


def rebuild(child: Path, job: dict[str, Any], saved: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    """Add only inspected pose hypotheses to the fusion graph, preserve reference files."""
    import shutil
    manifest = json.loads(Path(saved["raw_manifest"]["path"]).read_text())
    paths = retries._frames(manifest)
    _, _, original_ids, corrected = _motion(job, manifest)
    by_id = dict(zip(original_ids, corrected))
    rows = {r["frame_id"]: r for r in saved["frames"]}
    by_id.update({i: np.asarray(rows[i]["pose_hypothesis"]) for i in ids})
    fused_ids = sorted(by_id); poses = np.array([by_id[i] for i in fused_ids])
    module = jobs.core(); assert module is not None
    input_graph = child / "fusion-input.g2o"
    input_graph.write_text(module.PoseGraph.from_poses(poses, ids=fused_ids).to_g2o())
    input_artifact = jobs._artifact(input_graph)
    # A sparse original keyframe set must not trigger filename/order heuristics.
    # Keep original frame names and give fusion exactly the explicitly chosen set.
    directory = child / "fusion-scans"; directory.mkdir()
    copied = []
    for i in fused_ids:
        target = directory / paths[i].name; shutil.copyfile(paths[i], target)
        artifact = jobs._artifact(target)
        if (artifact["sha256"], artifact["bytes"]) != (manifest["frames"][i]["sha256"], manifest["frames"][i]["bytes"]):
            raise ValueError("fusion scan copy differs from the inspected original return bytes")
        copied.append(artifact)
    selected_manifest = {"frames": copied, "frame_ids": fused_ids, "source_manifest": saved["raw_manifest"]}
    jobs._save(child / "fusion-scans.json", selected_manifest)
    scans_artifact = jobs._artifact(child / "fusion-scans.json")
    result = fix_session(str(directory), str(child / "pointcloud"), poses=str(input_graph), loops=False, gravity=None,
        voxel=saved["pointcloud_options"]["scan_voxel_m"], map_voxel=saved["pointcloud_options"]["map_voxel_m"], remove_dynamic=job["pointcloud_options"]["remove_dynamic"])
    motion = np.loadtxt(result["outputs"]["kitti"]).reshape(-1, 3, 4)
    actual_graph = module.PoseGraph.from_g2o(Path(result["outputs"]["g2o"]).read_text())
    if (list(actual_graph.node_ids) != fused_ids or motion.shape != poses[:, :3].shape or not np.allclose(motion, poses[:, :3], atol=1e-10, rtol=0)
        or not np.allclose(actual_graph.poses(), poses, atol=1e-10, rtol=0)):
        raise ValueError("unused-frame fusion changed frozen or inspected poses")
    jobs._verify(input_artifact); jobs._verify(scans_artifact); retries._frames(selected_manifest)
    retries._frames(manifest); retries._verify_inputs(saved["inputs"])
    result["outputs"]["fusion_kitti"] = result["outputs"]["kitti"]
    result["outputs"]["fusion_g2o"] = result["outputs"]["g2o"]
    for key, output_key in (("trajectory", "kitti"), ("graph", "g2o")):
        target = child / ("reference.txt" if key == "trajectory" else "reference.g2o")
        shutil.copyfile(job["pointcloud"]["files"][key]["path"], target)
        result["outputs"][output_key] = str(target)
    result.update(added_frame_ids=ids, original_retained_frame_ids=original_ids, fusion_frame_ids=fused_ids,
        pose_hypothesis="interpolated original odometry correction; checked against both original bracket scans",
        pose_update_applied=False, motion_reoptimized=False, fusion_input=input_artifact,
        fusion_scans_manifest=scans_artifact, recording_raw_frames=len(paths),
        excluded_raw_frame_ids=[i for i in range(len(paths)) if i not in fused_ids],
        maximum_pose_roundtrip_difference=float(np.max(np.abs(motion - poses[:, :3]))))
    return cast(dict[str, Any], result)
