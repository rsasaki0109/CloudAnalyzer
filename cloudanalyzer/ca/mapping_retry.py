"""Evidence-driven density trials with frozen motion and one shared HD budget."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree

from ca import mapping_job as jobs
from ca.core.bag_ingest import materialize_pointcloud_bag
from ca.posegraph_fix import SCAN_SUFFIXES, fix_session
from ca.mapping_connections import route_metrics


def _parent(job: dict[str, Any], cid: int) -> dict[str, Any]:
    parent = next((a for a in job["attempts"] if a["id"] == cid), None)
    if parent is None or parent.get("kind") not in {"corridor_lanes", "connected_corridor_lanes", "patched_corridor_lanes"} or parent["status"] != "audited_draft":
        raise ValueError("inspect gaps on a retained audited corridor lane draft")
    for artifact in [*parent["files"].values(), *parent["geometry_inputs"].values(), parent["quality_report"], parent["corridor_proposal"],
                     *parent.get("connection_inputs", {}).values(), *parent.get("patch_inputs", {}).values(),
                     *([parent["patch_checks"], parent["patch_audits"]] if "patch_checks" in parent else [])]:
        jobs._verify(artifact)
    for key in ("connection_proposal", "connection_checks", "connection_audits"):
        if key in parent:
            jobs._verify(parent[key])
    if "patch_preview" in parent:
        jobs._verify(parent["patch_preview"])
    return cast(dict[str, Any], parent)


def _inputs(job: dict[str, Any], parent: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    correction = jobs._artifact(job["pointcloud"]["reports"]["correction"])
    return {"source": job["source"], "native": job["runtime"]["native"]["extension"], "layout": layout,
            "correction_report": correction, "proposal": parent["corridor_proposal"], "audits": parent["quality_report"],
            **{f"pointcloud_{k}": v for k, v in job["pointcloud"]["files"].items()},
            **{f"parent_{k}": v for k, v in parent["files"].items()},
            **{f"geometry_{k}": v for k, v in parent["geometry_inputs"].items()}}


def _verify_inputs(inputs: dict[str, Any]) -> None:
    for artifact in inputs.values():
        jobs._verify(artifact)


def _frames(manifest: dict[str, Any]) -> list[Path]:
    paths = [Path(a["path"]) for a in manifest["frames"]]
    if not paths or len(set(paths)) != len(paths) or any(p.parent != paths[0].parent for p in paths):
        raise ValueError("raw frame manifest must describe one complete scan directory")
    actual = {p for p in paths[0].parent.iterdir() if p.suffix.lower() in SCAN_SUFFIXES}
    if actual != set(paths):
        raise ValueError("raw frame directory changed since extraction")
    _verify_inputs({str(i): a for i, a in enumerate(manifest["frames"])})
    return paths


def _raw_source(root: Path, job: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Freshly decode the original recording, align raw returns at frozen keyframes."""
    directory = root / "gap-source"
    manifest_file = directory / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["source"] != job["source"] or manifest["topic"] != job["pointcloud_options"]["pointcloud_topic"]:
            raise ValueError("raw gap evidence belongs to another source")
        paths = _frames(manifest)
    else:
        if directory.exists():
            raise RuntimeError("raw evidence extraction was interrupted; retain it and inspect before recovery")
        paths, stamps = materialize_pointcloud_bag(job["source"]["path"], directory / "scans",
                              topic=job["pointcloud_options"]["pointcloud_topic"], kitti_bin=True, max_frames=4097)
        jobs._verify(job["source"])
        if not 1 <= len(paths) <= 4096:
            raise ValueError("raw gap inspection is limited to 4096 frames; use a smaller recording")
        manifest = {"source": job["source"], "topic": job["pointcloud_options"]["pointcloud_topic"],
                    "frames": [jobs._artifact(p) for p in paths], "timestamps_s": list(stamps)}
        jobs._save(manifest_file, manifest)
    module = jobs.core()
    assert module is not None
    graph = module.PoseGraph.from_g2o(Path(job["pointcloud"]["files"]["graph"]["path"]).read_text())
    poses = graph.poses()
    reference = np.loadtxt(job["pointcloud"]["files"]["trajectory"]["path"]).reshape(-1, 3, 4)
    if poses.shape[0] != reference.shape[0] or not np.allclose(poses[:, :3, :], reference, atol=1e-10, rtol=0):
        raise ValueError("saved graph does not reproduce the frozen trajectory")
    if any(p.name != f"frame_{i:06d}.bin" for i, p in enumerate(paths)):
        raise ValueError("decoded raw frames must retain original contiguous frame numbers")
    by_id = {node_id: i for i, node_id in enumerate(graph.node_ids)}
    matched = [by_id.get(i) for i in range(len(paths))]
    if set(n for n in matched if n is not None) != set(range(len(reference))):
        raise ValueError("raw scans do not cover every frozen keyframe")
    points, count = [], 0
    for path, node in zip(paths, matched):
        if node is None:
            continue
        xyz = np.asarray(module.read(str(path))["positions"])
        count += len(xyz)
        if count > 5_000_000:
            raise ValueError("raw gap inspection is limited to five million matched returns")
        if not np.isfinite(xyz).all():
            raise ValueError("raw scans contain invalid positions")
        points.append(xyz @ poses[node, :3, :3].T + poses[node, :3, 3])
    _frames(manifest)
    jobs._verify(job["source"])
    return jobs._artifact(manifest_file), np.concatenate(points)


def _neighborhood(tree: Any, xyz: np.ndarray, xy: Any) -> dict[str, Any]:
    indices = tree.query_ball_point(xy, .75)
    points = xyz[indices]
    return {"radius_m": .75, "returns": len(points),
            "occupied_xy_cells_0_2m": len(np.unique(np.floor(points[:, :2] / .2), axis=0)),
            "height_quantiles_m": dict(zip(("minimum", "p15", "median"), np.quantile(points[:, 2], [0., .15, .5]).tolist())) if len(points) else None}


def inspect_gaps(root: Path, cid: int, offset: int, layout: dict[str, Any]) -> dict[str, Any]:
    if type(offset) is not int or offset < 0:
        raise ValueError("gap offset must be a nonnegative integer")
    with jobs._locked(root):
        job = jobs._load(root)
        jobs._inputs(job)
        parent = _parent(job, cid)
        inputs = _inputs(job, parent, layout)
        path = root / f"gaps-{cid:02d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if inputs != saved["inputs"]:
                raise ValueError("gap evidence inputs changed")
        else:
            _verify_inputs(inputs)
            manifest, raw = _raw_source(root, job)
            module = jobs.core()
            assert module is not None
            cloud = np.asarray(module.read(job["pointcloud"]["files"]["map"]["path"])["positions"])
            raw_tree, map_tree = cKDTree(raw[:, :2]), cKDTree(cloud[:, :2])
            proposal = json.loads(Path(parent["corridor_proposal"]["path"]).read_text())
            report = json.loads(Path(parent["files"]["report"]["path"]).read_text())
            gaps: list[dict[str, Any]] = []
            for interval in report["station_disposition"]:
                if interval["status"] == "included_lane_hypothesis":
                    continue
                profiles = [p for p in proposal["profiles"] if interval["from_m"] - 2 <= p["station_m"] <= interval["to_m"] + 2]
                chosen = sorted(set(np.linspace(0, len(profiles) - 1, min(len(profiles), 8), dtype=int))) if profiles else []
                observations = [{**{k: p[k] for k in ("station_m", "trajectory", "heading_usable", "reference_ground_height_m")},
                                  "bands": p["bands"][:16], "bands_total": len(p["bands"]),
                                  "raw": _neighborhood(raw_tree, raw, p["trajectory"][:2]),
                                  "point_map": _neighborhood(map_tree, cloud, p["trajectory"][:2])} for p in (profiles[k] for k in chosen)]
                gaps.append({"id": len(gaps) + 1, **interval, "profiles": observations,
                             "profiles_total": len(profiles), "profiles_limited": len(profiles) > len(chosen)})
            correction = json.loads(Path(inputs["correction_report"]["path"]).read_text())
            saved = {"schema": "cloudanalyzer.mapping_gaps.v1", "candidate_id": cid, "inputs": inputs, "raw_manifest": manifest,
                     "gaps": gaps, "pointcloud_options": {"scan_voxel_m": job["pointcloud_options"].get("scan_voxel_m", .4),
                         "map_voxel_m": job["pointcloud_options"].get("map_voxel_m", .2)},
                     "processing": {k: correction.get(k) for k in ("nodes", "scans", "unmatched_scans", "scan_points", "map_points", "dynamic")},
                     "raw_matched_returns": len(raw), "protocol": proposal["protocol"], "source_extent": parent["extent"],
                     "note": "Raw neighborhoods contain repeated aligned returns before thinning/dynamic filtering. Return counts and height quantiles are descriptive, not coherent-ground tests or independent accuracy. Profiles include nearby context; gap reasons are observations, not proven causes. Unmatched non-keyframe scans remain excluded."}
            jobs._inputs(job)
            _verify_inputs(inputs)
            jobs._save(path, saved)
        _verify_inputs(saved["inputs"])
        jobs._verify(saved["raw_manifest"])
        return {"file": jobs._artifact(path), "candidate_id": cid, "gaps_total": len(saved["gaps"]),
                "gaps": saved["gaps"][offset:offset + 8], "next_offset": offset + 8 if offset + 8 < len(saved["gaps"]) else None,
                **{k: saved[k] for k in ("processing", "pointcloud_options", "raw_matched_returns", "protocol", "source_extent", "note")}}


def validate(evidence: dict[str, Any], gap_ids: Any, options: Any) -> dict[str, Any]:
    jobs._verify(evidence)
    saved = json.loads(Path(evidence["path"]).read_text())
    _verify_inputs(saved["inputs"])
    jobs._verify(saved["raw_manifest"])
    if not isinstance(gap_ids, list) or not 1 <= len(gap_ids) <= 8 or any(type(i) is not int for i in gap_ids) or len(set(gap_ids)) != len(gap_ids) or not set(gap_ids) <= {g["id"] for g in saved["gaps"]}:
        raise ValueError("choose 1..8 distinct inspected gap IDs")
    if not isinstance(options, dict) or set(options) != {"scan_voxel_m", "map_voxel_m"}:
        raise ValueError("density trial needs explicit scan_voxel_m and map_voxel_m")
    for key, lower in (("scan_voxel_m", .1), ("map_voxel_m", .05)):
        value = options[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= saved["pointcloud_options"][key]:
            raise ValueError("density trials can reduce thinning only within bounded resolution ranges")
    if options == saved["pointcloud_options"]:
        raise ValueError("density trial must change at least one thinning resolution")
    return cast(dict[str, Any], saved)


def retry(root: Path, cid: int, evidence: dict[str, Any], gap_ids: list[int], options: dict[str, Any], reason: str,
          frame_ids: list[int] | None = None, local_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    from ca.mapping_run import SCHEMA
    from ca import mapping_frames as frames
    from ca import mapping_local_points as local
    with jobs._locked(root):
        job = jobs._load(root)
        jobs._inputs(job)
        parent = _parent(job, cid)
        saved = validate(evidence, gap_ids, options) if frame_ids is None else frames.validate(evidence, frame_ids)
        strategy = "density" if frame_ids is None else "unused_frames"
        if local_evidence is not None:
            local_preview = (local.validate_density(local_evidence, evidence, cid, options) if frame_ids is None
                             else local.validate(local_evidence, evidence, cid, frame_ids))
            if gap_ids != local_preview['request']['gap_ids']:
                raise ValueError("local update gap choice changed")
            strategy = "local_density" if frame_ids is None else "local_unused_frames"
        if frame_ids is not None and options != saved["pointcloud_options"]:
            raise ValueError("unused-frame fusion must keep original thinning resolutions")
        if saved["candidate_id"] != cid:
            raise ValueError("gap evidence belongs to another candidate")
        stage = job.get("pointcloud_retry")
        if stage:
            if (stage["candidate_id"], stage["gap_ids"], stage["options"], stage["reason"], stage["evidence"]) != (cid, gap_ids, options, reason, evidence):
                raise ValueError("point-cloud retry was already allocated to another decision")
            if stage.get("strategy", "density") != strategy or stage.get("frame_ids") != frame_ids:
                raise ValueError("point-cloud retry strategy or chosen frames changed")
            if stage.get('local_evidence') != local_evidence:
                raise ValueError("point-cloud retry local region changed")
            if stage["status"] == "running":
                raise RuntimeError("point-cloud retry did not finish; inspect retained state before recovery")
            return cast(dict[str, Any], stage)
        available = jobs._remaining(job)
        if available < (3 if local_evidence is not None else 2) or "retry_inputs" in job:
            raise ValueError("one point-map retry requires enough shared HD attempts (three for local updates) and a root run")
        child = root / "pointcloud-retry"
        if child.exists():
            raise FileExistsError(f"retry directory already exists: {child}")
        stage = {"status": "running", "candidate_id": cid, "gap_ids": gap_ids, "options": options, "reason": reason,
                 "evidence": evidence, "child_job_dir": str(child), "allocated_attempts": available,
                 "strategy": strategy, "frame_ids": frame_ids}
        if local_evidence is not None:
            stage['local_evidence'] = local_evidence
        job["pointcloud_retry"] = stage
        jobs._save(root / "job.json", job)
        try:
            child.mkdir()
            shutil.copyfile(saved["inputs"]["layout"]["path"], child / "layout-hypothesis.json")
            child_job = {"schema": jobs.SCHEMA, "job_dir": str(child), "source": job["source"], "runtime": job["runtime"],
                "status": "pointcloud_running", "pointcloud": None, "attempts": [], "selected": None,
                "pointcloud_options": {**job["pointcloud_options"], **options}, "max_attempts": available,
                "minimum_retained_fraction": job["minimum_retained_fraction"], "scope": job["scope"],
                "retry_inputs": {**saved["inputs"], "retry_evidence": evidence, "raw_manifest": saved["raw_manifest"]},
                "retry_parent": {"job_dir": str(root), "candidate_id": cid, "gap_ids": gap_ids, "reason": reason}}
            if local_evidence is not None:
                child_job['retry_inputs']['local_preview'] = local_evidence
            jobs._save(child / "job.json", child_job)
            run = {"schema": SCHEMA, "revision": 0, "status": "preparing", "layout_file": jobs._artifact(child / "layout-hypothesis.json"),
                   "reviewed_candidates": [], "history": [], "output": None, "maximum_actions": 128, "pointcloud_retry_allowed": False}
            jobs._save(child / "run.json", run)
            manifest = json.loads(Path(saved["raw_manifest"]["path"]).read_text())
            raw_paths = _frames(manifest)
            result = frames.rebuild(child, job, saved, frame_ids) if frame_ids is not None else fix_session(str(raw_paths[0].parent), str(child / "pointcloud"),
                poses=job["pointcloud"]["files"]["graph"]["path"], voxel=options["scan_voxel_m"], map_voxel=options["map_voxel_m"],
                loops=False, gravity=None, remove_dynamic=job["pointcloud_options"]["remove_dynamic"])
            _frames(manifest)
            _verify_inputs(child_job["retry_inputs"])
            motion = np.loadtxt(result["outputs"]["kitti"])
            reference = np.loadtxt(job["pointcloud"]["files"]["trajectory"]["path"])
            if motion.shape != reference.shape or not np.allclose(motion, reference, atol=1e-10, rtol=0):
                raise ValueError("density trial changed frozen motion")
            if frame_ids is None and local_evidence is not None:
                module = jobs.core(); assert module is not None
                original_graph = module.PoseGraph.from_g2o(Path(job['pointcloud']['files']['graph']['path']).read_text())
                fused_graph = module.PoseGraph.from_g2o(Path(result['outputs']['g2o']).read_text())
                if (list(fused_graph.node_ids) != list(original_graph.node_ids)
                    or result['scans'] != len(original_graph.node_ids)
                    or not np.allclose(fused_graph.poses(), original_graph.poses(), atol=1e-10, rtol=0)):
                    raise ValueError("local density fusion changed retained frames or frozen poses")
                result['original_retained_frame_ids'] = list(original_graph.node_ids)
                result['added_frame_ids'] = []
            if result["map_points"] <= 0:
                raise ValueError("density trial produced an empty point map")
            # Keep the exact original station grid and graph, rather than a numerical roundtrip copy.
            for key, output_key in (("trajectory", "kitti"), ("graph", "g2o")):
                shutil.copyfile(job["pointcloud"]["files"][key]["path"], result["outputs"][output_key])
            result["motion_reoptimized"] = False
            result["maximum_pose_roundtrip_difference"] = max(result.get("maximum_pose_roundtrip_difference", 0.), float(np.max(np.abs(motion - reference))))
            if local_evidence is not None:
                update = local.apply(child, parent, local_evidence, result)
                stage['local_update'] = update
                if not update['passes']:
                    raise ValueError("local point update regressed retained HD source: " + ', '.join(update['holds']))
                result['outputs']['full_fusion_map'] = result['outputs']['map']
                result['outputs']['map'] = update['map']['path']
                result['full_fusion_map_points'] = result['map_points']
                result['map_points'] = update['map_points']
                result['local_update'] = update
            jobs._save(child / "pointcloud-report.json", result)
            child_job["pointcloud"] = {"files": {"map": jobs._artifact(result["outputs"]["map"]),
                "trajectory": jobs._artifact(result["outputs"]["kitti"]), "graph": jobs._artifact(result["outputs"]["g2o"])},
                "frames": job["pointcloud"]["frames"], "map_points": result["map_points"], "path_length_m": job["pointcloud"]["path_length_m"],
                "reports": {"correction": str(child / "pointcloud-report.json")},
                "quality_status": "generated_unverified", "coordinate_frame": job["pointcloud"]["coordinate_frame"]}
            if frame_ids is not None:
                child_job["pointcloud"]["files"].update(fusion_graph=jobs._artifact(result["outputs"]["fusion_g2o"]),
                    fusion_trajectory=jobs._artifact(result["outputs"]["fusion_kitti"]), fusion_scans=result["fusion_scans_manifest"])
                child_job["pointcloud"]["additional_frame_ids"] = frame_ids
                child_job["pointcloud"]["source_motion"] = job["pointcloud"]["source_motion"]
            if local_evidence is not None:
                child_job['pointcloud']['files'].update(local_update_report=update['report'],
                    local_update_checks=update['checks'], local_update_audits=update['audits'],
                    full_fusion_candidate=update['full_fusion_candidate'])
            child_job["status"] = "pointcloud_ready"
            jobs._save(child / "job.json", child_job)
            source_proposal = json.loads(Path(parent["corridor_proposal"]["path"]).read_text())
            source_options = source_proposal["protocol"]["options"]
            proposed = jobs.propose_mapping_corridors(str(child), source_options["search_radius_m"])
            if proposed["status"] != "ready":
                raise RuntimeError("retry corridor extraction failed; inspect child job")
            if source_options.get("association", "all_supported_bands") == "trajectory_containing":
                refined = jobs._refine_mapping_corridors(str(child), "Inherit the explicitly chosen parent association during density retry: " + reason)
                if refined["status"] != "ready":
                    raise RuntimeError("retry inherited association failed; inspect child job")
            active = jobs._load(child)["corridor_proposal"]
            new_proposal = json.loads(Path(active["file"]["path"]).read_text())
            if new_proposal["protocol"] != source_proposal["protocol"] or new_proposal["trajectory_length_m"] != source_proposal["trajectory_length_m"]:
                raise ValueError("density trial changed source thresholds, budgets or original input extent")
            jobs._inputs(child_job)
            _verify_inputs(saved["inputs"])
            run["status"] = "needs_agent"
            jobs._save(child / "run.json", run)
            stage.update(status="ready", pointcloud=child_job["pointcloud"], corridor_summary=active["summary"])
        except BaseException as error:
            stage.update(status="failed", error=str(error), error_type=type(error).__name__)
            if child.exists() and (child / "job.json").exists():
                failed_job = jobs._load(child)
                failed_job.update(status="pointcloud_retry_failed", error=str(error))
                jobs._save(child / "job.json", failed_job)
                if (child / "run.json").exists():
                    child_run = json.loads((child / "run.json").read_text())
                    child_run.update(status="processing_failed", error=str(error))
                    jobs._save(child / "run.json", child_run)
            if not isinstance(error, Exception) and not jobs._native_panic(error):
                jobs._save(root / "job.json", job)
                raise
        jobs._save(root / "job.json", job)
        return stage


def _routes(attempt: dict[str, Any]) -> dict[str, Any]:
    if "routes" in attempt:
        return cast(dict[str, Any], attempt["routes"]["after"])
    geometry = json.loads(Path(attempt["geometry_inputs"]["report"]["path"]).read_text())
    report = json.loads(Path(attempt["files"]["report"]["path"]).read_text())
    segments = {s["curve_ids"]["center"]: s for s in geometry["segments"]}
    intervals = {}
    for piece in report["built_segments"]:
        if len(piece["lane_ids"]) != 1:
            return {"available": False, "reason": "route station-span comparison currently requires single-lane pieces"}
        s = segments[piece["center_curve_id"]]
        intervals[piece["lane_ids"][0]] = (s["from_m"], s["to_m"])
    return route_metrics(json.loads(Path(attempt["files"]["editable_map"]["path"]).read_text()), intervals)


def compare(root: Path, cid: int) -> dict[str, Any]:
    """Compare audited maps on the identical frozen station denominator."""
    from ca.mapping_run import _load as load_run
    job = jobs._load(root)
    stage = job.get("pointcloud_retry", {})
    if stage.get("status") != "ready":
        raise ValueError("compare a ready density retry")
    child = Path(stage["child_job_dir"])
    child_job = jobs._load(child)
    jobs._inputs(job)
    jobs._inputs(child_job)
    parent = _parent(job, stage["candidate_id"])
    candidate = _parent(child_job, cid)
    if stage.get('strategy') in {'local_unused_frames', 'local_density'} and 'patch_inputs' not in candidate:
        raise ValueError("local point updates require a combined HD patch retaining the baseline map")
    child_run = load_run(child)
    if cid not in [h.get("lane_candidate_id") for h in child_run["history"]]:
        raise ValueError("compare an audited lane candidate generated by the retry run")
    if (job["source"] != child_job["source"] or job["runtime"]["native"] != child_job["runtime"]["native"]
        or child_run["layout_file"]["sha256"] != json.loads((root / "run.json").read_text())["layout_file"]["sha256"]
        or any(job["pointcloud"]["files"][k]["sha256"] != child_job["pointcloud"]["files"][k]["sha256"] for k in ("trajectory", "graph"))):
        raise ValueError("retry comparison changed source, native, layout or frozen reference motion")
    reports = [json.loads(Path(a["files"]["report"]["path"]).read_text()) for a in (parent, candidate)]
    proposals = [json.loads(Path(a["corridor_proposal"]["path"]).read_text()) for a in (parent, candidate)]
    if proposals[0]["protocol"] != proposals[1]["protocol"]:
        raise ValueError("retry comparison changed source extraction protocol")
    if parent["extent"]["trajectory_length_m"] != candidate["extent"]["trajectory_length_m"] or parent["extent"]["minimum_retained_fraction"] != candidate["extent"]["minimum_retained_fraction"]:
        raise ValueError("retry comparison changed the original denominator or extent goal")
    cuts = sorted({v for report in reports for i in report["station_disposition"] for v in (i["from_m"], i["to_m"])})
    generated = [[(i["from_m"], i["to_m"]) for i in r["station_disposition"] if i["status"] == "included_lane_hypothesis"] for r in reports]
    gained, lost = [], []
    for lo, hi in zip(cuts, cuts[1:]):
        before, after = [any(a <= (lo + hi) / 2 <= b for a, b in intervals) for intervals in generated]
        if after and not before:
            gained.append({"from_m": lo, "to_m": hi})
        elif before and not after:
            lost.append({"from_m": lo, "to_m": hi})
    diagnoses = [jobs.diagnose_mapping_candidate(str(path), a["id"]) for path, a in ((root, parent), (child, candidate))]
    def summary(d: dict[str, Any]) -> dict[str, Any]:
        audits = {"editable": d["editable"], "reopened_osm": d["reopened_osm"],
                  **{f"consensus_{k}": v for k, v in (d["ground_consensus"] or {}).items()}}
        return {"extent": d["extent"], "audits": {k: {"complete": v["complete"], "sample_totals": v["sample_totals"],
                "needs_review": v["needs_review"], "protocol": v["protocol"]} for k, v in audits.items()}}
    summaries = [summary(d) for d in diagnoses]
    expected_audits = {"editable", "reopened_osm", "consensus_editable", "consensus_reopened_osm"}
    if any(set(s["audits"]) != expected_audits for s in summaries) or any(
        summaries[0]["audits"][k]["protocol"] != summaries[1]["audits"][k]["protocol"] for k in expected_audits
    ):
        raise ValueError("retry comparison needs all four audits at identical quality protocols")
    inputs = {**{f"before_{k}": v for k, v in parent["files"].items()}, **{f"after_{k}": v for k, v in candidate["files"].items()},
              "before_audits": parent["quality_report"], "after_audits": candidate["quality_report"], "gap_evidence": stage["evidence"],
              **{f"after_lineage_{k}": v for k, v in child_job["retry_inputs"].items()},
              **{f"after_pointcloud_{k}": v for k, v in child_job["pointcloud"]["files"].items()}}
    _verify_inputs(inputs)
    report = {"schema": "cloudanalyzer.mapping_retry_comparison.v1", "parent_candidate_id": parent["id"], "retry_candidate_id": cid,
              "retry_strategy": stage.get("strategy", "density"), "added_frame_ids": stage.get("frame_ids"),
              "inputs": inputs, "before": {**summaries[0], "routes": _routes(parent), "map_points": job["pointcloud"]["map_points"]},
              "after": {**summaries[1], "routes": _routes(candidate), "map_points": child_job["pointcloud"]["map_points"]},
              "gained_source_intervals": gained, "lost_source_intervals": lost,
              "gained_source_length_m": sum(i["to_m"] - i["from_m"] for i in gained),
              "lost_source_length_m": sum(i["to_m"] - i["from_m"] for i in lost),
              "same_frozen_reference_motion": True, "same_fixed_layout": True, "same_original_extent_goal": True,
              "automatic_adoption": False, "deployment_ready": False}
    if stage.get('strategy') in {'local_unused_frames', 'local_density'}:
        local_report = json.loads(Path(child_job['pointcloud']['files']['local_update_report']['path']).read_text())
        report['local_point_update'] = {k: local_report[k] for k in ('effective_bounds_xy', 'before_inside_points',
            'after_inside_points', 'outside_points', 'outside_records_sha256', 'outside_records_bit_identical',
            'passes', 'full_candidate_generation_still_required')}
    path = root / f"retry-comparison-{cid:02d}.json"
    if path.exists():
        retained = json.loads(path.read_text())
        # Older completed density comparisons lack these optional strategy fields.
        expected = dict(report)
        if stage.get("strategy", "density") == "density":
            for key in ("retry_strategy", "added_frame_ids"):
                if key not in retained:
                    expected.pop(key)
        if retained != expected:
            raise ValueError("retained retry comparison changed")
        report = retained
    else:
        jobs._save(path, report)
    # Full chains and interval lists remain in the hashed report; bound the tool response.
    for mode in ("before", "after"):
        report[mode]["routes"].pop("routes", None)
    for key in ("gained_source_intervals", "lost_source_intervals"):
        report[key + "_total"] = len(report[key])
        report[key] = report[key][:16]
    report.pop("inputs")
    return {"file": jobs._artifact(path), **report}
