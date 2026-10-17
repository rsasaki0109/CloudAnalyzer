#!/usr/bin/env python3
"""Check published bytes, frozen source identity and all control accounting."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ["raw-legacy", "gyro-legacy", "raw-0.1875", "gyro-0.1875"]
ALL = ARMS + ["nominal-imu-legacy"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values = sorted(values)
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    location = .95*(len(values)-1)
    left = math.floor(location)
    return dict(count=len(values), rmse=math.sqrt(statistics.fmean(v*v for v in values)),
                mean=statistics.fmean(values), median=statistics.median(values),
                p95=values[left]+(location-left)*(values[min(left+1,len(values)-1)]-values[left]), max=values[-1])


def verify(root=None):
    source_root = Path(__file__).resolve().parents[3] if root is None else None
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    files = read(root/"packet-files.json")["files"]
    for name, expected in files.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        data = (root/name).read_bytes()
        assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"], name
    protocol, freeze, closure, boundary, summary, oracle = [read(root/name) for name in
        ["protocol.json", "controller-freeze.json", "estimates-receipt.json", "scoring-boundary.json", "summary.json", "oracle-receipt.json"]]
    assert protocol["schema"] == "cloudanalyzer.flatwall_frontend_controls.v1"
    assert protocol["physical_provenance_certified"] is summary["physical_provenance_certified"] is oracle["physical_provenance_certified"] is False
    assert summary["status"] == "exploratory_conditional_comparison"
    assert summary["reference_previously_inspected"] is closure["reference_already_inspected"] is freeze["reference_fully_inspected_before_this_experiment"] is True
    assert summary["no_initializer_state_adopted"] is closure["no_initializer_state_adopted"] is True
    assert summary["all_estimates_frozen_before_this_scoring"] is boundary["all_four_complete_estimates_frozen_before_this_scoring"] is True
    assert closure["protocol_sha256"] == freeze["protocol_sha256"] == files["protocol.json"]["sha256"]
    assert closure["controller_freeze_sha256"] == files["controller-freeze.json"]["sha256"]
    assert oracle["scoring_boundary_sha256"] == files["scoring-boundary.json"]["sha256"]
    assert datetime.fromisoformat(protocol["frozen_at_utc"]) <= datetime.fromisoformat(freeze["at_utc"]) <= datetime.fromisoformat(closure["started_at_utc"]) <= datetime.fromisoformat(closure["frozen_at_utc"]) <= datetime.fromisoformat(boundary["at_utc"])
    assert list(protocol["arms"]) == closure["arms"] == ARMS
    assert set(summary["controls"]) == set(oracle["controls"]) == set(ALL)
    assert protocol["original_scan_indices"] == closure["original_scan_indices"] == list(range(1,186))
    assert closure["skipped_scan_indices"] == protocol["excluded_original_scan_indices"] == [0]
    assert closure["points_per_arm"] == 4_440_000
    assert closure["all_64_prefix_pose_bits_and_reports_match"] is closure["prior_full_gyro_component_digest_matches"] is True
    assert closure["no_reference_poses_used_in_this_stage"] is True
    assert summary["original_estimates_per_arm"] == 185 and summary["paired_estimates_per_arm"] == 183
    assert summary["full_cohort_covered"] is False
    assert summary["coverage_counts"] == {"interpolated":183,"after_reference":2}
    assert summary["original_nominal_metrics_reproduced_exactly"] is True
    assert summary["controls"]["nominal-imu-legacy"]["metrics"] == read(root/"prior-nominal-summary.json")["metrics"]
    assert protocol["scoring_protocol_sha256"] == files["inherited-scoring-protocol.json"]["sha256"]
    inherited = read(root/"inherited-scoring-protocol.json")
    assert inherited["association"]["maximum_bracket_span_ns"] == 100_000_000
    assert protocol["lidar"] == dict(min_range=1.5,max_range=80.,map_voxel=1.,map_points=20,deskew=False,diagnostics=True)
    assert protocol["gyro_bias_rad_s"] == [0.,0.,0.] and protocol["max_gap_s"] == .05
    assert protocol["arms"] == {name:dict(compensation="gyro" if name.startswith("gyro") else "raw",
        registration_voxel=None if name.endswith("legacy") else .1875) for name in ARMS}
    for name in ARMS:
        for file in ["trajectory.npz","per-scan.json"]:
            assert closure["output_files_sha256"][name+"/"+file] == files[name+"/"+file]["sha256"]
    shared_times, shared_coverage = None, None
    ape_names = {"rigid_position_ape_m","origin_position_ape_m","origin_rotation_ape_deg"}
    for name in ALL:
        rows, edges, scans = [read(root/name/file) for file in ["samples.json","edges.json","per-scan.json"]]
        result = summary["controls"][name]
        assert len(rows) == len(scans) == 185
        assert [r["estimate_index"] for r in rows] == list(range(185))
        assert [r["scan_index"] for r in rows] == [s["scan_index"] for s in scans] == list(range(1,186))
        times = [r["timestamp_ns"] for r in rows]
        assert times == [s["reference_ns"] for s in scans] and all(b>a for a,b in zip(times[:-1],times[1:]))
        coverage = [{k:v for k,v in r.items() if k!="errors"} for r in rows]
        if shared_times is None:
            shared_times, shared_coverage = times, coverage
        assert times == shared_times and coverage == shared_coverage
        assert dict(Counter(r["status"] for r in rows)) == summary["coverage_counts"]
        covered = {r["estimate_index"] for r in rows if r["status"] == "interpolated"}
        assert covered == set(range(183))
        for row in rows:
            if row["estimate_index"] in covered:
                assert set(row["errors"]) == ape_names
                assert 0 < row["bracket_span_ns"] <= 100_000_000 and 0 < row["fraction"] < 1
            else:
                assert row["status"] == "after_reference" and row["errors"] is None
        assert [(e["left_estimate_index"],e["right_estimate_index"]) for e in edges] == [(i,i+1) for i in range(182)]
        assert len(edges) == oracle["controls"][name]["consecutive_edges"] == 182
        assert oracle["controls"][name]["paired_estimates"] == 183 and oracle["controls"][name]["prior_exact_pairing_reproduced"] is True
        for edge in edges:
            assert edge["elapsed_ns"] == times[edge["right_estimate_index"]]-times[edge["left_estimate_index"]] > 0
        values = {key:[r["errors"][key] for r in rows if r["estimate_index"] in covered] for key in ape_names}
        values.update(consecutive_rpe_translation_m=[e["translation_m"] for e in edges],consecutive_rpe_rotation_deg=[e["rotation_deg"] for e in edges])
        assert set(values) == set(result["metrics"])
        for key, value in values.items():
            for stat, expected in stats(value).items():
                assert math.isclose(result["metrics"][key][stat],expected,rel_tol=1e-12,abs_tol=1e-12), (name,key,stat)
        if name in ARMS:
            counts = dict(Counter(s["registration"]["termination"] for s in scans))
            assert counts == closure["termination_counts"][name]
            assert sum(counts.values()) == 185
        for key, difference in oracle["controls"][name]["maximum_abs_differences"].items():
            assert 0 <= difference <= (1e-5 if key.endswith("deg") else 2e-8)
    for name, expected in freeze["source_files_sha256"].items():
        if name.startswith(("scripts/","cloudanalyzer/tests/")):
            assert files["source-snapshot/"+name]["sha256"] == expected
            if source_root is not None:
                assert hashlib.sha256((source_root/name).read_bytes()).hexdigest() == expected, name
    for name in ["controller-revision.json","controller-revision-02.json"]:
        revision = read(root/"history"/name)
        assert revision["any_new_trajectory_produced_in_failed_attempt"] is False
        assert revision["estimator_settings_and_scoring_policy_changed"] is False
    return dict(arms=len(ALL),new_estimates_per_arm=185,paired_per_arm=183,edges_per_arm=182,
                conditional_rigid_position_rmse_m={n:summary["controls"][n]["metrics"]["rigid_position_ape_m"]["rmse"] for n in ALL},
                physical_provenance_certified=False, scope="Published bytes, source footprints and accounting; no independent raw-recording rerun or physical-accuracy certificate.")


if __name__ == "__main__":
    print(json.dumps(verify(),indent=2))
