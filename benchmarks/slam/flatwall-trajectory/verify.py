#!/usr/bin/env python3
"""Verify published bytes, source footprints and conditional-score accounting."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(p): return json.loads(p.read_text(encoding="utf-8"))


def stats(values):
    ordered = sorted(values)
    assert ordered and all(math.isfinite(v) and v >= 0 for v in ordered)
    index = .95*(len(ordered)-1)
    left = math.floor(index)
    p95 = ordered[left] + (index-left)*(ordered[min(left+1, len(ordered)-1)]-ordered[left])
    return dict(count=len(ordered), rmse=math.sqrt(statistics.fmean(v*v for v in ordered)),
                mean=statistics.fmean(ordered), median=statistics.median(ordered), p95=p95, max=ordered[-1])


def verify(root=None):
    source_root = Path(__file__).resolve().parents[3] if root is None else None
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = read(root/"packet-files.json")["files"]
    for name, item in manifest.items():
        assert Path(name).name == name
        data = (root/name).read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], name
    followup_files = read(root/"strict-time-files.json")["files"]
    for name, item in followup_files.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        data = (root/name).read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], name
    p, s, c, o = [read(root/name) for name in ["protocol.json", "summary.json", "controller-summary.json", "oracle-receipt.json"]]
    rows, edges, ref = [read(root/name) for name in ["samples.json", "edges.json", "reference-member.json"]]
    assert s["status"] == "conditional_numeric_comparison" and s["unresolved_provenance"]
    assert s["reference_inspection"].startswith("exploratory;")
    assert s["interpretation"] == p["conditional_interpretation"]
    assert s["estimate_commit"] == p["estimate_commit"]
    assert c["reference_full_trajectory_parsed"] is False
    assert o["full_reference_first_read_after_controller_freeze"] is True and o["physical_provenance_certified"] is False
    assert datetime.fromisoformat(p["frozen_at_utc"]) <= datetime.fromisoformat(c["at_utc"]) <= datetime.fromisoformat(o["started_at_utc"])
    assert c["protocol_sha256"] == o["protocol_sha256"] == manifest["protocol.json"]["sha256"]
    assert c["original_controller_freeze_sha256"] == o["controller_freeze_sha256"]
    assert c["analytic_controls_passed"] == 32
    archived = c["source_files_sha256"]["notes/audit-slam-flatwall-trajectory-oracle-20261017.py"]
    assert archived == manifest["oracle-controller.py"]["sha256"]
    correction, corrected_freeze, corrected_receipt = [read(root/name) for name in ["strict-time-comparison.json", "strict-time-controller-freeze.json", "strict-time-audit-receipt.json"]]
    assert correction["complete_reference_seen_before_correction"] is corrected_freeze["complete_reference_previously_inspected"] is True
    assert correction["policy_and_numeric_thresholds_unchanged"] is correction["original_results_unchanged"] is True
    assert correction["corrected_probe_rejected"] is True
    assert correction["controls_passed"] == corrected_freeze["analytic_controls_passed"] == 35
    assert corrected_freeze["protocol_sha256"] == correction["original_protocol_sha256"] == manifest["protocol.json"]["sha256"]
    assert corrected_receipt["controller_freeze_sha256"] == followup_files["strict-time-controller-freeze.json"]["sha256"]
    assert datetime.fromisoformat(corrected_freeze["at_utc"]) > datetime.fromisoformat(o["completed_at_utc"])
    assert correction["corrected_sources_sha256"] == corrected_freeze["source_files_sha256"]
    for name, item in correction["original_sources"].items():
        assert item["sha256"] == c["source_files_sha256"][name] == followup_files[item["path"]]["sha256"]
    assert sum(len(v) for v in correction["paired_and_error_arrays_bit_equal"].values()) == 9
    for name, expected in correction["original_reports_byte_equal"].items():
        assert expected == manifest[name]["sha256"] == corrected_receipt["output_files_sha256"][name]
    if source_root is not None:
        for name, expected in corrected_freeze["source_files_sha256"].items():
            assert hashlib.sha256((source_root/name).read_bytes()).hexdigest() == expected, name
    assert len(rows) == s["original_estimates"] == o["estimates_checked"] == 185
    assert [r["estimate_index"] for r in rows] == list(range(185))
    assert [r["scan_index"] for r in rows] == list(range(1, 186))
    assert all(b["timestamp_ns"] > a["timestamp_ns"] for a, b in zip(rows[:-1], rows[1:]))
    assert dict(Counter(r["status"] for r in rows)) == s["coverage_counts"] == {"interpolated": 183, "after_reference": 2}
    assert s["original_recording_excluded_scan_indices"] == [0]
    paired = {r["estimate_index"] for r in rows if r["status"] in {"exact", "interpolated"}}
    assert len(paired) == s["paired_estimates"] == o["paired_estimates"] == 183
    assert s["full_cohort_covered"] is False
    assert ref["rows"] == o["reference_rows"] == 729
    assert ref["maximum_original_quaternion_norm_error"] <= p["validation"]["quaternion_norm_atol"]
    names = {"origin_position_ape_m", "origin_rotation_ape_deg", "rigid_position_ape_m"}
    for row in rows:
        if row["estimate_index"] in paired:
            assert row["status"] == "interpolated" and set(row["errors"]) == names
            assert 0 <= row["left"] < row["right"] < ref["rows"]
            assert 0 < row["bracket_span_ns"] <= p["association"]["maximum_bracket_span_ns"]
            assert 0 < row["fraction"] < 1
            assert ref["first_timestamp_ns"] < row["timestamp_ns"] < ref["last_timestamp_ns"]
        else:
            assert row["errors"] is None and row["left"] is row["right"] is None
            assert row["timestamp_ns"] > ref["last_timestamp_ns"]
    expected_edges = [(i, i+1) for i in range(184) if i in paired and i+1 in paired]
    assert [(e["left_estimate_index"], e["right_estimate_index"]) for e in edges] == expected_edges
    assert len(edges) == o["consecutive_edges_checked"] == 182
    for e in edges:
        assert e["elapsed_ns"] == rows[e["right_estimate_index"]]["timestamp_ns"]-rows[e["left_estimate_index"]]["timestamp_ns"] > 0
    values = {name: [r["errors"][name] for r in rows if r["estimate_index"] in paired] for name in names}
    values["consecutive_rpe_translation_m"] = [e["translation_m"] for e in edges]
    values["consecutive_rpe_rotation_deg"] = [e["rotation_deg"] for e in edges]
    assert set(values) == set(s["metrics"])
    for name, items in values.items():
        for key, value in stats(items).items():
            assert math.isclose(value, s["metrics"][name][key], rel_tol=1e-12, abs_tol=1e-12), (name, key)
    fit = s["position_fit"]
    assert fit["status"] == "identified" and fit["scale"] == 1.0 and min(fit["ranks"].values()) >= 2
    for name, spectrum in fit["singular_values"].items():
        assert sum(v > spectrum[0]*fit["relative_rank_threshold"] for v in spectrum) == fit["ranks"][name]
    r = fit["world_alignment"]
    det = r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0])
    assert math.isclose(det, 1., abs_tol=1e-10)
    for name, value in o["maximum_abs_differences"].items():
        assert 0 <= value <= (1e-5 if name.endswith("deg") else 2e-8)
    assert "deviation" in read(root/"reference-policy-deviation.json")
    return dict(original_estimates=185, paired_estimates=183, excluded_after_reference=2, consecutive_edges=182,
                conditional_rigid_position_rmse_m=s["metrics"]["rigid_position_ape_m"]["rmse"], physical_provenance_certified=False,
                scope="Published bytes, source footprints and reported accounting; no independent raw-array rerun, missing body/clock provenance or SLAM accuracy certificate.")


if __name__ == "__main__": print(json.dumps(verify(), indent=2))
