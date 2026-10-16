#!/usr/bin/env python3
"""Verify native-shape packet identities and reported summaries; no data rerun."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent


def artifact(path):
    data = path.read_bytes()
    return dict(name=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def verify(packet=None):
    packet = PACKET if packet is None else Path(packet)
    def read(name):
        return json.loads((packet/name).read_text(encoding="utf-8"))
    manifest = read("packet-files.json")
    assert len(manifest["json_files"]) == 8
    for name, expected in manifest["json_files"].items():
        assert artifact(packet/name) == expected, f"changed {Path(name).name}"
    assert manifest["verifier"] == artifact(Path(__file__)), "changed verifier"
    scope, protocol, receipt = read("slice-protocol.json"), read("audit/protocol.json"), read("audit/receipt.json")
    assert protocol["slice"] == scope
    assert receipt["protocol"] == artifact(packet/"audit/protocol.json")
    assert receipt["controller"] == protocol["controller"] == artifact(ROOT/"scripts/audit_flatwall_gicp_shapes.py")
    assert receipt["native"] == protocol["native"]
    for name, expected in protocol["reference_dependencies"].items():
        assert artifact(ROOT/"scripts"/name) == expected
    expected_scope = artifact(packet/"slice-protocol.json")
    expected_scope["name"] = "protocol.json"
    assert protocol["slice_protocol"] == expected_scope
    assert protocol["build_inputs"] == artifact(packet/"native-build-inputs.json")
    cases = ["raw-legacy", "raw-fine", "rotation-legacy", "rotation-fine"]
    schedule = [0, 1, 2, 9, 17, 33, 49, 60, 63]
    assert protocol["cases"] == cases and protocol["rows"] == schedule
    assert receipt["verified_registrations"] == 36 and receipt["verified_point_sets"] == 72
    assert receipt["verified_shape_points"] == 47889
    assert receipt["frozen_input_artifacts_unchanged"] == len(protocol["input_artifacts"]) == 44
    assert receipt["serial_parallel_bit_equal"] and receipt["old_caller_shape_results_bit_equal"] and receipt["no_trajectory_or_frontend_adoption"]
    for name in ["frames.json", "summary.json"]:
        assert receipt["outputs"][name] == artifact(packet/"audit"/name), f"changed {name}"
    frames, summaries = read("audit/frames.json"), read("audit/summary.json")
    assert len(frames) == 36 and [s["case"] for s in summaries] == cases
    for summary in summaries:
        rows = [r for r in frames if r["case"] == summary["case"]]
        assert [r["row"] for r in rows] == schedule and summary["registrations"] == 9
        for field, status in [("statuses", "termination"), ("old_statuses", "old_termination")]:
            assert summary[field] == dict(Counter(r[status] for r in rows))
        assert summary["changed_terminations"] == sum(r["termination"] != r["old_termination"] for r in rows) == 0
        assert summary["iterations"] == sum(r["iterations"] for r in rows)
        assert summary["trial_count"] == sum(r["trial_count"] for r in rows)
        for row in rows:
            assert row["serial_parallel_bit_equal"] and row["old_caller_shape_result_bit_equal"]
            for name in ["native_reference_pose_max_error", "old_pose_max_difference", "old_pose_translation_difference_m"]:
                assert math.isfinite(row[name]) and row[name] >= 0
            for name in ["source", "target"]:
                stat = row[name]
                assert stat["supported"] + stat["unsupported"] == stat["points"]
                assert 0 <= stat["ambiguous_minimum_eigenspaces"] <= stat["supported"]
                assert 0 <= stat["boundary_ties"] <= stat["points"]
                for field in ["max_neighbor_distance_error", "max_eigenvalue_error", "max_separated_shape_error", "max_relative_eigenpair_residual", "max_old_scipy_shape_difference"]:
                    assert math.isfinite(stat[field]) and stat[field] >= 0
        for field in ["unsupported", "boundary_ties", "ambiguous_minimum_eigenspaces"]:
            assert summary[field] == sum(r[p][field] for r in rows for p in ["source", "target"]) == 0
        assert summary["shape_points"] == sum(r[p]["points"] for r in rows for p in ["source", "target"])
        assert summary["max_shape_error"] == max(r[p]["max_separated_shape_error"] for r in rows for p in ["source", "target"])
        assert summary["max_old_scipy_shape_difference"] == max(r[p]["max_old_scipy_shape_difference"] for r in rows for p in ["source", "target"])
        assert summary["max_pose_reference_error"] == max(r["native_reference_pose_max_error"] for r in rows)
        assert summary["max_old_pose_difference"] == max(r["old_pose_max_difference"] for r in rows)
    assert sum(s["shape_points"] for s in summaries) == 47889
    assert sum(s["iterations"] for s in summaries) == 374 and sum(s["trial_count"] for s in summaries) == 343
    build = read("native-build-inputs.json")
    assert len(build["sources_sha256"] | build["python_package_sources_sha256"] | build["additional_config_sha256"]) == build["unique_paths"] == 111
    return dict(public_json_files_verified=9, cases=4, registrations=36, point_sets=72, shape_points=47889,
                scope="Published bytes, controller identities, schedule and reported summaries only; no independent recording rerun, physical accuracy or timing result.")


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
