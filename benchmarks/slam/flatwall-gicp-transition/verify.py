#!/usr/bin/env python3
"""Check public transition evidence bytes, identities and reported accounting."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root=Path(__file__).resolve().parent):
    root = Path(root)
    manifest = read(root/"packet-files.json")
    expected = {"slice-protocol.json", "reused-native.json",
        *("audit/"+name for name in ["protocol.json", "receipt.json", "frames.json", "summary.json", "limited-tails.json"]),
        *("analysis/"+name for name in ["protocol.json", "receipt.json", "summary.json", "limited-effects.json", "recurrence.json"])}
    assert set(manifest["json_files"]) == expected
    for name, item in manifest["json_files"].items():
        data = (root/name).read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], name
    assert hashlib.sha256((root/"verify.py").read_bytes()).hexdigest() == manifest["verifier"]["sha256"]
    protocol, receipt, frames, summaries, tails = [read(root/"audit"/name) for name in ["protocol.json", "receipt.json", "frames.json", "summary.json", "limited-tails.json"]]
    analysis_protocol, analysis_receipt, analysis, limited, recurrence = [read(root/"analysis"/name) for name in ["protocol.json", "receipt.json", "summary.json", "limited-effects.json", "recurrence.json"]]
    assert receipt["schema"] == protocol["schema"] == "cloudanalyzer.flatwall_gicp_transition.v1"
    assert analysis_receipt["schema"] == analysis_protocol["schema"] == "cloudanalyzer.flatwall_gicp_transition_analysis.v1"
    assert protocol["slice"] == read(root/"slice-protocol.json")
    assert protocol["cube_axis_order"] == ["assignment", "source_metric_pose", "residual_pose"]
    assert receipt["native"]["sha256"] == read(root/"reused-native.json")["native_sha256"]
    assert receipt["controller"] == protocol["controller"]
    assert protocol["reused_native"] == manifest["json_files"]["reused-native.json"]
    assert receipt["histories"] == len(frames) == analysis["histories"] == 72
    assert receipt["accepted_transitions"] == sum(r["accepted_transitions"] for r in frames) == 686
    assert receipt["native_cube_cost_checks"] == 5488 and receipt["native_combined_shape_checks"] == 2744
    assert receipt["limited_tail_transitions"] == len(tails) == analysis_receipt["limited_tail_transitions"] == 100
    assert receipt["original_poses_and_statuses_unchanged"] and analysis_receipt["solver_and_termination_unchanged"]
    assert Counter(r["termination"] for r in frames) == analysis["statuses"] == {"converged_step": 54, "max_iterations": 10, "insufficient_correspondences": 8}
    for family in ["caller_shapes", "native_shapes"]:
        for case in ["raw-legacy", "raw-fine", "rotation-legacy", "rotation-fine"]:
            rows = [r for r in frames if (r["family"], r["case"]) == (family, case)]
            assert [r["row"] for r in rows] == [0, 1, 2, 9, 17, 33, 49, 60, 63]
            summary = next(r for r in summaries if (r["family"], r["case"]) == (family, case))
            assert summary["frames"] == len(rows) == 9
            assert summary["statuses"] == Counter(r["termination"] for r in rows)
            for key in ["accepted_transitions", "refreshed_increases", "common_increases", "support_changed_transitions"]:
                assert summary[key] == sum(r[key] for r in rows)
    assert analysis["all_transitions"]["transitions"] == analysis["all_transitions"]["strict_frozen_motion_decreases"] == 686
    assert analysis["all_transitions"]["strict_refreshed_increases"] == sum(r["refreshed_increases"] for r in frames) == 278
    assert analysis["all_transitions"]["support_changed_transitions"] == sum(r["support_changed_transitions"] for r in frames) == 0
    assert len(limited) == len(recurrence) == 10
    for group in limited:
        selected = [r for r in tails if all(r[key] == group[key] for key in ["family", "case", "row"])]
        assert len(selected) == group["transitions"] == 10
        assert group["strict_refreshed_increases"] == sum(r["full_delta"] > 0. for r in selected)
    for row in tails:
        assert len(row["cube"]) == 8 and row["motion_delta"] < 0.
        assert abs(row["accounting_error"]) < 2e-8
        assert abs(row["common_delta"]-row["motion_delta"]-row["source_metric_delta"]-row["assignment_delta"]) < 2e-8
        assert abs(row["full_delta"]-row["common_delta"]-row["added_cost"]+row["removed_cost"]) < 2e-8
    assert analysis["max_native_cost_error"] < 2.4e-12
    assert analysis["max_native_shape_error"] < 4.5e-16
    assert analysis["max_accounting_error"] < 1.5e-14
    for group in recurrence:
        assert group["last_ten_states"] == 10 and set(group["lags"]) == {"1", "2", "3", "4", "5"}
        assert all(0 <= row["equal_ordered_correspondence_sets"] <= row["comparisons"] == 10 for row in group["lags"].values())
    for name, item in receipt["outputs"].items():
        if name != "transitions.json":
            assert manifest["json_files"]["audit/"+name] == item
    for name, item in analysis_receipt["outputs"].items():
        assert manifest["json_files"]["analysis/"+name] == item
    assert analysis_protocol["audit_receipt"] == manifest["json_files"]["audit/receipt.json"]
    assert analysis_protocol["audit_outputs"] == receipt["outputs"]
    assert analysis_protocol["controller"] == analysis_receipt["controller"]
    for name, item in [("scripts/audit_flatwall_gicp_transition.py", receipt["controller"]),
                       ("scripts/analyze_flatwall_gicp_transitions.py", analysis_receipt["controller"])]:
        path = root.parents[2]/name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], name
    return dict(public_json_files_verified=13, histories=72, accepted_transitions=686, native_cube_cost_checks=5488,
        native_combined_shape_checks=2744, refreshed_increases=278, support_changed_transitions=0,
        limited_tail_transitions=100, scope="Published bytes, identities and reported accounting only; no independent real-input rerun, new optimizer, accepted pose, physical accuracy or performance claim.")


if __name__ == "__main__":
    print(json.dumps(verify()))
