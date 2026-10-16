#!/usr/bin/env python3
"""Verify published identities and iteration summaries; no real-data rerun."""
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
    assert receipt["controller"] == protocol["controller"] == artifact(ROOT/"scripts/audit_flatwall_gicp_registration.py")
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
    assert receipt["verified_registrations"] == 36
    assert receipt["frozen_input_artifacts_unchanged"] == len(protocol["inputs"]) == 272
    assert receipt["serial_parallel_bit_equal"] and receipt["no_trajectory_or_frontend_adoption"]
    for name in ["frames.json", "summary.json"]:
        assert receipt["outputs"][name] == artifact(packet/"audit"/name), f"changed {name}"
    frames, summaries = read("audit/frames.json"), read("audit/summary.json")
    assert len(frames) == 36 and [s["case"] for s in summaries] == cases
    for summary in summaries:
        rows = [r for r in frames if r["case"] == summary["case"]]
        assert [r["row"] for r in rows] == schedule
        assert summary["registrations"] == 9
        assert summary["statuses"] == dict(Counter(r["termination"] for r in rows))
        for name in ["iterations", "accepted_steps", "trial_count"]:
            assert summary[name] == sum(r[name] for r in rows)
        for row in rows:
            assert row["serial_parallel_bit_equal"] and row["final_matches"] >= 0
            assert row["unsupported_source_points"] >= 0 and row["unsupported_target_points"] >= 0
            assert len(row["iterations_summary"]) == row["iterations"]
            accepted, trials, scales = 0, 0, Counter()
            for step in row["iterations_summary"]:
                assert math.isfinite(step["cost_before"]) and step["cost_before"] >= 0
                assert len(step["trial_scales"]) == len(step["trial_costs"])
                assert step["trial_scales"] == [1., .5, .25, .125, .0625, .03125][:len(step["trial_scales"])]
                trials += len(step["trial_scales"])
                if step["accepted_scale"] is not None:
                    accepted += 1
                    scales[str(step["accepted_scale"])] += 1
                    assert step["trial_scales"][-1] == step["accepted_scale"]
                    assert step["trial_costs"][-1] == step["cost_after"] < step["cost_before"]
                    assert all(v is None or v >= step["cost_before"] for v in step["trial_costs"][:-1])
                else:
                    assert step["cost_after"] is None
            assert accepted == row["accepted_steps"] and trials == row["trial_count"]
            assert dict(scales) == row["accepted_scales"]
            for name in ["native_reference_pose_max_error", "max_trial_cost_error", "max_shape_error", "max_system_error"]:
                assert math.isfinite(row[name]) and row[name] >= 0
        assert summary["max_pose_error"] == max(r["native_reference_pose_max_error"] for r in rows)
        assert summary["max_trial_cost_error"] == max(r["max_trial_cost_error"] for r in rows)
        assert summary["max_system_error"] == max(r["max_system_error"] for r in rows)
    assert receipt["verified_outer_iterations"] == sum(r["iterations"] for r in frames)
    assert receipt["verified_trial_costs"] == sum(r["trial_count"] for r in frames)
    build = read("native-build-inputs.json")
    assert len(build["sources_sha256"] | build["python_package_sources_sha256"] | build["additional_config_sha256"]) == build["unique_paths"] == 110
    return dict(public_json_files_verified=9, cases=4, registrations=36,
                outer_iterations=receipt["verified_outer_iterations"], trial_costs=receipt["verified_trial_costs"],
                scope="Published bytes, controller identities, schedule and reported frozen-cost acceptance only; no independent real-data rerun or physical accuracy result.")


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
