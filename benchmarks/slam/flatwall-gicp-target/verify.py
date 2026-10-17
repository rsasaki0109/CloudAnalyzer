#!/usr/bin/env python3
"""Verify published bytes and reported immutable-target evidence summaries."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root=Path(__file__).resolve().parent):
    root = Path(root)
    manifest = read(root/"packet-files.json")
    expected = {"slice-protocol.json", "native-build-inputs.json", "deskew-regression/receipt.json",
                "integration/receipt.json", *("audit/"+name for name in ["protocol.json", "receipt.json", "frames.json", "summary.json"])}
    assert set(manifest["json_files"]) == expected
    for name, item in manifest["json_files"].items():
        data = (root/name).read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], name
    verifier = (root/"verify.py").read_bytes()
    assert hashlib.sha256(verifier).hexdigest() == manifest["verifier"]["sha256"]
    protocol, receipt, frames, summary = [read(root/"audit"/name) for name in ["protocol.json", "receipt.json", "frames.json", "summary.json"]]
    assert protocol["schema"] == receipt["schema"] == "cloudanalyzer.flatwall_gicp_target.v1"
    assert protocol["slice"] == read(root/"slice-protocol.json")
    assert protocol["families"] == ["caller_shapes", "native_shapes"]
    assert protocol["rows"] == [0, 1, 2, 9, 17, 33, 49, 60, 63]
    assert len(protocol["cases"]) == 4 and len(summary) == 8
    assert receipt["verified_registrations"] == len(frames) == 72
    assert receipt["standalone_frozen_bit_equal"] and receipt["prepared_standalone_bit_equal"] and receipt["reuse_parallel_owned_bit_equal"]
    assert receipt["no_trajectory_or_frontend_adoption"]
    assert receipt["frozen_input_artifacts_unchanged"] == len(protocol["input_artifacts"])
    for name in ["controller", "native"]:
        assert receipt[name] == protocol[name]
    assert protocol["build_inputs"]["sha256"] == manifest["json_files"]["native-build-inputs.json"]["sha256"]
    assert read(root/"native-build-inputs.json")["original_loop_and_python_result_serialization_unchanged"]
    for name in ["protocol.json", "frames.json", "summary.json"]:
        assert receipt["outputs"][name] == manifest["json_files"]["audit/"+name]
    for family in protocol["families"]:
        rows = [r for r in frames if r["family"] == family]
        assert Counter(r["termination"] for r in rows) == {"converged_step": 27, "max_iterations": 5, "insufficient_correspondences": 4}
        for case in protocol["cases"]:
            selected = [r for r in rows if r["case"] == case]
            assert [r["row"] for r in selected] == protocol["rows"]
            group = next(s for s in summary if (s["family"], s["case"]) == (family, case))
            assert group["registrations"] == len(selected) == 9
            assert group["statuses"] == Counter(r["termination"] for r in selected)
            for key in ["source_points", "target_points", "iterations", "trial_count"]:
                assert group[key] == sum(r[key] for r in selected)
            for row in selected:
                assert row["standalone_frozen_bit_equal"] and row["prepared_standalone_bit_equal"] and row["reuse_parallel_owned_bit_equal"]
                assert row["max_reference_pose_error"] < 2e-7
    controller = root.parents[2]/"scripts/audit_flatwall_gicp_target.py"
    if controller.exists():
        assert hashlib.sha256(controller.read_bytes()).hexdigest() == receipt["controller"]["sha256"]
    return dict(public_json_files_verified=9, families=2, registrations=len(frames),
                iterations=sum(r["iterations"] for r in frames), trial_costs=sum(r["trial_count"] for r in frames),
                scope="Published identities, bytes, schedules and reported summaries only; no independent recording rerun, physical accuracy or timing result.")


if __name__ == "__main__":
    print(json.dumps(verify()))
