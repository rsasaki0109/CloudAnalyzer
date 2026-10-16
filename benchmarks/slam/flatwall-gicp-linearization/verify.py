#!/usr/bin/env python3
"""Verify packet bytes and reported fixed-match geometry, without real data."""
from __future__ import annotations

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
    assert receipt["controller"] == protocol["inputs"]["controller"] == artifact(ROOT/"scripts/audit_flatwall_gicp_linearization.py")
    assert receipt["native"] == protocol["inputs"]["native"]
    assert protocol["inputs"]["map_replay_controller"] == artifact(ROOT/"scripts/audit_flatwall_odometry_linearization.py")
    expected_scope = artifact(packet/"slice-protocol.json")
    expected_scope["name"] = "protocol.json"
    assert protocol["inputs"]["slice_protocol"] == expected_scope
    assert protocol["inputs"]["build_inputs"] == artifact(packet/"native-build-inputs.json")
    cases = ["raw-legacy", "raw-fine", "rotation-legacy", "rotation-fine"]
    assert protocol["cases"] == cases
    assert receipt["verified_frames"] == 256 and receipt["frozen_input_artifacts_unchanged"] == 272
    assert receipt["serial_parallel_bit_equal"] and receipt["identity_controls_match_p2p"] and receipt["no_states_adopted"]
    for name in ["frames.json", "summary.json"]:
        assert receipt["outputs"][name] == artifact(packet/"audit"/name), f"changed {name}"
    frames, summaries = read("audit/frames.json"), read("audit/summary.json")
    assert len(frames) == 256 and [s["case"] for s in summaries] == cases
    for summary in summaries:
        rows = [r for r in frames if r["case"] == summary["case"]]
        assert len(rows) == summary["frames"] == 64
        assert [r["row"] for r in rows] == list(range(64))
        assert sum(r["original_matches"] for r in rows) == summary["original_pairs"]
        assert sum(r["retained_pairs"] for r in rows) == summary["retained_pairs"]
        assert summary["serial_parallel_bit_equal"] and summary["identity_controls_match_p2p"]
        supported = []
        for row in rows:
            assert 0 <= row["retained_pairs"] <= row["original_matches"]
            assert row["unsupported_source_pairs"] >= 0 and row["unsupported_target_pairs"] >= 0
            assert math.isfinite(row["weight_sum"]) and row["weight_sum"] >= 0
            for field in ["normal_matrix_max_error", "rhs_max_error", "cost_abs_error", "mahalanobis_max_error", "identity_control_normal_matrix_max_error"]:
                assert math.isfinite(row[field]) and row[field] >= 0
            if row["gicp_geometry"] is None:
                assert row["p2p_geometry"] is None and row["weight_sum"] == 0
                continue
            assert row["radius_m"] > 0 and row["weight_sum"] > 0
            for model in ["gicp", "p2p"]:
                geometry = row[model+"_geometry"]
                values = geometry["eigenvalues"]
                assert len(values) == 6 and sorted(values) == values and all(math.isfinite(v) for v in values)
                assert values[-1] > 0
                assert math.isclose(geometry["min_max_ratio"], values[0]/values[-1], abs_tol=1e-15)
                assert geometry["weak_directions_below_1e-3"] == sum(v/values[-1] < 1e-3 for v in values)
            supported.append(row)
        assert len(supported) == summary["supported_frames"] == 63
        for model in ["gicp", "p2p"]:
            ratios = sorted(r[model+"_geometry"]["min_max_ratio"] for r in supported)
            actual = summary["geometry"][model]
            assert actual["min_ratio"] == ratios[0] and actual["median_ratio"] == ratios[len(ratios)//2]
            assert actual["weak_frames"] == sum(r[model+"_geometry"]["weak_directions_below_1e-3"] > 0 for r in supported)
        assert summary["max_normal_matrix_error"] == max(r["normal_matrix_max_error"] for r in rows)
        assert summary["max_rhs_error"] == max(r["rhs_max_error"] for r in rows)
        assert summary["max_cost_error"] == max(r["cost_abs_error"] for r in rows)
    total = sum(s["retained_pairs"] for s in summaries)
    assert total == sum(s["original_pairs"] for s in summaries) == 118040
    build = read("native-build-inputs.json")
    assert len(build["sources_sha256"] | build["python_package_sources_sha256"] | build["additional_config_sha256"]) == build["unique_paths"] == 109
    return dict(public_json_files_verified=9, cases=4, frames=256, retained_pairs=total,
                scope="Published bytes, controller identities, schedule and reported fixed-match geometry only; no independent real-data rerun, calibration or physical accuracy result.")


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
