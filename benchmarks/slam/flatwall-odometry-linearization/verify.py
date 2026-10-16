#!/usr/bin/env python3
"""Check published bytes and reported geometry; does not rerun map replay."""
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
        return json.loads((packet / name).read_text(encoding="utf-8"))
    manifest = read("packet-files.json")
    assert len(manifest["json_files"]) == 8
    for name, expected in manifest["json_files"].items():
        assert artifact(packet / name) == expected, f"changed {Path(name).name}"
    assert manifest["verifier"] == artifact(Path(__file__)), "changed verifier"
    scope = read("slice-protocol.json")
    protocol, receipt = read("audit/protocol.json"), read("audit/receipt.json")
    assert receipt["protocol"] == artifact(packet / "audit/protocol.json"), "changed audit protocol"
    assert receipt["inputs"] == protocol["inputs"]
    assert protocol["slice"] == scope
    scope_artifact = artifact(packet / "slice-protocol.json")
    # Public naming differs from the original private protocol basename.
    scope_artifact["name"] = "protocol.json"
    assert protocol["inputs"]["slice_protocol"] == scope_artifact
    assert protocol["inputs"]["controller"] == artifact(ROOT / "scripts/audit_flatwall_odometry_linearization.py"), "changed controller"
    assert receipt["verified_frames"] == 256 and receipt["pose_sequences_compared"] == 8
    names = ["raw-legacy", "raw-fine", "rotation-legacy", "rotation-fine"]
    assert protocol["cases"] == receipt["cases"] == names
    assert protocol["scan_indices"] == list(range(1, 65))
    assert protocol["weak_ratio"] == 1e-3
    for name in ["summary.json", "frames.json"]:
        assert receipt["outputs"][name] == artifact(packet / "audit" / name), f"changed {name}"
    summaries, frames = read("audit/summary.json"), read("audit/frames.json")
    assert [s["case"] for s in summaries] == names
    assert len(frames) == 256
    for summary in summaries:
        rows = [row for row in frames if row["case"] == summary["case"]]
        assert len(rows) == summary["frames"] == 64
        assert [r["row"] for r in rows] == list(range(64))
        assert [r["scan_index"] for r in rows] == list(range(1, 65))
        assert all(b["reference_ns"] > a["reference_ns"] for a, b in zip(rows, rows[1:]))
        assert summary["capture_modes_exact_old_pose_bits_and_reports"]
        supported = []
        for row in rows:
            assert row["source_points"] >= row["matches"] >= 0
            assert math.isfinite(row["weight_sum"]) and row["weight_sum"] >= 0
            assert max(row["source_world_max_error_m"], row["insertion_max_error_m"]) <= 2e-8
            assert row["normal_matrix_max_error"] >= 0 and row["rhs_max_error"] >= 0
            geometry = row["geometry"]
            if geometry is None:
                assert row["weight_sum"] == 0 or row["radius_m"] in [None, 0.]
                continue
            assert row["weight_sum"] > 0 and row["radius_m"] > 0
            values = geometry["eigenvalues"]
            assert len(values) == 6 and all(math.isfinite(v) for v in values)
            assert sorted(values) == values and values[-1] > 0
            assert math.isclose(geometry["min_max_ratio"], values[0]/values[-1], abs_tol=1e-15)
            assert geometry["weak_directions_below_1e-3"] == sum(v/values[-1] < 1e-3 for v in values)
            supported.append(geometry)
        assert len(supported) == summary["supported_frames"] == 63
        assert summary["weak_frame_count"] == sum(g["weak_directions_below_1e-3"] > 0 for g in supported)
        ratios = sorted(g["min_max_ratio"] for g in supported)
        assert summary["min_ratio"] == ratios[0] and summary["median_ratio"] == ratios[len(ratios)//2]
        assert summary["max_normal_matrix_error"] == max(r["normal_matrix_max_error"] for r in rows)
        assert summary["max_rhs_error"] == max(r["rhs_max_error"] for r in rows)
    build = read("native-build-inputs.json")
    sources = build["sources_sha256"] | build["python_package_sources_sha256"] | build["additional_config_sha256"]
    assert len(sources) == build["unique_paths"] == 107
    return dict(public_json_files_verified=9, cases=4, frames=256, supported_frames=252,
                scope="Published bytes, controller, case schedule and reported conditional geometry only; no independent real-data rerun or physical accuracy claim.")


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
