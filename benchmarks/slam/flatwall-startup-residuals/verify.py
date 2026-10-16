#!/usr/bin/env python3
"""Verify the public packet's bytes, schedule and reported algebraic summaries.

This does not rerun IMU integration or establish physical calibration/accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent


def artifact(path):
    data = path.read_bytes()
    return dict(
        name=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
    )


def verify(packet=None):
    packet = PACKET if packet is None else Path(packet)
    read = lambda name: json.loads((packet / name).read_text(encoding="utf-8"))  # noqa: E731
    scope = read("slice-protocol.json")
    protocol, receipt = read("audit/protocol.json"), read("audit/receipt.json")
    assert receipt["protocol"] == artifact(packet / "audit/protocol.json"), (
        "changed audit protocol"
    )
    assert receipt["inputs"] == protocol["inputs"]
    assert protocol["parameters"] == scope["parameters"]
    assert protocol["reference_strides"] == scope["reference_strides"] == [1, 3, 7]
    assert (
        protocol["clock_offsets_ns"]
        == scope["clock_offsets_ns"]
        == list(range(-50_000_000, 50_000_001, 5_000_000))
    )
    controller = ROOT / "scripts/audit_flatwall_startup_residuals.py"
    assert protocol["inputs"]["controller"] == artifact(controller), (
        "changed controller"
    )
    for name in ["sensor.json", "baseline-summary.json", "sensitivity.json"]:
        assert receipt["outputs"][name] == artifact(packet / "audit" / name), (
            f"changed {name}"
        )
    results, baselines = (
        read("audit/sensitivity.json"),
        read("audit/baseline-summary.json"),
    )
    expected = {
        (name, stride, 0)
        for name in ["raw-legacy", "raw-fine", "rotation-legacy", "rotation-fine"]
        for stride in [1, 3, 7]
    }
    expected |= {
        (name, 1, offset)
        for name in ["raw-legacy", "raw-fine"]
        for offset in protocol["clock_offsets_ns"]
    }
    actual = [(r["source"], r["stride"], r["offset_ns"]) for r in results]
    assert (
        len(results)
        == receipt["cases"]
        == len(actual)
        == len(set(actual))
        == len(expected)
        == 52
    )
    assert set(actual) == expected, "changed case schedule"
    assert baselines == [r for r in results if r["stride"] == 1 and r["offset_ns"] == 0]
    assert len(baselines) == 4
    for result in results:
        assert result["reference_count"] == 63 // result["stride"] + 1
        count = result["reference_count"] - 2
        energy = result["decomposition_energy"]
        assert all(math.isfinite(v) for v in energy.values())
        assert all(energy[key] >= 0 for key in ["pose", "imu", "total"])
        assert math.isclose(
            energy["total"],
            energy["pose"] + energy["imu"] + energy["cross"],
            rel_tol=1e-10,
            abs_tol=2e-12,
        )
        for key, rms in [
            ("total", result["diagnostics"]["velocity_rms"]),
            ("pose", result["pose_term_rms"]),
            ("imu", result["imu_term_rms"]),
        ]:
            assert math.isclose(
                energy[key], rms**2 * count, rel_tol=1e-10, abs_tol=2e-12
            )
        assert result["decomposition_max_sum_error"] <= 2e-12
        limits, diag = protocol["parameters"], result["diagnostics"]
        failed = [
            key
            for key in [
                "gravity_error",
                "rotation_rms",
                "velocity_rms",
                "gyro_bias_change",
            ]
            if diag["max_gyro_bias_change" if key == "gyro_bias_change" else key]
            > limits["max_" + key]
        ]
        if diag["gyro_singular_ratio"] < limits["min_singular_ratio"]:
            failed.append("gyro_rank")
        if not diag["optimizer_success"]:
            failed.append("optimizer")
        assert result["failed_limits"] == failed
        assert result["accepted"] == result["native"]["accepted"] == (not failed)
        for field, rows, rms in [
            ("rotation_distribution", count + 1, diag["rotation_rms"]),
            ("velocity_distribution", count, diag["velocity_rms"]),
        ]:
            values = result[field]
            assert math.isclose(values["rms"], rms, rel_tol=1e-12, abs_tol=1e-12)
            assert (
                0
                <= values["top1_energy_share"]
                <= values["top5_energy_share"]
                <= 1 + 1e-12
            )
            assert 0 <= values["maximum_row"] < rows
            assert values["top_rows"][0] == values["maximum_row"]
            assert all(0 <= row < rows for row in values["top_rows"])
    assert not any(r["accepted"] for r in results)
    assert (
        receipt["baseline_values_match_previous"]
        and receipt["all_native_acceptance_matches"]
    )
    assert receipt["no_state_adopted"] and receipt["no_reference_poses_used"]
    return dict(
        public_json_files_verified=6,
        cases=52,
        accepted_cases=0,
        scope="Published bytes, controller, schedule, algebraic summaries and reported acceptance only; no real-data refit or physical accuracy result.",
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
