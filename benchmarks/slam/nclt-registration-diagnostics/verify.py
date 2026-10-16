#!/usr/bin/env python3
"""Verify frozen report aggregates, not the estimator or pose parity itself."""
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median


def checked_text(path, expected):
    data = path.read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    assert expected in {hashlib.sha256(v).hexdigest() for v in [lf, lf.replace(b"\n", b"\r\n")]}, path
    return lf.decode("utf-8")


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    index = .95 * (len(ordered) - 1)
    low, high = math.floor(index), math.ceil(index)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {"count": len(values), "min": ordered[0], "median": median(ordered), "p95": p95, "max": ordered[-1]}


def summarize(rows):
    return {"scans": len(rows), "terminations": dict(Counter(r["termination"] for r in rows)),
            "iterations": sum(r["iterations"] for r in rows),
            "residual_reports": sum(r["matched_points"] is not None for r in rows),
            **{key: distribution([r[key] for r in rows if r[key] is not None])
               for key in ["matched_fraction", "rmse_m", "max_residual_m"]}}


def equal(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            equal(actual[key], expected[key])
    elif isinstance(actual, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def verify(packet=None):
    packet = Path(__file__).resolve().parent if packet is None else Path(packet)
    result = json.loads((packet / "results.json").read_text(encoding="utf-8"))
    assert result["schema"] == "cloudanalyzer.nclt_registration_diagnostics_packet.v1"
    assert [s["session"] for s in result["sessions"]] == ["2012-04-29", "2012-06-15"]
    total = 0
    for session in result["sessions"]:
        label = "april" if session["session"] == "2012-04-29" else "june"
        assert session["csv"] == f"{label}-registrations.csv"
        previous = list(csv.DictReader(io.StringIO(checked_text(
            packet.parent / "nclt-imu-frontend" / f"{label}-poses.csv", session["frontend_csv_sha256"]))))
        raw = list(csv.DictReader(io.StringIO(checked_text(packet / session["csv"], session["csv_sha256"]))))
        assert len(raw) == 2 * len(previous)
        for arm_index, arm in enumerate(["baseline", "assisted"]):
            rows = []
            for i, source in enumerate(raw[arm_index * len(previous):(arm_index + 1) * len(previous)]):
                row = source.copy()
                for key in ["scan_index", "timestamp_ns", "iterations", "input_points", "registration_points"]:
                    row[key] = int(row[key])
                row["threshold_m"] = float(row["threshold_m"])
                for key in ["matched_fraction", "rmse_m", "max_residual_m"]:
                    row[key] = float(row[key]) if row[key] else None
                row["matched_points"] = int(row["matched_points"]) if row["matched_points"] else None
                assert row["arm"] == arm
                assert row["scan_index"] == i == int(previous[i]["scan_index"])
                assert row["timestamp_ns"] == int(previous[i]["timestamp_ns"])
                assert row["frontend_status"] == previous[i]["status"]
                assert row["iterations"] == int(previous[i][f"{arm}_icp_iterations"])
                assert row["input_points"] >= row["registration_points"] >= 0
                assert math.isfinite(row["threshold_m"]) and row["threshold_m"] > 0
                assert row["termination"] in {
                    "empty_map", "insufficient_points", "converged", "iteration_limit", "singular_system",
                    "non_finite_system", "non_finite_update", "unfinished"}
                if row["termination"] in {"empty_map", "insufficient_points"}:
                    assert row["iterations"] == 0
                    assert all(row[key] is None for key in ["matched_points", "matched_fraction", "rmse_m", "max_residual_m"])
                else:
                    matched, count = row["matched_points"], row["registration_points"]
                    assert count >= 10 and matched is not None and 0 <= matched <= count
                    equal(row["matched_fraction"], matched / count)
                    if matched:
                        assert all(math.isfinite(row[key]) for key in ["rmse_m", "max_residual_m"])
                        assert 0 <= row["rmse_m"] <= row["max_residual_m"] + 1e-14
                        assert row["max_residual_m"] <= row["threshold_m"] + 1e-12
                    else:
                        assert row["rmse_m"] is row["max_residual_m"] is None
                rows.append(row)
            expected = session["arms"][arm]
            equal(summarize(rows), expected["all"])
            equal({status: summarize([r for r in rows if r["frontend_status"] == status])
                   for status in {r["frontend_status"] for r in rows}}, expected["by_frontend_status"])
            assert expected["pose_bits_unchanged"] == expected["iteration_counts_unchanged"] == len(rows)
            total += len(rows)
    assert total == 948
    return total


if __name__ == "__main__":
    print(f"Verified {verify()} frozen registration reports and their aggregates.")
