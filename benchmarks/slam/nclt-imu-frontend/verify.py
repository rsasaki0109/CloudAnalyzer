#!/usr/bin/env python3
"""Verify complete frontend trajectories with frozen independent SE3 math."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent
MATH = ROOT / "benchmarks/slam/nclt-imu-pilot/verify.py"


def verify():
    spec = importlib.util.spec_from_file_location("frozen_nclt_frontend_math", MATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    receipt = json.loads((PACKET / "results.json").read_text(encoding="utf-8"))
    assert receipt["adopted_as_default"] is False
    for name, digest in receipt["files_sha256"].items():
        assert helper.text_digest_matches(ROOT / name, digest), f"changed input: {name}"
    verified = []
    for case in receipt["cases"]:
        with (PACKET / case["poses"]).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        times = [int(row["timestamp_ns"]) for row in rows]
        assert all(a < b for a, b in zip(times[:-1], times[1:]))
        assert [int(r["scan_index"]) for r in rows] == list(range(len(rows)))
        assert rows[0]["status"] == "lidar_start"
        active, expected_marginals = [], []
        for i, row in enumerate(rows):
            status = row["status"]
            if status in {"lidar_start", "lidar_gap"}:
                assert (i == 0) == (status == "lidar_start")
                active = []
                assert int(row["active_states"]) == 0
                continue
            assert status == ("imu_prediction" if active else "lidar_bootstrap")
            if not active:
                active = [i - 1]
            if len(active) == receipt["max_states"]:
                dropped = active.pop(0)
                expected_marginals.append((i, times[dropped], times[active[0]]))
            active.append(i)
            assert int(row["active_states"]) == len(active)
        masks = {"prediction_edges": np.array([r["status"] == "imu_prediction" for r in rows[1:]]),
                 "gap_edges": np.array([r["status"] == "lidar_gap" for r in rows[1:]])}
        masks["fallback_edges"] = ~masks["prediction_edges"]
        truth = helper.truth_at(ROOT / case["reference_rows"], times)
        for arm in ["baseline", "assisted"]:
            coverage = case["coverage"][arm]
            assert coverage["scans"] == len(rows) and coverage["all_edges"] == len(rows) - 1
            assert coverage["prediction_available"] == int(masks["prediction_edges"].sum())
            assert coverage["prediction_used"] == (coverage["prediction_available"] if arm == "assisted" else 0)
            assert coverage["gap_fallbacks"] == int(masks["gap_edges"].sum())
            assert coverage["bootstrap_fallbacks"] == sum(r["status"] == "lidar_bootstrap" for r in rows)
            assert coverage["startup_fallbacks"] == 1
            assert coverage["icp_iterations"] == sum(int(r[f"{arm}_icp_iterations"]) for r in rows)
            assert max(int(r["active_states"]) for r in rows) == coverage["max_active_states"] <= receipt["max_states"]
            marginals = [(m["scan_index"], m["dropped_timestamp_ns"], m["retained_timestamp_ns"])
                         for m in case["marginalizations"][arm]]
            assert marginals == expected_marginals and len(marginals) == coverage["marginalizations"]
            updates = case["optimization"][arm]
            assert [u["scan_index"] for u in updates] == [i for i, r in enumerate(rows) if r["status"] in {"imu_prediction", "lidar_bootstrap"}]
            assert sum(not u["converged"] for u in updates) == coverage["nonconverged_updates"]
            poses = np.tile(np.eye(4), (len(rows), 1, 1))
            poses[:, :3, 3] = [[float(r[f"{arm}_{k}"]) for k in ["x", "y", "z"]] for r in rows]
            poses[:, :3, :3] = Rotation.from_quat([[float(r[f"{arm}_{k}"]) for k in ["qx", "qy", "qz", "qw"]] for r in rows]).as_matrix()
            errors = helper.errors(poses, truth)
            for name, values in errors.items():
                helper.check_stats(values, case["metrics"][arm]["all"][name])
            for group, mask in masks.items():
                if not np.any(mask):
                    assert case["metrics"][arm][group] is None
                    continue
                for name in ["rpe_translation_m", "rpe_rotation_deg"]:
                    helper.check_stats(errors[name][mask], case["metrics"][arm][group][name])
        verified.append({"session": case["session"], "poses": len(rows), "edges": len(rows) - 1,
                         "predictions": int(masks["prediction_edges"].sum()), "gaps": int(masks["gap_edges"].sum())})
    return {"verified": verified, "scope": "Every saved ICP frontend pose and transition; no estimator rerun"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
