#!/usr/bin/env python3
"""Recheck every full-replay pose/edge with frozen independent SE3 math."""
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
    spec = importlib.util.spec_from_file_location("frozen_nclt_reference_math", MATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    receipt = json.loads((PACKET / "results.json").read_text(encoding="utf-8"))
    for name, digest in receipt["files_sha256"].items():
        assert helper.text_digest_matches(ROOT / name, digest), f"changed input: {name}"
    verified = []
    for case in receipt["cases"]:
        with (PACKET / case["poses"]).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        times = [int(r["timestamp_ns"]) for r in rows]
        assert all(a < b for a, b in zip(times[:-1], times[1:]))
        assert [int(r["scan_index"]) for r in rows] == list(range(len(rows)))
        coverage = case["coverage"]
        assert len(rows) == coverage["scans"] and len(rows) - 1 == coverage["all_edges"]
        assert rows[0]["status"] == "lidar_start"
        assert all(r["status"] in {"fused", "lidar_gap"} for r in rows[1:])
        gap = np.array([r["status"] == "lidar_gap" for r in rows[1:]])
        assert int(gap.sum()) == coverage["gap_fallbacks"]
        assert int((~gap).sum()) == coverage["fused_updates"]
        active, expected_marginals = [], []
        for i, row in enumerate(rows):
            if row["status"] != "fused":
                active = []
                assert int(row["active_states"]) == 0
                continue
            if not active:
                active = [i - 1]
            if len(active) == receipt["max_states"]:
                dropped = active.pop(0)
                expected_marginals.append((i, times[dropped], times[active[0]]))
            active.append(i)
            assert int(row["active_states"]) == len(active)
        assert max(int(r["active_states"]) for r in rows) == coverage["max_active_states"] <= receipt["max_states"]
        actual_marginals = [(m["scan_index"], m["dropped_timestamp_ns"], m["retained_timestamp_ns"]) for m in case["marginalizations"]]
        assert actual_marginals == expected_marginals and len(actual_marginals) == coverage["marginalizations"]
        truth = helper.truth_at(ROOT / case["reference_rows"], times)
        poses_by_method = {}
        for method in ["lidar", "replay"]:
            poses = np.tile(np.eye(4), (len(rows), 1, 1))
            poses[:, :3, 3] = [[float(r[f"{method}_{k}"]) for k in ["x", "y", "z"]] for r in rows]
            poses[:, :3, :3] = Rotation.from_quat([[float(r[f"{method}_{k}"]) for k in ["qx", "qy", "qz", "qw"]] for r in rows]).as_matrix()
            poses_by_method[method] = poses
            errors = helper.errors(poses, truth)
            for name, values in errors.items():
                helper.check_stats(values, case["metrics"][method]["all"][name])
            for group, mask in [("valid_imu_edges", ~gap), ("gap_edges", gap)]:
                for name in ["rpe_translation_m", "rpe_rotation_deg"]:
                    helper.check_stats(errors[name][mask], case["metrics"][method][group][name])
        np.testing.assert_array_equal(poses_by_method["replay"][1:][gap], poses_by_method["lidar"][1:][gap])
        verified.append({"session": case["session"], "poses": len(rows), "edges": len(rows) - 1,
                         "fused_updates": coverage["fused_updates"], "gap_fallbacks": coverage["gap_fallbacks"],
                         "marginalizations": coverage["marginalizations"]})
    return {"verified": verified, "scope": "All saved latest-state outputs and fallback transitions; no estimator rerun"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
