#!/usr/bin/env python3
"""Recompute the frozen pilot scores from public poses and original body rows.

No estimator or production evaluation helper is called. Relative errors use
explicit homogeneous SE3 products; reference rotations use SciPy directly.
"""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest_matches(path, captured):
    # The frozen receipt records the actual Windows bytes. Git can check out
    # the same source/CSV with LF on Linux. Permit only that newline conversion;
    # original recording/native binary artifact hashes are never normalized.
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return captured in {hashlib.sha256(b).hexdigest() for b in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def truth_at(path, times):
    rows = list(csv.reader(path.open(encoding="utf-8", newline="")))
    stamps = [int(Decimal(row[0]) * 1000) for row in rows]
    poses = []
    axes = np.diag([1., -1., -1.])
    for row in rows:
        pose = np.eye(4)
        values = np.array(row[1:], dtype=float)
        pose[:3, :3] = axes @ Rotation.from_euler("xyz", values[3:]).as_matrix() @ axes
        pose[:3, 3] = axes @ values[:3]
        poses.append(pose)
    result = []
    for t in times:
        right = int(np.searchsorted(stamps, t))
        if right < len(stamps) and stamps[right] == t:
            result.append(poses[right].copy())
            continue
        assert 0 < right < len(stamps), "reference extrapolation"
        a, b = stamps[right - 1:right + 1]
        assert max(t - a, b - t) * 1e-9 <= .05, "reference bracket gap"
        alpha = (t - a) / (b - a)
        pose = np.eye(4)
        pose[:3, 3] = (1 - alpha) * poses[right - 1][:3, 3] + alpha * poses[right][:3, 3]
        pair = Rotation.from_matrix(np.array([poses[right - 1][:3, :3], poses[right][:3, :3]]))
        pose[:3, :3] = Slerp([0., 1.], pair)(alpha).as_matrix()
        result.append(pose)
    return np.array(result)


def errors(estimated, reference):
    relative = [np.linalg.inv(np.linalg.inv(a) @ b) @ (np.linalg.inv(c) @ d)
                for a, b, c, d in zip(reference[:-1], reference[1:], estimated[:-1], estimated[1:], strict=True)]
    translation = [np.linalg.norm(p[:3, 3]) for p in relative]
    angles = [np.degrees(np.arccos(np.clip((np.trace(p[:3, :3]) - 1) / 2, -1, 1))) for p in relative]
    estimated_positions, reference_positions = estimated[:, :3, 3], reference[:, :3, 3]
    ep = estimated_positions - estimated_positions.mean(axis=0)
    rp = reference_positions - reference_positions.mean(axis=0)
    fit, _ = Rotation.align_vectors(rp, ep)
    ate = np.linalg.norm(fit.apply(ep) - rp, axis=1)
    end = np.linalg.inv(np.linalg.inv(reference[0]) @ reference[-1]) @ (np.linalg.inv(estimated[0]) @ estimated[-1])
    return {"ate_m": ate, "rpe_translation_m": np.array(translation),
            "rpe_rotation_deg": np.array(angles), "endpoint_translation_m": np.array([np.linalg.norm(end[:3, 3])])}


def check_stats(values, expected):
    assert len(values) == expected["count"]
    actual = [np.sqrt(np.mean(values**2)), np.mean(values), np.max(values)]
    np.testing.assert_allclose(actual, [expected[k] for k in ["rmse", "mean", "max"]], atol=1e-8, rtol=1e-9)


def verify():
    packet = json.loads((PACKET / "results.json").read_text(encoding="utf-8"))
    for relative, digest in packet["files_sha256"].items():
        assert text_digest_matches(ROOT / relative, digest), f"changed input: {relative}"
    verified = []
    for case in packet["cases"]:
        rows = list(csv.DictReader((PACKET / case["poses"]).open(encoding="utf-8", newline="")))
        groups = {}
        for row in rows:
            groups.setdefault(row["window"], []).append(row)
        assert len(rows) == case["coverage"]["eligible_nodes"]
        assert len(groups) == case["coverage"]["windows"]
        assert len({r["scan_index"] for r in rows}) == len(rows), "overlapping windows"
        pooled = {method: {} for method in ["lidar", "fused"]}
        for window in case["windows"]:
            group = groups[window["name"]]
            assert [int(r["scan_index"]) for r in group] == list(range(window["scan_start"], window["scan_stop"]))
            times = [int(r["timestamp_ns"]) for r in group]
            assert all(a < b for a, b in zip(times[:-1], times[1:]))
            truth = truth_at(ROOT / case["reference_rows"], times)
            for method in pooled:
                poses = np.tile(np.eye(4), (len(group), 1, 1))
                poses[:, :3, 3] = [[float(r[f"{method}_{k}"]) for k in ["x", "y", "z"]] for r in group]
                poses[:, :3, :3] = Rotation.from_quat([[float(r[f"{method}_{k}"]) for k in ["qx", "qy", "qz", "qw"]] for r in group]).as_matrix()
                for key, values in errors(poses, truth).items():
                    check_stats(values, window["metrics"][method][key])
                    pooled[method].setdefault(key, []).append(values)
        for method, blocks in pooled.items():
            for key, values in blocks.items():
                check_stats(np.concatenate(values), case["metrics"][method][key])
        assert case["metrics"]["lidar"]["rpe_translation_m"]["count"] == case["coverage"]["eligible_edges"]
        verified.append({"session": case["session"], "windows": len(groups), "nodes": len(rows),
                         "edges": case["coverage"]["eligible_edges"]})
    return {"verified": verified, "scope": "Frozen estimate/reference score verification; no estimator rerun"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
