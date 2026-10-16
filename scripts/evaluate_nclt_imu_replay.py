#!/usr/bin/env python3
"""Evaluate all outputs of a frozen NCLT bounded-window replay, including gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_nclt_imu_pilot as scoring
import pilot_nclt_imu_window as pilot
from replay_nclt_imu_window import SCHEMA


def evaluate(replay: Path, ground_truth: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    if output.resolve().is_relative_to(replay.resolve()):
        raise ValueError("write evaluation outside frozen replay")
    receipt_path = replay / "receipt.json"
    frozen = pilot.artifact(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["schema"] != SCHEMA or "trajectory.npz" not in receipt["outputs"]:
        raise ValueError("expected a frozen NCLT replay")
    for name, expected in receipt["outputs"].items():
        if Path(name).name != name or pilot.artifact(replay / name) != expected:
            raise ValueError("replay output hash mismatch")
    truth_input = pilot.artifact(ground_truth)
    with np.load(replay / "trajectory.npz", allow_pickle=False) as data:
        times, lidar, replayed = data["timestamps_ns"], data["lidar_body_poses"], data["replay_body_poses"]
        status = data["status"].copy()
    if len(times) != receipt["coverage"]["scans"] or status[0] != "lidar_start" or set(status[1:]) - {"lidar_gap", "fused"}:
        raise ValueError("invalid replay coverage/status")
    gap_mask = np.array([i in set(receipt["gap_resets"]) for i in range(len(times) - 1)])
    np.testing.assert_array_equal(status[1:] == "lidar_gap", gap_mask)
    np.testing.assert_array_equal(replayed[1:][gap_mask], lidar[1:][gap_mask])
    truth = scoring.reference_at(ground_truth, times)
    metrics, errors = {}, {}
    for method, poses in [("lidar", lidar), ("replay", replayed)]:
        block = scoring.pose_errors(poses, truth)
        metrics[method] = {"all": scoring.summarize(block)}
        for name, mask in [("valid_imu_edges", ~gap_mask), ("gap_edges", gap_mask)]:
            selected = {k: v[mask] for k, v in block.items() if k.startswith("rpe_")}
            metrics[method][name] = scoring.summarize(selected) if np.any(mask) else None
        errors.update({f"{method}_{k}": v for k, v in block.items()})
    if pilot.artifact(receipt_path) != frozen or pilot.artifact(ground_truth) != truth_input:
        raise ValueError("evaluation input changed")
    if any(pilot.artifact(replay / name) != a for name, a in receipt["outputs"].items()):
        raise ValueError("replay output changed")
    result = {"schema": "cloudanalyzer.nclt_imu_replay_evaluation.v1",
              "inputs": {"replay_receipt": frozen, "ground_truth": truth_input,
                         "evaluator": pilot.artifact(Path(__file__)), "scoring_module": pilot.artifact(Path(scoring.__file__))},
              "coverage": receipt["coverage"], "metrics": metrics,
              "primary": "Full chronological latest-state trajectory, all scan edges including LiDAR fallback transitions",
              "ate_policy": "One global rigid position fit per full session, no scale",
              "scope": "Diagnostic pose/IMU replay; no calibrated covariance, frontend feedback, independent truth or real-time speed claim"}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "errors.npz", **errors, gap_mask=gap_mask)
    result["outputs"] = {"errors.npz": pilot.artifact(output / "errors.npz")}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.replay, args.ground_truth, args.out)["metrics"], indent=2))


if __name__ == "__main__":
    main()
