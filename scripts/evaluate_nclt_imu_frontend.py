#!/usr/bin/env python3
"""Score every frozen frontend pose after estimation, including all fallbacks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_nclt_imu_pilot as scoring
import pilot_nclt_imu_window as pilot
from replay_nclt_imu_frontend import SCHEMA


def evaluate(replay: Path, ground_truth: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    if output.resolve().is_relative_to(replay.resolve()):
        raise ValueError("write evaluation outside frozen frontend replay")
    receipt_path = replay / "receipt.json"
    frozen = pilot.artifact(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["schema"] != SCHEMA:
        raise ValueError("expected a frozen NCLT frontend replay")

    def check_outputs():
        for name, expected in receipt["outputs"].items():
            if Path(name).name != name or pilot.artifact(replay / name) != expected:
                raise ValueError("frontend replay output hash mismatch")

    check_outputs()
    truth_input = pilot.artifact(ground_truth)
    with np.load(replay / "trajectory.npz", allow_pickle=False) as data:
        times = data["timestamps_ns"].copy()
        saved = {key: data[key].copy() for key in data.files if key != "timestamps_ns"}
    pilot.strictly_increasing(times)
    truth = scoring.reference_at(ground_truth, times)
    metrics, errors = {}, {}
    for arm in ["baseline", "assisted"]:
        coverage = receipt["arms"][arm]["coverage"]
        status = saved[f"{arm}_status"]
        if (coverage["scans"] != len(times) or status.shape != times.shape or status[0] != "lidar_start"
                or set(status[1:]) - {"lidar_gap", "lidar_bootstrap", "imu_prediction"}):
            raise ValueError("invalid frontend coverage/status")
        gap = np.array([i in set(receipt["arms"][arm]["gap_edges"]) for i in range(len(times) - 1)])
        prediction = status[1:] == "imu_prediction"
        np.testing.assert_array_equal(status[1:] == "lidar_gap", gap)
        assert int(prediction.sum()) == coverage["prediction_available"]
        assert int(gap.sum()) == coverage["gap_fallbacks"]
        assert int((status == "lidar_bootstrap").sum()) == coverage["bootstrap_fallbacks"]
        assert int(saved[f"{arm}_icp_iterations"].sum()) == coverage["icp_iterations"]
        # Independent maps evolve after different initial guesses. Even the gap
        # outputs can differ, so every fallback transition remains in the score.
        block = scoring.pose_errors(saved[f"{arm}_body_poses"], truth)
        metrics[arm] = {"all": scoring.summarize(block)}
        for group, mask in [("prediction_edges", prediction), ("fallback_edges", ~prediction), ("gap_edges", gap)]:
            selected = {k: v[mask] for k, v in block.items() if k.startswith("rpe_")}
            metrics[arm][group] = scoring.summarize(selected) if np.any(mask) else None
        errors.update({f"{arm}_{k}": v for k, v in block.items()})
    if pilot.artifact(receipt_path) != frozen or pilot.artifact(ground_truth) != truth_input:
        raise ValueError("frontend evaluation input changed")
    check_outputs()
    result = {"schema": "cloudanalyzer.nclt_imu_frontend_evaluation.v1",
              "inputs": {"replay_receipt": frozen, "ground_truth": truth_input,
                         "evaluator": pilot.artifact(Path(__file__)), "scoring_module": pilot.artifact(Path(scoring.__file__))},
              "coverage": {k: v["coverage"] for k, v in receipt["arms"].items()}, "metrics": metrics,
              "primary": "Full chronological ICP frontend trajectory, every pose and edge including all fallbacks",
              "ate_policy": "One global rigid position fit per full session, no scale",
              "scope": "Fixed diagnostic weights, same existing NCLT sample cohort; no held-out accuracy, calibrated covariance, real-time speed or default adoption claim"}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "errors.npz", **errors)
    result["outputs"] = {"errors.npz": pilot.artifact(output / "errors.npz")}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.replay, args.ground_truth, args.out)["metrics"], indent=2))


if __name__ == "__main__":
    main()
