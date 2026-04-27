"""Loop-closure before/after reporting helpers.

This is a lightweight integration layer: it compares "before" vs "after" artifacts
against the same reference to quantify whether a manual loop-closure pass improved
map and/or trajectory quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ca.evaluate import evaluate as evaluate_map
from ca.trajectory import evaluate_trajectory


@dataclass(slots=True)
class LoopClosureGate:
    """Optional gates. Values are interpreted as:

    - Improvements: higher is better
    - Error metrics: lower is better
    """

    min_auc_gain: float | None = None
    max_after_chamfer: float | None = None


def _best_f1(eval_result: dict) -> dict:
    return max(eval_result["f1_scores"], key=lambda s: s["f1"])


def _apply_gate(report: dict, gate: LoopClosureGate) -> dict | None:
    if gate.min_auc_gain is None and gate.max_after_chamfer is None:
        return None
    reasons: list[str] = []
    passed = True

    if gate.min_auc_gain is not None:
        gain = report["map"]["delta"]["auc"]
        if gain < gate.min_auc_gain:
            passed = False
            reasons.append(f"Map AUC gain {gain:.6f} < min {gate.min_auc_gain:.6f}")

    if gate.max_after_chamfer is not None:
        chamfer = report["map"]["after"]["chamfer_distance"]
        if chamfer > gate.max_after_chamfer:
            passed = False
            reasons.append(f"After chamfer {chamfer:.6f} > max {gate.max_after_chamfer:.6f}")

    return {
        "passed": passed,
        "reasons": reasons,
        "min_auc_gain": gate.min_auc_gain,
        "max_after_chamfer": gate.max_after_chamfer,
    }


def build_loop_closure_report(
    *,
    before_map: str,
    after_map: str,
    reference_map: str,
    thresholds: list[float] | None = None,
    before_trajectory: str | None = None,
    after_trajectory: str | None = None,
    reference_trajectory: str | None = None,
    trajectory_max_time_delta: float = 0.05,
    trajectory_align_origin: bool = False,
    trajectory_align_rigid: bool = False,
    gate: LoopClosureGate | None = None,
) -> dict[str, Any]:
    """Build a before/after report for manual loop-closure outcomes."""
    before_eval = evaluate_map(before_map, reference_map, thresholds=thresholds)
    after_eval = evaluate_map(after_map, reference_map, thresholds=thresholds)

    before_best = _best_f1(before_eval)
    after_best = _best_f1(after_eval)

    report: dict[str, Any] = {
        "map": {
            "reference": reference_map,
            "before": {
                "path": before_map,
                "chamfer_distance": before_eval["chamfer_distance"],
                "hausdorff_distance": before_eval["hausdorff_distance"],
                "auc": before_eval["auc"],
                "best_f1": before_best,
            },
            "after": {
                "path": after_map,
                "chamfer_distance": after_eval["chamfer_distance"],
                "hausdorff_distance": after_eval["hausdorff_distance"],
                "auc": after_eval["auc"],
                "best_f1": after_best,
            },
            "delta": {
                "auc": after_eval["auc"] - before_eval["auc"],
                "chamfer_distance": after_eval["chamfer_distance"] - before_eval["chamfer_distance"],
                "hausdorff_distance": after_eval["hausdorff_distance"] - before_eval["hausdorff_distance"],
                "best_f1": after_best["f1"] - before_best["f1"],
            },
        }
    }

    if (
        before_trajectory is not None
        and after_trajectory is not None
        and reference_trajectory is not None
    ):
        before_traj = evaluate_trajectory(
            before_trajectory,
            reference_trajectory,
            max_time_delta=trajectory_max_time_delta,
            align_origin=trajectory_align_origin,
            align_rigid=trajectory_align_rigid,
        )
        after_traj = evaluate_trajectory(
            after_trajectory,
            reference_trajectory,
            max_time_delta=trajectory_max_time_delta,
            align_origin=trajectory_align_origin,
            align_rigid=trajectory_align_rigid,
        )
        report["trajectory"] = {
            "reference": reference_trajectory,
            "before": {
                "path": before_trajectory,
                "ate_rmse": before_traj["ate"]["rmse"],
                "rpe_rmse": before_traj["rpe_translation"]["rmse"],
                "endpoint_drift": before_traj["drift"]["endpoint_drift_m"],
                "coverage": before_traj["matching"]["coverage_ratio"],
            },
            "after": {
                "path": after_trajectory,
                "ate_rmse": after_traj["ate"]["rmse"],
                "rpe_rmse": after_traj["rpe_translation"]["rmse"],
                "endpoint_drift": after_traj["drift"]["endpoint_drift_m"],
                "coverage": after_traj["matching"]["coverage_ratio"],
            },
            "delta": {
                "ate_rmse": after_traj["ate"]["rmse"] - before_traj["ate"]["rmse"],
                "rpe_rmse": after_traj["rpe_translation"]["rmse"] - before_traj["rpe_translation"]["rmse"],
                "endpoint_drift": after_traj["drift"]["endpoint_drift_m"] - before_traj["drift"]["endpoint_drift_m"],
                "coverage": after_traj["matching"]["coverage_ratio"] - before_traj["matching"]["coverage_ratio"],
            },
        }

    report["quality_gate"] = _apply_gate(report, gate or LoopClosureGate())
    return report

