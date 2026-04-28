"""Loop-closure before/after reporting helpers.

This is a lightweight integration layer: it compares "before" vs "after" artifacts
against the same reference to quantify whether a manual loop-closure pass improved
map and/or trajectory quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ca.evaluate import evaluate as evaluate_map
from ca.posegraph import validate_posegraph_session
from ca.trajectory import evaluate_trajectory


@dataclass(slots=True)
class LoopClosureGate:
    """Optional gates. Values are interpreted as:

    - Improvements: higher is better
    - Error metrics: lower is better
    """

    min_auc_gain: float | None = None
    max_after_chamfer: float | None = None
    # Trajectory gates (lower is better; gain means improvement, so higher is better).
    min_ate_gain: float | None = None
    max_after_ate: float | None = None
    require_posegraph_ok: bool = False


def _best_f1(eval_result: dict) -> dict:
    return max(eval_result["f1_scores"], key=lambda s: s["f1"])


def _apply_gate(report: dict, gate: LoopClosureGate) -> dict | None:
    if all(
        v is None
        for v in (
            gate.min_auc_gain,
            gate.max_after_chamfer,
            gate.min_ate_gain,
            gate.max_after_ate,
        )
    ) and not gate.require_posegraph_ok:
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

    if gate.min_ate_gain is not None:
        if "trajectory" not in report:
            passed = False
            reasons.append("Trajectory ATE gate configured but trajectory inputs are missing")
        else:
            gain = report["trajectory"]["before"]["ate_rmse"] - report["trajectory"]["after"]["ate_rmse"]
            if gain < gate.min_ate_gain:
                passed = False
                reasons.append(f"Trajectory ATE gain {gain:.6f} < min {gate.min_ate_gain:.6f}")

    if gate.max_after_ate is not None:
        if "trajectory" not in report:
            passed = False
            reasons.append("Trajectory max ATE gate configured but trajectory inputs are missing")
        else:
            ate = report["trajectory"]["after"]["ate_rmse"]
            if ate > gate.max_after_ate:
                passed = False
                reasons.append(f"After trajectory ATE {ate:.6f} > max {gate.max_after_ate:.6f}")

    if gate.require_posegraph_ok:
        sessions = report.get("posegraph_session")
        if not isinstance(sessions, dict):
            passed = False
            reasons.append("Posegraph gate configured but posegraph session inputs are missing")
        else:
            checked = 0
            for label in ("before", "after"):
                session = sessions.get(label)
                if session is None:
                    continue
                checked += 1
                summary = session.get("summary") if isinstance(session, dict) else None
                ok = summary.get("ok") if isinstance(summary, dict) else False
                if ok is not True:
                    passed = False
                    errors = summary.get("errors", []) if isinstance(summary, dict) else []
                    detail = f": {'; '.join(errors)}" if errors else ""
                    reasons.append(f"{label.capitalize()} posegraph session is not ok{detail}")
            if checked == 0:
                passed = False
                reasons.append("Posegraph gate configured but no posegraph sessions were validated")

    return {
        "passed": passed,
        "reasons": reasons,
        "min_auc_gain": gate.min_auc_gain,
        "max_after_chamfer": gate.max_after_chamfer,
        "min_ate_gain": gate.min_ate_gain,
        "max_after_ate": gate.max_after_ate,
        "require_posegraph_ok": gate.require_posegraph_ok,
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
    before_g2o: str | None = None,
    after_g2o: str | None = None,
    before_tum: str | None = None,
    after_tum: str | None = None,
    before_key_point_frame_dir: str | None = None,
    after_key_point_frame_dir: str | None = None,
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

    if before_g2o is not None or after_g2o is not None:
        report["posegraph_session"] = {
            "before": (
                validate_posegraph_session(
                    g2o_path=before_g2o,
                    tum_path=before_tum,
                    key_point_frame_dir=before_key_point_frame_dir,
                )
                if before_g2o is not None
                else None
            ),
            "after": (
                validate_posegraph_session(
                    g2o_path=after_g2o,
                    tum_path=after_tum,
                    key_point_frame_dir=after_key_point_frame_dir,
                )
                if after_g2o is not None
                else None
            ),
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
                "endpoint_drift": before_traj["drift"]["endpoint"],
                "coverage": before_traj["matching"]["coverage_ratio"],
            },
            "after": {
                "path": after_trajectory,
                "ate_rmse": after_traj["ate"]["rmse"],
                "rpe_rmse": after_traj["rpe_translation"]["rmse"],
                "endpoint_drift": after_traj["drift"]["endpoint"],
                "coverage": after_traj["matching"]["coverage_ratio"],
            },
            "delta": {
                "ate_rmse": after_traj["ate"]["rmse"] - before_traj["ate"]["rmse"],
                "rpe_rmse": after_traj["rpe_translation"]["rmse"] - before_traj["rpe_translation"]["rmse"],
                "endpoint_drift": after_traj["drift"]["endpoint"] - before_traj["drift"]["endpoint"],
                "coverage": after_traj["matching"]["coverage_ratio"] - before_traj["matching"]["coverage_ratio"],
            },
        }

    report["quality_gate"] = _apply_gate(report, gate or LoopClosureGate())
    return report
