"""State-estimation consistency evaluation from GT and pose covariance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

from ca.trajectory import _apply_alignment, load_trajectory


def load_covariance_trajectory(path: str) -> dict[str, Any]:
    """Load the explicit JSON covariance-trajectory interchange format.

    Schema: top-level ``metadata`` plus ``states``. Each state has
    ``timestamp``, ``position`` (3), and ``covariance`` (3x3). Six-dimensional
    pose covariance is reserved for a future orientation-aware revision.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("states"), list):
        raise ValueError("covariance trajectory must be JSON with a states list")
    metadata = raw.get("metadata", {})
    if metadata.get("covariance_frame") not in {"world", "estimate_world"}:
        raise ValueError("metadata.covariance_frame must be 'world' or 'estimate_world'")
    if metadata.get("error_convention", "estimated_minus_reference") != "estimated_minus_reference":
        raise ValueError("only estimated_minus_reference error_convention is supported")
    times, positions, covariances = [], [], []
    for index, state in enumerate(raw["states"]):
        try:
            times.append(float(state["timestamp"]))
            positions.append(np.asarray(state["position"], dtype=float))
            covariances.append(np.asarray(state["covariance"], dtype=float))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid covariance state at index {index}") from exc
    timestamps = np.asarray(times)
    positions_array = np.asarray(positions)
    covariance_array = np.asarray(covariances)
    if positions_array.shape != (len(times), 3) or covariance_array.shape != (len(times), 3, 3):
        raise ValueError("states require position[3] and covariance[3][3]")
    if len(times) < 1 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("state timestamps must be non-empty and strictly increasing")
    if not np.all(np.isfinite(positions_array)) or not np.all(np.isfinite(covariance_array)):
        raise ValueError("positions and covariances must be finite")
    return {"timestamps": timestamps, "positions": positions_array, "covariances": covariance_array, "metadata": metadata}


def _validate_covariance(covariance: np.ndarray, index: int, max_condition: float) -> None:
    scale = max(float(np.max(np.abs(covariance))), 1.0)
    if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-12 * scale):
        raise ValueError(f"covariance[{index}] is not symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0:
        raise ValueError(f"covariance[{index}] must be positive definite")
    condition = float(eigenvalues[-1] / eigenvalues[0])
    if not np.isfinite(condition) or condition > max_condition:
        raise ValueError(f"covariance[{index}] is ill-conditioned ({condition:.3g})")


def evaluate_uncertainty(
    estimated_covariance_path: str,
    reference_path: str,
    *,
    max_time_delta: float = 0.05,
    align_mode: str = "none",
    confidence: float = 0.95,
    max_condition: float = 1e12,
) -> dict[str, Any]:
    """Compute position NEES and chi-square coverage (DoF=3)."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if max_time_delta <= 0:
        raise ValueError("max_time_delta must be > 0")
    if align_mode not in {"none", "origin", "rigid"}:
        raise ValueError("align_mode must be none, origin, or rigid")
    estimate = load_covariance_trajectory(estimated_covariance_path)
    reference = load_trajectory(reference_path)
    reference_times = reference["timestamps"]
    matched_est, matched_ref, matched_cov, matched_times = [], [], [], []
    for time, position, covariance in zip(estimate["timestamps"], estimate["positions"], estimate["covariances"]):
        nearest = int(np.argmin(np.abs(reference_times - time)))
        if abs(float(reference_times[nearest] - time)) <= max_time_delta:
            matched_times.append(float(time))
            matched_est.append(position)
            matched_ref.append(reference["positions"][nearest])
            matched_cov.append(covariance)
    if not matched_est:
        raise ValueError("no covariance states matched the reference trajectory")
    est = np.asarray(matched_est)
    ref = np.asarray(matched_ref)
    aligned, alignment, translation, rotation = _apply_alignment(
        est, ref, align_mode == "origin", align_mode == "rigid"
    )
    covariances = np.asarray([rotation @ cov @ rotation.T for cov in matched_cov])
    nees: list[float] = []
    for index, (error, covariance) in enumerate(zip(aligned - ref, covariances)):
        _validate_covariance(covariance, index, max_condition)
        # Cholesky solve preserves the positive-definite contract; no pinv.
        factor = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(factor, error)
        nees.append(float(whitened @ whitened))
    values = np.asarray(nees)
    dof = 3
    threshold = float(chi2.ppf(confidence, dof))
    n = len(values)
    interpretation = "chi_square_descriptive" if alignment == "none" else "aligned_proxy"
    return {
        "mean_position_nees": float(values.mean()),
        "normalized_mean_position_nees": float(values.mean() / dof),
        "coverage_95": float(np.mean(values <= threshold)) if confidence == 0.95 else None,
        "coverage": float(np.mean(values <= threshold)),
        "confidence": confidence,
        "chi_square_threshold": threshold,
        "num_matched_states": n,
        "dof": dof,
        "statistical_interpretation": interpretation,
        "assumptions": [
            "reference trajectory is treated as exact",
            "per-state chi-square coverage is descriptive; temporal errors may be correlated",
            "aligned_proxy fits alignment on the evaluated samples and is not a formal consistency test",
            "NIS is unavailable without innovations and innovation covariance",
        ],
        "alignment": {"mode": alignment, "translation": translation.tolist(), "rotation": rotation.tolist()},
        "nees": values.tolist(),
        "metadata": estimate["metadata"],
    }


__all__ = ["evaluate_uncertainty", "load_covariance_trajectory"]
