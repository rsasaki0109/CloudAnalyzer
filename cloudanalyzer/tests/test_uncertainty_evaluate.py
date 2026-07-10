"""Tests for GT-based NEES uncertainty consistency evaluation."""

import json
from pathlib import Path

import pytest
import numpy as np
from typer.testing import CliRunner
from cloudanalyzer_cli.main import app

from ca.core.uncertainty_evaluate import evaluate_uncertainty, load_covariance_trajectory
from ca.core.checks import load_check_suite, run_check_suite
from ca.pr_comment import build_pr_comment


def _write_case(tmp_path: Path, covariance: list[list[float]] | None = None) -> tuple[Path, Path]:
    estimate = tmp_path / "estimated.json"
    reference = tmp_path / "reference.csv"
    covariance = covariance or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    states = [
        {"timestamp": float(i), "position": [float(i) + 1, 0, 0], "covariance": covariance}
        for i in range(10)
    ]
    estimate.write_text(json.dumps({
        "metadata": {"covariance_frame": "estimate_world", "error_convention": "estimated_minus_reference"},
        "states": states,
    }), encoding="utf-8")
    reference.write_text("timestamp,x,y,z\n" + "\n".join(f"{i},{i},0,0" for i in range(10)), encoding="utf-8")
    return estimate, reference


def test_position_nees_and_chi_square_coverage(tmp_path: Path) -> None:
    estimate, reference = _write_case(tmp_path)
    result = evaluate_uncertainty(str(estimate), str(reference))
    assert result["mean_position_nees"] == pytest.approx(1.0)
    assert result["normalized_mean_position_nees"] == pytest.approx(1 / 3)
    assert result["coverage_95"] == pytest.approx(1.0)
    assert result["dof"] == 3
    assert result["statistical_interpretation"] == "chi_square_descriptive"


def test_origin_alignment_removes_constant_offset(tmp_path: Path) -> None:
    estimate, reference = _write_case(tmp_path)
    result = evaluate_uncertainty(str(estimate), str(reference), align_mode="origin")
    assert result["mean_position_nees"] == pytest.approx(0.0)
    assert result["statistical_interpretation"] == "aligned_proxy"


@pytest.mark.parametrize(
    "covariance, message",
    [
        ([[1, 2, 0], [0, 1, 0], [0, 0, 1]], "symmetric"),
        ([[1, 0, 0], [0, 0, 0], [0, 0, 1]], "positive definite"),
    ],
)
def test_invalid_covariance_is_rejected(tmp_path: Path, covariance: list[list[float]], message: str) -> None:
    estimate, reference = _write_case(tmp_path, covariance)
    with pytest.raises(ValueError, match=message):
        evaluate_uncertainty(str(estimate), str(reference))


def test_uncertainty_check_gate_triage_and_comment(tmp_path: Path) -> None:
    estimate, reference = _write_case(tmp_path)
    config = tmp_path / "cloudanalyzer.yaml"
    config.write_text(f"""
checks:
  - id: covariance
    kind: uncertainty
    estimated: {estimate.name}
    reference: {reference.name}
    gate:
      max_mean_position_nees: 0.5
      min_coverage_95: 0.9
""", encoding="utf-8")
    result = run_check_suite(load_check_suite(str(config)))
    assert result["checks"][0]["passed"] is False
    assert "mean_position_nees" in result["summary"]["triage"]["items"][0]["failed_dimensions"]
    assert "Mean position NEES=1.0000" in build_pr_comment(result)


def test_loader_requires_explicit_covariance_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"states": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="covariance_frame"):
        load_covariance_trajectory(str(path))


def test_ill_conditioned_covariance_is_rejected(tmp_path: Path) -> None:
    estimate, reference = _write_case(tmp_path, [[1, 0, 0], [0, 1, 0], [0, 0, 1e-14]])
    with pytest.raises(ValueError, match="ill-conditioned"):
        evaluate_uncertainty(str(estimate), str(reference))


def test_no_timestamp_match_is_rejected(tmp_path: Path) -> None:
    estimate, reference = _write_case(tmp_path)
    payload = json.loads(estimate.read_text())
    for state in payload["states"]:
        state["timestamp"] += 100.0
    estimate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no covariance states matched"):
        evaluate_uncertainty(str(estimate), str(reference))


def test_anisotropic_covariance_is_global_rotation_invariant(tmp_path: Path) -> None:
    estimate, reference = _write_case(tmp_path, [[0.5, 0, 0], [0, 2, 0], [0, 0, 4]])
    baseline = evaluate_uncertainty(str(estimate), str(reference))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    payload = json.loads(estimate.read_text())
    covariance = np.asarray(payload["states"][0]["covariance"])
    for state in payload["states"]:
        state["position"] = (rotation @ np.asarray(state["position"])).tolist()
        state["covariance"] = (rotation @ covariance @ rotation.T).tolist()
    estimate.write_text(json.dumps(payload), encoding="utf-8")
    rows = ["timestamp,x,y,z"]
    for i in range(10):
        position = rotation @ np.array([i, 0, 0])
        rows.append(f"{i},{position[0]},{position[1]},{position[2]}")
    reference.write_text("\n".join(rows), encoding="utf-8")
    rotated = evaluate_uncertainty(str(estimate), str(reference))
    assert rotated["mean_position_nees"] == pytest.approx(baseline["mean_position_nees"])


def test_uncertainty_cli_happy_path_exits_zero(tmp_path: Path) -> None:
    estimate, reference = _write_case(tmp_path)
    result = CliRunner().invoke(app, ["uncertainty-evaluate", str(estimate), str(reference)])
    assert result.exit_code == 0, result.output
    assert "Mean position NEES" in result.output
