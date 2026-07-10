"""Phase 35 integration tests for AWD/SCS map-quality gates."""

from __future__ import annotations

from ca.core.check_triage import build_check_triage_request
from ca.core.checks import _artifact_quality_gate
from ca.pr_comment import build_pr_comment


def _failed_map_check() -> dict:
    gate = _artifact_quality_gate(
        1.0,
        0.0,
        awd_m=0.4,
        max_awd=0.2,
        scs=0.8,
        max_scs=0.5,
    )
    return {
        "id": "map-quality",
        "kind": "artifact",
        "passed": False,
        "summary": {
            "auc": 1.0,
            "chamfer_distance": 0.0,
            "hausdorff_distance": 0.0,
            "awd_m": 0.4,
            "scs": 0.8,
            "passed": False,
        },
        "result": {"quality_gate": gate},
    }


def test_awd_scs_gate_is_lower_is_better() -> None:
    gate = _failed_map_check()["result"]["quality_gate"]
    assert gate["passed"] is False
    assert any("AWD" in reason for reason in gate["reasons"])
    assert any("SCS" in reason for reason in gate["reasons"])


def test_unavailable_voxel_metrics_fail_configured_gates() -> None:
    gate = _artifact_quality_gate(
        1.0,
        0.0,
        awd_m=float("nan"),
        max_awd=0.2,
        scs=float("nan"),
        max_scs=0.5,
    )
    assert gate is not None and gate["passed"] is False
    assert all("unavailable" in reason for reason in gate["reasons"])


def test_map_metrics_surface_in_triage_dimensions() -> None:
    request = build_check_triage_request([_failed_map_check()])
    item = request.failed_items[0]
    assert item.metrics["awd"] == 0.4
    assert item.metrics["scs"] == 0.8
    assert item.gate["max_awd"] == 0.2
    assert item.gate["max_scs"] == 0.5


def test_map_metrics_surface_in_pr_comment_with_baseline_delta() -> None:
    current = _failed_map_check()
    baseline = _failed_map_check()
    baseline["summary"]["awd_m"] = 0.1
    baseline["summary"]["scs"] = 0.2
    payload = {
        "project": "phase35",
        "summary": {
            "passed": False,
            "total_checks": 1,
            "passed_checks": 0,
            "failed_checks": 1,
        },
        "checks": [current],
    }
    baseline_payload = {
        "summary": payload["summary"],
        "checks": [baseline],
    }
    comment = build_pr_comment(payload, baseline=baseline_payload)
    assert "AWD=0.4000 m" in comment
    assert "SCS=0.8000" in comment
    assert "↑" in comment
