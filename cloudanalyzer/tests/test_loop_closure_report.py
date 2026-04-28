"""Unit tests for ca.loop_closure_report (gates + report shape)."""

from __future__ import annotations

import json

import numpy as np
import open3d as o3d
import pytest

from ca.loop_closure_report import LoopClosureGate, build_loop_closure_report


def _write_pair(tmp_path, ref_pcd, before_pcd, after_pcd):
    ref = tmp_path / "ref.pcd"
    b = tmp_path / "before.pcd"
    a = tmp_path / "after.pcd"
    o3d.io.write_point_cloud(str(ref), ref_pcd)
    o3d.io.write_point_cloud(str(b), before_pcd)
    o3d.io.write_point_cloud(str(a), after_pcd)
    return str(b), str(a), str(ref)


def test_quality_gate_absent_when_no_constraints(tmp_path, simple_pcd):
    b, a, r = _write_pair(tmp_path, simple_pcd, simple_pcd, simple_pcd)
    report = build_loop_closure_report(
        before_map=b,
        after_map=a,
        reference_map=r,
        gate=LoopClosureGate(),
    )
    assert report["quality_gate"] is None
    assert report["map"]["delta"]["auc"] == pytest.approx(0.0, abs=1e-9)


def test_min_auc_gate_fails_when_no_improvement(tmp_path, simple_pcd, shifted_pcd):
    """Before and after both equal reference → AUC gain is zero."""
    b, a, r = _write_pair(tmp_path, simple_pcd, simple_pcd, simple_pcd)
    report = build_loop_closure_report(
        before_map=b,
        after_map=a,
        reference_map=r,
        thresholds=[0.05, 0.1, 0.2],
        gate=LoopClosureGate(min_auc_gain=0.001),
    )
    assert report["quality_gate"] is not None
    assert report["quality_gate"]["passed"] is False
    assert any("AUC gain" in x for x in report["quality_gate"]["reasons"])


def test_min_auc_gate_passes_when_after_better(tmp_path, simple_pcd):
    """Before strongly misaligned vs ref, after matches ref → clear AUC gain."""
    pts_bad = np.asarray(simple_pcd.points).copy()
    pts_bad[:, 0] += 0.55
    bad_pcd = o3d.geometry.PointCloud()
    bad_pcd.points = o3d.utility.Vector3dVector(pts_bad)
    b, a, r = _write_pair(tmp_path, simple_pcd, bad_pcd, simple_pcd)
    report = build_loop_closure_report(
        before_map=b,
        after_map=a,
        reference_map=r,
        thresholds=[0.05, 0.1, 0.2],
        gate=LoopClosureGate(min_auc_gain=0.001),
    )
    assert report["map"]["delta"]["auc"] > 0.01
    assert report["quality_gate"]["passed"] is True


def test_max_after_chamfer_gate_fails(tmp_path, simple_pcd, shifted_pcd):
    # Reference and before match; after is shifted → high chamfer(after, ref).
    b, a, r = _write_pair(tmp_path, simple_pcd, simple_pcd, shifted_pcd)
    report = build_loop_closure_report(
        before_map=b,
        after_map=a,
        reference_map=r,
        gate=LoopClosureGate(max_after_chamfer=1e-12),
    )
    assert report["quality_gate"]["passed"] is False
    assert any("chamfer" in x.lower() for x in report["quality_gate"]["reasons"])


def test_trajectory_ate_gate_without_trajectory_fails(tmp_path, simple_pcd):
    b, a, r = _write_pair(tmp_path, simple_pcd, simple_pcd, simple_pcd)
    report = build_loop_closure_report(
        before_map=b,
        after_map=a,
        reference_map=r,
        gate=LoopClosureGate(min_ate_gain=0.01),
    )
    assert report["quality_gate"]["passed"] is False
    assert any("missing" in x.lower() for x in report["quality_gate"]["reasons"])


def test_posegraph_section_when_g2o_provided(tmp_path, simple_pcd, shifted_pcd):
    g2o_b = tmp_path / "b.g2o"
    g2o_a = tmp_path / "a.g2o"
    body = "\n".join(
        [
            "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1",
            "VERTEX_SE3:QUAT 1 1 0 0 0 0 0 1",
            "EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 " + " ".join(["1"] * 21),
            "",
        ]
    )
    g2o_b.write_text(body, encoding="utf-8")
    g2o_a.write_text(body, encoding="utf-8")
    b, a, r = _write_pair(tmp_path, shifted_pcd, simple_pcd, simple_pcd)
    report = build_loop_closure_report(
        before_map=b,
        after_map=a,
        reference_map=r,
        before_g2o=str(g2o_b),
        after_g2o=str(g2o_a),
    )
    assert "posegraph_session" in report
    assert report["posegraph_session"]["before"]["summary"]["ok"] is True
    assert report["posegraph_session"]["after"]["summary"]["ok"] is True


def test_report_json_serializable(tmp_path, simple_pcd, shifted_pcd):
    b, a, r = _write_pair(tmp_path, shifted_pcd, simple_pcd, simple_pcd)
    report = build_loop_closure_report(
        before_map=b,
        after_map=a,
        reference_map=r,
        gate=LoopClosureGate(min_auc_gain=0.001),
    )
    json.dumps(report, default=str)
