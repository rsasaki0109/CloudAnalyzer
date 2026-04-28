"""Unit tests for ca.loop_closure_report (gates + report shape)."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_posegraph_gate_fails_when_validated_session_not_ok(tmp_path, simple_pcd):
    g2o = tmp_path / "bad.g2o"
    g2o.write_text(
        "\n".join(
            [
                "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1",
                "EDGE_SE3:QUAT 0 99 1 0 0 0 0 0 1 " + " ".join(["1"] * 21),
                "",
            ]
        ),
        encoding="utf-8",
    )
    b, a, r = _write_pair(tmp_path, simple_pcd, simple_pcd, simple_pcd)
    report = build_loop_closure_report(
        before_map=b,
        after_map=a,
        reference_map=r,
        after_g2o=str(g2o),
        gate=LoopClosureGate(require_posegraph_ok=True),
    )
    assert report["posegraph_session"]["after"]["summary"]["ok"] is False
    assert report["quality_gate"]["passed"] is False
    assert any("posegraph" in reason.lower() for reason in report["quality_gate"]["reasons"])


def test_posegraph_gate_requires_posegraph_inputs(tmp_path, simple_pcd):
    b, a, r = _write_pair(tmp_path, simple_pcd, simple_pcd, simple_pcd)
    report = build_loop_closure_report(
        before_map=b,
        after_map=a,
        reference_map=r,
        gate=LoopClosureGate(require_posegraph_ok=True),
    )
    assert report["quality_gate"]["passed"] is False
    assert any("missing" in reason.lower() for reason in report["quality_gate"]["reasons"])


def test_manual_loop_closure_demo_fixture_passes_gate():
    root = Path(__file__).resolve().parents[2] / "demo_assets" / "manual-loop-closure-minimal"
    report = build_loop_closure_report(
        before_map=str(root / "before" / "map.pcd"),
        after_map=str(root / "after" / "map.pcd"),
        reference_map=str(root / "reference" / "map.pcd"),
        thresholds=[0.05, 0.1, 0.2, 0.5],
        before_trajectory=str(root / "before" / "optimized_poses_tum.txt"),
        after_trajectory=str(root / "after" / "optimized_poses_tum.txt"),
        reference_trajectory=str(root / "reference" / "trajectory.tum"),
        before_g2o=str(root / "before" / "pose_graph.g2o"),
        after_g2o=str(root / "after" / "pose_graph.g2o"),
        before_tum=str(root / "before" / "optimized_poses_tum.txt"),
        after_tum=str(root / "after" / "optimized_poses_tum.txt"),
        before_key_point_frame_dir=str(root / "before" / "key_point_frame"),
        after_key_point_frame_dir=str(root / "after" / "key_point_frame"),
        gate=LoopClosureGate(
            min_auc_gain=0.01,
            min_ate_gain=0.05,
            require_posegraph_ok=True,
        ),
    )
    assert report["quality_gate"]["passed"] is True
    assert report["map"]["delta"]["auc"] > 0
    assert report["trajectory"]["after"]["ate_rmse"] < report["trajectory"]["before"]["ate_rmse"]
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
