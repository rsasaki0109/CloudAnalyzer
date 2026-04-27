"""Tests for posegraph validation utilities."""

from __future__ import annotations

from ca.posegraph import validate_posegraph_session


def test_validate_posegraph_session_detects_disconnected_graph(tmp_path):
    g2o = tmp_path / "pose_graph.g2o"
    g2o.write_text(
        "\n".join(
            [
                "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1",
                "VERTEX_SE3:QUAT 1 1 0 0 0 0 0 1",
                "VERTEX_SE3:QUAT 2 2 0 0 0 0 0 1",
                "VERTEX_SE3:QUAT 3 3 0 0 0 0 0 1",
                "EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 " + " ".join(["1"] * 21),
                "EDGE_SE3:QUAT 2 3 1 0 0 0 0 0 1 " + " ".join(["1"] * 21),
                "",
            ]
        ),
        encoding="utf-8",
    )
    out = validate_posegraph_session(str(g2o))
    assert out["summary"]["ok"] is False
    assert any("disconnected" in e for e in out["summary"]["errors"])


def test_validate_posegraph_session_duplicate_edges_are_warning_not_error(tmp_path):
    g2o = tmp_path / "pose_graph.g2o"
    g2o.write_text(
        "\n".join(
            [
                "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1",
                "VERTEX_SE3:QUAT 1 1 0 0 0 0 0 1",
                "EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 " + " ".join(["1"] * 21),
                "EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 " + " ".join(["1"] * 21),
                "",
            ]
        ),
        encoding="utf-8",
    )
    out = validate_posegraph_session(str(g2o))
    assert out["summary"]["ok"] is True
    assert out["g2o"]["duplicate_undirected_edges"] == 1
    assert any("duplicate" in w for w in out["summary"]["warnings"])
