"""Tests for posegraph validation utilities."""

from __future__ import annotations

from pathlib import Path

from ca.posegraph import discover_session_paths, parse_g2o_summary, validate_posegraph_session


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


def test_discover_session_paths_reports_expected_and_exists(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    (root / "pose_graph.g2o").write_text(
        "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    (root / "optimized_poses_tum.txt").write_text("0 0 0 0\n1 1 0 0\n", encoding="utf-8")
    kf = root / "key_point_frame"
    kf.mkdir()
    (kf / "a.pcd").write_text("x", encoding="utf-8")
    (root / "map.pcd").write_text("x", encoding="utf-8")

    d = discover_session_paths(str(root))
    assert d["g2o_path"] is not None
    assert d["tum_path"] is not None
    assert d["key_point_frame_dir"] is not None
    assert d["map_path"] is not None
    assert d["exists"]["g2o_path"] is True
    assert Path(d["expected"]["map_path"]) == root / "map.pcd"


def test_parse_g2o_flags_edge_referencing_missing_vertex(tmp_path):
    g2o = tmp_path / "p.g2o"
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
    summary = parse_g2o_summary(str(g2o))
    assert summary.vertex_ids == {0}
    assert summary.edge_pairs == [(0, 99)]
    out = validate_posegraph_session(str(g2o))
    assert out["summary"]["ok"] is False
    assert any("missing" in e.lower() for e in out["summary"]["errors"])
