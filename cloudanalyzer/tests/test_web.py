"""Tests for ca.web module."""

import json
import threading
from pathlib import Path
from urllib.request import urlopen

import open3d as o3d
import pytest

from ca.web import _VIEWER_HTML, _make_handler, _prepare_viewer_data


def _write_csv_trajectory(path: Path, rows: list[tuple[float, float, float, float]]) -> str:
    lines = ["timestamp,x,y,z"]
    lines.extend(f"{timestamp},{x},{y},{z}" for timestamp, x, y, z in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class TestWebHandler:
    def test_serves_html(self, sample_pcd_file):
        data_json = json.dumps({"positions": [0, 0, 0], "filename": "test.pcd"})
        handler_cls = _make_handler(_VIEWER_HTML, data_json)

        from http.server import HTTPServer
        server = HTTPServer(('127.0.0.1', 0), handler_cls)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        resp = urlopen(f"http://127.0.0.1:{port}/")
        html = resp.read().decode()
        assert "CloudAnalyzer" in html
        assert "Reference Overlay" in html
        assert "Estimated Trajectory" in html
        assert "Reference Trajectory" in html
        assert "Worst ATE Marker" in html
        assert "Worst RPE Segment" in html
        assert "Trajectory Inspection" in html
        assert "Click the worst marker or segment." in html
        assert "Trajectory Error Timeline" in html
        assert "Click a point to focus the viewer." in html
        assert "Reset View" in html
        assert "focusCameraOnSelection(selectionPositions)" in html
        assert "renderTrajectoryTimeline()" in html
        assert "data-trajectory-selection" in html
        assert "document.getElementById('resetView').addEventListener('click', resetView)" in html
        assert "Error Threshold" in html
        assert "Distance Legend" in html
        server.server_close()

    def test_serves_data(self, sample_pcd_file):
        data_json = json.dumps({"positions": [1.0, 2.0, 3.0], "filename": "test.pcd"})
        handler_cls = _make_handler(_VIEWER_HTML, data_json)

        from http.server import HTTPServer
        server = HTTPServer(('127.0.0.1', 0), handler_cls)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        resp = urlopen(f"http://127.0.0.1:{port}/data.json")
        data = json.loads(resp.read())
        assert data["positions"] == [1.0, 2.0, 3.0]
        server.server_close()


class TestPrepareViewerData:
    def test_standard_mode(self, tmp_path, simple_pcd):
        path = tmp_path / "cloud.pcd"
        o3d.io.write_point_cloud(str(path), simple_pcd)

        data = _prepare_viewer_data([str(path)], max_points=1000)

        assert data["viewer_mode"] == "standard"
        assert data["display_points"] == 100
        assert data["filename"] == "cloud.pcd"
        assert "distances" not in data

    def test_heatmap_mode(self, tmp_path, identical_pcd, shifted_pcd):
        source = tmp_path / "source.pcd"
        target = tmp_path / "target.pcd"
        o3d.io.write_point_cloud(str(source), identical_pcd)
        o3d.io.write_point_cloud(str(target), shifted_pcd)

        data = _prepare_viewer_data([str(source), str(target)], max_points=1000, heatmap=True)

        assert data["viewer_mode"] == "heatmap"
        assert data["source_filename"] == "source.pcd"
        assert data["target_filename"] == "target.pcd"
        assert len(data["distances"]) == data["display_points"]
        assert len(data["reference_positions"]) == data["reference_display_points"] * 3
        assert data["reference_display_points"] == 100
        assert data["distance_stats"]["mean"] > 0

    def test_standard_mode_with_trajectory_overlay(self, tmp_path, simple_pcd):
        path = tmp_path / "cloud.pcd"
        trajectory_path = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.5, 0.0), (2.0, 2.0, 1.0, 0.0)],
        )
        o3d.io.write_point_cloud(str(path), simple_pcd)

        data = _prepare_viewer_data(
            [str(path)],
            max_points=1000,
            trajectory_path=trajectory_path,
        )

        assert data["viewer_mode"] == "standard"
        assert data["trajectory"]["mode"] == "single"
        assert data["trajectory"]["estimated_filename"] == "traj.csv"
        assert data["trajectory"]["estimated_pose_count"] == 3

    def test_heatmap_mode_with_paired_trajectory_overlay(self, tmp_path, identical_pcd, shifted_pcd):
        source = tmp_path / "source.pcd"
        target = tmp_path / "target.pcd"
        estimated_trajectory = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.2, 0.0, 0.0)],
        )
        reference_trajectory = _write_csv_trajectory(
            tmp_path / "traj_ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        o3d.io.write_point_cloud(str(source), identical_pcd)
        o3d.io.write_point_cloud(str(target), shifted_pcd)

        data = _prepare_viewer_data(
            [str(source), str(target)],
            max_points=1000,
            heatmap=True,
            trajectory_path=estimated_trajectory,
            trajectory_reference_path=reference_trajectory,
        )

        assert data["viewer_mode"] == "heatmap"
        assert data["trajectory"]["mode"] == "paired"
        assert data["trajectory"]["matching"]["matched_poses"] == 3
        assert data["trajectory"]["alignment"]["mode"] == "none"
        assert data["trajectory"]["ate"]["rmse"] > 0
        assert data["trajectory"]["rpe"]["rmse"] > 0
        assert len(data["trajectory"]["timestamps"]) == 3
        assert len(data["trajectory"]["ate_errors"]) == 3
        assert len(data["trajectory"]["rpe_timestamps"]) == 2
        assert len(data["trajectory"]["rpe_errors"]) == 2
        assert data["trajectory"]["worst_ate_index"] is not None
        assert data["trajectory"]["worst_ate_sample"] is not None
        assert data["trajectory"]["worst_rpe_index"] is not None
        assert data["trajectory"]["worst_rpe_segment"] is not None

    def test_heatmap_requires_two_paths(self, tmp_path, simple_pcd):
        path = tmp_path / "cloud.pcd"
        o3d.io.write_point_cloud(str(path), simple_pcd)

        with pytest.raises(ValueError):
            _prepare_viewer_data([str(path)], heatmap=True)

    def test_trajectory_reference_requires_trajectory(self, tmp_path, simple_pcd):
        path = tmp_path / "cloud.pcd"
        trajectory_reference = _write_csv_trajectory(
            tmp_path / "traj_ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
        )
        o3d.io.write_point_cloud(str(path), simple_pcd)

        with pytest.raises(ValueError):
            _prepare_viewer_data(
                [str(path)],
                trajectory_reference_path=trajectory_reference,
            )
