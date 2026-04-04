"""Tests for ca.web module."""

import json
import threading
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import open3d as o3d
import pytest

from ca.core.web_progressive_loading import (
    WebProgressiveLoadingChunk,
    WebProgressiveLoadingResult,
)
from ca.core.web_sampling import WebSampleResult
from ca.core.web_trajectory_sampling import WebTrajectorySamplingResult
from ca.web import (
    _VIEWER_HTML,
    _downsample_for_web,
    export_static_bundle,
    _make_handler,
    _prepare_trajectory_viewer_data,
    _prepare_viewer_bundle,
    _prepare_viewer_data,
)


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
        assert "Point Inspection" in html
        assert "Click a point." in html
        assert "defaultPointInspectionHint()" in html
        assert "Distance to reference:" in html
        assert "Nearest displayed reference:" in html
        assert "updateCorrespondenceOverlay(position, referencePosition);" in html
        assert "Trajectory Inspection" in html
        assert "Click the worst marker or segment." in html
        assert "Trajectory Error Timeline" in html
        assert "Click a point to focus the viewer." in html
        assert "Trajectory displayed:" in html
        assert "Trajectory sampler:" in html
        assert "Trajectory ATE RMSE (display):" in html
        assert "Reset View" in html
        assert "focusCameraOnSelection(selectionPositions)" in html
        assert "renderTrajectoryTimeline()" in html
        assert "selectPointFeature(pointIntersections[0])" in html
        assert "data-trajectory-selection" in html
        assert "target instanceof Element" in html
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

    def test_serves_progressive_chunk_payload(self):
        data_json = json.dumps({"positions": [1.0, 2.0, 3.0], "filename": "test.pcd"})
        handler_cls = _make_handler(
            _VIEWER_HTML,
            data_json,
            {"chunks/source/0.json": json.dumps({"positions": [4.0, 5.0, 6.0]})},
        )

        from http.server import HTTPServer
        server = HTTPServer(('127.0.0.1', 0), handler_cls)
        port = server.server_address[1]

        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        urlopen(f"http://127.0.0.1:{port}/data.json").read()

        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        chunk = json.loads(urlopen(f"http://127.0.0.1:{port}/chunks/source/0.json").read())
        assert chunk["positions"] == [4.0, 5.0, 6.0]
        server.server_close()


class TestPrepareViewerData:
    def test_downsample_for_web_uses_core_reducer(self, simple_pcd, monkeypatch):
        seen = {}

        def fake_reduce(point_cloud, max_points, label):
            seen["point_cloud"] = point_cloud
            seen["max_points"] = max_points
            seen["label"] = label
            return WebSampleResult(
                point_cloud=point_cloud,
                strategy="dummy",
                design="test",
                original_points=len(point_cloud.points),
                reduced_points=len(point_cloud.points),
                metadata={},
            )

        monkeypatch.setattr("ca.web.reduce_point_cloud_for_web", fake_reduce)

        reduced = _downsample_for_web(simple_pcd, max_points=80, label="sample")

        assert reduced is simple_pcd
        assert seen == {
            "point_cloud": simple_pcd,
            "max_points": 80,
            "label": "sample",
        }

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

    def test_standard_mode_respects_max_points_budget(self, tmp_path):
        rng = np.random.default_rng(123)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rng.random((400, 3)))
        path = tmp_path / "large_cloud.pcd"
        o3d.io.write_point_cloud(str(path), pcd)

        data = _prepare_viewer_data([str(path)], max_points=40)

        assert data["display_points"] == 40
        assert data["original_points"] == 400

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
        assert data["trajectory"]["displayed_estimated_pose_count"] == 3

    def test_prepare_viewer_bundle_exposes_progressive_source_chunks(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(321)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rng.random((100, 3)))
        path = tmp_path / "cloud.pcd"
        o3d.io.write_point_cloud(str(path), pcd)

        def fake_plan_progressive_loading_for_web(
            positions,
            initial_points,
            chunk_points,
            distances=None,
            label="point cloud",
            strategy=None,
        ):
            return WebProgressiveLoadingResult(
                initial_positions=positions[:10],
                initial_distances=distances[:10] if distances is not None else None,
                chunks=(
                    WebProgressiveLoadingChunk(positions=positions[10:25]),
                    WebProgressiveLoadingChunk(positions=positions[25:40]),
                ),
                strategy="dummy",
                design="test",
                original_points=positions.shape[0],
                initial_points=10,
                chunk_points=15,
                metadata={},
            )

        monkeypatch.setattr("ca.web.plan_progressive_loading_for_web", fake_plan_progressive_loading_for_web)
        monkeypatch.setattr("ca.web._progressive_initial_budget", lambda max_points, display_points: 10)

        data, chunk_payloads = _prepare_viewer_bundle([str(path)], max_points=500000)

        assert data["display_points"] == 100
        assert data["initial_display_points"] == 10
        assert data["progressive_loading"]["enabled"] is True
        assert data["progressive_loading"]["source"]["strategy"] == "dummy"
        assert len(data["progressive_loading"]["source"]["chunks"]) == 2
        assert "chunks/source/0.json" in chunk_payloads
        assert data["scene_bounds"]["extent"] > 0

    def test_export_static_bundle_writes_relative_assets(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "bundle"

        monkeypatch.setattr(
            "ca.web._prepare_viewer_bundle",
            lambda *args, **kwargs: (
                {
                    "positions": [1.0, 2.0, 3.0],
                    "viewer_mode": "standard",
                    "display_points": 1,
                    "progressive_loading": {
                        "enabled": True,
                        "source": {
                            "enabled": True,
                            "strategy": "dummy",
                            "chunks": [{"path": "chunks/source/0.json", "points": 1}],
                            "total_points": 1,
                        },
                        "reference": None,
                    },
                },
                {"chunks/source/0.json": json.dumps({"positions": [4.0, 5.0, 6.0]})},
            ),
        )

        result = export_static_bundle(["source.pcd"], str(output_dir))

        assert result["output_dir"] == str(output_dir)
        assert (output_dir / "index.html").exists()
        assert (output_dir / "data.json").exists()
        assert (output_dir / "chunks" / "source" / "0.json").exists()
        assert "fetch('data.json')" in (output_dir / "index.html").read_text()
        exported_data = json.loads((output_dir / "data.json").read_text())
        assert exported_data["progressive_loading"]["source"]["chunks"][0]["path"] == "chunks/source/0.json"

    def test_large_trajectory_overlay_is_reduced_for_display(self, tmp_path, simple_pcd):
        path = tmp_path / "cloud.pcd"
        trajectory_rows = [
            (
                float(index),
                index * 0.05,
                float(np.sin(index * 0.08)),
                0.0,
            )
            for index in range(400)
        ]
        trajectory_path = _write_csv_trajectory(tmp_path / "traj.csv", trajectory_rows)
        o3d.io.write_point_cloud(str(path), simple_pcd)

        data = _prepare_viewer_data(
            [str(path)],
            max_points=40,
            trajectory_path=trajectory_path,
        )

        assert data["trajectory"]["estimated_pose_count"] == 400
        assert data["trajectory"]["displayed_estimated_pose_count"] <= 200
        assert data["trajectory"]["displayed_estimated_pose_count"] < 400

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
        assert data["trajectory"]["displayed_estimated_pose_count"] == 3
        assert len(data["trajectory"]["timestamps"]) == 3
        assert len(data["trajectory"]["ate_errors"]) == 3
        assert len(data["trajectory"]["rpe_timestamps"]) == 2
        assert len(data["trajectory"]["rpe_errors"]) == 2
        assert data["trajectory"]["worst_ate_index"] is not None
        assert data["trajectory"]["worst_ate_sample"] is not None
        assert data["trajectory"]["worst_rpe_index"] is not None
        assert data["trajectory"]["worst_rpe_segment"] is not None

    def test_single_trajectory_overlay_uses_core_reducer(self, monkeypatch):
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.2, 0.0],
                [2.0, 0.4, 0.0],
                [3.0, 0.6, 0.0],
            ]
        )
        seen = {}

        def fake_load_trajectory(path):
            seen["path"] = path
            return {"timestamps": timestamps, "positions": positions}

        def fake_reduce_trajectory_for_web(
            positions,
            max_points,
            timestamps,
            label,
            preserve_indices=(),
            strategy=None,
        ):
            seen["max_points"] = max_points
            seen["timestamps"] = timestamps
            seen["label"] = label
            seen["preserve_indices"] = preserve_indices
            return WebTrajectorySamplingResult(
                positions=positions[[0, 2, 3]],
                timestamps=timestamps[[0, 2, 3]],
                kept_indices=np.array([0, 2, 3]),
                strategy="dummy",
                design="test",
                original_points=positions.shape[0],
                reduced_points=3,
                metadata={},
            )

        monkeypatch.setattr("ca.web.load_trajectory", fake_load_trajectory)
        monkeypatch.setattr("ca.web.reduce_trajectory_for_web", fake_reduce_trajectory_for_web)

        data = _prepare_trajectory_viewer_data("traj.csv", max_points=1000)

        assert seen["path"] == "traj.csv"
        assert seen["max_points"] == 4
        assert seen["label"] == "trajectory traj.csv"
        assert seen["preserve_indices"] == ()
        assert data["estimated_pose_count"] == 4
        assert data["displayed_estimated_pose_count"] == 3
        assert data["sampling"]["strategy"] == "dummy"

    def test_paired_trajectory_overlay_preserves_worst_indices_when_reducing(self, monkeypatch):
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        estimated_positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [2.0, 1.4, 0.0],
                [3.0, 0.6, 0.0],
                [4.0, 0.4, 0.0],
                [5.0, 0.1, 0.0],
            ]
        )
        reference_positions = np.column_stack(
            [np.arange(6, dtype=float), np.zeros(6, dtype=float), np.zeros(6, dtype=float)]
        )
        seen = {}

        def fake_evaluate_trajectory(*args, **kwargs):
            return {
                "alignment": {"mode": "none"},
                "matching": {"matched_poses": 6, "coverage_ratio": 1.0},
                "drift": {"endpoint": 0.2},
                "worst_ate_samples": [
                    {"timestamp": 2.0, "position_error": 1.4, "time_delta": 0.0}
                ],
                "matched_trajectory": {
                    "timestamps": timestamps.tolist(),
                    "estimated_positions": estimated_positions.tolist(),
                    "reference_positions": reference_positions.tolist(),
                    "ate_errors": [0.0, 0.1, 1.4, 0.6, 0.4, 0.1],
                },
                "error_series": {
                    "rpe_timestamps": [0.5, 1.5, 2.5, 3.5, 4.5],
                    "rpe_translation": [0.1, 0.2, 1.0, 0.3, 0.2],
                },
            }

        def fake_reduce_trajectory_for_web(
            positions,
            max_points,
            timestamps,
            label,
            preserve_indices=(),
            strategy=None,
        ):
            seen["max_points"] = max_points
            seen["label"] = label
            seen["preserve_indices"] = preserve_indices
            return WebTrajectorySamplingResult(
                positions=positions[[0, 2, 3, 5]],
                timestamps=timestamps[[0, 2, 3, 5]],
                kept_indices=np.array([0, 2, 3, 5]),
                strategy="dummy",
                design="test",
                original_points=positions.shape[0],
                reduced_points=4,
                metadata={},
            )

        monkeypatch.setattr("ca.web.evaluate_trajectory", fake_evaluate_trajectory)
        monkeypatch.setattr("ca.web.reduce_trajectory_for_web", fake_reduce_trajectory_for_web)

        data = _prepare_trajectory_viewer_data(
            "traj.csv",
            trajectory_reference_path="traj_ref.csv",
            max_points=1000,
        )

        assert seen["max_points"] == 6
        assert seen["label"] == "matched trajectory traj.csv"
        assert 2 in seen["preserve_indices"]
        assert 3 in seen["preserve_indices"]
        assert data["estimated_pose_count"] == 6
        assert data["displayed_estimated_pose_count"] == 4
        assert data["worst_ate_index"] == 1
        assert len(data["timestamps"]) == 4
        assert len(data["rpe_errors"]) == 3
        assert data["worst_rpe_segment"]["start_timestamp"] == pytest.approx(
            data["timestamps"][data["worst_rpe_index"]]
        )
        assert data["sampling"]["strategy"] == "dummy"

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
