"""Tests for --output-json option across CLI commands."""

import json

import open3d as o3d
import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def _write_csv_trajectory(path, rows):
    lines = ["timestamp,x,y,z"]
    lines.extend(f"{timestamp},{x},{y},{z}" for timestamp, x, y, z in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class TestOutputJson:
    def test_info_output_json(self, sample_pcd_file, tmp_path):
        json_path = str(tmp_path / "info.json")
        result = runner.invoke(app, ["info", sample_pcd_file, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert data["num_points"] == 100

    def test_diff_output_json(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        json_path = str(tmp_path / "diff.json")
        result = runner.invoke(app, ["diff", src, tgt, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert "distance_stats" in data

    def test_downsample_output_json(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "down.pcd")
        json_path = str(tmp_path / "down.json")
        result = runner.invoke(app, ["downsample", sample_pcd_file, "-o", output, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert "reduction_ratio" in data

    def test_stats_output_json(self, sample_pcd_file, tmp_path):
        json_path = str(tmp_path / "stats.json")
        result = runner.invoke(app, ["stats", sample_pcd_file, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert "density" in data

    def test_filter_output_json(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "filtered.pcd")
        json_path = str(tmp_path / "filter.json")
        result = runner.invoke(app, ["filter", sample_pcd_file, "-o", output, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert "removed_points" in data

    def test_traj_evaluate_output_json(self, tmp_path):
        estimated = _write_csv_trajectory(
            tmp_path / "estimated.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0)],
        )
        reference = _write_csv_trajectory(
            tmp_path / "reference.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        json_path = str(tmp_path / "trajectory.json")

        result = runner.invoke(
            app,
            ["traj-evaluate", estimated, reference, "--output-json", json_path],
        )

        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert data["ate"]["rmse"] == pytest.approx(0.1)
        assert data["matching"]["matched_poses"] == 3

    def test_traj_batch_output_json(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        estimated = _write_csv_trajectory(
            estimated_dir / "a.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        json_path = str(tmp_path / "traj_batch.json")

        result = runner.invoke(
            app,
            [
                "traj-batch",
                str(estimated_dir),
                "--reference-dir",
                str(reference_dir),
                "--output-json",
                json_path,
            ],
        )

        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert len(data) == 1
        assert data[0]["path"] == estimated
        assert data[0]["ate"]["rmse"] == pytest.approx(0.1)

    def test_run_evaluate_output_json(self, tmp_path, identical_pcd):
        map_path = tmp_path / "map.pcd"
        map_reference = tmp_path / "map_ref.pcd"
        o3d.io.write_point_cloud(str(map_path), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference), identical_pcd)
        trajectory_path = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        trajectory_reference = _write_csv_trajectory(
            tmp_path / "traj_ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        json_path = str(tmp_path / "run.json")

        result = runner.invoke(
            app,
            [
                "run-evaluate",
                str(map_path),
                str(map_reference),
                trajectory_path,
                trajectory_reference,
                "--output-json",
                json_path,
            ],
        )

        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert data["map"]["auc"] == pytest.approx(1.0)
        assert data["trajectory"]["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)
        assert data["inspect"]["run_evaluate"].startswith("ca run-evaluate ")

    def test_run_batch_output_json(self, tmp_path, identical_pcd):
        map_dir = tmp_path / "maps"
        map_reference_dir = tmp_path / "map_refs"
        trajectory_dir = tmp_path / "trajs"
        trajectory_reference_dir = tmp_path / "traj_refs"
        map_dir.mkdir()
        map_reference_dir.mkdir()
        trajectory_dir.mkdir()
        trajectory_reference_dir.mkdir()
        o3d.io.write_point_cloud(str(map_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference_dir / "a.pcd"), identical_pcd)
        _write_csv_trajectory(
            trajectory_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        json_path = str(tmp_path / "run_batch.json")

        result = runner.invoke(
            app,
            [
                "run-batch",
                str(map_dir),
                "--map-reference-dir",
                str(map_reference_dir),
                "--trajectory-dir",
                str(trajectory_dir),
                "--trajectory-reference-dir",
                str(trajectory_reference_dir),
                "--output-json",
                json_path,
            ],
        )

        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert len(data) == 1
        assert data[0]["id"] == "a"
        assert data[0]["map"]["auc"] == pytest.approx(1.0)
        assert data[0]["inspect"]["run_evaluate"].startswith("ca run-evaluate ")
