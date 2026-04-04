"""Tests for config-driven QA check suites."""

from pathlib import Path
from textwrap import dedent

import numpy as np
import open3d as o3d
import pytest

from ca.core import load_check_suite, run_check_suite


def _write_csv_trajectory(path: Path, rows: list[tuple[float, float, float, float]]) -> str:
    lines = ["timestamp,x,y,z"]
    lines.extend(f"{timestamp},{x},{y},{z}" for timestamp, x, y, z in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def _write_pcd(path: Path, points: list[list[float]]) -> str:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    return str(path)


class TestLoadCheckSuite:
    def test_merges_defaults_and_resolves_relative_paths(self, tmp_path: Path):
        config = _write_config(
            tmp_path / "cloudanalyzer.yaml",
            """
            version: 1
            project: demo-stack
            summary_output_json: qa/summary.json
            defaults:
              thresholds: [0.2, 0.05, 0.1]
              max_time_delta: 0.1
              alignment: rigid
              report_dir: qa/reports
              json_dir: qa/results
              gate:
                min_auc: 0.95
                max_ate: 0.5
            checks:
              - id: perception-output
                kind: map
                source: outputs/map.pcd
                reference: refs/map_ref.pcd
              - kind: trajectory
                estimated: outputs/traj.csv
                reference: refs/traj_ref.csv
                gate:
                  max_rpe: 0.2
                  min_coverage: 0.9
            """,
        )

        suite = load_check_suite(str(config))

        assert suite.project == "demo-stack"
        assert suite.summary_output_json == str((tmp_path / "qa" / "summary.json").resolve())
        assert len(suite.checks) == 2

        artifact_check = suite.checks[0]
        assert artifact_check.kind == "artifact"
        assert artifact_check.thresholds == (0.05, 0.1, 0.2)
        assert artifact_check.alignment == "rigid"
        assert artifact_check.gate == {"min_auc": 0.95}
        assert artifact_check.inputs["source"] == str((tmp_path / "outputs" / "map.pcd").resolve())
        assert artifact_check.outputs.report_path == str(
            (tmp_path / "qa" / "reports" / "perception-output.html").resolve()
        )
        assert artifact_check.outputs.json_path == str(
            (tmp_path / "qa" / "results" / "perception-output.json").resolve()
        )

        trajectory_check = suite.checks[1]
        assert trajectory_check.kind == "trajectory"
        assert trajectory_check.alignment == "rigid"
        assert trajectory_check.max_time_delta == pytest.approx(0.1)
        assert trajectory_check.gate == {
            "max_ate": 0.5,
            "max_rpe": 0.2,
            "min_coverage": 0.9,
        }

    def test_rejects_unknown_kind(self, tmp_path: Path):
        config = _write_config(
            tmp_path / "cloudanalyzer.yaml",
            """
            checks:
              - kind: mystery
                source: a.pcd
                reference: b.pcd
            """,
        )

        with pytest.raises(ValueError, match="Unsupported check.kind"):
            load_check_suite(str(config))

    def test_requires_non_empty_checks(self, tmp_path: Path):
        config = _write_config(
            tmp_path / "cloudanalyzer.yaml",
            """
            checks: []
            """,
        )

        with pytest.raises(ValueError, match="checks must contain at least one item"):
            load_check_suite(str(config))


class TestRunCheckSuite:
    def test_runs_single_checks_and_writes_outputs(
        self,
        tmp_path: Path,
        identical_pcd,
    ):
        map_path = tmp_path / "artifacts" / "map.pcd"
        map_reference = tmp_path / "refs" / "map_ref.pcd"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_reference.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(map_path), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference), identical_pcd)

        trajectory_path = _write_csv_trajectory(
            tmp_path / "trajectories" / "estimated.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        trajectory_reference = _write_csv_trajectory(
            tmp_path / "trajectories" / "reference.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        config = _write_config(
            tmp_path / "cloudanalyzer.yaml",
            f"""
            version: 1
            project: qa-platform
            summary_output_json: qa/summary.json
            defaults:
              report_dir: qa/reports
              json_dir: qa/results
            checks:
              - id: perception-output
                kind: artifact
                source: {map_path.relative_to(tmp_path)}
                reference: {map_reference.relative_to(tmp_path)}
                gate:
                  min_auc: 0.95
                  max_chamfer: 0.01
              - id: localization-run
                kind: trajectory
                estimated: {Path(trajectory_path).relative_to(tmp_path)}
                reference: {Path(trajectory_reference).relative_to(tmp_path)}
                gate:
                  max_ate: 0.1
                  max_rpe: 0.1
                  max_drift: 0.1
                  min_coverage: 1.0
              - id: integrated-run
                kind: run
                map: {map_path.relative_to(tmp_path)}
                map_reference: {map_reference.relative_to(tmp_path)}
                trajectory: {Path(trajectory_path).relative_to(tmp_path)}
                trajectory_reference: {Path(trajectory_reference).relative_to(tmp_path)}
                gate:
                  min_auc: 0.95
                  max_chamfer: 0.01
                  max_ate: 0.1
                  max_rpe: 0.1
                  max_drift: 0.1
                  min_coverage: 1.0
            """,
        )

        result = run_check_suite(load_check_suite(str(config)))

        assert result["summary"]["passed"] is True
        assert result["summary"]["passed_checks"] == 3
        assert result["summary"]["failed_checks"] == 0
        assert [item["kind"] for item in result["checks"]] == [
            "artifact",
            "trajectory",
            "run",
        ]
        assert (tmp_path / "qa" / "summary.json").exists()
        assert (tmp_path / "qa" / "reports" / "perception-output.html").exists()
        assert (tmp_path / "qa" / "results" / "perception-output.json").exists()
        assert (tmp_path / "qa" / "reports" / "localization-run.html").exists()
        assert (tmp_path / "qa" / "reports" / "integrated-run.html").exists()
        assert result["checks"][0]["result"]["inspect"]["web_heatmap"].startswith("ca web ")

    def test_runs_batch_checks(self, tmp_path: Path, identical_pcd):
        artifact_dir = tmp_path / "artifacts"
        map_dir = tmp_path / "maps"
        map_reference_dir = tmp_path / "map_refs"
        trajectory_dir = tmp_path / "trajectories"
        trajectory_reference_dir = tmp_path / "trajectory_refs"
        trajectory_batch_dir = tmp_path / "trajectory_batch"
        trajectory_batch_reference_dir = tmp_path / "trajectory_batch_refs"
        for directory in [
            artifact_dir,
            map_dir,
            map_reference_dir,
            trajectory_dir,
            trajectory_reference_dir,
            trajectory_batch_dir,
            trajectory_batch_reference_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        reference_map = tmp_path / "reference_map.pcd"
        artifact_path = artifact_dir / "candidate.pcd"
        run_map_path = map_dir / "run1.pcd"
        run_map_reference_path = map_reference_dir / "run1.pcd"
        for path in [reference_map, artifact_path, run_map_path, run_map_reference_path]:
            o3d.io.write_point_cloud(str(path), identical_pcd)

        _write_csv_trajectory(
            trajectory_batch_dir / "run1.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_batch_reference_dir / "run1.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_dir / "run1.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "run1.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        config = _write_config(
            tmp_path / "cloudanalyzer.yaml",
            f"""
            defaults:
              report_dir: qa/reports
              json_dir: qa/results
            checks:
              - id: artifact-batch
                kind: artifact_batch
                directory: {artifact_dir.relative_to(tmp_path)}
                reference: {reference_map.relative_to(tmp_path)}
                gate:
                  min_auc: 0.95
              - id: trajectory-batch
                kind: trajectory_batch
                directory: {trajectory_batch_dir.relative_to(tmp_path)}
                reference_dir: {trajectory_batch_reference_dir.relative_to(tmp_path)}
                gate:
                  max_ate: 0.1
                  max_rpe: 0.1
                  max_drift: 0.1
                  min_coverage: 1.0
              - id: run-batch
                kind: run_batch
                map_dir: {map_dir.relative_to(tmp_path)}
                map_reference_dir: {map_reference_dir.relative_to(tmp_path)}
                trajectory_dir: {trajectory_dir.relative_to(tmp_path)}
                trajectory_reference_dir: {trajectory_reference_dir.relative_to(tmp_path)}
                gate:
                  min_auc: 0.95
                  max_ate: 0.1
                  max_rpe: 0.1
                  max_drift: 0.1
                  min_coverage: 1.0
            """,
        )

        result = run_check_suite(load_check_suite(str(config)))

        assert result["summary"]["passed"] is True
        assert result["summary"]["passed_checks"] == 3
        assert result["checks"][0]["summary"]["total_files"] == 1
        assert result["checks"][1]["summary"]["total_files"] == 1
        assert result["checks"][2]["summary"]["total_runs"] == 1
        assert (tmp_path / "qa" / "reports" / "artifact-batch.html").exists()
        assert (tmp_path / "qa" / "reports" / "trajectory-batch.html").exists()
        assert (tmp_path / "qa" / "reports" / "run-batch.html").exists()

    def test_marks_failed_checks(self, tmp_path: Path, identical_pcd, shifted_pcd):
        map_path = tmp_path / "map.pcd"
        map_reference = tmp_path / "map_ref.pcd"
        o3d.io.write_point_cloud(str(map_path), shifted_pcd)
        o3d.io.write_point_cloud(str(map_reference), identical_pcd)

        trajectory_path = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.4, 0.0, 0.0)],
        )
        trajectory_reference = _write_csv_trajectory(
            tmp_path / "traj_ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        config = _write_config(
            tmp_path / "cloudanalyzer.yaml",
            f"""
            checks:
              - id: bad-artifact
                kind: artifact
                source: {map_path.relative_to(tmp_path)}
                reference: {map_reference.relative_to(tmp_path)}
                gate:
                  min_auc: 0.99
                  max_chamfer: 0.01
              - id: bad-trajectory
                kind: trajectory
                estimated: {Path(trajectory_path).relative_to(tmp_path)}
                reference: {Path(trajectory_reference).relative_to(tmp_path)}
                gate:
                  max_ate: 0.05
                  max_rpe: 0.05
                  max_drift: 0.05
                  min_coverage: 1.0
            """,
        )

        result = run_check_suite(load_check_suite(str(config)))

        assert result["summary"]["passed"] is False
        assert result["summary"]["failed_checks"] == 2
        assert result["summary"]["failed_check_ids"] == ["bad-artifact", "bad-trajectory"]
        assert result["summary"]["triage"]["strategy"] == "severity_weighted"
        assert result["summary"]["triage"]["failed_count"] == 2
        assert set(result["summary"]["triage"]["ranked_ids"]) == {
            "bad-artifact",
            "bad-trajectory",
        }

    def test_runs_ground_check(self, tmp_path: Path):
        ground = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        nonground = [[0, 0, 2], [1, 0, 2], [2, 0, 2]]
        est_ground = _write_pcd(tmp_path / "est_g.pcd", ground)
        est_nonground = _write_pcd(tmp_path / "est_ng.pcd", nonground)
        ref_ground = _write_pcd(tmp_path / "ref_g.pcd", ground)
        ref_nonground = _write_pcd(tmp_path / "ref_ng.pcd", nonground)

        config = _write_config(
            tmp_path / "cloudanalyzer.yaml",
            f"""
            checks:
              - id: ground-seg
                kind: ground
                estimated_ground: {Path(est_ground).relative_to(tmp_path)}
                estimated_nonground: {Path(est_nonground).relative_to(tmp_path)}
                reference_ground: {Path(ref_ground).relative_to(tmp_path)}
                reference_nonground: {Path(ref_nonground).relative_to(tmp_path)}
                gate:
                  min_f1: 0.9
                  min_iou: 0.8
            """,
        )

        result = run_check_suite(load_check_suite(str(config)))

        assert result["summary"]["passed"] is True
        assert result["summary"]["gated_checks"] == 1
        ground_check = result["checks"][0]
        assert ground_check["id"] == "ground-seg"
        assert ground_check["kind"] == "ground"
        assert ground_check["passed"] is True
        assert ground_check["summary"]["f1"] == pytest.approx(1.0)
        assert ground_check["summary"]["iou"] == pytest.approx(1.0)
