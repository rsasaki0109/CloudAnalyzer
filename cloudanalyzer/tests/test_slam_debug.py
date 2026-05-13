"""Tests for SLAM run diagnostics."""

import json

from typer.testing import CliRunner

from ca.slam_debug import (
    analyze_slam_run,
    diagnose_slam_frame,
    render_slam_debug_markdown,
)
from cloudanalyzer_cli.main import app


def test_analyze_slam_run_ranks_failed_and_high_error_frames(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "scan_id,timestamp_sec,scan_match_failed,scan_match_error,scan_match_rmse_m,"
                "scan_match_correspondence_rejection_rate,prediction_delta_m,"
                "scan_match_vs_initial_pose_delta_m,registration_retry_count,"
                "consecutive_scan_match_failures,scan_quality_low,scan_quality_reason,"
                "initial_x_m,initial_y_m,initial_z_m",
                "scan_0,0.0,false,,0.05,0.1,0.2,0.1,0,0,false,,0.0,0.0,0.0",
                "scan_1,1.0,true,no_correspondences,3.5,0.9,1.2,1.1,2,1,true,sparse,1.0,2.0,0.0",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "scans_manifest.csv"
    manifest.write_text(
        "scan_id,timestamp_sec,points_csv\nscan_0,0.0,scan_0.pcd\nscan_1,1.0,scan_1.pcd\n",
        encoding="utf-8",
    )
    trajectory = tmp_path / "trajectory.csv"
    trajectory.write_text(
        "timestamp_sec,x_m,y_m,z_m,roll_rad,pitch_rad,yaw_rad\n"
        "1.0,1.1,2.1,0.0,0.0,0.0,0.1\n",
        encoding="utf-8",
    )

    result = analyze_slam_run(
        str(metrics),
        scans_manifest_csv=str(manifest),
        trajectory_csv=str(trajectory),
        map_path=str(tmp_path / "map.pcd"),
        top_k=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )

    assert result["total_frames"] == 2
    frame = result["selected_frames"][0]
    assert frame["scan_id"] == "scan_1"
    assert frame["scan_match_failed"] is True
    assert "scan_match_failed:no_correspondences" in frame["reasons"]
    assert frame["final_pose"]["yaw_rad"] == 0.1
    assert "scan-match-debug" in frame["scan_match_debug_command"]
    assert "--initial-matrix" in frame["scan_match_debug_command"]
    assert result["commands"]["web"].startswith("ca web")
    assert result["commands"]["web_export"].startswith("ca web-export")


def test_slam_debug_cli_outputs_json(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "scan_id,timestamp_sec,scan_match_failed,scan_match_rmse_m,"
        "scan_match_correspondence_rejection_rate,prediction_delta_m,"
        "scan_match_vs_initial_pose_delta_m,registration_retry_count,"
        "consecutive_scan_match_failures,scan_quality_low,initial_x_m,initial_y_m,initial_z_m\n"
        "scan_0,0.0,false,0.5,0.2,0.1,0.1,0,0,false,0.0,0.0,0.0\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["slam-debug", str(metrics), "--format-json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total_frames"] == 1
    assert payload["selected_frames"][0]["scan_id"] == "scan_0"
    assert payload["selected_frames"][0]["diagnosis"]["label"] == "needs_review"


def test_diagnose_slam_frame_detects_bad_initial_guess():
    frame = {
        "glim_metrics": {
            "scan_match_rmse_m": 3.0,
            "prediction_delta_m": 0.2,
            "initial_delta_m": 1.2,
            "scan_quality_low": False,
            "raw_points": 2000,
            "filtered_points": 1800,
        },
        "scan_match_debug_result": {
            "distance_before": {"stats": {"mean": 1.3}},
            "distance_after": {"stats": {"mean": 0.4}},
            "improvement": {"mean": 0.9},
            "registration": {"fitness": 0.9, "inlier_rmse": 0.3},
            "preprocess": {"map_points_used": 500, "scan_points_used": 1000},
        },
    }

    diagnosis = diagnose_slam_frame(frame)

    assert diagnosis["label"] == "bad_initial_guess"
    assert diagnosis["confidence"] == "high"
    assert diagnosis["signals"]["improvement_mean"] == 0.9


def test_diagnose_slam_frame_detects_sparse_map_before_initial_guess():
    frame = {
        "glim_metrics": {
            "scan_match_rmse_m": 3.0,
            "initial_delta_m": 1.2,
            "scan_quality_low": False,
            "raw_points": 2000,
            "filtered_points": 1800,
        },
        "scan_match_debug_result": {
            "distance_before": {"stats": {"mean": 1.3}},
            "distance_after": {"stats": {"mean": 0.4}},
            "improvement": {"mean": 0.9},
            "registration": {"fitness": 0.9, "inlier_rmse": 0.3},
            "preprocess": {"map_points_used": 30, "scan_points_used": 1000},
        },
    }

    diagnosis = diagnose_slam_frame(frame)

    assert diagnosis["label"] == "map_too_sparse"
    assert diagnosis["suggested_action"].startswith("Inspect keyframe insertion")


def test_diagnose_slam_frame_distinguishes_aggressive_filtering():
    frame = {
        "glim_metrics": {
            "scan_match_rmse_m": 3.0,
            "scan_quality_low": True,
            "raw_points": 5000,
            "downsampled_points": 20,
            "filtered_points": 20,
            "raw_range_mean_m": 12.5,
            "filtered_range_mean_m": 11.9,
        },
    }

    diagnosis = diagnose_slam_frame(frame)

    assert diagnosis["label"] == "filtering_too_aggressive"
    assert diagnosis["signals"]["filtered_ratio"] == 20 / 5000
    assert diagnosis["signals"]["raw_range_mean_m"] == 12.5


def test_diagnose_slam_frame_detects_sparse_raw_scan():
    frame = {
        "glim_metrics": {
            "scan_match_rmse_m": 3.0,
            "scan_quality_low": True,
            "raw_points": 12,
            "downsampled_points": 10,
        },
    }

    diagnosis = diagnose_slam_frame(frame)

    assert diagnosis["label"] == "sparse_raw_scan"


def test_analyze_slam_run_preserves_glim_scan_quality_metrics(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "scan_id,timestamp_sec,scan_match_failed,scan_match_rmse_m,"
        "scan_match_correspondence_rejection_rate,prediction_delta_m,"
        "scan_match_vs_initial_pose_delta_m,registration_retry_count,"
        "consecutive_scan_match_failures,scan_quality_low,raw_points,"
        "downsampled_points,filtered_points,raw_range_min_m,raw_range_max_m,"
        "raw_range_mean_m,filtered_range_min_m,filtered_range_max_m,"
        "filtered_range_mean_m,initial_x_m,initial_y_m,initial_z_m\n"
        "scan_0,0.0,false,2.0,0.2,0.1,0.1,0,0,true,5000,"
        "20,20,1.0,30.0,12.5,1.5,20.0,11.9,0.0,0.0,0.0\n",
        encoding="utf-8",
    )

    result = analyze_slam_run(str(metrics), top_k=1)

    metrics_payload = result["selected_frames"][0]["glim_metrics"]
    assert metrics_payload["downsampled_points"] == 20
    assert metrics_payload["raw_range_mean_m"] == 12.5
    assert metrics_payload["filtered_range_mean_m"] == 11.9
    assert result["selected_frames"][0]["diagnosis"]["label"] == "filtering_too_aggressive"


def test_analyze_slam_run_can_execute_scan_match_debug(source_and_target_files, tmp_path):
    scan_path, map_path = source_and_target_files
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "scan_id,timestamp_sec,scan_match_failed,scan_match_rmse_m,"
        "scan_match_correspondence_rejection_rate,prediction_delta_m,"
        "scan_match_vs_initial_pose_delta_m,registration_retry_count,"
        "consecutive_scan_match_failures,scan_quality_low,initial_x_m,initial_y_m,initial_z_m\n"
        "scan_0,0.0,false,0.5,0.2,0.1,0.1,0,0,false,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "scans_manifest.csv"
    manifest.write_text(
        f"scan_id,timestamp_sec,points_csv\nscan_0,0.0,{scan_path}\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"

    result = analyze_slam_run(
        str(metrics),
        scans_manifest_csv=str(manifest),
        map_path=map_path,
        top_k=1,
        artifact_dir=str(artifact_dir),
        run_scan_match_debug_frames=True,
        scan_match_method="icp",
        scan_match_max_correspondence_distance=0.5,
        scan_match_threshold=0.05,
    )

    frame = result["selected_frames"][0]
    assert result["scan_match_debug_ran"] is True
    assert frame["scan_match_debug_error"] is None
    assert frame["scan_match_debug_result"]["method"] == "icp"
    assert frame["scan_match_debug_result"]["registration"]["fitness"] >= 0.0
    assert (artifact_dir / "01_scan_0" / "scan_initial_error.ply").exists()

    markdown = render_slam_debug_markdown(result)
    assert "SLAM Debug Report" in markdown
    assert "Diagnosis:" in markdown
    assert "CloudAnalyzer scan-match" in markdown
    assert "scan_initial_error_ply" in markdown


def test_slam_debug_cli_writes_markdown_with_scan_match_debug(
    source_and_target_files,
    tmp_path,
):
    scan_path, map_path = source_and_target_files
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "scan_id,timestamp_sec,scan_match_failed,scan_match_rmse_m,"
        "scan_match_correspondence_rejection_rate,prediction_delta_m,"
        "scan_match_vs_initial_pose_delta_m,registration_retry_count,"
        "consecutive_scan_match_failures,scan_quality_low,initial_x_m,initial_y_m,initial_z_m\n"
        "scan_0,0.0,false,0.5,0.2,0.1,0.1,0,0,false,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "scans_manifest.csv"
    manifest.write_text(
        f"scan_id,timestamp_sec,points_csv\nscan_0,0.0,{scan_path}\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.md"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "slam-debug",
            str(metrics),
            "--scans-manifest-csv",
            str(manifest),
            "--map",
            map_path,
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--top-k",
            "1",
            "--run-scan-match-debug",
            "--scan-match-method",
            "icp",
            "--scan-match-max-correspondence-distance",
            "0.5",
            "--output-markdown",
            str(report),
        ],
    )

    assert result.exit_code == 0
    assert "Markdown:" in result.stdout
    assert "diagnosis:" in result.stdout
    assert "ca:" in result.stdout
    report_text = report.read_text(encoding="utf-8")
    assert "Diagnosis:" in report_text
    assert "CloudAnalyzer scan-match" in report_text
