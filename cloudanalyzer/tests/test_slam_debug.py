"""Tests for SLAM run diagnostics."""

import json

from typer.testing import CliRunner

from ca.slam_debug import analyze_slam_run
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

