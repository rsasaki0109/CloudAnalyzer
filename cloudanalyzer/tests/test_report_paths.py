"""Tests for portable public report path helpers."""

from pathlib import Path

from ca.report_paths import make_paths_portable


def test_make_paths_portable_rewrites_nested_paths_and_commands(tmp_path):
    repo = tmp_path / "CloudAnalyzer"
    run_dir = repo / "docs" / "leaderboard" / "runs" / "kiss-icp__demo"
    reference = repo / "benchmarks" / "slam" / "demo" / "reference" / "map.pcd"
    candidate = run_dir / "map.ply"
    command = f"ca web '{candidate}' '{reference}' --heatmap"

    data = {
        "source_path": str(candidate),
        "nested": {
            "reference_path": str(reference),
            "commands": [command],
        },
    }

    portable = make_paths_portable(data, roots=(repo,))

    assert portable["source_path"] == "docs/leaderboard/runs/kiss-icp__demo/map.ply"
    assert portable["nested"]["reference_path"] == "benchmarks/slam/demo/reference/map.pcd"
    assert portable["nested"]["commands"] == [
        (
            "ca web 'docs/leaderboard/runs/kiss-icp__demo/map.ply' "
            "'benchmarks/slam/demo/reference/map.pcd' --heatmap"
        )
    ]


def test_make_paths_portable_can_scrub_output_dir_outside_repo(tmp_path):
    repo = tmp_path / "CloudAnalyzer"
    output_dir = tmp_path / "public-output"
    run_path = output_dir / "runs" / "demo" / "trajectory.tum"
    reference_path = repo / "benchmarks" / "slam" / "demo" / "trajectory.tum"

    data = {
        "trajectory_path": str(run_path),
        "reference_path": str(reference_path),
    }

    portable = make_paths_portable(data, roots=(repo, output_dir))

    assert portable["trajectory_path"] == "runs/demo/trajectory.tum"
    assert portable["reference_path"] == "benchmarks/slam/demo/trajectory.tum"
