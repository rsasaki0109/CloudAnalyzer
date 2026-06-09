#!/usr/bin/env python3
"""Build the public SLAM leaderboard under docs/leaderboard/.

For each driver × dataset pair the script:

1. Runs ``ca slam-run`` on the bundled synthetic scans
2. Scores the output with ``ca benchmark eval``
3. Writes per-run JSON, HTML report, and a static ``ca web-export`` viewer
4. Emits ``results.json`` plus a static ``index.html`` table for GitHub Pages

The generated numbers are snapshots — they are not checked into the regular
CI no-diff gate because SLAM runtime varies by hardware.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ca import __version__ as cloudanalyzer_version  # noqa: E402
from ca.benchmark import evaluate_benchmark_run, load_benchmark_suite  # noqa: E402
from ca.core.slam_run import (  # noqa: E402
    SlamRunRequest,
    discover_frame_paths,
    get_driver,
    write_map_ply,
    write_tum_trajectory,
)
from ca.report import save_run_report  # noqa: E402
from ca.web import export_static_bundle  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "leaderboard"
DEFAULT_DRIVERS = ("kiss-icp", "kiss-slam", "small-gicp")


@dataclass(frozen=True, slots=True)
class LeaderboardDataset:
    id: str
    suite_path: Path
    scans_dir: Path
    sequence: str = "default"
    max_range: float = 25.0
    voxel_size: float = 0.5
    frame_period: float = 0.1


DEFAULT_DATASETS: tuple[LeaderboardDataset, ...] = (
    LeaderboardDataset(
        id="synthetic-figure8",
        suite_path=REPO_ROOT / "benchmarks/slam/synthetic-figure8/suite.yaml",
        scans_dir=REPO_ROOT / "benchmarks/slam/synthetic-figure8/scans",
    ),
    LeaderboardDataset(
        id="synthetic-oval",
        suite_path=REPO_ROOT / "benchmarks/slam/synthetic-oval/suite.yaml",
        scans_dir=REPO_ROOT / "benchmarks/slam/synthetic-oval/scans",
    ),
)

OPTIONAL_DATASETS: tuple[LeaderboardDataset, ...] = (
    LeaderboardDataset(
        id="kitti-mini",
        suite_path=REPO_ROOT / "benchmarks/slam/kitti-mini/suite.yaml",
        scans_dir=REPO_ROOT / "benchmarks/slam/kitti-mini/scans",
        sequence="sequence_00",
        max_range=80.0,
        voxel_size=0.5,
        frame_period=0.1,
    ),
    LeaderboardDataset(
        id="newer-college-mini",
        suite_path=REPO_ROOT / "benchmarks/slam/newer-college-mini/suite.yaml",
        scans_dir=REPO_ROOT / "benchmarks/slam/newer-college-mini/scans",
        sequence="short_experiment",
        max_range=30.0,
        voxel_size=0.2,
        frame_period=0.1,
    ),
)


def _dataset_is_ready(dataset: LeaderboardDataset) -> bool:
    return dataset.suite_path.is_file() and dataset.scans_dir.is_dir()


def discover_datasets(
    *,
    include_optional: bool = False,
    dataset_ids: tuple[str, ...] | None = None,
) -> tuple[LeaderboardDataset, ...]:
    """Return bundled synthetic suites plus optional locally-prepared ones."""

    catalog: dict[str, LeaderboardDataset] = {
        dataset.id: dataset for dataset in (*DEFAULT_DATASETS, *OPTIONAL_DATASETS)
    }
    if dataset_ids:
        missing = [name for name in dataset_ids if name not in catalog]
        if missing:
            allowed = ", ".join(sorted(catalog))
            raise ValueError(f"Unknown dataset(s): {', '.join(missing)}. Allowed: {allowed}")
        selected = [catalog[name] for name in dataset_ids]
    else:
        selected = list(DEFAULT_DATASETS)
        if include_optional:
            for dataset in OPTIONAL_DATASETS:
                if _dataset_is_ready(dataset):
                    selected.append(dataset)

    not_ready = [
        dataset.id
        for dataset in selected
        if dataset.id not in {item.id for item in DEFAULT_DATASETS} and not _dataset_is_ready(dataset)
    ]
    if not_ready:
        raise FileNotFoundError(
            "Optional dataset(s) are not prepared locally: "
            + ", ".join(not_ready)
            + ". See benchmarks/slam/*/README.md and scripts/prepare_leaderboard_kitti.py."
        )
    return tuple(selected)


def _run_id(driver: str, dataset_id: str) -> str:
    return f"{driver}__{dataset_id}"


def _slam_command(
    driver: str,
    dataset: LeaderboardDataset,
    output_dir: Path,
) -> str:
    rel_out = output_dir.relative_to(REPO_ROOT)
    return (
        f"ca slam-run {dataset.scans_dir.relative_to(REPO_ROOT)} "
        f"{rel_out} "
        f"--driver {driver} "
        f"--max-range {dataset.max_range:g} "
        f"--voxel-size {dataset.voxel_size:g} "
        f"--frame-period {dataset.frame_period:g}"
    )


def _benchmark_command(
    dataset: LeaderboardDataset,
    run_dir: Path,
) -> str:
    rel_run = run_dir.relative_to(REPO_ROOT)
    return (
        f"ca benchmark eval {dataset.suite_path.relative_to(REPO_ROOT)} "
        f"--map {rel_run / 'map.ply'} "
        f"--trajectory {rel_run / 'trajectory.tum'} "
        f"--sequence {dataset.sequence} "
        f"--report {rel_run / 'report.html'}"
    )


def _extract_row_metrics(
    *,
    driver: str,
    dataset: LeaderboardDataset,
    slam_summary: dict[str, Any],
    benchmark_result: dict[str, Any],
    run_dir: Path,
    pages_root: Path,
) -> dict[str, Any]:
    map_metrics = benchmark_result["map"]
    traj_metrics = benchmark_result["trajectory"]
    overall = benchmark_result["overall_quality_gate"]
    rel = run_dir.relative_to(pages_root)

    return {
        "id": _run_id(driver, dataset.id),
        "driver": driver,
        "dataset": dataset.id,
        "sequence": dataset.sequence,
        "gate_passed": bool(overall and overall.get("passed")),
        "metrics": {
            "ate_rmse_m": float(traj_metrics["ate"]["rmse"]),
            "rpe_rmse_m": float(traj_metrics["rpe_translation"]["rmse"]),
            "drift_m": float(traj_metrics["drift"]["endpoint"]),
            "map_auc": float(map_metrics["auc"]),
            "chamfer_m": float(map_metrics["chamfer_distance"]),
            "runtime_s": float(slam_summary["runtime_s"]),
            "frames_processed": int(slam_summary["frames_processed"]),
        },
        "cloudanalyzer_version": cloudanalyzer_version,
        "links": {
            "benchmark_json": f"{rel.as_posix()}/benchmark.json",
            "slam_summary_json": f"{rel.as_posix()}/slam_summary.json",
            "report_html": f"{rel.as_posix()}/report.html",
            "viewer_html": f"{rel.as_posix()}/viewer/index.html",
        },
        "commands": {
            "slam_run": _slam_command(driver, dataset, run_dir),
            "benchmark_eval": _benchmark_command(dataset, run_dir),
        },
    }


def _run_one(
    driver: str,
    dataset: LeaderboardDataset,
    run_dir: Path,
    *,
    export_viewer: bool,
    viewer_max_points: int,
) -> dict[str, Any]:
    if not dataset.scans_dir.is_dir():
        raise FileNotFoundError(f"Missing scans directory: {dataset.scans_dir}")
    if not dataset.suite_path.is_file():
        raise FileNotFoundError(f"Missing suite manifest: {dataset.suite_path}")

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = discover_frame_paths(dataset.scans_dir)
    request = SlamRunRequest(
        frame_paths=tuple(frame_paths),
        frame_period_s=dataset.frame_period,
        max_range_m=dataset.max_range,
        voxel_size_m=dataset.voxel_size,
        deskew=False,
    )

    try:
        slam_driver = get_driver(driver)
    except (ImportError, ValueError) as exc:
        raise RuntimeError(f"Driver {driver!r} unavailable: {exc}") from exc

    result = slam_driver.run(request)
    trajectory_path = run_dir / "trajectory.tum"
    map_path = run_dir / "map.ply"
    write_tum_trajectory(trajectory_path, result.poses, result.timestamps_s)
    write_map_ply(map_path, result.map_points)

    slam_summary = {
        "driver": result.driver,
        "frames_processed": result.frames_processed,
        "runtime_s": float(result.runtime_s),
        "map_points": int(result.map_points.shape[0]),
        "trajectory_path": str(trajectory_path),
        "map_path": str(map_path),
        "driver_metadata": result.metadata,
    }
    (run_dir / "slam_summary.json").write_text(
        json.dumps(slam_summary, indent=2),
        encoding="utf-8",
    )

    suite = load_benchmark_suite(str(dataset.suite_path))
    seq = suite.resolve_sequence(dataset.sequence)
    benchmark_result = evaluate_benchmark_run(
        suite,
        str(map_path),
        str(trajectory_path),
        sequence=dataset.sequence,
    )
    (run_dir / "benchmark.json").write_text(
        json.dumps(benchmark_result, indent=2, default=str),
        encoding="utf-8",
    )
    save_run_report(benchmark_result, str(run_dir / "report.html"))

    if export_viewer:
        viewer_dir = run_dir / "viewer"
        export_static_bundle(
            [str(map_path), str(seq.reference_map_path)],
            output_dir=str(viewer_dir),
            max_points=viewer_max_points,
            heatmap=True,
            trajectory_path=str(trajectory_path),
            trajectory_reference_path=str(seq.reference_trajectory_path),
        )

    return slam_summary, benchmark_result


def build_leaderboard(
    output_dir: Path,
    *,
    drivers: tuple[str, ...] = DEFAULT_DRIVERS,
    datasets: tuple[LeaderboardDataset, ...] = DEFAULT_DATASETS,
    export_viewer: bool = True,
    viewer_max_points: int = 80_000,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for dataset in datasets:
        for driver in drivers:
            run_dir = runs_root / _run_id(driver, dataset.id)
            try:
                slam_summary, benchmark_result = _run_one(
                    driver,
                    dataset,
                    run_dir,
                    export_viewer=export_viewer,
                    viewer_max_points=viewer_max_points,
                )
                rows.append(
                    _extract_row_metrics(
                        driver=driver,
                        dataset=dataset,
                        slam_summary=slam_summary,
                        benchmark_result=benchmark_result,
                        run_dir=run_dir,
                        pages_root=output_dir,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - collect per-run failures
                errors.append(
                    {
                        "id": _run_id(driver, dataset.id),
                        "driver": driver,
                        "dataset": dataset.id,
                        "error": str(exc),
                    }
                )

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = {
        "title": "CloudAnalyzer SLAM Leaderboard",
        "generated_at": generated_at,
        "cloudanalyzer_version": cloudanalyzer_version,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "drivers": list(drivers),
        "datasets": [dataset.id for dataset in datasets],
        "rows": rows,
        "errors": errors,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(_render_index_html(), encoding="utf-8")
    (output_dir / "README.md").write_text(_render_readme(), encoding="utf-8")
    return payload


def _render_readme() -> str:
    return "\n".join(
        [
            "# SLAM Leaderboard (generated)",
            "",
            "Static snapshot produced by `scripts/build_leaderboard.py`.",
            "Open [`index.html`](index.html) on GitHub Pages or regenerate locally:",
            "",
            "```bash",
            "pip install -e './cloudanalyzer[slam]'",
            "python scripts/build_leaderboard.py --output docs/leaderboard",
            "# add locally-prepared KITTI / Newer College rows when available:",
            "python scripts/prepare_leaderboard_kitti.py --velodyne-dir ... --kitti-poses ...",
            "python scripts/build_leaderboard.py --include-optional --output docs/leaderboard",
            "```",
            "",
            "Optional real-world datasets (`kitti-mini`, `newer-college-mini`) are",
            "not bundled because of size and upstream licenses. Prepare them locally,",
            "then rebuild with `--include-optional`.",
            "",
        ]
    )


def _render_index_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CloudAnalyzer SLAM Leaderboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Newsreader:wght@500;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f6efe3;
      --ink: #16110f;
      --muted: #6a5b52;
      --panel: rgba(255, 251, 245, 0.88);
      --line: rgba(22, 17, 15, 0.12);
      --accent: #c24f2d;
      --pass: #166534;
      --fail: #991b1b;
      --shadow: 0 24px 80px rgba(27, 17, 11, 0.14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Space Grotesk", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(194, 79, 45, 0.12), transparent 34%),
        radial-gradient(circle at top right, rgba(31, 95, 114, 0.14), transparent 28%),
        linear-gradient(180deg, #f8f2e8 0%, var(--bg) 52%, #efe7d8 100%);
    }
    a { color: var(--accent); }
    .shell { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 56px; }
    .hero, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }
    .hero { padding: 30px 32px 26px; margin-bottom: 22px; }
    .eyebrow {
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }
    h1 {
      margin: 16px 0 10px;
      font-family: "Newsreader", serif;
      font-size: clamp(2.2rem, 4vw, 3.8rem);
      line-height: 0.98;
      letter-spacing: -0.04em;
      max-width: 14ch;
    }
    .lede { margin: 0; max-width: 70ch; color: var(--muted); line-height: 1.7; }
    .meta { margin-top: 16px; color: var(--muted); font-size: 0.92rem; }
    .actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 20px; }
    .button {
      display: inline-flex; align-items: center; justify-content: center;
      min-height: 42px; padding: 0 16px; border-radius: 999px;
      text-decoration: none; font-weight: 700; border: 1px solid var(--line);
      background: rgba(255,255,255,0.45); color: var(--ink);
    }
    .panel { padding: 18px 18px 22px; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
    tbody tr:hover { background: rgba(255,255,255,0.45); }
    .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
    .badge {
      display: inline-flex; align-items: center; border-radius: 999px;
      padding: 0.18rem 0.55rem; font-size: 0.78rem; font-weight: 700;
    }
    .badge-pass { background: #ecfdf5; color: var(--pass); }
    .badge-fail { background: #fef2f2; color: var(--fail); }
    .link-list { display: flex; flex-wrap: wrap; gap: 8px; }
    .link-chip {
      display: inline-flex; padding: 0.18rem 0.55rem; border-radius: 999px;
      border: 1px solid var(--line); text-decoration: none; font-size: 0.78rem;
      background: rgba(255,255,255,0.55); color: var(--ink);
    }
    code { font-size: 0.78rem; white-space: pre-wrap; word-break: break-word; color: #374151; }
    .error-box {
      margin-top: 18px; padding: 14px 16px; border-radius: 18px;
      border: 1px solid #fecaca; background: #fef2f2; color: #7f1d1d;
    }
    @media (max-width: 900px) {
      .shell { width: min(100vw - 20px, 1180px); }
      .hero, .panel { border-radius: 22px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">CloudAnalyzer / SLAM Leaderboard</div>
      <h1>LiDAR SLAM regression snapshots on bundled synthetic suites.</h1>
      <p class="lede">
        Each row is a full <code>ca slam-run</code> followed by <code>ca benchmark eval</code>
        against a frozen reference map, trajectory, and gate. Numbers are hardware snapshots —
        regenerate locally or via the monthly GitHub Action when drivers change.
      </p>
      <div class="actions">
        <a class="button" href="https://github.com/rsasaki0109/CloudAnalyzer">Repository</a>
        <a class="button" href="../index.html">Public Demo Hub</a>
      </div>
      <div class="meta" id="meta"></div>
    </section>

    <section class="panel">
      <table>
        <thead>
          <tr>
            <th>Driver</th>
            <th>Dataset</th>
            <th>ATE</th>
            <th>RPE</th>
            <th>Drift</th>
            <th>Map AUC</th>
            <th>Chamfer</th>
            <th>Runtime</th>
            <th>Gate</th>
            <th>Links</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
      <div id="errors"></div>
    </section>
  </main>
  <script>
    function fmt(value, digits) {
      if (value === null || value === undefined || Number.isNaN(value)) return "—";
      return Number(value).toFixed(digits);
    }

    function renderRows(payload) {
      const tbody = document.getElementById("rows");
      tbody.innerHTML = "";
      for (const row of payload.rows) {
        const m = row.metrics;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${row.driver}</strong><div class="meta">v${row.cloudanalyzer_version}</div></td>
          <td>${row.dataset}</td>
          <td class="num">${fmt(m.ate_rmse_m, 3)} m</td>
          <td class="num">${fmt(m.rpe_rmse_m, 3)} m</td>
          <td class="num">${fmt(m.drift_m, 3)} m</td>
          <td class="num">${fmt(m.map_auc, 4)}</td>
          <td class="num">${fmt(m.chamfer_m, 4)} m</td>
          <td class="num">${fmt(m.runtime_s, 2)} s</td>
          <td><span class="badge ${row.gate_passed ? "badge-pass" : "badge-fail"}">${row.gate_passed ? "PASS" : "FAIL"}</span></td>
          <td>
            <div class="link-list">
              <a class="link-chip" href="${row.links.viewer_html}">viewer</a>
              <a class="link-chip" href="${row.links.report_html}">report</a>
              <a class="link-chip" href="${row.links.benchmark_json}">json</a>
            </div>
            <details><summary>commands</summary>
              <code>${row.commands.slam_run}\\n${row.commands.benchmark_eval}</code>
            </details>
          </td>`;
        tbody.appendChild(tr);
      }

      const meta = document.getElementById("meta");
      meta.textContent = `Generated ${payload.generated_at} · CloudAnalyzer v${payload.cloudanalyzer_version} · ${payload.rows.length} runs`;

      const errors = document.getElementById("errors");
      if (!payload.errors.length) {
        errors.innerHTML = "";
        return;
      }
      errors.innerHTML = `<div class="error-box"><strong>Failed runs</strong><ul>${payload.errors.map(e => `<li><code>${e.id}</code>: ${e.error}</li>`).join("")}</ul></div>`;
    }

    fetch("./results.json")
      .then(r => r.json())
      .then(renderRows)
      .catch(err => {
        document.getElementById("meta").textContent = `Failed to load results.json: ${err}`;
      });
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Leaderboard output directory (default: docs/leaderboard/)",
    )
    parser.add_argument(
        "--driver",
        action="append",
        dest="drivers",
        help="SLAM driver to include (repeatable; default: all built-ins)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset id to include (repeatable; default: bundled synthetic suites)",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also include locally-prepared kitti-mini / newer-college-mini when ready",
    )
    parser.add_argument(
        "--skip-viewer",
        action="store_true",
        help="Skip ca web-export viewer bundles (faster local rebuild)",
    )
    parser.add_argument(
        "--viewer-max-points",
        type=int,
        default=80_000,
        help="Point budget for each static viewer bundle",
    )
    args = parser.parse_args()

    drivers = tuple(args.drivers) if args.drivers else DEFAULT_DRIVERS
    dataset_ids = tuple(args.datasets) if args.datasets else None
    datasets = discover_datasets(
        include_optional=args.include_optional,
        dataset_ids=dataset_ids,
    )

    payload = build_leaderboard(
        args.output,
        drivers=drivers,
        datasets=datasets,
        export_viewer=not args.skip_viewer,
        viewer_max_points=args.viewer_max_points,
    )
    print(f"Wrote leaderboard to {args.output}")
    print(f"  rows:   {len(payload['rows'])}")
    print(f"  errors: {len(payload['errors'])}")
    if payload["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
