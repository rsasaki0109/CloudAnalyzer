"""Benchmark and document concrete slam_run drivers."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from statistics import fmean

from ca.experiments.slam_run import get_slam_run_drivers
from ca.experiments.slam_run.common import (
    SlamRunDatasetCase,
    absolute_trajectory_error_m,
    build_default_datasets,
    make_temp_dir,
)


def benchmark_driver_on_dataset(driver, dataset: SlamRunDatasetCase, repetitions: int = 1) -> dict:
    """Run ``driver`` on ``dataset`` and report ATE + qualitative outcome.

    Repetitions default to 1 because SLAM drivers are slow per run and the
    decision text in the doc is qualitative anyway. Tests can pass higher
    counts for stability checks.
    """

    ates: list[float] = []
    runtimes_ms: list[float] = []
    frames_processed = 0
    for _ in range(max(1, repetitions)):
        with make_temp_dir() as tmp:
            request = dataset.build_request(Path(tmp))
            start = time.perf_counter()
            result = driver.run(request)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            ate = absolute_trajectory_error_m(result.poses, dataset.gt_poses)
            ates.append(ate)
            runtimes_ms.append(elapsed_ms)
            frames_processed = result.frames_processed
    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "ate_m": float(fmean(ates)),
        "runtime_ms": float(fmean(runtimes_ms)),
        "frames_processed": frames_processed,
        "expected_kiss_icp_max_ate_m": dataset.expected_kiss_icp_max_ate_m,
    }


def run_slam_run_experiment(
    datasets: list[SlamRunDatasetCase] | None = None,
    repetitions: int = 1,
) -> dict:
    datasets = datasets or build_default_datasets()
    drivers = get_slam_run_drivers()
    rows: list[dict] = []

    for driver in drivers:
        module_path = Path(inspect.getsourcefile(driver.__class__) or "")
        for dataset in datasets:
            try:
                row = benchmark_driver_on_dataset(driver, dataset, repetitions=repetitions)
                row["driver"] = driver.name
                row["module"] = str(module_path)
                rows.append(row)
            except Exception as e:
                rows.append(
                    {
                        "dataset": dataset.name,
                        "description": dataset.description,
                        "driver": driver.name,
                        "module": str(module_path),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

    return {
        "problem": {
            "name": "slam_run",
            "statement": (
                "Drive a LiDAR-odometry pipeline end-to-end across a sequence of "
                "input scans and emit a trajectory + accumulated map that the "
                "rest of CloudAnalyzer's evaluation stack consumes."
            ),
            "experiment_package": "cloudanalyzer/ca/experiments/slam_run",
            "repetitions": repetitions,
        },
        "datasets": [
            {
                "name": d.name,
                "description": d.description,
                "n_frames": int(d.gt_poses.shape[0]),
            }
            for d in datasets
        ],
        "results": rows,
        "decision": {
            "selected_experiment": "kiss_icp",
            "stabilized_core_strategy": "KissICPSlamDriver",
            "reason": (
                "KISS-ICP wraps a well-known scan-to-map LiDAR-odometry pipeline "
                "available on PyPI under a BSD license. KISS-SLAM (its own pose-"
                "graph + loop-closure extension) is now also in the bake-off but "
                "does not consistently beat KISS-ICP on the synthetic cases the "
                "slice ships — the trajectories are too short to trigger a loop "
                "closure, so KISS-SLAM degrades to one round of pose-graph "
                "optimization over KISS-ICP's odometry chain. KISS-ICP stays "
                "core for the smaller, faster surface area; KISS-SLAM is held in "
                "experiments and will be promoted once dogfood data with real "
                "drift / revisits lands. identity_passthrough is the failure "
                "floor (zero-motion sentinel)."
            ),
        },
    }


def render_experiment_section(report: dict) -> str:
    dataset_lines = [
        "| Dataset | n_frames | Purpose |",
        "|---|---:|---|",
    ]
    for d in report["datasets"]:
        dataset_lines.append(
            f"| {d['name']} | {d['n_frames']} | {d['description']} |"
        )

    driver_lines = [
        "| Driver | Role |",
        "|---|---|",
        "| `kiss_icp` | Adopted — wraps the `kiss-icp` package (BSD, PyPI). |",
        "| `kiss_slam` | Experimental upgrade contender — wraps `kiss-slam` (KISS-ICP + pose-graph + MapClosures loop closure). Equivalent to `kiss_icp` on the synthetic cases that ship; promotion blocked on real-data dogfood with drift / revisits. |",
        "| `identity_passthrough` | Sentinel — identity poses + concatenated scans. Sets the failure floor. |",
    ]

    return (
        "## slam_run\n\n"
        f"{report['problem']['statement']}\n\n"
        "Experimental code lives in `cloudanalyzer/ca/experiments/slam_run/`.\n\n"
        "### Shared Inputs\n\n"
        + "\n".join(dataset_lines)
        + "\n\n### Drivers Compared\n\n"
        + "\n".join(driver_lines)
        + "\n\n### Notes\n\n"
        "- Datasets are synthesised into a temp directory at evaluation time and discarded after the run.\n"
        "- ATE / runtime numbers are kept out of the docs because driver runtime is hardware-sensitive; CI tests pin pass thresholds instead.\n"
        "- `KissICPSlamDriver` is also re-exported from `ca.core.slam_run` so the CLI stays on `ca.core`.\n"
    )


def render_decision_section(report: dict) -> str:
    return "\n".join(
        [
            "## slam_run",
            "",
            "### Adopted",
            "",
            "- `kiss_icp` (`KissICPSlamDriver`). Wraps `kiss-icp` and is also",
            "  exposed from `ca/core/slam_run.py` so `cloudanalyzer_cli` and",
            "  `ca.benchmark` can depend only on `ca.core`.",
            "",
            "### Not Adopted",
            "",
            "- `kiss_slam` (`KissSLAMSlamDriver`). Wraps `kiss-slam` (KISS-ICP",
            "  + pose-graph optimization + MapClosures loop closure). On the",
            "  short synthetic trajectories the slice ships, sensor displacement",
            "  from origin never crosses the local-map splitting distance, so",
            "  KISS-SLAM degenerates to one round of PGO over the KISS-ICP",
            "  odometry chain and produces the same trajectory KISS-ICP does.",
            "  Held in experiments and re-evaluated once real-drift / revisit",
            "  data lands.",
            "- `identity_passthrough` is a sentinel: it returns identity poses",
            "  and concatenates the input frames as the 'map'. Its job is to",
            "  fail loudly on any case that has non-trivial motion so that a",
            "  regression in the real driver doesn't slip through.",
            "",
            "### Triggers To Reconsider",
            "",
            "- KITTI / Newer-College mini fixtures get wired through",
            "  `ca slam-run` and KISS-SLAM's loop-closure / pose-graph kicks",
            "  in. The KISS-ICP vs KISS-SLAM gap on those sequences flips the",
            "  default driver.",
            "- A third real driver lands (e.g. `small_gicp` for faster GICP).",
            "  Different speed / accuracy operating point may justify a",
            "  driver-per-budget pick rather than a single core driver.",
        ]
    )


def render_interface_section(report: dict) -> str:
    return "\n".join(
        [
            "## slam_run",
            "",
            "### Current Minimal Interface",
            "",
            "Promoted to `ca/core/slam_run.py`. The request/result contract,",
            "frame loaders, trajectory/map writers, and the adopted",
            "`KissICPSlamDriver` re-export all live there.",
            "",
            "- `SlamRunRequest` — frame paths, optional per-frame timestamps,",
            "  driver knobs (`max_range_m`, `voxel_size_m`, `deskew`,",
            "  `max_frames`).",
            "- `SlamRunResult` — `(N, 4, 4)` poses, per-pose timestamps,",
            "  accumulated map ndarray, runtime, frames processed, and a",
            "  driver-specific metadata block snapshotted into the summary",
            "  JSON so two runs can be proven to share config.",
            "- `SlamRunDriver` — `@runtime_checkable` Protocol; concrete",
            "  drivers live under `ca/experiments/slam_run/` and are picked up",
            "  by the slice evaluator.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_slam_run_experiment(repetitions=args.repetitions), indent=2, default=str))


if __name__ == "__main__":
    main()
