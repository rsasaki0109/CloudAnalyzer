"""Benchmark and document concrete map evaluation strategies (MapEval-inspired)."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from statistics import fmean

from ca.experiments.map_evaluate import get_map_evaluate_strategies
from ca.experiments.map_evaluate.common import MapEvaluateDatasetCase, build_default_datasets


def benchmark_strategy_on_dataset(strategy, dataset: MapEvaluateDatasetCase, repetitions: int = 3) -> dict:
    runtimes_ms: list[float] = []
    last = None
    for _ in range(repetitions):
        start = time.perf_counter()
        last = strategy.evaluate(dataset.request)
        runtimes_ms.append((time.perf_counter() - start) * 1000.0)
    assert last is not None
    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "runtime_ms": float(fmean(runtimes_ms)),
        "metrics": last.metrics,
        "metric_family": last.metric_family,
        "reference_required": last.reference_required,
        "mode": last.mode,
        "sampling_policy": last.sampling_policy,
    }


def run_map_evaluate_experiment(
    datasets: list[MapEvaluateDatasetCase] | None = None,
    repetitions: int = 3,
) -> dict:
    datasets = datasets or build_default_datasets()
    strategies = get_map_evaluate_strategies()
    rows: list[dict] = []

    for strategy in strategies:
        module_path = Path(inspect.getsourcefile(strategy.__class__) or "")
        for dataset in datasets:
            try:
                row = benchmark_strategy_on_dataset(strategy, dataset, repetitions=repetitions)
                row["strategy"] = strategy.name
                row["design"] = strategy.design
                row["module"] = str(module_path)
                rows.append(row)
            except Exception as e:  # keep experiment report flowing
                rows.append(
                    {
                        "dataset": dataset.name,
                        "description": dataset.description,
                        "runtime_ms": None,
                        "strategy": strategy.name,
                        "design": strategy.design,
                        "module": str(module_path),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

    return {
        "problem": {
            "name": "map_evaluate",
            "statement": (
                "Evaluate point-cloud maps either against a reference (GT-based distance/coverage) "
                "or without GT (self-consistency proxies)."
            ),
            "experiment_package": "cloudanalyzer/ca/experiments/map_evaluate",
            "repetitions": repetitions,
        },
        "datasets": [
            {"name": d.name, "description": d.description, "has_gt": d.request.reference_points is not None}
            for d in datasets
        ],
        "results": rows,
        "decision": {
            "selected_experiment": None,
            "stabilized_core_strategy": None,
            "reason": "Not yet promoted to core; keep iterating on metrics and IO formats.",
        },
    }


def render_experiment_section(report: dict) -> str:
    dataset_lines = [
        "| Dataset | Has GT | Purpose |",
        "|---|---:|---|",
    ]
    for d in report["datasets"]:
        dataset_lines.append(f"| {d['name']} | {str(d['has_gt']).lower()} | {d['description']} |")

    family_lines = [
        "| Strategy | Metric family | Reference required | Mode |",
        "|---|---|---:|---|",
    ]
    seen_strategies: set[str] = set()
    for row in report.get("results", []):
        strat = row.get("strategy")
        # Skip error rows (they lack classification fields) and duplicates.
        if not strat or strat in seen_strategies or "metric_family" not in row:
            continue
        seen_strategies.add(strat)
        family_lines.append(
            "| {strat} | {family} | {ref} | {mode} |".format(
                strat=strat,
                family=row.get("metric_family", "unspecified"),
                ref=str(bool(row.get("reference_required", False))).lower(),
                mode=row.get("mode", "exact"),
            )
        )

    return (
        "## map_evaluate\n\n"
        f"{report['problem']['statement']}\n\n"
        "Experimental code lives in `cloudanalyzer/ca/experiments/map_evaluate/`.\n\n"
        "### Shared Inputs\n\n"
        + "\n".join(dataset_lines)
        + "\n\n### Metric Families\n\n"
        + "\n".join(family_lines)
        + "\n\n### Notes\n\n"
        "- This slice is MapEval-inspired: it keeps a threshold list (accuracy levels) and separates GT-based vs GT-free evaluation.\n"
        "- Reference-based and reference-free metrics live in separate `metric_family` lanes (`reference_based_nn_thresholds`, `reference_free_voxel_consistency`); CI gates should pick one family per check.\n"
        "- Current implementations are lightweight proxies; real-map scale should switch to KD-trees and richer outputs.\n"
    )


def render_decision_section(report: dict) -> str:
    return "\n".join(
        [
            "## map_evaluate",
            "",
            "### Adopted",
            "",
            "No strategy is adopted yet (still experimental).",
            "",
            "### Not Adopted",
            "",
            "- All strategies remain experimental until we settle IO formats and performance constraints.",
            "",
            "### Trigger To Promote",
            "",
            "- A stable request/result contract is needed by the CLI or library callers.",
            "- We have at least two strategies with clear trade-offs and representative datasets.",
        ]
    )


def render_interface_section(report: dict) -> str:
    return "\n".join(
        [
            "## map_evaluate",
            "",
            "### Current Minimal Interface (experimental)",
            "",
            "Not promoted to `ca/core` yet. Current request/result shapes live in "
            "`cloudanalyzer/ca/experiments/map_evaluate/common.py`.",
            "",
            "Result objects carry classification fields so reference-based and reference-free metrics stay in separate lanes:",
            "",
            "- `metric_family` — stable id (e.g. `reference_based_nn_thresholds`, `reference_free_voxel_consistency`).",
            "- `reference_required` — whether the strategy needs a reference map.",
            "- `mode` — `exact` / `voxelized` / `sampled`; records what approximation was used.",
            "- `sampling_policy` — structured record of voxel sizes, thresholds, and alignment used to produce the metrics.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_map_evaluate_experiment(repetitions=args.repetitions), indent=2, default=str))


if __name__ == "__main__":
    main()

