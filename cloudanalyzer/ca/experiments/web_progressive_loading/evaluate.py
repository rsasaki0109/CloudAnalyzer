"""Benchmark and compare concrete progressive-loading implementations."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
import time

import numpy as np

from ca.core.web_progressive_loading import WebProgressiveLoadingRequest
from ca.experiments.web_progressive_loading import get_strategies
from ca.experiments.web_progressive_loading.common import (
    chunk_size_std,
    coverage_p95,
    progressive_prefix_points,
)


@dataclass(slots=True)
class ProgressiveDatasetCase:
    """Comparable inputs for browser progressive-loading experiments."""

    name: str
    description: str
    positions: np.ndarray
    initial_points: int
    chunk_points: int


def _corridor_dataset() -> np.ndarray:
    xs = np.linspace(0.0, 24.0, 64)
    ys = np.linspace(-2.0, 2.0, 20)
    wall_z = np.linspace(0.0, 3.0, 10)
    floor = np.array([[x, y, 0.0] for x in xs for y in ys], dtype=float)
    left_wall = np.array([[x, -2.0, z] for x in xs for z in wall_z], dtype=float)
    right_wall = np.array([[x, 2.0, z] for x in xs for z in wall_z], dtype=float)
    return np.vstack([floor, left_wall, right_wall])


def _clustered_dataset() -> np.ndarray:
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [6.0, 2.5, 0.5],
            [2.0, 8.0, 1.0],
            [8.0, 7.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.linspace(-0.8, 0.8, 9)
    points = []
    for center in centers:
        for x in offsets:
            for y in offsets:
                for z in (-0.3, 0.0, 0.3):
                    points.append(center + np.array([x, y, z]))
    return np.asarray(points, dtype=float)


def _multi_level_dataset() -> np.ndarray:
    base_x = np.linspace(0.0, 10.0, 40)
    base_y = np.linspace(0.0, 10.0, 40)
    ground = np.array([[x, y, 0.0] for x in base_x for y in base_y], dtype=float)
    ramp = np.array(
        [[x, 5.0 + 0.2 * np.sin(x), 0.2 * x] for x in np.linspace(0.0, 10.0, 120)],
        dtype=float,
    )
    bridge = np.array(
        [[x, y, 2.0] for x in np.linspace(2.0, 8.0, 25) for y in np.linspace(3.0, 7.0, 15)],
        dtype=float,
    )
    return np.vstack([ground, ramp, bridge])


def build_default_datasets() -> list[ProgressiveDatasetCase]:
    """Create deterministic datasets stressing different initial-loading tradeoffs."""

    corridor = _corridor_dataset()
    clustered = _clustered_dataset()
    multi_level = _multi_level_dataset()
    return [
        ProgressiveDatasetCase(
            name="corridor_run",
            description="Long corridor where initial payload should cover the full extent.",
            positions=corridor,
            initial_points=180,
            chunk_points=220,
        ),
        ProgressiveDatasetCase(
            name="clustered_yard",
            description="Separated clusters where front-loading one region is visibly bad.",
            positions=clustered,
            initial_points=160,
            chunk_points=180,
        ),
        ProgressiveDatasetCase(
            name="multi_level_room",
            description="Ground plane plus elevated structure that benefits from spatial coverage.",
            positions=multi_level,
            initial_points=220,
            chunk_points=260,
        ),
    ]


def _count_annotated_functions(functions: list[ast.FunctionDef | ast.AsyncFunctionDef]) -> int:
    annotated = 0
    for func in functions:
        args = list(func.args.args) + list(func.args.kwonlyargs)
        if func.args.vararg is not None:
            args.append(func.args.vararg)
        if func.args.kwarg is not None:
            args.append(func.args.kwarg)
        if all(arg.annotation is not None for arg in args) and func.returns is not None:
            annotated += 1
    return annotated


def static_source_analysis(module_path: Path) -> dict[str, float | int]:
    """Produce heuristic readability/extensibility metrics from source shape."""

    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match))
    ]
    nonempty_loc = sum(
        1 for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )
    max_function_lines = max(
        ((func.end_lineno or func.lineno) - func.lineno + 1) for func in functions
    ) if functions else 0
    documented_nodes = int(ast.get_docstring(tree) is not None)
    for node in [*functions, *classes]:
        documented_nodes += int(ast.get_docstring(node) is not None)
    documentable_nodes = 1 + len(functions) + len(classes)
    docstring_coverage = documented_nodes / documentable_nodes if documentable_nodes else 0.0
    annotated_ratio = _count_annotated_functions(functions) / len(functions) if functions else 0.0

    readability_score = max(
        0.0,
        min(
            100.0,
            100.0
            - (nonempty_loc * 0.3)
            - (len(branches) * 2.5)
            - (max_function_lines * 0.35)
            + (docstring_coverage * 18.0),
        ),
    )
    extensibility_score = max(
        0.0,
        min(
            100.0,
            30.0
            + (max(len(functions) - 1, 0) * 4.5)
            + (len(classes) * 6.5)
            + (annotated_ratio * 28.0)
            + (docstring_coverage * 12.0)
            - (max_function_lines * 0.2),
        ),
    )
    return {
        "loc": nonempty_loc,
        "function_count": len(functions),
        "class_count": len(classes),
        "branch_count": len(branches),
        "max_function_lines": max_function_lines,
        "annotated_ratio": round(annotated_ratio, 4),
        "docstring_coverage": round(docstring_coverage, 4),
        "readability_score": round(readability_score, 2),
        "extensibility_score": round(extensibility_score, 2),
    }


def _rank(values: dict[str, float], reverse: bool = False) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
    return {name: rank + 1 for rank, (name, _) in enumerate(ordered)}


def benchmark_strategy_on_dataset(strategy, dataset: ProgressiveDatasetCase, repetitions: int = 3) -> dict:
    """Measure planning cost and progressive coverage quality on one dataset."""

    runtimes_ms: list[float] = []
    last_result = None
    for _ in range(repetitions):
        request = WebProgressiveLoadingRequest(
            positions=dataset.positions.copy(),
            initial_points=dataset.initial_points,
            chunk_points=dataset.chunk_points,
            label=dataset.name,
        )
        start = time.perf_counter()
        result = strategy.plan(request)
        runtimes_ms.append((time.perf_counter() - start) * 1000.0)
        last_result = result

    assert last_result is not None
    prefixes = progressive_prefix_points(last_result)
    coverage_series = np.asarray(
        [coverage_p95(dataset.positions, prefix) for prefix in prefixes],
        dtype=float,
    )
    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "original_points": int(dataset.positions.shape[0]),
        "initial_points": last_result.initial_points,
        "chunk_points": last_result.chunk_points,
        "chunk_count": len(last_result.chunks),
        "runtime_ms": float(fmean(runtimes_ms)),
        "initial_coverage_p95": float(coverage_series[0]) if coverage_series.size else 0.0,
        "progressive_coverage_auc": float(np.mean(coverage_series)) if coverage_series.size else 0.0,
        "final_coverage_p95": float(coverage_series[-1]) if coverage_series.size else 0.0,
        "chunk_size_std": chunk_size_std(last_result),
        "initial_ratio": (
            float(last_result.initial_points / last_result.original_points)
            if last_result.original_points > 0
            else 0.0
        ),
        "metadata": dict(last_result.metadata),
    }


def summarize_strategy_results(rows: list[dict], analysis_rows: dict[str, dict]) -> list[dict]:
    """Aggregate per-dataset measurements into comparable strategy summaries."""

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["strategy"], []).append(row)

    summaries = []
    for strategy_name, group in grouped.items():
        summaries.append(
            {
                "strategy": strategy_name,
                "design": group[0]["design"],
                "module": group[0]["module"],
                "avg_runtime_ms": round(float(fmean(row["runtime_ms"] for row in group)), 4),
                "avg_initial_coverage_p95": round(
                    float(fmean(row["initial_coverage_p95"] for row in group)),
                    6,
                ),
                "avg_progressive_coverage_auc": round(
                    float(fmean(row["progressive_coverage_auc"] for row in group)),
                    6,
                ),
                "avg_chunk_size_std": round(float(fmean(row["chunk_size_std"] for row in group)), 6),
                "avg_initial_ratio": round(float(fmean(row["initial_ratio"] for row in group)), 4),
                "readability_score": analysis_rows[strategy_name]["readability_score"],
                "extensibility_score": analysis_rows[strategy_name]["extensibility_score"],
            }
        )

    quality_ranks = _rank(
        {
            item["strategy"]: item["avg_initial_coverage_p95"] + (item["avg_progressive_coverage_auc"] * 0.35)
            for item in summaries
        },
        reverse=False,
    )
    runtime_ranks = _rank({item["strategy"]: item["avg_runtime_ms"] for item in summaries})
    chunk_balance_ranks = _rank(
        {item["strategy"]: item["avg_chunk_size_std"] for item in summaries},
        reverse=False,
    )
    readability_ranks = _rank(
        {item["strategy"]: item["readability_score"] for item in summaries},
        reverse=True,
    )
    extensibility_ranks = _rank(
        {item["strategy"]: item["extensibility_score"] for item in summaries},
        reverse=True,
    )

    for item in summaries:
        strategy_name = item["strategy"]
        item["quality_rank"] = quality_ranks[strategy_name]
        item["runtime_rank"] = runtime_ranks[strategy_name]
        item["chunk_balance_rank"] = chunk_balance_ranks[strategy_name]
        item["readability_rank"] = readability_ranks[strategy_name]
        item["extensibility_rank"] = extensibility_ranks[strategy_name]
        item["composite_rank"] = round(
            (item["quality_rank"] * 0.5)
            + (item["chunk_balance_rank"] * 0.15)
            + (item["runtime_rank"] * 0.15)
            + (item["readability_rank"] * 0.1)
            + (item["extensibility_rank"] * 0.1),
            3,
        )
    return sorted(summaries, key=lambda item: item["composite_rank"])


def run_web_progressive_loading_experiment(
    datasets: list[ProgressiveDatasetCase] | None = None,
    repetitions: int = 3,
) -> dict:
    """Evaluate every concrete progressive-loading implementation."""

    datasets = datasets or build_default_datasets()
    strategies = get_strategies()
    rows: list[dict] = []
    analysis_rows: dict[str, dict] = {}

    for strategy in strategies:
        module_path = Path(inspect.getsourcefile(strategy.__class__) or "")
        analysis_rows[strategy.name] = static_source_analysis(module_path)
        for dataset in datasets:
            metrics = benchmark_strategy_on_dataset(
                strategy=strategy,
                dataset=dataset,
                repetitions=repetitions,
            )
            metrics["strategy"] = strategy.name
            metrics["design"] = strategy.design
            metrics["module"] = str(module_path)
            rows.append(metrics)

    summaries = summarize_strategy_results(rows, analysis_rows)
    winner = summaries[0]
    return {
        "problem": {
            "name": "web_progressive_loading",
            "statement": "Load `ca web` point clouds with a small initial payload and bounded deferred chunks.",
            "stable_interface_path": "cloudanalyzer/ca/core/web_progressive_loading.py",
            "experiment_package": "cloudanalyzer/ca/experiments/web_progressive_loading",
            "repetitions": repetitions,
        },
        "datasets": [
            {
                "name": dataset.name,
                "description": dataset.description,
                "points": int(dataset.positions.shape[0]),
                "initial_points": dataset.initial_points,
                "chunk_points": dataset.chunk_points,
            }
            for dataset in datasets
        ],
        "results": rows,
        "analysis": analysis_rows,
        "strategy_summaries": summaries,
        "decision": {
            "selected_experiment": winner["strategy"],
            "stabilized_core_strategy": "distance_shells",
            "reason": "Best initial spatial coverage with acceptable planning cost and no extra chunk imbalance.",
        },
    }


def render_experiment_section(report: dict) -> str:
    """Render a markdown section for docs/experiments.md."""

    dataset_lines = [
        "| Dataset | Points | Initial | Chunk | Purpose |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in report["datasets"]:
        dataset_lines.append(
            f"| {dataset['name']} | {dataset['points']} | {dataset['initial_points']} | "
            f"{dataset['chunk_points']} | {dataset['description']} |"
        )

    summary_lines = [
        "| Strategy | Design | Avg runtime ms | Initial coverage p95 | Progressive coverage AUC | Chunk std | Readability | Extensibility | Composite rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["strategy_summaries"]:
        summary_lines.append(
            "| "
            f"{item['strategy']} | {item['design']} | {item['avg_runtime_ms']:.4f} | "
            f"{item['avg_initial_coverage_p95']:.6f} | {item['avg_progressive_coverage_auc']:.6f} | "
            f"{item['avg_chunk_size_std']:.6f} | {item['readability_score']:.2f} | "
            f"{item['extensibility_score']:.2f} | {item['composite_rank']:.3f} |"
        )

    return (
        "## web_progressive_loading\n\n"
        f"{report['problem']['statement']}\n\n"
        "Stable code lives in `cloudanalyzer/ca/core/web_progressive_loading.py`. "
        "Discardable variants live in `cloudanalyzer/ca/experiments/web_progressive_loading`.\n\n"
        "### Shared Inputs\n\n"
        + "\n".join(dataset_lines)
        + "\n\n### Strategy Comparison\n\n"
        + "\n".join(summary_lines)
        + "\n\n### Notes\n\n"
        "- Quality is dominated by how well the initial payload covers the whole cloud.\n"
        "- Progressive coverage AUC tracks how quickly spatial holes disappear as chunks arrive.\n"
        "- Chunk size standard deviation penalizes visibly uneven deferred loads.\n"
    )


def render_experiments_markdown(report: dict) -> str:
    """Render progressive-loading experiment comparison results."""
    return "# Experiments\n\n" + render_experiment_section(report)


def render_decision_section(report: dict) -> str:
    """Render the decision section for progressive-loading."""

    selected = report["strategy_summaries"][0]
    rejected = report["strategy_summaries"][1:]
    selected_name = report["decision"]["selected_experiment"]
    stabilized_name = report["decision"]["stabilized_core_strategy"]
    adopted_line = (
        f"`{stabilized_name}` is adopted directly as the current core strategy."
        if selected_name == stabilized_name
        else f"`{stabilized_name}` is the stabilized core form of `{selected_name}`."
    )
    lines = [
        "## web_progressive_loading",
        "",
        "### Adopted",
        "",
        adopted_line,
        "",
        f"Reason: {report['decision']['reason']}",
        "",
        "### Not Adopted",
        "",
    ]
    for item in rejected:
        lines.append(
            f"- `{item['strategy']}` remains experimental. "
            f"Quality rank={item['quality_rank']}, chunk balance rank={item['chunk_balance_rank']}, "
            f"runtime rank={item['runtime_rank']}, readability rank={item['readability_rank']}, "
            f"extensibility rank={item['extensibility_rank']}."
        )
    lines.extend(
        [
            "",
            "### Trigger To Re-run",
            "",
            "- Initial browser payload budget changes materially.",
            "- `ca web` adds camera-aware or view-dependent loading.",
            "- A new chunk planning strategy is proposed.",
        ]
    )
    return "\n".join(lines)


def render_decisions_markdown(report: dict) -> str:
    """Render adoption and rejection reasoning."""
    return "# Decisions\n\n" + render_decision_section(report)


def render_interface_section(report: dict) -> str:
    """Render the smallest stable progressive-loading interface."""

    selected = report["decision"]["selected_experiment"]
    stabilized = report["decision"]["stabilized_core_strategy"]
    lineage_line = (
        f"- Current stabilized lineage: `{selected}` adopted directly in core\n"
        if selected == stabilized
        else f"- Current stabilized lineage: `{selected}` -> `{stabilized}`\n"
    )
    return (
        "## web_progressive_loading\n\n"
        "### Current Minimal Interface\n\n"
        "The stable interface keeps only the data needed to serve an initial point payload plus deferred chunks.\n\n"
        "```python\n"
        "class WebProgressiveLoadingStrategy(Protocol):\n"
        "    name: str\n"
        "    design: str\n"
        "    def plan(self, request: WebProgressiveLoadingRequest) -> WebProgressiveLoadingResult: ...\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class WebProgressiveLoadingRequest:\n"
        "    positions: np.ndarray\n"
        "    initial_points: int\n"
        "    chunk_points: int\n"
        "    distances: np.ndarray | None = None\n"
        "    label: str = \"point cloud\"\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class WebProgressiveLoadingResult:\n"
        "    initial_positions: np.ndarray\n"
        "    initial_distances: np.ndarray | None\n"
        "    chunks: tuple[WebProgressiveLoadingChunk, ...]\n"
        "    strategy: str\n"
        "    design: str\n"
        "    original_points: int\n"
        "    initial_points: int\n"
        "    chunk_points: int\n"
        "    metadata: dict[str, Any]\n"
        "```\n\n"
        "### Stable Boundary\n\n"
        "- Stable core: `cloudanalyzer/ca/core/web_progressive_loading.py`\n"
        "- Experimental space: `cloudanalyzer/ca/experiments/web_progressive_loading/`\n"
        + lineage_line
    )


def render_interfaces_markdown(report: dict) -> str:
    """Render the smallest stable interface left after comparison."""
    return "# Interfaces\n\n" + render_interface_section(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_web_progressive_loading_experiment(repetitions=args.repetitions), indent=2))


if __name__ == "__main__":
    main()
