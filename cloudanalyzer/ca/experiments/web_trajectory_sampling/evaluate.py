"""Benchmark and compare concrete trajectory simplification implementations."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
import time

import numpy as np

from ca.core.web_trajectory_sampling import WebTrajectorySamplingRequest
from ca.experiments.web_trajectory_sampling import get_strategies
from ca.experiments.web_trajectory_sampling.common import path_length, reconstruct_positions


@dataclass(slots=True)
class TrajectoryDatasetCase:
    """Comparable input for trajectory simplification experiments."""

    name: str
    description: str
    timestamps: np.ndarray
    positions: np.ndarray
    max_points: int
    preserve_indices: tuple[int, ...] = ()


def _piecewise_path(segments: list[tuple[np.ndarray, np.ndarray, int]]) -> np.ndarray:
    chunks = []
    for index, (start, end, steps) in enumerate(segments):
        t = np.linspace(0.0, 1.0, steps, endpoint=(index == len(segments) - 1))
        chunk = start[None, :] + (end - start)[None, :] * t[:, None]
        if index > 0:
            chunk = chunk[1:]
        chunks.append(chunk)
    return np.vstack(chunks)


def build_default_datasets() -> list[TrajectoryDatasetCase]:
    """Create deterministic trajectory shapes with different simplification stress."""

    straight_positions = _piecewise_path(
        [
            (np.array([0.0, 0.0, 0.0]), np.array([25.0, 0.0, 0.0]), 1600),
        ]
    )
    turn_positions = _piecewise_path(
        [
            (np.array([0.0, 0.0, 0.0]), np.array([12.0, 0.0, 0.0]), 700),
            (np.array([12.0, 0.0, 0.0]), np.array([12.0, 9.0, 0.0]), 700),
            (np.array([12.0, 9.0, 0.0]), np.array([19.0, 9.0, 0.0]), 500),
        ]
    )

    zigzag_segments = []
    current = np.array([0.0, 0.0, 0.0])
    for step in range(8):
        next_point = np.array([current[0] + 4.0, 1.6 if step % 2 == 0 else -1.6, 0.0])
        zigzag_segments.append((current.copy(), next_point.copy(), 260))
        current = next_point
    zigzag_positions = _piecewise_path(zigzag_segments)

    datasets = [
        TrajectoryDatasetCase(
            name="straight_corridor",
            description="Mostly straight trajectory where aggressive decimation is acceptable.",
            timestamps=np.linspace(0.0, 159.9, straight_positions.shape[0]),
            positions=straight_positions,
            max_points=120,
            preserve_indices=(straight_positions.shape[0] // 2,),
        ),
        TrajectoryDatasetCase(
            name="right_angle_turn",
            description="Single sharp turn that should remain visible after simplification.",
            timestamps=np.linspace(0.0, 189.9, turn_positions.shape[0]),
            positions=turn_positions,
            max_points=140,
            preserve_indices=(699, 700),
        ),
        TrajectoryDatasetCase(
            name="switchback",
            description="Repeated turns where turn preservation matters more than pure stride.",
            timestamps=np.linspace(0.0, 207.9, zigzag_positions.shape[0]),
            positions=zigzag_positions,
            max_points=150,
            preserve_indices=(zigzag_positions.shape[0] // 3, (zigzag_positions.shape[0] * 2) // 3),
        ),
    ]
    return datasets


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


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


def benchmark_strategy_on_dataset(strategy, dataset: TrajectoryDatasetCase, repetitions: int = 3) -> dict:
    """Measure runtime and shape reconstruction quality for one strategy and one dataset."""

    runtimes_ms: list[float] = []
    last_result = None
    for _ in range(repetitions):
        request = WebTrajectorySamplingRequest(
            positions=dataset.positions.copy(),
            timestamps=dataset.timestamps.copy(),
            max_points=dataset.max_points,
            label=dataset.name,
            preserve_indices=dataset.preserve_indices,
        )
        start = time.perf_counter()
        result = strategy.reduce(request)
        runtimes_ms.append((time.perf_counter() - start) * 1000.0)
        last_result = result

    assert last_result is not None
    reconstructed = reconstruct_positions(
        original_timestamps=dataset.timestamps,
        sampled_timestamps=last_result.timestamps if last_result.timestamps is not None else dataset.timestamps[last_result.kept_indices],
        sampled_positions=last_result.positions,
    )
    point_errors = np.linalg.norm(reconstructed - dataset.positions, axis=1)
    error_summary = _summary(point_errors)
    original_length = path_length(dataset.positions)
    reduced_length = path_length(last_result.positions)
    requested_preserve = set(
        index
        for index in dataset.preserve_indices
        if 0 <= index < dataset.positions.shape[0]
    )
    kept = set(int(index) for index in last_result.kept_indices.tolist())
    preserve_ratio = (
        len(requested_preserve & kept) / len(requested_preserve)
        if requested_preserve
        else 1.0
    )
    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "original_points": int(dataset.positions.shape[0]),
        "target_points": dataset.max_points,
        "reduced_points": last_result.reduced_points,
        "runtime_ms": float(fmean(runtimes_ms)),
        "mean_error": error_summary["mean"],
        "p95_error": error_summary["p95"],
        "max_error": error_summary["max"],
        "path_length_delta": abs(reduced_length - original_length),
        "preserve_ratio": float(preserve_ratio),
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
                "avg_mean_error": round(float(fmean(row["mean_error"] for row in group)), 6),
                "avg_p95_error": round(float(fmean(row["p95_error"] for row in group)), 6),
                "avg_path_length_delta": round(float(fmean(row["path_length_delta"] for row in group)), 6),
                "avg_preserve_ratio": round(float(fmean(row["preserve_ratio"] for row in group)), 4),
                "readability_score": analysis_rows[strategy_name]["readability_score"],
                "extensibility_score": analysis_rows[strategy_name]["extensibility_score"],
            }
        )

    quality_ranks = _rank(
        {
            item["strategy"]: item["avg_mean_error"] + (item["avg_path_length_delta"] * 0.05)
            for item in summaries
        },
        reverse=False,
    )
    runtime_ranks = _rank(
        {item["strategy"]: item["avg_runtime_ms"] for item in summaries},
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
    preserve_ranks = _rank(
        {item["strategy"]: item["avg_preserve_ratio"] for item in summaries},
        reverse=True,
    )

    for item in summaries:
        strategy_name = item["strategy"]
        item["quality_rank"] = quality_ranks[strategy_name]
        item["runtime_rank"] = runtime_ranks[strategy_name]
        item["readability_rank"] = readability_ranks[strategy_name]
        item["extensibility_rank"] = extensibility_ranks[strategy_name]
        item["preserve_rank"] = preserve_ranks[strategy_name]
        item["composite_rank"] = round(
            (item["quality_rank"] * 0.5)
            + (item["preserve_rank"] * 0.2)
            + (item["runtime_rank"] * 0.1)
            + (item["readability_rank"] * 0.1)
            + (item["extensibility_rank"] * 0.1),
            3,
        )

    return sorted(summaries, key=lambda item: item["composite_rank"])


def run_web_trajectory_sampling_experiment(
    datasets: list[TrajectoryDatasetCase] | None = None,
    repetitions: int = 3,
) -> dict:
    """Evaluate every concrete trajectory simplification implementation."""

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
            "name": "web_trajectory_sampling",
            "statement": "Reduce `ca web` trajectory overlays without losing inspection-critical poses.",
            "stable_interface_path": "cloudanalyzer/ca/core/web_trajectory_sampling.py",
            "experiment_package": "cloudanalyzer/ca/experiments/web_trajectory_sampling",
            "repetitions": repetitions,
        },
        "datasets": [
            {
                "name": dataset.name,
                "description": dataset.description,
                "points": int(dataset.positions.shape[0]),
                "max_points": dataset.max_points,
                "preserve_indices": list(dataset.preserve_indices),
            }
            for dataset in datasets
        ],
        "results": rows,
        "analysis": analysis_rows,
        "strategy_summaries": summaries,
        "decision": {
            "selected_experiment": winner["strategy"],
            "stabilized_core_strategy": "turn_aware",
            "reason": "Best composite rank while preserving anchors and reducing geometric distortion.",
        },
    }


def render_experiment_section(report: dict) -> str:
    """Render a markdown section for docs/experiments.md."""

    dataset_lines = [
        "| Dataset | Points | Budget | Preserve | Purpose |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in report["datasets"]:
        dataset_lines.append(
            f"| {dataset['name']} | {dataset['points']} | {dataset['max_points']} | "
            f"{len(dataset['preserve_indices'])} | {dataset['description']} |"
        )

    summary_lines = [
        "| Strategy | Design | Avg runtime ms | Avg mean error | Avg p95 error | Preserve ratio | Readability | Extensibility | Composite rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["strategy_summaries"]:
        summary_lines.append(
            "| "
            f"{item['strategy']} | {item['design']} | {item['avg_runtime_ms']:.4f} | "
            f"{item['avg_mean_error']:.6f} | {item['avg_p95_error']:.6f} | "
            f"{item['avg_preserve_ratio']:.4f} | {item['readability_score']:.2f} | "
            f"{item['extensibility_score']:.2f} | {item['composite_rank']:.3f} |"
        )

    return (
        "## web_trajectory_sampling\n\n"
        f"{report['problem']['statement']}\n\n"
        "Stable code lives in `cloudanalyzer/ca/core/web_trajectory_sampling.py`. "
        "Discardable variants live in `cloudanalyzer/ca/experiments/web_trajectory_sampling`.\n\n"
        "### Shared Inputs\n\n"
        + "\n".join(dataset_lines)
        + "\n\n### Strategy Comparison\n\n"
        + "\n".join(summary_lines)
        + "\n\n### Notes\n\n"
        "- Quality uses reconstructed position error together with path-length preservation.\n"
        "- Preserve ratio tracks whether inspection-critical anchors survive simplification.\n"
        "- Readability and extensibility scores are heuristic and generated from AST/source-shape metrics.\n"
    )


def render_experiments_markdown(report: dict) -> str:
    """Render trajectory experiment comparison results."""
    return "# Experiments\n\n" + render_experiment_section(report)


def render_decision_section(report: dict) -> str:
    """Render the decision section for trajectory web reduction."""
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
        "## web_trajectory_sampling",
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
            f"Quality rank={item['quality_rank']}, preserve rank={item['preserve_rank']}, "
            f"runtime rank={item['runtime_rank']}, readability rank={item['readability_rank']}, "
            f"extensibility rank={item['extensibility_rank']}."
        )
    lines.extend(
        [
            "",
            "### Trigger To Re-run",
            "",
            "- Browser trajectory overlay budget changes materially.",
            "- `ca web` adds new trajectory inspection anchors or overlay modes.",
            "- A new simplification strategy is proposed.",
        ]
    )
    return "\n".join(lines)


def render_decisions_markdown(report: dict) -> str:
    """Render adoption and rejection reasoning."""
    return "# Decisions\n\n" + render_decision_section(report)


def render_interface_section(report: dict) -> str:
    """Render the smallest stable trajectory interface left after comparison."""
    selected = report["decision"]["selected_experiment"]
    stabilized = report["decision"]["stabilized_core_strategy"]
    lineage_line = (
        f"- Current stabilized lineage: `{selected}` adopted directly in core\n"
        if selected == stabilized
        else f"- Current stabilized lineage: `{selected}` -> `{stabilized}`\n"
    )
    return (
        "## web_trajectory_sampling\n\n"
        "### Current Minimal Interface\n\n"
        "The stable interface keeps only the contract needed to compare browser trajectory reducers.\n\n"
        "```python\n"
        "class WebTrajectorySamplingStrategy(Protocol):\n"
        "    name: str\n"
        "    design: str\n"
        "    def reduce(self, request: WebTrajectorySamplingRequest) -> WebTrajectorySamplingResult: ...\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class WebTrajectorySamplingRequest:\n"
        "    positions: np.ndarray\n"
        "    max_points: int\n"
        "    timestamps: np.ndarray | None = None\n"
        "    label: str = \"trajectory\"\n"
        "    preserve_indices: tuple[int, ...] = ()\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class WebTrajectorySamplingResult:\n"
        "    positions: np.ndarray\n"
        "    kept_indices: np.ndarray\n"
        "    strategy: str\n"
        "    design: str\n"
        "    original_points: int\n"
        "    reduced_points: int\n"
        "    timestamps: np.ndarray | None = None\n"
        "    metadata: dict[str, Any]\n"
        "```\n\n"
        "### Stable Boundary\n\n"
        "- Stable core: `cloudanalyzer/ca/core/web_trajectory_sampling.py`\n"
        "- Experimental space: `cloudanalyzer/ca/experiments/web_trajectory_sampling/`\n"
        + lineage_line
    )


def render_interfaces_markdown(report: dict) -> str:
    """Render the smallest stable interface left after comparison."""
    return "# Interfaces\n\n" + render_interface_section(report)
