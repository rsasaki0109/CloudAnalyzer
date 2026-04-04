"""Benchmark and document concrete web sampling implementations."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np
import open3d as o3d

from ca.core.web_sampling import WebSampleRequest
from ca.experiments.web_sampling import get_web_sampling_strategies
from ca.experiments.web_sampling.common import clone_point_cloud
from ca.metrics import compute_nn_distance


@dataclass(slots=True)
class DatasetCase:
    """A comparable experiment input."""

    name: str
    description: str
    point_cloud: o3d.geometry.PointCloud
    max_points: int


def _make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    return point_cloud


def build_default_datasets() -> list[DatasetCase]:
    """Create deterministic datasets with different spatial structure."""

    rng = np.random.default_rng(20260402)

    grid_axis = np.linspace(-6.0, 6.0, 90)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    zz = rng.normal(scale=0.015, size=xx.size)
    plane = np.column_stack([xx.ravel(), yy.ravel(), zz])

    cluster_centers = np.array(
        [
            [-4.0, -2.5, 0.5],
            [-1.0, 2.0, 1.2],
            [2.5, -1.5, 0.8],
            [4.0, 2.5, 1.8],
        ]
    )
    clustered = []
    for center in cluster_centers:
        clustered.append(rng.normal(loc=center, scale=[0.45, 0.35, 0.25], size=(1800, 3)))
    clustered_points = np.vstack(clustered)

    corridor_x = rng.uniform(0.0, 30.0, size=7000)
    corridor_y = rng.choice([-1.8, 1.8], size=7000) + rng.normal(scale=0.05, size=7000)
    corridor_z = rng.uniform(0.0, 2.5, size=7000)
    corridor_floor = np.column_stack(
        [
            rng.uniform(0.0, 30.0, size=2500),
            rng.uniform(-1.8, 1.8, size=2500),
            rng.normal(scale=0.02, size=2500),
        ]
    )
    corridor_points = np.vstack([np.column_stack([corridor_x, corridor_y, corridor_z]), corridor_floor])

    return [
        DatasetCase(
            name="structured_plane",
            description="Dense planar surface with low noise.",
            point_cloud=_make_point_cloud(plane),
            max_points=1800,
        ),
        DatasetCase(
            name="clustered_room",
            description="Separated spatial clusters with uneven density.",
            point_cloud=_make_point_cloud(clustered_points),
            max_points=1600,
        ),
        DatasetCase(
            name="corridor_scan",
            description="Long corridor geometry with walls and floor.",
            point_cloud=_make_point_cloud(corridor_points),
            max_points=2000,
        ),
    ]


def _summarize_distances(distances: np.ndarray) -> dict[str, float]:
    if distances.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(distances)),
        "p95": float(np.quantile(distances, 0.95)),
        "max": float(np.max(distances)),
    }


def benchmark_strategy_on_dataset(strategy, dataset: DatasetCase, repetitions: int = 3) -> dict:
    """Measure runtime and geometric quality for one strategy and one dataset."""

    runtimes_ms: list[float] = []
    last_result = None
    for _ in range(repetitions):
        request = WebSampleRequest(
            point_cloud=clone_point_cloud(dataset.point_cloud),
            max_points=dataset.max_points,
            label=dataset.name,
        )
        start = time.perf_counter()
        result = strategy.reduce(request)
        runtimes_ms.append((time.perf_counter() - start) * 1000.0)
        last_result = result

    assert last_result is not None
    reduced = last_result.point_cloud
    coverage = compute_nn_distance(dataset.point_cloud, reduced)
    fidelity = compute_nn_distance(reduced, dataset.point_cloud)
    coverage_summary = _summarize_distances(coverage)
    fidelity_summary = _summarize_distances(fidelity)

    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "original_points": len(dataset.point_cloud.points),
        "target_points": dataset.max_points,
        "reduced_points": last_result.reduced_points,
        "runtime_ms": float(fmean(runtimes_ms)),
        "coverage_mean": coverage_summary["mean"],
        "coverage_p95": coverage_summary["p95"],
        "fidelity_mean": fidelity_summary["mean"],
        "fidelity_p95": fidelity_summary["p95"],
        "chamfer_mean": coverage_summary["mean"] + fidelity_summary["mean"],
        "retained_ratio": (
            last_result.reduced_points / len(dataset.point_cloud.points)
            if len(dataset.point_cloud.points) > 0
            else 0.0
        ),
        "metadata": dict(last_result.metadata),
    }


def _count_annotated_functions(functions: list[ast.FunctionDef | ast.AsyncFunctionDef]) -> int:
    annotated = 0
    for func in functions:
        args = list(func.args.args) + list(func.args.kwonlyargs)
        if func.args.vararg is not None:
            args.append(func.args.vararg)
        if func.args.kwarg is not None:
            args.append(func.args.kwarg)
        all_args_annotated = all(arg.annotation is not None for arg in args)
        if all_args_annotated and func.returns is not None:
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
        1
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    max_function_lines = max(
        ((func.end_lineno or func.lineno) - func.lineno + 1) for func in functions
    ) if functions else 0
    documented_nodes = 0
    documentable_nodes = 1 + len(functions) + len(classes)
    if ast.get_docstring(tree):
        documented_nodes += 1
    for node in [*functions, *classes]:
        if ast.get_docstring(node):
            documented_nodes += 1
    docstring_coverage = documented_nodes / documentable_nodes if documentable_nodes else 0.0
    annotated_ratio = (
        _count_annotated_functions(functions) / len(functions) if functions else 0.0
    )
    helper_functions = max(len(functions) - 1, 0)
    class_count = len(classes)

    readability_score = max(
        0.0,
        min(
            100.0,
            100.0
            - (nonempty_loc * 0.35)
            - (len(branches) * 2.5)
            - (max_function_lines * 0.4)
            + (docstring_coverage * 20.0),
        ),
    )
    extensibility_score = max(
        0.0,
        min(
            100.0,
            25.0
            + (helper_functions * 5.0)
            + (class_count * 7.0)
            + (annotated_ratio * 30.0)
            + (docstring_coverage * 15.0)
            - (max_function_lines * 0.25),
        ),
    )

    return {
        "loc": nonempty_loc,
        "function_count": len(functions),
        "class_count": class_count,
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


def summarize_strategy_results(report_rows: list[dict], analysis_rows: dict[str, dict]) -> list[dict]:
    """Aggregate per-dataset measurements into comparable strategy summaries."""

    grouped: dict[str, list[dict]] = {}
    for row in report_rows:
        grouped.setdefault(row["strategy"], []).append(row)

    summaries = []
    for strategy_name, rows in grouped.items():
        summaries.append(
            {
                "strategy": strategy_name,
                "design": rows[0]["design"],
                "module": rows[0]["module"],
                "avg_runtime_ms": round(float(fmean(row["runtime_ms"] for row in rows)), 4),
                "avg_chamfer_mean": round(float(fmean(row["chamfer_mean"] for row in rows)), 6),
                "avg_coverage_p95": round(float(fmean(row["coverage_p95"] for row in rows)), 6),
                "avg_retained_ratio": round(float(fmean(row["retained_ratio"] for row in rows)), 4),
                "readability_score": analysis_rows[strategy_name]["readability_score"],
                "extensibility_score": analysis_rows[strategy_name]["extensibility_score"],
            }
        )

    quality_ranks = _rank(
        {item["strategy"]: item["avg_chamfer_mean"] for item in summaries},
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

    for item in summaries:
        strategy_name = item["strategy"]
        item["quality_rank"] = quality_ranks[strategy_name]
        item["runtime_rank"] = runtime_ranks[strategy_name]
        item["readability_rank"] = readability_ranks[strategy_name]
        item["extensibility_rank"] = extensibility_ranks[strategy_name]
        item["composite_rank"] = round(
            (item["quality_rank"] * 0.5)
            + (item["runtime_rank"] * 0.2)
            + (item["readability_rank"] * 0.15)
            + (item["extensibility_rank"] * 0.15),
            3,
        )

    return sorted(summaries, key=lambda item: item["composite_rank"])


def run_web_sampling_experiment(
    datasets: list[DatasetCase] | None = None,
    repetitions: int = 3,
) -> dict:
    """Evaluate every concrete implementation on shared inputs and metrics."""

    datasets = datasets or build_default_datasets()
    strategies = get_web_sampling_strategies()
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

    strategy_summaries = summarize_strategy_results(rows, analysis_rows)
    selected = strategy_summaries[0]

    return {
        "problem": {
            "name": "web_point_cloud_reduction",
            "statement": (
                "Reduce large point clouds for `ca web` without fixing the abstraction too early."
            ),
            "stable_interface_path": "cloudanalyzer/ca/core/web_sampling.py",
            "experiment_package": "cloudanalyzer/ca/experiments/web_sampling",
            "repetitions": repetitions,
        },
        "datasets": [
            {
                "name": dataset.name,
                "description": dataset.description,
                "points": len(dataset.point_cloud.points),
                "max_points": dataset.max_points,
            }
            for dataset in datasets
        ],
        "results": rows,
        "analysis": analysis_rows,
        "strategy_summaries": strategy_summaries,
        "decision": {
            "selected_experiment": selected["strategy"],
            "stabilized_core_strategy": "random_budget",
            "reason": (
                "Best composite rank with quality weighted ahead of speed, readability, and extensibility."
            ),
        },
    }


def _render_summary_table(strategy_summaries: list[dict]) -> str:
    lines = [
        "| Strategy | Design | Avg runtime ms | Avg chamfer | Avg coverage p95 | Readability | Extensibility | Composite rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in strategy_summaries:
        lines.append(
            "| "
            f"{item['strategy']} | {item['design']} | {item['avg_runtime_ms']:.4f} | "
            f"{item['avg_chamfer_mean']:.6f} | {item['avg_coverage_p95']:.6f} | "
            f"{item['readability_score']:.2f} | {item['extensibility_score']:.2f} | "
            f"{item['composite_rank']:.3f} |"
        )
    return "\n".join(lines)


def render_experiment_section(report: dict) -> str:
    """Render the experiment section for point-cloud web reduction."""
    dataset_lines = [
        "| Dataset | Points | Budget | Purpose |",
        "|---|---:|---:|---|",
    ]
    for dataset in report["datasets"]:
        dataset_lines.append(
            f"| {dataset['name']} | {dataset['points']} | {dataset['max_points']} | {dataset['description']} |"
        )

    return (
        "## web_point_cloud_reduction\n\n"
        f"{report['problem']['statement']}\n\n"
        "Stable code lives in `cloudanalyzer/ca/core/web_sampling.py`. "
        "Discardable variants live in `cloudanalyzer/ca/experiments/web_sampling`.\n\n"
        "### Shared Inputs\n\n"
        + "\n".join(dataset_lines)
        + "\n\n### Strategy Comparison\n\n"
        + _render_summary_table(report["strategy_summaries"])
        + "\n\n### Notes\n\n"
        "- Geometry quality uses original-to-reduced coverage plus reduced-to-original fidelity.\n"
        "- Readability and extensibility scores are heuristic and generated from AST/source-shape metrics.\n"
        "- The selected stable strategy is extracted only after comparing concrete implementations.\n"
    )


def render_experiments_markdown(report: dict) -> str:
    """Render experiment comparison results."""
    return "# Experiments\n\n" + render_experiment_section(report)


def render_decision_section(report: dict) -> str:
    """Render the decision section for point-cloud web reduction."""
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
        "## web_point_cloud_reduction",
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
            f"Quality rank={item['quality_rank']}, runtime rank={item['runtime_rank']}, "
            f"readability rank={item['readability_rank']}, extensibility rank={item['extensibility_rank']}."
        )
    lines.extend(
        [
            "",
            "### Trigger To Re-run",
            "",
            "- Browser point budget changes materially.",
            "- New sampling strategy is proposed.",
            "- `ca web` starts preserving additional attributes that change the reduction trade-off.",
        ]
    )
    return "\n".join(lines)


def render_decisions_markdown(report: dict) -> str:
    """Render adoption and rejection reasoning."""
    return "# Decisions\n\n" + render_decision_section(report)


def render_interface_section(report: dict) -> str:
    """Render the smallest stable interface left after comparison."""
    selected = report["decision"]["selected_experiment"]
    stabilized = report["decision"]["stabilized_core_strategy"]
    lineage_line = (
        f"- Current stabilized lineage: `{selected}` adopted directly in core\n"
        if selected == stabilized
        else f"- Current stabilized lineage: `{selected}` -> `{stabilized}`\n"
    )
    return (
        "## web_point_cloud_reduction\n\n"
        "### Current Minimal Interface\n\n"
        "The stable interface is intentionally small. It keeps only the contract required to compare and adopt reducers.\n\n"
        "```python\n"
        "class WebSamplingStrategy(Protocol):\n"
        "    name: str\n"
        "    design: str\n"
        "    def reduce(self, request: WebSampleRequest) -> WebSampleResult: ...\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class WebSampleRequest:\n"
        "    point_cloud: o3d.geometry.PointCloud\n"
        "    max_points: int\n"
        "    label: str = \"point cloud\"\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class WebSampleResult:\n"
        "    point_cloud: o3d.geometry.PointCloud\n"
        "    strategy: str\n"
        "    design: str\n"
        "    original_points: int\n"
        "    reduced_points: int\n"
        "    metadata: dict[str, Any]\n"
        "```\n\n"
        "### Stable Boundary\n\n"
        "- Stable core: `cloudanalyzer/ca/core/web_sampling.py`\n"
        "- Experimental space: `cloudanalyzer/ca/experiments/web_sampling/`\n"
        + lineage_line
    )


def render_interfaces_markdown(report: dict) -> str:
    """Render the smallest stable interface left after comparison."""
    return "# Interfaces\n\n" + render_interface_section(report)


def write_report_docs(report: dict, docs_root: Path) -> None:
    """Write the required docs files."""

    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "experiments.md").write_text(
        render_experiments_markdown(report) + "\n",
        encoding="utf-8",
    )
    (docs_root / "decisions.md").write_text(
        render_decisions_markdown(report) + "\n",
        encoding="utf-8",
    )
    (docs_root / "interfaces.md").write_text(
        render_interfaces_markdown(report) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--write-docs", action="store_true")
    parser.add_argument("--docs-root", type=Path, default=None)
    args = parser.parse_args()

    report = run_web_sampling_experiment(repetitions=args.repetitions)
    if args.write_docs:
        docs_root = args.docs_root or (Path(__file__).resolve().parents[4] / "docs")
        write_report_docs(report, docs_root)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
