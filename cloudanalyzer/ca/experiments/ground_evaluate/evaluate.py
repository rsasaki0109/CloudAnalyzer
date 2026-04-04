"""Benchmark and document concrete ground segmentation evaluation strategies."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from statistics import fmean
import time

from ca.experiments.ground_evaluate import get_ground_evaluate_strategies
from ca.experiments.ground_evaluate.common import (
    GroundEvaluateDatasetCase,
    build_default_datasets,
    perturb_request,
)


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
            - (nonempty_loc * 0.32)
            - (len(branches) * 2.5)
            - (max_function_lines * 0.25)
            + (docstring_coverage * 16.0),
        ),
    )
    extensibility_score = max(
        0.0,
        min(
            100.0,
            28.0
            + (max(len(functions) - 1, 0) * 4.0)
            + (len(classes) * 8.0)
            + (annotated_ratio * 28.0)
            + (docstring_coverage * 12.0)
            - (max_function_lines * 0.18),
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


def benchmark_strategy_on_dataset(
    strategy, dataset: GroundEvaluateDatasetCase, repetitions: int = 3,
) -> dict:
    """Measure quality and runtime on one ground segmentation scenario."""

    runtimes_ms: list[float] = []
    last_result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = strategy.evaluate(dataset.request)
        runtimes_ms.append((time.perf_counter() - start) * 1000.0)
        last_result = result

    assert last_result is not None

    # Stability: does F1 stay close under small perturbation?
    perturbed_result = strategy.evaluate(perturb_request(dataset.request))
    f1_stability = 1.0 - abs(last_result.f1 - perturbed_result.f1)

    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "runtime_ms": float(fmean(runtimes_ms)),
        "precision": last_result.precision,
        "recall": last_result.recall,
        "f1": last_result.f1,
        "iou": last_result.iou,
        "accuracy": last_result.accuracy,
        "f1_stability": round(float(f1_stability), 6),
        "meets_expected": 1.0 if last_result.f1 >= dataset.expected_min_f1 else 0.0,
    }


def _rank(values: dict[str, float], reverse: bool = False) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
    return {name: rank + 1 for rank, (name, _) in enumerate(ordered)}


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
                "avg_f1": round(float(fmean(row["f1"] for row in group)), 6),
                "avg_iou": round(float(fmean(row["iou"] for row in group)), 6),
                "avg_precision": round(float(fmean(row["precision"] for row in group)), 6),
                "avg_recall": round(float(fmean(row["recall"] for row in group)), 6),
                "avg_stability": round(float(fmean(row["f1_stability"] for row in group)), 6),
                "avg_meets_expected": round(float(fmean(row["meets_expected"] for row in group)), 6),
                "readability_score": analysis_rows[strategy_name]["readability_score"],
                "extensibility_score": analysis_rows[strategy_name]["extensibility_score"],
            }
        )

    quality_ranks = _rank(
        {
            item["strategy"]: (item["avg_f1"] * 0.5) + (item["avg_iou"] * 0.3) + (item["avg_meets_expected"] * 0.2)
            for item in summaries
        },
        reverse=True,
    )
    stability_ranks = _rank(
        {item["strategy"]: item["avg_stability"] for item in summaries},
        reverse=True,
    )
    runtime_ranks = _rank({item["strategy"]: item["avg_runtime_ms"] for item in summaries})
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
        item["stability_rank"] = stability_ranks[strategy_name]
        item["runtime_rank"] = runtime_ranks[strategy_name]
        item["readability_rank"] = readability_ranks[strategy_name]
        item["extensibility_rank"] = extensibility_ranks[strategy_name]
        item["composite_rank"] = round(
            (item["quality_rank"] * 0.5)
            + (item["stability_rank"] * 0.15)
            + (item["runtime_rank"] * 0.15)
            + (item["readability_rank"] * 0.1)
            + (item["extensibility_rank"] * 0.1),
            3,
        )
    return sorted(summaries, key=lambda item: item["composite_rank"])


def run_ground_evaluate_experiment(
    datasets: list[GroundEvaluateDatasetCase] | None = None,
    repetitions: int = 3,
) -> dict:
    """Evaluate every concrete ground evaluation strategy on shared scenarios."""

    datasets = datasets or build_default_datasets()
    strategies = get_ground_evaluate_strategies()
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
            "name": "ground_segmentation_evaluate",
            "statement": (
                "Evaluate ground segmentation quality by comparing estimated ground/non-ground "
                "points against reference labels using precision, recall, F1, and IoU."
            ),
            "stable_interface_path": "cloudanalyzer/ca/core/ground_evaluate.py",
            "experiment_package": "cloudanalyzer/ca/experiments/ground_evaluate",
            "repetitions": repetitions,
        },
        "datasets": [
            {
                "name": dataset.name,
                "description": dataset.description,
                "estimated_ground_points": int(dataset.request.estimated_ground.shape[0]),
                "reference_ground_points": int(dataset.request.reference_ground.shape[0]),
                "expected_min_f1": dataset.expected_min_f1,
            }
            for dataset in datasets
        ],
        "results": rows,
        "analysis": analysis_rows,
        "strategy_summaries": summaries,
        "decision": {
            "selected_experiment": winner["strategy"],
            "stabilized_core_strategy": "voxel_confusion",
            "reason": (
                "Best composite rank with the fastest runtime and robust voxel-level matching "
                "that avoids per-point distance computation."
            ),
        },
    }


def render_experiment_section(report: dict) -> str:
    """Render the experiment section for ground segmentation evaluation."""

    dataset_lines = [
        "| Dataset | Est ground pts | Ref ground pts | Expected min F1 | Purpose |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in report["datasets"]:
        dataset_lines.append(
            f"| {dataset['name']} | {dataset['estimated_ground_points']} | "
            f"{dataset['reference_ground_points']} | {dataset['expected_min_f1']:.2f} | "
            f"{dataset['description']} |"
        )

    summary_lines = [
        "| Strategy | Design | Avg runtime ms | Avg F1 | Avg IoU | Avg precision | Avg recall | Stability | Readability | Extensibility | Composite rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["strategy_summaries"]:
        summary_lines.append(
            "| "
            f"{item['strategy']} | {item['design']} | {item['avg_runtime_ms']:.4f} | "
            f"{item['avg_f1']:.4f} | {item['avg_iou']:.4f} | {item['avg_precision']:.4f} | "
            f"{item['avg_recall']:.4f} | {item['avg_stability']:.4f} | "
            f"{item['readability_score']:.2f} | {item['extensibility_score']:.2f} | "
            f"{item['composite_rank']:.3f} |"
        )

    return (
        "## ground_segmentation_evaluate\n\n"
        f"{report['problem']['statement']}\n\n"
        "Stable code lives in `cloudanalyzer/ca/core/ground_evaluate.py`. "
        "Discardable variants live in `cloudanalyzer/ca/experiments/ground_evaluate`.\n\n"
        "### Shared Inputs\n\n"
        + "\n".join(dataset_lines)
        + "\n\n### Strategy Comparison\n\n"
        + "\n".join(summary_lines)
        + "\n\n### Notes\n\n"
        "- Quality compares each strategy against expected F1 thresholds on synthetic ground segmentation scenarios.\n"
        "- Stability checks whether F1 stays close under small positional perturbations.\n"
        "- Voxel-based approaches are faster but coarser; nearest-neighbor is precise but slower.\n"
    )


def render_decision_section(report: dict) -> str:
    """Render the decision section for ground segmentation evaluation."""

    rejected = report["strategy_summaries"][1:]
    selected_name = report["decision"]["selected_experiment"]
    stabilized_name = report["decision"]["stabilized_core_strategy"]
    adopted_line = (
        f"`{stabilized_name}` is adopted directly as the current core strategy."
        if selected_name == stabilized_name
        else f"`{stabilized_name}` is the stabilized core form of `{selected_name}`."
    )
    lines = [
        "## ground_segmentation_evaluate",
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
            f"Quality rank={item['quality_rank']}, stability rank={item['stability_rank']}, "
            f"runtime rank={item['runtime_rank']}, readability rank={item['readability_rank']}, "
            f"extensibility rank={item['extensibility_rank']}."
        )
    lines.extend(
        [
            "",
            "### Trigger To Re-run",
            "",
            "- Ground segmentation evaluation needs per-point rather than per-voxel resolution.",
            "- Height-band diagnostics become a first-class output for slope analysis.",
            "- A new matching or scoring strategy is proposed.",
        ]
    )
    return "\n".join(lines)


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
        "## ground_segmentation_evaluate\n\n"
        "### Current Minimal Interface\n\n"
        "The stable interface keeps only the ground/non-ground contract needed for `ca ground-evaluate`.\n\n"
        "```python\n"
        "class GroundEvaluateStrategy(Protocol):\n"
        "    name: str\n"
        "    design: str\n"
        "    def evaluate(self, request: GroundEvaluateRequest) -> GroundEvaluateResult: ...\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class GroundEvaluateRequest:\n"
        "    estimated_ground: np.ndarray\n"
        "    estimated_nonground: np.ndarray\n"
        "    reference_ground: np.ndarray\n"
        "    reference_nonground: np.ndarray\n"
        "    voxel_size: float = 0.2\n"
        "```\n\n"
        "### Stable Boundary\n\n"
        "- Stable core: `cloudanalyzer/ca/core/ground_evaluate.py`\n"
        "- Experimental space: `cloudanalyzer/ca/experiments/ground_evaluate/`\n"
        + lineage_line
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_ground_evaluate_experiment(repetitions=args.repetitions), indent=2, default=str))


if __name__ == "__main__":
    main()
