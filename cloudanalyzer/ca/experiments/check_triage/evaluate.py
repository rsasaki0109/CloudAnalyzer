"""Benchmark and document concrete failed-check triage implementations."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from statistics import fmean
import time

from ca.experiments.check_triage import get_check_triage_strategies
from ca.experiments.check_triage.common import (
    TriageDatasetCase,
    build_default_datasets,
    perturb_request,
    ranking_ndcg,
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


def benchmark_strategy_on_dataset(strategy, dataset: TriageDatasetCase, repetitions: int = 3) -> dict:
    """Measure rank quality and runtime on one failure set."""

    runtimes_ms: list[float] = []
    last_result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = strategy.rank(dataset.request)
        runtimes_ms.append((time.perf_counter() - start) * 1000.0)
        last_result = result

    assert last_result is not None
    ranked_ids = [item.check_id for item in last_result.ranked_items]
    perturb_top1_hits: list[float] = []
    baseline_top1 = ranked_ids[0] if ranked_ids else None
    for factor in (0.98, 1.02):
        perturbed = strategy.rank(perturb_request(dataset.request, factor=factor))
        perturb_top1_hits.append(
            1.0 if baseline_top1 == (perturbed.ranked_items[0].check_id if perturbed.ranked_items else None) else 0.0
        )

    diversity_ids = {
        tuple(item.failed_dimensions)
        for item in last_result.ranked_items[: min(3, len(last_result.ranked_items))]
    }
    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "failed_items": len(dataset.request.failed_items),
        "runtime_ms": float(fmean(runtimes_ms)),
        "ndcg": float(ranking_ndcg(ranked_ids, dataset.expected_order)),
        "top1_hit": 1.0 if ranked_ids[:1] == list(dataset.expected_order[:1]) else 0.0,
        "stability_top1": float(fmean(perturb_top1_hits)) if perturb_top1_hits else 1.0,
        "diversity_top3": (
            float(len(diversity_ids) / max(min(3, len(last_result.ranked_items)), 1))
            if last_result.ranked_items
            else 1.0
        ),
        "ranked_ids": ranked_ids,
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
                "avg_ndcg": round(float(fmean(row["ndcg"] for row in group)), 6),
                "avg_top1_hit": round(float(fmean(row["top1_hit"] for row in group)), 6),
                "avg_stability_top1": round(float(fmean(row["stability_top1"] for row in group)), 6),
                "avg_diversity_top3": round(float(fmean(row["diversity_top3"] for row in group)), 6),
                "readability_score": analysis_rows[strategy_name]["readability_score"],
                "extensibility_score": analysis_rows[strategy_name]["extensibility_score"],
            }
        )

    quality_ranks = _rank(
        {
            item["strategy"]: (item["avg_ndcg"] * 0.7) + (item["avg_top1_hit"] * 0.3)
            for item in summaries
        },
        reverse=True,
    )
    stability_ranks = _rank(
        {item["strategy"]: item["avg_stability_top1"] for item in summaries},
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


def run_check_triage_experiment(
    datasets: list[TriageDatasetCase] | None = None,
    repetitions: int = 3,
) -> dict:
    """Evaluate every concrete triage implementation on shared failure sets."""

    datasets = datasets or build_default_datasets()
    strategies = get_check_triage_strategies()
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
            "name": "check_regression_triage",
            "statement": (
                "Rank failed mapping, localization, and perception checks so `ca check` surfaces the most informative regression first."
            ),
            "stable_interface_path": "cloudanalyzer/ca/core/check_triage.py",
            "experiment_package": "cloudanalyzer/ca/experiments/check_triage",
            "repetitions": repetitions,
        },
        "datasets": [
            {
                "name": dataset.name,
                "description": dataset.description,
                "failed_items": len(dataset.request.failed_items),
                "expected_order": list(dataset.expected_order),
            }
            for dataset in datasets
        ],
        "results": rows,
        "analysis": analysis_rows,
        "strategy_summaries": summaries,
        "decision": {
            "selected_experiment": winner["strategy"],
            "stabilized_core_strategy": "severity_weighted",
            "reason": (
                "Best composite rank while preserving direct threshold-based ordering and the lowest runtime for CLI use."
            ),
        },
    }


def render_experiment_section(report: dict) -> str:
    """Render the experiment section for failed-check triage."""

    dataset_lines = [
        "| Dataset | Failed checks | Expected top order | Purpose |",
        "|---|---:|---|---|",
    ]
    for dataset in report["datasets"]:
        dataset_lines.append(
            f"| {dataset['name']} | {dataset['failed_items']} | "
            f"{', '.join(dataset['expected_order'])} | {dataset['description']} |"
        )

    summary_lines = [
        "| Strategy | Design | Avg runtime ms | Avg NDCG | Top1 hit | Stability | Diversity | Readability | Extensibility | Composite rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["strategy_summaries"]:
        summary_lines.append(
            "| "
            f"{item['strategy']} | {item['design']} | {item['avg_runtime_ms']:.4f} | "
            f"{item['avg_ndcg']:.4f} | {item['avg_top1_hit']:.4f} | {item['avg_stability_top1']:.4f} | "
            f"{item['avg_diversity_top3']:.4f} | {item['readability_score']:.2f} | "
            f"{item['extensibility_score']:.2f} | {item['composite_rank']:.3f} |"
        )

    return (
        "## check_regression_triage\n\n"
        f"{report['problem']['statement']}\n\n"
        "Stable code lives in `cloudanalyzer/ca/core/check_triage.py`. "
        "Discardable variants live in `cloudanalyzer/ca/experiments/check_triage`.\n\n"
        "### Shared Inputs\n\n"
        + "\n".join(dataset_lines)
        + "\n\n### Strategy Comparison\n\n"
        + "\n".join(summary_lines)
        + "\n\n### Notes\n\n"
        "- Quality is scored against expected failure orderings using a small NDCG variant plus top-1 hit rate.\n"
        "- Stability checks whether the top-ranked failure stays stable under small metric perturbations.\n"
        "- Diversity is tracked but not weighted heavily in the final ranking because current CLI use favors severity-first triage.\n"
    )


def render_decision_section(report: dict) -> str:
    """Render the decision section for failed-check triage."""

    rejected = report["strategy_summaries"][1:]
    selected_name = report["decision"]["selected_experiment"]
    stabilized_name = report["decision"]["stabilized_core_strategy"]
    adopted_line = (
        f"`{stabilized_name}` is adopted directly as the current core strategy."
        if selected_name == stabilized_name
        else f"`{stabilized_name}` is the stabilized core form of `{selected_name}`."
    )
    lines = [
        "## check_regression_triage",
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
            "- `ca check` starts triaging root-cause groups instead of individual failures.",
            "- Batch checks expose richer aggregate metrics or per-item drill-down metadata.",
            "- A new failure-ranking strategy is proposed.",
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
        "## check_regression_triage\n\n"
        "### Current Minimal Interface\n\n"
        "The stable interface keeps only the failed-check contract needed to rank regressions for `ca check`.\n\n"
        "```python\n"
        "class CheckTriageStrategy(Protocol):\n"
        "    name: str\n"
        "    design: str\n"
        "    def rank(self, request: CheckTriageRequest) -> CheckTriageResult: ...\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class CheckTriageRequest:\n"
        "    failed_items: tuple[CheckTriageItem, ...]\n"
        "    project: str | None = None\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class CheckTriageItem:\n"
        "    check_id: str\n"
        "    kind: str\n"
        "    metrics: dict[str, float]\n"
        "    gate: dict[str, float]\n"
        "    reasons: tuple[str, ...] = ()\n"
        "```\n\n"
        "### Stable Boundary\n\n"
        "- Stable core: `cloudanalyzer/ca/core/check_triage.py`\n"
        "- Experimental space: `cloudanalyzer/ca/experiments/check_triage/`\n"
        + lineage_line
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_check_triage_experiment(repetitions=args.repetitions), indent=2))


if __name__ == "__main__":
    main()
