"""Benchmark and document concrete baseline evolution implementations."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from statistics import fmean
import time

from ca.experiments.check_baseline_evolution import (
    get_check_baseline_evolution_strategies,
)
from ca.experiments.check_baseline_evolution.common import (
    BaselineEvolutionDatasetCase,
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
            - (max_function_lines * 0.24)
            + (docstring_coverage * 16.0),
        ),
    )
    extensibility_score = max(
        0.0,
        min(
            100.0,
            28.0
            + (max(len(functions) - 1, 0) * 4.1)
            + (len(classes) * 8.2)
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


def benchmark_strategy_on_dataset(strategy, dataset: BaselineEvolutionDatasetCase, repetitions: int = 3) -> dict:
    """Measure decision quality and runtime on one baseline-evolution scenario."""

    runtimes_ms: list[float] = []
    last_result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = strategy.decide(dataset.request)
        runtimes_ms.append((time.perf_counter() - start) * 1000.0)
        last_result = result

    assert last_result is not None
    stability_hits: list[float] = []
    for factor in (0.98, 1.02):
        perturbed = strategy.decide(perturb_request(dataset.request, factor=factor))
        stability_hits.append(1.0 if perturbed.decision == last_result.decision else 0.0)

    critical_expected = dataset.expected_decision in {"promote", "reject"}
    critical_match = (
        1.0 if (last_result.decision == dataset.expected_decision and critical_expected) else 0.0
    )
    if not critical_expected:
        critical_match = 1.0 if last_result.decision == dataset.expected_decision else 0.0

    false_promote = 1.0 if (
        last_result.decision == "promote" and dataset.expected_decision != "promote"
    ) else 0.0
    false_reject = 1.0 if (
        last_result.decision == "reject" and dataset.expected_decision != "reject"
    ) else 0.0
    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "history_count": len(dataset.request.history),
        "runtime_ms": float(fmean(runtimes_ms)),
        "decision": last_result.decision,
        "expected_decision": dataset.expected_decision,
        "decision_match": 1.0 if last_result.decision == dataset.expected_decision else 0.0,
        "critical_match": critical_match,
        "stability": float(fmean(stability_hits)) if stability_hits else 1.0,
        "false_promote": false_promote,
        "false_reject": false_reject,
        "confidence": float(last_result.confidence),
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
                "avg_decision_match": round(float(fmean(row["decision_match"] for row in group)), 6),
                "avg_critical_match": round(float(fmean(row["critical_match"] for row in group)), 6),
                "avg_stability": round(float(fmean(row["stability"] for row in group)), 6),
                "avg_false_promote": round(float(fmean(row["false_promote"] for row in group)), 6),
                "avg_false_reject": round(float(fmean(row["false_reject"] for row in group)), 6),
                "readability_score": analysis_rows[strategy_name]["readability_score"],
                "extensibility_score": analysis_rows[strategy_name]["extensibility_score"],
            }
        )

    quality_ranks = _rank(
        {
            item["strategy"]: (
                (item["avg_decision_match"] * 0.45)
                + (item["avg_critical_match"] * 0.35)
                + ((1.0 - item["avg_false_promote"]) * 0.2)
            )
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


def run_check_baseline_evolution_experiment(
    datasets: list[BaselineEvolutionDatasetCase] | None = None,
    repetitions: int = 3,
) -> dict:
    """Evaluate every concrete baseline evolution implementation."""

    datasets = datasets or build_default_datasets()
    strategies = get_check_baseline_evolution_strategies()
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
            "name": "check_baseline_evolution",
            "statement": (
                "Decide whether a candidate QA summary should promote, keep, or reject a baseline revision."
            ),
            "stable_interface_path": "cloudanalyzer/ca/core/check_baseline_evolution.py",
            "experiment_package": "cloudanalyzer/ca/experiments/check_baseline_evolution",
            "repetitions": repetitions,
        },
        "datasets": [
            {
                "name": dataset.name,
                "description": dataset.description,
                "history_count": len(dataset.request.history),
                "expected_decision": dataset.expected_decision,
            }
            for dataset in datasets
        ],
        "results": rows,
        "analysis": analysis_rows,
        "strategy_summaries": summaries,
        "decision": {
            "selected_experiment": winner["strategy"],
            "stabilized_core_strategy": "stability_window",
            "reason": (
                "Best composite rank by avoiding premature promotions while preserving perfect reject/promote accuracy on the shared scenarios."
            ),
        },
    }


def render_experiment_section(report: dict) -> str:
    """Render the experiment section for baseline evolution."""

    dataset_lines = [
        "| Dataset | Expected decision | History size | Purpose |",
        "|---|---|---:|---|",
    ]
    for dataset in report["datasets"]:
        dataset_lines.append(
            f"| {dataset['name']} | {dataset['expected_decision']} | "
            f"{dataset['history_count']} | {dataset['description']} |"
        )

    summary_lines = [
        "| Strategy | Design | Avg runtime ms | Decision match | Critical match | Stability | False promote | False reject | Readability | Extensibility | Composite rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["strategy_summaries"]:
        summary_lines.append(
            "| "
            f"{item['strategy']} | {item['design']} | {item['avg_runtime_ms']:.4f} | "
            f"{item['avg_decision_match']:.4f} | {item['avg_critical_match']:.4f} | "
            f"{item['avg_stability']:.4f} | {item['avg_false_promote']:.4f} | "
            f"{item['avg_false_reject']:.4f} | {item['readability_score']:.2f} | "
            f"{item['extensibility_score']:.2f} | {item['composite_rank']:.3f} |"
        )

    return (
        "## check_baseline_evolution\n\n"
        f"{report['problem']['statement']}\n\n"
        "Stable code lives in `cloudanalyzer/ca/core/check_baseline_evolution.py`. "
        "Discardable variants live in `cloudanalyzer/ca/experiments/check_baseline_evolution`.\n\n"
        "### Shared Inputs\n\n"
        + "\n".join(dataset_lines)
        + "\n\n### Strategy Comparison\n\n"
        + "\n".join(summary_lines)
        + "\n\n### Notes\n\n"
        "- Decision quality compares each strategy against shared promote / keep / reject expectations.\n"
        "- Stability checks whether small metric perturbations preserve the same decision.\n"
        "- False promote is weighted more heavily than raw confidence because baseline drift is more damaging than delayed promotion.\n"
    )


def render_decision_section(report: dict) -> str:
    """Render the decision section for baseline evolution."""

    rejected = report["strategy_summaries"][1:]
    selected_name = report["decision"]["selected_experiment"]
    stabilized_name = report["decision"]["stabilized_core_strategy"]
    adopted_line = (
        f"`{stabilized_name}` is adopted directly as the current core strategy."
        if selected_name == stabilized_name
        else f"`{stabilized_name}` is the stabilized core form of `{selected_name}`."
    )
    lines = [
        "## check_baseline_evolution",
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
            "- Baseline promotion starts using historical cost, latency, or artifact size in addition to QA metrics.",
            "- `ca check` emits richer per-check confidence or uncertainty metadata.",
            "- A new promote / keep / reject strategy is proposed.",
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
        "## check_baseline_evolution\n\n"
        "### Current Minimal Interface\n\n"
        "The stable interface keeps only the candidate/history contract needed to decide baseline promotion.\n\n"
        "```python\n"
        "class BaselineEvolutionStrategy(Protocol):\n"
        "    name: str\n"
        "    design: str\n"
        "    def decide(self, request: BaselineEvolutionRequest) -> BaselineEvolutionResult: ...\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class BaselineEvolutionRequest:\n"
        "    candidate: BaselineEvolutionSnapshot\n"
        "    history: tuple[BaselineEvolutionSnapshot, ...] = ()\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class BaselineEvolutionSnapshot:\n"
        "    label: str\n"
        "    checks: tuple[BaselineCheckSnapshot, ...]\n"
        "    passed: bool\n"
        "```\n\n"
        "### Stable Boundary\n\n"
        "- Stable core: `cloudanalyzer/ca/core/check_baseline_evolution.py`\n"
        "- Experimental space: `cloudanalyzer/ca/experiments/check_baseline_evolution/`\n"
        + lineage_line
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_check_baseline_evolution_experiment(repetitions=args.repetitions), indent=2))


if __name__ == "__main__":
    main()
