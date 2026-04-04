"""Benchmark and document concrete starter-config scaffolding implementations."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from statistics import fmean
import time

import yaml  # type: ignore[import-untyped]

from ca.core import CheckScaffoldRequest
from ca.experiments.check_scaffolding import get_check_scaffolding_strategies
from ca.experiments.check_scaffolding.common import (
    ProfileCase,
    build_default_profile_cases,
    load_suite_from_result,
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
            - (len(branches) * 2.8)
            - (max_function_lines * 0.3)
            + (docstring_coverage * 18.0),
        ),
    )
    extensibility_score = max(
        0.0,
        min(
            100.0,
            28.0
            + (max(len(functions) - 1, 0) * 4.5)
            + (len(classes) * 7.5)
            + (annotated_ratio * 28.0)
            + (docstring_coverage * 14.0)
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


def benchmark_strategy_on_profile(strategy, case: ProfileCase, repetitions: int = 3) -> dict:
    """Measure render cost and output fidelity for one profile."""

    runtimes_ms: list[float] = []
    last_result = None
    for _ in range(repetitions):
        request = CheckScaffoldRequest(profile=case.profile)
        start = time.perf_counter()
        result = strategy.render(request)
        runtimes_ms.append((time.perf_counter() - start) * 1000.0)
        last_result = result

    assert last_result is not None
    raw = yaml.safe_load(last_result.yaml_text)
    suite = load_suite_from_result(last_result)
    actual_ids = tuple(check.check_id for check in suite.checks)
    actual_kinds = tuple(check.kind for check in suite.checks)
    output_ratio = float(
        sum(
            1
            for check in suite.checks
            if check.outputs.report_path is not None and check.outputs.json_path is not None
        )
        / len(suite.checks)
    )
    check_id_match_ratio = sum(
        actual == expected
        for actual, expected in zip(actual_ids, case.expected_check_ids)
    ) / max(len(case.expected_check_ids), 1)
    kind_match_ratio = sum(
        actual == expected
        for actual, expected in zip(actual_kinds, case.expected_kinds)
    ) / max(len(case.expected_kinds), 1)
    fidelity_score = (
        (1.0 if suite.project == case.expected_project else 0.0)
        + (1.0 if len(suite.checks) == len(case.expected_check_ids) else 0.0)
        + check_id_match_ratio
        + kind_match_ratio
        + output_ratio
    ) / 5.0

    return {
        "profile": case.profile,
        "description": case.description,
        "runtime_ms": float(fmean(runtimes_ms)),
        "yaml_lines": len(last_result.yaml_text.splitlines()),
        "yaml_bytes": len(last_result.yaml_text.encode("utf-8")),
        "project_match": suite.project == case.expected_project,
        "check_count": len(suite.checks),
        "expected_check_count": len(case.expected_check_ids),
        "check_id_match_ratio": float(check_id_match_ratio),
        "kind_match_ratio": float(kind_match_ratio),
        "output_ratio": output_ratio,
        "fidelity_score": float(fidelity_score),
        "metadata": dict(last_result.metadata),
        "yaml_keys": tuple(raw.keys()) if isinstance(raw, dict) else tuple(),
    }


def _rank(values: dict[str, float], reverse: bool = False) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
    return {name: rank + 1 for rank, (name, _) in enumerate(ordered)}


def summarize_strategy_results(rows: list[dict], analysis_rows: dict[str, dict]) -> list[dict]:
    """Aggregate per-profile measurements into comparable strategy summaries."""

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
                "avg_fidelity_score": round(float(fmean(row["fidelity_score"] for row in group)), 6),
                "avg_yaml_lines": round(float(fmean(row["yaml_lines"] for row in group)), 2),
                "avg_yaml_bytes": round(float(fmean(row["yaml_bytes"] for row in group)), 2),
                "readability_score": analysis_rows[strategy_name]["readability_score"],
                "extensibility_score": analysis_rows[strategy_name]["extensibility_score"],
            }
        )

    quality_ranks = _rank(
        {item["strategy"]: item["avg_fidelity_score"] for item in summaries},
        reverse=True,
    )
    runtime_ranks = _rank({item["strategy"]: item["avg_runtime_ms"] for item in summaries})
    compactness_ranks = _rank({item["strategy"]: item["avg_yaml_lines"] for item in summaries})
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
        item["compactness_rank"] = compactness_ranks[strategy_name]
        item["readability_rank"] = readability_ranks[strategy_name]
        item["extensibility_rank"] = extensibility_ranks[strategy_name]
        item["composite_rank"] = round(
            (item["quality_rank"] * 0.5)
            + (item["runtime_rank"] * 0.15)
            + (item["compactness_rank"] * 0.05)
            + (item["readability_rank"] * 0.15)
            + (item["extensibility_rank"] * 0.15),
            3,
        )
    return sorted(summaries, key=lambda item: item["composite_rank"])


def run_check_scaffolding_experiment(
    profiles: list[ProfileCase] | None = None,
    repetitions: int = 3,
) -> dict:
    """Evaluate every concrete starter-config generator on shared profiles."""

    profiles = profiles or build_default_profile_cases()
    strategies = get_check_scaffolding_strategies()
    rows: list[dict] = []
    analysis_rows: dict[str, dict] = {}

    for strategy in strategies:
        module_path = Path(inspect.getsourcefile(strategy.__class__) or "")
        analysis_rows[strategy.name] = static_source_analysis(module_path)
        for case in profiles:
            metrics = benchmark_strategy_on_profile(
                strategy=strategy,
                case=case,
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
            "name": "check_scaffolding",
            "statement": (
                "Generate starter `cloudanalyzer.yaml` files without locking config authoring into a single large abstraction."
            ),
            "stable_interface_path": "cloudanalyzer/ca/core/check_scaffolding.py",
            "experiment_package": "cloudanalyzer/ca/experiments/check_scaffolding",
            "repetitions": repetitions,
        },
        "datasets": [
            {
                "profile": case.profile,
                "description": case.description,
                "expected_project": case.expected_project,
                "expected_check_count": len(case.expected_check_ids),
            }
            for case in profiles
        ],
        "results": rows,
        "analysis": analysis_rows,
        "strategy_summaries": summaries,
        "decision": {
            "selected_experiment": winner["strategy"],
            "stabilized_core_strategy": "static_profiles",
            "reason": (
                "Literal profiles preserve full fidelity while keeping runtime and source complexity low for the current onboarding scope."
            ),
        },
    }


def render_experiment_section(report: dict) -> str:
    """Render the experiment section for starter-config scaffolding."""

    profile_lines = [
        "| Profile | Expected checks | Purpose |",
        "|---|---:|---|",
    ]
    for dataset in report["datasets"]:
        profile_lines.append(
            f"| {dataset['profile']} | {dataset['expected_check_count']} | {dataset['description']} |"
        )

    summary_lines = [
        "| Strategy | Design | Avg runtime ms | Fidelity | Avg yaml lines | Readability | Extensibility | Composite rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["strategy_summaries"]:
        summary_lines.append(
            "| "
            f"{item['strategy']} | {item['design']} | {item['avg_runtime_ms']:.4f} | "
            f"{item['avg_fidelity_score']:.4f} | {item['avg_yaml_lines']:.2f} | "
            f"{item['readability_score']:.2f} | {item['extensibility_score']:.2f} | "
            f"{item['composite_rank']:.3f} |"
        )

    return (
        "## check_scaffolding\n\n"
        f"{report['problem']['statement']}\n\n"
        "Stable code lives in `cloudanalyzer/ca/core/check_scaffolding.py`. "
        "Discardable variants live in `cloudanalyzer/ca/experiments/check_scaffolding`.\n\n"
        "### Shared Inputs\n\n"
        + "\n".join(profile_lines)
        + "\n\n### Strategy Comparison\n\n"
        + "\n".join(summary_lines)
        + "\n\n### Notes\n\n"
        "- Fidelity is measured by parsing rendered YAML through `load_check_suite` and checking expected ids, kinds, and output paths.\n"
        "- Runtime covers template rendering only, not file writing.\n"
        "- Readability and extensibility scores are heuristic and generated from AST/source-shape metrics.\n"
    )


def render_decision_section(report: dict) -> str:
    """Render the decision section for starter-config scaffolding."""

    rejected = report["strategy_summaries"][1:]
    selected_name = report["decision"]["selected_experiment"]
    stabilized_name = report["decision"]["stabilized_core_strategy"]
    adopted_line = (
        f"`{stabilized_name}` is adopted directly as the current core strategy."
        if selected_name == stabilized_name
        else f"`{stabilized_name}` is the stabilized core form of `{selected_name}`."
    )
    lines = [
        "## check_scaffolding",
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
            f"compactness rank={item['compactness_rank']}, readability rank={item['readability_rank']}, "
            f"extensibility rank={item['extensibility_rank']}."
        )
    lines.extend(
        [
            "",
            "### Trigger To Re-run",
            "",
            "- `ca init-check` gains user-supplied placeholders or path inference.",
            "- New starter profiles are added beyond mapping, localization, perception, and integrated.",
            "- Config generation needs structured customization beyond static scaffolds.",
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
        "## check_scaffolding\n\n"
        "### Current Minimal Interface\n\n"
        "The stable interface keeps only the contract needed by `ca init-check`: profile in, YAML text out.\n\n"
        "```python\n"
        "class CheckScaffoldingStrategy(Protocol):\n"
        "    name: str\n"
        "    design: str\n"
        "    def render(self, request: CheckScaffoldRequest) -> CheckScaffoldResult: ...\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class CheckScaffoldRequest:\n"
        "    profile: str = \"integrated\"\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class CheckScaffoldResult:\n"
        "    profile: str\n"
        "    yaml_text: str\n"
        "    strategy: str\n"
        "    design: str\n"
        "    metadata: dict[str, Any]\n"
        "```\n\n"
        "### Stable Boundary\n\n"
        "- Stable core: `cloudanalyzer/ca/core/check_scaffolding.py`\n"
        "- Experimental space: `cloudanalyzer/ca/experiments/check_scaffolding/`\n"
        + lineage_line
    )


def write_report_docs(report: dict, docs_root: Path) -> None:
    """Write the required docs files for this slice only."""

    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "experiments.md").write_text(
        "# Experiments\n\n" + render_experiment_section(report) + "\n",
        encoding="utf-8",
    )
    (docs_root / "decisions.md").write_text(
        "# Decisions\n\n" + render_decision_section(report) + "\n",
        encoding="utf-8",
    )
    (docs_root / "interfaces.md").write_text(
        "# Interfaces\n\n" + render_interface_section(report) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_check_scaffolding_experiment(repetitions=args.repetitions), indent=2))


if __name__ == "__main__":
    main()
