"""Write consolidated experiment-driven docs across active comparison slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ca.experiments.web_sampling.evaluate import (
    render_decision_section as render_web_sampling_decision_section,
    render_experiment_section as render_web_sampling_experiment_section,
    render_interface_section as render_web_sampling_interface_section,
    run_web_sampling_experiment,
)
from ca.experiments.check_scaffolding.evaluate import (
    render_decision_section as render_check_scaffolding_decision_section,
    render_experiment_section as render_check_scaffolding_experiment_section,
    render_interface_section as render_check_scaffolding_interface_section,
    run_check_scaffolding_experiment,
)
from ca.experiments.web_progressive_loading.evaluate import (
    render_decision_section as render_web_progressive_decision_section,
    render_experiment_section as render_web_progressive_experiment_section,
    render_interface_section as render_web_progressive_interface_section,
    run_web_progressive_loading_experiment,
)
from ca.experiments.web_trajectory_sampling.evaluate import (
    render_decision_section as render_web_trajectory_decision_section,
    render_experiment_section as render_web_trajectory_experiment_section,
    render_interface_section as render_web_trajectory_interface_section,
    run_web_trajectory_sampling_experiment,
)


def build_experiment_reports(repetitions: int = 3) -> list[dict]:
    """Run every active experiment slice and return reports in doc order."""

    return [
        run_web_sampling_experiment(repetitions=repetitions),
        run_web_trajectory_sampling_experiment(repetitions=repetitions),
        run_web_progressive_loading_experiment(repetitions=repetitions),
        run_check_scaffolding_experiment(repetitions=repetitions),
    ]


def render_experiments_markdown(reports: list[dict]) -> str:
    """Render the shared experiments doc across all slices."""

    sections = [
        render_web_sampling_experiment_section(reports[0]),
        render_web_trajectory_experiment_section(reports[1]),
        render_web_progressive_experiment_section(reports[2]),
        render_check_scaffolding_experiment_section(reports[3]),
    ]
    return (
        "# Experiments\n\n"
        "These comparisons keep stable `core/` contracts small and leave competing designs under `experiments/`.\n\n"
        + "\n\n".join(sections)
        + "\n"
    )


def render_decisions_markdown(reports: list[dict]) -> str:
    """Render the shared decisions doc across all slices."""

    sections = [
        render_web_sampling_decision_section(reports[0]),
        render_web_trajectory_decision_section(reports[1]),
        render_web_progressive_decision_section(reports[2]),
        render_check_scaffolding_decision_section(reports[3]),
    ]
    return "# Decisions\n\n" + "\n\n".join(sections) + "\n"


def render_interfaces_markdown(reports: list[dict]) -> str:
    """Render the shared interfaces doc across all slices."""

    sections = [
        render_web_sampling_interface_section(reports[0]),
        render_web_trajectory_interface_section(reports[1]),
        render_web_progressive_interface_section(reports[2]),
        render_check_scaffolding_interface_section(reports[3]),
    ]
    return (
        "# Interfaces\n\n"
        "Stable interfaces keep only the request/result shapes that production callers already need.\n\n"
        + "\n\n".join(sections)
        + "\n"
    )


def write_process_docs(reports: list[dict], docs_root: Path) -> None:
    """Write consolidated process docs for all experiment slices."""

    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "experiments.md").write_text(
        render_experiments_markdown(reports),
        encoding="utf-8",
    )
    (docs_root / "decisions.md").write_text(
        render_decisions_markdown(reports),
        encoding="utf-8",
    )
    (docs_root / "interfaces.md").write_text(
        render_interfaces_markdown(reports),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--write-docs", action="store_true")
    parser.add_argument("--docs-root", type=Path, default=None)
    args = parser.parse_args()

    reports = build_experiment_reports(repetitions=args.repetitions)
    if args.write_docs:
        docs_root = args.docs_root or (Path(__file__).resolve().parents[3] / "docs")
        write_process_docs(reports, docs_root)
    else:
        print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
