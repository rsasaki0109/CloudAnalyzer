"""CloudAnalyzer CLI entry point.

Commands live in ``cloudanalyzer_cli.commands``. Importing those modules
registers their commands on ``app``; the import order below is the order
``ca --help`` lists them.
"""

from cloudanalyzer_cli._app import app
from cloudanalyzer_cli.commands import (  # noqa: F401 -- imported to register commands, in order
    evaluate_debug,
    posegraph,
    mapping,
    vectormap,
    processing,
    batch_trajectory,
    runs,
    benchmark_bundle,
    quality_gate,
    task_evaluate,
    tiles_web,
    misc,
)


def main():
    app()


if __name__ == "__main__":
    main()
