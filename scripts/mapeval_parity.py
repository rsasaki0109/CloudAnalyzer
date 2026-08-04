#!/usr/bin/env python3
"""Run the reproducible CloudAnalyzer ↔ MapEval AWD/SCS parity harness.

The upstream executable is optional because its C++ dependency stack is not
available on the normal Python CI runner.  Without ``--official-executable``
the command still generates the common PCD fixture and a complete JSON report;
the external comparison is recorded as ``not_run``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ca.mapeval_parity import (  # noqa: E402
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    run_parity_harness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "qa" / "mapeval-parity",
        help="report directory (default: qa/mapeval-parity)",
    )
    parser.add_argument(
        "--official-executable",
        type=Path,
        help="optional fixed-commit MapEval executable",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        help="optional checkout used only to record and verify its git commit",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=DEFAULT_ABS_TOLERANCE,
        help=f"absolute AWD/SCS tolerance (default: {DEFAULT_ABS_TOLERANCE:g})",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=DEFAULT_REL_TOLERANCE,
        help=f"relative AWD/SCS tolerance (default: {DEFAULT_REL_TOLERANCE:g})",
    )
    parser.add_argument(
        "--format-json",
        action="store_true",
        help="print the complete JSON report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_parity_harness(
        args.output,
        official_executable=args.official_executable,
        upstream_dir=args.upstream_dir,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    if args.format_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"MapEval parity report: {args.output / 'mapeval_parity.json'}")
        print(f"Status: {report['status']}")
        print(f"External comparison: {report['comparison']['status']}")
    if args.official_executable is not None and report["comparison"]["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
