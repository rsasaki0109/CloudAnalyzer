#!/usr/bin/env python3
"""Benchmark streaming MapEval AWD/SCS at 1M/10M point-map scale."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ca.mapeval_benchmark import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SCALE_SIZES,
    run_scale_benchmark,
)


def _parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip().replace("_", "")) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must contain positive integers")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SCALE_SIZES,
        help="comma-separated points per map (default: 1000000,10000000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"stream chunk size (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "qa" / "mapeval-scale.json",
        help="JSON output path",
    )
    parser.add_argument("--format-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_scale_benchmark(
        args.sizes,
        chunk_size=args.chunk_size,
        seed=args.seed,
        output=args.output,
    )
    if args.format_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"MapEval scale report: {args.output}")
        for run in report["runs"]:
            print(
                f"{run['points_per_map']:,} points/map: "
                f"{run['runtime']['elapsed_seconds']:.3f}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
