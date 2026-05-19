#!/usr/bin/env python3
"""Materialize the `newer-college-mini` SLAM benchmark suite locally.

The Newer College Dataset (Ramezani et al., IROS 2020) is distributed
under CC-BY-NC-SA 4.0 and is too large to ship with CloudAnalyzer.
Users download the ground-truth bundle on their own machine, then run
this script once to convert it into a `ca benchmark eval`-ready suite
under `benchmarks/slam/newer-college-mini/`.

This is a thin wrapper around `ca benchmark init` with Newer-College
defaults baked in (name, description, license, gate thresholds, output
directory). See `benchmarks/slam/newer-college-mini/README.md` for the
full workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ca.benchmark import GATE_KEYS, materialize_suite  # noqa: E402


SUITE_DIR = REPO_ROOT / "benchmarks" / "slam" / "newer-college-mini"
DEFAULT_NAME = "newer-college-mini"
DEFAULT_LICENSE = (
    "CC-BY-NC-SA 4.0 (Newer College Dataset, Ramezani et al., IROS 2020)"
)
DEFAULT_GATE: dict[str, float] = {
    "min_auc": 0.90,
    "max_chamfer": 0.30,
    "max_ate": 0.50,
    "max_rpe": 0.20,
    "max_drift": 1.50,
    "min_coverage": 0.85,
}


def _parse_gate_overrides(values: list[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if not values:
        return overrides
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"--gate expects key=value; got {raw!r}")
        key, _, value = raw.partition("=")
        key = key.strip()
        if key not in GATE_KEYS:
            raise SystemExit(
                f"unknown gate key {key!r}; allowed: {', '.join(GATE_KEYS)}"
            )
        try:
            overrides[key] = float(value)
        except ValueError as exc:
            raise SystemExit(f"--gate {key} must be numeric: {value!r}") from exc
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-map",
        required=True,
        type=Path,
        help="Path to the Newer College ground-truth map (PCD/PLY)",
    )
    parser.add_argument(
        "--reference-trajectory",
        required=True,
        type=Path,
        help="Path to the Newer College ground-truth trajectory (TUM format)",
    )
    parser.add_argument(
        "--sequence",
        default="short_experiment",
        help="Sequence name used in the suite manifest (default: short_experiment)",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.10,
        help="Voxel size in meters for downsampling the GT map (default: 0.10)",
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=2000,
        help="Keep at most this many evenly-spaced trajectory poses (default: 2000)",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Override the suite description (default: auto-generated from settings)",
    )
    parser.add_argument(
        "--license",
        default=DEFAULT_LICENSE,
        help=f"License string (default: {DEFAULT_LICENSE!r})",
    )
    parser.add_argument(
        "--gate",
        action="append",
        metavar="KEY=VALUE",
        help="Override a gate threshold (repeatable). Defaults: "
        + ", ".join(f"{k}={v}" for k, v in DEFAULT_GATE.items()),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SUITE_DIR,
        help=f"Output suite directory (default: {SUITE_DIR.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    if not args.reference_map.is_file():
        raise SystemExit(f"--reference-map not found: {args.reference_map}")
    if not args.reference_trajectory.is_file():
        raise SystemExit(
            f"--reference-trajectory not found: {args.reference_trajectory}"
        )

    gate = dict(DEFAULT_GATE)
    gate.update(_parse_gate_overrides(args.gate))

    description = args.description or (
        f"Newer College Dataset {args.sequence} sequence "
        f"(GT map downsampled to {args.voxel:g} m, "
        f"GT trajectory subsampled to {args.max_poses} poses)."
    )

    suite = materialize_suite(
        args.output,
        name=DEFAULT_NAME,
        description=description,
        reference_map=args.reference_map,
        reference_trajectory=args.reference_trajectory,
        sequence_name=args.sequence,
        sequence_description=f"Newer College Dataset {args.sequence} GT bundle.",
        license=args.license,
        voxel_size=args.voxel,
        max_poses=args.max_poses,
        gate=gate,
    )

    print(f"Suite manifest: {suite.source_path}")
    print(f"Sequence:       {args.sequence}")
    seq = suite.resolve_sequence(args.sequence)
    print(f"Reference map:  {seq.reference_map_path}")
    print(f"Reference traj: {seq.reference_trajectory_path}")
    print("Gate:")
    for key, value in suite.gate.items():
        print(f"  {key}: {value}")
    print()
    print("Ready. Try:")
    try:
        suite_arg = suite.source_path.relative_to(REPO_ROOT)
    except ValueError:
        suite_arg = suite.source_path
    print(
        f"  ca benchmark eval {suite_arg} "
        f"--map <your_slam_map.pcd> --trajectory <your_slam_trajectory.tum>"
    )


if __name__ == "__main__":
    main()
