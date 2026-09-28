"""Write the trajectory parity fixture and print the Python module's results,
for the parity test in ``rust/crates/ca-core/tests/trajectory.rs``.

Usage::

    PYTHONPATH=cloudanalyzer python scripts/make_trajectory_fixtures.py

Writes ``rust/crates/ca-core/tests/trajectory/``: a reference (TUM, 10 Hz) and
an estimate (TUM, 20 Hz, offset by 20 ms so poses are interpolated) that is
the reference moved, slightly scaled and perturbed. Prints the numbers of
``ca.trajectory.evaluate_trajectory`` that the Rust test hardcodes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ca.trajectory import evaluate_trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rust" / "crates" / "ca-core" / "tests" / "trajectory"


def pose(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Position and rotation matrix of the reference path at time ``t``."""
    position = np.array([3 * np.sin(0.6 * t), 2 * (1 - np.cos(0.6 * t)), 0.3 * t])
    yaw, pitch = 0.5 * t, 0.1 * np.sin(t)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return position, rz @ ry


def quaternion(r: np.ndarray) -> np.ndarray:
    """``[x, y, z, w]`` of a rotation matrix (w > 0 for these small angles)."""
    w = np.sqrt(max(0.0, 1 + np.trace(r))) / 2
    return np.array(
        [(r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w]
    )


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def write(path: Path, rows: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for t, p, r in rows:
        values = [t, *p, *quaternion(r)]
        lines.append(" ".join(f"{v:.6f}" for v in values))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    # The estimate's frame: rotated, shifted and scaled against the reference.
    rotation = axis_angle(np.array([0.2, 0.1, 1.0]), 0.35)
    translation = np.array([1.0, -2.0, 0.5])
    scale = 0.97

    reference = [(0.1 * k, *pose(0.1 * k)) for k in range(20)]
    estimate = []
    for k in range(39):
        t = 0.02 + 0.05 * k
        p, r = pose(t)
        noise = axis_angle(rng.normal(size=3), rng.normal(scale=0.01))
        estimate.append(
            (t, scale * rotation @ p + translation + rng.normal(scale=0.02, size=3), rotation @ r @ noise)
        )
    write(OUT / "reference.tum", reference)
    write(OUT / "estimate.tum", estimate)

    results = {}
    for mode, flags in {
        "none": {},
        "origin": {"align_origin": True},
        "rigid": {"align_rigid": True},
    }.items():
        r = evaluate_trajectory(
            str(OUT / "estimate.tum"),
            str(OUT / "reference.tum"),
            max_time_delta=0.05,
            rpe_distances_m=[0.5],
            **flags,
        )
        results[mode] = {
            "matched": r["matching"]["matched_poses"],
            "ate": r["ate"],
            "ate_rotation": r["ate_rotation"],
            "rpe_translation": r["rpe_translation"],
            "rpe_rotation": r["rpe_rotation"],
            "rpe_0.5m": {k: r["rpe_distance"][0][k] for k in ("pairs", "translation", "rotation")},
            "endpoint_drift": r["drift"]["endpoint"],
            "alignment": r["alignment"],
        }
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
