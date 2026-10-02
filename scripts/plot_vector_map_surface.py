"""Plot actual before/after branch boundaries over original source points."""

import argparse
import json
from pathlib import Path

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vector_map_quality_audit import digest


def boundary_id(value):
    return value if isinstance(value, int) else value["boundary"]


def plot(source: Path, evaluation: Path, output: Path) -> None:
    report = json.loads((evaluation / "comparison.json").read_text())
    if digest(source) != report["source_sha256"] or report["reference_inputs"]:
        raise ValueError(
            "plot requires the audited source and reference-free generation"
        )
    cloud = laspy.read(source)
    origin = np.array([-9341.0, -40867.0])
    xyz = np.c_[cloud.x, cloud.y, cloud.z]
    # Display only; the native audit used every original point, not this subset.
    view = xyz[xyz[:, 2] < 32.0][::8]
    cases = [
        c
        for c in report["cases"]
        if c["name"] in {"west-south", "east-south", "west-north"}
    ]
    if len(cases) != 3 or any(c["after"].get("entire_path_deferred") for c in cases):
        raise ValueError("this plot requires the three retained frozen branch cases")
    fig, axes = plt.subplots(2, len(cases), figsize=(13, 8), squeeze=False)
    for col, case in enumerate(cases):
        before_map = json.loads((evaluation / case["before"]["map"]).read_text())
        selected = set(case["before"]["added_lane_ids"])
        ids = {
            boundary_id(lane[side])
            for lane in before_map["lanes"]
            if lane["id"] in selected
            for side in ("left", "right")
        }
        points = np.concatenate(
            [
                np.asarray(b["geometry"])[:, :2] - origin
                for b in before_map["boundaries"]
                if b["id"] in ids
            ]
        )
        lower, upper = points.min(axis=0) - 2, points.max(axis=0) + 2
        for row, mode in enumerate(("before", "after")):
            ax = axes[row, col]
            ax.scatter(*(view[:, :2] - origin).T, s=1, c="#c4c9cd", rasterized=True)
            result = case[mode]
            path = evaluation / result["map"]
            if digest(path) != result["map_sha256"]:
                raise ValueError("generated map differs from audited geometry")
            vector_map = json.loads(path.read_text())
            selected = set(result["added_lane_ids"])
            ids = {
                boundary_id(lane[side])
                for lane in vector_map["lanes"]
                if lane["id"] in selected
                for side in ("left", "right")
            }
            for boundary in vector_map["boundaries"]:
                if boundary["id"] in ids:
                    p = np.asarray(boundary["geometry"])[:, :2] - origin
                    ax.plot(*p.T, c="#d76818" if row == 0 else "#087d87", lw=2)
            review = len(result["source_quality"]["source_review_lane_ids"])
            suffix = ""
            if row == 1:
                fit = result["surface_fit"]
                suffix = f"; {fit['deferred_length_m']:.1f} m deferred"
            ax.set_title(
                f"{case['name']} / {mode}\n{result['generated_length_m']:.1f} m retained{suffix}; {review} review"
            )
            ax.set_xlim(lower[0], upper[0])
            ax.set_ylim(lower[1], upper[1])
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)
            ax.set_xlabel("Source X + 9341 (m)")
            ax.set_ylabel("Source Y + 40867 (m)")
    fig.suptitle(
        "Source-footprint drafting: repair supported branches, defer missing extent",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.025,
        f"Same six paths / explicit lane counts | {report['before']['generated_length_m']:.1f} → "
        f"{report['after']['generated_length_m']:.1f} m total road extent; "
        f"{report['after']['reported_deferred_length_m']:.1f} m deferred "
        f"({report['after']['reported_deferred_length_m'] / report['after']['evaluated_path_length_m']:.0%})\n"
        "Source gate, not independent accuracy; no reference-guided fitting | Display-only height filter / 1:8 decimation\n"
        "Hard Intersection Multimodal Samples, Dynamic Map Platform Co., Ltd. (2026), CC BY 4.0",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "evaluation", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    plot(args.source, args.evaluation, args.output)
