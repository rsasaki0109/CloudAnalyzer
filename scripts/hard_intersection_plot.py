"""Plot actual source geometry, proposals and held-out references for inspection.

This is a scientific audit figure, not an application screenshot or generated map demo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hard_intersection_evaluate import geometry, read_reference, reference_center


def plot(dataset, prepared, generated, evaluation, output):
    gen = json.loads((generated / "generation.json").read_text(encoding="utf-8"))
    report = json.loads((evaluation / "evaluation.json").read_text(encoding="utf-8"))
    refs, _, _ = read_reference(dataset / "maps/lanelet2/jp_tokyo_takanawadai.osm")
    with laspy.open(prepared / "geometry.las") as reader:
        lo, hi = reader.header.mins, reader.header.maxs
        background = [np.column_stack([c.x[::50], c.y[::50]]) for c in reader.chunk_iterator(300_000)]
    xy = np.concatenate(background) - lo[:2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    fig.subplots_adjust(left=.065, right=.99, bottom=.09, top=.85, wspace=.2)
    colors = {"repeated_paint": "#2078b4", "bright_bar": "#df7b20", "elevated_panel": "#8055ad"}
    for ax, kinds, title in zip(axes, [["repeated_paint", "bright_bar"], ["elevated_panel"]], ["Ground paint proposals", "Elevated panel proposals"]):
        ax.scatter(*xy.T, s=.15, color="#c3c9cb", rasterized=True)
        for kind in kinds:
            color = colors[kind]
            matched_ids = {m["reference"] for m in report["instances"][kind]["matches"]}
            for c in gen["candidates"]:
                if c["evidence"]["kind"] == kind:
                    points = geometry(c)[:, :2] - lo[:2]
                    ax.plot(*points.T, color=color, alpha=.45, linewidth=.65)
            for reference in refs[kind]:
                c = reference_center(reference, kind)
                if np.any(c[:2] < lo[:2]) or np.any(c[:2] > hi[:2]):
                    continue
                p = np.asarray(reference["geometry"])[:, :2] - lo[:2]
                ax.plot(*p.T, color="#172c36", linestyle="--", linewidth=1.4)
                c = c[:2] - lo[:2]
                if reference["id"] in matched_ids:
                    ax.scatter(*c, s=50, marker="o", facecolors="none", edgecolors="#172c36", linewidths=1.2)
                else:
                    ax.scatter(*c, s=50, marker="x", color="#172c36", linewidths=1.2)
            stats = report["instances"][kind]
            ax.plot([], [], color=color, label=f"{kind}: {stats['owned_proposals']} proposals, {stats['matched']}/{stats['reference_in_source_extent']} nearby")
        ax.plot([], [], "--", color="#172c36", label="Held-out map (circle: nearby; x: missed)")
        ax.set(xlim=(-2, hi[0] - lo[0] + 2), ylim=(-2, hi[1] - lo[1] + 2), xlabel="Source easting offset (m)", ylabel="Source northing offset (m)", title=title)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Hard Intersection: fixed source-only baseline\n2 m center gate; nearby geometry is not semantic correctness", fontsize=12)
    fig.text(.5, .012, "Data: Dynamic Map Platform Co., Ltd. (2026), CC BY 4.0 | one intersection; incomplete map-derived annotations", ha="center", fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset", "prepared", "generated", "evaluation", "output"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    plot(args.dataset, args.prepared, args.generated, args.evaluation, args.output)
