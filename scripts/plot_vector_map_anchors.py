"""Plot frozen anchor drafts and expose shared-interval gains and losses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vector_map_anchor_evaluate import reference_document
from vector_map_quality_audit import digest


def plot(source: Path, reference: Path, planning: Path, tokyo: Path, output: Path):
    reports = [json.loads((root / "evaluation.json").read_text(encoding="utf-8")) for root in (planning, tokyo)]
    if digest(source) != reports[0]["source_sha256"] or digest(reference) != reports[0]["reference_sha256"]:
        raise ValueError("plot inputs differ from the evaluated source/reference")
    for root, report in zip((planning, tokyo), reports):
        if any(digest(root / name) != sha for name, sha in report["generation_artifact_sha256"].items()):
            raise ValueError("frozen generation changed")
    with source.open("rb") as stream:
        header = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("missing PCD DATA header")
            line = raw.decode("ascii").strip()
            header.append(line)
            if line.startswith("DATA "):
                break
        offset = stream.tell()
    if "FIELDS x y z rgb" not in header or "DATA binary" not in header:
        raise ValueError("expected original planning XYZRGB binary PCD")
    count = int(next(line.split()[1] for line in header if line.startswith("POINTS ")))
    cloud = np.memmap(source, dtype=np.float32, offset=offset, shape=(count, 4), mode="r")
    drafts = [json.loads((planning / f"{mode}-0-profiles.json").read_text(encoding="utf-8")) for mode in ("before", "after")]
    surveyed = reference_document(reference, None)  # Posthoc plotting only.
    used = {ref if isinstance(ref, int) else ref["boundary"] for lane in surveyed["lanes"]
            if lane["kind"] == "driving" for ref in (lane["left"], lane["right"])}
    extent = np.concatenate([np.asarray(b)[:, :2] for r in drafts[0]["roads"] for b in r["boundaries"]])
    lo, hi = extent.min(axis=0) - 4, extent.max(axis=0) + 4
    origin = lo // 10 * 10
    ground = cloud[(cloud[:, 0] > lo[0]) & (cloud[:, 0] < hi[0]) &
                   (cloud[:, 1] > lo[1]) & (cloud[:, 1] < hi[1]) & (cloud[:, 2] < 22)][::7]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1, 1, 1.08]})
    colors = ["#db8721", "#147fa3"]
    for i, ax in enumerate(axes[:2]):
        ax.scatter(ground[:, 0] - origin[0], ground[:, 1] - origin[1], s=.5, color="#c4c9cd", rasterized=True)
        for b in surveyed["boundaries"]:
            if b["id"] in used:
                p = np.asarray(b["geometry"])[:, :2] - origin
                ax.plot(*p.T, color="#8d989f", lw=.65, alpha=.65)
        for road in drafts[i]["roads"]:
            for b in road["boundaries"]:
                p = np.asarray(b)[:, :2] - origin
                ax.plot(*p.T, color=colors[i], lw=1.8)
            p = np.asarray(road["reference"])[:, :2] - origin
            ax.plot(*p.T, color="#925fa2", lw=.8, ls="--")
        ax.set_xlim(lo[0] - origin[0], hi[0] - origin[0])
        ax.set_ylim(lo[1] - origin[1], hi[1] - origin[1])
        ax.set_aspect("equal")
        ax.set_xlabel(f"X - {origin[0]:.0f} (m)")
        ax.set_ylabel(f"Y - {origin[1]:.0f} (m)")
        ax.grid(alpha=.12)
        ax.set_title(["Before: coverage limits also shift priors", "After: curb/intensity anchors only"][i], fontsize=10)
        ax.text(.02, .02, ["41.11 m retained / 9.50 m deferred", "36.61 m retained / 14.00 m deferred"][i],
                transform=ax.transAxes, fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92})
    cases = [reports[0]["paired_source_intervals"][0], reports[1]["paired_source_intervals"][4]]
    ax = axes[2]
    y = np.array([1., 0.])
    for i, mode in enumerate(("before", "after")):
        values = [case[mode]["mean_xy_m"] for case in cases]
        ax.barh(y + (.16 if i == 0 else -.16), values, height=.28, color=colors[i], label=mode.capitalize())
        for row, value in zip(y + (.16 if i == 0 else -.16), values):
            ax.text(value + .03, row, f"{value:.3f} m", va="center", fontsize=9)
    ax.set_yticks(y, ["Planning path 0\n36.61 m / 297 samples", "Tokyo east-south\n4.78 m / 65 samples"])
    ax.set_xlim(0, 2.4)
    ax.set_ylim(-.55, 1.55)
    ax.set_xlabel("Mean distance to fixed before-selected survey samples (m)")
    ax.set_title("Identical source intervals and boundary slots\nFixed survey targets; no lane identity claim", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", alpha=.15)
    fig.suptitle("Optional anchor fix: measured gain on one path, slight loss on another", fontsize=13)
    fig.text(.5, .035, "Grey: later surveyed driving boundaries. Purple: explicit trace. Planning loses another 4.50 m; do not count that as accuracy.", ha="center", fontsize=9)
    fig.text(.5, .01, "Known development scenes; no reference input or registration fit. Planning sample: Copyright 2020 TIER IV, Inc. Tokyo data: Dynamic Map Platform Co., Ltd., CC BY 4.0.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "planning", "tokyo", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    plot(args.source, args.reference, args.planning, args.tokyo, args.output)
