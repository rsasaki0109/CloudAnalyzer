"""Plot frozen configured lane-edge inference with both fixed correspondences."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vector_map_anchor_evaluate import reference_document
from vector_map_quality_audit import digest


def plot(source, reference, proof, output):
    evaluation = json.loads((proof / "evaluation.json").read_text(encoding="utf-8"))
    if digest(source) != evaluation["source_sha256"] or digest(reference) != evaluation["reference_sha256"]:
        raise ValueError("plot inputs differ from frozen evaluation")
    if any(digest(proof / k) != v for k, v in evaluation["generation_artifact_sha256"].items()):
        raise ValueError("generation changed after freeze")
    with source.open("rb") as stream:
        header = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("missing PCD header")
            header.append(raw.decode("ascii").strip())
            if header[-1].startswith("DATA "):
                break
        offset = stream.tell()
    if "FIELDS x y z rgb" not in header or "DATA binary" not in header:
        raise ValueError("expected original binary XYZRGB")
    count = int(next(v.split()[1] for v in header if v.startswith("POINTS ")))
    cloud = np.memmap(source, dtype=np.float32, offset=offset, shape=(count, 4), mode="r")
    drafts = [json.loads((proof / f"{m}-1-profiles.json").read_text(encoding="utf-8")) for m in ("before", "after")]
    extent = np.concatenate([np.asarray(b)[:, :2] for p in drafts for r in p["roads"] for b in r["boundaries"]])
    lo, hi = extent.min(axis=0) - 2, extent.max(axis=0) + 2
    origin = lo // 10 * 10
    roi = cloud[(cloud[:, 0] > lo[0]) & (cloud[:, 0] < hi[0]) & (cloud[:, 1] > lo[1]) & (cloud[:, 1] < hi[1]) & (cloud[:, 2] < 20)]
    packed = roi[:, 3].copy().view(np.uint32)
    bright = ((packed >> 16) & 255 >= 180) & ((packed >> 8) & 255 >= 180) & ((packed & 255) >= 180)
    survey = reference_document(reference, None)
    used = {s if isinstance(s, int) else s["boundary"] for lane in survey["lanes"] if lane["kind"] == "driving" for s in (lane["left"], lane["right"])}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), gridspec_kw={"height_ratios": [1.4, 1]})
    colors = ["#db8623", "#1682a0"]
    for i, ax in enumerate(axes[0]):
        ax.scatter(roi[::4, 0] - origin[0], roi[::4, 1] - origin[1], s=.5, c="#c8cdd0", rasterized=True)
        ax.scatter(roi[bright, 0] - origin[0], roi[bright, 1] - origin[1], s=1, c="#677278", alpha=.5, rasterized=True)
        for b in survey["boundaries"]:
            if b["id"] in used:
                ax.plot(*(np.asarray(b["geometry"])[:, :2] - origin).T, color="#7a858b", lw=.8, alpha=.65)
        for road in drafts[i]["roads"]:
            for b, labels in zip(road["boundaries"], road["evidence"]):
                line = np.asarray(b)[:, :2] - origin
                ax.plot(*line.T, c=colors[i], lw=2, ls="--" if i else "-")
                if i:
                    observed = line[np.asarray(labels) == "rgb_paint"]
                    ax.scatter(*observed.T, c=colors[i], s=12, zorder=5)
            ax.plot(*(np.asarray(road["operator_reference"])[:, :2] - origin).T, c="#975b9a", ls=":", lw=1)
        if i:
            for edge in drafts[i]["extraction"]["lane_edge_inference"]["retained_road_edges"]:
                ax.plot(*(np.asarray(edge["geometry"])[:, :2] - origin).T, c="#db8623", ls=":", lw=1.5)
        report = drafts[i]["extraction"]
        ax.set(xlim=(lo[0]-origin[0], hi[0]-origin[0]), ylim=(lo[1]-origin[1], hi[1]-origin[1]),
               xlabel=f"X - {origin[0]:.0f} (m)", ylabel=f"Y - {origin[1]:.0f} (m)",
               title=["Before: corrected divider + outside candidates", "After: inferred lane edge; curb kept separately"][i])
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
        ax.text(.03, .02, f"{report['generated_length']:.3f} m retained / {report['surface_fit']['deferred_length_m']:.1f} m deferred", transform=ax.transAxes, fontsize=9, bbox={"facecolor": "white", "edgecolor": "none"})
    measures = [evaluation["fixed_reference_corridors"][1], evaluation["paired_source_intervals"][1], evaluation["fixed_reference_corridors"][1]["boundary_slots_left_to_right"][2]]
    ax = axes[1, 0]
    for i, mode in enumerate(("before", "after")):
        rows = np.array([2., 1., 0.]) + (.16 if i == 0 else -.16)
        values = [m[mode]["mean_xy_m"] for m in measures]
        ax.barh(rows, values, height=.28, color=colors[i], label=mode.capitalize())
        for y, v in zip(rows, values):
            ax.text(v+.04, y, f"{v:.3f} m", va="center", fontsize=9)
    ax.set(yticks=[2, 1, 0], yticklabels=["Ordered lane-pair slots", "Fixed before-nearest points", "Ordered right outside slot"], xlim=(0, max(m[mode]["mean_xy_m"] for m in measures for mode in ("before", "after"))*1.35), ylim=(-.55, 2.55), xlabel="Mean XY distance to SAME before-selected targets (m)", title=f"Common source path: {measures[0]['common_source_path_m']:.3f} m\nRetained and deferred extent reported separately")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=.15)
    ax = axes[1, 1]
    paint = drafts[1]["extraction"]["paint_divider"]
    t = paint["track"]
    length = drafts[1]["extraction"]["trajectory_length"]
    ax.broken_barh([(0, length)], (-.16, .32), facecolors="#e4edf0")
    for a, b in t["observed_intervals_m"]:
        ax.broken_barh([(a, b-a)], (-.16, .32), facecolors=colors[1])
    ax.text(.04, .87, f"{t['source_points']} points / {paint['curb_pair_sections']}/{paint['sampled_sections']} paired-curb sections\n{t['observed_length_m']:.3f} m observed components\n{t['interpolated_length_m']:.3f} m interpolated; {t['extrapolated_length_m']:.3f} m extended", transform=ax.transAxes, fontsize=9, va="top")
    ax.set(xlim=(0, length+1), ylim=(-.5, 1.1), yticks=[0], yticklabels=["Interior"], xlabel="Station along original trace (m)", title="Source intervals BEFORE footprint trimming\nLight portions stay inferred, not observed paint")
    ax.grid(axis="x", alpha=.15)
    fig.suptitle("Configured-width lane edge inside a distant curb; original road edge retained", fontsize=14)
    fig.text(.5, .045, "Actual generated geometry. Grey survey: posthoc only. Orange dotted: retained road-edge candidate. Blue dots: observed interior paint.", ha="center", fontsize=8)
    fig.text(.5, .024, "Known development scene, no held-out accuracy or certified lane roles. Missing paint stays inferred. Width is a manual prior, not observed outer paint or a certified shoulder.", ha="center", fontsize=8)
    fig.text(.5, .007, "Planning sample: Copyright 2020 TIER IV, Inc. Previous frozen comparisons and equipment remain unchanged.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "proof", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    plot(args.source, args.reference, args.proof, args.output)
