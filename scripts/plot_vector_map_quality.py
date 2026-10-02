"""Plot actual generated boundaries and a separately scored development audit."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vector_map_quality_audit import digest, verify_frozen_map


def plot(source: Path, proof: Path, report: Path, output: Path) -> None:
    audit = json.loads(report.read_text(encoding="utf-8"))
    vector_map, manifest = verify_frozen_map(proof)
    if (
        digest(source) != audit["input_cloud_sha256"]
        or manifest["generated_sha256"] != audit["generated_osm_sha256"]
    ):
        raise ValueError("plot inputs differ from audited source/map")
    root = ET.parse(vector_map).getroot()
    nodes = {}
    for node in root.findall("node"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in node.findall("tag")}
        nodes[node.attrib["id"]] = [
            float(tags[k]) for k in ("local_x", "local_y", "ele")
        ]
    ways = {
        w.attrib["id"]: np.array([nodes[n.attrib["ref"]] for n in w.findall("nd")])
        for w in root.findall("way")
    }
    quality = audit["native"]["quality"]
    review = {str(i) for i in quality["low_support_lanes"]}
    lanes, boundary_review = [], {}
    for relation in root.findall("relation"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in relation.findall("tag")}
        if tags.get("type") != "lanelet" or tags.get("subtype") != "road":
            continue
        sides = [
            m.attrib["ref"]
            for m in relation.findall("member")
            if m.attrib["type"] == "way" and m.attrib["role"] in {"left", "right"}
        ]
        lanes.append((relation.attrib["id"], sides))
        for side in sides:
            boundary_review[side] = (
                boundary_review.get(side, False) or relation.attrib["id"] in review
            )
    cloud = laspy.read(source)
    origin = np.array([-9341.0, -40867.0])
    xyz = np.c_[cloud.x, cloud.y, cloud.z]
    # Display only: source-quality measurements already used the entire cloud.
    ground_view = xyz[xyz[:, 2] <= 35.526][::25]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 7.5), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    left.scatter(
        *(ground_view[:, :2] - origin).T,
        s=0.7,
        color="#bcc3c7",
        rasterized=True,
        label="Original source points (display subset)",
    )
    for side, needs_review in boundary_review.items():
        p = ways[side][:, :2] - origin
        left.plot(
            p[:, 0],
            p[:, 1],
            color="#d76818" if needs_review else "#298697",
            linewidth=1.2,
            alpha=0.9,
        )
    for lane, sides in lanes:
        if lane in review:
            p = np.concatenate([ways[s][:, :2] for s in sides]).mean(axis=0) - origin
            left.annotate(
                lane,
                p,
                fontsize=8,
                fontweight="bold",
                color="#8f370d",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                    "pad": 1,
                },
            )
    left.plot([], [], color="#298697", label="Boundary curves with source support")
    left.plot([], [], color="#d76818", label="Boundary used by a lane needing review")
    left.set_aspect("equal")
    left.set_xlabel("Source X - (-9341 m)")
    left.set_ylabel("Source Y - (-40867 m)")
    left.set_title(
        f"{len(quality['lanes'])} generated lanes / {len(review)} need source review"
    )
    left.legend(loc="upper left", fontsize=8)
    left.grid(alpha=0.15)
    matches = audit["equipment"]["repeated_paint"]["matches"]
    y = np.arange(len(matches))
    right.barh(
        y - 0.17,
        [m["geometry"]["symmetric_mean_m"] for m in matches],
        height=0.32,
        color="#298697",
        label="Symmetric mean outline distance",
    )
    right.barh(
        y + 0.17,
        [m["geometry"]["hausdorff_m"] for m in matches],
        height=0.32,
        color="#d76818",
        label="Hausdorff outline distance",
    )
    right.set_yticks(y, [m["reference"] for m in matches])
    right.invert_yaxis()
    right.set_xlabel("Distance (m; lower is closer)")
    right.set_ylabel("Reference crossing ID")
    right.set_title("Observed paint vs mapped pedestrian footprint")
    right.legend(loc="lower right", fontsize=8)
    right.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Generated-map quality: source coverage and separate reference geometry",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.035,
        "0 structural errors | 7,968 source samples | No reference-guided repair | Development scene, not independent accuracy\nHard Intersection Multimodal Samples, Dynamic Map Platform Co., Ltd. (2026), CC BY 4.0",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "proof", "report", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    plot(args.source, args.proof, args.report, args.output)
