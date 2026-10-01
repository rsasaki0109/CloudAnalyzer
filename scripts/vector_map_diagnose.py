"""Plot an external map comparison written by the vector_map_evaluate Rust example.

Requires numpy, scipy and matplotlib. No reference geometry is fed to extraction.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def sample(points, spacing=0.1):
    points = np.asarray(points, dtype=float)
    result = []
    for a, b in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / spacing)))
        t = (np.arange(count) + 0.5) / count
        result.append(a + t[:, None] * (b - a))
    return np.concatenate(result) if result else np.empty((0, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    trajectory = np.array(data["trajectory"])
    origin = trajectory[0, :2]
    reference = [
        b for b in data["reference"] if b["scored"] and b["kind"]["type"] != "virtual"
    ]
    truth = np.concatenate([sample(b["points"]) for b in reference])
    tree = cKDTree(truth[:, :2])
    colors = {
        "curb": "#e69900",
        "intensity": "#0077bb",
        "support_edge": "#aa3377",
        "width_prior": "#cc3311",
    }
    distances = {key: [] for key in colors}
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1.8, 1, 1]}
    )
    cloud = np.asarray(data["cloud"])
    lo, hi = trajectory[:, :2].min(axis=0) - 12, trajectory[:, :2].max(axis=0) + 12
    cloud = cloud[((cloud[:, :2] >= lo) & (cloud[:, :2] <= hi)).all(axis=1)]
    axes[0].scatter(
        *(cloud[:, :2] - origin).T, c="#bbbbbb", s=0.3, alpha=0.4, rasterized=True
    )
    for i, boundary in enumerate(reference):
        line = np.asarray(boundary["points"])
        axes[0].plot(
            *(line[:, :2] - origin).T,
            c="#228833",
            lw=1,
            label="Reference boundaries" if i == 0 else None,
        )
    axes[0].plot(
        *(trajectory[:, :2] - origin).T, ".-", c="black", lw=1.2, label="Recorded GNSS"
    )
    used = set()
    generated_z = []
    for road in data["generated"]:
        ground = np.asarray(road["reference"])
        generated_z.extend(ground[:, 2])
        sources = road.get("source_boundaries") or road["boundaries"]
        for line, source, labels in zip(road["boundaries"], sources, road["evidence"]):
            line = np.asarray(line)
            axes[0].plot(*(line[:, :2] - origin).T, c="#cc3311", lw=1.2, alpha=0.65)
            source = np.asarray(source)
            error = tree.query(source[:, :2])[0]
            for evidence, color in colors.items():
                mask = np.array(labels) == evidence
                distances[evidence].extend(error[mask])
                if mask.any():
                    axes[0].scatter(
                        *(source[mask, :2] - origin).T,
                        c=color,
                        s=9,
                        label=evidence.replace("_", " ")
                        if evidence not in used
                        else None,
                    )
                    used.add(evidence)
    axes[0].set(
        xlim=(lo[0] - origin[0], hi[0] - origin[0]),
        ylim=(lo[1] - origin[1], hi[1] - origin[1]),
        xlabel="East from first fix (m)",
        ylabel="North from first fix (m)",
        title="Cloud, reference, trajectory and draft",
    )
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=7, loc="upper left")
    statistics = {}
    for evidence, values in distances.items():
        if not values:
            continue
        values = np.sort(values)
        axes[1].plot(
            values,
            np.arange(1, len(values) + 1) / len(values),
            label=evidence.replace("_", " "),
            color=colors[evidence],
        )
        statistics[evidence] = {
            "vertices": len(values),
            "median_error_m": float(np.median(values)),
            "p90_error_m": float(np.quantile(values, 0.9)),
            "within_0_3m": float(np.mean(values <= 0.3)),
        }
    axes[1].axvline(0.3, color="black", ls=":", lw=1)
    axes[1].set(
        xlim=(0, 5),
        ylim=(0, 1),
        xlabel="Distance to reference boundary (m)",
        ylabel="Fraction of vertices",
        title="Selected source errors before fitting",
    )
    axes[1].legend(fontsize=8)
    groups = [trajectory[:, 2], np.asarray(generated_z), truth[:, 2]]
    axes[2].boxplot(
        groups, tick_labels=["GNSS", "Cloud ground", "Reference"], showfliers=False
    )
    axes[2].set(
        ylabel="Input elevation (m)", title="Different elevations: XY scoring only"
    )
    fig.tight_layout()
    fig.savefig(args.out / "comparison.png", dpi=180)
    fig.savefig(args.out / "comparison.svg")
    (args.out / "diagnostics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
