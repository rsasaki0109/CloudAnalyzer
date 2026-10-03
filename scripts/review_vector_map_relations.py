"""Reproduce explicit equipment review on the frozen generated map, never labels.

Source geometry is read only; control choices are operator drafts. This is a
single development scene and does not establish legal or semantic accuracy.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "benchmarks/vector-map/hard-intersection/equipment-relations/operator-inputs.json"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def links(data: dict) -> list[dict]:
    return [
        {
            "id": r["id"],
            "rule": r["rule"],
            "lanes": r.get("lanes", []),
            "controlled_crosswalks": r.get("controlled_crosswalks", []),
        }
        for r in data["regulatory_elements"]
    ]


def warn(payload: dict) -> list[dict]:
    return [
        i
        for i in payload["report"]["validation"]["issues"]
        + payload["report"]["autoware_issues"]
        if i["severity"] == "warning" and not i["code"].startswith("lanelet2.")
    ]


def heading(edge: list) -> float:
    return math.atan2(edge[-1][1] - edge[0][1], edge[-1][0] - edge[0][0])


def angle(a: float, b: float) -> float:
    return math.degrees(math.acos(min(1.0, abs(math.cos(a - b)))))


def review(source: Path, cloud: Path, out: Path) -> dict:
    import cloudanalyzer_core as core

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if out.exists():
        raise FileExistsError(str(out))
    before = json.loads(source.read_text(encoding="utf-8"))
    source_hash, cloud_hash = digest(source), digest(cloud)
    if (
        source_hash != config["input_map_sha256"]
        or cloud_hash != config["input_cloud_sha256"]
    ):
        raise ValueError(
            "operator review belongs to another input map/source; fresh review required"
        )
    initial = json.loads(core.edit_vector_map_relations(str(source)))
    quality_before = json.loads(core.audit_vector_map_quality(str(cloud), str(source)))
    reports = []
    with tempfile.TemporaryDirectory(prefix="equipment-relations-") as temporary:
        current = source
        for edit in config["edits"]:
            payload = json.loads(
                core.edit_vector_map_relations(str(current), json.dumps(edit))
            )
            assert payload["report"]["edit"]["changed"]
            reports.append(payload["report"]["edit"])
            current = Path(temporary) / f"{edit['rule_id']}.json"
            current.write_text(payload["map_json"], encoding="utf-8", newline="\n")
        after = json.loads(payload["map_json"])
        for key in (
            "metadata",
            "boundaries",
            "lanes",
            "roads",
            "junctions",
            "stop_lines",
            "traffic_signals",
            "crosswalks",
        ):
            assert after.get(key, []) == before.get(key, []), key
        original = {r["id"]: r for r in before["regulatory_elements"]}
        modified = {r["rule_id"] for r in config["edits"]}
        for rule in after["regulatory_elements"]:
            if rule["id"] not in modified:
                assert rule == original[rule["id"]]
        for edit in config["edits"]:
            repeat = json.loads(
                core.edit_vector_map_relations(str(current), json.dumps(edit))
            )
            assert not repeat["report"]["edit"]["changed"]
            assert repeat["map_json"] == payload["map_json"]
        osm = Path(temporary) / "reviewed.osm"
        osm.write_text(payload["osm"], encoding="utf-8", newline="\n")
        reloaded = json.loads(core.edit_vector_map_relations(str(osm)))
        assert links(json.loads(reloaded["map_json"])) == links(after)
        assert warn(reloaded) == warn(payload)
        quality_after = json.loads(core.audit_vector_map_quality(str(cloud), str(osm)))
        assert quality_before["quality"] == quality_after["quality"]
        assert quality_after["quality"]["low_support_lanes"] == []
        assert payload["report"]["validation"]["counts"]["errors"] == 0
        root = ET.fromstring(payload["osm"])
        members = {
            int(r.attrib["id"]): [
                int(m.attrib["ref"])
                for m in r.findall("member")
                if m.attrib.get("role") == "regulatory_element"
            ]
            for r in root.findall("relation")
        }
        assert 173 in members[156] and 173 not in members[83]
        assert len(warn(initial)) == 19 and len(warn(payload)) == 18
        missing = [i for i in warn(payload) if "without_stop_line" in i["code"]]
        assert sorted(i["entity"]["id"] for i in missing) == config["unresolved_rules"]
        # Geometric direction is unsigned: the point-cloud panel does not prove
        # its front face or the legal control target. Report competing crossings.
        head = next(s for s in before["traffic_signals"] if s["id"] == 172)
        h = head["geometry"]
        mid = [(h[0][i] + h[-1][i]) / 2 for i in (0, 1)]
        candidates = []
        for c in before["crosswalks"]:
            edges = [c["left_edge"], c["right_edge"]]
            near = min(
                math.hypot(p[0] - mid[0], p[1] - mid[1])
                for e in edges
                for p in (e[0], e[-1])
            )
            candidates.append(
                {
                    "crosswalk": c["id"],
                    "nearest_edge_endpoint_xy_m": near,
                    "unsigned_housing_normal_vs_walking_degrees": angle(
                        heading(h) + math.pi / 2, heading(edges[0])
                    ),
                }
            )
        report = {
            "reference_inputs": [],
            "input_map_sha256": source_hash,
            "input_cloud_sha256": cloud_hash,
            "operator_inputs_sha256": digest(CONFIG),
            "native_sha256": digest(Path(core._core.__file__)),
            "edits": reports,
            "physical_map_exact": True,
            "unmodified_rules_exact": True,
            "repeated_edits_exact_noop": True,
            "lanelet2_targets_and_warnings_roundtrip": True,
            "source_quality_before_after_equal": True,
            "source_quality": quality_after["quality"],
            "warnings_before": warn(initial),
            "warnings_after": warn(payload),
            "export_warnings": [
                i
                for i in payload["report"]["autoware_issues"]
                if i["severity"] == "warning" and i["code"].startswith("lanelet2.")
            ],
            "unresolved_rules": config["unresolved_rules"],
            "pedestrian172_candidates": sorted(
                candidates, key=lambda c: c["nearest_edge_endpoint_xy_m"]
            ),
            "limitations": config["limitations"],
        }
        assert digest(source) == source_hash and digest(cloud) == cloud_hash
        out.mkdir(parents=True)
        for name, key in (
            ("reviewed.json", "map_json"),
            ("reviewed.osm", "osm"),
            ("map_projector_info.yaml", "projector_info"),
        ):
            (out / name).write_text(payload[key], encoding="utf-8", newline="\n")
        report["output_map_sha256"] = digest(out / "reviewed.osm")
        (out / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    return report


def plot_review(source: Path, cloud: Path, output: Path) -> None:
    """Plot original returns and unchanged equipment, with no teacher overlay."""
    import laspy
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(source.read_text(encoding="utf-8"))
    points = laspy.read(cloud)
    origin = np.array([-9341.0, -40867.0])
    xy = np.column_stack((points.x, points.y)) - origin
    z = np.asarray(points.z)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, limits, title, ids in zip(
        axes,
        [(62, 80, 42, 66), (32, 57, 70, 92)],
        [
            "Vehicle 168 / stop marking 164: reviewed draft",
            "Pedestrian 172 / crossing 156: reviewed draft",
        ],
        [(168, 164), (172, 156)],
    ):
        mask = (
            (xy[:, 0] > limits[0])
            & (xy[:, 0] < limits[1])
            & (xy[:, 1] > limits[2])
            & (xy[:, 1] < limits[3])
        )
        q, zz = xy[mask], z[mask]
        ax.scatter(
            q[::3, 0],
            q[::3, 1],
            c=zz[::3],
            s=0.4,
            cmap="gray",
            alpha=0.4,
            vmin=28,
            vmax=34,
        )
        for crossing in data["crosswalks"]:
            for edge in ("left_edge", "right_edge"):
                p = np.array(crossing[edge])[:, :2] - origin
                ax.plot(
                    p[:, 0],
                    p[:, 1],
                    c="#158fca",
                    lw=2 if crossing["id"] == ids[1] else 0.6,
                    alpha=0.9 if crossing["id"] == ids[1] else 0.4,
                )
            if ids[0] == 172 and crossing["id"] in (152, 156):
                center = (
                    np.array(crossing["left_edge"])[:, :2].mean(0)
                    + np.array(crossing["right_edge"])[:, :2].mean(0)
                ) / 2 - origin
                ax.text(*center, str(crossing["id"]), fontsize=9, clip_on=True)
        for stop in data["stop_lines"]:
            p = np.array(stop["geometry"])[:, :2] - origin
            ax.plot(p[:, 0], p[:, 1], c="#e48912", lw=3)
            ax.text(*p.mean(0), str(stop["id"]), fontsize=9, clip_on=True)
        for head in data["traffic_signals"]:
            p = np.array(head["geometry"])[:, :2] - origin
            mid = p.mean(0)
            ax.plot(p[:, 0], p[:, 1], c="#ac2380", lw=4)
            ax.text(
                mid[0] + 0.3, mid[1] + 0.3, str(head["id"]), fontsize=9, clip_on=True
            )
            if head["id"] == ids[0]:
                d = p[-1] - p[0]
                normal = np.array([-d[1], d[0]])
                normal = normal / np.linalg.norm(normal) * 2.5
                ax.plot(
                    [mid[0] - normal[0], mid[0] + normal[0]],
                    [mid[1] - normal[1], mid[1] + normal[1]],
                    "--",
                    c="#ac2380",
                    lw=1,
                )
        ax.set_xlim(limits[:2])
        ax.set_ylim(limits[2:])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X relative to (-9341, -40867) [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Original source returns and unchanged generated equipment\nUnsigned housing normal shows geometry only; legal control and phases remain unverified",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        type=Path,
        default=ROOT / "notes/supported-intersection-proof/reviewed.json",
    )
    parser.add_argument(
        "--cloud",
        type=Path,
        default=ROOT / "notes/hard-intersection-prepared-v2/geometry.las",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = review(args.map, args.cloud, args.out)
    if args.plot:
        plot_review(args.map, args.cloud, args.plot)
    print(
        json.dumps(
            {
                "physical_map_exact": result["physical_map_exact"],
                "warnings_before": len(result["warnings_before"]),
                "warnings_after": len(result["warnings_after"]),
                "unresolved_rules": result["unresolved_rules"],
            },
            indent=2,
        )
    )
