"""Check exact replay and overlapping trajectory subsets on external real data.

Requires an updated native Rust core and cloudanalyzer. Inputs are never modified.
This checks integration behavior, not accuracy or independent repeated surveys.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ca.vector_map import build_vector_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud", type=Path)
    parser.add_argument("trajectory", type=Path, help="timestamped XYZ CSV")
    parser.add_argument("out", type=Path, help="new results directory")
    parser.add_argument("--right-hand", action="store_true")
    parser.add_argument("--reference-map", help="coordinate metadata only")
    args = parser.parse_args()
    with args.trajectory.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    if len(rows) < 9 or [v.lower() for v in rows[0]] != ["timestamp", "x", "y", "z"]:
        parser.error(
            "trajectory must contain a timestamp,x,y,z header and at least 8 poses"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    options = {"left_hand_traffic": not args.right_hand}
    first = build_vector_map(
        str(args.cloud),
        str(args.trajectory),
        str(args.out / "first"),
        reference_map=args.reference_map,
        **options,
    )
    replay = build_vector_map(
        str(args.cloud),
        str(args.trajectory),
        str(args.out / "replay"),
        existing_map=str(args.out / "first/vector_map.json"),
        **options,
    )

    def read(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    exact = read(args.out / "first/vector_map.json") == read(
        args.out / "replay/vector_map.json"
    )
    poses = rows[1:]
    for name, subset in [
        ("a", poses[: (len(poses) * 5 + 7) // 8]),
        ("b", poses[len(poses) * 3 // 8 :]),
    ]:
        with (args.out / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
            csv.writer(file, lineterminator="\n").writerows([rows[0], *subset])
    a = build_vector_map(
        str(args.cloud),
        str(args.out / "a.csv"),
        str(args.out / "overlap-a"),
        reference_map=args.reference_map,
        **options,
    )
    b = build_vector_map(
        str(args.cloud),
        str(args.out / "b.csv"),
        str(args.out / "overlap-b"),
        existing_map=str(args.out / "overlap-a/vector_map.json"),
        **options,
    )
    old = read(args.out / "overlap-a/vector_map.json")
    combined = read(args.out / "overlap-b/vector_map.json")
    by_id = {
        key: {item["id"]: item for item in combined[key]}
        for key in ["lanes", "boundaries"]
    }
    retained = all(
        by_id[key].get(item["id"]) == item
        for key in ["lanes", "boundaries"]
        for item in old[key]
    )
    summary = {
        "experiment": "exact replay and overlapping subsets of one recorded drive",
        "accuracy_evaluated": False,
        "replay_map_equal": exact,
        "overlap_existing_geometry_and_rules_retained": retained,
        "first": first["extraction"],
        "replay": replay["extraction"],
        "overlap_a": a["extraction"],
        "overlap_b": b["extraction"],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not exact or replay["extraction"]["lanes"] != 0 or not retained:
        raise SystemExit("integration regression: inspect the saved maps and reports")


if __name__ == "__main__":
    main()
