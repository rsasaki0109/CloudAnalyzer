"""Draft Autoware Lanelet2 roads from a surveyed cloud and a measured drive."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from ca._rust import core


def build_vector_map(
    cloud: str,
    trajectory: str,
    out_dir: str,
    *,
    forward_lanes: int = 1,
    backward_lanes: int = 1,
    left_hand_traffic: bool = True,
    lane_width: float = 3.5,
    speed_limit: float = 40.0,
    segment_length: float = 50.0,
    anchor_width_prior: bool = True,
    merge_repeated_passes: bool = True,
    existing_map: str | None = None,
    reference_map: str | None = None,
    projection: str | None = None,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> dict[str, Any]:
    """Write a draft, projector metadata, editable IR and evidence report in a NEW directory.

    Inputs must already share a metre coordinate frame. The trajectory follows the outside
    forward lane. existing_map retains its geometry, IDs, rules and coordinates; matching
    intervals are reused and uncovered intervals are added. It can be editable IR JSON or OSM.
    Reference maps supply coordinate metadata only, never geometry. Explicit
    projections are mgrs, utm or transverse_mercator; an origin selects the MGRS tile or
    defines the local origin of the other projections. Omitted metadata uses Autoware Local.
    Review boundaries, repeated passes, travel directions and junctions before using the map.
    """
    module = core()
    if module is None or not hasattr(module, "build_vector_map"):
        raise RuntimeError(
            'vector map drafting needs an updated Rust core: pip install "cloudanalyzer[fast]" (or build rust/crates/ca-py with maturin)'
        )
    inputs = (
        [Path(cloud), Path(trajectory)]
        + ([Path(reference_map)] if reference_map else [])
        + ([Path(existing_map)] if existing_map else [])
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(
            f"output directory already exists: {out}; choose a new directory"
        )
    geo = None
    if existing_map and (
        reference_map or projection or origin_lat is not None or origin_lon is not None
    ):
        raise ValueError(
            "existing_map retains its coordinates; omit reference_map, projection and origin"
        )
    if reference_map and (
        projection or origin_lat is not None or origin_lon is not None
    ):
        raise ValueError("choose reference_map or explicit projection and origin")
    if projection is not None:
        if projection not in {"mgrs", "utm", "transverse_mercator"}:
            raise ValueError(
                "projection must be mgrs, utm or transverse_mercator; omit it for Local"
            )
        if origin_lat is None or origin_lon is None:
            raise ValueError("a projection requires origin_lat and origin_lon")
        if (
            not math.isfinite(origin_lat)
            or not -80 <= origin_lat <= 84
            or not math.isfinite(origin_lon)
            or not -180 <= origin_lon <= 180
        ):
            raise ValueError(
                "origin requires finite latitude [-80,84] and longitude [-180,180]"
            )
        geo = json.dumps(
            {"projection": projection, "origin": {"lat": origin_lat, "lon": origin_lon}}
        )
    elif origin_lat is not None or origin_lon is not None:
        raise ValueError("an origin requires a projection")
    options = json.dumps(
        {
            "forward_lanes": forward_lanes,
            "backward_lanes": backward_lanes,
            "left_hand_traffic": left_hand_traffic,
            "lane_width": lane_width,
            "speed_limit": speed_limit,
            "segment_length": segment_length,
            "anchor_width_prior": anchor_width_prior,
            "merge_repeated_passes": merge_repeated_passes,
        },
        allow_nan=False,
    )
    payload = json.loads(
        module.build_vector_map(
            str(inputs[0]), str(inputs[1]), options, reference_map, geo, existing_map
        )
    )
    report: dict[str, Any] = payload["report"]
    report["inputs"] = {
        "cloud": str(inputs[0].resolve()),
        "trajectory": str(inputs[1].resolve()),
        "reference_map": str(Path(reference_map).resolve()) if reference_map else None,
        "existing_map": str(Path(existing_map).resolve()) if existing_map else None,
    }
    report["options"] = json.loads(options)
    names = {
        "map": "lanelet2_map.osm",
        "projector": "map_projector_info.yaml",
        "editable_map": "vector_map.json",
        "report": "report.json",
    }
    report["files"] = {key: str(out / name) for key, name in names.items()}
    report["editing"] = {
        "command": "vectormap",
        "args": ["mcp", str(out / names["map"])],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stage all artifacts together; failed generation never leaves half a result.
    with tempfile.TemporaryDirectory(
        prefix=".vector-map-", dir=out.parent
    ) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        content = {
            names["map"]: payload["osm"],
            names["projector"]: payload["projector_info"],
            names["editable_map"]: payload["map_json"],
            names["report"]: json.dumps(report, indent=2) + "\n",
        }
        for name, text in content.items():
            with (stage / name).open("w", encoding="utf-8", newline="\n") as file:
                file.write(text)
        os.rename(stage, out)
    return report
