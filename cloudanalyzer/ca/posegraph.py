"""Pose graph validation utilities (g2o + TUM + keyframe PCD session layout).

This module is intentionally lightweight: it does not optimize graphs. It only
parses and validates common mapping-session artifacts used by manual loop-closure
workflows (e.g., pose_graph.g2o + optimized_poses_tum.txt + key_point_frame/*.pcd).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ca.trajectory import load_trajectory


@dataclass(slots=True)
class G2OParseSummary:
    path: str
    vertex_tags: dict[str, int]
    edge_tags: dict[str, int]
    vertex_ids: set[int]
    edge_pairs: list[tuple[int, int]]
    self_loop_count: int
    duplicate_undirected_edge_count: int
    connected_component_count: int
    isolated_vertex_count: int
    malformed_lines: int


def _connected_components(vertices: set[int], undirected_edges: set[tuple[int, int]]) -> int:
    """Count connected components in an undirected graph (vertices may be isolated)."""
    parent: dict[int, int] = {v: v for v in vertices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in undirected_edges:
        union(a, b)

    roots = {find(v) for v in vertices}
    return int(len(roots)) if vertices else 0


def discover_session_paths(session_root: str, map_name: str = "map.pcd") -> dict:
    """Discover common manual-loop-closure session artifacts under a root directory."""
    root = Path(session_root)
    if not root.exists():
        raise FileNotFoundError(session_root)
    if not root.is_dir():
        raise ValueError(f"session_root must be a directory: {session_root}")

    g2o = root / "pose_graph.g2o"
    tum = root / "optimized_poses_tum.txt"
    key_point_frame = root / "key_point_frame"
    map_path = root / map_name

    return {
        "session_root": str(root),
        "g2o_path": str(g2o) if g2o.exists() else None,
        "tum_path": str(tum) if tum.exists() else None,
        "key_point_frame_dir": str(key_point_frame) if key_point_frame.exists() else None,
        "map_path": str(map_path) if map_path.exists() else None,
        "map_name": map_name,
        "expected": {
            "g2o_path": str(g2o),
            "tum_path": str(tum),
            "key_point_frame_dir": str(key_point_frame),
            "map_path": str(map_path),
        },
        "exists": {
            "g2o_path": bool(g2o.exists()),
            "tum_path": bool(tum.exists()),
            "key_point_frame_dir": bool(key_point_frame.exists()),
            "map_path": bool(map_path.exists()),
        },
    }


def _iter_nonempty_lines(path: Path) -> Iterable[tuple[int, str]]:
    for idx, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield idx, line


def parse_g2o_summary(path: str) -> G2OParseSummary:
    """Parse a g2o file and return counts + vertex/edge connectivity.

    Supported (counted) tags include:
    - vertices: VERTEX_SE3:QUAT, VERTEX_SE2, ...
    - edges: EDGE_SE3:QUAT, EDGE_SE2, ...
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    vertex_tags: dict[str, int] = {}
    edge_tags: dict[str, int] = {}
    vertex_ids: set[int] = set()
    edge_pairs: list[tuple[int, int]] = []
    self_loops = 0
    undirected_counts: dict[tuple[int, int], int] = {}
    malformed = 0

    for _lineno, line in _iter_nonempty_lines(p):
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag.startswith("VERTEX_"):
            vertex_tags[tag] = vertex_tags.get(tag, 0) + 1
            # vertex id is typically the 2nd token
            if len(parts) < 2:
                malformed += 1
                continue
            try:
                vid = int(parts[1])
            except ValueError:
                malformed += 1
                continue
            vertex_ids.add(vid)
        elif tag.startswith("EDGE_"):
            edge_tags[tag] = edge_tags.get(tag, 0) + 1
            # edge endpoints are typically the 2nd and 3rd tokens
            if len(parts) < 3:
                malformed += 1
                continue
            try:
                a = int(parts[1])
                b = int(parts[2])
            except ValueError:
                malformed += 1
                continue
            edge_pairs.append((a, b))
            if a == b:
                self_loops += 1
            else:
                u, v = (a, b) if a <= b else (b, a)
                undirected_counts[u, v] = undirected_counts.get((u, v), 0) + 1
        else:
            # ignore other g2o records (PARAMS_*, FIX, etc.)
            continue

    duplicate_edges = sum(max(0, c - 1) for c in undirected_counts.values() if c > 1)
    all_vertices: set[int] = set(vertex_ids) | {v for pair in edge_pairs for v in pair}
    unique_undirected = {k for k, c in undirected_counts.items() if c > 0}
    comp_count = _connected_components(all_vertices, unique_undirected)
    endpoint_vertices: set[int] = {v for a, b in undirected_counts.keys() for v in (a, b)} if undirected_counts else set()
    isolated = (
        len([v for v in all_vertices if v not in endpoint_vertices])
        if all_vertices
        else 0
    )

    return G2OParseSummary(
        path=path,
        vertex_tags=vertex_tags,
        edge_tags=edge_tags,
        vertex_ids=vertex_ids,
        edge_pairs=edge_pairs,
        self_loop_count=int(self_loops),
        duplicate_undirected_edge_count=int(duplicate_edges),
        connected_component_count=int(comp_count),
        isolated_vertex_count=int(isolated),
        malformed_lines=malformed,
    )


def validate_key_point_frame_dir(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if not p.is_dir():
        raise ValueError(f"key_point_frame must be a directory: {path}")
    pcds = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".pcd"])
    return {
        "path": str(p),
        "pcd_count": int(len(pcds)),
        "sample_files": [str(x.name) for x in pcds[:5]],
    }


def validate_posegraph_session(
    g2o_path: str,
    tum_path: str | None = None,
    key_point_frame_dir: str | None = None,
) -> dict:
    """Validate common manual-loop-closure session inputs."""
    g2o = parse_g2o_summary(g2o_path)

    missing_vertices: int = 0
    if g2o.edge_pairs:
        endpoints = {v for pair in g2o.edge_pairs for v in pair}
        missing_vertices = len(endpoints - g2o.vertex_ids)

    result: dict = {
        "g2o": {
            "path": g2o.path,
            "vertex_tags": g2o.vertex_tags,
            "edge_tags": g2o.edge_tags,
            "vertex_count": int(len(g2o.vertex_ids)),
            "edge_count": int(len(g2o.edge_pairs)),
            "self_loops": int(g2o.self_loop_count),
            "duplicate_undirected_edges": int(g2o.duplicate_undirected_edge_count),
            "connected_components": int(g2o.connected_component_count),
            "isolated_vertices": int(g2o.isolated_vertex_count),
            "malformed_lines": int(g2o.malformed_lines),
            "missing_vertex_references": int(missing_vertices),
        }
    }

    if tum_path is not None:
        traj = load_trajectory(tum_path)
        result["tum"] = {
            "path": tum_path,
            "num_poses": traj["num_poses"],
            "timestamp_start": float(traj["timestamps"][0]),
            "timestamp_end": float(traj["timestamps"][-1]),
            "duration_s": float(traj["timestamps"][-1] - traj["timestamps"][0]),
            "bbox_min": [float(x) for x in np.min(traj["positions"], axis=0)],
            "bbox_max": [float(x) for x in np.max(traj["positions"], axis=0)],
        }

    if key_point_frame_dir is not None:
        result["key_point_frame"] = validate_key_point_frame_dir(key_point_frame_dir)

    # Status split into hard errors vs informational warnings.
    errors: list[str] = []
    warnings: list[str] = []
    if result["g2o"]["vertex_count"] == 0:
        errors.append("g2o: no vertices parsed")
    if result["g2o"]["edge_count"] == 0:
        errors.append("g2o: no edges parsed")
    if result["g2o"]["malformed_lines"] > 0:
        errors.append("g2o: malformed lines present")
    if result["g2o"]["missing_vertex_references"] > 0:
        errors.append("g2o: edges reference missing vertices")
    if g2o.self_loop_count > 0:
        warnings.append(f"g2o: {g2o.self_loop_count} self-loop edge(s) (from==to)")
    if g2o.duplicate_undirected_edge_count > 0:
        warnings.append(
            f"g2o: {g2o.duplicate_undirected_edge_count} duplicate undirected edge(s) beyond the first"
        )
    if g2o.edge_pairs and g2o.connected_component_count > 1:
        errors.append(f"g2o: graph is disconnected ({g2o.connected_component_count} components)")
    if g2o.isolated_vertex_count > 0 and g2o.edge_pairs:
        errors.append(f"g2o: {g2o.isolated_vertex_count} isolated vertex/vertices")
    if "tum" in result and result["tum"]["num_poses"] < 2:
        errors.append("tum: fewer than 2 poses")

    result["summary"] = {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
    return result

