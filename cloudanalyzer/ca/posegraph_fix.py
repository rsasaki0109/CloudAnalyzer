"""Fix a SLAM map from the command line, as the web app's pose graph panel does.

A session folder holds a poses file (a g2o graph, or a KITTI / TUM trajectory)
and one scan per pose (PCD, PLY, LAS/LAZ, XYZ or KITTI ``.bin``), named by frame
number. :func:`fix_session` loads it, finds loops with ICP, ties the keyframes to
IMU gravity, optimises, optionally leaves out dynamic points, and writes the
poses, the graph and the map, returning a JSON-ready report.

The algorithms are the Rust core's (``cloudanalyzer_core.PoseGraph``), the same
as the browser's, running natively on all cores without the browser's memory
limit.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np

from ca._rust import core

SCAN_SUFFIXES = {".pcd", ".ply", ".bin", ".las", ".laz", ".xyz", ".pts"}
POSE_SUFFIXES = {".g2o", ".tum", ".kitti", ".txt", ".csv"}
NOT_POSES = re.compile(r"^(calib|times)\.txt$", re.IGNORECASE)


def _core():
    module = core()
    if module is None or not hasattr(module, "PoseGraph"):
        raise RuntimeError(
            "the pose graph needs the Rust core: pip install \"cloudanalyzer[fast]\" "
            "(or build rust/crates/ca-py with maturin)"
        )
    return module


def _numbers(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def poses_file(folder: Path) -> Path | None:
    """The poses file in ``folder``: a g2o graph, else a trajectory, preferring names like "poses"."""
    candidates = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in POSE_SUFFIXES and not NOT_POSES.match(p.name)
    )
    for test in (
        lambda p: p.suffix.lower() == ".g2o",
        lambda p: re.search(r"pose|traj|odom|gt", p.name, re.IGNORECASE) is not None,
        lambda p: True,
    ):
        for p in candidates:
            if test(p):
                return p
    return None


def kitti_extrinsic(folder: Path) -> np.ndarray | None:
    """KITTI's Velodyne-to-camera transform (``Tr:`` in a ``calib.txt`` next to the scans), as 4x4."""
    calib = folder / "calib.txt"
    if not calib.exists():
        return None
    for line in calib.read_text().splitlines():
        if line.startswith("Tr:"):
            m = np.eye(4)
            m[:3] = np.array(_numbers(line[3:])).reshape(3, 4)
            return m
    return None


def _quaternion_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """KITTI (12 numbers a line) or TUM (``t x y z qx qy qz qw``) poses as (N, 4, 4), and TUM's timestamps."""
    rows = [
        _numbers(line)
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise ValueError(f"{path.name} holds no poses")
    width = len(rows[0])
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    if width == 12 and path.suffix.lower() != ".tum":
        for k, v in enumerate(rows):
            poses[k, :3] = np.array(v).reshape(3, 4)
        return poses, None
    if width == 8:
        for k, (_, x, y, z, qx, qy, qz, qw) in enumerate(rows):
            poses[k, :3, :3] = _quaternion_matrix(qx, qy, qz, qw)
            poses[k, :3, 3] = (x, y, z)
        return poses, np.array([v[0] for v in rows])
    raise ValueError(f"{path.name} is neither KITTI (12 numbers a line) nor TUM (8)")


def frame_number(name: str) -> int | None:
    """The last run of digits in a file's base name (``000123.pcd`` -> 123)."""
    digits = re.findall(r"\d+", Path(name).stem)
    return int(digits[-1]) if digits else None


def match_scans(scans: list[Path], node_ids: list[int]) -> list[int | None]:
    """Node index per scan: by the number in its name when that names a node for most scans, else in name order."""
    by_id = {node_id: i for i, node_id in enumerate(node_ids)}
    numbers = [frame_number(s.name) for s in scans]
    numbered = [None if n is None else by_id.get(n) for n in numbers]
    hits = sum(i is not None for i in numbered)
    if hits > 0 and hits >= len(scans) / 2:
        return numbered
    if len(scans) == len(node_ids):
        order = sorted(range(len(scans)), key=lambda k: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", scans[k].name)])
        out: list[int | None] = [None] * len(scans)
        for i, k in enumerate(order):
            out[k] = i
        return out
    raise ValueError(
        f"cannot match {len(scans)} scans to {len(node_ids)} poses: name the scans by frame number "
        "(e.g. 000042.pcd) or give one scan per pose"
    )


def read_ups(path: Path) -> dict[int, np.ndarray]:
    """Up directions in scan coordinates by frame number.

    From a KITTI OXTS folder (``data/0000000042.txt``: roll and pitch are fields 4
    and 5; ``calib_imu_to_velo.txt`` turns them into the Velodyne frame) or from
    text files of ``frame ux uy uz`` lines.
    """
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*.txt"))
    rotation = np.eye(3)
    for calib in (p for p in files if p.name.lower() == "calib_imu_to_velo.txt"):
        for line in calib.read_text().splitlines():
            if line.startswith("R:"):
                rotation = np.array(_numbers(line[2:])).reshape(3, 3)
    ups: dict[int, np.ndarray] = {}
    for file in files:
        if file.name.lower().startswith("calib"):
            continue
        text = file.read_text()
        frame = re.fullmatch(r"(\d+)\.txt", file.name)
        values = text.split()
        if frame and len(values) >= 30:
            # The IMU is turned by Rz(yaw) Ry(pitch) Rx(roll): up in its frame is that matrix's last row.
            roll, pitch = float(values[3]), float(values[4])
            imu = np.array([-math.sin(pitch), math.cos(pitch) * math.sin(roll), math.cos(pitch) * math.cos(roll)])
            ups[int(frame.group(1))] = rotation @ imu
            continue
        for line in text.splitlines():
            v = line.split()
            if len(v) == 4:
                try:
                    ups[int(float(v[0]))] = np.array([float(x) for x in v[1:]])
                except ValueError:
                    pass
    return ups


def write_ply(path: Path, positions: np.ndarray, fields: dict[str, np.ndarray]) -> None:
    """Binary PLY: double x, y, z (exact georeferenced coordinates) and float fields."""
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(positions)}",
        "property double x",
        "property double y",
        "property double z",
        *[f"property float {name}" for name in fields],
        "end_header",
    ]
    dtype = [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), *[(name, "<f4") for name in fields]]
    rows = np.empty(len(positions), dtype=dtype)
    rows["x"], rows["y"], rows["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    for name, values in fields.items():
        rows[name] = values
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(rows.tobytes())


def ate(estimate: np.ndarray, truth: np.ndarray) -> dict:
    """Trajectory error of positions, raw and after SE(3) alignment (metres)."""
    n = min(len(estimate), len(truth))
    est, ref = estimate[:n, :3, 3], truth[:n, :3, 3]
    raw = np.linalg.norm(est - ref, axis=1)
    mu_e, mu_r = est.mean(0), ref.mean(0)
    u, _, vt = np.linalg.svd((ref - mu_r).T @ (est - mu_e))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ d @ vt
    aligned = np.linalg.norm((est - mu_e) @ r.T + mu_r - ref, axis=1)
    return {
        "poses": n,
        "ate_rmse": float(np.sqrt((raw**2).mean())),
        "ate_rmse_aligned": float(np.sqrt((aligned**2).mean())),
        "end_error": float(raw[-1]),
    }


def fix_session(
    folder: str,
    out_dir: str | None = None,
    *,
    voxel: float = 0.4,
    sigma_t: float = 0.05,
    sigma_r_deg: float = 0.25,
    loops: bool = True,
    loop_options: dict | None = None,
    gravity: str | None = None,
    gravity_sigma_deg: float = 0.1,
    remove_dynamic: bool = False,
    map_voxel: float = 0.2,
    truth: str | None = None,
    progress=None,
) -> dict:
    """Load a session folder, fix it and write the results; returns the report."""
    cloudanalyzer_core = _core()
    say = progress or (lambda message: None)
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(folder)
    poses_path = poses_file(root)
    if poses_path is None:
        raise ValueError(f"no poses file (.g2o, .txt, .tum, .kitti) in {folder}")
    report: dict = {"session": str(root), "poses_file": poses_path.name, "timings_s": {}}
    clock = time.perf_counter()

    timestamps = None
    if poses_path.suffix.lower() == ".g2o":
        graph = cloudanalyzer_core.PoseGraph.from_g2o(poses_path.read_text())
    else:
        poses, timestamps = read_trajectory(poses_path)
        graph = cloudanalyzer_core.PoseGraph.from_poses(poses, sigma_t, sigma_r_deg)
    node_ids = list(graph.node_ids)

    scans = sorted(p for p in root.iterdir() if p.suffix.lower() in SCAN_SUFFIXES)
    if not scans:
        raise ValueError(f"no scans next to {poses_path.name}")
    extrinsic = kitti_extrinsic(root)
    matched = match_scans(scans, node_ids)
    scan_points = 0
    for k, (path, node) in enumerate(zip(scans, matched)):
        if node is None:
            continue
        cloud = cloudanalyzer_core.read(str(path))
        positions = cloud["positions"]
        if extrinsic is not None:
            positions = positions @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        scan_points += graph.set_scan(node, np.ascontiguousarray(positions), cloud.get("intensity"), voxel)
        if k % 200 == 0:
            say(f"scan {k + 1} of {len(scans)}")
    report.update(
        nodes=graph.node_count,
        scans=sum(m is not None for m in matched),
        unmatched_scans=sum(m is None for m in matched),
        scan_points=scan_points,
        scans_moved_by_calib_tr=extrinsic is not None,
    )
    report["timings_s"]["open"] = round(time.perf_counter() - clock, 2)

    if loops:
        clock = time.perf_counter()
        say("finding loops")
        found = graph.find_loops(**(loop_options or {}))
        report["loops"] = {
            "candidates": found["candidates"],
            "added": len(found["added"]),
            "implausible": found["implausible"],
            "same_place_retries": sum(1 for *_, retried in found["added"] if retried),
            "pairs": [[node_ids[a], node_ids[b], round(f, 3)] for a, b, f, _ in found["added"]],
        }
        if found["added"]:
            report["loops"]["optimized"] = graph.optimize()
        report["timings_s"]["loops"] = round(time.perf_counter() - clock, 2)

    if gravity:
        clock = time.perf_counter()
        say("tying to gravity")
        ups = read_ups(Path(gravity))
        nodes = [i for i, node_id in enumerate(node_ids) if node_id in ups]
        if not nodes:
            raise ValueError(f"no up direction in {gravity} matches a node")
        vectors = np.array([ups[node_ids[i]] for i in nodes])
        if extrinsic is not None:
            vectors = vectors @ extrinsic[:3, :3].T
        report["gravity"] = {"tied": graph.set_gravity(nodes, vectors, gravity_sigma_deg)}
        report["gravity"]["optimized"] = graph.optimize()
        report["timings_s"]["gravity"] = round(time.perf_counter() - clock, 2)

    if remove_dynamic:
        clock = time.perf_counter()
        say("finding dynamic points")
        dynamic, total = graph.detect_dynamic()
        report["dynamic"] = {"points": dynamic, "of": total, "share": dynamic / max(total, 1)}
        report["timings_s"]["dynamic"] = round(time.perf_counter() - clock, 2)

    if truth:
        truth_poses, _ = read_trajectory(Path(truth))
        report["ate"] = {"before": ate(graph.poses(initial=True), truth_poses), "after": ate(graph.poses(), truth_poses)}

    if out_dir:
        clock = time.perf_counter()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = poses_path.stem
        files = {"g2o": out / f"{stem}_fixed.g2o", "kitti": out / f"{stem}_fixed_kitti.txt"}
        files["g2o"].write_text(graph.to_g2o())
        files["kitti"].write_text(graph.to_kitti())
        if timestamps is not None:
            files["tum"] = out / f"{stem}_fixed.tum"
            files["tum"].write_text(graph.to_tum(list(timestamps)))
        say("building the map")
        part = "static" if remove_dynamic else "all"
        m = graph.map(voxel=map_voxel, part=part, correction=True)
        fields = {k: m[k] for k in ("intensity", "correction") if k in m}
        files["map"] = out / f"{stem}_map.ply"
        write_ply(files["map"], m["positions"], fields)
        report["map_points"] = int(len(m["positions"]))
        if remove_dynamic:
            d = graph.map(voxel=map_voxel, part="dynamic")
            files["dynamic"] = out / f"{stem}_dynamic.ply"
            write_ply(files["dynamic"], d["positions"], {k: d[k] for k in ("intensity",) if k in d})
        report["outputs"] = {k: str(v) for k, v in files.items()}
        report["timings_s"]["write"] = round(time.perf_counter() - clock, 2)
    return report
