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


def keyframes(poses: np.ndarray, spacing: float) -> list[int]:
    """Rows of ``poses`` at least ``spacing`` metres of travel apart (the first and last always); all when 0."""
    if spacing <= 0 or len(poses) < 3:
        return list(range(len(poses)))
    kept = [0]
    travel = 0.0
    for k in range(1, len(poses)):
        travel += float(np.linalg.norm(poses[k, :3, 3] - poses[k - 1, :3, 3]))
        if travel >= spacing:
            kept.append(k)
            travel = 0.0
    if kept[-1] != len(poses) - 1:
        kept.append(len(poses) - 1)
    return kept


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


class Session:
    """A session folder loaded into a ``cloudanalyzer_core.PoseGraph``."""

    def __init__(
        self,
        folder: str,
        *,
        poses: str | None = None,
        keyframe_spacing: float = 0.0,
        voxel: float = 0.4,
        sigma_t: float = 0.05,
        sigma_r_deg: float = 0.25,
        progress=None,
    ):
        cloudanalyzer_core = _core()
        say = progress or (lambda message: None)
        root = Path(folder)
        if not root.is_dir():
            raise FileNotFoundError(folder)
        poses_path = Path(poses) if poses else poses_file(root)
        if poses_path is None:
            raise ValueError(f"no poses file (.g2o, .txt, .tum, .kitti) in {folder}")
        if not poses_path.is_file():
            raise FileNotFoundError(str(poses_path))
        self.root = root
        self.poses_path = poses_path
        self.timestamps = None
        scans = sorted(p for p in root.iterdir() if p.suffix.lower() in SCAN_SUFFIXES)
        if not scans:
            raise ValueError(f"no scans next to {poses_path.name}")
        if poses_path.suffix.lower() == ".g2o":
            self.graph = cloudanalyzer_core.PoseGraph.from_g2o(poses_path.read_text())
            self.node_ids = list(self.graph.node_ids)
            matched = match_scans(scans, self.node_ids)
        else:
            rows, stamps = read_trajectory(poses_path)
            # Scans match rows (by frame number or in order), then only keyframes stay: nodes
            # keep their row as id, so scans and IMU frames still find them.
            by_row = match_scans(scans, list(range(len(rows))))
            kept = keyframes(rows, keyframe_spacing)
            node_of_row = {row: i for i, row in enumerate(kept)}
            matched = [None if row is None else node_of_row.get(row) for row in by_row]
            self.graph = cloudanalyzer_core.PoseGraph.from_poses(rows[kept], sigma_t, sigma_r_deg, kept)
            self.node_ids = kept
            self.timestamps = None if stamps is None else stamps[kept]
        self.extrinsic = kitti_extrinsic(root)
        self.scan_points = 0
        for k, (path, node) in enumerate(zip(scans, matched)):
            if node is None:
                continue
            cloud = cloudanalyzer_core.read(str(path))
            positions = cloud["positions"]
            if self.extrinsic is not None:
                positions = positions @ self.extrinsic[:3, :3].T + self.extrinsic[:3, 3]
            self.scan_points += self.graph.set_scan(node, np.ascontiguousarray(positions), cloud.get("intensity"), voxel)
            if k % 200 == 0:
                say(f"{root.name}: scan {k + 1} of {len(scans)}")
        self.scans = sum(m is not None for m in matched)
        self.unmatched = sum(m is None for m in matched)

    def summary(self) -> dict:
        return {
            "poses_file": self.poses_path.name,
            "nodes": len(self.node_ids),
            "scans": self.scans,
            "unmatched_scans": self.unmatched,
            "scan_points": self.scan_points,
            "scans_moved_by_calib_tr": self.extrinsic is not None,
        }

    def index(self, node_id: int) -> int:
        """The node index of a vertex id or frame number."""
        try:
            return self.node_ids.index(node_id)
        except ValueError:
            raise ValueError(f"{self.poses_path.name} has no node {node_id}") from None

    def ups(self, gravity: str) -> tuple[list[int], np.ndarray]:
        """The nodes with an up direction in ``gravity`` and those directions, in the pose frame."""
        ups = read_ups(Path(gravity))
        nodes = [i for i, node_id in enumerate(self.node_ids) if node_id in ups]
        if not nodes:
            raise ValueError(f"no up direction in {gravity} matches a node of {self.poses_path.name}")
        vectors = np.array([ups[self.node_ids[i]] for i in nodes])
        if self.extrinsic is not None:
            vectors = vectors @ self.extrinsic[:3, :3].T
        return nodes, vectors


def _loops(graph, options: dict | None) -> dict:
    found = graph.find_loops(**(options or {}))
    ids = list(graph.node_ids)
    out = {
        "candidates": found["candidates"],
        "added": len(found["added"]),
        "implausible": found["implausible"],
        "same_place_retries": sum(1 for *_, retried in found["added"] if retried),
        "pairs": [[ids[a], ids[b], round(f, 3)] for a, b, f, _ in found["added"]],
    }
    if found["added"]:
        out["optimized"] = graph.optimize()
    return out


def odometry(
    scans: str,
    out_dir: str,
    *,
    max_range: float = 80.0,
    voxel_size: float | None = None,
    max_frames: int | None = None,
    deskew: bool = False,
    pointcloud_topic: str | None = None,
    imu_topic: str | None = None,
    imu_to_lidar: list[float] | None = None,
) -> dict:
    """KISS-ICP odometry for raw scans (a folder, or a ROS bag whose scans are written out as
    KITTI ``.bin`` with intensity, with each scan's up direction from its IMU): writes
    ``trajectory.tum`` and ``map.ply`` to ``out_dir`` and returns where the scans, trajectory
    and gravity file are, ready for :func:`fix_session`."""
    from ca.core.bag_ingest import imu_ups, is_bag_path, materialize_pointcloud_bag
    from ca.core.slam_run import SlamRunRequest, discover_frame_paths, run_slam, write_map_ply, write_tum_trajectory

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    stamps: tuple[float, ...] | None = None
    gravity_file: Path | None = None
    if is_bag_path(scans):
        frames, stamps = materialize_pointcloud_bag(
            scans, out / "scans", topic=pointcloud_topic, max_frames=max_frames, kitti_bin=True
        )
        scan_folder = out / "scans"
        ups = imu_ups(scans, stamps, topic=imu_topic)
        if ups:
            rotation = np.eye(3) if imu_to_lidar is None else np.array(imu_to_lidar, dtype=float).reshape(3, 3)
            gravity_file = out / "gravity" / "gravity.txt"
            gravity_file.parent.mkdir(exist_ok=True)
            gravity_file.write_text(
                "".join(f"{k} {' '.join(f'{v:.9f}' for v in rotation @ up)}" + chr(10) for k, up in sorted(ups.items()))
            )
    else:
        frames = discover_frame_paths(Path(scans))
        scan_folder = Path(scans) if Path(scans).is_dir() else frames[0].parent
    request = SlamRunRequest(
        frame_paths=tuple(frames),
        timestamps_s=stamps,
        max_range_m=max_range,
        voxel_size_m=voxel_size,
        deskew=deskew,
        max_frames=max_frames,
    )
    result = run_slam(request)
    write_tum_trajectory(out / "trajectory.tum", result.poses, result.timestamps_s)
    write_map_ply(out / "map.ply", result.map_points)
    steps = result.poses[1:, :3, 3] - result.poses[:-1, :3, 3]
    return {
        "driver": result.driver,
        "frames": int(result.frames_processed),
        "path_length_m": round(float(np.sqrt((steps**2).sum(1)).sum()), 1),
        "runtime_s": round(time.perf_counter() - clock, 1),
        "scans": str(scan_folder),
        "trajectory": str(out / "trajectory.tum"),
        "gravity": None if gravity_file is None else str(gravity_file),
        "map": str(out / "map.ply"),
    }


def needs_odometry(folder: str, poses: str | None) -> bool:
    """Whether ``folder`` holds raw scans only: a ROS bag, or scans without a poses file."""
    from ca.core.bag_ingest import is_bag_path

    if poses:
        return False
    if is_bag_path(folder):
        return True
    return Path(folder).is_dir() and poses_file(Path(folder)) is None


def fix_session(
    folder: str,
    out_dir: str | None = None,
    *,
    poses: str | None = None,
    keyframe_spacing: float = 0.0,
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
    say = progress or (lambda message: None)
    clock = time.perf_counter()
    session = Session(
        folder,
        poses=poses,
        keyframe_spacing=keyframe_spacing,
        voxel=voxel,
        sigma_t=sigma_t,
        sigma_r_deg=sigma_r_deg,
        progress=progress,
    )
    graph, node_ids, poses_path, timestamps = session.graph, session.node_ids, session.poses_path, session.timestamps
    report: dict = {"session": str(session.root), **session.summary(), "timings_s": {}}
    report["timings_s"]["open"] = round(time.perf_counter() - clock, 2)

    if loops:
        clock = time.perf_counter()
        say("finding loops")
        report["loops"] = _loops(graph, loop_options)
        report["timings_s"]["loops"] = round(time.perf_counter() - clock, 2)

    if gravity:
        clock = time.perf_counter()
        say("tying to gravity")
        nodes, vectors = session.ups(gravity)
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
        rows = [i for i in node_ids if 0 <= i < len(truth_poses)]
        truth_poses = truth_poses[rows]
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


def _near(positions: np.ndarray, others: np.ndarray, reach: float) -> list[int]:
    """Indices of ``positions`` within ``reach`` of any of ``others``."""
    out = []
    for start in range(0, len(positions), 512):
        block = positions[start : start + 512]
        d2 = ((block[:, None, :] - others[None, :, :]) ** 2).sum(-1)
        out.extend((start + np.flatnonzero(d2.min(1) < reach * reach)).tolist())
    return out


def compare_sessions(
    first: str,
    second: str,
    out_dir: str | None = None,
    *,
    here: int,
    there: int,
    voxel: float = 0.4,
    loops: bool = True,
    loop_options: dict | None = None,
    gravity_first: str | None = None,
    gravity_second: str | None = None,
    gravity_sigma_deg: float = 0.1,
    map_voxel: float = 0.3,
    reach: float = 50.0,
    core_spacing: float = 0.5,
    min_change: float = 0.3,
    listed: int = 50,
    progress=None,
) -> dict:
    """Two drives through the same place, joined and compared: what changed between them.

    ``second`` joins ``first`` where its node ``there`` stands near ``first``'s
    node ``here`` (vertex ids or frame numbers); loops tie them together, IMU
    gravity levels both, and M3C2 compares their maps where they pass within
    ``reach`` metres of each other. Changes of at least ``min_change`` metres
    are grouped into objects, largest first.
    """
    cloudanalyzer_core = _core()
    say = progress or (lambda message: None)
    timings: dict = {}
    clock = time.perf_counter()
    a = Session(first, voxel=voxel, progress=progress)
    b = Session(second, voxel=voxel, progress=progress)
    report: dict = {"first": {"session": first, **a.summary()}, "second": {"session": second, **b.summary()}}
    timings["open"] = round(time.perf_counter() - clock, 2)
    graph = a.graph

    clock = time.perf_counter()
    if loops:
        say("finding loops in the first drive")
        report["first"]["loops"] = _loops(graph, loop_options)
    say("joining the second drive")
    joined = graph.join(b.graph, a.index(here), b.index(there))
    offset = joined["offset"]
    report["join"] = {"here": here, "there": there, "overlap": round(joined["fitness"], 3), "rms": joined["rms"]}
    if loops:
        say("finding loops across the drives")
        report["loops"] = _loops(graph, loop_options)
    else:
        report["join"]["optimized"] = graph.optimize()
    timings["loops"] = round(time.perf_counter() - clock, 2)

    if gravity_first or gravity_second:
        nodes: list[int] = []
        vectors = []
        for session, path, shift in ((a, gravity_first, 0), (b, gravity_second, offset)):
            if path:
                n, v = session.ups(path)
                nodes += [i + shift for i in n]
                vectors.append(v)
        report["gravity"] = {"tied": graph.set_gravity(nodes, np.vstack(vectors), gravity_sigma_deg)}
        report["gravity"]["optimized"] = graph.optimize()

    clock = time.perf_counter()
    say("building the maps where the drives meet")
    positions = graph.poses()[:, :3, 3]
    near_first = _near(positions[:offset], positions[offset:], reach)
    near_second = [offset + i for i in _near(positions[offset:], positions[:offset], reach)]
    if not near_first or not near_second:
        raise ValueError(f"the two drives never come within {reach} m of each other: nothing to compare")
    before = graph.map(voxel=map_voxel, nodes=near_first)
    after = graph.map(voxel=map_voxel, nodes=near_second)
    say("comparing them (M3C2)")
    core = after["positions"][cloudanalyzer_core.voxel_subsample(after["positions"], core_spacing)]
    distance, lod95, significant, _ = cloudanalyzer_core.m3c2(core, before["positions"], after["positions"])
    objects, labels = cloudanalyzer_core.changed_objects(core, distance, significant, min_change)
    measured = np.isfinite(distance)
    report["m3c2"] = {
        "keyframes": [len(near_first), len(near_second)],
        "core_points": int(len(core)),
        "measured": int(measured.sum()),
        "significant": int(significant.sum()),
        "significant_share": float(significant.sum() / max(len(core), 1)),
        "mean_change": float(np.nanmean(distance)) if measured.any() else None,
    }
    report["changes"] = {
        "objects": int(len(objects)),
        "min_change": min_change,
        "largest": [
            {
                "rank": k + 1,
                "points": int(o[0]),
                "centroid": [round(float(v), 3) for v in o[1:4]],
                "size": [round(float(o[7 + i] - o[4 + i]), 3) for i in range(3)],
                "mean_change": round(float(o[10]), 3),
            }
            for k, o in enumerate(objects[:listed])
        ],
    }
    timings["compare"] = round(time.perf_counter() - clock, 2)

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {
            "g2o": out / "joined.g2o",
            "first_map": out / "first_map.ply",
            "second_map": out / "second_map.ply",
            "m3c2": out / "m3c2.ply",
        }
        files["g2o"].write_text(graph.to_g2o())
        for key, m in (("first_map", before), ("second_map", after)):
            write_ply(files[key], m["positions"], {k: m[k] for k in ("intensity",) if k in m})
        write_ply(
            files["m3c2"],
            core,
            {
                "m3c2_distance": distance.astype(np.float32),
                "lod95": lod95.astype(np.float32),
                "significant": significant.astype(np.float32),
                "change_object": np.where(labels < 0, np.nan, labels + 1).astype(np.float32),
            },
        )
        report["outputs"] = {k: str(v) for k, v in files.items()}
    report["timings_s"] = timings
    return report
