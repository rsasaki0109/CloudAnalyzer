"""Load camera poses for ``ca rendered-evaluate``.

Initial formats:

- nerfstudio / Instant-NGP ``transforms.json``
- COLMAP text exports (``cameras.txt`` + ``images.txt``)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# nerfstudio / Instant-NGP use an OpenGL-style camera (+Y up, -Z forward).
# gsplat expects a COLMAP-style world-to-camera matrix, which we obtain via
# this fixed axis flip applied after inverting camera-to-world.
_OPENGL_TO_COLMAP = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One pinhole view to render or score."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray
    """4×4 camera-to-world matrix (nerfstudio / OpenGL convention)."""


@dataclass(frozen=True, slots=True)
class CameraSet:
    frames: tuple[CameraFrame, ...]
    source: str
    source_path: str


def _stem_from_file_path(file_path: str) -> str:
    return Path(file_path).name


def _focal_from_meta(meta: dict[str, Any], *, axis: str) -> float:
    width = int(meta["w"])
    height = int(meta["h"])

    def _fov_to_focal(radians: float, resolution: int) -> float:
        return 0.5 * float(resolution) / math.tan(0.5 * float(radians))

    if axis == "x":
        if "fl_x" in meta:
            return float(meta["fl_x"])
        if "x_fov" in meta:
            return _fov_to_focal(math.radians(float(meta["x_fov"])), width)
        if "camera_angle_x" in meta:
            return _fov_to_focal(float(meta["camera_angle_x"]), width)
        raise ValueError(
            "transforms.json is missing focal length for X "
            "(need fl_x, x_fov, or camera_angle_x)"
        )

    if "fl_y" in meta:
        return float(meta["fl_y"])
    if "y_fov" in meta:
        return _fov_to_focal(math.radians(float(meta["y_fov"])), height)
    if "camera_angle_y" in meta:
        return _fov_to_focal(float(meta["camera_angle_y"]), height)
    return _focal_from_meta(meta, axis="x")


def load_nerfstudio_transforms(path: str | Path) -> CameraSet:
    """Parse a nerfstudio / Instant-NGP ``transforms.json`` file."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "frames" not in payload or not isinstance(payload["frames"], list):
        raise ValueError(f"{path}: transforms.json missing non-empty 'frames' list")

    global_w = int(payload["w"])
    global_h = int(payload["h"])
    global_fx = _focal_from_meta(payload, axis="x")
    global_fy = _focal_from_meta(payload, axis="y")
    global_cx = float(payload.get("cx", global_w / 2.0))
    global_cy = float(payload.get("cy", global_h / 2.0))

    frames: list[CameraFrame] = []
    for index, frame in enumerate(payload["frames"]):
        if not isinstance(frame, dict):
            raise ValueError(f"{path}: frames[{index}] must be an object")
        matrix = frame.get("transform_matrix")
        if matrix is None:
            raise ValueError(f"{path}: frames[{index}] missing transform_matrix")
        c2w = np.asarray(matrix, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(
                f"{path}: frames[{index}].transform_matrix must be 4×4; got {c2w.shape}"
            )

        file_path = str(frame.get("file_path", f"frame_{index:04d}.png"))
        width = int(frame.get("w", global_w))
        height = int(frame.get("h", global_h))
        fx = float(frame.get("fl_x", global_fx))
        fy = float(frame.get("fl_y", global_fy))
        cx = float(frame.get("cx", global_cx))
        cy = float(frame.get("cy", global_cy))

        frames.append(
            CameraFrame(
                name=_stem_from_file_path(file_path),
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                c2w=c2w,
            )
        )

    if not frames:
        raise ValueError(f"{path}: transforms.json contains zero frames")
    return CameraSet(frames=tuple(frames), source="nerfstudio", source_path=str(path))


def _qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) → 3×3 rotation matrix."""

    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def _colmap_w2c_to_c2w(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """Convert COLMAP world-to-camera (q, t) into a 4×4 camera-to-world matrix."""

    rot_w2c = _qvec_to_rotmat(qw, qx, qy, qz)
    trans = np.array([tx, ty, tz], dtype=np.float64)
    rot_c2w = rot_w2c.T
    origin = -rot_c2w @ trans
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot_c2w
    c2w[:3, 3] = origin
    return c2w


def _parse_colmap_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        camera_id = int(tokens[0])
        model = tokens[1]
        width = int(tokens[2])
        height = int(tokens[3])
        params = [float(v) for v in tokens[4:]]
        cameras[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
            "params": params,
        }
    if not cameras:
        raise ValueError(f"{path}: no cameras parsed from COLMAP cameras.txt")
    return cameras


def _intrinsics_from_colmap(model: str, params: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    model_upper = model.upper()
    if model_upper in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
        return fx, fy, cx, cy
    if model_upper in {"PINHOLE", "OPENCV", "FULL_OPENCV", "RADIAL"}:
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
        return fx, fy, cx, cy
    raise ValueError(
        f"Unsupported COLMAP camera model {model!r}. "
        "Supported: SIMPLE_PINHOLE, SIMPLE_RADIAL, PINHOLE, OPENCV, FULL_OPENCV, RADIAL"
    )


def load_colmap_cameras(
    cameras_txt: str | Path,
    images_txt: str | Path,
) -> CameraSet:
    """Parse COLMAP text exports into :class:`CameraSet`."""

    cameras_path = Path(cameras_txt)
    images_path = Path(images_txt)
    camera_models = _parse_colmap_cameras(cameras_path)

    frames: list[CameraFrame] = []
    for raw in images_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 10:
            continue
        qw, qx, qy, qz = (float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]))
        tx, ty, tz = (float(tokens[5]), float(tokens[6]), float(tokens[7]))
        camera_id = int(tokens[8])
        name = tokens[9]
        if camera_id not in camera_models:
            raise ValueError(f"{images_path}: image {name!r} references unknown camera {camera_id}")
        cam = camera_models[camera_id]
        fx, fy, cx, cy = _intrinsics_from_colmap(
            cam["model"], cam["params"], int(cam["width"]), int(cam["height"])
        )
        frames.append(
            CameraFrame(
                name=name,
                width=int(cam["width"]),
                height=int(cam["height"]),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                c2w=_colmap_w2c_to_c2w(qw, qx, qy, qz, tx, ty, tz),
            )
        )

    if not frames:
        raise ValueError(f"{images_path}: no images parsed from COLMAP images.txt")
    return CameraSet(
        frames=tuple(frames),
        source="colmap",
        source_path=str(images_path),
    )


def load_cameras(path: str | Path) -> CameraSet:
    """Dispatch on file type / layout."""

    path = Path(path)
    if path.is_file() and path.name == "transforms.json":
        return load_nerfstudio_transforms(path)
    if path.is_file() and path.name == "images.txt":
        cameras_txt = path.with_name("cameras.txt")
        if not cameras_txt.is_file():
            raise ValueError(
                f"{path}: COLMAP images.txt requires cameras.txt in the same directory"
            )
        return load_colmap_cameras(cameras_txt, path)
    if path.is_dir():
        transforms = path / "transforms.json"
        if transforms.is_file():
            return load_nerfstudio_transforms(transforms)
        images_txt = path / "images.txt"
        cameras_txt = path / "cameras.txt"
        if images_txt.is_file() and cameras_txt.is_file():
            return load_colmap_cameras(cameras_txt, images_txt)
    raise ValueError(
        f"{path}: unsupported camera bundle. Expected transforms.json or "
        "COLMAP cameras.txt + images.txt"
    )


def c2w_to_viewmat(c2w: np.ndarray) -> np.ndarray:
    """Convert an OpenGL-style camera-to-world matrix to gsplat viewmat (w2c)."""

    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError(f"c2w must be 4×4; got {c2w.shape}")
    return (_OPENGL_TO_COLMAP @ np.linalg.inv(c2w)).astype(np.float32)


__all__ = [
    "CameraFrame",
    "CameraSet",
    "c2w_to_viewmat",
    "load_cameras",
    "load_colmap_cameras",
    "load_nerfstudio_transforms",
]
