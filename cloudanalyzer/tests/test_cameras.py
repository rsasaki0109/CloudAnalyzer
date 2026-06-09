"""Tests for ``ca.core.cameras``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ca.core.cameras import (
    c2w_to_viewmat,
    load_cameras,
    load_colmap_cameras,
    load_nerfstudio_transforms,
)


def test_load_nerfstudio_transforms(tmp_path: Path) -> None:
    path = tmp_path / "transforms.json"
    c2w = np.eye(4).tolist()
    c2w[2][3] = 5.0
    path.write_text(
        json.dumps(
            {
                "camera_angle_x": 0.8,
                "w": 640,
                "h": 480,
                "frames": [{"file_path": "images/a.png", "transform_matrix": c2w}],
            }
        ),
        encoding="utf-8",
    )
    cameras = load_nerfstudio_transforms(path)
    assert cameras.source == "nerfstudio"
    assert len(cameras.frames) == 1
    frame = cameras.frames[0]
    assert frame.name == "a.png"
    assert frame.width == 640
    assert frame.height == 480
    assert frame.fx > 0
    assert frame.fy > 0


def test_load_colmap_cameras(tmp_path: Path) -> None:
    cameras_txt = tmp_path / "cameras.txt"
    images_txt = tmp_path / "images.txt"
    cameras_txt.write_text(
        "# comment\n1 PINHOLE 640 480 500 500 320 240\n",
        encoding="utf-8",
    )
    images_txt.write_text(
        "1 1 0 0 0 0 0 0 1 image000.png\n",
        encoding="utf-8",
    )
    cameras = load_colmap_cameras(cameras_txt, images_txt)
    assert cameras.source == "colmap"
    assert cameras.frames[0].name == "image000.png"
    assert cameras.frames[0].fx == 500


def test_load_cameras_directory_dispatch(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    c2w = np.eye(4).tolist()
    (bundle / "transforms.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.8,
                "w": 128,
                "h": 128,
                "frames": [{"file_path": "000.png", "transform_matrix": c2w}],
            }
        ),
        encoding="utf-8",
    )
    cameras = load_cameras(bundle)
    assert len(cameras.frames) == 1


def test_c2w_to_viewmat_is_finite() -> None:
    c2w = np.eye(4)
    c2w[2, 3] = 4.0
    view = c2w_to_viewmat(c2w)
    assert view.shape == (4, 4)
    assert np.all(np.isfinite(view))


def test_load_nerfstudio_transforms_requires_focal(tmp_path: Path) -> None:
    path = tmp_path / "transforms.json"
    path.write_text(
        json.dumps({"w": 640, "h": 480, "frames": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="focal length"):
        load_nerfstudio_transforms(path)
