"""Tests for KITTI label format parser."""

import json
from pathlib import Path

import pytest

from ca.kitti import convert_kitti_labels, parse_kitti_label_file
from ca.object_eval import load_box_sequence


def _write_label(path: Path, lines: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class TestParseKittiLabelFile:
    def test_single_line(self, tmp_path: Path):
        label_path = _write_label(
            tmp_path / "000001.txt",
            ["Car 0.00 0 -1.58 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.27 1.67 46.70 -1.56"],
        )
        boxes = parse_kitti_label_file(label_path, camera_to_lidar=True)
        assert len(boxes) == 1
        box = boxes[0]
        assert box["label"] == "Car"
        assert len(box["center"]) == 3
        assert len(box["size"]) == 3
        assert "yaw" in box
        # KITTI dimensions: h=1.57, w=1.73, l=4.15
        assert box["size"] == [pytest.approx(4.15, abs=0.01), pytest.approx(1.73, abs=0.01), pytest.approx(1.57, abs=0.01)]

    def test_camera_to_lidar_transform(self, tmp_path: Path):
        # Simple case: camera (x=0, y=1, z=10), h=2
        label_path = _write_label(
            tmp_path / "000001.txt",
            ["Car 0.0 0 0 0 0 0 0 2.0 1.5 4.0 0.0 1.0 10.0 0.0"],
        )
        boxes = parse_kitti_label_file(label_path, camera_to_lidar=True)
        box = boxes[0]
        # lidar_x = cam_z = 10, lidar_y = -cam_x = 0, lidar_z = -(cam_y - h/2) = -(1.0 - 1.0) = 0
        assert box["center"][0] == pytest.approx(10.0)
        assert box["center"][1] == pytest.approx(0.0)
        assert box["center"][2] == pytest.approx(0.0)

    def test_no_camera_to_lidar(self, tmp_path: Path):
        label_path = _write_label(
            tmp_path / "000001.txt",
            ["Car 0.0 0 0 0 0 0 0 2.0 1.5 4.0 5.0 3.0 10.0 1.5"],
        )
        boxes = parse_kitti_label_file(label_path, camera_to_lidar=False)
        box = boxes[0]
        assert box["center"] == [pytest.approx(5.0), pytest.approx(3.0), pytest.approx(10.0)]
        assert box["yaw"] == pytest.approx(1.5)

    def test_dontcare_skipped(self, tmp_path: Path):
        label_path = _write_label(
            tmp_path / "000001.txt",
            [
                "Car 0.0 0 0 0 0 0 0 1.5 1.6 3.9 -1.0 1.8 30.0 -0.02",
                "DontCare -1 -1 -10 503 170 552 194 -1 -1 -1 -1000 -1000 -1000 -10",
            ],
        )
        boxes = parse_kitti_label_file(label_path)
        assert len(boxes) == 1
        assert boxes[0]["label"] == "Car"

    def test_multi_class(self, tmp_path: Path):
        label_path = _write_label(
            tmp_path / "000001.txt",
            [
                "Car 0.0 0 0 0 0 0 0 1.5 1.6 3.9 -1.0 1.8 30.0 -0.02",
                "Pedestrian 0.0 0 0 0 0 0 0 1.7 0.6 0.8 2.0 1.8 15.0 0.3",
                "Cyclist 0.0 0 0 0 0 0 0 1.8 0.6 1.8 0.5 1.7 20.0 -1.2",
            ],
        )
        boxes = parse_kitti_label_file(label_path)
        assert len(boxes) == 3
        labels = [b["label"] for b in boxes]
        assert labels == ["Car", "Pedestrian", "Cyclist"]

    def test_optional_score(self, tmp_path: Path):
        label_path = _write_label(
            tmp_path / "000001.txt",
            ["Car 0.0 0 0 0 0 0 0 1.5 1.6 3.9 -1.0 1.8 30.0 -0.02 0.95"],
        )
        boxes = parse_kitti_label_file(label_path)
        assert boxes[0]["score"] == pytest.approx(0.95)

    def test_no_score_field(self, tmp_path: Path):
        label_path = _write_label(
            tmp_path / "000001.txt",
            ["Car 0.0 0 0 0 0 0 0 1.5 1.6 3.9 -1.0 1.8 30.0 -0.02"],
        )
        boxes = parse_kitti_label_file(label_path)
        assert "score" not in boxes[0]


class TestConvertKittiLabels:
    def test_converts_directory(self, tmp_path: Path):
        label_dir = tmp_path / "labels"
        _write_label(
            label_dir / "000001.txt",
            ["Car 0.0 0 0 0 0 0 0 1.5 1.6 3.9 -1.0 1.8 30.0 -0.02"],
        )
        _write_label(
            label_dir / "000002.txt",
            [
                "Car 0.0 0 0 0 0 0 0 1.5 1.6 3.9 2.0 1.8 25.0 0.5",
                "Pedestrian 0.0 0 0 0 0 0 0 1.7 0.6 0.8 0.0 1.8 10.0 0.0",
            ],
        )
        output_path = str(tmp_path / "output.json")
        result = convert_kitti_labels(str(label_dir), output_path)
        assert result["frames"] == 2
        assert result["total_boxes"] == 3

        data = json.loads(Path(output_path).read_text())
        assert len(data["frames"]) == 2
        assert data["frames"][0]["frame_id"] == "000001"
        assert len(data["frames"][0]["boxes"]) == 1
        assert data["frames"][1]["frame_id"] == "000002"
        assert len(data["frames"][1]["boxes"]) == 2

    def test_roundtrip_with_load_box_sequence(self, tmp_path: Path):
        """KITTI labels converted to JSON should be loadable by load_box_sequence."""
        label_dir = tmp_path / "labels"
        _write_label(
            label_dir / "000001.txt",
            ["Car 0.0 0 0 0 0 0 0 1.5 1.6 3.9 -1.0 1.8 30.0 -1.57"],
        )
        output_path = str(tmp_path / "output.json")
        convert_kitti_labels(str(label_dir), output_path)

        seq = load_box_sequence(output_path)
        assert len(seq.frames) == 1
        box = seq.frames[0].boxes[0]
        assert box.label == "Car"
        assert box.yaw != 0.0  # rotation_y=-1.57 → yaw=1.57 after camera_to_lidar

    def test_rejects_missing_dir(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            convert_kitti_labels(str(tmp_path / "nonexistent"), str(tmp_path / "out.json"))

    def test_rejects_empty_dir(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No .txt files"):
            convert_kitti_labels(str(empty_dir), str(tmp_path / "out.json"))
