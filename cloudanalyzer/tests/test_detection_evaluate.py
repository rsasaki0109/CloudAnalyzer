"""Tests for 3D detection evaluation."""

import json
from pathlib import Path

import pytest

from ca.detection import evaluate_detection


def _write_json(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def _reference_sequence() -> dict:
    return {
        "frames": [
            {
                "frame_id": "0001",
                "boxes": [
                    {
                        "label": "car",
                        "center": [0.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                    },
                    {
                        "label": "pedestrian",
                        "center": [5.0, 0.0, 0.0],
                        "size": [1.0, 1.0, 2.0],
                    },
                ],
            },
            {
                "frame_id": "0002",
                "boxes": [
                    {
                        "label": "car",
                        "center": [10.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                    }
                ],
            },
        ]
    }


def _estimated_sequence_good() -> dict:
    return {
        "frames": [
            {
                "frame_id": "0001",
                "boxes": [
                    {
                        "label": "car",
                        "center": [0.1, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                        "score": 0.95,
                    },
                    {
                        "label": "pedestrian",
                        "center": [5.0, 0.1, 0.0],
                        "size": [1.0, 1.0, 2.0],
                        "score": 0.90,
                    },
                ],
            },
            {
                "frame_id": "0002",
                "boxes": [
                    {
                        "label": "car",
                        "center": [10.0, 0.0, 0.1],
                        "size": [2.0, 2.0, 2.0],
                        "score": 0.92,
                    }
                ],
            },
        ]
    }


def _estimated_sequence_bad() -> dict:
    return {
        "frames": [
            {
                "frame_id": "0001",
                "boxes": [
                    {
                        "label": "car",
                        "center": [2.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                        "score": 0.99,
                    },
                    {
                        "label": "pedestrian",
                        "center": [5.0, 0.0, 0.0],
                        "size": [0.5, 0.5, 1.0],
                        "score": 0.10,
                    },
                ],
            },
            {
                "frame_id": "0003",
                "boxes": [
                    {
                        "label": "car",
                        "center": [15.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                        "score": 0.80,
                    }
                ],
            },
        ]
    }


class TestDetectionEvaluate:
    def test_good_detection_sequence(self, tmp_path: Path):
        estimated = _write_json(tmp_path / "estimated.json", _estimated_sequence_good())
        reference = _write_json(tmp_path / "reference.json", _reference_sequence())

        result = evaluate_detection(estimated, reference, iou_thresholds=[0.25, 0.5])

        assert result["counts"]["shared_frames"] == 2
        assert result["mAP"] == pytest.approx(1.0)
        assert result["primary_threshold_result"]["precision"] == pytest.approx(1.0)
        assert result["primary_threshold_result"]["recall"] == pytest.approx(1.0)
        assert result["primary_threshold_result"]["f1"] == pytest.approx(1.0)
        assert result["per_class"]["car"]["mean_ap"] == pytest.approx(1.0)
        assert result["quality_gate"] is None

    def test_detection_gate_fails(self, tmp_path: Path):
        estimated = _write_json(tmp_path / "estimated.json", _estimated_sequence_bad())
        reference = _write_json(tmp_path / "reference.json", _reference_sequence())

        result = evaluate_detection(
            estimated,
            reference,
            iou_thresholds=[0.5],
            min_map=0.9,
            min_recall=0.8,
        )

        assert result["quality_gate"] is not None
        assert result["quality_gate"]["passed"] is False
        assert any("mAP" in reason for reason in result["quality_gate"]["reasons"])
        assert any("Recall" in reason for reason in result["quality_gate"]["reasons"])

    def test_rejects_primary_threshold_outside_set(self, tmp_path: Path):
        estimated = _write_json(tmp_path / "estimated.json", _estimated_sequence_good())
        reference = _write_json(tmp_path / "reference.json", _reference_sequence())

        with pytest.raises(ValueError, match="primary_iou_threshold"):
            evaluate_detection(
                estimated,
                reference,
                iou_thresholds=[0.25, 0.5],
                primary_iou_threshold=0.75,
            )

    def test_rejects_box_without_label(self, tmp_path: Path):
        reference = _write_json(
            tmp_path / "reference.json",
            {
                "frames": [
                    {
                        "frame_id": "0001",
                        "boxes": [
                            {"center": [0.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0]},
                        ],
                    },
                ]
            },
        )
        estimated = _write_json(
            tmp_path / "estimated.json",
            {
                "frames": [
                    {
                        "frame_id": "0001",
                        "boxes": [
                            {"label": "car", "center": [0.0, 0.0, 0.0], "size": [2.0, 2.0, 2.0], "score": 0.9},
                        ],
                    },
                ]
            },
        )

        with pytest.raises(ValueError, match="label.*class.*category"):
            evaluate_detection(estimated, reference, iou_thresholds=[0.5])
