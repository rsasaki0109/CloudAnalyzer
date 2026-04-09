"""Tests for 3D tracking evaluation."""

import json
from pathlib import Path

import pytest

from ca.tracking import evaluate_tracking


def _write_json(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def _reference_tracking_sequence() -> dict:
    return {
        "frames": [
            {
                "frame_id": "0001",
                "boxes": [
                    {
                        "label": "car",
                        "track_id": "gt-car",
                        "center": [0.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                    }
                ],
            },
            {
                "frame_id": "0002",
                "boxes": [
                    {
                        "label": "car",
                        "track_id": "gt-car",
                        "center": [1.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                    }
                ],
            },
            {
                "frame_id": "0003",
                "boxes": [
                    {
                        "label": "car",
                        "track_id": "gt-car",
                        "center": [2.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                    }
                ],
            },
        ]
    }


def _estimated_tracking_good() -> dict:
    return {
        "frames": [
            {
                "frame_id": "0001",
                "boxes": [
                    {
                        "label": "car",
                        "track_id": "pred-a",
                        "center": [0.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                    }
                ],
            },
            {
                "frame_id": "0002",
                "boxes": [
                    {
                        "label": "car",
                        "track_id": "pred-a",
                        "center": [1.0, 0.1, 0.0],
                        "size": [2.0, 2.0, 2.0],
                    }
                ],
            },
            {
                "frame_id": "0003",
                "boxes": [
                    {
                        "label": "car",
                        "track_id": "pred-a",
                        "center": [2.0, 0.0, 0.1],
                        "size": [2.0, 2.0, 2.0],
                    }
                ],
            },
        ]
    }


def _estimated_tracking_bad() -> dict:
    return {
        "frames": [
            {
                "frame_id": "0001",
                "boxes": [
                    {
                        "label": "car",
                        "track_id": "pred-a",
                        "center": [0.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                    }
                ],
            },
            {
                "frame_id": "0002",
                "boxes": [],
            },
            {
                "frame_id": "0003",
                "boxes": [
                    {
                        "label": "car",
                        "track_id": "pred-b",
                        "center": [2.0, 0.0, 0.0],
                        "size": [2.0, 2.0, 2.0],
                    }
                ],
            },
        ]
    }


class TestTrackingEvaluate:
    def test_good_tracking_sequence(self, tmp_path: Path):
        estimated = _write_json(tmp_path / "estimated.json", _estimated_tracking_good())
        reference = _write_json(tmp_path / "reference.json", _reference_tracking_sequence())

        result = evaluate_tracking(estimated, reference)

        assert result["detection"]["recall"] == pytest.approx(1.0)
        assert result["tracking"]["mota"] == pytest.approx(1.0)
        assert result["tracking"]["id_switches"] == 0
        assert result["quality_gate"] is None

    def test_tracking_gate_fails(self, tmp_path: Path):
        estimated = _write_json(tmp_path / "estimated.json", _estimated_tracking_bad())
        reference = _write_json(tmp_path / "reference.json", _reference_tracking_sequence())

        result = evaluate_tracking(
            estimated,
            reference,
            min_mota=0.8,
            min_recall=0.9,
            max_id_switches=0,
        )

        assert result["quality_gate"] is not None
        assert result["quality_gate"]["passed"] is False
        assert result["tracking"]["id_switches"] == 1
        assert any("MOTA" in reason for reason in result["quality_gate"]["reasons"])
        assert any("Recall" in reason for reason in result["quality_gate"]["reasons"])
        assert any("ID switches" in reason for reason in result["quality_gate"]["reasons"])

    def test_requires_track_ids(self, tmp_path: Path):
        estimated = _write_json(
            tmp_path / "estimated.json",
            {
                "frames": [
                    {
                        "frame_id": "0001",
                        "boxes": [
                            {
                                "label": "car",
                                "center": [0.0, 0.0, 0.0],
                                "size": [2.0, 2.0, 2.0],
                            }
                        ],
                    }
                ]
            },
        )
        reference = _write_json(tmp_path / "reference.json", _reference_tracking_sequence())

        with pytest.raises(ValueError, match="track_id"):
            evaluate_tracking(estimated, reference)
