"""Tests for detection/tracking report generation."""

from ca.report import (
    make_detection_html,
    make_detection_markdown,
    make_tracking_html,
    make_tracking_markdown,
    save_detection_report,
    save_tracking_report,
)


SAMPLE_DETECTION_RESULT = {
    "estimated_path": "estimated.json",
    "reference_path": "reference.json",
    "matching_policy": {
        "geometry": "axis_aligned_3d_boxes",
        "class_aware": True,
        "yaw_ignored": True,
        "iou_thresholds": [0.25, 0.5],
        "primary_iou_threshold": 0.5,
    },
    "counts": {
        "estimated_frames": 2,
        "reference_frames": 2,
        "shared_frames": 2,
        "estimated_boxes": 3,
        "reference_boxes": 3,
    },
    "mAP": 0.945,
    "primary_threshold_result": {
        "iou_threshold": 0.5,
        "precision": 1.0,
        "recall": 0.6667,
        "f1": 0.8,
        "mean_iou": 0.74,
        "mean_center_distance": 0.18,
    },
    "threshold_results": [
        {
            "iou_threshold": 0.25,
            "map": 0.98,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_iou": 0.81,
            "mean_center_distance": 0.12,
        },
        {
            "iou_threshold": 0.5,
            "map": 0.91,
            "precision": 1.0,
            "recall": 0.6667,
            "f1": 0.8,
            "mean_iou": 0.74,
            "mean_center_distance": 0.18,
        },
    ],
    "per_class": {
        "car": {
            "reference_boxes": 2,
            "estimated_boxes": 2,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_ap": 0.98,
        },
        "pedestrian": {
            "reference_boxes": 1,
            "estimated_boxes": 1,
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_ap": 0.91,
        },
    },
    "quality_gate": {
        "passed": False,
        "min_map": 0.95,
        "min_precision": None,
        "min_recall": 0.8,
        "min_f1": None,
        "reasons": ["mAP 0.9450 < min_map 0.9500", "Recall 0.6667 < min_recall 0.8000"],
    },
}

SAMPLE_TRACKING_RESULT = {
    "estimated_path": "estimated.json",
    "reference_path": "reference.json",
    "matching_policy": {
        "geometry": "axis_aligned_3d_boxes",
        "class_aware": True,
        "yaw_ignored": True,
        "iou_threshold": 0.5,
    },
    "counts": {
        "estimated_frames": 3,
        "reference_frames": 3,
        "shared_frames": 3,
        "estimated_detections": 3,
        "reference_detections": 3,
        "estimated_tracks": 2,
        "reference_tracks": 1,
        "matched_detections": 2,
    },
    "detection": {
        "precision": 1.0,
        "recall": 0.6667,
        "f1": 0.8,
    },
    "tracking": {
        "mota": 0.0,
        "id_switches": 1,
        "track_fragmentations": 1,
        "mostly_tracked_ratio": 0.0,
        "mean_iou": 0.83,
        "mean_center_distance": 0.11,
    },
    "per_class": {
        "car": {
            "reference_detections": 3,
            "estimated_detections": 3,
            "matched_detections": 2,
            "precision": 1.0,
            "recall": 0.6667,
            "f1": 0.8,
            "id_switches": 1,
            "false_positives": 0,
            "false_negatives": 1,
        }
    },
    "matched_samples": [
        {
            "frame_id": "0001",
            "label": "car",
            "reference_track_id": "gt-car",
            "estimated_track_id": "pred-a",
            "iou": 0.92,
            "center_distance": 0.04,
        }
    ],
    "quality_gate": {
        "passed": False,
        "min_mota": 0.8,
        "min_recall": 0.9,
        "max_id_switches": 0,
        "reasons": ["MOTA 0.0000 < min_mota 0.8000"],
    },
}


class TestDetectionTrackingReports:
    def test_detection_markdown(self, tmp_path):
        path = tmp_path / "detection.md"
        make_detection_markdown(SAMPLE_DETECTION_RESULT, str(path))
        content = path.read_text()
        assert "# CloudAnalyzer Detection Report" in content
        assert "## Threshold Sweep" in content
        assert "## Per-class Summary" in content
        assert "Status: FAIL" in content

    def test_detection_html(self, tmp_path):
        path = tmp_path / "detection.html"
        make_detection_html(SAMPLE_DETECTION_RESULT, str(path))
        content = path.read_text()
        assert "<title>CloudAnalyzer Detection Report</title>" in content
        assert "Threshold Sweep" in content
        assert "Per-class Summary" in content
        assert "Quality Gate" in content

    def test_detection_report_dispatch(self, tmp_path):
        path = tmp_path / "detection.html"
        save_detection_report(SAMPLE_DETECTION_RESULT, str(path))
        assert path.exists()

    def test_tracking_markdown(self, tmp_path):
        path = tmp_path / "tracking.md"
        make_tracking_markdown(SAMPLE_TRACKING_RESULT, str(path))
        content = path.read_text()
        assert "# CloudAnalyzer Tracking Report" in content
        assert "## Metrics" in content
        assert "ID switches: 1" in content
        assert "## Sample Matches" in content

    def test_tracking_html(self, tmp_path):
        path = tmp_path / "tracking.html"
        make_tracking_html(SAMPLE_TRACKING_RESULT, str(path))
        content = path.read_text()
        assert "<title>CloudAnalyzer Tracking Report</title>" in content
        assert "Per-class Summary" in content
        assert "Sample Matches" in content
        assert "Quality Gate" in content

    def test_tracking_report_dispatch(self, tmp_path):
        path = tmp_path / "tracking.html"
        save_tracking_report(SAMPLE_TRACKING_RESULT, str(path))
        assert path.exists()
