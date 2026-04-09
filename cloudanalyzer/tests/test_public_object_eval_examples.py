"""Checks for the checked-in public detection/tracking JSON examples."""

from pathlib import Path

from ca.detection import evaluate_detection
from ca.tracking import evaluate_tracking


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "demo_assets" / "public" / "rellis3d-frame-000001" / "object_eval"


def test_public_detection_examples_are_valid():
    reference = EXAMPLE_ROOT / "detection_reference.json"
    estimated_good = EXAMPLE_ROOT / "detection_estimated_good.json"
    estimated_regressed = EXAMPLE_ROOT / "detection_estimated_regressed.json"

    good = evaluate_detection(
        str(estimated_good),
        str(reference),
        iou_thresholds=[0.25, 0.5],
        min_map=0.9,
        min_recall=0.8,
    )
    regressed = evaluate_detection(
        str(estimated_regressed),
        str(reference),
        iou_thresholds=[0.25, 0.5],
        min_map=0.9,
        min_recall=0.8,
    )

    assert good["quality_gate"]["passed"] is True
    assert regressed["quality_gate"]["passed"] is False


def test_public_tracking_examples_are_valid():
    reference = EXAMPLE_ROOT / "tracking_reference.json"
    estimated_good = EXAMPLE_ROOT / "tracking_estimated_good.json"
    estimated_regressed = EXAMPLE_ROOT / "tracking_estimated_regressed.json"

    good = evaluate_tracking(
        str(estimated_good),
        str(reference),
        min_mota=0.8,
        min_recall=0.8,
        max_id_switches=1,
    )
    regressed = evaluate_tracking(
        str(estimated_regressed),
        str(reference),
        min_mota=0.8,
        min_recall=0.8,
        max_id_switches=1,
    )

    assert good["quality_gate"]["passed"] is True
    assert regressed["quality_gate"]["passed"] is False
