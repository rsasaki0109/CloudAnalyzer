"""Keep correspondence repeats separate from pose recurrence and raw cost signs."""
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2]/"scripts/analyze_flatwall_gicp_transitions.py"
spec = importlib.util.spec_from_file_location("gicp_transition_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_pose_distance_resolves_zero_small_and_half_turn_angles():
    identity = np.eye(4)
    for angle in [0., 1e-12, .1, np.pi]:
        pose = np.eye(4)
        c, s = np.cos(angle), np.sin(angle)
        pose[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
        pose[:3, 3] = [3., 4., 0.]
        result = analysis.pose_distance(pose, identity)
        np.testing.assert_allclose(result["rotation"], angle, atol=1e-16, rtol=1e-14)
        assert result["translation"] == 5.


def test_periodic_ids_do_not_imply_periodic_poses():
    history = []
    for index in range(30):
        pose = np.eye(4)
        pose[0, 3] = index*.1
        history.append(dict(pose_before=pose, source_indices=np.array([0], dtype=np.uint64),
            target_indices=np.array([index % 3], dtype=np.uint64), combined_shapes=np.eye(3)[None]))
    result = analysis.recurrence(history)
    assert result["last_ten_unique_correspondence_sets"] == 3
    lag = result["lags"]["3"]
    assert lag["equal_ordered_correspondence_sets"] == 10 and lag["exact_pose_bit_repeats"] == 0
    np.testing.assert_allclose(lag["translation_range"], [.3, .3])
    assert lag["same_ids_combined_shape_max_difference_range"] == [0., 0.]


def test_pose_and_metric_periods_are_reported_without_a_cutoff():
    history = []
    for index in range(30):
        pose = np.eye(4)
        pose[0, 3] = (index % 2)*.1
        history.append(dict(pose_before=pose, source_indices=np.array([0], dtype=np.uint64),
            target_indices=np.array([index % 2], dtype=np.uint64), combined_shapes=(1.+index)*np.eye(3)[None]))
    lag = analysis.recurrence(history)["lags"]["2"]
    assert lag["equal_ordered_correspondence_sets"] == lag["exact_pose_bit_repeats"] == 10
    assert lag["translation_range"] == lag["rotation_range"] == [0., 0.]
    assert lag["same_ids_combined_shape_max_difference_range"] == [2., 2.]


def test_empty_history_has_no_inferred_recurrence():
    result = analysis.recurrence([])
    assert result["last_ten_states"] == result["last_ten_unique_correspondence_sets"] == 0
    assert all(row["comparisons"] == 0 and row["translation_range"] is None for row in result["lags"].values())
