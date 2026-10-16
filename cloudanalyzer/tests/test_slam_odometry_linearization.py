"""Independent audit algebra, search ties and map lifecycle numerical controls."""
import importlib.util
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "scripts/audit_flatwall_odometry_linearization.py"
SPEC = importlib.util.spec_from_file_location("linearization_audit", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_search_checks_all_cells_center_first_and_retains_first_equal_distance():
    replay = audit.MapReplay(capacity=1)
    # Center wins an exact tie; cell capacity retains its first point.
    replay.insert(np.array([[.75, 0., 0.], [.875, 0., 0.], [-.25, 0., 0.]]), np.zeros(3))
    indices, targets, distances = replay.matches(np.array([[.25, 0., 0.], [5., 0., 0.]]), .25)
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_array_equal(targets, [[.75, 0., 0.]])
    np.testing.assert_array_equal(distances, [.25])


def test_pruning_uses_first_point_in_cell_and_closed_range_boundary():
    replay = audit.MapReplay(voxel=2., max_range=1.)
    replay.insert(np.array([[1.1, 0., 0.], [.1, 0., 0.], [-1., 0., 0.]]), np.zeros(3))
    assert set(replay.cells) == {(-1, 0, 0)}
    np.testing.assert_array_equal(replay.cells[(-1, 0, 0)], [[-1., 0., 0.]])


def test_sampling_keeps_source_order_and_uses_floor_for_negative_cells():
    points = np.array([[-.1, 0., 0.], [.1, 0., 0.], [-.2, 0., 0.], [1., 0., 0.]])
    np.testing.assert_array_equal(audit.selected(points, 1.), points[[0, 1, 3]])


def test_point_to_point_planar_correspondences_can_have_full_rank():
    points = np.array([[x, y, 0.] for x in [-1., 1.] for y in [-1., 1.]])
    s = {"source_world": points, "target_world": points.copy(),
         "matched_source_indices": np.arange(4), "center_world": np.zeros(3), "kernel": 1.}
    weights, h, b, support, radius = audit.reconstruct(s)
    np.testing.assert_array_equal(weights, np.ones(4))
    np.testing.assert_array_equal(h, np.diag([4., 4., 4., 4., 4., 8.]))
    np.testing.assert_array_equal(b, np.zeros(6))
    geometry = audit.spectrum(h, support, radius)
    np.testing.assert_allclose(geometry["eigenvalues"], [.5, .5, 1., 1., 1., 1.])
    assert geometry["weak_directions_below_1e-3"] == 0
    # This positive conditional matrix does not demonstrate plane observability.
    assert audit.spectrum(h, 0., radius) is None
    assert audit.spectrum(h, support, 0.) is None
