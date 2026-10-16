"""Independent fixed-match geometry checks, not trajectory accuracy benchmarks."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")


def scene():
    return np.array([[3. * (i % 4) + .2, 3. * (i // 4) + .2, float(i % 3) + .2]
                     for i in range(12)])


def system(snapshot):
    q = snapshot["source_world"][snapshot["matched_source_indices"]]
    arm = q - snapshot["center_world"]
    residual = q - snapshot["target_world"]
    jacobian = np.zeros((len(q), 3, 6))
    jacobian[:, :, :3] = np.eye(3)
    x, y, z = arm.T
    jacobian[:, 0, 4], jacobian[:, 0, 5] = z, -y
    jacobian[:, 1, 3], jacobian[:, 1, 5] = -z, x
    jacobian[:, 2, 3], jacobian[:, 2, 4] = y, -x
    d2 = np.sum(residual * residual, axis=1)
    weights = snapshot["kernel"] ** 2 / (snapshot["kernel"] + d2) ** 2
    h = np.einsum("nri,n,nrj->ij", jacobian, weights, jacobian)
    b = -np.einsum("nri,n,nr->i", jacobian, weights, residual)
    radius = np.sqrt(np.sum(weights * np.sum(arm * arm, axis=1)) / weights.sum())
    return jacobian, weights, h, b, radius


def test_default_flag_and_empty_or_insufficient_snapshots():
    plain = core.LidarOdometry()
    assert plain.capture_linearization is False
    assert plain.last_linearization() is None
    plain.register(scene())
    assert plain.last_linearization() is None
    measured = core.LidarOdometry(min_range=0., capture_linearization=True)
    assert measured.capture_linearization is True
    with pytest.raises(AttributeError):
        measured.capture_linearization = False
    assert measured.last_linearization() is None
    for points in [scene(), scene()[:1], np.empty((0, 3))]:
        measured.register(points)
        snapshot = measured.last_linearization()
        assert not snapshot["alignment_eligible"]
        assert snapshot["finite"]
        np.testing.assert_array_equal(snapshot["source_sensor"], points)
        np.testing.assert_array_equal(snapshot["inserted_world"], points)
        assert snapshot["target_world"].shape == (0, 3)
        assert snapshot["matched_source_indices"].dtype == np.int64
        assert snapshot["matched_source_indices"].shape == (0,)
        assert snapshot["weight_sum"] == 0.
        assert snapshot["weighted_rms_radius_m"] is None
        np.testing.assert_array_equal(snapshot["normal_matrix"], np.zeros((6, 6)))
        np.testing.assert_array_equal(snapshot["right_hand_side"], np.zeros(6))


@pytest.mark.parametrize("fine", [None, .125])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_capture_preserves_pose_bits_reports_predictions_and_sampling(fine, diagnostics):
    options = dict(min_range=0., diagnostics=diagnostics, registration_voxel=fine)
    plain = core.LidarOdometry(**options)
    measured = core.LidarOdometry(**options, capture_linearization=True)
    for i in range(5):
        points = scene() - [i * .1, i * -.02, 0.]
        prediction = np.eye(4)
        prediction[:3, :3] = Rotation.from_rotvec([.001, -.002, .003]).as_matrix()
        prediction[:3, 3] = [.12, -.03, .01]
        if i == 0:
            prediction = None
        a = plain.register(points, relative_prediction=prediction)
        b = measured.register(points, relative_prediction=prediction)
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        assert plain.last_report() == measured.last_report()
        assert plain.last_linearization() is None
        np.testing.assert_array_equal(measured.last_linearization()["pose"].view(np.uint64), b.view(np.uint64))
    np.testing.assert_array_equal(plain.poses().view(np.uint64), measured.poses().view(np.uint64))


def test_final_pose_matches_and_matrix_reconstruct_independently():
    points = scene()
    odometry = core.LidarOdometry(min_range=0., diagnostics=True, capture_linearization=True)
    odometry.register(points)
    scan = points.copy()
    scan[:, 0] += .10 + .003 * np.arange(len(scan))
    scan[:, 1] -= .02 * (np.arange(len(scan)) % 3)
    scan = np.vstack([scan, [35., 0., 0.]])
    prediction = np.eye(4)
    prediction[:3, :3] = Rotation.from_rotvec([.01, -.02, .03]).as_matrix()
    prediction[:3, 3] = [.12, -.08, .02]
    pose = odometry.register(scan, relative_prediction=prediction)
    s = odometry.last_linearization()
    np.testing.assert_array_equal(s["center_world"], prediction[:3, 3])
    np.testing.assert_allclose(s["source_world"], scan @ pose[:3, :3].T + pose[:3, 3], atol=1e-14)
    indices, targets = [], []
    for i, q in enumerate(s["source_world"]):
        allowed = np.all(np.abs(np.floor(points) - np.floor(q)) <= 1, axis=1)
        if np.any(allowed):
            candidates = points[allowed]
            d2 = np.sum((candidates - q) ** 2, axis=1)
            j = np.argmin(d2)
            if d2[j] <= s["threshold_sq"]:
                indices.append(i)
                targets.append(candidates[j])
    np.testing.assert_array_equal(s["matched_source_indices"], indices)
    np.testing.assert_array_equal(s["target_world"], targets)
    assert len(indices) == 12 < len(scan)
    assert s["squared_distances"].max() > 1e-6
    jacobian, weights, h, b, radius = system(s)
    np.testing.assert_allclose(s["weights"], weights, atol=1e-14)
    np.testing.assert_allclose(s["normal_matrix"], h, atol=1e-12)
    np.testing.assert_allclose(s["right_hand_side"], b, atol=1e-13)
    np.testing.assert_allclose(s["weight_sum"], weights.sum(), atol=1e-14)
    np.testing.assert_allclose(s["weighted_rms_radius_m"], radius, atol=1e-14)
    # Central differences of world-axis rotation around the fixed prediction center.
    q = s["source_world"][indices]
    for k in range(6):
        dx = np.zeros(6)
        dx[k] = 1e-6
        def moved(delta):
            return ((q - s["center_world"]) @ Rotation.from_rotvec(delta[3:]).as_matrix().T
                    + s["center_world"] + delta[:3])
        numerical = (moved(dx) - moved(-dx)) / 2e-6
        np.testing.assert_allclose(jacobian[:, :, k], numerical, atol=2e-9, rtol=2e-9)
    stats = odometry.last_report()["correspondences"]
    assert stats["matched_points"] == len(indices)
    np.testing.assert_allclose(stats["rmse_m"], np.sqrt(s["squared_distances"].mean()))


def test_snapshot_arrays_and_metadata_are_owned_across_calls_and_registrations():
    odometry = core.LidarOdometry(min_range=0., capture_linearization=True)
    odometry.register(scene())
    odometry.register(scene() - [.1, .02, .03])
    saved = odometry.last_linearization()
    mutated = odometry.last_linearization()
    for value in mutated.values():
        if isinstance(value, np.ndarray):
            value[...] = 123
    mutated["finite"] = False
    again = odometry.last_linearization()
    for key, value in saved.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(again[key], value)
        else:
            assert again[key] == value
    odometry.register(np.empty((0, 3)))
    assert len(saved["target_world"]) == 12
    assert odometry.last_linearization()["target_world"].shape == (0, 3)


def test_zero_support_and_singular_line_do_not_claim_pose_quality():
    for collinear in [False, True]:
        points = np.array([[3. * i, 0., 0.] for i in range(12)]) if collinear else scene()
        odometry = core.LidarOdometry(min_range=0., max_range=200., capture_linearization=True)
        odometry.register(points)
        odometry.register(points if collinear else points + [50., 0., 0.])
        s = odometry.last_linearization()
        assert s["alignment_eligible"] and s["finite"]
        assert odometry.last_report()["termination"] == "singular_system"
        if collinear:
            assert s["weight_sum"] == 12.
            assert np.linalg.eigvalsh(s["normal_matrix"])[0] == 0.
        else:
            assert s["weight_sum"] == 0.
            assert s["weighted_rms_radius_m"] is None


def test_invalid_prediction_does_not_replace_snapshot():
    odometry = core.LidarOdometry(min_range=0., capture_linearization=True)
    odometry.register(scene())
    saved = odometry.last_linearization()
    with pytest.raises(ValueError):
        odometry.register(scene(), relative_prediction=np.eye(4) * 2.)
    np.testing.assert_array_equal(odometry.last_linearization()["inserted_world"], saved["inserted_world"])
