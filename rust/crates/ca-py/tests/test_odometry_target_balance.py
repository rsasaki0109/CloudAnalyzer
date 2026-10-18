"""Opt-in target balancing, independent weight arithmetic and motion controls."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip('cloudanalyzer_core')


def scene():
    return np.array([[3.*(i % 4)+.2, 3.*(i//4)+.2, float(i % 3)+.2] for i in range(12)])


def reconstruct(s):
    q = s['source_world'][s['matched_source_indices']]
    target = s['target_world']
    _, inverse, counts = np.unique(target, axis=0, return_inverse=True, return_counts=True)
    count = counts[inverse.reshape(-1)]
    residual, arm = q-target, q-s['center_world']
    robust = s['kernel']**2/(s['kernel']+np.sum(residual**2, axis=1))**2
    weights = robust/count if s['target_balance'] else robust
    j = np.zeros((len(q), 3, 6))
    for i, (x, y, z) in enumerate(arm):
        j[i] = np.column_stack([np.eye(3), [[0., z, -y], [-z, 0., x], [y, -x, 0.]]])
    return count, robust, weights, np.einsum('nri,n,nrj->ij', j, weights, j), -np.einsum('nri,n,nr->i', j, weights, residual)


def test_default_is_exactly_explicit_false_and_policy_is_read_only():
    plain = core.LidarOdometry(min_range=0., capture_linearization=True)
    explicit = core.LidarOdometry(min_range=0., capture_linearization=True, target_balance=False)
    assert plain.target_balance is False
    with pytest.raises(AttributeError): plain.target_balance = True
    for i in range(5):
        points = scene()-[i*.1, i*-.02, 0.]
        a, b = plain.register(points), explicit.register(points)
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        assert plain.last_report() == explicit.last_report()


@pytest.mark.parametrize('balance', [False, True])
def test_native_snapshot_matches_independent_all_pair_system(balance):
    points = scene()
    odometry = core.LidarOdometry(min_range=0., registration_voxel=.03125, capture_linearization=True, target_balance=balance)
    odometry.register(points)
    dense = np.vstack([points+[.125, .03125, 0.], points+[.1875, .03125, 0.]])
    pose = odometry.register(dense)
    s = odometry.last_linearization()
    assert s['target_balance'] is balance
    assert len(s['matched_source_indices']) == len(dense)
    count, robust, weights, h, b = reconstruct(s)
    assert max(count) >= 2
    np.testing.assert_array_equal(s['target_multiplicity'], count)
    np.testing.assert_allclose(s['robust_weights'], robust, atol=1e-14)
    np.testing.assert_allclose(s['weights'], weights, atol=1e-14)
    np.testing.assert_allclose(s['normal_matrix'], h, atol=2e-11)
    np.testing.assert_allclose(s['right_hand_side'], b, atol=2e-12)
    np.testing.assert_allclose(s['source_world'], dense@pose[:3, :3].T+pose[:3, 3], atol=1e-13)


def test_balanced_motion_recovers_analytic_rigid_scan_and_capture_is_passive():
    options = dict(min_range=0., registration_voxel=.03125, target_balance=True, diagnostics=True)
    plain, captured = core.LidarOdometry(**options), core.LidarOdometry(**options, capture_linearization=True)
    world = scene()
    for i in range(5):
        truth = np.eye(4)
        truth[:3, :3] = Rotation.from_rotvec([.002*i, -.003*i, .004*i]).as_matrix()
        truth[:3, 3] = [.08*i, -.02*i, .01*i]
        scan = (world-truth[:3, 3])@truth[:3, :3]
        a, b = plain.register(scan), captured.register(scan)
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        np.testing.assert_allclose(a, truth, atol=3e-8)
        assert plain.last_report() == captured.last_report()


def test_failure_states_and_owned_arrays_are_retained():
    line = np.array([[3.*i, 0., 0.] for i in range(12)])
    odometry = core.LidarOdometry(min_range=0., max_range=200., capture_linearization=True, target_balance=True)
    odometry.register(line)
    odometry.register(line)
    saved = odometry.last_linearization()
    assert odometry.last_report()['termination'] == 'singular_system'
    assert saved['target_balance'] and saved['finite']
    mutated = odometry.last_linearization()
    mutated['weights'][:] = 123.
    mutated['target_multiplicity'][:] = 123
    np.testing.assert_array_equal(odometry.last_linearization()['weights'], saved['weights'])
    np.testing.assert_array_equal(odometry.last_linearization()['target_multiplicity'], saved['target_multiplicity'])
    odometry.register(np.empty((0, 3)))
    s = odometry.last_linearization()
    assert odometry.last_report()['termination'] == 'insufficient_points'
    assert s['target_balance'] and s['weight_sum'] == 0 and s['weighted_rms_radius_m'] is None
