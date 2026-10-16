"""Independent batch numerical oracles; these are not SLAM accuracy results."""

import numpy as np
import pytest
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from test_imu_factor import _initial, _oracle  # noqa: E402
from test_imu_preintegration import EPOCH, _write  # noqa: E402


@pytest.fixture
def chain(tmp_path):
    times = EPOCH + np.arange(61, dtype=np.int64) * 10_000_000
    k = np.arange(len(times))
    gyro = np.column_stack([.5 * np.sin(k * .2), .3 * np.cos(k * .3), .4 + .01 * k])
    accel = np.column_stack([.03 * k, np.cos(k * .3), 9.8 + .01 * k])
    reader = _write(tmp_path, times, gyro, accel)
    nodes = times[::20]
    bias = np.array([.03, -.02, .01, .1, .2, -.05])
    arguments = dict(gyro_noise_density=.02, accel_noise_density=.15, bias=bias)
    factors = [reader.imu_factor(int(a), int(b), **arguments) for a, b in zip(nodes[:-1], nodes[1:])]
    gravity = np.array([.1, -.2, -9.81])
    p, v = _initial()
    poses, velocities = [p], [v]
    for factor in factors:
        prediction = factor.predict(poses[-1], velocities[-1], gravity=gravity, bias=bias)
        poses.append(prediction["pose"])
        velocities.append(prediction["velocity"])
    return dict(times=nodes, poses=np.array(poses), velocities=np.array(velocities),
                biases=np.tile(bias, (4, 1)), factors=factors, gravity=gravity, reader=reader,
                arguments=arguments)


def _unpack(chain, x):
    delta = np.asarray(x).reshape(-1, 15)
    poses = chain["poses"].copy()
    poses[:, :3, :3] = poses[:, :3, :3] @ Rotation.from_rotvec(delta[:, :3]).as_matrix()
    poses[:, :3, 3] += delta[:, 6:9]
    return poses, chain["velocities"] + delta[:, 3:6], chain["biases"] + delta[:, 9:]


def _window(chain, x=None):
    if x is None:
        x = .03 * np.sin(np.arange(len(chain["times"]) * 15) + 1)
    poses, velocities, biases = _unpack(chain, x)
    return core.ImuWindow(chain["times"], poses, velocities, biases, gravity=chain["gravity"])


def _edges(chain, window):
    for i, factor in enumerate(chain["factors"]):
        window.add_imu_factor(i, factor, gyro_bias_random_walk=.005, accel_bias_random_walk=.05)


SIGMAS = np.r_[[.05] * 3, [.1] * 3, [.05] * 3, [.02] * 3, [.1] * 3]


def _anchor(chain, window, index=0):
    window.add_state_prior(index, chain["poses"][index], chain["velocities"][index],
                           chain["biases"][index], sigmas=SIGMAS)


def _observations(chain):
    observations = []
    for i in [1, 3]:
        pose = chain["poses"][i].copy()
        pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec([.015, -.02, .01]).as_matrix()
        pose[:3, 3] += [.02, -.03, .01 * i]
        observations.append((i, pose, .025, .03))
    return observations


def _residual_function(chain, anchor, observations):
    """SciPy rotation logs and NumPy Cholesky; does not use native linearization."""
    measurements = [factor.measurement() for factor in chain["factors"]]
    lower = [np.linalg.cholesky(m["covariance"]) for m in measurements]

    def residual(x):
        poses, velocities, biases = _unpack(chain, x)
        blocks = []
        for i, (m, chol) in enumerate(zip(measurements, lower)):
            raw = _oracle(m, poses[i], velocities[i], poses[i + 1], velocities[i + 1],
                          chain["gravity"], biases[i])
            blocks.append(solve_triangular(chol, raw, lower=True))
            sigma = np.r_[[.005] * 3, [.05] * 3] * np.sqrt(m["delta_time_s"])
            blocks.append((biases[i + 1] - biases[i]) / sigma)
        blocks.append(np.r_[
            Rotation.from_matrix(chain["poses"][anchor, :3, :3].T @ poses[anchor, :3, :3]).as_rotvec(),
            velocities[anchor] - chain["velocities"][anchor],
            poses[anchor, :3, 3] - chain["poses"][anchor, :3, 3],
            biases[anchor] - chain["biases"][anchor],
        ] / SIGMAS)
        for i, pose, sr, sp in observations:
            blocks.append(np.r_[
                Rotation.from_matrix(pose[:3, :3].T @ poses[i, :3, :3]).as_rotvec() / sr,
                (poses[i, :3, 3] - pose[:3, 3]) / sp,
            ])
        return np.concatenate(blocks)

    return residual


@pytest.mark.parametrize("anchor", [0, 2])
def test_inconsistent_pose_fusion_matches_independent_scipy_optimum(chain, anchor):
    window = _window(chain)
    _edges(chain, window)
    _anchor(chain, window, anchor)
    observations = _observations(chain)
    for i, pose, sr, sp in observations:
        window.add_pose_prior(i, pose, rotation_sigma=sr, position_sigma=sp)
    residual = _residual_function(chain, anchor, observations)
    x0 = .03 * np.sin(np.arange(60) + 1)
    expected = least_squares(residual, x0, jac="3-point", ftol=1e-12, xtol=1e-12,
                             gtol=1e-10, max_nfev=200)
    actual = window.optimize(max_iterations=80, gradient_tolerance=1e-9,
                             step_tolerance=1e-11, cost_tolerance=1e-14)
    assert expected.success and actual["converged"], (expected.message, actual)
    assert actual["initial_cost"] == pytest.approx(.5 * residual(x0) @ residual(x0), rel=1e-11)
    assert actual["final_cost"] == pytest.approx(expected.cost, rel=1e-9, abs=1e-11)
    for key, value in zip(["poses", "velocities", "biases"], _unpack(chain, expected.x)):
        np.testing.assert_allclose(actual[key], value, rtol=1e-7, atol=2e-7)
    assert np.all(np.diff(actual["cost_history"]) < 0)
    assert len(actual["cost_history"]) == actual["accepted_steps"] + 1
    expected_bias = np.max(np.abs(actual["biases"][:-1] - chain["biases"][:-1]), axis=0)
    np.testing.assert_allclose(actual["max_abs_bias_correction"], expected_bias, atol=1e-15)


def test_one_lm_step_matches_dense_numerical_normal_equations(chain):
    # At zero chart increments, the independent chart has precisely the native
    # right-local rotation/world velocity/world position/bias tangent convention.
    initial = .03 * np.sin(np.arange(60) + 1)
    p, v, b = _unpack(chain, initial)
    local_chain = dict(chain, poses=p, velocities=v, biases=b)
    # The mean prior must stay at the original reference, so use a residual
    # wrapper to compose a local right retraction with that original objective.
    reference_residual = _residual_function(chain, 0, [])

    def residual(d):
        pp, vv, bb = _unpack(local_chain, d)
        coordinates = np.zeros((4, 15))
        coordinates[:, :3] = Rotation.from_matrix(chain["poses"][:, :3, :3].transpose(0, 2, 1) @ pp[:, :3, :3]).as_rotvec()
        coordinates[:, 3:6] = vv - chain["velocities"]
        coordinates[:, 6:9] = pp[:, :3, 3] - chain["poses"][:, :3, 3]
        coordinates[:, 9:] = bb - chain["biases"]
        return reference_residual(coordinates.ravel())

    zero = np.zeros(60)
    h = 1e-6
    j = np.column_stack([(residual(zero + np.eye(60)[k] * h) - residual(zero - np.eye(60)[k] * h)) / (2 * h)
                         for k in range(60)])
    normal = j.T @ j
    damping = .002
    step = np.linalg.solve(normal + damping * np.diag(np.diag(normal)), -j.T @ residual(zero))
    expected = _unpack(local_chain, step)
    window = _window(chain, initial)
    _edges(chain, window)
    _anchor(chain, window)
    actual = window.optimize(max_iterations=1, initial_damping=damping,
                             cost_tolerance=0., gradient_tolerance=0., step_tolerance=0.)
    assert actual["accepted_steps"] == 1 and actual["rejected_steps"] == 0
    assert not actual["converged"] and actual["termination"] == "max_iterations"
    for key, value in zip(["poses", "velocities", "biases"], expected):
        np.testing.assert_allclose(actual[key], value, atol=3e-8, rtol=1e-7)


def test_consistent_chain_recovers_bias_and_states_and_owns_arrays(chain):
    window = _window(chain)
    _edges(chain, window)
    _anchor(chain, window)
    actual = window.optimize(max_iterations=60)
    assert actual["converged"] and actual["final_cost"] < 1e-14
    for key in ["poses", "velocities", "biases"]:
        np.testing.assert_allclose(actual[key], chain[key], atol=1e-7)
    original = window.states()
    for key in original:
        actual[key][...] = 42
    again = window.states()
    for key in original:
        np.testing.assert_array_equal(original[key], again[key])


def test_exact_interval_duplicates_missing_edges_and_anchor_are_rejected(chain):
    window = _window(chain)
    with pytest.raises(ValueError, match="every adjacent edge"):
        window.optimize()
    shifted = chain["reader"].imu_factor(int(chain["times"][0]) + 1, int(chain["times"][1]) + 1,
                                         **chain["arguments"])
    m = shifted.measurement()
    assert m["start_ns"] == int(chain["times"][0]) + 1
    assert m["end_ns"] == int(chain["times"][1]) + 1
    with pytest.raises(ValueError, match="exact timestamps"):
        window.add_imu_factor(0, shifted, gyro_bias_random_walk=.005, accel_bias_random_walk=.05)
    _edges(chain, window)
    with pytest.raises(ValueError, match="already has"):
        window.add_imu_factor(0, chain["factors"][0], gyro_bias_random_walk=.005, accel_bias_random_walk=.05)
    window.add_pose_prior(0, chain["poses"][0], rotation_sigma=.1, position_sigma=.1)
    with pytest.raises(ValueError, match="full 15D state prior"):
        window.optimize()


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf, 1e-200, 1e200])
def test_invalid_weights_do_not_mutate_window(chain, value):
    window = _window(chain)
    with pytest.raises(ValueError):
        window.add_imu_factor(0, chain["factors"][0], gyro_bias_random_walk=value, accel_bias_random_walk=.05)
    with pytest.raises(ValueError):
        window.add_state_prior(0, chain["poses"][0], chain["velocities"][0], chain["biases"][0], sigmas=[value] * 15)
    with pytest.raises(ValueError):
        window.add_pose_prior(0, chain["poses"][0], rotation_sigma=.1, position_sigma=value)
    _edges(chain, window)
    with pytest.raises(ValueError, match="full 15D state prior"):
        window.optimize()


def test_shapes_timestamps_finite_values_and_noncontiguous_inputs(chain):
    arguments = [chain["times"], chain["poses"], chain["velocities"], chain["biases"]]
    for index, bad in [(0, [0]), (0, [0, 0, 1, 2]), (0, [-(2**63), 0, 1, 2]),
                       (1, np.ones((4, 3, 4))), (2, np.ones((3, 3))), (3, np.ones((4, 9)))]:
        supplied = arguments.copy()
        supplied[index] = bad
        with pytest.raises(ValueError):
            core.ImuWindow(*supplied, gravity=chain["gravity"])
    for index in [1, 2, 3]:
        supplied = [np.array(a).copy() for a in arguments]
        supplied[index].flat[0] = np.nan
        with pytest.raises(ValueError):
            core.ImuWindow(*supplied, gravity=chain["gravity"])
    poses = np.tile(np.eye(4), (33, 1, 1))
    with pytest.raises(ValueError):
        core.ImuWindow(np.arange(33), poses, np.zeros((33, 3)), np.zeros((33, 6)), gravity=chain["gravity"])
    for bad in [np.diag([1., 1., -1., 1.]), np.eye(4) * 2]:
        supplied = chain["poses"].copy()
        supplied[1] = bad
        with pytest.raises(ValueError):
            core.ImuWindow(chain["times"], supplied, chain["velocities"], chain["biases"], gravity=chain["gravity"])
    strided = [np.repeat(np.array(a), 2, axis=0)[::2] for a in arguments]
    window = core.ImuWindow(*strided, gravity=chain["gravity"])
    for key, expected in zip(["timestamps_ns", "poses", "velocities", "biases"], arguments):
        np.testing.assert_array_equal(window.states()[key], expected)


@pytest.mark.parametrize("parameters", [dict(max_iterations=0), dict(max_iterations=101),
                                      dict(initial_damping=0.), dict(initial_damping=np.nan),
                                      dict(step_tolerance=-1.), dict(cost_tolerance=np.inf)])
def test_optimizer_errors_are_atomic(chain, parameters):
    window = _window(chain)
    _edges(chain, window)
    _anchor(chain, window)
    before = window.states()
    with pytest.raises(ValueError):
        window.optimize(**parameters)
    for key, expected in before.items():
        np.testing.assert_array_equal(window.states()[key], expected)
