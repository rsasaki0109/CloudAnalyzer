"""Independent QR elimination and nonlinear reduced-objective checks."""
import numpy as np
import pytest
from scipy.linalg import qr, solve_triangular
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")
from test_imu_factor import _oracle  # noqa: E402
from test_imu_window import SIGMAS, _anchor, _edges, _observations, _unpack, _window, chain as _chain  # noqa: E402


@pytest.fixture
def chain(tmp_path):
    return _chain.__wrapped__(tmp_path)


def local_states(base, delta):
    return _unpack(base, delta)


def residual_function(base, factors, gravity, *, anchor=None, observations=(), marginal=None):
    measurements = [f.measurement() for f in factors]
    lower = [np.linalg.cholesky(m["covariance"]) for m in measurements]

    def residual(delta):
        poses, velocities, biases = local_states(base, delta)
        blocks = []
        for i, (m, chol) in enumerate(zip(measurements, lower)):
            raw = _oracle(m, poses[i], velocities[i], poses[i + 1], velocities[i + 1], gravity, biases[i])
            blocks.append(solve_triangular(chol, raw, lower=True))
            blocks.append((biases[i + 1] - biases[i]) / (np.r_[[.005] * 3, [.05] * 3] * np.sqrt(m["delta_time_s"])))
        if anchor is not None:
            pose, velocity, bias = anchor
            blocks.append(np.r_[Rotation.from_matrix(pose[:3, :3].T @ poses[0, :3, :3]).as_rotvec(),
                                velocities[0] - velocity, poses[0, :3, 3] - pose[:3, 3], biases[0] - bias] / SIGMAS)
        if marginal is not None:
            reference = marginal["reference_pose"]
            local = np.r_[Rotation.from_matrix(reference[:3, :3].T @ poses[0, :3, :3]).as_rotvec(),
                          velocities[0] - marginal["reference_velocity"], poses[0, :3, 3] - reference[:3, 3],
                          biases[0] - marginal["reference_bias"]]
            blocks.append(marginal["sqrt_information"] @ local + marginal["residual_offset"])
        for i, pose, sr, sp in observations:
            blocks.append(np.r_[Rotation.from_matrix(pose[:3, :3].T @ poses[i, :3, :3]).as_rotvec() / sr,
                                (poses[i, :3, 3] - pose[:3, 3]) / sp])
        return np.concatenate(blocks)
    return residual


def jacobian(function, size):
    h = 1e-6
    directions = np.eye(size) * h
    return np.column_stack([(function(d) - function(-d)) / (2 * h) for d in directions])


def test_marginalized_nonlinear_objective_matches_qr_elimination_and_scipy(chain):
    window = _window(chain)
    _edges(chain, window)
    _anchor(chain, window)
    observations = [(0, chain["poses"][0], .04, .07), *_observations(chain)]
    for i, pose, sr, sp in observations:
        window.add_pose_prior(i, pose, rotation_sigma=sr, position_sigma=sp)
    current = window.states()
    original = residual_function(current, chain["factors"], chain["gravity"],
                                 anchor=(chain["poses"][0], chain["velocities"][0], chain["biases"][0]), observations=observations)
    r = original(np.zeros(60))
    j = jacobian(original, 60)
    # QR projects the eliminated columns out without normal equations or a Schur inverse.
    q, _ = qr(j[:, :15], mode="full")
    projected_j, projected_r = (q.T @ j[:, 15:])[15:], (q.T @ r)[15:]
    report = window.marginalize_oldest()
    assert report["dropped_timestamp_ns"] == chain["times"][0]
    assert report["retained_timestamp_ns"] == chain["times"][1]
    assert report["removed_priors"] == 2
    assert report["replaced_marginal_prior"] is False
    marginal = window.marginal_prior()
    retained = window.states()
    for key, old in current.items():
        np.testing.assert_array_equal(retained[key], old[1:])
    kept_observations = [(i - 1, p, sr, sp) for i, p, sr, sp in observations if i > 0]
    reduced = residual_function(retained, chain["factors"][1:], chain["gravity"],
                                marginal=marginal, observations=kept_observations)
    jr = jacobian(reduced, 45)
    expected_h = projected_j.T @ projected_j
    scale = np.sqrt(np.diag(expected_h))
    np.testing.assert_allclose((jr.T @ jr) / scale[:, None] / scale[None, :],
                               expected_h / scale[:, None] / scale[None, :], atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose((jr.T @ reduced(np.zeros(45))) / scale,
                               (projected_j.T @ projected_r) / scale, atol=2e-7, rtol=2e-7)
    expected_cost = .5 * projected_r @ projected_r
    actual = window.optimize(max_iterations=80, cost_tolerance=1e-14, step_tolerance=1e-11)
    assert actual["initial_cost"] + report["dropped_linearized_constant"] == pytest.approx(expected_cost, abs=1e-7, rel=1e-9)
    expected = least_squares(reduced, np.zeros(45), jac="3-point", ftol=1e-12, xtol=1e-12, gtol=1e-10, max_nfev=200)
    assert expected.success and actual["converged"]
    assert actual["final_cost"] == pytest.approx(expected.cost, abs=1e-9, rel=1e-8)
    for key, values in zip(["poses", "velocities", "biases"], local_states(retained, expected.x)):
        np.testing.assert_allclose(actual[key], values, atol=1e-6, rtol=1e-7)
    for key, values in marginal.items():
        np.testing.assert_array_equal(window.marginal_prior()[key], values)
    assert np.max(np.abs(marginal["sqrt_information"] - np.diag(np.diag(marginal["sqrt_information"])))) > 1.


def test_append_after_marginalization_matches_independent_reduced_problem(chain):
    base = {k: v[:3].copy() for k, v in zip(["times", "poses", "velocities", "biases"],
                                            [chain["times"], chain["poses"], chain["velocities"], chain["biases"]])}
    window = core.ImuWindow(base["times"], base["poses"], base["velocities"], base["biases"], gravity=chain["gravity"])
    for i in range(2):
        window.add_imu_factor(i, chain["factors"][i], gyro_bias_random_walk=.005, accel_bias_random_walk=.05)
    _anchor(chain, window)
    window.marginalize_oldest()
    window.append_state(int(chain["times"][3]), chain["poses"][3], chain["velocities"][3], chain["biases"][3], chain["factors"][2],
                        gyro_bias_random_walk=.005, accel_bias_random_walk=.05)
    pose = chain["poses"][3].copy()
    pose[:3, 3] += [.02, -.01, .03]
    window.add_pose_prior(2, pose, rotation_sigma=.03, position_sigma=.04)
    current = window.states()
    residual = residual_function(current, chain["factors"][1:], chain["gravity"],
                                  marginal=window.marginal_prior(), observations=[(2, pose, .03, .04)])
    expected = least_squares(residual, np.zeros(45), jac="3-point", ftol=1e-12, xtol=1e-12, gtol=1e-10, max_nfev=200)
    # The independent finite-difference optimum has an unscaled gradient of
    # about 2e-8; use a convergence tolerance above that numerical floor.
    actual = window.optimize(max_iterations=80, cost_tolerance=1e-14, gradient_tolerance=1e-7)
    assert expected.success and actual["converged"]
    assert actual["final_cost"] == pytest.approx(expected.cost, abs=1e-10, rel=1e-8)
    for key, values in zip(["poses", "velocities", "biases"], local_states(current, expected.x)):
        np.testing.assert_allclose(actual[key], values, atol=1e-6, rtol=1e-7)
    assert window.marginalize_oldest()["replaced_marginal_prior"] is True
    assert len(window.states()["timestamps_ns"]) == 2
    before = window.states()
    with pytest.raises(ValueError, match="at least two"):
        window.marginalize_oldest()
    for key in before:
        np.testing.assert_array_equal(window.states()[key], before[key])


def test_append_validation_and_prior_exports_are_atomic_owned_copies(chain):
    window = core.ImuWindow(chain["times"][:3], chain["poses"][:3], chain["velocities"][:3], chain["biases"][:3], gravity=chain["gravity"])
    for i in range(2):
        window.add_imu_factor(i, chain["factors"][i], gyro_bias_random_walk=.005, accel_bias_random_walk=.05)
    _anchor(chain, window)
    assert window.marginal_prior() is None
    window.marginalize_oldest()
    before = window.states()
    prior = window.marginal_prior()
    bad_bias = chain["biases"][3].copy()
    bad_bias[0] = np.nan
    for kwargs in [dict(timestamp_ns=int(chain["times"][3]) + 1), dict(bias=bad_bias), dict(gyro_bias_random_walk=0.)]:
        args = dict(timestamp_ns=int(chain["times"][3]), pose=chain["poses"][3], velocity=chain["velocities"][3],
                    bias=chain["biases"][3], factor=chain["factors"][2], gyro_bias_random_walk=.005, accel_bias_random_walk=.05)
        args.update(kwargs)
        with pytest.raises(ValueError):
            window.append_state(**args)
        for key in before:
            np.testing.assert_array_equal(window.states()[key], before[key])
        for key in prior:
            np.testing.assert_array_equal(window.marginal_prior()[key], prior[key])
    exported = window.marginal_prior()
    for key, value in exported.items():
        if isinstance(value, np.ndarray):
            value[...] = 123.
    for key in prior:
        np.testing.assert_array_equal(window.marginal_prior()[key], prior[key])
