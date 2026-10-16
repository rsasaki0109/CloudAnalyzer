"""Inertial-factor mathematics and optimizer interface, not SLAM accuracy claims."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from test_imu_preintegration import EPOCH, _write  # noqa: E402


@pytest.fixture
def inertial(tmp_path):
    times = EPOCH + np.array([0, 7, 19, 24, 39, 48, 60, 72, 83, 100], dtype=np.int64) * 1_000_000
    k = np.arange(len(times))
    gyro = np.column_stack([np.sin(k), .3 * np.cos(k), .4 + .1 * k])
    accel = np.column_stack([.2 * k, np.cos(k), 9.8 + .1 * k])
    reader = _write(tmp_path, times, gyro, accel)
    bias = (.03, -.02, .01, .1, .2, -.05)
    arguments = dict(gyro_noise_density=.02, accel_noise_density=.15, bias=bias)
    factor = reader.imu_factor(int(times[0]), int(times[-1]), **arguments)
    return reader, times, arguments, factor


def _initial():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([.3, -.2, .5]).as_matrix()
    pose[:3, 3] = [10., -2., 3.]
    return pose, np.array([.2, -.3, .1])


def _retract(pose, velocity, k, h):
    pose, velocity = pose.copy(), velocity.copy()
    if k < 3:
        theta = np.zeros(3)
        theta[k] = h
        pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(theta).as_matrix()
    elif k < 6:
        velocity[k - 3] += h
    else:
        pose[k - 6, 3] += h
    return pose, velocity


def _oracle(m, pose_i, velocity_i, pose_j, velocity_j, gravity, bias):
    correction = m["bias_jacobian"] @ (bias - m["linearization_bias"])
    rotation = m["delta_rotation"] @ Rotation.from_rotvec(correction[:3]).as_matrix()
    velocity = m["delta_velocity"] + correction[3:6]
    position = m["delta_position"] + correction[6:]
    dt = m["delta_time_s"]
    ri_t = pose_i[:3, :3].T
    return np.r_[Rotation.from_matrix(rotation.T @ ri_t @ pose_j[:3, :3]).as_rotvec(),
                 ri_t @ (velocity_j - velocity_i - gravity * dt) - velocity,
                 ri_t @ (pose_j[:3, 3] - pose_i[:3, 3] - velocity_i * dt - .5 * gravity * dt**2) - position]


@pytest.mark.parametrize("angle", [1e-8, .5, np.pi - 1e-3])
def test_jacobians_and_whitening_match_independent_oracle_at_nonzero_error(inertial, angle):
    _, _, _, factor = inertial
    m = factor.measurement()
    pose, velocity = _initial()
    gravity = np.array([.1, -.2, -9.81])
    bias = m["linearization_bias"] + np.array([.002, -.001, .003, .02, -.01, .03])
    predicted = factor.predict(pose, velocity, gravity=gravity, bias=bias)
    terminal = predicted["pose"].copy()
    terminal[:3, :3] = terminal[:3, :3] @ Rotation.from_rotvec(np.array([1., -2., 3.]) / np.sqrt(14) * angle).as_matrix()
    terminal[:3, 3] += [.1, -.2, .3]
    terminal_velocity = predicted["velocity"] + [.2, -.1, .3]
    linear = factor.linearize(pose, velocity, terminal, terminal_velocity, gravity=gravity, bias=bias)
    expected = _oracle(m, pose, velocity, terminal, terminal_velocity, gravity, bias)
    np.testing.assert_allclose(linear["residual"], expected, rtol=1e-9, atol=2e-12)
    lower = np.linalg.cholesky(m["covariance"])
    np.testing.assert_allclose(linear["whitened_residual"], np.linalg.solve(lower, expected), rtol=1e-9, atol=1e-9)
    assert linear["cost"] == pytest.approx(.5 * expected @ np.linalg.solve(m["covariance"], expected), rel=1e-9)
    h = 1e-5
    for group, width in [("i", 9), ("j", 9), ("bias", 6), ("gravity", 3)]:
        numerical = np.zeros((9, width))
        for k in range(width):
            pa, ma, pz, mz = [pose.copy(), pose.copy(), terminal.copy(), terminal.copy()]
            pv, mv, pzv, mzv = [velocity.copy(), velocity.copy(), terminal_velocity.copy(), terminal_velocity.copy()]
            pb, mb, pg, mg = [bias.copy(), bias.copy(), gravity.copy(), gravity.copy()]
            if group == "i":
                pa, pv = _retract(pose, velocity, k, h)
                ma, mv = _retract(pose, velocity, k, -h)
            elif group == "j":
                pz, pzv = _retract(terminal, terminal_velocity, k, h)
                mz, mzv = _retract(terminal, terminal_velocity, k, -h)
            elif group == "bias":
                pb[k] += h
                mb[k] -= h
            else:
                pg[k] += h
                mg[k] -= h
            numerical[:, k] = (_oracle(m, pa, pv, pz, pzv, pg, pb) - _oracle(m, ma, mv, mz, mzv, mg, mb)) / (2 * h)
        np.testing.assert_allclose(linear[f"jacobian_{group}"], numerical, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(linear[f"whitened_jacobian_{group}"], np.linalg.solve(lower, linear[f"jacobian_{group}"]), rtol=1e-10, atol=1e-9)


def test_bias_correction_is_first_order_and_measurement_owns_its_arrays(inertial):
    reader, times, arguments, factor = inertial
    m = factor.measurement()
    np.testing.assert_array_equal(m["linearization_bias"], arguments["bias"])
    direct = reader.preintegrate_imu(int(times[0]), int(times[-1]), **arguments)
    for key in direct:
        np.testing.assert_equal(m[key], direct[key])
    pose, velocity = _initial()
    errors = []
    for scale in [1., .5, .25]:
        bias = np.array(arguments["bias"]) + scale * np.array([.02, -.03, .01, .04, .05, -.02])
        predicted = factor.predict(pose, velocity, gravity=(0., 0., -9.81), bias=bias)
        corrected_arguments = {**arguments, "bias": tuple(bias)}
        exact = reader.imu_factor(int(times[0]), int(times[-1]), **corrected_arguments).predict(pose, velocity, gravity=(0., 0., -9.81))
        errors.append(np.linalg.norm(predicted["pose"] - exact["pose"]) + np.linalg.norm(predicted["velocity"] - exact["velocity"]))
    assert errors[1] < .3 * errors[0] and errors[2] < .3 * errors[1]
    m["delta_rotation"][:] = np.nan
    m["linearization_bias"][:] = np.nan
    assert np.isfinite(factor.predict(pose, velocity, gravity=(0., 0., -9.81))["pose"]).all()


def test_factor_linearization_can_drive_a_gauss_newton_update(inertial):
    factor = inertial[-1]
    pose, velocity = _initial()
    expected = factor.predict(pose, velocity, gravity=(0., 0., -9.81))
    terminal, terminal_velocity = expected["pose"].copy(), expected["velocity"].copy()
    terminal[:3, :3] = terminal[:3, :3] @ Rotation.from_rotvec([.3, -.2, .1]).as_matrix()
    terminal[:3, 3] += [.2, -.1, .3]
    terminal_velocity += [.1, .2, -.3]
    initial_cost = factor.evaluate(pose, velocity, terminal, terminal_velocity, gravity=(0., 0., -9.81))["cost"]
    for _ in range(2):
        linear = factor.linearize(pose, velocity, terminal, terminal_velocity, gravity=(0., 0., -9.81))
        step = np.linalg.lstsq(linear["whitened_jacobian_j"], -linear["whitened_residual"], rcond=None)[0]
        terminal[:3, :3] = terminal[:3, :3] @ Rotation.from_rotvec(step[:3]).as_matrix()
        terminal_velocity += step[3:6]
        terminal[:3, 3] += step[6:]
    final = factor.evaluate(pose, velocity, terminal, terminal_velocity, gravity=(0., 0., -9.81))
    assert initial_cost > 1. and final["cost"] < 1e-20
    np.testing.assert_allclose(terminal, expected["pose"], atol=1e-12)
    np.testing.assert_allclose(terminal_velocity, expected["velocity"], atol=1e-12)


def test_world_frame_change_preserves_residuals(inertial):
    factor = inertial[-1]
    pose, velocity = _initial()
    gravity = np.array([.1, -.2, -9.81])
    predicted = factor.predict(pose, velocity, gravity=gravity)
    terminal = predicted["pose"].copy()
    terminal[:3, 3] += [.1, .2, -.1]
    evaluated = factor.evaluate(pose, velocity, terminal, predicted["velocity"], gravity=gravity)
    world = Rotation.from_rotvec([-.5, .2, -.8]).as_matrix()
    offset = np.array([1e5, -2e5, 3e5])

    def change(p):
        p = p.copy()
        p[:3, :3] = world @ p[:3, :3]
        p[:3, 3] = world @ p[:3, 3] + offset
        return p

    reframed = factor.evaluate(change(pose), world @ velocity, change(terminal), world @ predicted["velocity"], gravity=world @ gravity)
    np.testing.assert_allclose(reframed["residual"], evaluated["residual"], atol=1e-10)


@pytest.mark.parametrize("problem", ["shape", "homogeneous", "reflection", "rotation", "position", "velocity", "gravity", "bias"])
def test_rejects_invalid_state_and_parameters(inertial, problem):
    factor = inertial[-1]
    pose, velocity = _initial()
    gravity = np.array([0., 0., -9.81])
    bias = np.zeros(6)
    if problem == "shape":
        pose = np.eye(3)
    elif problem == "homogeneous":
        pose[3, 0] = .1
    elif problem == "reflection":
        pose[:3, 0] *= -1
    elif problem == "rotation":
        pose[0, 0] += .1
    elif problem == "position":
        pose[0, 3] = np.nan
    elif problem == "velocity":
        velocity[0] = np.inf
    elif problem == "gravity":
        gravity[0] = np.nan
    else:
        bias[0] = np.nan
    with pytest.raises(ValueError):
        factor.predict(pose, velocity, gravity=gravity, bias=bias)


def test_weighted_factor_rejects_singular_covariance_while_preintegration_remains_available(tmp_path):
    times = EPOCH + np.array([0, 10_000_000], dtype=np.int64)
    reader = _write(tmp_path, times, np.zeros((2, 3)), np.array([[0., 0., 9.81]] * 2))
    for density in [0., .01]:
        args = dict(gyro_noise_density=density, accel_noise_density=density)
        assert reader.preintegrate_imu(int(times[0]), int(times[-1]), **args)["integration_steps"] == 1
        with pytest.raises(ValueError, match="positive definite"):
            reader.imu_factor(int(times[0]), int(times[-1]), **args)


def test_stationary_sensor_with_rotated_world_cancels_gravity(tmp_path):
    times = EPOCH + np.arange(11, dtype=np.int64) * 10_000_000
    reader = _write(tmp_path, times, np.zeros((11, 3)), np.array([[0., 0., 9.81]] * 11))
    factor = reader.imu_factor(int(times[0]), int(times[-1]), gyro_noise_density=.02, accel_noise_density=.15)
    pose, velocity = _initial()
    gravity = pose[:3, :3] @ [0., 0., -9.81]
    result = factor.predict(np.asfortranarray(pose), velocity, gravity=gravity)
    np.testing.assert_allclose(result["velocity"], velocity, atol=1e-14)
    np.testing.assert_allclose(result["pose"][:3, 3], pose[:3, 3] + velocity * .1, atol=1e-14)
