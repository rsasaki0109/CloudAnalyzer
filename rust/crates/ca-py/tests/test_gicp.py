"""Independent fixed-metric derivatives and frame changes; no accuracy fixture."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")


def data(n=9):
    rng = np.random.default_rng(53918)
    source = rng.normal(size=(n, 3)) * [3., 2., 1.]
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
    pose[:3, 3] = [4., -2., 1.]
    target = source @ pose[:3, :3].T + pose[:3, 3] + rng.normal(size=(n, 3)) * .05
    shapes = []
    for _ in range(2):
        a = rng.normal(size=(n, 3, 3))
        shapes.append(a @ a.transpose(0, 2, 1) + np.eye(3) * .2)
    weights = rng.uniform(.1, 1., n)
    if n:
        weights[0] = 0.
    return source, target, *shapes, pose, np.array([2., -1., .5]), weights


def call(values, parallel=False):
    return core.gicp_linearization(*values[:6], weights=values[6], parallel=parallel)


def reference(values):
    source, target, cs, ct, pose, center, weights = values
    world = source @ pose[:3, :3].T + pose[:3, 3]
    residual = world - target
    shapes = ct + pose[:3, :3] @ cs @ pose[:3, :3].T
    arm = world - center
    j = np.zeros((len(source), 3, 6))
    j[:, :, :3] = np.eye(3)
    x, y, z = arm.T
    j[:, 0, 4], j[:, 0, 5] = z, -y
    j[:, 1, 3], j[:, 1, 5] = -z, x
    j[:, 2, 3], j[:, 2, 4] = y, -x
    precision_r = np.linalg.solve(shapes, residual[..., None])[..., 0]
    precision_j = np.linalg.solve(shapes, j)
    d2 = np.einsum("ni,ni->n", residual, precision_r)
    return {"normal_matrix": np.einsum("nri,n,nrj->ij", j, weights, precision_j),
            "right_hand_side": -np.einsum("nri,n,nr->i", j, weights, precision_r),
            "cost": .5 * float(weights @ d2), "source_world": world,
            "combined_shapes": shapes, "mahalanobis_squared": d2,
            "weight_sum": float(weights.sum()), "positive_weight_points": np.count_nonzero(weights)}, j


@pytest.mark.parametrize("n", [0, 1, 9, 257])
def test_against_independent_numpy_inverse_action(n):
    values = data(n)
    expected, _ = reference(values)
    result = call(values)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key], atol=2e-11, rtol=2e-12)
    assert result["source_world"].shape == (n, 3)
    assert result["combined_shapes"].shape == (n, 3, 3)
    assert result["mahalanobis_squared"].shape == (n,)


def test_finite_differences_use_frozen_transported_metric_and_declared_center():
    values = data()
    expected, jacobian = reference(values)
    native = call(values)
    world, target, center = expected["source_world"], values[1], values[5]
    lower = np.linalg.cholesky(expected["combined_shapes"])
    weights = values[6]
    def residual(delta):
        moved = ((world-center) @ Rotation.from_rotvec(delta[3:]).as_matrix().T
                 + center + delta[:3])
        return moved-target
    def cost(delta):
        white = np.linalg.solve(lower, residual(delta)[..., None])[..., 0]
        return .5 * float(weights @ np.sum(white*white, axis=1))
    for k in range(6):
        delta = np.eye(6)[k] * 1e-6
        np.testing.assert_allclose((residual(delta)-residual(-delta))/2e-6,
                                   jacobian[:, :, k], atol=3e-9, rtol=2e-9)
        derivative = (cost(delta)-cost(-delta))/2e-6
        np.testing.assert_allclose(derivative, -native["right_hand_side"][k], atol=3e-9, rtol=3e-8)


def test_world_basis_rotation_and_translation_transform_the_system():
    values = list(data())
    base = call(values)
    q = Rotation.from_rotvec([-.3, .4, .1]).as_matrix()
    shift = np.array([13., -7., 4.])
    basis = np.zeros((6, 6))
    basis[:3, :3] = basis[3:, 3:] = q
    changed = [v.copy() for v in values]
    changed[1] = values[1] @ q.T + shift
    changed[3] = q @ values[3] @ q.T
    changed[4][:3, :3] = q @ values[4][:3, :3]
    changed[4][:3, 3] = q @ values[4][:3, 3] + shift
    changed[5] = q @ values[5] + shift
    transformed = call(changed)
    np.testing.assert_allclose(transformed["normal_matrix"], basis @ base["normal_matrix"] @ basis.T, atol=3e-11)
    np.testing.assert_allclose(transformed["right_hand_side"], basis @ base["right_hand_side"], atol=1e-12)
    np.testing.assert_allclose(transformed["cost"], base["cost"], atol=1e-13)


def test_large_world_origin_preserves_local_parameterization_with_declared_tolerance():
    values = list(data())
    base = call(values)
    changed = [v.copy() for v in values]
    shift = np.array([1e6, -2e6, 3e6])
    changed[1] += shift
    changed[4][:3, 3] += shift
    changed[5] += shift
    translated = call(changed)
    for key in ["normal_matrix", "right_hand_side", "cost", "mahalanobis_squared"]:
        np.testing.assert_allclose(translated[key], base[key], atol=2e-7, rtol=1e-8)


def test_source_frame_reparameterization_includes_rotation_and_lever_arm():
    values = list(data())
    base = call(values)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec([.4, -.2, .1]).as_matrix()
    extrinsic[:3, 3] = [.5, -.3, .2]
    changed = [v.copy() for v in values]
    changed[0] = values[0] @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    changed[2] = extrinsic[:3, :3] @ values[2] @ extrinsic[:3, :3].T
    changed[4] = values[4] @ np.linalg.inv(extrinsic)
    # The physical rotation center stays in world axes, independently of sensor origin.
    result = call(changed)
    for key in base:
        np.testing.assert_allclose(result[key], base[key], atol=2e-11, rtol=2e-12)


def test_identity_combined_metric_reproduces_point_to_point_system():
    values = list(data())
    values[2] = np.tile(np.eye(3) * .5, (len(values[0]), 1, 1))
    values[3] = values[2].copy()
    values[4] = np.eye(4)
    result = call(values)
    expected, _ = reference(values)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key], atol=1e-12, rtol=2e-12)
    np.testing.assert_array_equal(result["combined_shapes"], np.tile(np.eye(3), (len(values[0]), 1, 1)))


@pytest.mark.parametrize("n", [0, 1, 9, 1025])
def test_serial_parallel_outputs_match_every_bit(n):
    values = data(n)
    serial, parallel = call(values), call(values, True)
    for key in serial:
        if isinstance(serial[key], np.ndarray):
            np.testing.assert_array_equal(serial[key].view(np.uint64), parallel[key].view(np.uint64))
        elif isinstance(serial[key], float):
            assert np.float64(serial[key]).view(np.uint64) == np.float64(parallel[key]).view(np.uint64)
        else:
            assert serial[key] == parallel[key]


@pytest.mark.parametrize("kind", ["source", "target"])
@pytest.mark.parametrize("bad", ["asymmetric", "singular", "indefinite", "nan", "infinite"])
def test_invalid_shapes_are_rejected_even_with_zero_weight(kind, bad):
    values = list(data())
    index = 2 if kind == "source" else 3
    matrix = values[index][4]
    matrix[:] = np.eye(3)
    if bad == "asymmetric":
        matrix[0, 1] = .1
    else:
        matrix[0, 0] = {"singular": 0., "indefinite": -1., "nan": np.nan, "infinite": np.inf}[bad]
    values[6][4] = 0.
    for parallel in [False, True]:
        with pytest.raises(ValueError, match=f"pair 4: {kind} shape"):
            call(values, parallel)


@pytest.mark.parametrize("weight", [-1., np.nan, np.inf, -np.inf])
def test_invalid_weights_fail_at_their_index(weight):
    values = list(data())
    values[6][3] = weight
    with pytest.raises(ValueError, match="pair 3: weight"):
        call(values)


def test_parallel_validation_reports_first_row_not_first_completed_task():
    values = list(data(1025))
    values[2][7, 0, 0] = np.nan
    values[1][333, 1] = np.inf
    values[6][1000] = -1.
    for parallel in [False, True]:
        with pytest.raises(ValueError, match="pair 7: source shape"):
            call(values, parallel)


@pytest.mark.parametrize("field", ["source", "target", "pose", "reflection", "center", "last_row"])
def test_invalid_points_and_common_frames(field):
    values = list(data())
    if field in ["source", "target"]:
        values[field == "target"][2, 0] = np.nan
    elif field == "pose":
        values[4][0, 0] *= 1.01
    elif field == "reflection":
        values[4][:3, 0] *= -1
    elif field == "center":
        values[5][1] = np.inf
    else:
        values[4][3, 0] = .01
    with pytest.raises(ValueError):
        call(values)


def test_roundoff_symmetry_is_documented_and_no_eigenvalue_floor_is_applied():
    values = list(data(1))
    values[6][:] = 1.
    values[4] = np.eye(4)
    values[2][:] = np.diag([1e-18, 1., 2.])
    values[3][:] = values[2]
    result = call(values)
    assert result["combined_shapes"][0, 0, 0] == 2e-18
    values[2][:] = np.eye(3)
    values[2][0, 0, 1] = 1e-14
    first = call(values)
    values[2] = values[2].transpose(0, 2, 1).copy()
    second = call(values)
    np.testing.assert_array_equal(first["normal_matrix"].view(np.uint64), second["normal_matrix"].view(np.uint64))


def test_strided_inputs_are_copied_and_outputs_are_independent():
    values = list(data())
    values[:4] = [v[::-1] for v in values[:4]]
    values[4] = np.asfortranarray(values[4])
    values[6] = values[6][::-1]
    before = [v.copy() for v in values]
    result = call(values)
    expected, _ = reference(values)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key], atol=2e-11)
    saved = {k: v.copy() for k, v in result.items() if isinstance(v, np.ndarray)}
    for actual, original in zip(values, before):
        np.testing.assert_array_equal(actual, original)
    for v in values:
        v[...] = 123.
    for key, old in saved.items():
        np.testing.assert_array_equal(result[key], old)
    result["normal_matrix"][:] = 123.
    fresh = call(before)
    np.testing.assert_allclose(fresh["normal_matrix"], expected["normal_matrix"], atol=2e-11)


@pytest.mark.parametrize("index,value", [
    (0, np.zeros((3, 2))), (1, np.zeros((3, 3))), (2, np.zeros((9, 3, 2))),
    (3, np.zeros((8, 3, 3))), (4, np.eye(3)), (5, np.zeros(2)), (6, np.ones(8)),
])
def test_incompatible_shapes_are_rejected(index, value):
    values = list(data())
    values[index] = value
    with pytest.raises(ValueError):
        call(values)


def test_finite_input_overflow_is_rejected_without_a_partial_result():
    values = list(data(1))
    values[0][:] = 1e308
    values[6][:] = 1.
    with pytest.raises(ValueError, match="pair 0: nonfinite"):
        call(values)
