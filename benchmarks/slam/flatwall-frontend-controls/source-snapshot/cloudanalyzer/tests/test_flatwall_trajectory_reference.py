"""Analytic controls for conditional evaluation, not physical accuracy evidence."""
import io
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import flatwall_trajectory_reference as scoring


def poses(points):
    result = np.tile(np.eye(4), (len(points), 1, 1))
    result[:, :3, 3] = points
    return result


def rigid(angles, translation):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    t[:3, 3] = translation
    return t


POINTS = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [1., 2., 3.]])


def test_decimal_unix_seconds_preserve_single_nanoseconds_and_quaternion_normalization():
    text = "# assumed TUM\n\n1656505431.000000001 0 0 0 0 0 0 0.999999\n1656505431.000000002 1 2 3 0 0 0 1\n"
    stamps, actual, error = scoring.parse_tum_assumption(text)
    assert stamps.tolist() == [1656505431000000001, 1656505431000000002]
    np.testing.assert_array_equal(actual[:, :3, :3], np.tile(np.eye(3), (2, 1, 1)))
    assert abs(error-1e-6) < 1e-15
    np.testing.assert_array_equal(actual[1, :3, 3], [1., 2., 3.])


@pytest.mark.parametrize("token", ["1656505431.0000000000000000000000000001", "1e-999999999"])
def test_subnanosecond_digits_cannot_disappear_under_decimal_context(token):
    with pytest.raises(ValueError, match="integer nanoseconds"):
        scoring.exact_nanoseconds(token)


def test_long_trailing_zeroes_and_int64_boundaries_remain_exact():
    assert scoring.exact_nanoseconds("1656505431.000000001000000000000000000000") == 1656505431000000001
    assert scoring.exact_nanoseconds("0e-999999999") == 0
    assert scoring.exact_nanoseconds("-9223372036.854775808") == -(2**63)
    assert scoring.exact_nanoseconds("9223372036.854775807") == 2**63-1


@pytest.mark.parametrize("stamp", ["nan", "inf", "1656505431.0000000001", "9223372036.854775808", "bad"])
def test_invalid_reference_time_is_not_rounded_or_repaired(stamp):
    with pytest.raises(ValueError, match="reference line"):
        scoring.parse_tum_assumption(f"{stamp} 0 0 0 0 0 0 1")


@pytest.mark.parametrize("text", ["", "1 0 0 0 0 0 0", "1 0 0 0 0 0 0 0", "1 0 0 0 0 0 0 2", "1 nan 0 0 0 0 0 1", "1 0 0 0 0 0 nan 1"])
def test_bad_rows_and_quaternions_are_rejected(text):
    with pytest.raises(ValueError):
        scoring.parse_tum_assumption(text)


@pytest.mark.parametrize("times", [np.array([1., 2.]), np.array([True, False]), np.array([1, 1], dtype=np.int64), np.array([2, 1], dtype=np.int64), np.array([[1]], dtype=np.int64)])
def test_timestamp_contract(times):
    with pytest.raises(ValueError):
        scoring.timestamps(times)


def test_integer_brackets_exact_endpoints_gap_coverage_and_shortest_arc():
    base = 1656505431000000000
    ref_times = np.array([base+1, base+11, base+100000012], dtype=np.int64)
    ref = poses([[0, 0, 0], [2, 0, 0], [3, 0, 0]])
    ref[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    times = np.array([base, base+1, base+6, base+11, base+12, base+100000012, base+100000013], dtype=np.int64)
    before = [a.copy() for a in [times, ref_times, ref]]
    rows, indices, paired = scoring.associate(times, ref_times, ref)
    assert [r["status"] for r in rows] == ["before_reference", "exact", "interpolated", "exact", "reference_gap", "exact", "after_reference"]
    assert indices.tolist() == [1, 2, 3, 5]
    assert rows[4]["bracket_span_ns"] == 100000001
    np.testing.assert_array_equal(paired[0], ref[0])
    np.testing.assert_array_equal(paired[2], ref[1])
    np.testing.assert_allclose(paired[1, :3, 3], [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(paired[1, :3, :3], Rotation.from_euler("z", 45, degrees=True).as_matrix(), atol=1e-15)
    for actual, saved in zip([times, ref_times, ref], before):
        np.testing.assert_array_equal(actual, saved)
    # +170 to -170 degrees interpolates through 180, not through zero.
    ref[:2, :3, :3] = Rotation.from_euler("z", np.array([[170], [-170]]), degrees=True).as_matrix()
    _, _, result = scoring.associate(times[2:3], ref_times, ref)
    np.testing.assert_allclose(result[0, :3, :3], np.diag([-1., -1., 1.]), atol=1e-15)


def test_known_rigid_gauge_has_zero_errors_and_does_not_mutate_inputs():
    est = poses(POINTS)
    transform = rigid([20., -30., 70.], [2., -4., 8.])
    ref = transform @ est
    original = [a.copy() for a in [est, ref]]
    result, errors = scoring.score(est, ref, np.arange(5, dtype=np.int64), np.arange(5, dtype=np.int64))
    assert result["position_fit"]["status"] == "identified"
    np.testing.assert_allclose(result["position_fit"]["world_alignment"], transform, atol=3e-15)
    for values in errors.values():
        np.testing.assert_allclose(values, 0., atol=1e-12)
    for actual, saved in zip([est, ref], original):
        np.testing.assert_array_equal(actual, saved)


@pytest.mark.parametrize("mode", ["scale", "reflection"])
def test_fit_cannot_hide_scale_or_reflection(mode):
    est, ref = poses(POINTS), poses(POINTS.copy())
    if mode == "scale":
        ref[:, :3, 3] *= 2.
    else:
        ref[:, 0, 3] *= -1.
    result, errors = scoring.score(est, ref, np.arange(5, dtype=np.int64), np.arange(5, dtype=np.int64))
    fit = np.array(result["position_fit"]["world_alignment"])
    assert abs(np.linalg.det(fit[:3, :3])-1) < 1e-12
    assert result["position_fit"]["scale"] == 1.
    assert scoring.summarize(errors["rigid_position_ape_m"])["rmse"] > .1


def test_collinear_fit_is_explicitly_unavailable_and_single_pose_has_no_edges():
    est = poses([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    ref = rigid([10, 20, 30], [1, 2, 3]) @ est
    result, errors = scoring.score(est, ref, np.arange(3, dtype=np.int64), np.arange(3, dtype=np.int64))
    assert result["position_fit"]["status"] == "degenerate_rank"
    assert result["position_fit"]["ranks"]["estimate"] == 1
    assert "rigid_position_ape_m" not in errors
    np.testing.assert_allclose(errors["origin_position_ape_m"], 0, atol=1e-15)
    result, _ = scoring.score(est[:1], ref[:1], np.array([0], dtype=np.int64), np.array([10], dtype=np.int64))
    assert result["edges"] == [] and result["metrics"]["consecutive_rpe_translation_m"] is None


def test_excluded_original_scan_is_never_bridged_by_rpe_or_path_length():
    est = poses([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 2, 3]])
    indices = np.array([0, 1, 3, 4], dtype=np.int64)
    result, _ = scoring.score(est, est.copy(), indices, indices*10)
    assert [(r["left_estimate_index"], r["right_estimate_index"]) for r in result["edges"]] == [(0, 1), (3, 4)]
    assert result["consecutive_path_lengths"] == {"estimate_m": 4., "reference_m": 4.}
    assert [r["elapsed_ns"] for r in result["edges"]] == [10, 10]


def test_rotation_errors_include_half_turn_and_are_invariant_to_world_gauge():
    est, ref = poses(POINTS), poses(POINTS)
    est[:, :3, :3] = Rotation.from_euler("z", np.array([[0], [90], [180], [90], [0]]), degrees=True).as_matrix()
    ids = np.arange(5, dtype=np.int64)
    _, before = scoring.score(est, ref, ids, ids)
    np.testing.assert_allclose(before["origin_rotation_ape_deg"], [0, 90, 180, 90, 0], atol=1e-12)
    np.testing.assert_allclose(before["consecutive_rpe_rotation_deg"], 90, atol=1e-12)
    _, after = scoring.score(rigid([13, 29, 37], [1e6, -2e6, 3e6]) @ est, rigid([-7, 9, 17], [4e6, 5e6, -6e6]) @ ref, ids, ids)
    for name in before:
        np.testing.assert_allclose(after[name], before[name], atol=1e-8)


@pytest.mark.parametrize("kind", ["reflection", "nonorthogonal", "last_row", "nan"])
def test_improper_estimates_are_rejected(kind):
    est = poses(POINTS)
    if kind == "reflection": est[1, 0, 0] = -1
    if kind == "nonorthogonal": est[1, 0, 0] = 2
    if kind == "last_row": est[1, 3, 0] = .1
    if kind == "nan": est[1, 1, 3] = np.nan
    with pytest.raises(ValueError): scoring.validate_poses(est)


@pytest.mark.parametrize("kind", ["valid", "duplicate", "symlink", "missing"])
def test_archive_reads_only_bounded_literal_regular_member_without_extraction(tmp_path, kind):
    path = tmp_path / "reference.tar.gz"
    content = b"1 0 0 0 0 0 0 1\n2 1 0 0 0 0 0 1\n"
    with tarfile.open(path, "w:gz") as archive:
        other = tarfile.TarInfo("../../untrusted")
        other.size = 3
        archive.addfile(other, io.BytesIO(b"\xff\x00\xfe"))
        if kind != "missing":
            for _ in range(2 if kind == "duplicate" else 1):
                member = tarfile.TarInfo("gt/flatwall_01.txt")
                if kind == "symlink":
                    member.type, member.linkname = tarfile.SYMTYPE, "../../untrusted"
                    archive.addfile(member)
                else:
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
    if kind == "valid":
        times, result, receipt = scoring.read_selected_reference(path)
        assert times.tolist() == [1000000000, 2000000000] and receipt["rows"] == 2
        np.testing.assert_array_equal(result[:, :3, 3], [[0, 0, 0], [1, 0, 0]])
    else:
        with pytest.raises(ValueError, match="regular"): scoring.read_selected_reference(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["reference.tar.gz"]
