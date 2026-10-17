"""Analytic reciprocal checks, ties, full source coverage and system controls."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('consistency', ROOT/'scripts/flatwall_correspondence_consistency.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def exhaustive(source, target):
    return np.array([min(range(len(source)), key=lambda i: (float(np.sum((source[i]-p)**2)), i)) for p in target], dtype=np.int64)


def test_unmatched_source_can_disqualify_a_captured_match():
    source = np.array([[0., 0, 0], [1., 0, 0]])
    result = audit.diagnose(source, np.array([0]), np.array([[.9, 0, 0]]))
    assert result['reverse_source_indices'].tolist() == [1]
    assert result['reciprocal'].tolist() == [False]


def test_target_reuse_is_separate_from_reciprocity():
    source = np.array([[0., 0, 0], [.2, 0, 0], [2., 0, 0]])
    target = np.array([[.01, 0, 0], [.01, 0, 0], [1.9, 0, 0]])
    result = audit.diagnose(source, np.arange(3), target)
    assert result['target_multiplicity'].tolist() == [2, 2, 1]
    assert result['reciprocal'].tolist() == [True, False, True]
    assert result['unique_targets'] == 2 and result['saved_reverse_queries'] == 1


@pytest.mark.parametrize('workers', [1, 4])
def test_parallel_deduplicated_queries_match_exhaustive_ties(workers):
    source = np.array([[-1., 0, 0], [1., 0, 0], [1., 0, 0], [0., 2, 0]])
    target = np.array([[0., 0, 0], [1., 0, 0], [1., 0, 0], [0., 2, 0]])
    before = source.copy(), target.copy()
    result = audit.diagnose(source, np.arange(4), target, workers=workers)
    np.testing.assert_array_equal(result['reverse_source_indices'], exhaustive(source, target))
    assert result['reverse_source_indices'].tolist() == [0, 1, 1, 3]
    np.testing.assert_array_equal(source, before[0]); np.testing.assert_array_equal(target, before[1])


def test_sign_zero_target_coordinates_share_a_group():
    result = audit.diagnose(np.zeros((2, 3)), np.arange(2), np.array([[0., 0, 0], [-0., 0, 0]]))
    assert result['unique_targets'] == 1
    assert result['reciprocal'].tolist() == [True, False]


def test_empty_matches_retain_explicit_empty_arrays_and_system():
    result = audit.diagnose(np.ones((3, 3)), np.array([], dtype=np.int64), np.empty((0, 3)))
    assert result['unique_targets'] == result['saved_reverse_queries'] == 0
    assert result['reciprocal'].shape == (0,)
    system = audit.point_system(np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.zeros(3))
    assert system['geometry'] is system['radius_m'] is None
    assert system['weight_sum'] == system['weighted_cost'] == 0
    np.testing.assert_array_equal(system['normal_matrix'], np.zeros((6, 6)))


def test_filtered_system_is_exact_sum_of_retained_scalar_rows():
    q = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    p, w, c = q+[.1, -.2, .3], np.arange(1., 5.), np.array([.2, -.1, .3])
    mask = np.array([True, False, True, True])
    result = audit.point_system(q[mask], p[mask], w[mask], c)
    h, b = np.zeros((6, 6)), np.zeros(6)
    for source, target, weight in zip(q[mask], p[mask], w[mask]):
        x, y, z = source-c
        j = np.column_stack([np.eye(3), [[0., z, -y], [-z, 0., x], [y, -x, 0.]]])
        h += weight*j.T@j
        b -= weight*j.T@(source-target)
    np.testing.assert_allclose(result['normal_matrix'], h)
    np.testing.assert_allclose(result['right_hand_side'], b)
    assert result['geometry']['weak_directions_below_1e_3'] == 0


def test_common_world_translation_preserves_reciprocity_and_centered_system():
    q = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    p, c = q+[.1, -.1, .1], np.array([.2, -.1, .3])
    r = audit.diagnose(q, np.arange(4), p)
    shifted = audit.diagnose(q+[20., -30, 40], np.arange(4), p+[20., -30, 40])
    np.testing.assert_array_equal(r['reciprocal'], shifted['reciprocal'])
    a = audit.point_system(q, p, np.ones(4), c)
    b = audit.point_system(q+[20., -30, 40], p+[20., -30, 40], np.ones(4), c+[20., -30, 40])
    for key in ['normal_matrix', 'right_hand_side']: np.testing.assert_allclose(a[key], b[key], atol=1e-12)


@pytest.mark.parametrize('kind', ['duplicate', 'negative', 'outside', 'float', 'nan', 'workers'])
def test_invalid_reverse_inputs_are_rejected(kind):
    q, p, indices, workers = np.zeros((2, 3)), np.zeros((2, 3)), np.arange(2), 4
    if kind == 'duplicate': indices = np.zeros(2, dtype=np.int64)
    elif kind == 'negative': indices[0] = -1
    elif kind == 'outside': indices[1] = 2
    elif kind == 'float': indices = indices.astype(float)
    elif kind == 'nan': p[0, 0] = np.nan
    else: workers = 2
    with pytest.raises(ValueError): audit.diagnose(q, indices, p, workers=workers)
