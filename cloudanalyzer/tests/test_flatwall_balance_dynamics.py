"""Ordered nearest-search boundaries and independent recurrence diagnostics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('balance_dynamics', ROOT/'scripts/flatwall_balance_dynamics.py')
dynamics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamics)


def test_global_nearest_outside_native_neighborhood_does_not_hide_a_valid_local_target():
    search = dynamics.Search([[2.01,.99,.99], [1.99,1.99,1.99]])
    q = np.array([[.99,.99,.99]])
    ids, target, distance = search.matches(q, 4.)
    np.testing.assert_array_equal(ids, [0])
    np.testing.assert_array_equal(target, [1])
    np.testing.assert_allclose(distance, [3.])
    for actual, expected in zip((ids, target, distance), search.matches(q, 4., exhaustive=True)):
        np.testing.assert_array_equal(actual, expected)


def test_equal_distance_tie_visits_center_cell_before_neighbor_cells():
    search = dynamics.Search([[-.5,0.,0.], [.5,0.,0.]])
    _, target, _ = search.matches(np.zeros((1,3)), .25)
    np.testing.assert_array_equal(target, [1])
    assert len(search.matches(np.zeros((1,3)), np.nextafter(.25, 0.))[0]) == 0


def test_broad_phase_agrees_with_all_ordered_cell_candidates():
    # Analytic math/search fixture only; runtime results use the real recordings.
    grid = np.array([[x*.5+.125,y*.5+.125,z*.5+.125]
        for x in range(-3,5) for y in range(-3,5) for z in range(-2,3)])
    q = np.array([[i*.1875-.75,j*.1875-.75,k*.1875]
        for i in range(11) for j in range(9) for k in range(3)])
    search = dynamics.Search(grid)
    for actual, expected in zip(search.matches(q, 1.), search.matches(q, 1., exhaustive=True)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('points', [[], [[np.nan,0.,0.]], [[0.,0.]]])
def test_invalid_map_is_rejected(points):
    with pytest.raises(ValueError):
        dynamics.Search(points)


def test_exact_target_count_normalizes_signed_zero_in_weight_arithmetic():
    world = np.array([[.125,0.,0.], [.25,0.,0.]])
    h, b, count, weight, cost, unique = dynamics.system(world, np.array([0,1]),
        np.array([[0.,0.,0.],[-0.,0.,0.]]), np.zeros(3), 1.)
    np.testing.assert_array_equal(count, [2,2])
    np.testing.assert_allclose(weight, 1/(1+world[:,0]**2)**2/2)
    assert unique == 1 and h.shape == (6,6) and b.shape == (6,) and cost > 0


def tail_state(i):
    pose = np.eye(4)
    pose[0,3] = (i%2)*.001
    return dict(before=pose, source_indices=np.array([0,1]), target_indices=np.array([i%2,1-i%2]),
        multiplicity=np.array([1,1]), weights=np.array([.5,.25]), step=np.array([.001,0.,0.,0.,0.,0.]))


def test_recurrence_distinguishes_alternation_from_convergence():
    report = dynamics.tail_summary([tail_state(i) for i in range(24)], np.zeros(3))
    assert report['lags'][0]['identical_ordered_pair_sets'] == 0
    assert report['lags'][1]['identical_ordered_pair_sets'] == 22
    assert report['lags'][1]['maximum_translation_m'] == 0
    assert report['step_translation_m_range'] == [.001,.001]
    assert report['stopping_policy_changed'] is False


def test_count_changes_are_visible_even_when_a_source_keeps_its_target():
    states = [tail_state(i) for i in range(24)]
    states[0]['target_indices'] = np.array([0,0])
    states[0]['multiplicity'] = np.array([2,2])
    states[1]['target_indices'] = np.array([0,1])
    report = dynamics.tail_summary(states, np.zeros(3))
    assert report['consecutive_changes'][0]['changed_target_or_support'] == 1
    assert report['consecutive_changes'][0]['unchanged_target_count_changed'] == 1


def test_tail_cannot_be_shortened_to_hide_a_missing_state():
    with pytest.raises(ValueError, match='24'):
        dynamics.tail_summary([tail_state(i) for i in range(23)], np.zeros(3))
