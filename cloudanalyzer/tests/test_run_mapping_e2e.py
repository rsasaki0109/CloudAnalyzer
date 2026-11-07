"""Tests for the reference client in scripts/run_mapping_e2e.py (no native core needed)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_mapping_e2e as e2e  # noqa: E402


def _observation(start=0.0, stop=6.0, *, contained=True, direction=True, level=(True, True, True, True),
                 issues_limited=False, preview_limited=False):
    sections = [{"station_m": 2.0 * i, "path_level_supported": ok} for i, ok in enumerate(level)]
    return {
        "sections": sections,
        "section_preview_limited": preview_limited,
        "layout_width_preview": {"interpolated_ranges": [
            {"from_m": start, "to_m": stop, "path_contained_at_all_sections": contained,
             "endpoint_policy": "linear_interpolation"}]},
        "layout_direction_preview": {"ranges": [
            {"from_m": start, "to_m": stop, "direction_checks_passed": direction,
             "issues_limited": issues_limited}]},
    }


def test_interval_accepted_when_every_preview_passes():
    assert e2e.accepted_intervals(_observation()) == [(0.0, 6.0)]


def test_interval_rejected_when_any_preview_fails():
    assert e2e.accepted_intervals(_observation(contained=False)) == []
    assert e2e.accepted_intervals(_observation(direction=False)) == []
    assert e2e.accepted_intervals(_observation(issues_limited=True)) == []
    assert e2e.accepted_intervals(_observation(level=(True, False, True, True))) == []
    assert e2e.accepted_intervals(_observation(start=4.0, stop=4.0)) == []


def test_direction_result_must_match_the_width_interval():
    observation = _observation()
    observation["layout_direction_preview"]["ranges"][0]["to_m"] = 4.0
    assert e2e.accepted_intervals(observation) == []


def test_interval_beyond_truncated_section_preview_is_rejected():
    assert e2e.accepted_intervals(_observation(stop=8.0, level=(True,) * 4, preview_limited=True)) == []
    assert e2e.accepted_intervals(_observation(stop=6.0, preview_limited=True)) == [(0.0, 6.0)]


def test_overlaps_resolved_longest_first_with_deterministic_ties():
    observations = {
        3: _observation(0.0, 4.0, level=(True,) * 3),
        1: _observation(0.0, 6.0),
        2: _observation(6.0, 10.0, level=(True,) * 6),
        4: _observation(2.0, 6.0, level=(True,) * 4),
    }
    # 1 (6 m) wins over 3 and 4, which overlap it; 2 only touches 1's end.
    assert e2e.choose(observations) == [(1, 0.0, 6.0), (2, 6.0, 10.0)]

    tied = {7: _observation(0.0, 4.0, level=(True,) * 3), 5: _observation(2.0, 6.0, level=(True,) * 4)}
    assert e2e.choose(tied) == [(5, 2.0, 6.0)]


def test_rejection_reports_the_first_failing_check_in_order():
    width = lambda obs: obs["layout_width_preview"]["interpolated_ranges"][0]  # noqa: E731
    assert e2e.rejection(_observation(), width(_observation())) is None
    obs = _observation(contained=False, direction=False)
    assert e2e.rejection(obs, width(obs)) == "path_outside_profile"
    obs = _observation(direction=False, level=(True, False, True, True))
    assert e2e.rejection(obs, width(obs)) == "direction_unresolved"
    obs = _observation(stop=8.0, preview_limited=True)
    assert e2e.rejection(obs, width(obs)) == "sections_not_previewed"
    obs = _observation(level=(True, False, True, True))
    assert e2e.rejection(obs, width(obs)) == "surface_level_unsupported"


def _row(start, stop, status, candidates=(), reasons=()):
    return {"from_m": start, "to_m": stop, "status": status,
            "available_candidate_ids": list(candidates), "source_reasons": list(reasons)}


def test_breakdown_explains_every_unresolved_metre():
    observations = {
        1: _observation(0.0, 6.0),                        # accepted
        2: _observation(0.0, 6.0, direction=False),       # reaches the direction check
        3: _observation(0.0, 6.0, contained=False),       # fails earlier, at the path
    }
    observations[4] = _observation(0.0, 6.0)
    observations[4]["layout_width_preview"]["interpolated_ranges"] = []  # never wide enough
    disposition = [
        _row(0.0, 2.0, "included_geometry", [1]),
        _row(2.0, 4.0, "agent_deferred", [2, 3]),         # furthest check wins: direction
        _row(4.0, 6.0, "agent_deferred", [1]),            # accepted but not chosen here
        _row(6.0, 8.0, "agent_deferred", [4]),
        _row(8.0, 10.0, "agent_deferred", []),
        _row(10.0, 14.0, "source_deferred", [], ["branching_bands"]),
        _row(14.0, 15.0, "source_deferred", [], ["missing_ground_anchor", "branching_bands"]),
    ]
    breakdown = e2e.disposition_breakdown(disposition, observations)
    assert breakdown == {
        "included_m": 2.0,
        "source_deferred_m": {"branching_bands": 4.0, "branching_bands+missing_ground_anchor": 1.0},
        "client_deferred_m": {"direction_unresolved": 2.0, "narrower_than_layout": 2.0,
                              "no_inspected_candidate": 2.0, "overlapping_interval_chosen": 2.0},
        "other_m": {},
    }
    total = breakdown["included_m"] + sum(breakdown["source_deferred_m"].values()) \
        + sum(breakdown["client_deferred_m"].values())
    assert total == 15.0


def test_review_hold_summary_keeps_held_traces_and_floating_samples():
    def trace(lane, side, status, consensus_mismatches=0):
        return {"lane": lane, "trace": side, "status": status,
                "low_quantile": {"fraction": 0.8, "holds": ["below_support_threshold"]},
                "ground_consensus": {"fraction": 0.99, "holds": [], "height_mismatches": consensus_mismatches}}
    diagnosis = {"review_holds": {
        "traces": [trace(3, "left", "held", 1), trace(3, "center", "explained_by_returns_above")],
        "traces_total": 2, "unexplained_traces_total": 1, "ground_estimators_disagree": True,
        "below_trace_samples": [{"estimator": "ground_consensus", "lane": 3, "trace": "left",
                                 "lane_station_m": 165.89252, "xyz": [2.63758, -72.7398, -2.74145],
                                 "source_height_m": -3.118, "below_trace_m": 0.37671}],
        "below_trace_samples_total": 1,
    }}
    summary = e2e.review_hold_summary(diagnosis)
    assert summary["traces_total"] == 2 and summary["held_traces_total"] == 1
    assert summary["held"] == [{"lane": 3, "trace": "left", "low_quantile_fraction": 0.8,
                                "ground_consensus_fraction": 0.99, "ground_consensus_mismatches": 1}]
    assert summary["below_trace_samples"] == [{"estimator": "ground_consensus", "lane": 3, "trace": "left",
                                               "lane_station_m": 165.893, "xyz": [2.638, -72.74, -2.741],
                                               "below_trace_m": 0.377}]
    assert e2e.review_hold_summary({}) is None
