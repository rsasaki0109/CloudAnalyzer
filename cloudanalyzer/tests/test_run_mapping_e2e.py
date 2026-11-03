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
