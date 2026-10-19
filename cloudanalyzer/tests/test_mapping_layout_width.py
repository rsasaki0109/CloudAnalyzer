"""Width observations preserve usable subranges without adopting lane drafts."""
from copy import deepcopy
import json

import pytest

from ca import mapping_job as jobs, mapping_run as runs
from ca.mapping_geometry import preview_layout_width
from tests.test_mapping_job import job_backend, _corridor_report, _run_layout


def _candidate(widths):
    return {"id": 37, "total_sections": len(widths), "sections": [
        {"station_m": i * 2., "left": [i * 2., w / 2, 0.],
         "right": [i * 2., -w / 2, 0.], "center": [i * 2., 0., 0.],
         "intersects_trajectory": True, "support_span_m": w,
         "left_evidence": "support_gap", "right_evidence": "support_gap"}
        for i, w in enumerate(widths)]}


def test_narrow_section_keeps_separate_wider_observed_ranges():
    candidate = _candidate([3.5, 3.5, 1., 3.5, 3.5])
    before = deepcopy(candidate)
    preview = preview_layout_width(candidate, _run_layout(width=2.5))
    assert [(r["from_m"], r["to_m"]) for r in preview["ranges"]] == [(0., 2.), (6., 8.)]
    assert preview["width_matching_station_length_m"] == 4.
    assert preview["checked_section_pairs"] == 4 and preview["uninspected_section_pairs"] == 0
    assert candidate == before and not preview["adopted"]
    assert not preview["source_support_checked"] and not preview["lane_generation_checked"]


def test_fixed_lane_fractions_determine_required_total_span():
    layout = _run_layout(width=2.)
    layout["lanes"] = [{**layout["lanes"][0], "fraction": fraction} for fraction in (.4, .6)]
    before = deepcopy(layout)
    preview = preview_layout_width(_candidate([4.5, 6., 5.]), layout)
    assert preview["required_span_m"] == 5.
    assert [(r["from_m"], r["to_m"]) for r in preview["ranges"]] == [(2., 4.)]
    assert layout == before


@pytest.mark.parametrize("end", [(0., 4.), (-4., 0.)])
def test_interpolated_pinches_are_not_missed_between_wide_sections(end):
    candidate = _candidate([4., 4.])
    for section, span in zip(candidate["sections"], [(4., 0.), end]):
        section["left"] = [span[0] / 2, span[1] / 2, 0.]
        section["right"] = [-span[0] / 2, -span[1] / 2, 0.]
    assert preview_layout_width(candidate, _run_layout(width=3.))["ranges"] == []


def test_width_uses_xy_geometry_and_retains_off_path_evidence():
    candidate = _candidate([2., 2.])
    for section in candidate["sections"]:
        section["support_span_m"] = 100.
        section["left"][2], section["right"][2] = 10., -10.
    assert preview_layout_width(candidate, _run_layout(width=3.))["ranges"] == []
    candidate = _candidate([3., 3.])
    candidate["sections"][1]["intersects_trajectory"] = False
    preview = preview_layout_width(candidate, _run_layout(width=2.5))
    assert len(preview["ranges"]) == 1 and not preview["ranges"][0]["path_contained_at_all_sections"]


@pytest.mark.parametrize("width,fits", [(2.5 - 5e-7, True), (2.5 - 2e-6, False)])
def test_width_tolerance_matches_existing_profile_width_checks(width, fits):
    preview = preview_layout_width(_candidate([width, width]), _run_layout(width=2.5))
    assert bool(preview["ranges"]) is fits


@pytest.mark.parametrize("width,fits", [(5. - 2e-6, True), (5. - 3e-6, False)])
def test_width_tolerance_applies_to_each_lane_not_to_the_total_span(width, fits):
    layout = _run_layout(width=2.)
    layout["lanes"] = [{**layout["lanes"][0], "fraction": fraction} for fraction in (.4, .6)]
    preview = preview_layout_width(_candidate([width, width]), layout)
    assert bool(preview["ranges"]) is fits


def test_uninspected_sections_are_not_counted_or_offered():
    candidate = _candidate([3.] * 128)
    candidate.update(total_sections=180, section_preview_limited=True)
    preview = preview_layout_width(candidate, _run_layout(width=2.5))
    assert preview["checked_section_pairs"] == 127 and preview["total_section_pairs"] == 179
    assert preview["uninspected_section_pairs"] == 52
    assert preview["ranges"][0]["to_m"] == 254.
    assert preview["width_matching_station_length_m"] == 254.


def test_range_response_is_bounded_without_hiding_range_count():
    preview = preview_layout_width(_candidate([3., 3., 1.] * 40), _run_layout(width=2.5))
    assert preview["ranges_total"] == 40 and len(preview["ranges"]) == 16
    assert preview["ranges_limited_in_response"] and preview["width_matching_station_length_m"] == 80.


def _proposal(candidate):
    report = _corridor_report()
    candidate = deepcopy(candidate)
    candidate.update(from_m=0., to_m=candidate["sections"][-1]["station_m"],
        minimum_support_span_m=min(s["support_span_m"] for s in candidate["sections"]),
        maximum_support_span_m=max(s["support_span_m"] for s in candidate["sections"]),
        paired_curb_sections=0, curb_width_range_m=None, review_required=True)
    report["candidates"] = [candidate]
    report["trajectory_length_m"] = candidate["to_m"]
    return report


def test_run_inspection_records_width_ranges_without_processing_attempts(job_backend, tmp_path):
    source, _ = job_backend
    report = _proposal(_candidate([3.5, 3.5, 1., 3.5, 3.5]))
    jobs.core().propose_road_corridors = lambda *args: json.dumps(report)
    root = tmp_path / "run"
    run = runs.start_mapping_run(str(source), str(root), _run_layout(width=2.5))
    before = [(root / name).read_bytes() for name in ("job.json", "corridor-proposals.json")]
    assert run["reviewed_candidates"] == [] and jobs.inspect_mapping_job(str(root))["attempts"] == []
    run = runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [37]}, "Read partial-width ranges", 0)
    preview = run["observation"][0]["layout_width_preview"]
    assert [(r["from_m"], r["to_m"]) for r in preview["ranges"]] == [(0., 2.), (6., 8.)]
    assert run["remaining_attempts"] == 6 and "layout_width_preview" in run["guidance"]
    assert before == [(root / name).read_bytes() for name in ("job.json", "corridor-proposals.json")]
    saved = json.loads((root / "run.json").read_text())
    assert saved["history"][-1]["observation"][0]["layout_width_preview"] == preview
    assert saved["output"] is None


def test_width_preview_does_not_authorize_uninspected_tail_stations(job_backend, tmp_path):
    source, _ = job_backend
    jobs.core().propose_road_corridors = lambda *args: json.dumps(_proposal(_candidate([3.] * 180)))
    root = tmp_path / "run"
    runs.start_mapping_run(str(source), str(root), _run_layout(width=2.5))
    run = runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [37]}, "Read the bounded preview", 0)
    assert run["observation"][0]["layout_width_preview"]["uninspected_section_pairs"] == 52
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    with pytest.raises(ValueError, match="inspected preview stations"):
        runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [
            {"candidate_id": 37, "action": "include", "from_m": 0., "to_m": 358., "reason": "Unseen tail"}]},
            "A width observation must not authorize uninspected endpoints", run["revision"])
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]
