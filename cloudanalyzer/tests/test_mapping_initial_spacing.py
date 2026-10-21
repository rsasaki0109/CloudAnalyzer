"""Freeze one corridor grid before the original and path-associated proposals."""
import json

import pytest
from typer.testing import CliRunner

from ca import mapping_job as jobs, mapping_run as runs
from cloudanalyzer_cli.main import app
from tests.test_mapping_import import external_pair
from tests.test_mapping_job import _corridor_report, _run_layout, job_backend


def _source(calls, fault=None):
    def propose(cloud, trajectory, encoded):
        options = json.loads(encoded)
        calls.append(options)
        report = _corridor_report()
        report["protocol"] = {"options": options.copy(), "sample_spacing_m": options.get("sample_spacing_m", 2.)}
        if fault == "missing":
            del report["protocol"]["sample_spacing_m"]
        elif fault == "wrong_grid":
            report["protocol"]["sample_spacing_m"] = 2.
        elif fault == "wrong_options":
            report["protocol"]["options"]["sample_spacing_m"] = 2.
        return json.dumps(report)
    return propose


def _start(origin, pair, source, root, spacing):
    if origin == "raw":
        return runs.start_mapping_run(str(source), str(root), _run_layout(), sample_spacing_m=spacing)
    cloud, trajectory, provenance = pair
    return runs.start_mapping_run_from_slam(str(cloud), str(trajectory), str(root), provenance,
                                            _run_layout(), sample_spacing_m=spacing)


@pytest.mark.parametrize("origin", ["raw", "import"])
@pytest.mark.parametrize("spacing", [None, .5, 1., 2.])
def test_initial_grid_is_frozen_and_inherited_by_association_only_refinement(external_pair, job_backend, tmp_path, origin, spacing):
    calls = []
    jobs.core().propose_road_corridors = _source(calls)
    root = tmp_path / "run"
    run = _start(origin, external_pair, job_backend[0], root, spacing)
    assert run["status"] == "needs_agent" and run["remaining_attempts"] == 6
    expected = {"search_radius_m": 8.}
    if spacing is not None:
        expected["sample_spacing_m"] = spacing
    original = jobs._load(root)["corridor_proposal"]
    assert calls == [expected] and original["options"] == expected
    original_bytes = open(original["file"]["path"], "rb").read()
    refined = runs.advance_mapping_run(str(root), {"type": "refine", "association": "trajectory_containing"},
        "Change path association on the already frozen original grid", run["revision"])
    assert refined["refine_result"]["status"] == "ready" and refined["remaining_attempts"] == 6
    assert calls == [expected, {**expected, "association": "trajectory_containing"}]
    assert open(original["file"]["path"], "rb").read() == original_bytes


@pytest.mark.parametrize("origin", ["raw", "import"])
@pytest.mark.parametrize("spacing", [.49, 2.01, True, float("nan"), float("inf"), "1"])
def test_invalid_initial_spacing_is_rejected_before_creating_a_job(external_pair, job_backend, tmp_path, origin, spacing):
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="sample_spacing_m"):
        _start(origin, external_pair, job_backend[0], root, spacing)
    assert not root.exists()


@pytest.mark.parametrize("fault", ["missing", "wrong_grid", "wrong_options"])
def test_explicit_initial_grid_requires_native_acknowledgement(external_pair, job_backend, tmp_path, fault):
    calls = []
    jobs.core().propose_road_corridors = _source(calls, fault)
    root = tmp_path / "unapplied"
    run = _start("import", external_pair, job_backend[0], root, 1.)
    assert run["status"] == "processing_failed" and len(calls) == 1
    job = jobs._load(root)
    assert job["corridor_proposal"]["status"] == "failed"
    assert "did not apply" in job["corridor_proposal"]["error"] and not job["attempts"]


@pytest.mark.parametrize("origin", ["raw", "import"])
def test_cli_can_choose_the_initial_grid(external_pair, job_backend, tmp_path, origin):
    calls = []
    jobs.core().propose_road_corridors = _source(calls)
    layout, provenance = tmp_path / "layout.json", tmp_path / "provenance.json"
    layout.write_text(json.dumps(_run_layout()))
    provenance.write_text(json.dumps(external_pair[2]))
    command = ["mapping-run", str(job_backend[0])] if origin == "raw" else [
        "mapping-import", str(external_pair[0]), "--trajectory", str(external_pair[1]), "--provenance", str(provenance)]
    result = CliRunner().invoke(app, [*command, "--layout", str(layout), "--out", str(tmp_path / "cli"),
                                    "--corridor-sample-spacing", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "needs_agent"
    assert calls == [{"search_radius_m": 8., "sample_spacing_m": 1.}]
