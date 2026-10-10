"""Delivered map pairs remain reviewable without their source workspace."""

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ca import mapping_bundle as bundles, mapping_job as jobs, mapping_run as runs
from cloudanalyzer_cli.main import app
from tests.test_mapping_job import job_backend, unused_frame_run, _advance, _patch_pairs
from tests.test_mapping_hd_repair import hd_only_run, child_run, draft


@pytest.fixture
def finished(hd_only_run):
    _advance(hd_only_run, {"type": "finish", "candidate_id": 2})
    return hd_only_run


def _export(root, target):
    return bundles.export_mapping_run(
        str(root), str(target), "Synthetic fixture; test data"
    )


def _mutate(source, target, change):
    with zipfile.ZipFile(source) as archive:
        entries = {i.filename: archive.read(i) for i in archive.infolist()}
    change(entries)
    with zipfile.ZipFile(target, "w") as archive:
        for key, value in entries.items():
            archive.writestr(key, value)


def test_exact_pair_and_full_audits_are_portable_without_native_or_runs(
    finished, tmp_path, monkeypatch
):
    target = tmp_path / "review.zip"
    before = (finished / "run.json").read_bytes(), (finished / "job.json").read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("review export must not regenerate or audit maps")

    for name in (
        "build_corridor_lanes",
        "edit_vector_map_relations",
        "audit_vector_map_quality_details",
        "audit_vector_map_ground_consensus_details",
        "propose_road_corridors",
    ):
        monkeypatch.setattr(jobs.core(), name, forbidden)
    monkeypatch.setattr(jobs, "fix_session", forbidden)
    monkeypatch.setattr(jobs, "odometry", forbidden)
    exported = _export(finished, target)
    assert before == (
        (finished / "run.json").read_bytes(),
        (finished / "job.json").read_bytes(),
    )
    assert exported["integrity_verified"] and not exported["authenticity_verified"]
    assert not exported["resumable_mapping_job"] and not exported["deployment_ready"]
    output = runs._load(finished)["output"]
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for role, artifact in output["artifacts"].items():
            data = archive.read(manifest["roles"][role])
            assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
            assert len(data) == artifact["bytes"]
        assert len(manifest["files"]) < len(manifest["roles"])
        assert json.loads(
            archive.read(manifest["roles"]["hd_source_audits"])
        ) == json.loads(
            Path(output["artifacts"]["hd_source_audits"]["path"]).read_text()
        )
        assert (
            manifest["review"]["layout_hypothesis"]["path"]
            == manifest["roles"]["layout_hypothesis"]
        )
        assert (
            manifest["review"]["decision_history"]
            == manifest["roles"]["decision_history"]
        )
        assert not manifest["raw_logs_included"]
    # Only a copied stdlib module and ZIP are available to the isolated interpreter.
    isolated = tmp_path / "foreign-machine"
    isolated.mkdir()
    shutil.copyfile(bundles.__file__, isolated / "review.py")
    shutil.copyfile(target, isolated / "review.zip")
    hidden = finished.with_name("unavailable-run")
    finished.rename(hidden)
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import runpy,json; m=runpy.run_path('review.py'); print(json.dumps(m['inspect_mapping_bundle']('review.zip')))",
            ],
            cwd=isolated,
            capture_output=True,
            text=True,
            check=True,
        )
        portable = json.loads(process.stdout)
        assert portable["review"] == exported["review"]
        assert portable["integrity_verified"]
    finally:
        hidden.rename(finished)


def test_exports_adopted_child_pair_and_decision_not_baseline(hd_only_run, tmp_path):
    root = hd_only_run
    baseline = jobs._load(root)["attempts"][1]
    child = child_run(root)
    draft(child)
    _advance(
        child, {"type": "inspect_patch", "candidate_id": 2, "gap_ids": [1], "offset": 0}
    )
    _advance(
        child,
        {
            "type": "patch_gaps",
            "candidate_id": 2,
            "gap_ids": [1],
            "pairs": _patch_pairs(child),
        },
    )
    _advance(root, {"type": "compare_retry", "candidate_id": 3})
    _advance(child, {"type": "finish", "candidate_id": 3})
    _advance(root, {"type": "finish_retry", "candidate_id": 3})
    result = _export(root, tmp_path / "adopted.zip")
    assert result["review"]["candidate_id"] == 3
    assert result["review"]["hd_repair_decision"]["adopted"]
    assert (
        result["review"]["artifacts"]["hd_map"]["sha256"]
        != baseline["files"]["map"]["sha256"]
    )
    with zipfile.ZipFile(tmp_path / "adopted.zip") as archive:
        assert (
            json.loads(archive.read(result["roles"]["owner_decision_history"]))[
                "status"
            ]
            == "finished"
        )
        assert (
            result["roles"]["retry_comparison"] == result["roles"]["retry_comparison_3"]
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("bytes", "hash"),
        ("missing", "members differ"),
        ("extra", "members differ"),
        ("path", "unsafe"),
        ("descriptor", "identity"),
        ("role", "roles"),
    ],
)
def test_tampered_packages_are_rejected(finished, tmp_path, mutation, match):
    target = tmp_path / "review.zip"
    result = _export(finished, target)

    def change(entries):
        manifest = json.loads(entries["manifest.json"])
        point = result["roles"]["map"]
        if mutation == "bytes":
            data = entries[point]
            entries[point] = bytes([data[0] ^ 1]) + data[1:]
        elif mutation == "missing":
            del entries[point]
        elif mutation == "extra":
            entries["files/unlisted.txt"] = b"unlisted"
        elif mutation == "path":
            entries["../escape"] = b"outside"
        elif mutation == "descriptor":
            manifest["review"]["artifacts"]["map"]["sha256"] = "0" * 64
        elif mutation == "role":
            manifest["roles"]["map"] = "missing"
        entries["manifest.json"] = json.dumps(manifest).encode()

    altered = tmp_path / "tampered.zip"
    _mutate(target, altered, change)
    with pytest.raises(ValueError, match=match):
        bundles.inspect_mapping_bundle(str(altered))


@pytest.mark.parametrize("kind", [0o120000, 0o010000])
def test_duplicate_and_symlink_members_are_rejected_before_reads(
    finished, tmp_path, monkeypatch, kind
):
    target = tmp_path / "review.zip"
    _export(finished, target)
    with zipfile.ZipFile(target, "a") as archive, pytest.warns(UserWarning):
        archive.writestr("manifest.json", b"{}")
    monkeypatch.setattr(
        zipfile.ZipFile,
        "read",
        lambda *args: pytest.fail("invalid archive must not read data"),
    )
    with pytest.raises(ValueError, match="duplicate"):
        bundles.inspect_mapping_bundle(str(target))
    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as archive:
        info = zipfile.ZipInfo("manifest.json")
        info.external_attr = (kind | 0o777) << 16
        archive.writestr(info, "outside")
    with pytest.raises(ValueError, match="regular"):
        bundles.inspect_mapping_bundle(str(symlink))


def test_limits_prevent_copy_or_inspection_and_existing_output_is_preserved(
    finished, tmp_path, monkeypatch
):
    target = tmp_path / "review.zip"
    result = _export(finished, target)
    previous = target.read_bytes()
    with pytest.raises(FileExistsError):
        _export(finished, target)
    assert target.read_bytes() == previous
    with pytest.raises(ValueError, match="attribution"):
        bundles.export_mapping_run(str(finished), str(tmp_path / "empty.zip"), " ")
    for limit in (True, 0, 4 * 1024**3 + 1):
        with pytest.raises(ValueError, match="max_bundle_bytes"):
            bundles.inspect_mapping_bundle(str(target), limit)
    limited = tmp_path / "limited.zip"
    with pytest.raises(ValueError, match="before copying"):
        bundles.export_mapping_run(str(finished), str(limited), "test", 1024)
    assert not limited.exists() and not list(tmp_path.glob(".mapping-review-*"))
    monkeypatch.setattr(
        zipfile.ZipFile,
        "read",
        lambda *args: pytest.fail("over-budget archive must not read data"),
    )
    with pytest.raises(ValueError, match="limits"):
        bundles.inspect_mapping_bundle(str(target), result["uncompressed_bytes"] - 1)


def test_export_refuses_unfinished_and_modified_exact_pair(hd_only_run, tmp_path):
    target = tmp_path / "review.zip"
    with pytest.raises(ValueError, match="finished"):
        _export(hd_only_run, target)
    _advance(hd_only_run, {"type": "finish", "candidate_id": 2})
    run = runs._load(hd_only_run)
    run["output"]["artifacts"]["map"] = run["output"]["artifacts"]["trajectory"]
    jobs._save(hd_only_run / "run.json", run)
    with pytest.raises(ValueError, match="exact delivered point"):
        _export(hd_only_run, target)
    assert not target.exists()


def test_copy_detects_changed_source_without_publishing_partial_zip(
    finished, tmp_path, monkeypatch
):
    artifact = runs._load(finished)["output"]["artifacts"]["map"]
    original = zipfile.ZipFile.writestr

    def change_after_validation(self, name, *args, **kwargs):
        result = original(self, name, *args, **kwargs)
        if name == "manifest.json":
            path = Path(artifact["path"])
            data = path.read_bytes()
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        return result

    monkeypatch.setattr(zipfile.ZipFile, "writestr", change_after_validation)
    target = tmp_path / "review.zip"
    with pytest.raises(ValueError, match="changed while copying"):
        _export(finished, target)
    assert not target.exists() and not list(tmp_path.glob(".mapping-review-*"))


def test_cli_exports_and_inspects_final_holds(finished, tmp_path):
    target = tmp_path / "review.zip"
    runner = CliRunner()
    exported = runner.invoke(
        app,
        [
            "mapping-run-export",
            str(finished),
            "--out",
            str(target),
            "--attribution",
            "Synthetic test",
        ],
    )
    assert exported.exit_code == 0, exported.output
    inspected = runner.invoke(app, ["mapping-bundle-inspect", str(target)])
    assert inspected.exit_code == 0, inspected.output
    result = json.loads(inspected.stdout)
    assert (
        result["review"]["diagnosis"]["extent"]
        == runs._load(finished)["output"]["diagnosis"]["extent"]
    )
    assert not result["review"]["diagnosis"]["extent"]["passes_requested_extent"]


def test_cli_reports_an_invalid_zip(tmp_path):
    target = tmp_path / "bad.zip"
    target.write_bytes(b"not a ZIP")
    result = CliRunner().invoke(app, ["mapping-bundle-inspect", str(target)])
    assert result.exit_code == 1 and "not a zip file" in result.output.lower()
