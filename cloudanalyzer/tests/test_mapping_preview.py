"""Display subsets retain exact rows; full-map audits remain bound to the source."""

import hashlib
import json
import struct
import subprocess
import sys
import shutil
import zipfile
from pathlib import Path

import pytest

from ca import (
    mapping_preview as previews,
    mapping_bundle as bundles,
    mapping_job as jobs,
    mapping_run as runs,
)
from tests.test_mapping_job import job_backend, unused_frame_run
from tests.test_mapping_hd_repair import hd_only_run
from tests.test_mapping_bundle import finished, _mutate


def _canonical(path, count, attributes=False):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {count}\n"
        + "property double x\nproperty double y\nproperty double z\n"
        + (
            "property float intensity\nproperty float correction\n"
            if attributes
            else ""
        )
        + "end_header\n"
    ).encode()
    with path.open("wb") as stream:
        stream.write(header)
        if attributes:
            for i in range(count):
                stream.write(
                    struct.pack(
                        "<dddff",
                        500000.00001 + i * 0.000003,
                        4000000.00002,
                        2.123456789,
                        i / count,
                        -0.1234567,
                    )
                )
        else:
            row = struct.pack("<ddd", 500000.00001, 4000000.00002, 2.123456789)
            for i in range(0, count, 16384):
                stream.write(row * min(16384, count - i))
    return jobs._artifact(path)


def test_streaming_rows_preserve_large_coordinates_and_all_attribute_bytes(tmp_path):
    source = tmp_path / "original.ply"
    descriptor = _canonical(source, 70001, True)
    target = tmp_path / "preview.ply"
    result = previews.write(descriptor, target, 70001, 17773)
    with source.open("rb") as stream:
        _, count, width = previews.header(stream)
        original = stream.read()
    with target.open("rb") as stream:
        _, kept, size = previews.header(stream)
        actual = stream.read()
    assert size == width == 32 and kept == result["preview_count"] == 17501
    assert result["every_nth_record"] == 4
    assert actual == b"".join(
        original[i * width : (i + 1) * width] for i in range(0, count, 4)
    )
    assert jobs._artifact(source) == descriptor
    assert not result["coordinate_or_attribute_quantization"]


def test_source_exceeding_browser_limit_streams_to_a_bounded_display_subset(tmp_path):
    source = tmp_path / "large.ply"
    descriptor = _canonical(source, 3000001)
    assert descriptor["bytes"] > 64 * 1024**2
    target = tmp_path / "preview.ply"
    result = previews.write(descriptor, target, 3000001, 321)
    assert result["preview_count"] <= 321 and target.stat().st_size < 10000
    assert jobs._artifact(source) == descriptor


def test_export_links_original_full_source_and_exact_hd_audits(
    finished, tmp_path, monkeypatch
):
    target = tmp_path / "preview.zip"
    before = (finished / "run.json").read_bytes(), (finished / "job.json").read_bytes()
    output = runs._load(finished)["output"]
    for name in (
        "build_corridor_lanes",
        "audit_vector_map_quality_details",
        "audit_vector_map_ground_consensus_details",
        "propose_road_corridors",
    ):
        monkeypatch.setattr(
            jobs.core(),
            name,
            lambda *args: pytest.fail("preview must not regenerate or reaudit"),
        )
    result = bundles.export_mapping_preview(
        str(finished), str(target), "Synthetic data", 1
    )
    preview = result["preview_pointcloud"]
    assert result["schema"] == bundles.PREVIEW_SCHEMA
    assert (
        preview["source"]
        == result["review"]["artifacts"]["map"]
        == output["artifacts"]["map"]
    )
    assert "map" not in result["roles"] and "preview_map" in result["roles"]
    assert preview["preview_count"] == 1 and not preview["full_point_map_included"]
    with zipfile.ZipFile(target) as archive:
        assert (
            hashlib.sha256(
                archive.read(result["roles"]["hd_source_audits"])
            ).hexdigest()
            == output["artifacts"]["hd_source_audits"]["sha256"]
        )
        assert (
            archive.read(result["roles"]["hd_map"])
            == Path(output["artifacts"]["hd_map"]["path"]).read_bytes()
        )
        assert preview["file"]["path"] == result["roles"]["preview_map"]
    assert before == (
        (finished / "run.json").read_bytes(),
        (finished / "job.json").read_bytes(),
    )
    # Inspector remains one standalone standard-library module, including v2 metadata.
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    shutil.copyfile(bundles.__file__, isolated / "review.py")
    shutil.copyfile(target, isolated / "review.zip")
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
    assert json.loads(process.stdout)["preview_pointcloud"] == preview


def test_a_map_fitting_the_cap_keeps_the_exact_full_map_format(finished, tmp_path):
    result = bundles.export_mapping_preview(
        str(finished), str(tmp_path / "full.zip"), "Synthetic data", 1000000
    )
    assert result["schema"] == bundles.SCHEMA and result["preview_pointcloud"] is None
    assert "map" in result["roles"] and "preview_map" not in result["roles"]


def test_cli_exports_a_preview_and_reports_original_source(finished, tmp_path):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app

    target = tmp_path / "cli-preview.zip"
    result = CliRunner().invoke(
        app,
        [
            "mapping-run-preview",
            str(finished),
            "--out",
            str(target),
            "--attribution",
            "Synthetic data",
            "--max-preview-points",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    state = json.loads(result.output)
    assert state["preview_pointcloud"]["preview_count"] == 1
    assert (
        state["preview_pointcloud"]["source_for_saved_audits"]
        == "original_full_point_map"
    )
    assert bundles.inspect_mapping_bundle(str(target))["review"] == state["review"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_count", 0),
        ("preview_count", 9),
        ("every_nth_record", 0),
        ("source_for_saved_audits", "preview"),
        ("full_point_map_included", True),
        ("coordinate_frame_changed", True),
    ],
)
def test_changed_preview_provenance_is_rejected(finished, tmp_path, field, value):
    target = tmp_path / "preview.zip"
    bundles.export_mapping_preview(str(finished), str(target), "Synthetic data", 1)
    altered = tmp_path / "changed.zip"

    def change(entries):
        manifest = json.loads(entries["manifest.json"])
        manifest["preview_pointcloud"][field] = value
        entries["manifest.json"] = json.dumps(manifest).encode()

    _mutate(target, altered, change)
    with pytest.raises(ValueError, match="preview"):
        bundles.inspect_mapping_bundle(str(altered))


def test_invalid_budget_and_source_count_leave_no_output_or_scratch(finished, tmp_path):
    target = tmp_path / "preview.zip"
    for cap in (0, True, 1000001):
        with pytest.raises(ValueError, match="max_preview_points"):
            bundles.export_mapping_preview(str(finished), str(target), "Synthetic", cap)
    with pytest.raises(ValueError, match="before copying"):
        bundles.export_mapping_preview(str(finished), str(target), "Synthetic", 1, 1024)
    assert not target.exists() and not list(tmp_path.glob(".mapping-preview-*"))


def test_canonical_source_and_full_hash_are_checked(tmp_path):
    source = tmp_path / "source.ply"
    descriptor = _canonical(source, 10)
    with pytest.raises(ValueError, match="count"):
        previews.write(descriptor, tmp_path / "wrong-count.ply", 11, 2)
    bad = {**descriptor, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="hash"):
        previews.write(bad, tmp_path / "wrong-hash.ply", 10, 2)
    source.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    with pytest.raises(ValueError, match="canonical"):
        previews.write(jobs._artifact(source), tmp_path / "ascii.ply", 10, 2)
