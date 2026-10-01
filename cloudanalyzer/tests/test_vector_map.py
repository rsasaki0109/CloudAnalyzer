"""File publication and CLI use the native draft builder, including failures."""

import json

import pytest
from typer.testing import CliRunner

from ca.vector_map import build_vector_map
from cloudanalyzer_cli.main import app


@pytest.fixture
def survey(tmp_path):
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "build_vector_map"):
        pytest.skip("installed core predates vector map support")
    cloud = tmp_path / "ground.xyz"
    cloud.write_text(
        "".join(
            f"{50000 + x / 5} {50000 + y / 5} 2\n"
            for x in range(201)
            for y in range(-40, 21)
        )
    )
    trajectory = tmp_path / "drive.csv"
    trajectory.write_text(
        "timestamp,x,y,z\n0,50003,50000,50\n1,50020,50000,50\n2,50037,50000,50\n"
    )
    return cloud, trajectory


def test_native_draft_is_published_together_and_never_replaced(survey, tmp_path):
    cloud, trajectory = survey
    out = tmp_path / "draft"
    report = build_vector_map(str(cloud), str(trajectory), str(out))
    assert report["status"] == "draft"
    assert report["extraction"]["lanes"] == 2
    assert report["extraction"]["width_prior_vertices"] > 0
    assert json.loads((out / "report.json").read_text()) == report
    assert set(p.name for p in out.iterdir()) == {
        "lanelet2_map.osm",
        "map_projector_info.yaml",
        "vector_map.json",
        "report.json",
    }
    xml = (out / "lanelet2_map.osm").read_text()
    assert '<tag k="ele" v="2"/>' in xml and '<tag k="ele" v="50"/>' not in xml
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        build_vector_map(str(cloud), str(trajectory), str(out))
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_cli_writes_the_same_report_with_mgrs_and_reference_metadata_only(
    survey, tmp_path
):
    cloud, trajectory = survey
    out = tmp_path / "mgrs"
    result = CliRunner().invoke(
        app,
        [
            "vectormap-build",
            str(cloud),
            str(trajectory),
            "--out",
            str(out),
            "--projection",
            "mgrs",
            "--origin-lat",
            "35.681236",
            "--origin-lon",
            "139.767125",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == json.loads((out / "report.json").read_text())
    assert "projector_type: MGRS" in (out / "map_projector_info.yaml").read_text()
    copied = tmp_path / "copy"
    report = build_vector_map(
        str(cloud),
        str(trajectory),
        str(copied),
        reference_map=str(out / "lanelet2_map.osm"),
    )
    assert report["extraction"]["lanes"] == 2  # No reference geometry is added.
    assert (copied / "map_projector_info.yaml").read_bytes() == (
        out / "map_projector_info.yaml"
    ).read_bytes()


def test_invalid_input_leaves_no_output(survey, tmp_path):
    cloud, trajectory = survey
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        build_vector_map(str(cloud), str(trajectory), str(out), forward_lanes=0)
    assert not out.exists()
    with pytest.raises(ValueError, match="requires origin"):
        build_vector_map(str(cloud), str(trajectory), str(out), projection="mgrs")
    assert not out.exists()
    trajectory.write_text("timestamp,x,y,z\n0,3000,0,50\n1,3037,0,50\n")
    with pytest.raises(ValueError, match="no continuous road surface"):
        build_vector_map(str(cloud), str(trajectory), str(out))
    assert not out.exists()


def test_mgrs_outside_selected_tile_leaves_no_output(survey, tmp_path):
    cloud, trajectory = survey
    rows = [line.split() for line in cloud.read_text().splitlines()]
    cloud.write_text("".join(f"{float(x) + 100000} {y} {z}\n" for x, y, z in rows))
    trajectory.write_text(
        "timestamp,x,y,z\n0,150003,50000,50\n1,150020,50000,50\n2,150037,50000,50\n"
    )
    out = tmp_path / "outside"
    with pytest.raises(ValueError, match="cannot export this coordinate frame"):
        build_vector_map(
            str(cloud),
            str(trajectory),
            str(out),
            projection="mgrs",
            origin_lat=35.681236,
            origin_lon=139.767125,
        )
    assert not out.exists()
