"""File publication and CLI use the native draft builder, including failures."""

import json

import pytest
from typer.testing import CliRunner

from ca.vector_map import build_vector_map, connect_vector_map_junctions, measure_vector_map_signal
from cloudanalyzer_cli.main import app


@pytest.fixture
def junction_survey(tmp_path):
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "connect_vector_map_junctions"):
        pytest.skip("installed core predates junction support")
    cloud = tmp_path / "junction.xyz"
    cloud.write_text("".join(
        f"{50000 + x / 5} {50000 + y / 5} 2\n"
        for x in range(-110, 11) for y in range(-110, 111)
    ))
    paths = [([(-20, 1.75), (-10, 1.75)], [(-20, -1.75), (-10, -1.75)]),
             ([(-1.75, 10), (-1.75, 20)], [(1.75, 10), (1.75, 20)]),
             ([(1.75, -10), (1.75, -20)], [(-1.75, -10), (-1.75, -20)])]
    boundaries = [
        {"id": i * 2 + side + 1, "kind": {"type": "virtual"},
         "geometry": [[50000 + x, 50000 + y, 2] for x, y in path]}
        for i, sides in enumerate(paths) for side, path in enumerate(sides)
    ]
    data = {"format": "vectormap-ir", "version": 1,
            "metadata": {"georeference": {"projection": "mgrs", "origin": {"lat": 35.681236, "lon": 139.767125}}},
            "lanes": [{"id": i + 7, "kind": "driving", "left": 2 * i + 1, "right": 2 * i + 2,
                       "speed_limit": {"kmh": 20}, "attributes": {"survey": "retained"}} for i in range(3)],
            "boundaries": boundaries}
    path = tmp_path / "legs.json"
    path.write_text(json.dumps(data))
    return cloud, path


def test_junction_preview_selection_atomic_publication_and_replay(junction_survey, tmp_path):
    cloud, source = junction_survey
    before = source.read_bytes()
    preview = tmp_path / "preview"
    report = connect_vector_map_junctions(str(cloud), str(source), str(preview), preview_only=True)
    assert report["status"] == "preview"
    candidates = report["junctions"]["candidates"]
    assert {(c["from"], c["to"]) for c in candidates} == {(7, 8), (7, 9)}
    assert all(c["ambiguous"] for c in candidates)
    assert report["junctions"]["added"] == []
    original = json.loads((preview / "vector_map.json").read_text())
    selected = tmp_path / "selected"
    report = connect_vector_map_junctions(str(cloud), str(source), str(selected), lane_pairs=[(7, 9)])
    assert json.loads((selected / "report.json").read_text()) == report
    assert len(report["junctions"]["added"]) == 1
    changed = json.loads((selected / "vector_map.json").read_text())
    for field in ("boundaries", "lanes"):
        retained = {item["id"]: item for item in changed[field]}
        assert all(retained[item["id"]] == item for item in original[field])
    assert changed["metadata"] == original["metadata"]
    assert source.read_bytes() == before
    assert "projector_type: MGRS" in (selected / "map_projector_info.yaml").read_text()
    assert 'k="cloudanalyzer_review_required" v="yes"' in (selected / "lanelet2_map.osm").read_text()
    replay = tmp_path / "replay"
    report = connect_vector_map_junctions(str(cloud), str(selected / "vector_map.json"), str(replay))
    assert report["junctions"]["added"] == []
    assert json.loads((replay / "vector_map.json").read_text()) == changed
    invalid = tmp_path / "invalid-junction"
    with pytest.raises(ValueError, match="not supported"):
        connect_vector_map_junctions(str(cloud), str(source), str(invalid), lane_pairs=[(7, 9), (9, 7)])
    assert not invalid.exists()
    with pytest.raises(FileExistsError):
        connect_vector_map_junctions(str(cloud), str(source), str(selected))
    empty = tmp_path / "empty-selection"
    connect_vector_map_junctions(str(cloud), str(source), str(empty), lane_pairs=[])
    assert json.loads((empty / "vector_map.json").read_text()) == original


def test_junction_cli_selects_branches_and_rejects_malformed_pairs(junction_survey, tmp_path):
    cloud, source = junction_survey
    runner = CliRunner()
    preview = tmp_path / "cli-preview"
    result = runner.invoke(app, ["vectormap-connect", str(cloud), str(source), "--out", str(preview), "--preview"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["status"] == "preview"
    assert len(report["junctions"]["candidates"]) == 2 and report["junctions"]["added"] == []
    automatic = tmp_path / "cli-automatic"
    result = runner.invoke(app, ["vectormap-connect", str(cloud), str(source), "--out", str(automatic)])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["junctions"]["added"]) == 2
    out = tmp_path / "cli-connections"
    result = runner.invoke(app, ["vectormap-connect", str(cloud), str(source), "--out", str(out), "--pair", "7:8", "--pair", "7:9"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report == json.loads((out / "report.json").read_text())
    assert len(report["junctions"]["added"]) == 2
    invalid = tmp_path / "bad-pair"
    result = runner.invoke(app, ["vectormap-connect", str(cloud), str(source), "--out", str(invalid), "--pair", "7:8:9"])
    assert result.exit_code != 0
    assert not invalid.exists()


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
            "--no-track-boundaries",
            "--no-fit-boundaries",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == json.loads((out / "report.json").read_text())
    report = json.loads(result.stdout)
    assert report["options"]["track_boundaries"] is False
    assert report["options"]["fit_boundaries"] is False
    assert report["extraction"]["tracked_vertices"] == 0
    assert report["extraction"]["fitted_vertices"] == 0
    assert "projector_type: MGRS" in (out / "map_projector_info.yaml").read_text()
    copied = tmp_path / "copy"
    report = build_vector_map(
        str(cloud),
        str(trajectory),
        str(copied),
        reference_map=str(out / "lanelet2_map.osm"),
    )
    assert report["extraction"]["lanes"] == 2  # No reference geometry is added.
    assert (copied / "lanelet2_map.osm").read_text().count(
        '<tag k="subtype" v="road"/>'
    ) == 2
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


def test_existing_ir_reuses_lanes_rules_coordinates_and_does_not_touch_inputs(
    survey, tmp_path
):
    cloud, trajectory = survey
    first = tmp_path / "first"
    build_vector_map(
        str(cloud),
        str(trajectory),
        str(first),
        projection="mgrs",
        origin_lat=35.681236,
        origin_lon=139.767125,
    )
    existing = first / "vector_map.json"
    document = json.loads(existing.read_text())
    for lane in document["lanes"]:
        lane["speed_limit"] = {"kmh": 18.0}
    existing.write_text(json.dumps(document))
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    out = tmp_path / "replayed"
    result = CliRunner().invoke(
        app,
        [
            "vectormap-build",
            str(cloud),
            str(trajectory),
            "--out",
            str(out),
            "--existing-map",
            str(existing),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["extraction"]["lanes"] == 0
    assert report["extraction"]["reused_length"] > 30
    assert json.loads((out / "vector_map.json").read_text()) == document
    assert (out / "map_projector_info.yaml").read_bytes() == before[
        "map_projector_info.yaml"
    ]
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before
    assert report["inputs"]["existing_map"] == str(existing.resolve())
    assert report["import_issues"] == []
    # Lanelet2 remains a supported append input; IDs/projector survive a replay.
    third = tmp_path / "osm-replay"
    osm_report = build_vector_map(
        str(cloud),
        str(trajectory),
        str(third),
        existing_map=str(out / "lanelet2_map.osm"),
    )
    assert osm_report["extraction"]["lanes"] == 0
    assert (third / "map_projector_info.yaml").read_bytes() == before[
        "map_projector_info.yaml"
    ]
    conflict = tmp_path / "conflict"
    with pytest.raises(ValueError, match="existing_map retains its coordinates"):
        build_vector_map(
            str(cloud),
            str(trajectory),
            str(conflict),
            existing_map=str(existing),
            projection="mgrs",
        )
    assert not conflict.exists()
    with pytest.raises(ValueError, match="choose it without"):
        import cloudanalyzer_core

        cloudanalyzer_core.build_vector_map(
            str(cloud),
            str(trajectory),
            "{}",
            str(out / "lanelet2_map.osm"),
            None,
            str(existing),
        )
    # An explicitly disabled merge adds lanes but still preserves old IDs/rules.
    off = tmp_path / "off"
    disabled = build_vector_map(
        str(cloud),
        str(trajectory),
        str(off),
        existing_map=str(existing),
        merge_repeated_passes=False,
    )
    assert disabled["extraction"]["lanes"] == 2
    assert disabled["extraction"]["reused_intervals"] == 0


def test_existing_map_failure_is_atomic(survey, tmp_path):
    cloud, trajectory = survey
    first = tmp_path / "first"
    build_vector_map(str(cloud), str(trajectory), str(first))
    existing = first / "vector_map.json"
    before = existing.read_bytes()
    trajectory.write_text("timestamp,x,y,z\n0,3000,0,50\n1,3037,0,50\n")
    out = tmp_path / "failure"
    with pytest.raises(ValueError, match="no continuous road surface"):
        build_vector_map(
            str(cloud), str(trajectory), str(out), existing_map=str(existing)
        )
    assert existing.read_bytes() == before
    assert not out.exists()


def test_signal_measurement_publication_cli_replay_and_sparse_rejection(junction_survey, tmp_path):
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "measure_vector_map_signal"):
        pytest.skip("installed core predates measured signals")
    _, source = junction_survey
    cloud = tmp_path / "head.xyz"
    cloud.write_text("".join(f"{49994.4 + x * .05} 50001 7.{z:02d}\n" for x in range(25) for z in range(0, 51, 5)))
    bounds = [49994.3, 50000.9, 6.9, 49995.7, 50001.1, 7.6]
    preview = tmp_path / "signal-preview"
    report = measure_vector_map_signal(str(cloud), str(source), str(preview), bounds=bounds, lanes=[7])
    assert report["status"] == "preview"
    assert report["signal"]["points"] == 275
    assert report["signal"]["added"] is None
    before = json.loads((preview / "vector_map.json").read_text())
    result = CliRunner().invoke(app, ["vectormap-signal", str(cloud), str(source), "--out", str(tmp_path / "signal-added"),
                                    "--box", ",".join(map(str, bounds)), "--lane", "7", "--add"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    added = tmp_path / "signal-added"
    changed = json.loads((added / "vector_map.json").read_text())
    for key in ("lanes", "boundaries", "metadata"):
        assert changed[key] == before[key]
    assert len(changed["traffic_signals"]) == 1 and not changed.get("stop_lines")
    assert not changed["traffic_signals"][0].get("bulbs")
    assert report["signal"]["classification_source"] == "user_identified_box"
    assert 'k="cloudanalyzer_geometry_source" v="point_cloud_box_fit"' in (added / "lanelet2_map.osm").read_text()
    replay = tmp_path / "signal-replay"
    again = measure_vector_map_signal(str(cloud), str(added / "vector_map.json"), str(replay), bounds=bounds, lanes=[7], preview_only=False)
    assert again["signal"]["reused"] == report["signal"]["added"]
    assert json.loads((replay / "vector_map.json").read_text()) == changed
    cloud.write_text("49995 50001 7\n")
    invalid = tmp_path / "signal-invalid"
    with pytest.raises(ValueError, match="at least 12"):
        measure_vector_map_signal(str(cloud), str(source), str(invalid), bounds=bounds, lanes=[7], preview_only=False)
    assert not invalid.exists()


def test_existing_explicit_centres_are_respected_and_invalid_imports_rejected(
    survey, tmp_path
):
    cloud, trajectory = survey
    first = tmp_path / "explicit"
    build_vector_map(str(cloud), str(trajectory), str(first))
    existing = first / "vector_map.json"
    document = json.loads(existing.read_text())
    boundaries = {b["id"]: b["geometry"] for b in document["boundaries"]}
    for lane in document["lanes"]:
        ref = lane["left"]
        line = boundaries[ref if isinstance(ref, int) else ref["boundary"]]
        lane["centerline"] = [[x, y, z + 3] for x, y, z in line]
        if isinstance(ref, dict) and ref.get("reversed"):
            lane["centerline"].reverse()
    existing.write_text(json.dumps(document))
    report = build_vector_map(
        str(cloud), str(trajectory), str(tmp_path / "new"), existing_map=str(existing)
    )
    assert report["extraction"]["reused_intervals"] == 0
    document["lanes"].append(document["lanes"][0])
    existing.write_text(json.dumps(document))
    before = existing.read_bytes()
    out = tmp_path / "duplicate-id"
    with pytest.raises(ValueError, match="cannot retain existing_map"):
        build_vector_map(
            str(cloud), str(trajectory), str(out), existing_map=str(existing)
        )
    assert not out.exists()
    assert existing.read_bytes() == before
