"""File publication and CLI use the native draft builder, including failures."""

import json
import asyncio
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from typer.testing import CliRunner
from cloudanalyzer_cli.main import app

from ca.vector_map import build_vector_map, connect_vector_map_junctions, measure_vector_map_signal, measure_vector_map_crosswalk, discover_vector_map_features


def test_source_footprint_cli_keeps_explicit_lanes_and_failed_build_publishes_nothing(tmp_path):
    import numpy as np
    import laspy
    native = pytest.importorskip("cloudanalyzer_core")
    x, y = np.meshgrid(np.arange(0,20.01,.1),np.arange(-4.5,2.01,.1))
    header=laspy.LasHeader(point_format=3,version="1.2");header.scales=[.001]*3
    data=laspy.LasData(header);data.x=x.ravel();data.y=y.ravel();data.z=np.where(y.ravel()>.2,4.,2.)
    cloud=tmp_path/"narrow.las";data.write(cloud)
    poses=tmp_path/"drive.csv";poses.write_text("timestamp,x,y,z\n0,0,0,50\n1,20,0,50\n")
    output=tmp_path/"fitted"
    result=CliRunner().invoke(app,["vectormap-build",str(cloud),str(poses),"--out",str(output),"--fit-source-surface"])
    assert result.exit_code==0,result.output
    report=json.loads(result.stdout);fit=report["extraction"]["surface_fit"]
    assert fit["maximum_lane_width_m"]<3.5 and report["extraction"]["lanes"]==2
    quality=json.loads(native.audit_vector_map_quality(str(cloud),report["files"]["editable_map"]))
    assert not quality["quality"]["low_support_lanes"]
    original=cloud.read_bytes()
    rejected=tmp_path/"unsupported"
    result=CliRunner().invoke(app,["vectormap-build",str(cloud),str(poses),"--out",str(rejected),"--fit-source-surface","--forward-lanes","8","--backward-lanes","0"])
    assert result.exit_code!=0 and not rejected.exists()
    assert cloud.read_bytes()==original


def test_native_source_quality_distinguishes_edges_and_preserves_files(tmp_path):
    import numpy as np
    import laspy
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "audit_vector_map_quality"):
        pytest.skip("installed core predates source quality audit")
    x, y = np.meshgrid(np.arange(-1, 11.01, .2), np.arange(-.2, .201, .2))
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [.001] * 3
    data = laspy.LasData(header)
    data.x, data.y, data.z = x.ravel(), y.ravel(), np.full(x.size, 2.)
    cloud = tmp_path / "source.las"
    data.write(cloud)
    vector_map = tmp_path / "map.json"
    vector_map.write_text(json.dumps({"format": "vectormap-ir", "version": 1,
        "lanes": [{"id": 3, "kind": "driving", "left": 1, "right": 2}],
        "boundaries": [{"id": i+1, "kind": {"type": "lane_marking", "pattern": "solid"},
                        "geometry": [[0, offset, 2], [10, offset, 2]]} for i, offset in enumerate([2, -2])]}), encoding="utf-8")
    original = [p.read_bytes() for p in (cloud, vector_map)]
    result = json.loads(native.audit_vector_map_quality(str(cloud), str(vector_map)))
    q = result["quality"]
    assert q["low_support_lanes"] == [3] and not q["limited"]
    assert q["lanes"][0]["center"]["fraction"] == 1
    assert q["lanes"][0]["left"]["fraction"] == 0
    assert q["lanes"][0]["left"]["insufficient_returns"] > 0
    assert [p.read_bytes() for p in (cloud, vector_map)] == original


def test_source_only_equipment_preview_confirm_cli_replay_and_failed_publication(tmp_path):
    import numpy as np
    import laspy
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "discover_vector_map_features"):
        pytest.skip("installed core predates automatic equipment discovery")
    x, y = np.meshgrid(np.arange(-5, 35.001, .1), np.arange(-5, 5.001, .1))
    x, y = x.ravel(), y.ravel()
    bright = (((x >= 8) & (x < 12) & (((x - 8) % 1) < .5)) | ((x >= 20) & (x < 20.6))) & (abs(y) <= 3)
    head = np.array([[25., -.6 + i * .05, 6 + j * .05] for i in range(25) for j in range(13)])
    points = np.vstack([np.column_stack([x, y, np.full(len(x), 2.)]), head])
    colors = np.r_[np.where(bright, 220, 60), np.full(len(head), 80)].astype(np.uint16) * 256
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.full(3, .001)
    data = laspy.LasData(header)
    data.x, data.y, data.z = points.T
    data.red = data.green = data.blue = colors
    cloud = tmp_path / "source.las"
    data.write(cloud)
    poses = tmp_path / "drive.csv"
    poses.write_text("timestamp,x,y,z\n0,-5,0,4\n1,35,0,4\n")
    built = build_vector_map(str(cloud), str(poses), str(tmp_path / "roads"))
    source = Path(built["files"]["editable_map"])
    preview = discover_vector_map_features(str(cloud), str(tmp_path / "preview"), vector_map=str(source))
    before = source.read_text()
    assert Path(preview["files"]["editable_map"]).read_text() == before
    candidates = preview["discovery"]["candidates"]
    assert {c["evidence"]["kind"] for c in candidates} == {"repeated_paint", "bright_bar", "elevated_panel"}
    lanes = [json.loads(before)["lanes"][0]["id"]]
    classifications = {"repeated_paint": "crosswalk", "bright_bar": "stop_line", "elevated_panel": "vehicle_signal"}
    confirmations = [{"candidate": c["id"], "key": c["key"], "classification": classifications[c["evidence"]["kind"]], "lanes": lanes} for c in candidates]
    selection = tmp_path / "reviewed.json"
    selection.write_text(json.dumps(confirmations))
    result = CliRunner().invoke(app, ["vectormap-discover", str(cloud), "--map", str(source), "--out", str(tmp_path / "added"), "--confirmations", str(selection)])
    assert result.exit_code == 0, result.output
    added = json.loads(result.stdout)
    changed = json.loads(Path(added["files"]["editable_map"]).read_text())
    assert changed["lanes"] == json.loads(before)["lanes"]
    assert len(changed["traffic_signals"]) == 1 and len(changed["crosswalks"]) >= 1 and len(changed["stop_lines"]) >= 1
    assert all(not s.get("bulbs") for s in changed["traffic_signals"])
    assert "user_confirmed_automatic_proposal" in Path(added["files"]["map"]).read_text()
    replay = discover_vector_map_features(str(cloud), str(tmp_path / "replay"), vector_map=added["files"]["map"], confirmations=confirmations)
    assert all(a["reused"] for a in replay["additions"])
    assert source.read_text() == before
    bad = [{**confirmations[0], "key": "stale"}]
    with pytest.raises(ValueError, match="changed"):
        discover_vector_map_features(str(cloud), str(tmp_path / "invalid"), vector_map=str(source), confirmations=bad)
    assert not (tmp_path / "invalid").exists()
    full = discover_vector_map_features(str(cloud), str(tmp_path / "without-map"), scope="ground_surface")
    assert full["status"] == "preview"
    assert full["discovery"]["candidates"] and all(not c["nearby_lanes"] for c in full["discovery"]["candidates"])
    with pytest.raises(FileExistsError):
        discover_vector_map_features(str(cloud), str(tmp_path / "added"), vector_map=str(source))



def test_crosswalk_rgb_preview_cli_add_replay_and_atomic_publication(junction_survey, tmp_path):
    import numpy as np
    import laspy
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "measure_vector_map_crosswalk"):
        pytest.skip("installed core predates paint measurement")
    _, source = junction_survey
    s, t = np.meshgrid(np.arange(-5, 5.001, .05), np.arange(-4, 4.001, .05))
    s, t = s.ravel(), t.ravel()
    a = np.deg2rad(28)
    x, y = s * np.cos(a) - t * np.sin(a), s * np.sin(a) + t * np.cos(a)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.full(3, .001)
    data = laspy.LasData(header)
    data.x, data.y, data.z = 49985 + x, 50000 + y, 2 + .03 * x - .02 * y
    white = (s >= -2) & (s < 2) & (((s + 2) % 1) < .5) & (np.abs(t) <= 3)
    data.red = data.green = data.blue = np.where(white, 210, 70).astype(np.uint16) * 256
    cloud = tmp_path / "paint.las"
    data.write(cloud)
    bounds = [49978, 49993, 1.5, 49992, 50007, 2.5]
    preview = tmp_path / "preview"
    report = measure_vector_map_crosswalk(str(cloud), str(source), str(preview), bounds=bounds)
    candidate = report["crosswalk"]["candidates"][0]
    assert candidate["stripe_count"] == 4
    assert abs(candidate["angle_degrees"] - 28) <= 2
    before = json.loads((preview / "vector_map.json").read_text())
    result = CliRunner().invoke(app, ["vectormap-crosswalk", str(cloud), str(source), "--out", str(tmp_path / "added"),
                                    "--box", ",".join(map(str, bounds)), "--lane", "7", "--candidate", "0", "--add"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    added = tmp_path / "added"
    changed = json.loads((added / "vector_map.json").read_text())
    for key in ("lanes", "boundaries", "metadata"):
        assert changed[key] == before[key]
    assert len(changed["crosswalks"]) == 1 and not changed.get("stop_lines")
    assert "point_cloud_brightness_stripes" in (added / "lanelet2_map.osm").read_text()
    assert "cloudanalyzer_lanes_source" in (added / "lanelet2_map.osm").read_text()
    assert "cloudanalyzer_paint_bands" in (added / "lanelet2_map.osm").read_text()
    replay = tmp_path / "replay"
    again = measure_vector_map_crosswalk(str(cloud), str(added / "lanelet2_map.osm"), str(replay),
        bounds=bounds, lanes=[7], preview_only=False)
    assert again["crosswalk"]["reused"] == report["crosswalk"]["added"]
    assert "cloudanalyzer_paint_bands" in Path(again["files"]["map"]).read_text()
    # Reject invalid confirmation without creating even a partial artifact dir.
    for options in ({"lanes": []}, {"lanes": [7], "candidate": 99}, {"lanes": [7, 7]}, {"lanes": [7], "brightness_fraction": float("nan")}, {"lanes": [7], "brightness_fraction": 0.99}):
        with pytest.raises(ValueError):
            measure_vector_map_crosswalk(str(cloud), str(source), str(tmp_path / "invalid"),
                bounds=bounds, preview_only=False, **options)
        assert not (tmp_path / "invalid").exists()
    with pytest.raises(FileExistsError):
        measure_vector_map_crosswalk(str(cloud), str(source), str(added), bounds=bounds)
    assert json.loads((added / "vector_map.json").read_text()) == changed


def test_signal_las_spatial_reader_matches_native_file_and_cli(junction_survey, tmp_path):
    import laspy
    import numpy as np
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "measure_vector_map_signal_points"):
        pytest.skip("installed core predates spatial signals")
    _, source = junction_survey
    bounds = [49994.3, 50000.9, 6.9, 49995.7, 50001.1, 7.6]
    points = np.array([[49994.4 + x * .05, 50001, 7 + z * .01] for x in range(25) for z in range(0, 51, 5)])
    xyz = np.vstack([np.tile([50020., 50020., 0.], (30000, 1)), points])
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.full(3, .001)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz.T
    cloud = tmp_path / "head.las"
    las.write(cloud)
    options = json.dumps({"min": bounds[:3], "max": bounds[3:], "lanes": [7], "kind": "vehicle"})
    expected = json.loads(native.measure_vector_map_signal(str(cloud), str(source), options))["report"]["signal"]
    report = measure_vector_map_signal(str(cloud), str(source), str(tmp_path / "preview"), bounds=bounds, lanes=[7])
    assert report["signal"] == expected
    assert report["processing"] == {"strategy": "sequential-filtered-chunks", "selected_points": 275,
                                    "selected_limit": 200000, "chunk_points": 10000}
    result = CliRunner().invoke(app, ["vectormap-signal", str(cloud), str(source), "--out", str(tmp_path / "cli"),
                                    "--box", ",".join(map(str, bounds)), "--lane", "7"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["signal"] == expected
    with pytest.raises(ValueError, match="shape"):
        native.measure_vector_map_signal_points(np.zeros((4, 2)), str(source), options)
    with pytest.raises(ValueError, match="200000"):
        native.measure_vector_map_signal_points(np.zeros((200001, 3)), str(source), options)
    with pytest.raises(ValueError, match="finite"):
        native.measure_vector_map_signal_points(np.array([[0., 0., np.nan]]), str(source), options)
    # Strided arrays are copied correctly; no contiguous-only requirement.
    doubled = np.repeat(points, 2, axis=0)
    result = json.loads(native.measure_vector_map_signal_points(doubled[::2], str(source), options))
    assert result["report"]["signal"]["points"] == 275


def test_signal_copc_local_and_http_all_levels(junction_survey, tmp_path):
    import laspy
    import numpy as np
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "measure_vector_map_signal_points"):
        pytest.skip("installed core predates spatial signals")
    _, source = junction_survey
    fixture = Path(__file__).parent / "data" / "signal.copc.laz"
    data = fixture.read_bytes()
    bounds = [49994.3, 50000.9, 6.9, 49995.7, 50001.1, 7.6]
    # Independent laspy/lazrs full-source reader; filter only after reading all nodes.
    # The minimal test writer omits the ordinary LAZ chunk table.
    with laspy.CopcReader.open(str(fixture)) as reader:
        all_points = reader.query()
    xyz = np.column_stack([all_points.x, all_points.y, all_points.z])
    selected = xyz[np.all((xyz >= bounds[:3]) & (xyz <= bounds[3:]), axis=1)]
    assert len(xyz) == 5275 and len(selected) == 275
    options = json.dumps({"min": bounds[:3], "max": bounds[3:], "lanes": [7], "kind": "vehicle"})
    expected = json.loads(native.measure_vector_map_signal_points(selected, str(source), options))["report"]["signal"]
    local = measure_vector_map_signal(str(fixture), str(source), str(tmp_path / "local"), bounds=bounds, lanes=[7])
    assert local["signal"] == expected
    assert local["processing"]["strategy"] == "copc-full-density-box"
    ranges = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            span = self.headers["Range"].removeprefix("bytes=").split("-")
            a, b = int(span[0]), min(int(span[1]), len(data) - 1)
            ranges.append((a, b))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            self.send_header("Content-Length", str(b - a + 1))
            self.send_header("ETag", '"fixture-v1"')
            self.end_headers()
            self.wfile.write(data[a:b + 1])
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/signal.copc.laz?token=secret#fragment"
        remote = measure_vector_map_signal(url, str(source), str(tmp_path / "http"), bounds=bounds, lanes=[7])
        assert remote["signal"] == expected
        assert remote["processing"]["selected_points"] == 275
        assert "secret" not in remote["inputs"]["cloud"] and "fragment" not in remote["inputs"]["cloud"]
        assert len(ranges) > 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_signal_spatial_caps_close_reader_and_invalid_box_never_reads(junction_survey, tmp_path, monkeypatch):
    import numpy as np
    import ca.io
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "measure_vector_map_signal_points"):
        pytest.skip("installed core predates spatial signals")
    _, source = junction_survey
    cloud = tmp_path / "source.las"
    cloud.touch()
    closed = []
    calls = []
    def chunks(*args, **kwargs):
        calls.append((args, kwargs))
        try:
            for _ in range(21): yield np.tile([49995., 50001., 7.25], (10000, 1))
            pytest.fail("reader continued after point limit")
        finally: closed.append(True)
    monkeypatch.setattr(ca.io, "iter_point_chunks", chunks)
    bounds = [49994.3, 50000.9, 6.9, 49995.7, 50001.1, 7.6]
    out = tmp_path / "overflow"
    with pytest.raises(ValueError, match="200000"):
        measure_vector_map_signal(str(cloud), str(source), str(out), bounds=bounds, lanes=[7])
    assert closed == [True] and not out.exists()
    assert calls[0][1] == {"chunk_size": 10000, "bounds": tuple(bounds)}
    calls.clear()
    for bad in [[0, 0, 0, 11, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, float("nan"), 1, 1]]:
        with pytest.raises(ValueError, match="bounds"):
            measure_vector_map_signal(str(cloud), str(source), str(out), bounds=bad, lanes=[7])
    with pytest.raises(ValueError, match="lane"):
        measure_vector_map_signal(str(cloud), str(source), str(out), bounds=bounds, lanes=[7, 7])
    assert calls == [] and not out.exists()


def test_signal_mcp_spatial_copc_publishes_preview(junction_survey, tmp_path):
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "measure_vector_map_signal_points"):
        pytest.skip("installed core predates spatial signals")
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _, source = junction_survey
    fixture = Path(__file__).parent / "data" / "signal.copc.laz"
    server = StdioServerParameters(command=sys.executable,
        args=["-c", "from ca.mcp_server import main; main()"], cwd=str(Path(__file__).resolve().parents[1]))
    async def run():
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("measure_vector_map_signal", {
                    "cloud": str(fixture), "vector_map": str(source), "out_dir": str(tmp_path / "mcp-preview"),
                    "bounds": [49994.3, 50000.9, 6.9, 49995.7, 50001.1, 7.6], "lanes": [7]})
    result = asyncio.run(run())
    assert not getattr(result, "is_error", getattr(result, "isError", False))
    text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    report = json.loads(text)
    assert report["status"] == "preview" and report["signal"]["points"] == 275
    assert report["processing"]["strategy"] == "copc-full-density-box"
    assert Path(report["files"]["editable_map"]).is_file()

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


def test_tall_roadside_returns_are_rejected_through_the_public_builder(survey, tmp_path):
    cloud, trajectory = survey
    # Roadside vehicle/wall returns produce positive steps but not curb profiles.
    cloud.write_text("".join(
        f"{50000 + x / 5} {50000 + y / 5} {2 if -5.25 <= y / 5 <= 1.75 else 3.5}\n"
        for x in range(201) for y in range(-40, 21)
    ))
    legacy = build_vector_map(str(cloud), str(trajectory), str(tmp_path / "unchecked"),
                              verify_curb_profiles=False)
    guarded = build_vector_map(str(cloud), str(trajectory), str(tmp_path / "checked"))
    assert legacy["extraction"]["curb_vertices"] > 0
    assert guarded["options"]["verify_curb_profiles"] is True
    assert guarded["extraction"]["curb_vertices"] == 0
    assert guarded["extraction"]["rejected_curb_candidates"] > 0
    assert guarded["extraction"]["width_prior_vertices"] > 0
    assert any("height transitions" in warning for warning in guarded["extraction"]["warnings"])


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
            "--no-verify-curb-profiles",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == json.loads((out / "report.json").read_text())
    report = json.loads(result.stdout)
    assert report["options"]["track_boundaries"] is False
    assert report["options"]["fit_boundaries"] is False
    assert report["options"]["verify_curb_profiles"] is False
    assert report["extraction"]["rejected_curb_candidates"] == 0
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
