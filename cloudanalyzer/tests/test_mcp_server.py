"""`ca mcp`: CloudAnalyzer's tools for AI agents."""

import asyncio
import json
import sys
import os
from pathlib import Path

import pytest

from ca.mcp_server import TOOLS, session_layout


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "drive"
    folder.mkdir()
    rows = [f"1 0 0 {3 * k} 0 1 0 0 0 0 1 0" for k in range(4)]
    (folder / "poses.txt").write_text("\n".join(rows) + "\n")
    for k in range(4):
        (folder / f"{k:06d}.xyz").write_text("0 0 0\n1 0 0\n")
    (tmp_path / "oxts").mkdir()
    return folder


def test_a_session_folder_is_described_without_loading_it(tmp_path):
    layout = session_layout(str(_folder(tmp_path)))
    assert layout["poses_file"] == "poses.txt"
    assert (layout["poses"], layout["scans"], layout["scans_matched"]) == (4, 4, 4)
    assert layout["path_length_m"] == 9.0 and layout["scan_formats"] == [".xyz"]
    assert layout["gravity_candidates"] == [str(tmp_path / "oxts")]


def test_every_tool_is_documented():
    for tool in TOOLS:
        assert tool.__doc__ and len(tool.__doc__.split()) > 5, tool.__name__


def test_the_server_answers_over_stdio(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    folder = _folder(tmp_path)
    server = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from ca.mcp_server import main; main()"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env={"PYTHONPATH": os.environ["PYTHONPATH"]} if "PYTHONPATH" in os.environ else None,
    )

    async def run():
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                names = {t.name for t in tools}
                assert {"export_mapping_run", "inspect_mapping_bundle"} <= names
                export = next(t for t in tools if t.name == "export_mapping_run")
                assert set(export.model_dump(by_alias=True)["inputSchema"]["required"]) == {"finished_job_dir", "bundle_path", "attribution"}
                bundle = next(t for t in tools if t.name == "inspect_mapping_bundle")
                assert bundle.model_dump(by_alias=True)["inputSchema"]["required"] == ["bundle_path"]
                assert {"start_mapping_run", "continue_mapping_run", "inspect_mapping_run", "advance_mapping_run"} <= names
                continuation = next(t for t in tools if t.name == "continue_mapping_run")
                assert set(continuation.model_dump(by_alias=True)["inputSchema"]["required"]) == {"finished_job_dir", "out_dir", "max_attempts", "reason"}
                start_run = next(t for t in tools if t.name == "start_mapping_run")
                assert set(start_run.model_dump(by_alias=True)["inputSchema"]["required"]) == {"source", "out_dir", "layout_hypothesis"}
                advance_run = next(t for t in tools if t.name == "advance_mapping_run")
                assert set(advance_run.model_dump(by_alias=True)["inputSchema"]["required"]) == {"job_dir", "action", "reason", "expected_revision"}
                assert {"start_mapping_job", "inspect_mapping_job", "propose_mapping_corridors", "inspect_mapping_corridors", "diagnose_mapping_candidate", "generate_mapping_candidate", "select_mapping_candidate"} <= names
                corridors = next(t for t in tools if t.name == "propose_mapping_corridors")
                schema = corridors.model_dump(by_alias=True)["inputSchema"]
                assert schema["required"] == ["job_dir"]
                assert schema["properties"]["search_radius_m"]["default"] == 8.0
                inspect_corridors = next(t for t in tools if t.name == "inspect_mapping_corridors")
                schema = inspect_corridors.model_dump(by_alias=True)["inputSchema"]
                geometry = next(t for t in tools if t.name == "generate_mapping_geometry")
                geometry_schema = geometry.model_dump(by_alias=True)["inputSchema"]
                assert set(geometry_schema["required"]) == {"job_dir", "decisions", "reason"}
                assert geometry_schema["properties"]["decisions"]["type"] == "array"
                geometry_inspect = next(t for t in tools if t.name == "inspect_mapping_geometry")
                assert geometry_inspect.model_dump(by_alias=True)["inputSchema"]["properties"]["offset"]["default"] == 0
                lane_draft = next(t for t in tools if t.name == "generate_mapping_corridor_lanes")
                assert set(lane_draft.model_dump(by_alias=True)["inputSchema"]["required"]) == {"job_dir", "geometry_candidate_id", "lane_specs", "boundary_policy", "reason"}
                assert schema["required"] == ["job_dir"]
                assert schema["properties"]["offset"]["default"] == 0
                diagnose = next(t for t in tools if t.name == "diagnose_mapping_candidate")
                assert set(diagnose.model_dump(by_alias=True)["inputSchema"]["required"]) == {"job_dir", "candidate_id"}
                mapping = next(t for t in tools if t.name == "generate_mapping_candidate")
                schema = mapping.model_dump(by_alias=True)["inputSchema"]
                assert set(schema["required"]) == {"job_dir", "road_options", "reason"}
                build = next(t for t in tools if t.name == "build_vector_map")
                properties = build.model_dump(by_alias=True)["inputSchema"]["properties"]
                for field in ("track_boundaries", "fit_boundaries", "verify_curb_profiles"):
                    assert properties[field]["type"] == "boolean"
                    assert properties[field]["default"] is True
                assert properties["fit_source_surface"]["type"] == "boolean"
                assert properties["fit_source_surface"]["default"] is False
                assert properties["local_ground_height"]["type"] == "boolean"
                assert properties["local_ground_height"]["default"] is False
                assert properties["physical_anchors_only"]["type"] == "boolean"
                assert properties["physical_anchors_only"]["default"] is False
                assert properties["align_trace_to_curbs"]["type"] == "boolean"
                assert properties["align_trace_to_curbs"]["default"] is False
                assert properties["infer_lane_edges"]["type"] == "boolean"
                assert properties["infer_lane_edges"]["default"] is False
                assert properties["fit_paint_divider"]["type"] == "boolean"
                assert properties["fit_paint_divider"]["default"] is False
                assert properties["fit_paint_corridor"]["type"] == "boolean"
                assert properties["fit_paint_corridor"]["default"] is False
                assert properties["paint_channel"]["default"] == "rgb"
                assert properties["paint_channel"]["enum"] == ["rgb", "intensity"]
                junction = next(t for t in tools if t.name == "connect_vector_map_junctions")
                properties = junction.model_dump(by_alias=True)["inputSchema"]["properties"]
                assert properties["preview_only"]["type"] == "boolean"
                assert properties["preview_only"]["default"] is False
                assert properties["max_gap"]["default"] == 30.0
                assert properties["check_boundary_support"]["type"] == "boolean"
                assert properties["check_boundary_support"]["default"] is False
                assert "lane_pairs" in properties
                association = next(t for t in tools if t.name == "edit_vector_map_relations")
                properties = association.model_dump(by_alias=True)["inputSchema"]["properties"]
                assert {"rule_id", "lanes", "controlled_crosswalks", "stop_lines", "out_dir"} <= properties.keys()
                proposal = next(t for t in tools if t.name == "propose_vector_map_relations")
                properties = proposal.model_dump(by_alias=True)["inputSchema"]["properties"]
                assert {"rule_id", "candidate_key", "map_snapshot", "out_dir"} <= properties.keys()
                signal = next(t for t in tools if t.name == "measure_vector_map_signal")
                properties = signal.model_dump(by_alias=True)["inputSchema"]["properties"]
                assert properties["preview_only"]["default"] is True
                assert properties["bounds"]["type"] == "array"
                assert properties["lanes"]["type"] == "array"
                tiles = next(t for t in tools if t.name == "tile_copc")
                crossing = next(t for t in tools if t.name == "measure_vector_map_crosswalk")
                properties = crossing.model_dump(by_alias=True)["inputSchema"]["properties"]
                assert properties["preview_only"]["default"] is True
                assert properties["candidate"]["default"] == 0
                assert properties["brightness_fraction"]["default"] == 0.75
                assert "lanes" in properties
                assert "bounds" in properties
                properties = tiles.model_dump(by_alias=True)["inputSchema"]["properties"]
                assert properties["resume"]["default"] is False
                assert properties["chunk_size"]["default"] == 10000
                assert "stop_after_nodes" in properties
                export = next(t for t in tools if t.name == "export_copc_tile")
                assert export.model_dump(by_alias=True)["inputSchema"]["properties"]["include_halo"]["default"] is False
                result = await session.call_tool("session_layout", {"folder": str(folder)})
                return names, result

    names, result = asyncio.run(run())
    assert {"session_layout", "posegraph_fix", "posegraph_compare", "evaluate_map", "build_vector_map"} <= names
    text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    assert json.loads(text)["scans_matched"] == 4
