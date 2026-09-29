"""`ca mcp`: CloudAnalyzer's tools for AI agents."""

import asyncio
import json
import sys
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
    )

    async def run():
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                result = await session.call_tool("session_layout", {"folder": str(folder)})
                return names, result

    names, result = asyncio.run(run())
    assert {"session_layout", "posegraph_fix", "posegraph_compare", "evaluate_map"} <= names
    text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    assert json.loads(text)["scans_matched"] == 4
