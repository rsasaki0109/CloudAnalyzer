"""Owner/halo, durable recovery and streaming export contracts."""

import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ca import copc_tiles as jobs
from cloudanalyzer_cli.main import app

cc = pytest.importorskip("cloudanalyzer_core")
pytestmark = pytest.mark.skipif(not hasattr(cc, "CopcStream"), reason="requires spatial core")
SOURCE = Path(__file__).resolve().parents[2] / "web/e2e/fixtures/small.copc.laz"


def canonical(records):
    return sorted(row.tobytes() for row in records)


def test_halo_owner_counts_identities_and_export(tmp_path):
    laspy = pytest.importorskip("laspy")
    box = [8, 8, -1, 40, 40, 10]
    root = tmp_path / "tiles"
    report = jobs.tile_copc(str(SOURCE), str(root), 16, bounds=box, halo=2, chunk_size=197)
    expected = cc.read_copc(SOURCE, max_points=100_000)["positions"]
    core = expected[np.all((expected >= box[:3]) & (expected <= box[3:]), axis=1)]
    assert report["status"] == "complete"
    assert report["core_points"] == len(core)
    assert report["halo_copies"] > 0
    db = sqlite3.connect(root / "manifest.sqlite3")
    tiles = list(db.execute("SELECT i,j,core_points,points FROM tiles ORDER BY i,j"))
    db.close()
    owned_ids, tile_records = [], []
    for i, j, owned, total in tiles:
        batches = list(jobs.iter_tile_batches(str(root), i, j))
        assert sum(len(b["records"]) for b in batches) == total
        identities = [(b["node_offset"], int(n)) for b in batches for n in b["ordinals"]]
        assert len(identities) == len(set(identities))
        points = np.vstack([b["positions"] for b in batches])
        halo = np.concatenate([b["halo"] for b in batches])
        expected_core = core[np.all(np.floor(core[:, :2] / 16) == [i, j], axis=1)]
        assert np.count_nonzero(halo == 0) == owned == len(expected_core)
        assert np.all(points[:, :2] >= np.array([i, j]) * 16 - 2)
        assert np.all(points[:, :2] <= (np.array([i, j]) + 1) * 16 + 2)
        expanded = np.asarray(box) + [-2, -2, 0, 2, 2, 0]
        expected_tile = expected[np.all((expected >= expanded[:3]) & (expected <= expanded[3:]), axis=1)]
        expected_tile = expected_tile[np.all((expected_tile[:, :2] >= np.array([i, j]) * 16 - 2) &
                                            (expected_tile[:, :2] <= (np.array([i, j]) + 1) * 16 + 2), axis=1)]
        assert len(points) == len(expected_tile)
        for b in batches:
            owned_ids.extend((b["node_offset"], int(n)) for n in b["ordinals"][b["halo"] == 0])
        if (i, j) == (1, 1):
            tile_records = batches
    assert len(owned_ids) == len(set(owned_ids)) == len(core)
    source_records = np.concatenate([b["records"].array[b["halo"] == 0] for b in tile_records])
    output = tmp_path / "tile.las"
    result = jobs.export_copc_tile(str(root), 1, 1, str(output))
    saved = laspy.read(output)
    assert result["points"] == len(source_records)
    assert canonical(saved.points.array) == canonical(source_records)
    assert all(v.user_id != "copc" for v in saved.header.vlrs)
    with pytest.raises(FileExistsError):
        jobs.export_copc_tile(str(root), 1, 1, str(output))
    if any(b.is_available() for b in laspy.LazBackend):
        halo_output = tmp_path / "tile-with-halo.laz"
        jobs.export_copc_tile(str(root), 1, 1, str(halo_output), include_halo=True)
        saved = laspy.read(halo_output)
        assert len(saved.points) == sum(len(b["records"]) for b in tile_records)
        assert np.count_nonzero(saved.ca_halo == 0) == len(source_records)
        expected_ids = sorted((b["node_offset"], int(n), int(h)) for b in tile_records for n, h in zip(b["ordinals"], b["halo"], strict=True))
        assert sorted(zip(saved.ca_source_node_offset, saved.ca_source_ordinal, saved.ca_halo, strict=True)) == expected_ids


def test_pause_resume_skips_decoding_and_refuses_changed_options_or_corruption(tmp_path, monkeypatch):
    root = tmp_path / "resume"
    first = jobs.tile_copc(str(SOURCE), str(root), 16, stop_after_nodes=2)
    assert first["status"] == "paused" and first["nodes"] == 2
    with pytest.raises(ValueError, match="not complete"):
        jobs.export_copc_tile(str(root), 0, 0, str(tmp_path / "partial.las"))
    with pytest.raises(FileExistsError):
        jobs.tile_copc(str(SOURCE), str(root), 16)
    with pytest.raises(ValueError, match="options"):
        jobs.tile_copc(str(SOURCE), str(root), 17, resume=True)
    original = cc.CopcStream.read_batches
    decoded = []
    def record(stream, node):
        decoded.append(node.offset)
        yield from original(stream, node)
    monkeypatch.setattr(cc.CopcStream, "read_batches", record)
    resumed = jobs.tile_copc(str(SOURCE), str(root), 16., resume=True)
    assert resumed["status"] == "complete" and resumed["core_points"] == 42_000
    assert resumed["nodes"] == 21 and len(decoded) == 19
    assert resumed["verified_pack_bytes"] > 0 and resumed["written_pack_bytes"] > 0
    unchanged = jobs.tile_copc(str(SOURCE), str(root), 16, resume=True)
    assert unchanged["committed_nodes_this_call"] == 0
    assert unchanged["written_pack_bytes"] == 0 and unchanged["verified_pack_bytes"] > 0
    assert len(decoded) == 19
    pack = next((root / "nodes").glob("*.pack"))
    with pack.open("r+b") as file:
        file.seek(-1, 2)
        value = file.read(1)
        file.seek(-1, 2)
        file.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="corrupt"):
        jobs.tile_copc(str(SOURCE), str(root), 16, resume=True)


def test_publish_before_commit_failure_and_cancel_recovery(tmp_path, monkeypatch):
    root = tmp_path / "orphan"
    original = jobs._commit_node
    def fail(*args):
        raise OSError("interrupted after pack publication")
    monkeypatch.setattr(jobs, "_commit_node", fail)
    with pytest.raises(OSError, match="publication"):
        jobs.tile_copc(str(SOURCE), str(root), 16)
    assert len(list((root / "nodes").glob("*.pack"))) == 1
    db = sqlite3.connect(root / "manifest.sqlite3")
    assert db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM fragments").fetchone()[0] == 0
    db.close()
    monkeypatch.setattr(jobs, "_commit_node", original)
    result = jobs.tile_copc(str(SOURCE), str(root), 16, resume=True)
    assert result["status"] == "complete" and result["core_points"] == 42_000
    cancelled = False
    def commit_then_cancel(*args):
        nonlocal cancelled
        original(*args)
        cancelled = True
    monkeypatch.setattr(jobs, "_commit_node", commit_then_cancel)
    cancelled_root = tmp_path / "cancel"
    result = jobs.tile_copc(str(SOURCE), str(cancelled_root), 16, cancel=lambda: cancelled)
    assert result["status"] == "cancelled" and result["nodes"] == 1
    monkeypatch.setattr(jobs, "_commit_node", original)
    result = jobs.tile_copc(str(SOURCE), str(cancelled_root), 16, resume=True)
    assert result["core_points"] == 42_000 and result["nodes"] == 21


def test_cli_and_mcp_bounded_calls(tmp_path):
    from ca.mcp_server import tile_copc, export_copc_tile
    root = tmp_path / "cli"
    runner = CliRunner()
    result = runner.invoke(app, ["copc-tile", str(SOURCE), "--out", str(root), "--grid", "16", "--stop-after-nodes", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "paused"
    result = tile_copc(str(SOURCE), str(root), 16, resume=True)
    assert result["status"] == "complete" and result["core_points"] == 42_000
    saved = export_copc_tile(str(root), 0, 0, str(tmp_path / "mcp.las"))
    assert saved["points"] > 0


def test_quota_prefix_recovery_and_exclusive_writer_lock(tmp_path):
    root = tmp_path / "quota"
    with pytest.raises(ValueError, match="output/fragment limit"):
        jobs.tile_copc(str(SOURCE), str(root), 16, max_node_output_bytes=100)
    db = sqlite3.connect(root / "manifest.sqlite3")
    assert db.execute("SELECT COUNT(*) FROM fragments").fetchone()[0] == 0
    db.close()
    assert not list((root / "nodes").glob("*.part"))
    lock = jobs._lock_job(root)
    try:
        with pytest.raises(OSError):
            jobs.tile_copc(str(SOURCE), str(root), 16, resume=True)
    finally:
        lock.close()
    result = jobs.tile_copc(str(SOURCE), str(root), 16, resume=True)
    assert result["core_points"] == 42_000 and result["status"] == "complete"


def test_evlr_export_and_full_source_count_integrity(tmp_path):
    laspy = pytest.importorskip("laspy")
    data = bytearray(SOURCE.read_bytes())
    payload = b"original application metadata"
    evlr = struct.pack("<H16sHQ32s", 0, b"test_metadata", 42, len(payload), b"preserve")
    struct.pack_into("<Q", data, 235, len(data))
    struct.pack_into("<I", data, 243, 1)
    source = tmp_path / "metadata.copc.laz"
    source.write_bytes(data + evlr + payload)
    root = tmp_path / "metadata"
    jobs.tile_copc(str(source), str(root), 100)
    assert (root / "source-metadata-evlrs.bin").read_bytes() == evlr + payload
    output = tmp_path / "metadata.las"
    jobs.export_copc_tile(str(root), 0, 0, str(output))
    assert laspy.read(output).header.evlrs[0].record_data_bytes() == payload
    # Metadata-only oversized count is not treated as processed points.
    struct.pack_into("<Q", data, 247, 10_000_000_000)
    bad = tmp_path / "bad-count.copc.laz"
    bad.write_bytes(data + evlr + payload)
    with pytest.raises(ValueError, match="source point count"):
        jobs.tile_copc(str(bad), str(tmp_path / "bad-count"), 100)
