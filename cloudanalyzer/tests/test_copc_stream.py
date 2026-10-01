"""Bounded full-density contracts; independent public-data checks are documented."""

import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from ca.io import iter_point_chunks

cc = pytest.importorskip("cloudanalyzer_core")
pytestmark = pytest.mark.skipif(not hasattr(cc, "CopcStream"), reason="requires spatial core build")
FIXTURE = Path(__file__).resolve().parents[2] / "web/e2e/fixtures/small.copc.laz"


def canonical_xyz(points):
    return points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]


def test_full_density_box_preserves_raw_records_and_source_identity():
    expected = cc.read_copc(FIXTURE, max_points=100_000)["positions"]
    bounds = (8, 8, -1, 20, 24, 10)
    mask = np.all((expected >= bounds[:3]) & (expected <= bounds[3:]), axis=1)
    with cc.CopcStream(FIXTURE, bounds=bounds, chunk_size=137) as stream:
        assert stream.total_points == 42_000
        assert stream.header.point_count == 42_000
        points, ids = [], []
        for batch in stream:
            assert 0 < len(batch.positions) <= 137
            np.testing.assert_array_equal(batch.positions, np.column_stack((batch.records.x, batch.records.y, batch.records.z)))
            assert batch.records.array.dtype.itemsize == stream.header.point_format.size
            points.append(batch.positions)
            ids.extend((batch.node_offset, int(i)) for i in batch.ordinals)
        np.testing.assert_array_equal(canonical_xyz(np.vstack(points)), canonical_xyz(expected[mask]))
        assert len(ids) == len(set(ids))
        assert stream.ranges.bytes_read < stream.file_size * 2
    assert stream.closed and stream.ranges.handle.closed
    actual = np.vstack(list(iter_point_chunks(str(FIXTURE), bounds=bounds, chunk_size=137)))
    np.testing.assert_array_equal(canonical_xyz(actual), canonical_xyz(expected[mask]))


def test_empty_query_cancel_close_and_concurrent_traversal():
    with cc.CopcStream(FIXTURE, bounds=(1000, 1000, 1000, 1001, 1001, 1001)) as stream:
        assert list(stream) == []
    cancelled = False
    with pytest.raises(cc.CopcCancelled):
        with cc.CopcStream(FIXTURE, chunk_size=31, cancel=lambda: cancelled) as stream:
            batches = iter(stream)
            assert len(next(batches).positions) == 31
            other = stream.nodes()
            with pytest.raises(ValueError, match="one COPC traversal"):
                next(other)
            cancelled = True
            next(batches)
    assert stream.ranges.handle.closed
    batches = cc.iter_copc_batches(FIXTURE)
    next(batches)
    batches.close()


def test_item_budget_and_modified_local_source(tmp_path):
    with cc.CopcStream(FIXTURE, limits=cc.CopcLimits(raw_node_bytes=1024)) as stream:
        with pytest.raises(ValueError, match="byte limit"):
            next(iter(stream))
    path = tmp_path / "copy.copc.laz"
    shutil.copyfile(FIXTURE, path)
    with cc.CopcStream(path) as stream:
        nodes = stream.nodes()
        node = next(nodes)
        pending = stream.query.next_item()
        with pytest.raises(ValueError, match="pending node"):
            stream.query.decode_records(node.offset + 1, b"")
        assert stream.query.next_item() == pending
        with path.open("ab") as file:
            file.write(b"changed")
        with pytest.raises(OSError, match="source changed"):
            next(stream.read_batches(node))
        assert stream.query.next_item() == pending
        nodes.close()


def test_non_index_evlr_preservation_and_metadata_limit(tmp_path):
    data = bytearray(FIXTURE.read_bytes())
    payload = b"original metadata including CRS or extra application fields"
    evlr = struct.pack("<H16sHQ32s", 0, b"test_metadata", 42, len(payload), b"round trip")
    struct.pack_into("<Q", data, 235, len(data))
    struct.pack_into("<I", data, 243, 1)
    path = tmp_path / "metadata.copc.laz"
    path.write_bytes(data + evlr + payload)
    with cc.CopcStream(path) as stream:
        assert len(stream.header.evlrs) == 1
        assert stream.header.evlrs[0].record_data_bytes() == payload
        assert stream.header.evlrs[0].user_id == "test_metadata"
    with pytest.raises(ValueError, match="metadata byte limit"):
        cc.CopcStream(path, limits=cc.CopcLimits(metadata_bytes=int.from_bytes(data[96:100], "little")))
