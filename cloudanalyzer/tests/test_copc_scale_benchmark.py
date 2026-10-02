"""Exercise the actual benchmark subprocesses, oracle caps and crash recovery."""

import io
import json
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

cc = pytest.importorskip("cloudanalyzer_core")
laspy = pytest.importorskip("laspy")
lazrs = pytest.importorskip("lazrs")
pytest.importorskip("psutil")
pytestmark = pytest.mark.skipif(not hasattr(cc, "CopcStream"), reason="requires spatial core")
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/benchmark_copc_scale.py"


def complete_fixture(path):
    # The synthetic Web fixture has independently compressed nodes but omits
    # the ordinary LAZ chunk-table pointer/table. Add them to a temporary copy
    # so the independent sequential laspy oracle can read the same records.
    original = bytearray((ROOT / "web/e2e/fixtures/small.copc.laz").read_bytes())
    header = laspy.LasHeader.read_from(io.BytesIO(original))
    info = header.vlrs[0]
    data_offset = header.offset_to_point_data
    raw = original[:data_offset] + bytearray(8) + original[data_offset:]
    struct.pack_into("<Q", raw, 375 + 54 + 40, info.hierarchy_root_offset + 8)
    chunks = []
    for pos in range(info.hierarchy_root_offset + 8,
                     info.hierarchy_root_offset + 8 + info.hierarchy_root_size, 32):
        _, _, _, _, offset, size, count = struct.unpack_from("<iiiiQii", raw, pos)
        assert count > 0
        struct.pack_into("<Q", raw, pos + 16, offset + 8)
        chunks.append((offset, count, size))
    output = io.BytesIO(raw)
    output.seek(0, 2)
    table_offset = output.tell()
    lazrs.write_chunk_table(output, [(count, size) for _, count, size in sorted(chunks)],
                           lazrs.LazVlr(header.vlrs[1].record_data))
    output.seek(data_offset)
    output.write(struct.pack("<q", table_offset))
    path.write_bytes(output.getvalue())


def command(source, output):
    return [sys.executable, str(SCRIPT), str(source), "--bounds", "8", "8", "-1", "20", "24", "10",
            "--grid-size", "16", "--halo", "2", "--output", str(output)]


def test_real_subprocess_oracle_full_tiles_and_interruption(tmp_path):
    source, output = tmp_path / "synthetic.copc.laz", tmp_path / "report.json"
    complete_fixture(source)
    data = source.read_bytes()
    requests = []
    class Ranges(BaseHTTPRequestHandler):
        def do_GET(self):
            span = self.headers["Range"].removeprefix("bytes=")
            start, stop = map(int, span.split("-"))
            stop = min(stop, len(data) - 1)
            requests.append((start, stop, self.headers.get("If-Match")))
            if self.headers.get("If-Match") not in {None, '"synthetic"'}:
                self.send_error(412)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{stop}/{len(data)}")
            self.send_header("Content-Length", str(stop - start + 1))
            self.send_header("ETag", '"synthetic"')
            self.end_headers()
            self.wfile.write(data[start:stop + 1])

        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Ranges)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/source.copc.laz?token=private-test-value"
    try:
        result = subprocess.run(command(source, output) + ["--full-tiles", "--http-source", url],
                                capture_output=True, text=True, timeout=120)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    runs = {run["mode"]: run for run in report["runs"]}
    assert runs["oracle"]["total_points"] == runs["stream-full"]["selected_points"] == 42_000
    assert runs["stream-box"]["selected_points"] == runs["tile-verify"]["oracle_core_points"] == 1_965
    assert runs["stream-http-box"]["selected_points"] == 1_965
    assert requests and any(etag == '"synthetic"' for _, _, etag in requests)
    assert "private-test-value" not in output.read_text()
    assert runs["tile-verify"]["raw_record_multiset_equal"]
    assert runs["tile-verify"]["xyz64_equal"]
    assert runs["tile-verify"]["export_scales_offsets_vlrs_equal"]
    assert runs["tile-verify"]["core_export_points"] == 1_965
    assert runs["tile-interrupt"]["published_uncommitted_pack"]
    assert runs["tile-full-resume"]["core_points"] == 42_000
    assert runs["tile-full-complete"]["committed_nodes_this_call"] == 0
    assert runs["tile-full-complete"]["verified_pack_bytes"] == runs["tile-full-resume"]["pack_bytes"]
    assert all(run["peak_rss_bytes"] > 0 for run in runs.values())
    original = output.read_bytes()
    repeat = subprocess.run(command(source, output), capture_output=True, text=True, timeout=20)
    assert repeat.returncode != 0 and "already exist" in repeat.stderr
    assert output.read_bytes() == original


def test_oracle_cap_stops_before_stream_or_tile_processing(tmp_path):
    source, output = tmp_path / "synthetic.copc.laz", tmp_path / "capped.json"
    complete_fixture(source)
    result = subprocess.run(command(source, output) + ["--max-oracle-points", "100"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["runs"] == []
    artifacts = output.with_suffix(".artifacts")
    assert "exceeds --max-oracle-points" in (artifacts / "oracle.stderr").read_text()
    assert not (artifacts / "tiles-box").exists()
