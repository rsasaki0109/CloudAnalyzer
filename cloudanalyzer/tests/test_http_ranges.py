"""Exercise range transport independently of the optional native extension."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

_path = Path(__file__).resolve().parents[2] / "rust/crates/ca-py/python/cloudanalyzer_core/_ranges.py"
_spec = importlib.util.spec_from_file_location("http_ranges_under_test", _path)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
HttpRangeReader = _module.HttpRangeReader


def response(monkeypatch, *, status=206, span="bytes 5-7/8", body=b"abc", headers=None):
    reply = Mock(status=status, headers={"Content-Range": span, **(headers or {})})
    reply.read.return_value = body
    reply.__enter__ = Mock(return_value=reply)
    reply.__exit__ = Mock(return_value=False)
    opener = Mock(return_value=reply)
    monkeypatch.setattr(_module.urllib.request, "urlopen", opener)
    return reply, opener


def test_range_clips_at_eof_and_pins_identity(monkeypatch):
    reply, opener = response(monkeypatch, headers={"ETag": '"v1"', "Content-Length": "3"})
    reader = HttpRangeReader("https://example.test/cloud.copc.laz")
    assert reader(5, 10) == b"abc"
    reply.read.assert_called_once_with(4)
    assert reader.total_bytes == 8 and reader.bytes_read == 3 and reader.requests == 1
    assert reader(5, 3) == b"abc"
    assert opener.call_args.args[0].get_header("If-match") == '"v1"'
    reply.headers["ETag"] = '"v2"'
    reply.read.reset_mock()
    with pytest.raises(OSError, match="ETag changed"):
        reader(5, 3)
    reply.read.assert_not_called()


@pytest.mark.parametrize("kwargs", [
    {"status": 200}, {"span": ""}, {"span": "bytes 4-6/8"},
    {"span": "bytes 5-8/8"}, {"headers": {"Content-Length": "1000000000000"}},
    {"headers": {"Content-Encoding": "gzip"}},
])
def test_bad_headers_never_read_body(monkeypatch, kwargs):
    reply, _ = response(monkeypatch, **kwargs)
    with pytest.raises(OSError):
        HttpRangeReader("https://example.test/cloud")(5, 3)
    reply.read.assert_not_called()


@pytest.mark.parametrize("body", [b"ab", b"abcd"])
def test_bad_body_is_bounded(monkeypatch, body):
    reply, _ = response(monkeypatch, body=body)
    with pytest.raises(OSError, match="body"):
        HttpRangeReader("https://example.test/cloud")(5, 3)
    reply.read.assert_called_once_with(4)


def test_invalid_requests_never_open_a_connection(monkeypatch):
    _, opener = response(monkeypatch)
    reader = HttpRangeReader("https://example.test/cloud")
    for offset, length in [(-1, 3), (0, -1), (0, (64 << 20) + 1), (0.5, 3), (True, 3)]:
        with pytest.raises(ValueError):
            reader(offset, length)
    assert reader(0, 0) == b""
    opener.assert_not_called()
