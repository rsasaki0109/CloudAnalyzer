"""Bounded HTTP byte ranges; never download a full response as a fallback."""

from __future__ import annotations

import re
import threading
import urllib.request

MAX_RANGE_BYTES = 64 << 20


class HttpRangeReader:
    """Read a stable HTTP object, rejecting unsupported or malformed ranges.

    Each response is limited to 64 MiB. A request past EOF may return the
    remaining bytes, which is useful when probing a small file's header.
    ``total_bytes`` and a strong ETag, when exposed, are pinned on first use.
    """

    def __init__(self, url: str, *, timeout: float = 60):
        self.url = url
        self.timeout = timeout
        self.total_bytes: int | None = None
        self.etag: str | None = None
        self.bytes_read = 0
        self.requests = 0
        self._lock = threading.Lock()

    def __call__(self, offset: int, length: int) -> bytes:
        if type(offset) is not int or type(length) is not int or offset < 0 or length < 0:
            raise ValueError("range offset and length must be nonnegative integers")
        if length > MAX_RANGE_BYTES:
            raise ValueError("range exceeds the 64 MiB response limit")
        if not length:
            return b""
        headers = {"Range": f"bytes={offset}-{offset + length - 1}", "Accept-Encoding": "identity"}
        with self._lock:
            if self.etag is not None:
                headers["If-Match"] = self.etag
        request = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - caller URL
            # Check before reading: a 200 may be a terabyte object.
            if response.status != 206:
                raise OSError(f"{self.url}: HTTP byte ranges require 206, received {response.status}")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if match is None:
                raise OSError("missing or invalid Content-Range (expose it for browser clients)")
            start, stop, total = map(int, match.groups())
            if total <= offset or start != offset or stop != min(offset + length, total) - 1:
                raise OSError("Content-Range does not match the requested byte span")
            expected = stop - start + 1
            encoding = response.headers.get("Content-Encoding", "identity").lower()
            if encoding != "identity":
                raise OSError("encoded HTTP range responses are unsupported")
            size = response.headers.get("Content-Length")
            if size is not None and (not size.isdigit() or int(size) != expected):
                raise OSError("Content-Length does not match Content-Range")
            etag = response.headers.get("ETag")
            with self._lock:
                if self.total_bytes is not None and self.total_bytes != total:
                    raise OSError("remote file size changed during reading")
                if self.etag is not None and self.etag != etag:
                    raise OSError("remote file ETag changed during reading")
                self.total_bytes = total
                if etag and not etag.startswith("W/"):
                    self.etag = etag
            data = response.read(expected + 1)
            if len(data) != expected:
                raise OSError("HTTP range body is truncated or exceeds its declared span")
            with self._lock:
                self.bytes_read += len(data)
                self.requests += 1
            return data
