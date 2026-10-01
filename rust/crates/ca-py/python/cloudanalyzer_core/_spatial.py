"""Full-density COPC batches, with bounded IO, decoding and metadata."""

from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np

from ._core import CopcReader, CopcSpatialQuery
from ._ranges import MAX_RANGE_BYTES, HttpRangeReader

Bounds = tuple[float, float, float, float, float, float]


class CopcCancelled(OSError):
    """Reading stopped at an IO/node/batch boundary by the caller."""


@dataclass(frozen=True)
class CopcLimits:
    """Per-item limits, not an RSS cap or a limit on caller-retained output."""

    page_bytes: int = 1 << 20
    compressed_node_bytes: int = 16 << 20
    raw_node_bytes: int = 32 << 20
    pending_entries: int = 16_384
    page_depth: int = 32
    metadata_bytes: int = 8 << 20

    def __post_init__(self) -> None:
        for name, maximum in (
            ("page_bytes", MAX_RANGE_BYTES), ("compressed_node_bytes", MAX_RANGE_BYTES),
            ("raw_node_bytes", MAX_RANGE_BYTES), ("pending_entries", 65_536),
            ("page_depth", 64), ("metadata_bytes", 8 << 20),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")


@dataclass(frozen=True)
class CopcNode:
    offset: int
    size: int
    count: int


@dataclass(frozen=True)
class CopcBatch:
    """Unchanged LAS records, XYZ64, and their identity in the source node.

    ``records`` is a laspy ScaleAwarePointRecord. ``ordinals`` are original
    record indices, preserving duplicate point multiplicity across batches.
    Holding returned batches increases caller memory; consume and discard.
    """

    records: Any
    positions: np.ndarray
    node_offset: int
    ordinals: np.ndarray


class _LocalRanges:
    def __init__(self, source: str):
        self.handle = open(source, "rb")  # noqa: SIM115 - owned by CopcStream
        self.stamp = self._stamp()
        self.total_bytes = self.stamp[2]
        self.bytes_read = 0
        self.requests = 0

    def _stamp(self) -> tuple[int, int, int, int, int]:
        stat = os.fstat(self.handle.fileno())
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def __call__(self, offset: int, length: int) -> bytes:
        if type(offset) is not int or type(length) is not int or offset < 0 or length < 0:
            raise ValueError("range offset and length must be nonnegative integers")
        if length > MAX_RANGE_BYTES or offset + length > self.total_bytes:
            raise ValueError("range exceeds file or response limit")
        if self._stamp() != self.stamp:
            raise OSError("local COPC source changed during reading")
        self.handle.seek(offset)
        data = self.handle.read(length)
        if len(data) != length or self._stamp() != self.stamp:
            raise OSError("local COPC source changed or range is truncated")
        self.bytes_read += len(data)
        self.requests += 1
        return data

    def close(self) -> None:
        self.handle.close()


class CopcStream:
    """Context-managed full-density box reader for local/HTTP(S) COPC.

    Walk all overlapping octree levels, decode one node at a time, and
    filter its raw records in batches. There is no global output/hierarchy
    list. Metadata includes source VLRs and non-index EVLRs, within its cap;
    large COPC hierarchy EVLR payloads are skipped. ``cancel`` is checked
    between ranges/nodes/batches; a blocking HTTP request has ``timeout``.
    HTTP objects without a strong ETag are size-pinned only and cannot be
    used for an immutable resumable job.
    """

    def __init__(
        self,
        source: str | os.PathLike[str],
        *,
        bounds: Bounds | None = None,
        chunk_size: int = 100_000,
        limits: CopcLimits | None = None,
        cancel: Callable[[], bool] | None = None,
        timeout: float = 60,
    ):
        import laspy

        if type(chunk_size) is not int or not 1 <= chunk_size <= 1_000_000:
            raise ValueError("chunk_size must be an integer in 1..1000000")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if bounds is not None:
            values = np.asarray(bounds, dtype=np.float64)
            if values.shape != (6,) or not np.isfinite(values).all() or np.any(values[:3] > values[3:]):
                raise ValueError("bounds must contain finite min XYZ <= max XYZ")
        self.source = os.fspath(source)
        self.limits = limits or CopcLimits()
        self.chunk_size = chunk_size
        self.cancel = cancel
        self.closed = False
        self._iterating = False
        self._remote = urlparse(self.source).scheme in {"http", "https"}
        scheme = urlparse(self.source).scheme
        if scheme and scheme not in {"http", "https"} and len(scheme) != 1:
            raise ValueError("use a local file or HTTP(S) COPC URL (including a presigned URL)")
        self.ranges = HttpRangeReader(self.source, timeout=timeout) if self._remote else _LocalRanges(self.source)
        try:
            self._check()
            length = min(1 << 16, self.ranges.total_bytes) if not self._remote else 1 << 16
            head = self._read(0, length)
            needed = CopcReader.header_length(head)
            if not CopcReader.is_copc(head) or needed is None:
                raise ValueError("input is not a COPC file")
            self.file_size = self.ranges.total_bytes
            if self.file_size is None or not 589 <= needed <= min(self.file_size, self.limits.metadata_bytes):
                raise ValueError("COPC header exceeds metadata/file limit")
            self.raw_header = self._read(0, needed) if needed > len(head) else head[:needed]
            # Use the root cube by default, padded for floating-point faces.
            center = np.frombuffer(self.raw_header, dtype="<f8", count=3, offset=429)
            halfsize = np.frombuffer(self.raw_header, dtype="<f8", count=1, offset=453)[0]
            if bounds is None:
                margin = (np.abs(center) + abs(halfsize)) * np.finfo(float).eps * 8
                values = np.concatenate((center - halfsize - margin, center + halfsize + margin))
            self.bounds = cast(Bounds, tuple(float(value) for value in values))
            self.query = CopcSpatialQuery(
                self.raw_header, self.file_size, self.bounds,
                self.limits.page_bytes, self.limits.compressed_node_bytes,
                self.limits.raw_node_bytes, self.limits.pending_entries, self.limits.page_depth,
            )
            self.header = laspy.LasHeader.read_from(io.BytesIO(self.raw_header), read_evlrs=False)
            if self.header.point_format.size != self.query.record_length:
                raise ValueError("LAS schema does not match the raw record length")
            self.header.point_count = self.query.total_points
            self.total_points = self.query.total_points
            self._read_metadata_evlrs()
            self.header_sha256 = hashlib.sha256(self.raw_header).hexdigest()
        except BaseException:
            self.close()
            raise

    def _check(self) -> None:
        if self.closed:
            raise ValueError("COPC stream is closed")
        if self.cancel is not None and self.cancel():
            raise CopcCancelled("COPC reading cancelled")

    def _read(self, offset: int, length: int) -> bytes:
        self._check()
        result = self.ranges(offset, length)
        self._check()
        return result

    def _read_metadata_evlrs(self) -> None:
        from laspy.vlrs.vlrlist import VLRList

        count = self.header.number_of_evlrs
        offset = self.header.start_of_first_evlr
        if count > 16_384 or (count and offset < len(self.raw_header)):
            raise ValueError("invalid EVLR count or offset")
        used = len(self.raw_header)
        kept = VLRList()
        raw_metadata: list[bytes] = []
        self.index_evlrs: list[tuple[int, int]] = []
        for _ in range(count):
            if offset + 60 > self.file_size:
                raise ValueError("EVLR header is outside the source")
            header = self._read(offset, 60)
            size = int.from_bytes(header[20:28], "little")
            end = offset + 60 + size
            if end > self.file_size:
                raise ValueError("EVLR payload is outside the source")
            # COPC hierarchy data is an index, not output LAS metadata.
            if header[2:18].split(b"\0")[0] == b"copc" and int.from_bytes(header[18:20], "little") == 1000:
                self.index_evlrs.append((offset, size))
            else:
                used += 60 + size
                if used > self.limits.metadata_bytes:
                    raise ValueError("non-index EVLRs exceed the metadata byte limit")
                payload = self._read(offset + 60, size)
                raw_metadata.append(header + payload)
                kept.extend(VLRList.read_from(io.BytesIO(header + payload), 1, extended=True))
            offset = end
        self.header.evlrs = kept
        # Preserve original bytes as well as parsed metadata for durable jobs.
        self.raw_metadata_evlrs = b"".join(raw_metadata)

    @property
    def identity(self) -> dict[str, Any]:
        """Source identity for checkpoints; local stamps, or a strong HTTP ETag."""
        result: dict[str, Any] = {"file_size": self.file_size, "header_sha256": self.header_sha256}
        if isinstance(self.ranges, _LocalRanges):
            result["local_stamp"] = self.ranges.stamp
            result["source_sha256"] = hashlib.sha256(os.path.abspath(self.source).encode()).hexdigest()
        else:
            result["etag"] = self.ranges.etag
            result["source_sha256"] = hashlib.sha256(self.source.encode()).hexdigest()
        return result

    def nodes(self) -> Iterator[CopcNode]:
        """Yield one pending node; acknowledge it when iteration advances.

        A consumer can skip a committed node without reading its compressed
        bytes. Only one traversal iterator may be active on this stream.
        """
        self._check()
        if self._iterating:
            raise ValueError("only one COPC traversal iterator may be active")
        self._iterating = True
        try:
            while True:
                self._check()
                item = self.query.next_item()
                if item is None:
                    return
                kind, offset, size, count = item
                if kind == "page":
                    self.query.supply_page(offset, self._read(offset, size))
                else:
                    yield CopcNode(offset, size, count)
                    self._check()
                    self.query.advance_node()
        finally:
            self._iterating = False

    def read_batches(self, node: CopcNode) -> Iterator[CopcBatch]:
        """Decode the pending node; filter/output arrays are batch-sized."""
        import laspy

        self._check()
        if self.query.next_item() != ("node", node.offset, node.size, node.count):
            raise ValueError("node does not match the pending query item")
        compressed = self._read(node.offset, node.size)
        raw = self.query.decode_records(node.offset, compressed)
        del compressed
        records = np.frombuffer(raw, dtype=self.header.point_format.dtype())
        bounds = np.asarray(self.bounds)
        for start in range(0, len(records), self.chunk_size):
            self._check()
            part = records[start : start + self.chunk_size]
            positions = np.column_stack((part["X"], part["Y"], part["Z"])).astype(np.float64)
            positions *= self.header.scales
            positions += self.header.offsets
            mask = np.isfinite(positions).all(axis=1) & np.all(
                (positions >= bounds[:3]) & (positions <= bounds[3:]), axis=1,
            )
            if mask.any():
                selected = part[mask]
                yield CopcBatch(
                    laspy.ScaleAwarePointRecord(selected, self.header.point_format, self.header.scales, self.header.offsets),
                    positions[mask], node.offset, np.flatnonzero(mask).astype(np.uint64) + start,
                )

    def __iter__(self) -> Iterator[CopcBatch]:
        for node in self.nodes():
            yield from self.read_batches(node)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            close = getattr(self.ranges, "close", None)
            if close is not None:
                close()

    def __enter__(self) -> CopcStream:
        self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def iter_copc_batches(source: str | os.PathLike[str], **kwargs: Any) -> Iterator[CopcBatch]:
    """Yield full-density records; close the generator to release its source."""
    with CopcStream(source, **kwargs) as stream:
        yield from stream
