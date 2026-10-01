"""Read COPC files, local or remote, down to a point budget."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ._core import CopcReader
from ._ranges import MAX_RANGE_BYTES, HttpRangeReader

#: Nodes this close together in the file are fetched in one read.
_MERGE_GAP = 64 << 10


def _file_reader(path: str | os.PathLike[str]) -> Callable[[int, int], bytes]:
    handle = open(path, "rb")  # noqa: SIM115 - closed with the reader's owner
    lock = threading.Lock()

    def read(offset: int, length: int) -> bytes:
        with lock:
            handle.seek(offset)
            return handle.read(length)

    read.close = handle.close  # type: ignore[attr-defined]
    return read


def _url_reader(url: str) -> Callable[[int, int], bytes]:
    return HttpRangeReader(url)


def read_copc(
    source: str | os.PathLike[str],
    max_points: int = 10_000_000,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    """Read a COPC file (a path or an ``http(s)`` URL) node by node.

    Every octree level is read while the points so far fit ``max_points``
    (the root level always is), so the density stays even and the rest of
    the file is never read; from a URL only those byte ranges are fetched.

    Returns the same dictionary as :func:`read` (``positions``, ``colors``,
    ``intensity``, ``classification``) plus ``total_points`` (in the file)
    and ``levels`` (octree levels read).
    """
    remote = str(source).startswith(("http://", "https://"))
    read = _url_reader(str(source)) if remote else _file_reader(source)
    try:
        head = read(0, 1 << 16)
        needed = CopcReader.header_length(head) or len(head)
        head = read(0, needed) if needed > len(head) else head[:needed]
        if not CopcReader.is_copc(head):
            raise ValueError(f"{source}: not a COPC file")
        reader = CopcReader(head)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            level, chosen = 0, 0
            while True:
                pages = reader.pages_for(level)
                for (offset, _), data in zip(pages, pool.map(lambda p: read(*p), pages), strict=True):
                    reader.add_page(offset, data)
                points = reader.level_points(level)
                if points == 0 or (level > 0 and chosen + points > max_points):
                    break
                chosen += points
                level += 1
                if not reader.deeper_than(level - 1):
                    break

            # Nodes in file order, fetched in runs of neighbouring ones.
            nodes = sorted(reader.nodes_to(level - 1))
            runs: list[list[int]] = []
            for offset, size, _ in nodes:
                if runs and offset <= runs[-1][1] + _MERGE_GAP and offset + size - runs[-1][0] <= MAX_RANGE_BYTES:
                    runs[-1][1] = max(runs[-1][1], offset + size)
                else:
                    runs.append([offset, offset + size])
            data = list(pool.map(lambda r: read(r[0], r[1] - r[0]), runs))
        chunks, counts = [], []
        run = 0
        for offset, size, count in nodes:
            while offset >= runs[run][1]:
                run += 1
            start = offset - runs[run][0]
            chunks.append(data[run][start : start + size])
            counts.append(count)
        out = reader.decode(chunks, counts)
        out["total_points"] = reader.total_points
        out["levels"] = level
        return out
    finally:
        close = getattr(read, "close", None)
        if close is not None:
            close()
