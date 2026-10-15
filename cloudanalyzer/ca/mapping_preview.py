"""Display-only subsets preserve canonical PLY record bytes and original provenance."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, BinaryIO


def header(stream: BinaryIO) -> tuple[bytes, int, int]:
    lines: list[bytes] = []
    size = 0
    while size < 16384:
        line = stream.readline(16384)
        if not line:
            raise ValueError("preview requires a complete canonical binary PLY header")
        lines.append(line)
        size += len(line)
        if line == b"end_header\n":
            break
    else:
        raise ValueError("preview PLY header exceeds its limit")
    if lines[:2] != [b"ply\n", b"format binary_little_endian 1.0\n"] or lines[3:6] != [
        b"property double x\n",
        b"property double y\n",
        b"property double z\n",
    ]:
        raise ValueError("preview requires canonical binary PLY with double XYZ")
    match = re.fullmatch(rb"element vertex ([0-9]+)\n", lines[2])
    if not match or not 1 <= int(match[1]) <= 10**9:
        raise ValueError("invalid preview source vertex count")
    names = ["x", "y", "z"]
    for line in lines[6:-1]:
        field = re.fullmatch(rb"property float ([A-Za-z_][A-Za-z_0-9]*)\n", line)
        if not field:
            raise ValueError("unsupported preview source attribute schema")
        names.append(field[1].decode("ascii"))
    if len(names) > 19 or len(set(names)) != len(names):
        raise ValueError("invalid preview attribute names")
    return b"".join(lines), int(match[1]), 24 + 4 * (len(names) - 3)


def write(
    source: dict[str, Any], target: Path, source_count: int, max_points: int
) -> dict[str, Any]:
    """Stream source rows; retain every kth complete record without quantization."""
    if type(max_points) is not int or not 1 <= max_points <= 10**6:
        raise ValueError("max_preview_points must be an integer from 1 to 1000000")
    digest = hashlib.sha256()
    with Path(source["path"]).open("rb") as incoming:
        original, count, width = header(incoming)
        if count != source_count or source["bytes"] != len(original) + count * width:
            raise ValueError(
                "preview source count/record bytes differ from delivered point map"
            )
        if max_points >= count:
            raise ValueError(
                "preview sampling needs fewer points than the original map"
            )
        stride = math.ceil(count / max_points)
        kept = math.ceil(count / stride)
        replacement = re.sub(
            rb"element vertex [0-9]+\n",
            f"element vertex {kept}\n".encode(),
            original,
            count=1,
        )
        digest.update(original)
        consumed = 0
        with target.open("xb") as output:
            output.write(replacement)
            while consumed < count:
                records = min(count - consumed, max(1, 1024**2 // width))
                block = incoming.read(records * width)
                if len(block) != records * width:
                    raise ValueError("preview source changed while streaming records")
                digest.update(block)
                start = (-consumed) % stride
                # Allocation is bounded by one input chunk; retained attribute bytes stay exact.
                output.write(
                    b"".join(
                        block[i * width : (i + 1) * width]
                        for i in range(start, records, stride)
                    )
                )
                consumed += records
            if incoming.read(1):
                raise ValueError(
                    "preview source changed after its last recorded vertex"
                )
    if digest.hexdigest() != source["sha256"]:
        raise ValueError("preview source hash changed while streaming records")
    return {
        "purpose": "display_only",
        "source": source,
        "source_count": count,
        "preview_count": kept,
        "max_preview_points": max_points,
        "every_nth_record": stride,
        "first_record": 0,
        "record_size_bytes": width,
        "coordinate_frame_changed": False,
        "coordinate_or_attribute_quantization": False,
        "original_record_bytes_preserved": True,
        "source_for_saved_audits": "original_full_point_map",
        "full_point_map_included": False,
    }
