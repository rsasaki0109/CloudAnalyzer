"""Bounded PointCloud2 structure/XYZ inspection without altering sensor data."""
from __future__ import annotations

import numpy as np

MAX_POINTS = 5_000_000
MAX_DATA_BYTES = 256 * 1024**2
SIZES = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}


def inspect_pointcloud(message) -> dict:
    width, height = int(message.width), int(message.height)
    step, row_step = int(message.point_step), int(message.row_step)
    count = width * height
    if width < 1 or height < 1:
        raise ValueError('empty point cloud')
    if count > MAX_POINTS:
        raise ValueError('point count exceeds the 5000000-point diagnostic limit')
    if step < 1 or row_step < width * step:
        raise ValueError('invalid point_step or row_step')
    data = memoryview(message.data)
    if not data.contiguous or data.nbytes > MAX_DATA_BYTES:
        raise ValueError('point data is noncontiguous or exceeds the 256 MiB diagnostic limit')
    if data.nbytes != row_step * height:
        raise ValueError('point data length differs from row_step * height')
    if not 3 <= len(message.fields) <= 256:
        raise ValueError('missing XYZ fields or excessive field count')
    fields = {}
    for field in message.fields:
        if not isinstance(field.name, str) or not field.name or len(field.name) > 128 or field.name in fields:
            raise ValueError('invalid or duplicate point field name')
        size = SIZES.get(int(field.datatype))
        offset, elements = int(field.offset), int(field.count)
        if size is None or offset < 0 or elements < 1 or offset + size * elements > step:
            raise ValueError('point field extends outside point_step or has unsupported type/count')
        fields[field.name] = field
    if not {'x', 'y', 'z'} <= fields.keys():
        raise ValueError('missing XYZ fields')
    spans = []
    for name in 'xyz':
        field = fields[name]
        if field.datatype not in (7, 8) or field.count != 1:
            raise ValueError('XYZ must be scalar FLOAT32 or FLOAT64')
        spans.append((field.offset, field.offset + SIZES[field.datatype]))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('XYZ fields overlap')
    valid = np.ones((height, width), dtype=bool)
    endian = '>' if message.is_bigendian else '<'
    for name in 'xyz':
        field = fields[name]
        values: np.ndarray = np.ndarray((height, width), dtype=endian + ('f4' if field.datatype == 7 else 'f8'),
                            buffer=data, offset=field.offset, strides=(row_step, step))
        valid &= np.isfinite(values)
    finite = int(np.count_nonzero(valid))
    return {'points': count, 'finite_points': finite, 'nonfinite_points': count - finite}
