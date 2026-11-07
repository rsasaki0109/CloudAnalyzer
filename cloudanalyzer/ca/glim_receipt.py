"""Bounded, unambiguous JSON loading for saved GLIM execution receipts."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MAX_RECEIPT_BYTES = 4 * 1024**2


def read_glim_receipt(path: Path) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_RECEIPT_BYTES:
        raise ValueError('GLIM run report must be a regular file of at most 4 MiB')
    with path.open('rb') as stream:
        raw = stream.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError('GLIM run report exceeds 4 MiB')

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate object key')
            result[key] = value
        return result

    def invalid_constant(token):
        raise ValueError('nonfinite JSON number')

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        return value

    try:
        report = json.loads(raw, object_pairs_hook=unique_object,
                            parse_constant=invalid_constant, parse_float=finite_float)
    except (ValueError, RecursionError) as error:
        raise ValueError(f'invalid GLIM run report JSON: {error}') from error
    if not isinstance(report, dict) or report.get('schema') != 'cloudanalyzer.glim_run.v1':
        raise ValueError('unsupported GLIM run report')
    return raw, report
