"""Scale benchmark helpers for the streaming MapEval AWD/SCS lane."""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ca import __version__
from ca.core.map_evaluate import MapEvalProtocol, evaluate_map_streaming


SCALE_BENCHMARK_SCHEMA_VERSION = "cloudanalyzer.mapeval_scale.v1"
DEFAULT_SCALE_SIZES = (1_000_000, 10_000_000)
DEFAULT_CHUNK_SIZE = 100_000


def iter_synthetic_map_chunks(
    point_count: int,
    *,
    chunk_size: int,
    seed: int,
    estimated: bool,
) -> Iterator[np.ndarray]:
    """Yield a deterministic dense 6m cube without a resident full map."""

    if point_count < 1:
        raise ValueError("point_count must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    rng = np.random.default_rng(seed)
    remaining = int(point_count)
    while remaining:
        count = min(remaining, chunk_size)
        points = rng.uniform(0.05, 5.95, size=(count, 3)).astype(np.float64)
        if estimated:
            points += rng.normal(0.0, 0.002, size=points.shape)
        yield points
        remaining -= count


def _rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    return value * 1024 if sys.platform != "darwin" else value


def run_scale_benchmark(
    sizes: tuple[int, ...] = DEFAULT_SCALE_SIZES,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    seed: int = 2026,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run streaming AWD/SCS for each requested map size and record resources."""

    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("sizes must contain positive point counts")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    protocol = MapEvalProtocol()
    results: list[dict[str, Any]] = []
    for index, size in enumerate(sizes):
        before = _rss_bytes()
        started = time.perf_counter()
        metrics = evaluate_map_streaming(
            iter_synthetic_map_chunks(
                size,
                chunk_size=chunk_size,
                seed=seed + index,
                estimated=True,
            ),
            iter_synthetic_map_chunks(
                size,
                chunk_size=chunk_size,
                seed=seed + index,
                estimated=False,
            ),
            protocol=protocol,
        )
        elapsed = time.perf_counter() - started
        after = _rss_bytes()
        results.append(
            {
                "points_per_map": int(size),
                "chunk_size": int(chunk_size),
                "seed": int(seed + index),
                "metrics": metrics.metrics,
                "runtime": {
                    "elapsed_seconds": float(elapsed),
                    "peak_rss_bytes": after,
                    "peak_rss_delta_bytes": (
                        None
                        if before is None or after is None
                        else max(0, after - before)
                    ),
                    "memory_measurement": "resource.RUSAGE_SELF.ru_maxrss high-water mark",
                },
            }
        )

    report: dict[str, Any] = {
        "schema_version": SCALE_BENCHMARK_SCHEMA_VERSION,
        "status": "completed",
        "runtime": {
            "cloudanalyzer_version": __version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "protocol": {
            **protocol.as_dict(),
            "execution_mode": "streaming",
            "input_generation": "uniform dense cube with optional 2mm estimate noise",
        },
        "plan": {
            "ci_smoke_points_per_map": 100_000,
            "target_points_per_map": [1_000_000, 10_000_000],
            "ci_policy": "run the 100k smoke only; run 1M/10M on a labeled benchmark runner",
            "comparison_policy": "compare elapsed time and peak RSS on the same runner; do not gate across hardware",
        },
        "runs": results,
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_SCALE_SIZES",
    "SCALE_BENCHMARK_SCHEMA_VERSION",
    "iter_synthetic_map_chunks",
    "run_scale_benchmark",
]
