"""Bag ingest strategy: connection metadata only (no deserialization)."""

from __future__ import annotations

from ca.experiments.bag_ingest.common import inspect_bag_metadata


def inspect(path: str) -> dict:
    """Inspect a bag using topic headers only."""
    return inspect_bag_metadata(path, decode_sample=False)
