"""Bag ingest strategy: connection metadata only (no deserialization)."""

from __future__ import annotations

from ca.core.bag_ingest import inspect_bag


def inspect(path: str) -> dict:
    """Inspect a bag using topic headers only."""
    return inspect_bag(path, decode_sample=False)
