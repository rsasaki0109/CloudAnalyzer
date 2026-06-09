"""Bag ingest strategy: metadata plus optional first-message decode."""

from __future__ import annotations

from ca.core.bag_ingest import inspect_bag


def inspect(path: str) -> dict:
    """Inspect a bag by reading connections and decoding one message per topic."""
    return inspect_bag(path, decode_sample=True)
