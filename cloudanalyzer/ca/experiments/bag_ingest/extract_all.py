"""Bag ingest strategy: metadata plus optional first-message decode."""

from __future__ import annotations

from ca.experiments.bag_ingest.common import inspect_bag_metadata


def inspect(path: str) -> dict:
    """Inspect a bag by reading connections and decoding one message per topic."""
    return inspect_bag_metadata(path, decode_sample=True)
