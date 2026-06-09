"""Compare concrete ROS bag ingest inspection strategies."""

from __future__ import annotations

from ca.experiments.bag_ingest import extract_all, stream_decode

STRATEGIES = {
    "extract_all": extract_all.inspect,
    "stream_decode": stream_decode.inspect,
}
