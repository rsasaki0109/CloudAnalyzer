#!/usr/bin/env python3
"""Verify published bytes, decision witnesses and reported accounting only."""
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    for name, item in read(root/"packet-files.json")["files"].items():
        data = (root/name).read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], name
    original = read(root/"prototype/frames.json")
    frames = read(root/"audit/frames.json")
    summary = read(root/"audit/summary.json")
    identity = lambda row: (row["family"], row["case"], row["row"])
    indexed = {identity(row): row for row in original}
    assert len(indexed) == len(original) == len(frames) == summary["histories"] == 72
    assert len({identity(row) for row in frames}) == 72
    statuses, original_statuses = {}, {}
    cycles = 0
    for row in frames:
        previous = indexed[identity(row)]
        assert row["original_termination"] == previous["original_termination"]
        for key in ["disabled_frozen_bit_equal", "accepted_prefix_bit_equal", "serial_parallel_prepared_owned_bits_equal", "history_capture_independent"]:
            assert row[key] is True
        report = row["cycle"]
        assert (report is None) == (previous["cycle"] is None)
        if report is None:
            assert row["termination"] == row["original_termination"]
        else:
            cycles += 1
            assert row["termination"] == "cycle_detected" and row["original_termination"] == "max_iterations"
            for key in ["period", "confirmations", "first_iteration", "last_iteration", "compared_state_pairs"]:
                assert report[key] == previous["cycle"][key]
            assert report["compared_state_pairs"] == report["confirmations"]*report["period"]
            assert report["last_iteration"]-report["first_iteration"]+1 == (report["confirmations"]+1)*report["period"]
            assert row["iterations"] == report["last_iteration"]
            assert row["accepted_steps"] == row["iterations"]-1
            params = read(root/"audit/protocol.json")["options"]
            for name, bound in [("translation", "translation_tolerance"), ("rotation", "rotation_tolerance"),
                                ("relative_shape", "relative_shape_tolerance"), ("weight", "weight_tolerance")]:
                assert 0. <= report["max_"+name] <= params[bound]
        statuses[row["termination"]] = statuses.get(row["termination"], 0)+1
        original_statuses[row["original_termination"]] = original_statuses.get(row["original_termination"], 0)+1
    assert statuses == summary["statuses"] and original_statuses == summary["original_statuses"]
    assert cycles == summary["cycles"] == read(root/"prototype/summary.json")["cycles"]
    for name in ["iterations", "accepted_steps"]: assert sum(r[name] for r in frames) == summary[name]
    assert summary["independent_decisions_agree"] is True
    return dict(histories=len(frames), cycles=cycles, statuses=statuses,
                scope="Published bytes and reported witnesses only; no independent real-input rerun, convergence recovery, physical accuracy or performance certificate.")


if __name__ == "__main__": print(json.dumps(verify(), indent=2))
