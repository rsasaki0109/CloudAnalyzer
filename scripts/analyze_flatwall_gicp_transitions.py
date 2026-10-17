#!/usr/bin/env python3
"""Summarize verified transition costs and pose/correspondence recurrence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def pose_distance(a, b):
    """Translation norm and stable principal relative angle, without a cutoff."""
    relative = a[:3, :3]@b[:3, :3].T
    skew = np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0], relative[1, 0]-relative[0, 1]])/2.
    return dict(translation=float(np.hypot.reduce(a[:3, 3]-b[:3, 3])),
                rotation=float(np.arctan2(np.hypot.reduce(skew), (np.trace(relative)-1.)/2.)))


def correspondence_identity(row):
    return hashlib.sha256(np.asarray(row["source_indices"], dtype="<u8").tobytes()
                          + np.asarray(row["target_indices"], dtype="<u8").tobytes()).hexdigest()


def value_range(values):
    return [float(min(values)), float(max(values))] if values else None


def recurrence(history):
    """Describe last-ten baseline states at lags 1..5; never decide convergence."""
    identities = [correspondence_identity(row) for row in history]
    start = max(0, len(history)-10)
    lags = {}
    for lag in range(1, 6):
        distances, metric_differences, exact_poses, same_ids = [], [], 0, 0
        for index in range(max(start, lag), len(history)):
            a, b = history[index], history[index-lag]
            distances.append(pose_distance(a["pose_before"], b["pose_before"]))
            exact_poses += a["pose_before"].tobytes() == b["pose_before"].tobytes()
            if identities[index] == identities[index-lag]:
                same_ids += 1
                metric_differences.append(float(np.max(np.abs(a["combined_shapes"]-b["combined_shapes"]), initial=0.)))
        lags[str(lag)] = dict(comparisons=len(distances), equal_ordered_correspondence_sets=same_ids,
            exact_pose_bit_repeats=exact_poses, translation_range=value_range([d["translation"] for d in distances]),
            rotation_range=value_range([d["rotation"] for d in distances]),
            same_ids_combined_shape_max_difference_range=value_range(metric_differences))
    return dict(last_ten_states=min(len(history), 10), last_ten_unique_correspondence_sets=len(set(identities[start:])), lags=lags)


def effects(rows):
    """Raw signed observations; no significance or acceptance threshold."""
    names = ["motion_delta", "source_metric_delta", "assignment_delta", "assignment_first_delta", "source_metric_second_delta",
             "assignment_metric_interaction", "full_delta", "added_cost", "removed_cost"]
    return dict(transitions=len(rows), strict_frozen_motion_decreases=sum(r["motion_delta"] < 0. for r in rows),
        strict_refreshed_increases=sum(r["full_delta"] > 0. for r in rows),
        support_changed_transitions=sum(bool(r["added_sources"] or r["removed_sources"]) for r in rows),
        terms={name: dict(signed_sum=float(sum(r[name] for r in rows)),
            mean=float(sum(r[name] for r in rows)/len(rows)) if rows else None,
            range=value_range([r[name] for r in rows])) for name in names})


def analyze(audit_directory, original, output):
    import audit_flatwall_gicp_shapes as previous
    import audit_flatwall_raw_point_time as fields

    if output.exists() or output.is_relative_to(audit_directory) or output.is_relative_to(original):
        raise ValueError("use a fresh output outside original evidence")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    audit_receipt = read(audit_directory/"receipt.json")
    assert audit_receipt["accepted_transitions"] == 686 and audit_receipt["histories"] == 72
    audit_protocol = read(audit_directory/"protocol.json")
    for name, item in audit_protocol["input_artifacts"].items():
        assert fields.artifact(original/name) == item, name
    for name, item in audit_receipt["outputs"].items():
        assert fields.artifact(audit_directory/name) == item, name
    controller = fields.artifact(Path(__file__))
    output.mkdir(parents=True)
    frozen = {p.relative_to(original).as_posix(): fields.artifact(p) for p in sorted(original.rglob("*")) if p.is_file()}
    fields.save(output/"protocol.json", dict(schema="cloudanalyzer.flatwall_gicp_transition_analysis.v1", controller=controller,
        original_input_artifacts=frozen, audit_receipt=fields.artifact(audit_directory/"receipt.json"),
        audit_outputs=audit_receipt["outputs"],
        reference_dependencies={p.name: fields.artifact(p) for p in [Path(previous.__file__), Path(fields.__file__)]},
        population="All686 verified transitions and last-ten transitions of all ten limited histories (five unique real problems, two shape variants).",
        recurrence="For last-ten captured baseline poses/ordered source-target IDs, report lags1..5: exact ID/pose repeats and translation/rotation ranges. Compare combined shapes only when ordered correspondence IDs match. No approximate-cycle threshold or convergence decision. Principal rotation angle uses atan2 of skew norm and half trace-minus-one.",
        effects="Raw strict signs, signed sums/means/min/max of every declared accounting term, without selecting a significance threshold. Assignment includes target point and target shape; decompositions are order-dependent. Signed tail sums need not span a whole recurrence period. No causal or physical-accuracy claim."))
    rows = read(audit_directory/"transitions.json")
    tails = read(audit_directory/"limited-tails.json")
    frames = read(audit_directory/"frames.json")
    limited, repeated = [], []
    for item in frames:
        if item["termination"] != "max_iterations":
            continue
        identity = {key: item[key] for key in ["family", "case", "row"]}
        selected = [row for row in tails if all(row[key] == value for key, value in identity.items())]
        assert len(selected) == 10
        limited.append(dict(**identity, **effects(selected)))
        path = original/item["family"]/item["case"]/f"frame-{item['row']:02d}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            captured = previous.load_registration({key: arrays[key].copy() for key in arrays.files})
        assert captured["termination"] == "max_iterations" and len(captured["history"]) == 30
        repeated.append(dict(**identity, **recurrence(captured["history"])))
    assert len(limited) == len(repeated) == 10
    summary = dict(histories=len(frames), statuses=dict(Counter(r["termination"] for r in frames)), all_transitions=effects(rows),
        limited_last_ten=effects(tails), max_native_cost_error=max(r["max_native_cost_error"] for r in rows),
        max_native_shape_error=max(r["max_native_shape_error"] for r in rows),
        max_accounting_error=max(abs(r["accounting_error"]) for r in rows),
        scope="Observed fixed-support costs and recurrence descriptors only; no new optimizer/pose, significance cutoff, convergence/failure policy, physical accuracy or speed claim.")
    for name, value in [("limited-effects.json", limited), ("recurrence.json", repeated), ("summary.json", summary)]:
        fields.save(output/name, value)
    for name, item in frozen.items():
        assert fields.artifact(original/name) == item, name
    for name, item in audit_receipt["outputs"].items():
        assert fields.artifact(audit_directory/name) == item, name
    assert fields.artifact(Path(__file__)) == controller
    fields.save(output/"receipt.json", dict(schema="cloudanalyzer.flatwall_gicp_transition_analysis.v1", controller=controller,
        histories=72, transitions=686, limited_tail_transitions=100, limited_histories=10, original_input_artifacts_unchanged=len(frozen),
        outputs={name: fields.artifact(output/name) for name in ["protocol.json", "limited-effects.json", "recurrence.json", "summary.json"]},
        solver_and_termination_unchanged=True))
    print(json.dumps({"summary": summary, "limited_effects": limited}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    analyze(*(p.resolve() for p in [args.audit, args.input, args.out]))


if __name__ == "__main__":
    main()
