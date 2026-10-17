#!/usr/bin/env python3
"""Independent bounded recurrence checks on captured GICP baseline states."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(max_period=8, confirmations=2, translation_tolerance=1e-9,
                rotation_tolerance=1e-10, relative_shape_tolerance=1e-9, weight_tolerance=1e-12)
MEASURES = ("translation", "rotation", "relative_shape", "weight")


def validate(params, translation_tolerance=1e-4, rotation_tolerance=1e-5):
    if not isinstance(params["max_period"], int) or not 2 <= params["max_period"] <= 16:
        raise ValueError("max_period must be in 2..16")
    if not isinstance(params["confirmations"], int) or not 2 <= params["confirmations"] <= 4:
        raise ValueError("confirmations must be in 2..4")
    if any(not np.isfinite(params[k]) or params[k] <= 0. for k in DEFAULTS if "tolerance" in k):
        raise ValueError("cycle tolerances must be finite and positive")
    if (params["translation_tolerance"] > .01*translation_tolerance
            or params["rotation_tolerance"] > .01*rotation_tolerance):
        raise ValueError("cycle pose tolerances must be <= 0.01 of convergence tolerances")


def centered_distance(a, b, center):
    """Left relative motion about center, invariant to the source coordinate frame."""
    # For rotations D = Rrel-I. Subtract before multiplying: identical poses
    # then return exact zero even when the source origin has a large lever arm.
    delta = (a[:3, :3]-b[:3, :3]) @ b[:3, :3].T
    displacement = (a[:3, 3]-b[:3, 3])-delta @ (b[:3, 3]-center)
    skew = np.array([delta[2, 1]-delta[1, 2], delta[0, 2]-delta[2, 0],
                     delta[1, 0]-delta[0, 1]])/2.
    values = (float(np.hypot.reduce(displacement)),
              float(np.arctan2(np.hypot.reduce(skew), 1.+np.trace(delta)/2.)))
    if not np.isfinite(delta).all() or not np.isfinite(displacement).all() or not np.isfinite(values).all():
        raise FloatingPointError("nonfinite cycle comparison")
    return values


def differences(a, b, center):
    if any(not np.array_equal(a[k], b[k]) for k in ["source_indices", "target_indices"]):
        return None
    translation, rotation = centered_distance(a["pose_before"], b["pose_before"], center)
    ca, cb = a["combined_shapes"], b["combined_shapes"]
    if ca.shape != cb.shape or a["weights"].shape != b["weights"].shape:
        return None
    na, nb = [np.hypot.reduce(c.reshape(-1, 9), axis=1) for c in (ca, cb)]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        shape = np.hypot.reduce((ca-cb).reshape(-1, 9), axis=1)/np.maximum(na, nb)
        weight = np.abs(a["weights"]-b["weights"])
    values = np.array([translation, rotation, np.max(shape, initial=0.), np.max(weight, initial=0.)])
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite cycle comparison")
    return values


def cycle_at(history, center, *, convergence_translation=1e-4, convergence_rotation=1e-5, **options):
    """Return the smallest witnessed period at this baseline, never convergence."""
    params = DEFAULTS | options
    validate(params, convergence_translation, convergence_rotation)
    bounds = np.array([params[k] for k in ["translation_tolerance", "rotation_tolerance",
                                        "relative_shape_tolerance", "weight_tolerance"]])
    n = len(history)
    for period in range(2, params["max_period"]+1):
        count = params["confirmations"]*period
        if n < count+period:
            continue
        maxima = np.zeros(4)
        for index in range(n-count, n):
            values = differences(history[index], history[index-period], center)
            if values is None or np.any(values > bounds):
                break
            maxima = np.maximum(maxima, values)
        else:
            motions = [centered_distance(history[j]["pose_before"], history[j-1]["pose_before"], center)
                       for j in range(n-period+1, n)]
            if any(t > convergence_translation or r > convergence_rotation for t, r in motions):
                return dict(period=period, confirmations=params["confirmations"], first_iteration=n-count-period+1,
                            last_iteration=n, compared_state_pairs=count,
                            **{"max_"+k: float(v) for k, v in zip(MEASURES, maxima)})
    return None


def first_cycle(history, center, **options):
    """Mirror only the declared ordering, using already captured full GN steps."""
    for index, row in enumerate(history):
        delta = row["step"]
        if delta is None or (np.hypot.reduce(delta[:3]) <= 1e-4 and np.hypot.reduce(delta[3:]) <= 1e-5):
            return None
        report = cycle_at(history[:index+1], center, **options)
        if report is not None:
            return report
    return None


def prototype(inputs, output, slice_protocol):
    import audit_flatwall_gicp_shapes as previous
    import audit_flatwall_raw_point_time as fields

    if output.exists() or output.is_relative_to(inputs):
        raise ValueError("use a fresh output outside frozen inputs")
    controller = fields.artifact(Path(__file__))
    frozen = {p.relative_to(inputs).as_posix(): fields.artifact(p) for p in sorted(inputs.rglob("*")) if p.is_file()}
    output.mkdir(parents=True)
    fields.save(output/"protocol.json", dict(schema="cloudanalyzer.flatwall_gicp_cycle_prototype.v1",
        controller=controller, input_artifacts=frozen, slice_protocol=fields.artifact(slice_protocol),
        options=DEFAULTS, reference_dependencies={p.name: fields.artifact(p) for p in [Path(previous.__file__), Path(fields.__file__)]},
        scope="Saved-history independent numerical recurrence decisions; no new estimator execution or convergence claim."))
    frames = []
    for family in ["caller_shapes", "native_shapes"]:
        for path in sorted((inputs/family).glob("*/frame-*.npz")):
            with np.load(path, allow_pickle=False) as arrays:
                data = {k: arrays[k].copy() for k in arrays.files}
            result = previous.load_registration(data)
            frames.append(dict(family=family, case=path.parent.name, row=int(path.stem[6:]),
                original_termination=result["termination"], cycle=first_cycle(result["history"], data["center_world"])))
    fields.save(output/"frames.json", frames)
    summary = dict(histories=len(frames), cycles=sum(r["cycle"] is not None for r in frames),
        original_statuses=dict(Counter(r["original_termination"] for r in frames)),
        scope="All frozen histories, with predeclared tolerances and bounds; a cycle is a failure diagnosis.")
    fields.save(output/"summary.json", summary)
    for name, item in frozen.items(): assert fields.artifact(inputs/name) == item, name
    assert fields.artifact(Path(__file__)) == controller
    fields.save(output/"receipt.json", dict(controller=controller, frozen_inputs_unchanged=len(frozen),
        outputs={name: fields.artifact(output/name) for name in ["protocol.json", "frames.json", "summary.json"]}))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slice-protocol", type=Path, required=True)
    args = parser.parse_args()
    prototype(*(p.resolve() for p in [args.input, args.out, args.slice_protocol]))


if __name__ == "__main__":
    main()
