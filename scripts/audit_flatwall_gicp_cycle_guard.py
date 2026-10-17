#!/usr/bin/env python3
"""Replay an explicit GICP cycle guard against frozen real inputs and history."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

import audit_flatwall_gicp_cycles as independent
import audit_flatwall_gicp_registration as registration
import audit_flatwall_gicp_shapes as previous
import audit_flatwall_raw_point_time as fields


def compare_report(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        if key.startswith("max_"):
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-14, rtol=1e-5)
        else:
            assert actual[key] == expected[key], key


def check_prefix(actual, frozen, report):
    n = report["last_iteration"]
    assert actual["termination"] == "cycle_detected" and not actual["converged"]
    assert actual["failure_reason"] == "repeated nonstationary GICP outer states"
    assert actual["iterations"] == len(actual["history"]) == n
    assert actual["accepted_steps"] == sum(r["accepted_scale"] is not None for r in frozen["history"][:n-1])
    previous.equal_bits(actual["history"][:-1], frozen["history"][:n-1])
    terminal = dict(frozen["history"][n-1])
    terminal.update(pose_after=terminal["pose_before"].copy(), cost_after=None,
                    trial_scales=np.empty(0), trial_costs=[], accepted_scale=None)
    previous.equal_bits(actual["history"][-1], terminal)
    previous.equal_bits(actual["pose"], terminal["pose_before"])
    for out, row in [("matched_source_indices", "source_indices"), ("matched_target_indices", "target_indices"),
                     ("squared_distances", "squared_distances"), ("weights", "weights"), ("final_cost", "cost_before")]:
        previous.equal_bits(actual[out], terminal[row])


def audit(inputs, prototype, output, slice_protocol, build_inputs):
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native

    if output.exists() or output.is_relative_to(inputs) or output.is_relative_to(prototype):
        raise ValueError("use a fresh output outside previous evidence")
    if os.environ.get("RAYON_NUM_THREADS") != "4" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("use four Rayon threads and one OpenBLAS thread")
    frozen = {p.relative_to(inputs).as_posix(): fields.artifact(p) for p in sorted(inputs.rglob("*")) if p.is_file()}
    proto_files = {p.name: fields.artifact(p) for p in sorted(prototype.iterdir()) if p.is_file()}
    controller, binary = fields.artifact(Path(__file__)), fields.artifact(Path(native.__file__))
    reference = {(r["family"], r["case"], r["row"]): r for r in json.loads((prototype/"frames.json").read_text(encoding="utf-8"))}
    output.mkdir(parents=True)
    fields.save(output/"protocol.json", dict(schema="cloudanalyzer.flatwall_gicp_cycle_guard.v1",
        controller=controller, native=binary, slice_protocol=fields.artifact(slice_protocol), build_inputs=fields.artifact(build_inputs),
        input_artifacts=frozen, prototype_artifacts=proto_files,
        reference_dependencies={p.name: fields.artifact(p) for p in [Path(independent.__file__), Path(registration.__file__), Path(previous.__file__), Path(fields.__file__)]},
        options=independent.DEFAULTS, report_comparison=dict(atol=2e-14, rtol=1e-5),
        scope="All frozen real problems, disabled exact outputs, enabled exact accepted prefix and independently predicted guard decisions. No accuracy or performance claim."))
    params = dict(max_correspondence_distance=2., max_iterations=30, translation_tolerance=1e-4,
                  rotation_tolerance=1e-5, robust_scale=None)
    guard = core.GicpCycleGuard(**independent.DEFAULTS)
    frames = []
    for family in ["caller_shapes", "native_shapes"]:
        for path in sorted((inputs/family).glob("*/frame-*.npz")):
            with np.load(path, allow_pickle=False) as arrays:
                data = {k: arrays[k].copy() for k in arrays.files}
            names = ["source", "target", "source_shapes", "target_shapes", "initial_pose", "center_world"]
            values = [data[k] for k in names]
            original = previous.load_registration(data)
            disabled = core.gicp_register(*values, capture_history=True, **params)
            previous.equal_bits(disabled, original)
            previous.equal_bits(core.gicp_register(*values, capture_history=True, cycle_guard=None, parallel=True, **params), original)
            identity = (family, path.parent.name, int(path.stem[6:]))
            predicted = reference[identity]["cycle"]
            # Recompute the independent decision; do not trust a classification file alone.
            assert independent.first_cycle(original["history"], data["center_world"]) == predicted
            actual = core.gicp_register(*values, cycle_guard=guard, capture_history=True, **params)
            if predicted is None:
                previous.equal_bits(actual, original)
            else:
                compare_report(actual["cycle"], predicted)
                check_prefix(actual, original, predicted)
            target_points, target_shapes = values[1].copy(), values[3].copy()
            target = core.GicpTarget(target_points, target_shapes)
            target_points[:], target_shapes[:] = np.nan, 0.
            for parallel in [False, True]:
                previous.equal_bits(core.gicp_register(*values, cycle_guard=guard, parallel=parallel, capture_history=True, **params), actual)
                previous.equal_bits(target.register(values[0], values[2], values[4], values[5],
                    cycle_guard=guard, parallel=parallel, capture_history=True, **params), actual)
            uncaptured = core.gicp_register(*values, cycle_guard=guard, **params)
            previous.equal_bits(uncaptured, dict(actual, history=[]))
            if predicted is not None:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = pool.map(lambda _: target.register(values[0], values[2], values[4], values[5],
                        cycle_guard=guard, capture_history=True, **params), range(8))
                    for result in results: previous.equal_bits(result, actual)
                mutated = target.register(values[0], values[2], values[4], values[5], cycle_guard=guard, capture_history=True, **params)
                mutated["pose"][:] = 0.
                mutated["cycle"]["period"] = 99
                mutated["history"][0]["combined_shapes"][:] = 0.
                previous.equal_bits(target.register(values[0], values[2], values[4], values[5],
                    cycle_guard=guard, capture_history=True, **params), actual)
            destination = output/family/path.parent.name
            destination.mkdir(parents=True, exist_ok=True)
            registration.save_result(destination/path.name, {k: data[k] for k in names}, actual)
            item = dict(family=family, case=identity[1], row=identity[2], original_termination=original["termination"],
                termination=actual["termination"], iterations=actual["iterations"], accepted_steps=actual["accepted_steps"],
                final_cost=actual["final_cost"], final_matches=len(actual["weights"]), cycle=actual.get("cycle"),
                disabled_frozen_bit_equal=True, accepted_prefix_bit_equal=True,
                serial_parallel_prepared_owned_bits_equal=True, history_capture_independent=True)
            frames.append(item)
            print(json.dumps(item), flush=True)
    summary = dict(histories=len(frames), cycles=sum(r["cycle"] is not None for r in frames),
        original_statuses=dict(Counter(r["original_termination"] for r in frames)),
        statuses=dict(Counter(r["termination"] for r in frames)), iterations=sum(r["iterations"] for r in frames),
        accepted_steps=sum(r["accepted_steps"] for r in frames),
        disabled_frozen_bit_equal=True, independent_decisions_agree=True, accepted_prefix_bit_equal=True,
        serial_parallel_prepared_owned_bits_equal=True, history_capture_independent=True,
        shared_target_cycle_calls=32, returned_cycle_outputs_owned=True,
        scope="Bounded failure diagnosis on fixed real inputs. Remaining limited cases stay limited. No convergence recovery, frontend adoption, physical accuracy or speed claim.")
    fields.save(output/"frames.json", frames)
    fields.save(output/"summary.json", summary)
    for name, item in frozen.items(): assert fields.artifact(inputs/name) == item, name
    for name, item in proto_files.items(): assert fields.artifact(prototype/name) == item, name
    assert fields.artifact(Path(__file__)) == controller and fields.artifact(Path(native.__file__)) == binary
    fields.save(output/"receipt.json", dict(schema="cloudanalyzer.flatwall_gicp_cycle_guard.v1", controller=controller,
        native=binary, frozen_inputs_unchanged=len(frozen), prototype_artifacts_unchanged=len(proto_files),
        outputs={p.relative_to(output).as_posix(): fields.artifact(p) for p in sorted(output.rglob("*")) if p.is_file()}))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["input", "prototype", "out", "slice-protocol", "build-inputs"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    audit(*(p.resolve() for p in [args.input, args.prototype, args.out, args.slice_protocol, args.build_inputs]))


if __name__ == "__main__":
    main()
