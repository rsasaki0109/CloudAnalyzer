#!/usr/bin/env python3
"""Audit an explicit native transported gradient and both rejected prototypes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

import audit_flatwall_gicp_registration as registration
import audit_flatwall_gicp_shapes as previous
import audit_flatwall_raw_point_time as fields
import gicp_transported_reference as independent


def audit(inputs, cycles, rematched, fixed, output, slice_protocol, build_inputs):
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native

    roots = dict(original=inputs, cycles=cycles, rematched=rematched, fixed=fixed)
    if output.exists() or any(output.is_relative_to(p) for p in roots.values()):
        raise ValueError("use a fresh output outside original results")
    if os.environ.get("RAYON_NUM_THREADS") != "4" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("use four Rayon threads and one OpenBLAS thread")
    frozen = {family+"/"+p.relative_to(root).as_posix(): fields.artifact(p)
              for family, root in roots.items() for p in sorted(root.rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts}
    controller, binary = fields.artifact(Path(__file__)), fields.artifact(Path(native.__file__))
    dependencies = {p.name: fields.artifact(p) for p in [Path(registration.__file__), Path(previous.__file__), Path(fields.__file__), Path(independent.__file__)]}
    output.mkdir(parents=True)
    fields.save(output/"protocol.json", dict(schema="cloudanalyzer.flatwall_gicp_transported.v1", controller=controller,
        native=binary, input_artifacts=frozen, reference_dependencies=dependencies,
        slice_protocol=fields.artifact(slice_protocol), build_inputs=fields.artifact(build_inputs),
        comparisons=dict(gradient_atol=2e-7, gradient_rtol=1e-9, cost_atol=2e-6, cost_rtol=1e-9),
        scope="New diagnostic derivative only. Existing defaults/registration and cycle policies stay unchanged; both proposed optimizer variants remain reference experiments."))
    summaries, states = [], []
    groups = {"original": inputs, "rematched": rematched, "fixed": fixed}
    old_cycles = {}
    for path in sorted(cycles.glob("*/*/frame-*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            old_cycles[tuple(path.parts[-3:])] = previous.load_registration({k: arrays[k].copy() for k in arrays.files})
    max_gradient_error, max_metric_error = 0., 0.
    checked_baselines = checked_trials = nearest_checks = 0
    def compare(source, target, cs, ct, pose, center, identity):
        nonlocal max_gradient_error, max_metric_error
        actual = core.gicp_transported_linearization(source, target, cs, ct, pose, center)
        baseline = core.gicp_linearization(source, target, cs, ct, pose, center)
        previous.equal_bits({k: actual[k] for k in baseline}, baseline)
        previous.equal_bits(actual, core.gicp_transported_linearization(source, target, cs, ct, pose, center, parallel=True))
        metric = independent.metric_gradient(source, target, cs, pose, baseline["combined_shapes"])
        gradient = -baseline["right_hand_side"].copy()
        gradient[3:] += metric
        np.testing.assert_allclose(actual["metric_gradient_rotation"], metric, atol=2e-7, rtol=1e-9)
        np.testing.assert_allclose(actual["objective_gradient"], gradient, atol=2e-7, rtol=1e-9)
        np.testing.assert_array_equal(actual["objective_right_hand_side"], -actual["objective_gradient"])
        error = float(np.max(np.abs(actual["objective_gradient"]-gradient), initial=0.))
        metric_error = float(np.max(np.abs(actual["metric_gradient_rotation"]-metric), initial=0.))
        max_gradient_error, max_metric_error = max(max_gradient_error, error), max(max_metric_error, metric_error)
        states.append(dict(**identity, pairs=len(source), cost=actual["cost"], gradient=actual["objective_gradient"].tolist(),
                           metric_gradient_rotation=actual["metric_gradient_rotation"].tolist(), independent_gradient_error=error,
                           independent_metric_error=metric_error, frozen_fields_bit_equal=True, parallel_bit_equal=True))
        return actual
    params = dict(max_correspondence_distance=2., max_iterations=30, translation_tolerance=1e-4, rotation_tolerance=1e-5)
    names = ["source", "target", "source_shapes", "target_shapes", "initial_pose", "center_world"]
    for group, root in groups.items():
        for path in sorted(root.glob("*/*/frame-*.npz")):
            with np.load(path, allow_pickle=False) as arrays:
                data = {k: arrays[k].copy() for k in arrays.files}
            result = previous.load_registration(data)
            source, target, cs, ct, initial, center = [data[k] for k in names]
            family, case, filename = path.parts[-3:]
            prepared = core.GicpTarget(target, ct)
            if group == "original":
                previous.equal_bits(core.gicp_register(source, target, cs, ct, initial, center, capture_history=True, **params), result)
                previous.equal_bits(prepared.register(source, cs, initial, center, parallel=True, capture_history=True, **params), result)
                guarded = core.gicp_register(source, target, cs, ct, initial, center,
                    cycle_guard=core.GicpCycleGuard(), capture_history=True, **params)
                previous.equal_bits(guarded, old_cycles[(family, case, filename)])
            for index, row in enumerate(result["history"]):
                si, ti = row["source_indices"], row["target_indices"]
                identity = dict(group=group, family=family, case=case, row=int(path.stem[6:]), iteration=index+1, kind="baseline", scale=None)
                actual = compare(source[si], target[ti], cs[si], ct[ti], row["pose_before"], center, identity)
                checked_baselines += 1
                np.testing.assert_allclose(actual["cost"], row["cost_before"], atol=2e-6, rtol=1e-9)
                if group != "original":
                    np.testing.assert_allclose(actual["objective_gradient"], row["objective_gradient"], atol=2e-7, rtol=1e-9)
                if group == "original": continue
                for scale, cost in zip(row["trial_scales"], row["trial_costs"]):
                    if cost is None: continue
                    pose = registration.retract(row["pose_before"], center, row["step"], scale)
                    if group == "rematched":
                        candidate = independent.score(source, target, cs, ct, pose, 2.)
                        ci, cj = candidate["source_indices"], candidate["target_indices"]
                        query = prepared.register(source, cs, pose, center, max_correspondence_distance=2., max_iterations=1, capture_history=True)
                        initial_row = query["history"][0]
                        np.testing.assert_array_equal(initial_row["source_indices"], ci)
                        np.testing.assert_array_equal(initial_row["target_indices"], cj)
                        nearest_checks += 1
                    else:
                        ci, cj = si, ti
                    state = compare(source[ci], target[cj], cs[ci], ct[cj], pose, center, dict(identity, kind="trial", scale=float(scale)))
                    np.testing.assert_allclose(state["cost"], cost, atol=2e-6, rtol=1e-9)
                    checked_trials += 1
            summaries.append(dict(group=group, family=family, case=case, row=int(path.stem[6:]),
                termination=result["termination"], iterations=result["iterations"], accepted_steps=result["accepted_steps"], final_cost=result["final_cost"],
                gradient_baselines=len(result["history"]), finite_trials=sum(c is not None for r in result["history"] for c in r["trial_costs"]) if group != "original" else 0))
            print(json.dumps(summaries[-1]), flush=True)
    assert len(summaries) == 216
    summary = dict(real_histories=len(summaries), distinct_problems=36, shape_families=2,
        checked_gradient_baselines=checked_baselines, checked_gradient_trials=checked_trials, checked_gradient_states=len(states),
        rematched_candidate_native_ID_checks=nearest_checks, max_independent_gradient_error=max_gradient_error,
        max_independent_metric_error=max_metric_error,
        group_statuses={g: dict(Counter(r["termination"] for r in summaries if r["group"] == g)) for g in groups},
        default_registrations_bit_equal=72, cycle_enabled_registrations_bit_equal=72,
        existing_frozen_factor_fields_bit_equal=True, serial_parallel_gradient_bits_equal=True,
        native_registration_policy_added=False,
        scope="Unit-weight fixed-pair derivative validation and exact old regression. Experimental reference policies show regressions and are not native/default/frontend behavior. No physical accuracy, convergence repair or speed claim.")
    fields.save(output/"frames.json", summaries)
    fields.save(output/"states.json", states)
    fields.save(output/"summary.json", summary)
    for key, item in frozen.items():
        family, relative = key.split("/", 1)
        assert fields.artifact(roots[family]/relative) == item, key
    for name, item in dependencies.items():
        assert fields.artifact(Path(independent.__file__).parent/name) == item, name
    assert fields.artifact(Path(__file__)) == controller and fields.artifact(Path(native.__file__)) == binary
    fields.save(output/"receipt.json", dict(controller=controller, native=binary, frozen_input_artifacts_unchanged=len(frozen),
        outputs={p.name: fields.artifact(p) for p in sorted(output.iterdir()) if p.is_file()}, no_registration_policy_added=True))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["input", "cycles", "rematched", "fixed", "out", "slice-protocol", "build-inputs"]:
        parser.add_argument("--"+name, type=Path, required=True)
    a = parser.parse_args()
    audit(*(p.resolve() for p in [a.input, a.cycles, a.rematched, a.fixed, a.out, a.slice_protocol, a.build_inputs]))


if __name__ == "__main__": main()
