#!/usr/bin/env python3
"""Check immutable GICP target reuse against real fixed-map inputs and oracle."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np

import audit_flatwall_gicp_linearization as factor
import audit_flatwall_gicp_registration as registration
import audit_flatwall_gicp_shapes as shapes


def audit(inputs, shape_inputs, output, slice_protocol, build_inputs):
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    import audit_flatwall_raw_point_time as fields

    directories = {"caller_shapes": inputs, "native_shapes": shape_inputs}
    if output.exists() or any(output.is_relative_to(path) for path in directories.values()):
        raise ValueError("use a fresh output outside frozen evidence")
    if os.environ.get("RAYON_NUM_THREADS") != "4" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("use a fresh four-Rayon, one-OpenBLAS process")
    frozen = {f"{family}/{p.relative_to(root).as_posix()}": fields.artifact(p)
              for family, root in directories.items() for p in sorted(root.rglob("*")) if p.is_file()}
    controller, binary = fields.artifact(Path(__file__)), fields.artifact(Path(native.__file__))
    protocol = dict(schema="cloudanalyzer.flatwall_gicp_target.v1", slice=factor.read(slice_protocol),
                    input_artifacts=frozen, controller=controller, native=binary,
                    build_inputs=fields.artifact(build_inputs), slice_protocol=fields.artifact(slice_protocol),
                    cases=registration.CASES, rows=registration.ROWS, families=list(directories),
                    reference_dependencies={p.name: fields.artifact(p) for p in [Path(factor.__file__), Path(registration.__file__), Path(shapes.__file__)]},
                    versions={name: version(name) for name in ["numpy", "scipy"]})
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output/"protocol.json", protocol)
    params = dict(max_correspondence_distance=2., max_iterations=30, translation_tolerance=1e-4,
                  rotation_tolerance=1e-5, robust_scale=None)
    frames, summaries = [], []
    names = ["source", "target", "source_shapes", "target_shapes", "initial_pose", "center_world"]
    for family, root in directories.items():
        for case in registration.CASES:
            destination = output/family/case
            destination.mkdir(parents=True)
            case_rows = []
            for row in registration.ROWS:
                with np.load(root/case/f"frame-{row:02d}.npz", allow_pickle=False) as file:
                    data = {k: file[k].copy() for k in file.files}
                source, target, cs, ct, initial, center = [data[name] for name in names]
                frozen_result = shapes.load_registration(data)
                standalone = core.gicp_register(source, target, cs, ct, initial, center, capture_history=True, **params)
                shapes.equal_bits(standalone, frozen_result)
                prepared = core.GicpTarget(target, ct)
                actual = prepared.register(source, cs, initial, center, capture_history=True, **params)
                shapes.equal_bits(actual, standalone)
                target_copy, shapes_copy = target.copy(), ct.copy()
                owned = core.GicpTarget(target_copy, shapes_copy)
                target_copy[:] = np.nan
                shapes_copy[:] = 0.
                assert len(owned) == len(prepared) == len(target)
                for obj, parallel in [(prepared, True), (prepared, False), (owned, False), (owned, True)]:
                    shapes.equal_bits(obj.register(source, cs, initial, center, parallel=parallel, capture_history=True, **params), actual)
                expected = registration.reference(source, target, cs, ct, initial, center, **params)
                registration.compare(actual, expected)
                registration.save_result(destination/f"frame-{row:02d}.npz", {name: data[name] for name in names}, actual)
                item = dict(family=family, case=case, row=row, source_points=len(source), target_points=len(target),
                            termination=actual["termination"], iterations=actual["iterations"], accepted_steps=actual["accepted_steps"],
                            trial_count=sum(len(step["trial_scales"]) for step in actual["history"]),
                            matched_points=len(actual["weights"]), final_cost=actual["final_cost"],
                            max_reference_pose_error=float(np.max(np.abs(actual["pose"]-expected["pose"]))),
                            standalone_frozen_bit_equal=True, prepared_standalone_bit_equal=True,
                            reuse_parallel_owned_bit_equal=True)
                case_rows.append(item)
                print(json.dumps(item), flush=True)
            fields.save(destination/"frames.json", case_rows)
            frames.extend(case_rows)
            summaries.append(dict(family=family, case=case, registrations=len(case_rows),
                                  statuses=dict(Counter(r["termination"] for r in case_rows)),
                                  source_points=sum(r["source_points"] for r in case_rows),
                                  target_points=sum(r["target_points"] for r in case_rows),
                                  iterations=sum(r["iterations"] for r in case_rows),
                                  trial_count=sum(r["trial_count"] for r in case_rows),
                                  max_reference_pose_error=max(r["max_reference_pose_error"] for r in case_rows),
                                  artifacts={p.name: fields.artifact(p) for p in sorted(destination.iterdir()) if p.is_file()}))
    for name, expected in frozen.items():
        family, relative = name.split("/", 1)
        assert fields.artifact(directories[family]/relative) == expected, name
    assert fields.artifact(Path(__file__)) == controller
    assert fields.artifact(Path(native.__file__)) == binary
    fields.save(output/"frames.json", frames)
    fields.save(output/"summary.json", summaries)
    fields.save(output/"receipt.json", dict(schema=protocol["schema"], protocol=fields.artifact(output/"protocol.json"),
        controller=controller, native=binary, verified_registrations=len(frames),
        standalone_frozen_bit_equal=True, prepared_standalone_bit_equal=True, reuse_parallel_owned_bit_equal=True,
        frozen_input_artifacts_unchanged=len(frozen), no_trajectory_or_frontend_adoption=True,
        outputs={name: fields.artifact(output/name) for name in ["protocol.json", "frames.json", "summary.json"]}))
    print(json.dumps({"verified_registrations": len(frames), "statuses": dict(Counter(r["termination"] for r in frames))}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shape-input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slice-protocol", type=Path, required=True)
    parser.add_argument("--build-inputs", type=Path, required=True)
    args = parser.parse_args()
    audit(*(p.resolve() for p in [args.input, args.shape_input, args.out, args.slice_protocol, args.build_inputs]))


if __name__ == "__main__":
    main()
