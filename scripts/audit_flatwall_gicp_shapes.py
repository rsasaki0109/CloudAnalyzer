#!/usr/bin/env python3
"""Independent exact-neighbor/PCA and registration audit on real fixed maps."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

import audit_flatwall_gicp_linearization as factor
import audit_flatwall_gicp_registration as registration


def equal_bits(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_bits(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_bits(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, float):
        assert np.float64(a).tobytes() == np.float64(b).tobytes()
    else:
        assert a == b


def pca_reference(points, indices):
    """Independent NumPy population scatter/eigensystem on selected neighbors."""
    n, k = indices.shape
    shapes, supported, values, scatter = np.zeros((n, 3, 3)), np.zeros(n, dtype=bool), np.zeros((n, 3)), np.zeros((n, 3, 3))
    if k < 5 or not n:
        return shapes, supported, values, scatter
    offsets = points[indices] - points[:, None, :]
    centered = offsets - offsets.mean(axis=1)[:, None, :]
    scatter = centered.transpose(0, 2, 1) @ centered / k
    values, vectors = np.linalg.eigh(scatter)
    supported = (values[:, 2] > 0) & (values[:, 1] > 1e-12*values[:, 2])
    normals = vectors[:, :, 0]
    shapes[supported] = np.eye(3) + (.001-1.)*np.einsum("ni,nj->nij", normals[supported], normals[supported])
    return shapes, supported, values, scatter


def check_shapes(points, actual):
    n, k = len(points), min(20, len(points))
    ids = actual["neighbor_indices"]
    assert ids.shape == actual["neighbor_squared_distances"].shape == (n, k)
    np.testing.assert_array_equal(actual["neighbor_counts"], np.full(n, k))
    boundary_ties, distance_error = 0, 0.
    for start in range(0, n, 64):
        d = points[start:start+64, None, :] - points[None, :, :]
        d2 = d[..., 0]**2 + d[..., 1]**2 + d[..., 2]**2
        ordered = np.sort(d2, axis=1)
        chosen = ids[start:start+64]
        assert all(len(np.unique(row)) == k for row in chosen)
        assert ((chosen >= 0) & (chosen < n)).all()
        selected = d2[np.arange(len(chosen))[:, None], chosen]
        native = actual["neighbor_squared_distances"][start:start+64]
        np.testing.assert_allclose(native, selected, atol=2e-12, rtol=1e-12)
        np.testing.assert_allclose(native, ordered[:, :k], atol=2e-12, rtol=1e-12)
        distance_error = max(distance_error, float(np.max(np.abs(native-selected), initial=0)))
        if k < n:
            boundary_ties += int(np.count_nonzero(ordered[:, k-1] == ordered[:, k]))
    shapes, good, eigenvalues, scatter = pca_reference(points, ids)
    np.testing.assert_array_equal(actual["supported"], good)
    np.testing.assert_allclose(actual["scatter_eigenvalues"], eigenvalues, atol=1e-9, rtol=1e-9)
    separated = good & ((eigenvalues[:, 1]-eigenvalues[:, 0]) > 1e-10*eigenvalues[:, 2])
    np.testing.assert_allclose(actual["shapes"][separated], shapes[separated], atol=1e-9, rtol=1e-8)
    np.testing.assert_array_equal(actual["shapes"][~good], np.zeros((int((~good).sum()), 3, 3)))
    residual_error = 0.
    if good.any():
        shape_values, shape_vectors = np.linalg.eigh(actual["shapes"][good])
        np.testing.assert_allclose(shape_values, np.tile([.001, 1., 1.], (int(good.sum()), 1)), atol=2e-12, rtol=1e-12)
        normal = shape_vectors[:, :, 0]
        residual = (scatter[good] @ normal[..., None])[..., 0] - eigenvalues[good, 0, None]*normal
        residual_error = float(np.max(np.linalg.norm(residual, axis=1)/eigenvalues[good, 2]))
        assert residual_error <= 1e-10
    return dict(points=n, supported=int(good.sum()), unsupported=int((~good).sum()),
                boundary_ties=boundary_ties, ambiguous_minimum_eigenspaces=int((good & ~separated).sum()),
                max_neighbor_distance_error=distance_error,
                max_eigenvalue_error=float(np.max(np.abs(actual["scatter_eigenvalues"]-eigenvalues), initial=0)),
                max_separated_shape_error=float(np.max(np.abs(actual["shapes"][separated]-shapes[separated]), initial=0)),
                max_relative_eigenpair_residual=residual_error)


def load_registration(data):
    result = json.loads(str(data["metadata_json"]))
    for name in ["pose", "matched_source_indices", "matched_target_indices", "squared_distances", "weights"]:
        result[name] = data[name].copy()
    for i, row in enumerate(result["history"]):
        prefix = f"iteration_{i:02d}_"
        row.update({name[len(prefix):]: data[name].copy() for name in data if name.startswith(prefix)})
    return result


def audit(inputs_directory, output, slice_protocol, build_inputs):
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    import audit_flatwall_raw_point_time as fields
    from importlib.metadata import version

    if output.exists() or output.is_relative_to(inputs_directory):
        raise ValueError("use a fresh output outside frozen evidence")
    if os.environ.get("RAYON_NUM_THREADS") != "4" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("use a fresh four-Rayon, one-OpenBLAS process")
    frozen = {p.relative_to(inputs_directory).as_posix(): fields.artifact(p) for p in sorted(inputs_directory.rglob("*")) if p.is_file()}
    controller, binary = fields.artifact(Path(__file__)), fields.artifact(Path(native.__file__))
    protocol = dict(schema="cloudanalyzer.flatwall_gicp_shapes.v1", slice=factor.read(slice_protocol),
                    input_artifacts=frozen, controller=controller, native=binary,
                    build_inputs=fields.artifact(build_inputs), slice_protocol=fields.artifact(slice_protocol),
                    cases=registration.CASES, rows=registration.ROWS,
                    reference_dependencies={p.name: fields.artifact(p) for p in [Path(factor.__file__), Path(registration.__file__)]},
                    versions={name: version(name) for name in ["numpy", "scipy"]})
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output/"protocol.json", protocol)
    params = dict(max_correspondence_distance=2., max_iterations=30, translation_tolerance=1e-4,
                  rotation_tolerance=1e-5, robust_scale=None)
    frames, summaries = [], []
    for case in registration.CASES:
        destination = output/case
        destination.mkdir()
        case_rows = []
        for row in registration.ROWS:
            with np.load(inputs_directory/case/f"frame-{row:02d}.npz", allow_pickle=False) as file:
                data = {k: file[k].copy() for k in file.files}
            source, target = data["source"], data["target"]
            initial, center = data["initial_pose"], data["center_world"]
            old = load_registration(data)
            control = core.gicp_register(source, target, data["source_shapes"], data["target_shapes"], initial, center, capture_history=True, **params)
            equal_bits(control, old)
            shape_outputs, shape_stats = [], []
            saved = {}
            for name, points, old_shapes in [("source", source, data["source_shapes"]), ("target", target, data["target_shapes"])]:
                serial = core.gicp_shapes(points, capture_neighbors=True)
                parallel = core.gicp_shapes(points, parallel=True, capture_neighbors=True)
                equal_bits(serial, parallel)
                plain = core.gicp_shapes(points)
                equal_bits({k:v for k,v in serial.items() if k not in ["neighbor_indices", "neighbor_squared_distances"]},
                           {k:v for k,v in plain.items() if k not in ["neighbor_indices", "neighbor_squared_distances"]})
                assert plain["neighbor_indices"].shape == (len(points), 0)
                stats = check_shapes(points, serial)
                stats["max_old_scipy_shape_difference"] = float(np.max(np.abs(serial["shapes"]-old_shapes), initial=0))
                shape_stats.append(stats)
                shape_outputs.append(serial)
                saved.update({name+"_"+key: value for key, value in serial.items()})
            cs, ct = shape_outputs
            sg, tg = cs["supported"], ct["supported"]
            args = (source[sg], target[tg], cs["shapes"][sg], ct["shapes"][tg], initial, center)
            actual = core.gicp_register(*args, capture_history=True, **params)
            parallel = core.gicp_register(*args, capture_history=True, parallel=True, **params)
            equal_bits(actual, parallel)
            expected = registration.reference(*args, **params)
            registration.compare(actual, expected)
            np.savez_compressed(destination/f"frame-{row:02d}-shapes.npz", **saved)
            registration.save_result(destination/f"frame-{row:02d}.npz", dict(source=args[0], target=args[1], source_shapes=args[2],
                target_shapes=args[3], initial_pose=initial, center_world=center, retained_source_indices=np.flatnonzero(sg),
                retained_target_indices=np.flatnonzero(tg)), actual)
            item = dict(case=case, row=row, source=shape_stats[0], target=shape_stats[1],
                        termination=actual["termination"], old_termination=old["termination"],
                        iterations=actual["iterations"], old_iterations=old["iterations"], accepted_steps=actual["accepted_steps"],
                        trial_count=sum(len(s["trial_scales"]) for s in actual["history"]), final_matches=len(actual["weights"]),
                        native_reference_pose_max_error=float(np.max(np.abs(actual["pose"]-expected["pose"]))),
                        old_pose_max_difference=float(np.max(np.abs(actual["pose"]-old["pose"]))),
                        old_pose_translation_difference_m=float(np.linalg.norm(actual["pose"][:3, 3]-old["pose"][:3, 3])),
                        serial_parallel_bit_equal=True, old_caller_shape_result_bit_equal=True)
            case_rows.append(item)
            print(json.dumps(item), flush=True)
        fields.save(destination/"frames.json", case_rows)
        frames.extend(case_rows)
        summaries.append(dict(case=case, registrations=len(case_rows), statuses=dict(Counter(r["termination"] for r in case_rows)),
            old_statuses=dict(Counter(r["old_termination"] for r in case_rows)),
            changed_terminations=sum(r["termination"] != r["old_termination"] for r in case_rows),
            shape_points=sum(r[p]["points"] for r in case_rows for p in ["source", "target"]),
            unsupported=sum(r[p]["unsupported"] for r in case_rows for p in ["source", "target"]),
            boundary_ties=sum(r[p]["boundary_ties"] for r in case_rows for p in ["source", "target"]),
            ambiguous_minimum_eigenspaces=sum(r[p]["ambiguous_minimum_eigenspaces"] for r in case_rows for p in ["source", "target"]),
            max_shape_error=max(r[p]["max_separated_shape_error"] for r in case_rows for p in ["source", "target"]),
            max_old_scipy_shape_difference=max(r[p]["max_old_scipy_shape_difference"] for r in case_rows for p in ["source", "target"]),
            max_pose_reference_error=max(r["native_reference_pose_max_error"] for r in case_rows),
            max_old_pose_difference=max(r["old_pose_max_difference"] for r in case_rows),
            iterations=sum(r["iterations"] for r in case_rows), trial_count=sum(r["trial_count"] for r in case_rows),
            artifacts={p.name: fields.artifact(p) for p in sorted(destination.iterdir())}))
    fields.save(output/"frames.json", frames)
    fields.save(output/"summary.json", summaries)
    for name, expected in frozen.items():
        assert fields.artifact(inputs_directory/name) == expected, name
    assert controller == fields.artifact(Path(__file__)) and binary == fields.artifact(Path(native.__file__))
    receipt = dict(schema=protocol["schema"], protocol=fields.artifact(output/"protocol.json"),
        controller=controller, native=binary, verified_registrations=len(frames), verified_point_sets=2*len(frames),
        verified_shape_points=sum(s["shape_points"] for s in summaries),
        serial_parallel_bit_equal=True, old_caller_shape_results_bit_equal=True,
        frozen_input_artifacts_unchanged=len(frozen), no_trajectory_or_frontend_adoption=True,
        outputs={p.name: fields.artifact(p) for p in sorted(output.iterdir()) if p.is_file()})
    fields.save(output/"receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["input", "out", "slice-protocol", "build-inputs"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input.resolve(), args.out.resolve(), args.slice_protocol.resolve(), args.build_inputs.resolve())
    print(json.dumps({k:v for k,v in result.items() if k != "outputs"}))


if __name__ == "__main__":
    main()
