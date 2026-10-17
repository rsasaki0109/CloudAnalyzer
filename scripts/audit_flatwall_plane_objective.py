#!/usr/bin/env python3
"""Evaluate a target-normal surrogate on frozen matches and reconstructed maps."""
from collections import Counter
import argparse
import json
import os
from pathlib import Path

import numpy as np

import audit_flatwall_frontend_controls as common
import audit_flatwall_odometry_linearization as independent
import flatwall_plane_objective as candidate

ROOT = Path(__file__).resolve().parents[1]


def summary(rows):
    usable = [r for r in rows if r["plane_geometry"] is not None]
    counts = Counter()
    for row in rows: counts.update(row["normal_status_counts"])
    return dict(scans=len(rows), original_matches=sum(r["matches"] for r in rows),
        accepted_normal_matches=sum(r["accepted_normal_matches"] for r in rows), normal_status_counts=dict(counts),
        supported_systems=len(usable), no_plane_system_scan_indices=[r["scan_index"] for r in rows if r["plane_geometry"] is None],
        plane_weak_systems=sum(r["plane_geometry"]["weak_directions_below_1e_3"] > 0 for r in usable),
        same_subset_point_weak_systems=sum(r["point_geometry"]["weak_directions_below_1e_3"] > 0 for r in usable),
        plane_more_weak_directions=sum(r["plane_geometry"]["weak_directions_below_1e_3"] > r["point_geometry"]["weak_directions_below_1e_3"] for r in usable),
        median_normal_fraction=float(np.median([r["accepted_normal_matches"]/r["matches"] for r in rows if r["matches"]])) if any(r["matches"] for r in rows) else None,
        maximum_plane_matrix_oracle_error=max(r["plane_matrix_oracle_error"] for r in rows),
        maximum_plane_rhs_oracle_error=max(r["plane_rhs_oracle_error"] for r in rows))


def exhaustive_neighbors(cloud, target):
    difference = cloud-target
    squared = np.einsum("ni,ni->n", difference, difference)
    ids = np.flatnonzero(squared <= 1.)
    order = np.lexsort((ids, squared[ids]))[:12]
    result = np.full(12, -1, dtype=np.int64)
    result[:len(order)] = ids[order]
    return result


def audit(protocol_path, freeze_path, out):
    protocol, freeze = common.read(protocol_path), common.read(freeze_path)
    if (protocol["schema"] != "cloudanalyzer.flatwall_plane_objective.v1" or protocol["normal_policy"] != candidate.NORMAL_POLICY
            or protocol["workers"] != 4 or protocol["state_feedback"] is not False
            or len(protocol["cases"]) != 16 or os.environ.get("OPENBLAS_NUM_THREADS") != "1"):
        raise ValueError("unsupported frozen plane-objective policy")
    expected = dict(freeze["files_sha256"])
    expected[protocol_path.resolve().relative_to(ROOT).as_posix()] = freeze["protocol_sha256"]
    expected[freeze_path.resolve().relative_to(ROOT).as_posix()] = common.sha(freeze_path)
    for source in [__file__, candidate.__file__, common.__file__, independent.__file__]:
        if Path(source).resolve().relative_to(ROOT).as_posix() not in expected:
            raise ValueError("freeze every source before real geometry analysis")
    common.verify(expected)
    out.mkdir(parents=True, exist_ok=False)
    results, catalog, oracle_rows, total = {}, [], [], 0
    try:
        for case in protocol["cases"]:
            folder, replay, rows = out/case["name"], independent.MapReplay(), []
            folder.mkdir(parents=True)
            source = common.checked_path(case["directory"])
            frames = [json.loads(s) for s in common.checked_path(case["frames"]).read_text(encoding="utf-8").splitlines()]
            if [r["scan_index"] for r in frames] != case["scan_indices"]: raise ValueError("frozen frame identities changed")
            sample_ordinals = {1, len(frames)//2, len(frames)-1}
            for ordinal, frame in enumerate(frames):
                index = frame["scan_index"]
                original = source/f"scan-{index:03d}.npz"
                scalars = common.read(original.with_suffix(".json"))
                if scalars["map_voxel"] != 1. or scalars["map_points"] != 20 or scalars["max_range"] != 80.:
                    raise ValueError("map replay policy differs from captured map")
                with np.load(original, allow_pickle=False) as saved: snapshot = {k: saved[k].copy() for k in saved.files}
                snapshot.update(scalars)
                np.testing.assert_array_equal(snapshot["pose"].view(np.uint64), np.array(frame["returned_pose"]).view(np.uint64))
                weights, old_h, old_b, _, _ = independent.reconstruct(snapshot)
                np.testing.assert_allclose(snapshot["weights"], weights, rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(snapshot["normal_matrix"], old_h, rtol=2e-12, atol=2e-8)
                np.testing.assert_allclose(snapshot["right_hand_side"], old_b, rtol=2e-12, atol=2e-8)
                cloud = candidate.canonical_map(np.array([p for cell in replay.cells.values() for p in cell]).reshape(-1, 3))
                q, p = snapshot["source_world"][snapshot["matched_source_indices"]], snapshot["target_world"]
                membership = {point.tobytes() for point in cloud}
                if any(point.tobytes() not in membership for point in p): raise ValueError("captured target absent from reconstructed pre-insertion map")
                neighbors = candidate.neighborhoods(cloud, p, workers=4)
                normal = candidate.normal_geometry(cloud, neighbors)
                mask = normal["accepted"]
                system = candidate.systems(q[mask], p[mask], normal["normals"][mask], weights[mask], snapshot["center_world"])
                # Independent scalar-column Jacobian and einsum accumulation.
                arm = q[mask]-snapshot["center_world"]
                nx, ny, nz = normal["normals"][mask].T
                ax, ay, az = arm.T
                jacobian = np.array([nx, ny, nz, ay*nz-az*ny, az*nx-ax*nz, ax*ny-ay*nx]).T
                residual = np.einsum("ni,ni->n", normal["normals"][mask], q[mask]-p[mask])
                h = np.einsum("ni,n,nj->ij", jacobian, weights[mask], jacobian)
                b = -np.einsum("ni,n,n->i", jacobian, weights[mask], residual)
                np.testing.assert_allclose(system["plane_normal_matrix"], h, rtol=2e-12, atol=2e-8)
                np.testing.assert_allclose(system["plane_right_hand_side"], b, rtol=2e-12, atol=2e-8)
                arrays = dict(source_world=q, target_world=p, center_world=snapshot["center_world"], weights=weights,
                    neighbors=neighbors, **{k: v for k, v in normal.items() if isinstance(v, np.ndarray)},
                    **{k: v for k, v in system.items() if isinstance(v, np.ndarray)})
                # Public samples include complete canonical maps; full private outputs need no duplicate map bytes.
                if ordinal in sample_ordinals: arrays["map_points"] = cloud
                target = folder/f"scan-{index:03d}.npz"
                with target.open("xb") as stream: np.savez_compressed(stream, **arrays)
                row = dict(case=case["name"], scan_index=index, timestamp_ns=frame["reference_ns"],
                    original_snapshot_sha256=common.sha(original), map_points=len(cloud), matches=len(q), accepted_normal_matches=int(mask.sum()),
                    normal_status_counts=dict(Counter(normal["labels"].tolist())),
                    center_world=snapshot["center_world"].tolist(),
                    **{k: v.tolist() if isinstance(v, np.ndarray) else ({a: b for a, b in v.items() if a != "eigenvectors"} if isinstance(v, dict) else v)
                       for k, v in system.items() if k not in {"plane_residual", "plane_jacobian"}},
                    plane_matrix_oracle_error=float(np.max(np.abs(h-system["plane_normal_matrix"]), initial=0.)),
                    plane_rhs_oracle_error=float(np.max(np.abs(b-system["plane_right_hand_side"]), initial=0.)))
                rows.append(row)
                with (folder/"frames.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                catalog.append(dict(case=case["name"], scan_index=index, file=target.relative_to(out).as_posix(),
                    bytes=target.stat().st_size, sha256=common.sha(target), public_sample=ordinal in sample_ordinals))
                if ordinal in sample_ordinals:
                    chosen = sorted({0, len(q)//2, len(q)-1}) if len(q) else []
                    for query in chosen:
                        exact = exhaustive_neighbors(cloud, p[query])
                        np.testing.assert_array_equal(neighbors[query], exact)
                    oracle_rows.append(dict(case=case["name"], scan_index=index, selected_match_rows=chosen,
                        exhaustive_neighborhoods_checked=len(chosen), no_matches=len(q) == 0))
                replay.insert(snapshot["inserted_world"], snapshot["pose"][:3, 3])
                total += 1
            results[case["name"]] = summary(rows)
            print(json.dumps(dict(case=case["name"], scans=len(rows), supported=results[case["name"]]["supported_systems"], total=total)), flush=True)
        common.verify(expected)
        result = dict(schema="cloudanalyzer.flatwall_plane_objective_result.v1", frames=total, cases=results,
            original_poses_and_matches_unchanged=True, state_feedback=False, reference_poses_used=False,
            map_insertion_and_membership_replayed=True, map_nearest_match_search_independently_replayed=False,
            physical_observability_certified=False, speed_claim=False)
        common.save(out/"summary.json", result)
        common.save(out/"snapshot-catalog.json", catalog)
        common.save(out/"neighborhood-oracle.json", dict(samples=oracle_rows,
            selected_rule="Cohort ordinal 1, floor(N/2), N-1; match rows 0, floor(M/2), M-1, without replacement. Empty samples remain explicit.",
            queries_checked=sum(r["exhaustive_neighborhoods_checked"] for r in oracle_rows)))
        common.save(out/"receipt.json", dict(at_utc=common.now(), protocol_sha256=common.sha(protocol_path), controller_freeze_sha256=common.sha(freeze_path),
            output_files_sha256={p.relative_to(out).as_posix(): common.sha(p) for p in sorted(out.rglob("*")) if p.is_file()}))
    except Exception as exc:
        common.save(out/"failure.json", dict(at_utc=common.now(), error_type=type(exc).__name__, error=str(exc), completed_frames=total))
        common.save(out/"partial-summary.json", results)
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["protocol", "freeze", "out"]: parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.protocol, args.freeze, args.out), indent=2))
