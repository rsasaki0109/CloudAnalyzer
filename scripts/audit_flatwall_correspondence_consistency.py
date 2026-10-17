#!/usr/bin/env python3
"""Audit reuse and reciprocal checks on every frozen real forward match."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import audit_flatwall_frontend_controls as common
import audit_flatwall_odometry_linearization as independent
import flatwall_correspondence_consistency as candidate

ROOT = Path(__file__).resolve().parents[1]


def exhaustive_reverse(source, target):
    result = np.empty(len(target), dtype=np.int64)
    for start in range(0, len(target), 64):
        delta = source[None, :, :]-target[start:start+64, None, :]
        squared = np.sum(delta*delta, axis=2)
        result[start:start+64] = np.argmin(squared, axis=1)
    return result


def summarize(rows):
    matched = [r for r in rows if r['matches']]
    return dict(scans=len(rows), matches=sum(r['matches'] for r in rows),
        unique_target_queries=sum(r['unique_targets'] for r in rows),
        saved_reverse_queries=sum(r['saved_reverse_queries'] for r in rows),
        reciprocal_matches=sum(r['reciprocal_matches'] for r in rows),
        reverse_to_unmatched_source=sum(r['reverse_to_unmatched_source'] for r in rows),
        median_reciprocal_fraction=float(np.median([r['reciprocal_matches']/r['matches'] for r in matched])) if matched else None,
        median_saved_query_fraction=float(np.median([r['saved_reverse_queries']/r['matches'] for r in matched])) if matched else None,
        median_retained_weight_fraction=float(np.median([r['reciprocal']['weight_sum']/r['original']['weight_sum'] for r in matched])) if matched else None,
        supported_original_systems=sum(r['original']['geometry'] is not None for r in rows),
        supported_reciprocal_systems=sum(r['reciprocal']['geometry'] is not None for r in rows),
        weak_reciprocal_systems=sum(r['reciprocal']['geometry'] is not None and r['reciprocal']['geometry']['weak_directions_below_1e_3'] > 0 for r in rows),
        no_reciprocal_system_scan_indices=[r['scan_index'] for r in rows if r['reciprocal']['geometry'] is None],
        maximum_matrix_oracle_error=max(r['matrix_oracle_error'] for r in rows),
        maximum_rhs_oracle_error=max(r['rhs_oracle_error'] for r in rows))


def audit(protocol_path, freeze_path, out):
    protocol, freeze = common.read(protocol_path), common.read(freeze_path)
    if (protocol['schema'] != 'cloudanalyzer.flatwall_correspondence_consistency.v1' or protocol['policy'] != candidate.POLICY
            or protocol['state_feedback'] is not False or len(protocol['cases']) != 16 or os.environ.get('OPENBLAS_NUM_THREADS') != '1'):
        raise ValueError('unsupported consistency protocol')
    expected = dict(freeze['files_sha256'])
    expected[protocol_path.resolve().relative_to(ROOT).as_posix()] = freeze['protocol_sha256']
    expected[freeze_path.resolve().relative_to(ROOT).as_posix()] = common.sha(freeze_path)
    for source in [__file__, common.__file__, independent.__file__, candidate.__file__]:
        if Path(source).resolve().relative_to(ROOT).as_posix() not in expected: raise ValueError('freeze every controller source')
    common.verify(expected)
    out.mkdir(parents=True, exist_ok=False)
    results, catalog, total, exhaustive_queries = {}, [], 0, 0
    try:
        for case in protocol['cases']:
            folder = out/case['name']
            folder.mkdir(parents=True)
            rows = []
            frames = [json.loads(s) for s in common.checked_path(case['frames']).read_text(encoding='utf-8').splitlines()]
            if [r['scan_index'] for r in frames] != case['scan_indices']: raise ValueError('original identities differ')
            for ordinal, frame in enumerate(frames):
                index = frame['scan_index']
                original = common.checked_path(case['directory'])/f'scan-{index:03d}.npz'
                with np.load(original, allow_pickle=False) as saved: snapshot = {k: saved[k].copy() for k in saved.files}
                snapshot.update(common.read(original.with_suffix('.json')))
                source, indices, target = (snapshot[k] for k in ['source_world', 'matched_source_indices', 'target_world'])
                np.testing.assert_array_equal(snapshot['pose'].view(np.uint64), np.array(frame['returned_pose']).view(np.uint64))
                weights, old_h, old_b, _, _ = independent.reconstruct(snapshot)
                for actual, expected_value in [(snapshot['weights'], weights), (snapshot['normal_matrix'], old_h), (snapshot['right_hand_side'], old_b)]:
                    np.testing.assert_allclose(actual, expected_value, rtol=2e-12, atol=2e-8)
                diagnostics = candidate.diagnose(source, indices, target, workers=4)
                reverse = exhaustive_reverse(source, target)
                np.testing.assert_array_equal(diagnostics['reverse_source_indices'], reverse)
                np.testing.assert_array_equal(diagnostics['reciprocal'], reverse == indices)
                exhaustive_queries += len(target)
                mask = diagnostics['reciprocal']
                systems = {name: candidate.point_system(source[indices[keep]], target[keep], weights[keep], snapshot['center_world'])
                           for name, keep in [('original', np.ones(len(target), dtype=bool)), ('reciprocal', mask)]}
                subset = dict(snapshot, matched_source_indices=indices[mask], target_world=target[mask])
                _, h, b, _, _ = independent.reconstruct(subset)
                for actual, desired in [(systems['original']['normal_matrix'], old_h), (systems['original']['right_hand_side'], old_b),
                                        (systems['reciprocal']['normal_matrix'], h), (systems['reciprocal']['right_hand_side'], b)]:
                    np.testing.assert_allclose(actual, desired, rtol=2e-12, atol=2e-8)
                arrays = dict(source_world=source, matched_source_indices=indices, target_world=target,
                    center_world=snapshot['center_world'], weights=weights,
                    **{k: v for k, v in diagnostics.items() if isinstance(v, np.ndarray)})
                for name, system in systems.items():
                    arrays.update({name+'_'+k: v for k, v in system.items() if isinstance(v, np.ndarray)})
                path = folder/f'scan-{index:03d}.npz'
                with path.open('xb') as stream: np.savez_compressed(stream, **arrays)
                public_sample = ordinal in {1, len(frames)//2, len(frames)-1}
                row = dict(case=case['name'], scan_index=index, timestamp_ns=frame['reference_ns'],
                    original_snapshot_sha256=common.sha(original), source_points=len(source), matches=len(target),
                    center_world=snapshot['center_world'].tolist(), unique_targets=diagnostics['unique_targets'],
                    saved_reverse_queries=diagnostics['saved_reverse_queries'],
                    reciprocal_matches=int(mask.sum()), reused_target_rows=int(sum(diagnostics['target_multiplicity'] > 1)),
                    maximum_target_multiplicity=int(np.max(diagnostics['target_multiplicity'], initial=0)),
                    reverse_to_unmatched_source=int(sum(~np.isin(reverse, indices))),
                    original={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in systems['original'].items()},
                    reciprocal={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in systems['reciprocal'].items()},
                    matrix_oracle_error=float(np.max(np.abs(h-systems['reciprocal']['normal_matrix']), initial=0.)),
                    rhs_oracle_error=float(np.max(np.abs(b-systems['reciprocal']['right_hand_side']), initial=0.)))
                rows.append(row)
                with (folder/'frames.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                catalog.append(dict(case=case['name'], scan_index=index, file=path.relative_to(out).as_posix(),
                    bytes=path.stat().st_size, sha256=common.sha(path), public_sample=public_sample))
                total += 1
            results[case['name']] = summarize(rows)
            print(json.dumps(dict(case=case['name'], scans=len(rows), reciprocal_matches=results[case['name']]['reciprocal_matches'], total=total)), flush=True)
        common.verify(expected)
        result = dict(schema='cloudanalyzer.flatwall_correspondence_consistency_result.v1', frames=total, cases=results,
            exhaustive_reverse_queries=exhaustive_queries, original_poses_and_forward_pairs_unchanged=True,
            all_source_points_in_reverse_search=True, all_reverse_indices_independently_checked=True,
            forward_nearest_search_replayed=False, state_feedback=False, reference_poses_used=False,
            outlier_labels_certified=False, trajectory_accuracy_claim=False, speed_claim=False)
        common.save(out/'summary.json', result)
        common.save(out/'snapshot-catalog.json', catalog)
        common.save(out/'receipt.json', dict(at_utc=common.now(), protocol_sha256=common.sha(protocol_path), controller_freeze_sha256=common.sha(freeze_path),
            output_files_sha256={p.relative_to(out).as_posix(): common.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    except Exception as exc:
        common.save(out/'failure.json', dict(error_type=type(exc).__name__, error=str(exc), completed_frames=total))
        common.save(out/'partial-summary.json', results)
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'freeze', 'out']: parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.protocol, args.freeze, args.out), indent=2))
