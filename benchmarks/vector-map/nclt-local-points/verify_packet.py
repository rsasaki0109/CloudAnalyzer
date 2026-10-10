"""Verify saved evidence; optionally check full generated records/native audits."""
import argparse
import hashlib
import json
from pathlib import Path

from ca.mapping_connections import edges, route_metrics
from ca.mapping_local_points import inside_geometry, mask, records
from ca.mapping_patch import _checks, _preserved

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--generated-root', type=Path)
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(root / 'files-sha256.json')
actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
          and p.name != 'files-sha256.json' and '__pycache__' not in p.parts}
assert actual == manifest.keys(), 'packet membership changed'
for name, expected in manifest.items():
    assert sha(root / name) == expected, name

for run in read(root / 'verification.json')['runs']:
    label = run['label']
    folder = root / label
    before = read(folder / 'before/vector_map.json')
    after = read(folder / 'after/vector_map.json')
    old_ids = {l['id'] for l in before['lanes']}
    new_ids = {l['id'] for l in after['lanes']} - old_ids
    assert new_ids == ({42} if label == 'april' else set())
    _preserved(before, after)
    assert edges(before) == edges(after)
    box = run['effective_bounds_xy']
    old_boundaries = {b['id'] for b in before['boundaries']}
    assert all(inside_geometry(b['geometry'], box) for b in after['boundaries']
               if b['id'] not in old_boundaries)
    before_audits = read(folder / 'before/source-quality.json')
    local_audits = read(folder / 'local-audits.json')
    checks = _checks(before_audits, local_audits, old_ids, set())
    assert checks == read(folder / 'local-checks.json') == run['checks']
    assert checks['passes'] == run['adopted'] == (label == 'april')
    preview, update = read(folder / 'local-preview.json'), read(folder / 'local-update.json')
    assert preview['outside_records_sha256'] == update['outside_records_sha256'] == run['outside_records_sha256']
    assert update['holds'] == checks['holds'] and update['effective_bounds_xy'] == box
    for name, count in [('before', run['before_inside_points']), ('candidate', run['candidate_inside_points'])]:
        _, rows = records(folder / f'inside-{name}.ply')
        assert len(rows) == count and mask(rows, box).all()
    for mode, ir in [('before', before), ('after', after)]:
        report = read(folder / mode / 'report.json')
        intervals = {int(k): tuple(v) for k, v in report['lane_intervals'].items()}
        assert route_metrics(ir, intervals) == run[f'{mode}_routes'] == report['routes']['after']
        assert report['extent']['generated_length_m'] == run[f'{mode}_hd_extent_m']
    if label == 'april':
        final_checks = _checks(before_audits, read(folder / 'after/source-quality.json'), old_ids, new_ids)
        assert final_checks == read(folder / 'patch-checks.json') and final_checks['passes']
        comparison = read(folder / 'comparison.json')
        assert comparison['gained_source_length_m'] == 4 and comparison['lost_source_length_m'] == 0
    else:
        assert any('new_retained_failure_location:39:left' in h for h in checks['holds'])
        for filename in ('vector_map.json', 'lanelet2_map.osm', 'map_projector_info.yaml', 'source-quality.json', 'report.json'):
            assert (folder / 'before' / filename).read_bytes() == (folder / 'after' / filename).read_bytes()
    frame_checks = read(folder / 'selected-frame-checks.json')['frames']
    assert [f['frame_id'] for f in frame_checks] == run['selected_frames']
    assert all(f['eligible'] for f in frame_checks)
    actions = read(folder / 'agent-actions.json')
    assert actions['root']['output']['pointcloud_retry_decision']['adopted'] == run['adopted']

    if args.generated_root:
        import cloudanalyzer_core as native
        generated = args.generated_root / f'nclt-local-points-{label}'

        def generated_file(artifact):
            relative = Path(artifact['path']).relative_to('run')
            path = generated / relative
            assert path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256']
            return path

        base_path = generated_file(update['baseline_map'])
        candidate_path = generated_file(update['full_fusion_candidate'])
        local_path = generated_file(update['updated_map'])
        _, base = records(base_path)
        _, candidate = records(candidate_path)
        _, merged = records(local_path)
        a, b, c = mask(base, box), mask(candidate, box), mask(merged, box)
        assert base.dtype == candidate.dtype == merged.dtype
        assert base[~a].tobytes() == merged[~c].tobytes()
        assert candidate[b].tobytes() == merged[c].tobytes()
        assert hashlib.sha256(merged[~c].tobytes()).hexdigest() == run['outside_records_sha256']
        assert len(merged[~c]) == run['outside_points'] and len(merged) == run['candidate_local_map_points']
        for name, rows in [('before', base[a]), ('candidate', candidate[b])]:
            assert records(folder / f'inside-{name}.ply')[1].tobytes() == rows.tobytes()
        repeated = {}
        for estimator, audit_function in [('legacy', native.audit_vector_map_quality_details),
                                           ('coherent', native.audit_vector_map_ground_consensus_details)]:
            audits = {key: json.loads(audit_function(str(local_path), str(generated_file(update['baseline_hd_files'][fkey]))))
                      for key, fkey in [('editable', 'editable_map'), ('reopened_osm', 'map')]}
            if estimator == 'legacy':
                repeated.update(audits)
            else:
                repeated['ground_consensus'] = audits
        # Saved reports normalize machine paths; compare the audit's actual quality evidence.
        for key in ('editable', 'reopened_osm'):
            assert repeated[key]['quality'] == local_audits[key]['quality']
            assert repeated['ground_consensus'][key]['quality'] == local_audits['ground_consensus'][key]['quality']
        assert _checks(before_audits, repeated, old_ids, set()) == checks
        assert generated_file(run['final_point_map']) == (local_path if run['adopted'] else base_path)

print('Verified saved hashes, inside records, retained HD geometry/edges, station routes and audit decisions.')
print('Also verified full outside records and repeated four native audits.' if args.generated_root else
      'Full outside equality and native audits were not recomputed; use --generated-root with the full outputs.')
