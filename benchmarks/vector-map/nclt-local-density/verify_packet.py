"""Check saved density evidence; optionally verify full generated maps/native audits."""
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


def totals(audits):
    return {name: {'samples': audit['quality']['sampled_points'],
                   'supported': sum(lane[side]['supported'] for lane in audit['quality']['lanes']
                                    for side in ('center', 'left', 'right'))}
            for name, audit in [('legacy', audits['editable']),
                                ('consensus', audits['ground_consensus']['editable'])]}


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
    old_ids = {lane['id'] for lane in before['lanes']}
    new_ids = {lane['id'] for lane in after['lanes']} - old_ids
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
    assert checks == read(folder / 'local-checks.json') == run['checks'] and checks['passes']
    assert totals(before_audits) == run['baseline_hd_audits']
    assert totals(local_audits) == run['local_attempt_hd_audits']
    preview, update = read(folder / 'local-preview.json'), read(folder / 'local-update.json')
    assert preview['request']['strategy'] == 'density' and preview['eligible_frames'] == []
    assert preview['outside_records_sha256'] == update['outside_records_sha256'] == run['outside_records_sha256']
    assert not update['holds'] and update['effective_bounds_xy'] == box
    for name, count in [('before', run['before_inside_points']), ('candidate', run['candidate_inside_points'])]:
        _, rows = records(folder / f'inside-{name}.ply')
        assert len(rows) == count and mask(rows, box).all()
    point_report = read(folder / 'pointcloud-report.json')
    assert point_report['added_frame_ids'] == [] and run['added_frames'] == 0
    ids = point_report['original_retained_frame_ids']
    assert ids == sorted(set(ids)) and len(ids) == run['original_retained_frames'] == point_report['nodes'] == point_report['scans']
    assert point_report['full_fusion_map_points'] == run['full_fusion_map_points']
    for mode, ir in [('before', before), ('after', after)]:
        report = read(folder / mode / 'report.json')
        intervals = {int(k): tuple(v) for k, v in report['lane_intervals'].items()}
        assert route_metrics(ir, intervals) == run[f'{mode}_routes'] == report['routes']['after']
        assert report['extent']['generated_length_m'] == run[f'{mode}_hd_extent_m']
    addition_audits = read(folder / 'addition/source-quality.json')
    assert totals(addition_audits) == run['new_addition_audits']
    assert addition_audits['editable']['quality'] == addition_audits['reopened_osm']['quality']
    assert addition_audits['ground_consensus']['editable']['quality'] == addition_audits['ground_consensus']['reopened_osm']['quality']
    if label == 'april':
        final_checks = _checks(before_audits, read(folder / 'after/source-quality.json'), old_ids, new_ids)
        assert final_checks == read(folder / 'patch-checks.json') and final_checks['passes']
        comparison = read(folder / 'comparison.json')
        assert comparison['retry_strategy'] == 'local_density'
        assert comparison['gained_source_length_m'] == 4 and comparison['lost_source_length_m'] == 0
    else:
        assert run['new_addition_audits'] == {'legacy': {'samples': 46, 'supported': 45}, 'consensus': {'samples': 46, 'supported': 46}}
        problems = addition_audits['editable']['quality']['problems']
        assert len(problems) == 1 and problems[0]['curve'] == 'right' and problems[0]['reason'] == 'height_mismatch'
        for filename in ('vector_map.json', 'lanelet2_map.osm', 'map_projector_info.yaml', 'source-quality.json', 'report.json'):
            assert (folder / 'before' / filename).read_bytes() == (folder / 'after' / filename).read_bytes()
    actions = read(folder / 'agent-actions.json')
    assert actions['root']['output']['pointcloud_retry_decision']['adopted'] == run['adopted'] == (label == 'april')
    if label == 'june':
        assert actions['child']['output']['candidate_id'] is None
        assert not any(h['action']['type'] == 'patch_gaps' for h in actions['child']['actions'])

    if args.generated_root:
        import cloudanalyzer_core as native
        generated = args.generated_root / f'nclt-local-density-{label}'

        def generated_file(artifact):
            path = generated / Path(artifact['path']).relative_to('run')
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
        graph = native.PoseGraph.from_g2o(generated_file(run['baseline_point_files']['graph']).read_text())
        assert list(graph.node_ids) == ids
        for key in ('graph', 'trajectory'):
            relative = Path(point_report['outputs']['g2o' if key == 'graph' else 'kitti']).relative_to('run')
            assert sha(generated / relative) == run['baseline_point_files'][key]['sha256']
        repeated = {}
        for estimator, function in [('legacy', native.audit_vector_map_quality_details),
                                    ('coherent', native.audit_vector_map_ground_consensus_details)]:
            audits = {key: json.loads(function(str(local_path), str(generated_file(update['baseline_hd_files'][fkey]))))
                      for key, fkey in [('editable', 'editable_map'), ('reopened_osm', 'map')]}
            if estimator == 'legacy':
                repeated.update(audits)
            else:
                repeated['ground_consensus'] = audits
        for key in ('editable', 'reopened_osm'):
            assert repeated[key]['quality'] == local_audits[key]['quality']
            assert repeated['ground_consensus'][key]['quality'] == local_audits['ground_consensus'][key]['quality']
        assert _checks(before_audits, repeated, old_ids, set()) == checks
        assert generated_file(run['final_point_map']) == (local_path if run['adopted'] else base_path)

print('Verified saved hashes, inside records, fixed frame IDs, retained HD, routes and actual adoption decisions.')
print('Also verified full outside records/reference bytes and repeated four native audits.' if args.generated_root else
      'Full outside equality and native audits were not recomputed; use --generated-root with the full outputs.')
