"""Verify saved map/audit evidence; full point clouds are not re-audited here."""
import hashlib
import json
from pathlib import Path

import numpy as np

from ca.mapping_connections import _inside, _width, edges, route_metrics
from ca.mapping_patch import _checks, _preserved

root = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

manifest = read(root / 'files-sha256.json')
actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
          and p.name != 'files-sha256.json' and '__pycache__' not in p.parts}
assert actual == manifest.keys(), 'packet membership changed'
for name, expected in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

for label in ('april', 'june'):
    folder = root / label
    before = read(folder / 'before/vector_map.json')
    after = read(folder / 'after/vector_map.json')
    reports = [read(folder / mode / 'report.json') for mode in ('before', 'after')]
    old_ids = {l['id'] for l in before['lanes']}
    new_ids = {l['id'] for l in after['lanes']} - old_ids
    extra = edges(after) - edges(before)
    assert new_ids == (set() if label == 'april' else {54})
    assert extra == (set() if label == 'april' else {(6, 54), (54, 51)})
    _preserved(before, after, extra)
    for ir, report in zip((before, after), reports):
        intervals = {int(k): tuple(v) for k, v in report['lane_intervals'].items()}
        assert route_metrics(ir, intervals) == report['routes']['after']
    assert reports[0]['extent'] == reports[1]['extent']
    assert reports[0]['station_disposition'] == reports[1]['station_disposition']
    audits = [read(folder / mode / 'source-quality.json') for mode in ('before', 'after')]
    checks = _checks(*audits, old_ids, new_ids)
    assert checks == read(folder / 'connection-checks.json') and checks['passes']
    candidates = read(folder / 'connection-proposal.json')['candidates']
    assert len(candidates) == (0 if label == 'april' else 1)
    if label == 'june':
        c = candidates[0]
        assert (c['from'], c['to'], c['from_m'], c['to_m']) == (6, 51, 18, 22)
        assert _inside(np.array(c['trajectory_xy']), np.array(c['left'] + c['right'][::-1])[:, :2])
        assert _width(c) >= 2.
        routes = reports[1]['routes']['after']
        assert routes['connected_components'] == 12
        assert routes['longest_route_station_span_m'] == 66
        assert any(r['lane_ids'] == [6, 54, 51] and r['station_span_m'] == 14 for r in routes['routes'])
print('Verified both saved packets: hashes, retained geometry/edges, station routes and four audits.')
