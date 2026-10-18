#!/usr/bin/env python3
"""Prepare fixed binary states from the already published real recordings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(root, output):
    output.mkdir(parents=True, exist_ok=False)
    trial = root/'benchmarks/slam/flatwall-target-balance'
    dynamics = root/'benchmarks/slam/flatwall-balance-dynamics'
    catalog = [r for r in read(trial/'snapshot-catalog.json') if r['public_sample']]
    traces = read(dynamics/'summary.json')['traces']
    assert len(catalog) == 48 and len(traces) == 8
    definitions, origins, inputs = [], [], {}

    def add(name, path, source, cloud, pose, center, threshold, kernel, origin, expected):
        assert source.ndim == cloud.ndim == 2 and source.shape[1] == cloud.shape[1] == 3
        assert pose.shape == (4, 4) and center.shape == (3,)
        values = np.concatenate([pose.ravel(), center, [threshold, kernel, 1., 20.], source.ravel(), cloud.ravel()]).astype('<f8')
        assert np.isfinite(values).all()
        binary = f'state-{len(definitions):03d}.f64'
        with (output/binary).open('xb') as stream:
            stream.write(values.tobytes())
        definitions.append(dict(name=name, file=binary, source_points=len(source), map_points=len(cloud)))
        origin.update(name=name, original_npz=path.relative_to(root).as_posix(), original_npz_sha256=sha(path),
            binary_file=binary, binary_sha256=sha(output/binary), expected_system=expected)
        origins.append(origin)
        inputs[path.relative_to(root).as_posix()] = sha(path)

    for row in catalog:
        relative = next(n for n in row['files'] if n.endswith('.npz'))
        path = trial/relative
        assert sha(path) == row['files'][relative]['sha256']
        metadata = path.with_suffix('.json')
        inputs[metadata.relative_to(root).as_posix()] = sha(metadata)
        config = read(metadata)
        with np.load(path, allow_pickle=False) as data:
            add(row['case']+f"/scan-{row['scan_index']:03d}/final", path,
                data['source_sensor'], data['map_points_world'], data['pose'], data['center_world'],
                config['threshold_sq'], config['kernel'],
                dict(case=row['case'], scan_index=row['scan_index'], kind='final', metadata=metadata.relative_to(root).as_posix()),
                'final-pose captured balanced H/b and ordered pairs')
    for trace in traces:
        path = dynamics/trace['file']
        assert sha(path) == trace['sha256']
        with np.load(path, allow_pickle=False) as data:
            for step in [0, 499]:
                add(trace['case']+f"/scan-{trace['scan_index']:03d}/step-{step:04d}", path,
                    data['source_at_prediction'], data['map_points_world'], data['before'][step], data['center_world'],
                    float(data['threshold_sq']), float(data['kernel']),
                    dict(case=trace['case'], scan_index=trace['scan_index'], kind='step', step=step),
                    'attempted native balanced H/b; tail ordered pairs also retained for step 499')
    for path in [trial/'snapshot-catalog.json', dynamics/'summary.json']:
        inputs[path.relative_to(root).as_posix()] = sha(path)
    save(output/'input.json', dict(schema='cloudanalyzer.odometry_fixed_states.v1', repetitions=32,
        warmup_rounds=1, measured_rounds=5, states=definitions))
    save(output/'origins.json', origins)
    save(output/'receipt.json', dict(inputs_sha256=inputs, states=64,
        outputs_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        binary_layout='little-endian f64: row-major 4x4 pose, center xyz, threshold_sq, kernel, map_voxel=1, max_points=20, source xyz, map xyz',
        selection='All 48 predetermined public final-pose states, then steps 0 and 499 of all eight capped traces; no timing-based selection.',
        reconstruction='Insert each stored map in its stored order; verify no voxel capacity loss and verify ordered selected targets against original snapshots.',
        reference_poses_read=False, estimator_or_default_changed=False))
    return dict(states=len(definitions), source_points=sum(d['source_points'] for d in definitions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve(), args.output.resolve())))


if __name__ == '__main__':
    main()
