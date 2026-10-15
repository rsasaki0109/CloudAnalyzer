#!/usr/bin/env python3
"""Verify the portable NCLT reference packet using only Python's standard library.

With --ground-truth-root, additionally check full downloaded CSV hashes and that
every published bracket row occurs numerically in the authoritative CSV.
No native core, mapping jobs, raw recordings or network are needed.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def close(a, b, tolerance=1e-9):
    assert abs(a - b) <= tolerance, (a, b)


def matrix_product(a, b):
    return [[sum(x * y for x, y in zip(row, column)) for column in zip(*b)] for row in a]


def quaternion_matrix(q):
    x, y, z, w = q
    norm = math.sqrt(sum(v*v for v in q))
    x, y, z, w = (v / norm for v in q)
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]


def text_rows(path):
    return [[float(v) for v in line.split()] for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def transformed(position, rotation, translation):
    return [sum(a*b for a, b in zip(row, position)) + t for row, t in zip(rotation, translation)]


def verify(root, ground_truth_root=None):
    lines = (root / 'SHA256SUMS').read_text().splitlines()
    named = set()
    for line in lines:
        expected, filename = line.split('  ', 1)
        assert filename not in named and not Path(filename).is_absolute() and '..' not in Path(filename).parts
        named.add(filename)
        assert digest(root / filename) == expected, filename
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'SHA256SUMS'}
    assert named == actual, (named ^ actual)
    receipt = json.loads((root / 'receipt.json').read_text())
    for name in ('april', 'june'):
        folder = root / name
        preparation = json.loads((folder / 'preparation.json').read_text())
        download = json.loads((folder / 'download.json').read_text())
        assert download['sha256'] == preparation['inputs']['ground_truth']['sha256']
        assert download['bytes'] == preparation['inputs']['ground_truth']['bytes']
        assert preparation['body_vel_xyz_m_rpy_deg'] == [.002, -.004, -.957, .807, .166, 0.]
        assert preparation['ground_truth_rpy_units'] == 'radians'
        raw = [[float(v) for v in row] for row in csv.reader((folder / 'reference-raw-rows.csv').open())]
        reference = text_rows(folder / 'reference.tum')
        original = text_rows(folder / 'original.tum')
        corrected = text_rows(folder / 'corrected-kitti.txt')
        assert len(raw) == len(reference) == preparation['selected_bracket_rows']
        calibration = rpy_matrix(*(math.radians(v) for v in [.807, .166, 0.]))
        flip = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        for row, pose in zip(raw, reference):
            close(row[0] / 1e6, pose[0], 0.)
            body = rpy_matrix(*row[4:])
            expected_position = transformed([.002, -.004, -.957], body, row[1:4])
            expected_rotation = matrix_product(matrix_product(matrix_product(flip, body), calibration), flip)
            for a, b in zip(pose[1:4], [expected_position[0], -expected_position[1], -expected_position[2]]):
                close(a, b, 1e-11)
            for a, b in zip(sum(quaternion_matrix(pose[4:]), []), sum(expected_rotation, [])):
                close(a, b, 1e-11)
        if ground_truth_root:
            source = ground_truth_root / Path(download['path']).name
            assert source.stat().st_size == download['bytes'] and digest(source) == download['sha256']
            wanted = {row[0]: row for row in raw}
            with source.open() as stream:
                for source_row in csv.reader(stream):
                    values = [float(v) for v in source_row]
                    if values[0] in wanted:
                        assert values == wanted.pop(values[0])
            assert not wanted
        graph_ids, graph_positions = [], []
        for line in (folder / 'corrected.g2o').read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == 'VERTEX_SE3:QUAT':
                graph_ids.append(int(fields[1])); graph_positions.append([float(v) for v in fields[2:5]])
        assert graph_ids == sorted(set(graph_ids)) and len(graph_ids) == len(corrected)
        for row, position in zip(corrected, graph_positions):
            for a, b in zip([row[3], row[7], row[11]], position):
                close(a, b, 1e-10)
        reference_times = [row[0] for row in reference]
        for mode in ('full', 'prefix-30'):
            path = folder / f'comparison-{mode}.json'
            report = json.loads(path.read_text())
            saved = receipt['sessions'][name]['results'][mode]
            assert digest(path) == saved['report']['sha256'] and path.stat().st_size == saved['report']['bytes']
            for key, filename in (('job', 'job-state.json'), ('source_motion_trajectory', 'original.tum'),
                                  ('pointcloud_graph', 'corrected.g2o'), ('pointcloud_trajectory', 'corrected-kitti.txt'),
                                  ('reference', 'reference.tum')):
                assert digest(folder / filename) == report['inputs'][key]['sha256']
            assert report['reference_independence'] == 'caller_declared_not_used_for_generation'
            assert report['reference_provenance']['used_for_generation'] is False
            assert report['reference_matches_recorded_inputs'] == []
            coverage = report['coverage']
            ids = coverage['matched_original_frame_ids']
            fit_ids, test_ids = coverage['alignment_original_frame_ids'], coverage['evaluated_original_frame_ids']
            assert ids == graph_ids
            assert coverage['matched_poses'] == coverage['retained_poses'] == len(ids)
            if mode == 'prefix-30':
                assert report['protocol']['held_out_alignment'] is True
                assert fit_ids + test_ids == ids and not set(fit_ids).intersection(test_ids)
                assert len(fit_ids) == int(len(ids) * .3) and len(test_ids) >= 3
            else:
                assert report['protocol']['held_out_alignment'] is False and fit_ids == test_ids == ids
            expected_reference = {}
            for frame_id in ids:
                time = original[frame_id][0]
                right = bisect.bisect_left(reference_times, time)
                assert 0 < right < len(reference_times)
                lo, hi = reference[right-1], reference[right]
                assert 0 <= time - lo[0] <= .05 and 0 <= hi[0] - time <= .05
                fraction = (time - lo[0]) / (hi[0] - lo[0])
                expected_reference[frame_id] = [(1-fraction)*a+fraction*b for a, b in zip(lo[1:4], hi[1:4])]
            for estimate in ('original', 'corrected'):
                result = report['results'][estimate]
                assert 'quality_gate' not in result
                matched = result['matched_trajectory']
                assert matched['timestamps'] == [original[i][0] for i in test_ids]
                rotation, translation = result['alignment']['rotation_matrix'], result['alignment']['translation']
                identity = matrix_product(rotation, list(zip(*rotation)))
                for i in range(3):
                    for j in range(3):
                        close(identity[i][j], float(i == j), 1e-10)
                errors = []
                for index, frame_id in enumerate(test_ids):
                    row = original[frame_id] if estimate == 'original' else corrected[graph_ids.index(frame_id)]
                    source_position = row[1:4] if estimate == 'original' else [row[3], row[7], row[11]]
                    estimated_position = transformed(source_position, rotation, translation)
                    reference_position = expected_reference[frame_id]
                    for a, b in zip(estimated_position, matched['estimated_positions'][index]):
                        close(a, b, 1e-9)
                    for a, b in zip(reference_position, matched['reference_positions'][index]):
                        close(a, b, 1e-9)
                    errors.append(math.dist(estimated_position, reference_position))
                rmse = math.sqrt(sum(e*e for e in errors) / len(errors))
                close(rmse, result['ate']['rmse'])
                close(rmse, saved[f'{estimate}_ate_rmse_m'])
                source_fit, reference_fit = [], []
                for frame_id in fit_ids:
                    row = original[frame_id] if estimate == 'original' else corrected[graph_ids.index(frame_id)]
                    source_fit.append(transformed(row[1:4] if estimate == 'original' else [row[3], row[7], row[11]], rotation, translation))
                    reference_fit.append(expected_reference[frame_id])
                for axis in range(3):
                    close(sum(p[axis] for p in source_fit) / len(fit_ids), sum(p[axis] for p in reference_fit) / len(fit_ids), 1e-9)
                positions, truths = matched['estimated_positions'], matched['reference_positions']
                rpe = [math.sqrt(sum(((b[j]-a[j])-(d[j]-c[j]))**2 for j in range(3)))
                       for a, b, c, d in zip(positions, positions[1:], truths, truths[1:])]
                close(math.sqrt(sum(v*v for v in rpe) / len(rpe)), result['rpe_translation']['rmse'])
            close(report['results']['corrected']['ate']['rmse'] - report['results']['original']['ate']['rmse'], saved['delta_ate_rmse_m'])
    print('Verified: hashes, reference calibration/clock, original IDs, alignment split, transformed samples, ATE/RPE and receipt metrics.'
          + (' Full source CSV hashes and bracket rows also verified.' if ground_truth_root else ' Full source CSVs are external.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth-root', type=Path)
    args = parser.parse_args()
    verify(Path(__file__).resolve().parent, args.ground_truth_root)
