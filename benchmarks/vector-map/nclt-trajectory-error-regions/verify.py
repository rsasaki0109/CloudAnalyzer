#!/usr/bin/env python3
"""Recompute the portable MCP localization evidence without NumPy or native code."""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isfinite(actual) and abs(actual - expected) <= 1e-9, (actual, expected)


def norm(vector):
    return math.sqrt(sum(v * v for v in vector))


def rmse(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


def difference(first, second):
    return [a - b for a, b in zip(first, second)]


def verify():
    root = Path(__file__).resolve().parent
    source = root.parent / 'nclt-reference-accuracy'
    subprocess.run([sys.executable, str(source / 'verify.py')], check=True)
    manifest = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert name not in manifest and (root / name).resolve().is_relative_to(root)
        assert sha(root / name) == digest, name
        manifest[name] = digest
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.name != 'SHA256SUMS'}
    assert set(manifest) == actual, (set(manifest) ^ actual)
    receipt = json.loads((root / 'receipt.json').read_text())
    old_receipt = json.loads((source / 'receipt.json').read_text())
    assert receipt['schema'] == 'cloudanalyzer.nclt_trajectory_regions_receipt.v1'
    assert set(receipt['sessions']) == {'april', 'june'}
    for name, entry in receipt['sessions'].items():
        folder = source / name
        report_path = folder / 'comparison-prefix-30.json'
        report = json.loads(report_path.read_text())
        assert entry['report'] == old_receipt['sessions'][name]['results']['prefix-30']['report']
        assert sha(report_path) == entry['report']['sha256']
        assert report_path.stat().st_size == entry['report']['bytes']
        assert entry['unchanged_job_files'] == old_receipt['sessions'][name]['unchanged_job_files']
        assert entry['unchanged_manifest_sha256'] == old_receipt['sessions'][name]['unchanged_manifest_sha256']
        assert set(entry['pages']) == {'regression', 'corrected_ate'}
        graph_ids = sorted(int(row[1]) for line in (folder / 'corrected.g2o').read_text().splitlines()
                           if (row := line.split()) and row[0] == 'VERTEX_SE3:QUAT')
        rows = [[float(v) for v in line.split()] for line in (folder / 'corrected-kitti.txt').read_text().splitlines() if line.strip()]
        map_poses = [[row[3], row[7], row[11]] for row in rows]
        assert len(map_poses) == len(graph_ids)
        graph_index = {frame: i for i, frame in enumerate(graph_ids)}
        distance = [0.]
        for previous, current in zip(map_poses, map_poses[1:]):
            distance.append(distance[-1] + norm(difference(current, previous)))
        ids = report['coverage']['evaluated_original_frame_ids']
        samples = {key: report['results'][key]['matched_trajectory'] for key in ('original', 'corrected')}
        times = samples['original']['timestamps']
        assert report['protocol']['held_out_alignment'] is True
        assert not set(ids) & set(report['coverage']['alignment_original_frame_ids'])
        expected = {}
        for start in range(0, len(ids), 12):
            end = min(start + 12, len(ids))
            metrics = {}
            for estimate in ('original', 'corrected'):
                poses = samples[estimate]['estimated_positions'][start:end]
                truth = samples[estimate]['reference_positions'][start:end]
                ate = [norm(difference(p, q)) for p, q in zip(poses, truth)]
                relative = [norm(difference(difference(p1, p0), difference(q1, q0)))
                            for p0, p1, q0, q1 in zip(poses, poses[1:], truth, truth[1:])]
                metrics[estimate] = {'ate_rmse_m': rmse(ate), 'rpe_translation_rmse_m': rmse(relative) if relative else None}
            indices = [graph_index[frame] for frame in ids[start:end]]
            raw = [map_poses[i] for i in indices]
            expected[start // 12] = {'ids': ids[start:end], 'times': [times[start], times[end - 1]],
                'metrics': metrics, 'delta': metrics['corrected']['ate_rmse_m'] - metrics['original']['ate_rmse_m'],
                'bounds': [min(p[0] for p in raw), min(p[1] for p in raw), max(p[0] for p in raw), max(p[1] for p in raw)],
                'distance': [distance[indices[0]], distance[indices[-1]]], 'missing': indices[-1] - indices[0] + 1 - len(indices)}
        for ranking, pages in entry['pages'].items():
            windows = []
            for page_index, filename in enumerate(pages):
                page = json.loads((root / filename).read_text())
                assert page['schema'] == 'cloudanalyzer.mapping_trajectory_regions.v1'
                assert page['report'] == entry['report']
                assert page['job'] == report['inputs']['job'] and page['point_map'] == report['inputs']['pointcloud_map']
                assert page['reference_provenance'] == report['reference_provenance']
                assert page['reference_independence'] == report['reference_independence']
                assert page['protocol']['ranking'] == ranking and page['protocol']['window_poses'] == 12
                assert page['protocol']['held_out_alignment'] is True
                assert page['protocol']['alignment_prefix_fraction'] == .3
                assert page['protocol']['bounds_frame'] == 'unaligned_corrected_point_map_sensor_origins_only'
                assert page['offset'] == page_index * 8 and len(page['windows']) <= 8
                assert page['total_windows'] == len(expected)
                assert page['next_offset'] == ((page_index + 1) * 8 if page_index + 1 < len(pages) else None)
                for key, value in page['coverage'].items():
                    assert value == report['coverage'][key]
                assert page['global_change'] == report['change']
                windows.extend(page['windows'])
            order = sorted(expected, key=lambda i: (-(expected[i]['delta'] if ranking == 'regression' else expected[i]['metrics']['corrected']['ate_rmse_m']), i))
            assert [window['window_id'] for window in windows] == order
            assert sorted(frame for window in windows for frame in window['original_frame_ids']) == ids
            assert entry['top_windows'][ranking] == windows[0]
            for window in windows:
                value = expected[window['window_id']]
                assert window['original_frame_ids'] == value['ids'] and window['timestamp_range_s'] == value['times']
                assert window['evaluated_poses'] == len(value['ids'])
                assert window['unevaluated_retained_poses_within_frame_span'] == value['missing']
                for actual, target in zip(window['evaluated_corrected_pose_bounds_xy'], value['bounds']):
                    close(actual, target)
                for actual, target in zip(window['corrected_graph_distance_range_m'], value['distance']):
                    close(actual, target)
                close(window['ate_rmse_m_corrected_minus_original'], value['delta'])
                for estimate, metrics in value['metrics'].items():
                    for metric, target in metrics.items():
                        close(window['results'][estimate][metric], target)
            for estimate in ('original', 'corrected'):
                weighted = math.sqrt(sum(window['evaluated_poses'] * window['results'][estimate]['ate_rmse_m']**2 for window in windows) / len(ids))
                close(weighted, report['results'][estimate]['ate']['rmse'])
        print(name, len(ids), 'evaluated poses, both complete rankings and map bounds verified')
    print('NCLT trajectory error regions verified')


if __name__ == '__main__':
    verify()
