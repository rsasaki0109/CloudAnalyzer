#!/usr/bin/env python3
"""Recompute temporal station correspondence and pair-selection evidence without native code."""
import bisect
import hashlib
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) < 1e-8, (a, b)


def stations(points):
    result = [0.]
    for before, after in zip(points, points[1:]):
        step = math.hypot(after[0]-before[0], after[1]-before[1])
        assert step > 1e-9
        result.append(result[-1]+step)
    return result


def interpolate(value, axis, values):
    if value <= axis[0]:
        return values[0]
    if value >= axis[-1]:
        return values[-1]
    i = bisect.bisect_right(axis, value)
    fraction = (value-axis[i-1])/(axis[i]-axis[i-1])
    return values[i-1]+fraction*(values[i]-values[i-1])


def verify():
    root = Path(__file__).resolve().parent
    trial_root, reference_root = root.parent/'nclt-motion-trials', root.parent/'nclt-reference-accuracy'
    subprocess.run([sys.executable, str(trial_root/'verify.py')], check=True)
    helpers = runpy.run_path(str(reference_root/'verify.py'))
    text_rows = helpers['text_rows']
    named = set()
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected, filename = line.split('  ', 1)
        assert filename not in named and (root/filename).resolve().is_relative_to(root)
        assert digest(root/filename) == expected, filename
        named.add(filename)
    assert named == {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts}
    receipt = read(root/'receipt.json')
    for label, entry in receipt['sessions'].items():
        folder = root/label
        comparison = read(folder/'comparison.json')
        selection = read(folder/'selection.json')
        run, job = read(folder/'run.json'), read(folder/'job.json')
        manifest = read(folder/'motion-run.json')
        assert run['status'] == 'finished' and run['output'] == comparison['pairs']['candidate']
        assert manifest['old_hd_reused'] is False and job['selected'] is None
        assert len(job['attempts']) == 2 and job['max_attempts'] == 4
        for original_path, filename in entry['artifact_paths'].items():
            local = root/filename
            for inputs in (comparison['inputs'], manifest['inputs']):
                if original_path in inputs:
                    artifact = inputs[original_path]
                    assert artifact['path'] == original_path
                    assert digest(local) == artifact['sha256'] and local.stat().st_size == artifact['bytes']
        assert digest(folder/'comparison.json') == selection['comparison']['sha256'] == entry['comparison']['sha256']
        assert selection['revision'] == entry['final_revision'] == 2
        assert selection['choice'] == entry['final_choice'] == 'baseline'
        assert [h['choice'] for h in selection['history']] == ['baseline', 'candidate', 'baseline']
        assert read(folder/'mcp-choose.json')['output'] == comparison['pairs']['candidate']
        assert read(folder/'mcp-restore.json')['output'] == comparison['pairs']['baseline']
        assert all(v['all_files_unchanged_during_comparison_and_selection'] for v in entry['preservation'].values())
        axis = comparison['station_protocol']
        ids = axis['original_frame_ids']
        original = text_rows(reference_root/label/'original.tum')
        common = stations([original[i][1:4] for i in ids])
        timestamps = [original[i][0] for i in ids]
        assert axis['original_timestamps_s'] == timestamps
        for actual, expected in zip(axis['original_stations_m'], common):
            close(actual, expected)
        included = []
        for role, evidence_file, poses_file, quality_file in (
            ('baseline', folder/'baseline-report.json', reference_root/label/'corrected-kitti.txt', folder/'baseline-audits.json'),
            ('candidate', folder/'attempt-2-report.json', trial_root/label/'loops-only/corrected-kitti.txt', folder/'attempt-2-audits.json'),
        ):
            evidence = read(evidence_file)
            poses = text_rows(poses_file)
            axis_map = stations([[row[3],row[7],row[11]] for row in poses])
            close(evidence['extraction']['trajectory_length'], axis_map[-1])
            rows = [row for row in evidence['station_disposition'] if row['status']=='included_lane_hypothesis']
            actual = comparison[role]['included_intervals']
            assert len(rows) == len(actual)
            for source, mapped in zip(rows, actual):
                for edge in ('from', 'to'):
                    value = source[edge+'_m']
                    close(mapped[edge+'_m'], interpolate(value, axis_map, common))
                    close(mapped[edge+'_timestamp_s'], interpolate(value, axis_map, timestamps))
                    close(mapped[edge+'_original_frame_coordinate'], interpolate(value, axis_map, ids))
            included.append(actual)
            generated = sum(row['to_m']-row['from_m'] for row in actual)
            close(comparison[role]['common_extent']['generated_length_m'], generated)
            close(comparison[role]['common_extent']['trajectory_length_m'], common[-1])
            close(comparison[role]['common_extent']['retained_fraction'], generated/common[-1])
            assert comparison[role]['common_extent']['passes_requested_extent'] is False
            quality = read(quality_file)
            for key, audit in [('editable',quality['editable']), ('reopened_osm',quality['reopened_osm']),
                    *[(f'consensus_{key}', value) for key,value in quality['ground_consensus'].items()]]:
                raw = audit['quality']
                assert raw['limited'] is False and not raw['omitted_lanes'] and not raw['malformed_lanes']
                traces = [lane[trace] for lane in raw['lanes'] for trace in ['center','left','right']]
                totals = {metric:sum(trace[metric] for trace in traces) for metric in ['samples','supported','height_mismatches','insufficient_returns']}
                assert totals == comparison[role]['audits'][key]['sample_totals']
                assert comparison[role]['audits'][key]['needs_review'] == [lane['lane'] for lane in raw['lanes'] if lane['needs_review']]
                for metric in ['sampling_step_m','ground_radius_m','ground_height_tolerance_m','minimum_support_fraction','sample_budget']:
                    assert comparison[role]['audits'][key]['protocol'][metric] == raw[metric]
        changes = {'gained': [], 'lost': []}
        cuts = sorted({0., common[-1], *[v for rows in included for row in rows for v in [row['from_m'],row['to_m']]]})
        for lo, hi in zip(cuts,cuts[1:]):
            before, after = [any(row['from_m'] <= (lo+hi)/2 <= row['to_m'] for row in rows) for rows in included]
            if before != after:
                changes['gained' if after else 'lost'].append({'from_m':lo,'to_m':hi})
        for key, rows in changes.items():
            assert len(rows) == len(comparison['source_changes'][key])
            for row, actual in zip(rows,comparison['source_changes'][key]):
                close(row['from_m'], actual['from_m']); close(row['to_m'], actual['to_m'])
        pages = [read(trial_root/label/'loops-only'/f'comparison-with-baseline-{offset}.json') for offset in (0,8)]
        assert comparison['motion']['global_results'] == pages[0]['global_results']
        assert comparison['motion']['windows'] == [window for page in pages for window in page['windows']]
        assert comparison['motion']['worsened_windows'] == 4
        assert comparison['deployment_ready'] is False and comparison['independent_accuracy_established'] is False
        print(label, 'fresh pair, original-frame intervals, gains/losses, four audit totals and exact restoration verified')


if __name__ == '__main__':
    verify()
