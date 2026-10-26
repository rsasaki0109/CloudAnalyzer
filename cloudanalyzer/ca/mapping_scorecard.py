"""Comparable evidence summaries for saved mapping runs (no processing rerun)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _read(path: Path) -> dict:
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError(f"metadata exceeds 32 MiB: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    return value


def scorecard(directory: str) -> dict:
    root = Path(directory)
    run, job = _read(root / 'run.json'), _read(root / 'job.json')
    if run.get('schema') != 'cloudanalyzer.mapping_run.v1':
        raise ValueError('unsupported mapping run')
    source, layout = job.get('source', {}), run.get('layout_file', {})
    def digest(value):
        result = value.get('sha256')
        if not isinstance(result, str) or len(result) != 64 or any(c not in '0123456789abcdef' for c in result):
            raise ValueError('source and layout require SHA-256 identities')
        return result
    output = run.get('output') or {}
    diagnosis = output.get('diagnosis') or {}
    holds, extent = diagnosis.get('review_holds', {}), diagnosis.get('extent', {})
    def number(value):
        return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None
    identity = {'source_sha256': digest(source), 'layout_sha256': digest(layout)}
    from ca.mapping_measure import SCHEMA, PROTOCOL, file_hash
    measured = {}
    measurement_status = 'unavailable'
    if (root / 'measurement.json').is_file():
        report = _read(root / 'measurement.json')
        if (report.get('schema') == SCHEMA and report.get('protocol') == PROTOCOL
                and report.get('run_sha256') == file_hash(root / 'run.json')
                and report.get('job_sha256') == file_hash(root / 'job.json')):
            measurement_status = report.get('status', 'unavailable')
            interval = number(report.get('sample_interval_seconds'))
            if (measurement_status == 'process_succeeded' and type(report.get('exit_code')) is int and report['exit_code'] == 0
                    and interval is not None and .01 <= interval <= 1):
                measured = report
        else:
            measurement_status = 'stale_or_unsupported'
    return {'schema': 'cloudanalyzer.mapping_scorecard.v1', 'identity': identity,
            'run_sha256': hashlib.sha256((root / 'run.json').read_bytes()).hexdigest(),
            'status': output.get('status', run.get('status')),
            'generated_length_m': number(extent.get('generated_length_m')),
            'trajectory_length_m': number(extent.get('trajectory_length_m')),
            'unexplained_traces': number(holds.get('unexplained_traces_total')),
            'recorded_actions': len(run.get('history', [])),
            'human_interventions': None, 'wall_time_seconds': number(measured.get('wall_time_seconds')),
            'sampled_peak_tree_rss_bytes': number(measured.get('sampled_peak_tree_rss_bytes')),
            'logical_output_bytes': number(measured.get('logical_output_bytes')),
            'measurement_status': measurement_status,
            'measurement_protocol': measured.get('protocol'),
            'measurement_context': {key: measured.get(key) for key in ('command', 'script_sha256', 'platform', 'rss_samples', 'unavailable_process_reads')},
            'sample_interval_seconds': number(measured.get('sample_interval_seconds')),
            'independent_accuracy': None,
            'scope': 'Saved evidence only. Missing measurements are null; action count is not human intervention count.'}


def compare(before: dict, after: dict) -> dict:
    if before['identity'] != after['identity']:
        raise ValueError('comparison requires identical source and layout identities')
    keys = ['generated_length_m', 'trajectory_length_m', 'unexplained_traces', 'recorded_actions']
    if (before.get('status') == after.get('status') and before.get('measurement_protocol') and before.get('measurement_protocol') == after.get('measurement_protocol')
            and before.get('sample_interval_seconds') == after.get('sample_interval_seconds')):
        keys += ['wall_time_seconds', 'sampled_peak_tree_rss_bytes', 'logical_output_bytes']
    return {'before': before, 'after': after,
            'delta': {key: after[key] - before[key] if before[key] is not None and after[key] is not None else None for key in keys}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--baseline')
    args = parser.parse_args()
    current = scorecard(args.run)
    print(json.dumps(compare(scorecard(args.baseline), current) if args.baseline else current, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
