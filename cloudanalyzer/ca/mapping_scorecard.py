"""Comparable evidence summaries for saved mapping runs (no processing rerun)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MAX_METADATA_BYTES = 32 * 1024**2
MAX_METADATA_DEPTH = 128


def _read(path: Path) -> tuple[dict, str]:
    """Parse and identify the same bounded bytes, without a second file read."""
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds 32 MiB: {path.name}")
    with path.open('rb') as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata exceeds 32 MiB: {path.name}")
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('metadata contains a duplicate object key')
            result[key] = item
        return result

    def invalid_constant(token):
        raise ValueError('metadata contains a nonfinite JSON number')

    def finite_float(token):
        result = float(token)
        if not math.isfinite(result):
            raise ValueError('metadata contains a nonfinite JSON number')
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique_object,
                           parse_constant=invalid_constant, parse_float=finite_float)
    except RecursionError as error:
        raise ValueError("metadata nesting exceeds parser limits") from error
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    # Bound nesting independently of the Python JSON parser's recursion policy.
    stack = [iter(value.values())]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(item, (dict, list)):
            if len(stack) >= MAX_METADATA_DEPTH:
                raise ValueError('metadata nesting exceeds 128 levels')
            stack.append(iter(item.values() if isinstance(item, dict) else item))
    return value, hashlib.sha256(raw).hexdigest()


def scorecard(directory: str) -> dict:
    root = Path(directory)
    run, run_sha256 = _read(root / 'run.json')
    job, job_sha256 = _read(root / 'job.json')
    if run.get('schema') != 'cloudanalyzer.mapping_run.v1':
        raise ValueError('unsupported mapping run')
    def object_field(parent, key, path, *, nullable=False):
        value = parent.get(key, {})
        if value is None and nullable:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f'{path} must be an object' + (' or null' if nullable else ''))
        return value

    source = object_field(job, 'source', 'job.json.source')
    layout = object_field(run, 'layout_file', 'run.json.layout_file')
    def digest(value):
        result = value.get('sha256')
        if not isinstance(result, str) or len(result) != 64 or any(c not in '0123456789abcdef' for c in result):
            raise ValueError('source and layout require SHA-256 identities')
        return result
    output = object_field(run, 'output', 'run.json.output', nullable=True)
    diagnosis = object_field(output, 'diagnosis', 'run.json.output.diagnosis', nullable=True)
    holds = object_field(diagnosis, 'review_holds', 'run.json.output.diagnosis.review_holds', nullable=True)
    extent = object_field(diagnosis, 'extent', 'run.json.output.diagnosis.extent', nullable=True)
    history = run.get('history', [])
    if not isinstance(history, list) or any(not isinstance(entry, dict) for entry in history):
        raise ValueError('run.json.history must be an array of objects')
    def number(value):
        if type(value) is int:
            return value if value >= 0 else None
        return value if type(value) is float and math.isfinite(value) and value >= 0 else None
    identity = {'source_sha256': digest(source), 'layout_sha256': digest(layout)}
    from ca.mapping_measure import SCHEMA, PROTOCOL
    measured = {}
    diagnostics: dict = {'exit_code': None, 'error': None, 'cleanup_error': None, 'output_size_error': None}
    measurement_status = 'unavailable'
    measurement_report_error = None
    binding_reasons = []
    if (root / 'measurement.json').is_file():
        try:
            report, _ = _read(root / 'measurement.json')
        except (OSError, ValueError) as error:
            report = None
            measurement_status = 'invalid_report'
            measurement_report_error = str(error)[:4096]
        if report is not None:
            if report.get('schema') != SCHEMA:
                binding_reasons.append('Measurement schema is missing or unsupported.')
            if report.get('protocol') != PROTOCOL:
                binding_reasons.append('Measurement protocol is missing or unsupported.')
            for name, expected in (('run', run_sha256), ('job', job_sha256)):
                if (root / f'{name}.json').is_symlink():
                    binding_reasons.append(f'{name}.json is a symbolic link.')
                if report.get(f'{name}_sha256') != expected:
                    binding_reasons.append(f'Measurement {name}_sha256 is missing or does not match the parsed {name}.json.')
        if report is not None and not binding_reasons:
            measurement_status = report.get('status', 'unavailable')
            diagnostics['exit_code'] = report.get('exit_code') if type(report.get('exit_code')) is int else None
            for key in ('error', 'cleanup_error', 'output_size_error'):
                value = report.get(key)
                diagnostics[key] = value[:4096] if isinstance(value, str) else None
            interval = number(report.get('sample_interval_seconds'))
            if (measurement_status == 'process_succeeded' and type(report.get('exit_code')) is int and report['exit_code'] == 0
                    and report.get('error') is None and report.get('cleanup_error') is None
                    and interval is not None and .01 <= interval <= 1):
                measured = report
        elif report is not None:
            measurement_status = 'stale_or_unsupported'
    return {'schema': 'cloudanalyzer.mapping_scorecard.v1', 'identity': identity,
            'run_sha256': run_sha256,
            'status': output.get('status', run.get('status')),
            'generated_length_m': number(extent.get('generated_length_m')),
            'trajectory_length_m': number(extent.get('trajectory_length_m')),
            'unexplained_traces': number(holds.get('unexplained_traces_total')),
            'recorded_actions': len(history),
            'human_interventions': None, 'wall_time_seconds': number(measured.get('wall_time_seconds')),
            'sampled_peak_tree_rss_bytes': number(measured.get('sampled_peak_tree_rss_bytes')),
            'logical_output_bytes': number(measured.get('logical_output_bytes')) if measured.get('output_size_error') is None else None,
            'measurement_status': measurement_status,
            'measurement_report_error': measurement_report_error,
            'measurement_binding_reasons': binding_reasons,
            'measurement_diagnostics': diagnostics,
            'measurement_protocol': measured.get('protocol'),
            'measurement_context': {key: measured.get(key) for key in ('command', 'script_sha256', 'platform', 'rss_samples', 'unavailable_process_reads')},
            'sample_interval_seconds': number(measured.get('sample_interval_seconds')),
            'independent_accuracy': None,
            'scope': 'Saved evidence only. Missing measurements are null; action count is not human intervention count.'}


def compare(before: dict, after: dict) -> dict:
    if before['identity'] != after['identity']:
        raise ValueError('comparison requires identical source and layout identities')
    keys = ['generated_length_m', 'trajectory_length_m', 'unexplained_traces', 'recorded_actions']
    reasons = []
    statuses = [card.get('status') for card in (before, after)]
    if any(not isinstance(value, str) or not value.strip() for value in statuses):
        reasons.append('Generation stage is missing or invalid.')
    elif statuses[0] != statuses[1]:
        reasons.append('Generation stages differ.')
    if any(card.get('measurement_status') != 'process_succeeded' for card in (before, after)):
        reasons.append('Both runs need successful, current measurements.')
    from ca.mapping_measure import PROTOCOL
    if any(card.get('measurement_protocol') != PROTOCOL for card in (before, after)):
        reasons.append('Measurement protocols are missing, unsupported or different.')
    intervals = [card.get('sample_interval_seconds') for card in (before, after)]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not .01 <= value <= 1 for value in intervals):
        reasons.append('Sampling intervals are missing or invalid.')
    elif intervals[0] != intervals[1]:
        reasons.append('Sampling intervals differ.')
    if not reasons:
        keys += ['wall_time_seconds', 'sampled_peak_tree_rss_bytes', 'logical_output_bytes']
    delta: dict[str, int | float | None] = {}
    delta_errors = {}
    for key in keys:
        left, right = before[key], after[key]
        if left is None or right is None:
            delta[key] = None
            if left is None and right is None:
                delta_errors[key] = 'Baseline and current values are unavailable.'
            elif left is None:
                delta_errors[key] = 'Baseline value is unavailable.'
            else:
                delta_errors[key] = 'Current value is unavailable.'
            continue
        try:
            difference = right - left
            if type(difference) is float and not math.isfinite(difference):
                raise OverflowError('nonfinite difference')
        except OverflowError:
            delta[key] = None
            delta_errors[key] = 'Difference exceeds the finite floating-point range.'
        else:
            delta[key] = difference
    return {'before': before, 'after': after,
            'performance_comparison': {'eligible': not reasons, 'reasons': reasons},
            'delta': delta, 'delta_errors': delta_errors}


def format_text(result: dict) -> str:
    """Render saved evidence without hiding missing values or eligibility limits."""
    metrics = (
        ('generated_length_m', 'Generated length (m)'),
        ('trajectory_length_m', 'Trajectory length (m)'),
        ('unexplained_traces', 'Unexplained traces'),
        ('recorded_actions', 'Recorded actions'),
        ('wall_time_seconds', 'Wall time (s)'),
        ('sampled_peak_tree_rss_bytes', 'Sampled peak RSS (bytes)'),
        ('logical_output_bytes', 'Logical output (bytes)'),
    )
    def display(value):
        # Escape saved strings so control characters cannot affect the terminal.
        return 'unavailable' if value is None else json.dumps(value, ensure_ascii=True, allow_nan=False)

    def summary(card, title):
        lines = [title, f"  Generation: {display(card['status'])}",
                 f"  Measurement: {display(card['measurement_status'])}",
                 f"  Run SHA-256: {card['run_sha256']}"]
        lines += [f'  {label}: {display(card[key])}' for key, label in metrics]
        if card['measurement_report_error']:
            lines.append(f"  Report error: {display(card['measurement_report_error'])}")
        lines += [f'  Binding: {display(reason)}' for reason in card['measurement_binding_reasons']]
        lines += [f'  {key}: {display(value)}' for key, value in card['measurement_diagnostics'].items() if value is not None]
        return lines

    if 'before' not in result:
        lines = summary(result, 'Mapping scorecard')
    else:
        lines = summary(result['before'], 'Baseline') + [''] + summary(result['after'], 'Current')
        eligibility = result['performance_comparison']
        lines += ['', 'Delta (current - baseline)',
                  '  Performance comparison: ' + ('eligible' if eligibility['eligible'] else 'ineligible')]
        lines += [f'  Reason: {display(reason)}' for reason in eligibility['reasons']]
        for key, label in metrics:
            if key not in result['delta']:
                continue
            value = result['delta'][key]
            rendered = display(value)
            if value is not None and value > 0:
                rendered = '+' + rendered
            reason = result['delta_errors'].get(key)
            lines.append(f'  {label}: {rendered}' + (f' ({display(reason)})' if reason else ''))
    lines += ['', 'Saved evidence only; independent accuracy and human intervention counts are unknown.',
              'Performance comparability does not establish identical hardware or system load.']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--baseline')
    parser.add_argument('--text', action='store_true', help='Show a readable summary instead of JSON')
    args = parser.parse_args()
    current = scorecard(args.run)
    result = compare(scorecard(args.baseline), current) if args.baseline else current
    print(format_text(result) if args.text else json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
