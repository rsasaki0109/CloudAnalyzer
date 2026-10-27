"""Inspect saved GLIM file identities without Docker or estimator execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from ca.mapping_import import inspect_tum_trajectory
from ca.mapping_job import _artifact
from ca.glim_run import inventory


def _identity(value):
    if (not isinstance(value, dict) or not isinstance(value.get('path'), str)
            or not value['path'] or not isinstance(value.get('sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', value['sha256'])
            or type(value.get('bytes')) is not int or value['bytes'] < 0):
        raise ValueError('invalid recorded file identity')
    return value


def inspect_glim_run(directory: str) -> dict:
    root = Path(directory).resolve()
    receipt = root/'glim-run.json'
    if receipt.is_symlink() or not receipt.is_file() or not 0 < receipt.stat().st_size <= 4 * 1024**2:
        raise ValueError('GLIM run report must be a regular file of at most 4 MiB')
    with receipt.open('rb') as stream:
        raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError('GLIM run report exceeds 4 MiB')
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get('schema') != 'cloudanalyzer.glim_run.v1':
        raise ValueError('unsupported GLIM run report')
    inputs = report.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError('missing GLIM input inventory')
    for key in ('bag','configuration'):
        if not isinstance(inputs.get(key),list) or not 1 <= len(inputs[key]) <= 10000:
            raise ValueError('invalid GLIM input inventory')
        for entry in inputs[key]: _identity(entry)
    errors = []
    count = 0
    def verify(path, expected, role):
        nonlocal count
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError('file missing or replaced by a symlink')
            current = _artifact(path)
            if any(current[k] != expected[k] for k in ('bytes','sha256')):
                raise ValueError('recorded bytes changed')
            count += 1
        except (OSError, ValueError) as error:
            errors.append(f'{role}: {error}')
    # New receipts record roots so additions can be detected as well as mutations.
    roots = report.get('input_roots')
    inventory_status = 'not_recorded'
    if roots is None:
        for key in ('bag', 'configuration'):
            for index, entry in enumerate(inputs[key]):
                verify(Path(entry['path']), entry, f'{key}[{index}]')
    else:
        if not isinstance(roots, dict) or set(roots) != {'bag', 'configuration'}:
            raise ValueError('invalid input roots')
        inventory_status = 'matched'
        for key in ('bag', 'configuration'):
            value = roots[key]
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError('input roots must be absolute paths')
            try:
                current = inventory(Path(value))
                # Inventory already hashes every file. Do not reread large bags
                # for the individual-file check when complete roots are available.
                by_path = {entry['path']: entry for entry in current}
                count += sum(by_path.get(entry['path']) == entry for entry in inputs[key])
                if current != inputs[key]:
                    raise ValueError('input directory membership or file identities differ from inventory')
            except (OSError, ValueError) as error:
                inventory_status = 'changed_or_missing'
                errors.append(f'{key} inventory: {error}')
    # Only runs that reached container command creation require a copied config.
    if report.get('command'):
        expected = {}
        for entry in inputs['configuration']:
            relative = entry.get('relative_path')
            if (not isinstance(relative,str) or not relative or '\\' in relative
                    or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts
                    or relative in expected):
                raise ValueError('invalid configuration relative path')
            expected[relative] = entry
        try:
            if (root/'config').is_symlink():
                raise ValueError('configuration copy is a symlink')
            copied = inventory(root/'config')
            actual = {entry['relative_path']: entry for entry in copied}
            if actual.keys() != expected.keys() or any(
                actual[name][key] != value[key] for name,value in expected.items() for key in ('sha256','bytes')):
                raise ValueError('configuration copy differs from inventory')
            count += len(copied)
        except (OSError,ValueError) as error:
            errors.append(f'configuration copy: {error}')
    trajectory = None
    if report.get('trajectory') is not None:
        reference = _identity(report['trajectory'])
        path = root/'dump'/'traj_lidar.txt'
        verify(path,reference,'trajectory')
        try:
            trajectory = inspect_tum_trajectory(path)
            if any(trajectory[key] != reference[key] for key in ('sha256','bytes')):
                raise ValueError('trajectory differs from recorded identity')
        except (OSError,ValueError) as error:
            trajectory = None
            errors.append(f'trajectory validation: {error}')
    elif report.get('status') == 'process_succeeded':
        errors.append('successful execution report has no trajectory identity')
    digest = hashlib.sha256(raw).hexdigest()
    if _artifact(receipt)['sha256'] != digest:
        errors.append('run report changed during inspection')
    return {'schema':'cloudanalyzer.glim_inspection.v1',
            'status':'recorded_files_match' if not errors else 'changed_or_missing',
            'recorded_execution_status':report.get('status'), 'recorded_exit_code':report.get('exit_code'),
            'recorded_image_id':report.get('image_id'), 'report_sha256':digest,
            'input_inventory_status': inventory_status,
            'recorded_failure_stage':report.get('failure_stage'), 'recorded_error':report.get('error'),
            'recorded_cleanup_error':report.get('cleanup_error'),
            'verified_file_count':count, 'issue_count':len(errors), 'issues':errors[:32],
            'issues_truncated':len(errors)>32, 'trajectory_check':trajectory,
            'accuracy_verified':False,
            'scope':'Read-only check of recorded input files, complete input inventories when roots were recorded, copied configuration when a command was recorded, and recorded trajectory. No Docker calls. Older reports without input roots cannot detect added input files. Process logs and other dump files are not verified. Matching file hashes do not prove execution, calibration, PLY consistency or accuracy.'}
