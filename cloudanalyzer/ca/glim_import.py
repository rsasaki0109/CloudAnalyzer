"""Prepare documented GLIM global LiDAR exports for the existing SLAM importer."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
import tempfile
from typing import Any

from ca import mapping_job as jobs
from ca.glim_receipt import read_glim_receipt
from ca.mapping_import import MAX_MAP_BYTES, MAX_TRAJECTORY_BYTES, _provenance, _space, inspect_slam_inputs

SCHEMA = 'cloudanalyzer.glim_export.v1'
KEY = 'cloudanalyzer_glim_export'


def _execution_report(path: str, trajectory: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    raw, report = read_glim_receipt(Path(path))
    if (report.get('status') != 'process_succeeded' or type(report.get('exit_code')) is not int
            or report['exit_code'] != 0 or report.get('inputs_unchanged') is not True
            or report.get('configuration_copy_matches') is not True):
        raise ValueError('GLIM run report must record successful execution with unchanged inputs/configuration')
    image_id = report.get('image_id')
    if not isinstance(image_id, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
        raise ValueError('GLIM run report needs an immutable image identity')
    for key in ('trajectory', 'trajectory_check'):
        recorded = report.get(key)
        if (not isinstance(recorded, dict) or type(recorded.get('bytes')) is not int
                or any(recorded.get(k) != trajectory[k] for k in ('sha256', 'bytes'))):
            raise ValueError('GLIM run report trajectory does not match the selected export')
    return raw, {'schema': 'cloudanalyzer.glim_execution_link.v1', 'image_id': image_id,
                 'report_sha256': hashlib.sha256(raw).hexdigest(), 'report_bytes': len(raw),
                 'trajectory_sha256': trajectory['sha256'],
                 'scope': 'Local execution report matched by content hash; not a signed attestation or proof of PLY same-state/frame alignment.'}


def prepare_glim_export(dump_dir: str, map_path: str, out_dir: str, provenance: dict[str, Any], *, run_report: str | None = None) -> dict[str, Any]:
    """Preserve traj_lidar.txt bytes as trajectory.tum; reference the PLY in place.

    The caller declares a map and trajectory saved from the same optimized GLIM
    state, in a Z-up metre frame. No SLAM, alignment, clock conversion or IMU
    extrinsic conversion is performed. Sensor calibration/accuracy is unverified.
    """
    declared = _provenance(provenance)
    if declared['backend'].strip().lower() != 'glim':
        raise ValueError('the producer backend must explicitly be GLIM')
    if KEY in declared['configuration']:
        raise ValueError('producer configuration already contains GLIM adapter metadata')
    root = Path(out_dir).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError('GLIM preparation needs a new output directory')
    trajectory = Path(dump_dir).resolve() / 'traj_lidar.txt'
    cloud = Path(map_path).resolve()
    if trajectory.is_symlink():
        raise ValueError('traj_lidar.txt must be the exported file, not a symbolic link')
    if cloud.suffix.lower() != '.ply':
        raise ValueError('export the same optimized GLIM map as PLY')
    for path, limit in [(trajectory, MAX_TRAJECTORY_BYTES), (cloud, MAX_MAP_BYTES)]:
        if not path.is_file() or not 0 < path.stat().st_size <= limit:
            raise ValueError(f'missing, empty or oversized GLIM export: {path}')
    _space(root, trajectory.stat().st_size + 1024**2)
    inputs = {'map': jobs._artifact(cloud), 'trajectory': jobs._artifact(trajectory)}
    trajectory_bytes = trajectory.read_bytes()
    metadata = {'schema': SCHEMA, 'inputs': inputs, 'trajectory_kind': 'global_lidar',
                'trajectory_bytes_preserved': True, 'map_referenced_in_place': True,
                'alignment_applied': False, 'timestamps_rebased': False}
    run_bytes = None
    if run_report is not None:
        run_bytes, execution = _execution_report(run_report, inputs['trajectory'])
        execution['report'] = {'path': str(root / 'glim-run.json'),
                               'sha256': execution.pop('report_sha256'), 'bytes': execution.pop('report_bytes')}
        metadata['execution'] = execution
    declared['configuration'][KEY] = metadata
    declared = _provenance(declared)
    with tempfile.TemporaryDirectory(prefix='cloudanalyzer-glim-') as scratch:
        normalized = Path(scratch) / 'trajectory.tum'
        normalized.write_bytes(trajectory_bytes)
        if jobs._artifact(normalized)['sha256'] != inputs['trajectory']['sha256']:
            raise ValueError('GLIM trajectory changed during preparation')
        preflight = inspect_slam_inputs(str(cloud), str(normalized), declared)
    for artifact in inputs.values():
        jobs._verify(artifact)
    result = {**metadata, 'map': str(cloud), 'trajectory': str(root / 'trajectory.tum'),
              'provenance': str(root / 'provenance.json'), 'preflight': preflight,
              'scope': 'Documented GLIM export adapter. Same-state/frame declarations and independent accuracy are not verified. No GLIM estimator is executed.'}
    encoded = json.dumps(result, indent=2, allow_nan=False)
    _space(root, len(trajectory_bytes) + len(encoded.encode()) + len(run_bytes or b'') + 1024**2)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'trajectory.tum').write_bytes(trajectory_bytes)
    if run_bytes is not None:
        (root / 'glim-run.json').write_bytes(run_bytes)
    (root / 'provenance.json').write_text(json.dumps(declared, indent=2, allow_nan=False))
    (root / 'glim-export.json').write_text(encoded)
    return result
