"""Run the official ROS2 GLIM bag executable in a local Docker image."""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time
import uuid

from ca.mapping_job import _artifact, _save
from ca.mapping_import import _space, inspect_tum_trajectory
from ca.glim_preflight import inspect_glim_bag, imu_gap_limit_ns
from ca.glim_config import inspect_config_topics

SCHEMA = 'cloudanalyzer.glim_run.v1'
MAX_INVENTORY_ENTRIES = 10000


def inventory(path: Path) -> list[dict]:
    def kind(item):
        mode = item.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError('input symlinks are not supported')
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError('input must contain only regular files and directories')
        return stat.S_ISDIR(mode)

    paths = []
    if not kind(path):
        paths.append(path)
    else:
        pending = [path]
        count = 0
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    count += 1
                    if count > MAX_INVENTORY_ENTRIES:
                        raise ValueError(f'input inventory exceeds {MAX_INVENTORY_ENTRIES} entries')
                    item = Path(entry.path)
                    if kind(item):
                        pending.append(item)
                    else:
                        paths.append(item)
    result = [{'relative_path': str(item.relative_to(path)) if item != path else item.name,
               **_artifact(item)} for item in sorted(paths)]
    if not result:
        raise ValueError('input is empty or missing')
    return result


def run_glim(bag: str, config: str, out: str, image: str, timeout: float = 3600, *, points_topic: str | None = None,
             imu_topic: str | None = None, max_imu_gap_s: float = .05) -> dict:
    """Record a container run, without claiming calibration or map accuracy."""
    if (points_topic is None) != (imu_topic is None):
        raise ValueError('supply both points_topic and imu_topic for input preflight')
    imu_gap_limit_ns(max_imu_gap_s)
    source, settings, root = Path(bag).absolute(), Path(config).absolute(), Path(out).absolute()
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    if root.exists() or root.is_symlink():
        raise FileExistsError('GLIM execution needs a new output directory')
    if source.is_symlink() or settings.is_symlink():
        raise ValueError('input symlinks are not supported')
    if not (source.suffix == '.mcap' and source.is_file() or source.is_dir() and (source/'metadata.yaml').is_file()):
        raise ValueError('supply an MCAP file or a ROS2 bag directory with metadata.yaml')
    if not (settings/'config.json').is_file():
        raise ValueError('config directory must contain config.json')
    for parent in [source.resolve(), settings.resolve()]:
        if root.resolve().is_relative_to(parent):
            raise ValueError('output must be outside the inputs')
    if any(',' in str(p) for p in [source, settings, root]):
        raise ValueError('Docker bind paths must not contain commas')
    inputs = {'bag': inventory(source), 'configuration': inventory(settings)}
    _space(root, sum(x['bytes'] for x in inputs['configuration']) + 1024**2)
    checked = None
    topic_check = None
    if points_topic is not None:
        assert imu_topic is not None
        checked = inspect_glim_bag(str(source), points_topic, imu_topic, max_imu_gap_s=max_imu_gap_s)
        try:
            topic_check = inspect_config_topics(settings, points_topic, imu_topic)
        except (OSError, ValueError) as error:
            topic_check = {'status': 'blocked', 'error': str(error)}
    report: dict = {'schema': SCHEMA, 'image_requested': image, 'image_id': None,
              'inputs': inputs, 'command': [], 'status': 'initializing', 'exit_code': None,
              'input_preflight_status': checked['status'] if checked else 'not_requested',
              'input_preflight': checked, 'config_topics': topic_check, 'accuracy_verified': False, 'map_export_required': True}
    if checked:
        assert topic_check is not None
        if (any(checked['input'][k] != inputs['bag'][0][k] for k in ('sha256', 'bytes'))
                or inputs != {'bag': inventory(source), 'configuration': inventory(settings)}):
            report['status'] = 'inputs_changed'
        elif topic_check['status'] != 'matched':
            report['status'] = 'config_blocked'
        elif checked['status'] != 'basic_checks_passed':
            report['status'] = 'input_blocked'
    root.mkdir(parents=True, exist_ok=False)
    report_path = root/'glim-run.json'
    _save(report_path, report)
    if report['status'] in ('inputs_changed', 'input_blocked', 'config_blocked'):
        return report
    # Pin the already installed image to its immutable local ID; never pull implicitly.
    docker = ['docker', '--host=unix:///var/run/docker.sock']
    env = {k: v for k, v in os.environ.items() if k not in
           {'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH'}}
    try:
        identity = subprocess.check_output(docker + ['image', 'inspect', '--format', '{{.Id}}', image], env=env, text=True, timeout=30).strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', identity):
            raise ValueError('Docker did not return an immutable image ID')
    except (OSError, subprocess.SubprocessError, ValueError, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'runtime_unavailable',
                      failure_stage='image_lookup', error=str(error) or 'Interrupted during image lookup')
        _save(report_path, report)
        return report
    report['image_id'] = identity
    try:
        shutil.copytree(settings, root/'config')
        copied = inventory(root/'config')
        expected = [(x['relative_path'], x['sha256']) for x in inputs['configuration']]
        if ([(x['relative_path'], x['sha256']) for x in copied] != expected or inventory(source) != inputs['bag']):
            report.update(status='inputs_changed', image_id=identity)
            _save(report_path, report)
            return report
        (root/'dump').mkdir()
    except (OSError, ValueError, KeyboardInterrupt) as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'preparation_failed',
                      failure_stage='configuration_copy', error=str(error) or 'Interrupted during configuration preparation')
        _save(report_path, report)
        return report
    name = 'cloudanalyzer-glim-' + uuid.uuid4().hex
    bag_mount = '/input/bag.mcap' if source.is_file() else '/input/bag'
    command = docker + ['run', '--rm', '--name', name, '--network=none',
        '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
        '--mount', f'type=bind,src={source},dst={bag_mount},readonly',
        '--mount', f'type=bind,src={root / "config"},dst=/input/config,readonly',
        '--mount', f'type=bind,src={root / "dump"},dst=/output', identity,
        'ros2', 'run', 'glim_ros', 'glim_rosbag', bag_mount, '--ros-args',
        '-p', 'config_path:=/input/config', '-p', 'dump_path:=/output', '-p', 'auto_quit:=true']
    report.update(image_id=identity, command=command, status='running')
    _save(report_path, report)
    start = time.perf_counter()
    try:
        with (root/'glim.log').open('wb') as log:
            process = subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        report['exit_code'] = process.returncode
        report['status'] = 'process_succeeded' if process.returncode == 0 else 'process_failed'
    except (subprocess.TimeoutExpired, KeyboardInterrupt, OSError) as error:
        report['status'] = 'timed_out' if isinstance(error, subprocess.TimeoutExpired) else 'interrupted' if isinstance(error, KeyboardInterrupt) else 'launch_failed'
        report['error'] = str(error) or 'Execution interrupted'
        report['failure_stage'] = 'container_process'
    finally:
        # Docker CLI termination does not stop the container. Always clean up its unique name.
        try:
            subprocess.run(docker + ['rm', '-f', name], env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
        except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            report['cleanup_error'] = str(error) or 'Cleanup interrupted; removal is unconfirmed'
            if isinstance(error, KeyboardInterrupt):
                report.update(status='interrupted', failure_stage='container_cleanup')
        report['wall_time_seconds'] = time.perf_counter() - start
        try:
            report['inputs_unchanged'] = inputs == {'bag': inventory(source), 'configuration': inventory(settings)}
            copied = inventory(root/'config')
            report['configuration_copy_matches'] = [(x['relative_path'], x['sha256']) for x in copied] == [(x['relative_path'], x['sha256']) for x in inputs['configuration']]
            if not report['inputs_unchanged'] or not report['configuration_copy_matches']:
                report['status'] = 'inputs_changed'
            trajectory = root/'dump'/'traj_lidar.txt'
            report['trajectory'] = None
            report['trajectory_check'] = None
            if trajectory.is_file() and not trajectory.is_symlink() and trajectory.stat().st_size > 0:
                try:
                    checked = inspect_tum_trajectory(trajectory)
                    artifact = _artifact(trajectory)
                    if any(checked[key] != artifact[key] for key in ('bytes', 'sha256')):
                        raise ValueError('GLIM trajectory changed during validation')
                    report['trajectory'] = artifact
                    report['trajectory_check'] = checked
                except (OSError, ValueError) as error:
                    report['trajectory_error'] = str(error)
                    if report['status'] == 'process_succeeded':
                        report['status'] = 'output_invalid'
            elif report['status'] == 'process_succeeded':
                report['status'] = 'output_missing'
        except (OSError, ValueError, KeyboardInterrupt) as error:
            report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'verification_failed',
                          failure_stage='output_verification', error=str(error) or 'Verification interrupted')
        _save(report_path, report)
    return report
