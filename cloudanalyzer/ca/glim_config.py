"""Read GLIM's comment-bearing JSON topic selection without rewriting settings."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

MAX_CONFIG_BYTES = 1024**2
TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|//[^\r\n]*|/\*[\s\S]*?\*/')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate configuration key')
        result[key] = value
    return result


def _read(path: Path) -> tuple[dict, dict]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_CONFIG_BYTES:
        raise ValueError('GLIM config must be a regular file of at most 1 MiB')
    with path.open('rb') as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError('GLIM config exceeds 1 MiB')
    text = raw.decode('utf-8')
    # Preserve string literals (including comment markers) and token separation.
    text = TOKENS.sub(lambda m: m[0] if m[0].startswith('"') else ' ', text)
    def invalid_constant(value):
        raise ValueError('nonfinite JSON number')
    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        return value
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=invalid_constant,
                           parse_float=finite_float)
    except (ValueError, RecursionError) as error:
        raise ValueError(f'invalid GLIM config JSON: {error}') from error
    if not isinstance(value, dict):
        raise ValueError('GLIM config root must be an object')
    return value, {'path': str(path.resolve()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def inspect_config_topics(directory: str | Path, points_topic: str, imu_topic: str) -> dict:
    root = Path(directory).resolve()
    global_config, first = _read(root/'config.json')
    selection = global_config.get('global', {})
    if not isinstance(selection, dict):
        raise ValueError('GLIM global settings must be an object')
    name = selection.get('config_ros', 'config_ros.json')
    if (not isinstance(name, str) or not name or len(name) > 1024 or Path(name).is_absolute()
            or '..' in Path(name).parts or '\\' in name):
        raise ValueError('config_ros must name a relative file inside the configuration directory')
    target = root/name
    if not target.resolve().is_relative_to(root) or target.is_symlink():
        raise ValueError('config_ros must stay inside the configuration directory')
    ros, second = _read(target)
    settings = ros.get('glim_ros', {})
    if not isinstance(settings, dict):
        raise ValueError('glim_ros settings must be an object')
    # These defaults follow glim_rosbag.cpp, not the sample config's sensor names.
    topics = {'points_topic': settings.get('points_topic', '/points'), 'imu_topic': settings.get('imu_topic', '/imu')}
    for key, value in topics.items():
        if not isinstance(value, str) or not value or len(value) > 2048:
            raise ValueError(f'invalid configured {key}')
    if topics != {'points_topic': points_topic, 'imu_topic': imu_topic}:
        raise ValueError(f'diagnostic topics differ from GLIM config: points={topics["points_topic"]}, imu={topics["imu_topic"]}')
    return {'schema': 'cloudanalyzer.glim_config_topics.v1', 'status': 'matched', **topics,
            'files': [first, second],
            'scope': 'Configured bag-reader topic names only; sensor calibration, modules, offsets and scaling are not validated.'}
