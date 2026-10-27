"""Read-only sensor checks for GLIM LiDAR/inertial input, not calibration approval."""
from __future__ import annotations

from pathlib import Path
import math
from typing import TypedDict


class StreamStats(TypedDict):
    messages: int
    first_stamp_ns: int | None
    last_stamp_ns: int | None
    non_increasing_stamps: int
    invalid_stamps: int


class ImuGaps(TypedDict):
    limit_ns: int
    max_observed_ns: int | None
    count: int
    examples: list[dict[str, int]]
    example_limit: int
    scope: str

from ca.core.bag_ingest import require_rosbags
from ca.mapping_job import _artifact
from ca.glim_points import inspect_pointcloud


def imu_gap_limit_ns(seconds: float) -> int:
    if not math.isfinite(seconds) or not 1e-9 <= seconds <= 3600:
        raise ValueError('max IMU gap must be finite and between 1 ns and 3600 seconds')
    return round(seconds * 1_000_000_000)


def inspect_glim_bag(path: str, points_topic: str, imu_topic: str, *, max_imu_gap_s: float = .05) -> dict:
    limit_ns = imu_gap_limit_ns(max_imu_gap_s)
    if not points_topic or not imu_topic or points_topic == imu_topic:
        raise ValueError('select distinct, nonempty points and IMU topics')
    bag = Path(path)
    if not bag.is_file() or bag.suffix != '.mcap':
        raise ValueError('this preflight currently requires a ROS2 MCAP file')
    before = _artifact(bag)
    issues = set()
    stats: dict[str, StreamStats] = {name: {'messages': 0, 'first_stamp_ns': None, 'last_stamp_ns': None,
                    'non_increasing_stamps': 0, 'invalid_stamps': 0} for name in ['points', 'imu']}
    missing = {'angular_velocity': 0, 'linear_acceleration': 0}
    nonfinite = 0
    gaps: ImuGaps = {'limit_ns': limit_ns, 'max_observed_ns': None, 'count': 0,
            'examples': [], 'example_limit': 32, 'scope': 'all selected IMU messages'}
    point_checks = {'checked_messages': 0, 'invalid_layout_messages': 0, 'points': 0,
                    'finite_points': 0, 'nonfinite_points': 0, 'nondense_messages_with_nonfinite': 0}
    fields_seen: set[str] = set()
    frames: dict[str, set[str]] = {'points': set(), 'imu': set()}
    AnyReader = require_rosbags()
    try:
        with AnyReader([bag]) as reader:
            selected = []
            for role, topic, msgtype in [('points', points_topic, 'sensor_msgs/msg/PointCloud2'),
                                         ('imu', imu_topic, 'sensor_msgs/msg/Imu')]:
                connections = [c for c in reader.connections if c.topic == topic]
                if not connections:
                    issues.add(f'{role}: topic not found: {topic}')
                for connection in connections:
                    if connection.msgtype != msgtype:
                        issues.add(f'{role}: unexpected message type: {connection.msgtype}')
                    else:
                        selected.append(connection)
            for connection, _, raw in (reader.messages(connections=selected) if selected else []):
                role = 'points' if connection.topic == points_topic else 'imu'
                message = reader.deserialize(raw, connection.msgtype)
                row = stats[role]
                row['messages'] += 1
                stamp = message.header.stamp
                if not 0 <= stamp.nanosec < 1_000_000_000:
                    row['invalid_stamps'] += 1
                ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                if role == 'imu' and row['last_stamp_ns'] is not None:
                    delta = ns - row['last_stamp_ns']
                    if delta > 0:
                        gaps['max_observed_ns'] = max(gaps['max_observed_ns'] or 0, delta)
                        if delta > limit_ns:
                            gaps['count'] += 1
                            if len(gaps['examples']) < gaps['example_limit']:
                                gaps['examples'].append({'start_ns': row['last_stamp_ns'],
                                                         'end_ns': ns, 'duration_ns': delta})
                if row['last_stamp_ns'] is not None and ns <= row['last_stamp_ns']:
                    row['non_increasing_stamps'] += 1
                if row['first_stamp_ns'] is None:
                    row['first_stamp_ns'] = ns
                row['last_stamp_ns'] = ns
                frames[role].add(message.header.frame_id)
                if role == 'imu':
                    for name in missing:
                        if getattr(message, name + '_covariance')[0] == -1:
                            missing[name] += 1
                    values = [getattr(getattr(message, name), axis) for name in missing for axis in 'xyz']
                    if not all(math.isfinite(float(v)) for v in values):
                        nonfinite += 1
                else:
                    try:
                        checked = inspect_pointcloud(message)
                        fields_seen.update(f.name for f in message.fields)
                        point_checks['checked_messages'] += 1
                        for key in ('points', 'finite_points', 'nonfinite_points'):
                            point_checks[key] += checked[key]
                        if not checked['finite_points']:
                            issues.add('points: point cloud has no finite XYZ records')
                        if checked['nonfinite_points']:
                            if message.is_dense:
                                issues.add('points: dense cloud contains nonfinite XYZ records')
                            else:
                                point_checks['nondense_messages_with_nonfinite'] += 1
                    except (ValueError, TypeError, BufferError) as error:
                        point_checks['invalid_layout_messages'] += 1
                        issues.add(f'points: {error}')
    except Exception as error:
        # Retain partial counts but never label a partial/undecodable scan as checked.
        issues.add(f'bag decoding failed: {type(error).__name__}: {error}')
    for role, row in stats.items():
        if row['messages'] < 2:
            issues.add(f'{role}: fewer than two messages')
        if row['non_increasing_stamps'] or row['invalid_stamps']:
            issues.add(f'{role}: invalid or non-increasing header timestamps')
        if len(frames[role]) != 1 or '' in frames[role]:
            issues.add(f'{role}: missing or changing frame_id')
    for name, count in missing.items():
        if count:
            issues.add(f'imu: {name} marked unavailable in {count} messages')
    if nonfinite:
        issues.add(f'imu: nonfinite measurements in {nonfinite} messages')
    if gaps['count']:
        issues.add(f"imu: {gaps['count']} header timestamp gaps exceed {limit_ns} ns")
    p, i = stats['points'], stats['imu']
    p_first, p_last = p['first_stamp_ns'], p['last_stamp_ns']
    i_first, i_last = i['first_stamp_ns'], i['last_stamp_ns']
    if p_first is not None and p_last is not None and i_first is not None and i_last is not None:
        if i_first > p_first or i_last < p_last:
            issues.add('imu: header timestamp span does not cover the point cloud span')
    after = _artifact(bag)
    if before != after:
        issues.add('bag changed during inspection')
    return {'schema': 'cloudanalyzer.glim_preflight.v1', 'input': before,
            'status': 'blocked' if issues else 'basic_checks_passed',
            'points_topic': points_topic, 'imu_topic': imu_topic, 'streams': stats,
            'imu_gaps': gaps, 'imu_unavailable': missing, 'imu_nonfinite_messages': nonfinite,
            'point_fields_seen': sorted(fields_seen), 'point_checks': point_checks,
            'warnings': ['Non-dense clouds contain nonfinite XYZ records; no filtering was applied.'] if point_checks['nondense_messages_with_nonfinite'] else [],
            'frames': {k: sorted(v) for k, v in frames.items()}, 'issues': sorted(issues),
            'calibration_verified': False,
            'scope': 'All selected messages checked for basic structure, IMU availability and header timing. Does not verify per-point timing, units, clock alignment, extrinsics or GLIM configuration. Passing is not SLAM readiness or accuracy approval.'}
