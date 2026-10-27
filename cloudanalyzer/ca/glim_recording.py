"""Join explicit point/IMU MCAP streams without inventing inertial measurements."""
from __future__ import annotations

import heapq
from pathlib import Path

from ca.glim_preflight import inspect_glim_bag, imu_gap_limit_ns
from ca.mapping_import import _space
from ca.mapping_job import _artifact, _save


def prepare_glim_recording(lidar: str, imu: str, out: str, points_topic: str, imu_topic: str, *, max_imu_gap_s: float = .05) -> dict:
    imu_gap_limit_ns(max_imu_gap_s)
    try:
        from mcap.reader import make_reader
        from mcap.writer import Writer, CompressionType
    except ImportError as error:
        raise ValueError('Install cloudanalyzer[ros] to prepare MCAP recordings') from error
    if not points_topic or not imu_topic or points_topic == imu_topic:
        raise ValueError('select distinct, nonempty point and IMU topics')
    paths = [Path(lidar).resolve(), Path(imu).resolve()]
    root = Path(out).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError('recording preparation requires a new output directory')
    for path in paths:
        if not path.is_file() or path.suffix.lower() != '.mcap':
            raise ValueError('both inputs must be MCAP files')
    inputs = [_artifact(path) for path in paths]
    _space(root, sum(item['bytes'] for item in inputs))
    # Open and inspect both inputs before creating outputs. Independent readers also
    # allow selecting both streams from one source without double-reading messages.
    from contextlib import ExitStack
    with ExitStack() as stack:
        readers = [make_reader(stack.enter_context(path.open('rb')), validate_crcs=True) for path in paths]
        topics = [points_topic, imu_topic]
        types = ['sensor_msgs/msg/PointCloud2', 'sensor_msgs/msg/Imu']
        channels = []
        schemas = []
        for reader, topic, expected in zip(readers, topics, types):
            summary = reader.get_summary()
            if reader.get_header().profile != 'ros2' or summary is None:
                raise ValueError('inputs need indexed ROS2 MCAP recordings')
            matches = [c for c in summary.channels.values() if c.topic == topic]
            if len(matches) != 1:
                raise ValueError(f'expected exactly one channel for {topic}')
            channel = matches[0]
            schema = summary.schemas.get(channel.schema_id)
            if (schema is None or schema.name != expected or schema.encoding != 'ros2msg'
                    or not schema.data or channel.message_encoding != 'cdr'):
                raise ValueError(f'{topic} needs a complete {expected} ROS2 schema and CDR data')
            channels.append(channel)
            schemas.append(schema)
        root.mkdir(parents=True, exist_ok=False)
        output = root/'recording.mcap'
        report: dict = {'schema': 'cloudanalyzer.glim_recording.v1', 'inputs': inputs,
                  'max_imu_gap_s': max_imu_gap_s, 'points_topic': points_topic, 'imu_topic': imu_topic, 'status': 'preparing',
                  'message_counts': [0, 0], 'calibration_verified': False,
                  'scope': 'Selected channels only; CDR payloads, log/header/publish timestamps, sequences and channel metadata preserved. No resampling, axis conversion, synchronization, deskew or calibration. Other channels, attachments and file metadata are not copied. Basic preflight success is not GLIM readiness.'}
        receipt = root/'preparation.json'
        _save(receipt, report)
        stage = 'recording_write'
        try:
            with output.open('xb') as stream:
                writer = Writer(stream, compression=CompressionType.ZSTD)
                writer.start(profile='ros2', library='CloudAnalyzer GLIM recording preparation')
                ids = []
                for schema, channel in zip(schemas, channels):
                    sid = writer.register_schema(schema.name, schema.encoding, schema.data)
                    ids.append(writer.register_channel(channel.topic, channel.message_encoding, sid, channel.metadata))
                def messages(index):
                    for _, _, message in readers[index].iter_messages(topics=[topics[index]], log_time_order=True):
                        yield message.log_time, index, message
                for _, index, message in heapq.merge(messages(0), messages(1), key=lambda row: (row[0], row[1])):
                    writer.add_message(ids[index], message.log_time, message.data, message.publish_time, message.sequence)
                    report['message_counts'][index] += 1
                writer.finish()
            stage = 'input_verification'
            if inputs != [_artifact(path) for path in paths]:
                raise ValueError('input changed during recording preparation')
            stage = 'output_verification'
            report['output'] = _artifact(output)
            stage = 'input_preflight'
            report['preflight'] = inspect_glim_bag(str(output), points_topic, imu_topic, max_imu_gap_s=max_imu_gap_s)
            report['status'] = report['preflight']['status']
        except (Exception, KeyboardInterrupt) as error:
            report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'preparation_failed'
            report['failure_stage'] = stage
            report['error'] = str(error) or 'Recording preparation interrupted'
            raise
        finally:
            _save(receipt, report)
    return report
