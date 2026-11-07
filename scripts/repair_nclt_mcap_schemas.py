"""Repair only the known empty ROS schemas in an NCLT demo MCAP.

Usage: python scripts/repair_nclt_mcap_schemas.py OLD.mcap NEW.mcap
Requires mcap, zstandard and rosbags. The source is never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer
from rosbags.typesys import Stores, get_typestore

from nclt_ros2_schemas import schemas


def repair(source: Path, output: Path) -> int:
    definitions = schemas()
    store = get_typestore(Stores.ROS2_HUMBLE)
    with source.open('rb') as stream:
        reader = make_reader(stream, validate_crcs=True)
        summary = reader.get_summary()
        if (summary is None or summary.statistics is None or reader.get_header().profile != 'ros2'
                or summary.statistics.attachment_count or summary.statistics.metadata_count):
            raise ValueError('expected an indexed ROS2 demo without attachments or metadata')
        if not summary.schemas or any(s.name not in definitions or s.encoding != 'ros2msg' or s.data
                                      for s in summary.schemas.values()):
            raise ValueError('only known empty PointCloud2/Imu ROS2 schemas can be repaired')
        if any(c.schema_id not in summary.schemas or c.message_encoding != 'cdr'
               for c in summary.channels.values()):
            raise ValueError('expected only schema-backed CDR channels')
        # Exclusive creation protects source and existing output files.
        with output.open('xb') as target:
            writer = Writer(target, compression=CompressionType.ZSTD, chunk_size=1 << 20)
            writer.start(profile='ros2', library='CloudAnalyzer repair_nclt_mcap_schemas.py')
            schema_ids = {key: writer.register_schema(s.name, s.encoding, definitions[s.name])
                          for key, s in summary.schemas.items()}
            channel_ids = {key: writer.register_channel(c.topic, c.message_encoding, schema_ids[c.schema_id], c.metadata)
                           for key, c in summary.channels.items()}
            count = 0
            for schema, channel, message in reader.iter_messages(log_time_order=False):
                # Check the unchanged payload against the standard schema before emitting it.
                store.deserialize_cdr(message.data, schema.name)
                writer.add_message(channel_ids[channel.id], message.log_time, message.data,
                                   message.publish_time, message.sequence)
                count += 1
            if count != summary.statistics.message_count:
                raise ValueError('message count differs from original summary')
            writer.finish()
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(f'Repaired schemas for {repair(args.source, args.output)} unchanged messages')
