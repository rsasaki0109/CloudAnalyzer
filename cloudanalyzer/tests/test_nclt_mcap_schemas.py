"""MCAP schema repair preserves CDR data and refuses unrelated schemas."""
from pathlib import Path

import pytest

pytest.importorskip('mcap')
pytest.importorskip('rosbags')


@pytest.fixture
def repair_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    import repair_nclt_mcap_schemas
    return repair_nclt_mcap_schemas


def test_repairs_empty_schema_without_changing_message(tmp_path, repair_module):
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from rosbags.highlevel import AnyReader
    sample = Path(__file__).resolve().parents[2]/'web/public/samples/nclt-2012-04-29.mcap'
    with sample.open('rb') as f:
        schema, channel, message = next(make_reader(f).iter_messages(topics=['/imu']))
    old, new = tmp_path/'old.mcap', tmp_path/'new.mcap'
    with old.open('wb') as f:
        w=Writer(f); w.start(profile='ros2')
        sid=w.register_schema(schema.name, 'ros2msg', b'')
        cid=w.register_channel(channel.topic, 'cdr', sid, {'test': 'preserve'})
        w.add_message(cid, message.log_time, message.data, message.publish_time, message.sequence)
        w.finish()
    before=old.read_bytes()
    assert repair_module.repair(old,new)==1
    assert old.read_bytes()==before
    with new.open('rb') as f:
        schema,channel,actual=next(make_reader(f).iter_messages())
    assert schema.data
    assert channel.metadata=={'test':'preserve'}
    assert (actual.data, actual.log_time, actual.publish_time, actual.sequence)==(message.data,message.log_time,message.publish_time,message.sequence)
    with AnyReader([new]) as reader:
        for c,_,raw in reader.messages():
            assert reader.deserialize(raw,c.msgtype).angular_velocity_covariance[0]==-1
    with pytest.raises(FileExistsError): repair_module.repair(old,new)
    with pytest.raises(ValueError,match='only known empty'): repair_module.repair(new,tmp_path/'again.mcap')


def test_generator_uses_complete_schemas(repair_module):
    from rosbags.typesys import get_types_from_msg
    for name, raw in repair_module.schemas().items():
        types=get_types_from_msg(raw.decode(),name)
        assert 'std_msgs/msg/Header' in types
        assert types[name][1]  # Root message has fields, not an empty definition.
