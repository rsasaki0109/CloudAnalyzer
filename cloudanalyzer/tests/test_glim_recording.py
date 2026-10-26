"""Real point payloads and synthetic inertial fixtures; not SLAM accuracy tests."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mcap')
pytest.importorskip('rosbags')
from mcap.reader import make_reader
from mcap.writer import Writer
from rosbags.typesys import Stores, get_typestore
from ca.glim_recording import prepare_glim_recording


@pytest.fixture
def recordings(tmp_path, monkeypatch):
    monkeypatch.setattr('ca.glim_recording._space', lambda *args: None)
    source = Path(__file__).resolve().parents[2]/'web/public/samples/nclt-2012-04-29.mcap'
    points = tmp_path/'points.mcap'
    selected = []
    with source.open('rb') as src, points.open('wb') as dst:
        writer = Writer(dst); writer.start(profile='ros2')
        for schema, channel, msg in make_reader(src).iter_messages(topics=['/velodyne_points']):
            if not selected:
                sid = writer.register_schema(schema.name, schema.encoding, schema.data)
                cid = writer.register_channel('/points', 'cdr', sid, {'source': 'fixture'})
            writer.add_message(cid, msg.log_time, msg.data, msg.publish_time, msg.sequence)
            selected.append(msg)
            if len(selected) == 2: break
        writer.finish()
    ts = get_typestore(Stores.ROS2_HUMBLE)
    first, last = [ts.deserialize_cdr(m.data, 'sensor_msgs/msg/PointCloud2').header.stamp for m in selected]
    times = [s.sec * 10**9 + s.nanosec for s in [first, last]]
    def raw_imu(available=True, cover=True):
        path = tmp_path/f'imu-{available}-{cover}.mcap'
        def obj(name, **kw): return ts.types[name](**kw)
        with path.open('wb') as f:
            w = Writer(f); w.start(profile='ros2')
            name = 'sensor_msgs/msg/Imu'
            sid = w.register_schema(name, 'ros2msg', ts.generate_msgdef(name, ros_version=2)[0].encode())
            cid = w.register_channel('/raw', 'cdr', sid)
            for seq, ns in enumerate([*range(times[0] - 1 if cover else times[0] + 1, times[1], 10_000_000), times[1] + 1]):
                sec, nanos = divmod(ns, 10**9)
                msg = obj(name, header=obj('std_msgs/msg/Header', stamp=obj('builtin_interfaces/msg/Time', sec=sec, nanosec=nanos), frame_id='nclt_ms25_native'),
                          orientation=obj('geometry_msgs/msg/Quaternion', x=0., y=0., z=0., w=0.), orientation_covariance=np.array([-1.]+[0.]*8),
                          angular_velocity=obj('geometry_msgs/msg/Vector3', x=.1, y=.2, z=.3), angular_velocity_covariance=np.array([0. if available else -1.]+[0.]*8),
                          linear_acceleration=obj('geometry_msgs/msg/Vector3', x=0., y=0., z=-9.81), linear_acceleration_covariance=np.zeros(9))
                w.add_message(cid, ns, ts.serialize_cdr(msg, name).tobytes(), ns, seq)
            w.finish()
        return path
    return points, raw_imu


def records(path, topic):
    with path.open('rb') as f:
        return [(c.topic, c.metadata, s.data, m.data, m.log_time, m.publish_time, m.sequence)
                for s,c,m in make_reader(f).iter_messages(topics=[topic])]


def test_join_preserves_every_selected_message(recordings, tmp_path):
    points, make_imu = recordings; imu = make_imu(); out = tmp_path/'prepared'
    result = prepare_glim_recording(str(points), str(imu), str(out), '/points', '/raw')
    assert result['status'] == 'basic_checks_passed'
    assert result['message_counts'] == [2,len(records(imu,'/raw'))]
    assert result['calibration_verified'] is False
    for source, topic in [(points, '/points'), (imu, '/raw')]:
        assert records(source,topic) == records(out/'recording.mcap',topic)
    assert json.loads((out/'preparation.json').read_text()) == result
    with pytest.raises(FileExistsError):
        prepare_glim_recording(str(points),str(imu),str(out),'/points','/raw')


@pytest.mark.parametrize('available,cover,needle', [(False,True,'unavailable'),(True,False,'does not cover')])
def test_invalid_imu_is_retained_as_blocked(recordings,tmp_path,available,cover,needle):
    points, make_imu = recordings; out = tmp_path/'blocked'
    result = prepare_glim_recording(str(points),str(make_imu(available,cover)),str(out),'/points','/raw')
    assert result['status'] == 'blocked'
    assert any(needle in issue for issue in result['preflight']['issues'])
    assert (out/'preparation.json').exists()


def test_missing_topic_does_not_create_output(recordings,tmp_path):
    points,make_imu=recordings;out=tmp_path/'bad'
    with pytest.raises(ValueError,match='exactly one channel'):
        prepare_glim_recording(str(points),str(make_imu()),str(out),'/missing','/raw')
    assert not out.exists()


def test_cli_reports_blocked(recordings,tmp_path):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    points,make_imu=recordings
    result=CliRunner().invoke(app,['mapping-prepare-glim-recording',str(points),'--imu',str(make_imu(False)),
                                 '--out',str(tmp_path/'cli'),'--points-topic','/points','--imu-topic','/raw'])
    assert result.exit_code==1
    assert json.loads(result.stdout)['status']=='blocked'


def test_join_propagates_gap_policy(recordings,tmp_path):
    points,make_imu=recordings
    result=prepare_glim_recording(str(points),str(make_imu()),str(tmp_path/'gap-policy'),
                                  '/points','/raw',max_imu_gap_s=.005)
    assert result['status']=='blocked'
    assert result['preflight']['imu_gaps']['count']>0
    assert result['preflight']['imu_gaps']['limit_ns']==5_000_000
    assert result['max_imu_gap_s']==.005
