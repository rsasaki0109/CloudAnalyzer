from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from ca import glim_preflight as check


def message(t, role):
    sec, nano = divmod(round(t * 10**9), 10**9)
    result = NS(header=NS(stamp=NS(sec=sec, nanosec=nano), frame_id=role))
    if role == 'points':
        result.fields = [NS(name=x, offset=i*4, datatype=7, count=1) for i,x in enumerate('xyz')]
        result.point_step, result.row_step = 12, 120
        result.data = bytes(120)
        result.is_bigendian, result.is_dense = False, True
        result.width, result.height = 10, 1
    else:
        for name in ['angular_velocity', 'linear_acceleration']:
            setattr(result, name, NS(x=0., y=0., z=9.81 if name == 'linear_acceleration' else 0.))
            setattr(result, name + '_covariance', [0.] * 9)
    return result


@pytest.fixture
def recording(tmp_path, monkeypatch):
    bag = tmp_path/'bag.mcap'; bag.write_bytes(b'test recording')
    connections = [NS(topic='/points', msgtype='sensor_msgs/msg/PointCloud2'),
                   NS(topic='/imu', msgtype='sensor_msgs/msg/Imu')]
    rows = [(connections[1], 0, message(1, 'imu')), (connections[0], 0, message(1.01, 'points')),
            (connections[0], 0, message(1.02, 'points')), (connections[1], 0, message(1.03, 'imu'))]
    class Reader:
        def __init__(self, paths): self.connections = connections
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def messages(self, connections):
            yield from (r for r in rows if r[0] in connections)
        def deserialize(self, raw, msgtype): return raw
    monkeypatch.setattr(check, 'require_rosbags', lambda: Reader)
    return bag, rows


def test_basic_pass_does_not_claim_calibration(recording):
    bag, _ = recording
    result = check.inspect_glim_bag(str(bag), '/points', '/imu')
    assert result['status'] == 'basic_checks_passed'
    assert result['calibration_verified'] is False
    assert result['streams']['imu']['messages'] == 2


@pytest.mark.parametrize('failure,needle', [('unavailable', 'marked unavailable'),
    ('nonfinite', 'nonfinite'), ('timestamp', 'non-increasing'),
    ('coverage', 'does not cover'), ('frame', 'frame_id'), ('xyz', 'XYZ')])
def test_sensor_faults_block(recording, failure, needle):
    bag, rows = recording
    imu, points = rows[0][2], rows[1][2]
    if failure == 'unavailable': imu.angular_velocity_covariance[0] = -1
    if failure == 'nonfinite': imu.linear_acceleration.z = float('nan')
    if failure == 'timestamp':
        rows[-1][2].header.stamp.sec = 1
        rows[-1][2].header.stamp.nanosec = 0
    if failure == 'coverage': imu.header.stamp.sec = 3
    if failure == 'frame': imu.header.frame_id = ''
    if failure == 'xyz': points.fields = []
    result = check.inspect_glim_bag(str(bag), '/points', '/imu')
    assert result['status'] == 'blocked'
    assert any(needle in s for s in result['issues'])


def test_missing_topics_not_decoded(recording):
    bag, _ = recording
    result = check.inspect_glim_bag(str(bag), '/absent', '/also_absent')
    assert result['status'] == 'blocked'
    assert result['streams']['points']['messages'] == 0


@pytest.mark.parametrize('date', ['2012-04-29', '2012-06-15'])
def test_real_demo_bags_are_not_raw_imu_inputs(date):
    pytest.importorskip('rosbags')
    bag = Path(__file__).resolve().parents[2]/'web/public/samples'/f'nclt-{date}.mcap'
    result = check.inspect_glim_bag(str(bag), '/velodyne_points', '/imu')
    assert result['status'] == 'blocked'
    assert not any('decoding failed' in s for s in result['issues'])
    assert result['imu_unavailable']['angular_velocity'] == result['streams']['imu']['messages'] > 0
    assert result['imu_unavailable']['linear_acceleration'] == result['streams']['imu']['messages']
    assert result['streams']['points']['messages'] > 0
    assert result['point_checks']['invalid_layout_messages'] == 0
    assert result['point_checks']['checked_messages'] == result['streams']['points']['messages']
    assert result['point_checks']['nonfinite_points'] == 0
    assert result['calibration_verified'] is False


@pytest.mark.parametrize('available', [True, False])
def test_ros2_serialized_imu(tmp_path, available):
    pytest.importorskip('rosbags')
    import numpy as np
    from rosbags.rosbag2 import Writer, StoragePlugin
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    def obj(name, **kw): return ts.types[name](**kw)
    with Writer(tmp_path/'bag', version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        c = writer.add_connection('/imu', 'sensor_msgs/msg/Imu', typestore=ts)
        for stamp in [1, 2]:
            v = obj('geometry_msgs/msg/Vector3', x=0., y=0., z=0.)
            msg = obj('sensor_msgs/msg/Imu',
                      header=obj('std_msgs/msg/Header', stamp=obj('builtin_interfaces/msg/Time', sec=stamp, nanosec=0), frame_id='imu'),
                      orientation=obj('geometry_msgs/msg/Quaternion', x=0., y=0., z=0., w=1.),
                      orientation_covariance=np.zeros(9), angular_velocity=v,
                      angular_velocity_covariance=np.array([0. if available else -1.] + [0.]*8),
                      linear_acceleration=v, linear_acceleration_covariance=np.zeros(9))
            writer.write(c, stamp * 10**9, ts.serialize_cdr(msg, c.msgtype))
    bag = next((tmp_path/'bag').glob('*.mcap'))
    result = check.inspect_glim_bag(str(bag), '/points', '/imu')
    assert result['streams']['imu']['messages'] == 2
    assert result['imu_unavailable']['angular_velocity'] == (0 if available else 2)
    assert not any('decoding failed' in s for s in result['issues'])


def test_cli_blocked_exit(recording):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    result = CliRunner().invoke(app, ['mapping-check-glim', str(recording[0]),
                                     '--points-topic', '/missing', '--imu-topic', '/imu'])
    assert result.exit_code == 1
    assert 'topic not found' in result.stdout


@pytest.mark.parametrize('delta,blocked', [(50_000_000,False),(50_000_001,True)])
def test_gap_boundary_at_large_epoch_uses_integer_nanoseconds(recording,delta,blocked):
    bag,rows=recording
    # Point coverage is valid in both cases; only a 1 ns interval difference blocks.
    for index,nano in [(0,0),(1,10_000_000),(2,20_000_000),(3,delta)]:
        rows[index][2].header.stamp.sec=1_335_706_505
        rows[index][2].header.stamp.nanosec=nano
    result=check.inspect_glim_bag(str(bag),'/points','/imu')
    assert (result['status']=='blocked') == blocked
    assert result['imu_gaps']['max_observed_ns']==delta
    assert result['imu_gaps']['count']==int(blocked)


def test_gap_examples_are_bounded_but_all_counted(recording):
    bag,rows=recording
    connection=rows[0][0]
    rows[:]=[rows[1],rows[2]]+[(connection,0,message(1+k*.1,'imu')) for k in range(41)]
    result=check.inspect_glim_bag(str(bag),'/points','/imu')
    assert result['imu_gaps']['count']==40
    assert len(result['imu_gaps']['examples'])==32
    assert result['status']=='blocked'


@pytest.mark.parametrize('limit',[0,-1,float('nan'),float('inf'),3601,1e-10])
def test_invalid_gap_limit_rejected(recording,limit):
    with pytest.raises(ValueError,match='max IMU gap'):
        check.inspect_glim_bag(str(recording[0]),'/points','/imu',max_imu_gap_s=limit)


def test_cli_gap_policy_is_explicit(recording):
    import json
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    args=['mapping-check-glim',str(recording[0]),'--points-topic','/points','--imu-topic','/imu']
    blocked=CliRunner().invoke(app,args+['--max-imu-gap','0.01'])
    passed=CliRunner().invoke(app,args+['--max-imu-gap','0.03'])
    assert blocked.exit_code==1
    assert passed.exit_code==0
    assert json.loads(passed.stdout)['imu_gaps']['limit_ns']==30_000_000


@pytest.mark.parametrize('dense,all_invalid,blocked', [(True,False,True),(False,False,False),(False,True,True)])
def test_nonfinite_point_policy(recording,dense,all_invalid,blocked):
    import struct
    bag,rows=recording
    msg=rows[1][2]
    msg.is_dense=dense
    bad=struct.pack('<fff',float('nan'),float('inf'),0)
    msg.data=bad*10 if all_invalid else bad+msg.data[12:]
    result=check.inspect_glim_bag(str(bag),'/points','/imu')
    assert (result['status']=='blocked')==blocked
    assert result['point_checks']['nonfinite_points']==(10 if all_invalid else 1)
    assert bool(result['warnings']) == (not dense)


def test_bad_layout_does_not_prevent_checking_later_messages(recording):
    bag,rows=recording
    rows[1][2].data=b'bad'
    result=check.inspect_glim_bag(str(bag),'/points','/imu')
    assert result['status']=='blocked'
    assert result['point_checks']['invalid_layout_messages']==1
    assert result['point_checks']['checked_messages']==1
    assert result['streams']['imu']['messages']==2
    assert not any('bag decoding failed' in item for item in result['issues'])
