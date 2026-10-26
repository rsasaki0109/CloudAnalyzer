from types import SimpleNamespace as NS
import struct

import pytest
from ca import glim_points as points


def cloud(big=False, double=False):
    width = 8 if double else 4
    code = 'd' if double else 'f'
    row = b''.join(struct.pack(('>' if big else '<')+code*3, *xyz) for xyz in [(1,2,3),(4,5,6)])
    # Padding is deliberately nonfinite-looking; it is not a point record.
    data = (row + b'\xff'*8)*2
    return NS(width=2,height=2,point_step=3*width,row_step=len(row)+8,data=data,
              is_bigendian=big,is_dense=True,
              fields=[NS(name=name,offset=i*width,count=1,datatype=8 if double else 7) for i,name in enumerate('xyz')])


@pytest.mark.parametrize('big',[False,True])
@pytest.mark.parametrize('double',[False,True])
def test_padded_organized_clouds_and_both_byte_orders(big,double):
    message=cloud(big,double)
    assert points.inspect_pointcloud(message)=={'points':4,'finite_points':4,'nonfinite_points':0}


@pytest.mark.parametrize('fault', ['short','long','row_step','field_offset','count','duplicate','missing',
                                    'integer_xyz','overlap','empty','too_many'])
def test_invalid_structure_rejected(fault):
    msg=cloud()
    if fault=='short': msg.data=msg.data[:-1]
    if fault=='long': msg.data+=b'\0'
    if fault=='row_step': msg.row_step=1
    if fault=='field_offset': msg.fields[0].offset=msg.point_step
    if fault=='count': msg.fields[0].count=0
    if fault=='duplicate': msg.fields[1].name='x'
    if fault=='missing': msg.fields[1].name='intensity'
    if fault=='integer_xyz': msg.fields[0].datatype=5
    if fault=='overlap': msg.fields[1].offset=0
    if fault=='empty': msg.width=0
    if fault=='too_many': msg.width=points.MAX_POINTS+1
    with pytest.raises(ValueError): points.inspect_pointcloud(msg)


def test_nonfinite_xyz_counted_per_record_not_per_axis():
    msg=cloud()
    msg.data=struct.pack('<fff',float('nan'),float('inf'),1)+msg.data[12:]
    assert points.inspect_pointcloud(msg)=={'points':4,'finite_points':3,'nonfinite_points':1}


def test_byte_budget_checked_before_array_views(monkeypatch):
    monkeypatch.setattr(points,'MAX_DATA_BYTES',10)
    with pytest.raises(ValueError,match='diagnostic limit'):
        points.inspect_pointcloud(cloud())
