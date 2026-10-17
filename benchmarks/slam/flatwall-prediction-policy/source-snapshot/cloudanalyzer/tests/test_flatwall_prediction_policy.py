"""Boundary controls for last-pose initialization and exact cohort identity."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
spec=importlib.util.spec_from_file_location("flatwall_prediction",ROOT/"scripts/audit_flatwall_prediction_policy.py")
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_first_scan_has_no_prediction_and_later_scan_uses_body_identity():
    class Spy:
        def __init__(self):
            self.calls=[]
        def register(self,points,**kwargs):
            self.calls.append(kwargs)
            return np.eye(4)
    native=Spy()
    points=np.zeros((12,3))
    _,first=audit.last_pose_step(native,points,None)
    assert native.calls[0]=={}
    np.testing.assert_array_equal(first,np.eye(4))
    previous=np.eye(4)
    previous[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    previous[:3,3]=[2.,3.,4.]
    _,predicted=audit.last_pose_step(native,points,previous)
    np.testing.assert_array_equal(native.calls[1]["relative_prediction"],np.eye(4))
    np.testing.assert_array_equal(predicted,previous)
    assert not np.shares_memory(predicted,previous)


def make_cohort(directory,count=185):
    times=np.arange(count,dtype=np.int64)+1_656_505_431_000_000_000
    np.savez(directory/"trajectory.npz",timestamps_ns=times,poses=np.tile(np.eye(4),(count,1,1)))
    rows=[dict(scan_index=i+1,reference_ns=int(time)) for i,time in enumerate(times)]
    (directory/"per-scan.json").write_text(json.dumps(rows),encoding="utf-8")
    return times,rows


@pytest.mark.parametrize("count",[64,184,186])
def test_partial_or_extended_cohort_is_rejected(tmp_path,count):
    make_cohort(tmp_path,count)
    with pytest.raises(ValueError,match="185"):
        audit.cohort(tmp_path)


def test_relabelled_original_scan_is_rejected(tmp_path):
    _,rows=make_cohort(tmp_path)
    rows[75]["scan_index"]=77
    (tmp_path/"per-scan.json").write_text(json.dumps(rows),encoding="utf-8")
    with pytest.raises(ValueError,match="scan identities"):
        audit.cohort(tmp_path)


def test_single_nanosecond_time_change_is_rejected(tmp_path):
    times,_=make_cohort(tmp_path)
    changed=times.copy()
    changed[70]+=1
    with pytest.raises(ValueError,match="cohort"):
        audit.cohort(tmp_path,changed)
