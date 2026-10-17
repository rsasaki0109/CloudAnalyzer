"""Analytic controls for centered equations and body-frame motion prediction."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
spec=importlib.util.spec_from_file_location("flatwall_constraints",ROOT/"scripts/audit_flatwall_frontend_constraints.py")
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_prediction_composes_the_previous_body_displacement_on_the_right():
    earlier,last=np.eye(4),np.eye(4)
    earlier[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    earlier[:3,3]=[2,3,4]
    last[:3,:3]=[[-1,0,0],[0,-1,0],[0,0,1]]
    last[:3,3]=[1,5,4]
    found=audit.prediction(np.asarray([earlier,last]))
    np.testing.assert_array_equal(found[:3,:3],[[0,1,0],[-1,0,0],[0,0,1]])
    np.testing.assert_array_equal(found[:3,3],[-1,4,4])
    np.testing.assert_array_equal(audit.prediction([]),np.eye(4))
    np.testing.assert_array_equal(audit.prediction(np.asarray([earlier])),earlier)


def fixture(supported=True):
    points=np.concatenate([np.eye(3),-np.eye(3)]*2) if supported else np.empty((0,3))
    count=len(points)
    delta=np.array([.1,-.2,.3])
    kernel=.5
    squared=np.full(count,float(delta@delta))
    weight=kernel**2/(kernel+float(delta@delta))**2
    h=np.diag([12*weight]*3+[8*weight]*3) if supported else np.zeros((6,6))
    b=np.r_[12*weight*delta,np.zeros(3)] if supported else np.zeros(6)
    snapshot=dict(finite=True,source_sensor=points,source_world=points,matched_source_indices=np.arange(count,dtype=np.int64),
                  target_world=points+delta,squared_distances=squared,weights=np.full(count,weight),
                  kernel=kernel,normal_matrix=h,right_hand_side=b,weight_sum=count*weight,
                  weighted_rms_radius_m=1. if supported else None,center_world=np.zeros(3),pose=np.eye(4),
                  threshold_sq=(kernel*9)**2,alignment_eligible=supported)
    report=dict(termination="converged" if supported else "empty_map",iterations=1 if supported else 0,
                threshold_m=kernel*9,correspondences=dict(matched_points=count) if supported else None)
    return snapshot,report


def test_isotropic_centered_geometry_has_known_equations_and_spectrum():
    snapshot,report=fixture()
    row=audit.describe(snapshot,report,np.eye(4))
    assert row["prediction_only"] is False
    np.testing.assert_allclose(row["geometry"]["eigenvalues"],[2/3]*3+[1.]*3,atol=1e-14)
    assert row["geometry"]["weak_directions_below_1e-3"]==0
    assert row["weighted_rmse_m"]==pytest.approx(np.sqrt(.14))


def test_empty_support_retains_absent_radius_and_spectrum():
    snapshot,report=fixture(False)
    row=audit.describe(snapshot,report,np.eye(4))
    assert row["prediction_only"] is True
    assert row["radius_m"] is row["geometry"] is row["weighted_rmse_m"] is None
    assert row["matches"]==0 and row["weight_sum"]==0.


@pytest.mark.parametrize("field",["normal_matrix","right_hand_side","weights"])
def test_inconsistent_native_equation_is_rejected(field):
    snapshot,report=fixture()
    snapshot[field].flat[0]+=.1
    with pytest.raises(AssertionError):
        audit.describe(snapshot,report,np.eye(4))


def test_unregistered_pose_cannot_masquerade_as_a_measurement_update():
    snapshot,report=fixture(False)
    snapshot["pose"][0,3]=.1
    with pytest.raises(ValueError,match="motion prediction"):
        audit.describe(snapshot,report,np.eye(4))
