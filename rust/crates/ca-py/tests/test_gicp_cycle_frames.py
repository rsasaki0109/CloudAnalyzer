"""Coordinate invariants with the explicit guard enabled."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]/"scripts"))
import audit_flatwall_gicp_cycles as independent
import audit_flatwall_gicp_registration as reference


@pytest.mark.parametrize("source_frame", [False, True])
@pytest.mark.parametrize("origin", [np.zeros(3), np.array([1e6, -2e6, 3e6])])
def test_recurrence_witness_and_guarded_alignment_under_frame_changes(source_frame, origin):
    q = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
    frame = np.eye(4)
    frame[:3, :3], frame[:3, 3] = q, origin
    source = np.concatenate([np.eye(3)*2., -np.eye(3)*2.])
    target = source+[.1, -.03, .04]
    cs = np.tile(np.diag([.1, 1., 2.]), (len(source), 1, 1))
    ct, pose, center = cs.copy(), np.eye(4), np.zeros(3)
    options = dict(max_correspondence_distance=1., cycle_guard=core.GicpCycleGuard())
    baseline = core.gicp_register(source, target, cs, ct, pose, center, **options)
    history = []
    for i in range(9):
        p = np.eye(4)
        p[:3, :3] = Rotation.from_rotvec([.03*(i % 3), .01*(i % 3), 0.]).as_matrix()
        p[0, 3] = .01*(i % 3)
        history.append(dict(pose_before=p, source_indices=np.array([0, 1, 2]), target_indices=np.array([0, i % 3+1, 2]),
                            combined_shapes=cs[:3].copy(), weights=np.ones(3)))
    if source_frame:
        source, cs = source@q.T+origin, q@cs@q.T
        inverse = np.linalg.inv(frame)
        pose = pose@inverse
        for row in history: row["pose_before"] = row["pose_before"]@inverse
    else:
        target, ct, center = target@q.T+origin, q@ct@q.T, q@center+origin
        pose = frame@pose
        for row in history:
            row["pose_before"] = frame@row["pose_before"]
            row["combined_shapes"] = q@row["combined_shapes"]@q.T
    witness = independent.cycle_at(history, center)
    assert witness is not None and witness["period"] == 3
    actual = core.gicp_register(source, target, cs, ct, pose, center, **options)
    disabled = core.gicp_register(source, target, cs, ct, pose, center, max_correspondence_distance=1.)
    reference.equal_bits(actual, disabled)
    assert actual["termination"] == baseline["termination"] == "converged_step"
    # Compare physical transforms in a common frame: raw source-frame t contains
    # a rotation times the source origin and amplifies tiny rotation differences.
    common = actual["pose"]@frame if source_frame else np.linalg.inv(frame)@actual["pose"]
    np.testing.assert_allclose(common, baseline["pose"], atol=2e-7, rtol=1e-12)
