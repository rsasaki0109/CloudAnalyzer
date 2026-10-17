"""Independent evo error arithmetic with the previously audited exact pairing."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from evo.core import geometry, lie_algebra as lie, metrics, trajectory

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'notes/slam-flatwall-prediction-policy-20261017'
ARMS=['raw-legacy','gyro-legacy','raw-0.1875','gyro-0.1875']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


freeze=read(STUDY/'controller-freeze.json')
boundary=read(STUDY/'scoring-boundary.json')
def verify():
    for name,digest in freeze['files_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        assert sha(path)==digest,name
    for name,digest in boundary['files_sha256'].items():
        assert sha(ROOT/name)==digest,name
verify()
out=STUDY/'oracle'
out.mkdir(exist_ok=False)
with np.load(ROOT/'notes/slam-flatwall-frontend-controls-20261017/revision-02/results/raw-legacy/paired.npz',allow_pickle=False) as old:
    reference,indices,times=[old[name].copy() for name in ['reference','original_indices','timestamps_ns']]
results={policy:{} for policy in ['repeat-motion','last-pose']}
for policy in results:
    for arm in ARMS:
        folder=STUDY/'results'/policy/arm
        summary=read(folder/'summary.json')
        with np.load(folder/'paired.npz',allow_pickle=False) as paired:
            for name,expected in [('reference',reference),('original_indices',indices),('timestamps_ns',times)]:
                np.testing.assert_array_equal(paired[name],expected)
            estimate=paired['estimated'].copy()
        values,maximum={},{}
        def ape(name,poses,relation):
            metric=metrics.APE(relation)
            metric.process_data((trajectory.PosePath3D(poses_se3=reference),trajectory.PosePath3D(poses_se3=poses)))
            values[name]=metric.error
        origin=reference[0]@lie.se3_inverse(estimate[0])
        ape('origin_position_ape_m',origin@estimate,metrics.PoseRelation.translation_part)
        ape('origin_rotation_ape_deg',origin@estimate,metrics.PoseRelation.rotation_angle_deg)
        if summary['position_fit']['status']=='identified':
            rotation,translation,scale=geometry.umeyama_alignment(estimate[:,:3,3].T,reference[:,:3,3].T,with_scale=False)
            assert scale==1.
            transform=lie.se3(rotation,translation)
            np.testing.assert_allclose(transform,summary['position_fit']['world_alignment'],rtol=0,atol=2e-8)
            maximum['world_alignment_matrix']=float(np.max(np.abs(transform-summary['position_fit']['world_alignment'])))
            ape('rigid_position_ape_m',transform@estimate,metrics.PoseRelation.translation_part)
        edges=np.flatnonzero(np.diff(indices)==1)
        deltas=[metrics.RPE.rpe_base(reference[i],reference[i+1],estimate[i],estimate[i+1]) for i in edges]
        values['consecutive_rpe_translation_m']=np.array([np.linalg.norm(delta[:3,3]) for delta in deltas])
        values['consecutive_rpe_rotation_deg']=np.array([abs(lie.so3_log_angle(delta[:3,:3],degrees=True)) for delta in deltas])
        with np.load(folder/'errors.npz',allow_pickle=False) as actual:
            assert set(actual.files)==set(values)
            for name,expected in values.items():
                np.testing.assert_allclose(actual[name],expected,rtol=0,atol=1e-5 if name.endswith('deg') else 2e-8)
                maximum[name]=float(np.max(np.abs(actual[name]-expected)))
        np.savez_compressed(out/(policy+'--'+arm+'.npz'),**values)
        results[policy][arm]=dict(paired_estimates=len(estimate),consecutive_edges=len(deltas),maximum_abs_differences=maximum,
                                 prior_exact_pairing_reproduced=True)
verify()
receipt=dict(at_utc=datetime.now(timezone.utc).isoformat(),controls=results,controller_freeze_sha256=sha(STUDY/'controller-freeze.json'),
    scoring_boundary_sha256=sha(STUDY/'scoring-boundary.json'),input_hashes_unchanged=True,physical_provenance_certified=False,
    methods='Installed evo APE/RPE and no-scale Umeyama on eight paired cohorts. All reference matrices, integer times and indices must equal the previous independently audited pairing; no new independent raw-reference association claim.',
    outputs_sha256={p.name:sha(p) for p in out.iterdir()})
with (out/'receipt.json').open('x',encoding='utf-8',newline='\n') as f:
    json.dump(receipt,f,indent=2,allow_nan=False)
    f.write('\n')
print(json.dumps(receipt,indent=2))
