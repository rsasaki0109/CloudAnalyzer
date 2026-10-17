"""Independent evo error arithmetic on the fixed, previously audited pairing."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from evo.core import geometry, lie_algebra as lie, metrics, trajectory

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'notes/slam-flatwall-frontend-controls-20261017/revision-02'
ARMS = ['raw-legacy','gyro-legacy','raw-0.1875','gyro-0.1875','nominal-imu-legacy']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    freeze = read(STUDY / 'scoring-boundary.json')
    def verify():
        assert sha(Path(__file__)) == freeze['oracle_controller_sha256']
        for name, expected in freeze['files_sha256'].items():
            assert sha(ROOT / name) == expected, name
    verify()
    out = STUDY / 'oracle'
    out.mkdir(exist_ok=False)
    baseline = ROOT / 'notes/slam-flatwall-trajectory-20261017/strict-time-regression/results/paired.npz'
    with np.load(baseline, allow_pickle=False) as old:
        reference, indices, times = [old[n].copy() for n in ['reference','original_indices','timestamps_ns']]
    results = {}
    for name in ARMS:
        folder = STUDY / 'results' / name
        summary = read(folder / 'summary.json')
        with np.load(folder / 'paired.npz', allow_pickle=False) as pairs:
            for key, expected in [('reference',reference),('original_indices',indices),('timestamps_ns',times)]:
                np.testing.assert_array_equal(pairs[key], expected)
            estimate = pairs['estimated'].copy()
        values, maximum = {}, {}
        def ape(key, poses, relation):
            metric = metrics.APE(relation)
            metric.process_data((trajectory.PosePath3D(poses_se3=reference), trajectory.PosePath3D(poses_se3=poses)))
            values[key] = metric.error
        origin = reference[0] @ lie.se3_inverse(estimate[0])
        ape('origin_position_ape_m', origin @ estimate, metrics.PoseRelation.translation_part)
        ape('origin_rotation_ape_deg', origin @ estimate, metrics.PoseRelation.rotation_angle_deg)
        if summary['position_fit']['status'] == 'identified':
            rotation, translation, scale = geometry.umeyama_alignment(estimate[:,:3,3].T, reference[:,:3,3].T, with_scale=False)
            assert scale == 1.
            transform = lie.se3(rotation, translation)
            np.testing.assert_allclose(transform, summary['position_fit']['world_alignment'], rtol=0, atol=2e-8)
            maximum['world_alignment_matrix'] = float(np.max(np.abs(transform-summary['position_fit']['world_alignment'])))
            ape('rigid_position_ape_m', transform @ estimate, metrics.PoseRelation.translation_part)
        edges = np.flatnonzero(np.diff(indices)==1)
        deltas = [metrics.RPE.rpe_base(reference[i],reference[i+1],estimate[i],estimate[i+1]) for i in edges]
        values['consecutive_rpe_translation_m'] = np.array([np.linalg.norm(delta[:3,3]) for delta in deltas])
        values['consecutive_rpe_rotation_deg'] = np.array([abs(lie.so3_log_angle(delta[:3,:3], degrees=True)) for delta in deltas])
        with np.load(folder / 'errors.npz', allow_pickle=False) as actual:
            assert set(actual.files) == set(values)
            for key, expected in values.items():
                np.testing.assert_allclose(actual[key], expected, rtol=0, atol=1e-5 if key.endswith('deg') else 2e-8)
                maximum[key] = float(np.max(np.abs(actual[key]-expected)))
        np.savez_compressed(out / (name+'.npz'), **values)
        results[name] = dict(paired_estimates=len(estimate), consecutive_edges=len(deltas), maximum_abs_differences=maximum,
                             prior_exact_pairing_reproduced=True)
    verify()
    receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), controls=results, scoring_boundary_sha256=sha(STUDY/'scoring-boundary.json'),
                   methods='Installed evo APE/RPE and no-scale Umeyama on the exact paired arrays; all reference matrices/indices/times must equal the previous independently audited pairing bit for bit. No new independent raw-reference association claim.',
                   physical_provenance_certified=False, input_hashes_unchanged=True,
                   outputs_sha256={p.name: sha(p) for p in out.iterdir()})
    save(out / 'receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
