"""Independent selected-member/decimal association and installed evo arithmetic."""
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation
from evo.core import geometry, lie_algebra as lie, metrics, trajectory

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'notes/slam-flatwall02-prediction-20261017'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


freeze = read(STUDY/'controller-freeze.json')
boundary = read(STUDY/'results/scoring-boundary.json')
def verify():
    for name, digest in freeze['files_sha256'].items():
        assert sha(Path(name) if Path(name).is_absolute() else ROOT/name) == digest, name
    for name, digest in boundary['files_sha256'].items():
        assert sha(ROOT/name) == digest, name
verify()
assert boundary['all_eight_estimates_and_snapshots_closed'] is True
out = STUDY/'oracle'
out.mkdir(exist_ok=False)
with tarfile.open(ROOT/'notes/slam-flatwall-inputs-20261016/gt.tar.gz', 'r:gz') as archive:
    members = [m for m in archive.getmembers() if m.name == 'gt/flatwall_02.txt']
    assert len(members) == 1 and members[0].isfile() and 0 < members[0].size <= 8_388_608
    payload = archive.extractfile(members[0]).read()
assert payload == (STUDY/'results/reference-source.txt').read_bytes()
stamps, reference_rows = [], []
with localcontext() as context:
    context.prec = 80
    for line in payload.decode('utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        tokens = line.split()
        assert len(tokens) == 8
        scaled = Decimal(tokens[0])*Decimal(1_000_000_000)
        assert scaled.is_finite() and scaled == scaled.to_integral_value()
        stamps.append(int(scaled))
        reference_rows.append([float(x) for x in tokens[1:]])
assert all(b > a for a, b in zip(stamps, stamps[1:]))
raw = np.asarray(reference_rows)
norms = np.linalg.norm(raw[:, 3:], axis=1)
assert np.isfinite(raw).all() and np.all(np.abs(norms-1.) <= 1e-5)
rotations = Rotation.from_quat(raw[:, 3:]/norms[:, None])
with np.load(STUDY/'estimates/repeat-motion/raw-legacy/trajectory.npz', allow_pickle=False) as arrays:
    cohort_times, cohort_ids = arrays['timestamps_ns'].copy(), arrays['original_scan_indices'].copy()
indices, paired = [], []
from bisect import bisect_left
for index, stamp in enumerate(cohort_times):
    right = bisect_left(stamps, int(stamp))
    if right < len(stamps) and stamps[right] == int(stamp):
        p = lie.se3(rotations[right].as_matrix(), raw[right, :3])
    elif right == 0 or right == len(stamps) or stamps[right]-stamps[right-1] > 100_000_000:
        continue
    else:
        left = right-1
        alpha = float((Decimal(int(stamp))-Decimal(stamps[left]))/Decimal(stamps[right]-stamps[left]))
        r = rotations[left] * Rotation.from_rotvec((rotations[left].inv()*rotations[right]).as_rotvec()*alpha)
        p = lie.se3(r.as_matrix(), raw[left, :3]+alpha*(raw[right, :3]-raw[left, :3]))
    indices.append(index)
    paired.append(p)
indices, reference = np.asarray(indices, dtype=np.int64), np.asarray(paired)
with np.load(STUDY/'results/paired-reference.npz', allow_pickle=False) as actual:
    np.testing.assert_array_equal(actual['cohort_indices'], indices)
    np.testing.assert_array_equal(actual['scan_indices'], cohort_ids[indices])
    np.testing.assert_array_equal(actual['timestamps_ns'], cohort_times[indices])
    np.testing.assert_allclose(actual['reference'], reference, rtol=0, atol=2e-8)
    pairing_error = float(np.max(np.abs(actual['reference']-reference)))
results = {p: {} for p in ['repeat-motion', 'last-pose']}
for policy in results:
    for arm in ['raw-legacy', 'gyro-legacy', 'raw-0.1875', 'gyro-0.1875']:
        folder = STUDY/'results'/policy/arm
        summary = read(folder/'summary.json')
        with np.load(STUDY/'estimates'/policy/arm/'trajectory.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['timestamps_ns'], cohort_times)
            np.testing.assert_array_equal(saved['original_scan_indices'], cohort_ids)
            estimate = saved['poses'][indices]
        values, maximum = {}, {}
        def ape(name, poses, relation):
            metric = metrics.APE(relation)
            metric.process_data((trajectory.PosePath3D(poses_se3=reference), trajectory.PosePath3D(poses_se3=poses)))
            values[name] = metric.error
        origin = reference[0]@lie.se3_inverse(estimate[0])
        ape('origin_position_ape_m', origin@estimate, metrics.PoseRelation.translation_part)
        ape('origin_rotation_ape_deg', origin@estimate, metrics.PoseRelation.rotation_angle_deg)
        if summary['position_fit']['status'] == 'identified':
            r, t, scale = geometry.umeyama_alignment(estimate[:, :3, 3].T, reference[:, :3, 3].T, with_scale=False)
            assert scale == 1.
            transform = lie.se3(r, t)
            np.testing.assert_allclose(transform, summary['position_fit']['world_alignment'], rtol=0, atol=2e-8)
            maximum['world_alignment_matrix'] = float(np.max(np.abs(transform-summary['position_fit']['world_alignment'])))
            ape('rigid_position_ape_m', transform@estimate, metrics.PoseRelation.translation_part)
        edges = np.flatnonzero(np.diff(cohort_ids[indices]) == 1)
        delta = [metrics.RPE.rpe_base(reference[i], reference[i+1], estimate[i], estimate[i+1]) for i in edges]
        values['consecutive_rpe_translation_m'] = np.array([np.linalg.norm(d[:3, 3]) for d in delta])
        values['consecutive_rpe_rotation_deg'] = np.array([abs(lie.so3_log_angle(d[:3, :3], degrees=True)) for d in delta])
        with np.load(folder/'errors.npz', allow_pickle=False) as actual:
            assert set(actual.files) == set(values)
            for name, expected in values.items():
                np.testing.assert_allclose(actual[name], expected, rtol=0, atol=1e-5 if name.endswith('deg') else 2e-8)
                maximum[name] = float(np.max(np.abs(actual[name]-expected), initial=0.))
        np.savez_compressed(out/(policy+'--'+arm+'.npz'), **values)
        results[policy][arm] = dict(paired_estimates=len(indices), consecutive_edges=len(edges), maximum_abs_differences=maximum)
verify()
receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), controls=results,
    selected_reference_sha256=hashlib.sha256(payload).hexdigest(), maximum_pairing_matrix_error=pairing_error,
    independent_decimal_bracketing_and_rotation_interpolation=True,
    controller_freeze_sha256=sha(STUDY/'controller-freeze.json'), scoring_boundary_sha256=sha(STUDY/'results/scoring-boundary.json'),
    physical_provenance_certified=False, input_hashes_unchanged=True, outputs_sha256={p.name: sha(p) for p in out.iterdir()})
with (out/'receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(receipt, indent=2))
