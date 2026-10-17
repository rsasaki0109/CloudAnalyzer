from datetime import datetime, timezone
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'notes/slam-flatwall-frontend-controls-20261017/revision-02'
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_flatwall_frontend_controls as control


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def artifact(name):
    path = ROOT / name
    return dict(path=name, bytes=path.stat().st_size, sha256=sha(path))


def main():
    control.verify(read(STUDY.parent / 'prior-artifacts.json'))
    native_root = 'notes/slam-gicp-transported-20261017'
    native = read(ROOT / native_root / 'native-artifact.json')
    build_receipt = read(ROOT / native_root / 'native-build-inputs.json')
    build = dict(build_receipt['sources_sha256'])
    build.update(build_receipt['additional_config_sha256'])
    comparison = {}
    for name, expected in build.items():
        archived = ROOT / native_root / 'source-snapshot' / name
        current = ROOT / name
        assert sha(archived) == expected, name
        same = current.read_bytes() == archived.read_bytes()
        assert current.read_bytes().replace(b'\r\n', b'\n') == archived.read_bytes().replace(b'\r\n', b'\n'), name
        comparison[name] = dict(archived_sha256=expected, current_sha256=sha(current),
                                byte_identical=same, identical_after_crlf_to_lf=True)
    save(STUDY / 'native-source-comparison.json', dict(inputs=comparison,
         original_source_files_verified=len(build), byte_identical=sum(v['byte_identical'] for v in comparison.values()),
         correction='First freeze attempt stopped before creating the protocol because a merged checkout had CRLF conversion. Preserve that failed attempt; verify archived original build bytes plus current text identity, and freeze current exact bytes separately.'))
    package = native_root + '/native/cloudanalyzer_core'
    files = {name: sha(ROOT / name) for name in build}
    files.update({native_root+'/source-snapshot/'+name: digest for name, digest in build.items()})
    for name, digest in native['package_sha256'].items():
        path = f'{package}/{name}'
        assert sha(ROOT / path) == digest
        files[path] = digest
    prefixes = {
        'raw-legacy': 'notes/slam-registration-sampling-20261016/after/omitted',
        'gyro-legacy': 'notes/slam-imu-rotation-20261017/rotation/legacy',
        'raw-0.1875': 'notes/slam-registration-sampling-20261016/after/explicit-0.1875',
        'gyro-0.1875': 'notes/slam-imu-rotation-20261017/rotation/explicit-0.1875',
    }
    protocol = dict(schema='cloudanalyzer.flatwall_frontend_controls.v1', frozen_at_utc=control.now(),
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        question='Compare raw and gyro-only point compensation at both existing spatial sampling policies on the complete original cohort, alongside the previously frozen nominal IMU integration baseline.',
        reference_boundary='Exploratory: selected reference fully inspected and scored before designing this control experiment. No blind or held-out claim.',
        recording='notes/slam-flatwall-inputs-20261016/flatwall_01.bag',
        reference_archive='notes/slam-flatwall-inputs-20261016/gt.tar.gz',
        scoring_protocol='notes/slam-flatwall-trajectory-20261017/protocol.json',
        scoring_protocol_sha256='ac414581f94797cde51c80d246f91d66bd76591c05405d678a4fbab32b7a9452',
        nominal_trajectory=native_root+'/integration/trajectory.npz',
        nominal_evaluation_summary='notes/slam-flatwall-trajectory-20261017/strict-time-regression/results/summary.json',
        native_package=package, native_binary_sha256=native['sha256'], archived_native_build_inputs_verified=len(build),
        gyro_component_receipt='notes/slam-imu-rotation-20261017/rotation/receipt.json',
        prefix_directories=prefixes, arms=control.ARMS, lidar=control.LIDAR,
        original_scan_indices=list(range(1, 186)), excluded_original_scan_indices=[0],
        cohort='All original scans 1..185, each 24000 points; retain first-scan exclusion for incomplete IMU coverage. No reference-driven scan selection.',
        reference_time='Maximum integer point timestamp; original UINT32 t in nanoseconds from exact header. Point order unchanged.',
        model='Gyro-only uses held body gyro, zero declared bias, identity declared IMU-from-LiDAR and fixed IMU origin within each scan; no acceleration/translation compensation or external pose prediction. Both arms use the unchanged internal constant-motion ICP prediction, deskew=False and independent evolving maps.',
        raw_time_limitation='Raw points span an acquisition interval; the estimated pose is assigned to scan end for the common comparison. This is an approximation, not a claim that the uncompensated cloud is instantaneous.',
        gyro_bias_rad_s=[0., 0., 0.], max_gap_s=.05, imu_from_lidar=control.np.eye(4).tolist(),
        threads=dict(RAYON_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'),
        execution='One Python process, four sequential independent odometry arms per scan; native parallel gyro compensation uses four Rayon threads. No runtime or GPU comparison.',
        sampling='Legacy omitted source sampling and explicit 0.1875 m source voxel were selected in pre-reference prefix studies; retain both for both controls. No choice of winner or defaults from these results.',
        initialization='No rejected initializer gravity/velocity/bias is used; no initializer thresholds or state acceptance are changed. The frozen nominal baseline is only a diagnostic comparator.',
        failures='Any unexpected cohort, prefix-bit/report, component-digest, finite-pose or registration failure aborts estimation and preserves partial rows/poses and a failure receipt. Never score an incomplete run.',
        freeze_boundary='Freeze this protocol, controller, scoring source, analytic tests, native package/build inputs and selected dependency identities before reading measurements for new estimates. Write all four complete estimates and diagnostics plus a hash closure before invoking the scoring stage.',
        reference_policy='Reuse the entire prior scoring protocol without changes: declared TUM layout, reference-world-from-LiDAR, fixed zero clock offset, identity body transform, unit scale, exact integer timestamps, 100 ms maximum interpolation span, no extrapolation, all covered poses and consecutive original edges, proper rigid fit with existing rank guard.',
        interpretation='Archive-specific body axes/pose direction and numeric clock/extrinsic mapping remain unverified. Arithmetic is conditional. Control differences may reveal model sensitivity but cannot alone identify a physical cause, prove calibration/SLAM accuracy, accept an initial state, or compare with GLIM.',
        physical_provenance_certified=False,
        packages={name: version(name) for name in ['numpy', 'scipy', 'rosbags', 'evo']})
    artifacts = {}
    for key in ['recording','reference_archive','scoring_protocol','nominal_trajectory','nominal_evaluation_summary','gyro_component_receipt']:
        name = protocol[key]
        artifacts[key] = artifact(name)
        files[name] = sha(ROOT / name)
    for directory in prefixes.values():
        for name in ['startup-poses.npz','lidar-diagnostics.json']:
            files[directory+'/'+name] = sha(ROOT / directory / name)
    sources = ['scripts/audit_flatwall_frontend_controls.py', 'scripts/flatwall_trajectory_reference.py',
               'cloudanalyzer/tests/test_flatwall_frontend_controls.py', 'cloudanalyzer/tests/test_flatwall_trajectory_reference.py',
               Path(__file__).relative_to(ROOT).as_posix()]
    snapshot = STUDY / 'source-snapshot'
    for name in sources:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        files[name] = sha(ROOT / name)
    # Preserve the selected dependency inventory already used for the evo cross-check.
    dependencies = read(ROOT / 'notes/slam-flatwall-trajectory-20261017/controller-freeze.json')['dependency_files_sha256']
    control.verify(dependencies)
    for module in ['numpy', 'numpy._core._multiarray_umath', 'scipy', 'scipy.spatial.transform._rotation', 'scipy.spatial.transform._rotation_cy']:
        path = Path(import_module(module).__file__)
        dependencies[path.as_posix()] = sha(path)
    dependencies[Path(sys.executable).as_posix()] = sha(Path(sys.executable))
    dependency_snapshot = STUDY / 'dependency-snapshot'
    dependency_snapshot.mkdir()
    for index, (name, digest) in enumerate(sorted(dependencies.items())):
        target = dependency_snapshot / f'{index:03d}-{Path(name).name}'
        shutil.copyfile(name, target)
        assert sha(target) == digest
    files.update(dependencies)
    control.validate_protocol(protocol)
    protocol['input_artifacts'] = artifacts
    save(STUDY / 'protocol.json', protocol)
    control.verify(files)
    save(STUDY / 'controller-freeze.json', dict(at_utc=control.now(), protocol_sha256=sha(STUDY/'protocol.json'),
         files_sha256=files, source_files_sha256={n: files[n] for n in sources},
         analytic_controls_passed=41, unit_log_sha256=sha(STUDY/'unit-01.log'),
         native_package_files=len(native['package_sha256']), native_build_inputs=len(build),
         reference_fully_inspected_before_this_experiment=True))
    print(json.dumps(dict(protocol_sha256=sha(STUDY/'protocol.json'), frozen_files=len(files),
                         native_build_inputs=len(build), dependencies=len(dependencies))))


if __name__ == '__main__':
    main()
