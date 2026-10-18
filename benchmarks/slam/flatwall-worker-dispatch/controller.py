"""Freeze matched native sources, run real recordings, and close the comparison."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'notes/slam-repeat-workers-20261018'
PREVIOUS = ROOT/'notes/slam-single-lookup-counts-20261018'
BASE = '1a31df6d63063f210d98293b9a8cde09f3c25a95'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(source.read_bytes())


def sources():
    paths = sorted((ROOT/'rust/crates/ca-core/src').rglob('*.rs'))
    paths += [ROOT/name for name in ['rust/Cargo.toml', 'rust/Cargo.lock',
        'rust/crates/ca-core/Cargo.toml', 'rust/crates/ca-core/examples/audit_odometry_steps.rs']]
    assert len(paths) == 84
    return {p.relative_to(ROOT).as_posix():dict(sha256=sha(p),
        normalized_sha256=hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()) for p in paths}


def check(mapping):
    for name, digest in mapping.items():
        assert sha(ROOT/name) == digest, name


def build_inputs(implementation):
    out = STUDY/('build-'+implementation)
    out.mkdir()
    if implementation == 'baseline':
        original = PREVIOUS/'build-candidate'
        snapshot = read(original/'inputs.json')['sources']
        for name, item in snapshot.items():
            assert sha(original/'source'/name) == item['sha256'], name
            raw = subprocess.check_output(['git', 'show', BASE+':'+name], cwd=ROOT)
            assert hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == item['normalized_sha256'], name
            copy(original/'source'/name, out/'source'/name)
    else:
        snapshot = sources()
        for name in snapshot:
            copy(ROOT/name, out/'source'/name)
    env = dict(os.environ, PATH=str(Path.home()/'.cargo/bin')+os.pathsep+os.environ['PATH'])
    common = ['--release', '--offline', '--manifest-path', 'rust/Cargo.toml', '-p', 'ca-core',
        '--features', 'parallel', '--message-format=json']
    commands = [['build', *common, '--example', 'audit_odometry_steps'],
        ['test', *common, '--lib', '--no-run']]
    versions = {name:subprocess.check_output([str(Path.home()/'.cargo/bin'/(name+'.exe')), '--version'],
        cwd=ROOT, env=env, text=True).strip() for name in ['rustc', 'cargo']}
    save(out/'inputs.json', dict(implementation=implementation, sources=snapshot, commands=commands,
        versions=versions, protocol_sha256=sha(STUDY/'protocol.json'), controller_sha256=sha(Path(__file__))))
    return out, env, commands


def setup():
    probe_path = ROOT/'notes/slam-outlined-workers-20261018/analysis-01/summary.json'
    probe = read(probe_path)
    assert probe['original_pose_and_threshold_bits_match']
    save(STUDY/'protocol.json', dict(schema='cloudanalyzer.gated_workers.v1', base=BASE,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        candidate='Keep pool entry out of line to preserve the compact caller-side path. Keep the first iteration on the caller. Only after a nonterminating step, move the remaining automatic ICP iterations into the current Rayon pool once when alignment is active, the captured policy is target-balanced or the source has more than the existing 256-point chunk, and the caller is not already a Rayon worker.',
        gate='needs_alignment && (registration.target_balance || registration.source.len() > 256) && iterations > 0 && termination.is_none() && current_thread_index().is_none()',
        arithmetic='Keep begin/finish on the caller; preserve correspondence search/order, exact counts, all weights, 256-point chunks, reductions, solver, stopping, map updates, diagnostics and defaults.',
        selection='All 16 previous real recording cases, both default and balanced modes, all frames; raw and gyro, both motion policies, both sampling modes.',
        replay_process_order=['baseline', 'candidate', 'candidate', 'baseline'], replay_threads=4,
        replay_rounds=dict(warmup=1, measured=3),
        replay_scope='Complete native automatic register windows on preloaded real scans, including begin, gate, worker dispatch, all ICP iterations, solve and finish/map insertion. Input loading, output and checks are outside clocks.',
        replay_summary='Per case/mode/implementation median of six complete measured recording totals, then sum of case medians and candidate/baseline ratio. Preserve every recording and per-frame timing; include all subgroups and regressions.',
        validation='Check every output pose and learned threshold against original reference bits and both binaries; iterations and termination must match. No trajectory tolerance can substitute for strict independent original-bit checks.',
        baseline_reuse='Reuse the preserved PR315 candidate executables only after confirming every one of the 84 normalized source files, toolchain versions and original build commands match the new baseline. Retain the original receipts and executable bytes.',
        limitations='One host, four Rayon workers, uncontrolled CPU placement/frequency. Native registration timing is not whole application throughput or GPU performance. No accuracy improvement is claimed.',
        private_probe_analysis_sha256=sha(probe_path),
        defaults_changed=False, installed_extension_replaced=False, raw_bags_redistributed=False))
    out, env, commands = build_inputs('baseline')
    old = PREVIOUS/'build-candidate'
    current, previous = read(out/'inputs.json'), read(old/'inputs.json')
    assert current['commands'] == previous['commands'] and current['versions'] == previous['versions']
    assert set(current['sources']) == set(previous['sources'])
    for name, item in current['sources'].items():
        assert item['normalized_sha256'] == previous['sources'][name]['normalized_sha256'], name
        assert sha(old/'source'/name) == previous['sources'][name]['sha256'], name
    old_receipt = read(old/'receipt.json')
    assert old_receipt['inputs_sha256'] == sha(old/'inputs.json')
    for name, item in old_receipt['artifacts'].items():
        assert sha(old/name) == item['sha256']
        copy(old/name, out/name)
    for name in ['inputs.json', 'receipt.json', 'messages-00.jsonl', 'messages-01.jsonl', 'build-00.log', 'build-01.log']:
        copy(old/name, out/'original-build'/name)
    save(out/'receipt.json', dict(implementation='baseline', artifacts=old_receipt['artifacts'],
        inputs_sha256=sha(out/'inputs.json'), sources_checked=84, reused_preserved_executables=True,
        original_inputs_sha256=sha(old/'inputs.json'), original_receipt_sha256=sha(old/'receipt.json'),
        installed_extension_replaced=False))
    print(json.dumps(dict(baseline_frozen=True, native_sources=84)), flush=True)


def candidate():
    out, env, commands = build_inputs('candidate')
    baseline = read(STUDY/'build-baseline/inputs.json')['sources']
    current = read(out/'inputs.json')['sources']
    changed = sorted(name for name, item in current.items()
        if baseline[name]['normalized_sha256'] != item['normalized_sha256'])
    assert changed == ['rust/crates/ca-core/src/odometry.rs'], changed
    artifacts = {}
    for index, command in enumerate(commands):
        log, messages = out/f'build-{index:02}.log', out/f'messages-{index:02}.jsonl'
        with messages.open('xb') as stdout, log.open('xb') as stderr:
            result = subprocess.run([str(Path.home()/'.cargo/bin/cargo.exe'), *command],
                cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        if result.returncode:
            save(out/f'failure-{index:02}.json', dict(returncode=result.returncode, log_sha256=sha(log)))
            raise RuntimeError(log.read_text(encoding='utf-8', errors='replace')[-5000:])
        outputs = [json.loads(line) for line in messages.read_text(encoding='utf-8').splitlines()]
        selected = [Path(r['executable']) for r in outputs if r.get('reason')=='compiler-artifact'
            and r.get('executable') and ((index==0 and r['target']['name']=='audit_odometry_steps')
            or (index==1 and r['target']['name']=='ca_core' and r['profile']['test']))]
        assert len(selected) == 1, selected
        name = 'audit_odometry_steps.exe' if index==0 else 'odometry-stage-profile.exe'
        copy(selected[0], out/name)
        artifacts[name] = dict(sha256=sha(out/name), bytes=(out/name).stat().st_size)
        print(json.dumps(dict(build_step=index, implementation='candidate', artifact=artifacts[name])), flush=True)
    for name, item in current.items():
        assert sha(ROOT/name) == item['sha256'], name
    save(out/'receipt.json', dict(implementation='candidate', artifacts=artifacts,
        inputs_sha256=sha(out/'inputs.json'), sources_checked=84, installed_extension_replaced=False))


def inputs():
    mapping = dict(read(STUDY/'initial-preservation.json')['real_replay_inputs_sha256'])
    for implementation in ['baseline', 'candidate']:
        out = STUDY/('build-'+implementation)
        artifact = read(out/'receipt.json')
        assert artifact['inputs_sha256'] == sha(out/'inputs.json')
        for name, item in artifact['artifacts'].items():
            mapping[(out/name).relative_to(ROOT).as_posix()] = item['sha256']
        for name, item in read(out/'inputs.json')['sources'].items():
            mapping[(out/'source'/name).relative_to(ROOT).as_posix()] = item['sha256']
            if implementation=='candidate':
                assert sha(ROOT/name) == item['sha256'], name
        for name in ['inputs.json', 'receipt.json']:
            mapping[(out/name).relative_to(ROOT).as_posix()] = sha(out/name)
    for path in [STUDY/'protocol.json', STUDY/'initial-preservation.json', Path(__file__)]:
        mapping[path.relative_to(ROOT).as_posix()] = sha(path)
    return mapping


def time():
    mapping = inputs()
    check(mapping)
    out = STUDY/'time-01'
    out.mkdir()
    save(out/'invocation.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=mapping, protocol_sha256=sha(STUDY/'protocol.json'), controller_sha256=sha(Path(__file__))))
    for index, implementation in enumerate(read(STUDY/'protocol.json')['replay_process_order']):
        binary = STUDY/('build-'+implementation)/'audit_odometry_steps.exe'
        stem = f'run-{index:02}-{implementation}-threads-4'
        command = [str(binary), 'time',
            str(ROOT/'notes/slam-flatwall-target-balance-dynamics-20261018/prepared-01/input.json'), str(out/stem)]
        save(out/(stem+'-invocation.json'), dict(index=index, implementation=implementation,
            threads=4, command=command, binary_sha256=sha(binary), environment=dict(RAYON_NUM_THREADS='4')))
        log = out/(stem+'.log')
        with log.open('xb') as stream:
            result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, RAYON_NUM_THREADS='4'),
                stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            save(out/(stem+'-failure.json'), dict(returncode=result.returncode, log_sha256=sha(log)))
            raise RuntimeError(log.read_text(encoding='utf-8', errors='replace')[-4000:])
        print(json.dumps(dict(index=index, implementation=implementation, complete=True)), flush=True)
    check(mapping)
    save(out/'receipt.json', dict(closed_at_utc=datetime.now(timezone.utc).isoformat(),
        invocation_sha256=sha(out/'invocation.json'), input_hashes_unchanged=True,
        output_files_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))


def analyze():
    out = STUDY/'analysis-01'
    out.mkdir()
    spec = importlib.util.spec_from_file_location('native_worker_comparison', ROOT/'scripts/flatwall_count_comparison.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    time_root = STUDY/'time-01'
    receipt = read(time_root/'receipt.json')
    for name, digest in receipt['output_files_sha256'].items():
        assert sha(time_root/name) == digest, name
    originals = comparison.original_cases(ROOT)
    records, reports = [], []
    for index, implementation in enumerate(read(STUDY/'protocol.json')['replay_process_order']):
        run = read(time_root/f'run-{index:02}-{implementation}-threads-4/runs.json')
        reports.append(comparison.check_replay(run, originals, 'time'))
        records.append(dict(index=index, implementation=implementation, data=run))
    for record in records[1:]:
        comparison.compare_replays(record['data'], records[0]['data'])
    summary = dict(schema='cloudanalyzer.gated_workers_summary.v1',
        registration=comparison.summarize_time(records, [r['name'] for r in originals]),
        time_verification=reports, original_pose_and_threshold_bits_match=True,
        replay_frames_checked=sum(r['frames'] for r in reports), defaults_changed=False,
        installed_extension_replaced=False, whole_application_speedup_established=False)
    save(out/'summary.json', summary)
    save(out/'receipt.json', dict(controller_sha256=sha(Path(__file__)),
        comparison_sha256=sha(ROOT/'scripts/flatwall_count_comparison.py'),
        time_receipt_sha256=sha(time_root/'receipt.json'), summary_sha256=sha(out/'summary.json')))
    print(json.dumps(dict(frames=summary['replay_frames_checked'], groups=summary['registration']['groups'])), flush=True)


if __name__ == '__main__':
    {'setup':setup, 'candidate':candidate, 'time':time, 'analyze':analyze}[sys.argv[1]]()
