"""Preserve the previous study and freeze matched native builds for count optimization."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'notes/slam-single-lookup-counts-20261018'
PREVIOUS = ROOT/'notes/slam-flatwall-balance-stages-20261018'
BASE = '3de34d50808ac09a6c86a7fbf9cb5118947741aa'
CHANGED = [
    'rust/crates/ca-core/src/odometry_target_balance.rs',
    'rust/crates/ca-core/src/odometry_profile.rs',
    'cloudanalyzer/tests/test_flatwall_target_balance_packet.py',
    'benchmarks/slam/flatwall-target-balance/manifest.json',
    'benchmarks/slam/flatwall-balance-dynamics/manifest.json',
    'benchmarks/slam/flatwall-balance-dynamics/source-footprints.json',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def copy(path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(path.read_bytes())


def initialize():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
    STUDY.mkdir()
    previous = read(PREVIOUS/'completion.json')
    assert previous['merge_commit']==BASE and previous['worktree_clean']
    ledgers = ['merge-integrity-03-inputs.json','merge-final-integrity-03-inputs.json']
    preserved = {}
    for name in ledgers:
        ledger = read(PREVIOUS/name)
        assert isinstance(ledger,dict)
        entries = ledger.get('files_sha256',ledger)
        def check(item):
            path,digest = item
            assert sha(ROOT/path)==digest,path
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(check,entries.items()))
        preserved[name] = dict(sha256=sha(PREVIOUS/name),files=len(entries))
    originals = {}
    for name in CHANGED:
        copy(ROOT/name,STUDY/'original-public'/name)
        originals[name]=sha(ROOT/name)
    prepared = ROOT/'notes/slam-flatwall-target-balance-dynamics-20261018/prepared-01'
    inputs = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(prepared.iterdir()) if p.is_file()}
    fixed = PREVIOUS/'prepared-02'
    fixed_inputs = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(fixed.rglob('*')) if p.is_file()}
    save(STUDY/'initial-preservation.json',dict(base=BASE,previous_completion_sha256=sha(PREVIOUS/'completion.json'),
        previous_integrity_ledgers=preserved,original_public_sha256=originals,
        real_replay_inputs_sha256=inputs,fixed_state_inputs_sha256=fixed_inputs,
        previous_outputs_rewritten=False,installed_extension_replaced=False))
    save(STUDY/'protocol.json',dict(schema='cloudanalyzer.single_lookup_counts.v1',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),base=BASE,
        candidate='Assign exact coordinate IDs with one hash lookup per accepted pair, count IDs, replace IDs in the output vector with completed frequencies; count directly from pair records.',
        arithmetic='Preserve signed-zero normalization, exact coordinate bits, acceptance, pair ordering, robust weights, 256-pair chunks, H/b arithmetic, solve and registration policies.',
        selection='Reuse all 16 previous closed-loop cases and all 64 previous fixed states; retain 4 alignment-ineligible empty controls separately from 60 eligible states.',
        replay_process_order=['baseline','candidate','candidate','baseline'],
        replay_threads=4,replay_rounds=dict(warmup=1,measured=3),
        replay_control='Run target_balance=false as an unchanged-path timing/trajectory control in both binaries.',
        replay_scope='Native register calls on preloaded real scans, including sampling, ICP, solve, learned thresholds and map insertion/pruning; excludes input preparation, bag I/O, gyro deskew, checks, output and application/GPU work.',
        replay_summary='For each case/mode take the median of 6 complete measured recording totals per implementation, then report candidate/baseline and sum-of-case-medians ratio. Keep per-frame and per-round measurements.',
        fixed_process_order=[dict(implementation=i,threads=t) for t in [1,4] for i in ['baseline','candidate','candidate','baseline']],
        fixed_rounds=dict(warmup=1,measured=5,evaluations_per_block=32),
        validation='Compare all actual replay poses, iterations, terminations and learned thresholds bit-for-bit between builds; compare all 64 fixed native H/b/count/pair rows with the independent published-data oracle.',
        trace_validation='Replay a trace pass per implementation and compare all 4000 capped step systems/poses plus final associations; checks are outside timing runs.',
        limitations='Single host, uncontrolled CPU placement/frequency. Fixed-state/profile clocks have code-layout effects. No GPU or full application throughput claim. No accuracy change is expected or inferred.',
        defaults_changed=False,installed_extension_replaced=False,raw_bags_redistributed=False))
    print(json.dumps(dict(initialized=True,preserved=preserved,inputs=len(inputs),fixed_inputs=len(fixed_inputs))))


def build(implementation):
    assert implementation in ['baseline','candidate']
    out = STUDY/('build-'+implementation)
    out.mkdir()
    paths = sorted((ROOT/'rust/crates/ca-core/src').rglob('*.rs'))
    paths += [ROOT/name for name in ['rust/Cargo.toml','rust/Cargo.lock','rust/crates/ca-core/Cargo.toml','rust/crates/ca-core/examples/audit_odometry_steps.rs']]
    sources = {}
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        copy(path,out/'source'/name)
        sources[name]=dict(sha256=sha(path),normalized_sha256=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    if implementation=='baseline':
        original=read(STUDY/'initial-preservation.json')['original_public_sha256']
        for name in CHANGED[:2]:
            assert sha(ROOT/name)==original[name]
    env=dict(os.environ,PATH=str(Path.home()/'.cargo/bin')+os.pathsep+os.environ['PATH'])
    cargo=str(Path.home()/'.cargo/bin/cargo.exe')
    common=['--release','--offline','--manifest-path','rust/Cargo.toml','-p','ca-core','--features','parallel','--message-format=json']
    commands=[['build',*common,'--example','audit_odometry_steps'],['test',*common,'--lib','--no-run']]
    versions={name:subprocess.check_output([str(Path.home()/'.cargo/bin'/(name+'.exe')),'--version'],cwd=ROOT,env=env,text=True).strip() for name in ['rustc','cargo']}
    save(out/'inputs.json',dict(implementation=implementation,sources=sources,commands=commands,versions=versions,protocol_sha256=sha(STUDY/'protocol.json'),controller_sha256=sha(Path(__file__))))
    artifacts={}
    for index,command in enumerate(commands):
        messages=out/f'messages-{index:02}.jsonl'
        log=out/f'build-{index:02}.log'
        with messages.open('xb') as stdout,log.open('xb') as stderr:
            result=subprocess.run([cargo,*command],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        if result.returncode:
            save(out/f'failure-{index:02}.json',dict(returncode=result.returncode,log_sha256=sha(log)))
            raise RuntimeError(log.read_text(encoding='utf-8',errors='replace')[-5000:])
        outputs=[json.loads(line) for line in messages.read_text(encoding='utf-8').splitlines()]
        selected=[Path(r['executable']) for r in outputs if r.get('reason')=='compiler-artifact' and r.get('executable') and ((index==0 and r['target']['name']=='audit_odometry_steps') or (index==1 and r['target']['name']=='ca_core' and r['profile']['test']))]
        assert len(selected)==1,selected
        name='audit_odometry_steps.exe' if index==0 else 'odometry-stage-profile.exe'
        copy(selected[0],out/name)
        artifacts[name]=dict(sha256=sha(out/name),bytes=(out/name).stat().st_size)
    for name,digest in sources.items():
        assert sha(ROOT/name)==digest['sha256'],name
    save(out/'receipt.json',dict(implementation=implementation,artifacts=artifacts,inputs_sha256=sha(out/'inputs.json'),sources_checked=len(sources),installed_extension_replaced=False))
    print(json.dumps(dict(implementation=implementation,sources=len(sources),artifacts=artifacts)))


if sys.argv[1]=='initialize':
    initialize()
else:
    build(sys.argv[1])
