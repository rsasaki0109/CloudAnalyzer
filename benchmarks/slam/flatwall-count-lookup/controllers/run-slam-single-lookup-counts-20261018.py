"""Run immutable native builds against the predeclared real-data comparison."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'notes/slam-single-lookup-counts-20261018'
PREVIOUS=ROOT/'notes/slam-flatwall-balance-stages-20261018'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def inputs():
    initial=read(STUDY/'initial-preservation.json')
    files=initial['real_replay_inputs_sha256']|initial['fixed_state_inputs_sha256']
    for implementation in ['baseline','candidate']:
        build=STUDY/('build-'+implementation)
        receipt=read(build/'receipt.json')
        for name,item in receipt['artifacts'].items():
            files[(build/name).relative_to(ROOT).as_posix()]=item['sha256']
        for path in [build/'inputs.json',build/'receipt.json']:
            files[path.relative_to(ROOT).as_posix()]=sha(path)
        assert sha(build/'inputs.json')==receipt['inputs_sha256']
        for name,item in read(build/'inputs.json')['sources'].items():
            archive=build/'source'/name
            files[archive.relative_to(ROOT).as_posix()]=item['sha256']
            if implementation=='candidate':
                assert sha(ROOT/name)==item['sha256'],name
    for path in [STUDY/'initial-preservation.json',STUDY/'protocol.json',Path(__file__)]:
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    return files


def check(files):
    for name,digest in files.items():
        assert sha(ROOT/name)==digest,name


def run(mode):
    assert mode in ['fixed','trace','time']
    files=inputs()
    check(files)
    protocol=read(STUDY/'protocol.json')
    out=STUDY/(mode+'-01')
    out.mkdir()
    save(out/'invocation.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),mode=mode,
        inputs_sha256=files,protocol_sha256=sha(STUDY/'protocol.json'),installed_extension_replaced=False))
    if mode=='fixed':
        order=protocol['fixed_process_order']
    else:
        implementations=['baseline','candidate'] if mode=='trace' else protocol['replay_process_order']
        order=[dict(implementation=i,threads=4) for i in implementations]
    for index,item in enumerate(order):
        implementation,threads=item['implementation'],item['threads']
        build=STUDY/('build-'+implementation)
        env=dict(os.environ,RAYON_NUM_THREADS=str(threads))
        stem=f'run-{index:02}-{implementation}-threads-{threads}'
        if mode=='fixed':
            destination=out/(stem+'.json')
            env.update(CA_ODOMETRY_PROFILE_INPUT=str(PREVIOUS/'prepared-02/input.json'),
                CA_ODOMETRY_PROFILE_OUTPUT=str(destination),CA_ODOMETRY_PROFILE_THREADS=str(threads))
            command=[str(build/'odometry-stage-profile.exe'),'odometry::profile::record_fixed_states','--exact','--ignored','--nocapture']
        else:
            destination=out/stem
            command=[str(build/'audit_odometry_steps.exe'),mode,
                str(ROOT/'notes/slam-flatwall-target-balance-dynamics-20261018/prepared-01/input.json'),str(destination)]
        log=out/(stem+'.log')
        save(out/(stem+'-invocation.json'),dict(index=index,implementation=implementation,threads=threads,
            command=command,binary_sha256=sha(Path(command[0])),environment={k:env[k] for k in ['RAYON_NUM_THREADS',*(['CA_ODOMETRY_PROFILE_INPUT','CA_ODOMETRY_PROFILE_OUTPUT','CA_ODOMETRY_PROFILE_THREADS'] if mode=='fixed' else [])]}))
        with log.open('xb') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            save(out/(stem+'-failure.json'),dict(returncode=result.returncode,log_sha256=sha(log)))
            raise RuntimeError(log.read_text(encoding='utf-8',errors='replace')[-4000:])
        if mode=='fixed':
            assert read(destination)['threads']==threads
        print(json.dumps(dict(mode=mode,index=index,implementation=implementation,threads=threads,complete=True)),flush=True)
    check(files)
    save(out/'receipt.json',dict(closed_at_utc=datetime.now(timezone.utc).isoformat(),
        invocation_sha256=sha(out/'invocation.json'),input_hashes_unchanged=True,
        output_files_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))


run(sys.argv[1])
