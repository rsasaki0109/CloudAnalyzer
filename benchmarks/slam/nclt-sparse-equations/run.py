import argparse, json, subprocess, os, hashlib, time
from pathlib import Path
parser=argparse.ArgumentParser(description='Replay both native binaries on all bundled NCLT scans, with alternating order.')
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--candidate',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[3]
work=args.out.resolve()
work.mkdir(parents=True,exist_ok=False)
binaries={name:getattr(args,name).resolve() for name in ['baseline','candidate']}
identities={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in binaries.items()}
identities.update({date:hashlib.sha256((root/f'web/public/samples/nclt-{date}.mcap').read_bytes()).hexdigest() for date in ['2012-04-29','2012-06-15']})
(work/'identities.json').write_text(json.dumps(identities,indent=2))
rows=[]
for date in ['2012-04-29','2012-06-15']:
    for repeat in range(3):
        for arm in (['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']):
            path=work/f'{date}-{repeat}-{arm}.json'
            started=time.perf_counter()
            subprocess.run([str(binaries[arm]),str(root/f'web/public/samples/nclt-{date}.mcap'),str(path)],env={**os.environ,'RAYON_NUM_THREADS':'1'},check=True)
            data=json.loads(path.read_text())['frames']
            row=dict(session=date,repeat=repeat,arm=arm,file=path.name,registration_seconds=sum(x['seconds'] for x in data),wall_seconds=time.perf_counter()-started,iterations=sum(x['iterations'] for x in data),frames=len(data))
            rows.append(row)
            print(json.dumps(row),flush=True)
(work/'summary.json').write_text(json.dumps(rows,indent=2))
for date in ['2012-04-29','2012-06-15']:
    baseline=json.loads((work/f'{date}-0-baseline.json').read_text())['frames']
    for repeat in range(3):
        for arm in ['baseline','candidate']:
            frames=json.loads((work/f'{date}-{repeat}-{arm}.json').read_text())['frames']
            assert len(frames)==len(baseline)
            for a,b in zip(baseline,frames):
                assert {k:v for k,v in a.items() if k!='seconds'} == {k:v for k,v in b.items() if k!='seconds'}, (date,repeat,arm)
print('All poses, iteration counts, terminations, timestamps and registration point counts are exactly equal',flush=True)
