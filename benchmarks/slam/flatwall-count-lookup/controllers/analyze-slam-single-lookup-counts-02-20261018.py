"""Independently check every retained native comparison and recompute summaries."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'notes/slam-single-lookup-counts-20261018'
spec=importlib.util.spec_from_file_location('count_comparison',ROOT/'scripts/flatwall_count_comparison.py')
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def closed(mode):
    folder=STUDY/(mode+'-01')
    receipt=comparison.read(folder/'receipt.json')
    assert sha(folder/'invocation.json')==receipt['invocation_sha256']
    for name,digest in receipt['output_files_sha256'].items():
        assert sha(folder/name)==digest,name
    invocation=comparison.read(folder/'invocation.json')
    for name,digest in invocation['inputs_sha256'].items():
        assert sha(ROOT/name)==digest,name
    return folder


def records(mode,folder):
    result=[]
    protocol=comparison.read(STUDY/'protocol.json')
    order=protocol['fixed_process_order'] if mode=='fixed' else [dict(implementation=i,threads=4)
        for i in (['baseline','candidate'] if mode=='trace' else protocol['replay_process_order'])]
    for index,item in enumerate(order):
        stem=f"run-{index:02}-{item['implementation']}-threads-{item['threads']}"
        path=folder/(stem+'.json') if mode=='fixed' else folder/stem/'runs.json'
        result.append(dict(index=index,**item,file=path.relative_to(ROOT).as_posix(),data=comparison.read(path)))
    return result


mode=sys.argv[1]
assert mode in ['validate','summarize']
out=STUDY/('validation-02' if mode=='validate' else 'analysis-01')
out.mkdir()
save(out/'invocation.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
    controller_sha256=sha(Path(__file__)),analysis_source_sha256=sha(ROOT/'scripts/flatwall_count_comparison.py'),
    protocol_sha256=sha(STUDY/'protocol.json')))
fixed=records('fixed',closed('fixed'))
trace_folder=closed('trace')
traces=records('trace',trace_folder)
originals=comparison.original_cases(ROOT)
trace_reports=[comparison.check_replay(r['data'],originals,'trace') for r in traces]
comparison.compare_replays(traces[1]['data'],traces[0]['data'])
catalog=comparison.read(ROOT/'benchmarks/slam/flatwall-balance-dynamics/summary.json')['traces']
trace_data=[]
for index,implementation in enumerate(['baseline','candidate']):
    folder=trace_folder/f'run-{index:02}-{implementation}-threads-4'
    trace_data.append({(r['case'],r['scan_index']):comparison.read(folder/r['case']/f"scan-{r['scan_index']:03}.json") for r in catalog})
validation=dict(fixed=comparison.check_fixed(fixed,ROOT),
    trace_replay=trace_reports,trace_systems=comparison.check_traces(*trace_data,ROOT),
    defaults_changed=False,installed_extension_replaced=False)
save(out/'verification.json',validation)
if mode=='summarize':
    timed=records('time',closed('time'))
    time_reports=[comparison.check_replay(r['data'],originals,'time') for r in timed]
    for r in timed[1:]:
        comparison.compare_replays(r['data'],timed[0]['data'])
    summary=dict(fixed=comparison.summarize_fixed(fixed,comparison.read(ROOT/'benchmarks/slam/flatwall-balance-stages/origins.json')),
        registration=comparison.summarize_time(timed,[r['name'] for r in originals]),time_verification=time_reports)
    save(out/'summary.json',summary)
    print(json.dumps(dict(registration=summary['registration']['groups'],fixed=summary['fixed']['groups'],validation=validation)))
else:
    print(json.dumps(validation))
save(out/'receipt.json',dict(outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
    inputs_changed=False,original_outputs_rewritten=False))
