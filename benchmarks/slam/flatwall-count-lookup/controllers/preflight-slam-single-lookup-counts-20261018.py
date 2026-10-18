"""Check the new comparison reader against already-verified original native outputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'notes/slam-single-lookup-counts-20261018'
OLD=ROOT/'notes/slam-flatwall-target-balance-dynamics-20261018'
spec=importlib.util.spec_from_file_location('comparison_preflight',ROOT/'scripts/flatwall_count_comparison.py')
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
out=STUDY/'preflight-01'
out.mkdir()
files={}


def read(path):
    files[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return comparison.read(path)


for path in [Path(__file__),ROOT/'scripts/flatwall_count_comparison.py']:
    files[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
originals=comparison.original_cases(ROOT)
reports={mode:comparison.check_replay(read(OLD/(mode+'-01')/'runs.json'),originals,mode) for mode in ['time','trace']}
catalog=read(ROOT/'benchmarks/slam/flatwall-balance-dynamics/summary.json')['traces']
traces={(r['case'],r['scan_index']):read(OLD/'trace-01'/r['case']/f"scan-{r['scan_index']:03}.json") for r in catalog}
reports['traces']=comparison.check_traces(traces,traces,ROOT)
with (out/'receipt.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(dict(reports=reports,files_sha256=files,production_measurements_started=False),stream,indent=2)
    stream.write('\n')
print(json.dumps(reports))
