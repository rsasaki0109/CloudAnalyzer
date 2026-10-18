"""Keep earlier benchmark sources bound to their unchanged executed source archive."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'notes/slam-single-lookup-counts-20261018'
PACKET=ROOT/'benchmarks/slam/flatwall-count-lookup'
SOURCE='rust/crates/ca-core/src/odometry_target_balance.rs'
ARCHIVE='benchmarks/slam/flatwall-balance-stages/executed-source/'+SOURCE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


PACKET.mkdir()
originals=read(STUDY/'initial-preservation.json')['original_public_sha256']
for name in originals:
    if name.startswith('rust/'):
        continue
    assert sha(ROOT/name)==originals[name],name
    destination=PACKET/'history'/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        stream.write((ROOT/name).read_bytes())
    assert sha(destination)==originals[name]
assert sha(ROOT/ARCHIVE)=='f961927f13fe7886b2026944e6830f11f5f60d24cf8056f29188d9d5a02fa686'
target=ROOT/'benchmarks/slam/flatwall-target-balance/manifest.json'
manifest=read(target)
assert manifest['normalized_native_sources_sha256'][SOURCE]==sha(ROOT/ARCHIVE)
manifest['native_source_aliases'][SOURCE]=ARCHIVE
manifest['dependencies'][ARCHIVE]=sha(ROOT/ARCHIVE)
save(target,manifest)

folder=ROOT/'benchmarks/slam/flatwall-balance-dynamics'
footprint=read(folder/'source-footprints.json')
footprint['normalized_native_sources_sha256'][ARCHIVE]=footprint['normalized_native_sources_sha256'].pop(SOURCE)
footprint['native_source_aliases'][SOURCE]=ARCHIVE
save(folder/'source-footprints.json',footprint)
manifest=read(folder/'manifest.json')
manifest['normalized_native_sources_sha256'][ARCHIVE]=manifest['normalized_native_sources_sha256'].pop(SOURCE)
manifest['files_sha256']['source-footprints.json']=sha(folder/'source-footprints.json')
save(folder/'manifest.json',manifest)

test=ROOT/'cloudanalyzer/tests/test_flatwall_target_balance_packet.py'
old="    path=target.parents[2]/'rust/crates/ca-core/src/odometry_target_balance.rs'"
new="    name='rust/crates/ca-core/src/odometry_target_balance.rs'\n    alias=packet.read(target/'manifest.json').get('native_source_aliases',{}).get(name,name)\n    path=target.parents[2]/alias"
source=test.read_text(encoding='utf-8')
assert source.count(old)==1
test.write_text(source.replace(old,new),encoding='utf-8',newline='\n')
with (STUDY/'source-projection.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(dict(executed_source=SOURCE,unchanged_archive=ARCHIVE,archive_sha256=sha(ROOT/ARCHIVE),
        original_metadata_preserved={name:'benchmarks/slam/flatwall-count-lookup/history/'+name for name in originals if not name.startswith('rust/')},
        previous_measurements_changed=False),stream,indent=2)
    stream.write('\n')
print(json.dumps(dict(previous_native_source_unchanged=True,archived_original_public_files=4)))
