import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {collectionSource,transformCollectionSource} from '../../benchmarks/wasm-kdtree-collection/executed-source/web/scripts/kdtree-collection-source.mjs';

const root=fileURLToPath(new URL('../../',import.meta.url)),directory='benchmarks/wasm-kdtree-collection';
const bytes=n=>readFileSync(resolve(root,n)),hash=b=>createHash('sha256').update(b).digest('hex'),json=n=>JSON.parse(bytes(n));
const manifest=json(`${directory}/manifest.json`);
function record(e){assert.ok(!e.file.includes('..'));const raw=bytes(`${directory}/${e.file}`);assert.equal(raw.length,e.bytes);assert.equal(hash(raw),e.sha256);return raw;}
const protocol=JSON.parse(record(manifest.protocol)),build=JSON.parse(record(manifest.build));
const median=a=>{const s=a.toSorted((x,y)=>x-y);return(s[s.length/2-1]+s[s.length/2])/2;};

test('collection packet pins matched complete Rust builds and the sole collection source change',()=>{
 assert.equal(manifest.adopted,false);assert.equal(manifest.screenGatePassed,false);assert.equal(manifest.productionRustChanged,false);assert.equal(manifest.backendSelectionChanged,false);assert.equal(manifest.finalDefault,'two-borrowed-scans');
 assert.equal(protocol.status,'frozen_before_timing');assert.equal(protocol.screenReportBudget,4);assert.equal(protocol.warmups,6);assert.equal(protocol.repeats,12);assert.equal(build.purpose,'isolated-kdtree-collection-build');assert.equal(build.offline,true);assert.equal(build.productionWorkspaceUnchanged,true);
 for(const[n,e]of Object.entries(manifest.sourceArchives))assert.equal(hash(record(e)),protocol.normalizedSources[n]);
 for(const[n,h]of Object.entries(build.helpers))assert.equal(hash(record(manifest.sourceArchives[n])),h);
 const original=record(manifest.variantSourceArchives.baseline).toString('utf8');
 const normal=manifest.normalProductionBuildUnchanged;
 assert.deepEqual(build.rustInputs,Object.fromEntries(Object.entries(normal.normalizedInputs).filter(([n])=>n.startsWith('rust/'))));
 assert.equal(Object.keys(build.rustInputs).length,94);assert.equal(Object.keys(normal.artifacts).length,9);
 for(const mode of protocol.controls){
  const v=build.variants[mode],source=record(manifest.variantSourceArchives[mode]);
  assert.equal(source.toString('utf8'),transformCollectionSource(original,mode));assert.equal(hash(source),v.sourceSha256);assert.equal(protocol.variantSourceSha256[mode],v.sourceSha256);
  assert.deepEqual(v.normalizedRustInputs,{...build.rustInputs,[collectionSource]:v.sourceSha256});assert.deepEqual(protocol.variantArtifacts[mode],v.artifacts);assert.equal(Object.keys(v.artifacts).length,4);
  for(const[n,h]of Object.entries(v.artifacts))if(mode==='baseline'||!n.endsWith('.wasm'))assert.equal(h,normal.artifacts['web/src/wasm/'+n.split('/').at(-1)]);
 }
 const wasm=mode=>Object.entries(build.variants[mode].artifacts).find(([n])=>n.endsWith('.wasm'))[1];assert.notEqual(wasm('baseline'),wasm('unzip'));
 assert.equal(Object.keys(manifest.priorPacketFilesUnchanged).length,167);for(const[n,h]of Object.entries(manifest.priorPacketFilesUnchanged))assert.equal(hash(bytes(n)),h);
 assert.equal(hash(bytes(manifest.previousPacket.file)),manifest.previousPacket.sha256);
});

test('collection packet keeps every balanced pair, exact result witness and rejected screen gate',()=>{
 assert.deepEqual(manifest.reports.map(e=>e.dataset),protocol.datasetOrder);assert.deepEqual(manifest.reports.map(e=>e.first),protocol.firstOrder);
 const identities=new Map();let pairs=0,wins=0,passed=0;
 for(const[i,e]of manifest.reports.entries()){
  const r=JSON.parse(record(e)),m=r.measurement;
  assert.equal(r.status,'passed');assert.equal(r.purpose,'constructor-screen');assert.equal(r.commit,protocol.baseCommit);assert.equal(r.wasmKdTreeCollectionScreen,true);assert.equal(r.warmups,6);assert.equal(r.repeats,12);assert.equal(r.first,protocol.firstOrder[i]);assert.equal(r.filesUnchangedDuringMeasurement,true);assert.deepEqual(r.build,build);assert.equal(r.buildInfoSha256,manifest.build.sha256);
  assert.deepEqual(r.sources,{'web/scripts/benchmark-kdtree-collection.mjs':protocol.normalizedSources['web/scripts/benchmark-kdtree-collection.mjs']});
  const input=record(manifest.inputSpecifications[e.dataset]);assert.equal(hash(input),r.inputManifestSha256);assert.deepEqual(JSON.parse(input),r.inputManifest);
  for(const role of ['reference','compared'])for(const field of ['points','bytes','sha256'])assert.equal(r.inputs[role][field],r.inputManifest.inputs[role][field]);
  for(const flag of ['workerScope','validationUnchanged','distanceBitsExact','snapshotBitsExact','gpuExportBitsExact','queryBitsExact','ownershipChecksPassed','sourceCloudsReleased'])assert.equal(m[flag],true);
  assert.equal(m.gpuRequests,0);assert.equal(m.wasmSharedMemory,false);assert.equal(m.liveReferences,0);assert.equal(m.liveJobs,0);
  assert.deepEqual(m.validation.baseline,m.validation.unzip);assert.deepEqual(m.validation.baseline,manifest.validationByDataset[e.dataset]);
  const v=m.validation.baseline;assert.equal(v.referencePoints,r.inputs.reference.points);assert.equal(v.comparedPoints,r.inputs.compared.points);assert.equal(v.hintChecks,12);assert.equal(v.rangeChecks,2);assert.equal(v.independentQueryExport,true);assert.equal(v.referenceReleasedBeforeVerification,true);
  for(const[n,h]of Object.entries(v))if(n.endsWith('Sha256'))assert.match(h,/^[0-9a-f]{64}$/);
  if(identities.has(e.dataset))assert.deepEqual(v,identities.get(e.dataset));else identities.set(e.dataset,v);
  assert.equal(m.warming.length,6);assert.equal(m.trials.length,12);
  for(const[j,t]of [...m.warming,...m.trials].entries()){
   const first=j%2?(r.first==='baseline'?'unzip':'baseline'):r.first;assert.deepEqual(t.order,first==='baseline'?['baseline','unzip']:['unzip','baseline']);
   assert.deepEqual(Object.keys(t.results).sort(),['baseline','unzip']);for(const value of Object.values(t.results)){assert.ok(value.constructorMillis>0);assert.ok(value.wasmLinearMemoryBytes>0);}
  }
  assert.deepEqual(m.counts,{created:{baseline:20,unzip:20},freed:{baseline:20,unzip:20},jobs:4,jobReleases:4});
  const b=m.trials.map(t=>t.results.baseline.constructorMillis),c=m.trials.map(t=>t.results.unzip.constructorMillis),ratios=c.map((v,j)=>v/b[j]);
  assert.equal(e.pairs,12);assert.equal(e.baselineMedianMillis,median(b));assert.equal(e.unzipMedianMillis,median(c));assert.equal(e.medianPairedRatio,median(ratios));assert.equal(e.medianPairedReductionPercent,100*(1-median(ratios)));assert.deepEqual(e.pairedRatios,ratios);assert.equal(e.unzipWins,c.filter((v,j)=>v<b[j]).length);
  assert.equal(e.baselineMinimumMillis,Math.min(...b));assert.equal(e.baselineMaximumMillis,Math.max(...b));assert.equal(e.unzipMinimumMillis,Math.min(...c));assert.equal(e.unzipMaximumMillis,Math.max(...c));
  const memory=Object.fromEntries(protocol.controls.map(mode=>[mode,Math.max(...m.trials.map(t=>t.results[mode].wasmLinearMemoryBytes))]));assert.deepEqual(e.committedWasmMemoryBytes,memory);
  const memoryRatio=Math.max(...m.trials.map(t=>t.results.unzip.wasmLinearMemoryBytes/t.results.baseline.wasmLinearMemoryBytes));assert.equal(e.maximumCommittedMemoryRatio,memoryRatio);
  const g=protocol.screenGate,gates={reduction:e.medianPairedRatio<=1-g.minimumMedianPairedReduction,wins:e.unzipWins>=g.minimumFasterPairsPerReport,committedMemory:memoryRatio<=g.maximumCommittedMemoryRatio};assert.deepEqual(e.gates,gates);assert.equal(e.gatePassed,Object.values(gates).every(Boolean));
  pairs+=e.pairs;wins+=e.unzipWins;passed+=Number(e.gatePassed);
 }
 assert.equal(pairs,48);assert.equal(wins,32);assert.equal(passed,2);assert.equal(manifest.reportsPassingGate,passed);assert.equal(manifest.unzipWins,wins);
 assert.deepEqual(manifest.screenBudget,{allowed:4,used:4,remaining:0,closed:true,allCorrectnessChecksPassed:true,warmupPairs:24,measuredPairs:48,measuredConstructors:96,validationConstructors:16,gpuRequests:0});
 const rpc=manifest.conditionalRpcBudget;assert.equal(rpc.allowed,4);assert.equal(rpc.used,0);assert.equal(rpc.discardedUnused,4);assert.equal(rpc.remaining,0);assert.equal(rpc.closed,true);
 assert.equal(protocol.screenGate.allFourReportsRequired,true);assert.equal(protocol.conditionalRpc.onlyIfAllScreensPass,true);
});
