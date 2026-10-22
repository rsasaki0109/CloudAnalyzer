import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=fileURLToPath(new URL('../../',import.meta.url)),directory='benchmarks/wasm-reference-preparation';
const bytes=n=>readFileSync(resolve(root,n)),json=n=>JSON.parse(bytes(n)),hash=b=>createHash('sha256').update(b).digest('hex'),normal=n=>hash(bytes(n).toString('utf8').replaceAll('\r\n','\n'));
const median=a=>{a=a.toSorted((x,y)=>x-y);return a.length%2?a[Math.floor(a.length/2)]:(a[a.length/2-1]+a[a.length/2])/2;};
const describe=a=>({count:a.length,minimum:Math.min(...a),median:median(a),maximum:Math.max(...a)});
function phases(c){
 const events=c.preparationProbe.events,at=n=>events.filter(e=>e.phase===n),result={};
 for(const[key,a,b]of[['tree','tree-start','tree-end'],['querySnapshot','query-snapshot-start','query-snapshot-end'],['verifierSnapshot','verifier-snapshot-start','verifier-snapshot-end'],['candidateExport','candidate-export-start','candidate-export-end'],['pointsCopy','points-copy-start','points-copy-end'],['nodesCopy','nodes-copy-start','nodes-copy-end'],['nodeEncoding','node-encode-start','node-encode-end'],['pointsAllocationCall','points-allocate-start','points-allocate-end'],['pointsWriteCall','points-write-start','points-write-end'],['nodesAllocationCall','nodes-allocate-start','nodes-allocate-end'],['nodesWriteCall','nodes-write-start','nodes-write-end'],['uploadScopeQueue','reference-upload-queued','reference-upload-enter'],['uploadReadiness','reference-upload-queued','reference-upload-ready'],['nodeReadiness','node-readiness-start','node-readiness-end'],['verifierReadiness','verifier-snapshot-start','verifiers-ready']]){
  const start=at(a),end=at(b);if(!start.length&&(!end.length||b==='verifiers-ready'))continue;
  assert.equal(start.length,1,a);assert.equal(end.length,1,b);assert.ok(end[0].at>=start[0].at);result[key]=end[0].at-start[0].at;
 }
 const upload=at('reference-upload-enter')[0];if(upload){const scope=at('device-scope-enter').filter(e=>e.at<=upload.at).at(-1),start=at('scope-pop-start').find(e=>e.id===scope.id),end=at('scope-pop-end').find(e=>e.id===scope.id);assert.ok(start&&end);result.uploadScopePop=end.at-start.at;}
 return result;
}

test('reference preparation packet retains balanced cold/warm observations, unchanged math and bounded phase provenance',async()=>{
 const m=json(`${directory}/manifest.json`),priorBytes=bytes('benchmarks/wasm-gpu-packed-nodes/manifest.json'),prior=JSON.parse(priorBytes);
 const source=(n,h)=>{if(normal(n)===h)return n;const a=m.sourceAliases[h];assert.equal(a?.original,n);assert.ok(a.archived.startsWith(`${directory}/executed-source/`)&&!a.archived.includes('..'));assert.equal(normal(a.archived),h);return a.archived;};
 assert.equal(m.purpose,'explanatory-only-no-adoption');assert.equal(m.productionDefaultChanged,false);assert.equal(m.backendSelectionChanged,false);assert.equal(m.normalProductionArtifactsUnchanged,true);assert.deepEqual(m.normalProductionBuild,prior.finalBuild);
 assert.equal(m.timingBudget.allowed,4);assert.equal(m.timingBudget.used,4);assert.equal(m.timingBudget.remaining,0);assert.equal(m.timingBudget.coldSamples,32);assert.equal(m.timingBudget.cachedWarmFollowups,32);assert.equal(m.timingBudget.gpuTimestampQueries,0);
 assert.equal(hash(Object.entries(m.normalizedSourceFootprint).sort(([a],[b])=>a<b?-1:1).map(([n,h])=>n+'\0'+h+'\n').join('')),m.normalizedSourceFingerprintSha256);
 for(const[n,h]of Object.entries(m.normalizedSourceFootprint))source(n,h);
 for(const[h,a]of Object.entries(m.sourceAliases))assert.equal(normal(a.archived),h);
 const helperHash=m.normalizedSourceFootprint['web/scripts/gpu-reference-preparation-probe.mjs'],helper=await import(pathToFileURL(resolve(root,m.sourceAliases[helperHash].archived)).href);
 for(const[n,v]of Object.entries(m.instrumentedSources)){
  const original=bytes(source(n,v.originalSha256)).toString('utf8').replaceAll('\r\n','\n');assert.equal(hash(helper.instrumentPreparationSource(n,original)),v.sha256);assert.equal(normal(v.archived),v.sha256);assert.equal(m.profileBuild.instrumentedSources[n].sha256,v.sha256);assert.ok(m.profileBuild.transformContexts.includes('worker:'+n));
 }
 assert.equal(m.profileBuild.wasmReuse.priorMetadataSha256,hash(priorBytes));
 assert.deepEqual(Object.keys(m.profileBuild.wasmReuse.unchangedRustInputs).sort(),Object.keys(prior.finalBuild.normalizedInputs).filter(n=>n.startsWith('rust/')).sort());
 for(const[n,h]of Object.entries(m.profileBuild.wasmReuse.unchangedRustInputs))assert.equal(h,prior.finalBuild.normalizedInputs[n]);
 for(const[n,h]of Object.entries(m.profileBuild.wasmReuse.unchangedArtifacts))assert.equal(h,prior.finalBuild.artifacts[n]);
 const all=[],successors=new Map(),positions=new Map();let cold=0,warm=0;
 assert.deepEqual(m.reports.map(e=>e.dataset),['pandaset','autoware','autoware','pandaset']);
 for(const[index,entry]of m.reports.entries()){
  const raw=bytes(`${directory}/${entry.file}`),r=JSON.parse(raw),measurement=r.measurement;assert.equal(hash(raw),entry.sha256);assert.equal(raw.length,entry.bytes);assert.equal(r.status,'passed');assert.equal(r.purpose,'explanatory-reference-preparation-profile');assert.equal(r.gpuReferencePreparationProfile,true);assert.equal(r.gpu,'physical');assert.equal(r.row,index);assert.equal(r.cycles,2);assert.equal(r.threads,4);assert.equal(r.filesUnchangedDuringMeasurement,true);
  assert.deepEqual(r.build,m.profileBuild);assert.equal(hash(JSON.stringify(r.build,null,2)+'\n'),r.buildInfoSha256);for(const[n,h]of Object.entries({...r.sources,...r.build.normalizedInputs}))source(n,h);
  const spec=m.inputs[entry.dataset],specBytes=bytes(`${directory}/${spec.file}`);assert.equal(hash(specBytes),spec.sha256);assert.equal(specBytes.length,spec.bytes);assert.equal(r.inputManifestSha256,spec.sha256);
  const previous=prior.reports.find(e=>e.dataset===entry.dataset);assert.equal(measurement.queryBitsSha256,previous.queryBitsSha256);assert.equal(measurement.displayedF32Sha256,previous.displayedF32Sha256);assert.equal(measurement.certifiedByMode.flat,previous.certifiedByMode.flat);assert.equal(measurement.certifiedByMode.flat,measurement.certifiedByMode.packed);assert.deepEqual(measurement.certifiedByMode,entry.certifiedByMode);
  for(const f of ['physicalGpuRequired','physicalWorkerAdapterChecked','queryBitsExact','queryRangeGuardsChecked','ownerSnapshotAndExportIndependenceChecked','distanceBitsExact','summaryFieldsExact','certificateCountsStablePerMode','sourceRemovalCompleted'])assert.equal(measurement[f],true);
  assert.equal(measurement.deviceRequests,1);assert.equal(measurement.gpuTimestampRequests,0);assert.equal(measurement.trials.length,2);
  const samples=measurement.trials.flatMap(t=>t.samples);
  for(const[cycle,t]of measurement.trials.entries()){
   assert.deepEqual(t.order,cycle?m.protocol.orders[index].toReversed():m.protocol.orders[index]);assert.deepEqual(t.samples.map(s=>s.arm),t.order);
   t.order.forEach((arm,i)=>{const key=arm+':'+i;positions.set(key,(positions.get(key)||0)+1);if(i){const pair=t.order[i-1]+':'+arm;successors.set(pair,(successors.get(pair)||0)+1);}});
  }
  const observed=[];
  for(const s of samples){
   const mode=s.arm.split('-')[0],traced=s.arm.endsWith('-on');
   for(const[phase,reused]of [['cold',false],['warm',true]]){
    phase==='cold'?cold++:warm++;const value=s[phase],c=value.compute;assert.ok(value.elapsedMillis>=value.workerMillis);assert.equal(c.backend,'webgpu');assert.equal(c.fallback,undefined);assert.equal(c.verifyFallback,undefined);assert.equal(c.verifyWorkers,4);assert.equal(c.workgroupSize,64);assert.equal(c.batchSize,131072);assert.equal(c.packedNodes,mode==='packed');
    for(const flag of ['compactStack','inheritedBounds','boxBounds','parentTraversal','querySliceCopy','queryOwnedCopy','verifyStagedQueries','verifyMainWorker','leafOnly'])assert.equal(c[flag],false);
    for(const flag of ['prefetch','verifyBounds','verifySimd','verifySnapshot','deviceReused'])assert.equal(c[flag],true);
    for(const flag of ['referenceReused','verifyReferenceReused','nodeRepresentationReused'])assert.equal(c[flag],reused);
    assert.equal(c.boxBoundsBytes,0);assert.equal(c.parentTraversalBytes,0);assert.equal(c.nodeBytes,c.nodeCount*(mode==='packed'?8:20));assert.equal(c.retainedNodeBytes,c.nodeBytes);assert.equal(c.certified,measurement.certifiedByMode[mode]);assert.equal(c.batches,Math.ceil(measurement.inputPoints.compared/131072));assert.equal(Object.hasOwn(c,'diagnostics'),false);
    if(!traced){assert.equal(c.preparationProbe,undefined);continue;}
    const p=c.preparationProbe;assert.equal(p.omitted,0);assert.ok(p.events.length>0&&p.events.length<=256);assert.equal(p.events[0].phase,'c2c-start');assert.equal(p.events.at(-1).phase,'c2c-return');assert.ok(p.events.every((e,i)=>Number.isFinite(e.at)&&(!i||e.at>=p.events[i-1].at)));
    for(const enter of p.events.filter(e=>e.phase==='device-scope-enter'))for(const name of ['scope-pop-start','scope-pop-end'])assert.equal(p.events.filter(e=>e.phase===name&&e.id===enter.id).length,1);
    if(reused)assert.equal(p.events.some(e=>['tree-start','candidate-export-start','reference-upload-queued','node-encode-start'].includes(e.phase)),false);
   }
   if(traced)observed.push({arm:s.arm,cold:phases(s.cold.compute),warm:phases(s.warm.compute)});
  }
  assert.deepEqual(observed,entry.summary.instrumented);all.push(...observed.map(s=>({report:entry.file.replace('.json',''),dataset:entry.dataset,...s})));
  for(const arm of m.protocol.arms)for(const phase of ['cold','warm'])assert.deepEqual(entry.summary.arms[arm][phase],describe(samples.filter(s=>s.arm===arm).map(s=>s[phase].elapsedMillis)));
 }
 assert.equal(cold,32);assert.equal(warm,32);assert.equal(all.length,16);assert.equal(positions.size,16);assert.ok([...positions.values()].every(n=>n===2));assert.equal(successors.size,12);assert.ok([...successors.values()].every(n=>n===2));
 for(const dataset of ['pandaset','autoware']){
  const observed=all.filter(s=>s.dataset===dataset);for(const key of Object.keys(observed[0].cold))assert.deepEqual(m.datasets[dataset].coldPhaseMillis[key],describe(observed.map(s=>s.cold[key])));
  const samples=m.reports.filter(e=>e.dataset===dataset).flatMap(e=>json(`${directory}/${e.file}`).measurement.trials.flatMap(t=>t.samples));for(const arm of m.protocol.arms)for(const phase of ['cold','warm'])assert.deepEqual(m.datasets[dataset].arms[arm][phase],describe(samples.filter(s=>s.arm===arm).map(s=>s[phase].elapsedMillis)));
 }
 const long=all.filter(s=>s.cold.nodeReadiness>=50).map(s=>({report:s.report,arm:s.arm,nodeReadinessMillis:s.cold.nodeReadiness,uploadScopePopMillis:s.cold.uploadScopePop,popFraction:s.cold.uploadScopePop/s.cold.nodeReadiness,nodeEncodingMillis:s.cold.nodeEncoding,scopePopAtLeast80Percent:s.cold.uploadScopePop>=s.cold.nodeReadiness*.8}));
 assert.deepEqual(m.longReadinessIntervals,long);assert.equal(m.longReadinessCount,long.length);assert.equal(m.popDominatedLongReadiness,long.filter(s=>s.scopePopAtLeast80Percent).length);assert.equal(long.length,9);
});
