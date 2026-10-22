import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../../',import.meta.url)),directory='benchmarks/wasm-gpu-mapped-reference';
const bytes=n=>readFileSync(resolve(root,n)),hash=b=>createHash('sha256').update(b).digest('hex'),json=n=>JSON.parse(bytes(n)),normal=n=>hash(bytes(n).toString('utf8').replaceAll('\r\n','\n'));
const median=a=>{a=a.toSorted((x,y)=>x-y);return (a[a.length/2-1]+a[a.length/2])/2;};
function compare(pairs){const q=pairs.map(p=>p.queued.elapsedMillis),m=pairs.map(p=>p.mapped.elapsedMillis),ratios=q.map((v,i)=>m[i]/v);return{pairs:pairs.length,queuedMedianMillis:median(q),mappedMedianMillis:median(m),medianPairedRatio:median(ratios),medianPairedReductionPercent:100*(1-median(ratios)),mappedWins:q.filter((v,i)=>m[i]<v).length,queuedMinimumMillis:Math.min(...q),queuedMaximumMillis:Math.max(...q),mappedMinimumMillis:Math.min(...m),mappedMaximumMillis:Math.max(...m),pairedRatios:ratios};}

test('mapped reference packet retains exact production outputs, cache isolation, every sample and the failed adoption gate',()=>{
 const m=json(`${directory}/manifest.json`),p=m.protocol;assert.equal(m.adopted,false);assert.equal(m.adoptionGatePassed,false);assert.equal(m.finalDefault,'queued');assert.equal(m.backendSelectionChanged,false);assert.deepEqual(m.finalBuild,m.measuredBuild);
 assert.equal(m.timingBudget.allowed,4);assert.equal(m.timingBudget.used,4);assert.equal(m.timingBudget.remaining,0);assert.equal(m.timingBudget.closed,true);assert.equal(m.timingBudget.allPassed,true);assert.equal(m.timingBudget.gpuTimestampQueries,0);
 assert.equal(hash(JSON.stringify(p,null,2)+'\n'),m.protocolSha256);assert.equal(p.timingReportBudget,4);assert.equal(p.warmups,6);assert.equal(p.repeats,12);assert.equal(p.coldPairs,4);assert.equal(p.nodes,'flat');
 assert.equal(hash(Object.entries(m.normalizedSourceFootprint).sort(([a],[b])=>a<b?-1:1).map(([n,h])=>n+'\0'+h+'\n').join('')),m.normalizedSourceFingerprintSha256);
 const source=(n,h)=>{if(normal(n)===h)return;const a=m.sourceAliases[h];assert.equal(a?.original,n);assert.ok(a.archived.startsWith(`${directory}/executed-source/`)&&!a.archived.includes('..'));assert.equal(normal(a.archived),h);};
 for(const[n,h]of Object.entries(m.normalizedSourceFootprint))source(n,h);
 for(const[h,a]of Object.entries(m.sourceAliases))assert.equal(normal(a.archived),h);
 for(const[n,h]of Object.entries(m.priorPacketFilesUnchanged))assert.equal(hash(bytes(n)),h);assert.equal(Object.keys(m.priorPacketFilesUnchanged).length,154);
 assert.equal(hash(bytes(m.previousPacket.file)),m.previousPacket.sha256);
 const reused=m.measuredBuild.wasmReuse;assert.deepEqual(Object.keys(reused.unchangedRustInputs).sort(),Object.keys(m.measuredBuild.normalizedInputs).filter(n=>n.startsWith('rust/')).sort());
 for(const[n,h]of Object.entries(reused.unchangedRustInputs))assert.equal(h,m.measuredBuild.normalizedInputs[n]);for(const[n,h]of Object.entries(reused.unchangedArtifacts))assert.equal(h,m.measuredBuild.artifacts[n]);
 const identities=new Map();let warmPairs=0,coldPairs=0,passed=0,coldWins=0;
 assert.deepEqual(m.reports.map(e=>e.dataset),p.datasetOrder);
 for(const[i,e]of m.reports.entries()){
  const raw=bytes(`${directory}/${e.file}`),r=JSON.parse(raw),v=r.measurement;assert.equal(raw.length,e.bytes);assert.equal(hash(raw),e.sha256);assert.equal(r.status,'passed');assert.equal(r.purpose,'timing-comparison');assert.equal(r.gpu,'physical');assert.equal(r.gpuMappedReferenceComparison,true);assert.equal(r.threads,4);assert.equal(r.warmups,6);assert.equal(r.repeats,12);assert.equal(r.coldPairs,4);assert.equal(r.nodes,'flat');assert.equal(r.coldFirst,p.coldFirstOrder[i]);assert.equal(r.filesUnchangedDuringMeasurement,true);assert.deepEqual(r.build,m.measuredBuild);assert.equal(hash(JSON.stringify(r.build,null,2)+'\n'),r.buildInfoSha256);
  for(const[n,h]of Object.entries({...r.sources,...r.build.normalizedInputs}))source(n,h);
  assert.equal(hash(bytes(`${directory}/inputs-${e.dataset}-01.json`)),r.inputManifestSha256);assert.deepEqual(json(`${directory}/inputs-${e.dataset}-01.json`),r.inputManifest);
  for(const role of ['reference','compared']){assert.equal(r.inputs[role].sha256,r.inputManifest.inputs[role].sha256);assert.equal(r.inputs[role].bytes,r.inputManifest.inputs[role].bytes);assert.equal(v.inputPoints[role],r.inputManifest.inputs[role].points);}
  for(const flag of ['physicalGpuRequired','physicalWorkerAdapterChecked','queryBitsExact','queryRangeGuardsChecked','ownerSnapshotAndExportIndependenceChecked','distanceBitsExact','summaryFieldsExact','certificateCountsStablePerMode','sourceRemovalCompleted'])assert.equal(v[flag],true);
  assert.equal(v.cappedWorkerHardwareConcurrency,5);assert.equal(v.deviceRequests,1);assert.equal(v.gpuTimestampRequests,0);assert.equal(v.defaultMappedReference,false);assert.deepEqual(v.certifiedByMode,{queued:e.dataset==='pandaset'?235714:421096,mapped:e.dataset==='pandaset'?235714:421096});
  const identity=[v.displayedF32Sha256,v.queryBitsSha256,v.stats];if(identities.has(e.dataset))assert.deepEqual(identity,identities.get(e.dataset));else identities.set(e.dataset,identity);
  const check=(value,mode,reused)=>{assert.ok(value.elapsedMillis>=value.workerMillis);const c=value.compute;assert.equal(c.backend,'webgpu');assert.equal(c.fallback,undefined);assert.equal(c.verifyFallback,undefined);assert.equal(c.mappedReference,mode==='mapped');assert.equal(c.verifyWorkers,4);assert.equal(c.workgroupSize,64);assert.equal(c.batchSize,131072);assert.equal(c.certified,v.certifiedByMode[mode]);assert.equal(c.batches,Math.ceil(v.inputPoints.compared/131072));assert.equal(c.nodeBytes,c.nodeCount*20);assert.equal(c.retainedNodeBytes,c.nodeBytes);assert.equal(c.boxBoundsBytes,0);assert.equal(c.parentTraversalBytes,0);
   for(const f of ['packedNodes','compactStack','inheritedBounds','boxBounds','parentTraversal','querySliceCopy','queryOwnedCopy','verifyStagedQueries','verifyMainWorker','leafOnly','bufferedReadback'])assert.equal(c[f],false);
   for(const f of ['prefetch','verifyBounds','verifySimd','verifySnapshot','verifyReuseImportBounds','deviceReused'])assert.equal(c[f],true);
   assert.equal(Object.hasOwn(c,'diagnostics'),false);assert.equal(Object.hasOwn(c,'preparationProbe'),false);
   for(const f of ['referenceReused','verifyReferenceReused','nodeRepresentationReused'])assert.equal(c[f],reused);
  };
  assert.equal(v.warming.length,6);assert.equal(v.trials.length,12);assert.equal(v.cold.length,4);
  for(const[j,t]of [...v.warming,...v.trials].entries()){assert.deepEqual(t.order,j%2?['mapped','queued']:['queued','mapped']);for(const mode of p.controls)check(t.results[mode],mode,true);}
  for(const[j,t]of v.cold.entries()){const first=j%2?(r.coldFirst==='queued'?'mapped':'queued'):r.coldFirst;assert.deepEqual(t.order,first==='queued'?['queued','mapped']:['mapped','queued']);for(const mode of p.controls){check(t.results[mode].cold,mode,false);check(t.results[mode].cached,mode,true);}}
  const warm=compare(v.trials.map(t=>t.results)),cold=compare(v.cold.map(t=>Object.fromEntries(p.controls.map(mode=>[mode,t.results[mode].cold])))),cached=compare(v.cold.map(t=>Object.fromEntries(p.controls.map(mode=>[mode,t.results[mode].cached]))));assert.deepEqual(e.warm,warm);assert.deepEqual(e.cold,cold);assert.deepEqual(e.cachedFollowup,cached);
  const g=p.adoptionGate,gates={coldReduction:cold.medianPairedRatio<=1-g.minimumMedianPairedColdReduction,coldWins:cold.mappedWins>=g.minimumColdWinsPerReport,warmRegression:warm.medianPairedRatio<=g.maximumMedianPairedWarmRatio,cachedFollowupRegression:cached.medianPairedRatio<=g.maximumMedianPairedCachedFollowupRatio};assert.deepEqual(e.gates,gates);assert.equal(e.gatePassed,Object.values(gates).every(Boolean));
  assert.deepEqual(e.coldNodeReadinessMillis,Object.fromEntries(p.controls.map(mode=>[mode,v.cold.map(t=>t.results[mode].cold.compute.nodePrepareMillis)])));
  warmPairs+=warm.pairs;coldPairs+=cold.pairs;coldWins+=cold.mappedWins;passed+=Number(e.gatePassed);
 }
 assert.equal(warmPairs,48);assert.equal(coldPairs,16);assert.equal(coldWins,12);assert.equal(passed,2);assert.equal(m.reportsPassingGate,passed);assert.equal(m.coldMappedWins,coldWins);assert.equal(m.coldOperations,32);assert.equal(m.cachedFollowups,32);
});
