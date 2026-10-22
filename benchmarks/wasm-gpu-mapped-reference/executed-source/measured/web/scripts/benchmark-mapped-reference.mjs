// Same-binary comparison through the ordinary production distance Worker.
import { chromium } from '@playwright/test';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve, basename, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2), allowed = ['--output', '--build-info', '--reference', '--compared', '--input-manifest', '--threads', '--warmups', '--repeats', '--gpu', '--cold-first', '--cold-pairs', '--nodes'];
for (let i = 0; i < args.length; i += 2) if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith('--') || args.indexOf(args[i]) !== i)
  throw new Error(`Invalid option: ${args[i]}`);
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = option('--output'), buildFile = option('--build-info'), reference = option('--reference'), compared = option('--compared');
if (!output || !buildFile || !reference || !compared || existsSync(output)) throw new Error('Supply a NEW --output, --build-info, --reference and --compared');
const threads = Number(option('--threads', '4')), warmups = Number(option('--warmups', '6')), repeats = Number(option('--repeats', '12')), gpu = option('--gpu', 'physical');
if (!Number.isInteger(threads) || threads < 2 || threads > 8 || !Number.isInteger(warmups) || warmups < 0 || !Number.isInteger(repeats) || repeats < 1 || !['physical', 'software'].includes(gpu))
  throw new Error('Invalid trial settings');
const coldFirst=option('--cold-first','queued'),coldPairs=Number(option('--cold-pairs','4')),nodes=option('--nodes','flat');
if(!['queued','mapped'].includes(coldFirst)||!Number.isInteger(coldPairs)||coldPairs<1||coldPairs>8||!['flat','packed'].includes(nodes))throw new Error('Invalid cold settings');
const root = fileURLToPath(new URL('../../', import.meta.url)), dist = resolve(root, 'web/dist');
const hash = data => createHash('sha256').update(data).digest('hex');
const buildBytes = readFileSync(buildFile), build = JSON.parse(buildBytes);
function verifyBuild() {
  for (const [name, expected] of Object.entries(build.normalizedInputs)) if (hash(readFileSync(resolve(root, name), 'utf8').replaceAll('\r\n', '\n')) !== expected) throw new Error(`Source changed: ${name}`);
  for (const [name, expected] of Object.entries(build.artifacts)) if (hash(readFileSync(resolve(root, name))) !== expected) throw new Error(`Artifact changed: ${name}`);
}
verifyBuild();
const inputs = Object.fromEntries([['reference', reference], ['compared', compared]].map(([role, path]) => {
  const bytes = readFileSync(path); return [role, { name: basename(path), path: resolve(path), bytes: bytes.length, sha256: hash(bytes) }];
}));
const specFile = option('--input-manifest'), specBytes = specFile ? readFileSync(specFile) : undefined, inputManifest = specBytes ? JSON.parse(specBytes) : undefined;
if (inputManifest) for (const [role, input] of Object.entries(inputs)) {
  if (inputManifest.inputs[role].sha256 !== input.sha256 || inputManifest.inputs[role].bytes !== input.bytes) throw new Error('Input specification mismatch');
  input.points = inputManifest.inputs[role].points;
}
const harnessSource = 'web/scripts/benchmark-mapped-reference.mjs', harnessHash = hash(readFileSync(resolve(root, harnessSource), 'utf8').replaceAll('\r\n', '\n'));
const workerName = readdirSync(resolve(dist, 'assets')).find(n => /^worker-.*\.js$/.test(n));
if (!workerName) throw new Error('No production worker');
const bundledWasm = Object.entries(build.artifacts).filter(([name]) => name.startsWith('web/dist/assets/') && name.endsWith('.wasm'));
if (bundledWasm.length !== 1 || bundledWasm[0][1] !== build.artifacts['web/src/wasm/ca_wasm_bg.wasm']) throw new Error('Diagnostic and production WASM differ');
const server = createServer((req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/__query-export') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>GPU mapped reference comparison</title>'); return; }
  for (const [role, input] of Object.entries(inputs)) if (url.pathname === `/__${role}`) {
    const data = readFileSync(input.path); if (hash(data) !== input.sha256) { res.writeHead(409).end(); return; }
    res.end(data); return;
  }
  const wasm = url.pathname.startsWith('/__wasm/'), directory = wasm ? resolve(root, 'web/src/wasm') : dist;
  const name = wasm ? url.pathname.slice('/__wasm/'.length) : url.pathname.replace(/^\/CloudAnalyzer\/app/, '').slice(1);
  const path = resolve(directory, name);
  if (!path.startsWith(directory + sep) || (wasm && !['ca_wasm.js', 'ca_wasm_bg.wasm'].includes(name))) { res.writeHead(403).end(); return; }
  try { res.setHeader('Content-Type', ({ '.js':'text/javascript', '.wasm':'application/wasm' })[extname(path)] || 'application/octet-stream'); res.end(readFileSync(path)); }
  catch { res.writeHead(404).end(); }
});
async function measure({ inputs, workerName, threads, physical, warmups, repeats, coldFirst, coldPairs, nodes }) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || physical === adapter.info.isFallbackAdapter) throw new Error('Requested GPU adapter class unavailable');
  // This measurement-only wrapper caps the pool before importing production code.
  const workerUrl = new URL(`/assets/${workerName}`, location.href).href;
  const wrapper = `Object.defineProperty(navigator,'hardwareConcurrency',{value:${threads+1}});
    const requestDevice=GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice=function(...args){
      if(${physical}===this.info.isFallbackAdapter)throw new Error('Wrong Worker GPU adapter class');
      const timestampRequested=args[0]?.requiredFeatures?.includes('timestamp-query')??false;
      if(timestampRequested)throw new Error('Mapped reference comparison must not request timestamps');
      self.postMessage({prepDeviceRequested:true});return requestDevice.apply(this,args);
    };
    const queued=[];self.onmessage=e=>queued.push(e);await import(${JSON.stringify(workerUrl)});const handle=self.onmessage;
    for(const event of queued)handle(event);`;
  const wrapperUrl = URL.createObjectURL(new Blob([wrapper], { type: 'text/javascript' }));
  const worker = new Worker(wrapperUrl, { type: 'module' }), pending = new Map(); let seq = 0, deviceRequests = 0;
  const fail = error => { for (const p of pending.values()) { clearTimeout(p.timer); p.reject(error); } pending.clear(); };
  worker.onerror = e => fail(new Error(e.message || 'Worker error'));
  worker.onmessageerror = () => fail(new Error('Worker message error'));
  worker.onmessage = ({ data }) => {
    if(data.prepDeviceRequested){deviceRequests++;return;}
    if (!data.response) return;
    const p = pending.get(data.seq); if (!p) return;
    pending.delete(data.seq); clearTimeout(p.timer);
    if (data.response.ok) p.resolve({ value: data.response.value, elapsedMillis: performance.now() - p.start });
    else p.reject(new Error(data.response.error));
  };
  const call = req => new Promise((resolve, reject) => {
    const id = ++seq, start = performance.now(), timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${req.kind}`)); }, 60_000);
    pending.set(id, { resolve, reject, start, timer }); worker.postMessage({ seq: id, req });
  });
  const digest = async array => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(array.buffer, array.byteOffset, array.byteLength).slice()))).map(v => v.toString(16).padStart(2, '0')).join('');
  const files = {};
  try {
    for (const role of ['reference', 'compared']) { const response = await fetch(`/__${role}`); if (!response.ok) throw new Error('Input fetch failed'); files[role] = new Uint8Array(await response.arrayBuffer()); }
    const wasm = await import('/__wasm/ca_wasm.js'); await wasm.default();
    const r = wasm.Cloud.parse(inputs.reference.name, files.reference), q = wasm.Cloud.parse(inputs.compared.name, files.compared);
    const index = new wasm.GpuC2cReference(r), job = new wasm.GpuC2cJob(q, index);
    r.free(); q.free(); index.free();
    let queryBitsSha256, copiedQueryBytes;
    try {
      const iterator = job.queriesF64WithCopy(0, job.length, false), slice = job.queriesF64WithCopy(0, job.length, true);
      const a = new BigUint64Array(iterator.buffer), b = new BigUint64Array(slice.buffer);
      const compatible = job.queriesF64(0, job.length);
      if (new BigUint64Array(compatible.buffer).some((v, i) => v !== a[i])) throw new Error('Two-argument query API changed');
      if (a.length !== b.length || a.some((v, i) => v !== b[i])) throw new Error('Exact query export bits differ');
      queryBitsSha256 = await digest(iterator); copiedQueryBytes = iterator.byteLength;
      slice.fill(42);
      const owned = job.queriesF64WithCopy(0, job.length, true);
      if (new BigUint64Array(owned.buffer).some((v, i) => v !== a[i])) throw new Error('Query export borrowed owner memory');
      for (const range of [[job.length, 1], [0xffffffff, 2]]) for (const mode of [false, true]) {
        let rejected = false; try { job.queriesF64WithCopy(...range, mode); } catch { rejected = true; }
        if (!rejected) throw new Error('Invalid query range accepted');
      }
      for (const mode of [false, true]) if (job.queriesF64WithCopy(job.length, 0, mode).length) throw new Error('Empty range changed');
    } finally { job.free(); }
    const prewarm = await call({ kind: 'cpu-prepare' }), loaded = {};
    for (const role of ['reference', 'compared']) {
      loaded[role] = (await call({ kind: 'load', file: new File([files[role]], inputs[role].name), maxPoints: 2_000_000 })).value;
      const cloud = loaded[role];
      if (cloud.kind !== 'cloud' || cloud.keepEvery !== 1 || cloud.count !== cloud.filePoints || (inputs[role].points && cloud.count !== inputs[role].points)) throw new Error('Input points changed or thinned');
    }
    const req = { kind: 'c2c', reference: loaded.reference.id, compared: loaded.compared.id, signed: false };
    const oracle = (await call({ ...req, backend: 'cpu', cpuCache: false })).value;
    const expected = new Uint32Array(oracle.distances.buffer);
    function check(result,mode,reused){
      const bits=new Uint32Array(result.distances.buffer),c=result.compute;
      if(bits.length!==expected.length||bits.some((v,i)=>v!==expected[i])||!Object.keys(oracle.stats).every(k=>Object.is(oracle.stats[k],result.stats[k])))throw new Error('CPU bits or ordered statistics changed');
      if(c?.backend!=='webgpu'||c.fallback||c.verifyFallback||c.verifyWorkers!==threads||c.batchSize!==131072||c.workgroupSize!==64||c.mappedReference!==(mode==='mapped')||c.packedNodes!==(nodes==='packed')||c.compactStack||c.boxBounds||c.parentTraversal||c.inheritedBounds||c.querySliceCopy||c.queryOwnedCopy||c.verifyStagedQueries||c.verifyMainWorker||c.leafOnly||!c.verifyBounds||!c.verifySimd||!c.prefetch||!c.verifySnapshot||Object.hasOwn(c,'diagnostics')||Object.hasOwn(c,'preparationProbe'))throw new Error('Wrong ordinary production GPU path');
      if(c.nodeBytes!==c.nodeCount*(nodes==='packed'?8:20)||c.retainedNodeBytes!==c.nodeBytes||c.boxBoundsBytes||c.parentTraversalBytes)throw new Error('Unexpected reference node storage');
      if(reused===true&&(!c.referenceReused||!c.deviceReused||!c.verifyReferenceReused||!c.nodeRepresentationReused))throw new Error('Cached reference unexpectedly rebuilt');
      if(reused===false&&(c.referenceReused||c.verifyReferenceReused||c.nodeRepresentationReused||!c.deviceReused))throw new Error('Cold reference unexpectedly reused');
    }
    const operation=mode=>call({...req,reference:loaded.reference.id,backend:'webgpu',gpuVerifyWorkers:threads,gpuBatchSize:131072,gpuMappedReference:mode==='mapped',gpuPackedNodes:nodes==='packed',gpuDiagnostics:false,gpuInheritedBounds:false,gpuBoxBounds:false,gpuParentTraversal:false,gpuCompactStack:false,gpuQuerySlice:false,gpuQueryOwned:false});
    const compact=r=>({elapsedMillis:r.elapsedMillis,workerMillis:r.value.millis,compute:r.value.compute});
    const defaultRun=await call({...req,backend:'webgpu',gpuVerifyWorkers:threads,gpuBatchSize:131072,gpuPackedNodes:nodes==='packed'});
    check(defaultRun.value,defaultRun.value.compute.mappedReference?'mapped':'queued');
    const setup={},certifiedByMode={},warming=[],trials=[],cold=[];
    for(const mode of ['queued','mapped']){const r=await operation(mode);check(r.value,mode);certifiedByMode[mode]=r.value.compute.certified;setup[mode]=compact(r);}
    if(certifiedByMode.queued!==certifiedByMode.mapped||deviceRequests!==1)throw new Error('Setup changed certificates or ordinary device reuse');
    for(let i=0;i<warmups+repeats;i++){
      const order=i%2?['mapped','queued']:['queued','mapped'],results={};
      for(const mode of order){const r=await operation(mode);check(r.value,mode,true);if(r.value.compute.certified!==certifiedByMode[mode])throw new Error('Warm certificates changed');results[mode]=compact(r);}
      (i<warmups?warming:trials).push({order,results});
      await globalThis.benchmarkProgress(`${i<warmups?'warm-up':'trial'} ${i<warmups?i+1:i-warmups+1}: queued ${results.queued.elapsedMillis.toFixed(1)}, mapped ${results.mapped.elapsedMillis.toFixed(1)} ms`);
    }
    for(let pair=0;pair<coldPairs;pair++){
      const first=pair%2?(coldFirst==='queued'?'mapped':'queued'):coldFirst,order=first==='queued'?['queued','mapped']:['mapped','queued'],results={};
      for(const mode of order){
        await call({kind:'remove',id:loaded.reference.id});
        loaded.reference=(await call({kind:'load',file:new File([files.reference],inputs.reference.name),maxPoints:2_000_000})).value;
        if(loaded.reference.keepEvery!==1||loaded.reference.count!==loaded.reference.filePoints||(inputs.reference.points&&loaded.reference.count!==inputs.reference.points))throw new Error('Cold reference changed');
        const initial=await operation(mode);check(initial.value,mode,false);
        const cached=await operation(mode);check(cached.value,mode,true);
        if(initial.value.compute.certified!==certifiedByMode[mode]||cached.value.compute.certified!==certifiedByMode[mode])throw new Error('Cold certificates changed');
        results[mode]={cold:compact(initial),cached:compact(cached)};
      }
      cold.push({order,results});await globalThis.benchmarkProgress(`cold pair ${pair+1}: queued ${results.queued.cold.elapsedMillis.toFixed(1)}, mapped ${results.mapped.cold.elapsedMillis.toFixed(1)} ms`);
    }
    if(deviceRequests!==1)throw new Error('Ordinary device was recreated');
    for(const role of ['compared','reference'])await call({kind:'remove',id:loaded[role].id});
    return{inputPoints:{reference:loaded.reference.count,compared:loaded.compared.count},cappedWorkerHardwareConcurrency:threads+1,physicalGpuRequired:physical,physicalWorkerAdapterChecked:true,gpuTimestampRequests:0,deviceRequests,queryBitsSha256,copiedQueryBytes,queryBitsExact:true,queryRangeGuardsChecked:true,ownerSnapshotAndExportIndependenceChecked:true,distanceBitsExact:true,summaryFieldsExact:true,certificateCountsStablePerMode:true,sourceRemovalCompleted:true,displayedF32Sha256:await digest(oracle.distances),stats:oracle.stats,certifiedByMode,defaultMappedReference:defaultRun.value.compute.mappedReference,defaultValidation:compact(defaultRun),prewarmMillis:prewarm.elapsedMillis,setup,warming,trials,cold};
  }finally{worker.terminate();URL.revokeObjectURL(wrapperUrl);}
}
const report = { schemaVersion: 1, gpuMappedReferenceComparison: true, createdUtc: new Date().toISOString(), threads, warmups, repeats, gpu, coldFirst, coldPairs, nodes,
  purpose: gpu === 'physical' && inputManifest && warmups === 6 && repeats === 12 && coldPairs === 4 && nodes === 'flat' ? 'timing-comparison' : 'compatibility-smoke',
  inputs, inputManifest, inputManifestSha256: specBytes ? hash(specBytes) : undefined, build, buildInfoSha256: hash(buildBytes),
  sources: { [harnessSource]: harnessHash }, commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8', windowsHide: true }).trim(),
  operationScope:'Ordinary production C2C RPC with diagnostics OFF: query snapshot, GPU hints/readback, exact CPU workers, ordered statistics, f32 export and response transfer. Warm pairs reuse separate queued/mapped searches sharing one ordinary GPU device and exact verifier pool. Cold pairs reload each reference outside RPC, then include complete reference setup and an immediate cached follow-up. Parse/index and pool/device/pipeline initialization excluded.',
  note:'Primary comparison uses ordinary flat nodes. Explicit packed-node smokes are separate compatibility checks. Mapped mode initializes immutable reference points and selected nodes with exact byte-range copies and unmap; batch/readback and optional box/parent uploads retain their paths. Both modes retain the same selected node bytes per search. Separate search caches are bounded by two upload modes and two diagnostic settings per reference; this is not a measurement of total device memory. No GPU timestamp negotiation or adapter identity collection. Every cold/warm tail is retained. Timing does not isolate physical transfer time or explain browser/driver/GC causes. All earlier study budgets remain closed.'};
let browser;
try {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  browser = await chromium.launch({ channel: 'chromium', headless: true, args: gpu === 'software' ? ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'] : [] });
  report.browserVersion = browser.version();
  const page = await browser.newPage(); await page.exposeFunction('benchmarkProgress', s => console.log(s));
  await page.goto(`http://127.0.0.1:${server.address().port}/__query-export`);
  report.measurement = await page.evaluate(measure, { inputs, workerName, threads, warmups, repeats, physical: gpu === 'physical', coldFirst, coldPairs, nodes });
  verifyBuild();
  for (const input of Object.values(inputs)) if (hash(readFileSync(input.path)) !== input.sha256) throw new Error('Input changed');
  if (specFile && hash(readFileSync(specFile)) !== hash(specBytes) || hash(readFileSync(buildFile)) !== hash(buildBytes) || hash(readFileSync(resolve(root, harnessSource), 'utf8').replaceAll('\r\n', '\n')) !== harnessHash) throw new Error('Specification, build record or harness changed');
  report.filesUnchangedDuringMeasurement = true; report.status = 'passed';
} catch (error) { report.status = 'failed'; report.error = String(error); process.exitCode = 1; }
finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' }); console.log(JSON.stringify({ status: report.status, output, error: report.error })); }
