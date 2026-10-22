// Same-binary comparison through the ordinary production distance Worker.
import { chromium } from '@playwright/test';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve, basename, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2), allowed = ['--output', '--build-info', '--reference', '--compared', '--input-manifest', '--threads', '--gpu', '--row', '--cycles'];
for (let i = 0; i < args.length; i += 2) if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith('--') || args.indexOf(args[i]) !== i)
  throw new Error(`Invalid option: ${args[i]}`);
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = option('--output'), buildFile = option('--build-info'), reference = option('--reference'), compared = option('--compared');
if (!output || !buildFile || !reference || !compared || existsSync(output)) throw new Error('Supply a NEW --output, --build-info, --reference and --compared');
const threads=Number(option('--threads','4')),cycles=Number(option('--cycles','2')),row=Number(option('--row','0')),gpu=option('--gpu','physical');
if(!Number.isInteger(threads)||threads<2||threads>8||![1,2].includes(cycles)||!Number.isInteger(row)||row<0||row>3||!['physical','software'].includes(gpu))throw new Error('Invalid reference preparation settings');
const root = fileURLToPath(new URL('../../', import.meta.url));
const hash = data => createHash('sha256').update(data).digest('hex');
const buildBytes = readFileSync(buildFile), build = JSON.parse(buildBytes), dist = resolve(root, build.outputDirectory);
if(build.purpose!=="measurement-only-host-phase-build")throw new Error("A reference preparation profile build is required");
function verifyBuild() {
  for (const [name, expected] of Object.entries(build.normalizedInputs)) if (hash(readFileSync(resolve(root, name), 'utf8').replaceAll('\r\n', '\n')) !== expected) throw new Error(`Source changed: ${name}`);
  for(const [name,expected] of Object.entries(build.wasmReuse.unchangedArtifacts))if(hash(readFileSync(resolve(root,name)))!==expected)throw new Error(`WASM/glue changed: ${name}`);
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
const harnessSource = 'web/scripts/benchmark-reference-preparation.mjs', harnessHash = hash(readFileSync(resolve(root, harnessSource), 'utf8').replaceAll('\r\n', '\n'));
const workerName = readdirSync(resolve(dist, 'assets')).find(n => /^worker-.*\.js$/.test(n));
if (!workerName) throw new Error('No production worker');
const bundledWasm = Object.entries(build.artifacts).filter(([name]) => name.startsWith(build.outputDirectory+'/assets/') && name.endsWith('.wasm'));
if (bundledWasm.length !== 1 || bundledWasm[0][1] !== build.wasmReuse.unchangedArtifacts['web/src/wasm/ca_wasm_bg.wasm']) throw new Error('Diagnostic and production WASM differ');
const server = createServer((req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/__query-export') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>GPU reference preparation profile</title>'); return; }
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
async function measure({ inputs, workerName, threads, physical, row, cycles }) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || physical === adapter.info.isFallbackAdapter) throw new Error('Requested GPU adapter class unavailable');
  // This measurement-only wrapper caps the pool before importing production code.
  const workerUrl = new URL(`/assets/${workerName}`, location.href).href;
  const wrapper = `Object.defineProperty(navigator,'hardwareConcurrency',{value:${threads+1}});
    const requestDevice=GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice=function(...args){
      if(${physical}===this.info.isFallbackAdapter)throw new Error('Wrong Worker GPU adapter class');
      const timestampRequested=args[0]?.requiredFeatures?.includes('timestamp-query')??false;
      if(timestampRequested)throw new Error('Preparation profile must not request timestamps');
      self.postMessage({prepDeviceRequested:true});return requestDevice.apply(this,args);
    };
    const queued=[];self.onmessage=e=>queued.push(e);await import(${JSON.stringify(workerUrl)});const handle=self.onmessage;
    self.onmessage=e=>{const req=e.data.req;globalThis.__caPreparationTrace=req?.kind==='c2c'&&req.preparationProbe?{events:[],omitted:0,nextScope:0}:undefined;return handle.call(self,e);};
    for(const event of queued)self.onmessage(event);`;
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
    function check(result, arm, expectedReuse) {
      const bits=new Uint32Array(result.distances.buffer),c=result.compute,packed=arm.startsWith('packed'),trace=arm.endsWith('-on');
      if(bits.length!==expected.length||bits.some((v,i)=>v!==expected[i])||!Object.keys(oracle.stats).every(k=>Object.is(oracle.stats[k],result.stats[k])))throw new Error('CPU bits or ordered statistics changed');
      if(c?.backend!=='webgpu'||c.fallback||c.verifyFallback||c.verifyWorkers!==threads||c.batchSize!==131072||c.workgroupSize!==64||c.packedNodes!==packed||c.compactStack||c.boxBounds||c.parentTraversal||c.inheritedBounds||c.boxBoundsBytes||c.parentTraversalBytes||c.querySliceCopy||c.queryOwnedCopy||c.verifyStagedQueries||c.verifyMainWorker||c.leafOnly||!c.verifyBounds||!c.verifySimd||!c.prefetch||!c.verifySnapshot||Object.hasOwn(c,'diagnostics'))throw new Error('Wrong ordinary production path');
      if(c.nodeBytes!==c.nodeCount*(packed?8:20)||c.retainedNodeBytes<c.nodeBytes)throw new Error('Node bytes changed');
      if(expectedReuse!==undefined&&[c.referenceReused,c.verifyReferenceReused,c.nodeRepresentationReused].some(v=>v!==expectedReuse))throw new Error('Unexpected reference cache state');
      if(expectedReuse!==undefined&&(!c.deviceReused||c.retainedNodeBytes!==c.nodeBytes))throw new Error('Cold/warm profile duplicated reference nodes or device');
      const p=c.preparationProbe;
      if(!trace){if(p!==undefined)throw new Error('Disabled phase logging returned records');return;}
      if(!p||p.omitted||p.events.length<4||p.events.length>256||p.events.some((e,i)=>!Number.isFinite(e.at)||(i&&e.at<p.events[i-1].at))||p.events[0].phase!=='c2c-start'||p.events.at(-1).phase!=='c2c-return')throw new Error('Invalid or incomplete bounded phase events');
      const names=p.events.map(e=>e.phase);
      if(expectedReuse===false)for(const phase of ['tree-start','tree-end','query-snapshot-start','query-snapshot-end','verifier-snapshot-start','verifier-snapshot-end','candidate-export-start','candidate-export-end','points-copy-start','points-copy-end','nodes-copy-start','nodes-copy-end','reference-upload-queued','reference-upload-enter','points-allocate-start','points-allocate-end','points-write-start','points-write-end','node-encode-start','node-encode-end','nodes-allocate-start','nodes-allocate-end','nodes-write-start','nodes-write-end','reference-upload-work-end','reference-upload-ready','node-readiness-start','node-readiness-end','first-dispatch','verifiers-ready'])if(names.filter(n=>n===phase).length!==1)throw new Error('Missing/duplicate cold phase '+phase);
      if(expectedReuse===true&&names.some(n=>['tree-start','candidate-export-start','reference-upload-queued','node-encode-start','points-allocate-start'].includes(n)))throw new Error('Warm reference unexpectedly prepared again');
      for(const event of p.events.filter(e=>e.phase==='device-scope-enter')){
        const pops=p.events.filter(e=>e.id===event.id&&e.phase==='scope-pop-end'),starts=p.events.filter(e=>e.id===event.id&&e.phase==='scope-pop-start');
        if(pops.length!==1||starts.length!==1||event.at>starts[0].at||starts[0].at>pops[0].at)throw new Error('Unclosed or reordered device scope');
      }
    }
    const operation=arm=>call({...req,reference:loaded.reference.id,backend:'webgpu',gpuVerifyWorkers:threads,gpuBatchSize:131072,gpuInheritedBounds:false,gpuBoxBounds:false,gpuParentTraversal:false,gpuCompactStack:false,gpuPackedNodes:arm.startsWith('packed'),gpuDiagnostics:false,gpuQuerySlice:false,gpuQueryOwned:false,preparationProbe:arm.endsWith('-on')});
    const compact=r=>({elapsedMillis:r.elapsedMillis,workerMillis:r.value.millis,compute:r.value.compute});
    const setup={},certifiedByMode={};
    for(const mode of ['flat','packed']){const arm=mode+'-off',r=await operation(arm);check(r.value,arm);certifiedByMode[mode]=r.value.compute.certified;setup[mode]=compact(r);}
    if(deviceRequests!==1)throw new Error('Expected one ordinary GPU device');
    const rows=[['flat-off','flat-on','packed-on','packed-off'],['flat-on','packed-off','flat-off','packed-on'],['packed-off','packed-on','flat-on','flat-off'],['packed-on','flat-off','packed-off','flat-on']],trials=[];
    for(let cycle=0;cycle<cycles;cycle++){
      const order=cycle?rows[row].toReversed():rows[row],samples=[];
      for(const arm of order){
        await call({kind:'remove',id:loaded.reference.id});
        loaded.reference=(await call({kind:'load',file:new File([files.reference],inputs.reference.name),maxPoints:2_000_000})).value;
        if(loaded.reference.keepEvery!==1||loaded.reference.count!==loaded.reference.filePoints||(inputs.reference.points&&loaded.reference.count!==inputs.reference.points))throw new Error('Fresh reference changed');
        const cold=await operation(arm);check(cold.value,arm,false);
        const warm=await operation(arm);check(warm.value,arm,true);
        const mode=arm.split('-')[0];if(cold.value.compute.certified!==certifiedByMode[mode]||warm.value.compute.certified!==certifiedByMode[mode])throw new Error('Certificate count changed');
        samples.push({arm,cold:compact(cold),warm:compact(warm)});
        await globalThis.benchmarkProgress(`cycle ${cycle+1} ${arm}: cold ${cold.elapsedMillis.toFixed(1)}, cached ${warm.elapsedMillis.toFixed(1)} ms`);
      }
      trials.push({order,samples});
    }
    if(deviceRequests!==1)throw new Error('GPU device was recreated');
    for(const role of ['compared','reference'])await call({kind:'remove',id:loaded[role].id});
    return {inputPoints:{reference:loaded.reference.count,compared:loaded.compared.count},cappedWorkerHardwareConcurrency:threads+1,physicalGpuRequired:physical,physicalWorkerAdapterChecked:true,gpuTimestampRequests:0,deviceRequests,queryBitsSha256,copiedQueryBytes,queryBitsExact:true,queryRangeGuardsChecked:true,ownerSnapshotAndExportIndependenceChecked:true,distanceBitsExact:true,summaryFieldsExact:true,certificateCountsStablePerMode:true,sourceRemovalCompleted:true,displayedF32Sha256:await digest(oracle.distances),stats:oracle.stats,certifiedByMode,setup,trials};
  } finally {worker.terminate();URL.revokeObjectURL(wrapperUrl);}
}
const report={schemaVersion:1,gpuReferencePreparationProfile:true,createdUtc:new Date().toISOString(),threads,cycles,row,gpu,
 purpose:gpu==='physical'&&inputManifest&&cycles===2&&threads===4?'explanatory-reference-preparation-profile':'compatibility-smoke',
 inputs,inputManifest,inputManifestSha256:specBytes?hash(specBytes):undefined,build,buildInfoSha256:hash(buildBytes),sources:{[harnessSource]:harnessHash},commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8',windowsHide:true}).trim(),
 operationScope:'Fresh-reference ordinary production C2C RPC and immediate cached follow-up, with CPU pool and GPU device/pipelines prewarmed. Parse/index excluded. Four balanced format/logging arms share one measurement-only binary. No GPU timestamps.',
 note:'Host phase events use one Worker clock and are bounded at 256. Allocation/write spans are synchronous host calls. Error-scope completion spans include browser scheduling and possible driver/GC/other waiting; they are not isolated physical transfer durations. Phase spans overlap and must not be added or subtracted across page/worker clocks. Off arms skip phase clock reads but still execute instrumentation guards, so this does not quantify overhead against a byte-identical uninstrumented production binary. No optimization/adoption conclusion or backend/default change. Every tail retained. Real input subsets unchanged; no adapter identity collected.'};
let browser;
try{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 browser=await chromium.launch({channel:'chromium',headless:true,args:gpu==='software'?['--enable-unsafe-webgpu','--use-webgpu-adapter=swiftshader','--enable-unsafe-swiftshader']:[]});report.browserVersion=browser.version();
 const page=await browser.newPage();await page.exposeFunction('benchmarkProgress',s=>console.log(s));await page.goto(`http://127.0.0.1:${server.address().port}/__query-export`);
 report.measurement=await page.evaluate(measure,{inputs,workerName,threads,physical:gpu==='physical',row,cycles});
 verifyBuild();for(const input of Object.values(inputs))if(hash(readFileSync(input.path))!==input.sha256)throw new Error('Input changed');
 if(specFile&&hash(readFileSync(specFile))!==hash(specBytes)||hash(readFileSync(buildFile))!==hash(buildBytes)||hash(readFileSync(resolve(root,harnessSource),'utf8').replaceAll('\r\n','\n'))!==harnessHash)throw new Error('Specification, build or harness changed');
 report.filesUnchangedDuringMeasurement=true;report.status='passed';
}catch(error){report.status='failed';report.error=String(error);process.exitCode=1;}
finally{await browser?.close();await new Promise(resolve=>server.close(resolve));writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({status:report.status,output,error:report.error}));}
