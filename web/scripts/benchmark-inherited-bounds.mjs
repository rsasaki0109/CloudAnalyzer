// Same-binary comparison through the ordinary production distance Worker.
import { chromium } from '@playwright/test';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve, basename, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2), allowed = ['--output', '--build-info', '--reference', '--compared', '--input-manifest', '--threads', '--warmups', '--repeats', '--gpu'];
for (let i = 0; i < args.length; i += 2) if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith('--') || args.indexOf(args[i]) !== i)
  throw new Error(`Invalid option: ${args[i]}`);
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = option('--output'), buildFile = option('--build-info'), reference = option('--reference'), compared = option('--compared');
if (!output || !buildFile || !reference || !compared || existsSync(output)) throw new Error('Supply a NEW --output, --build-info, --reference and --compared');
const threads = Number(option('--threads', '4')), warmups = Number(option('--warmups', '6')), repeats = Number(option('--repeats', '12')), gpu = option('--gpu', 'physical');
if (!Number.isInteger(threads) || threads < 2 || threads > 8 || !Number.isInteger(warmups) || warmups < 0 || !Number.isInteger(repeats) || repeats < 1 || !['physical', 'software'].includes(gpu))
  throw new Error('Invalid trial settings');
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
const harnessSource = 'web/scripts/benchmark-inherited-bounds.mjs', harnessHash = hash(readFileSync(resolve(root, harnessSource), 'utf8').replaceAll('\r\n', '\n'));
const workerName = readdirSync(resolve(dist, 'assets')).find(n => /^worker-.*\.js$/.test(n));
if (!workerName) throw new Error('No production worker');
const bundledWasm = Object.entries(build.artifacts).filter(([name]) => name.startsWith('web/dist/assets/') && name.endsWith('.wasm'));
if (bundledWasm.length !== 1 || bundledWasm[0][1] !== build.artifacts['web/src/wasm/ca_wasm_bg.wasm']) throw new Error('Diagnostic and production WASM differ');
const server = createServer((req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/__query-export') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>Inherited GPU bound comparison</title>'); return; }
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
async function measure({ inputs, workerName, threads, warmups, repeats, physical }) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || physical === adapter.info.isFallbackAdapter) throw new Error('Requested GPU adapter class unavailable');
  // This measurement-only wrapper caps the pool before importing production code.
  const workerUrl = new URL(`/assets/${workerName}`, location.href).href;
  const wrapper = `Object.defineProperty(navigator, 'hardwareConcurrency', {value:${threads + 1}});\nconst queued=[];self.onmessage=e=>queued.push(e);await import(${JSON.stringify(workerUrl)});const handle=self.onmessage;for(const event of queued)handle(event);`;
  const wrapperUrl = URL.createObjectURL(new Blob([wrapper], { type: 'text/javascript' }));
  const worker = new Worker(wrapperUrl, { type: 'module' }), pending = new Map(); let seq = 0;
  const fail = error => { for (const p of pending.values()) { clearTimeout(p.timer); p.reject(error); } pending.clear(); };
  worker.onerror = e => fail(new Error(e.message || 'Worker error'));
  worker.onmessageerror = () => fail(new Error('Worker message error'));
  worker.onmessage = ({ data }) => {
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
    function check(result, mode, warm) {
      const bits = new Uint32Array(result.distances.buffer), compute = result.compute;
      if (bits.length !== expected.length || bits.some((v, i) => v !== expected[i]) || !Object.keys(oracle.stats).every(k => Object.is(oracle.stats[k], result.stats[k]))) throw new Error('Distances or ordered statistics differ from CPU');
      if (compute?.backend !== 'webgpu' || compute.fallback || compute.verifyFallback || compute.verifyWorkers !== threads || compute.verifyStagedQueries ||
          compute.inheritedBounds !== (mode === 'inherited') || compute.querySliceCopy !== false || compute.queryOwnedCopy !== false || compute.leafOnly !== false || compute.workgroupSize !== 64 || compute.batchSize !== 131072 || !compute.verifyBounds || !compute.verifySimd || !compute.prefetch || !compute.verifySnapshot)
        throw new Error('Wrong production GPU traversal or copy path');
      if (warm && (!compute.referenceReused || !compute.deviceReused || !compute.verifyReferenceReused)) throw new Error('Timed reference unexpectedly rebuilt');
    }
    const operation = mode => call({ ...req, backend: 'webgpu', gpuVerifyWorkers: threads, gpuBatchSize: 131072,
      gpuInheritedBounds: mode === 'inherited', gpuQuerySlice: false, gpuQueryOwned: false });
    const compact = r => ({ elapsedMillis: r.elapsedMillis, workerMillis: r.value.millis, compute: r.value.compute });
    const defaultRun = await call({ ...req, backend: 'webgpu', gpuVerifyWorkers: threads, gpuBatchSize: 131072 });
    check(defaultRun.value, defaultRun.value.compute.inheritedBounds ? 'inherited' : 'local', false);
    const defaultInheritedBounds = defaultRun.value.compute.inheritedBounds;
    const setup = {}, warming = [], trials = [], certifiedByMode = {};
    for (const mode of ['local', 'inherited']) { const r = await operation(mode); check(r.value, mode, false); certifiedByMode[mode] = r.value.compute.certified; setup[mode] = compact(r); }
    for (let i = 0; i < warmups + repeats; i++) {
      const order = i % 2 ? ['inherited', 'local'] : ['local', 'inherited'], results = {};
      for (const mode of order) { const r = await operation(mode); check(r.value, mode, true); if (r.value.compute.certified !== certifiedByMode[mode]) throw new Error('Certificate counts changed'); results[mode] = compact(r); }
      (i < warmups ? warming : trials).push({ order, results });
      await globalThis.benchmarkProgress(`${i < warmups ? 'warm-up' : 'trial'} ${i < warmups ? i + 1 : i - warmups + 1}: local ${results.local.elapsedMillis.toFixed(1)}, inherited ${results.inherited.elapsedMillis.toFixed(1)} ms`);
    }
    for (const role of ['compared', 'reference']) await call({ kind: 'remove', id: loaded[role].id });
    return { inputPoints: { reference: loaded.reference.count, compared: loaded.compared.count }, cappedWorkerHardwareConcurrency: threads + 1,
      physicalGpuRequired: physical, queryBitsSha256, copiedQueryBytes, queryBitsExact: true, queryRangeGuardsChecked: true,
      ownerSnapshotAndExportIndependenceChecked: true, distanceBitsExact: true, summaryFieldsExact: true, certificateCountsStablePerMode: true,
      sourceRemovalCompleted: true, displayedF32Sha256: await digest(oracle.distances), stats: oracle.stats, certifiedByMode,
      defaultInheritedBounds, defaultValidation: compact(defaultRun), prewarmMillis: prewarm.elapsedMillis, setup, warming, trials };
  } finally { worker.terminate(); URL.revokeObjectURL(wrapperUrl); }
}
const report = { schemaVersion: 1, gpuInheritedBoundsComparison: true, createdUtc: new Date().toISOString(), threads, warmups, repeats, gpu,
  purpose: gpu === 'physical' && inputManifest && warmups >= 6 && repeats >= 12 ? 'timing-comparison' : 'compatibility-smoke',
  inputs, inputManifest, inputManifestSha256: specBytes ? hash(specBytes) : undefined, build, buildInfoSha256: hash(buildBytes),
  sources: { [harnessSource]: harnessHash }, commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8', windowsHide: true }).trim(),
  operationScope: 'Ordinary production loaded-cloud distance RPC: fresh query snapshot, physical GPU dispatch/readback, streaming exact CPU workers, ordered statistics, f32 export, asynchronous release submission and result transfer. Parsing, pool/device/reference/pipeline initialization and query-copy diagnostics excluded.',
  note: 'Both shader controls run in the same compiled production JS/WASM and cached production worker pool. Both inherited-bound settings are explicit; coordinate-copy controls are fixed false; candidate hints and certificate counts may differ across shader controls but must remain stable within each control; a separate untimed request records and checks the application default. Pool capacity is capped in a measurement-only worker wrapper before production module import. Query release uses normal asynchronous submission, not shared-experiment acknowledgements. CPU/GPU clocks, host activity, GC and power are uncontrolled. Host GPU wait is not shader execution time. Real inputs are unchanged point subsets, not independent surveys or SLAM ground truth. No adapter identity is collected.' };
let browser;
try {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  browser = await chromium.launch({ channel: 'chromium', headless: true, args: gpu === 'software' ? ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'] : [] });
  report.browserVersion = browser.version();
  const page = await browser.newPage(); await page.exposeFunction('benchmarkProgress', s => console.log(s));
  await page.goto(`http://127.0.0.1:${server.address().port}/__query-export`);
  report.measurement = await page.evaluate(measure, { inputs, workerName, threads, warmups, repeats, physical: gpu === 'physical' });
  verifyBuild();
  for (const input of Object.values(inputs)) if (hash(readFileSync(input.path)) !== input.sha256) throw new Error('Input changed');
  if (specFile && hash(readFileSync(specFile)) !== hash(specBytes) || hash(readFileSync(buildFile)) !== hash(buildBytes) || hash(readFileSync(resolve(root, harnessSource), 'utf8').replaceAll('\r\n', '\n')) !== harnessHash) throw new Error('Specification, build record or harness changed');
  report.filesUnchangedDuringMeasurement = true; report.status = 'passed';
} catch (error) { report.status = 'failed'; report.error = String(error); process.exitCode = 1; }
finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' }); console.log(JSON.stringify({ status: report.status, output, error: report.error })); }
