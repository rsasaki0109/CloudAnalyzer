// Same-binary comparison through the ordinary production distance Worker.
import { chromium } from '@playwright/test';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve, basename, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2), allowed = ['--output', '--build-info', '--reference', '--compared', '--input-manifest', '--threads', '--warmups', '--repeats', '--gpu', '--cold-first'];
for (let i = 0; i < args.length; i += 2) if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith('--') || args.indexOf(args[i]) !== i)
  throw new Error(`Invalid option: ${args[i]}`);
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = option('--output'), buildFile = option('--build-info'), reference = option('--reference'), compared = option('--compared');
if (!output || !buildFile || !reference || !compared || existsSync(output)) throw new Error('Supply a NEW --output, --build-info, --reference and --compared');
const threads = Number(option('--threads', '4')), warmups = Number(option('--warmups', '6')), repeats = Number(option('--repeats', '12')), gpu = option('--gpu', 'physical');
if (!Number.isInteger(threads) || threads < 2 || threads > 8 || !Number.isInteger(warmups) || warmups < 0 || !Number.isInteger(repeats) || repeats < 1 || !['physical', 'software'].includes(gpu))
  throw new Error('Invalid trial settings');
const coldFirst = option('--cold-first', 'local');
if (!['local', 'boxes'].includes(coldFirst)) throw new Error('Invalid cold order');
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
const harnessSource = 'web/scripts/benchmark-box-bounds.mjs', harnessHash = hash(readFileSync(resolve(root, harnessSource), 'utf8').replaceAll('\r\n', '\n'));
const workerName = readdirSync(resolve(dist, 'assets')).find(n => /^worker-.*\.js$/.test(n));
if (!workerName) throw new Error('No production worker');
const bundledWasm = Object.entries(build.artifacts).filter(([name]) => name.startsWith('web/dist/assets/') && name.endsWith('.wasm'));
if (bundledWasm.length !== 1 || bundledWasm[0][1] !== build.artifacts['web/src/wasm/ca_wasm_bg.wasm']) throw new Error('Diagnostic and production WASM differ');
const server = createServer((req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/__query-export') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>GPU subtree box comparison</title>'); return; }
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
async function measure({ inputs, workerName, threads, warmups, repeats, physical, coldFirst }) {
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
    function check(result, mode, warm, diagnostic = false) {
      const bits = new Uint32Array(result.distances.buffer), compute = result.compute;
      if (bits.length !== expected.length || bits.some((v, i) => v !== expected[i]) || !Object.keys(oracle.stats).every(k => Object.is(oracle.stats[k], result.stats[k]))) throw new Error('Distances or ordered statistics differ from CPU');
      if (compute?.backend !== 'webgpu' || compute.fallback || compute.verifyFallback || compute.verifyWorkers !== threads || compute.verifyStagedQueries ||
          compute.inheritedBounds !== false || compute.boxBounds !== (mode === 'boxes') || compute.querySliceCopy !== false || compute.queryOwnedCopy !== false || compute.leafOnly !== false || compute.workgroupSize !== 64 || compute.batchSize !== 131072 || !compute.verifyBounds || !compute.verifySimd || !compute.prefetch || !compute.verifySnapshot)
        throw new Error('Wrong production GPU traversal or copy path');
      if (mode === 'boxes' && (!(compute.boxBoundsBytes > 0) || compute.boxBoundsBytes > 16 * 1024 * 1024 || (warm && !compute.boxBoundsReused))) throw new Error('Box bounds missing or rebuilt');
      if (diagnostic) {
        const d = compute.diagnostics, enabled = adapter.features.has('timestamp-query');
        if (!d || d.timestampQuery !== (enabled ? 'enabled' : 'unsupported') || d.gpuBatchesOmitted || d.verificationBatchesOmitted || d.gpu.length !== compute.batches) throw new Error('Incomplete explanatory diagnostics');
        if (physical && !enabled) throw new Error('Explanatory physical samples require timestamps');
        if (d.gpu.reduce((n, b) => n + b.count, 0) !== expected.length || d.verification.reduce((n, b) => n + b.count, 0) !== expected.length) throw new Error('Diagnostic coverage changed');
        if (enabled && d.gpu.some(b => !Number.isFinite(b.shaderMillis) || b.shaderMillis < 0)) throw new Error('Invalid GPU timestamp duration');
      } else if (Object.hasOwn(compute, 'diagnostics')) throw new Error('Adoption trials must leave diagnostics off');
      if (warm && (!compute.referenceReused || !compute.deviceReused || !compute.verifyReferenceReused)) throw new Error('Timed reference unexpectedly rebuilt');
    }
    const operation = (mode, diagnostic = false) => call({ ...req, reference: loaded.reference.id, backend: 'webgpu', gpuVerifyWorkers: threads, gpuBatchSize: 131072,
      gpuInheritedBounds: false, gpuBoxBounds: mode === 'boxes', gpuDiagnostics: diagnostic, gpuQuerySlice: false, gpuQueryOwned: false });
    const compact = r => ({ elapsedMillis: r.elapsedMillis, workerMillis: r.value.millis, compute: r.value.compute });
    const defaultRun = await call({ ...req, backend: 'webgpu', gpuVerifyWorkers: threads, gpuBatchSize: 131072 });
    check(defaultRun.value, defaultRun.value.compute.boxBounds ? 'boxes' : 'local', false);
    const defaultBoxBounds = defaultRun.value.compute.boxBounds;
    const setup = {}, warming = [], trials = [], certifiedByMode = {};
    for (const mode of ['local', 'boxes']) { const r = await operation(mode); check(r.value, mode, false); certifiedByMode[mode] = r.value.compute.certified; setup[mode] = compact(r); }
    for (let i = 0; i < warmups + repeats; i++) {
      const order = i % 2 ? ['boxes', 'local'] : ['local', 'boxes'], results = {};
      for (const mode of order) { const r = await operation(mode); check(r.value, mode, true); if (r.value.compute.certified !== certifiedByMode[mode]) throw new Error('Certificate counts changed'); results[mode] = compact(r); }
      (i < warmups ? warming : trials).push({ order, results });
      await globalThis.benchmarkProgress(`${i < warmups ? 'warm-up' : 'trial'} ${i < warmups ? i + 1 : i - warmups + 1}: local ${results.local.elapsedMillis.toFixed(1)}, boxes ${results.boxes.elapsedMillis.toFixed(1)} ms`);
    }
    const coldOrder = coldFirst === 'local' ? ['local', 'boxes'] : ['boxes', 'local'], cold = {};
    for (const mode of coldOrder) {
      await call({ kind: 'remove', id: loaded.reference.id });
      loaded.reference = (await call({ kind: 'load', file: new File([files.reference], inputs.reference.name), maxPoints: 2_000_000 })).value;
      if (loaded.reference.keepEvery !== 1 || loaded.reference.count !== loaded.reference.filePoints || (inputs.reference.points && loaded.reference.count !== inputs.reference.points)) throw new Error('Cold reference changed');
      const r = await operation(mode); check(r.value, mode, false);
      const c = r.value.compute;
      if (c.referenceReused || c.verifyReferenceReused || !c.deviceReused || c.boxBoundsReused || c.certified !== certifiedByMode[mode]) throw new Error('Cold reference unexpectedly reused or changed');
      cold[mode] = compact(r);
    }
    const diagnosticWarming = [], diagnosticTrials = [];
    for (let i = 0; i < 4; i++) {
      const order = i % 2 ? ['boxes', 'local'] : ['local', 'boxes'], results = {};
      for (const mode of order) {
        const r = await operation(mode, true); check(r.value, mode, i > 0, true);
        if (r.value.compute.certified !== certifiedByMode[mode]) throw new Error('Diagnostic certificates changed');
        results[mode] = compact(r);
      }
      (i < 2 ? diagnosticWarming : diagnosticTrials).push({ order, results });
    }
    for (const role of ['compared', 'reference']) await call({ kind: 'remove', id: loaded[role].id });
    return { inputPoints: { reference: loaded.reference.count, compared: loaded.compared.count }, cappedWorkerHardwareConcurrency: threads + 1,
      physicalGpuRequired: physical, queryBitsSha256, copiedQueryBytes, queryBitsExact: true, queryRangeGuardsChecked: true,
      ownerSnapshotAndExportIndependenceChecked: true, distanceBitsExact: true, summaryFieldsExact: true, certificateCountsStablePerMode: true,
      sourceRemovalCompleted: true, displayedF32Sha256: await digest(oracle.distances), stats: oracle.stats, certifiedByMode,
      defaultBoxBounds, coldOrder, cold, diagnosticWarming, diagnosticTrials, defaultValidation: compact(defaultRun), prewarmMillis: prewarm.elapsedMillis, setup, warming, trials };
  } finally { worker.terminate(); URL.revokeObjectURL(wrapperUrl); }
}
const report = { schemaVersion: 1, gpuBoxBoundsComparison: true, createdUtc: new Date().toISOString(), threads, warmups, repeats, gpu, coldFirst,
  purpose: gpu === 'physical' && inputManifest && warmups >= 6 && repeats >= 12 ? 'timing-comparison' : 'compatibility-smoke',
  inputs, inputManifest, inputManifestSha256: specBytes ? hash(specBytes) : undefined, build, buildInfoSha256: hash(buildBytes),
  sources: { [harnessSource]: harnessHash }, commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8', windowsHide: true }).trim(),
  operationScope: 'Ordinary production loaded-cloud distance RPC: fresh query snapshot, physical GPU dispatch/readback, streaming exact CPU workers, ordered statistics, f32 export, asynchronous release submission and result transfer. Parsing, pool/device/reference/pipeline initialization and query-copy diagnostics excluded.',
  note: 'Primary adoption pairs have diagnostics OFF. The shader variants share ordinary point/node buffers and exact CPU caches. Boxes are lazily built from rounded tree-order points, uploaded once and retained per GPU reference; report cold activation, preparation and storage separately. Cold samples use fresh parsed/indexed references with warmed devices, pipelines and pool; only the C2C RPC is timed, and single samples are a conservative regression guard, not a cold-speedup estimate. Two diagnostic-on explanatory pairs follow two warm-up pairs and are excluded from the adoption gate. GPU pass timestamps and host timings overlap, use distinct clocks and may be quantized; no phase sums/subtractions isolate transfer time. Candidate hints/certificate counts may differ between shaders but must be stable within each control. Exact f64 verification stays authoritative. Real maps are unchanged measured subsets, not independent surveys or SLAM ground truth. All tails remain and no adapter identity is collected.' };
let browser;
try {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  browser = await chromium.launch({ channel: 'chromium', headless: true, args: gpu === 'software' ? ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'] : [] });
  report.browserVersion = browser.version();
  const page = await browser.newPage(); await page.exposeFunction('benchmarkProgress', s => console.log(s));
  await page.goto(`http://127.0.0.1:${server.address().port}/__query-export`);
  report.measurement = await page.evaluate(measure, { inputs, workerName, threads, warmups, repeats, physical: gpu === 'physical', coldFirst });
  verifyBuild();
  for (const input of Object.values(inputs)) if (hash(readFileSync(input.path)) !== input.sha256) throw new Error('Input changed');
  if (specFile && hash(readFileSync(specFile)) !== hash(specBytes) || hash(readFileSync(buildFile)) !== hash(buildBytes) || hash(readFileSync(resolve(root, harnessSource), 'utf8').replaceAll('\r\n', '\n')) !== harnessHash) throw new Error('Specification, build record or harness changed');
  report.filesUnchangedDuringMeasurement = true; report.status = 'passed';
} catch (error) { report.status = 'failed'; report.error = String(error); process.exitCode = 1; }
finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' }); console.log(JSON.stringify({ status: report.status, output, error: report.error })); }
