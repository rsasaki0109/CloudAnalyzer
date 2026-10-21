// Fixed real WebGPU hints and complete prefetched GPU pipeline comparisons.
import { chromium } from '@playwright/test';
import { verifierTrialOrder } from '../experiments/shared-gpu-c2c/trial-order.mjs';
import { createServer } from 'node:http';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { resolve, extname, sep, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus, platform, release, totalmem } from 'node:os';

const args = process.argv.slice(2);
const allowed = ['--output', '--reference', '--compared', '--input-manifest', '--threads', '--warmups', '--repeats', '--gpu', '--batch-size', '--jobs'];
for (let i = 0; i < args.length; i += 2) if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith('--') || args.indexOf(args[i]) !== i)
  throw new Error(`Invalid or repeated option: ${args[i]}`);
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = option('--output'), reference = option('--reference'), compared = option('--compared');
if (!output || !reference || !compared) throw new Error('Supply --output NEW.json --reference FILE --compared FILE [--threads 4] [--gpu physical|disabled] [--jobs prepared|fresh]');
if (existsSync(output)) throw new Error('Report already exists');
const threads = Number(option('--threads', '4')), warmups = Number(option('--warmups', '5')), repeats = Number(option('--repeats', '10'));
const batchSize = Number(option('--batch-size', '131072')), gpu = option('--gpu', 'physical'), jobs = option('--jobs', 'prepared');
if (!Number.isInteger(threads) || threads < 1 || threads > 8 || !Number.isInteger(warmups) || warmups < 0 ||
    !Number.isInteger(repeats) || repeats < 1 || !Number.isInteger(batchSize) || batchSize < 1 || batchSize > 262_144 ||
    !['physical', 'disabled'].includes(gpu) || !['prepared', 'fresh'].includes(jobs)) throw new Error('Invalid trial options');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const repository = fileURLToPath(new URL('../../', import.meta.url));
const experiment = resolve(repository, 'web/experiments/shared-gpu-c2c');
const inputs = Object.fromEntries([['reference', reference], ['compared', compared]].map(([role, path]) => {
  const bytes = readFileSync(path);
  return [role, { path: resolve(path), name: basename(path), bytes: bytes.length, sha256: hash(bytes) }];
}));
const manifestPath = option('--input-manifest'), manifestBytes = manifestPath ? readFileSync(manifestPath) : undefined;
const inputManifest = manifestBytes ? JSON.parse(manifestBytes) : undefined;
for (const [role, input] of Object.entries(inputs)) if (inputManifest) {
  if (inputManifest.inputs?.[role]?.sha256 !== input.sha256 || inputManifest.inputs[role].bytes !== input.bytes) throw new Error(`Manifest mismatch: ${role}`);
  input.points = inputManifest.inputs[role].points;
}
function verifyBuild(mode, metadata) {
  const directory = resolve(experiment, `pkg-${mode}`);
  for (const [name, expected] of Object.entries(metadata.normalizedInputs)) {
    const bytes = name.endsWith('.wasm') ? readFileSync(resolve(repository, name)) : readFileSync(resolve(repository, name), 'utf8').replaceAll('\r\n', '\n');
    if (hash(bytes) !== expected) throw new Error(`Source changed since build: ${name}`);
  }
  for (const [name, expected] of Object.entries(metadata.artifacts)) if (hash(readFileSync(resolve(directory, name))) !== expected) throw new Error(`Build changed: ${mode}/${name}`);
}
const builds = Object.fromEntries(['serial', 'threads', 'driver'].map(mode => {
  const metadata = JSON.parse(readFileSync(resolve(experiment, `pkg-${mode}/build-info.json`), 'utf8'));
  verifyBuild(mode, metadata);
  return [mode, metadata];
}));
const prefix = '/CloudAnalyzer/app/shared-gpu-c2c/';
const server = createServer((request, response) => {
  const pathname = new URL(request.url, 'http://localhost').pathname;
  response.setHeader('Cache-Control', 'no-store');
  if (pathname === prefix) { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>Shared GPU C2C experiment</title>'); return; }
  for (const [role, input] of Object.entries(inputs)) if (pathname === `${prefix}${role}`) {
    const bytes = readFileSync(input.path);
    if (hash(bytes) !== input.sha256) { response.writeHead(409).end('Input changed'); return; }
    response.setHeader('Content-Type', 'application/octet-stream'); response.end(bytes); return;
  }
  if (!pathname.startsWith(prefix)) { response.writeHead(404).end(); return; }
  const path = resolve(experiment, pathname.slice(prefix.length));
  if (!path.startsWith(experiment + sep)) { response.writeHead(403).end(); return; }
  try { response.setHeader('Content-Type', ({ '.js':'text/javascript', '.mjs':'text/javascript', '.wasm':'application/wasm' })[extname(path)] || 'application/octet-stream'); response.end(readFileSync(path)); }
  catch { response.writeHead(404).end(); }
});

async function inPage({ isolated, inputs, threads, warmups, repeats, batchSize, gpu, jobs, trialOrders }) {
  function rpc(url) {
    const worker = new Worker(url, { type: 'module' }); let seq = 0;
    const pending = new Map();
    const fail = error => { for (const waiting of pending.values()) { clearTimeout(waiting.timer); waiting.reject(error); } pending.clear(); };
    worker.onmessage = ({ data }) => {
      const waiting = pending.get(data.seq); if (!waiting) return;
      clearTimeout(waiting.timer); pending.delete(data.seq);
      if (data.ok) waiting.resolve({ value: data.value, elapsedMillis: performance.now() - waiting.started });
      else waiting.reject(new Error(data.error));
    };
    worker.onerror = event => fail(new Error(event.message || 'Worker failed'));
    worker.onmessageerror = () => fail(new Error('Worker message failed'));
    return {
      call: (req, transfer = []) => new Promise((resolve, reject) => {
        const id = ++seq, started = performance.now();
        const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${req.kind}`)); }, 60_000);
        pending.set(id, { resolve, reject, timer, started });
        try { worker.postMessage({ seq: id, req }, { transfer }); } catch (error) { clearTimeout(timer); pending.delete(id); reject(error); }
      }),
      close: () => { fail(new Error('Worker closed')); worker.terminate(); },
    };
  }
  const digest = async values => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice()))).map(v => v.toString(16).padStart(2, '0')).join('');
  const reject = async (operation, message) => { try { await operation(); } catch (error) { if (String(error).includes(message)) return; throw error; } throw new Error(`Expected rejection: ${message}`); };
  const stable = rpc('./worker.mjs');
  if (!isolated) {
    try {
      await reject(() => stable.call({ kind: 'init', build: 'threads', threads }), 'isolation');
      const init = (await stable.call({ kind: 'init', build: 'serial', threads: 1 })).value;
      if (init.sharedMemory || init.threads) throw new Error('Serial fallback initialized shared threads');
      return { rejectedUnisolatedThreads: true, serialFallbackInitialized: true };
    } finally { stable.close(); }
  }
  const driver = rpc('./pkg-driver/driver.js');
  try {
    const initialization = await driver.call({ kind: 'init', threads, gpu: gpu === 'physical', sharedWorkerUrl: new URL('./worker.mjs', location.href).href });
    const files = {};
    for (const role of ['reference', 'compared']) {
      const response = await fetch(`./${role}`); if (!response.ok) throw new Error('Input fetch failed');
      files[role] = { name: inputs[role].name, bytes: new Uint8Array(await response.arrayBuffer()) };
    }
    const preparation = await driver.call({ kind: 'prepare', ...files, batchSize, retainQueries: jobs === 'fresh' }, Object.values(files).map(f => f.bytes.buffer));
    if (inputs.reference.points && inputs.reference.points !== preparation.value.referenceCount ||
        inputs.compared.points && inputs.compared.points !== preparation.value.queryCount) throw new Error('Input point count changed');
    if (!preparation.value.poolReferenceReused || initialization.value.effectivePoolSize !== threads) throw new Error('Expected pool capacity and cached references');
    if (gpu === 'physical' && preparation.value.candidateSource !== 'physical-webgpu') throw new Error('Physical GPU was not used');
    const expected = (await driver.call({ kind: 'expected' })).value;
    const exactExpected = new BigUint64Array(expected.exact.buffer), displayExpected = new Uint32Array(expected.distances.buffer);
    function parity(value, exact = false) {
      const displayed = new Uint32Array(value.distances.buffer, value.distances.byteOffset, value.distances.length);
      if (displayed.length !== displayExpected.length || displayed.some((v, i) => v !== displayExpected[i]) ||
          !Object.keys(expected.stats).every(key => Object.is(expected.stats[key], value.stats[key])) || value.certified !== expected.certified)
        throw new Error('Displayed distance, summary or certificate count changed');
      if (exact) {
        const values = new BigUint64Array(value.exact.buffer, value.exact.byteOffset, value.exact.length);
        if (values.length !== exactExpected.length || values.some((v, i) => v !== exactExpected[i])) throw new Error('Authoritative f64 bits changed');
      }
    }
    await stable.call({ kind: 'init', build: 'serial', threads: 1 });
    const snapshot = (await driver.call({ kind: 'snapshot' })).value;
    await stable.call({ kind: 'prepare', ...snapshot }, [snapshot.snapshot.buffer, snapshot.queries.buffer]);
    const stableExact = new Float64Array(expected.exact.length); let stableCertified = 0;
    for (let first = 0; first < stableExact.length; first += preparation.value.batchSize) {
      const candidates = expected.hints.slice(first, first + preparation.value.batchSize);
      const result = (await stable.call({ kind: 'verify', first, candidates, parallel: false, bounds: true, simd: true }, [candidates.buffer])).value;
      stableExact.set(result.distances, first); stableCertified += result.certified;
    }
    if (new BigUint64Array(stableExact.buffer).some((v, i) => v !== exactExpected[i]) || stableCertified !== expected.certified) throw new Error('Stable serial verifier bits or certificates changed');
    await reject(() => stable.call({ kind: 'verify', first: 0, candidates: new Uint32Array([0]), parallel: true, bounds: true, simd: true }), 'threads build');
    const guards = (await driver.call({ kind: 'guards' })).value;
    parity((await driver.call({ kind: 'run', mode: 'shared', pipeline: 'fixed', exact: true })).value, true);
    guards.failedReplacementPreservedSnapshot = true;
    const freshSetup = jobs === 'fresh' ? (await driver.call({ kind: 'begin-fresh-jobs' })).value : undefined;
    let lastJobSequence = 0, lastSharedQueryGeneration = 0;
    const validateLifecycle = value => {
      if (jobs !== 'fresh') return;
      if (!value.freshJob || !value.queryReleaseAcknowledged || !Number.isInteger(value.jobSequence) || value.jobSequence <= lastJobSequence)
        throw new Error('Fresh query ownership or acknowledged release was not established');
      lastJobSequence = value.jobSequence;
      if (value.sharedReferenceGeneration !== undefined) {
        if (value.sharedReferenceGeneration !== 1 || value.sharedQueryGeneration <= lastSharedQueryGeneration)
          throw new Error('Shared query reused an old job or rebuilt the cached reference');
        lastSharedQueryGeneration = value.sharedQueryGeneration;
      }
    };
    const modes = ['pool', 'pool-staged', 'shared-serial', 'shared'], pipelines = gpu === 'physical' ? ['fixed', 'gpu'] : ['fixed'];
    const compact = result => ({ elapsedMillis: result.elapsedMillis, workerMillis: result.value.millis,
      verifyMillis: result.value.verifyMillis, gpuWaitMillis: result.value.gpuWaitMillis,
      poolReferenceReused: result.value.poolReferenceReused, mainCommittedBytes: result.value.mainCommittedBytes,
      poolCommittedBytes: result.value.poolCommittedBytes, sharedCommittedBytes: result.value.sharedCommittedBytes,
      freshJob: result.value.freshJob, jobSequence: result.value.jobSequence, queryJobMillis: result.value.queryJobMillis,
      queryStageMillis: result.value.queryStageMillis, queryReleaseMillis: result.value.queryReleaseMillis,
      queryReleaseAcknowledged: result.value.queryReleaseAcknowledged, stagedQueryBytes: result.value.stagedQueryBytes,
      sharedReferenceGeneration: result.value.sharedReferenceGeneration, sharedQueryGeneration: result.value.sharedQueryGeneration });
    const measurements = {};
    for (const pipeline of pipelines) {
      const cold = {}, warming = [], trials = [];
      for (const mode of modes) {
        const result = await driver.call({ kind: 'run', mode, pipeline, exact: true }); parity(result.value, true); validateLifecycle(result.value); cold[mode] = compact(result);
      }
      for (let i = 0; i < warmups + repeats; i++) {
        const order = trialOrders[i], results = {};
        for (const mode of order) {
          const result = await driver.call({ kind: 'run', mode, pipeline }); parity(result.value); validateLifecycle(result.value);
          if (!result.value.poolReferenceReused) throw new Error('Verifier reference unexpectedly rebuilt');
          results[mode] = compact(result);
        }
        (i < warmups ? warming : trials).push({ order, results });
        await globalThis.benchmarkProgress(`${pipeline} ${i < warmups ? 'warm-up' : 'trial'} ${i < warmups ? i + 1 : i - warmups + 1}: ` + modes.map(mode => `${mode} ${results[mode].elapsedMillis.toFixed(1)} ms`).join(', '));
      }
      measurements[pipeline] = { cold, warming, trials };
    }
    const finished = (await driver.call({ kind: 'finish' })).value;
    return { isolated: self.crossOriginIsolated, actualHardwareConcurrency: navigator.hardwareConcurrency,
      initialization, preparation, guards, freshSetup, freshJobsValidated: jobs === 'fresh', ...finished, inputPointCountsVerified: !!inputs.reference.points && !!inputs.compared.points,
      candidateSource: preparation.value.candidateSource, candidateSha256: await digest(expected.hints),
      exactF64Sha256: await digest(expected.exact), displayedF32Sha256: await digest(expected.distances),
      stats: expected.stats, certified: expected.certified, exactDistancesMatchFullSearch: true,
      distanceBitsExact: true, summaryFieldsExact: true, certificateCountsExact: true, stableSerialBitsExact: true,
      measurements };
  } finally { stable.close(); driver.close(); }
}

const trialOrders = Array.from({ length: warmups + repeats }, (_, i) => verifierTrialOrder(i));
const report = { schemaVersion: 2, queryJobs: jobs, operationScope: jobs === 'fresh' ?
  'Fresh owned queries from an already parsed cloud, per-operation staging, selected pipeline, ordered summaries, f32 export, acknowledged query release and output transfer; immutable reference caches stay fixed. Excludes file parsing, initialization, reference import and diagnostic f64 export.' :
  'Prepared immutable query job reused across requests; per-job query preparation and release excluded.', orderScheme: 'williams-4', createdUtc: new Date().toISOString(), inputs, inputManifest,
  purpose: gpu === 'physical' && warmups >= 5 && repeats >= 10 ? 'timing-comparison' : 'compatibility-smoke',
  machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length,
    memoryBytes: totalmem(), node: process.version }, threads, warmups, repeats, batchSize, gpu, builds,
  inputManifestSha256: manifestBytes ? hash(manifestBytes) : undefined,
  sources: Object.fromEntries(['web/scripts/benchmark-shared-gpu-c2c.mjs', 'web/experiments/shared-gpu-c2c/worker.mjs',
    'web/experiments/shared-gpu-c2c/coi-sw.js', 'web/experiments/shared-gpu-c2c/trial-order.mjs'].map(name => [name, hash(readFileSync(resolve(repository, name), 'utf8').replaceAll('\r\n', '\n'))])),
  commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8', windowsHide: true }).trim(),
  note: 'Standalone shared candidate-verification experiment. No production defaults or isolation registration change. A plain HTTP static server and experiment-scoped service worker isolate the page and nested workers. Driver caps production pool capacity to threads+1 before initialization. Production candidate shader, f64 verification and ordered summary arithmetic are reused. Fixed pipeline reuses identical captured physical-WebGPU hints; GPU pipeline repeats rounded query packing, actual dispatch/readback and one-batch prefetch before verification. All modes retain identical owner reference and query order. Pool streams exact query coordinates per batch; pool-staged retains distributed queries once per job. The queryJobs and operationScope fields distinguish prepared reuse from fresh per-request jobs. Fresh jobs retain the already parsed cloud, create a new production owner snapshot on every call, stage only the current queries, and await one release acknowledgement per participating worker before freeing the owner job. Shared jobs replace only owned queries; reference generation must remain one. The fixed oracle and stable fallback are checked before releasing the initial prepared job. Diagnostic f64 export is outside warm trials. Both share the same production per-lane reference trees. Shared serial and parallel hold one extra imported tree and full query snapshot in the same threaded binary; stable serial is an untimed correctness/fallback control. Initialization and reference imports are separate; validation calls do not establish cold first-call speedup. The legacy cold field contains post-guard validation requests. Warm RPC times include the complete chosen pipeline, exact verification, ordered f64 summaries, f32 export and output transfer; diagnostic f64 exports occur only outside timed trials. GPU wait and verification intervals can overlap and must not be summed into wall time. Snapshot transfer bytes describe wire payload, not retained tree allocation. WASM committed pages are high-water counters, not RSS; all alternatives coexist, so no whole-application memory-saving claim is made. Measured point subsets are not independent surveys or SLAM-accuracy ground truth. Other host activity, GC, clocks and power are uncontrolled. No adapter identity is collected.' };
let browser;
try {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  browser = await chromium.launch({ channel: 'chromium', headless: true }); report.browserVersion = browser.version();
  const url = `http://127.0.0.1:${server.address().port}${prefix}`;
  for (const isolated of [false, true]) {
    const context = await browser.newContext();
    try {
      const page = await context.newPage(); await page.exposeFunction('benchmarkProgress', message => console.log(message));
      const response = await page.goto(url);
      if (response.headers()['cross-origin-opener-policy'] || response.headers()['cross-origin-embedder-policy']) throw new Error('Server supplied isolation headers');
      if (isolated) {
        await page.evaluate(async () => { await navigator.serviceWorker.register('./coi-sw.js'); await navigator.serviceWorker.ready; });
        await page.reload(); await page.waitForFunction(() => self.crossOriginIsolated, undefined, { timeout: 15_000 });
      }
      report[isolated ? 'measurement' : 'preflight'] = await page.evaluate(inPage, { isolated, inputs, threads, warmups, repeats, batchSize, gpu, jobs, trialOrders });
    } finally { await context.close(); }
  }
  for (const input of Object.values(inputs)) if (hash(readFileSync(input.path)) !== input.sha256) throw new Error('Input changed during measurement');
  if (manifestPath && hash(readFileSync(manifestPath)) !== hash(manifestBytes)) throw new Error('Manifest changed during measurement');
  for (const [mode, metadata] of Object.entries(builds)) verifyBuild(mode, metadata);
  for (const [name, expected] of Object.entries(report.sources)) if (hash(readFileSync(resolve(repository, name), 'utf8').replaceAll('\r\n', '\n')) !== expected) throw new Error(`Source changed during measurement: ${name}`);
  report.filesUnchangedDuringMeasurement = true;
  report.status = 'passed';
} catch (error) { report.status = 'failed'; report.error = String(error); process.exitCode = 1; }
finally {
  await browser?.close(); await new Promise(resolve => server.close(resolve));
  writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ status: report.status, output, error: report.error }));
}
