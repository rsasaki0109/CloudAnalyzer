// Real-file C2C comparison, including RPC, output copies and exact summaries.
import { chromium } from '@playwright/test';
import { createServer } from 'node:http';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { resolve, extname, sep, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus, platform, release, totalmem } from 'node:os';

const args = process.argv.slice(2);
const allowed = ['--output', '--reference', '--compared', '--input-manifest', '--threads', '--warmups', '--repeats'];
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith('--') || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
}
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = option('--output'), reference = option('--reference'), compared = option('--compared');
if (!output || !reference || !compared)
  throw new Error('Supply --output NEW.json --reference FILE --compared FILE [--input-manifest JSON] [--threads 4] [--warmups 5] [--repeats 10]');
if (existsSync(output)) throw new Error('Report already exists');
const threads = Number(option('--threads', '4')), warmups = Number(option('--warmups', '5')), repeats = Number(option('--repeats', '10'));
if (!Number.isInteger(threads) || threads < 1 || threads > 8 || !Number.isInteger(warmups) || warmups < 0 ||
    !Number.isInteger(repeats) || repeats < 1) throw new Error('Invalid thread or trial count');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const repository = fileURLToPath(new URL('../../', import.meta.url));
const experiment = resolve(repository, 'web/experiments/shared-c2c');
const dist = resolve(repository, 'web/dist');
const assets = resolve(dist, 'assets');
const workerName = readdirSync(assets).find(name => /^worker-.*\.js$/.test(name));
if (!workerName) throw new Error('Build the production app first');
const inputs = Object.fromEntries([['reference', reference], ['compared', compared]].map(([role, path]) => {
  const bytes = readFileSync(path);
  return [role, { path: resolve(path), name: basename(path), bytes: bytes.length, sha256: sha256(bytes) }];
}));
const inputManifestPath = option('--input-manifest');
const manifestBytes = inputManifestPath ? readFileSync(inputManifestPath) : undefined;
const inputManifest = manifestBytes ? JSON.parse(manifestBytes) : undefined;
for (const [role, input] of Object.entries(inputs)) {
  if (inputManifest && (inputManifest.inputs?.[role]?.sha256 !== input.sha256 || inputManifest.inputs[role].bytes !== input.bytes))
    throw new Error(`Manifest mismatch: ${role}`);
  if (inputManifest) input.points = inputManifest.inputs[role].points;
}
const builds = Object.fromEntries(['serial', 'threads'].map(mode => {
  const directory = resolve(experiment, `pkg-${mode}`);
  const metadata = JSON.parse(readFileSync(resolve(directory, 'build-info.json'), 'utf8'));
  for (const [name, hash] of Object.entries(metadata.normalizedInputs)) {
    if (sha256(readFileSync(resolve(repository, name), 'utf8').replaceAll('\r\n', '\n')) !== hash)
      throw new Error(`Source changed since build: ${name}`);
  }
  for (const [name, hash] of Object.entries(metadata.artifacts)) {
    if (sha256(readFileSync(resolve(directory, name))) !== hash) throw new Error(`Build changed: ${mode}/${name}`);
  }
  return [mode, metadata];
}));
const prefix = '/CloudAnalyzer/app/shared-c2c/';
const server = createServer((request, response) => {
  const pathname = new URL(request.url, 'http://localhost').pathname;
  response.setHeader('Cache-Control', 'no-store');
  if (pathname === prefix) { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>Shared C2C experiment</title>'); return; }
  for (const [role, input] of Object.entries(inputs)) if (pathname === `${prefix}${role}`) {
    const bytes = readFileSync(input.path);
    if (sha256(bytes) !== input.sha256) { response.writeHead(409).end('Input changed'); return; }
    response.setHeader('Content-Type', 'application/octet-stream'); response.end(bytes); return;
  }
  const production = pathname.startsWith(`${prefix}production/`);
  const directory = production ? dist : experiment;
  const mount = production ? `${prefix}production/` : prefix;
  if (!pathname.startsWith(mount)) { response.writeHead(404).end(); return; }
  const path = resolve(directory, pathname.slice(mount.length));
  if (!path.startsWith(directory + sep)) { response.writeHead(403).end(); return; }
  try {
    response.setHeader('Content-Type', ({ '.js':'text/javascript', '.mjs':'text/javascript', '.wasm':'application/wasm' })[extname(path)] || 'application/octet-stream');
    const bytes = readFileSync(path);
    // Benchmark-only pool capacity. Production files on disk are untouched.
    response.end(production && basename(path) === workerName
      ? `Object.defineProperty(navigator, 'hardwareConcurrency', { value: ${threads + 1} });\n${bytes.toString('utf8')}` : bytes);
  } catch { response.writeHead(404).end(); }
});

async function runInPage({ isolated, inputs, threads, warmups, repeats, workerName }) {
  function rpc(url, production = false) {
    const worker = new Worker(url, { type: 'module' });
    let seq = 0;
    const pending = new Map();
    worker.onmessage = ({ data }) => {
      if (production && !data.response) return;
      const request = pending.get(data.seq);
      if (!request) return;
      clearTimeout(request.timer); pending.delete(data.seq);
      const ok = production ? data.response.ok : data.ok;
      if (ok) request.resolve({ value: production ? data.response.value : data.value,
        elapsedMillis: performance.now() - request.started, mainMemory: data.memory, poolMemory: data.poolMemory });
      else request.reject(new Error(production ? data.response.error : data.error));
    };
    worker.onerror = event => {
      for (const request of pending.values()) { clearTimeout(request.timer); request.reject(new Error(event.message || 'Worker failed')); }
      pending.clear();
    };
    const call = (req, transfer = []) => new Promise((resolve, reject) => {
      const id = ++seq, started = performance.now();
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${req.kind}`)); }, 60_000);
      pending.set(id, { resolve, reject, timer, started });
      try { worker.postMessage({ seq: id, req }, { transfer }); }
      catch (error) { clearTimeout(timer); pending.delete(id); reject(error); }
    });
    return { call, close: () => worker.terminate() };
  }
  const digest = async values => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice())))
    .map(v => v.toString(16).padStart(2, '0')).join('');
  const reject = async (operation, message) => {
    try { await operation(); } catch (error) { if (String(error).includes(message)) return; throw error; }
    throw new Error(`Expected rejection: ${message}`);
  };
  const standalone = rpc('./worker.mjs');
  if (!isolated) {
    try {
      await reject(() => standalone.call({ kind: 'init', build: 'threads', threads }), 'isolation');
      const fallback = await standalone.call({ kind: 'init', build: 'serial', threads: 1 });
      if (fallback.value.sharedMemory || fallback.value.threads) throw new Error('Serial fallback initialized shared threads');
      return { rejectedUnisolatedThreads: true, serialFallbackInitialized: true };
    } finally { standalone.close(); }
  }
  const production = rpc(`./production/assets/${workerName}`, true);
  const threaded = rpc('./worker.mjs');
  try {
    const poolInit = await production.call({ kind: 'cpu-prepare' });
    const stableInit = await standalone.call({ kind: 'init', build: 'serial', threads: 1 });
    const sharedInit = await threaded.call({ kind: 'init', build: 'threads', threads });
    if (!sharedInit.value.sharedMemory || JSON.stringify(sharedInit.value.indices) !== JSON.stringify(Array.from({ length: threads }, (_, i) => i)))
      throw new Error('Not every shared-memory lane responded');

    function exportedXyz(bytes) {
      const text = new TextDecoder().decode(bytes.subarray(0, 16_384));
      const end = text.indexOf('end_header\n') + 'end_header\n'.length;
      if (end < 'end_header\n'.length || !text.startsWith('ply\nformat binary_little_endian 1.0')) throw new Error('Unexpected production PLY');
      const head = text.slice(0, end), count = Number(head.match(/element vertex (\d+)/)?.[1]);
      const sizes = { char:1, uchar:1, short:2, ushort:2, int:4, uint:4, float:4, double:8 };
      let stride = 0;
      const offsets = {};
      for (const [, type, name] of head.matchAll(/^property (\w+) (\w+)$/gm)) {
        if (!sizes[type]) throw new Error('Unknown PLY scalar');
        if (['x', 'y', 'z'].includes(name)) { if (type !== 'double') throw new Error('XYZ precision changed'); offsets[name] = stride; }
        stride += sizes[type];
      }
      if (!count || Object.keys(offsets).length !== 3 || bytes.length !== end + stride * count) throw new Error('Incomplete exported XYZ');
      const view = new DataView(bytes.buffer, bytes.byteOffset + end), xyz = new Float64Array(count * 3);
      for (let i = 0; i < count; i++) for (const [axis, name] of ['x', 'y', 'z'].entries()) xyz[3 * i + axis] = view.getFloat64(i * stride + offsets[name], true);
      return xyz;
    }
    const clouds = {}, xyz = {};
    for (const role of ['reference', 'compared']) {
      const response = await fetch(`./${role}`);
      if (!response.ok) throw new Error('Input fetch failed');
      clouds[role] = (await production.call({ kind: 'load', file: new File([await response.arrayBuffer()], inputs[role].name), maxPoints: 2_000_000 })).value;
      const c = clouds[role];
      if (c.count !== c.filePoints || c.keepEvery !== 1 || inputs[role].points && inputs[role].points !== c.count)
        throw new Error('Input was thinned or differs from the manifest');
      xyz[role] = exportedXyz((await production.call({ kind: 'export', id: c.id, format: 'ply' })).value);
      if (xyz[role].length !== c.count * 3) throw new Error('Export changed point count');
    }
    const prepare = async client => {
      const start = performance.now(), reference = xyz.reference.slice(), queries = xyz.compared.slice();
      const result = await client.call({ kind: 'prepare', reference, queries }, [reference.buffer, queries.buffer]);
      return { ...result, elapsedMillis: performance.now() - start };
    };
    const stablePrepare = await prepare(standalone), sharedPrepare = await prepare(threaded);
    const run = mode => mode === 'pool'
      ? production.call({ kind: 'c2c', reference: clouds.reference.id, compared: clouds.compared.id, signed: false, backend: 'cpu' })
      : (mode === 'stable' ? standalone : threaded).call({ kind: 'compute', parallel: mode === 'shared' });
    const cold = {};
    for (const mode of ['pool', 'stable', 'shared']) cold[mode] = await run(mode);
    const expected = cold.pool.value;
    const bits = result => new Uint32Array(result.distances.buffer, result.distances.byteOffset, result.distances.length);
    const parity = result => {
      const a = bits(expected), b = bits(result);
      if (a.length !== b.length || a.some((v, i) => v !== b[i]) || !Object.keys(expected.stats).every(key => Object.is(expected.stats[key], result.stats[key])))
        throw new Error('Distance or summary bits changed');
    };
    parity(cold.stable.value); parity(cold.shared.value);
    const expectedWorkers = Math.max(1, Math.min(threads, Math.floor(clouds.compared.count / 100_000)));
    if (expected.workers !== expectedWorkers) throw new Error('Production worker count differs from requested benchmark cap');
    await reject(() => standalone.call({ kind: 'compute', parallel: true }), 'threads build');
    // Failed replacement must leave the old immutable snapshot usable.
    await reject(() => threaded.call({ kind: 'prepare', reference: new Float64Array([NaN, 0, 0]), queries: new Float64Array([0, 0, 0]) }), 'finite');
    parity((await run('shared')).value);
    const exact = [];
    for (const [client, parallel] of [[standalone, false], [threaded, false], [threaded, true]]) {
      const values = (await client.call({ kind: 'exact', parallel })).value;
      exact.push(await digest(values));
    }
    if (new Set(exact).size !== 1) throw new Error('Authoritative f64 distance bits changed');
    const compact = result => ({ elapsedMillis: result.elapsedMillis, workerMillis: result.value.millis,
      workers: result.value.workers, cacheReused: result.value.cpuCacheReused,
      cacheInputBytes: result.value.cpuCacheInputBytes, mainCommittedBytes: result.mainMemory,
      poolCommittedBytes: result.poolMemory, sharedCommittedBytes: result.value.committedBytes,
      sharedRetainedBytes: result.value.retainedBytes });
    const trials = [], warming = [], modes = ['stable', 'serial', 'pool', 'shared'];
    for (let i = 0; i < warmups + repeats; i++) {
      const order = [...modes.slice(i % 4), ...modes.slice(0, i % 4)], results = {};
      for (const mode of order) {
        const result = await run(mode); parity(result.value);
        if (mode === 'pool' && expectedWorkers > 1 && result.value.cpuCacheReused !== true) throw new Error('Expected retained production preparation');
        results[mode] = compact(result);
      }
      (i < warmups ? warming : trials).push({ order, results });
      await globalThis.benchmarkProgress(`${i < warmups ? 'warm-up' : 'trial'} ${i < warmups ? i + 1 : i - warmups + 1}: ` +
        modes.map(mode => `${mode} ${results[mode].elapsedMillis.toFixed(1)} ms`).join(', '));
    }
    const released = (await threaded.call({ kind: 'release' })).value;
    if (released.retainedBytes !== 0) throw new Error('Snapshot was not released');
    await reject(() => run('shared'), 'No prepared snapshot');
    return { isolated: self.crossOriginIsolated, actualHardwareConcurrency: navigator.hardwareConcurrency,
      inputClouds: Object.fromEntries(Object.entries(clouds).map(([role, c]) => [role,
        { count: c.count, filePoints: c.filePoints, keepEvery: c.keepEvery, bounds: c.bounds }])),
      inputXyzSha256: { reference: await digest(xyz.reference), compared: await digest(xyz.compared) },
      initialization: { pool: { elapsedMillis: poolInit.elapsedMillis, mainCommittedBytes: poolInit.mainMemory,
        poolCommittedBytes: poolInit.poolMemory }, serial: stableInit, shared: sharedInit },
      preparation: { serial: stablePrepare, shared: sharedPrepare },
      cold: Object.fromEntries(Object.entries(cold).map(([mode, result]) => [mode, compact(result)])),
      exactF64Sha256: exact[0], displayedF32Sha256: await digest(expected.distances), stats: expected.stats,
      distanceBitsExact: true, summaryFieldsExact: true, rejectedReplacementPreservedSnapshot: true,
      releaseVerified: true, warming, trials };
  } finally { standalone.close(); threaded.close(); production.close(); }
}

const report = { schemaVersion: 1, createdUtc: new Date().toISOString(), inputs, inputManifest,
  purpose: warmups >= 5 && repeats >= 10 ? 'timing-comparison' : 'compatibility-smoke',
  machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model,
    logicalCpus: cpus().length, memoryBytes: totalmem(), node: process.version },
  inputManifestSha256: manifestBytes ? sha256(manifestBytes) : undefined, threads, warmups, repeats, builds,
  benchmarkSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
  workerSha256: sha256(readFileSync(resolve(experiment, 'worker.mjs'))),
  serviceWorkerSha256: sha256(readFileSync(resolve(experiment, 'coi-sw.js'))),
  productionWorkerSha256: sha256(readFileSync(resolve(assets, workerName))),
  productionPoolSha256: sha256(readFileSync(resolve(assets, readdirSync(assets).find(name => /^c2c-worker-.*\.js$/.test(name))))),
  productionWasmSha256: sha256(readFileSync(resolve(assets, readdirSync(assets).find(name => name.endsWith('.wasm'))))),
  commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8', windowsHide: true }).trim(),
  note: 'Measured-file compatibility and timing experiment. Static HTTP server adds no isolation headers; scoped service worker isolates only this experiment. Pool capacity is capped by a served-only hardwareConcurrency override to threads+1. Production defaults are unchanged; an additive opt-in PreparedC2c method supplies the experiment. Inputs are loaded/indexed once, then production PLY exports supply identical f64 XYZ and query order to both standalone builds. Cold initialization, snapshot preparation and first calls are separate. Warm RPC times include search, ordered f64 summaries, f32 export and transfer. Same-nightly serial vs shared holds the binary and tree fixed; stable serial is a separate compiler/build control. Production retains spatial partitions and query order; both standalone modes retain one full reference tree, query snapshot and Morton order. Pool comparison covers the whole alternative design, including partition differences. Retained payload capacities and committed WASM pages are reported separately; committed pages are high-water reservations, not RSS or peak process memory. Both alternatives coexist for paired timing, so these measurements do not establish whole-application memory savings. No GPU timing is included. Input geometry is not independent-survey or SLAM-accuracy ground truth. Other processes, clocks, power and GC are uncontrolled.' };
let browser;
try {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  browser = await chromium.launch({ channel: 'chromium', headless: true });
  report.browserVersion = browser.version();
  const url = `http://127.0.0.1:${server.address().port}${prefix}`;
  for (const isolated of [false, true]) {
    const context = await browser.newContext();
    try {
      const page = await context.newPage();
      await page.exposeFunction('benchmarkProgress', message => console.log(message));
      const response = await page.goto(url);
      if (response.headers()['cross-origin-opener-policy'] || response.headers()['cross-origin-embedder-policy']) throw new Error('Server supplied isolation headers');
      if (isolated) {
        await page.evaluate(async () => { await navigator.serviceWorker.register('./coi-sw.js'); await navigator.serviceWorker.ready; });
        await page.reload();
        await page.waitForFunction(() => self.crossOriginIsolated, undefined, { timeout: 15_000 });
      }
      const result = await page.evaluate(runInPage, { isolated, inputs, threads, warmups, repeats, workerName });
      report[isolated ? 'measurement' : 'preflight'] = result;
    } finally { await context.close(); }
  }
  for (const input of Object.values(inputs)) if (sha256(readFileSync(input.path)) !== input.sha256) throw new Error('Source changed during measurement');
  if (inputManifestPath && sha256(readFileSync(inputManifestPath)) !== sha256(manifestBytes)) throw new Error('Manifest changed during measurement');
  report.status = 'passed';
} catch (error) { report.status = 'failed'; report.error = String(error); process.exitCode = 1; }
finally {
  await browser?.close(); await new Promise(resolve => server.close(resolve));
  writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ status: report.status, output, error: report.error }));
}
