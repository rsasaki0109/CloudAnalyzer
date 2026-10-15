// Production C2C: CPU leaves/joining, summaries and WebGPU verification controls.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output");
if (!output) throw new Error("Supply --output REPORT.json; choose --compare-cache, --compare-join, --compare-summary, --compare-backend, --compare-verify, --compare-verify-setup, --compare-verify-queries, --compare-verify-batches [--batch-size 131072|262144] [--control-batch-size 65536|131072] [--verify-workers 2..8] or default SIMD; optional --reference FILE --compared FILE --input-manifest JSON --max-points N, or --fixture grid|scatter|independent --sizes N,...; --backend cpu|webgpu --hardware --repeats N");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const referencePath = value("--reference"), comparedPath = value("--compared");
const suppliedFiles = !!referencePath || !!comparedPath;
if (suppliedFiles && (!referencePath || !comparedPath || args.includes("--fixture") || args.includes("--sizes")))
  throw new Error("Supply both input files; --fixture/--sizes are synthetic-only controls");
const maxPoints = Number(value("--max-points", "2000000"));
if (!Number.isInteger(maxPoints) || maxPoints < 1 || maxPoints > 2_000_000) throw new Error("--max-points must be in 1..2000000");
if (!suppliedFiles && (args.includes("--max-points") || args.includes("--input-manifest"))) throw new Error("Input controls require --reference and --compared");
const fileHash = path => createHash("sha256").update(readFileSync(path)).digest("hex");
const inputs = suppliedFiles ? Object.fromEntries([["reference", referencePath], ["compared", comparedPath]].map(([role, path]) => {
  const bytes = readFileSync(path);
  return [role, { path, name: basename(path), bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") }];
})) : undefined;
const manifestPath = value("--input-manifest");
const manifestBytes = manifestPath ? readFileSync(manifestPath) : undefined;
const inputManifest = manifestBytes ? JSON.parse(manifestBytes) : undefined;
if (inputManifest && ["reference", "compared"].some(role => inputManifest.inputs?.[role]?.sha256 !== inputs[role].sha256 ||
    inputManifest.inputs[role].bytes !== inputs[role].bytes)) throw new Error("Input files differ from the supplied manifest");
const sizes = value("--sizes", args.includes("--compare-cache") ? "200000,1000000,2000000" : "10000,100000,1000000,2000000").split(",").map(Number);
const repeats = Number(value("--repeats", "10"));
const fixtureKind = suppliedFiles ? "files" : value("--fixture", "grid"), warmupRounds = 5;
const compareJoin = args.includes("--compare-join");
const compareSummary = args.includes("--compare-summary");
const compareBackend = args.includes("--compare-backend");
const compareVerify = args.includes("--compare-verify");
const compareSetup = args.includes("--compare-verify-setup");
const compareBatches = args.includes("--compare-verify-batches");
const compareQueries = args.includes("--compare-verify-queries");
const compareCache = args.includes("--compare-cache");
const batchSize = Number(value("--batch-size", "131072"));
if (args.includes("--batch-size") && !compareBatches || ![131_072, 262_144].includes(batchSize))
  throw new Error("--batch-size requires --compare-verify-batches and 131072 or 262144 (control: 65536)");
const controlBatchSize = Number(value("--control-batch-size", "65536"));
if (args.includes("--control-batch-size") && !compareBatches || ![65_536, 131_072].includes(controlBatchSize) || controlBatchSize >= batchSize)
  throw new Error("--control-batch-size requires --compare-verify-batches and 65536 or 131072, smaller than --batch-size");
const verifyWorkers = Number(value("--verify-workers", "4"));
if (!Number.isInteger(verifyWorkers) || verifyWorkers < 2 || verifyWorkers > 8 || args.includes("--verify-workers") && !compareVerify && !compareSetup && !compareBatches && !compareQueries)
  throw new Error("--verify-workers requires a verification comparison and an integer in 2..8");
const backend = value("--backend", compareVerify || compareSetup || compareBatches || compareQueries ? "webgpu" : "cpu"), hardware = args.includes("--hardware");
const usesGpu = backend === "webgpu" || compareBackend;
if ([compareJoin, compareSummary, compareBackend, compareVerify, compareSetup, compareBatches, compareQueries, compareCache].filter(Boolean).length > 1 || !["cpu", "webgpu"].includes(backend) ||
    backend === "webgpu" && !compareSummary && !compareVerify && !compareSetup && !compareBatches && !compareQueries || hardware && !usesGpu || compareBackend && args.includes("--backend") || (compareVerify || compareSetup || compareBatches || compareQueries) && backend !== "webgpu")
  throw new Error("Choose one comparison; --compare-backend chooses CPU/GPU itself; single-backend WebGPU requires a summary or verification comparison");
if (!(suppliedFiles || ["grid", "scatter", "independent"].includes(fixtureKind)) || !sizes.every(n => Number.isInteger(n) && n > 0 && n <= 2_000_000) ||
    !Number.isInteger(repeats) || repeats < 1) throw new Error("Invalid fixture, sizes or repeats");
if (compareCache && !suppliedFiles && sizes.some(n => n < 200_000)) throw new Error("CPU cache comparisons require at least 200000 query points");
const root = fileURLToPath(new URL("../dist/", import.meta.url));
const assets = resolve(root, "assets");
const workerName = readdirSync(assets).find(n => /^worker-.*\.js$/.test(n));
if (!workerName) throw new Error("Build the production app first");
const server = createServer((req, res) => {
  if (req.url === "/__c2c-benchmark") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (suppliedFiles && ["/__reference", "/__compared"].includes(req.url)) {
    const input = inputs[req.url === "/__reference" ? "reference" : "compared"];
    const bytes = readFileSync(input.path);
    if (createHash("sha256").update(bytes).digest("hex") !== input.sha256) { res.writeHead(409).end("Input changed after hashing"); return; }
    res.setHeader("Content-Type", "application/octet-stream"); res.end(bytes); return;
  }
  const path = resolve(root, "." + new URL(req.url, "http://localhost").pathname);
  if (!path.startsWith(root.endsWith(sep) ? root : root + sep)) { res.writeHead(403).end(); return; }
  try {
    res.setHeader("Content-Type", ({ ".js": "text/javascript", ".wasm": "application/wasm" })[extname(path)] || "application/octet-stream");
    res.end(readFileSync(path));
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ channel: "chromium", headless: true,
    args: !usesGpu ? [] : hardware
      ? ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"]
      : ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader"] });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/__c2c-benchmark`);
  const report = await page.evaluate(async ({ workerPath, sizes, repeats, warmupRounds, fixtureKind, compareJoin, compareSummary, compareBackend, compareVerify, compareSetup, compareBatches, compareQueries, compareCache, batchSize, controlBatchSize, verifyWorkers, backend, hardware, inputs, maxPoints }) => {
    let adapterInfo;
    if (backend === "webgpu" || compareBackend) {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter || hardware && adapter.info.isFallbackAdapter) throw new Error("Expected the requested WebGPU adapter");
      const info = adapter.info;
      // Public reports identify the adapter family without publishing a model.
      adapterInfo = { vendor: info.vendor, architecture: info.architecture,
        isFallbackAdapter: info.isFallbackAdapter };
    }
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const p = waiting.get(data.seq);
      if (!p) return;
      waiting.delete(data.seq); clearTimeout(p.timer);
      data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = event => {
      for (const p of waiting.values()) { clearTimeout(p.timer); p.reject(new Error(event.message)); }
      waiting.clear();
    };
    const call = req => new Promise((resolve, reject) => {
      const id = ++seq;
      const timer = setTimeout(() => { waiting.delete(id); reject(new Error(`Worker timeout: ${req.kind}`)); }, 60_000);
      waiting.set(id, { resolve, reject, timer });
      worker.postMessage({ seq: id, req });
    });
    const fixture = (n, lift) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 24);
      bytes.set(header);
      const view = new DataView(bytes.buffer, header.length), width = Math.ceil(Math.sqrt(n));
      let state = fixtureKind === "independent" && lift ? 0xc0ffee : 0x5eed;
      const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 0x1_0000_0000; };
      for (let i = 0; i < n; i++) {
        const x = i % width, y = Math.floor(i / width);
        const point = fixtureKind !== "grid"
          ? [368_000 + random() * 50, 3_955_000 + random() * 50, 40 + random() * 10 + lift]
          : [368_000 + x * 0.1, 3_955_000 + y * 0.1, 40 + Math.sin(x / 7) * 0.2 + lift];
        point.forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    const parity = (scalar, simd, n) => {
      if ((!compareBackend && !compareVerify && scalar.workers !== simd.workers) || scalar.summarySort !== compareSummary || simd.summarySort !== false)
        throw new Error("Summary comparison did not preserve controls/workers");
      if (backend === "webgpu" || compareBackend) {
        for (const result of compareBackend ? [simd] : [scalar, simd]) if (result.compute?.backend !== "webgpu" || !result.compute.verifySimd ||
          !result.compute.verifyBounds || !result.compute.prefetch || result.compute.workgroupSize !== 64)
          throw new Error("Summary trial did not use the default WebGPU path");
        if (!compareBackend && (scalar.compute.certified !== simd.compute.certified || !compareBatches && scalar.compute.batchSize !== simd.compute.batchSize))
          throw new Error("Summary trial changed GPU verification/batching");
        if (compareVerify && (scalar.compute.verifyWorkers !== 1 || simd.compute.verifyWorkers !== Math.min(verifyWorkers, Math.max(1, navigator.hardwareConcurrency - 1)) ||
          simd.compute.verifyWorkers < 2 || scalar.compute.verifyFallback || simd.compute.verifyFallback))
          throw new Error("Verification trial did not use the requested worker paths");
        if (compareSetup && ([scalar, simd].some(r => r.compute.verifyWorkers !== Math.min(verifyWorkers, Math.max(1, navigator.hardwareConcurrency - 1)) ||
          r.compute.verifyWorkers < 2 || r.compute.verifyFallback) || scalar.compute.verifySnapshot !== false || simd.compute.verifySnapshot !== true))
          throw new Error("Setup trial did not preserve workers or preparation controls");
        if (compareBatches && ([scalar, simd].some(r => r.compute.verifyWorkers !== Math.min(verifyWorkers, Math.max(1, navigator.hardwareConcurrency - 1)) ||
          r.compute.verifyWorkers < 2 || r.compute.verifyFallback || !r.compute.verifySnapshot || !Number.isFinite(r.compute.gpuWaitMillis)) ||
          scalar.compute.batchSize !== controlBatchSize || simd.compute.batchSize !== batchSize ||
          scalar.compute.batches !== Math.ceil(n / controlBatchSize) || simd.compute.batches !== Math.ceil(n / batchSize)))
          throw new Error("Batch trial changed workers/controls or requested sizes were device-clamped");
        if (compareQueries && ([scalar, simd].some(r => r.compute.verifyWorkers !== Math.min(verifyWorkers, Math.max(1, navigator.hardwareConcurrency - 1)) ||
          r.compute.verifyWorkers < 2 || r.compute.verifyFallback || !r.compute.verifySnapshot) ||
          scalar.compute.verifyStagedQueries !== false || simd.compute.verifyStagedQueries !== true || simd.compute.verifyQueryBytes !== n * 24))
          throw new Error("Query trial changed workers/controls or did not stage each exact coordinate once");
        if (compareBackend && (scalar.compute?.backend === "webgpu" || scalar.cpuSimd !== false ||
            scalar.cpuStreaming !== (scalar.workers > 1) || scalar.cpuCached !== (scalar.workers > 1)))
          throw new Error("Backend control did not use production scalar CPU with progressive joining");
      } else if (scalar.cpuSimd !== false || simd.cpuSimd !== (!compareJoin && !compareSummary && !compareCache) ||
          scalar.cpuStreaming !== ((compareSummary || compareCache) && scalar.workers > 1) ||
          simd.cpuStreaming !== ((compareJoin || compareSummary || compareCache) && simd.workers > 1))
        throw new Error("CPU comparison did not preserve selected kernels/workers");
      if (compareCache && (scalar.workers < 2 || scalar.cpuCached || scalar.cpuCacheReused || !simd.cpuCached))
        throw new Error("Cache comparison requires parallel uncached/cached CPU paths");
      if (compareCache && (!Number.isSafeInteger(simd.cpuCacheInputBytes) || simd.cpuCacheInputBytes < 28 * n))
        throw new Error("Missing CPU preparation input size");
      const a = new Uint32Array(scalar.distances.buffer), b = new Uint32Array(simd.distances.buffer);
      if (a.length !== n || b.length !== n) throw new Error("Incomplete CPU distances");
      let mismatches = 0;
      for (let i = 0; i < n; i++) if (a[i] !== b[i]) mismatches++;
      const statsEqual = Object.keys(scalar.stats).every(k => Object.is(scalar.stats[k], simd.stats[k]));
      if (mismatches || !statsEqual) throw new Error("CPU C2C parity failed");
      return { mismatches, statsEqual };
    };
    const cases = [];
    try {
      await globalThis.benchmarkProgress("Waiting for the automatic CPU pool warm-up...");
      const prepareStarted = performance.now();
      await call({ kind: "cpu-prepare" });
      const poolPrepareMillis = performance.now() - prepareStarted;
      const sourceFile = async role => {
        const response = await fetch(`/__${role}`);
        if (!response.ok) throw new Error(`Cannot fetch ${role}: ${response.status}`);
        return new File([await response.arrayBuffer()], inputs[role].name);
      };
      for (const size of inputs ? [maxPoints] : sizes) {
        await globalThis.benchmarkProgress(`Measuring ${fixtureKind}, point budget ${size}; ${warmupRounds} warm-up rounds...`);
        let reference = await call({ kind: "load", file: inputs ? await sourceFile("reference") : fixture(size, 0), maxPoints: size });
        const compared = await call({ kind: "load", file: inputs ? await sourceFile("compared") : fixture(size, 0.05), maxPoints: size });
        const n = compared.count;
        if (reference.kind !== "cloud" || compared.kind !== "cloud" || !n || !reference.count) throw new Error("C2C requires nonempty point clouds");
        if (!inputs && (reference.count !== size || n !== size)) throw new Error("Fixture was unexpectedly thinned");
        const c2c = (optimized, workers = (compareSetup || compareBatches || compareQueries || compareVerify && optimized) ? verifyWorkers : 1) => call({ kind: "c2c", gpuReuseImportBounds: false, reference: reference.id, compared: compared.id,
          signed: false, backend: compareBackend ? optimized ? "webgpu" : "cpu" : backend,
          cpuNearestSimd: !compareJoin && !compareSummary && !compareBackend && !compareVerify && !compareSetup && !compareBatches && !compareQueries && !compareCache && optimized,
          cpuCache: compareBackend || compareCache && optimized,
          cpuStream: compareBackend || compareSummary || compareCache || compareJoin && optimized, summarySort: compareSummary && !optimized,
          gpuVerifyWorkers: workers, gpuVerifySnapshot: !compareSetup || optimized, gpuStageQueries: compareQueries && optimized,
          // Keep concurrency comparisons independent of the tuned parallel cap.
          gpuBatchSize: compareBatches ? optimized ? batchSize : controlBatchSize :
            compareVerify ? n >= 1_000_000 ? 131_072 : 65_536 : undefined });
        const describe = cloud => ({ points: cloud.count, filePoints: cloud.filePoints, keepEvery: cloud.keepEvery, bounds: cloud.bounds, timings: cloud.timings });
        if (compareSetup) {
          const trials = [];
          const firstReference = describe(reference);
          for (let i = -warmupRounds; i < repeats; i++) {
            if (i !== -warmupRounds) {
              await call({ kind: "remove", id: reference.id });
              reference = await call({ kind: "load", file: inputs ? await sourceFile("reference") : fixture(size, 0), maxPoints: size });
              if (reference.count !== firstReference.points || reference.keepEvery !== firstReference.keepEvery)
                throw new Error("Reloaded reference changed point selection");
            }
            // Exact same bytes, a fresh reference identity and empty worker
            // caches. Warm the main GPU tree/device outside both timed runs.
            const serial = await c2c(false, 1);
            const modes = ["rebuild", "snapshot"], order = Math.abs(i) % 2 ? [...modes].reverse() : modes;
            const results = {};
            for (const mode of order) results[mode] = await c2c(mode === "snapshot");
            const a = results.rebuild, b = results.snapshot;
            const coldParity = parity(a, b, n);
            if ([a, b].some(r => !r.compute.referenceReused || !r.compute.deviceReused || r.compute.verifyReferenceReused))
              throw new Error("Expected warm GPU data/device and cold worker references");
            const serialBits = new Uint32Array(serial.distances.buffer), resultBits = new Uint32Array(b.distances.buffer);
            if (serial.compute.verifyFallback || serial.compute.backend !== "webgpu" || serial.compute.certified !== b.compute.certified ||
              serialBits.some((v, j) => v !== resultBits[j]) || !Object.keys(serial.stats).every(k => Object.is(serial.stats[k], b.stats[k])))
              throw new Error("Cold setup changed serial exact results");
            const warmResults = {};
            for (const mode of [...order].reverse()) warmResults[mode] = await c2c(mode === "snapshot");
            const wa = warmResults.rebuild, wb = warmResults.snapshot;
            const warmParity = parity(wa, wb, n);
            if ([wa, wb].some(r => !r.compute.verifyReferenceReused)) throw new Error("Expected reused worker references");
            if (i >= 0) trials.push({ order, rebuildMillis: a.millis, snapshotMillis: b.millis,
              serialPrepareMillis: serial.millis, serialCompute: serial.compute,
              workers: b.workers, controlWorkers: a.workers, stats: a.stats,
              controlCompute: a.compute, optimizedCompute: b.compute, serialParity: true, ...coldParity,
              warm: { order: [...order].reverse(), rebuildMillis: wa.millis, snapshotMillis: wb.millis,
                controlCompute: wa.compute, optimizedCompute: wb.compute, ...warmParity } });
            await globalThis.benchmarkProgress(`Setup ${i < 0 ? "warm-up" : "timed"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: rebuild ${Math.round(a.millis)}, snapshot ${Math.round(b.millis)} ms; exact parity`);
          }
          cases.push({ pointsPerCloud: n, reference: firstReference, compared: describe(compared), trials });
          await call({ kind: "remove", id: reference.id });
          await call({ kind: "remove", id: compared.id });
          continue;
        }
        let verificationSetup;
        let cpuCacheSetup;
        if (compareCache) {
          const uncached = await c2c(false), cached = await c2c(true);
          cpuCacheSetup = { uncachedMillis: uncached.millis, cachedFirstMillis: cached.millis,
            workers: cached.workers, cached: cached.cpuCached, reused: cached.cpuCacheReused,
            inputBytes: cached.cpuCacheInputBytes,
            ...parity(uncached, cached, n) };
          if (cached.cpuCacheReused) throw new Error("Expected cold CPU preparation");
        }
        if (compareVerify) {
          await c2c(false); // Warm only the main GPU reference/device first.
          const serial = await c2c(false), parallel = await c2c(true);
          verificationSetup = { serialWarmMillis: serial.millis, parallelFirstMillis: parallel.millis,
            controlCompute: serial.compute, optimizedCompute: parallel.compute, ...parity(serial, parallel, n) };
          if (parallel.compute.verifyReferenceReused) throw new Error("Expected cold verification references");
        }
        for (let i = 0; i < warmupRounds; i++) parity(await c2c(false), await c2c(true), n);
        const trials = [];
        for (let i = 0; i < repeats; i++) {
          const modes = compareCache ? ["uncached", "cached"] : compareQueries ? ["streamed", "staged"] : compareBatches ? ["smallBatch", "largeBatch"] : compareVerify ? ["serial", "parallel"] : compareBackend ? ["cpu", "webgpu"] : compareSummary ? ["sorted", "selected"] : compareJoin ? ["collected", "streaming"] : ["scalar", "simd"];
          const order = i % 2 ? [...modes].reverse() : modes, results = {};
          for (const mode of order) results[mode] = await c2c(mode === modes[1]);
          const a = results[modes[0]], b = results[modes[1]];
          if ((backend === "webgpu" || compareBackend) && (compareBackend ? [b] : [a, b]).some(r => !r.compute?.referenceReused || !r.compute.deviceReused))
            throw new Error("Timed summary trials must reuse the GPU reference/device");
          if (compareVerify && !b.compute.verifyReferenceReused) throw new Error("Timed verification must reuse worker references");
          if ((compareBatches || compareQueries) && [a, b].some(r => !r.compute.verifyReferenceReused)) throw new Error("Timed verification must reuse worker references");
          if (compareCache && !b.cpuCacheReused) throw new Error("Timed CPU preparation must be reused");
          if (compareBackend && a.workers > 1 && !a.cpuCacheReused) throw new Error("Timed backend CPU preparation must be reused");
          trials.push({ order, [modes[0] + "Millis"]: a.millis, [modes[1] + "Millis"]: b.millis,
            workers: b.workers, controlWorkers: a.workers, stats: a.stats, controlCompute: a.compute, optimizedCompute: b.compute,
            ...(compareCache ? { controlCached: a.cpuCached, optimizedCached: b.cpuCached, cacheReused: b.cpuCacheReused,
              cacheInputBytes: b.cpuCacheInputBytes } : compareBackend ? { cpuCached: a.cpuCached,
              cpuCacheReused: a.cpuCacheReused, cpuCacheInputBytes: a.cpuCacheInputBytes } : {}), ...parity(a, b, n) });
        }
        cases.push({ pointsPerCloud: n, reference: describe(reference), compared: describe(compared), verificationSetup, cpuCacheSetup, trials });
        await globalThis.benchmarkProgress(`Completed ${n}: control ${trials.map(t => Math.round(compareCache ? t.uncachedMillis : compareQueries ? t.streamedMillis : compareBatches ? t.smallBatchMillis : compareVerify ? t.serialMillis : compareBackend ? t.cpuMillis : compareSummary ? t.sortedMillis : compareJoin ? t.collectedMillis : t.scalarMillis))}; optimized ${trials.map(t => Math.round(compareCache ? t.cachedMillis : compareQueries ? t.stagedMillis : compareBatches ? t.largeBatchMillis : compareVerify ? t.parallelMillis : compareBackend ? t.webgpuMillis : compareSummary ? t.selectedMillis : compareJoin ? t.streamingMillis : t.simdMillis))} ms; exact parity`);
        await call({ kind: "remove", id: reference.id });
        await call({ kind: "remove", id: compared.id });
      }
      return { fixture: fixtureKind, backend: compareBackend ? "cpu-vs-webgpu" : backend, adapter: adapterInfo, poolPrepareMillis, userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, cases };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, sizes, repeats, warmupRounds, fixtureKind, compareJoin, compareSummary, compareBackend, compareVerify, compareSetup, compareBatches, compareQueries, compareCache, batchSize, controlBatchSize, verifyWorkers, backend, hardware, inputs, maxPoints });
  report.schemaVersion = 1;
  report.createdAt = new Date().toISOString();
  report.warmupRounds = warmupRounds;
  report.poolPrewarm = true;
  report.comparison = compareCache ? "cpu-preparation-cache" : compareQueries ? "gpu-verification-queries" : compareBatches ? "gpu-verification-batches" : compareSetup ? "gpu-verifier-setup" : compareVerify ? "gpu-verification" : compareBackend ? "backend" : compareSummary ? "summary" : compareJoin ? "joining" : "simd";
  report.verifyWorkers = compareVerify || compareSetup || compareBatches || compareQueries ? verifyWorkers : undefined;
  report.batchSizes = compareBatches ? [controlBatchSize, batchSize] : undefined;
  report.gpuModelOmitted = usesGpu ? true : undefined;
  report.inputs = inputs;
  report.inputManifest = inputManifest;
  report.inputManifestSha256 = manifestBytes ? createHash("sha256").update(manifestBytes).digest("hex") : undefined;
  report.benchmarkScriptSha256 = fileHash(fileURLToPath(import.meta.url));
  if (inputManifest && ["reference", "compared"].some(role => report.cases[0][role].filePoints !== inputManifest.inputs[role].points))
    throw new Error("Decoded file counts differ from the supplied manifest");
  report.commit = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim();
  report.workingTreeDirty = !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim();
  const hash = name => createHash("sha256").update(readFileSync(resolve(assets, name))).digest("hex");
  report.workerSha256 = hash(workerName);
  report.poolWorkerSha256 = hash(readdirSync(assets).find(n => /^c2c-worker-.*\.js$/.test(n)));
  report.wasmSha256 = hash(readdirSync(assets).find(n => /^ca_wasm.*\.wasm$/.test(n)));
  report.browserVersion = browser.version();
  report.machine = { platform: platform(), release: release(), cpu: cpus()[0]?.model,
    logicalCpus: cpus().length, memoryBytes: totalmem() };
  report.note = "Synthetic UTM wavy grid or seeded 50x50x10m scatter, 0.05m vertical offset. Ordinary production CPU C2C, same snapshots, spatial partitions, Morton query order, reference construction and worker counts. Await the already-started automatic CPU pool warm-up before any fixture/trial; its time is recorded separately and excluded. Five unmeasured scalar/SIMD warm-up pairs precede alternating timed pairs. Times include planning, copies/transfers, tree builds, distances, joining and exact summaries; file load/rendering are excluded. Every timed pair must match displayed f32 distance bits and exact f64 statistics. Native builds/non-SIMD WASM remain scalar. CPU clocks, power and other applications are uncontrolled; no real-source or cross-machine speed claim.";
  if (fixtureKind === "independent") report.note += " Independent scatter uses reference seed 0x5eed and compared seed 0xc0ffee, yielding varied nearest distances instead of mostly constant paired-point offsets.";
  if (inputs) report.note = report.note.replace("Synthetic UTM wavy grid or seeded 50x50x10m scatter, 0.05m vertical offset.", "Supplied input files, unchanged by the benchmark. Source-file hashes, optional provenance, decoded/file point counts, thinning factors and loaded bounds are recorded.")
    .replace("no real-source or cross-machine speed claim", "no cross-machine speed guarantee");
  if (compareCache) {
    report.note = "Parallel CPU C2C preparation reuse for the same immutable compared/reference identities. The uncached control repeats spatial partitioning, coordinate transfers, reference tree construction and Morton ordering on every call. The cached control retains each lane's exact queries, Morton order and reference tree, plus main-worker output indices. Both controls use identical partitions, worker counts, scalar leaves, progressive joining and adaptive exact summaries. Each call starts a new nearest-hit seed and computes every distance; result distances are not cached. cpuCacheSetup separately records the first uncached and first cached call, including preparation. Five warm-up pairs precede alternating timed pairs; every timed cached call must reuse preparation. Times include all per-call work and exact summaries; CPU pool prewarming, loading and rendering are excluded. Every pair must match displayed f32 distance bits and reported f64 statistics. One idle prepared pair is retained, with independent active snapshots during source edits/removal or pair replacement. Retention adds resident memory; WASM capacity may remain at its high-water mark after release. CPU clocks, power and other applications are uncontrolled; no cross-machine speed guarantee. Input hashes and provenance are recorded for supplied files.";
    report.note += " The prepared pair remains resident during uncached control calls in the same pool; allocation/locality and host effects are not isolated. cacheInputBytes reports coordinate and output-index element bytes, excluding derived tree nodes, slot indices, Morton order, spare capacity and allocator metadata.";
  } else if (compareQueries) {
    report.note = "Hybrid WebGPU C2C query preparation: streaming exact f64 coordinates with each batch versus staging each lane's exact query snapshot once per job in WASM. Both controls use identical cached reference snapshots, verification worker count, GPU batch size, prefetch, bounds, SIMD, workgroup and adaptive-summary controls. CPU pool prewarming, file loading and initial reference setup are excluded. Five warm-up pairs precede ten (or requested) alternating timed pairs. Staged coordinates are not reused across jobs: every timed call includes packing, query transfers, WASM ownership, GPU readbacks, verification, ordered joining, query release submission and exact summaries. Coordinates retain original f64 bits; each source query is staged to one lane only. The total coordinate bytes are unchanged; allocation/copy scheduling and per-batch payloads differ. Every pair must match displayed f32 distance bits, exact reported f64 statistics and certificate counts. Timed worker/GPU fallback is rejected. Preparation can overlap the first GPU dispatch, and gpuMillis includes overlapping verification; phase intervals must not be summed. CPU/GPU clocks, power and other applications are uncontrolled; no cross-machine speed guarantee. Input hashes and provenance are recorded for supplied files.";
  } else if (compareBatches) {
    report.note = "Hybrid WebGPU C2C parallel verification: the two explicit batchSizes, with identical exact-snapshot worker references, worker count, prefetch, bounds, SIMD, workgroup and adaptive-summary controls. The CPU pool is prewarmed (time recorded separately). Five warm-up pairs precede alternating timed pairs; timed pairs reuse GPU and worker references. Times include job snapshots, query copies/transfers, readbacks, exact verification, joining and summaries; file loading is excluded. Every pair must match displayed f32 distance bits, exact reported f64 statistics and certificate counts. Fallback and device-clamped batch sizes are rejected. gpuWaitMillis measures host time awaiting GPU setup/readbacks after overlapped verification, including host scheduling and queue delays; it is not shader time. gpuMillis includes overlapping intervals and must not be added to verifyMillis. CPU/GPU clocks, power and other applications are uncontrolled; no cross-machine speed guarantee. Input hashes and optional provenance are recorded for supplied files; synthetic inputs use the selected fixture otherwise.";
  } else if (compareSetup) {
    report.note = "Hybrid WebGPU C2C worker reference setup: original-XYZ tree rebuilding versus importing a validated exact f64 tree snapshot, with identical verification worker count and GPU/query/summary controls. CPU workers are prewarmed (time recorded separately). Every pair reloads identical reference bytes under a fresh identity; the main GPU tree/device is warmed by an excluded serial run, while both worker reference caches are empty. Five preparation warm-up pairs precede alternating timed pairs. Cold times include snapshot export, copies/transfers, worker import or tree construction, query snapshots, GPU readbacks, exact verification, joining and summaries. File loading and the main GPU reference construction are excluded. Each pair also records warm runs reusing both worker caches, in reverse order. Every cold and warm pair must match displayed f32 distance bits, exact reported f64 statistics and certificate counts; cold output also matches serial verification. Fallback is rejected. Phase intervals overlap and must not be summed. CPU/GPU clocks, power and other applications are uncontrolled; no cross-machine speed guarantee. Input hashes and optional provenance are recorded for supplied files.";
  } else if (compareVerify) {
    report.note = "Hybrid WebGPU C2C serial versus independent f64 WASM verification workers. Only GPU verification concurrency differs; identical hint, bounds, SIMD, batch/workgroup and adaptive-summary controls. The CPU pool is prewarmed (time recorded separately). verificationSetup records one warm serial run and the first parallel run, which includes exporting/copying/transferring the exact reference snapshot and validating/importing worker trees; the main GPU reference/device is already warm in both. Five warm-up pairs precede alternating timed pairs; timed pairs reuse GPU and worker references. Times include job snapshots, query copies/transfers, readbacks, exact verification, joining and summaries; file loading is excluded. Every pair must match all displayed f32 distance bits, exact reported f64 statistics and certificate counts. Timed fallback is rejected. Phase intervals overlap and must not be summed. CPU/GPU clocks, power and other applications are uncontrolled; no cross-machine speed guarantee. Input hashes and optional provenance are recorded for supplied files; synthetic inputs use the selected fixture otherwise.";
  } else if (compareBackend) {
    report.note = report.note.replace("Ordinary production CPU C2C", "Production parallel CPU and hybrid WebGPU C2C").replace("scalar/SIMD", "CPU/WebGPU")
      .replace("same snapshots, spatial partitions, Morton query order, reference construction and worker counts", "same input clouds and snapshots, with backend-specific query planning, trees and worker counts");
    report.note += " Both backends use adaptive exact summaries. CPU uses scalar leaves, progressive joining and default preparation reuse for parallel jobs; timed CPU calls must reuse prepared exact queries/trees and report retained input bytes. GPU uses default prefetch, bounds, SIMD verification, batch caps and 64-invocation workgroups, with warmed reference/device reuse. Timed GPU fallback is rejected. Backend timing includes CPU parallel workers versus one hybrid main worker; worker counts need not match. GPU phase intervals exclude final summaries and can overlap.";
  } else if (compareSummary) {
    report.note = report.note.replace("scalar/SIMD", "full-sort/median-selection");
    report.note += " Only median calculation differs: sortedMillis uses full total_cmp sorting, selectedMillis selects the upper middle and (for even sizes) the maximum lower-partition value. Equal nonzero extrema with a non-NaN sum skip median scratch allocation; a complete total-order scan also permits directly reading ordered inputs when endpoints match the extrema. The input reduction order and midpoint arithmetic are identical. Both CPU controls use scalar leaves and progressive joining; GPU controls use default prefetch, bounds, SIMD verification, batches and workgroups with reference/device reuse. GPU phase intervals exclude final summaries and can overlap; do not add them to infer wall time.";
    if (backend === "webgpu") report.note = report.note.replace("Ordinary production CPU C2C", "Production hybrid WebGPU C2C");
  } else if (compareJoin) report.note = report.note.replace("scalar/SIMD", "collect-all/progressive-join") + " Both joining controls use scalar leaves; collectedMillis waits for every result before joining, streamingMillis joins each result on arrival. Serial cases have identical joining behavior. Summary arithmetic/order is unchanged.";
  else report.note += " Both leaf controls collect all pool results before joining, isolating SIMD from the new progressive-join default.";
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
