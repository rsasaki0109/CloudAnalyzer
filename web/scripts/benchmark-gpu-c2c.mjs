// Production-worker C2C benchmark. Includes preparation, upload, readback,
// f64 verification and summaries. Refuses to overwrite reports.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output");
if (!output) throw new Error("Usage: node scripts/benchmark-gpu-c2c.mjs --output REPORT.json [--hardware] [--prewarm] [--compare-overlap | --compare-bounds | --compare-setup | --compare-simd | --compare-batches | --compare-leaf | --compare-workgroups] [--batch-size 262144] [--fixture grid | scatter] [--sizes 10000,100000,1000000] [--repeats 3]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const sizes = value("--sizes", "10000,100000,1000000").split(",").map(Number);
const repeats = Number(value("--repeats", "3"));
const compareOverlap = args.includes("--compare-overlap");
const compareBounds = args.includes("--compare-bounds");
const compareSetup = args.includes("--compare-setup");
const compareSimd = args.includes("--compare-simd");
const compareBatches = args.includes("--compare-batches");
const compareLeaf = args.includes("--compare-leaf");
const compareWorkgroups = args.includes("--compare-workgroups");
const workgroupWarmupRounds = compareWorkgroups ? 5 : 0;
const batchSize = args.includes("--batch-size") ? Number(value("--batch-size")) : undefined;
if (batchSize !== undefined && (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 262_144))
  throw new Error("Batch size must be an integer in 1..262144");
if (compareBatches && batchSize !== undefined && batchSize <= 65_536)
  throw new Error("An explicit batch comparison requires a size larger than 65536");
const prewarm = args.includes("--prewarm");
if ([compareOverlap, compareBounds, compareSetup, compareSimd, compareBatches, compareLeaf, compareWorkgroups].filter(Boolean).length > 1) throw new Error("Choose one comparison mode");
const fixtureKind = value("--fixture", "grid");
if (!["grid", "scatter"].includes(fixtureKind)) throw new Error("Fixture must be grid or scatter");
if (!sizes.every(n => Number.isInteger(n) && n > 0 && n <= 2_000_000) || !Number.isInteger(repeats) || repeats < 1) {
  throw new Error("Sizes must be 1..2000000 and repeats must be positive");
}
const root = fileURLToPath(new URL("../dist/", import.meta.url));
const workerName = readdirSync(resolve(root, "assets")).find(n => /^worker-.*\.js$/.test(n));
if (!workerName) throw new Error("Build the production app first");
const workerPath = `/assets/${workerName}`;
const server = createServer((req, res) => {
  if (req.url === "/__gpu-benchmark") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  const path = resolve(root, "." + new URL(req.url, "http://localhost").pathname.replace(/\/$/, "/index.html"));
  if (!path.startsWith(root.endsWith(sep) ? root : root + sep)) { res.writeHead(403).end(); return; }
  try {
    res.setHeader("Content-Type", ({ ".js": "text/javascript", ".wasm": "application/wasm", ".html": "text/html" })[extname(path)] || "application/octet-stream");
    res.end(readFileSync(path));
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ channel: "chromium", headless: true, args: args.includes("--hardware")
    ? ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"]
    : ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader"] });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/__gpu-benchmark`);
  const report = await page.evaluate(async ({ workerPath, sizes, repeats, hardware, compareOverlap, compareBounds, compareSetup, compareSimd, compareBatches, compareLeaf, compareWorkgroups, workgroupWarmupRounds, batchSize, prewarm, fixtureKind }) => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter");
    const info = adapter.info;
    if (hardware && info.isFallbackAdapter) throw new Error("Expected a physical GPU, got a fallback adapter");
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const pending = waiting.get(data.seq);
      if (!pending) return;
      waiting.delete(data.seq);
      clearTimeout(pending.timer);
      data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
    };
    worker.onerror = event => {
      for (const pending of waiting.values()) {
        clearTimeout(pending.timer);
        pending.reject(new Error(event.message));
      }
      waiting.clear();
    };
    const call = req => new Promise((resolve, reject) => {
      const id = ++seq;
      const timer = setTimeout(() => { waiting.delete(id); reject(new Error(`Worker request timed out: ${req.kind}`)); }, 60_000);
      waiting.set(id, { resolve, reject, timer });
      worker.postMessage({ seq: id, req });
    });
    const fixture = (n, lift) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 24);
      bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      const width = Math.ceil(Math.sqrt(n));
      let state = 0x5eed;
      const random = () => {
        state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
        return (state >>> 0) / 0x1_0000_0000;
      };
      for (let i = 0; i < n; i++) {
        const x = i % width, y = Math.floor(i / width);
        const point = fixtureKind === "scatter"
          ? [368_000 + random() * 50, 3_955_000 + random() * 50, 40 + random() * 10 + lift]
          : [368_000 + x * 0.1, 3_955_000 + y * 0.1, 40 + Math.sin(x / 7) * 0.2 + lift];
        point.forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    const cases = [];
    const parity = (cpu, gpu, n) => {
      if (gpu.compute?.backend !== "webgpu") throw new Error(`GPU trial fell back: ${gpu.compute?.fallback}`);
      const cpuBits = new Uint32Array(cpu.distances.buffer), gpuBits = new Uint32Array(gpu.distances.buffer);
      if (cpuBits.length !== n || gpuBits.length !== n) throw new Error("Incomplete C2C distances");
      let mismatches = 0;
      for (let j = 0; j < n; j++) if (cpuBits[j] !== gpuBits[j]) mismatches++;
      const statsEqual = JSON.stringify(cpu.stats) === JSON.stringify(gpu.stats);
      if (mismatches || !statsEqual) throw new Error(`C2C parity failed at ${n} points`);
      return { mismatches, statsEqual };
    };
    try {
      let prewarmMillis;
      for (const n of sizes) {
        await globalThis.benchmarkProgress(`Measuring ${fixtureKind}, ${n} points per cloud...`);
        const reference = await call({ kind: "load", file: fixture(n, 0), maxPoints: n });
        const compared = await call({ kind: "load", file: fixture(n, 0.05), maxPoints: n });
        if (reference.count !== n || compared.count !== n) throw new Error("Fixture was unexpectedly thinned");
        const c2c = (backend, ref = reference.id) => call({ kind: "c2c", gpuReuseImportBounds: false, reference: ref, compared: compared.id, signed: false,
          backend: backend.startsWith("wg") || ["serial", "planes", "isolated", "scalar", "small", "leaf", "full"].includes(backend) ? "webgpu" : backend,
          gpuPrefetch: backend !== "serial", gpuVerifyBounds: backend !== "planes", gpuReuseDevice: backend !== "isolated",
          gpuVerifySimd: backend !== "scalar", gpuBatchSize: backend === "small" ? 65_536 : batchSize,
          gpuLeafOnly: backend === "leaf" ? true : backend === "full" ? false : undefined,
          gpuWorkgroupSize: backend.startsWith("wg") ? Number(backend.slice(2)) : undefined });
        const freshReference = async backend => {
          const fresh = await call({ kind: "load", file: fixture(n, 0), maxPoints: n });
          if (fresh.count !== n) throw new Error("Setup fixture was unexpectedly thinned");
          try { return await c2c(backend, fresh.id); }
          finally { await call({ kind: "remove", id: fresh.id }); }
        };
        const warmedCpu = await c2c("cpu");
        if (prewarm && prewarmMillis === undefined) {
          const started = performance.now();
          await call({ kind: "gpu-prepare" });
          prewarmMillis = performance.now() - started;
        }
        const firstGpu = await c2c("webgpu");
        if (firstGpu.compute?.referenceReused) throw new Error("Expected fresh reference preparation");
        if (!firstGpu.compute?.verifyBounds) throw new Error("Expected default bounding-box verification");
        if (!firstGpu.compute.verifySimd) throw new Error("Expected default pairwise WASM SIMD");
        if (prewarm && !firstGpu.compute.deviceReused) throw new Error("Expected the prepared device to be reused");
        const coldGpu = { millis: firstGpu.millis, compute: firstGpu.compute, ...parity(warmedCpu, firstGpu, n) };
        if (compareOverlap) parity(warmedCpu, await c2c("serial"), n);
        if (compareBounds) parity(warmedCpu, await c2c("planes"), n);
        if (compareSetup) parity(warmedCpu, await freshReference("isolated"), n);
        if (compareSimd) parity(warmedCpu, await c2c("scalar"), n);
        if (compareBatches) parity(warmedCpu, await c2c("small"), n);
        if (compareLeaf) {
          parity(warmedCpu, await c2c("full"), n);
          parity(warmedCpu, await c2c("leaf"), n);
        }
        if (compareWorkgroups) {
          await globalThis.benchmarkProgress(`Warming all workgroups for ${workgroupWarmupRounds} rounds...`);
          for (let round = 0; round < workgroupWarmupRounds; round++) {
            const cpu = await c2c("cpu");
            for (const size of [32, 64, 128, 256]) parity(cpu, await c2c(`wg${size}`), n);
          }
        }
        // Subsequent GPU trials reuse the reference; alternate CPU/GPU order.
        const trials = [];
        for (let i = 0; i < repeats; i++) {
          const results = {};
          const modes = compareOverlap ? ["cpu", "serial", "webgpu"]
            : compareBounds ? ["cpu", "planes", "webgpu"]
            : compareSetup ? ["cpu", "isolated", "webgpu"]
            : compareSimd ? ["cpu", "scalar", "webgpu"]
            : compareBatches ? ["cpu", "small", "webgpu"]
            : compareLeaf ? ["cpu", "full", "leaf"]
            : compareWorkgroups ? ["cpu", "wg32", "wg64", "wg128", "wg256"] : ["cpu", "webgpu"];
          // Rotate all modes; multiples of modes.length balance every position.
          const order = modes.slice(i % modes.length).concat(modes.slice(0, i % modes.length));
          for (const backend of order) {
            results[backend] = compareSetup && backend !== "cpu" ? await freshReference(backend) : await c2c(backend);
          }
          if (compareLeaf) results.webgpu = results.leaf;
          if (compareWorkgroups) results.webgpu = results.wg64;
          if (results.webgpu.compute?.referenceReused !== !compareSetup) throw new Error("Unexpected reference reuse");
          if (!results.webgpu.compute.deviceReused) throw new Error("GPU device/pipeline was not reused");
          if (!results.webgpu.compute.verifyBounds || !results.webgpu.compute.prefetch) throw new Error("Expected bounding-box verification with prefetch");
          if (!results.webgpu.compute.verifySimd) throw new Error("Expected pairwise WASM SIMD");
          const trial = { cpuMillis: results.cpu.millis, gpuMillis: results.webgpu.millis,
            workers: results.cpu.workers, compute: results.webgpu.compute, ...parity(results.cpu, results.webgpu, n) };
          if (compareOverlap) {
            if (!results.serial.compute?.referenceReused || results.serial.compute.prefetch) throw new Error("Expected cached serial GPU mode");
            trial.serialGpu = { millis: results.serial.millis, compute: results.serial.compute, ...parity(results.cpu, results.serial, n) };
          }
          if (compareBounds) {
            if (!results.planes.compute?.referenceReused || results.planes.compute.verifyBounds ||
              !results.planes.compute.prefetch) throw new Error("Expected cached, prefetched plane-only verification");
            trial.planeGpu = { millis: results.planes.millis, compute: results.planes.compute, ...parity(results.cpu, results.planes, n) };
          }
          if (compareSetup) {
            if (results.isolated.compute?.referenceReused || results.isolated.compute.deviceReused) throw new Error("Expected a fresh isolated reference/device");
            trial.isolatedGpu = { millis: results.isolated.millis, compute: results.isolated.compute, ...parity(results.cpu, results.isolated, n) };
          }
          if (compareSimd) {
            const scalar = results.scalar.compute;
            if (!scalar?.referenceReused || !scalar.deviceReused || scalar.verifySimd || !scalar.verifyBounds || !scalar.prefetch) throw new Error("Expected a cached scalar-leaf GPU control");
            if (scalar.certified !== results.webgpu.compute.certified) throw new Error("SIMD changed certificate decisions");
            trial.scalarGpu = { millis: results.scalar.millis, compute: scalar, ...parity(results.cpu, results.scalar, n) };
          }
          if (compareBatches) {
            const small = results.small.compute, large = results.webgpu.compute;
            if (!small?.referenceReused || !small.deviceReused || !small.prefetch || !small.verifyBounds || !small.verifySimd)
              throw new Error("Expected the same cached, prefetched verifier for the small batch control");
            if (small.batchSize !== 65_536 || (batchSize !== undefined && large.batchSize !== batchSize) ||
              large.batchSize < small.batchSize)
              throw new Error("Device cannot support the requested batch comparison");
            if (small.batches !== Math.ceil(n / small.batchSize) || large.batches !== Math.ceil(n / large.batchSize))
              throw new Error("Unexpected GPU batch count");
            if (small.certified !== large.certified) throw new Error("Batch size changed certificate decisions");
            trial.smallBatchGpu = { millis: results.small.millis, compute: small, ...parity(results.cpu, results.small, n) };
          }
          if (compareLeaf) {
            const full = results.full.compute, leaf = results.leaf.compute;
            if (!full?.referenceReused || !full.deviceReused || full.leafOnly || !leaf.leafOnly ||
                full.batchSize !== leaf.batchSize || full.batches !== leaf.batches)
              throw new Error("Expected identical batching/cache with only GPU hint mode changed");
            trial.fullSearchGpu = { millis: results.full.millis, compute: full, ...parity(results.cpu, results.full, n) };
          }
          if (compareWorkgroups) {
            trial.order = order;
            trial.workgroups = {};
            for (const size of [32, 64, 128, 256]) {
              const result = results[`wg${size}`], compute = result.compute, control = results.webgpu.compute;
              if (!compute?.referenceReused || !compute.deviceReused || !compute.verifyBounds || !compute.verifySimd ||
                  !compute.prefetch || compute.leafOnly || compute.workgroupSize !== size ||
                  compute.batchSize !== control.batchSize || compute.batches !== control.batches || compute.certified !== control.certified)
                throw new Error("Workgroup comparison changed cache, batching or exact verification");
              trial.workgroups[size] = { millis: result.millis, compute, ...parity(results.cpu, result, n) };
            }
          }
          trials.push(trial);
        }
        cases.push({ pointsPerCloud: n, fixture: fixtureKind === "scatter"
          ? "Seed 0x5eed, UTM 50x50x10m uniform 3D scatter, 0.05m vertical offset"
          : "UTM wavy grid, 0.05m vertical offset", coldGpu, trials });
        const medians = compareWorkgroups ? [32, 64, 128, 256].map(size => {
          const times = trials.map(t => t.workgroups[size].millis).sort((a, b) => a - b);
          return `${size}=${((times[Math.floor(times.length / 2)] + times[Math.floor((times.length - 1) / 2)]) / 2).toFixed(1)}`;
        }).join(", ") : "";
        await globalThis.benchmarkProgress(`Completed ${n}: GPU ${trials.map(t => Math.round(t.gpuMillis))}${compareLeaf ? `; full-search control ${trials.map(t => Math.round(t.fullSearchGpu.millis))}` : ""} ms${compareWorkgroups ? `; workgroup medians ${medians} ms` : ""}`);
        await call({ kind: "remove", id: reference.id });
        await call({ kind: "remove", id: compared.id });
      }
      return { prewarmMillis, adapter: { vendor: info.vendor, architecture: info.architecture,
        isFallbackAdapter: info.isFallbackAdapter },
        userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, cases };
    } finally { worker.terminate(); }
  }, { workerPath, sizes, repeats, hardware: args.includes("--hardware"), compareOverlap, compareBounds, compareSetup, compareSimd, compareBatches, compareLeaf, compareWorkgroups, workgroupWarmupRounds, batchSize, prewarm, fixtureKind });
  report.createdAt = new Date().toISOString();
  report.commit = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim();
  report.workingTreeDirty = !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim();
  report.workerSha256 = createHash("sha256").update(readFileSync(resolve(root, "." + workerPath))).digest("hex");
  report.wasmSha256 = createHash("sha256").update(readFileSync(resolve(root, "assets",
    readdirSync(resolve(root, "assets")).find(n => /^ca_wasm.*\.wasm$/.test(n))))).digest("hex");
  report.browserVersion = browser.version();
  report.machine = { platform: platform(), release: release(), cpu: cpus()[0]?.model,
    logicalCpus: cpus().length, memoryBytes: totalmem() };
  try {
    report.nvidiaSmi = execFileSync("nvidia-smi", ["--query-gpu=driver_version,memory.total", "--format=csv,noheader"],
      { encoding: "utf8", windowsHide: true }).trim();
  } catch { /* Adapter information remains available on non-NVIDIA hosts. */ }
  report.schemaVersion = 9;
  report.gpuModelOmitted = true;
  report.compareOverlap = compareOverlap;
  report.compareBounds = compareBounds;
  report.compareSetup = compareSetup;
  report.compareSimd = compareSimd;
  report.compareBatches = compareBatches;
  report.compareLeaf = compareLeaf;
  report.compareWorkgroups = compareWorkgroups;
  report.workgroupWarmupRounds = workgroupWarmupRounds;
  report.requestedBatchSize = batchSize;
  report.batchSelection = batchSize === undefined ? "automatic" : "fixed";
  report.prewarm = prewarm;
  report.note = "Synthetic data. coldGpu uses a fresh reference; deviceReused records whether its device/pipeline was already ready. A shared device persists across sizes, or can be prepared separately with --prewarm. Ordinary warm trials reuse the reference. --compare-setup loads a new reference before each GPU trial: shared-device vs fresh isolated device; load time is excluded, reference build/upload/verification and summaries included. Exact f64 verification uses axis certificates plus sibling subtree boxes and pairwise WASM SIMD by default; planeGpu disables boxes; scalarGpu disables only leaf SIMD. --compare-batches rotates CPU, a 65536-query control and automatic batch selection (or an explicit --batch-size), reusing the same reference/device/verifier. Effective query caps and completed dispatch counts are recorded. Batch buffers retain their largest capacity, so both controls use the same retained allocations after warm-up. Next-batch GPU dispatch overlaps verification unless disabled by serial mode. gpuMillis intervals overlap verifyMillis and are not additive. Other applications and CPU/GPU power/clocks are uncontrolled. No real-source throughput claim.";
  if (compareLeaf) report.note += " --compare-leaf rotates CPU, backtracking full-search GPU hints and one near-leaf GPU hints; gpuMillis is the leaf-hint end-to-end time, fullSearchGpu is the control. Both modes share the same reference/device/batches and complete f64 verifier; certificate counts may change. coldGpu records the current default hint mode, which may differ from the compared leaf mode.";
  if (compareWorkgroups) report.note += " --compare-workgroups rotates CPU and 32/64/128/256 GPU invocations per workgroup. All four variants and CPU receive five unmeasured warm-up rounds first and reuse the same reference/device/buffers, batch caps, full-search hints and exact verifier. Ten repeats balance each of five ordering positions twice. workgroups records each variant including verification and summaries; gpuMillis is the 64-invocation control. coldGpu records the default 64-invocation pipeline. Every variant must match CPU distance bits, f64 stats and certificate counts.";
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
  for (const c of report.cases) console.log(`${c.pointsPerCloud} points: CPU ${c.trials.map(t => Math.round(t.cpuMillis))} ms; GPU first-reference ${Math.round(c.coldGpu.millis)} ms, ${compareSetup ? "new-reference shared-device" : "reused"} ${c.trials.map(t => Math.round(t.gpuMillis))} ms${compareOverlap ? `, serial ${c.trials.map(t => Math.round(t.serialGpu.millis))} ms` : compareBounds ? `, planes ${c.trials.map(t => Math.round(t.planeGpu.millis))} ms` : compareSetup ? `, isolated-device ${c.trials.map(t => Math.round(t.isolatedGpu.millis))} ms` : compareSimd ? `, scalar-leaf ${c.trials.map(t => Math.round(t.scalarGpu.millis))} ms` : compareBatches ? `, 65536-query control ${c.trials.map(t => Math.round(t.smallBatchGpu.millis))} ms` : ""}; exact parity`);
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
