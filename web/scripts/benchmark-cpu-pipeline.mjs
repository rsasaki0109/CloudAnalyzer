// Pair-cold CPU preparations: acknowledgement-gated vs immediately queued searches.
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
const allowed = ["--output", "--reference", "--compared", "--input-manifest", "--repeats", "--threshold", "--voxel", "--min-points", "--method"];
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
}
const output = value("--output"), referencePath = value("--reference"), comparedPath = value("--compared");
if (!output || !referencePath || !comparedPath)
  throw new Error("Supply --output REPORT.json --reference FILE --compared FILE [--input-manifest JSON] [--method both|c2c|map-quality] [--repeats 10] [--threshold 0.15] [--voxel 1] [--min-points 10]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats = Number(value("--repeats", "10")), warmupRounds = 5;
const method = value("--method", "both");
if (!["both", "c2c", "map-quality"].includes(method)) throw new Error("Invalid method");
const params = { threshold: Number(value("--threshold", "0.15")), voxel: Number(value("--voxel", "1")),
  minPoints: Number(value("--min-points", "10")) };
if (!Number.isInteger(repeats) || repeats < 1 || !Number.isFinite(params.threshold) || params.threshold < 0 ||
  !Number.isFinite(params.voxel) || params.voxel <= 0 || !Number.isInteger(params.minPoints) || params.minPoints < 2)
  throw new Error("Invalid repeats or Map quality parameters");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const inputs = Object.fromEntries([["reference", referencePath], ["compared", comparedPath]].map(([role, path]) => {
  const bytes = readFileSync(path);
  return [role, { path, name: basename(path), bytes: bytes.length, sha256: digest(bytes) }];
}));
const manifestPath = value("--input-manifest"), manifestBytes = manifestPath ? readFileSync(manifestPath) : undefined;
const inputManifest = manifestBytes ? JSON.parse(manifestBytes) : undefined;
if (inputManifest && ["reference", "compared"].some(role => inputManifest.inputs?.[role]?.sha256 !== inputs[role].sha256 ||
  inputManifest.inputs[role].bytes !== inputs[role].bytes)) throw new Error("Input files differ from the manifest");
const root = fileURLToPath(new URL("../dist/", import.meta.url)), assets = resolve(root, "assets");
const workerName = readdirSync(assets).find(n => /^worker-.*\.js$/.test(n));
if (!workerName) throw new Error("Build the production app first");
const server = createServer((req, res) => {
  if (req.url === "/__map-cache") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (["/__reference", "/__compared"].includes(req.url)) {
    const input = inputs[req.url === "/__reference" ? "reference" : "compared"], bytes = readFileSync(input.path);
    if (digest(bytes) !== input.sha256) { res.writeHead(409).end("Input changed after hashing"); return; }
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
  browser = await chromium.launch({ channel: "chromium", headless: true });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/__map-cache`);
  const report = await page.evaluate(async ({ workerPath, inputs, inputManifest, params, warmupRounds, repeats, method }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const pending = waiting.get(data.seq);
      if (!pending) return;
      waiting.delete(data.seq); clearTimeout(pending.timer);
      data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
    };
    worker.onerror = event => {
      for (const pending of waiting.values()) { clearTimeout(pending.timer); pending.reject(new Error(event.message)); }
      waiting.clear();
    };
    const call = req => new Promise((resolve, reject) => {
      const id = ++seq;
      const timer = setTimeout(() => { waiting.delete(id); reject(new Error(`Worker timeout: ${req.kind}`)); }, 60_000);
      waiting.set(id, { resolve, reject, timer }); worker.postMessage({ seq: id, req });
    });
    const parity = (a, b, warm = false) => {
      if (a.cpuCached !== true || b.cpuCached !== true || a.cpuCacheReused !== warm || b.cpuCacheReused !== warm ||
        a.cpuPipelined !== false || b.cpuPipelined !== true || a.workers !== b.workers || b.workers < 2 ||
        !Number.isSafeInteger(b.cpuCacheInputBytes) || a.cpuCacheInputBytes !== b.cpuCacheInputBytes)
        throw new Error("Pipeline comparison changed scheduling/preparation controls");
      const first = new Uint32Array(a.distances.buffer), second = new Uint32Array(b.distances.buffer);
      if (first.length !== second.length || first.some((bits, i) => bits !== second[i])) throw new Error("Distance bit parity failed");
      if (!Object.keys(a.stats).every(key => Object.is(a.stats[key], b.stats[key]))) throw new Error("Statistics changed");
      let maxVoxelDifference = 0;
      for (const key of Object.keys(a.quality ?? {})) {
        const x = a.quality[key], y = b.quality[key];
        if (Object.is(x, y)) continue;
        const delta = Math.abs(x - y);
        if (["awd", "scs"].includes(key) && delta <= 1e-12 * Math.max(1, Math.abs(x), Math.abs(y))) {
          maxVoxelDifference = Math.max(maxVoxelDifference, delta); continue;
        }
        throw new Error(`Quality changed: ${key}`);
      }
      return { distanceBitMismatches: 0, statsExact: true, inlierMetricsExact: true, maxVoxelDifference };
    };
    try {
      await call({ kind: "cpu-prepare" });
      const clouds = {}, files = {};
      const load = async role => {
        const cloud = await call({ kind: "load", file: files[role], maxPoints: 2_000_000 });
        if (cloud.kind !== "cloud" || !cloud.count || cloud.count !== cloud.filePoints || cloud.keepEvery !== 1 ||
          inputManifest && cloud.count !== inputManifest.inputs[role].points) throw new Error(`Expected the full unchanged input: ${role}`);
        return cloud;
      };
      for (const role of ["reference", "compared"]) {
        const response = await fetch(`/__${role}`);
        if (!response.ok) throw new Error(`Input fetch failed: ${role}`);
        files[role] = new File([await response.arrayBuffer()], inputs[role].name);
        clouds[role] = await load(role);
      }
      const describe = cloud => ({ points: cloud.count, filePoints: cloud.filePoints, keepEvery: cloud.keepEvery, bounds: cloud.bounds });
      const reference = describe(clouds.reference), compared = describe(clouds.compared);
      const compute = async (method, cpuPipeline) => {
        const start = performance.now();
        const common = { compared: clouds.compared.id, reference: clouds.reference.id, cpuCache: true, cpuPipeline };
        const result = await call(method === "c2c" ? { kind: "c2c", ...common, signed: false, backend: "cpu",
          cpuNearestSimd: false, cpuStream: true, summarySort: false } :
          { kind: "map-evaluate", voxelCache: false, ...common, ...params, streamSummaries: true });
        return { ...result, elapsedMillis: performance.now() - start };
      };
      const cases = [];
      for (const kind of method === "both" ? ["c2c", "map-quality"] : [method]) {
        const warmups = [], trials = [];
        for (let i = -warmupRounds; i < repeats; i++) {
          const order = Math.abs(i) % 2 ? ["pipelined", "waiting"] : ["waiting", "pipelined"], results = {}, warm = {};
          for (const mode of order) {
            // Both controls get a fresh reference identity and no prepared pair.
            // Reloads and release submissions precede timed computation.
            // Lane order keeps release handling before the new preparation.
            await call({ kind: "remove", id: clouds.reference.id });
            clouds.reference = await load("reference");
            if (JSON.stringify(describe(clouds.reference)) !== JSON.stringify(reference)) throw new Error("Reload changed geometry/order metadata");
            results[mode] = await compute(kind, mode === "pipelined");
            warm[mode] = await compute(kind, mode === "pipelined");
            if (results[mode].cpuCacheReused || !warm[mode].cpuCacheReused) throw new Error("Expected cold-then-warm preparation");
            // Exact distances/stats must also survive repeated searches.
            const cold = results[mode], repeated = warm[mode];
            const first = new Uint32Array(cold.distances.buffer), second = new Uint32Array(repeated.distances.buffer);
            if (first.length !== second.length || first.some((bits, k) => bits !== second[k]) ||
              !Object.keys(cold.stats).every(key => Object.is(cold.stats[key], repeated.stats[key]))) throw new Error("Warm search changed exact results");
          }
          const a = results.waiting, b = results.pipelined, wa = warm.waiting, wb = warm.pipelined;
          const trial = { order, waitingMillis: a.elapsedMillis, pipelinedMillis: b.elapsedMillis,
            waitingWorkerMillis: a.millis, pipelinedWorkerMillis: b.millis, workers: b.workers,
            cacheInputBytes: b.cpuCacheInputBytes, stats: b.stats, quality: b.quality, ...parity(a, b),
            warm: { waitingMillis: wa.elapsedMillis, pipelinedMillis: wb.elapsedMillis, ...parity(wa, wb, true) } };
          (i < 0 ? warmups : trials).push(trial);
          await globalThis.benchmarkProgress(`${kind} ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: waiting ${Math.round(a.elapsedMillis)}, pipelined ${Math.round(b.elapsedMillis)} ms; exact parity`);
        }
        cases.push({ method: kind, warmups, trials });
      }
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, reference, compared, cases };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, inputs, inputManifest, params, warmupRounds, repeats, method });
  for (const input of Object.values(inputs)) if (digest(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (manifestPath && digest(readFileSync(manifestPath)) !== digest(manifestBytes)) throw new Error("Manifest changed during timing");
  const hash = name => digest(readFileSync(resolve(assets, name)));
  Object.assign(report, { schemaVersion: 1, comparison: "cpu-preparation-pipeline", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    workerSha256: hash(workerName), poolWorkerSha256: hash(readdirSync(assets).find(n => /^c2c-worker-.*\.js$/.test(n))),
    wasmSha256: hash(readdirSync(assets).find(n => /^ca_wasm.*\.wasm$/.test(n))), scriptSha256: digest(readFileSync(fileURLToPath(import.meta.url))),
    browserVersion: browser.version(), machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model,
      logicalCpus: cpus().length, memoryBytes: totalmem() }, inputs, inputManifest,
    inputManifestSha256: manifestBytes ? digest(manifestBytes) : undefined, params, method, warmupRounds, repeats,
    note: "Full supplied point clouds, no thinning, one production analysis worker and pool. Five warm-up pairs precede alternating sequential waiting/pipelined cold prepared CPU computations. Every mode reloads identical reference bytes under a fresh identity, submitting release of the preceding CPU preparation before timing; the compared cloud stays unchanged. Both controls build the same exact partitions, reference trees and Morton order and use scalar f64 searches, progressive joins, adaptive summaries, f32 inlier arithmetic and the existing Map quality voxel kernel. Only cold search dispatch differs: waiting requires the lane preparation ack; pipelined queues search on the same live worker generation immediately behind preparation, allowing it to search without another main-worker turn. Both messages must succeed and are drained on failure. Each cold call is followed by an independently recomputed warm call with the same mode/preparation; those times and parity are separate and warm order follows cold order. Wall times include planning, transfers, tree construction, search, joins, metrics and result RPCs; pool initialization/prewarming, loading, old-state release submission and validation are excluded. Release acknowledgements are not awaited; a lane may still handle its preceding release before the new preparation. Worker millis are recorded separately. This measures cold source-pair preparation in an already warmed worker pool, not browser/process startup. Neither result distances nor voxel scores are cached. Every cold and warm pair must preserve all distance bits and f64 summary fields; inlier metrics/voxel counts are exact and existing HashMap voxel reductions allow 1e-12 relative/absolute roundoff. Retained input bytes exclude derived tree/order storage, allocator overhead, source clouds and outputs and are equal between controls. Other applications, CPU clocks, power and GC are uncontrolled. Measured-source even/odd subsets are not independent survey ground truth." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
