// Real measured-source inputs; only CPU preparation reuse differs in each pair.
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
const allowed = ["--output", "--reference", "--compared", "--input-manifest", "--repeats", "--threshold", "--voxel", "--min-points"];
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
}
const output = value("--output"), referencePath = value("--reference"), comparedPath = value("--compared");
if (!output || !referencePath || !comparedPath)
  throw new Error("Supply --output REPORT.json --reference FILE --compared FILE [--input-manifest JSON] [--repeats 10] [--threshold 0.15] [--voxel 1] [--min-points 10]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats = Number(value("--repeats", "10")), warmupRounds = 5;
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
  const report = await page.evaluate(async ({ workerPath, inputs, inputManifest, params, warmupRounds, repeats }) => {
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
    const parity = (a, b) => {
      if (a.cpuCached !== false || a.cpuCacheReused !== false || b.cpuCached !== true || a.workers !== b.workers ||
        b.workers < 2 || !Number.isSafeInteger(b.cpuCacheInputBytes)) throw new Error("Cache comparison changed scheduling/controls");
      const first = new Uint32Array(a.distances.buffer), second = new Uint32Array(b.distances.buffer);
      if (first.length !== second.length || first.some((bits, i) => bits !== second[i])) throw new Error("Distance bit parity failed");
      if (!Object.keys(a.stats).every(key => Object.is(a.stats[key], b.stats[key]))) throw new Error("Statistics changed");
      let maxVoxelDifference = 0;
      for (const key of Object.keys(a.quality)) {
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
      const clouds = {};
      for (const role of ["reference", "compared"]) {
        const response = await fetch(`/__${role}`);
        if (!response.ok) throw new Error(`Input fetch failed: ${role}`);
        clouds[role] = await call({ kind: "load", file: new File([await response.arrayBuffer()], inputs[role].name), maxPoints: 2_000_000 });
        const cloud = clouds[role];
        if (cloud.kind !== "cloud" || !cloud.count || cloud.count !== cloud.filePoints || cloud.keepEvery !== 1 ||
          inputManifest && cloud.count !== inputManifest.inputs[role].points) throw new Error(`Expected the full unchanged input: ${role}`);
      }
      const compute = async cpuCache => {
        const start = performance.now();
        const result = await call({ kind: "map-evaluate", voxelCache: false, compared: clouds.compared.id, reference: clouds.reference.id,
          ...params, streamSummaries: true, cpuCache });
        return { ...result, elapsedMillis: performance.now() - start };
      };
      const firstUncached = await compute(false), firstCached = await compute(true);
      if (firstCached.cpuCacheReused) throw new Error("Expected first-use preparation");
      const setup = { uncachedMillis: firstUncached.elapsedMillis, cachedFirstMillis: firstCached.elapsedMillis,
        workers: firstCached.workers, reused: firstCached.cpuCacheReused, inputBytes: firstCached.cpuCacheInputBytes,
        ...parity(firstUncached, firstCached) };
      const warmups = [];
      for (let i = 0; i < warmupRounds; i++) warmups.push(parity(await compute(false), await compute(true)));
      const trials = [];
      for (let i = 0; i < repeats; i++) {
        const order = i % 2 ? ["cached", "uncached"] : ["uncached", "cached"], results = {};
        for (const mode of order) results[mode] = await compute(mode === "cached");
        const a = results.uncached, b = results.cached;
        if (!b.cpuCacheReused || b.cpuCacheInputBytes !== setup.inputBytes) throw new Error("Timed preparation was not reused");
        trials.push({ order, uncachedMillis: a.elapsedMillis, cachedMillis: b.elapsedMillis,
          uncachedWorkerMillis: a.millis, cachedWorkerMillis: b.millis, workers: b.workers,
          cacheReused: b.cpuCacheReused, cacheInputBytes: b.cpuCacheInputBytes, quality: b.quality, ...parity(a, b) });
        await globalThis.benchmarkProgress(`Trial ${i + 1}: uncached ${Math.round(a.elapsedMillis)}, cached ${Math.round(b.elapsedMillis)} ms; exact distances/stats/inliers`);
      }
      const describe = cloud => ({ points: cloud.count, filePoints: cloud.filePoints, keepEvery: cloud.keepEvery, bounds: cloud.bounds });
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent,
        reference: describe(clouds.reference), compared: describe(clouds.compared), setup, warmups, trials };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, inputs, inputManifest, params, warmupRounds, repeats });
  for (const input of Object.values(inputs)) if (digest(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (manifestPath && digest(readFileSync(manifestPath)) !== digest(manifestBytes)) throw new Error("Manifest changed during timing");
  const hash = name => digest(readFileSync(resolve(assets, name)));
  Object.assign(report, { schemaVersion: 1, comparison: "map-quality-cpu-preparation-cache", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    workerSha256: hash(workerName), poolWorkerSha256: hash(readdirSync(assets).find(n => /^c2c-worker-.*\.js$/.test(n))),
    wasmSha256: hash(readdirSync(assets).find(n => /^ca_wasm.*\.wasm$/.test(n))), scriptSha256: digest(readFileSync(fileURLToPath(import.meta.url))),
    browserVersion: browser.version(), machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model,
      logicalCpus: cpus().length, memoryBytes: totalmem() }, inputs, inputManifest,
    inputManifestSha256: manifestBytes ? digest(manifestBytes) : undefined, params, warmupRounds, repeats,
    note: "Full supplied point clouds with no thinning. Five warm-up pairs, then alternating sequential uncached/cached Map quality calls on the same production analysis worker and pool. Both controls use identical bidirectional partitions, scalar f64 searches, progressive joins, adaptive forward summaries, f32 inlier arithmetic and the existing voxel kernel; only immutable CPU preparation reuse differs. All distances and all voxel scores are recomputed every call. Wall times include RPC/result transfers, voxel work, exact nearest searches, joins and metrics; file loading, viewer rendering and validation are excluded. Worker millis are recorded separately. Setup is one fixed-order uncached-then-cached pair and is not a paired first-use performance claim. The prepared pair remains resident during uncached control calls; memory/allocator/locality effects are not isolated. Retained input bytes cover exact query/reference XYZ and output indices, excluding derived trees/Morton order, allocator overhead, source clouds and outputs. Voxel HashMap reductions allow only 1e-12 relative/absolute roundoff; voxel counts and inlier metrics are exact. Other applications and CPU power/clocks are uncontrolled. Supplied input provenance determines whether the pair represents independent surveys; measured-source even/odd subsets are not accuracy ground truth." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
