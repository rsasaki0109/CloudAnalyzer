// Real-source scalar vs SIMD voxel moments, standalone and with warm C2C.
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
  throw new Error("Supply --output REPORT.json --reference FILE --compared FILE [--input-manifest JSON] [--method both|voxel-scores|map-quality] [--repeats 10] [--threshold 0.15] [--voxel 1] [--min-points 10]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats = Number(value("--repeats", "10")), warmupRounds = 5;
const method = value("--method", "both");
if (!["both", "voxel-scores", "map-quality"].includes(method)) throw new Error("Invalid method");
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
const wasmName = readdirSync(assets).find(n => /^ca_wasm.*\.wasm$/.test(n));
const simd128Build = WebAssembly.Module.customSections(
  new WebAssembly.Module(readFileSync(resolve(assets, wasmName))), "target_features",
).some(section => Buffer.from(section).includes(Buffer.from("+\u0007simd128")));
if (!simd128Build) throw new Error("Expected a simd128 production WASM build with target_features metadata");
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
    const parity = (a, b, timed) => {
      if (a.distances) {
        if (a.cpuCached !== true || b.cpuCached !== true || timed && (!a.cpuCacheReused || !b.cpuCacheReused) ||
          a.cpuPipelined !== true || b.cpuPipelined !== true || a.workers !== b.workers || b.workers < 2 ||
          !Number.isSafeInteger(b.cpuCacheInputBytes) || a.cpuCacheInputBytes !== b.cpuCacheInputBytes)
          throw new Error("SIMD comparison changed nearest preparation/scheduling controls");
        const first = new Uint32Array(a.distances.buffer), second = new Uint32Array(b.distances.buffer);
        if (first.length !== second.length || first.some((bits, i) => bits !== second[i])) throw new Error("Distance bit parity failed");
        if (!Object.keys(a.stats).every(key => Object.is(a.stats[key], b.stats[key]))) throw new Error("Statistics changed");
      }
      let maxVoxelDifference = 0;
      const firstQuality = a.quality ?? a, secondQuality = b.quality ?? b;
      for (const key of a.quality ? Object.keys(a.quality) : ["awd", "scs", "voxels"]) {
        const x = firstQuality[key], y = secondQuality[key];
        if (Object.is(x, y)) continue;
        const delta = Math.abs(x - y);
        if (["awd", "scs"].includes(key) && delta <= 1e-12 * Math.max(1, Math.abs(x), Math.abs(y))) {
          maxVoxelDifference = Math.max(maxVoxelDifference, delta); continue;
        }
        throw new Error(`Quality changed: ${key}`);
      }
      return { distanceBitMismatches: a.distances ? 0 : undefined, statsExact: a.stats ? true : undefined,
        inlierMetricsExact: a.quality ? true : undefined, voxelCountExact: true, maxVoxelDifference };
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
      const compute = async (method, voxelSimd) => {
        const start = performance.now();
        const common = { compared: clouds.compared.id, reference: clouds.reference.id, ...params, voxelSimd, voxelCache: false, voxelReferenceCache: false };
        const result = await call(method === "voxel-scores" ? { kind: "map-quality", voxelCache: false, ...common } :
          { kind: "map-evaluate", voxelCache: false, ...common, cpuCache: true, cpuPipeline: true, streamSummaries: true });
        return { ...result, elapsedMillis: performance.now() - start };
      };
      const cases = [];
      for (const kind of method === "both" ? ["voxel-scores", "map-quality"] : [method]) {
        const warmups = [], trials = [];
        for (let i = -warmupRounds; i < repeats; i++) {
          const order = Math.abs(i) % 2 ? ["simd", "scalar"] : ["scalar", "simd"], results = {};
          for (const mode of order) {
            results[mode] = await compute(kind, mode === "simd");
          }
          const a = results.scalar, b = results.simd;
          const trial = { order, scalarMillis: a.elapsedMillis, simdMillis: b.elapsedMillis,
            scalarWorkerMillis: a.millis, simdWorkerMillis: b.millis, workers: b.workers,
            cacheInputBytes: b.cpuCacheInputBytes, stats: b.stats, quality: b.quality ?? { awd: b.awd, scs: b.scs, voxels: b.voxels },
            ...parity(a, b, i >= 0) };
          (i < 0 ? warmups : trials).push(trial);
          await globalThis.benchmarkProgress(`${kind} ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: scalar ${Math.round(a.elapsedMillis)}, SIMD ${Math.round(b.elapsedMillis)} ms; parity passed`);
        }
        cases.push({ method: kind, warmups, trials });
      }
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, reference, compared, cases };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, inputs, inputManifest, params, warmupRounds, repeats, method });
  for (const input of Object.values(inputs)) if (digest(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (manifestPath && digest(readFileSync(manifestPath)) !== digest(manifestBytes)) throw new Error("Manifest changed during timing");
  const hash = name => digest(readFileSync(resolve(assets, name)));
  Object.assign(report, { schemaVersion: 1, comparison: "map-quality-voxel-simd", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    workerSha256: hash(workerName), poolWorkerSha256: hash(readdirSync(assets).find(n => /^c2c-worker-.*\.js$/.test(n))),
    wasmSha256: hash(readdirSync(assets).find(n => /^ca_wasm.*\.wasm$/.test(n))), scriptSha256: digest(readFileSync(fileURLToPath(import.meta.url))),
    browserVersion: browser.version(), machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model,
      logicalCpus: cpus().length, memoryBytes: totalmem() }, inputs, inputManifest,
    inputManifestSha256: manifestBytes ? digest(manifestBytes) : undefined, params, method, warmupRounds, repeats,
    simd128Build,
    note: "Full supplied measured point clouds, no thinning, one production analysis worker and pool. Five warm-up pairs precede alternating sequential scalar/SIMD comparisons for each workflow. Standalone voxel scores rebuild both Gaussian maps and calculate fresh AWD/SCS. Timed combined Map quality uses warm retained nearest preparation, scalar exact f64 searches, queued dispatch, progressive joins, adaptive summaries and original f32 inlier arithmetic. The first combined warm-up pair includes initial nearest preparation in its first mode; that pair is excluded from timed comparisons. No Gaussian maps, distance arrays or voxel scores are cached. Only independent voxel moment channels change: SIMD performs two f64 components per instruction, with the same per-channel point order and separate multiply/add operations; voxel keys, occupancy, covariance conversion, Wasserstein eigensolvers and SCS arithmetic stay the same. Wall times include fresh computation and RPCs, plus searches, joins, summaries and metrics for combined Map quality. Loading, pool startup/warming, validation and exports are excluded; combined worker millis are recorded separately. Every comparison preserves voxel counts and allows only existing 1e-12 relative/absolute HashMap reduction roundoff in AWD/SCS. Combined comparisons also require all displayed distance bits, f64 statistics and inlier metrics to match exactly, with identical lane counts and retained input bytes. Source/manifest and artifact hashes are recorded; production WASM target_features metadata must declare simd128. Other applications, CPU clocks, power and GC are uncontrolled. Measured-source even/odd subsets are not independent surveys or accuracy ground truth." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
