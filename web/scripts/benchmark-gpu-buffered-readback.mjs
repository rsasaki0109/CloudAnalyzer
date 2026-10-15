// Real-source comparison: one or two reusable WebGPU readback slots.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2);
const allowed = ["--output", "--reference", "--compared", "--input-manifest", "--repeats", "--workers", "--max-points"];
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
}
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output"), referencePath = value("--reference"), comparedPath = value("--compared");
if (!output || !referencePath || !comparedPath) throw new Error("Supply --output REPORT.json --reference FILE --compared FILE [--input-manifest JSON] [--repeats 10] [--workers 4] [--max-points 2000000]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats = Number(value("--repeats", "10")), workers = Number(value("--workers", "4"));
const maxPoints = Number(value("--max-points", "2000000")), warmupRounds = 5;
if (!Number.isInteger(repeats) || repeats < 1 || !Number.isInteger(workers) || workers < 2 || workers > 8 ||
  !Number.isInteger(maxPoints) || maxPoints < 1 || maxPoints > 2_000_000) throw new Error("Invalid repeats, workers or point budget");
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
const names = readdirSync(assets).filter(n => /^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n));
const workerName = names.find(n => /^worker-/.test(n));
if (!workerName || names.length !== 3) throw new Error("Build the production app first; expected three worker/WASM artifacts");
const artifactHashes = Object.fromEntries(names.map(name => [name, digest(readFileSync(resolve(assets, name)))]));
const scriptSha256 = digest(readFileSync(fileURLToPath(import.meta.url)));
const server = createServer((req, res) => {
  if (req.url === "/__gpu-buffered-readback") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (["/__reference", "/__compared"].includes(req.url)) {
    const input = inputs[req.url === "/__reference" ? "reference" : "compared"], bytes = readFileSync(input.path);
    if (digest(bytes) !== input.sha256) { res.writeHead(409).end("Input changed after hashing"); return; }
    res.setHeader("Content-Type", "application/octet-stream"); res.end(bytes); return;
  }
  const path = resolve(root, "." + new URL(req.url, "http://localhost").pathname);
  if (!path.startsWith(root.endsWith(sep) ? root : root + sep)) { res.writeHead(403).end(); return; }
  try { res.setHeader("Content-Type", ({ ".js": "text/javascript", ".wasm": "application/wasm" })[extname(path)] || "application/octet-stream"); res.end(readFileSync(path)); }
  catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ channel: "chromium", headless: true,
    args: ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"] });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/__gpu-buffered-readback`);
  const report = await page.evaluate(async ({ workerPath, inputs, inputManifest, repeats, warmupRounds, workers, maxPoints }) => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error("A physical WebGPU adapter is required");
    const adapterInfo = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, isFallbackAdapter: adapter.info.isFallbackAdapter };
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const pending = waiting.get(data.seq); if (!pending) return;
      waiting.delete(data.seq); clearTimeout(pending.timer);
      data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const pending of waiting.values()) { clearTimeout(pending.timer); pending.reject(new Error(event.message)); } waiting.clear(); };
    const call = req => new Promise((resolve, reject) => {
      const id = ++seq, timer = setTimeout(() => { waiting.delete(id); reject(new Error(`Worker timeout: ${req.kind}`)); }, 180_000);
      waiting.set(id, { resolve, reject, timer }); worker.postMessage({ seq: id, req });
    });
    const sameBits = (a, b) => {
      const x = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      return x.length === y.length && x.every((v, i) => v === y[i]);
    };
    const parity = (a, b) => {
      if (!sameBits(a.distances, b.distances) || Object.keys(a.stats).some(k => !Object.is(a.stats[k], b.stats[k])) ||
        a.compute.certified !== b.compute.certified) throw new Error("Distance bits, exact statistics or certificate counts changed");
    };
    const expectedWorkers = Math.min(workers, Math.max(1, navigator.hardwareConcurrency - 1));
    if (expectedWorkers < 2) throw new Error("At least two verification lanes are required");
    const controls = (result, bufferedReadback, reused = true) => {
      const c = result.compute;
      if (c?.backend !== "webgpu" || c.fallback || c.verifyFallback || !c.referenceReused || !c.deviceReused || c.verifyReferenceReused !== reused ||
        c.verifyWorkers !== expectedWorkers || c.verifyPoolWorkers !== expectedWorkers || c.verifyMainWorker !== false || !c.verifyReuseImportBounds ||
        !c.verifySnapshot || c.verifyStagedQueries || !c.verifyBounds || !c.verifySimd || !c.prefetch || c.workgroupSize !== 64 || c.bufferedReadback !== bufferedReadback)
        throw new Error("Readback controls changed or fell back");
    };
    try {
      const start = performance.now(); await call({ kind: "cpu-prepare" });
      const poolPrepareMillis = performance.now() - start;
      const files = {};
      for (const role of ["reference", "compared"]) {
        const response = await fetch(`/__${role}`); if (!response.ok) throw new Error(`Input fetch failed: ${role}`);
        files[role] = new File([await response.arrayBuffer()], inputs[role].name);
      }
      const load = async role => {
        const cloud = await call({ kind: "load", file: files[role], maxPoints });
        if (cloud.kind !== "cloud" || !cloud.count || cloud.keepEvery !== Math.max(1, Math.ceil(cloud.filePoints / maxPoints)) ||
          cloud.count !== Math.ceil(cloud.filePoints / cloud.keepEvery) || inputManifest && cloud.filePoints !== inputManifest.inputs[role].points)
          throw new Error(`Point selection changed: ${role}`);
        return cloud;
      };
      const compared = await load("compared"), describe = cloud => ({ points: cloud.count, filePoints: cloud.filePoints, keepEvery: cloud.keepEvery, bounds: cloud.bounds });
      const reference = await load("reference");
      const compute = async (bufferedReadback, lanes = workers) => {
        const start = performance.now();
        const result = await call({ kind: "c2c", reference: reference.id, compared: compared.id, signed: false, backend: "webgpu",
          gpuVerifyWorkers: lanes, gpuVerifySnapshot: true, gpuStageQueries: false, gpuMainWorker: false,
          gpuReuseImportBounds: true, gpuBufferedReadback: bufferedReadback });
        return { ...result, elapsedMillis: performance.now() - start };
      };
      // Prepare the same immutable reference, exact pool and both buffer slots
      // before timing. No fresh source identities or cold worker imports in pairs.
      const serial = await compute(false, 1);
      const prepared = await compute(false); controls(prepared, false, false); parity(serial, prepared);
      const buffered = await compute(true); controls(buffered, true); parity(serial, buffered);
      const warmups = [], trials = [];
      for (let i = -warmupRounds; i < repeats; i++) {
        const order = Math.abs(i) % 2 ? ["buffered", "single"] : ["single", "buffered"], results = {};
        for (const mode of order) {
          const result = await compute(mode === "buffered"); controls(result, mode === "buffered"); parity(serial, result);
          results[mode] = result;
        }
        const a = results.single, b = results.buffered; parity(a, b);
        if (a.compute.batchSize !== b.compute.batchSize || a.compute.batches !== b.compute.batches)
          throw new Error("Batch scheduling changed");
        const trial = { order, workers: expectedWorkers, stats: a.stats, certified: a.compute.certified,
          distanceBitMismatches: 0, exactStatsEqual: true, certifiedEqual: true,
          singleMillis: a.elapsedMillis, bufferedMillis: b.elapsedMillis,
          controlCompute: a.compute, optimizedCompute: b.compute };
        (i < 0 ? warmups : trials).push(trial);
        await globalThis.benchmarkProgress(`${i < 0 ? "Warm-up" : "Trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: single ${Math.round(a.elapsedMillis)}/buffered ${Math.round(b.elapsedMillis)} ms; exact parity`);
      }
      return { adapter: adapterInfo, hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, poolPrepareMillis,
        reference: describe(reference), compared: describe(compared),
        excludedSetup: { serialMillis: serial.elapsedMillis, singleMillis: prepared.elapsedMillis, bufferedMillis: buffered.elapsedMillis }, warmups, trials };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, inputs, inputManifest, repeats, warmupRounds, workers, maxPoints });
  for (const input of Object.values(inputs)) if (digest(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (manifestPath && digest(readFileSync(manifestPath)) !== digest(manifestBytes)) throw new Error("Manifest changed during timing");
  for (const [name, hash] of Object.entries(artifactHashes)) if (digest(readFileSync(resolve(assets, name))) !== hash) throw new Error("Production artifact changed during timing");
  if (digest(readFileSync(fileURLToPath(import.meta.url))) !== scriptSha256) throw new Error("Benchmark script changed during timing");
  Object.assign(report, { schemaVersion: 1, comparison: "gpu-c2c-buffered-readback", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    artifactHashes, scriptSha256, browserVersion: browser.version(), machine: { platform: platform(), release: release(),
      cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() }, inputs, inputManifest,
    inputManifestSha256: manifestBytes ? digest(manifestBytes) : undefined, workers, maxPoints, warmupRounds, repeats,
    note: "Measured file coordinates, physical WebGPU candidates and authoritative f64 verification. Both modes use the same immutable reference identity and ready exact worker snapshots, validated import-bound reuse, pool lane count, bounds, SIMD, batch size, prefetch, workgroup and summaries. Single mode retains historical submission/readback serialization. Buffered mode retains two independent input/output/readback buffer triples, closes each submission's device scopes before waiting for mapping, and queues at most two batches ahead of CPU verification. Excluded setup prepares the main tree/device/upload, pool references and both buffer capacities. Each timed call includes query snapshots, candidate upload/submission/readback, exact verification, joining, summaries and RPC. Five excluded alternating warm-up pairs precede ten alternating timed pairs unless repeats overridden. Every result matches serial displayed f32 distance bits, exact reported f64 statistics and certificate counts; cross-mode parity and all controls are checked. GPU phase spans can overlap other GPU batches and CPU verification; do not sum them or interpret gpuWaitMillis as shader time. Inputs, manifests, selected artifacts and script hashes are checked before/after; output is never overwritten. Adapter metadata omits model/description/device identifiers. CPU/GPU clocks, power, GC and other applications are uncontrolled. Source splits are not independent surveys or ground truth." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
