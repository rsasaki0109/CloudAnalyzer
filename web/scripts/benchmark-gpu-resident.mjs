// Real-source comparison: the same exact lane count, with/without a resident lane.
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
  if (req.url === "/__gpu-resident") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
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
  await page.goto(`http://127.0.0.1:${server.address().port}/__gpu-resident`);
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
    const controls = (result, main, reused) => {
      const c = result.compute;
      if (c?.backend !== "webgpu" || c.fallback || c.verifyFallback || !c.referenceReused || !c.deviceReused || c.verifyReferenceReused !== reused ||
        c.verifyWorkers !== expectedWorkers || c.verifyPoolWorkers !== expectedWorkers - Number(main) || c.verifyMainWorker !== main ||
        !c.verifySnapshot || c.verifyStagedQueries || !c.verifyBounds || !c.verifySimd || !c.prefetch || c.workgroupSize !== 64)
        throw new Error("Resident comparison changed verification controls or fell back");
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
      let referenceDescription;
      const compute = async (ref, main, lanes = workers) => {
        const start = performance.now();
        const result = await call({ kind: "c2c", gpuReuseImportBounds: false, reference: ref.id, compared: compared.id, signed: false, backend: "webgpu",
          gpuVerifyWorkers: lanes, gpuVerifySnapshot: true, gpuStageQueries: false, gpuMainWorker: main });
        return { ...result, elapsedMillis: performance.now() - start };
      };
      const warmups = [], trials = [];
      for (let i = -warmupRounds; i < repeats; i++) {
        const order = Math.abs(i) % 2 ? ["resident", "poolOnly"] : ["poolOnly", "resident"], results = {};
        let exactBaseline;
        for (const mode of order) {
          // Each mode gets a fresh immutable reference, so resident setup never
          // retains the extra pool-only lane. Main GPU setup is excluded equally.
          const ref = await load("reference"); referenceDescription = describe(ref);
          const serial = await compute(ref, false, 1);
          const cold = await compute(ref, mode === "resident"); controls(cold, mode === "resident", false); parity(serial, cold);
          const warm = await compute(ref, mode === "resident"); controls(warm, mode === "resident", true); parity(cold, warm);
          if (exactBaseline) parity(exactBaseline, cold); else exactBaseline = cold;
          results[mode] = { cold, warm };
          await call({ kind: "remove", id: ref.id });
        }
        const a = results.poolOnly, b = results.resident;
        if (a.cold.compute.batchSize !== b.cold.compute.batchSize || a.cold.compute.batches !== b.cold.compute.batches)
          throw new Error("Batch scheduling changed");
        const trial = { order, workers: expectedWorkers, stats: a.cold.stats, certified: a.cold.compute.certified,
          distanceBitMismatches: 0, exactStatsEqual: true, certifiedEqual: true,
          cold: { poolOnlyMillis: a.cold.elapsedMillis, residentMillis: b.cold.elapsedMillis,
            controlCompute: a.cold.compute, optimizedCompute: b.cold.compute },
          warm: { poolOnlyMillis: a.warm.elapsedMillis, residentMillis: b.warm.elapsedMillis,
            controlCompute: a.warm.compute, optimizedCompute: b.warm.compute } };
        (i < 0 ? warmups : trials).push(trial);
        await globalThis.benchmarkProgress(`${i < 0 ? "Warm-up" : "Trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: cold ${Math.round(a.cold.elapsedMillis)}/${Math.round(b.cold.elapsedMillis)}, warm ${Math.round(a.warm.elapsedMillis)}/${Math.round(b.warm.elapsedMillis)} ms; exact parity`);
      }
      return { adapter: adapterInfo, hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, poolPrepareMillis,
        reference: referenceDescription, compared: describe(compared), warmups, trials };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, inputs, inputManifest, repeats, warmupRounds, workers, maxPoints });
  for (const input of Object.values(inputs)) if (digest(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (manifestPath && digest(readFileSync(manifestPath)) !== digest(manifestBytes)) throw new Error("Manifest changed during timing");
  for (const [name, hash] of Object.entries(artifactHashes)) if (digest(readFileSync(resolve(assets, name))) !== hash) throw new Error("Production artifact changed during timing");
  if (digest(readFileSync(fileURLToPath(import.meta.url))) !== scriptSha256) throw new Error("Benchmark script changed during timing");
  Object.assign(report, { schemaVersion: 1, comparison: "gpu-c2c-resident-main-lane", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    artifactHashes, scriptSha256, browserVersion: browser.version(), machine: { platform: platform(), release: release(),
      cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() }, inputs, inputManifest,
    inputManifestSha256: manifestBytes ? digest(manifestBytes) : undefined, workers, maxPoints, warmupRounds, repeats,
    note: "Measured file coordinates, physical WebGPU candidates and authoritative f64 verification. Five warm-up pairs precede alternating timed pairs. Identical total lane counts; resident mode replaces the last pool lane with the analysis worker's existing immutable tree/queries. Each mode reloads identical reference bytes under a fresh identity and warms main GPU preparation with an excluded serial run; cold verification copies/imports only the required pool references. A warm verification call immediately follows on that same reference. Timings include job snapshots, required reference/query transfers, GPU readback, exact verification, joining, summaries and RPC; loading, CPU pool/device/main tree preparation and parity checks are excluded. Every pair, including cold/warm/serial controls, must match displayed f32 distance bits, reported f64 statistics and exact certificate counts. GPU phase intervals overlap and must not be summed. Input/manifest/artifact/script hashes are checked before/after. CPU/GPU clocks, power, GC and other applications are uncontrolled; no cross-machine guarantee. Measured source splits are not independent surveys or ground truth." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
