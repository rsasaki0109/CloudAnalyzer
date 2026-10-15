// Compare separate and shared reference radius searches on measured inputs.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2);
const allowed = ["--output", "--reference", "--compared", "--input-manifest", "--repeats", "--normal-radius", "--projection-radius", "--max-depth", "--core-spacing", "--max-points"];
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
}
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output"), referencePath = value("--reference"), comparedPath = value("--compared");
if (!output || !referencePath || !comparedPath)
  throw new Error("Supply --output REPORT.json --reference FILE --compared FILE [--input-manifest JSON] [--repeats 10] [--normal-radius 0.4] [--projection-radius 0.25] [--max-depth 0.5] [--core-spacing 0.25] [--max-points 2000000]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats = Number(value("--repeats", "10")), warmupRounds = 5;
const maxPoints = Number(value("--max-points", "2000000"));
const params = { normalRadius: Number(value("--normal-radius", "0.4")), projectionRadius: Number(value("--projection-radius", "0.25")),
  maxDepth: Number(value("--max-depth", "0.5")), coreSpacing: Number(value("--core-spacing", "0.25")) };
if (!Number.isInteger(repeats) || repeats < 1 || !Number.isInteger(maxPoints) || maxPoints < 1 || maxPoints > 2_000_000 ||
  Object.entries(params).some(([key, v]) => !Number.isFinite(v) || (key === "coreSpacing" ? v < 0 : v <= 0)))
  throw new Error("Invalid repeats or M3C2 parameters");
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
const artifactNames = readdirSync(assets).filter(n => /^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n));
const workerName = artifactNames.find(n => /^worker-/.test(n));
if (!workerName || artifactNames.length !== 3) throw new Error("Build the production app first; expected exactly three worker/WASM artifacts");
const artifactHashes = Object.fromEntries(artifactNames.map(name => [name, digest(readFileSync(resolve(assets, name)))]));
const scriptSha256 = digest(readFileSync(fileURLToPath(import.meta.url)));
const server = createServer((req, res) => {
  if (req.url === "/__m3c2-radius") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
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
  await page.goto(`http://127.0.0.1:${server.address().port}/__m3c2-radius`);
  const report = await page.evaluate(async ({ workerPath, inputs, inputManifest, params, repeats, warmupRounds, maxPoints }) => {
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
      if (a.cloud.count !== b.cloud.count || a.radiusBounds || b.radiusBounds ||
        ["distance", "lod95", "significant"].some(key => !sameBits(a[key], b[key]))) throw new Error("M3C2 result bits changed");
    };
    const compute = async (clouds, parallel, sharedSearch) => {
      const start = performance.now();
      const result = await call({ kind: "m3c2", reference: clouds.reference.id, compared: clouds.compared.id, ...params,
        parallel, interleaved: true, radiusBounds: false, mainWorker: true, preparedTrees: true, cacheNormals: false, sharedSearch });
      return { ...result, elapsedMillis: performance.now() - start };
    };
    try {
      await call({ kind: "cpu-prepare" });
      const clouds = {};
      for (const role of ["reference", "compared"]) {
        const response = await fetch(`/__${role}`); if (!response.ok) throw new Error(`Input fetch failed: ${role}`);
        clouds[role] = await call({ kind: "load", file: new File([await response.arrayBuffer()], inputs[role].name), maxPoints });
        const cloud = clouds[role];
        if (cloud.kind !== "cloud" || !cloud.count || cloud.keepEvery !== Math.max(1, Math.ceil(cloud.filePoints / maxPoints)) ||
          cloud.count !== Math.ceil(cloud.filePoints / cloud.keepEvery) ||
          inputManifest && cloud.filePoints !== inputManifest.inputs[role].points) throw new Error(`Input point selection changed: ${role}`);
      }
      const describe = cloud => ({ points: cloud.count, filePoints: cloud.filePoints, keepEvery: cloud.keepEvery, bounds: cloud.bounds });
      const referenceBytes = await (await fetch("/__reference")).arrayBuffer();
      const serial = await compute(clouds, false, false);
      const serialBytes = await call({ kind: "export", id: serial.cloud.id, format: "ply" });
      const warmups = [], trials = [];
      for (let i = -warmupRounds; i < repeats; i++) {
        const order = Math.abs(i) % 2 ? ["shared", "separate"] : ["separate", "shared"], results = {};
        for (const mode of order) {
          // Each mode gets a fresh immutable identity; both cold calls build and transfer trees.
          const reference = await call({ kind: "load", file: new File([referenceBytes], inputs.reference.name), maxPoints });
          if (reference.count !== clouds.reference.count || reference.keepEvery !== clouds.reference.keepEvery) throw new Error("Fresh reference changed selection");
          const pair = { ...clouds, reference }, shared = mode === "shared";
          const cold = await compute(pair, true, shared), warm = await compute(pair, true, shared);
          if (!cold.preparedTrees || cold.treeCacheHit || !warm.preparedTrees || !warm.treeCacheHit ||
            !cold.mainWorker || !warm.mainWorker || cold.workers !== warm.workers || cold.workers < 2 ||
            cold.workers > Math.min(4, Math.max(1, navigator.hardwareConcurrency - 1)) ||
            cold.sharedSearch !== shared || warm.sharedSearch !== shared || cold.normalCacheHit || warm.normalCacheHit ||
            cold.normalCacheBytes || warm.normalCacheBytes || cold.radiusBounds || warm.radiusBounds)
            throw new Error("Shared-search/cache/scheduling controls changed");
          for (const result of [cold, warm]) {
            parity(serial, result);
            const exported = await call({ kind: "export", id: result.cloud.id, format: "ply" });
            if (exported.length !== serialBytes.length || !exported.every((v, j) => v === serialBytes[j])) throw new Error("Exported core coordinates/attributes differ");
          }
          results[mode] = { cold, warm };
          for (const result of [cold, warm]) await call({ kind: "remove", id: result.cloud.id });
          await call({ kind: "remove", id: reference.id });
        }
        const { separate, shared } = results;
        if (separate.cold.workers !== shared.cold.workers) throw new Error("Lane counts changed between controls");
        const inputPoints = clouds.reference.count + clouds.compared.count, pools = shared.cold.workers - 1;
        const inputBudgetBytes = inputPoints * Math.max(80, 108 + (pools - 1) * 104, 36 + pools * 104);
        if (inputBudgetBytes > 256 * 1024 * 1024) throw new Error("Input estimate exceeded budget");
        const trial = { order, separateColdMillis: separate.cold.elapsedMillis, sharedColdMillis: shared.cold.elapsedMillis,
          separateWarmMillis: separate.warm.elapsedMillis, sharedWarmMillis: shared.warm.elapsedMillis,
          separateColdWorkerMillis: separate.cold.millis, sharedColdWorkerMillis: shared.cold.millis,
          separateWarmWorkerMillis: separate.warm.millis, sharedWarmWorkerMillis: shared.warm.millis,
          workers: shared.cold.workers, mainWorker: true, coldCacheHit: false, warmCacheHit: true,
          normalCacheHit: false, normalCacheBytes: 0, inputBudgetBytes, cores: shared.cold.cloud.count,
          finiteDistances: shared.cold.distance.filter(Number.isFinite).length, undefinedDistances: shared.cold.distance.filter(Number.isNaN).length,
          distanceBitMismatches: 0, lod95BitMismatches: 0, significantBitMismatches: 0, exportedBytesEqual: true, exportedBytes: serialBytes.length };
        (i < 0 ? warmups : trials).push(trial);
        await globalThis.benchmarkProgress(`${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: cold ${Math.round(trial.separateColdMillis)}/${Math.round(trial.sharedColdMillis)}, fresh normals on warm trees ${Math.round(trial.separateWarmMillis)}/${Math.round(trial.sharedWarmMillis)} ms; ${trial.workers} workers, exact parity`);
      }
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent,
        reference: describe(clouds.reference), compared: describe(clouds.compared), serialMillis: serial.elapsedMillis,
        cases: [{ scheduling: "auto", warmups, trials }] };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, inputs, inputManifest, params, repeats, warmupRounds, maxPoints });
  for (const input of Object.values(inputs)) if (digest(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (manifestPath && digest(readFileSync(manifestPath)) !== digest(manifestBytes)) throw new Error("Manifest changed during timing");
  for (const [name, hash] of Object.entries(artifactHashes)) if (digest(readFileSync(resolve(assets, name))) !== hash) throw new Error("Production artifact changed during timing");
  if (digest(readFileSync(fileURLToPath(import.meta.url))) !== scriptSha256) throw new Error("Benchmark script changed during timing");
  Object.assign(report, { schemaVersion: 1, comparison: "m3c2-shared-reference-search", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    artifactHashes, scriptSha256, browserVersion: browser.version(), machine: { platform: platform(), release: release(),
      cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() }, inputs, inputManifest,
    inputManifestSha256: manifestBytes ? digest(manifestBytes) : undefined, params, maxPoints, warmupRounds, repeats,
    note: "Supplied measured coordinates and file order, with original measured core representatives. The default 2000000-point budget loads these inputs completely. Each mode uses a new reference identity for a timed cold prepared-tree call followed by a timed warm-tree call. Both fit fresh normals and recompute projections/results on every call; normal/result caches are disabled. Separate control issues reference normal-radius and reference projection-sphere queries independently. Shared mode combines those two searches, preserving each radius's own pruning and unsorted hit order while computing each visited point's squared distance once. Each lane uses an additional temporary vector of projection-sphere hits; it is reused within a slice and dropped after computation, with no persistent added cache. Full RPC timing includes core selection/attributes, tree construction/export/import when cold, bindings, fitting, projections, joining and output LOD indexing. Loading, removal, pool startup, serial baseline, parity validation and exports are excluded. Both modes must use the same lane count and match serial f32 distance/LoD95/significance bits and complete exported PLY bytes. Five excluded warm-up trials precede ten timed alternating sequential blocks unless repeats overridden. Radius bounds stay false. The existing 256 MiB input-storage estimate excludes source/core/results, temporary hit vectors, allocator overhead, retained WASM pages, other workflow caches and concurrent jobs; it is not a heap cap. Inputs, manifests, artifacts and script are hash-checked before and after. Other apps, power, CPU clocks and GC are uncontrolled. Even/odd measured-source subsets are not independent surveys or change-detection ground truth." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
