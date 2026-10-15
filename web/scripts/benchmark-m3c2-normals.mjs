// Compare bounded exact-normal reuse with fresh fitting on the same prepared trees.
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
    const warmParams = { ...params, projectionRadius: params.projectionRadius * 1.4, maxDepth: params.maxDepth * 1.4 };
    const compute = async (clouds, profile, cacheNormals, parallel = true) => {
      const start = performance.now();
      const result = await call({ kind: "m3c2", sharedSearch: false, fusedProjection: false, reference: clouds.reference.id, compared: clouds.compared.id, ...profile,
        parallel, interleaved: true, radiusBounds: false, mainWorker: true, preparedTrees: true, cacheNormals });
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
      const serial = [await compute(clouds, params, false, false), await compute(clouds, warmParams, false, false)];
      const serialBytes = [];
      for (const result of serial) serialBytes.push(await call({ kind: "export", id: result.cloud.id, format: "ply" }));
      const warmups = [], trials = [];
      for (let i = -warmupRounds; i < repeats; i++) {
        const reference = await call({ kind: "load", file: new File([referenceBytes], inputs.reference.name), maxPoints });
        if (reference.count !== clouds.reference.count || reference.keepEvery !== clouds.reference.keepEvery) throw new Error("Fresh reference changed point selection");
        const pair = { ...clouds, reference };
        // Prepare immutable trees outside timers, with no fitted-normal cache.
        const preparation = await compute(pair, params, false);
        if (!preparation.preparedTrees || preparation.treeCacheHit || preparation.normalCacheHit || preparation.normalCacheBytes)
          throw new Error("Expected a fresh tree pair with no normal cache");
        await call({ kind: "remove", id: preparation.cloud.id });
        const order = Math.abs(i) % 2 ? ["cached", "control"] : ["control", "cached"], results = {};
        for (const mode of order) {
          if (mode === "control") {
            results.controlCold = await compute(pair, params, false);
            results.controlWarm = await compute(pair, warmParams, false);
          } else {
            results.cold = await compute(pair, params, true);
            results.warm = await compute(pair, warmParams, true);
          }
        }
        const { controlCold, controlWarm, cold, warm } = results;
        if ([controlCold, controlWarm, cold, warm].some(v => !v.preparedTrees || !v.treeCacheHit || !v.mainWorker || v.workers !== cold.workers) ||
          controlCold.normalCacheHit || controlWarm.normalCacheHit || cold.normalCacheHit || !warm.normalCacheHit ||
          cold.normalCacheBytes !== cold.cloud.count * 56 || warm.normalCacheBytes !== cold.normalCacheBytes ||
          warm.normalCacheBytes > 64 * 1024 * 1024 || cold.workers < 2 || cold.workers > Math.min(4, Math.max(1, navigator.hardwareConcurrency - 1)))
          throw new Error("Normal reuse controls, allocation or worker limits changed");
        for (const [result, profile] of [[controlCold, 0], [cold, 0], [controlWarm, 1], [warm, 1]]) {
          parity(serial[profile], result);
          const exported = await call({ kind: "export", id: result.cloud.id, format: "ply" });
          if (exported.length !== serialBytes[profile].length || !exported.every((v, j) => v === serialBytes[profile][j])) throw new Error("Exported core coordinates/attributes differ");
        }
        const trial = { order, controlColdMillis: controlCold.elapsedMillis, coldMillis: cold.elapsedMillis,
          controlWarmMillis: controlWarm.elapsedMillis, warmMillis: warm.elapsedMillis,
          controlColdWorkerMillis: controlCold.millis, coldWorkerMillis: cold.millis,
          controlWarmWorkerMillis: controlWarm.millis, warmWorkerMillis: warm.millis,
          workers: cold.workers, treeCacheHit: true, coldNormalCacheHit: cold.normalCacheHit, warmNormalCacheHit: warm.normalCacheHit,
          normalCacheBytes: cold.normalCacheBytes, cores: cold.cloud.count,
          finiteDistances: [cold, warm].map(v => v.distance.filter(Number.isFinite).length),
          undefinedDistances: [cold, warm].map(v => v.distance.filter(Number.isNaN).length),
          distanceBitMismatches: 0, lod95BitMismatches: 0, significantBitMismatches: 0, exportedBytesEqual: true,
          exportedBytes: serialBytes.map(v => v.length) };
        (i < 0 ? warmups : trials).push(trial);
        for (const result of Object.values(results)) await call({ kind: "remove", id: result.cloud.id });
        await call({ kind: "remove", id: reference.id });
        await globalThis.benchmarkProgress(`${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: cold ${Math.round(controlCold.elapsedMillis)} -> ${Math.round(cold.elapsedMillis)}, changed projection ${Math.round(controlWarm.elapsedMillis)} -> ${Math.round(warm.elapsedMillis)} ms; ${cold.workers} workers, exact parity`);
      }
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent,
        reference: describe(clouds.reference), compared: describe(clouds.compared), warmParams,
        serialMillis: serial.map(v => v.elapsedMillis), cases: [{ scheduling: "auto", warmups, trials }] };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, inputs, inputManifest, params, repeats, warmupRounds, maxPoints });
  for (const input of Object.values(inputs)) if (digest(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (manifestPath && digest(readFileSync(manifestPath)) !== digest(manifestBytes)) throw new Error("Manifest changed during timing");
  for (const [name, hash] of Object.entries(artifactHashes)) if (digest(readFileSync(resolve(assets, name))) !== hash) throw new Error("Production artifact changed during timing");
  if (digest(readFileSync(fileURLToPath(import.meta.url))) !== scriptSha256) throw new Error("Benchmark script changed during timing");
  Object.assign(report, { schemaVersion: 1, comparison: "m3c2-bounded-normal-reuse", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    artifactHashes, scriptSha256, browserVersion: browser.version(), machine: { platform: platform(), release: release(),
      cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() }, inputs, inputManifest,
    inputManifestSha256: manifestBytes ? digest(manifestBytes) : undefined, params, maxPoints, warmupRounds, repeats,
    note: "Supplied measured point clouds with unchanged coordinates and original measured core selection. Whole files load by default; smaller max-points budgets select original file records and report counts/stride. One production analysis worker plus pool, unchanged lane policy capped at min(4, hardwareConcurrency-1). Each trial loads a fresh reference identity and prepares its exact immutable trees with normal reuse disabled outside timers. All four timed calls reuse these trees. Control fits normals afresh for profile A and for profile B (projection radius and half-depth multiplied by 1.4). Candidate cold A captures one ordered-core/normal set; warm B reuses those normals but recomputes both cylinders, distance and LoD95 for the changed projection. Five excluded warm-up trials precede ten trials with alternating control/candidate block order. The cache key includes every original f64 core bit in order, normal-radius bits and radius-pruning flag; it is local to the immutable tree pair. Normal fits including undefined results are retained with their validity tags; no distances, confidence bounds or final clouds are cached. Additional core/normal storage is bounded at 64 MiB across a prepared pair, 56 bytes per selected core; the unchanged 256 MiB input-tree scheduling estimate remains separate and excludes source/core/result storage, allocator/GC/retained heap pages, other workflows and concurrent jobs. Complete RPC timings include new core selection/attributes, binding copies, normal fitting or exact-key check, fresh projection/summary, joining and output indexing. Input loading, tree preparation, pool startup, validation and exports are excluded. Four timed results must match their corresponding excluded serial profile in f32 distance/LoD95/significance bits and complete PLY bytes. Source/manifest, script and production artifact hashes are checked before/after measurement; existing reports cannot be overwritten. Disabled controls skip normal reuse and can retain an already allocated normal set. CPU clocks, power, GC and other applications are uncontrolled. Measured-source subsets are not independent surveys or change-detection ground truth." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
