// Compare the M3C2 resident analysis lane with the existing pool-only policy.
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
const scheduling = "auto";
const maxPoints = Number(value("--max-points", "2000000"));
const params = { normalRadius: Number(value("--normal-radius", "0.4")), projectionRadius: Number(value("--projection-radius", "0.25")),
  maxDepth: Number(value("--max-depth", "0.5")), coreSpacing: Number(value("--core-spacing", "0.25")) };
if (!Number.isInteger(repeats) || repeats < 1 || !Number.isInteger(maxPoints) || maxPoints < 1 || maxPoints > 2_000_000 || !["auto", "serial", "both"].includes(scheduling) ||
  Object.entries(params).some(([key, v]) => !Number.isFinite(v) || (key === "coreSpacing" ? v < 0 : v <= 0)))
  throw new Error("Invalid repeats, scheduling or M3C2 parameters");
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
  const report = await page.evaluate(async ({ workerPath, inputs, inputManifest, params, repeats, warmupRounds, scheduling, maxPoints }) => {
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
      if (a.cloud.count !== b.cloud.count || a.mainWorker !== false || a.radiusBounds !== false || b.radiusBounds !== false ||
        b.mainWorker && b.workers !== a.workers + 1 || !b.mainWorker && b.workers !== a.workers ||
        ["distance", "lod95", "significant"].some(key => !sameBits(a[key], b[key]))) throw new Error("M3C2 controls or result bits changed");
      return { distanceBitMismatches: 0, lod95BitMismatches: 0, significantBitMismatches: 0 };
    };
    const compute = async (clouds, parallel, mainWorker) => {
      const start = performance.now();
      const result = await call({ kind: "m3c2", sharedSearch: false, reference: clouds.reference.id, compared: clouds.compared.id, ...params, parallel, interleaved: true, radiusBounds: false, mainWorker, preparedTrees: false });
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
      const cases = [];
      for (const mode of scheduling === "both" ? ["auto", "serial"] : [scheduling]) {
        const warmups = [], trials = [];
        for (let i = -warmupRounds; i < repeats; i++) {
          const order = Math.abs(i) % 2 ? ["resident", "poolOnly"] : ["poolOnly", "resident"], results = {};
          for (const kernel of order) results[kernel] = await compute(clouds, true, kernel === "resident");
          const a = results.poolOnly, b = results.resident;
          const match = parity(a, b);
          // Check result positions, attributes and LOD order outside the timer.
          const exportA = await call({ kind: "export", id: a.cloud.id, format: "ply" });
          const exportB = await call({ kind: "export", id: b.cloud.id, format: "ply" });
          if (exportA.length !== exportB.length || !exportA.every((v, j) => v === exportB[j])) throw new Error("Exported core coordinates/attributes differ");
          if (mode === "serial" && b.workers !== 1) throw new Error("Expected serial control");
          const inputPoints = clouds.reference.count + clouds.compared.count;
          const inputBudgetBytes = inputPoints * (b.mainWorker ? 80 + (b.workers - 1) * 128 : b.workers * 128);
          if (b.workers > 1 && inputBudgetBytes > 256 * 1024 * 1024) throw new Error("Parallel input estimate exceeds the existing budget");
          const trial = { order, poolOnlyMillis: a.elapsedMillis, residentMillis: b.elapsedMillis,
            poolOnlyWorkerMillis: a.millis, residentWorkerMillis: b.millis, controlWorkers: a.workers, workers: b.workers,
            mainWorker: b.mainWorker, inputBudgetBytes, cores: b.cloud.count,
            finiteDistances: b.distance.filter(Number.isFinite).length, undefinedDistances: b.distance.filter(Number.isNaN).length,
            ...match, exportedBytesEqual: true, exportedBytes: exportB.length };
          (i < 0 ? warmups : trials).push(trial);
          for (const result of [a, b]) await call({ kind: "remove", id: result.cloud.id });
          await globalThis.benchmarkProgress(`${mode} ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: pool-only ${Math.round(a.elapsedMillis)}, resident ${Math.round(b.elapsedMillis)} ms; ${a.workers}/${b.workers} workers, main ${b.mainWorker}, ${b.cloud.count} cores, exact parity`);
        }
        cases.push({ scheduling: mode, warmups, trials });
      }
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, reference: describe(clouds.reference), compared: describe(clouds.compared), cases };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, inputs, inputManifest, params, repeats, warmupRounds, scheduling, maxPoints });
  for (const input of Object.values(inputs)) if (digest(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (manifestPath && digest(readFileSync(manifestPath)) !== digest(manifestBytes)) throw new Error("Manifest changed during timing");
  for (const [name, hash] of Object.entries(artifactHashes)) if (digest(readFileSync(resolve(assets, name))) !== hash) throw new Error("Production artifact changed during timing");
  if (digest(readFileSync(fileURLToPath(import.meta.url))) !== scriptSha256) throw new Error("Benchmark script changed during timing");
  Object.assign(report, { schemaVersion: 1, comparison: "m3c2-resident-main-lane", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    artifactHashes, scriptSha256, browserVersion: browser.version(), machine: { platform: platform(), release: release(),
      cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() }, inputs, inputManifest,
    inputManifestSha256: manifestBytes ? digest(manifestBytes) : undefined, params, scheduling, maxPoints, warmupRounds, repeats,
    note: "Supplied measured point clouds, one production analysis worker and pool. The default 2000000-point budget loads these inputs completely; a smaller explicit budget selects every keepEvery-th original file point, recorded with original/loaded counts. Five warm-up pairs precede alternating sequential comparisons. Core spacing selects original measured points without averaging or generating coordinates. Every call times plan/core selection, snapshots/copying, tree construction, exact f64 M3C2 computation, joining, attributes, result LOD indexing and RPCs; loading, pool startup, validation and exports are excluded. No trees, normals or results are cached. Both modes disable experimental radius pruning and preserve the same kernel, cyclic partitioning and core selection. Pool-only uses the previous 128 bytes per input point per pool lane estimate. Resident mode uses the analysis worker only when its borrowed Cloud inputs admit one additional lane within the same 256 MiB estimate, using 80 bytes per input point for that lane's tree construction and 128 for each pool lane. Core/result/source storage, retained heap pages and concurrent jobs are excluded, as in the existing scheduling estimate; it is not a total memory cap. Worker totals stay within min(4, hardwareConcurrency-1). Every pair must match distance, LoD95, significance bits and complete exported PLY bytes. Source/manifest, production artifact and script hashes are checked before and after timing. Other applications, CPU clocks, power and GC are uncontrolled. These measured-source subsets are not independent surveys or change-detection ground truth." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
