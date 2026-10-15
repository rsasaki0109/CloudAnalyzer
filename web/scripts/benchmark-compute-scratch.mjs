// Compare immutable production builds: scratch, f64 SIMD and immutable reference/score reuse.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2);
const allowed = ["--output", "--reference", "--compared", "--input-manifest", "--baseline-dist", "--repeats", "--methods"];
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
}
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output"), referencePath = value("--reference"), comparedPath = value("--compared");
const manifestPath = value("--input-manifest"), baselineRoot = value("--baseline-dist");
if (!output || !referencePath || !comparedPath || !manifestPath || !baselineRoot)
  throw new Error("Supply --baseline-dist DIST --reference FILE --compared FILE --input-manifest JSON --output REPORT.json [--repeats 10] [--methods m3c2,normals,sor,icp,map-evaluate]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats = Number(value("--repeats", "10")), warmupRounds = 5;
const methods = value("--methods", "m3c2,normals,sor,icp,map-evaluate").split(",");
if (!Number.isInteger(repeats) || repeats < 1 || new Set(methods).size !== methods.length ||
  methods.some(m => !["m3c2", "normals", "sor", "icp", "map-evaluate"].includes(m))) throw new Error("Invalid repeats or methods");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const inputs = Object.fromEntries([["reference", referencePath], ["compared", comparedPath]].map(([role, path]) => {
  const bytes = readFileSync(path);
  return [role, { path, name: basename(path), bytes: bytes.length, sha256: digest(bytes) }];
}));
const manifestBytes = readFileSync(manifestPath), inputManifest = JSON.parse(manifestBytes);
if (["reference", "compared"].some(role => inputManifest.inputs?.[role]?.sha256 !== inputs[role].sha256 ||
  inputManifest.inputs[role].bytes !== inputs[role].bytes || !inputManifest.inputs[role].points || inputManifest.inputs[role].points > 2_000_000))
  throw new Error("Expected full measured inputs matching their manifest, at most two million points each");
const builds = {};
for (const [mode, root] of [["baseline", resolve(baselineRoot)], ["optimized", fileURLToPath(new URL("../dist/", import.meta.url))]]) {
  const assets = resolve(root, "assets");
  const names = readdirSync(assets).filter(n => /^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n));
  const worker = names.find(n => /^worker-/.test(n));
  if (!worker || names.length !== 3) throw new Error(`Expected three production worker/WASM artifacts in ${mode}`);
  builds[mode] = { root, worker, artifactHashes: Object.fromEntries(names.map(n => [n, digest(readFileSync(resolve(assets, n)))])) };
}
if (JSON.stringify(builds.baseline.artifactHashes) === JSON.stringify(builds.optimized.artifactHashes)) throw new Error("Builds are identical");
const scriptSha256 = digest(readFileSync(fileURLToPath(import.meta.url)));
const server = createServer((req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  if (path === "/") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (["/__reference", "/__compared"].includes(path)) {
    const input = inputs[path === "/__reference" ? "reference" : "compared"], bytes = readFileSync(input.path);
    if (digest(bytes) !== input.sha256) { res.writeHead(409).end("Input changed"); return; }
    res.setHeader("Content-Type", "application/octet-stream"); res.end(bytes); return;
  }
  const mode = path.split("/")[1], build = builds[mode];
  if (!build) { res.writeHead(404).end(); return; }
  const target = resolve(build.root, "." + path.slice(mode.length + 1));
  if (!target.startsWith(build.root.endsWith(sep) ? build.root : build.root + sep)) { res.writeHead(403).end(); return; }
  try { res.setHeader("Content-Type", ({ ".js": "text/javascript", ".wasm": "application/wasm" })[extname(target)] || "application/octet-stream"); res.end(readFileSync(target)); }
  catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ channel: "chromium", headless: true });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report = await page.evaluate(async ({ builds, inputs, inputManifest, methods, repeats, warmupRounds }) => {
    const files = {};
    for (const role of ["reference", "compared"]) {
      const response = await fetch(`/__${role}`); if (!response.ok) throw new Error(`Input fetch failed: ${role}`);
      files[role] = new File([await response.arrayBuffer()], inputs[role].name);
    }
    const context = async mode => {
      const worker = new Worker(`/${mode}/assets/${builds[mode].worker}`, { type: "module" });
      let seq = 0;
      const waiting = new Map();
      worker.onmessage = ({ data }) => {
        if (!data.response) return;
        const p = waiting.get(data.seq); if (!p) return;
        waiting.delete(data.seq); clearTimeout(p.timer);
        data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
      };
      worker.onerror = event => { for (const p of waiting.values()) { clearTimeout(p.timer); p.reject(new Error(event.message)); } waiting.clear(); };
      const call = req => new Promise((resolve, reject) => {
        const id = ++seq, timer = setTimeout(() => { waiting.delete(id); reject(new Error(`Worker timeout: ${req.kind}`)); }, 180_000);
        waiting.set(id, { resolve, reject, timer }); worker.postMessage({ seq: id, req });
      });
      const load = async role => {
        const cloud = await call({ kind: "load", file: files[role], maxPoints: 2_000_000 });
        if (cloud.kind !== "cloud" || cloud.count !== inputManifest.inputs[role].points || cloud.filePoints !== cloud.count || cloud.keepEvery !== 1)
          throw new Error("Input was thinned or changed");
        return cloud;
      };
      try {
        await call({ kind: "cpu-prepare" });
        return { worker, call, load, reference: await load("reference"), compared: await load("compared") };
      } catch (error) { worker.terminate(); throw error; }
    };
    const sameBits = (a, b) => {
      const x = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      return x.length === y.length && x.every((v, i) => v === y[i]);
    };
    const exact = (a, b) => Object.keys(a).every(k => Object.is(a[k], b[k]));
    const cases = [];
    let description;
    for (const method of methods) {
      const contexts = {};
      try {
        for (const mode of ["baseline", "optimized"]) contexts[mode] = await context(mode);
        description = Object.fromEntries(["reference", "compared"].map(role => {
          const c = contexts.baseline[role]; return [role, { points: c.count, filePoints: c.filePoints, keepEvery: c.keepEvery, bounds: c.bounds }];
        }));
        const compute = async ctx => {
          const { reference, compared, call } = ctx;
          let moving;
          if (method === "icp") moving = await ctx.load("compared");
          const req = method === "m3c2" ? { kind: method, reference: reference.id, compared: compared.id,
            normalRadius: 0.4, projectionRadius: 0.25, maxDepth: 0.5, coreSpacing: 0.25,
            parallel: true, interleaved: true, radiusBounds: false, mainWorker: true, preparedTrees: true, sharedSearch: true } :
            method === "normals" ? { kind: method, id: compared.id, k: 24, orientation: "up" } :
            method === "sor" ? { kind: "filter", id: compared.id, op: "sor", a: 24, b: 1.0 } :
            method === "icp" ? { kind: method, moving: moving.id, reference: reference.id, maxIterations: 10, overlap: 0.8, matchCentroids: false, pointToPlane: true, icpTemporal: false } :
            { kind: method, compared: compared.id, reference: reference.id, voxel: 1.0, minPoints: 5, threshold: 0.5,
              voxelSimd: false, cpuCache: true, cpuPipeline: true, streamSummaries: true };
          const start = performance.now();
          const result = await call(req);
          return { result, elapsedMillis: performance.now() - start, outputId: method === "sor" ? result.id : result.cloud?.id };
        };
        const verify = async (a, b, timed) => {
          let maxVoxelDifference = 0;
          if (method === "normals") {
            if (!sameBits(a.result, b.result)) throw new Error("Normal f32 bits changed");
          } else if (method === "m3c2") {
            for (const key of ["distance", "lod95", "significant"]) if (!sameBits(a.result[key], b.result[key])) throw new Error(`M3C2 ${key} bits changed`);
            for (const x of [a.result, b.result]) if (!x.treeCacheHit || !x.preparedTrees || !x.mainWorker || !x.sharedSearch || x.radiusBounds)
              throw new Error("M3C2 scheduling/cache controls changed");
            if (a.result.workers !== b.result.workers) throw new Error("M3C2 lane counts differ");
            if (a.result.normalCacheHit || a.result.normalCacheBytes) throw new Error("Baseline normal cache changed");
            if (timed && (!b.result.normalCacheHit || !(b.result.normalCacheBytes > 0) || b.result.normalCacheBytes > 64*1024*1024)) throw new Error("Expected bounded automatic normal reuse");
          } else if (method === "icp") {
            if (!a.result.matrix.every((v, i) => Object.is(v, b.result.matrix[i])) ||
              !["rmsInitial", "rmsFinal", "iterations", "converged"].every(k => Object.is(a.result[k], b.result[k]))) throw new Error("ICP transform/RMS/convergence changed");
            if (!b.result.referenceCached || !b.result.referenceReused || !(b.result.referenceBytes > 0) || b.result.referenceBytes > 128*1024*1024) throw new Error("ICP reference cache control changed");
          } else if (method === "map-evaluate") {
            const x = a.result, y = b.result;
            if (!sameBits(x.distances, y.distances) || !exact(x.stats, y.stats) || x.workers !== y.workers || !x.cpuCacheReused || !y.cpuCacheReused ||
              x.cpuCacheInputBytes !== y.cpuCacheInputBytes) throw new Error("Map quality distance/statistics/cache controls changed");
            if (!y.voxelCacheHit) throw new Error("Expected warm voxel score reuse");
            for (const key of Object.keys(x.quality)) {
              const u = x.quality[key], v = y.quality[key];
              if (["awd", "scs"].includes(key)) {
                const d = Math.abs(u - v);
                if (!Object.is(u, v) && !(Number.isNaN(u) && Number.isNaN(v)) && !(d <= 1e-12 * Math.max(1, Math.abs(u), Math.abs(v)))) throw new Error("Voxel reduction precision changed");
                if (Number.isFinite(d)) maxVoxelDifference = Math.max(maxVoxelDifference, d);
              } else if (!Object.is(u, v)) throw new Error(`Map quality ${key} changed`);
            }
          }
          let exportedBytes = 0;
          if (["m3c2", "sor", "icp"].includes(method)) {
            const x = await contexts.baseline.call({ kind: "export", id: a.outputId, format: "ply" });
            const y = await contexts.optimized.call({ kind: "export", id: b.outputId, format: "ply" });
            if (x.length !== y.length || x.some((v, i) => v !== y[i])) throw new Error(`${method} exported coordinates/attributes changed`);
            exportedBytes = x.length;
          }
          return { resultParity: true, exportedBytes, maxVoxelDifference };
        };
        const disposeResult = async (mode, x) => {
          if (x.outputId !== undefined) await contexts[mode].call({ kind: "remove", id: x.outputId });
        };
        // Record excluded first-use observations, then compare complete warm RPCs.
        const setup = {};
        for (const mode of ["baseline", "optimized"]) {
          const x = await compute(contexts[mode]);
          setup[mode] = { elapsedMillis: x.elapsedMillis, referenceCached: x.result.referenceCached,
            referenceReused: x.result.referenceReused, referenceBytes: x.result.referenceBytes, voxelCacheHit: x.result.voxelCacheHit };
          await disposeResult(mode, x);
        }
        const warmups = [], trials = [];
        for (let i = -warmupRounds; i < repeats; i++) {
          const order = Math.abs(i) % 2 ? ["optimized", "baseline"] : ["baseline", "optimized"], values = {};
          for (const mode of order) values[mode] = await compute(contexts[mode]);
          const a = values.baseline, b = values.optimized, parity = await verify(a, b, i >= 0);
          const trial = { order, baselineMillis: a.elapsedMillis, optimizedMillis: b.elapsedMillis, ...parity,
            points: method === "normals" ? a.result.length / 3 : method === "sor" ? a.result.count : a.result.cloud?.count,
            workers: a.result.workers, baselineWorkerMillis: a.result.millis, optimizedWorkerMillis: b.result.millis,
            optimizedReferenceReused: b.result.referenceReused, optimizedReferenceBytes: b.result.referenceBytes, optimizedVoxelCacheHit: b.result.voxelCacheHit,
            optimizedNormalCacheHit: b.result.normalCacheHit, optimizedNormalCacheBytes: b.result.normalCacheBytes, quality: a.result.quality, iterations: a.result.iterations, rmsInitial: a.result.rmsInitial, rmsFinal: a.result.rmsFinal };
          (i < 0 ? warmups : trials).push(trial);
          for (const mode of ["baseline", "optimized"]) await disposeResult(mode, values[mode]);
          await globalThis.benchmarkProgress(`${method} ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: baseline ${Math.round(a.elapsedMillis)}/optimized ${Math.round(b.elapsedMillis)} ms; parity`);
        }
        cases.push({ method, excludedSetup: setup, warmups, trials });
      } finally { for (const c of Object.values(contexts)) c.worker.terminate(); }
    }
    return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, ...description, cases };
  }, { builds: Object.fromEntries(Object.entries(builds).map(([mode, b]) => [mode, { worker: b.worker }])), inputs, inputManifest, methods, repeats, warmupRounds });
  for (const input of Object.values(inputs)) if (digest(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (digest(readFileSync(manifestPath)) !== digest(manifestBytes)) throw new Error("Manifest changed during timing");
  for (const b of Object.values(builds)) for (const [name, hash] of Object.entries(b.artifactHashes))
    if (digest(readFileSync(resolve(b.root, "assets", name))) !== hash) throw new Error("Production artifact changed during timing");
  if (digest(readFileSync(fileURLToPath(import.meta.url))) !== scriptSha256) throw new Error("Script changed during timing");
  Object.assign(report, { schemaVersion: 3, comparison: "compute-scratch-simd-and-immutable-reuse", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    builds, scriptSha256, browserVersion: browser.version(), machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model,
      logicalCpus: cpus().length, memoryBytes: totalmem() }, inputs, inputManifest, inputManifestSha256: digest(manifestBytes), repeats, warmupRounds,
    note: "Full unchanged measured-source files, two isolated production analysis workers with the same pool policy and calculation parameters. Baseline is an immutable archived production build from immediately before these core changes; optimized is the current production build. The adopted candidate retains nearest-k/coordinate scratch and SCS neighbourhood scratch. Paired f64 SIMD leaf/covariance kernels and invariant cylinder geometry remain comparison APIs after mixed whole-call timings. Automatic bounded M3C2 normal reuse activates only after a successful call with the same normal radius, core spacing, pruning and scheduling settings; the first repeated call captures normals and later calls reuse them under the existing 64 MiB budget. The baseline omits cacheNormals too, using its historical default false. Timed M3C2 requires normal cache hits in the adopted build; projection queries/statistics are recomputed. Its worker also retains one immutable ICP reference/tree/lazy-normal cache under a 128 MiB capacity budget, and three voxel scores for one immutable pair/settings. Every coordinate edit/removal invalidates the relevant entry. Timed ICP uses the warmed reference and lazy normals; timed Map quality recomputes nearest distances/statistics/inliers and reuses matching AWD/SCS/voxel scores. First-use observations are recorded as excluded setup, not cold-start medians. Earlier prototype reports retain their original schema/fingerprints. Both reuse exact M3C2 trees and Map quality nearest preparation after excluded setup. Each method gets fresh worker contexts and full inputs; five excluded alternating warm-up pairs precede ten timed pairs unless repeats overridden. Wall times include complete computation and RPC; SOR includes output indexing and ICP includes source transformation. ICP reloads an unchanged measured moving cloud before each call, with loading excluded, then fits point-to-plane for at most ten iterations at 0.8 overlap. Normal/SOR k=24, M3C2 normal/projection/depth/core spacing=0.4/0.25/0.5/0.25 m; map voxel/minimum occupancy/inlier threshold=1 m/5/0.5 m, scalar moments. Every pair checks all normal/M3C2/display distance bits, ICP transform/RMS/iterations/convergence, exact inlier metrics/statistics and byte-identical PLY for M3C2/SOR/ICP. Map AWD/SCS retain the existing 1e-12 relative/absolute HashMap reduction tolerance. Validation/exports/cleanup and loading/pool startup/setup are outside timing; no other builds/tests/scans run concurrently. Source/manifest/both-build/script hashes are checked before/after. Two contexts retain independent trees/heaps; each existing copy budget applies separately and is not a total process-memory cap. CPU clocks, power, GC and other applications are uncontrolled. Even/odd measured-source splits are not independent surveys or accuracy/registration ground truth. No GPU adapter or model information is collected." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
