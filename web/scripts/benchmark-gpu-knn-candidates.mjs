// Current prepared CPU / GPU block hints / GPU nearest-k candidates, with an archived output oracle.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { readFileSync, readdirSync, existsSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2), allowed = ["--output", "--input", "--input-manifest", "--baseline-dist", "--repeats"];
for (let i = 0; i < args.length; i += 2) if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
  throw new Error(`Invalid or repeated option: ${args[i]}`);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output"), inputPath = value("--input"), manifestPath = value("--input-manifest"), baselinePath = value("--baseline-dist");
const repeats = Number(value("--repeats", "10")), warmupRounds = 5;
if (!output || !inputPath || !manifestPath || !baselinePath) throw new Error("Supply --baseline-dist DIST --input COMPARED.ply --input-manifest JSON --output NEW.json [--repeats 10]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
if (!Number.isInteger(repeats) || repeats < 1 || repeats > 100) throw new Error("Invalid repeats");
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const inputBytes = readFileSync(inputPath), manifestBytes = readFileSync(manifestPath), inputManifest = JSON.parse(manifestBytes);
const input = { path: inputPath, name: basename(inputPath), bytes: inputBytes.length, sha256: hash(inputBytes) };
if (inputManifest.inputs?.compared?.sha256 !== input.sha256 || inputManifest.inputs.compared.bytes !== input.bytes ||
    !Number.isInteger(inputManifest.inputs.compared.points) || inputManifest.inputs.compared.points < 200_000 || inputManifest.inputs.compared.points > 1_000_000)
  throw new Error("Expected a full measured compared input, 200000..1000000 points, matching its manifest");
const sourcePath = resolve(inputManifest.source.path);
if (hash(readFileSync(sourcePath)) !== inputManifest.source.sha256) throw new Error("Source differs from manifest");
const builds = {};
for (const [name, root] of [["baseline", resolve(baselinePath)], ["current", fileURLToPath(new URL("../dist/", import.meta.url))]]) {
  const assets = resolve(root, "assets"), files = readdirSync(assets).filter(n => /^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n));
  const worker = files.find(n => /^worker-/.test(n));
  if (!worker || files.length !== 3) throw new Error(`Expected three production artifacts in ${name}`);
  builds[name] = { root, worker, artifactHashes: Object.fromEntries(files.map(n => [n, hash(readFileSync(resolve(assets, n)))])) };
}
if (JSON.stringify(builds.baseline.artifactHashes) === JSON.stringify(builds.current.artifactHashes)) throw new Error("Expected an archived pre-prototype oracle");
const scriptPath = fileURLToPath(import.meta.url), scriptSha256 = hash(readFileSync(scriptPath));
const sourceFiles = ["web/src/gpu-knn-candidates.ts", "web/src/gpu-knn-blocks.ts", "web/src/gpu-candidates.ts", "web/src/knn-preparation.ts", "web/src/worker.ts", "web/src/c2c-worker.ts", "web/src/pool.ts",
  "rust/crates/ca-core/src/kdtree.rs", "rust/crates/ca-core/src/filter.rs", "rust/crates/ca-core/src/normals.rs", "rust/crates/ca-wasm/src/lib.rs"];
const sourceHashes = Object.fromEntries(sourceFiles.map(p => [p, hash(readFileSync(resolve("..", p)))]));
const server = createServer((req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  if (path === "/") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (path === "/__input") { res.setHeader("Content-Type", "application/octet-stream"); res.end(inputBytes); return; }
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
  browser = await chromium.launch({ channel: "chromium", headless: true,
    args: ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"] });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report = await page.evaluate(async ({ builds, input, points, repeats, warmupRounds }) => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error("Physical WebGPU is required");
    const adapterInfo = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, isFallbackAdapter: adapter.info.isFallbackAdapter };
    const file = new File([await (await fetch("/__input")).arrayBuffer()], input.name);
    const same = (a, b) => { const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength); return x.length === y.length && x.every((v, i) => v === y[i]); };
    const digest = async a => [...new Uint8Array(await crypto.subtle.digest("SHA-256", a))].map(v => v.toString(16).padStart(2, "0")).join("");
    const context = async mode => {
      const worker = new Worker(`/${mode}/assets/${builds[mode].worker}`, { type: "module" });
      let seq = 0; const pending = new Map();
      worker.onmessage = ({ data }) => {
        if (!data.response) return; const p = pending.get(data.seq); if (!p) return;
        pending.delete(data.seq); clearTimeout(p.timer); data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
      };
      worker.onerror = e => { for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new Error(e.message)); } pending.clear(); };
      const call = req => new Promise((resolve, reject) => {
        const id = ++seq, timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timeout: ${req.kind}`)); }, 180_000);
        pending.set(id, { resolve, reject, timer }); worker.postMessage({ seq: id, req });
      });
      try {
        await call({ kind: "cpu-prepare" });
        const cloud = await call({ kind: "load", file, maxPoints: 2_000_000 });
        if (cloud.count !== points || cloud.filePoints !== points || cloud.keepEvery !== 1 || cloud.kind !== "cloud") throw new Error("Input was thinned or changed");
        return { worker, cloud, call };
      } catch (error) { worker.terminate(); throw error; }
    };
    const orders = [["original", "webgpu-knn", "webgpu"], ["webgpu", "webgpu-knn", "original"], ["webgpu-knn", "original", "webgpu"],
      ["webgpu", "original", "webgpu-knn"], ["original", "webgpu", "webgpu-knn"], ["webgpu-knn", "webgpu", "original"]];
    const cases = []; let description;
    for (const method of ["normals", "sor"]) {
      const contexts = [];
      try {
        const baseline = await context("baseline"); contexts.push(baseline);
        const req = (ctx, mode) => method === "normals" ? { kind: "normals", id: ctx.cloud.id, k: 24, orientation: "up", knnHint: mode === "original" ? undefined : mode } :
          { kind: "filter", id: ctx.cloud.id, op: "sor", a: 24, b: 1.0, knnHint: mode === "original" ? undefined : mode };
        const oracle = await baseline.call(req(baseline, "original"));
        const expected = method === "normals" ? oracle : await baseline.call({ kind: "export", id: oracle.id, format: "ply" });
        const expectedHash = await digest(expected);
        baseline.worker.terminate();
        const current = await context("current"); contexts.push(current);
        description = { points, bounds: current.cloud.bounds, keepEvery: 1 };
        const compute = async mode => {
          const start = performance.now(), result = await current.call(req(current, mode)), elapsedMillis = performance.now() - start;
          const state = await current.call({ kind: "knn-hint-state" }), cache = await current.call({ kind: "knn-cache-state" });
          if (!cache || cache.pending || cache.points !== points || cache.bytes > 128 * 1024 * 1024) throw new Error("Expected bounded prepared CPU state");
          if (mode === "original" ? state !== null : !state || state.backend !== mode || state.fallback || state.points !== points || state.k !== (method === "normals" ? 24 : 25))
            throw new Error(`Unexpected backend/fallback for ${mode}: ${JSON.stringify(state)}`);
          if (mode === "webgpu-knn" && state.completeRows !== points) throw new Error("GPU nearest-k search returned incomplete rows");
          const bytes = method === "normals" ? result : await current.call({ kind: "export", id: result.id, format: "ply" });
          if (!same(expected, bytes)) throw new Error(`${method}/${mode} differs from archived production output`);
          if (method === "sor") await current.call({ kind: "remove", id: result.id });
          return { elapsedMillis, state, cache, resultParity: true, outputPoints: method === "normals" ? result.length / 3 : result.count };
        };
        const setup = {};
        for (const mode of orders[0]) setup[mode] = await compute(mode);
        const warmups = [], trials = [];
        for (let i = -warmupRounds; i < repeats; i++) {
          const order = orders[(i + warmupRounds) % orders.length], values = {};
          for (const mode of order) values[mode] = await compute(mode);
          if (!values.webgpu.state.referenceReused || !values["webgpu-knn"].state.referenceReused) throw new Error("Expected warm GPU buffers");
          (i < 0 ? warmups : trials).push({ order, ...values });
          await globalThis.benchmarkProgress(`${method} ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: CPU ${values.original.elapsedMillis.toFixed(1)}, GPU candidates ${values["webgpu-knn"].elapsedMillis.toFixed(1)}, GPU blocks ${values.webgpu.elapsedMillis.toFixed(1)} ms; parity`);
        }
        cases.push({ method, expectedHash, setup, warmups, trials });
      } finally { for (const c of contexts) c.worker.terminate(); }
    }
    return { adapter: adapterInfo, hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, ...description, cases };
  }, { builds: Object.fromEntries(Object.entries(builds).map(([k, b]) => [k, { worker: b.worker }])), input, points: inputManifest.inputs.compared.points, repeats, warmupRounds });
  for (const [name, build] of Object.entries(builds)) for (const [file, expected] of Object.entries(build.artifactHashes))
    if (hash(readFileSync(resolve(build.root, "assets", file))) !== expected) throw new Error(`Changed ${name} artifact`);
  for (const [p, expected] of Object.entries(sourceHashes)) if (hash(readFileSync(resolve("..", p))) !== expected) throw new Error(`Changed source ${p}`);
  if (hash(readFileSync(inputPath)) !== input.sha256 || hash(readFileSync(manifestPath)) !== hash(manifestBytes) || hash(readFileSync(sourcePath)) !== inputManifest.source.sha256 ||
      hash(readFileSync(scriptPath)) !== scriptSha256) throw new Error("Input/provenance/script changed during timing");
  const median = a => { a = [...a].sort((x, y) => x - y); return a.length % 2 ? a[a.length >> 1] : (a[a.length/2 - 1] + a[a.length/2]) / 2; };
  for (const c of report.cases) {
    c.summary = Object.fromEntries(["original", "webgpu-knn", "webgpu"].map(mode => [mode, { medianMillis: median(c.trials.map(t => t[mode].elapsedMillis)) }]));
    c.comparisons = Object.fromEntries([["original", "webgpu"], ["original", "webgpu-knn"], ["webgpu", "webgpu-knn"]].map(([control, candidate]) => {
      const reductions = c.trials.map(t => 100 * (1 - t[candidate].elapsedMillis / t[control].elapsedMillis));
      return [`${candidate}-vs-${control}`, { medianPairedReductionPercent: median(reductions), fasterTrials: reductions.filter(x => x > 0).length, trials: repeats }];
    }));
  }
  Object.assign(report, { schemaVersion: 1, comparison: "knn-nearest-candidates", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    builds, scriptSha256, sourceHashes, input, inputManifest, inputManifestSha256: hash(manifestBytes), repeats, warmupRounds, browserVersion: browser.version(),
    machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
    note: "Full unchanged measured compared input; no added/thinned coordinates. Normals k=24/up, SOR k=24/ratio=1.0; each method gets fresh contexts. Archived production output is a bit/PLY oracle. Current prepared CPU, physical GPU rounded blocks and physical GPU nearest-k candidates share one immutable source with six counterbalanced orders. Candidate search uses rounded f32 coordinates, private sorted arrays of up to 32 hits, a 32-entry DFS stack and at most 256 node visits per query; output is k tree-order u32 slots. CPU validates distinct finite in-range points, computes the maximum original f64 seed distance and traverses the original tree in historical order with the original split-plane tests plus strict upper-radius exclusions, including before the hit list fills. The candidate verifier does not read per-node boxes; the block control still does. Incomplete/invalid candidate rows use the original query; backend success proves GPU dispatch, not that every row supplied a valid radius. Timed runs require k non-sentinel GPU slots for every row; this completeness count is not an exact-neighbour certificate. Final membership, distances, tie order, normal PCA and SOR arithmetic remain authoritative on CPU. GPU blocks remain one u32 node per query. Whole RPC includes GPU transfer/readback, CPU verification, joining and SOR indexing. GPU phase time measures host submission/readback, not device timestamps. Retained optional bounds and both GPU reference sets are shared after warm-up; no peak-memory claim. Warm reuse required; job-level fallback fails. Loading, pool startup, first setup, five warm-up rounds, output comparison/export/removal excluded. All outputs match archived normal bits/complete SOR PLY bytes. Input/source/manifest/script/source-code/build hashes checked before and after. No builds/tests/scans alongside timing; clocks, power, other applications, scheduling and GC uncontrolled. Even/odd source splits are workload geometry, not accuracy ground truth. GPU identifiers limited to vendor/architecture/fallback." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" }); console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
