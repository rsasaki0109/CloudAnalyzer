// Compare a shared bounded-query state with archived original or previous-radius ICP.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2), allowed = ["--output", "--reference", "--compared", "--input-manifest", "--baseline-dist", "--repeats", "--control"];
for (let i = 0; i < args.length; i += 2)
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output"), referencePath = value("--reference"), comparedPath = value("--compared");
const manifestPath = value("--input-manifest"), baselineRoot = value("--baseline-dist");
if (!output || !referencePath || !comparedPath || !manifestPath || !baselineRoot)
  throw new Error("Supply --baseline-dist DIST --reference FILE --compared FILE --input-manifest JSON --output NEWREPORT.json [--repeats 10]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats = Number(value("--repeats", "10")), warmupRounds = 5;
if (!Number.isInteger(repeats) || repeats < 1 || repeats > 100) throw new Error("Invalid repeats");
const controlMode = value('--control', 'original');
if (!['original','radius'].includes(controlMode)) throw new Error('Invalid control; choose original or radius');
if (!readFileSync('rust/crates/ca-core/src/kdtree.rs','utf8').includes('struct BoundedNearest')) throw new Error('Expected the shared bounded-query state');
if (!/const USE_SHARED_NEAREST_STATE:\s*bool\s*=\s*true\s*;/.test(readFileSync('rust/crates/ca-core/src/kdtree.rs','utf8')))
  throw new Error('Set USE_SHARED_NEAREST_STATE = true and rebuild matching production WASM/app for the shared-state comparison; the standard is false');
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const inputBytes = Object.fromEntries([["reference", referencePath], ["compared", comparedPath]].map(([role, path]) => [role, readFileSync(path)]));
const inputs = Object.fromEntries([["reference", referencePath], ["compared", comparedPath]].map(([role, path]) =>
  [role, { path, name: basename(path), bytes: inputBytes[role].length, sha256: digest(inputBytes[role]) }]));
const manifestBytes = readFileSync(manifestPath), inputManifest = JSON.parse(manifestBytes);
for (const role of ["reference", "compared"])
  if (inputManifest.inputs?.[role]?.sha256 !== inputs[role].sha256 || inputManifest.inputs[role].bytes !== inputs[role].bytes ||
    !Number.isInteger(inputManifest.inputs[role].points) || inputManifest.inputs[role].points < 1 || inputManifest.inputs[role].points > 2_000_000)
    throw new Error("Expected full measured inputs matching their manifest, at most two million points each");
const sourcePath = resolve('web', inputManifest.source?.path ?? '');
const sourceBytes = readFileSync(sourcePath);
if (sourceBytes.length !== inputManifest.source?.bytes || digest(sourceBytes) !== inputManifest.source.sha256)
  throw new Error('Original measured source differs from manifest');
const builds = {};
for (const [mode, root] of [["archived", resolve(baselineRoot)], ["baseline", resolve(baselineRoot)], ["optimized", fileURLToPath(new URL("../dist/", import.meta.url))]]) {
  const names = readdirSync(resolve(root, "assets")).filter(n => /^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n));
  const worker = names.find(n => /^worker-/.test(n));
  if (!worker || names.length !== 3) throw new Error(`Expected three production worker/WASM artifacts in ${mode}`);
  builds[mode] = { root, worker, artifactHashes: Object.fromEntries(names.map(n => [n, digest(readFileSync(resolve(root, "assets", n)))])) };
}
if (JSON.stringify(builds.archived.artifactHashes) === JSON.stringify(builds.optimized.artifactHashes)) throw new Error("Builds are identical");
const sourceFiles = ["rust/crates/ca-core/src/icp.rs", "rust/crates/ca-core/src/kdtree.rs", "rust/crates/ca-core/src/normals.rs",
  "rust/crates/ca-wasm/src/lib.rs", "web/src/worker.ts", "web/src/protocol.ts", "web/src/icp-preparation.ts"];
const sourceHashes = Object.fromEntries(sourceFiles.map(path => [path, digest(readFileSync(path))]));
const scriptSha256 = digest(readFileSync(fileURLToPath(import.meta.url)));
const settings = { maxIterations: 10, overlap: 0.8, matchCentroids: false, pointToPlane: true, icpCache: true, icpSimd: false, sampleLimit: 50_000 };
const server = createServer((req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  if (path === "/") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (path === "/__reference" || path === "/__compared") {
    res.setHeader("Content-Type", "application/octet-stream"); res.end(inputBytes[path.slice(3)]); return;
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
  const report = await page.evaluate(async ({ builds, inputs, inputManifest, settings, repeats, warmupRounds, controlMode }) => {
    const files = {};
    for (const role of ["reference", "compared"]) files[role] = new File([await (await fetch(`/__${role}`)).arrayBuffer()], inputs[role].name);
    const contexts = {};
    const context = async mode => {
      const worker = new Worker(`/${mode}/assets/${builds[mode].worker}`, { type: "module" });
      let seq = 0; const waiting = new Map();
      worker.onmessage = ({ data }) => { if (!data.response) return; const p = waiting.get(data.seq); if (!p) return;
        waiting.delete(data.seq); clearTimeout(p.timer); data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error)); };
      worker.onerror = e => { for (const p of waiting.values()) { clearTimeout(p.timer); p.reject(new Error(e.message)); } waiting.clear(); };
      const call = req => new Promise((resolve, reject) => { const id = ++seq, timer = setTimeout(() => { waiting.delete(id); reject(new Error(`Timeout: ${req.kind}`)); }, 180_000);
        waiting.set(id, { resolve, reject, timer }); worker.postMessage({ seq: id, req }); });
      const load = async role => {
        const cloud = await call({ kind: "load", file: files[role], maxPoints: 2_000_000 });
        if (cloud.kind !== "cloud" || cloud.count !== inputManifest.inputs[role].points || cloud.filePoints !== cloud.count || cloud.keepEvery !== 1)
          throw new Error("Input was thinned or changed");
        return cloud;
      };
      try { await call({ kind: "cpu-prepare" }); return { worker, call, load, reference: await load("reference") }; }
      catch (e) { worker.terminate(); throw e; }
    };
    const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
    const record = values => ({ baselineMillis: values.baseline.millis, optimizedMillis: values.optimized.millis,
      baselineWorkerMillis: values.baseline.result.millis, optimizedWorkerMillis: values.optimized.result.millis,
      referenceBytes: values.optimized.result.referenceBytes, iterations: values.optimized.result.iterations,
      rmsInitial: values.optimized.result.rmsInitial, rmsFinal: values.optimized.result.rmsFinal,
      converged: values.optimized.result.converged, matrix: values.optimized.result.matrix });
    try {
      for (const mode of ["archived", "baseline", "optimized"]) contexts[mode] = await context(mode);
      const compute = async mode => {
        const ctx = contexts[mode], moving = await ctx.load("compared");
        const start = performance.now();
        const result = await ctx.call({ kind: "icp", moving: moving.id, reference: ctx.reference.id, ...settings, icpTemporal: mode === "optimized" || (mode === "baseline" && controlMode === "radius") });
        return { result, millis: performance.now() - start };
      };
      const verify = async (values, warm) => {
        const a = values.baseline.result, b = values.optimized.result;
        if (a.correspondenceSimd !== false || b.correspondenceSimd !== false ||
          (a.correspondenceTemporal ?? false) !== (controlMode === "radius") || b.correspondenceTemporal !== true ||
          (controlMode === "radius" ? a.temporalQueries !== b.temporalQueries || a.temporalScratchBytes !== b.temporalScratchBytes : (a.temporalQueries ?? 0) !== 0 || (a.temporalScratchBytes ?? 0) !== 0) ||
          !(b.temporalQueries > 0) || !Number.isInteger(b.temporalQueries) ||
          b.temporalQueries > b.iterations * b.temporalScratchBytes / 4 ||
          b.temporalScratchBytes !== 4 * Math.ceil(inputManifest.inputs.compared.points / Math.ceil(inputManifest.inputs.compared.points / 50000)))
          throw new Error('Explicit temporal controls were not executed or accounting changed');
        if (a.matrix.length !== 16 || b.matrix.length !== 16 || !a.matrix.every((v, i) => Object.is(v, b.matrix[i])) ||
          !["rmsInitial", "rmsFinal", "iterations", "converged"].every(k => Object.is(a[k], b[k]))) throw new Error("ICP transform/RMS/convergence changed");
        for (const x of [a, b]) if (!x.referenceCached || x.referenceReused !== warm || !(x.referenceBytes > 0) || x.referenceBytes > 128 * 1024 * 1024)
          throw new Error("ICP reference cache control changed");
        if (a.referenceBytes !== b.referenceBytes || a.cloud.count !== b.cloud.count || a.cloud.count !== inputManifest.inputs.compared.points)
          throw new Error("ICP capacities or point counts differ");
        const x = await contexts.baseline.call({ kind: "export", id: a.cloud.id, format: "ply" });
        const y = await contexts.optimized.call({ kind: "export", id: b.cloud.id, format: "ply" });
        if (!same(x, y)) throw new Error("ICP transformed PLY bytes changed");
        return { resultParity: true, exportedBytes: x.length, points: a.cloud.count,
          baselineReferenceReused: a.referenceReused, optimizedReferenceReused: b.referenceReused, controlTemporal: a.correspondenceTemporal ?? false, controlQueries: a.temporalQueries ?? 0, contextTemporal: b.correspondenceTemporal, boundedQueries: b.temporalQueries, temporalScratchBytes: b.temporalScratchBytes };
      };
      const cleanup = async values => { for (const mode of ["baseline", "optimized"]) await contexts[mode].call({ kind: "remove", id: values[mode].result.cloud.id }); };
      const archive = await compute('archived');
      const setup = { baseline: await compute("baseline"), optimized: await compute("optimized") };
      const a = archive.result, b = setup.baseline.result;
      if (!a.matrix.every((v, i) => Object.is(v, b.matrix[i])) ||
        !['rmsInitial', 'rmsFinal', 'iterations', 'converged', 'referenceBytes'].every(k => Object.is(a[k], b[k])) ||
        !a.referenceCached || a.referenceReused || a.cloud.count !== b.cloud.count) throw new Error('Archived scalar result changed');
      const oldPly = await contexts.archived.call({ kind: 'export', id: a.cloud.id, format: 'ply' });
      const newPly = await contexts.baseline.call({ kind: 'export', id: b.cloud.id, format: 'ply' });
      if (!same(oldPly, newPly)) throw new Error('Archived transformed PLY changed');
      const archivedOracle = { millis: archive.millis, iterations: a.iterations, matrix: a.matrix,
        rmsInitial: a.rmsInitial, rmsFinal: a.rmsFinal, converged: a.converged, referenceBytes: a.referenceBytes,
        exportedBytes: oldPly.length, resultParity: true };
      await contexts.archived.call({ kind: 'remove', id: a.cloud.id });
      const excludedFirstUse = { ...record(setup), ...await verify(setup, false) }; await cleanup(setup);
      const retained = setup.optimized.result.referenceBytes, warmups = [], trials = [];
      for (let i = -warmupRounds; i < repeats; i++) {
        const order = Math.abs(i) % 2 ? ["optimized", "baseline"] : ["baseline", "optimized"], values = {};
        for (const mode of order) values[mode] = await compute(mode);
        const parity = await verify(values, true);
        if (values.optimized.result.referenceBytes !== retained) throw new Error("ICP retained capacity changed");
        (i < 0 ? warmups : trials).push({ order, ...record(values), ...parity }); await cleanup(values);
        await globalThis.benchmarkProgress(`ICP ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: ${controlMode} ${Math.round(values.baseline.millis)}/context ${Math.round(values.optimized.millis)} ms; result/PLY parity`);
      }
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent,
        reference: { points: contexts.baseline.reference.count, bounds: contexts.baseline.reference.bounds },
        compared: { points: inputManifest.inputs.compared.points, keepEvery: 1 }, archivedOracle, excludedFirstUse, warmups, trials };
    } finally { for (const c of Object.values(contexts)) c.worker.terminate(); }
  }, { builds: Object.fromEntries(Object.entries(builds).map(([mode, b]) => [mode, { worker: b.worker }])), inputs, inputManifest, settings, repeats, warmupRounds, controlMode });
  for (const [path, hash] of [...Object.values(inputs).map(x => [x.path, x.sha256]), [manifestPath, digest(manifestBytes)], [sourcePath, digest(sourceBytes)],
    [fileURLToPath(import.meta.url), scriptSha256], ...Object.entries(sourceHashes)])
    if (digest(readFileSync(path)) !== hash) throw new Error(`Input/source/script changed during timing: ${path}`);
  for (const build of Object.values(builds)) for (const [name, hash] of Object.entries(build.artifactHashes))
    if (digest(readFileSync(resolve(build.root, "assets", name))) !== hash) throw new Error("Production artifact changed during timing");
  Object.assign(report, { schemaVersion: 1, comparison: "cpu-icp-shared-radius-context", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    builds, sourceHashes, scriptSha256, browserVersion: browser.version(),
    machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
    inputs, inputManifest, inputManifestSha256: digest(manifestBytes), settings, repeats, warmupRounds, controlMode,
    note: "The current explicit temporal-radius kernel shares one call-local incumbent/radius state across recursive tree nodes instead of forwarding another f64 parameter at each node. The archived original control selects icpTemporal:false; --control radius enables the previous radius implementation in its archived build. Both use scalar leaves, the existing unchanged reference/tree/lazy-normal cache under 128 MiB, maximum ten iterations and the existing 50000-point stride sample limit. Every call loads a fresh unchanged full measured moving cloud before timing; previous-hit slots are reused only within a call and freed before return. An excluded archived scalar oracle verifies exact transform/RMS/iterations/convergence and transformed PLY. One excluded first-use pair, five excluded alternating warm-up pairs and ten alternating timed pairs unless overridden; every pair verifies exact results/all PLY bytes, full input counts, equal bounded cache capacities and expected controls. Radius-control comparisons additionally require identical tighter-radius counts and temporary slot bytes. A tighter radius is not a branch-saving count. Whole ICP RPC includes matching, overlap selection, normals, fitting, transformation and indexing; input load/pool startup, exports, validation and removal are excluded. Input/manifest/original measured source/seven implementation sources/script/current and archived builds are hashed before/after. No builds/tests/scans during timing. CPU clocks/power/GC/other applications are uncontrolled. Source subsets are not independent surveys or registration ground truth. No GPU adapter/model metadata collected." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" }); console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
