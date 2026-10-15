// Compare generation stamps / compact cell bits in immutable production builds.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2), allowed = ["--output", "--reference", "--compared", "--input-manifest", "--baseline-dist", "--repeats", "--methods"];
for (let i = 0; i < args.length; i += 2)
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output"), referencePath = value("--reference"), comparedPath = value("--compared");
const manifestPath = value("--input-manifest"), baselineRoot = value("--baseline-dist");
if (!output || !referencePath || !comparedPath || !manifestPath || !baselineRoot)
  throw new Error("Supply --baseline-dist DIST --reference FILE --compared FILE --input-manifest JSON --output NEWREPORT.json [--repeats 10] [--methods load,transform,icp]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats = Number(value("--repeats", "10")), warmupRounds = 5;
if (!Number.isInteger(repeats) || repeats < 1) throw new Error("Invalid repeats");
const methods = value("--methods", "load,transform,icp").split(",");
if (new Set(methods).size !== methods.length || methods.some(m => !["load", "transform", "icp"].includes(m))) throw new Error("Invalid methods");
if (!/^const COMPACT_CELL_MARKS: bool = true;/m.test(readFileSync("rust/crates/ca-core/src/octree.rs", "utf8")))
  throw new Error("Comparison requires enabled COMPACT_CELL_MARKS and a matching rebuilt SIMD production build");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const inputBytes = Object.fromEntries([["reference", referencePath], ["compared", comparedPath]].map(([role, path]) => [role, readFileSync(path)]));
const inputs = Object.fromEntries([["reference", referencePath], ["compared", comparedPath]].map(([role, path]) =>
  [role, { path, name: basename(path), bytes: inputBytes[role].length, sha256: digest(inputBytes[role]) }]));
const manifestBytes = readFileSync(manifestPath), inputManifest = JSON.parse(manifestBytes);
for (const role of ["reference", "compared"])
  if (inputManifest.inputs?.[role]?.sha256 !== inputs[role].sha256 || inputManifest.inputs[role].bytes !== inputs[role].bytes ||
    !Number.isInteger(inputManifest.inputs[role].points) || inputManifest.inputs[role].points < 1 || inputManifest.inputs[role].points > 2_000_000)
    throw new Error("Expected full measured inputs matching their manifest, at most two million points each");
const builds = {};
for (const [mode, root] of [["baseline", resolve(baselineRoot)], ["optimized", fileURLToPath(new URL("../dist/", import.meta.url))]]) {
  const names = readdirSync(resolve(root, "assets")).filter(n => /^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n));
  const worker = names.find(n => /^worker-/.test(n));
  if (!worker || names.length !== 3) throw new Error(`Expected three production worker/WASM artifacts in ${mode}`);
  builds[mode] = { root, worker, artifactHashes: Object.fromEntries(names.map(n => [n, digest(readFileSync(resolve(root, "assets", n)))])) };
}
if (JSON.stringify(builds.baseline.artifactHashes) === JSON.stringify(builds.optimized.artifactHashes)) throw new Error("Builds are identical");
const sourceFiles = ["rust/crates/ca-core/src/octree.rs", "rust/crates/ca-core/src/lib.rs", "rust/crates/ca-core/src/icp.rs",
  "rust/crates/ca-wasm/src/lib.rs", "web/src/worker.ts", "web/src/icp-preparation.ts"];
const sourceHashes = Object.fromEntries(sourceFiles.map(path => [path, digest(readFileSync(path))]));
const scriptSha256 = digest(readFileSync(fileURLToPath(import.meta.url)));
const settings = { maxIterations: 10, overlap: 0.8, matchCentroids: false, pointToPlane: true, icpCache: true, icpTemporal: false, sampleLimit: 50_000 };
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
  const report = await page.evaluate(async ({ builds, inputs, inputManifest, settings, repeats, warmupRounds, methods }) => {
    const files = {};
    for (const role of ["reference", "compared"]) files[role] = new File([await (await fetch(`/__${role}`)).arrayBuffer()], inputs[role].name);
    const context = async (mode, method) => {
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
      try { await call({ kind: "cpu-prepare" }); return { worker, call, load, reference: method === "load" ? undefined : await load("reference") }; }
      catch (e) { worker.terminate(); throw e; }
    };
    const same = (a, b) => {
      if (a == null || b == null) return a === b;
      if (!ArrayBuffer.isView(a) || !ArrayBuffer.isView(b) || a.constructor !== b.constructor) return false;
      const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
      return x.length === y.length && x.every((v, i) => v === y[i]);
    };
    const sameNumbers = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
    const record = (values, method) => {
      const a = values.baseline.result, b = values.optimized.result;
      const out = { baselineMillis: values.baseline.millis, optimizedMillis: values.optimized.millis };
      if (method === "load") Object.assign(out, { baselineIndexMillis: a.timings.index, optimizedIndexMillis: b.timings.index,
        baselineParseMillis: a.timings.parse, optimizedParseMillis: b.timings.parse,
        baselinePrepareMillis: a.timings.prepare, optimizedPrepareMillis: b.timings.prepare });
      if (method === "icp") Object.assign(out, { baselineWorkerMillis: a.millis, optimizedWorkerMillis: b.millis,
        referenceBytes: b.referenceBytes, iterations: b.iterations, rmsInitial: b.rmsInitial,
        rmsFinal: b.rmsFinal, converged: b.converged, matrix: b.matrix });
      return out;
    };
    const cases = [];
    for (const method of methods) {
      const contexts = {};
      try {
        for (const mode of ["baseline", "optimized"]) contexts[mode] = await context(mode, method);
        let matrix;
        const compute = async (mode, operation = method) => {
          const ctx = contexts[mode];
          const moving = operation === "load" ? undefined : await ctx.load("compared");
          const start = performance.now();
          const result = operation === "load" ? await ctx.load("compared") : operation === "transform" ?
            await ctx.call({ kind: "transform", id: moving.id, matrix }) :
            await ctx.call({ kind: "icp", moving: moving.id, reference: ctx.reference.id, ...settings });
          return { result, millis: performance.now() - start };
        };
        const verify = async (values, operation, warm) => {
          const a = values.baseline.result, b = values.optimized.result, x = a.cloud ?? a, y = b.cloud ?? b;
          if (x.kind !== "cloud" || y.kind !== "cloud" || x.count !== inputManifest.inputs.compared.points || x.count !== y.count ||
            !sameNumbers(x.bounds, y.bounds) || !sameNumbers(x.shift, y.shift) || x.lodGrid !== y.lodGrid ||
            JSON.stringify(x.scalarNames) !== JSON.stringify(y.scalarNames)) throw new Error("Cloud metadata changed");
          for (const cloud of [x, y]) if (!(cloud.positions instanceof Float32Array) || cloud.positions.length !== cloud.count * 3 ||
            !(cloud.lodNodes instanceof Float64Array) || !cloud.lodNodes.length) throw new Error("Complete indexed geometry unavailable");
          for (const key of ["positions", "colors", "intensity", "classification", "normals", "opacity", "lodNodes"])
            if (!same(x[key], y[key])) throw new Error(`Cloud ${key} bits changed`);
          const controls = {};
          if (operation === "load") for (const cloud of [x, y])
            if (!["parse", "index", "prepare"].every(k => Number.isFinite(cloud.timings[k]) && cloud.timings[k] >= 0)) throw new Error("Index timings unavailable");
          if (operation === "icp") {
            if (a.matrix.length !== 16 || b.matrix.length !== 16 || !sameNumbers(a.matrix, b.matrix) ||
              !["rmsInitial", "rmsFinal", "iterations", "converged"].every(k => Object.is(a[k], b[k]))) throw new Error("ICP fit changed");
            for (const fit of [a, b]) if (!fit.referenceCached || fit.referenceReused !== warm || !(fit.referenceBytes > 0) || fit.referenceBytes > 128 * 1024 * 1024)
              throw new Error("ICP reference cache control changed");
            if (a.referenceBytes !== b.referenceBytes) throw new Error("ICP retained capacity changed");
            Object.assign(controls, { baselineReferenceReused: a.referenceReused, optimizedReferenceReused: b.referenceReused });
          }
          const oldPly = await contexts.baseline.call({ kind: "export", id: x.id, format: "ply" });
          const newPly = await contexts.optimized.call({ kind: "export", id: y.id, format: "ply" });
          if (!same(oldPly, newPly)) throw new Error("Complete PLY bits changed");
          return { resultParity: true, exportedBytes: oldPly.length, points: x.count, ...controls };
        };
        const cleanup = async values => { for (const mode of ["baseline", "optimized"]) {
          const r = values[mode].result; await contexts[mode].call({ kind: "remove", id: (r.cloud ?? r).id });
        } };
        let fittedTransform;
        if (method === "transform") {
          const fits = { baseline: await compute("baseline", "icp"), optimized: await compute("optimized", "icp") };
          fittedTransform = { matrix: fits.baseline.result.matrix, ...await verify(fits, "icp", false) };
          matrix = fittedTransform.matrix; await cleanup(fits);
        }
        const setup = { baseline: await compute("baseline"), optimized: await compute("optimized") };
        const excludedFirstUse = { ...record(setup, method), ...await verify(setup, method, false) }; await cleanup(setup);
        const retained = setup.optimized.result.referenceBytes, warmups = [], trials = [];
        for (let i = -warmupRounds; i < repeats; i++) {
          const order = Math.abs(i) % 2 ? ["optimized", "baseline"] : ["baseline", "optimized"], values = {};
          for (const mode of order) values[mode] = await compute(mode);
          const parity = await verify(values, method, true);
          if (method === "icp" && values.optimized.result.referenceBytes !== retained) throw new Error("ICP capacity changed");
          (i < 0 ? warmups : trials).push({ order, ...record(values, method), ...parity }); await cleanup(values);
          await globalThis.benchmarkProgress(`${method} ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: stamps ${Math.round(values.baseline.millis)}/bits ${Math.round(values.optimized.millis)} ms; geometry/LOD/PLY parity`);
        }
        cases.push({ method, fittedTransform, excludedFirstUse, warmups, trials });
      } finally { for (const c of Object.values(contexts)) c.worker.terminate(); }
    }
    return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent,
      reference: { points: inputManifest.inputs.reference.points }, compared: { points: inputManifest.inputs.compared.points, keepEvery: 1 }, cases };
  }, { builds: Object.fromEntries(Object.entries(builds).map(([mode, b]) => [mode, { worker: b.worker }])), inputs, inputManifest, settings, repeats, warmupRounds, methods });
  for (const [path, hash] of [...Object.values(inputs).map(x => [x.path, x.sha256]), [manifestPath, digest(manifestBytes)],
    [fileURLToPath(import.meta.url), scriptSha256], ...Object.entries(sourceHashes)])
    if (digest(readFileSync(path)) !== hash) throw new Error(`Input/source/script changed during timing: ${path}`);
  for (const build of Object.values(builds)) for (const [name, hash] of Object.entries(build.artifactHashes))
    if (digest(readFileSync(resolve(build.root, "assets", name))) !== hash) throw new Error("Production artifact changed during timing");
  Object.assign(report, { schemaVersion: 1, comparison: "cpu-octree-compact-occupancy", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    builds, sourceHashes, scriptSha256, browserVersion: browser.version(),
    machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
    inputs, inputManifest, inputManifestSha256: digest(manifestBytes), settings, repeats, warmupRounds, methods,
    note: "Two immutable production builds compare generation-stamp versus compact-bit occupancy in the same octree builder. Full measured inputs unchanged, no thinning. Both ICP workers retain unchanged reference/tree/lazy-normal caches under the existing 128 MiB capacity budget; full-matrix accumulation remains standard. Load timing covers full load/parse/index/description/RPC and also records existing worker index/parse/prepare fields. Transform timing covers applying the actual ICP-fitted matrix, rebuilding the octree, description and RPC; its fresh load and untimed fitted-matrix derivation are excluded. ICP timing includes exact fitting, applying the transform, rebuilding the octree, description and RPC, with every fresh moving-cloud load outside timing. One excluded first-use pair per method, five excluded alternating warm-up pairs, ten alternating timed pairs unless overridden. Every pair requires identical full coordinate PLY bytes, render/attribute/LOD bits, bounds/shift/counts; ICP also requires identical matrix/RMS/iterations/convergence and cache capacity/reuse controls. Export/verification/removal excluded. Inputs/manifest/six sources/script/both production builds hashed before/after. No builds/tests/scans during timing. CPU clocks/power/GC/other apps uncontrolled. Even/odd source splits are not independent surveys or accuracy ground truth. No GPU adapter/model metadata collected." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" }); console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
