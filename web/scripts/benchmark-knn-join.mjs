// Whole production RPCs and isolated joins on unchanged measured-source inputs.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";
import { stripTypeScriptTypes } from "node:module";

const args = process.argv.slice(2);
const allowed = ["--output", "--input", "--input-manifest", "--baseline-dist", "--repeats"];
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
}
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output"), inputPath = value("--input"), manifestPath = value("--input-manifest"), baselineRoot = value("--baseline-dist");
if (!output || !inputPath || !manifestPath || !baselineRoot)
  throw new Error("Supply --baseline-dist DIST --input COMPARED.ply --input-manifest JSON --output NEWREPORT.json [--repeats 10]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats = Number(value("--repeats", "10")), warmupRounds = 5;
if (!Number.isInteger(repeats) || repeats < 1) throw new Error("Invalid repeats");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const inputBytes = readFileSync(inputPath), manifestBytes = readFileSync(manifestPath), inputManifest = JSON.parse(manifestBytes);
const input = { path: inputPath, name: basename(inputPath), bytes: inputBytes.length, sha256: digest(inputBytes) };
if (inputManifest.inputs?.compared?.sha256 !== input.sha256 || inputManifest.inputs.compared.bytes !== input.bytes ||
  !inputManifest.inputs.compared.points || inputManifest.inputs.compared.points > 2_000_000)
  throw new Error("Expected the full measured compared input matching its manifest, at most two million points");
const builds = {};
for (const [mode, root] of [["baseline", resolve(baselineRoot)], ["optimized", fileURLToPath(new URL("../dist/", import.meta.url))]]) {
  const names = readdirSync(resolve(root, "assets")).filter(n => /^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n));
  const worker = names.find(n => /^worker-/.test(n)), wasm = names.find(n => n.endsWith(".wasm"));
  if (!worker || !wasm || names.length !== 3) throw new Error(`Expected three worker/WASM artifacts in ${mode}`);
  builds[mode] = { root, worker, wasm, artifactHashes: Object.fromEntries(names.map(n => [n, digest(readFileSync(resolve(root, "assets", n)))])) };
}
const hashOf = (build, pattern) => Object.entries(build.artifactHashes).find(([n]) => pattern.test(n))?.[1];
if (hashOf(builds.baseline, /^worker-/) === hashOf(builds.optimized, /^worker-/) ||
  hashOf(builds.baseline, /^c2c-worker-/) !== hashOf(builds.optimized, /^c2c-worker-/) ||
  hashOf(builds.baseline, /\.wasm$/) !== hashOf(builds.optimized, /\.wasm$/))
  throw new Error("The comparison must change only the main worker, with identical lane kernels and WASM");
const sourcePath = fileURLToPath(new URL("../src/knn-join.ts", import.meta.url));
const gluePath = fileURLToPath(new URL("../src/wasm/ca_wasm.js", import.meta.url));
const sourceBytes = readFileSync(sourcePath), glueBytes = readFileSync(gluePath);
const sourceSha256 = digest(sourceBytes), glueSha256 = digest(glueBytes), scriptSha256 = digest(readFileSync(fileURLToPath(import.meta.url)));
const helperModule = stripTypeScriptTypes(sourceBytes.toString("utf8"), { mode: "strip" });
const server = createServer((req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  if (path === "/") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (path === "/__input") { res.setHeader("Content-Type", "application/octet-stream"); res.end(inputBytes); return; }
  if (path === "/__join.js" || path === "/__wasm.js") { res.setHeader("Content-Type", "text/javascript"); res.end(path === "/__join.js" ? helperModule : glueBytes); return; }
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
  const report = await page.evaluate(async ({ builds, input, points, repeats, warmupRounds }) => {
    const bytes = new Uint8Array(await (await fetch("/__input")).arrayBuffer()), file = new File([bytes], input.name);
    const sameBits = (a, b) => {
      const x = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      return x.length === y.length && x.every((v, i) => v === y[i]);
    };
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
      worker.onerror = e => { for (const p of waiting.values()) { clearTimeout(p.timer); p.reject(new Error(e.message)); } waiting.clear(); };
      const call = req => new Promise((resolve, reject) => {
        const id = ++seq, timer = setTimeout(() => { waiting.delete(id); reject(new Error(`Worker timeout: ${req.kind}`)); }, 180_000);
        waiting.set(id, { resolve, reject, timer }); worker.postMessage({ seq: id, req });
      });
      try {
        await call({ kind: "cpu-prepare" });
        const cloud = await call({ kind: "load", file, maxPoints: 2_000_000 });
        if (cloud.kind !== "cloud" || cloud.count !== points || cloud.filePoints !== points || cloud.keepEvery !== 1) throw new Error("Input was thinned or changed");
        return { worker, call, cloud };
      } catch (e) { worker.terminate(); throw e; }
    };
    const cases = [];
    let description;
    for (const method of ["normals", "sor"]) {
      const contexts = {};
      try {
        for (const mode of ["baseline", "optimized"]) contexts[mode] = await context(mode);
        description = { points: contexts.baseline.cloud.count, bounds: contexts.baseline.cloud.bounds, keepEvery: 1 };
        const compute = async mode => {
          const { call, cloud } = contexts[mode], start = performance.now();
          const result = await call(method === "normals" ? { kind: method, id: cloud.id, k: 24, orientation: "up" } : { kind: "filter", id: cloud.id, op: "sor", a: 24, b: 1.0 });
          return { result, elapsedMillis: performance.now() - start };
        };
        const verify = async values => {
          const a = values.baseline.result, b = values.optimized.result;
          if (method === "normals") { if (!sameBits(a, b)) throw new Error("Normal f32 bits changed"); return { resultParity: true, outputPoints: a.length / 3 }; }
          const x = await contexts.baseline.call({ kind: "export", id: a.id, format: "ply" }), y = await contexts.optimized.call({ kind: "export", id: b.id, format: "ply" });
          if (a.count !== b.count || x.length !== y.length || x.some((v, i) => v !== y[i])) throw new Error("SOR kept points/coordinates/attributes changed");
          return { resultParity: true, outputPoints: a.count, exportedBytes: x.length };
        };
        const cleanup = async values => { if (method === "sor") for (const mode of ["baseline", "optimized"]) await contexts[mode].call({ kind: "remove", id: values[mode].result.id }); };
        const setup = { baseline: await compute("baseline"), optimized: await compute("optimized") };
        const setupParity = await verify(setup);
        await cleanup(setup);
        const warmups = [], trials = [];
        for (let i = -warmupRounds; i < repeats; i++) {
          const order = Math.abs(i) % 2 ? ["optimized", "baseline"] : ["baseline", "optimized"], values = {};
          for (const mode of order) values[mode] = await compute(mode);
          const parity = await verify(values);
          (i < 0 ? warmups : trials).push({ order, baselineMillis: values.baseline.elapsedMillis, optimizedMillis: values.optimized.elapsedMillis, ...parity });
          await cleanup(values);
          await globalThis.benchmarkProgress(`${method} ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: ${Math.round(values.baseline.elapsedMillis)}/${Math.round(values.optimized.elapsedMillis)} ms; parity`);
        }
        cases.push({ method, excludedFirstUse: { baselineMillis: setup.baseline.elapsedMillis, optimizedMillis: setup.optimized.elapsedMillis, ...setupParity }, warmups, trials });
      } finally { for (const c of Object.values(contexts)) c.worker.terminate(); }
    }
    // Isolate the JavaScript joins using real partition/kernel outputs. No RPC
    // trial or worker pool is running during these sequential diagnostics.
    const wasm = await import("/__wasm.js"), join = await import("/__join.js");
    await wasm.default({ module_or_path: `/optimized/assets/${builds.optimized.wasm}` });
    const cloud = wasm.Cloud.parse(input.name, bytes);
    const prepared = [];
    let plan;
    const legacyPack = (local, indices, n) => {
      const means = new Float64Array(n), lists = [], openIndex = [], asked = Array.from({ length: local.length }, () => []);
      local.forEach((part, i) => {
        for (let j = 0; j < part.means.length; j++) if (!Number.isNaN(part.means[j])) means[indices[i][j]] = part.means[j];
        for (let o = 0; o < part.open.length; o++) {
          const q = lists.length; lists.push([part.candidates.subarray(o * 25, (o + 1) * 25)]); openIndex.push(indices[i][part.open[o]]);
          const [x, y, z] = part.openPoints.subarray(o * 3, o * 3 + 3);
          for (let j = 0; j < local.length; j++) if ((part.reach[o] >>> j) & 1) asked[j].push({ query: q, at: [x, y, z] });
        }
      });
      return { means, lists, openIndex, asked, queries: asked.map(a => Float64Array.from(a.flatMap(x => x.at))) };
    };
    const legacyFinish = (answers, p) => {
      answers.forEach((d, j) => { if (d) p.asked[j].forEach((a, row) => p.lists[a.query].push(d.subarray(row * 25, (row + 1) * 25))); });
      p.lists.forEach((candidates, q) => {
        const all = new Float64Array(candidates.reduce((s, c) => s + c.length, 0)); let at = 0;
        for (const c of candidates) { all.set(c, at); at += c.length; } all.sort();
        let sum = 0; for (let j = 1; j <= 24; j++) sum += Math.sqrt(all[j]); p.means[p.openIndex[q]] = sum / 24;
      });
      return p.means;
    };
    let phases;
    try {
      cloud.buildIndex();
      const parts = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 2) - 1), Math.floor(points / 100_000));
      if (parts < 2) throw new Error("Expected a parallel-eligible input");
      let t = performance.now(); plan = wasm.planSor(cloud, parts);
      const indices = Array.from({ length: plan.length }, (_, j) => plan.indices(j)), regions = plan.regions(), partitionMillis = performance.now() - t;
      const local = [], normals = [], preparation = [], orientation = new Float64Array(cloud.normalsOrientation("up"));
      for (let j = 0; j < plan.length; j++) {
        t = performance.now(); const pts = plan.points(cloud, j), pointCopyMillis = performance.now() - t;
        t = performance.now(); const part = new wasm.SorPart(pts); prepared.push(part); const treeMillis = performance.now() - t;
        t = performance.now(); const r = part.local(24, j, regions); const localMillis = performance.now() - t;
        local.push({ means: r.means(), open: r.open(), openPoints: r.openPoints(), candidates: r.candidates(), reach: r.reach() }); r.free();
        t = performance.now(); normals.push(wasm.normalsOf(pts, 24, orientation)); const normalKernelIncludingTreeMillis = performance.now() - t;
        preparation.push({ pointCopyMillis, treeMillis, localMillis, normalKernelIncludingTreeMillis });
      }
      const first = legacyPack(local, indices, points); const answers = [], crossMillis = [];
      for (let j = 0; j < plan.length; j++) { t = performance.now(); answers.push(first.queries[j].length ? prepared[j].within(first.queries[j], 24) : null); crossMillis.push(performance.now() - t); }
      const warmups = [], trials = [];
      for (let i = -warmupRounds; i < repeats; i++) {
        const order = Math.abs(i) % 2 ? ["optimized", "baseline"] : ["baseline", "optimized"], values = {};
        for (const mode of order) {
          t = performance.now();
          const p = mode === "baseline" ? legacyPack(local, indices, points) : join.makeSorQueries(local, indices, points);
          const packingMillis = performance.now() - t;
          t = performance.now(); const means = mode === "baseline" ? legacyFinish(answers, p) : join.finishSorMeans(local, answers, p, 24); const mergeMillis = performance.now() - t;
          t = performance.now(); let out;
          if (mode === "baseline") { out = new Float32Array(points * 3); normals.forEach((part, j) => { for (let r = 0; r < indices[j].length; r++) out.set(part.subarray(r * 3, r * 3 + 3), indices[j][r] * 3); }); }
          else out = join.scatterNormalParts(normals, indices, points);
          values[mode] = { packingMillis, mergeMillis, scatterMillis: performance.now() - t, means, normals: out, queries: p.queries };
        }
        if (!sameBits(values.baseline.means, values.optimized.means) || !sameBits(values.baseline.normals, values.optimized.normals) ||
          values.baseline.queries.some((q, j) => !sameBits(q, values.optimized.queries[j]))) throw new Error("Isolated join/query bits changed");
        const times = Object.fromEntries(Object.entries(values).map(([mode, v]) => [mode, { packingMillis: v.packingMillis, mergeMillis: v.mergeMillis, scatterMillis: v.scatterMillis }]));
        (i < 0 ? warmups : trials).push({ order, ...times, resultParity: true });
      }
      phases = { parts: plan.length, openPoints: local.reduce((n, p) => n + p.open.length, 0), crossQueries: first.queries.map(q => q.length / 3),
        singleObservations: { partitionMillis, preparation, crossMillis }, warmups, trials,
        note: "Sequential diagnostic kernels on the full measured cloud indexed via Cloud.buildIndex; these are not production pool phase timings. Isolated JavaScript packing, merge and scatter compare identical real kernel outputs; no message transfer is timed here. Preparation observations are single samples, not medians; normal kernel includes its own tree and Morton order. Production whole RPCs above include packing, message transfer, kernels, merge/scatter, output copies, and SOR filtered-cloud indexing. Do not add these isolated sequential observations to estimate parallel latency." };
    } finally { prepared.forEach(p => p.free()); plan?.free(); cloud.free(); }
    return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, ...description, cases, phases };
  }, { builds: Object.fromEntries(Object.entries(builds).map(([mode, b]) => [mode, { worker: b.worker, wasm: b.wasm }])), input, points: inputManifest.inputs.compared.points, repeats, warmupRounds });
  for (const [path, hash] of [[inputPath, input.sha256], [manifestPath, digest(manifestBytes)], [sourcePath, sourceSha256], [gluePath, glueSha256], [fileURLToPath(import.meta.url), scriptSha256]])
    if (digest(readFileSync(path)) !== hash) throw new Error(`Input/source/script changed during timing: ${path}`);
  for (const b of Object.values(builds)) for (const [name, hash] of Object.entries(b.artifactHashes))
    if (digest(readFileSync(resolve(b.root, "assets", name))) !== hash) throw new Error("Production artifact changed during timing");
  Object.assign(report, { schemaVersion: 1, comparison: "cpu-knn-packed-joins", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(), builds, scriptSha256, sourceSha256, glueSha256,
    browserVersion: browser.version(), machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
    input, inputManifest, inputManifestSha256: digest(manifestBytes), repeats, warmupRounds,
    note: "Unchanged full measured input, k=24, normals up, SOR ratio=1. Each method has two isolated production analysis workers and identical pool policy; lane worker and WASM hashes must match. Pool startup/loading excluded. One excluded first-use pair, five excluded alternating warm-up pairs and ten timed pairs unless repeats overridden. Whole RPC wall time includes computation, messages, typed-array copies and SOR output indexing. Every pair checks all normal f32 bits or byte-identical SOR PLY/kept count; exports, validation, result removal excluded. Separate join diagnostics run only after production workers terminate. No builds/tests/scans run during timing. All input, manifest, source, glue, script and artifact hashes checked before/after. CPU clocks/power/GC and other apps uncontrolled; source subsets are not independent surveys or ground truth. No GPU information collected." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
