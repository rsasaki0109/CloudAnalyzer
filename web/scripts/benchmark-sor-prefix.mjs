// SOR whole RPCs and isolated sorted-prefix merges on unchanged measured inputs.
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
const allowed = ["--output", "--input", "--input-manifest", "--baseline-dist", "--baseline-helper", "--repeats"];
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
if (!Number.isInteger(repeats) || repeats < 1 || repeats > 100) throw new Error("Invalid repeats");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const inputBytes = readFileSync(inputPath), manifestBytes = readFileSync(manifestPath), inputManifest = JSON.parse(manifestBytes);
const input = { path: inputPath, name: basename(inputPath), bytes: inputBytes.length, sha256: digest(inputBytes) };
if (inputManifest.inputs?.compared?.sha256 !== input.sha256 || inputManifest.inputs.compared.bytes !== input.bytes ||
  !Number.isInteger(inputManifest.inputs.compared.points) || inputManifest.inputs.compared.points < 1 || inputManifest.inputs.compared.points > 2_000_000)
  throw new Error("Expected the full measured compared input matching its manifest, at most two million points");
const originalPath = resolve('web', inputManifest.source?.path ?? '');
const originalBytes = readFileSync(originalPath);
if (originalBytes.length !== inputManifest.source?.bytes || digest(originalBytes) !== inputManifest.source.sha256)
  throw new Error('Original measured source differs from manifest');
const baselineHelperPath = value('--baseline-helper', resolve(baselineRoot, '../web/src/knn-join.ts'));
const baselineHelperBytes = readFileSync(baselineHelperPath), baselineHelperSha256 = digest(baselineHelperBytes);
const sourceFiles = ['web/src/knn-join.ts','web/src/worker.ts','web/src/knn-preparation.ts','web/src/protocol.ts','web/src/c2c-worker.ts','web/src/pool.ts'];
const sourceHashes = Object.fromEntries(sourceFiles.map(path => [path,digest(readFileSync(path))]));
if (!/const USE_PREFIX_SOR_MERGE\s*=\s*true\s*;/.test(readFileSync('web/src/knn-join.ts','utf8')))
  throw new Error('Set USE_PREFIX_SOR_MERGE = true and rebuild the matching app before comparison');
if (baselineHelperBytes.includes(Buffer.from('mergeSorPrefix')))
  throw new Error('Expected a pre-prefix full-sort helper');
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
const baselineHelperModule = stripTypeScriptTypes(baselineHelperBytes.toString('utf8'), { mode:'strip' });
const helperModule = stripTypeScriptTypes(sourceBytes.toString("utf8"), { mode: "strip" });
const server = createServer((req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  if (path === "/") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (path === "/__input") { res.setHeader("Content-Type", "application/octet-stream"); res.end(inputBytes); return; }
  if (path === '/__old-join.js') { res.setHeader('Content-Type','text/javascript'); res.end(baselineHelperModule); return; }
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
    for (const method of ["sor"]) {
      const contexts = {};
      try {
        for (const mode of ["baseline", "optimized"]) contexts[mode] = await context(mode);
        description = { points: contexts.baseline.cloud.count, bounds: contexts.baseline.cloud.bounds, keepEvery: 1 };
        const compute = async mode => {
          const { call, cloud } = contexts[mode], start = performance.now();
          const result = await call(method === "normals" ? { kind: method, id: cloud.id, k: 24, orientation: "up", knnCache:true, knnBounds:false } : { kind: "filter", id: cloud.id, op: "sor", a: 24, b: 1.0, knnCache:true, knnBounds:false });
          const elapsedMillis = performance.now() - start, cache = await call({kind:'knn-cache-state'});
          if (!cache || cache.pending || cache.points !== points || !(cache.bytes > 0) || cache.bytes > 128*1024*1024)
            throw new Error('Expected a bounded live unchanged preparation');
          return { result, elapsedMillis, cache };
        };
        const verify = async values => {
          const a = values.baseline.result, b = values.optimized.result;
          if (values.baseline.cache.bytes !== values.optimized.cache.bytes || values.baseline.cache.parts !== values.optimized.cache.parts)
            throw new Error('Preparation capacities or lanes differ');
          if (method === "normals") { if (!sameBits(a, b)) throw new Error("Normal f32 bits changed"); return { resultParity: true, outputPoints: a.length / 3 }; }
          const x = await contexts.baseline.call({ kind: "export", id: a.id, format: "ply" }), y = await contexts.optimized.call({ kind: "export", id: b.id, format: "ply" });
          if (a.count !== b.count || x.length !== y.length || x.some((v, i) => v !== y[i])) throw new Error("SOR kept points/coordinates/attributes changed");
          return { resultParity: true, outputPoints: a.count, exportedBytes: x.length };
        };
        const cleanup = async values => { if (method === "sor") for (const mode of ["baseline", "optimized"]) await contexts[mode].call({ kind: "remove", id: values[mode].result.id }); };
        const setup = { baseline: await compute("baseline"), optimized: await compute("optimized") };
        const setupParity = await verify(setup);
        if (setup.baseline.cache.reuses !== 0 || setup.optimized.cache.reuses !== 0) throw new Error('First use was already warm');
        const jobs = {baseline:setup.baseline.cache.job, optimized:setup.optimized.cache.job};
        await cleanup(setup);
        const warmups = [], trials = [];
        for (let i = -warmupRounds; i < repeats; i++) {
          const order = Math.abs(i) % 2 ? ["optimized", "baseline"] : ["baseline", "optimized"], values = {};
          for (const mode of order) values[mode] = await compute(mode);
          for (const mode of ['baseline','optimized']) if (values[mode].cache.job !== jobs[mode] || values[mode].cache.reuses !== i+warmupRounds+1)
            throw new Error('Preparation rebuilt or exact reuse count changed');
          const parity = await verify(values);
          (i < 0 ? warmups : trials).push({ order, baselineMillis: values.baseline.elapsedMillis, optimizedMillis: values.optimized.elapsedMillis, baselineCache:values.baseline.cache, optimizedCache:values.optimized.cache, ...parity });
          await cleanup(values);
          await globalThis.benchmarkProgress(`${method} ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: ${Math.round(values.baseline.elapsedMillis)}/${Math.round(values.optimized.elapsedMillis)} ms; parity`);
        }
        cases.push({ method, excludedFirstUse: { baselineMillis: setup.baseline.elapsedMillis, optimizedMillis: setup.optimized.elapsedMillis, baselineCache:setup.baseline.cache, optimizedCache:setup.optimized.cache, ...setupParity }, warmups, trials });
      } finally { for (const c of Object.values(contexts)) c.worker.terminate(); }
    }
    // Sequential isolated joins use unchanged real WASM kernel outputs.
    const wasm = await import('/__wasm.js'), join = await import('/__join.js'), oldJoin = await import('/__old-join.js');
    await wasm.default({module_or_path:'/optimized/assets/'+builds.optimized.wasm});
    const cloud = wasm.Cloud.parse(input.name,bytes), prepared=[];
    let plan, phases;
    try {
      cloud.buildIndex();
      const partCount=Math.min(8,Math.max(1,(navigator.hardwareConcurrency||2)-1),Math.floor(points/100_000));
      if(partCount<2) throw new Error('Expected a parallel-eligible input');
      plan=wasm.planSor(cloud,partCount);
      const indices=Array.from({length:plan.length},(_,j)=>plan.indices(j)),regions=plan.regions(),local=[];
      for(let j=0;j<plan.length;j++) {
        const part=wasm.SorPart.prepared(plan.points(cloud,j)); prepared.push(part);
        const r=part.localWithBounds(24,j,regions,false);
        local.push({means:r.means(),open:r.open(),openPoints:r.openPoints(),candidates:r.candidates(),reach:r.reach()}); r.free();
      }
      const packet=join.makeSorQueries(local,indices,points);
      const answers=prepared.map((p,j)=>packet.queries[j].length?p.withinWithBounds(packet.queries[j],24,false):null);
      let candidateRows=0;
      for(const array of [...local.map(p=>p.candidates),...answers.filter(Boolean)]) for(let row=0;row<array.length;row+=25) {
        let previous=-Infinity;
        for(let i=0;i<25;i++) { const value=array[row+i]; if(!(value>=0)||value<previous||Object.is(value,-0)) throw new Error('Expected ordinary sorted nonnegative real kernel rows'); previous=value; }
        candidateRows++;
      }
      const warmups=[],trials=[];
      for(let i=-warmupRounds;i<repeats;i++) {
        const order=Math.abs(i)%2?['optimized','baseline']:['baseline','optimized'],values={};
        for(const mode of order) {
          const helper=mode==='baseline'?oldJoin:join,p=helper.makeSorQueries(local,indices,points);
          const start=performance.now(),means=helper.finishSorMeans(local,answers,p,24);
          values[mode]={millis:performance.now()-start,means};
        }
        if(!sameBits(values.baseline.means,values.optimized.means)) throw new Error('Full-sort/prefix mean bits differ');
        (i<0?warmups:trials).push({order,baselineMillis:values.baseline.millis,optimizedMillis:values.optimized.millis,resultParity:true});
      }
      phases={parts:plan.length,openPoints:local.reduce((n,p)=>n+p.open.length,0),crossQueries:packet.queries.map(q=>q.length/3),
        candidateRows,warmups,trials,note:'Only finishSorMeans is timed on identical real kernel outputs. Independent old/current helper modules retain the exact full-sort oracle. Every f64 mean bit matches. Packing, sequential preparation/kernels, parsing, checking and cleanup are excluded. These isolated joins omit message transfer, concurrent pool scheduling and SOR output indexing; do not infer whole-RPC savings from them.'};
    } finally {prepared.forEach(p=>p.free());plan?.free();cloud.free();}
    return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, ...description, cases, phases };
  }, { builds: Object.fromEntries(Object.entries(builds).map(([mode, b]) => [mode, { worker: b.worker, wasm: b.wasm }])), input, points: inputManifest.inputs.compared.points, repeats, warmupRounds });
  for (const [path, hash] of [[inputPath, input.sha256], [originalPath,digest(originalBytes)], [baselineHelperPath,baselineHelperSha256], ...Object.entries(sourceHashes), [manifestPath, digest(manifestBytes)], [sourcePath, sourceSha256], [gluePath, glueSha256], [fileURLToPath(import.meta.url), scriptSha256]])
    if (digest(readFileSync(path)) !== hash) throw new Error(`Input/source/script changed during timing: ${path}`);
  for (const b of Object.values(builds)) for (const [name, hash] of Object.entries(b.artifactHashes))
    if (digest(readFileSync(resolve(b.root, "assets", name))) !== hash) throw new Error("Production artifact changed during timing");
  Object.assign(report, { schemaVersion: 1, comparison: "cpu-sor-sorted-prefix", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(), builds, scriptSha256, sourceSha256, glueSha256, sourceHashes, baselineHelperPath, baselineHelperSha256, settings:{k:24,ratio:1,knnCache:true,knnBounds:false},
    browserVersion: browser.version(), machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
    input, inputManifest, inputManifestSha256: digest(manifestBytes), repeats, warmupRounds,
    note: "Full unchanged measured input, k=24, SOR ratio=1. Two independent production analysis workers compare the preceding reusable full-sort join with the guarded sorted-prefix join. Both request knnCache:true, knnBounds:false; exact incrementing reuse counters, stable preparation jobs, equal lanes/capacities and nonpending caches under 128 MiB are required. Only the main worker differs; lane and WASM hashes must match. Each pair checks kept count and every exported PLY byte. One excluded first-use pair and five excluded alternating warm-up pairs precede ten alternating timed pairs unless overridden. First use is not a cold median. Whole RPC includes local/cross-part kernels, packing, messages, merge, summary/filtering and output indexing; startup/load, cache diagnostics, validation/export/removal excluded. Separate isolated joins follow worker termination and check every f64 mean bit on the same real sorted kernel lists. No builds/tests/scans during timing. Full input, manifest, original measured source, six current sources, archived helper, glue, script and both worker/pool/WASM builds hashed before/after. CPU clocks/power/GC/other apps uncontrolled; even-odd source subsets are not independent surveys or accuracy ground truth. No GPU metadata or request fields added." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
