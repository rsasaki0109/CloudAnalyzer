// Retained CPU C2M / GPU triangle hints, with an archived retained-CPU oracle.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { readFileSync, readdirSync, existsSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { cpus, platform, release, totalmem } from "node:os";

const args=process.argv.slice(2), allowed=["--output","--reference","--compared","--input-manifest","--baseline-dist","--repeats"];
for(let i=0;i<args.length;i+=2)if(!allowed.includes(args[i])||!args[i+1]||args[i+1].startsWith("--")||args.indexOf(args[i])!==i)throw new Error(`Invalid or repeated option: ${args[i]}`);
const value=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const output=value("--output"),manifestPath=value("--input-manifest"),baselinePath=value("--baseline-dist"),repeats=Number(value("--repeats","10")),warmupRounds=5;
if(!output||!manifestPath||!baselinePath||!value("--reference")||!value("--compared"))throw new Error("Supply --baseline-dist DIST --reference FILE --compared FILE --input-manifest JSON --output NEW.json [--repeats 10]");
if(existsSync(output))throw new Error(`Report already exists: ${output}`);
if(!Number.isInteger(repeats)||repeats<1||repeats>100)throw new Error("Invalid repeats");
const hash=b=>createHash("sha256").update(b).digest("hex"),manifestBytes=readFileSync(manifestPath),inputManifest=JSON.parse(manifestBytes);
const inputData={},inputs={};
for(const role of ["reference","compared"]){
  const path=value(`--${role}`),bytes=readFileSync(path);inputData[role]=bytes;
  inputs[role]={path,name:basename(path),bytes:bytes.length,sha256:hash(bytes)};
  const expected=inputManifest.inputs?.[role];
  if(!expected||expected.sha256!==inputs[role].sha256||expected.bytes!==bytes.length||!Number.isInteger(expected.points)||expected.points<200_000||expected.points>1_000_000)throw new Error(`Expected full measured ${role} input matching its manifest`);
}
const sourcePath=resolve(inputManifest.source.path);
if(hash(readFileSync(sourcePath))!==inputManifest.source.sha256)throw new Error("Source differs from manifest");
const builds={};
for(const [name,root] of [["baseline",resolve(baselinePath)],["current",fileURLToPath(new URL("../dist/",import.meta.url))]]){
  const assets=resolve(root,"assets"),files=readdirSync(assets).filter(n=>/^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n)),worker=files.find(n=>/^worker-/.test(n));
  if(!worker||files.length!==3)throw new Error(`Expected three production artifacts in ${name}`);
  builds[name]={root,worker,artifactHashes:Object.fromEntries(files.map(n=>[n,hash(readFileSync(resolve(assets,n)))]))};
}
if(JSON.stringify(builds.baseline.artifactHashes)===JSON.stringify(builds.current.artifactHashes))throw new Error("Expected an archived production oracle");
const scriptPath=fileURLToPath(import.meta.url),scriptSha256=hash(readFileSync(scriptPath));
const sourceFiles=["web/src/gpu-mesh-candidates.ts","web/src/gpu-candidates.ts","web/src/mesh-preparation.ts","web/src/worker.ts","web/src/c2c-worker.ts","web/src/pool.ts","web/src/protocol.ts","rust/crates/ca-core/src/mesh.rs","rust/crates/ca-core/src/distance.rs","rust/crates/ca-wasm/src/lib.rs"];
const sourceHashes=Object.fromEntries(sourceFiles.map(p=>[p,hash(readFileSync(resolve("..",p)))]));
const server=createServer((req,res)=>{
  const path=new URL(req.url,"http://localhost").pathname;
  if(path==="/"){res.setHeader("Content-Type","text/html");res.end("<!doctype html>");return;}
  if(path==="/__reference"||path==="/__compared"){res.setHeader("Content-Type","application/octet-stream");res.end(inputData[path.slice(3)]);return;}
  const mode=path.split("/")[1],build=builds[mode];if(!build){res.writeHead(404).end();return;}
  const target=resolve(build.root,"."+path.slice(mode.length+1));
  if(!target.startsWith(build.root.endsWith(sep)?build.root:build.root+sep)){res.writeHead(403).end();return;}
  try{res.setHeader("Content-Type",({".js":"text/javascript",".wasm":"application/wasm"})[extname(target)]||"application/octet-stream");res.end(readFileSync(target));}catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));let browser;
try {
  browser = await chromium.launch({ channel: "chromium", headless: true, args: ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"] });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report = await page.evaluate(async ({ builds, inputs, inputManifest, repeats, warmupRounds }) => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error("Physical WebGPU adapter required");
    const gpu = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, isFallbackAdapter: adapter.info.isFallbackAdapter };
    const files = {};
    for (const role of ["reference", "compared"]) files[role] = new File([await (await fetch(`/__${role}`)).arrayBuffer()], inputs[role].name);
    const digest = async a => [...new Uint8Array(await crypto.subtle.digest("SHA-256", a))].map(v => v.toString(16).padStart(2, "0")).join("");
    const same = (a, b) => {
      const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
      return x.length === y.length && x.every((v, i) => v === y[i]);
    };
    const context = async mode => {
      const worker = new Worker(`/${mode}/assets/${builds[mode].worker}`, { type: "module" }); let seq = 0; const pending = new Map();
      worker.onmessage = ({ data }) => {
        if (!data.response) return; const p = pending.get(data.seq); if (!p) return; pending.delete(data.seq); clearTimeout(p.timer);
        data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
      };
      worker.onerror = e => { for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new Error(e.message)); } pending.clear(); };
      const call = req => new Promise((resolve, reject) => {
        const id = ++seq, timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timeout ${req.kind}`)); }, 180_000);
        pending.set(id, { resolve, reject, timer }); worker.postMessage({ seq: id, req });
      });
      try {
        await call({ kind: "cpu-prepare" }); const clouds = {};
        for (const role of ["reference", "compared"]) {
          clouds[role] = await call({ kind: "load", file: files[role], maxPoints: 2_000_000 }); const c = clouds[role];
          if (c.kind !== "cloud" || c.count !== inputManifest.inputs[role].points || c.filePoints !== c.count || c.keepEvery !== 1) throw new Error("Input point selection changed");
        }
        const built = await call({ kind: "mesh", id: clouds.reference.id, maxEdge: null });
        if (!built.mesh.triangles || built.voxel !== 0) throw new Error("Expected full-input measured TIN without voxel thinning");
        const mesh = { vertices: built.mesh.count, triangles: built.mesh.triangles, points: built.points, voxel: built.voxel, maxEdge: built.maxEdge, removed: built.removed,
          positionsSha256: await digest(built.mesh.positions), indicesSha256: await digest(built.mesh.indices) };
        return { worker, call, clouds, mesh, meshId: built.mesh.id };
      } catch (error) { worker.terminate(); throw error; }
    };
    const contexts = [];
    try {
      const baseline = await context("baseline"); contexts.push(baseline);
      const request = (ctx, meshGpu = false) => ({ kind: "c2c", compared: ctx.clouds.compared.id, reference: ctx.meshId, signed: false, meshGpu });
      const oracle = await baseline.call(request(baseline));
      if (!oracle.meshCached || oracle.meshCacheFallback) throw new Error("Expected archived retained-CPU oracle");
      const oracleBytes = await baseline.call({ kind: "export", id: baseline.clouds.compared.id, format: "ply", scalar: { name: "C2M", values: oracle.distances } });
      const distanceSha256 = await digest(oracle.distances), plySha256 = await digest(oracleBytes); baseline.worker.terminate();
      const current = await context("current"); contexts.push(current);
      if (JSON.stringify(current.mesh) !== JSON.stringify(baseline.mesh)) throw new Error("Production TIN changed");
      const compute = async mode => {
        const start = performance.now(), r = await current.call(request(current, mode === "gpu")), elapsedMillis = performance.now() - start;
        const cache = await current.call({ kind: "mesh-cache-state" });
        if (r.kind !== "c2m" || r.signed || r.meshCacheFallback || r.workers !== oracle.workers || r.meshSlices !== oracle.meshSlices || !r.meshCached || !cache || cache.pending ||
          cache.bytes !== r.meshCacheBytes || cache.bytes > cache.reservation || cache.reservedTotal > cache.budget) throw new Error("Unexpected C2M layout, cache budget or fallback");
        if (mode === "gpu" && (r.meshCompute?.backend !== "webgpu" || r.meshCompute.fallback || r.meshCompute.completeRows !== current.clouds.compared.count ||
          r.meshCompute.batches !== Math.ceil(current.clouds.compared.count / 65_536) || r.meshCompute.gpuBytes > 256*1024*1024)) throw new Error("Expected physical GPU hints for every point: " + JSON.stringify(r.meshCompute));
        if (mode === "cpu" && r.meshCompute) throw new Error("CPU control used GPU hints");
        if (!same(oracle.distances, r.distances) || Object.keys(oracle.stats).some(k => !Object.is(oracle.stats[k], r.stats[k]))) throw new Error("C2M result bits/summaries changed");
        const bytes = await current.call({ kind: "export", id: current.clouds.compared.id, format: "ply", scalar: { name: "C2M", values: r.distances } });
        if (!same(oracleBytes, bytes)) throw new Error("C2M scalar PLY changed");
        return { elapsedMillis, workerMillis: r.millis, workers: r.workers, slices: r.meshSlices, reused: r.meshCacheReused, retainedBytes: r.meshCacheBytes, cache, compute: r.meshCompute, resultParity: true };
      };
      const firstUse = { cpu: await compute("cpu"), gpu: await compute("gpu") };
      if (firstUse.cpu.reused || firstUse.gpu.compute.referenceReused) throw new Error("Unexpected cold reuse");
      const warmups = [], trials = [];
      for (let i = -warmupRounds; i < repeats; i++) {
        const order = (i + warmupRounds) % 2 ? ["gpu", "cpu"] : ["cpu", "gpu"], values = {};
        for (const mode of order) values[mode] = await compute(mode);
        if (!values.cpu.reused || !values.gpu.reused || !values.gpu.compute.referenceReused) throw new Error("Expected warm CPU/GPU mesh reuse");
        (i < 0 ? warmups : trials).push({ order, ...values });
        await globalThis.benchmarkProgress(`C2M ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: CPU ${values.cpu.elapsedMillis.toFixed(1)}, GPU ${values.gpu.elapsedMillis.toFixed(1)} ms; GPU ${values.gpu.compute.gpuMillis.toFixed(1)}, verify ${values.gpu.compute.verifyMillis.toFixed(1)} ms; parity`);
      }
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, gpu, mesh: baseline.mesh, points: baseline.clouds.compared.count,
        oracleWorkers: oracle.workers, oracleSlices: oracle.meshSlices, distanceSha256, plySha256, firstUse, warmups, trials };
    } finally { for (const c of contexts) c.worker.terminate(); }
  }, { builds: Object.fromEntries(Object.entries(builds).map(([k, b]) => [k, { worker: b.worker }])), inputs, inputManifest, repeats, warmupRounds });
  for (const [name, build] of Object.entries(builds)) for (const [file, expected] of Object.entries(build.artifactHashes)) if (hash(readFileSync(resolve(build.root, "assets", file))) !== expected) throw new Error(`Changed ${name} artifact`);
  for (const [p, expected] of Object.entries(sourceHashes)) if (hash(readFileSync(resolve("..", p))) !== expected) throw new Error(`Changed source ${p}`);
  for (const role of ["reference", "compared"]) if (hash(readFileSync(inputs[role].path)) !== inputs[role].sha256) throw new Error("Input changed during timing");
  if (hash(readFileSync(manifestPath)) !== hash(manifestBytes) || hash(readFileSync(sourcePath)) !== inputManifest.source.sha256 || hash(readFileSync(scriptPath)) !== scriptSha256) throw new Error("Provenance/script changed during timing");
  const median = a => { a = [...a].sort((x, y) => x-y); return a.length%2 ? a[a.length>>1] : (a[a.length/2-1]+a[a.length/2])/2; };
  const reductions = report.trials.map(t => 100*(1-t.gpu.elapsedMillis/t.cpu.elapsedMillis));
  Object.assign(report, { schemaVersion: 1, comparison: "retained-c2m-gpu-triangle-candidates", createdAt: new Date().toISOString(), builds, scriptSha256, sourceHashes, inputs, inputManifest, inputManifestSha256: hash(manifestBytes), repeats, warmupRounds, browserVersion: browser.version(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(), workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
    summary: { cpuMedianMillis: median(report.trials.map(t => t.cpu.elapsedMillis)), gpuMedianMillis: median(report.trials.map(t => t.gpu.elapsedMillis)), medianPairedReductionPercent: median(reductions),
      fasterTrials: reductions.filter(x => x>0).length, trials: repeats, gpuPhaseMedianMillis: median(report.trials.map(t => t.gpu.compute.gpuMillis)), verifyPhaseMedianMillis: median(report.trials.map(t => t.gpu.compute.verifyMillis)) },
    note: "Unchanged measured source-index geometry splits, not independent accuracy ground truth. Full-input production 2.5D TIN, recorded XY duplicates/edge rejection; no voxel thinning, new coordinates or input resampling. Archived retained-CPU production worker supplies exact f32 distance, f64 summary and scalar PLY oracles; its first output is excluded. Current CPU/GPU controls share one prewarmed context and the same 1 GiB bounded immutable CPU mesh/BVH cache, resident workers and unchanged logical slices. GPU geometry remains resident during the CPU control. GPU candidate search uses shifted f32 triangles/BVH, returns one untrusted triangle slot per point in batches of at most 65536, with a 256 MiB aggregate GPU-buffer budget. Every candidate's upper radius is recomputed on original f64 geometry; CPU root traversal retains old warm guesses, child order and strict insertion, with fallback for invalid hints/incomplete bounded search. Timed fallback, missing candidate, changed layout or lack of CPU/GPU reuse fails. Whole RPC includes query copies, packing/transfers, GPU search/readback, exact CPU verification, ordering, joining, summaries and returned arrays. GPU phase includes host packing, serialized device scopes, dispatch and readback; verification phase includes worker transfers/query traversal/join. Loading/meshing, pool startup, first observations, five warm-up rounds, parity/export checks and cleanup excluded. All relevant source/artifact/input/manifest/script hashes verified before/after. No builds/tests/scans alongside timing; clocks, power, other applications, scheduling and GC uncontrolled. Retained sizes cover owned vectors/allocated GPU buffers, not temporary copies, allocator/driver overhead or peak process memory. Only GPU vendor/architecture/fallback recorded; no model identifiers." });
  writeFileSync(output, JSON.stringify(report, null, 2)+"\n", { flag: "wx" }); console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
