// Measure current heavy operations and, separately, their WASM export / pool boundaries.
// No GPU speedup is inferred from CPU timings. The existing hybrid C2C is measured directly.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2);
const allowed = ["--output", "--reference", "--compared", "--input-manifest", "--repeats", "--profile-repeats", "--methods"];
for (let i = 0; i < args.length; i += 2) {
  if (!allowed.includes(args[i]) || !args[i + 1] || args[i + 1].startsWith("--") || args.indexOf(args[i]) !== i)
    throw new Error(`Invalid or repeated option: ${args[i]}`);
}
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output"), manifestPath = value("--input-manifest");
const referencePath = value("--reference"), comparedPath = value("--compared");
const available = ["c2c", "normals", "sor", "m3c2-fresh", "m3c2-reused", "icp", "map-evaluate", "map-voxels", "c2m"];
const methods = value("--methods", available.join(",")).split(",");
const repeats = Number(value("--repeats", "10")), profileRepeats = Number(value("--profile-repeats", "3")), warmupRounds = 5;
if (!output || !manifestPath || !referencePath || !comparedPath)
  throw new Error("Supply --reference FILE --compared FILE --input-manifest JSON --output REPORT.json [--repeats 10] [--profile-repeats 3] [--methods c2c,normals,sor,m3c2-fresh,m3c2-reused,icp,map-evaluate,map-voxels,c2m]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
if (![repeats, profileRepeats].every(n => Number.isInteger(n) && n >= 1 && n <= 100) ||
    new Set(methods).size !== methods.length || methods.some(m => !available.includes(m))) throw new Error("Invalid repeats or methods");
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const scriptPath = fileURLToPath(import.meta.url), scriptSha256 = hash(readFileSync(scriptPath));
const inputs = Object.fromEntries([["reference", referencePath], ["compared", comparedPath]].map(([role, path]) => {
  const bytes = readFileSync(path);
  return [role, { path, name: basename(path), bytes: bytes.length, sha256: hash(bytes) }];
}));
const manifestBytes = readFileSync(manifestPath), inputManifest = JSON.parse(manifestBytes);
if (["reference", "compared"].some(role => inputManifest.inputs?.[role]?.sha256 !== inputs[role].sha256 ||
    inputManifest.inputs[role].bytes !== inputs[role].bytes || !Number.isInteger(inputManifest.inputs[role].points) ||
    inputManifest.inputs[role].points < 1 || inputManifest.inputs[role].points > 2_000_000))
  throw new Error("Expected full measured files matching their manifest, at most two million points each");
if (!inputManifest.source?.sha256 || !inputManifest.split?.disjointSourceIndices)
  throw new Error("Expected measured-source provenance");
const sourcePath = resolve(inputManifest.source.path);
if (hash(readFileSync(sourcePath)) !== inputManifest.source.sha256) throw new Error("Original source differs from manifest");
const root = fileURLToPath(new URL("../dist/", import.meta.url)), assets = resolve(root, "assets");
const names = readdirSync(assets).filter(n => /^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n));
const workerName = names.find(n => /^worker-/.test(n));
if (!workerName || names.length !== 3) throw new Error("Expected three production worker/WASM artifacts; build first");
const artifactHashes = Object.fromEntries(names.map(n => [n, hash(readFileSync(resolve(assets, n)))]));
const wasmName = names.find(n => /\.wasm$/.test(n));
const gluePath = fileURLToPath(new URL("../src/wasm/ca_wasm.js", import.meta.url));
const glueBytes = readFileSync(gluePath), glueSha256 = hash(glueBytes);

// Served only by this benchmark. The measured baseline uses ordinary production workers.
// Diagnostic runs wrap JS -> WASM calls and nested-worker RPCs; they never change WASM bytes.
const probeSource = String.raw`
const params = new URL(import.meta.url).searchParams;
const lane = params.get("lane"), modulePath = params.get("module");
const nativePost = self.postMessage.bind(self), NativeWorker = self.Worker;
let active = false, ready = false, nextControl = 0;
const deferred = [];
const calls = new Map(), rpcs = [], children = [], pendingControls = new Map();
const controlChild = (child, action) => new Promise((resolve, reject) => {
  const id = ++nextControl, timer = setTimeout(() => { pendingControls.delete(id); reject(new Error("Probe control timeout")); }, 180000);
  pendingControls.set(id, { resolve, timer }); child.postMessage({ __opportunityControl: action, id });
});
self.Worker = class extends NativeWorker {
  constructor(url, options) {
    const target = new URL(url, self.location.href);
    const childLane = lane + "/" + children.length;
    super("/__probe-worker?module=" + encodeURIComponent(target.pathname) + "&lane=" + encodeURIComponent(childLane), options);
    children.push(this);
    const pending = new Map();
    this.addEventListener("message", ({ data }) => {
      if (data.__opportunityReply) {
        const wait = pendingControls.get(data.id);
        if (wait) { clearTimeout(wait.timer); pendingControls.delete(data.id); wait.resolve(data.profile); }
        return;
      }
      const request = pending.get(data.seq);
      if (request) {
        pending.delete(data.seq);
        if (active) rpcs.push({ lane: childLane, kind: request.kind, elapsedMillis: performance.now() - request.started });
      }
    });
    const post = this.postMessage.bind(this);
    this.postMessage = (data, ...rest) => {
      if (active && data.seq !== undefined) pending.set(data.seq, { kind: data.kind, started: performance.now() });
      return post(data, ...rest);
    };
  }
};
const wrapInstance = instance => ({ exports: Object.fromEntries(Object.entries(instance.exports).map(([name, fn]) => {
  if (typeof fn !== "function") return [name, fn];
  return [name, (...args) => {
    if (!active) return fn(...args);
    const t = performance.now();
    try { return fn(...args); }
    finally {
      const duration = performance.now() - t;
      const s = calls.get(name) ?? { name, calls: 0, totalMillis: 0, maxMillis: 0 };
      s.calls++; s.totalMillis += duration; s.maxMillis = Math.max(s.maxMillis, duration); calls.set(name, s);
    }
  }];
})) });
for (const name of ["instantiate", "instantiateStreaming"]) {
  const instantiate = WebAssembly[name].bind(WebAssembly);
  WebAssembly[name] = async (...args) => {
    const result = await instantiate(...args);
    // The production glue uses a Response/byte source and the {instance,module} form.
    if (!result.instance) throw new Error("Unexpected synchronous WASM module instantiation in probe");
    return { ...result, instance: wrapInstance(result.instance) };
  };
}
self.addEventListener("message", event => {
  if (!ready && !event.data.__opportunityControl) {
    event.stopImmediatePropagation(); deferred.push(event.data); return;
  }
  if (!event.data.__opportunityControl) return;
  event.stopImmediatePropagation();
  const { __opportunityControl: action, id } = event.data;
  (async () => {
    if (action === "begin") {
      calls.clear(); rpcs.length = 0;
      await Promise.all(children.map(child => controlChild(child, action)));
      active = true;
      nativePost({ __opportunityReply: true, id });
    } else if (action === "end") {
      active = false;
      const profiles = await Promise.all(children.map(child => controlChild(child, action)));
      nativePost({ __opportunityReply: true, id, profile: { lane, exports: [...calls.values()], rpcs, children: profiles } });
    } else throw new Error("Unknown probe action");
  })().catch(error => { throw error; });
});
await import(modulePath);
ready = true;
for (const data of deferred) self.dispatchEvent(new MessageEvent("message", { data }));
deferred.length = 0;
`;

const server = createServer((req, res) => {
  const path = new URL(req.url, "http://localhost").pathname;
  if (path === "/") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (path === "/__probe-worker") { res.setHeader("Content-Type", "text/javascript"); res.end(probeSource); return; }
  if (path === "/__wasm-glue.js") { res.setHeader("Content-Type", "text/javascript"); res.end(glueBytes); return; }
  if (["/__reference", "/__compared"].includes(path)) {
    const input = inputs[path === "/__reference" ? "reference" : "compared"], bytes = readFileSync(input.path);
    if (hash(bytes) !== input.sha256) { res.writeHead(409).end("Input changed"); return; }
    res.setHeader("Content-Type", "application/octet-stream"); res.end(bytes); return;
  }
  const target = resolve(root, "." + path);
  if (!target.startsWith(root.endsWith(sep) ? root : root + sep)) { res.writeHead(403).end(); return; }
  try { res.setHeader("Content-Type", ({ ".js": "text/javascript", ".wasm": "application/wasm" })[extname(target)] || "application/octet-stream"); res.end(readFileSync(target)); }
  catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ channel: "chromium", headless: true,
    args: methods.includes("c2c") ? ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"] : [] });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report = await page.evaluate(async ({ workerName, wasmName, inputs, inputManifest, methods, repeats, profileRepeats, warmupRounds }) => {
    const files = {};
    for (const role of ["reference", "compared"]) {
      const response = await fetch(`/__${role}`); if (!response.ok) throw new Error(`Input fetch failed: ${role}`);
      files[role] = new File([await response.arrayBuffer()], inputs[role].name);
    }
    let adapter;
    if (methods.includes("c2c")) {
      const physical = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
      if (!physical || physical.info?.isFallbackAdapter) throw new Error("A physical WebGPU adapter is required");
      adapter = { vendor: physical.info?.vendor, architecture: physical.info?.architecture, isFallbackAdapter: physical.info?.isFallbackAdapter };
    }
    const digest = async data => [...new Uint8Array(await crypto.subtle.digest("SHA-256", data))].map(v => v.toString(16).padStart(2, "0")).join("");
    const numberBits = n => {
      const b = new ArrayBuffer(8); new DataView(b).setFloat64(0, n, true);
      return [...new Uint8Array(b)].map(v => v.toString(16).padStart(2, "0")).join("");
    };
    const exact = value => typeof value === "number" ? numberBits(value) : Array.isArray(value) ? value.map(exact) :
      value && typeof value === "object" ? Object.fromEntries(Object.keys(value).sort().map(k => [k, exact(value[k])])) : value;
    const create = async (method, instrumented) => {
      const worker = new Worker(instrumented ? `/__probe-worker?module=${encodeURIComponent(`/assets/${workerName}`)}&lane=main` : `/assets/${workerName}`, { type: "module" });
      let seq = 0, controlSeq = 0;
      const waiting = new Map(), controls = new Map();
      worker.onmessage = ({ data }) => {
        const map = data.__opportunityReply ? controls : waiting;
        const p = map.get(data.__opportunityReply ? data.id : data.seq);
        if (!p || (!data.__opportunityReply && !data.response)) return;
        map.delete(data.__opportunityReply ? data.id : data.seq); clearTimeout(p.timer);
        if (data.__opportunityReply) p.resolve(data.profile);
        else data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
      };
      worker.onerror = event => {
        for (const map of [waiting, controls]) { for (const p of map.values()) { clearTimeout(p.timer); p.reject(new Error(event.message)); } map.clear(); }
      };
      const send = (map, key, payload) => new Promise((resolve, reject) => {
        const timer = setTimeout(() => { map.delete(key); reject(new Error(`Worker timeout: ${JSON.stringify(payload)}`)); }, 180_000);
        map.set(key, { resolve, reject, timer }); worker.postMessage(payload);
      });
      const call = req => { const id = ++seq; return send(waiting, id, { seq: id, req }); };
      const control = action => { const id = ++controlSeq; return send(controls, id, { __opportunityControl: action, id }); };
      const load = async role => {
        const cloud = await call({ kind: "load", file: files[role], maxPoints: 2_000_000 });
        if (cloud.kind !== "cloud" || cloud.count !== inputManifest.inputs[role].points || cloud.filePoints !== cloud.count || cloud.keepEvery !== 1)
          throw new Error("Input was thinned or changed");
        return cloud;
      };
      try {
        await call({ kind: "cpu-prepare" });
        const reference = await load("reference"), compared = await load("compared");
        let mesh;
        if (method === "c2m") {
          const built = await call({ kind: "mesh", id: reference.id, maxEdge: null });
          mesh = { id: built.mesh.id, vertices: built.mesh.count, triangles: built.mesh.indices.length / 3,
            sourcePoints: built.points, voxel: built.voxel, maxEdge: built.maxEdge, removed: built.removed,
            positionsSha256: await digest(built.mesh.positions), indicesSha256: await digest(built.mesh.indices) };
          if (!(mesh.triangles > 0) || built.voxel !== 0) throw new Error("Expected nonempty TIN from full measured source without voxel thinning");
        }
        return { worker, call, control, load, reference, compared, mesh };
      } catch (error) { worker.terminate(); throw error; }
    };
    const cases = [];
    let description;
    for (const method of methods) {
      const contexts = [];
      try {
        const ordinary = await create(method, false); contexts.push(ordinary);
        description = Object.fromEntries(["reference", "compared"].map(role => {
          const c = ordinary[role]; return [role, { points: c.count, filePoints: c.filePoints, keepEvery: c.keepEvery, bounds: c.bounds }];
        }));
        const compute = async (ctx, backend = "cpu", profiled = false) => {
          const { call, reference, compared } = ctx;
          const moving = method === "icp" ? await ctx.load("compared") : undefined;
          const req = method.startsWith("m3c2-") ? { kind: "m3c2", reference: reference.id, compared: compared.id,
            normalRadius: 0.4, projectionRadius: 0.25, maxDepth: 0.5, coreSpacing: 0.25, cacheNormals: method === "m3c2-fresh" ? false : undefined } :
            method === "normals" ? { kind: "normals", id: compared.id, k: 24, orientation: "up" } :
            method === "sor" ? { kind: "filter", id: compared.id, op: "sor", a: 24, b: 1.0 } :
            method === "icp" ? { kind: "icp", moving: moving.id, reference: reference.id, maxIterations: 10, overlap: 0.8, matchCentroids: false, pointToPlane: true } :
            method === "map-evaluate" ? { kind: method, reference: reference.id, compared: compared.id, voxel: 1.0, minPoints: 5, threshold: 0.5 } :
            method === "map-voxels" ? { kind: "map-quality", reference: reference.id, compared: compared.id, voxel: 1.0, minPoints: 5, voxelCache: false } :
            { kind: "c2c", reference: method === "c2m" ? ctx.mesh.id : reference.id, compared: compared.id, signed: false, backend, gpuVerifyWorkers: 4 };
          if (profiled) await ctx.control("begin");
          const start = performance.now();
          const result = await call(req), elapsedMillis = performance.now() - start;
          const profile = profiled ? await ctx.control("end") : undefined;
          if (method === "c2c" && backend === "webgpu" && (result.compute?.backend !== "webgpu" || result.compute.fallback || result.compute.verifyWorkers !== 4))
            throw new Error("Expected four-lane physical WebGPU execution without fallback");
          const outputId = method === "sor" ? result.id : result.cloud?.id;
          return { result, elapsedMillis, profile, outputId, request: req };
        };
        let expected, meshQuerySignature, meshQueryPoints;
        const verify = async (ctx, value) => {
          const { result, outputId } = value, signature = {}, quality = result.quality ?? (method === "map-voxels" ? result : undefined);
          if (method === "normals") signature.normals = await digest(result);
          else if (method.startsWith("m3c2-")) {
            for (const key of ["distance", "lod95", "significant"]) signature[key] = await digest(result[key]);
            signature.points = result.cloud.count;
          } else if (method === "icp") {
            for (const key of ["matrix", "rmsInitial", "rmsFinal", "iterations", "converged"]) signature[key] = exact(result[key]);
          } else if (["c2c", "c2m", "map-evaluate"].includes(method)) {
            signature.distances = await digest(result.distances); signature.stats = exact(result.stats);
          }
          if (quality) {
            signature.quality = Object.fromEntries(Object.keys(quality).filter(k => !["awd", "scs", "voxelCacheHit"].includes(k)).sort().map(k => [k, exact(quality[k])]));
            if (expected) for (const key of ["awd", "scs"]) {
              const a = quality[key], b = expected.quality[key];
              if (!Object.is(a, b) && !(Number.isNaN(a) && Number.isNaN(b)) && !(Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b))))
                throw new Error(`Map ${key} reduction changed`);
            }
          }
          if (outputId !== undefined) {
            const ply = await ctx.call({ kind: "export", id: outputId, format: "ply" });
            signature.plySha256 = await digest(ply); signature.plyBytes = ply.length;
          }
          const key = JSON.stringify(signature);
          if (expected && key !== expected.key) throw new Error(`Result parity failed for ${method}`);
          expected ??= { key, signature, quality };
          if (outputId !== undefined) await ctx.call({ kind: "remove", id: outputId });
          return { resultParity: true, outputPoints: result.cloud?.count ?? result.count ?? (method === "normals" ? result.length / 3 : undefined),
            workers: result.workers, workerMillis: result.millis, compute: result.compute,
            treeCacheHit: result.treeCacheHit, normalCacheHit: result.normalCacheHit, normalCacheBytes: result.normalCacheBytes,
            referenceReused: result.referenceReused, referenceBytes: result.referenceBytes, voxelCacheHit: result.voxelCacheHit,
            cpuCacheReused: result.cpuCacheReused, iterations: result.iterations };
        };
        const backends = method === "c2c" ? ["cpu", "webgpu"] : ["cpu"];
        const setup = [], warmups = [], trials = [];
        let parameters;
        for (const backend of backends) {
          const x = await compute(ordinary, backend);
          parameters ??= Object.fromEntries(Object.entries(x.request).filter(([key]) => !["id", "moving", "reference", "compared", "backend"].includes(key)));
          if (method === "c2m") {
            meshQueryPoints = Math.ceil(ordinary.compared.count / x.result.workers);
            meshQuerySignature = await digest(x.result.distances.slice(0, meshQueryPoints));
          }
          setup.push({ backend, elapsedMillis: x.elapsedMillis, ...(await verify(ordinary, x)) });
        }
        for (let i = -warmupRounds; i < repeats; i++) {
          const order = Math.abs(i) % 2 ? [...backends].reverse() : backends;
          const trial = { order, measurements: [] };
          for (const backend of order) {
            const x = await compute(ordinary, backend);
            const parity = await verify(ordinary, x);
            if (i >= 0 && ((method === "c2c" && backend === "cpu" || method === "map-evaluate") && !x.result.cpuCacheReused ||
                method === "m3c2-reused" && !x.result.normalCacheHit || method === "icp" && !x.result.referenceReused ||
                method === "map-evaluate" && !x.result.voxelCacheHit || method === "c2c" && backend === "webgpu" &&
                  (!x.result.compute.referenceReused || !x.result.compute.deviceReused))) throw new Error("Expected current warm cache policy");
            trial.measurements.push({ backend, elapsedMillis: x.elapsedMillis, ...parity });
          }
          (i < 0 ? warmups : trials).push(trial);
          await globalThis.benchmarkProgress(`${method} ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: ${trial.measurements.map(x => `${x.backend} ${x.elapsedMillis.toFixed(1)} ms`).join(" / ")}; parity`);
        }
        ordinary.worker.terminate();
        const diagnostic = await create(method, true); contexts.push(diagnostic);
        if (method === "c2m" && JSON.stringify({ ...ordinary.mesh, id: 0 }) !== JSON.stringify({ ...diagnostic.mesh, id: 0 }))
          throw new Error("TIN mesh differs between ordinary/probe contexts");
        for (let i = 0; i < warmupRounds + 1; i++) for (const backend of backends)
          await verify(diagnostic, await compute(diagnostic, backend));
        const profiles = [];
        for (let i = 0; i < profileRepeats; i++) for (const backend of i % 2 ? [...backends].reverse() : backends) {
          const x = await compute(diagnostic, backend, true);
          profiles.push({ backend, elapsedMillis: x.elapsedMillis, profile: x.profile, ...(await verify(diagnostic, x)) });
          await globalThis.benchmarkProgress(`${method} diagnostic ${i + 1}: ${backend} ${x.elapsedMillis.toFixed(1)} ms; parity`);
        }
        let meshPreparationProbe;
        if (method === "c2m") {
          diagnostic.worker.terminate();
          const wasm = await import("/__wasm-glue.js");
          await wasm.default({ module_or_path: `/assets/${wasmName}` });
          const reference = wasm.Cloud.parse(inputs.reference.name, new Uint8Array(await files.reference.arrayBuffer()));
          const compared = wasm.Cloud.parse(inputs.compared.name, new Uint8Array(await files.compared.arrayBuffer()));
          let built, mesh;
          try {
            reference.buildIndex(); compared.buildIndex();
            built = reference.meshDelaunay(undefined, 5_000_000); mesh = built.takeMesh();
            const vertices = mesh.rawVertices(), indices = mesh.indices(), queries = compared.rawPositions().slice(0, meshQueryPoints * 3);
            if (vertices.length / 3 !== ordinary.mesh.vertices || await digest(indices) !== ordinary.mesh.indicesSha256)
              throw new Error("Raw C2M probe uses a different TIN");
            const empty = new Float64Array(), rows = [];
            for (let i = -warmupRounds; i < repeats; i++) {
              const order = Math.abs(i) % 2 ? ["queries", "empty"] : ["empty", "queries"], row = { order, excluded: i < 0 };
              for (const kind of order) {
                const start = performance.now();
                const distances = wasm.meshDistances(vertices, indices, kind === "empty" ? empty : queries, false);
                row[`${kind}Millis`] = performance.now() - start;
                if (kind === "empty" ? distances.length !== 0 : await digest(new Float32Array(distances)) !== meshQuerySignature)
                  throw new Error("C2M preparation probe result differs from production slice");
              }
              row.resultParity = true; rows.push(row);
              await globalThis.benchmarkProgress(`c2m preparation ${i < 0 ? "warm-up" : "trial"} ${i < 0 ? i + warmupRounds + 1 : i + 1}: empty ${row.emptyMillis.toFixed(1)} / one-slice queries ${row.queriesMillis.toFixed(1)} ms; parity`);
            }
            meshPreparationProbe = { queryPoints: meshQueryPoints, queryResultSha256: meshQuerySignature, verticesSha256: await digest(vertices), rows,
              note: "Separate single-WASM invocation of the exact production meshDistances export, on the same indexed-source TIN and first production query slice. Both controls copy/validate the full mesh and build/free its BVH; the empty control performs no nearest queries. Five warm-up pairs precede alternating timed pairs. Includes JS/WASM input copies and returned-array copies. No worker messages/pool parallelism. This is a preparation-cost probe, not a directly subtractable component of the full production RPC and not a GPU speedup prediction." };
          } finally { mesh?.free(); built?.free(); compared.free(); reference.free(); }
        }
        cases.push({ method, mesh: ordinary.mesh, parameters, excludedFirstUse: setup, warmups, trials, profiles, meshPreparationProbe, result: expected.signature });
      } finally { for (const ctx of contexts) ctx.worker.terminate(); }
    }
    return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, adapter, ...description, cases };
  }, { workerName, wasmName, inputs, inputManifest, methods, repeats, profileRepeats, warmupRounds });
  for (const input of Object.values(inputs)) if (hash(readFileSync(input.path)) !== input.sha256) throw new Error("Input changed during timing");
  if (hash(readFileSync(sourcePath)) !== inputManifest.source.sha256 || hash(readFileSync(manifestPath)) !== hash(manifestBytes)) throw new Error("Source/manifest changed during timing");
  for (const [name, expected] of Object.entries(artifactHashes)) if (hash(readFileSync(resolve(assets, name))) !== expected) throw new Error("Production artifact changed during timing");
  if (hash(readFileSync(scriptPath)) !== scriptSha256) throw new Error("Script changed during timing");
  if (hash(readFileSync(gluePath)) !== glueSha256) throw new Error("Diagnostic WASM glue changed during timing");
  const median = values => { const s = [...values].sort((a, b) => a - b), n = s.length; return n % 2 ? s[n >> 1] : (s[n/2 - 1] + s[n/2]) / 2; };
  for (const entry of report.cases) {
    entry.summary = Object.fromEntries([...new Set(entry.trials.flatMap(t => t.measurements.map(m => m.backend)))].map(backend => {
      const times = entry.trials.map(t => t.measurements.find(m => m.backend === backend).elapsedMillis);
      return [backend, { medianMillis: median(times), minMillis: Math.min(...times), maxMillis: Math.max(...times) }];
    }));
    if (entry.method === "c2c") {
      const reductions = entry.trials.map(t => { const cpu = t.measurements.find(m => m.backend === "cpu").elapsedMillis;
        return 100 * (1 - t.measurements.find(m => m.backend === "webgpu").elapsedMillis / cpu); });
      entry.gpuVsCpu = { medianPairedReductionPercent: median(reductions), fasterPairs: reductions.filter(x => x > 0).length, pairs: reductions.length };
    }
    if (entry.meshPreparationProbe) {
      const rows = entry.meshPreparationProbe.rows.filter(x => !x.excluded);
      entry.meshPreparationProbe.summary = { emptyMedianMillis: median(rows.map(x => x.emptyMillis)),
        queriesMedianMillis: median(rows.map(x => x.queriesMillis)),
        medianPairedEmptyPercent: median(rows.map(x => 100 * x.emptyMillis / x.queriesMillis)) };
    }
  }
  Object.assign(report, { schemaVersion: 1, measurement: "gpu-opportunity-triage", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    artifactHashes, scriptSha256, glueSha256, browserVersion: browser.version(), inputs, inputManifest, inputManifestSha256: hash(manifestBytes),
    repeats, profileRepeats, warmupRounds, machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
    note: "Current production CPU/WASM operation times use ordinary, uninstrumented workers and defaults. Five warm-up rounds precede ten timed rounds unless overridden. C2C alternates current prepared CPU and physical WebGPU with four exact f64 verification lanes in the same analysis worker; all other GPU conversions remain unimplemented. Separate diagnostic worker contexts wrap JS-to-WASM exports and pool-worker RPC boundaries, after their own warm-up; their timings are not production wall measurements. Per-export durations include all internal WASM work and cannot isolate nearest search from PCA/projection/reduction. Cross-worker sums are aggregate CPU service time, not wall time; RPC windows overlap WASM, each other, scheduling and transfers, and must not be added. Initial observations are single excluded first-use calls after pool prewarming/loading, not cold-app medians. ICP reloads unchanged measured moving input outside timing; the timed call includes the fitted transformation and output indexing. M3C2-fresh disables normal reuse; M3C2-reused uses automatic normal reuse. Map-evaluate reuses voxel scores and distance preparation; Map-voxels recomputes only voxel scores. C2M uses the production full-input 2.5D TIN, with original source vertices and recorded duplicate-XY/edge rejection; mesh construction is excluded. These are workload profiles, not GPU speedup predictions. All calls check exact f32 result bits, f64 summaries, ICP matrix/RMS/iterations and PLY hashes; Map AWD/SCS keep the existing 1e-12 tolerance. Loading, validation/export/hashing, cleanup and worker startup are excluded. Sources are unchanged measured geometry splits, not independent surveys or accuracy ground truth. Source/manifest/input/script/artifact hashes are checked before/after. No builds/tests/scans run during timing; clocks, power, other applications and GC are uncontrolled. GPU identifiers are limited to vendor/architecture/fallback status." });
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
