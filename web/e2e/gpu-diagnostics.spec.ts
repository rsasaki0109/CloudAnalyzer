import { expect, test, type Page } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import type { C2cOutput, LoadedCloud, Request } from "../src/protocol";

test.use({ channel: "chromium", launchOptions: { args: process.env.CA_GPU_HARDWARE === "1"
  ? ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"]
  : ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader"] } });

async function inputs(page: Page) {
  for (const [role, name] of [["reference", "lidar_reference.pcd"], ["compared", "lidar_candidate.pcd"]])
    await page.route(`**/__diagnostic-${role}`, route => route.fulfill({ body: readFileSync(new URL(`../public/samples/${name}`, import.meta.url)) }));
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
}

const metricsPrelude = `
  Object.defineProperty(navigator, "hardwareConcurrency", {value:5});
  const metrics={timestampRequests:0,createdQueries:0,liveQueries:0};
  const publish=()=>self.postMessage({diagnosticMetrics:{...metrics}});
  const requestDevice=GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice=function(desc){
    if(desc?.requiredFeatures?.includes("timestamp-query"))metrics.timestampRequests++;
    publish();return requestDevice.call(this,desc);
  };
  const createQuerySet=GPUDevice.prototype.createQuerySet,destroyQuerySet=GPUQuerySet.prototype.destroy;
  const destroyed=new WeakSet();
  GPUDevice.prototype.createQuerySet=function(desc){const q=createQuerySet.call(this,desc);metrics.createdQueries++;metrics.liveQueries++;publish();return q;};
  GPUQuerySet.prototype.destroy=function(){if(!destroyed.has(this)){destroyed.add(this);metrics.liveQueries--;publish();}return destroyQuerySet.call(this);};
`;

async function production(page: Page, prelude: string, requests: Partial<Extract<Request, { kind: "c2c" }>>[]) {
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: metricsPrelude + prelude + "\n" + await response.text() });
  });
  await inputs(page);
  const workerPath = "/assets/" + readdirSync(new URL("../dist/assets/", import.meta.url)).find(n => /^worker-.*\.js$/.test(n));
  return page.evaluate(async ({ workerPath, requests }) => {
    const worker = new Worker(workerPath, { type: "module" });
    const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
    let seq = 0, metrics = { timestampRequests: 0, createdQueries: 0, liveQueries: 0 };
    worker.onmessage = ({ data }) => {
      if (data.diagnosticMetrics) metrics = data.diagnosticMetrics;
      const p = pending.get(data.seq);
      if (!data.response || !p) return;
      pending.delete(data.seq);
      data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = e => { for (const p of pending.values()) p.reject(new Error(e.message)); pending.clear(); };
    const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
      pending.set(++seq, { resolve: v => resolve(v as T), reject }); worker.postMessage({ seq, req });
    });
    const loaded: LoadedCloud[] = [];
    try {
      for (const role of ["reference", "compared"]) {
        const response = await fetch(`/__diagnostic-${role}`);
        loaded.push(await call<LoadedCloud>({ kind: "load", file: new File([await response.arrayBuffer()], role + ".pcd"), maxPoints: 4096 }));
      }
      const req = { kind: "c2c" as const, reference: loaded[0].id, compared: loaded[1].id, signed: false };
      const cpu = await call<C2cOutput>({ ...req, backend: "cpu", cpuCache: false });
      const expected = new Uint32Array(cpu.distances.buffer), outputs = [];
      for (const options of requests) {
        const value = await call<C2cOutput>({ ...req, backend: "webgpu", gpuVerifyWorkers: 4, gpuBatchSize: 512, ...options });
        const bits = new Uint32Array(value.distances.buffer);
        if (bits.length !== expected.length || bits.some((v, i) => v !== expected[i]) || !Object.keys(cpu.stats).every(k => Object.is(cpu.stats[k as keyof typeof cpu.stats], value.stats[k as keyof typeof cpu.stats])))
          throw new Error("Instrumentation changed exact distances or summary");
        outputs.push({ compute: value.compute!, metrics: { ...metrics } });
      }
      for (const cloud of loaded) await call({ kind: "remove", id: cloud.id });
      return { outputs, finalMetrics: metrics, points: expected.length };
    } finally { worker.terminate(); }
  }, { workerPath, requests });
}

test("GPU diagnostics preserve production bits, cached switches, resident lanes and bounded traces", async ({ page }) => {
  const { outputs, finalMetrics, points } = await production(page, "", [
    {}, { gpuDiagnostics: true, gpuBufferedReadback: true },
    { gpuDiagnostics: true, gpuStageQueries: true, gpuMainWorker: true },
    { gpuDiagnostics: true, gpuVerifyWorkers: 1 }, {},
    { gpuDiagnostics: true, gpuBatchSize: 8 },
  ]);
  for (const [i, { compute: c }] of outputs.entries()) {
    expect(c.backend).toBe("webgpu"); expect(c.fallback).toBeUndefined(); expect(c.verifyFallback).toBeUndefined();
    expect(c.certified).toBe(outputs[0].compute.certified);
    if (i === 0 || i === 4) expect(c.diagnostics).toBeUndefined();
    else {
      const d = c.diagnostics!;
      expect(["enabled", "unsupported"]).toContain(d.timestampQuery);
      for (const b of d.gpu) {
        for (const value of [b.queryExportMillis, b.uploadCallMillis, b.readbackMillis]) expect(value).toBeGreaterThanOrEqual(0);
        if (d.timestampQuery === "enabled") expect(b.shaderMillis).toBeGreaterThanOrEqual(0);
        else expect(b.shaderMillis).toBeUndefined();
      }
      if (i !== 5) {
        expect(d.gpu.reduce((n, b) => n + b.count, 0)).toBe(points);
        expect(d.verification.reduce((n, l) => n + l.count, 0)).toBe(points);
        expect(d.gpuBatchesOmitted).toBe(0); expect(d.verificationBatchesOmitted).toBe(0);
      }
      for (const lane of d.verification) {
        for (const value of [lane.verifyCallMillis, lane.distanceExportMillis, lane.queryExportMillis, lane.roundTripMillis]) expect(value).toBeGreaterThanOrEqual(0);
      }
    }
    if (i > 1) { expect(c.referenceReused).toBe(true); expect(c.deviceReused).toBe(true); }
  }
  expect(outputs[0].metrics).toEqual({ timestampRequests: 0, createdQueries: 0, liveQueries: 0 });
  expect(outputs[2].compute.diagnostics!.verification.some(l => l.resident)).toBe(true);
  expect(outputs[3].compute.diagnostics!.verification.every(l => l.resident)).toBe(true);
  expect(outputs[5].compute.diagnostics!.gpu).toHaveLength(256);
  expect(outputs[5].compute.diagnostics!.gpuBatchesOmitted).toBe(Math.ceil(points / 8) - 256);
  expect(outputs[5].compute.diagnostics!.verificationBatchesOmitted).toBe(Math.ceil(points / 8) - 256);
  expect(finalMetrics.liveQueries).toBe(0);
});

test("GPU diagnostics without timestamp support keep exact GPU hints and CPU lane timings", async ({ page }) => {
  const { outputs, finalMetrics } = await production(page, `
    const requestAdapter=navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter=async(...args)=>{const adapter=await requestAdapter(...args);if(adapter)Object.defineProperty(adapter,"features",{value:new Set([...adapter.features].filter(f=>f!=="timestamp-query"))});return adapter;};
  `, [{ gpuDiagnostics: true }, {}, { gpuDiagnostics: true }]);
  for (const { compute: c } of outputs) { expect(c.backend).toBe("webgpu"); expect(c.verifyFallback).toBeUndefined(); }
  expect(outputs[0].compute.diagnostics!.timestampQuery).toBe("unsupported");
  expect(outputs[0].compute.diagnostics!.gpu.every(b => b.shaderMillis === undefined)).toBe(true);
  expect(outputs[0].compute.diagnostics!.verification.length).toBeGreaterThan(0);
  expect(finalMetrics).toEqual({ timestampRequests: 0, createdQueries: 0, liveQueries: 0 });
});

test("failed GPU timestamp readback falls back exactly and retries without leaking queries", async ({ page }) => {
  const { outputs, finalMetrics } = await production(page, `
    let failed=false;
    const map=GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync=function(...args){if(this.size===16&&(this.usage&GPUBufferUsage.MAP_READ)&&!failed){failed=true;return Promise.reject(new Error("injected timestamp map failure"));}return map.apply(this,args);};
  `, [{ gpuDiagnostics: true, gpuBufferedReadback: true }, { gpuDiagnostics: true }, {}]);
  test.skip(outputs[0].compute.diagnostics!.timestampQuery === "unsupported", "No timestamp-query feature");
  expect(outputs[0].compute.backend).toBe("cpu");
  expect(outputs[0].compute.fallback).toContain("injected timestamp map failure");
  expect(outputs[1].compute.backend).toBe("webgpu"); expect(outputs[1].compute.fallback).toBeUndefined();
  expect(outputs[2].compute.backend).toBe("webgpu"); expect(outputs[2].compute.diagnostics).toBeUndefined();
  expect(finalMetrics.liveQueries).toBe(0);
});

test("GPU diagnostic slots drain both failed map directions and retain full f64 parity", async ({ page }) => {
  await inputs(page);
  const modulePath = "/__diagnostic-wasm.js";
  const wasmPath = "/assets/" + readdirSync(new URL("../dist/assets/", import.meta.url)).find(n => /^ca_wasm.*\.wasm$/.test(n));
  await page.route("**" + modulePath, route => route.fulfill({ contentType: "text/javascript", body: readFileSync(new URL("../src/wasm/ca_wasm.js", import.meta.url), "utf8") }));
  const source = stripTypeScriptTypes(readFileSync(new URL("../src/gpu-candidates.ts", import.meta.url), "utf8"), { mode: "transform" });
  const gpuModule = "data:text/javascript;base64," + Buffer.from(source).toString("base64");
  const result = await page.evaluate(async ({ modulePath, wasmPath, gpuModule }) => {
    const wasm = await import(modulePath); await wasm.default({ module_or_path: wasmPath });
    const { GpuCandidateDevice, GpuCandidateSearch } = await import(gpuModule);
    const r = wasm.Cloud.parse("r.pcd", new Uint8Array(await (await fetch("/__diagnostic-reference")).arrayBuffer()));
    const q = wasm.Cloud.parse("q.pcd", new Uint8Array(await (await fetch("/__diagnostic-compared")).arrayBuffer()));
    const index = new wasm.GpuC2cReference(r), job = new wasm.GpuC2cJob(q, index), packed = index.candidateBuffers();
    const points = packed.takePoints(), nodes = packed.takeNodes(); packed.free(); r.free(); q.free(); index.free();
    const runtime = await GpuCandidateDevice.create(false, false, true);
    if (!runtime.timestampQueries) { job.free(); runtime.dispose(); return { supported: false, checked: 0, drained: 0 }; }
    const search = await GpuCandidateSearch.create(points, nodes, runtime);
    const xyz = job.queries(0, 259), expected = job.verify(0, new Uint32Array(259).fill(0xffffffff));
    const bits = new BigUint64Array(expected.takeDistances().buffer); expected.free();
    const map = GPUBuffer.prototype.mapAsync;
    let direction = "", injected = false, waiting = 0, checked = 0, drained = 0;
    GPUBuffer.prototype.mapAsync = async function(...args: Parameters<GPUBuffer["mapAsync"]>) {
      const timestamp = this.size === 16;
      if (direction && !injected && timestamp === (direction === "timestamp")) { injected = true; throw new Error("injected map"); }
      if (direction && timestamp !== (direction === "timestamp")) {
        waiting++;
        try { await map.apply(this, args); await new Promise(resolve => setTimeout(resolve, 40)); }
        finally { waiting--; }
      } else await map.apply(this, args);
    };
    const verify = (candidates: Uint32Array) => {
      const result = job.verify(0, candidates);
      try { const actual = new BigUint64Array(result.takeDistances().buffer); if (actual.some((v, i) => v !== bits[i])) throw new Error("Full f64 parity changed"); checked++; }
      finally { result.free(); }
    };
    const timing = () => ({ first: 0, count: 259, queryExportMillis: 0 });
    try {
      for (const fail of ["timestamp", "candidate"]) {
        direction = fail; injected = false;
        await search.find(xyz, false, 64, false, false, timing()).then(() => { throw new Error("Missing injected failure"); }, (e: Error) => { if (!e.message.includes("injected map")) throw e; });
        if (!injected || waiting !== 0) throw new Error("Slot released before the other map settled");
        drained++; direction = "";
        verify(await search.find(xyz, false, 64, false, false, timing()));
      }
      const pending = await Promise.all(Array.from({ length: 5 }, () => search.find(xyz, false, 64, true, false, timing())));
      pending.forEach(verify);
      return { supported: true, checked, drained };
    } finally { GPUBuffer.prototype.mapAsync = map; search.dispose(); runtime.dispose(); job.free(); }
  }, { modulePath, wasmPath, gpuModule });
  test.skip(!result.supported, "No timestamp-query feature");
  expect(result).toEqual({ supported: true, checked: 7, drained: 2 });
});
