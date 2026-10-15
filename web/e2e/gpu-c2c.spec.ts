import { expect, type Page, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import type { C2cOutput, LoadedCloud, Request } from "../src/protocol";
import { summaryScenario } from "./distance-summary-scenario";

// Software WebGPU is portable in CI; opt into the physical adapter locally.
test.use({ channel: "chromium", launchOptions: { args: process.env.CA_GPU_HARDWARE === "1"
  ? ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"]
  : ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader"] } });

function ply(count: number, lift: number): Buffer {
  const header = Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
  const body = Buffer.alloc(count * 24);
  for (let i = 0; i < count; i++) {
    // Duplicate points, UTM offsets, a wide span and submillimetre differences.
    const j = i % 4000;
    const xyz = [368_000 + (j % 80) * 0.1, 3_955_000 + Math.floor(j / 80) * 0.1,
      40 + Math.sin(j / 7) * 0.2 + lift];
    if (i === 0) xyz[0] += 100_000;
    xyz.forEach((v, a) => body.writeDoubleLE(v, i * 24 + a * 8));
  }
  return Buffer.concat([header, body]);
}

async function open(page: Page, count = 4000): Promise<void> {
  await page.goto("/");
  await page.locator("#file-input").setInputFiles([0, 0.000_009].map((lift, i) => ({
    name: i ? "compared.ply" : "reference.ply", mimeType: "application/octet-stream", buffer: ply(count, lift),
  })));
  await expect(page.locator("#status")).toContainText(`Loaded compared.ply: ${count.toLocaleString()} points`);
}

async function runAndExport(page: Page, backend: string, expectedStatus?: string): Promise<Buffer> {
  await page.locator("#c2c-backend").selectOption(backend);
  await page.locator("#c2c-run").click();
  await expect(page.locator("#status")).toContainText("C2C distance computed");
  if (expectedStatus) await expect(page.locator("#status")).toContainText(expectedStatus);
  const download = page.waitForEvent("download");
  await page.locator("#export-ply").click();
  const chunks: Buffer[] = [];
  for await (const chunk of await (await download).createReadStream()) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks);
}

async function alterWorker(page: Page, prelude: string): Promise<void> {
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    // Exercise lane growth through eight slots independently of the host's
    // reported CPU count. Actual low-capacity scheduling is checked below.
    const capacity = 'Object.defineProperty(navigator, "hardwareConcurrency", { value: 9, configurable: true });\n';
    await route.fulfill({ response, body: capacity + prelude + "\n" + await response.text() });
  });
}

test.beforeEach(async ({ page }) => { await alterWorker(page, ""); });

const gpuMetrics = `
  const metrics = { devices: 0, pipelines: 0, buffers: 0, maxScopes: 0 };
  let scopes = 0;
  const publish = () => self.postMessage({ gpuMetrics: { ...metrics } });
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function(...args) {
    const device = await requestDevice.apply(this, args);
    metrics.devices++; publish(); return device;
  };
  const pipeline = GPUDevice.prototype.createComputePipelineAsync;
  GPUDevice.prototype.createComputePipelineAsync = function(...args) {
    metrics.pipelines++; publish(); return pipeline.apply(this, args);
  };
  const createBuffer = GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer = function(...args) {
    const buffer = createBuffer.apply(this, args);
    metrics.buffers++; publish(); return buffer;
  };
  const destroyed = new WeakSet();
  const destroyBuffer = GPUBuffer.prototype.destroy;
  GPUBuffer.prototype.destroy = function(...args) {
    if (!destroyed.has(this)) { destroyed.add(this); metrics.buffers--; publish(); }
    return destroyBuffer.apply(this, args);
  };
  const push = GPUDevice.prototype.pushErrorScope, pop = GPUDevice.prototype.popErrorScope;
  GPUDevice.prototype.pushErrorScope = function(...args) {
    const result = push.apply(this, args);
    metrics.maxScopes = Math.max(metrics.maxScopes, ++scopes); publish(); return result;
  };
  GPUDevice.prototype.popErrorScope = function(...args) { scopes--; return pop.apply(this, args); };
`;

async function workerScenario(page: Page, concurrent: boolean, gpuVerifyWorkers = 1, gpuStageQueries = false, gpuMainWorker = false, gpuBufferedReadback = false) {
  const workerPath = "/assets/" + readdirSync(new URL("../dist/assets/", import.meta.url)).find(n => /^worker-.*\.js$/.test(n));
  return page.evaluate(async ({ workerPath, concurrent, gpuVerifyWorkers, gpuStageQueries, gpuMainWorker, gpuBufferedReadback }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    let metrics = { devices: 0, pipelines: 0, buffers: 0, maxScopes: 0 };
    const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    let submitted!: () => void;
    const firstSubmit = new Promise<void>(resolve => { submitted = resolve; });
    worker.onmessage = ({ data }) => {
      if (data.gpuMetrics) metrics = data.gpuMetrics;
      if (data.gpuSubmitted) submitted();
      const request = pending.get(data.seq);
      if (!data.response || !request) return;
      pending.delete(data.seq);
      data.response.ok ? request.resolve(data.response.value) : request.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const p of pending.values()) p.reject(new Error(event.message)); pending.clear(); };
    const call = <T>(req: Request): Promise<T> => new Promise((resolve, reject) => {
      pending.set(++seq, { resolve: value => resolve(value as T), reject });
      worker.postMessage({ seq, req });
    });
    const fixture = (lift: number) => {
      const header = new TextEncoder().encode("ply\nformat binary_little_endian 1.0\nelement vertex 4096\nproperty double x\nproperty double y\nproperty double z\nend_header\n");
      const bytes = new Uint8Array(header.length + 4096 * 24);
      bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < 4096; i++) {
        [368_000 + (i % 64) * 0.1, 3_955_000 + Math.floor(i / 64) * 0.1, 40 + lift]
          .forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    const parity = (cpu: C2cOutput, gpu: C2cOutput) => {
      if (gpu.compute?.backend !== "webgpu") throw new Error(`GPU fell back: ${gpu.compute?.fallback}`);
      const a = new Uint32Array(cpu.distances.buffer), b = new Uint32Array(gpu.distances.buffer);
      if (a.length !== 4096 || b.length !== a.length || a.some((v, i) => v !== b[i]) ||
        JSON.stringify(cpu.stats) !== JSON.stringify(gpu.stats)) throw new Error("CPU/GPU parity failed");
      return gpu.compute;
    };
    try {
      let reference = await call<LoadedCloud>({ kind: "load", file: fixture(0), maxPoints: 4096 });
      let compared = await call<LoadedCloud>({ kind: "load", file: fixture(0.05), maxPoints: 4096 });
      const c2c = (backend: "cpu" | "webgpu", ref = reference.id, gpuPrefetch = true, gpuVerifyBounds = true, gpuVerifySimd = true) => call<C2cOutput>({
        kind: "c2c", compared: compared.id, reference: ref, backend, signed: false, gpuPrefetch, gpuVerifyBounds, gpuVerifySimd, gpuVerifyWorkers, gpuStageQueries, gpuMainWorker,
        gpuBufferedReadback, gpuBatchSize: gpuBufferedReadback ? 512 : undefined,
      });
      const before = await c2c("cpu");
      const transform = () => call({ kind: "transform", id: reference.id,
        matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.03, 0, 0, 0, 1] });
      const computed = [];
      if (concurrent) {
        const first = c2c("webgpu"), second = c2c("webgpu");
        await firstSubmit; // Both query snapshots exist; first readback is paused.
        await transform();
        const after = await c2c("cpu");
        const third = c2c("webgpu");
        await call({ kind: "remove", id: reference.id });
        await call({ kind: "remove", id: compared.id });
        computed.push(parity(before, await first), parity(before, await second), parity(after, await third));
        return { computed, metrics, changed: JSON.stringify(before.stats) !== JSON.stringify(after.stats) };
      }
      computed.push(parity(before, await c2c("webgpu", reference.id, false, false, false)), parity(before, await c2c("webgpu")));
      await transform();
      const after = await c2c("cpu");
      computed.push(parity(after, await c2c("webgpu")), parity(after, await c2c("webgpu")));
      const self = await c2c("cpu", compared.id);
      computed.push(parity(self, await c2c("webgpu", compared.id)));
      computed.push(parity(after, await c2c("webgpu")));
      // A fresh compared cloud can reuse the same unchanged reference.
      await call({ kind: "remove", id: compared.id });
      compared = await call<LoadedCloud>({ kind: "load", file: fixture(0.05), maxPoints: 4096 });
      computed.push(parity(await c2c("cpu"), await c2c("webgpu")));
      await call({ kind: "remove", id: reference.id });
      reference = await call<LoadedCloud>({ kind: "load", file: fixture(0), maxPoints: 4096 });
      computed.push(parity(await c2c("cpu"), await c2c("webgpu")));
      return { computed, metrics, changed: JSON.stringify(before.stats) !== JSON.stringify(after.stats) };
    } finally { worker.terminate(); }
  }, { workerPath, concurrent, gpuVerifyWorkers, gpuStageQueries, gpuMainWorker, gpuBufferedReadback });
}

test("GPU batches preserve exact CPU summaries through median-control switches", async ({ page }) => {
  const cases = await summaryScenario(page, "webgpu");
  expect(cases.map(c => [c.n, c.workers, c.reused])).toEqual([
    [200_013, 2, [false, true, true, true]], [200_014, 2, [false, true, true, true]],
  ]);
});

test("cached references are reused and invalidated after transform, switch and removal", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, gpuMetrics);
  const { computed, metrics, changed } = await workerScenario(page, false);
  expect(computed.map(c => c.referenceReused)).toEqual([false, true, false, true, false, false, true, false]);
  expect(computed.map(c => c.prefetch)).toEqual([false, true, true, true, true, true, true, true]);
  expect(computed.map(c => c.verifyBounds)).toEqual([false, true, true, true, true, true, true, true]);
  expect(computed.map(c => c.verifySimd)).toEqual([false, true, true, true, true, true, true, true]);
  expect(computed.map(c => c.deviceReused)).toEqual([false, true, true, true, true, true, true, true]);
  expect(metrics).toEqual({ devices: 1, pipelines: 1, buffers: 5, maxScopes: 3 });
  expect(computed.every(c => c.certified > 0 && c.certified <= 4096)).toBe(true);
  expect(changed).toBe(true);
});

test("WASM SIMD leaf verification preserves f64 bits against scalar and brute force", async ({ page }) => {
  const modulePath = "/__simd-wasm.js";
  const wasmPath = "/assets/" + readdirSync(new URL("../dist/assets/", import.meta.url)).find(n => /^ca_wasm.*\.wasm$/.test(n));
  await page.route("**" + modulePath, route => route.fulfill({ contentType: "text/javascript",
    body: readFileSync(new URL("../src/wasm/ca_wasm.js", import.meta.url), "utf8") }));
  await page.goto("/");
  const checked = await page.evaluate(async ({ modulePath, wasmPath }) => {
    const { default: init, Cloud, GpuC2cReference, GpuC2cJob, GpuC2cVerifier } = await import(modulePath);
    await init({ module_or_path: wasmPath });
    const cloud = (points: number[][]) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + points.length * 24);
      bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      points.forEach((p, i) => p.forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true)));
      return Cloud.parse("simd.ply", bytes);
    };
    const bits = (values: Float64Array) => [...new Uint32Array(values.buffer, values.byteOffset, values.length * 2)];
    let checked = 0;
    for (const count of [1, 2, 3, 15, 16, 17, 31, 33, 257]) {
      for (const scale of [1e-160, 1e-150, 1, 1e150, 1e200]) {
        const origin = scale === 1 ? [368_000, 3_955_000, 40] : [0, 0, 0];
        const points = Array.from({ length: count }, (_, i) => [
          origin[0] + (i % 17 - 8) * scale,
          origin[1] + (i % 5 - 2) * scale,
          origin[2] + (i % 3 - 1) * scale,
        ]);
        const queries = points.map((p, i) => [p[0] + (i % 2 ? 1e-6 : 0.35) * scale,
          p[1] - 0.25 * scale, p[2] + 9e-6 * scale]);
        // Duplicate ties, exact hits, diagonal exclusions and far/wrong seeds.
        queries.push(...points.slice(0, 3), origin.map(v => v + 20 * scale));
        const reference = cloud(points), compared = cloud(queries);
        const index = new GpuC2cReference(reference);
        const copy = GpuC2cReference.fromXyz(index.xyz());
        const snapshot = index.snapshot();
        const imported = GpuC2cReference.fromSnapshot(snapshot);
        const reused = GpuC2cReference.fromSnapshot(snapshot, true);
        const recomputed = GpuC2cReference.fromSnapshot(snapshot, false);
        snapshot.fill(0); // Each verifier owns its decoded tree.
        const vector = new GpuC2cJob(compared, index), scalar = new GpuC2cJob(compared, index);
        try {
          scalar.useSimd(false);
          if (!vector.simdEnabled || scalar.simdEnabled) throw new Error("WASM SIMD selection was not applied");
          const q = compared.positionsRange(0, compared.length), p = reference.positionsRange(0, reference.length);
          const expected = Float64Array.from({ length: compared.length }, (_, i) => {
            let best = Infinity;
            for (let j = 0; j < reference.length; j++) {
              const dx = p[j * 3] - q[i * 3], dy = p[j * 3 + 1] - q[i * 3 + 1], dz = p[j * 3 + 2] - q[i * 3 + 2];
              best = Math.min(best, (dx * dx + dy * dy) + dz * dz);
            }
            return Math.sqrt(best);
          });
          const seeds = Uint32Array.from({ length: compared.length }, (_, i) => [0, count - 1, Math.floor(count / 2), 0xffffffff][i % 4]);
          const a = vector.complete(0, seeds), b = scalar.complete(0, seeds);
          const resident = vector.verify(0, seeds), certificatesBefore = vector.certified;
          try {
            if (JSON.stringify(bits(resident.takeDistances())) !== JSON.stringify(bits(a)) ||
              resident.certified !== certificatesBefore || vector.certified !== certificatesBefore)
              throw new Error("Resident f64 verification changed results or mutated serial certificates");
          } finally { resident.free(); }
          if (JSON.stringify(bits(a)) !== JSON.stringify(bits(b)) || JSON.stringify(bits(a)) !== JSON.stringify(bits(expected)) ||
            vector.certified !== scalar.certified) throw new Error(`SIMD parity failed at ${count} points, scale ${scale}`);
          for (const worker of [copy, imported, reused, recomputed]) {
            const verified = worker.verify(vector.queriesF64(0, seeds.length), seeds, true, true);
            try {
              if (JSON.stringify(bits(verified.takeDistances())) !== JSON.stringify(bits(a)) || verified.certified !== vector.certified)
                throw new Error("Independent f64 reference verification changed distance bits/certificates");
            } finally { verified.free(); }
            const partial = worker.verify(vector.queriesF64(1, seeds.length - 2), seeds.slice(1, -1), false, false);
            try {
              if (JSON.stringify(bits(partial.takeDistances())) !== JSON.stringify(bits(a.slice(1, -1))))
                throw new Error("Partial scalar verification changed distance bits");
            } finally { partial.free(); }
            for (const bounds of [false, true]) for (const simd of [false, true]) {
              const old = recomputed.verify(vector.queriesF64(0, seeds.length), seeds, bounds, simd);
              const next = worker.verify(vector.queriesF64(0, seeds.length), seeds, bounds, simd);
              try {
                if (JSON.stringify(bits(old.takeDistances())) !== JSON.stringify(bits(next.takeDistances())) || old.certified !== next.certified)
                  throw new Error("Reused validated bounds changed f64 bits or certificates");
              } finally { old.free(); next.free(); }
            }
            const xyz = vector.queriesF64(0, seeds.length);
            const staged = new GpuC2cVerifier(worker, xyz);
            xyz.fill(0); // Constructor owns the original values in WASM.
            try {
              const verified = staged.verify(0, seeds, true, true);
              try {
                if (JSON.stringify(bits(verified.takeDistances())) !== JSON.stringify(bits(a)) || verified.certified !== vector.certified)
                  throw new Error("Staged f64 queries changed distance bits/certificates");
              } finally { verified.free(); }
              const partial = staged.verify(1, seeds.slice(1, -1), false, false);
              try {
                if (JSON.stringify(bits(partial.takeDistances())) !== JSON.stringify(bits(a.slice(1, -1))))
                  throw new Error("Partial staged verification changed distance bits");
              } finally { partial.free(); }
              let rejected = false;
              try { staged.verify(seeds.length, new Uint32Array([0]), true, true); }
              catch { rejected = true; }
              if (!rejected) throw new Error("Out-of-range staged verification was accepted");
            } finally { staged.free(); }
          }
          if (count === 1 && scale === 1) {
            let rejectedResident = false;
            try { vector.verify(seeds.length, new Uint32Array([0])); } catch { rejectedResident = true; }
            if (!rejectedResident || vector.certified !== certificatesBefore) throw new Error("Invalid resident range changed certificates");
            for (const [lane, workers, batch] of [[0, 0, 1], [4, 4, 1], [0, 9, 1], [0, 4, 0], [0, 4, 262145]]) {
              let rejected = false;
              try { vector.workerQueries(lane, workers, batch); } catch { rejected = true; }
              if (!rejected) throw new Error("Invalid staged partition was accepted");
            }
            for (const xyz of [new Float64Array(0), new Float64Array(4)]) {
              let rejected = false;
              try { new GpuC2cVerifier(index, xyz); } catch { rejected = true; }
              if (!rejected) throw new Error("Invalid staged coordinates were accepted");
            }
          }
          checked += a.length;
        } finally { vector.free(); scalar.free(); copy.free(); imported.free(); reused.free(); recomputed.free(); index.free(); compared.free(); reference.free(); }
      }
    }
    return checked;
  }, { modulePath, wasmPath });
  expect(checked).toBeGreaterThan(1500);
});

test("overlapping GPU jobs retain snapshots through reference edits and source removal", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, gpuMetrics + `
    const submit = GPUQueue.prototype.submit;
    GPUQueue.prototype.submit = function(...args) { submit.apply(this, args); self.postMessage({ gpuSubmitted: true }); };
    const map = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = async function(...args) {
      await map.apply(this, args);
      await new Promise(resolve => setTimeout(resolve, 150));
    };`);
  const { computed, metrics, changed } = await workerScenario(page, true);
  expect(computed.map(c => c.referenceReused)).toEqual([false, true, false]);
  expect(computed[2].deviceReused).toBe(true);
  expect(metrics).toEqual({ devices: 1, pipelines: 1, buffers: 0, maxScopes: 3 });
  expect(changed).toBe(true);
});

test("selecting WebGPU prepares one device before computing and retries failed preparation", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, `
    const pipeline = GPUDevice.prototype.createComputePipelineAsync;
    let calls = 0;
    GPUDevice.prototype.createComputePipelineAsync = function(...args) {
      if (++calls === 1) {
        self.postMessage({ preparationFailed: true });
        return Promise.reject(new Error("Injected pipeline preparation failure"));
      }
      return pipeline.apply(this, args).then(result => { self.postMessage({ preparationReady: true }); return result; });
    };`);
  // Observe preparation messages from the app's actual worker.
  await page.addInitScript(() => {
    const OriginalWorker = window.Worker;
    let failed = false, ready = 0;
    window.Worker = class extends OriginalWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener("message", ({ data }) => {
          if (data.preparationFailed) failed = true;
          if (data.preparationReady) ready++;
        });
      }
    };
    Object.defineProperty(window, "gpuPreparation", { get: () => ({ failed, ready }) });
  });
  await open(page);
  const cpu = await runAndExport(page, "cpu");
  await page.locator("#c2c-backend").selectOption("webgpu");
  await expect.poll(() => page.evaluate(() => (window as unknown as { gpuPreparation: { failed: boolean } }).gpuPreparation.failed)).toBe(true);
  // The failed setup is removed from the cache; selecting again can prepare
  // successfully before the first GPU job, without changing the CPU result.
  await page.locator("#c2c-backend").selectOption("webgpu");
  await expect.poll(() => page.evaluate(() => (window as unknown as { gpuPreparation: { ready: number } }).gpuPreparation.ready)).toBe(1);
  expect(await runAndExport(page, "webgpu", "with WebGPU")).toEqual(cpu);
  expect(await page.evaluate(() => (window as unknown as { gpuPreparation: { ready: number } }).gpuPreparation.ready)).toBe(1);
});

test("WebGPU C2C export matches CPU exactly across batches and repeated runs", async ({ page }) => {
  await page.goto("/");
  const supported = await page.evaluate(async () => !!await navigator.gpu?.requestAdapter());
  test.skip(!supported, "No WebGPU adapter; CPU fallback is tested separately");
  await open(page, 262_145);
  const cpu = await runAndExport(page, "cpu");
  for (let i = 0; i < 2; i++) {
    expect(await runAndExport(page, "webgpu", "with WebGPU")).toEqual(cpu);
  }
});

test("missing WebGPU uses CPU with an identical export", async ({ page }) => {
  await alterWorker(page, `Object.defineProperty(navigator, 'gpu', { value: undefined });`);
  await open(page);
  const cpu = await runAndExport(page, "cpu");
  expect(await runAndExport(page, "webgpu", "CPU fallback: WebGPU is unavailable")).toEqual(cpu);
  expect(await runAndExport(page, "webgpu-parallel", "CPU fallback: WebGPU is unavailable")).toEqual(cpu);
});

for (const backend of ["webgpu", "webgpu-parallel"]) test(`device loss after the first batch preserves the complete CPU result (${backend})`, async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, `
    // Exercise loss after a full maximum-sized batch, then a partial batch.
    self.addEventListener("message", ({ data }) => {
      if (data.req?.kind === "c2c" && data.req.backend === "webgpu") data.req.gpuBatchSize = 262144;
    });
    const requestDevice = GPUAdapter.prototype.requestDevice;
    let devices = 0;
    GPUAdapter.prototype.requestDevice = async function(...args) {
      const device = await requestDevice.apply(this, args);
      const lose = ++devices === 1;
      const submit = device.queue.submit.bind(device.queue);
      let calls = 0;
      device.queue.submit = (...args) => { submit(...args); if (++calls === 2 && lose) device.destroy(); };
      return device;
    };`);
  await open(page, 262_145);
  const cpu = await runAndExport(page, "cpu");
  expect(await runAndExport(page, backend, "CPU fallback:")).toEqual(cpu);
  expect(await runAndExport(page, backend, "with WebGPU")).toEqual(cpu);
});

test("a WebGPU buffer limit uses CPU instead of a partial result", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, `
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function(...args) {
      const device = await requestDevice.apply(this, args);
      const limits = device.limits;
      Object.defineProperty(device, 'limits', { value: {
        maxStorageBufferBindingSize: 16384, maxBufferSize: limits.maxBufferSize,
        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
      } });
      return device;
    };`);
  await open(page);
  const cpu = await runAndExport(page, "cpu");
  expect(await runAndExport(page, "webgpu", "CPU fallback: WebGPU buffer limit exceeded")).toEqual(cpu);
});

async function batchScenario(page: Page, count: number, limitedBy?: string, compareHints = false, compareWorkgroups = false,
  compareVerify = false, failVerify?: "prepare" | "batch" | "snapshot" | "queries" | "staged-batch" | "release", compareBatches = false,
  compareQueries = false, idleLoss = false, mainWorker = false, bufferedReadback = false) {
  const failure = failVerify ? `
    const post = Worker.prototype.postMessage;
    let attempts = 0, failed = false;
    Worker.prototype.postMessage = function(request, ...args) {
      if (!failed && request.kind === "${failVerify === "release" ? "gpu-query-release" : failVerify === "queries" ? "gpu-query-prepare" : ["batch", "staged-batch"].includes(failVerify) ? "gpu-verify" : "gpu-verify-prepare"}" &&
        ${failVerify === "staged-batch" ? '"queryJob" in request &&' : ""}
        ++attempts === ${["batch", "staged-batch"].includes(failVerify) ? 5 : 1}) {
        failed = true;
        ${failVerify === "snapshot" ? "request.reference[4] = 0;" : `queueMicrotask(() => this.dispatchEvent(new ErrorEvent("error", { message: "injected verification worker failure" })));
        return;`}
      }
      return post.call(this, request, ...args);
    };` : "";
  const idle = idleLoss ? `
    const NativeWorker = Worker, lanes = [];
    Worker = class extends NativeWorker { constructor(...args) { super(...args); lanes.push(this); } };
    self.addEventListener("message", event => {
      if (event.data.killIdleGpuLane) {
        event.stopImmediatePropagation();
        lanes[0].dispatchEvent(new ErrorEvent("error", { message: "Injected idle verification worker loss" }));
        self.postMessage({ idleGpuLaneLost: true });
      }
    });` : "";
  await alterWorker(page, gpuMetrics + failure + idle + (limitedBy ? `
    const originalDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function(...args) {
      const device = await originalDevice.apply(this, args);
      const limits = device.limits;
      Object.defineProperty(device, "limits", { value: {
        maxComputeWorkgroupsPerDimension: ${limitedBy === "dispatch" ? 256 : "limits.maxComputeWorkgroupsPerDimension"},
        maxStorageBufferBindingSize: ${limitedBy === "storage" ? 262144 : "limits.maxStorageBufferBindingSize"},
        maxBufferSize: ${limitedBy === "buffer" ? 262144 : "limits.maxBufferSize"},
        maxComputeInvocationsPerWorkgroup: ${limitedBy === "invocations" ? 128 : "limits.maxComputeInvocationsPerWorkgroup"},
        maxComputeWorkgroupSizeX: ${limitedBy === "workgroup-x" ? 128 : "limits.maxComputeWorkgroupSizeX"},
      } });
      return device;
    };` : ""));
  await page.route("**/__gpu-batches", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__gpu-batches");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n))!;
  return page.evaluate(async ({ workerPath, count, compareHints, compareWorkgroups, limitedBy, compareVerify, failVerify, compareBatches, compareQueries, idleLoss, mainWorker, bufferedReadback }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0, metrics = { devices: 0, pipelines: 0, buffers: 0, maxScopes: 0 };
    let idleLost!: () => void;
    const idleDone = new Promise<void>(resolve => { idleLost = resolve; });
    const waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (data.idleGpuLaneLost) { idleLost(); return; }
      if (data.gpuMetrics) { metrics = data.gpuMetrics; return; }
      if (!data.response) return;
      const p = waiting.get(data.seq)!;
      waiting.delete(data.seq);
      data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const p of waiting.values()) p.reject(new Error(event.message)); waiting.clear(); };
    const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
      const id = ++seq;
      waiting.set(id, { resolve: value => resolve(value as T), reject });
      worker.postMessage({ seq: id, req });
    });
    const fixture = (n: number, lift: number) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 24);
      bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < n; i++) {
        const j = i % 4096;
        [368_000 + (j % 64) * 0.1, 3_955_000 + Math.floor(j / 64) * 0.1, 40 + Math.sin(j / 20) * 0.1 + lift]
          .forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    try {
      const reference = await call<LoadedCloud>({ kind: "load", file: fixture(4096, 0), maxPoints: 4096 });
      const compared = await call<LoadedCloud>({ kind: "load", file: fixture(count, 0.000_009), maxPoints: count });
      const c2c = (backend: "cpu" | "webgpu", gpuBatchSize?: number, gpuLeafOnly?: boolean, gpuWorkgroupSize?: number, gpuVerifyWorkers = 1,
        controls: { gpuPrefetch?: boolean; gpuVerifyBounds?: boolean; gpuVerifySimd?: boolean; gpuVerifySnapshot?: boolean; gpuStageQueries?: boolean; gpuReuseImportBounds?: boolean } = {}) => call<C2cOutput>({ kind: "c2c",
        reference: reference.id, compared: compared.id, signed: false, backend, gpuBatchSize, gpuLeafOnly, gpuWorkgroupSize, gpuVerifyWorkers, gpuMainWorker: mainWorker,
        gpuReuseImportBounds: compareVerify ? true : undefined, gpuBufferedReadback: bufferedReadback, ...controls });
      const cpu = await c2c("cpu");
      const parity = (result: C2cOutput) => {
        const a = new Uint32Array(cpu.distances.buffer), b = new Uint32Array(result.distances.buffer);
        if (a.length !== count || b.length !== count || a.some((v, i) => v !== b[i]) ||
          JSON.stringify(cpu.stats) !== JSON.stringify(result.stats)) throw new Error("Batch C2C parity failed");
      };
      const computed = [];
      // Grow twice, then shrink bindings while retaining the largest buffers.
      for (const size of [65_536, 131_072, 262_144, 65_536]) {
        const result = await c2c("webgpu", size);
        parity(result);
        if (result.compute?.backend !== "webgpu") throw new Error(`Unexpected fallback: ${result.compute?.fallback}`);
        computed.push(result.compute);
      }
      const hints = [];
      if (compareHints) for (const mode of [false, true, false, true]) {
        const result = await c2c("webgpu", 65_536, mode);
        parity(result);
        if (result.compute?.backend !== "webgpu") throw new Error(`Unexpected hint fallback: ${result.compute?.fallback}`);
        hints.push(result.compute);
      }
      const workgroups = [];
      const verifications = [];
      const setups = [];
      const parallelBatches = [];
      const defaults = [];
      const queryModes = [];
      if (compareQueries) {
        for (const [size, workers] of [[65_536, 4], [131_072, 4], [262_144, 4], [65_536, 2], [65_536, 8]]) {
          for (const gpuStageQueries of [true, false]) {
            const result = await c2c("webgpu", size, false, 64, workers, { gpuStageQueries });
            parity(result);
            if (result.compute?.backend !== "webgpu" || result.compute.verifyFallback && !failVerify)
              throw new Error("Query staging lost GPU candidates or verification workers");
            queryModes.push(result.compute);
          }
        }
        // More requested lanes than queries per batch, including a tiny tail.
        if (count <= 19) for (const size of [1, 3, 7, 17]) {
          const result = await c2c("webgpu", size, true, 32, 8,
            { gpuStageQueries: true, gpuPrefetch: false, gpuVerifyBounds: false, gpuVerifySimd: false });
          parity(result);
          if (!result.compute?.verifyStagedQueries || result.compute.verifyFallback) throw new Error("Tiny staged partition failed");
          queryModes.push(result.compute);
        }
      }
      if (compareBatches) for (const size of [65_536, 131_072, 262_144, 65_536]) {
        const result = await c2c("webgpu", size, false, 64, 4);
        parity(result);
        if (result.compute?.backend !== "webgpu" || result.compute.verifyWorkers !== 4 || result.compute.verifyFallback)
          throw new Error("Parallel batch comparison lost GPU or verification workers");
        parallelBatches.push(result.compute);
      }
      if (compareBatches) for (const workers of [1, 2, 4, 8]) {
        const result = await c2c("webgpu", undefined, false, 64, workers);
        parity(result);
        if (result.compute?.backend !== "webgpu" || result.compute.verifyWorkers !== workers || result.compute.verifyFallback)
          throw new Error("Default batch comparison lost GPU or verification workers");
        defaults.push(result.compute);
      }
      if (compareVerify) {
        for (const workers of [4, 2, 4, 1, 8, 4]) {
          const result = await c2c("webgpu", 65_536, false, 64, workers);
          parity(result);
          if (result.compute?.backend !== "webgpu") throw new Error("Verification unexpectedly lost GPU candidates");
          if (result.compute.verifyFallback && !failVerify) throw new Error(result.compute.verifyFallback);
          verifications.push(result.compute);
          if (idleLoss && verifications.length === 1) {
            worker.postMessage({ killIdleGpuLane: true }); await idleDone;
          }
        }
        for (const controls of [{ gpuPrefetch: false }, { gpuVerifyBounds: false }, { gpuVerifySimd: false }]) {
          const result = await c2c("webgpu", 65_536, true, 32, 4, controls);
          parity(result);
          if (result.compute?.verifyWorkers !== 4 || result.compute.verifyFallback) throw new Error("Verification controls did not use pool");
        }
        for (const [snapshot, reuse] of [[false, false], [true, false], [true, true], [true, false], [true, true], [false, true]]) {
          const result = await c2c("webgpu", 65_536, false, 64, 4, { gpuVerifySnapshot: snapshot, gpuReuseImportBounds: reuse });
          parity(result);
          if (result.compute?.verifyWorkers !== 4 || result.compute.verifyFallback) throw new Error("Setup controls did not use pool");
          setups.push(result.compute);
        }
        const concurrentImports = await Promise.all([false, true].map(gpuReuseImportBounds =>
          c2c("webgpu", 65_536, false, 64, 4, { gpuReuseImportBounds })));
        concurrentImports.forEach((result, i) => {
          parity(result);
          if (result.compute?.verifyReuseImportBounds !== Boolean(i) || !result.compute.verifyReferenceReused || result.compute.verifyFallback)
            throw new Error("Concurrent import modes mixed their cache state");
        });
        for (const workers of [0, -1, 1.5, 9, NaN]) {
          const result = await c2c("webgpu", 65_536, false, 64, workers);
          parity(result);
          if (result.compute?.backend !== "cpu" || !result.compute.fallback?.includes("verification workers"))
            throw new Error("Invalid verification workers must use complete CPU fallback");
        }
      }
      if (compareWorkgroups) {
        for (const leafOnly of [false, true]) for (const size of [64, 32, 128, 256, 64]) {
          const result = await c2c("webgpu", 65_536, leafOnly, size);
          parity(result);
          const unsupported = size === 256 && ["invocations", "workgroup-x"].includes(limitedBy ?? "");
          if (result.compute?.backend !== (unsupported ? "cpu" : "webgpu"))
            throw new Error(`Unexpected workgroup fallback: ${result.compute?.fallback}`);
          if (unsupported && !result.compute?.fallback?.includes("device limits")) throw new Error("Missing workgroup limit fallback");
          workgroups.push(result.compute!);
        }
        for (const size of [0, -1, 1.5, 31, NaN, 512]) {
          const result = await c2c("webgpu", 65_536, false, size);
          parity(result);
          if (result.compute?.backend !== "cpu" || !result.compute.fallback?.includes("workgroup size") || result.compute.workgroupSize !== 0)
            throw new Error("Invalid workgroup must use the complete CPU fallback");
        }
      }
      for (const size of [0, -1, 1.5, NaN, 262_145]) {
        const result = await c2c("webgpu", size);
        parity(result);
        if (result.compute?.backend !== "cpu" || !result.compute.fallback?.includes("batch size") || result.compute.batchSize !== 0)
          throw new Error("Invalid batch must use the complete CPU fallback");
      }
      const recovered = await c2c("webgpu", 65_536);
      parity(recovered);
      const active = { ...metrics };
      await call({ kind: "remove", id: reference.id });
      await call({ kind: "remove", id: compared.id });
      return { computed, hints, workgroups, verifications, setups, parallelBatches, defaults, queryModes, active, released: metrics, recovered: recovered.compute };
    } finally { worker.terminate(); }
  }, { workerPath, count, compareHints, compareWorkgroups, limitedBy, compareVerify, failVerify, compareBatches, compareQueries, idleLoss, mainWorker, bufferedReadback });
}

for (const [count, limit] of [[19, undefined], [65_537, undefined], [262_145, "dispatch"]] as const) {
  test(`buffered readbacks preserve exact verification, staging and device limits (${count})`, async ({ page }) => {
    const { computed, queryModes, active, released } = await batchScenario(page, count, limit,
      false, false, false, undefined, false, true, false, false, true);
    expect([...computed, ...queryModes].every(c => c.bufferedReadback === c.prefetch)).toBe(true);
    expect(new Set([...computed, ...queryModes].map(c => c.certified)).size).toBe(1);
    expect(active.maxScopes).toBe(3);
    expect(active.buffers).toBe(8);
    expect(released.buffers).toBe(0);
  });
}

test("buffered readbacks survive overlapping jobs, source edits and removal", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, gpuMetrics + `
    const submit = GPUQueue.prototype.submit; let notified = false;
    GPUQueue.prototype.submit = function(...args) {
      submit.apply(this, args);
      if (!notified) { notified = true; self.postMessage({ gpuSubmitted: true }); }
    };
    const map = GPUBuffer.prototype.mapAsync; let first = true;
    GPUBuffer.prototype.mapAsync = function(...args) {
      const result = map.apply(this, args);
      if (!first) return result;
      first = false; return result.then(() => new Promise(resolve => setTimeout(resolve, 150)));
    };
  `);
  const result = await workerScenario(page, true, 4, true, false, true);
  expect(result.changed).toBe(true);
  expect(result.computed.every(c => c.bufferedReadback && c.verifyStagedQueries && !c.verifyFallback)).toBe(true);
  expect(result.metrics).toEqual({ devices: 1, pipelines: 1, buffers: 0, maxScopes: 3 });
});

test("buffered readbacks drain failed maps and retry with exact CPU fallback", async ({ page }) => {
  await alterWorker(page, `
    self.addEventListener("message", event => {
      if (event.data.req?.kind === "c2c" && event.data.req.backend === "webgpu") {
        event.data.req.gpuBufferedReadback = true; event.data.req.gpuBatchSize = 512;
        event.data.req.gpuVerifyWorkers = 4;
      }
    });
    const map = GPUBuffer.prototype.mapAsync; let failed = false;
    GPUBuffer.prototype.mapAsync = function(...args) {
      if (!failed) { failed = true; return Promise.reject(new Error("injected buffered map failure")); }
      return map.apply(this, args);
    };
  `);
  await open(page);
  const cpu = await runAndExport(page, "cpu");
  expect(await runAndExport(page, "webgpu", "CPU fallback: injected buffered map failure")).toEqual(cpu);
  expect(await runAndExport(page, "webgpu", "with WebGPU")).toEqual(cpu);
});

test("buffered slot allocation failure releases partial buffers before retry", async ({ page }) => {
  const source = stripTypeScriptTypes(readFileSync(new URL("../src/gpu-candidates.ts", import.meta.url), "utf8"), { mode: "transform" });
  const moduleUrl = "data:text/javascript;base64," + Buffer.from(source).toString("base64");
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  const counts = await page.evaluate(async moduleUrl => {
    const { GpuCandidateSearch } = await import(moduleUrl);
    const create = GPUDevice.prototype.createBuffer, destroy = GPUBuffer.prototype.destroy;
    let calls = 0, live = 0;
    const destroyed = new WeakSet<GPUBuffer>();
    GPUDevice.prototype.createBuffer = function(desc) {
      if (++calls === 4) throw new Error("injected slot allocation failure");
      const buffer = create.call(this, desc); live++; return buffer;
    };
    GPUBuffer.prototype.destroy = function() {
      if (!destroyed.has(this)) { destroyed.add(this); live--; }
      destroy.call(this);
    };
    const search = await GpuCandidateSearch.create(new Float32Array([0, 0, 0, 0]), new Uint32Array([0, 1, 0, 255, 0]));
    try {
      const q = new Float32Array([0, 0, 0, 0]);
      await search.find(q, false, 64, true).then(() => { throw new Error("Expected allocation failure"); }, error => {
        if (!error.message.includes("slot allocation failure")) throw error;
      });
      const counts = [live];
      for (let i = 0; i < 3; i++) {
        const result = await search.find(q, false, 64, true);
        if (result[0] !== 0) throw new Error("Allocation retry changed candidate");
        counts.push(live);
      }
      search.dispose(); counts.push(live);
      return counts;
    } finally {
      search.dispose(); GPUDevice.prototype.createBuffer = create; GPUBuffer.prototype.destroy = destroy;
    }
  }, moduleUrl);
  expect(counts).toEqual([2, 5, 8, 8, 0]);
});

test("buffered readbacks recover from device loss with outstanding submissions", async ({ page }) => {
  await alterWorker(page, `
    self.addEventListener("message", event => {
      if (event.data.req?.kind === "c2c" && event.data.req.backend === "webgpu") {
        event.data.req.gpuBufferedReadback = true; event.data.req.gpuBatchSize = 512;
      }
    });
    const request = GPUAdapter.prototype.requestDevice; let first = true;
    GPUAdapter.prototype.requestDevice = async function(...args) {
      const device = await request.apply(this, args);
      if (first) {
        first = false; const submit = device.queue.submit.bind(device.queue); let count = 0;
        device.queue.submit = (...args) => { submit(...args); if (++count === 2) device.destroy(); };
      }
      return device;
    };
  `);
  await open(page);
  const cpu = await runAndExport(page, "cpu");
  expect(await runAndExport(page, "webgpu", "CPU fallback:")).toEqual(cpu);
  expect(await runAndExport(page, "webgpu", "with WebGPU")).toEqual(cpu);
});

for (const scopeFailure of [false, true]) {
  test(`two readback slots submit ahead and keep leases through mapping (scope failure: ${scopeFailure})`, async ({ page }) => {
    const source = stripTypeScriptTypes(readFileSync(new URL("../src/gpu-candidates.ts", import.meta.url), "utf8"), { mode: "transform" });
    const moduleUrl = "data:text/javascript;base64," + Buffer.from(source).toString("base64");
    await page.goto("/");
    test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
    const result = await page.evaluate(async ({ moduleUrl, scopeFailure }) => {
      const { GpuCandidateSearch } = await import(moduleUrl);
      const points = new Float32Array([0, 0, 0, 0, 10, 0, 0, 0]);
      const search = await GpuCandidateSearch.create(points, new Uint32Array([0, 2, 0, 255, 0]));
      const originalMap = GPUBuffer.prototype.mapAsync, originalPop = GPUDevice.prototype.popErrorScope;
      let release!: () => void, mapped = 0, scopes = 0;
      const gate = new Promise<void>(resolve => { release = resolve; });
      let second!: () => void;
      const secondMap = new Promise<void>(resolve => { second = resolve; });
      GPUBuffer.prototype.mapAsync = function(...args: Parameters<GPUBuffer["mapAsync"]>) {
        const n = ++mapped, result = originalMap.apply(this, args);
        if (n === 2) second();
        return n === 1 ? result.then(() => gate) : result;
      };
      GPUDevice.prototype.popErrorScope = function() {
        const result = originalPop.call(this);
        return scopeFailure && ++scopes === 1 ? result.then(() => new GPUValidationError("injected submission scope failure")) : result;
      };
      try {
        const first = search.find(new Float32Array([0, 0, 0, 0]), false, 64, true);
        // Attach rejection handlers immediately while the first mapping is gated.
        const settled = Promise.allSettled([first,
          search.find(new Float32Array([10, 0, 0, 0]), false, 64, true),
          search.find(Float32Array.from(Array.from({ length: 67 }, () => [10, 0, 0, 0]).flat()), false, 64, true)]);
        await Promise.race([secondMap, new Promise((_, reject) => setTimeout(() => reject(new Error("second submission waited for readback")), 5000))]);
        await new Promise(resolve => setTimeout(resolve, 30));
        const beforeRelease = mapped;
        release();
        const values = await settled;
        const retry = await search.find(new Float32Array([0, 0, 0, 0]), false, 64, true);
        return { beforeRelease, statuses: values.map(v => v.status),
          values: values.map(v => v.status === "fulfilled" ? [...v.value] : v.reason.message), retry: [...retry] };
      } finally {
        release(); GPUBuffer.prototype.mapAsync = originalMap;
        GPUDevice.prototype.popErrorScope = originalPop; search.dispose();
      }
    }, { moduleUrl, scopeFailure });
    expect(result.beforeRelease).toBe(2);
    expect(result.statuses).toEqual([scopeFailure ? "rejected" : "fulfilled", "fulfilled", "fulfilled"]);
    expect(result.values).toEqual([scopeFailure ? "WebGPU: injected submission scope failure" : [0], [1], Array(67).fill(1)]);
    expect(result.retry).toEqual([0]);
  });
}

test("resident GPU verification preserves controls, lane growth and exact certificates", async ({ page }) => {
  const { verifications, setups, released } = await batchScenario(page, 65_537, undefined, false, false, true, undefined, false, false, false, true);
  expect(verifications.map(c => c.verifyPoolWorkers)).toEqual([3, 1, 3, 0, 7, 3]);
  expect(verifications.map(c => c.verifyMainWorker)).toEqual([true, true, true, false, true, true]);
  expect(verifications.map(c => c.verifyReferenceReused)).toEqual([false, true, true, false, false, true]);
  expect(new Set([...verifications, ...setups].map(c => c.certified)).size).toBe(1);
  expect(released.buffers).toBe(0);
});

test("GPU verification default retains pool-only lanes after mixed resident timings", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, gpuMetrics + `
    self.addEventListener("message", ({ data }) => {
      if (data.req?.kind === "c2c") delete data.req.gpuMainWorker;
    });`);
  const { computed } = await workerScenario(page, false, 4);
  expect(computed.every(c => !c.verifyMainWorker && c.verifyPoolWorkers === 4 && !c.verifyFallback)).toBe(true);
});

for (const count of [19, 65_537]) test(`resident GPU staged queries exclude the local lane and preserve tails (${count})`, async ({ page }) => {
  const { queryModes, released } = await batchScenario(page, count, undefined, false, false, false, undefined, false, true, false, true);
  for (const compute of queryModes.filter(c => c.verifyStagedQueries)) {
    const workers = compute.verifyWorkers!, batch = compute.batchSize;
    let poolQueries = 0;
    for (let first = 0; first < count; first += batch) {
      const length = Math.min(batch, count - first), lanes = Math.min(workers, length);
      poolQueries += Math.floor((lanes - 1) * length / lanes);
    }
    expect(compute.verifyQueryBytes).toBe(poolQueries * 24);
    expect(compute.verifyMainWorker).toBe(true);
    expect(compute.verifyPoolWorkers).toBe(workers - 1);
  }
  expect(new Set(queryModes.map(c => c.certified)).size).toBe(1);
  expect(released.buffers).toBe(0);
});

for (const failure of ["prepare", "batch", "queries", "staged-batch"] as const)
test(`resident GPU ${failure} failure drains and recomputes certificates exactly`, async ({ page }) => {
  const staged = failure === "queries" || failure === "staged-batch";
  const result = await batchScenario(page, 131_073, undefined, false, false, !staged, failure, false, staged, false, true);
  const outputs = staged ? result.queryModes : result.verifications;
  expect(outputs[0].verifyWorkers).toBe(1);
  expect(outputs[0].verifyMainWorker).toBe(false);
  expect(outputs[0].verifyPoolWorkers).toBe(0);
  expect(outputs[0].verifyFallback).toContain("injected verification worker failure");
  expect(outputs.slice(1).every(c => !c.verifyFallback)).toBe(true);
  expect(new Set(outputs.map(c => c.certified)).size).toBe(1);
});

test("resident GPU snapshots survive concurrent source edits and removal", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, gpuMetrics + `
    const submit = GPUQueue.prototype.submit;
    GPUQueue.prototype.submit = function(...args) { submit.apply(this, args); self.postMessage({ gpuSubmitted: true }); };
    const map = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = async function(...args) {
      await map.apply(this, args); await new Promise(resolve => setTimeout(resolve, 150));
    };`);
  const { computed, changed } = await workerScenario(page, true, 4, true, true);
  expect(computed.every(c => c.verifyMainWorker && c.verifyPoolWorkers === 3 && c.verifyQueryBytes === 3072 * 24 && !c.verifyFallback)).toBe(true);
  expect(changed).toBe(true);
});

test("resident GPU parallel UI exports exactly the CPU PLY", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, `
    self.addEventListener("message", ({ data }) => {
      if (data.req?.kind === "c2c") data.req.gpuMainWorker = true;
    });`);
  await open(page);
  const cpu = await runAndExport(page, "cpu");
  expect(await runAndExport(page, "webgpu-parallel", "with WebGPU")).toEqual(cpu);
});

test("resident GPU main failure drains pool lanes and lost generations never spawn replacements", async ({ page }) => {
  const source = readFileSync(new URL("../src/gpu-verifiers.ts", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");
  const transformed = stripTypeScriptTypes(source, { mode: "transform" });
  await page.goto("/");
  const outputs = await page.evaluate(async source => {
    const results = [];
    for (const failure of ["main", "pool", "lost"]) {
      const pending: { resolve: (value: unknown) => void; reject: (error: Error) => void }[] = [];
      let generation: number | undefined = 1, spawned = 0, settled = false;
      const runOn = () => { spawned++; return Promise.resolve(); };
      const runOnGeneration = (_lane: number, expected: number | undefined) => {
        if (expected === undefined || expected !== generation) return Promise.reject(new Error("generation lost"));
        return new Promise((resolve, reject) => pending.push({ resolve, reject }));
      };
      Object.assign(globalThis, { mockRunOn: runOn, mockRunOnGeneration: runOnGeneration, mockGeneration: () => generation });
      const module = await import("data:text/javascript;base64," + btoa(`
        const runOn = globalThis.mockRunOn, runOnGeneration = globalThis.mockRunOnGeneration;
        const workerGeneration = () => globalThis.mockGeneration(), releaseOn = () => Promise.resolve(true);
        ${source}\n// ${failure}`));
      const verifier = new module.GpuVerifiers();
      await verifier.prepare({ snapshot: () => new Uint8Array([1]) }, 3);
      spawned = 0;
      if (failure === "lost") generation = undefined;
      let freed = 0;
      const distances = new Float64Array(8).fill(-1);
      const job = {
        queriesF64: (_first: number, count: number) => new Float64Array(count * 3),
        verify: () => {
          if (failure === "main") throw new Error("main lane failed");
          return { takeDistances: () => new Float64Array(2), certified: 2, free: () => { freed++; } };
        },
      };
      const completion = verifier.complete(job, 0, new Uint32Array(8), 4, true, true, distances, undefined, true)
        .then(() => { settled = true; return "success"; }, (error: Error) => { settled = true; return error.message; });
      if (pending.length !== (failure === "lost" ? 0 : 3)) throw new Error("Unexpected pool submissions");
      if (failure !== "lost") {
        if (failure === "pool") pending[0].reject(new Error("pool lane failed"));
        else pending[0].resolve({ distances: new Float64Array(2), certified: 2 });
        await Promise.resolve(); await Promise.resolve();
        if (settled) throw new Error("Batch ended before all pool lanes drained");
        pending[1].resolve({ distances: new Float64Array(2), certified: 2 });
        pending[2].resolve({ distances: new Float64Array(2), certified: 2 });
      }
      results.push({ failure, error: await completion, spawned, freed, unchanged: distances.every(v => v === -1) });
    }
    return results;
  }, transformed);
  expect(outputs).toEqual([
    { failure: "main", error: "main lane failed", spawned: 0, freed: 0, unchanged: true },
    { failure: "pool", error: "pool lane failed", spawned: 0, freed: 1, unchanged: true },
    { failure: "lost", error: "generation lost", spawned: 0, freed: 1, unchanged: true },
  ]);
});

for (const [count, limit] of [[19, undefined], [262_145, undefined], [65_537, "dispatch"]] as const)
test(`staged exact queries preserve partial batches and worker switches (${count}, ${limit ?? "uncapped"})`, async ({ page }) => {
  const { queryModes, released } = await batchScenario(page, count, limit, false, false, false, undefined, false, true);
  expect(queryModes.slice(0, 10).map(c => c.verifyStagedQueries)).toEqual([true, false, true, false, true, false, true, false, true, false]);
  expect(new Set(queryModes.slice(0, 10).map(c => c.certified)).size).toBe(1);
  expect(queryModes.filter(c => c.verifyStagedQueries).every(c => c.verifyQueryBytes === count * 24)).toBe(true);
  if (limit) expect(queryModes.every(c => c.batchSize === 16_384)).toBe(true);
  expect(released.buffers).toBe(0);
});

for (const failure of ["queries", "staged-batch"] as const) test(`staged query ${failure} failure recomputes exact output and retries`, async ({ page }) => {
  const { queryModes } = await batchScenario(page, 65_537, undefined, false, false, false, failure, false, true);
  expect(queryModes[0].verifyWorkers).toBe(1);
  expect(queryModes[0].verifyFallback).toContain("injected verification worker failure");
  expect(queryModes[0].verifyStagedQueries).toBe(false);
  expect(queryModes.slice(1).every(c => !c.verifyFallback)).toBe(true);
  expect(new Set(queryModes.map(c => c.certified)).size).toBe(1);
});

test("staged queries retain separate snapshots through overlapping jobs and source removal", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, gpuMetrics + `
    const submit = GPUQueue.prototype.submit;
    GPUQueue.prototype.submit = function(...args) { submit.apply(this, args); self.postMessage({ gpuSubmitted: true }); };
    const map = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = async function(...args) {
      await map.apply(this, args); await new Promise(resolve => setTimeout(resolve, 150));
    };`);
  const { computed, changed } = await workerScenario(page, true, 4, true);
  expect(computed.every(c => c.verifyStagedQueries && c.verifyQueryBytes === 4096 * 24 && !c.verifyFallback)).toBe(true);
  expect(changed).toBe(true);
});

test("a worker lost during query release invalidates the cached reference before retry", async ({ page }) => {
  const { queryModes } = await batchScenario(page, 65_537, undefined, false, false, false, "release", false, true);
  expect(queryModes.every(c => c.verifyWorkers >= 2 && !c.verifyFallback)).toBe(true);
  expect(queryModes[1].verifyReferenceReused).toBe(false);
  expect(new Set(queryModes.map(c => c.certified)).size).toBe(1);
});

for (const [count, limitedBy] of [[524_287, undefined], [524_289, undefined], [65_537, "dispatch"]] as const) {
  test(`parallel batch defaults preserve exact output and cached workers (${count}, ${limitedBy ?? "uncapped"})`, async ({ page }) => {
    const { parallelBatches, defaults, released } = await batchScenario(page, count, limitedBy, false, false, false, undefined, true);
    expect(parallelBatches.map(c => c.batchSize)).toEqual(limitedBy ? [16_384, 16_384, 16_384, 16_384] : [65_536, 131_072, 262_144, 65_536]);
    expect(parallelBatches.map(c => c.batches)).toEqual(limitedBy ? [5, 5, 5, 5] : count === 524_287 ? [8, 4, 2, 8] : [9, 5, 3, 9]);
    expect(parallelBatches.map(c => c.verifyReferenceReused)).toEqual([false, true, true, true]);
    expect(defaults.map(c => c.batchSize)).toEqual(limitedBy ? [16_384, 16_384, 16_384, 16_384] :
      count === 524_287 ? [65_536, 65_536, 65_536, 65_536] : [65_536, 65_536, 131_072, 131_072]);
    expect(new Set([...parallelBatches, ...defaults].map(c => c.certified)).size).toBe(1);
    expect([...parallelBatches, ...defaults].every(c => Number.isFinite(c.gpuWaitMillis) && c.gpuWaitMillis! >= 0 && c.gpuWaitMillis! <= c.gpuMillis + 1)).toBe(true);
    expect(released.buffers).toBe(0);
  });
}

test("parallel exact verification preserves partial batches, controls and cached lane growth", async ({ page }) => {
  const { verifications, setups, released } = await batchScenario(page, 65_537, undefined, false, false, true);
  expect(verifications.map(c => c.verifyWorkers)).toEqual([4, 2, 4, 1, 8, 4]);
  expect(verifications.map(c => c.verifyReferenceReused)).toEqual([false, true, true, false, false, true]);
  expect(new Set(verifications.map(c => c.certified)).size).toBe(1);
  expect(verifications.every(c => c.verifySnapshot && c.verifyReuseImportBounds)).toBe(true);
  expect(setups.map(c => c.verifySnapshot)).toEqual([false, true, true, true, true, false]);
  expect(setups.map(c => c.verifyReuseImportBounds)).toEqual([false, false, true, false, true, false]);
  expect(setups.map(c => c.verifyReferenceReused)).toEqual([false, false, true, true, true, true]);
  expect(new Set([...verifications, ...setups].map(c => c.certified)).size).toBe(1);
  expect(released.buffers).toBe(0);
});

test("idle verification worker loss reimports exact references without serial fallback", async ({ page }) => {
  const { verifications, released } = await batchScenario(page, 65_537, undefined, false, false,
    true, undefined, false, false, true);
  expect(verifications.map(c => c.verifyWorkers)).toEqual([4, 2, 4, 1, 8, 4]);
  expect(verifications.map(c => c.verifyReferenceReused)).toEqual([false, false, true, false, false, true]);
  expect(verifications.every(c => !c.verifyFallback)).toBe(true);
  expect(new Set(verifications.map(c => c.certified)).size).toBe(1);
  expect(released.buffers).toBe(0);
});

for (const failure of ["prepare", "batch"] as const) {
  test(`verification worker ${failure} failure recomputes exact output and retries`, async ({ page }) => {
    const { verifications } = await batchScenario(page, 65_537, undefined, false, false, true, failure);
    expect(verifications[0].verifyWorkers).toBe(1);
    expect(verifications[0].verifyFallback).toContain("injected verification worker failure");
    expect(verifications.slice(1).map(c => c.verifyWorkers)).toEqual([2, 4, 1, 8, 4]);
    expect(verifications.slice(1).every(c => !c.verifyFallback)).toBe(true);
    expect(new Set(verifications.map(c => c.certified)).size).toBe(1);
  });
}

test("invalid exact tree snapshots use complete verification and retry cleanly", async ({ page }) => {
  const { verifications } = await batchScenario(page, 65_537, undefined, false, false, true, "snapshot");
  expect(verifications[0].verifyWorkers).toBe(1);
  expect(verifications[0].verifyFallback).toContain("invalid exact k-d tree snapshot");
  expect(verifications.slice(1).map(c => c.verifyWorkers)).toEqual([2, 4, 1, 8, 4]);
  expect(verifications.slice(1).every(c => !c.verifyFallback)).toBe(true);
  expect(new Set(verifications.map(c => c.certified)).size).toBe(1);
});

test("parallel verifier snapshots survive overlapping jobs, reference edits and removal", async ({ page }) => {
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await alterWorker(page, gpuMetrics + `
    const submit = GPUQueue.prototype.submit;
    GPUQueue.prototype.submit = function(...args) { submit.apply(this, args); self.postMessage({ gpuSubmitted: true }); };
    const map = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = async function(...args) {
      await map.apply(this, args); await new Promise(resolve => setTimeout(resolve, 150));
    };`);
  const { computed, changed } = await workerScenario(page, true, 4);
  expect(computed.map(c => c.verifyWorkers)).toEqual([4, 4, 4]);
  expect(computed.map(c => c.verifyReferenceReused)).toEqual([false, true, false]);
  expect(computed.every(c => !c.verifyFallback)).toBe(true);
  expect(changed).toBe(true);
});

test("WebGPU plus CPU workers can be selected and exports identical distances", async ({ page }) => {
  await open(page, 65_537);
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  const cpu = await runAndExport(page, "cpu");
  expect(await runAndExport(page, "webgpu-parallel", "with WebGPU + 4 CPU workers")).toEqual(cpu);
});

for (const concurrency of [1, 4]) test(`GPU verification respects reported CPU capacity (${concurrency})`, async ({ page }) => {
  await alterWorker(page, `Object.defineProperty(navigator, "hardwareConcurrency", { value: ${concurrency} });` + gpuMetrics);
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  const { computed } = await workerScenario(page, false, 8);
  const workers = Math.max(1, concurrency - 1);
  expect(computed.every(c => c.verifyWorkers === workers && !c.verifyFallback)).toBe(true);
  expect(computed.every(c => !c.verifyMainWorker && c.verifyPoolWorkers === (workers > 1 ? workers : 0))).toBe(true);
});

test("batch caps preserve exact results through growth, partial batches and invalid controls", async ({ page }) => {
  const { computed, active, released, recovered } = await batchScenario(page, 262_145);
  expect(computed.map(c => c.batchSize)).toEqual([65_536, 131_072, 262_144, 65_536]);
  expect(computed.map(c => c.batches)).toEqual([5, 3, 2, 5]);
  expect(computed.map(c => c.referenceReused)).toEqual([false, true, true, true]);
  expect(new Set(computed.map(c => c.certified)).size).toBe(1);
  expect(active).toEqual({ devices: 1, pipelines: 1, buffers: 5, maxScopes: 3 });
  expect(released.buffers).toBe(0);
  expect(recovered?.backend).toBe("webgpu");
  expect(recovered?.referenceReused).toBe(true);
});

test("batch caps respect each device dispatch, storage and buffer limit", async ({ page }) => {
  for (const limit of ["dispatch", "storage", "buffer"]) {
    const { computed, active, released } = await batchScenario(page, 65_537, limit);
    expect(computed.map(c => c.batchSize)).toEqual([16_384, 16_384, 16_384, 16_384]);
    expect(computed.map(c => c.batches)).toEqual([5, 5, 5, 5]);
    expect(active).toEqual({ devices: 1, pipelines: 1, buffers: 5, maxScopes: 3 });
    expect(released.buffers).toBe(0);
  }
});

test("GPU hint modes preserve full C2C results and reuse buffers across pipeline switches", async ({ page }) => {
  const { hints, active, released } = await batchScenario(page, 65_537, undefined, true);
  expect(hints.map(h => h.leafOnly)).toEqual([false, true, false, true]);
  expect(hints.every(h => h.referenceReused && h.deviceReused && h.batches === 2)).toBe(true);
  expect(active).toEqual({ devices: 1, pipelines: 2, buffers: 5, maxScopes: 3 });
  expect(released.buffers).toBe(0);
});

test("workgroup switches preserve exact C2C results, pipeline caching and partial groups", async ({ page }) => {
  const { workgroups, active, released, recovered } = await batchScenario(page, 65_537, undefined, false, true);
  expect(workgroups.map(c => c.workgroupSize)).toEqual([64, 32, 128, 256, 64, 64, 32, 128, 256, 64]);
  expect(workgroups.every(c => c.referenceReused && c.deviceReused && c.batches === 2 && c.batchSize === 65_536)).toBe(true);
  expect(active).toEqual({ devices: 1, pipelines: 8, buffers: 5, maxScopes: 3 });
  expect(released.buffers).toBe(0);
  expect(recovered?.workgroupSize).toBe(64);
});

test("workgroup caps respect dispatch and invocation limits with complete CPU recovery", async ({ page }) => {
  const dispatched = await batchScenario(page, 65_537, "dispatch", false, true);
  expect(dispatched.workgroups.map(c => c.batchSize)).toEqual([16_384, 8192, 32_768, 65_536, 16_384, 16_384, 8192, 32_768, 65_536, 16_384]);
  expect(dispatched.workgroups.map(c => c.batches)).toEqual([5, 9, 3, 2, 5, 5, 9, 3, 2, 5]);
  expect(dispatched.active).toEqual({ devices: 1, pipelines: 8, buffers: 5, maxScopes: 3 });
  expect(dispatched.released.buffers).toBe(0);
  for (const limit of ["invocations", "workgroup-x"]) {
    const { active, released, recovered } = await batchScenario(page, 65_537, limit, false, true);
    expect(active).toEqual({ devices: 1, pipelines: 6, buffers: 5, maxScopes: 3 });
    expect(released.buffers).toBe(0);
    expect(recovered?.backend).toBe("webgpu");
  }
});

test("near-leaf hints can differ from nearest candidates while shared query arrays remain intact", async ({ page }) => {
  const source = stripTypeScriptTypes(readFileSync(new URL("../src/gpu-candidates.ts", import.meta.url), "utf8"), { mode: "transform" });
  const moduleUrl = "data:text/javascript;base64," + Buffer.from(source).toString("base64");
  await page.route("**/__gpu-hints", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__gpu-hints");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  const result = await page.evaluate(async moduleUrl => {
    const { GpuCandidateSearch } = await import(moduleUrl);
    const points = Float32Array.from(Array.from({ length: 32 }, (_, i) => [i, i % 3, 0, 0]).flat());
    const split = new Uint32Array(new Float32Array([16]).buffer)[0];
    const nodes = new Uint32Array([0, 32, 2, 0, split, 0, 16, 0, 255, 0, 16, 32, 0, 255, 0]);
    const storage = new Float32Array([123, 456, 789, 0, 15.8, 1.1, 0, 0]);
    const bits = new Uint32Array(storage.buffer);
    bits[7] = 0x7fc00123; // Preserve an arbitrary NaN padding payload, too.
    const before = [...bits], queries = storage.subarray(4);
    const search = await GpuCandidateSearch.create(points, nodes);
    try {
      const values = await Promise.all([search.find(queries, false, 32), search.find(queries, true, 128), search.find(queries, false, 256)]);
      return { candidates: values.map(v => v[0]), preserved: bits.every((v, i) => v === before[i]) };
    } finally { search.dispose(); }
  }, moduleUrl);
  expect(result).toEqual({ candidates: [16, 15, 16], preserved: true });
});

test("a failed optional hint pipeline completes on CPU and can retry on the shared device", async ({ page }) => {
  await alterWorker(page, `
    self.addEventListener("message", event => {
      if (event.data.req?.kind === "c2c" && event.data.req.backend === "webgpu") event.data.req.gpuLeafOnly = true;
    });
    let failed = false;
    const create = GPUDevice.prototype.createComputePipelineAsync;
    GPUDevice.prototype.createComputePipelineAsync = function(desc) {
      if (desc.compute.entryPoint === "leafNearest" && !failed) {
        failed = true;
        return Promise.reject(new Error("synthetic hint pipeline failure"));
      }
      return create.call(this, desc);
    };
  `);
  await open(page);
  const cpu = await runAndExport(page, "cpu");
  expect(await runAndExport(page, "webgpu", "CPU fallback: synthetic hint pipeline failure")).toEqual(cpu);
  expect(await runAndExport(page, "webgpu", "with WebGPU")).toEqual(cpu);
});

test("a failed optional workgroup pipeline completes on CPU and retries on the shared device", async ({ page }) => {
  await alterWorker(page, `
    self.addEventListener("message", event => {
      if (event.data.req?.kind === "c2c" && event.data.req.backend === "webgpu") event.data.req.gpuWorkgroupSize = 128;
    });
    let failed = false;
    const create = GPUDevice.prototype.createComputePipelineAsync;
    GPUDevice.prototype.createComputePipelineAsync = function(desc) {
      if (desc.compute.constants?.workgroupSize === 128 && !failed) {
        failed = true;
        return Promise.reject(new Error("synthetic workgroup pipeline failure"));
      }
      return create.call(this, desc);
    };
  `);
  await open(page);
  const cpu = await runAndExport(page, "cpu");
  expect(await runAndExport(page, "webgpu", "CPU fallback: synthetic workgroup pipeline failure")).toEqual(cpu);
  expect(await runAndExport(page, "webgpu", "with WebGPU")).toEqual(cpu);
});

test("GPU candidates match brute force on both sides of a split", async ({ page }) => {
  // Run the actual standalone host module in the browser, without a test
  // endpoint in the production app. Node 22 transforms its TypeScript.
  const source = stripTypeScriptTypes(readFileSync(new URL("../src/gpu-candidates.ts", import.meta.url), "utf8"), { mode: "transform" });
  const moduleUrl = "data:text/javascript;base64," + Buffer.from(source).toString("base64");
  await page.goto("/");
  test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  const { runs, allocations } = await page.evaluate(async moduleUrl => {
    const { GpuCandidateSearch } = await import(moduleUrl);
    const points = Float32Array.from(Array.from({ length: 32 }, (_, i) => [i, i % 3, 0, 0]).flat());
    const splitBits = new Uint32Array(new Float32Array([16]).buffer)[0];
    const nodes = new Uint32Array([0, 32, 2, 0, splitBits, 0, 16, 0, 255, 0, 16, 32, 0, 255, 0]);
    const queries = Float32Array.from(Array.from({ length: 67 }, (_, i) => [i * 0.5 - 1, 1.1, 0, 0]).flat());
    let allocated = 0;
    const createBuffer = GPUDevice.prototype.createBuffer;
    GPUDevice.prototype.createBuffer = function(desc) { allocated++; return createBuffer.call(this, desc); };
    const search = await GpuCandidateSearch.create(points, nodes);
    try {
      const runs = [[...await search.find(queries)]];
      const allocations = [allocated];
      // A smaller batch and concurrent requests must use the retained buffers
      // with live binding ranges, while a larger batch grows them once.
      runs.push(...(await Promise.all([search.find(queries.slice(0, 12)), search.find(queries)])).map(a => [...a]));
      allocations.push(allocated);
      const doubled = new Float32Array(queries.length * 2);
      doubled.set(queries); doubled.set(queries, queries.length);
      runs.push([...await search.find(doubled)]);
      allocations.push(allocated);
      runs.push([...await search.find(queries.slice(0, 4))]);
      allocations.push(allocated);
      return { runs, allocations };
    } finally { search.dispose(); GPUDevice.prototype.createBuffer = createBuffer; }
  }, moduleUrl);
  const expected = Array.from({ length: 67 }, (_, i) => {
    const q = [Math.fround(i * 0.5 - 1), Math.fround(1.1)];
    let best = Infinity, slot = -1;
    for (let j = 0; j < 32; j++) {
      const d = (j - q[0]) ** 2 + (j % 3 - q[1]) ** 2;
      if (d < best) { best = d; slot = j; }
    }
    return slot;
  });
  expect(runs).toEqual([expected, expected.slice(0, 3), expected, [...expected, ...expected], expected.slice(0, 1)]);
  expect(allocations).toEqual([5, 5, 8, 8]);
});
