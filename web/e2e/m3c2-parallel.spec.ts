import { expect, type Page, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import type { LoadedCloud, M3c2Output, Request } from "../src/protocol";

async function setup(page: Page, prelude = ""): Promise<string> {
  // No viewer or extra worker pool competing with the workers under test.
  await page.route("**/__m3c2-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__m3c2-test");
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `Object.defineProperty(navigator, "hardwareConcurrency", { value: 5 });\n${prelude}\n` + await response.text() });
  });
  return "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n))!;
}

async function scenario(page: Page, workerPath: string, n: number, coreSpacing: number, mutate = false, compareLayouts = false, preparedTrees = false, defaultTrees = false) {
  return page.evaluate(async ({ workerPath, n, coreSpacing, mutate, compareLayouts, preparedTrees, defaultTrees }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    let announce: () => void;
    const started = new Promise<void>(resolve => { announce = resolve; });
    worker.onmessage = ({ data }) => {
      if (data.m3c2Started) { announce(); return; }
      if (!data.response) return;
      const pending = waiting.get(data.seq)!;
      waiting.delete(data.seq);
      data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const p of waiting.values()) p.reject(new Error(event.message)); waiting.clear(); };
    const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
      const id = ++seq;
      waiting.set(id, { resolve: value => resolve(value as T), reject });
      worker.postMessage({ seq: id, req });
    });
    const fixture = (lift: number) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float intensity\nproperty float m3c2_distance\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 35);
      bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      const width = Math.ceil(Math.sqrt(n));
      for (let i = 0; i < n; i++) {
        const x = i % width, y = Math.floor(i / width);
        const xyz = i === n - 1 ? [400_000, 4_000_000, 50 + lift]
          : [368_000 + x * 0.1, 3_955_000 + y * 0.1, 40 + Math.sin(x / 20) * 0.1 + lift];
        xyz.forEach((v, a) => view.setFloat64(i * 35 + a * 8, v, true));
        [i % 256, 30, 90].forEach((v, a) => view.setUint8(i * 35 + 24 + a, v));
        view.setFloat32(i * 35 + 27, i / 10, true);
        view.setFloat32(i * 35 + 31, 123, true);
      }
      return new File([bytes], lift ? "after.ply" : "before.ply");
    };
    const bits = (array: Float32Array) => new Uint32Array(array.buffer);
    try {
      const reference = await call<LoadedCloud>({ kind: "load", file: fixture(0), maxPoints: n });
      const compared = await call<LoadedCloud>({ kind: "load", file: fixture(0.07), maxPoints: n });
      const compute = (parallel: boolean, normalRadius = 0.4, interleaved = true, radiusBounds?: boolean) => call<M3c2Output>({ kind: "m3c2", reference: reference.id,
        compared: compared.id, normalRadius, projectionRadius: 0.25, maxDepth: 0.5, coreSpacing, parallel, interleaved, radiusBounds, sharedSearch: parallel && defaultTrees ? undefined : parallel,
        ...(defaultTrees ? {} : { preparedTrees }) });
      const serial = await compute(false, 0.4, true, false);
      const before = await call<Uint8Array>({ kind: "export", id: serial.cloud.id, format: "ply" });
      const pending = compute(true, 0.4, true, true);
      if (mutate) {
        await started;
        for (const id of [reference.id, compared.id]) {
          await call({ kind: "transform", id, matrix: [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 0.1, 0, 0, 0, 1] });
          await call({ kind: "remove", id });
        }
      }
      const parallel = await pending;
      if (serial.sharedSearch !== false || parallel.sharedSearch !== true || serial.radiusBounds !== false || parallel.radiusBounds !== true) throw new Error("Radius pruning controls were not applied");
      const after = await call<Uint8Array>({ kind: "export", id: parallel.cloud.id, format: "ply" });
      let errors = 0;
      for (const key of ["distance", "lod95", "significant"] as const) {
        const a = bits(serial[key]), b = bits(parallel[key]);
        if (a.length !== b.length) throw new Error("Missing core results");
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) errors++;
      }
      let bytesEqual = before.length === after.length;
      for (let i = 0; bytesEqual && i < before.length; i++) bytesEqual = before[i] === after[i];
      let layoutsEqual = true;
      let cacheReused = false;
      if (preparedTrees && !mutate && parallel.workers > 1) {
        const warm = await compute(true, 0.4, true, false);
        const bytes = await call<Uint8Array>({ kind: "export", id: warm.cloud.id, format: "ply" });
        cacheReused = warm.treeCacheHit && warm.preparedTrees && warm.workers === parallel.workers &&
          bytes.length === after.length && bytes.every((v, i) => v === after[i]);
        const changed = await compute(true, 0.5, true, true);
        const control = await compute(false, 0.5, true, false);
        cacheReused &&= changed.treeCacheHit && ["distance", "lod95", "significant"].every(key => {
          const a = bits(changed[key as "distance"]), b = bits(control[key as "distance"]);
          return a.length === b.length && a.every((v, i) => v === b[i]);
        });
      }
      if (compareLayouts) {
        const contiguous = await compute(true, 0.4, false, false);
        const controlBytes = await call<Uint8Array>({ kind: "export", id: contiguous.cloud.id, format: "ply" });
        layoutsEqual = contiguous.workers === parallel.workers && controlBytes.length === after.length &&
          controlBytes.every((v, i) => v === after[i]);
      }
      let recovered = false;
      if (!mutate) {
        try { await compute(true, 0); } catch (error) { recovered = String(error).includes("positive radii"); }
        const again = await compute(true);
        recovered &&= again.cloud.count === parallel.cloud.count;
      }
      return { errors, bytesEqual, layoutsEqual, serialWorkers: serial.workers, workers: parallel.workers,
        count: parallel.cloud.count, undefined: parallel.distance.filter(Number.isNaN).length,
        measured: parallel.distance.filter(Number.isFinite).length, name: parallel.cloud.name, recovered,
        mainWorker: parallel.mainWorker, preparedTrees: parallel.preparedTrees, treeCacheHit: parallel.treeCacheHit, cacheReused,
        normalCacheHit: parallel.normalCacheHit, normalCacheBytes: parallel.normalCacheBytes };
    } finally { worker.terminate(); }
  }, { workerPath, n, coreSpacing, mutate, compareLayouts, preparedTrees, defaultTrees });
}

for (const spacing of [0, 0.13]) {
  test(`parallel M3C2 preserves boundary results, attributes and undefined cores (spacing ${spacing})`, async ({ page }) => {
    const result = await scenario(page, await setup(page), 40_013, spacing, false, true);
    expect(result.errors).toBe(0);
    expect(result.bytesEqual).toBe(true);
    expect(result.layoutsEqual).toBe(true);
    expect(result.serialWorkers).toBe(1);
    expect(result.workers).toBeGreaterThan(1);
    expect(result.workers).toBeLessThanOrEqual(4);
    if (spacing) expect(result.count).toBeLessThan(40_013);
    else expect(result.count).toBe(40_013);
    expect(result.undefined).toBeGreaterThan(0);
    expect(result.measured).toBeGreaterThan(0);
    expect(result.recovered).toBe(true);
  });
}

test("prepared exact trees preserve cyclic/contiguous exports with four and two lanes", async ({ page }) => {
  test.setTimeout(120_000);
  const path = await setup(page);
  for (const n of [40_013, 880_013]) {
    const result = await scenario(page, path, n, n > 100_000 ? 0.4 : 0, false, true, true, true);
    expect(result.workers).toBe(n > 100_000 ? 2 : 4);
    expect(result.preparedTrees).toBe(true);
    expect(result.treeCacheHit).toBe(false);
    expect(result.normalCacheHit).toBe(false);
    expect(result.normalCacheBytes).toBe(0);
    expect(result.cacheReused).toBe(true);
    expect(result.mainWorker).toBe(true);
    expect(result.errors).toBe(0);
    expect(result.bytesEqual).toBe(true);
    expect(result.layoutsEqual).toBe(true);
    expect(result.recovered).toBe(true);
  }
});

test("prepared M3C2 leases keep geometry and core attributes after source removal", async ({ page }) => {
  const path = await setup(page, `
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(request, options) {
      if (request.kind === "m3c2-prepared") {
        self.postMessage({ m3c2Started: true });
        setTimeout(() => post.call(this, request, options), 200);
      } else post.call(this, request, options);
    };
  `);
  const result = await scenario(page, path, 40_013, 0.13, true, false, true);
  expect(result.preparedTrees).toBe(true);
  expect(result.errors).toBe(0);
  expect(result.bytesEqual).toBe(true);
  expect(result.name).toBe("after_m3c2");
});

test("small and heavily spaced jobs keep the serial path", async ({ page }) => {
  const path = await setup(page);
  for (const [n, spacing] of [[2017, 0], [20_017, 1]]) {
    const result = await scenario(page, path, n, spacing);
    expect(result.workers).toBe(1);
    expect(result.errors).toBe(0);
    expect(result.bytesEqual).toBe(true);
  }
});

test("cyclic core blocks preserve two- and three-worker threshold jobs", async ({ page }) => {
  const path = await setup(page);
  for (const n of [20_001, 30_001]) {
    const result = await scenario(page, path, n, 0, false, true);
    expect(result.workers).toBe(Math.floor(n / 10_000));
    expect(result.count).toBe(n);
    expect(result.errors).toBe(0);
    expect(result.bytesEqual).toBe(true);
    expect(result.layoutsEqual).toBe(true);
    expect(result.recovered).toBe(true);
  }
});

test("parallel M3C2 owns snapshots when both source clouds change and disappear", async ({ page }) => {
  const path = await setup(page, `
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(request, options) {
      if (request.kind === "m3c2") {
        self.postMessage({ m3c2Started: true });
        setTimeout(() => post.call(this, request, options), 200);
      } else post.call(this, request, options);
    };
  `);
  const result = await scenario(page, path, 40_013, 0, true);
  expect(result.workers).toBe(4);
  expect(result.errors).toBe(0);
  expect(result.bytesEqual).toBe(true);
  expect(result.name).toBe("after_m3c2");
});

test("M3C2 worker count is bounded by input copying and CPU availability", async ({ page }) => {
  // Test policy with real plan/dispatch/result assembly, without allocating
  // millions of points just to exercise the conservative copy estimate.
  const source = readFileSync("src/m3c2-workers.ts", "utf8");
  const { stripTypeScriptTypes } = await import("node:module");
  const instrumented = source.replace(/import .*?;\r?\n/g, "")
    .replace("export function parallelM3c2", "function parallelM3c2");
  await page.route("**/__m3c2-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__m3c2-test");
  const result = await page.evaluate(async code => {
    const run = new Function("M3c2Plan", "poolSize", "runOn", "m3c2CloudDistances",
      "const planM3c2Cores = (cloud, spacing) => ({ plan: new M3c2Plan(cloud, spacing), hit: false, bytes: 0 });\n" + code + "\nreturn parallelM3c2;");
    class Plan {
      length = 40_013;
      points(first: number, count: number) { return new Float64Array([first, count]); }
      interleavedPoints(part: number, parts: number, _block: number) {
        return new Float64Array([part, Math.floor(this.length / parts) + (part < this.length % parts ? 1 : 0)]);
      }
      finish(values: Float64Array) { return values; }
      free() {}
    }
    const slice = async (_lane: number, request: { core: Float64Array }) => new Float64Array(request.core[1] * 2);
    const outcomes = [];
    for (const [points, lanes] of [[2_000_000, 8], [1_000_000, 8], [1, 1], [1, 8], [1_100_000, 8], [1_100_000, 1]]) {
      const parallel = run(Plan, () => lanes, slice, (_compared: unknown, _reference: unknown, core: Float64Array) => new Float64Array(core[1] * 2));
      const input = { length: 40_013, rawPositions: () => new Float64Array(0) };
      const ref = { length: points, rawPositions: input.rawPositions };
      const job = parallel(input, ref, 0.4, 0.25, 0.5, 0, true, false, true, false);
      outcomes.push(job ? (await job).workers : 1);
    }
    return outcomes;
  }, stripTypeScriptTypes(instrumented));
  expect(result).toEqual([1, 2, 1, 4, 2, 1]);
});

test("resident analysis lane admits large inputs within the copying budget with exact exports", async ({ page }) => {
  const result = await scenario(page, await setup(page), 540_013, 0.35, false, true);
  expect(result.workers).toBe(2);
  expect(result.mainWorker).toBe(true);
  expect(result.errors).toBe(0);
  expect(result.bytesEqual).toBe(true);
  expect(result.layoutsEqual).toBe(true);
  expect(result.recovered).toBe(true);
});

test("resident third lane preserves cyclic tails and contiguous exports against serial", async ({ page }) => {
  const result = await scenario(page, await setup(page), 360_013, 0.25, false, true);
  expect(result.workers).toBe(3);
  expect(result.mainWorker).toBe(true);
  expect(result.errors).toBe(0);
  expect(result.bytesEqual).toBe(true);
  expect(result.layoutsEqual).toBe(true);
  expect(result.recovered).toBe(true);
});

test("resident analysis lane keeps core snapshots through source edits and removal", async ({ page }) => {
  const path = await setup(page, `
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(request, options) {
      if (request.kind === "m3c2") {
        self.postMessage({ m3c2Started: true });
        setTimeout(() => post.call(this, request, options), 200);
      } else post.call(this, request, options);
    };
  `);
  const result = await scenario(page, path, 540_013, 0.35, true);
  expect(result.workers).toBe(2);
  expect(result.mainWorker).toBe(true);
  expect(result.errors).toBe(0);
  expect(result.bytesEqual).toBe(true);
  expect(result.name).toBe("after_m3c2");
});

test("resident lane and pool failures drain every dispatched lane before freeing the plan", async ({ page }) => {
  const { stripTypeScriptTypes } = await import("node:module");
  const source = readFileSync("src/m3c2-workers.ts", "utf8").replace(/import .*?;\r?\n/g, "")
    .replace("export function parallelM3c2", "function parallelM3c2");
  await page.route("**/__m3c2-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__m3c2-test");
  const outcomes = await page.evaluate(async code => {
    const construct = new Function("M3c2Plan", "poolSize", "runOn", "m3c2CloudDistances",
      "const planM3c2Cores = (cloud, spacing) => ({ plan: new M3c2Plan(cloud, spacing), hit: false, bytes: 0 });\n" + code + "\nreturn parallelM3c2;");
    const outcomes = [];
    for (const failure of ["main", "pool"]) {
      let freed = 0, finished = 0, mainCalls = 0, settled = false;
      class Plan {
        length = 40_013;
        interleavedPoints(part: number, parts: number) { return new Float64Array([part, Math.floor(this.length / parts)]); }
        finish() { finished++; throw new Error("Unexpected partial result"); }
        free() { freed++; }
      }
      const pending: { resolve: (values: Float64Array) => void; reject: (error: Error) => void }[] = [];
      const runOn = () => new Promise<Float64Array>((resolve, reject) => pending.push({ resolve, reject }));
      const main = () => { mainCalls++; if (failure === "main") throw new Error("main lane failed"); return new Float64Array(0); };
      const compute = construct(Plan, () => 4, runOn, main);
      const cloud = { length: 40_013, rawPositions: () => new Float64Array(0) };
      const reference = { length: 700_000, rawPositions: cloud.rawPositions };
      const job = compute(cloud, reference, 0.4, 0.25, 0.5, 0, true, false, true, false).then(() => { settled = true; return "success"; }, (error: Error) => { settled = true; return error.message; });
      if (pending.length !== 2 || mainCalls !== 1 || freed || settled) throw new Error("Expected one main and two pool lanes before yielding");
      if (failure === "pool") pending[0].reject(new Error("pool lane failed"));
      else pending[0].resolve(new Float64Array(0));
      await Promise.resolve(); await Promise.resolve();
      if (freed || settled) throw new Error("Plan released before the remaining lane drained");
      pending[1].resolve(new Float64Array(0));
      outcomes.push({ error: await job, freed, finished, mainCalls });
    }
    return outcomes;
  }, stripTypeScriptTypes(source));
  expect(outcomes).toEqual([
    { error: "main lane failed", freed: 1, finished: 0, mainCalls: 1 },
    { error: "pool lane failed", freed: 1, finished: 0, mainCalls: 1 },
  ]);
});
