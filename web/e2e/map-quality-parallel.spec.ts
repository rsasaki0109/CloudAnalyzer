import { expect, type Page, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import type { C2cOutput, LoadedCloud, MapQuality, Request } from "../src/protocol";

async function setup(page: Page, cores = 5, prelude = "") {
  await page.route("**/__quality-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__quality-test");
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: `Object.defineProperty(navigator, "hardwareConcurrency", { value: ${cores} });\n${prelude}\n` + await response.text() });
  });
  return "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n))!;
}

async function scenario(page: Page, workerPath: string, counts: number[], mutate = false, fail = false, threshold = 0.15, nearest = false) {
  return page.evaluate(async ({ workerPath, counts, mutate, fail, threshold, nearest }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    let announce: () => void;
    const started = new Promise<void>(resolve => { announce = resolve; });
    worker.onmessage = ({ data }) => {
      if (data.qualityStarted) { announce(); return; }
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
    const fixture = (n: number, changed: boolean) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 24);
      bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < n; i++) {
        const x = i % 401, y = Math.floor(i / 401);
        // Close neighbours, a changed region, duplicate/tied points, and a
        // far point test partition borders and original f64 coordinates.
        const xyz = i === n - 1 ? [400_000, 4_000_000, changed ? 51 : 50]
          : [368_000 + (x - x % 2) * 0.1, 3_955_000 + y * 0.1,
            40 + Math.sin(x / 20) * 0.1 + (changed ? x < 200 ? 0.07 : 1.1 : 0)];
        xyz.forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], changed ? "compared.ply" : "truth.ply");
    };
    // The former UI calculation is an independent baseline, including its
    // float32 inlier distances and NaN semantics for an empty inlier set.
    const inliers = (ds: Float32Array) => {
      let n = 0, sum = 0, squares = 0;
      for (const d of ds) if (d <= threshold) { n++; sum += d; squares += d * d; }
      return { share: n / ds.length, mean: n ? sum / n : NaN, rms: n ? Math.sqrt(squares / n) : NaN };
    };
    try {
      const compared = await call<LoadedCloud>({ kind: "load", file: fixture(counts[0], true), maxPoints: counts[0] });
      const reference = await call<LoadedCloud>({ kind: "load", file: fixture(counts[1], false), maxPoints: counts[1] });
      const ours = await call<C2cOutput>({ kind: "c2c", compared: compared.id, reference: reference.id, signed: false, cpuNearestSimd: false, summarySort: true });
      const theirs = await call<C2cOutput>({ kind: "c2c", compared: reference.id, reference: compared.id, signed: false, cpuNearestSimd: false, summarySort: true });
      const params = { compared: compared.id, reference: reference.id, threshold, voxel: 1, minPoints: 10 };
      const voxels = await call<{ awd: number; scs: number; voxels: number }>({ kind: "map-quality", ...params });
      const a = inliers(ours.distances), b = inliers(theirs.distances);
      const expected: MapQuality = { threshold, accuracy: a.rms, precision: a.share, completeness: b.share,
        f1: a.share + b.share ? 2 * a.share * b.share / (a.share + b.share) : 0,
        chamfer: (a.mean + b.mean) / 2, ...voxels };
      let recovered = !fail;
      if (fail) {
        try { await call({ kind: "map-evaluate", ...params }); }
        catch (error) { recovered = String(error).includes("multiple of 3"); }
      }
      const pending = call<C2cOutput & { quality: MapQuality }>(nearest
        ? { kind: "c2c", compared: compared.id, reference: reference.id, signed: false }
        : { kind: "map-evaluate", ...params });
      if (mutate) {
        await started;
        for (const id of [reference.id, compared.id]) {
          await call({ kind: "transform", id, matrix: [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 0.1, 0, 0, 0, 1] });
          await call({ kind: "remove", id });
        }
      }
      const result = await pending;
      const bits = (ds: Float32Array) => new Uint32Array(ds.buffer, ds.byteOffset, ds.length);
      const left = bits(ours.distances), right = bits(result.distances);
      let mismatches = left.length === right.length ? 0 : 1;
      left.forEach((v, i) => { if (v !== right[i]) mismatches++; });
      const statsEqual = Object.keys(ours.stats).every(key => Object.is(ours.stats[key as keyof typeof ours.stats], result.stats[key as keyof typeof result.stats]));
      const exactKeys = ["threshold", "accuracy", "precision", "completeness", "f1", "chamfer", "voxels"] as const;
      const quality = result.quality ?? expected;
      const qualityExact = exactKeys.every(key => Object.is(expected[key], quality[key]));
      // HashMap reduction order already varies between repeated serial voxel
      // calls; retain that kernel and allow only floating reduction roundoff.
      const voxelEqual = ["awd", "scs"].every(key => {
        const a = expected[key as "awd" | "scs"], b = quality[key as "awd" | "scs"];
        return Object.is(a, b) || Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b));
      });
      return { mismatches, statsEqual, qualityExact, voxelEqual, recovered,
        workers: result.workers, quality };
    } finally { worker.terminate(); }
  }, { workerPath, counts, mutate, fail, threshold, nearest });
}

for (const counts of [[100_013, 100_017], [200_013, 200_017], [200_013, 1_013]]) {
  test(`map quality matches the former bidirectional workflow (${counts.join("/")} points)`, async ({ page }) => {
    const result = await scenario(page, await setup(page), counts);
    expect(result).toMatchObject({ mismatches: 0, statsEqual: true, qualityExact: true, voxelEqual: true, recovered: true });
    expect(result.workers).toBe(counts[0] === 100_013 || counts[1] === 1_013 ? 2 : 4);
  });
}

test("small maps and a single available CPU retain serial results, including empty inliers", async ({ page }) => {
  for (const [counts, cores] of [[[1_013, 1_017], 5], [[100_013, 100_017], 2]] as const) {
    const result = await scenario(page, await setup(page, cores), [...counts], false, false, 0.001);
    expect(result).toMatchObject({ workers: 1, mismatches: 0, statsEqual: true, qualityExact: true, voxelEqual: true });
    expect(result.quality.f1).toBe(0);
    expect(result.quality.accuracy).toBeNaN();
    expect(result.quality.chamfer).toBeNaN();
  }
});

test("map quality completes its snapshots after both clouds move and disappear", async ({ page }) => {
  const path = await setup(page, 5, `
    let evaluating = false;
    self.addEventListener("message", event => { if (event.data.req?.kind === "map-evaluate") evaluating = true; });
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(request, options) {
      if (evaluating && ["cloud-prepare", "cloud-prepared"].includes(request.kind)) {
        self.postMessage({ qualityStarted: true });
        setTimeout(() => post.call(this, request, options), 200);
      } else post.call(this, request, options);
    };
  `);
  const result = await scenario(page, path, [200_013, 200_017], true);
  expect(result).toMatchObject({ workers: 4, mismatches: 0, statsEqual: true, qualityExact: true, voxelEqual: true });
});

test("a failed distance slice rejects the evaluation and the same workers recover", async ({ page }) => {
  const path = await setup(page, 5, `
    let armed = false, failed = false;
    self.addEventListener("message", event => { if (event.data.req?.kind === "map-evaluate" && !failed) armed = true; });
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(request, options) {
      if (armed && request.kind === "cloud-prepare") {
        armed = false; failed = true;
        request.queries = request.queries.subarray(0, request.queries.length - 1);
      }
      post.call(this, request, options);
    };
  `);
  const result = await scenario(page, path, [200_013, 200_017], false, true);
  expect(result).toMatchObject({ workers: 4, mismatches: 0, statsEqual: true, qualityExact: true, voxelEqual: true, recovered: true });
});

test("parallel nearest C2C also retains the point count after its source is removed", async ({ page }) => {
  const path = await setup(page, 5, `
    let evaluations = 0, evaluating = false;
    self.addEventListener("message", event => {
      if (event.data.req?.kind === "c2c" && ++evaluations === 3) evaluating = true;
    });
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(request, options) {
      if (evaluating && ["cloud", "cloud-prepare", "cloud-prepared"].includes(request.kind)) {
        self.postMessage({ qualityStarted: true });
        setTimeout(() => post.call(this, request, options), 200);
      } else post.call(this, request, options);
    };
  `);
  const result = await scenario(page, path, [200_013, 200_017], true, false, 0.15, true);
  expect(result).toMatchObject({ workers: 2, mismatches: 0, statsEqual: true });
});

test("the copy budget keeps oversized inputs and references off the pool", async ({ page }) => {
  // Exercise the production preparation/fallback/freeing flow with reported
  // buffer sizes, avoiding a large allocation just to cross the copy budget.
  const { stripTypeScriptTypes } = await import("node:module");
  const source = readFileSync("src/map-quality-workers.ts", "utf8").replace(/import .*?;\r?\n/g, "")
    .replace("export async function evaluateMapQuality", "async function evaluateMapQuality");
  await page.route("**/__quality-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__quality-test");
  const result = await page.evaluate(async code => {
    const make = new Function("MIN_PARALLEL_QUERIES", "cloudToCloud", "mapQuality", "planCloudToCloud", "summarizeDistances", "poolSize", "runSlices", "eachSlice", "findCpuPreparation", "prepareCpuSlices", code.replace("snapshotVoxelScores", "((...args) => { const [awd,scs,voxels]=mapQuality(...args); return {scores:{awd,scs,voxels},reused:false}; })") + "\nreturn evaluateMapQuality;");
    const outcomes = [];
    for (const cpuCache of [false, true]) for (const count of [4_000_000, 200_013]) {
      let dispatched = 0, serial = 0, freed = 0, plansFreed = 0;
      const cloud = { length: count, rawPositions: () => { throw new Error("Unexpected full input copy"); } };
      const native = () => { serial++; return { distances: () => new Float32Array([0.07]), free: () => { freed++; } }; };
      const plan = () => ({ length: 2, takeQueries: () => new Float64Array(3),
        takeQueryIndices: () => new Uint32Array([0]),
        takeReference: () => ({ byteLength: 64 * 1024 * 1024 }), free: () => { plansFreed++; } });
      const evaluate = make(100_000, native, () => [0.1, 0.01, 1], plan,
        () => { throw new Error("Unexpected pool summary"); }, () => 4,
        async (n: number) => { dispatched += n; return []; },
        async (n: number) => { dispatched += n; }, () => undefined,
        () => { throw new Error("Unexpected retained preparation above the copy budget"); });
      const result = await evaluate(cloud, cloud, 0.15, 1, 10, true, cpuCache);
      result.result.free();
      outcomes.push({ dispatched, serial, freed, plansFreed, workers: result.workers });
    }
    return outcomes;
  }, stripTypeScriptTypes(source));
  expect(result).toEqual([
    { dispatched: 0, serial: 2, freed: 2, plansFreed: 0, workers: 1 },
    { dispatched: 0, serial: 2, freed: 2, plansFreed: 2, workers: 1 },
    { dispatched: 0, serial: 2, freed: 2, plansFreed: 0, workers: 1 },
    { dispatched: 0, serial: 2, freed: 2, plansFreed: 2, workers: 1 },
  ]);
});

test("forward summaries overlap unfinished reverse slices and late failed results are ignored", async ({ page }) => {
  const { stripTypeScriptTypes } = await import("node:module");
  const source = readFileSync("src/map-quality-workers.ts", "utf8").replace(/import .*?;\r?\n/g, "")
    .replace("export async function evaluateMapQuality", "async function evaluateMapQuality");
  await page.route("**/__quality-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__quality-test");
  const result = await page.evaluate(async code => {
    const make = new Function("MIN_PARALLEL_QUERIES", "cloudToCloud", "mapQuality", "planCloudToCloud", "summarizeDistances", "poolSize", "runSlices", "eachSlice", code.replace("snapshotVoxelScores", "((...args) => { const [awd,scs,voxels]=mapQuality(...args); return {scores:{awd,scs,voxels},reused:false}; })") + "\nreturn evaluateMapQuality;");
    const cloud = { length: 200_013, rawPositions: () => new Float64Array(3) };
    const plan = () => ({ length: 2, takeQueries: () => new Float64Array(3),
      takeReference: () => new Float64Array(3), takeQueryIndices: (k: number) => new Uint32Array([k]), free() {} });
    const unexpected = () => { throw new Error("Unexpected serial C2C"); };
    const outcomes = [];
    for (const streaming of [false, true]) {
      let summaries = 0, freed = 0, complete = false;
      let release!: () => void, announce!: () => void;
      const gate = new Promise<void>(resolve => { release = resolve; });
      const forward = new Promise<void>(resolve => { announce = resolve; });
      const summary = (values: Float64Array) => {
        summaries++;
        const distances = Float32Array.from(values);
        return { distances: () => distances, free: () => { freed++; } };
      };
      const pump = async (done: (i: number, values: Float64Array) => void) => {
        await Promise.resolve();
        for (const i of [0, 2]) done(i, new Float64Array([0.07]));
        announce();
        await gate;
        for (const i of [1, 3]) done(i, new Float64Array([0.11]));
      };
      const collect = async () => {
        const results: Float64Array[] = [];
        await pump((k, value) => { results[k] = value; });
        return results;
      };
      const each = async (_n: number, _make: unknown, done: (k: number, values: Float64Array) => void) => pump(done);
      const evaluate = make(100_000, unexpected, () => [0.1, 0.01, 3], plan, summary, () => 4, collect, each);
      const pending = evaluate(cloud, cloud, 0.15, 1, 10, streaming, false).then((result: unknown) => { complete = true; return result; });
      await forward;
      const beforeRelease = { summaries, complete };
      release();
      const result = await pending;
      result.result.free();
      outcomes.push({ streaming, beforeRelease, summaries, freed, quality: result.quality });
    }
    let late!: (k: number, values: Float64Array) => void, lateSummaries = 0, failed = false;
    const evaluate = make(100_000, unexpected, () => [0.1, 0.01, 3], plan,
      () => { lateSummaries++; throw new Error("Late summary"); }, () => 4, unexpected,
      async (_n: number, _make: unknown, done: typeof late) => { late = done; throw new Error("Failed slice"); });
    const small = { length: 100_013, rawPositions: cloud.rawPositions };
    try { await evaluate(small, small, 0.15, 1, 10, true, false); }
    catch (error) { failed = String(error).includes("Failed slice"); }
    late(0, new Float64Array(small.length));
    return { outcomes, failed, lateSummaries };
  }, stripTypeScriptTypes(source));
  expect(result.outcomes[0]).toMatchObject({ beforeRelease: { summaries: 0, complete: false }, summaries: 2, freed: 2 });
  expect(result.outcomes[1]).toMatchObject({ beforeRelease: { summaries: 1, complete: false }, summaries: 1, freed: 1 });
  expect(result.outcomes[0].quality).toEqual(result.outcomes[1].quality);
  expect(result).toMatchObject({ failed: true, lateSummaries: 0 });
});
