import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import type { LoadedCloud, M3c2Output, Request } from "../src/protocol";

for (const fault of ["import", "search", "generation"]) {
  test(`prepared M3C2 recovers ${fault} failure, reuses changed parameters and invalidates moved sources`, async ({ page }) => {
    await page.route("**/__m3c2-preparation", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
    await page.goto("/__m3c2-preparation");
    await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
      const response = await route.fetch();
      await route.fulfill({ response, body: `
        Object.defineProperty(navigator, "hardwareConcurrency", { value: 5 });
        const originalPost = Worker.prototype.postMessage;
        let injected = false;
        Worker.prototype.postMessage = function(request, options) {
          const fault = ${JSON.stringify(fault)};
          if (!injected && request.kind === (fault === "import" ? "m3c2-prepare" : "m3c2-prepared")) {
            injected = true;
            if (fault === "import") request.reference[0] = 0;
            if (fault === "search") request.core = request.core.subarray(0, 1);
            if (fault === "generation") {
              this.dispatchEvent(new ErrorEvent("error", { message: "Injected M3C2 worker loss" }));
              return;
            }
          }
          return originalPost.call(this, request, options);
        };
      ` + await response.text() });
    });
    const path = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n));
    const result = await page.evaluate(async workerPath => {
      const worker = new Worker(workerPath, { type: "module" });
      let seq = 0;
      const waiting = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
      worker.onmessage = ({ data }) => {
        if (!data.response) return;
        const pending = waiting.get(data.seq)!; waiting.delete(data.seq);
        data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
      };
      worker.onerror = event => { for (const p of waiting.values()) p.reject(new Error(event.message)); waiting.clear(); };
      const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
        const id = ++seq; waiting.set(id, { resolve: v => resolve(v as T), reject }); worker.postMessage({ seq: id, req });
      });
      const fixture = (lift: number) => {
        const n = 40_013;
        const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
        const bytes = new Uint8Array(header.length + n * 24); bytes.set(header);
        const view = new DataView(bytes.buffer, header.length);
        for (let i = 0; i < n; i++) {
          const x = i % 201 * 0.1, y = Math.floor(i / 201) * 0.1;
          [368_000 + x, 3_955_000 + y, x * 0.2 + lift].forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
        }
        return new File([bytes], "boundary-fixture.ply");
      };
      try {
        const reference = await call<LoadedCloud>({ kind: "load", file: fixture(0), maxPoints: 40_013 });
        const compared = await call<LoadedCloud>({ kind: "load", file: fixture(0.07), maxPoints: 40_013 });
        const compute = (normalRadius = 0.4, parallel = true, coreSpacing = 0) => call<M3c2Output>({ kind: "m3c2", reference: reference.id,
          compared: compared.id, normalRadius, projectionRadius: 0.25, maxDepth: 0.5, coreSpacing, parallel, preparedTrees: true, cacheNormals: true, sharedSearch: parallel });
        let failed = false;
        try { await compute(); } catch { failed = true; }
        const recovered = await compute();
        const warm = await compute(0.5);
        const concurrent = await Promise.all([compute(0.5), compute(0.4)]);
        const control = await compute(0.5, false), controlOther = await compute(0.4, false);
        const sameExport = async (a: M3c2Output, b: M3c2Output) => {
          const x = await call<Uint8Array>({ kind: "export", id: a.cloud.id, format: "ply" });
          const y = await call<Uint8Array>({ kind: "export", id: b.cloud.id, format: "ply" });
          return x.length === y.length && x.every((v, i) => v === y[i]);
        };
        const equal = await sameExport(warm, control) && await sameExport(concurrent[0], control) && await sameExport(concurrent[1], controlOther);
        const spaced = await compute(0.5, true, 0.08), spacedControl = await compute(0.5, false, 0.08);
        if (!spaced.treeCacheHit || !await sameExport(spaced, spacedControl)) throw new Error("Changed core selection did not reuse exact geometry");
        await call({ kind: "transform", id: reference.id, matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.01, 0, 0, 0, 1] });
        const moved = await compute(0.5), movedControl = await compute(0.5, false);
        return { failed, recoveredHit: recovered.treeCacheHit, warmHit: warm.treeCacheHit,
          concurrentHits: concurrent.map(v => v.treeCacheHit), concurrentNormalHits: concurrent.map(v => v.normalCacheHit), equal, movedHit: moved.treeCacheHit, movedNormalHit: moved.normalCacheHit,
          movedEqual: await sameExport(moved, movedControl), changed: warm.distance.some((v, i) =>
            Number.isFinite(v) && Number.isFinite(moved.distance[i]) && v !== moved.distance[i]), workers: moved.workers };
      } finally { worker.terminate(); }
    }, path);
    expect(result).toEqual({ failed: true, recoveredHit: false, warmHit: true, concurrentHits: [true, true], concurrentNormalHits: [true, false],
      equal: true, movedHit: false, movedNormalHit: false, movedEqual: true, changed: true, workers: 4 });
  });
}

test("prepared M3C2 drains import/search/main failures and holds retired concurrent leases", async ({ page }) => {
  const { stripTypeScriptTypes } = await import("node:module");
  const source = readFileSync("src/m3c2-preparation.ts", "utf8").replace(/import .*?;\r?\n/g, "")
    .replace(/export (class|function)/g, "$1");
  await page.route("**/__m3c2-preparation", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__m3c2-preparation");
  const outcomes = await page.evaluate(async code => {
    const create = new Function("M3c2Prepared", "runOn", "runOnGeneration", "workerGeneration", "releaseOn",
      code + "\nreturn { prepareM3c2, invalidateM3c2Preparation }; ");
    const results = [];
    for (const failure of ["import", "search", "main", "cores", "generation", "none"]) {
      let freed = 0, mainCalls = 0, releases = 0, generation = 1;
      const imports: { resolve: (v: Float64Array) => void; reject: (e: Error) => void }[] = [];
      const searches: { resolve: (v: Float64Array) => void; reject: (e: Error) => void }[] = [];
      class Prepared {
        referenceSnapshot() { return new Uint8Array(16); }
        comparedSnapshot() { return new Uint8Array(16); }
        free() { freed++; }
        compute() { mainCalls++; if (failure === "main") throw new Error("main failed"); return new Float64Array(2); }
      }
      const { prepareM3c2, invalidateM3c2Preparation } = create(Prepared,
        () => new Promise<Float64Array>((resolve, reject) => imports.push({ resolve, reject })),
        (_lane: number, g: number) => g !== generation ? Promise.reject(new Error("generation lost")) :
          new Promise<Float64Array>((resolve, reject) => searches.push({ resolve, reject })),
        () => generation, () => { releases++; return Promise.resolve(true); });
      const a = {}, b = {};
      const { preparation, hit } = prepareM3c2(a, b, 2);
      if (hit || imports.length !== 2 || !prepareM3c2(a, b, 2).hit) throw new Error("Immutable pair was not cached");
      if (failure === "generation") generation++;
      let settled = false;
      const core = (lane: number) => { if (failure === "cores" && lane === 0) throw new Error("cores failed"); return new Float64Array(3); };
      const pending = preparation.compute(core, 0.4, 0.25, 0.5, false).then(() => { settled = true; return "success"; }, (e: Error) => { settled = true; return e.message; });
      // Invalidate during the await: sources can disappear, but active trees
      // must survive until every import and search response has drained.
      invalidateM3c2Preparation(a);
      if (freed || releases || settled || mainCalls !== 1) throw new Error("Retired active state was freed early");
      if (failure === "import") imports[0].reject(new Error("import failed"));
      else imports[0].resolve(new Float64Array(0));
      if (failure === "search") searches[0].reject(new Error("search failed"));
      else if (searches[0]) searches[0].resolve(new Float64Array(2));
      await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
      if (freed || releases || settled) throw new Error("Remaining lane was not drained");
      imports[1].resolve(new Float64Array(0));
      if (searches[1]) searches[1].resolve(new Float64Array(2));
      const outcome = await pending;
      if (freed !== 1 || releases !== 2 || (failure === "none" ? outcome !== "success" : outcome === "success"))
        throw new Error("Failed state cleanup or publication");
      results.push(outcome);
    }
    // Two calls share immutable state. Retiring between their responses
    // frees neither the main nor pool trees while the second lease remains.
    let freed = 0, releases = 0;
    const searches: ((v: Float64Array) => void)[] = [];
    class Prepared {
      referenceSnapshot() { return new Uint8Array(0); }
      comparedSnapshot() { return new Uint8Array(0); }
      compute() { return new Float64Array(2); }
      free() { freed++; }
    }
    const { prepareM3c2, invalidateM3c2Preparation } = create(Prepared, () => Promise.resolve(new Float64Array(0)),
      () => new Promise<Float64Array>(resolve => searches.push(resolve)), () => 1,
      () => { releases++; return Promise.resolve(true); });
    const a = {}, b = {}, state = prepareM3c2(a, b, 1).preparation;
    const first = state.compute(() => new Float64Array(3), 0.4, 0.25, 0.5, false);
    const second = state.compute(() => new Float64Array(3), 0.5, 0.3, 0.6, true);
    invalidateM3c2Preparation(b);
    searches[0](new Float64Array(2)); await first;
    if (freed || releases) throw new Error("Other active lease was released");
    searches[1](new Float64Array(2)); await second;
    if (freed !== 1 || releases !== 1) throw new Error("Last lease was leaked");
    return results;
  }, stripTypeScriptTypes(source, { mode: "transform" }));
  expect(outcomes).toEqual(["import failed", "search failed", "main failed", "cores failed", "generation lost", "success"]);
});

test("exact-tree memory estimate admits full real-sized pairs and retains serial thresholds", async ({ page }) => {
  const { stripTypeScriptTypes } = await import("node:module");
  const source = readFileSync("src/m3c2-workers.ts", "utf8").replace(/import .*?;\r?\n/g, "")
    .replace("export function parallelM3c2", "function parallelM3c2");
  await page.route("**/__m3c2-preparation", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__m3c2-preparation");
  const outcomes = await page.evaluate(async code => {
    const create = new Function("M3c2Plan", "poolSize", "prepareM3c2",
      "const planM3c2Cores = (cloud, spacing) => ({ plan: new M3c2Plan(cloud, spacing), hit: false, bytes: 0 });\n" + code + "\nreturn parallelM3c2;");
    class Plan {
      length = 40_013;
      interleavedPoints(lane: number, parts: number) { return new Float64Array([Math.floor(this.length / parts) + Number(lane < this.length % parts)]); }
      finish(v: Float64Array) { return v; }
      free() {}
    }
    const outcomes = [];
    for (const [points, lanes] of [[1_063_816, 8], [1_757_841, 8], [1_883_866, 8], [2_000_000, 8], [40_026, 8], [40_026, 1]]) {
      const compute = create(Plan, () => lanes, (_a: unknown, _b: unknown, pools: number) => ({ hit: false,
        preparation: { compute: async (cores: (i: number) => Float64Array) => Array.from({ length: pools + 1 }, (_, i) => ({ distances: new Float64Array(cores(i)[0] * 2), normalCacheHit: false, normalCacheBytes: 0 })), retire() {} } }));
      const cloud = { length: points / 2, rawPositions: () => { throw new Error("Raw inputs copied"); } };
      const pending = compute(cloud, cloud, 0.4, 0.25, 0.5, 0, true, false, true, true);
      outcomes.push(pending ? (await pending).workers : 1);
    }
    return outcomes;
  }, stripTypeScriptTypes(source));
  expect(outcomes).toEqual([3, 2, 2, 1, 4, 1]);
});

test("normal cache uses a fixed per-lane ceiling across core-layout changes", async ({ page }) => {
  const { stripTypeScriptTypes } = await import("node:module");
  const source = readFileSync("src/m3c2-workers.ts", "utf8").replace(/import .*?;\r?\n/g, "")
    .replace("export function parallelM3c2", "function parallelM3c2");
  await page.route("**/__m3c2-preparation", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__m3c2-preparation");
  const outcomes = await page.evaluate(async code => {
    const construct = new Function("M3c2Plan", "poolSize", "prepareM3c2",
      "const planM3c2Cores = (cloud, spacing) => ({ plan: new M3c2Plan(cloud, spacing), hit: false, bytes: 0 });\n" + code + "\nreturn parallelM3c2;");
    const outcomes = [];
    for (const count of [1_198_371, 1_198_372, 1_198_373]) {
      let allowance = -1;
      class Plan {
        length = count;
        interleavedPoints(lane: number, parts: number) { return new Float64Array([Math.floor(count / parts) + Number(lane < count % parts)]); }
        finish() { return {}; }
        free() {}
      }
      const compute = construct(Plan, () => 4, (_a: unknown, _b: unknown, pools: number) => ({ hit: false,
        preparation: { retire() {}, compute: async (cores: (i: number) => Float64Array, _r: number, _p: number, _d: number, _b: boolean, enabled: boolean, max: number) => {
          if (!enabled) throw new Error("Normal-cache control lost");
          allowance = max;
          return Array.from({ length: pools + 1 }, (_, i) => ({ distances: new Float64Array(cores(i)[0] * 2),
            normalCacheHit: false, normalCacheBytes: max ? cores(i)[0] * 56 : 0 }));
        } } }));
      // Exercise the policy without allocating a million source points.
      const cloud = { length: 531_908 };
      const result = await compute(cloud, cloud, 0.4, 0.25, 0.5, 0, true, false, true, true, true);
      outcomes.push({ allowance, bytes: result.normalCacheBytes });
    }
    return outcomes;
  }, stripTypeScriptTypes(source));
  expect(outcomes).toEqual([{ allowance: 399_457, bytes: 1_198_371 * 56 }, { allowance: 0, bytes: 0 }, { allowance: 0, bytes: 0 }]);
});
