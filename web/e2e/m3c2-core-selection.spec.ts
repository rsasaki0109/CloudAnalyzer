import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import type { LoadedCloud, M3c2Output, Request } from "../src/protocol";

test("M3C2 core indices are bounded, replaced and invalidated while plans keep snapshots", async ({ page }) => {
  const { stripTypeScriptTypes } = await import("node:module");
  const source = readFileSync("src/m3c2-core-selection.ts", "utf8").replace(/import .*?;\r?\n/g, "")
    .replace(/export function/g, "function");
  test.skip(!source.includes("const USE_M3C2_CORE_SELECTION = true;"), "Candidate core reuse disabled");
  await page.route("**/__core-selection-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__core-selection-test");
  const result = await page.evaluate(code => {
    let selections = 0, allocations = 0;
    class Plan {
      readonly revision: number;
      constructor(cloud: { revision: number }) { allocations++; this.revision = cloud.revision; }
      static selectedIndices() { selections++; return new Uint32Array([0, 2]); }
      static fromIndices(cloud: { revision: number }, indices: Uint32Array) {
        if (indices.join() !== "0,2") throw new Error("Changed selection");
        return new Plan(cloud);
      }
    }
    const { planM3c2Cores: plan, invalidateM3c2CoreSelection: invalidate } = new Function("M3c2Plan", code +
      "\nreturn { planM3c2Cores, invalidateM3c2CoreSelection };")(Plan);
    const a = { length: 3, revision: 1 }, b = { length: 3, revision: 7 };
    const cold = plan(a, 0.25), warm = plan(a, 0.25);
    a.revision = 2; // Attribute-only updates must be copied even on index hits.
    const attributes = plan(a, 0.25);
    invalidate(b);
    const unrelated = plan(a, 0.25), changedSpacing = plan(a, 0.5);
    invalidate(a);
    const moved = plan(a, 0.5), different = plan(b, 0.5), previous = plan(a, 0.5);
    const disabled = plan(a, 0.5, false), afterDisabled = plan(a, 0.5);
    const controls = [0, -0, -1, NaN, Infinity].map(spacing => plan(a, spacing));
    const oversized = plan({ length: 2_097_153, revision: 3 }, 0.25);
    const admitted = plan({ length: 2_097_152, revision: 4 }, 0.25);
    return {
      hits: [cold, warm, attributes, unrelated, changedSpacing, moved, different, previous, disabled, afterDisabled].map(v => v.hit),
      oldRevision: cold.plan.revision, attributeRevision: attributes.plan.revision,
      bytes: cold.bytes, controls: controls.map(v => [v.hit, v.bytes]),
      oversized: [oversized.hit, oversized.bytes], admitted: [admitted.hit, admitted.bytes], selections, allocations,
    };
  }, stripTypeScriptTypes(source));
  expect(result).toEqual({ hits: [false, true, true, true, false, false, false, false, false, false],
    oldRevision: 1, attributeRevision: 2, bytes: 8, controls: Array(5).fill([false, 0]),
    oversized: [false, 0], admitted: [false, 8], selections: 7, allocations: 17 });
});

test("retained M3C2 core selections preserve complete exports through controls, edits and removal", async ({ page }) => {
  await page.route("**/__core-selection-worker", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__core-selection-worker");
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n));
  const result = await page.evaluate(async workerPath => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const pending = waiting.get(data.seq); if (!pending) return; waiting.delete(data.seq);
      data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const p of waiting.values()) p.reject(new Error(event.message)); waiting.clear(); };
    const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
      const id = ++seq; waiting.set(id, { resolve: v => resolve(v as T), reject }); worker.postMessage({ seq: id, req });
    });
    const fixture = (lift: number) => {
      const n = 40_013;
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float intensity\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 31); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < n; i++) {
        const x = i % 201 * 0.1, y = Math.floor(i / 201) * 0.1;
        [368_000 + x, 3_955_000 + y, x * 0.2 + lift].forEach((v, a) => view.setFloat64(i * 31 + a * 8, v, true));
        [i % 255, 10, 20].forEach((v, a) => view.setUint8(i * 31 + 24 + a, v));
        view.setFloat32(i * 31 + 27, i, true);
      }
      return new File([bytes], "core-fixture.ply");
    };
    try {
      const reference = await call<LoadedCloud>({ kind: "load", file: fixture(0), maxPoints: 40_013 });
      const compared = await call<LoadedCloud>({ kind: "load", file: fixture(0.07), maxPoints: 40_013 });
      const compute = (options: Partial<Extract<Request, { kind: "m3c2" }>> = {}) => call<M3c2Output>({ kind: "m3c2", reference: reference.id,
        compared: compared.id, normalRadius: 0.4, projectionRadius: 0.25, maxDepth: 0.5, coreSpacing: 0.08, cacheNormals: false, ...options });
      let comparisons = 0;
      const sameExport = async (a: M3c2Output, b: M3c2Output) => {
        const x = await call<Uint8Array>({ kind: "export", id: a.cloud.id, format: "ply" });
        const y = await call<Uint8Array>({ kind: "export", id: b.cloud.id, format: "ply" });
        if (x.length !== y.length || !x.every((v, i) => v === y[i])) throw new Error("Complete M3C2 core export changed");
        comparisons++;
      };
      const metadata = (r: M3c2Output) => [r.coreSelectionCacheHit, r.coreSelectionCacheBytes];
      const cold = await compute(), warm = await compute(), fresh = await compute({ cacheCoreSelection: false });
      await sameExport(warm, fresh);
      const recaptured = await compute(), changedRadius = await compute({ normalRadius: 0.5 });
      await sameExport(changedRadius, await compute({ normalRadius: 0.5, parallel: false }));
      const changedSpacing = await compute({ coreSpacing: 0.09 }), spacingWarm = await compute({ coreSpacing: 0.09 });
      await sameExport(spacingWarm, await compute({ coreSpacing: 0.09, parallel: false }));
      const layout = await compute({ coreSpacing: 0.09, interleaved: false });
      await sameExport(layout, await compute({ coreSpacing: 0.09, parallel: false }));
      const cachedNormal = await compute({ coreSpacing: 0.09, interleaved: false, cacheNormals: true });
      const bothWarm = await compute({ coreSpacing: 0.09, interleaved: false, cacheNormals: true });
      await sameExport(bothWarm, await compute({ coreSpacing: 0.09, parallel: false }));
      await call({ kind: "transform", id: reference.id, matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.01, 0, 0, 0, 1] });
      const movedReference = await compute({ coreSpacing: 0.09 });
      await sameExport(movedReference, await compute({ coreSpacing: 0.09, parallel: false }));
      await call({ kind: "transform", id: compared.id, matrix: [1, 0, 0, 0.01, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
      const movedCompared = await compute({ coreSpacing: 0.09 });
      await sameExport(movedCompared, await compute({ coreSpacing: 0.09, parallel: false }));
      const concurrent = await Promise.all([compute({ coreSpacing: 0.08 }), compute({ coreSpacing: 0.09 })]);
      for (const [i, r] of concurrent.entries()) await sameExport(r, await compute({ coreSpacing: i ? 0.09 : 0.08, parallel: false }));
      const pending = compute({ coreSpacing: 0.09 });
      await call({ kind: "remove", id: compared.id });
      const removed = await pending;
      await sameExport(removed, movedCompared);
      const replacement = await call<LoadedCloud>({ kind: "load", file: fixture(0.07), maxPoints: 40_013 });
      const reloaded = await compute({ compared: replacement.id, coreSpacing: 0.09 });
      await sameExport(reloaded, await compute({ compared: replacement.id, coreSpacing: 0.09, parallel: false }));
      const zero = await compute({ compared: replacement.id, coreSpacing: 0 });
      await sameExport(zero, await compute({ compared: replacement.id, coreSpacing: 0, parallel: false }));
      return { cold: metadata(cold), warm: metadata(warm), fresh: metadata(fresh), recaptured: metadata(recaptured),
        changedRadius: metadata(changedRadius), changedSpacing: metadata(changedSpacing), spacingWarm: metadata(spacingWarm),
        layout: metadata(layout), bothWarm: metadata(bothWarm), normals: [cachedNormal.normalCacheHit, bothWarm.normalCacheHit],
        reference: metadata(movedReference), compared: metadata(movedCompared), removed: metadata(removed),
        reloaded: metadata(reloaded), zero: metadata(zero), comparisons };
    } finally { worker.terminate(); }
  }, workerPath);
  const bytes = 40_013 * 4;
  expect(result).toEqual({ cold: [false, bytes], warm: [true, bytes], fresh: [false, 0], recaptured: [false, bytes],
    changedRadius: [true, bytes], changedSpacing: [false, bytes], spacingWarm: [true, bytes], layout: [true, bytes],
    bothWarm: [true, bytes], normals: [false, true], reference: [true, bytes], compared: [false, bytes],
    removed: [true, bytes], reloaded: [false, bytes], zero: [false, 0], comparisons: 12 });
});

test("WASM rejects malformed M3C2 core indices before selecting attributes", async ({ page }) => {
  await page.route("**/__core-indices", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__core-indices.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__core-indices");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const rejected = await page.evaluate(async wasmPath => {
    const wasm = await import("/__core-indices.js"); await wasm.default({ module_or_path: wasmPath });
    const cloud = wasm.Cloud.parse("points.xyz", new TextEncoder().encode("0 0 0\n1 0 0\n2 0 0\n"));
    try {
      let rejected = 0;
      for (const indices of [[3], [0, 0], [1, 0], [0, 2, 1], [0xffffffff]]) {
        try { const plan = wasm.M3c2Plan.fromIndices(cloud, Uint32Array.from(indices)); plan.free(); }
        catch (error) { if (!String(error).includes("Invalid ordered M3C2 core indices")) throw error; rejected++; }
      }
      for (const indices of [[], [0], [0, 2], [0, 1, 2]]) wasm.M3c2Plan.fromIndices(cloud, Uint32Array.from(indices)).free();
      return rejected;
    } finally { cloud.free(); }
  }, wasmPath);
  expect(rejected).toBe(5);
});
