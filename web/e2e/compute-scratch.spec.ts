import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";

test("paired k-neighbours and covariance SIMD preserve scalar distance and normal bits", async ({ page }) => {
  await page.route("**/__scratch-wasm.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const wasm = await import("/__scratch-wasm.js");
    await wasm.default({ module_or_path: wasmPath });
    const same = (a: Float32Array | Float64Array, b: Float32Array | Float64Array) => {
      const x = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      return x.length === y.length && x.every((v, i) => v === y[i]);
    };
    let checked = 0;
    for (const count of [1, 17, 65, 257]) for (const mode of ["local", "utm", "tiny", "huge", "coincident", "nonfinite"]) {
      const points = Float64Array.from(Array.from({ length: count }, (_, i) => {
        const j = i % 23, p = [j % 7 * 0.1, Math.floor(j / 7) * 0.1, Math.sin(j) * 0.02];
        if (mode === "utm") return p.map((v, a) => v + [368000, 3955000, 40][a]);
        if (mode === "tiny") return p.map(v => v * 1e-160);
        if (mode === "huge") return p.map(v => v * 1e155);
        if (mode === "coincident") return [1, -0, 0];
        if (mode === "nonfinite" && i === 0) return [NaN, Infinity, -Infinity];
        return p;
      }).flat());
      const queries = new Float64Array([...points.slice(0, Math.min(points.length, 15)), NaN, 0, 0, Infinity, -0, 0]);
      const part = new wasm.SorPart(points);
      try {
        for (const k of [1, 3, 12, 24, count + 2]) {
          if (!same(part.withinWithScratch(queries, k, false), part.withinWithScratch(queries, k, true))) throw new Error("Neighbour distance bits changed");
          for (const orientation of [new Float64Array([0]), new Float64Array([1, 0, 0, 0])]) {
            if (!same(wasm.normalsOfWithScratch(points, k, orientation, false), wasm.normalsOfWithScratch(points, k, orientation, true))) throw new Error(`Normal bits changed: ${count}/${mode}/${k}`);
          }
          checked++;
        }
      } finally { part.free(); }
    }
    return checked;
  }, wasmPath);
  expect(checked).toBe(120);
});

async function cacheScenario(page: import("@playwright/test").Page, kind: "icp" | "voxel" | "m3c2") {
  await page.goto("/");
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n));
  return page.evaluate(async ({ workerPath, kind }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const p = waiting.get(data.seq)!; waiting.delete(data.seq);
      data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = e => { for (const p of waiting.values()) p.reject(new Error(e.message)); waiting.clear(); };
    const call = (req: any): Promise<any> => new Promise((resolve, reject) => {
      const id = ++seq; waiting.set(id, { resolve, reject }); worker.postMessage({ seq: id, req });
    });
    const file = (lift: number) => {
      const count = kind === "m3c2" ? 40017 : 6017, header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + count * 24); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < count; i++) {
        const x = (i % 80) * 0.1, y = Math.floor(i / 80) * 0.1;
        [x + 368000, y + 3955000, 40 + Math.sin(x) * 0.2 + y * y * 0.01 + lift].forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], "geometry-fixture.ply");
    };
    const load = (lift: number) => call({ kind: "load", file: file(lift), maxPoints: kind === "m3c2" ? 40017 : 6017 });
    const transform = (id: number) => call({ kind: "transform", id, matrix: [1,0,0,0,0,1,0,0,0,0,1,0.03,0,0,0,1] });
    try {
      let reference = await load(0);
      if (kind === "m3c2") {
        await call({ kind: "cpu-prepare" });
        const compared = await load(0.02);
        const compute = async (normalRadius = 0.4, projectionRadius = 0.25, cacheNormals?: boolean) => {
          const x = await call({ kind: "m3c2", compared: compared.id, reference: reference.id,
            normalRadius, projectionRadius, maxDepth: 0.5, coreSpacing: 0, cacheNormals });
          await call({ kind: "remove", id: x.cloud.id });
          return x;
        };
        const a = await compute(), b = await compute(), c = await compute(0.4,0.3);
        const raw = await compute(0.4,0.3,false);
        for (const key of ["distance","lod95","significant"]) {
          const x = new Uint32Array(c[key].buffer), y = new Uint32Array(raw[key].buffer);
          if (x.length !== y.length || x.some((v,i)=>v!==y[i])) throw new Error("Automatic normal reuse changed M3C2 bits");
        }
        const changed = await compute(0.5), capture = await compute(0.5), hit = await compute(0.5);
        await transform(reference.id); const moved = await compute(0.5);
        return { hits: [a,b,c,raw,changed,capture,hit,moved].map(x=>x.normalCacheHit),
          bytes: [a,b,c,raw,changed,capture,hit,moved].map(x=>x.normalCacheBytes) };
      }
      if (kind === "icp") {
        const compute = async (cache = true) => {
          const moving = await load(0.02);
          const result = await call({ kind: "icp", moving: moving.id, reference: reference.id, maxIterations: 8, overlap: 0.8, matchCentroids: false, pointToPlane: true, icpCache: cache });
          const bytes = await call({ kind: "export", id: moving.id, format: "ply" });
          await call({ kind: "remove", id: moving.id });
          return { result, bytes };
        };
        const a = await compute(), b = await compute(), control = await compute(false);
        const equal = (x: any, y: any) => {
          if (!x.result.matrix.every((v: number, i: number) => Object.is(v, y.result.matrix[i])) ||
            !["rmsInitial", "rmsFinal", "iterations", "converged"].every(k => Object.is(x.result[k], y.result[k])) ||
            x.bytes.length !== y.bytes.length || x.bytes.some((v: number, i: number) => v !== y.bytes[i])) throw new Error("Cached ICP parity failed");
        };
        equal(a, b); equal(a, control);
        await compute(); await transform(reference.id);
        const moved = await compute(); equal(moved, await compute(false));
        await call({ kind: "remove", id: reference.id }); reference = await load(0);
        const replaced = await compute(); equal(a, replaced);
        return { hits: [a.result.referenceReused, b.result.referenceReused, control.result.referenceReused, moved.result.referenceReused, replaced.result.referenceReused],
          cached: [a.result.referenceCached, control.result.referenceCached], bytes: a.result.referenceBytes };
      }
      const compared = await load(0.02);
      const standalone = (voxel = 1, minPoints = 5, voxelSimd = false, voxelCache = true) => call({ kind: "map-quality", compared: compared.id, reference: reference.id, voxel, minPoints, voxelSimd, voxelCache });
      const a = await standalone(), b = await standalone(), raw = await standalone(1,5,false,false);
      const equal = (x: any, y: any) => {
        if (x.voxels !== y.voxels || ["awd", "scs"].some(k => !Object.is(x[k], y[k]) && Math.abs(x[k] - y[k]) > 1e-12 * Math.max(1, Math.abs(x[k]), Math.abs(y[k])))) throw new Error("Voxel cache precision changed");
      };
      equal(a,b); equal(a,raw);
      const changed = [await standalone(0.5), await standalone(0.5), await standalone(0.5,6), await standalone(0.5,6,true)];
      await standalone();
      const evaluate = (threshold: number) => call({ kind: "map-evaluate", compared: compared.id, reference: reference.id, voxel: 1, minPoints: 5, threshold });
      const small = await evaluate(0.001), large = await evaluate(0.5);
      equal(a, small.quality); equal(a, large.quality);
      await transform(reference.id);
      const moved = await standalone(); equal(moved, await standalone(1,5,false,false));
      await call({ kind: "remove", id: reference.id }); reference = await load(0);
      const replaced = await standalone(); equal(a,replaced);
      return { hits: [a.voxelCacheHit,b.voxelCacheHit,raw.voxelCacheHit,...changed.map(x=>x.voxelCacheHit),moved.voxelCacheHit,replaced.voxelCacheHit],
        thresholdHits: [small.voxelCacheHit,large.voxelCacheHit], shares: [small.quality.precision,large.quality.precision] };
    } finally { worker.terminate(); }
  }, { workerPath, kind });
}

test("ICP reference reuse preserves fits and invalidates edits, controls and removed sources", async ({ page }) => {
  const r = await cacheScenario(page,"icp");
  expect(r.hits).toEqual([false,true,false,false,false]);
  expect(r.cached).toEqual([true,false]);
  expect(r.bytes).toBeGreaterThan(0);
  expect(r.bytes).toBeLessThanOrEqual(128*1024*1024);
});

test("voxel score reuse distinguishes geometry/settings and retains fresh inlier thresholds", async ({ page }) => {
  const r = await cacheScenario(page,"voxel");
  expect(r.hits).toEqual([false,true,false,false,true,false,false,false,false]);
  expect(r.thresholdHits).toEqual([true,true]);
  expect(r.shares![0]).toBeLessThan(r.shares![1]);
});

test("automatic M3C2 normal reuse waits for repeated fitting settings and respects controls and edits", async ({ page }) => {
  const r = await cacheScenario(page,"m3c2");
  expect(r.hits).toEqual([false,false,true,false,false,false,true,false]);
  expect(r.bytes).toEqual([0,expect.any(Number),expect.any(Number),expect.any(Number),expect.any(Number),expect.any(Number),expect.any(Number),0]);
  // Disabling use retains the bounded records until replacement or retirement.
  expect(r.bytes![3]).toBe(r.bytes![2]);
  expect(r.bytes![4]).toBe(r.bytes![2]);
  expect(r.bytes![1]).toBeGreaterThan(0);
  expect(r.bytes!.every(n=>n<=64*1024*1024)).toBe(true);
});
