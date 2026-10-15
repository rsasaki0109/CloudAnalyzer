import { expect, type Page, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import type { C2cOutput, LoadedCloud, Request } from "../src/protocol";

async function blank(page: Page) {
  await page.route("**/__c2c-simd", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__c2c-simd");
}

test("ordinary C2C SIMD preserves f64 distances against scalar and brute force", async ({ page }) => {
  await blank(page);
  const modulePath = "/__c2c-simd-wasm.js";
  await page.route("**" + modulePath, route => route.fulfill({ contentType: "text/javascript",
    body: readFileSync(new URL("../src/wasm/ca_wasm.js", import.meta.url), "utf8") }));
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const result = await page.evaluate(async ({ modulePath, wasmPath }) => {
    const { default: init, nearestDistances, c2cSimdEnabled, C2cPrepared } = await import(modulePath);
    await init({ module_or_path: wasmPath });
    if (!c2cSimdEnabled()) throw new Error("Expected the SIMD WASM build");
    const bits = (ds: Float64Array) => [...new Uint32Array(ds.buffer, ds.byteOffset, ds.length * 2)];
    const same = (a: Float64Array, b: Float64Array) => JSON.stringify(bits(a)) === JSON.stringify(bits(b));
    let checked = 0;
    for (const count of [1, 2, 3, 15, 16, 17, 31, 33, 257]) {
      for (const scale of [1e-160, 1e-150, 1, 1e150, 1e200]) {
        const origin = scale === 1 ? [368_000, 3_955_000, 40] : [0, 0, 0];
        const points = Array.from({ length: count }, (_, i) => [origin[0] + (i % 17 - 8) * scale,
          origin[1] + (i % 5 - 2) * scale, origin[2] + (i % 3 - 1) * scale]);
        if (count > 1) points[1] = [...points[0]];
        const queries = points.map((p, i) => [p[0] + (i % 2 ? 9e-6 : 0.5) * scale,
          p[1] - 0.25 * scale, p[2] + 0.75 * scale]);
        queries.push(...points.slice(0, 3), origin.map(v => v + 20 * scale));
        const reference = Float64Array.from(points.flat()), q = Float64Array.from(queries.flat());
        const expected = Float64Array.from(queries, query => {
          let best = Infinity;
          for (const p of points) {
            const dx = p[0] - query[0], dy = p[1] - query[1], dz = p[2] - query[2];
            best = Math.min(best, (dx * dx + dy * dy) + dz * dz);
          }
          return Math.sqrt(best);
        });
        const scalar = nearestDistances(reference, q, false), vector = nearestDistances(reference, q, true);
        if (!same(scalar, vector) || !same(vector, expected) || !same(vector, nearestDistances(reference, q)))
          throw new Error(`SIMD distance mismatch: ${count} points, scale ${scale}`);
        const prepared = new C2cPrepared(reference, q);
        reference.fill(0); q.fill(0); // The prepared job owns both inputs.
        try {
          for (const simd of [false, true, false]) if (!same(prepared.distances(simd), expected))
            throw new Error(`Prepared distance mismatch: ${count} points, scale ${scale}`);
        } finally { prepared.free(); }
        checked += vector.length;
      }
    }
    for (const reference of [new Float64Array([0, 0, 0, 1, 1, 1, NaN, 0, 0, Infinity, 1, 1, -0, 0, 0]),
      new Float64Array(1999 * 3).fill(4)]) {
      const queries = new Float64Array([NaN, 0, 0, Infinity, 1, 0, 0, -0, 0, 1e300, 1e300, 1e300, 3, 4, 5]);
      const a = nearestDistances(reference, queries, false), b = nearestDistances(reference, queries, true);
      if (!same(a, b)) throw new Error("SIMD nonfinite/coincident parity failed");
      const prepared = new C2cPrepared(reference, queries);
      try { if (!same(prepared.distances(false), a) || !same(prepared.distances(true), a))
        throw new Error("Prepared nonfinite/coincident parity failed"); }
      finally { prepared.free(); }
      checked += b.length;
    }
    for (const [reference, queries] of [[[], [0, 0, 0]], [[0, 0], []], [[0, 0, 0], [1, 2]]]) {
      let rejected = false;
      try { new C2cPrepared(Float64Array.from(reference), Float64Array.from(queries)).free(); }
      catch { rejected = true; }
      if (!rejected) throw new Error("Prepared malformed inputs were accepted");
    }
    const empty = new C2cPrepared(new Float64Array([0, 0, 0]), new Float64Array(0));
    try { if (empty.distances(false).length) throw new Error("Prepared empty query result"); }
    finally { empty.free(); }
    return checked;
  }, { modulePath, wasmPath });
  expect(result).toBeGreaterThan(1500);
});

test("CPU C2C propagates scalar and SIMD controls through serial and parallel workers", async ({ page }) => {
  await blank(page);
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n))!;
  const cases = await page.evaluate(async workerPath => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    worker.onmessage = ({ data }) => {
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
        const x = i % 401, y = Math.floor(i / 401);
        [368_000 + (x - x % 2) * 0.1, 3_955_000 + y * 0.1, 40 + Math.sin(x / 20) * 0.1 + lift]
          .forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    const cases = [];
    try {
      await call({ kind: "cpu-prepare" });
      for (const n of [1013, 200_013]) {
        const reference = await call<LoadedCloud>({ kind: "load", file: fixture(n + 4, 0), maxPoints: n + 4 });
        const compared = await call<LoadedCloud>({ kind: "load", file: fixture(n, 9e-6), maxPoints: n });
        const c2c = (cpuNearestSimd?: boolean) => call<C2cOutput>({ kind: "c2c", compared: compared.id,
          reference: reference.id, signed: false, backend: "cpu", cpuNearestSimd });
        const scalar = await c2c(false);
        if (scalar.cpuCached !== (scalar.workers > 1) || scalar.cpuCacheReused)
          throw new Error("CPU preparation default did not propagate");
        for (const flag of [true, undefined, false, true]) {
          const result = await c2c(flag);
          const a = new Uint32Array(scalar.distances.buffer), b = new Uint32Array(result.distances.buffer);
          if (a.length !== n || b.length !== n || a.some((v, i) => v !== b[i]) ||
              !Object.keys(scalar.stats).every(k => Object.is(scalar.stats[k as keyof typeof scalar.stats], result.stats[k as keyof typeof result.stats])))
            throw new Error("CPU SIMD production parity failed");
          if (result.cpuSimd !== (flag === true) || scalar.cpuSimd || scalar.workers !== result.workers)
            throw new Error("CPU SIMD control did not propagate");
        }
        cases.push({ count: n, workers: scalar.workers });
        await call({ kind: "remove", id: compared.id });
        await call({ kind: "remove", id: reference.id });
      }
      return cases;
    } finally { worker.terminate(); }
  }, workerPath);
  expect(cases).toEqual([{ count: 1013, workers: 1 }, { count: 200_013, workers: 2 }]);
});
