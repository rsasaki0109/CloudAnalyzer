import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import type { C2cOutput, LoadedCloud, Request } from "../src/protocol";

test("voxel SIMD preserves scalar AWD/SCS across precision, occupancy and empty cases", async ({ page }) => {
  await page.route("**/__voxel-simd", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__voxel-simd");
  const modulePath = "/__voxel-simd-wasm.js";
  await page.route("**" + modulePath, route => route.fulfill({ contentType: "text/javascript",
    body: readFileSync(new URL("../src/wasm/ca_wasm.js", import.meta.url), "utf8") }));
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async ({ modulePath, wasmPath }) => {
    const { default: init, Cloud, mapQuality, c2cSimdEnabled } = await import(modulePath);
    await init({ module_or_path: wasmPath });
    if (!c2cSimdEnabled()) throw new Error("Expected production SIMD support");
    const cloud = (count: number, scale: number, lifted: boolean) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + count * 24); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      const offset = scale === 1 ? [368_000, 3_955_000, 40] : [0, 0, 0];
      for (let i = 0; i < count; i++) {
        const x = i % 41 - 20, y = Math.floor(i / 41) - 13;
        const p = [x * 0.1, y * 0.1, Math.sin(x / 5) * 0.1 + (lifted ? x < 0 ? 0.07 : 0.21 : 0)];
        p.forEach((v, a) => view.setFloat64(i * 24 + a * 8, offset[a] + v * scale, true));
      }
      return Cloud.parse("moments.ply", bytes);
    };
    const same = (a: Float64Array, b: Float64Array) => a.length === b.length && a.every((x, i) =>
      Object.is(x, b[i]) || i < 2 && Math.abs(x - b[i]) <= 1e-12 * Math.max(1, Math.abs(x), Math.abs(b[i])));
    let checked = 0;
    for (const count of [1, 2, 19, 1057]) for (const scale of [1e-150, 1, 1e100]) {
      const a = cloud(count, scale, false), b = cloud(count, scale, true);
      try {
        for (const voxel of [0.25, 1, 5].map(v => v * scale)) for (const occupancy of [0, 2, 10, 10000]) {
          const scalar = mapQuality(a, b, voxel, occupancy, false);
          const vector = mapQuality(a, b, voxel, occupancy, true);
          if (!same(scalar, vector) || !same(vector, mapQuality(a, b, voxel, occupancy)))
            throw new Error(`Voxel SIMD parity failed: ${count}, ${scale}, ${voxel}, ${occupancy}`);
          checked++;
        }
      } finally { a.free(); b.free(); }
    }
    return checked;
  }, { modulePath, wasmPath });
  expect(checked).toBe(144);
});

test("production Map quality carries voxel SIMD controls through serial and cached parallel paths", async ({ page }) => {
  await page.route("**/__voxel-controls", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__voxel-controls");
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: 'Object.defineProperty(navigator, "hardwareConcurrency", { value: 5 });\n' + await response.text() });
  });
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n))!;
  const outcomes = await page.evaluate(async workerPath => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const pending = waiting.get(data.seq)!; waiting.delete(data.seq);
      data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const pending of waiting.values()) pending.reject(new Error(event.message)); waiting.clear(); };
    const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
      const id = ++seq; waiting.set(id, { resolve: value => resolve(value as T), reject }); worker.postMessage({ seq: id, req });
    });
    const file = (n: number, lift: number) => {
      const head = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(head.length + n * 24); bytes.set(head);
      const view = new DataView(bytes.buffer, head.length);
      for (let i = 0; i < n; i++) [368_000 + i % 401 * 0.1, 3_955_000 + Math.floor(i / 401) * 0.1,
        40 + Math.sin(i % 401 / 20) * 0.1 + lift].forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      return new File([bytes], "map.ply");
    };
    const qualitySame = (a: object, b: object) => Object.keys(a).filter(key => key !== "voxelCacheHit" && key !== "voxelReference").every(key => {
      const x = (a as Record<string, number>)[key], y = (b as Record<string, number>)[key];
      return Object.is(x, y) || ["awd", "scs"].includes(key) &&
        Math.abs(x - y) <= 1e-12 * Math.max(1, Math.abs(x), Math.abs(y));
    });
    const outcomes = [];
    try {
      await call({ kind: "cpu-prepare" });
      for (const n of [1013, 200_013]) {
        const reference = await call<LoadedCloud>({ kind: "load", file: file(n + 4, 0), maxPoints: n + 4 });
        const compared = await call<LoadedCloud>({ kind: "load", file: file(n, 0.07), maxPoints: n });
        for (const [voxel, minPoints, threshold] of [[1, 10, 0.15], [0.25, 2, 0.01]]) {
          const params = { reference: reference.id, compared: compared.id, voxel, minPoints, threshold };
          const compute = (voxelSimd?: boolean, cpuCache = true, streamSummaries = true) =>
            call<C2cOutput>({ kind: "map-evaluate", voxelCache: false, voxelReferenceCache: false, ...params, voxelSimd, cpuCache, streamSummaries });
          const scalar = await compute(false);
          const bits = new Uint32Array(scalar.distances.buffer);
          for (const value of [await compute(true), await compute(), await compute(true, false, false)]) {
            const actual = new Uint32Array(value.distances.buffer);
            if (bits.length !== actual.length || !bits.every((v, i) => v === actual[i]) ||
              !Object.keys(scalar.stats).every(key => Object.is(scalar.stats[key as keyof typeof scalar.stats], value.stats[key as keyof typeof value.stats])) ||
              !qualitySame(scalar.quality!, value.quality!)) throw new Error("Combined voxel SIMD results changed");
          }
          const direct = await call<Record<string, number>>({ kind: "map-quality", voxelCache: false, voxelReferenceCache: false, ...params, voxelSimd: true });
          const control = await call<Record<string, number>>({ kind: "map-quality", voxelCache: false, voxelReferenceCache: false, ...params, voxelSimd: false });
          if (!qualitySame(direct, control) || !qualitySame(direct, scalar.quality!)) throw new Error("Standalone voxel SIMD results changed");
          outcomes.push({ n, workers: scalar.workers, cached: scalar.cpuCached });
        }
        for (const id of [reference.id, compared.id]) await call({ kind: "remove", id });
      }
      return outcomes;
    } finally { worker.terminate(); }
  }, workerPath);
  expect(outcomes).toEqual([
    { n: 1013, workers: 1, cached: false }, { n: 1013, workers: 1, cached: false },
    { n: 200_013, workers: 4, cached: true }, { n: 200_013, workers: 4, cached: true },
  ]);
});
