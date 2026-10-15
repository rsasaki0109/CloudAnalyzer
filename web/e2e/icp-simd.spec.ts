import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";

test("ICP correspondence SIMD preserves complete fits and transformed coordinates", async ({ page }) => {
  await page.route("**/__icp-simd", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__icp-wasm.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__icp-simd");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const wasm = await import("/__icp-wasm.js");
    await wasm.default({ module_or_path: wasmPath });
    if (!wasm.c2cSimdEnabled()) throw new Error("Expected SIMD WASM");
    const bits = (a: Float64Array) => [...new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4)];
    const cloud = (xyz: Float64Array) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${xyz.length / 3}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + xyz.byteLength); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      xyz.forEach((v, i) => view.setFloat64(i * 8, v, true));
      return wasm.Cloud.parseThinned("fixture.ply", bytes, 1);
    };
    const same = (a: any, b: any) => {
      if (JSON.stringify(bits(a.matrix())) !== JSON.stringify(bits(b.matrix())) ||
        !["rmsInitial", "rmsFinal", "iterations", "converged"].every(k => Object.is(a[k], b[k])))
        throw new Error("ICP matrix/RMS/iterations/convergence changed");
    };
    let checked = 0;
    for (const count of [3, 17, 65, 257]) for (const mode of ["utm", "local", "tiny", "huge", "coincident", "nonfinite"]) {
      const positions = Array.from({ length: count }, (_, i) => {
        const j = i % 31, x = (j % 7) * 0.1, y = Math.floor(j / 7) * 0.1;
        let p = [x, y, Math.sin(x) * 0.2 + y * y * 0.03];
        if (mode === "utm") p = p.map((v, a) => v + [368000, 3955000, 40][a]);
        if (mode === "tiny") p = p.map(v => v * 1e-160);
        if (mode === "huge") p = p.map(v => v * 1e155);
        if (mode === "coincident") p = [1, -0, 0];
        if (mode === "nonfinite" && i === count - 1) p = [NaN, Infinity, -Infinity];
        return p;
      });
      if (count > 3) positions[1] = [...positions[0]];
      const reference = cloud(Float64Array.from(positions.flat()));
      const moving = cloud(Float64Array.from(positions.map(p => [p[0] + 0.01, p[1] - 0.02, p[2] + 0.025]).flat()));
      const prepared = new wasm.IcpPrepared(reference), retained = prepared.retainedBytes;
      try {
        for (const pointToPlane of [false, true]) for (const [overlap, centroids] of [[1, false], [count === 3 ? 1 : 0.8, true]] as const) {
          const outcomes = [wasm.registerIcp(moving, reference, 8, overlap, centroids, pointToPlane, false),
            wasm.registerIcp(moving, reference, 8, overlap, centroids, pointToPlane, true),
            prepared.compute(moving, 8, overlap, centroids, pointToPlane, true),
            prepared.compute(moving, 8, overlap, centroids, pointToPlane, false),
            prepared.compute(moving, 8, overlap, centroids, pointToPlane)];
          try {
            for (const outcome of outcomes.slice(1)) same(outcomes[0], outcome);
            const copies = outcomes.map(() => cloud(moving.rawPositions()));
            try {
              copies.forEach((copy, i) => copy.transform(outcomes[i].matrix()));
              const expected = JSON.stringify(bits(copies[0].rawPositions()));
              if (copies.some(copy => JSON.stringify(bits(copy.rawPositions())) !== expected)) throw new Error("Transformed coordinates changed");
            } finally { copies.forEach(copy => copy.free()); }
            if (prepared.retainedBytes !== retained) throw new Error("Retained reference capacity changed");
            checked++;
          } finally { outcomes.forEach(outcome => outcome.free()); }
        }
      } finally { prepared.free(); reference.free(); moving.free(); }
    }
    return checked;
  }, wasmPath);
  expect(checked).toBe(96);
});

test("production ICP switches paired leaves without changing reference reuse or PLY", async ({ page }) => {
  await page.goto("/");
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n));
  const result = await page.evaluate(async workerPath => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const pending = new Map<number, { resolve: (x: any) => void; reject: (e: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const p = pending.get(data.seq)!; pending.delete(data.seq);
      data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = e => { for (const p of pending.values()) p.reject(new Error(e.message)); pending.clear(); };
    const call = (req: any): Promise<any> => new Promise((resolve, reject) => {
      const id = ++seq; pending.set(id, { resolve, reject }); worker.postMessage({ seq: id, req });
    });
    const file = (lift: number) => {
      const count = 6017, header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + count * 24); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < count; i++) {
        const x = i % 80 * 0.1, y = Math.floor(i / 80) * 0.1;
        [368000 + x, 3955000 + y, 40 + Math.sin(x) * 0.2 + y * y * 0.01 + lift].forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], "correctness.ply");
    };
    try {
      const reference = await call({ kind: "load", file: file(0), maxPoints: 6017 });
      const flags: boolean[] = [], hits: boolean[] = [];
      for (const pointToPlane of [true, false]) {
        let control: any;
        for (const [icpCache, icpSimd] of [[true, false], [true, true], [true, undefined], [false, true], [false, false]] as const) {
          const moving = await call({ kind: "load", file: file(0.02), maxPoints: 6017 });
          const x = await call({ kind: "icp", reference: reference.id, moving: moving.id, maxIterations: 8,
            overlap: 0.8, matchCentroids: false, pointToPlane, icpCache, icpSimd });
          const bytes = await call({ kind: "export", id: moving.id, format: "ply" });
          if (control && (!x.matrix.every((v: number, i: number) => Object.is(v, control.x.matrix[i])) ||
            !["rmsInitial", "rmsFinal", "iterations", "converged"].every(k => Object.is(x[k], control.x[k])) ||
            bytes.length !== control.bytes.length || bytes.some((v: number, i: number) => v !== control.bytes[i]))) throw new Error("Production ICP parity failed");
          control ??= { x, bytes };
          flags.push(x.correspondenceSimd); hits.push(x.referenceReused);
          await call({ kind: "remove", id: moving.id });
        }
      }
      return { flags, hits };
    } finally { worker.terminate(); }
  }, workerPath);
  expect(result.flags).toEqual([false, true, false, true, false, false, true, false, true, false]);
  expect(result.hits).toEqual([false, true, true, false, false, false, true, true, false, false]);
});
