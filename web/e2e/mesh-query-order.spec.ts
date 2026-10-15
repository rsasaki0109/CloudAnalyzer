import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const baseline = process.env.MESH_QUERY_ORDER_BASELINE;
test("mesh query orders retain exact archived f64 results across signs, meshes, edits and budgets", async ({ page }) => {
  test.skip(!baseline, "Set MESH_QUERY_ORDER_BASELINE to an archived cached-bound production snapshot root");
  await page.route("**/__mesh-order", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__mesh-order-old.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync(resolve(baseline!, "web/src/wasm/ca_wasm.js"), "utf8") }));
  const oldWasm = readdirSync(resolve(baseline!, "dist/assets")).find(n => /^ca_wasm.*\.wasm$/.test(n))!;
  await page.route("**/__mesh-order-old.wasm", route => route.fulfill({ contentType: "application/wasm", body: readFileSync(resolve(baseline!, "dist/assets", oldWasm)) }));
  await page.route("**/__mesh-order-new.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__mesh-order");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const old = await import("/__mesh-order-old.js"), current = await import("/__mesh-order-new.js");
    await old.default({ module_or_path: "/__mesh-order-old.wasm" });
    await current.default({ module_or_path: wasmPath });
    const same = (a: Float64Array, b: Float64Array) => {
      const x = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      if (x.length !== y.length || !x.every((v, i) => v === y[i])) throw new Error("Cached-order C2M f64/sign bits changed");
    };
    let checked = 0;
    for (const mode of ["local", "utm", "tiny", "small", "large"]) {
      const scale = ({ tiny: 1e-160, small: 1e-60, large: 1e60 } as Record<string, number>)[mode] ?? 1;
      const origin = mode === "utm" ? [368000, 3955000, 40] : [0, 0, 0];
      const points = Array.from({ length: 73 }, (_, i) => [i % 9 * 0.1, Math.floor(i / 9) * 0.1, i % 9 * 0.02].map((v, a) => v * scale + origin[a]));
      const vertices = Float64Array.from(points.flat()), faces: number[][] = [];
      for (let i = 0; i + 10 < points.length; i++) if (i % 9 < 8) faces.push([i, i + 1, i + 9], [i + 9, i + 1, i + 10]);
      faces.push(...faces.slice(0, 17).map(([a, b, c]) => [c, b, a]), [0, 0, 0]);
      const indices = Uint32Array.from(faces.flat());
      const a = new old.MeshPrepared(vertices, indices), b = new current.MeshPrepared(vertices, indices);
      const reversed = new current.MeshPrepared(vertices, Uint32Array.from(faces.map(([a, b, c]) => [c, b, a]).flat()));
      try {
        for (const count of [0, 1, 3, 17, 64, 103]) for (const signed of [false, true]) {
          const queries = Float64Array.from(Array.from({ length: count }, (_, i) => {
            const p = [...points[i % points.length]]; p[2] += (i % 5 - 2) * 0.013 * scale; return p;
          }).flat());
          const cache = new current.MeshQueryOrder(8 * 1024 * 1024);
          try {
            same(a.distances(queries, signed), cache.distances(b, queries, signed));
            if (cache.hit || cache.retainedBytes !== count * 28) throw new Error("Cold order/budget changed");
            same(a.distances(queries, !signed), cache.distances(b, queries, !signed));
            if (cache.hit !== (count > 0)) throw new Error("Sign switch missed exact order");
            same(reversed.distances(queries, signed), cache.distances(reversed, queries, signed));
            if (cache.hit !== (count > 0)) throw new Error("Mesh switch missed exact query order");
            const changed = queries.slice(); if (changed.length) changed[changed.length - 1] += 0.017 * scale;
            same(a.distances(changed, signed), cache.distances(b, changed, signed));
            const rekeyed = count > 0 && changed[changed.length - 1] !== queries[queries.length - 1];
            if (cache.hit === rekeyed && count > 0) throw new Error("Changed last coordinate incorrectly reused order");
            const permuted = Float64Array.from(Array.from({ length: count }, (_, i) => [...queries.subarray((count - 1 - i) * 3, (count - i) * 3)]).flat());
            same(a.distances(permuted, signed), cache.distances(b, permuted, signed));
            if (cache.retainedBytes > 8 * 1024 * 1024 || a.retainedBytes !== b.retainedBytes) throw new Error("Retained allowances changed");
            cache.clear(); if (cache.hit || cache.retainedBytes || cache.entryCount) throw new Error("Order release failed");
            const small = new current.MeshQueryOrder(Math.max(0, count * 28 - 1));
            try { same(a.distances(queries, signed), small.distances(b, queries, signed)); if (small.hit || small.retainedBytes) throw new Error("Oversized order retained"); }
            finally { small.free(); }
            checked++;
          } finally { cache.free(); }
        }
      } finally { a.free(); b.free(); reversed.free(); }
    }
    return checked;
  }, wasmPath);
  expect(checked).toBe(60);
});

test("query-order release queues only on live lanes and preserves geometry leases", async () => {
  const { stripTypeScriptTypes } = await import("node:module");
  const source = stripTypeScriptTypes(readFileSync("src/mesh-preparation.ts", "utf8"), { mode: "transform" }).replace(/^import\b[\s\S]*?;\s*/gm, "");
  const released: { lane: number; kind: string }[] = [];
  const m = new Function("workerGeneration", "releaseOn", source.replace(/export /g, "") + "\nreturn { invalidateMeshSources };")(
    (lane: number) => [1, 4, 7].includes(lane) ? 1 : undefined,
    (lane: number, slice: { kind: string }) => { released.push({ lane, kind: slice.kind }); return Promise.resolve(true); });
  m.invalidateMeshSources({});
  expect(released).toEqual([1, 4, 7].map(lane => ({ lane, kind: "mesh-order-release" })));
});
