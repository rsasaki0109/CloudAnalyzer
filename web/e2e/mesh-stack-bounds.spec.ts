import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const baseline = process.env.MESH_STACK_BOUNDS_BASELINE;
test("cached mesh traversal bounds preserve archived f64 distances, signs and repeated queries", async ({ page }) => {
  test.skip(!baseline, "Set MESH_STACK_BOUNDS_BASELINE to an archived production snapshot root");
  await page.route("**/__mesh-bounds", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__mesh-bounds-old.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync(resolve(baseline!, "web/src/wasm/ca_wasm.js"), "utf8") }));
  const oldWasm = readdirSync(resolve(baseline!, "dist/assets")).find(n => /^ca_wasm.*\.wasm$/.test(n))!;
  await page.route("**/__mesh-bounds-old.wasm", route => route.fulfill({ contentType: "application/wasm", body: readFileSync(resolve(baseline!, "dist/assets", oldWasm)) }));
  await page.route("**/__mesh-bounds-new.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__mesh-bounds");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const old = await import("/__mesh-bounds-old.js"), current = await import("/__mesh-bounds-new.js");
    await old.default({ module_or_path: "/__mesh-bounds-old.wasm" });
    await current.default({ module_or_path: wasmPath });
    const same = (a: Float64Array, b: Float64Array) => {
      const x = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      if (x.length !== y.length || !x.every((v, i) => v === y[i])) throw new Error("C2M f64 distance/sign bits changed");
    };
    let checked = 0;
    for (const mode of ["local", "utm", "tiny", "small", "large"]) {
      const scale = ({ tiny: 1e-160, small: 1e-60, large: 1e60 } as Record<string, number>)[mode] ?? 1;
      const origin = mode === "utm" ? [368000, 3955000, 40] : [0, 0, 0];
      for (const geometry of ["slope", "opposite", "line", "coincident"]) {
        const points = Array.from({ length: 73 }, (_, i) => {
          let p = [i % 9 * 0.1, Math.floor(i / 9) * 0.1, i % 9 * 0.02 + (i % 3 - 1) * 0.003];
          if (geometry === "line") p = [i * 0.1, 0, -0];
          if (geometry === "coincident") p = [1, -0, 0];
          return p.map((v, a) => v * scale + origin[a]);
        });
        const vertices = Float64Array.from(points.flat()), faces: number[][] = [];
        for (let i = 0; i + 10 < points.length; i++) if (i % 9 < 8) faces.push([i, i + 1, i + 9], [i + 9, i + 1, i + 10]);
        if (geometry === "opposite") faces.push(...faces.slice(0, 17).map(([a, b, c]) => [c, b, a]));
        faces.push([0, 0, 0], [0, 1, 0], [0, 1, 9999]); // The last face is validated out.
        const indices = Uint32Array.from(faces.flat());
        const a = new old.MeshPrepared(vertices, indices), b = new current.MeshPrepared(vertices, indices);
        try {
          if (a.retainedBytes !== b.retainedBytes) throw new Error("Retained mesh allowance changed");
          for (const count of [0, 1, 3, 17, 64, 103]) for (const signed of [false, true]) {
            const queries = Float64Array.from(Array.from({ length: count }, (_, i) => {
              const p = [...points[i % points.length]];
              p[2] += (i % 5 - 2) * 0.013 * scale;
              return p;
            }).flat());
            same(old.meshDistances(vertices, indices, queries, signed), current.meshDistances(vertices, indices, queries, signed));
            for (const q of [queries, Float64Array.from(queries, (v, i) => v + (i % 3 === 2 ? 0.017 * scale : 0)), queries]) {
              const before = a.distances(q, signed), after = b.distances(q, signed);
              same(before, after);
              same(before, old.meshDistances(vertices, indices, q, signed));
              if (a.retainedBytes !== b.retainedBytes) throw new Error("Queries retained new mesh storage");
            }
            checked++;
          }
        } finally { a.free(); b.free(); }
      }
    }
    return checked;
  }, wasmPath);
  expect(checked).toBe(240);
});
