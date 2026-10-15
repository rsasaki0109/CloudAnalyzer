import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";

test("combined split bounds preserve normal bits and every SOR local/query field", async ({ page }) => {
  await page.route("**/__knn-bounds-wasm.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const wasm = await import("/__knn-bounds-wasm.js"); await wasm.default({ module_or_path: wasmPath });
    const same = (a: ArrayBufferView, b: ArrayBufferView) => {
      const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
      return x.length === y.length && x.every((v, i) => v === y[i]);
    };
    const regions = new Float64Array([0, -1e200, -1e200, -1e200, 1e200, 1e200, 1e200, 0.2, -1e200, -1e200, 1e200, 1e200, 1e200]);
    let checked = 0;
    for (const count of [0, 1, 17, 257]) for (const mode of ["local", "utm", "tiny", "huge", "coincident", "nonfinite"]) {
      const points = Float64Array.from(Array.from({ length: count }, (_, i) => {
        const j = i % 37, p = [j % 7 * 0.1, Math.floor(j / 7) * 0.1, Math.sin(j) * 0.02];
        if (mode === "utm") return p.map((v, a) => v + [368000, 3955000, 40][a]);
        if (mode === "tiny") return p.map(v => v * 1e-160);
        if (mode === "huge") return p.map(v => v * 1e155);
        if (mode === "coincident") return [1, -0, 0];
        if (mode === "nonfinite" && i % 17 === 0) return [NaN, Infinity, -Infinity];
        return p;
      }).flat());
      const part = wasm.SorPart.prepared(points), retained = part.retainedBytes;
      const queries = new Float64Array([...points.slice(0, 15), NaN, 0, 0, Infinity, -0, 1]);
      try {
        for (const k of [1, 3, 12, 24, count + 2]) {
          if (!same(part.within(queries, k), part.withinWithBounds(queries, k, false)) || !same(part.within(queries, k), part.withinWithBounds(queries, k, true))) throw new Error("Query distance bits changed");
          for (const orientation of [new Float64Array([0]), new Float64Array([1, 0, 0, 0])]) {
            const old = wasm.normalsOf(points, k, orientation);
            if (!same(old, part.normalsWithBounds(k, orientation, false)) || !same(old, part.normalsWithBounds(k, orientation, true))) throw new Error("Normal bits changed");
          }
          const old = part.local(k, 0, regions), off = part.localWithBounds(k, 0, regions, false), on = part.localWithBounds(k, 0, regions, true);
          try {
            for (const field of ["means", "open", "openPoints", "candidates", "reach"]) if (!same(old[field](), off[field]()) || !same(old[field](), on[field]())) throw new Error(`Local ${field} changed`);
          } finally { old.free(); off.free(); on.free(); }
          if (retained !== part.retainedBytes) throw new Error("Pruning retained extra capacity");
          checked++;
        }
      } finally { part.free(); }
    }
    return checked;
  }, wasmPath);
  expect(checked).toBe(120);
});
