import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const baseline = process.env.SOR_SLOT_BASELINE;
test("distance-only SOR matches archived indexed queries, all local fields and normals", async ({ page }) => {
  test.skip(!baseline, "Set SOR_SLOT_BASELINE to a pre-slot production snapshot root");
  await page.route("**/__slot-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__slot-old.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync(resolve(baseline!, "web/src/wasm/ca_wasm.js"), "utf8") }));
  const oldWasm = readdirSync(resolve(baseline!, "dist/assets")).find(n => /^ca_wasm.*\.wasm$/.test(n))!;
  await page.route("**/__slot-old.wasm", route => route.fulfill({ contentType: "application/wasm", body: readFileSync(resolve(baseline!, "dist/assets", oldWasm)) }));
  await page.route("**/__slot-new.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__slot-test");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const old = await import("/__slot-old.js"), current = await import("/__slot-new.js");
    await old.default({ module_or_path: "/__slot-old.wasm" });
    await current.default({ module_or_path: wasmPath });
    const bits = (a: Float64Array | Float32Array | Uint32Array) => JSON.stringify([...new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4)]);
    const same = (a: Float64Array | Float32Array | Uint32Array, b: Float64Array | Float32Array | Uint32Array) => {
      if (bits(a) !== bits(b)) throw new Error("Archived indexed/slot output bits changed");
    };
    const regions = new Float64Array([0, 0, 0, -1, 1, 2, 1, 1, 0, -1, 2, 2, 1]);
    let checked = 0;
    for (const count of [0, 1, 17, 257]) for (const mode of ["local", "utm", "tiny", "huge", "coincident", "nonfinite"]) {
      const xyz = Float64Array.from(Array.from({ length: count }, (_, i) => {
        const j = i % 37, x = j % 7 * 0.1, y = Math.floor(j / 7) * 0.1;
        let p = [x, y, Math.sin(j) * 0.02];
        if (mode === "utm") p = p.map((v, a) => v + [368000, 3955000, 40][a]);
        if (mode === "tiny") p = p.map(v => v * 1e-160);
        if (mode === "huge") p = p.map(v => v * 1e155);
        if (mode === "coincident") p = [1, -0, 0];
        if (mode === "nonfinite" && i === 0) p = [NaN, Infinity, -Infinity];
        return p;
      }).flat());
      const reference = old.SorPart.prepared(xyz), part = current.SorPart.prepared(xyz);
      const retained = part.retainedBytes;
      try {
        for (const k of [0, 1, 3, 12, 24, count + 2]) for (const bounds of [false, true]) {
          const a = reference.localWithBounds(k, 0, regions, bounds), b = part.localWithBounds(k, 0, regions, bounds);
          try {
            for (const field of ["means", "open", "openPoints", "candidates", "reach"]) same(a[field](), b[field]());
          } finally { a.free(); b.free(); }
          const queries = new Float64Array([0.1, 0, 0.02, NaN, 0, 0, Infinity, -0, 1, 1e155, 1e155, 1e155]);
          same(reference.withinWithBounds(queries, k, bounds), part.withinWithBounds(queries, k, bounds));
          for (const orientation of [new Float64Array([0, 0, 0, 0]), new Float64Array([1, 368000, 3955000, 40])])
            same(reference.normalsWithBounds(k, orientation, bounds), part.normalsWithBounds(k, orientation, bounds));
          if (part.retainedBytes !== retained || retained !== reference.retainedBytes) throw new Error("Capacity changed");
          checked++;
        }
      } finally { reference.free(); part.free(); }
    }
    return checked;
  }, wasmPath);
  expect(checked).toBe(288);
});
