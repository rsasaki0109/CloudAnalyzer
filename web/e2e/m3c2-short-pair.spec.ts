import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const baseline = process.env.M3C2_SHORT_PAIR_BASELINE;
test("short paired M3C2 guard preserves adopted paired f64 results and cache capture", async ({ page }) => {
  test.skip(!baseline, "Set M3C2_SHORT_PAIR_BASELINE to a preceding production snapshot root");
  await page.route("**/__m3c2-short-pair-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__m3c2-short-pair-old.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync(resolve(baseline!, "web/src/wasm/ca_wasm.js"), "utf8") }));
  const oldWasm = readdirSync(resolve(baseline!, "dist/assets")).find(n => /^ca_wasm.*\.wasm$/.test(n))!;
  await page.route("**/__m3c2-short-pair-old.wasm", route => route.fulfill({ contentType: "application/wasm", body: readFileSync(resolve(baseline!, "dist/assets", oldWasm)) }));
  await page.route("**/__m3c2-short-pair-new.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__m3c2-short-pair-test");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const old = await import("/__m3c2-short-pair-old.js"), current = await import("/__m3c2-short-pair-new.js");
    await old.default({ module_or_path: "/__m3c2-short-pair-old.wasm" });
    await current.default({ module_or_path: wasmPath });
    if (!current.c2cSimdEnabled()) throw new Error("Expected SIMD-enabled production WASM");
    const same = (a: Float64Array, b: Float64Array) => {
      const x = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      if (x.length !== y.length || !x.every((v, i) => v === y[i])) throw new Error("M3C2 f64 output bits changed");
    };
    let checked = 0;
    for (const mode of ["local", "utm", "tiny", "large", "huge", "line", "coincident"]) {
      const scale = ({ tiny: 1e-160, large: 1e60, huge: 1e155 } as Record<string, number>)[mode] ?? 1;
      const offset = mode === "utm" ? [368000, 3955000, 40] : [0, 0, 0];
      const points = Array.from({ length: 73 }, (_, i) => {
        let p = [i % 9 * 0.1, Math.floor(i / 9) * 0.1, Math.sin(i * 0.7) * 0.01];
        if (mode === "line") p = [i * 0.1, 0, -0];
        if (mode === "coincident") p = [1, -0, 0];
        return p.map((v, a) => v * scale + offset[a]);
      });
      const bytes = (lift: number) => {
        const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
        const result = new Uint8Array(header.length + points.length * 24); result.set(header);
        const view = new DataView(result.buffer, header.length);
        points.forEach((p, i) => p.forEach((v, a) => view.setFloat64(i * 24 + a * 8, v + (a === 2 ? lift * scale : 0), true)));
        return result;
      };
      const referenceOld = old.Cloud.parse("reference.ply", bytes(0)), referenceNew = current.Cloud.parse("reference.ply", bytes(0));
      const comparedOld = old.Cloud.parse("compared.ply", bytes(0.02)), comparedNew = current.Cloud.parse("compared.ply", bytes(0.02));
      try {
        for (const count of [0, 1, 3, 17, 64]) for (const radius of [0.15, 0.4, 1]) for (const bounds of [false, true]) for (const shared of [false, true]) {
          const cores = Float64Array.from(points.slice(0, count).flat());
          if (count >= 17) cores.set([NaN, Infinity, -Infinity], (count - 1) * 3);
          const a = new old.M3c2Prepared(comparedOld, referenceOld), b = new current.M3c2Prepared(comparedNew, referenceNew);
          try {
            for (const fused of [false, true]) same(a.compute(cores, radius * scale, 0.25 * scale, 0.5 * scale, bounds, shared, fused), b.compute(cores, radius * scale, 0.25 * scale, 0.5 * scale, bounds, shared, fused));
            for (const [i, fused] of [false, true, undefined].entries()) {
              same(a.computeCached(cores, radius * scale, (0.25 + i * 0.02) * scale, 0.5 * scale, bounds, 100, shared, fused), b.computeCached(cores, radius * scale, (0.25 + i * 0.02) * scale, 0.5 * scale, bounds, 100, shared, fused));
              if (a.normalCacheHit !== b.normalCacheHit || b.normalCacheHit !== (count > 0 && i > 0) || a.normalCacheBytes !== b.normalCacheBytes || b.normalCacheBytes > 64 * 1024 * 1024) throw new Error("Normal capture/reuse/capacity changed");
            }
            checked++;
          } finally { a.free(); b.free(); }
        }
      } finally { referenceOld.free(); referenceNew.free(); comparedOld.free(); comparedNew.free(); }
    }
    return checked;
  }, wasmPath);
  expect(checked).toBe(420);
});
