import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const baseline = process.env.VOXEL_PAIR_BASELINE;
test("paired voxel cross roots preserve archived scores, odd tails and owned reference reuse", async ({ page }) => {
  test.skip(!baseline, "Set VOXEL_PAIR_BASELINE to a pre-pair production snapshot root");
  await page.route("**/__voxel-pair-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__voxel-pair-old.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync(resolve(baseline!, "web/src/wasm/ca_wasm.js"), "utf8") }));
  const oldWasm = readdirSync(resolve(baseline!, "dist/assets")).find(n => /^ca_wasm.*\.wasm$/.test(n))!;
  await page.route("**/__voxel-pair-old.wasm", route => route.fulfill({ contentType: "application/wasm", body: readFileSync(resolve(baseline!, "dist/assets", oldWasm)) }));
  await page.route("**/__voxel-pair-new.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__voxel-pair-test");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const old = await import("/__voxel-pair-old.js"), current = await import("/__voxel-pair-new.js");
    await old.default({ module_or_path: "/__voxel-pair-old.wasm" });
    await current.default({ module_or_path: wasmPath });
    if (!current.c2cSimdEnabled()) throw new Error("Expected production SIMD support");
    const fixture = (count: number, mode: string, lift: number) => {
      const scale = ({ tiny: 1e-160, large: 1e50, huge: 1e100, overflow: 1e155 } as Record<string, number>)[mode] ?? 1;
      const offset = mode === "utm" ? [368000, 3955000, 40] : [0, 0, 0];
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count * 5}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + count * 5 * 24); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < count * 5; i++) {
        const cell = Math.floor(i / 5), j = i % 5;
        let p = [cell * 2 + 0.2 + j * 0.02, 0.2 + j % 2 * 0.03 + lift, 0.3 + j * 0.01 + (cell % 3 ? j * j * 0.003 : 0)];
        if (mode === "coincident") p = [cell * 2 + 0.2, 0.2 + lift, -0];
        if (mode === "nonfinite" && cell === 0 && j === 0) p = [NaN, Infinity, -Infinity];
        p.forEach((v, a) => view.setFloat64(i * 24 + a * 8, offset[a] + v * scale, true));
      }
      return { bytes, scale };
    };
    const same = (a: Float64Array, b: Float64Array) => {
      if (a.length !== b.length || !a.every((x, i) => Object.is(x, b[i]) || i < 2 && Math.abs(x - b[i]) <= 1e-12 * Math.max(1, Math.abs(x), Math.abs(b[i]))))
        throw new Error("Archived paired/scalar voxel scores changed");
    };
    // Cloud.parse rejects empty files before the voxel kernel runs. Empty
    // Gaussian maps are exercised below via occupancy 6 and in native tests.
    let emptyRejected = 0;
    for (const module of [old, current]) {
      try { module.Cloud.parse("empty.ply", fixture(0, "local", 0).bytes).free(); }
      catch (error) { if (!String(error).includes("no points found")) throw error; emptyRejected++; }
    }
    let checked = 0;
    for (const count of [1, 2, 3, 17, 64]) for (const mode of ["local", "utm", "tiny", "large", "huge", "overflow", "coincident", "nonfinite"])
      for (const minPoints of [0, 5, 6]) for (const simd of [false, true]) {
        const { bytes, scale } = fixture(count, mode, 0);
        const a = old.Cloud.parse("reference.ply", bytes), b = current.Cloud.parse("reference.ply", bytes);
        const scalar = new old.VoxelReference(a, scale, minPoints, simd), paired = new current.VoxelReference(b, scale, minPoints, simd);
        const retained = paired.retainedBytes;
        a.free(); b.free();
        try {
          for (const lift of [0, 0.01, 0.02]) {
            const query = fixture(count, mode, lift).bytes;
            const x = old.Cloud.parse("query.ply", query), y = current.Cloud.parse("query.ply", query);
            try {
              same(scalar.compute(x), paired.compute(y));
              if (scalar.rootsComputed !== paired.rootsComputed || scalar.rootsReused !== paired.rootsReused ||
                paired.retainedBytes !== retained || scalar.retainedBytes !== retained) throw new Error("Reference capacity/root reuse changed");
              checked++;
            } finally { x.free(); y.free(); }
          }
        } finally { scalar.free(); paired.free(); }
      }
    return { checked, emptyRejected };
  }, wasmPath);
  expect(checked).toEqual({ checked: 720, emptyRejected: 2 });
});
