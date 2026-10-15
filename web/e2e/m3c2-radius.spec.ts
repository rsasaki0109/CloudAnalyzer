import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";

test("WASM radius pruning keeps raw f64 M3C2 pairs across boundaries and missing data", async ({ page }) => {
  await page.route("**/__m3c2-radius", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__m3c2-radius");
  const modulePath = "/__m3c2-radius-wasm.js";
  await page.route("**" + modulePath, route => route.fulfill({ contentType: "text/javascript",
    body: readFileSync(new URL("../src/wasm/ca_wasm.js", import.meta.url), "utf8") }));
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const result = await page.evaluate(async ({ modulePath, wasmPath }) => {
    const { default: init, m3c2Distances, Cloud, M3c2Prepared } = await import(modulePath);
    await init({ module_or_path: wasmPath });
    const sameBits = (a: Float64Array, b: Float64Array) => {
      const x = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4);
      return x.length === y.length && x.every((v, i) => v === y[i]);
    };
    let cases = 0, finite = 0, missing = 0;
    for (const count of [1, 17, 2057]) for (const offset of [0, 368_000]) {
      const cloud = (lift: number) => Float64Array.from(Array.from({ length: count }, (_, i) => {
        const x = i % 47 * 0.1, y = Math.floor(i / 47) * 0.1;
        return [offset + x, offset * 10 + y, 40 + x * 0.2 + Math.sin(y * 4) * 0.01 + lift];
      }).flat());
      const a = cloud(0), b = cloud(0.07);
      const source = (points: Float64Array) => {
        const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${points.length / 3}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
        const bytes = new Uint8Array(header.length + points.byteLength);
        bytes.set(header); bytes.set(new Uint8Array(points.buffer), header.length);
        return Cloud.parse("cloud.ply", bytes);
      };
      const reference = source(a), compared = source(b);
      const prepared = new M3c2Prepared(compared, reference);
      const snapshots = [prepared.referenceSnapshot(), prepared.comparedSnapshot()];
      const imported = M3c2Prepared.fromSnapshots(...snapshots, count, count);
      reference.free(); compared.free();
      const core = Float64Array.from([...Array.from(b).filter((_, i) => Math.floor(i / 3) % 37 === 0),
        offset + 100, offset * 10 + 100, 100, Number.NaN, Number.NaN, Number.NaN]);
      for (const [normal, projection, depth] of [[0.1, 0.1, 0.2], [0.4, 0.25, 0.5], [2, 0.1, 0.2], [Math.hypot(0.5, 2), 0.5, 2], [1, 0.5, 2]]) {
        const planes = m3c2Distances(a, b, core, normal, projection, depth, false, false, false);
        const bounded = m3c2Distances(a, b, core, normal, projection, depth, true, false);
        const defaultResult = m3c2Distances(a, b, core, normal, projection, depth);
        if (!sameBits(planes, bounded) || !sameBits(planes, defaultResult)) throw new Error("Raw f64 M3C2 result bits changed");
        for (const bounds of [false, true]) {
          for (const shared of [false, true]) for (const fused of [false, true]) {
            if (!sameBits(planes, m3c2Distances(a, b, core, normal, projection, depth, bounds, shared, fused)))
              throw new Error("Projection/search controls changed raw f64 pairs");
          }
        }
        for (const state of [prepared, imported]) for (const bounds of [false, true]) {
          if (!sameBits(planes, state.compute(core, normal, projection, depth, bounds))) throw new Error("Transferred M3C2 f64 pairs changed");
          if (!sameBits(planes, state.compute(core, normal, projection, depth, bounds, true))) throw new Error("Transferred shared-search f64 pairs changed");
          const cold = state.computeCached(core, normal, projection, depth, bounds, core.length / 3, true);
          if (state.normalCacheHit || !sameBits(planes, cold)) throw new Error("Cold fitted-normal cache changed f64 pairs");
          const expected = state.compute(core, normal, projection * 1.4, depth * 1.4, bounds);
          const warm = state.computeCached(core, normal, projection * 1.4, depth * 1.4, bounds, core.length / 3, false);
          if (!state.normalCacheHit || state.normalCacheBytes !== core.length / 3 * 56 || !sameBits(expected, warm))
            throw new Error("Cached normals changed fresh projection results or exceeded their allocation");
          const ordinary = state.computeCached(core, normal, projection * 1.4, depth * 1.4, bounds, core.length / 3, false, false);
          if (!state.normalCacheHit || !sameBits(warm, ordinary)) throw new Error("Projection control invalidated normal reuse or changed bits");
        }
        finite += bounded.filter(Number.isFinite).length; missing += bounded.filter(Number.isNaN).length;
        cases++;
      }
      let rejected = 0;
      for (const invalid of [new Uint8Array(0), snapshots[0].slice(0, -1), snapshots[0].map((v: number, i: number) => i === 0 ? 0 : v)]) {
        try { M3c2Prepared.fromSnapshots(invalid, snapshots[1], count, count).free(); } catch { rejected++; }
      }
      try { M3c2Prepared.fromSnapshots(...snapshots, count + 1, count).free(); } catch { rejected++; }
      try { imported.compute(new Float64Array(1), 0.4, 0.25, 0.5); } catch { rejected++; }
      try { imported.compute(core, 0, 0.25, 0.5); } catch { rejected++; }
      try { imported.computeCached(new Float64Array(1), 1, 0.5, 2, true, core.length / 3); } catch { rejected++; }
      try { imported.computeCached(core, 0, 0.5, 2, true, core.length / 3); } catch { rejected++; }
      if (rejected !== 8 || !sameBits(prepared.compute(core, 0.4, 0.25, 0.5), imported.compute(core, 0.4, 0.25, 0.5)))
        throw new Error("Invalid snapshots or computations corrupted prepared state");
      if (!sameBits(imported.compute(core, 1, 0.5, 2, true), imported.computeCached(core, 1, 0.5, 2, true, core.length / 3)) || !imported.normalCacheHit)
        throw new Error("Failed cached calls corrupted the last valid normal set");
      const shifted = core.slice(); shifted[0] += 0.001;
      const shiftedExpected = imported.compute(shifted, 1, 0.5, 2, true);
      if (!sameBits(shiftedExpected, imported.computeCached(shifted, 1, 0.5, 2, true, shifted.length / 3)) || imported.normalCacheHit)
        throw new Error("Changed core positions reused stale normals");
      imported.computeCached(shifted, 1, 0.5, 2, true, shifted.length / 3 - 1);
      if (imported.normalCacheHit || imported.normalCacheBytes) throw new Error("Normal cache budget was ignored");
      prepared.free(); imported.free();
    }
    return { cases, finite, missing };
  }, { modulePath, wasmPath });
  expect(result.cases).toBe(30);
  expect(result.finite).toBeGreaterThan(0);
  expect(result.missing).toBeGreaterThan(0);
});
