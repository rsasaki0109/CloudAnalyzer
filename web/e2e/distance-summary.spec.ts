import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { summaryScenario } from "./distance-summary-scenario";

test("WASM median selection matches total-order sorting without changing inputs or coloring", async ({ page }) => {
  await page.route("**/__summary", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__summary");
  const modulePath = "/__summary-wasm.js";
  await page.route("**" + modulePath, route => route.fulfill({ contentType: "text/javascript",
    body: readFileSync(new URL("../src/wasm/ca_wasm.js", import.meta.url), "utf8") }));
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const count = await page.evaluate(async ({ modulePath, wasmPath }) => {
    const { default: init, summarizeDistances } = await import(modulePath);
    await init({ module_or_path: wasmPath });
    let checked = 0;
    const fields = ["count", "min", "max", "mean", "rms", "stdDev", "median"];
    const key = (bits: bigint) => bits & 0x8000_0000_0000_0000n ? ~bits & 0xffff_ffff_ffff_ffffn : bits ^ 0x8000_0000_0000_0000n;
    const check = (distances: Float64Array) => {
      const before = [...new BigUint64Array(distances.buffer, distances.byteOffset, distances.length)];
      const sorted = [...before].sort((a, b) => key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0);
      const oracle = new Float64Array(BigUint64Array.from(sorted).buffer);
      const mid = oracle.length >> 1;
      const expected = oracle.length % 2 ? oracle[mid] : 0.5 * (oracle[mid - 1] + oracle[mid]);
      const control = summarizeDistances(distances, true);
      try {
        const colors = new Uint32Array(control.distances().buffer);
        for (const flag of [undefined, false]) {
          const selected = summarizeDistances(distances, flag);
          try {
            if (fields.some(f => !Object.is(selected[f], control[f])) || !Object.is(selected.median, expected))
              throw new Error("WASM selected median/statistic mismatch");
            const result = new Uint32Array(selected.distances().buffer);
            if (result.length !== colors.length || colors.some((v, i) => v !== result[i])) throw new Error("Changed f32 coloring");
          } finally { selected.free(); }
        }
        const after = new BigUint64Array(distances.buffer, distances.byteOffset, distances.length);
        if (before.some((v, i) => v !== after[i])) throw new Error("Summary mutated input bits");
      } finally { control.free(); }
      checked++;
    };
    const extreme = new Float64Array(new BigUint64Array([
      0xfff8_0000_0000_0003n, 0xfff0_0000_0000_0000n, 0xffef_ffff_ffff_ffffn, 0x8000_0000_0000_0001n,
      0x8000_0000_0000_0000n, 0n, 1n, 0x7fef_ffff_ffff_ffffn, 0x7ff0_0000_0000_0000n,
      0x7ff0_0000_0000_0001n, 0x7ff8_0000_0000_0002n,
    ]).buffer);
    check(extreme); check(new Float64Array(extreme.buffer, 8, 9));
    for (const trap of [[1, 4, 2, 3, 5], [5, 3, 2, 4, 1], [1, 4, 2, 5], [5, 2, 4, 1]]) check(Float64Array.from(trap));
    for (const a of extreme) for (const b of extreme) { check(Float64Array.of(a, b)); check(Float64Array.of(a, b, a)); }
    let state = 0x5eed;
    for (const n of [1, 2, 3, 15, 16, 17, 257, 4095, 4096, 100_013]) {
      const values = Float64Array.from({ length: n }, () => {
        state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
        return ((state >>> 0) / 0x1_0000_0000 - 0.5) * 100;
      });
      check(values); check(Float64Array.from(values).sort()); check(Float64Array.from(values).sort().reverse());
      check(new Float64Array(n).fill(-0));
      check(Float64Array.from({ length: n }, (_, i) => i % 7));
    }
    let rejected = false;
    try { summarizeDistances(new Float64Array()); } catch (error) { rejected = String(error).includes("compared cloud is empty"); }
    if (!rejected) throw new Error("Expected empty-summary rejection");
    return checked;
  }, { modulePath, wasmPath });
  expect(count).toBeGreaterThan(250);
});

test("CPU serial and pool C2C preserve exact summaries through median-control switches", async ({ page }) => {
  const cases = await summaryScenario(page, "cpu");
  expect(cases.map(c => [c.n, c.workers])).toEqual([[1013, 1], [1014, 1], [200_013, 2], [200_014, 2]]);
});

test("signed serial and pool C2M preserve exact summaries through median-control switches", async ({ page }) => {
  const cases = await summaryScenario(page, "mesh");
  expect(cases.map(c => [c.n, c.workers, c.signed, c.crossesPlane])).toEqual([[1014, 1, true, true], [200_013, 2, true, true]]);
});
