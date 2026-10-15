import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

// Supply a pre-pairing production snapshot to exercise an independent oracle.
const baseline = process.env.ICP_PLANE_BASELINE;
test("paired ICP matrix preserves archived scalar fits and all transformed bytes", async ({ page }) => {
  test.skip(!baseline, "Set ICP_PLANE_BASELINE to a pre-pairing production snapshot root");
  const oldWasm = readdirSync(resolve(baseline!, "dist/assets")).find(n => /^ca_wasm.*\.wasm$/.test(n))!;
  await page.route("**/__plane-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__plane-old.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync(resolve(baseline!, "web/src/wasm/ca_wasm.js"), "utf8") }));
  await page.route("**/__plane-old.wasm", route => route.fulfill({ contentType: "application/wasm", body: readFileSync(resolve(baseline!, "dist/assets", oldWasm)) }));
  await page.route("**/__plane-new.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__plane-test");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const old = await import("/__plane-old.js"), current = await import("/__plane-new.js");
    await old.default({ module_or_path: "/__plane-old.wasm" });
    await current.default({ module_or_path: wasmPath });
    if (!old.c2cSimdEnabled() || !current.c2cSimdEnabled()) throw new Error("Expected SIMD-capable builds");
    const bits = (a: Float64Array) => JSON.stringify([...new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4)]);
    const cloud = (wasm: any, xyz: Float64Array) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${xyz.length / 3}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + xyz.byteLength); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      xyz.forEach((v, i) => view.setFloat64(i * 8, v, true));
      return wasm.Cloud.parseThinned("correctness.ply", bytes, 1);
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
      const xyz = Float64Array.from(positions.flat());
      const movingXyz = Float64Array.from(positions.map(p => [p[0] + 0.01, p[1] - 0.02, p[2] + 0.025]).flat());
      const references = [cloud(old, xyz), cloud(current, xyz)];
      const movings = [cloud(old, movingXyz), cloud(current, movingXyz)];
      const prepared = new current.IcpPrepared(references[1]), retained = prepared.retainedBytes;
      try {
        for (const pointToPlane of [false, true]) for (const iterations of [0, 8])
          for (const [overlap, centroids] of [[1, false], [count === 3 ? 1 : 0.8, true]] as const) {
            const outcomes = [old.registerIcp(movings[0], references[0], iterations, overlap, centroids, pointToPlane, false, false),
              current.registerIcp(movings[1], references[1], iterations, overlap, centroids, pointToPlane, false, false),
              prepared.compute(movings[1], iterations, overlap, centroids, pointToPlane, false, false),
              prepared.compute(movings[1], iterations, overlap, centroids, pointToPlane, false, false)];
            try {
              for (const outcome of outcomes.slice(1)) if (bits(outcomes[0].matrix()) !== bits(outcome.matrix()) ||
                !["rmsInitial", "rmsFinal", "iterations", "converged"].every(k => Object.is(outcomes[0][k], outcome[k])))
                throw new Error(`Scalar/SIMD fit changed: ${count}/${mode}/${pointToPlane}/${iterations}`);
              const copies = outcomes.map((_, i) => cloud(i === 0 ? old : current, movingXyz));
              try {
                copies.forEach((copy, i) => copy.transform(outcomes[i].matrix()));
                const expected = bits(copies[0].rawPositions()), ply: Uint8Array = copies[0].export("ply");
                for (const copy of copies.slice(1)) {
                  const actual: Uint8Array = copy.export("ply");
                  if (bits(copy.rawPositions()) !== expected || actual.length !== ply.length || actual.some((v, i) => v !== ply[i]))
                    throw new Error("Transformed coordinates/PLY bytes changed");
                }
              } finally { copies.forEach(copy => copy.free()); }
              if (prepared.retainedBytes !== retained) throw new Error("Retained capacity changed");
              checked++;
            } finally { outcomes.forEach(outcome => outcome.free()); }
          }
      } finally { prepared.free(); references.forEach(c => c.free()); movings.forEach(c => c.free()); }
    }
    return checked;
  }, wasmPath);
  expect(checked).toBe(192);
});
