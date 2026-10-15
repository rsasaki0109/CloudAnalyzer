import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";

test("C2M missing hits preserve later signed queries, repeated calls and WASM ownership", async ({ page }) => {
  await page.route("**/__mesh-missing", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__mesh-missing.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.goto("/__mesh-missing");
  const wasmPath = "/assets/" + readdirSync("dist/assets").find(n => /^ca_wasm.*\.wasm$/.test(n));
  const checked = await page.evaluate(async wasmPath => {
    const wasm = await import("/__mesh-missing.js");
    await wasm.default({ module_or_path: wasmPath });
    const vertices = Float64Array.of(0,0,0, 1,0,0, 0,1,0), faces = Uint32Array.of(0,1,2);
    const mesh = new wasm.MeshPrepared(vertices, faces), cache = new wasm.MeshQueryOrder(8 * 1024 * 1024);
    const retained = mesh.retainedBytes;
    const bits = (a: Float64Array) => new BigUint64Array(a.buffer, a.byteOffset, a.length);
    const same = (a: Float64Array, b: Float64Array) => {
      const x = bits(a), y = bits(b);
      if (x.length !== y.length || !x.every((v,i) => v === y[i])) throw new Error("Missing-hit or finite distance bits changed");
    };
    const points = [[0.25,0.25,-0.5]];
    for (const value of [NaN,Infinity,-Infinity]) for (let axis = 0; axis < 3; axis++) {
      const p = [0.25,0.25,0.5]; p[axis] = value;
      points.push(p, [0.25,0.25,0.5], [0.25,0.25,-0.5]);
    }
    points.push([1e308,0.25,0.5], [0.25,0.25,0.5]);
    let checked = 0;
    try {
      for (const source of [points, [...points].reverse(), [[NaN,0,0]], [[0,0,Infinity]]]) {
        const queries = Float64Array.from(source.flat());
        for (let i = 0; i < queries.length; i++) if (Number.isNaN(queries[i])) bits(queries)[i] = 0x7ff8000000000023n;
        const before = new Uint8Array(queries.buffer).slice();
        for (const signed of [false,true,false,true]) {
          const expected = Float64Array.from(source, p => p.some(v => !Number.isFinite(v)) || p[0] === 1e308 ? Infinity : signed ? p[2] : Math.abs(p[2]));
          same(expected, mesh.distances(queries,signed));
          same(expected, wasm.meshDistances(vertices,faces,queries,signed));
          for (const candidate of [0,0xffffffff]) same(expected, mesh.distancesHinted(queries,signed,new Uint32Array(source.length).fill(candidate)));
          same(expected, cache.distances(mesh,queries,signed));
          same(expected, cache.distances(mesh,queries,signed));
          if (!cache.hit) throw new Error("Repeated nonfinite query bits did not reuse their order");
          same(Float64Array.of(-0.5), mesh.distances(Float64Array.of(0.25,0.25,-0.5),true));
          if (mesh.retainedBytes !== retained) throw new Error("Missing hits retained mesh storage");
          checked++;
        }
        if (!before.every((v,i) => v === new Uint8Array(queries.buffer)[i])) throw new Error("Caller coordinates mutated");
      }
      const invalidMesh = new wasm.MeshPrepared(new Float64Array(9).fill(NaN),faces);
      try {
        for (const signed of [false,true]) {
          const q = Float64Array.of(0,0,0, 1,1,1, NaN,0,0), expected = Float64Array.of(Infinity,Infinity,Infinity);
          same(expected,invalidMesh.distances(q,signed));
          same(expected,invalidMesh.distancesHinted(q,signed,Uint32Array.of(0,0,0xffffffff)));
          same(expected,wasm.meshDistances(new Float64Array(9).fill(NaN),faces,q,signed));
        }
      } finally { invalidMesh.free(); }
    } finally {
      // A trapped Rust call used to leave a live borrow, making free() fail.
      cache.free(); mesh.free();
    }
    return checked;
  }, wasmPath);
  expect(checked).toBe(16);
});
