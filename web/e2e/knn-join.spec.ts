import { expect, test } from "@playwright/test";
import { finishSorMeans, makeSorQueries, scatterNormalParts } from "../src/knn-join";
import type { SorLocalResult } from "../src/c2c-worker";
import { readFileSync, readdirSync } from "node:fs";

const bits = (v: Float32Array | Float64Array) => Array.from(new Uint32Array(v.buffer, v.byteOffset, v.byteLength / 4));

test("normal scatter preserves arbitrary f32 bits, offsets and original point order", () => {
  const raw = new Uint32Array([0, 0x80000000, 0x7fc01234, 0x7f801234, 0xff800000, 0x3f800000, 0x7fc05678, 0x40000000, 0x40400000, 0]);
  const a = new Float32Array(raw.buffer, 4, 6), b = new Float32Array(raw.buffer, 0, 3);
  const parts = [a, new Float32Array(0), b], indices = [new Uint32Array([2, 0]), new Uint32Array(0), new Uint32Array([1])];
  const expected = new Float32Array(12);
  parts.forEach((part, j) => indices[j].forEach((idx, i) => expected.set(part.subarray(i * 3, i * 3 + 3), idx * 3)));
  expect(bits(scatterNormalParts(parts, indices, 4))).toEqual(bits(expected));
});

/** The previous production join, kept independently as the precision control. */
function allocatingJoin(local: SorLocalResult[], indices: Uint32Array[], count: number, k: number) {
  const means = new Float64Array(count), lists: Float64Array[][] = [], openIndex: number[] = [];
  const asked: { query: number; at: number[] }[][] = Array.from({ length: local.length }, () => []);
  local.forEach((part, i) => {
    for (let j = 0; j < part.means.length; j++) if (!Number.isNaN(part.means[j])) means[indices[i][j]] = part.means[j];
    for (let o = 0; o < part.open.length; o++) {
      const q = lists.length;
      lists.push([part.candidates.subarray(o * (k + 1), (o + 1) * (k + 1))]);
      openIndex.push(indices[i][part.open[o]]);
      const at = Array.from(part.openPoints.subarray(o * 3, o * 3 + 3));
      for (let j = 0; j < local.length; j++) if ((part.reach[o] >>> j) & 1) asked[j].push({ query: q, at });
    }
  });
  const queries = asked.map(a => Float64Array.from(a.flatMap(x => x.at)));
  return { queries, finish(answers: (Float64Array | null)[]) {
    answers.forEach((d, j) => { if (d) asked[j].forEach((a, row) => lists[a.query].push(d.subarray(row * (k + 1), (row + 1) * (k + 1)))); });
    lists.forEach((candidates, q) => {
      const all = new Float64Array(candidates.reduce((s, c) => s + c.length, 0));
      let at = 0;
      for (const c of candidates) { all.set(c, at); at += c.length; }
      all.sort();
      let sum = 0;
      for (let j = 1; j <= k; j++) sum += Math.sqrt(all[j]);
      means[openIndex[q]] = sum / k;
    });
    return means;
  } };
}

test("packed SOR queries and reusable merges match allocating joins for boundaries, ties and nonfinite distances", () => {
  const distances = [0, -0, 0.001, 1, 1, 1e-310, 1e300, Infinity, NaN, -Infinity];
  let checked = 0;
  for (const parts of [1, 2, 4, 8]) for (const k of [1, 3, 12, 24]) for (const mode of ["closed", "mixed", "all", "nonfinite"]) {
    const indices = Array.from({ length: parts }, (_, j) => Uint32Array.from({ length: 17 }, (_, i) => i * parts + j));
    const local = indices.map((idx, j): SorLocalResult => {
      const open = Uint32Array.from(Array.from({ length: idx.length }, (_, i) => i).filter(i => mode !== "closed" && (mode !== "mixed" || i % 3 === 0)));
      return {
        means: Float64Array.from(idx, (_, i) => open.includes(i) ? NaN : (i + j) / 13), open,
        openPoints: new Float64Array(open.length * 3),
        candidates: Float64Array.from({ length: open.length * (k + 1) }, (_, i) => mode === "nonfinite" ? distances[(i + j) % distances.length] : (i % (k + 1)) / 7),
        reach: Uint32Array.from(open, i => mode === "mixed" ? ((1 << ((i + j + 1) % parts)) & ~(1 << j)) : ((1 << parts) - 1) & ~(1 << j)),
      };
    });
    local.forEach((part, j) => {
      part.openPoints = Float64Array.from(Array.from(part.open, i => [368000 + i, 3955000 + j, i === 0 ? -0 : 40 + j]).flat());
    });
    const legacy = allocatingJoin(local, indices, parts * 17, k), packed = makeSorQueries(local, indices, parts * 17);
    expect(packed.queries.map(bits)).toEqual(legacy.queries.map(bits));
    const answers = packed.queries.map((q, j) => q.length ? Float64Array.from({ length: q.length / 3 * (k + 1) }, (_, i) => mode === "nonfinite" ? distances[(i + j * 3) % distances.length] : i % (k + 1) / 11) : null);
    // Sending query buffers detaches them; the row mapping remains sufficient.
    packed.queries.forEach(q => structuredClone(q, { transfer: [q.buffer] }));
    expect(bits(finishSorMeans(local, answers, packed, k))).toEqual(bits(legacy.finish(answers)));
    checked++;
  }
  expect(checked).toBe(64);
});

test("SOR sorted-prefix merge preserves full-sort means and falls back for unusual rows", () => {
  let checked = 0, seed = 0x73102491;
  const random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; };
  for (const parts of [1, 2, 4, 8]) for (const k of [1, 3, 12, 24, 64])
    for (const mode of ["sorted", "infinity", "unsorted", "nan", "negative", "signed-zero", "short", "tiny-huge"]) {
      const width = k + 1, indices = Array.from({ length: parts }, (_, j) => Uint32Array.from({ length: 9 }, (_, i) => i * parts + j));
      const rows = (count: number) => {
        const backing = new Float64Array(count * width + 2), values = new Float64Array(backing.buffer, 8, count * width);
        for (let row = 0; row < count; row++) {
          for (let i = 0; i < width; i++) values[row * width + i] = i === 0 ? 0 : Math.floor(random() * 12) / 7;
          values.subarray(row * width, (row + 1) * width).sort();
          if (mode === "infinity" && row % 3 === 0) values.fill(Infinity, row * width + 1, (row + 1) * width);
          if (mode === "tiny-huge") for (let i = 1; i < width; i++) values[row * width + i] *= row % 2 ? 1e-310 : 1e300;
        }
        return values;
      };
      const local = indices.map((idx, j): SorLocalResult => ({
        means: new Float64Array(idx.length).fill(NaN), open: Uint32Array.from(idx, (_, i) => i),
        openPoints: new Float64Array(idx.length * 3), candidates: rows(idx.length),
        reach: Uint32Array.from(idx, (_, i) => i % 3 === 0 ? 0 : ((1 << parts) - 1) & ~(1 << j)),
      }));
      const make = () => makeSorQueries(local, indices, parts * 9);
      const base = make(), answers = base.queries.map(q => q.length ? rows(q.length / 3) : null);
      // Corrupt a later reached lane, after some prefix merges have succeeded.
      const unusual = [...answers].reverse().find(a => a && a.length > width) ?? local[parts - 1].candidates;
      if (mode === "unsorted") { unusual[width] = 2; unusual[width + 1] = 1; }
      if (mode === "nan") {
        const raw = new Uint32Array(unusual.buffer, unusual.byteOffset, unusual.byteLength / 4);
        raw[width * 2] = 0x12345678; raw[width * 2 + 1] = 0x7ff80000;
      }
      if (mode === "negative") unusual[width] = -1;
      if (mode === "signed-zero") unusual[width] = -0;
      if (mode === "short") {
        const lane = answers.findIndex(a => a === unusual);
        if (lane >= 0) answers[lane] = unusual.subarray(0, unusual.length - 1);
        else local[parts - 1].candidates = unusual.subarray(0, unusual.length - 1);
      }
      const expected = bits(finishSorMeans(local, answers, make(), k, false));
      const inputBits = local.map(p => bits(p.candidates)), answerBits = answers.map(a => a && bits(a));
      for (let repeat = 0; repeat < 3; repeat++) {
        const packed = make(); packed.queries.forEach(q => structuredClone(q, { transfer: [q.buffer] }));
        expect(bits(finishSorMeans(local, answers, packed, k, true))).toEqual(expected);
      }
      expect(local.map(p => bits(p.candidates))).toEqual(inputBits);
      expect(answers.map(a => a && bits(a))).toEqual(answerBits);
      checked++;
    }
  expect(checked).toBe(160);
});

test("parallel normals and SOR exports preserve the WASM partition/global controls", async ({ page }) => {
  test.setTimeout(120_000);
  await page.route("**/__knn-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route("**/__knn-wasm.js", route => route.fulfill({ contentType: "text/javascript", body: readFileSync("src/wasm/ca_wasm.js", "utf8") }));
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: 'Object.defineProperty(navigator, "hardwareConcurrency", { value: 5 });\n' + await response.text() });
  });
  await page.goto("/__knn-test");
  const names = readdirSync("dist/assets");
  const result = await page.evaluate(async ({ workerPath, wasmPath }) => {
    const wasm = await import("/__knn-wasm.js");
    await wasm.default({ module_or_path: wasmPath });
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const p = waiting.get(data.seq)!; waiting.delete(data.seq);
      data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = e => { for (const p of waiting.values()) p.reject(new Error(e.message)); waiting.clear(); };
    const call = (req: any): Promise<any> => new Promise((resolve, reject) => {
      const id = ++seq; waiting.set(id, { resolve, reject }); worker.postMessage({ seq: id, req });
    });
    const same = (a: Float32Array | Uint8Array, b: Float32Array | Uint8Array) => {
      const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
      return x.length === y.length && x.every((v, i) => v === y[i]);
    };
    const count = 200319, header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
    const bytes = new Uint8Array(header.length + count * 24); bytes.set(header);
    const view = new DataView(bytes.buffer, header.length);
    for (let i = 0; i < count; i++) {
      const j = i - i % 2, x = (j % 801) * 0.1, y = Math.floor(j / 801) * 0.1;
      const xyz = i === count - 1 ? [369000, 3956000, 200] : [368000 + x, 3955000 + y, 40 + Math.sin(x) * 0.1 + y * y * 0.002];
      xyz.forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
    }
    const cloud = wasm.Cloud.parse("fixture.ply", bytes);
    let plan;
    try {
      cloud.buildIndex();
      plan = wasm.planSor(cloud, 2);
      const indices = Array.from({ length: plan.length }, (_, j) => plan.indices(j));
      const points = Array.from({ length: plan.length }, (_, j) => plan.points(cloud, j));
      const loaded = await call({ kind: "load", file: new File([bytes], "fixture.ply"), maxPoints: count });
      let normalCases = 0, sorCases = 0;
      for (const k of [3, 12]) for (const orientation of ["up", "outward"]) {
        const expected = new Float32Array(count * 3), how = new Float64Array(cloud.normalsOrientation(orientation));
        points.forEach((p, j) => {
          const part = wasm.normalsOf(p, k, how);
          for (let i = 0; i < indices[j].length; i++) expected.set(part.subarray(i * 3, i * 3 + 3), indices[j][i] * 3);
        });
        const actual = await call({ kind: "normals", id: loaded.id, k, orientation });
        if (!same(expected, actual)) throw new Error(`Normal bits changed: ${k}/${orientation}`);
        cloud.setNormals(expected);
        if (!same(cloud.export("ply"), await call({ kind: "export", id: loaded.id, format: "ply" }))) throw new Error("Normal export changed");
        normalCases++;
      }
      for (const k of [1, 12]) for (const ratio of [0.5, 2]) {
        const control = cloud.filter("sor", k, ratio);
        try {
          control.buildIndex();
          const filtered = await call({ kind: "filter", id: loaded.id, op: "sor", a: k, b: ratio });
          if (!same(control.export("ply"), await call({ kind: "export", id: filtered.id, format: "ply" }))) throw new Error(`SOR export changed: ${k}/${ratio}`);
          await call({ kind: "remove", id: filtered.id }); sorCases++;
        } finally { control.free(); }
      }
      await call({ kind: "remove", id: loaded.id });
      return { normalCases, sorCases, parts: plan.length };
    } finally { plan?.free(); cloud.free(); worker.terminate(); }
  }, { workerPath: "/assets/" + names.find(n => /^worker-.*\.js$/.test(n)), wasmPath: "/assets/" + names.find(n => /^ca_wasm.*\.wasm$/.test(n)) });
  expect(result).toEqual({ normalCases: 4, sorCases: 4, parts: 2 });
});
