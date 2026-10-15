import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";

for (const slots of [false, true]) test(`GPU ${slots ? "nearest candidates" : "blocks"} leases drain failures and keep retired buffers until all users finish`, async () => {
  let generation = 1, failed = false, finish: (() => void) | undefined;
  const destroyed: number[] = [], released: number[] = [];
  class FakeBlocks {
    static serial = 0;
    readonly id = ++FakeBlocks.serial;
    bytes = 100;
    static async create() { return new FakeBlocks(); }
    find() {
      if (failed && this.id % 2) return Promise.reject(new Error("GPU readback failed"));
      if (failed) return new Promise<any>(resolve => { finish = () => resolve({ hints: new Uint32Array(2), batches: 1 }); });
      return Promise.resolve({ hints: new Uint32Array(2), batches: 1 });
    }
    dispose() { destroyed.push(this.id); }
  }
  const hooks = {
    planSor: () => ({ length: 2, indices: (i: number) => new Uint32Array([i, i + 2]), regions: () => new Float64Array(1), points: () => new Float64Array(6), free() {} }),
    workerGeneration: () => generation,
    runOn: () => Promise.resolve(100),
    runOnGeneration: (_lane: number, expected: number, s: any) => {
      if (expected !== generation) return Promise.reject(new Error("lost generation"));
      return Promise.resolve(s.kind === "knn-gpu-buffers" ? { nodes: new Uint32Array(5), queries: new Float32Array(8), points: new Float32Array(8) } : 120);
    },
    releaseOn: (_lane: number, s: any) => { released.push(s.job); return Promise.resolve(true); },
    GpuKnnBlocks: FakeBlocks, GpuKnnCandidates: FakeBlocks, knnGpuDevice: () => Promise.resolve({}),
  };
  (globalThis as any).__gpuKnnHooks = hooks;
  const source = stripTypeScriptTypes(readFileSync("src/knn-preparation.ts", "utf8"), { mode: "transform" }).replace(/^import\b[\s\S]*?;\s*/gm, "");
  const prefix = `// isolated mode ${slots}\nconst {planSor,workerGeneration,runOn,runOnGeneration,releaseOn,GpuKnnBlocks,GpuKnnCandidates,knnGpuDevice}=globalThis.__gpuKnnHooks;\n`;
  const m = await import("data:text/javascript;base64," + Buffer.from(prefix + source).toString("base64"));
  const a = { length: 200003 }, b = { length: 200007 };
  try {
    const first = m.acquireKnn(a, 2), initial = await first.gpuHints(24, slots);
    expect(initial.referenceReused).toBe(false); expect(initial.batches).toBe(2);
    expect((await first.gpuHints(25, slots)).referenceReused).toBe(true);
    expect(m.knnCacheState().bytes).toBe(264);
    const second = m.acquireKnn(b, 2); await second.gpuHints(24, slots);
    expect(destroyed).toEqual([]);
    m.invalidateKnnPreparation(a); expect(() => first.assertSource()).toThrow("coordinates changed");
    first.release(); expect(destroyed).toEqual([1, 2]);
    failed = true;
    let settled = false;
    const work = second.gpuHints(24, slots).catch(() => { settled = true; });
    await new Promise(resolve => setTimeout(resolve, 10));
    expect(finish).toBeDefined(); expect(settled).toBe(false);
    second.retire(); expect(destroyed).toEqual([1, 2]);
    finish!(); await work; second.release(); expect(destroyed).toEqual([1, 2, 3, 4]);
    expect(released).toHaveLength(4);
    failed = false;
    const lost = m.acquireKnn(a, 2); await lost.gpuHints(24, slots); generation++;
    await expect(lost.run((job: number) => ({ kind: "knn-normals", job, k: 24, orientation: new Float64Array(4) }))).rejects.toThrow("lost generation");
    lost.release(); expect(destroyed).toEqual([1, 2, 3, 4, 5, 6]);
  } finally { m.invalidateKnnPreparation(a); m.invalidateKnnPreparation(b); delete (globalThis as any).__gpuKnnHooks; }
});

const hardware = process.env.CA_GPU_KNN_HARDWARE === "1";
test.use({ channel: "chromium", launchOptions: { args: ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"] } });
for (const hintMode of ["webgpu", "webgpu-knn"]) test.describe(`physical GPU ${hintMode} hints`, () => {
  test.skip(!hardware, "Opt in with CA_GPU_KNN_HARDWARE=1 on a physical WebGPU adapter");
  test("normals/SOR match original bits and recover after a verification lane fails", async ({ page }) => {
    test.setTimeout(120_000);
    await page.route("**/__gpu-knn-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
    await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
      const response = await route.fetch();
      const prelude = `Object.defineProperty(navigator,"hardwareConcurrency",{value:3});
        const NativeWorker=Worker;
        globalThis.Worker=class extends NativeWorker { postMessage(request,options) {
          if (self.__failGpuKnn && request.kind==="knn-normals" && request.gpuHints) {
            self.__failGpuKnn=false; this.dispatchEvent(new Event("error")); return;
          }
          super.postMessage(request,options);
        }};
        self.addEventListener("message",event=>{if(event.data.failGpuKnn){event.stopImmediatePropagation();self.__failGpuKnn=true;self.postMessage({armedGpuKnn:true});}});\n`;
      await route.fulfill({ response, body: prelude + await response.text() });
    });
    await page.goto("/__gpu-knn-test");
    const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n));
    const result = await page.evaluate(async ({ workerPath, hintMode }) => {
      const worker = new Worker(workerPath, { type: "module" }), pending = new Map<number, any>(); let seq = 0, armed: (() => void) | undefined;
      worker.onmessage = ({ data }) => {
        if (data.armedGpuKnn) { armed?.(); return; }
        if (!data.response) return; const p = pending.get(data.seq); if (!p) return; pending.delete(data.seq);
        data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
      };
      worker.onerror = e => { for (const p of pending.values()) p.reject(new Error(e.message)); pending.clear(); };
      const call = (req: any): Promise<any> => new Promise((resolve, reject) => { const id = ++seq; pending.set(id, { resolve, reject }); worker.postMessage({ seq: id, req }); });
      const same = (a: ArrayBufferView, b: ArrayBufferView) => {
        const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
        return x.length === y.length && x.every((v, i) => v === y[i]);
      };
      const count = 200319, header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + count * 24), view = new DataView(bytes.buffer, header.length); bytes.set(header);
      for (let i = 0; i < count; i++) { const j = i % 100003, x = j % 701 * 0.1, y = Math.floor(j / 701) * 0.1;
        [368000 + x, 3955000 + y, 40 + Math.sin(x) * 0.1 + y * y * 0.002].forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true)); }
      try {
        await call({ kind: "cpu-prepare" });
        const cloud = await call({ kind: "load", file: new File([bytes], "gpu-knn.ply"), maxPoints: count });
        const normals = (k: number, knnHint?: string, orientation = "up") => call({ kind: "normals", id: cloud.id, k, orientation, knnHint });
        for (const k of [3, 24, 32]) for (const orientation of ["up", "outward"]) {
          const baseline = await normals(k, undefined, orientation);
          if (!same(baseline, await normals(k, "cpu", orientation)) || !same(baseline, await normals(k, hintMode, orientation))) throw new Error("Normal bits differ");
          const state = await call({ kind: "knn-hint-state" });
          if (state.backend !== hintMode || state.fallback || !state.gpuBytes) throw new Error("GPU execution did not happen: " + JSON.stringify(state));
          if (hintMode === "webgpu-knn" && state.completeRows !== count) throw new Error("Incomplete GPU candidate rows");
        }
        const filter = (knnHint?: string) => call({ kind: "filter", id: cloud.id, op: "sor", a: 24, b: 1.0, knnHint });
        const first = await filter(), expected = await call({ kind: "export", id: first.id, format: "ply" }); await call({ kind: "remove", id: first.id });
        for (const mode of ["cpu", hintMode]) {
          const x = await filter(mode); if (!same(expected, await call({ kind: "export", id: x.id, format: "ply" }))) throw new Error("SOR PLY differs");
          await call({ kind: "remove", id: x.id });
        }
        const baseline = await normals(24);
        await new Promise<void>(resolve => { armed = resolve; worker.postMessage({ failGpuKnn: true }); });
        if (!same(baseline, await normals(24, hintMode))) throw new Error("Lane-failure recovery changed normals");
        const fallback = await call({ kind: "knn-hint-state" });
        if (fallback.backend !== "cpu" || !fallback.fallback) throw new Error("Missing CPU failure recovery");
        if (!same(baseline, await normals(24, hintMode))) throw new Error("Retry changed normals");
        if ((await call({ kind: "knn-hint-state" })).backend !== hintMode) throw new Error("GPU retry failed");
        return { parity: true, recovered: true };
      } finally { worker.terminate(); }
    }, { workerPath, hintMode });
    expect(result).toEqual({ parity: true, recovered: true });
  });
});
