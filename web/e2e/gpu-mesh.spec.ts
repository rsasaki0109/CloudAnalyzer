import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";

test("mesh GPU finds serialize failure cleanup and retain pending leases", async () => {
  let finish: (() => void) | undefined, finds = 0, disposed = 0, fail = true;
  class Reference {
    bytes = 64;
    static async create() { return new Reference(); }
    find() {
      finds++;
      return new Promise<any>((resolve, reject) => {
        finish = () => fail ? reject(new Error("readback failed")) : resolve({ hints: new Uint32Array(2), batches: 1, completeRows: 2 });
      });
    }
    dispose() { disposed++; }
  }
  const releases: number[] = [];
  (globalThis as any).__meshGpuHooks = {
    GpuMeshReference: Reference, meshGpuDevice: async () => ({}), workerGeneration: () => 1,
    runOn: async () => 128,
    runOnGeneration: async () => ({ origin: new Float64Array(3), nodes: new Uint32Array(8), triangles: new Float32Array(12) }),
    releaseOn: async (_lane: number, s: any) => { if (s.kind === "mesh-release") releases.push(s.job); },
  };
  const source = stripTypeScriptTypes(readFileSync("src/mesh-preparation.ts", "utf8"), { mode: "transform" }).replace(/^import\b[\s\S]*?;\s*/gm, "");
  const prefix = "const {GpuMeshReference,meshGpuDevice,workerGeneration,runOn,runOnGeneration,releaseOn}=globalThis.__meshGpuHooks;\n";
  const m = await import("data:text/javascript;base64," + Buffer.from(prefix + source).toString("base64"));
  const mesh = { vertexCount: 3, triangleCount: 1, rawVertices: () => new Float64Array(9), indices: () => new Uint32Array(3) };
  try {
    const lease = m.acquireMesh(mesh, 2), points = new Float64Array(6);
    const first = lease.gpuHints(points).catch((e: Error) => e.message), second = lease.gpuHints(points);
    await new Promise(resolve => setTimeout(resolve, 10));
    expect(finds).toBe(1); expect(disposed).toBe(0);
    finish!(); expect(await first).toBe("readback failed"); fail = false;
    await new Promise(resolve => setTimeout(resolve, 10));
    expect(finds).toBe(2); expect(disposed).toBe(1);
    lease.retire(); lease.release(); expect(releases).toHaveLength(0);
    finish!(); expect((await second).completeRows).toBe(2);
    expect(disposed).toBe(2); expect(releases).toHaveLength(2);
    const edited = m.acquireMesh(mesh, 2), work = edited.gpuHints(points).catch((e: Error) => e.message);
    await new Promise(resolve => setTimeout(resolve, 10));
    m.invalidateMeshSources(mesh); edited.release(); expect(disposed).toBe(2);
    finish!(); expect(await work).toContain("source changed"); expect(disposed).toBe(3);
  } finally { m.invalidateMeshSources(mesh); delete (globalThis as any).__meshGpuHooks; }
});

test("mesh GPU quota rejects overlapping allocations and rolls back partial references", async () => {
  const live = new Set<{ size: number; destroy(): void }>(), mib = 1024*1024;
  const runtime = {
    device: {
      limits: { maxBufferSize: 128*mib, maxStorageBufferBindingSize: 128*mib },
      createShaderModule: () => ({}), createComputePipelineAsync: async () => ({}),
      createBuffer: ({ size }: { size: number }) => {
        const b = { size, destroy() { live.delete(b); } }; live.add(b); return b;
      },
      queue: { writeBuffer() {} },
    },
    run: async (work: () => Promise<unknown>) => work(),
  };
  (globalThis as any).__meshQuotaHooks = { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, UNIFORM: 4 } };
  const source = stripTypeScriptTypes(readFileSync("src/gpu-mesh-candidates.ts", "utf8"), { mode: "transform" }).replace(/^import\b[\s\S]*?;\s*/gm, "");
  const prefix = "const {GPUBufferUsage}=globalThis.__meshQuotaHooks;\n";
  const m = await import("data:text/javascript;base64," + Buffer.from(prefix + source).toString("base64"));
  // Virtual snapshots/buffers exercise aggregate accounting without reserving
  // hundreds of MiB in this unit test. Production snapshots are typed arrays.
  const snapshot = (byteLength: number) => ({ byteLength, length: byteLength/4 });
  const make = (nodeBytes: number, triangleBytes: number) => m.GpuMeshReference.create(runtime, new Float64Array(3), snapshot(nodeBytes), snapshot(triangleBytes));
  let first: any, second: any;
  try {
    first = await make(128*mib, 96*mib);
    await expect(make(16*mib, 63*mib)).rejects.toThrow("budget exceeded");
    expect(live.size).toBe(3);
    second = await make(24*mib, 48); expect(live.size).toBe(6);
    second.dispose(); second.dispose(); expect(live.size).toBe(3);
    await expect(make(128*mib+32, 48)).rejects.toThrow("device limits");
    first.dispose(); expect(live.size).toBe(0);
    first = await make(128*mib, 96*mib); expect(live.size).toBe(3);
  } finally { first?.dispose(); second?.dispose(); delete (globalThis as any).__meshQuotaHooks; }
  expect(live.size).toBe(0);
});

test.use({ channel: "chromium", launchOptions: { args: ["--enable-unsafe-webgpu", "--force-high-performance-gpu", "--disable-software-rasterizer"] } });
test.describe("physical GPU mesh triangle hints", () => {
  test.skip(process.env.CA_GPU_MESH_HARDWARE !== "1", "Opt in with CA_GPU_MESH_HARDWARE=1 on a physical WebGPU adapter");
  for (const fault of ["none", "edit", "remove-mesh"] as const) test(`mesh hints preserve output and handle ${fault}`, async ({ page }) => {
    test.setTimeout(120_000);
    await page.route("**/__gpu-mesh-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
    await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
      const response = await route.fetch();
      const prelude = `Object.defineProperty(navigator,"hardwareConcurrency",{value:3});
        const NativeWorker=Worker;
        globalThis.Worker=class extends NativeWorker { postMessage(request,options) {
          if(self.__meshFault==="loss" && request.kind==="mesh-prepared" && request.hints){self.__meshFault="";this.dispatchEvent(new Event("error"));return;}
          super.postMessage(request,options);
        }};
        if(typeof GPUBuffer!=="undefined"){
          const map=GPUBuffer.prototype.mapAsync;
          GPUBuffer.prototype.mapAsync=function(...args){
            const fault=self.__meshFault;
            if(fault==="readback"){self.__meshFault="";return Promise.reject(new Error("injected readback failure"));}
            const result=map.apply(this,args);
            if(fault==="edit"||fault==="remove-mesh"){self.__meshFault="";self.postMessage({meshStarted:true});return result.then(()=>new Promise(resolve=>setTimeout(resolve,100)));}
            return result;
          };
        }
        self.addEventListener("message",event=>{if(event.data.meshFault){event.stopImmediatePropagation();self.__meshFault=event.data.meshFault;self.postMessage({armedMesh:true});}});\n`;
      await route.fulfill({ response, body: prelude + await response.text() });
    });
    await page.goto("/__gpu-mesh-test");
    const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n));
    const result = await page.evaluate(async ({ workerPath, fault }) => {
      const worker = new Worker(workerPath, { type: "module" }), pending = new Map<number, any>();
      let seq = 0, armed: (() => void) | undefined, started: (() => void) | undefined;
      worker.onmessage = ({ data }) => {
        if (data.armedMesh) { armed?.(); return; } if (data.meshStarted) { started?.(); return; }
        if (!data.response) return; const p = pending.get(data.seq); if (!p) return; pending.delete(data.seq);
        data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
      };
      worker.onerror = e => { for (const p of pending.values()) p.reject(new Error(e.message)); pending.clear(); };
      const call = (req: any): Promise<any> => new Promise((resolve, reject) => { const id = ++seq; pending.set(id, { resolve, reject }); worker.postMessage({ seq: id, req }); });
      const arm = (meshFault: string) => new Promise<void>(resolve => { armed = resolve; worker.postMessage({ meshFault }); });
      const same = (a: ArrayBufferView, b: ArrayBufferView) => {
        const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
        return x.length === y.length && x.every((v, i) => v === y[i]);
      };
      const count = 200319, header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + count * 24), view = new DataView(bytes.buffer, header.length); bytes.set(header);
      for (let i = 0; i < count; i++) [368000 + (i % 701) * 0.01, 3955000 + Math.floor(i / 701) * 0.01, 40 + (i % 5 - 2) * 0.001].forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      const obj: string[] = [];
      for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) obj.push(`v ${368000+x*0.25} ${3955000+y*0.25} 40`);
      for (let y = 0; y < 31; y++) for (let x = 0; x < 31; x++) {
        const a = y*32+x+1, b = a+1, c = a+32, d = c+1;
        obj.push(`f ${a} ${b} ${c}`, `f ${c} ${b} ${a}`, `f ${c} ${b} ${d}`, `f ${d} ${b} ${c}`);
      }
      try {
        await call({ kind: "cpu-prepare" });
        const cloud = await call({ kind: "load", file: new File([bytes], "queries.ply"), maxPoints: count });
        let mesh = await call({ kind: "load", file: new File([obj.join("\n")], "surface.obj"), maxPoints: count });
        const distance = (gpu: boolean, signed = true) => call({ kind: "c2c", compared: cloud.id, reference: mesh.id, signed, meshGpu: gpu });
        const expected = await distance(false);
        const check = (r: any, control = expected) => {
          if (!same(r.distances, control.distances) || Object.keys(control.stats).some(k => !Object.is(r.stats[k], control.stats[k]))) throw new Error("C2M bits/summary changed");
        };
        const gpuCheck = (r: any) => { if (r.meshCompute?.backend !== "webgpu" || r.meshCompute.fallback || r.meshCompute.completeRows !== count || r.meshCompute.batches !== 4 || !r.meshCached) throw new Error("Expected real GPU mesh dispatch: " + JSON.stringify(r.meshCompute)); };
        const first = await distance(true); check(first); gpuCheck(first);
        if (first.meshCompute.referenceReused) throw new Error("Unexpected cold GPU reuse");
        if (fault !== "none") {
          await arm(fault); const start = new Promise<void>(resolve => { started = resolve; });
          const work = distance(true).then(() => "published", (e: Error) => e.message); await start;
          if (fault === "edit") await call({ kind: "transform", id: cloud.id, matrix: [1,0,0,0,0,1,0,0,0,0,1,0.01,0,0,0,1] });
          else await call({ kind: "remove", id: mesh.id });
          if (!(await work).includes("source changed")) throw new Error("Stale mesh GPU result published");
          if (fault === "remove-mesh") mesh = await call({ kind: "load", file: new File([obj.join("\n")], "retry.obj"), maxPoints: count });
          const control = await distance(false), retry = await distance(true); check(retry, control); gpuCheck(retry);
          return { fault, matched: true };
        }
        for (const r of await Promise.all([distance(true), distance(true)])) { check(r); gpuCheck(r); if (!r.meshCompute.referenceReused) throw new Error("GPU mesh reuse lost"); }
        const control = await distance(false, false), unsigned = await distance(true, false); check(unsigned, control); gpuCheck(unsigned);
        const ply = (r: any) => call({ kind: "export", id: cloud.id, format: "ply", scalar: { name: "C2M", values: r.distances } });
        if (!same(await ply(unsigned), await ply(control))) throw new Error("Scalar PLY changed");
        for (const injected of ["readback", "loss"]) {
          await arm(injected); const recovered = await distance(true); check(recovered);
          if (recovered.meshCompute?.backend !== "cpu" || !recovered.meshCompute.fallback) throw new Error("Mesh GPU failure did not recover");
          const retry = await distance(true); check(retry); gpuCheck(retry);
        }
        return { fault, matched: true };
      } finally { worker.terminate(); }
    }, { workerPath, fault });
    expect(result).toEqual({ fault, matched: true });
  });
});
