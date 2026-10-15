import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";

test("neighbourhood leases respect budgets, overlapping retirement, generations and drained failures", async () => {
  const jobs = new Set<string>(), released: string[] = [], generations = [1, 1];
  let plans = 0, bytes = 1000, fail = false, finish: (() => void) | undefined;
  const hooks = {
    planSor: () => { plans++; return { length: 2, indices: (j: number) => new Uint32Array([j, j + 2]), regions: () => new Float64Array([0]), points: () => new Float64Array(6), free() {} }; },
    workerGeneration: (lane: number) => generations[lane],
    runOn: (lane: number, s: any) => { jobs.add(`${lane}:${s.job}`); return Promise.resolve(bytes); },
    runOnGeneration: (lane: number, generation: number, s: any) => {
      if (generation !== generations[lane]) return Promise.reject(new Error("lost generation"));
      if (fail && lane === 0) return Promise.reject(new Error("failed search"));
      if (fail) return new Promise<Float32Array>(resolve => { finish = () => resolve(new Float32Array(3)); });
      return Promise.resolve(new Float32Array(3));
    },
    releaseOn: (lane: number, s: any) => { const key = `${lane}:${s.job}`; released.push(key); jobs.delete(key); return Promise.resolve(true); },
  };
  (globalThis as any).__knnTestHooks = hooks;
  const source = stripTypeScriptTypes(readFileSync("src/knn-preparation.ts", "utf8"), { mode: "transform" }).replace(/^import\b[\s\S]*?;\s*/gm, "");
  const prefix = "const {planSor,workerGeneration,runOn,runOnGeneration,releaseOn}=globalThis.__knnTestHooks;\n";
  const m = await import("data:text/javascript;base64," + Buffer.from(prefix + source).toString("base64"));
  const a = { length: 200003 }, b = { length: 200007 };
  const normals = (job: number) => ({ kind: "knn-normals", job, k: 12, orientation: new Float64Array([0]) });
  try {
    expect(m.acquireKnn({ length: 2_000_000 }, 8)).toBeUndefined(); expect(plans).toBe(0);
    const first = m.acquireKnn(a, 2); await first.run(normals);
    const initial = m.knnCacheState(); expect(initial.pending).toBe(false); expect(initial.bytes).toBe(2024);
    const repeat = m.acquireKnn(a, 2); await repeat.run(normals); expect(m.knnCacheState().job).toBe(initial.job); expect(m.knnCacheState().reuses).toBe(1);
    repeat.release(); repeat.release();
    const next = m.acquireKnn(b, 2); await next.run(normals);
    expect(jobs.size).toBe(4); // Retired A is still leased while B owns the idle slot.
    m.invalidateKnnPreparation(a); expect(() => first.assertSource()).toThrow("coordinates changed");
    expect(m.knnCacheState().points).toBe(b.length);
    first.release(); expect(jobs.size).toBe(2);
    next.release(); m.invalidateKnnPreparation(b); expect(jobs.size).toBe(0);

    const lost = m.acquireKnn(a, 2); await lost.run(normals); const oldId = m.knnCacheState().job;
    generations[0]++; jobs.delete(`0:${oldId}`);
    expect(m.knnCacheState()).toBeNull();
    const replacement = m.acquireKnn(a, 2); await replacement.run(normals);
    await expect(lost.run(normals)).rejects.toThrow("lost generation"); lost.release();
    expect(released).not.toContain(`0:${oldId}`); expect(jobs.size).toBe(2);
    replacement.release(); m.invalidateKnnPreparation(a); expect(jobs.size).toBe(0);

    const broken = m.acquireKnn(a, 2); fail = true;
    let settled = false;
    const work = broken.run(normals).catch(() => { settled = true; });
    await new Promise(resolve => setTimeout(resolve, 10));
    expect(settled).toBe(false); expect(finish).toBeDefined();
    broken.retire(); expect(jobs.size).toBe(2);
    finish!(); await work; broken.release(); expect(jobs.size).toBe(0);
    fail = false; bytes = 128 * 1024 * 1024;
    const oversize = m.acquireKnn(a, 2); await oversize.run(normals);
    expect(m.knnCacheState()).toBeNull(); oversize.release(); expect(jobs.size).toBe(0);
  } finally { m.invalidateKnnPreparation(a); m.invalidateKnnPreparation(b); delete (globalThis as any).__knnTestHooks; }
});

async function scenario(page: import("@playwright/test").Page, fault: "none" | "edit" | "remove" | "loss") {
  await page.route("**/__knn-cache-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    const prelude = `Object.defineProperty(navigator,"hardwareConcurrency",{value:5});
      const NativeWorker=Worker;
      let intercepted=false;
      globalThis.Worker=class extends NativeWorker {
        postMessage(request,options) {
          if (${JSON.stringify(fault)}!=="none" && !intercepted && request.kind==="knn-normals") {
            intercepted=true; self.postMessage({knnStarted:true});
            if (${JSON.stringify(fault)}==="loss") { this.dispatchEvent(new Event("error")); return; }
            setTimeout(()=>super.postMessage(request,options),80); return;
          }
          super.postMessage(request,options);
        }
      };\n`;
    await route.fulfill({ response, body: prelude + await response.text() });
  });
  await page.goto("/__knn-cache-test");
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n));
  return page.evaluate(async ({ workerPath, fault }) => {
    const worker = new Worker(workerPath, { type: "module" }); let seq = 0;
    let started!: () => void; const start = new Promise<void>(resolve => { started = resolve; });
    const pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (data.knnStarted) { started(); return; }
      if (!data.response) return;
      const p = pending.get(data.seq)!; pending.delete(data.seq); data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = e => { for (const p of pending.values()) p.reject(new Error(e.message)); pending.clear(); };
    const call = (req: any): Promise<any> => new Promise((resolve, reject) => { const id = ++seq; pending.set(id, { resolve, reject }); worker.postMessage({ seq: id, req }); });
    const count = 200319, header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
    const bytes = new Uint8Array(header.length + count * 24); bytes.set(header); const view = new DataView(bytes.buffer, header.length);
    for (let i = 0; i < count; i++) { const x = i % 801 * 0.1, y = Math.floor(i / 801) * 0.1; [368000+x,3955000+y,40+Math.sin(x)*0.1+y*y*0.002].forEach((v,a) => view.setFloat64(i*24+a*8,v,true)); }
    const file = new File([bytes], "fixture.ply"), load = () => call({ kind: "load", file, maxPoints: count });
    const state = () => call({ kind: "knn-cache-state" });
    const normals = (id: number, k: number, knnCache = true) => call({ kind: "normals", id, k, orientation: k===3?"outward":"up", knnCache });
    const same = (a: Uint8Array | Float32Array, b: Uint8Array | Float32Array) => { const x = new Uint8Array(a.buffer,a.byteOffset,a.byteLength), y = new Uint8Array(b.buffer,b.byteOffset,b.byteLength); return x.length===y.length && x.every((v,i)=>v===y[i]); };
    try {
      await call({ kind: "cpu-prepare" }); const cloud = await load();
      if (fault !== "none") {
        const work = normals(cloud.id,12).then(value => ({value,error:""}), error => ({value:null,error:error.message}));
        await start;
        if (fault === "edit") await call({kind:"transform",id:cloud.id,matrix:[1,0,0,0,0,1,0,0,0,0,1,0.03,0,0,0,1]});
        if (fault === "remove") await call({kind:"remove",id:cloud.id});
        const result = await work;
        if (fault === "loss") {
          if (result.error || await state() !== null) throw new Error("Worker loss did not recover via fresh computation");
          if (!same(result.value, await normals(cloud.id,12,false))) throw new Error("Recovered normals changed");
        } else if (!result.error.includes("coordinates changed") || await state() !== null) throw new Error("Stale source result was published");
        const target = fault === "remove" ? await load() : cloud;
        await normals(target.id,12); const rebuilt = await state();
        if (!rebuilt || rebuilt.pending || rebuilt.reuses!==0 || rebuilt.bytes>128*1024*1024) throw new Error("Preparation did not rebuild");
        await call({kind:"remove",id:target.id}); return {fault,rebuilt:true,cleared:await state()===null};
      }
      const fresh = await normals(cloud.id,12,false); if (await state()!==null) throw new Error("Fresh control cached trees");
      if (!same(fresh,await normals(cloud.id,12))) throw new Error("Prepared normal bits changed");
      const initial = await state();
      if (!initial || initial.pending || initial.reuses!==0 || initial.parts!==2 || initial.bytes>128*1024*1024) throw new Error("Invalid first-use cache state");
      const jobs = await Promise.all([normals(cloud.id,3),normals(cloud.id,12)]);
      const reused = await state(); if (reused.job!==initial.job || reused.reuses!==2 || !same(jobs[1],fresh)) throw new Error("Concurrent fits did not reuse immutable preparation");
      const filtered = await call({kind:"filter",id:cloud.id,op:"sor",a:12,b:1});
      if ((await state()).job!==initial.job) throw new Error("SOR did not share normal preparation");
      const ply = await call({kind:"export",id:filtered.id,format:"ply"}); await call({kind:"remove",id:filtered.id});
      const control = await call({kind:"filter",id:cloud.id,op:"sor",a:12,b:1,knnCache:false});
      if (!same(ply,await call({kind:"export",id:control.id,format:"ply"})) || await state()!==null) throw new Error("Fresh/prepared SOR outputs differ");
      await call({kind:"remove",id:control.id});
      if (!same(jobs[0],await normals(cloud.id,3,false))) throw new Error("Concurrent orientation fit changed");
      await normals(cloud.id,12); const beforeReplace=await state(); const second=await load();
      await normals(second.id,12); if ((await state()).job===beforeReplace.job) throw new Error("Different sources shared a preparation");
      await call({kind:"remove",id:cloud.id}); if (await state()===null) throw new Error("Removing previous source invalidated current source");
      await call({kind:"remove",id:second.id}); return {fault,reused:true,cleared:await state()===null};
    } finally { worker.terminate(); }
  }, {workerPath,fault});
}

test("normals/SOR share bounded preparation across parameters and retire controls and source replacements", async ({page}) => {
  expect(await scenario(page,"none")).toEqual({fault:"none",reused:true,cleared:true});
});
for (const fault of ["edit","remove","loss"] as const) test(`prepared jobs handle ${fault} and rebuild safely`, async ({page}) => {
  expect(await scenario(page,fault)).toEqual({fault,rebuilt:true,cleared:true});
});
