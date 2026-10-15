import { expect, test, type Page } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";

test("mesh leases bound reservations and drain failures without releasing active BVHs", async () => {
  let generation = 1, fail = false, finish: (() => void) | undefined, actualBytes = 100;
  const releases: number[] = [], prepared: number[] = [];
  const hooks = {
    runOn: (_lane: number, s: any) => { prepared.push(s.job); return Promise.resolve(actualBytes); },
    workerGeneration: () => generation,
    runOnGeneration: (lane: number, expected: number, s: any) => {
      if (expected !== generation) return Promise.reject(new Error("lost generation"));
      if (fail && lane === 0) return Promise.reject(new Error("failed mesh query"));
      if (fail && lane === 1) return new Promise(resolve => { finish = () => resolve({ distances: new Float64Array(2), queryOrderHit: false, queryOrderBytes: 0, queryOrderEntries: 0 }); });
      return Promise.resolve({ distances: new Float64Array(s.queries.length / 3), queryOrderHit: false, queryOrderBytes: 0, queryOrderEntries: 0 });
    },
    releaseOn: (_lane: number, s: any) => { if (s.kind === "mesh-release") releases.push(s.job); return Promise.resolve(true); },
  };
  (globalThis as any).__meshHooks = hooks;
  const source = stripTypeScriptTypes(readFileSync("src/mesh-preparation.ts", "utf8"), { mode: "transform" }).replace(/^import\b[\s\S]*?;\s*/gm, "");
  const prefix = "const {runOn,runOnGeneration,releaseOn,workerGeneration}=globalThis.__meshHooks;\n";
  const m = await import("data:text/javascript;base64," + Buffer.from(prefix + source).toString("base64"));
  const mesh = (triangles = 100) => ({ vertexCount: 100, triangleCount: triangles, rawVertices: () => new Float64Array(300), indices: () => new Uint32Array(300) });
  const a = mesh(), b = mesh(), queries = () => new Float64Array(6);
  try {
    const first = m.acquireMesh(a, 4); await first.run(8, queries, true);
    expect(m.meshCacheState().bytes).toBe(400);
    const again = m.acquireMesh(a, 4); expect(again.reused).toBe(true); again.release();
    const guard = m.trackMeshSources(a, {}), second = m.acquireMesh(b, 2); await second.run(2, queries, false);
    expect(releases).toEqual([]);
    m.invalidateMeshSources(a); expect(() => guard.assertSource()).toThrow("source changed"); guard.release();
    first.release(); expect(releases).toHaveLength(4);
    fail = true; let settled = false;
    const pending = second.run(2, queries, false).catch(() => { settled = true; });
    await new Promise(resolve => setTimeout(resolve, 10));
    expect(finish).toBeDefined(); expect(settled).toBe(false);
    second.retire(); expect(releases).toHaveLength(4); finish!(); await pending; second.release(); expect(releases).toHaveLength(6);
    fail = false;
    const lost = m.acquireMesh(a, 2); await lost.run(2, queries, true); generation++;
    await expect(lost.run(2, queries, true)).rejects.toThrow("lost generation"); lost.release();
    const large = mesh(5_000_000), other = mesh(2_500_000), lease = m.acquireMesh(large, 8); await lease.run(2, queries, false);
    expect(lease.workers).toBe(1); expect(m.meshCacheState().reservedTotal).toBeLessThanOrEqual(1024*1024*1024);
    const before = prepared.length; expect(m.acquireMesh(other, 8)).toBeUndefined(); expect(prepared).toHaveLength(before);
    lease.release();
    actualBytes = 1024*1024*1024;
    const over = m.acquireMesh(a,2); await expect(over.run(2,queries,false)).rejects.toThrow("reserved memory"); over.release();
    expect(m.meshCacheState()).toBeNull();
    expect(m.acquireMesh(mesh(8_000_000),8)).toBeUndefined();
  } finally { m.invalidateMeshSources(a); m.invalidateMeshSources(b); delete (globalThis as any).__meshHooks; }
});

async function scenario(page: Page, fault: "none" | "edit" | "remove-query" | "remove-mesh" | "loss") {
  const orderEnabled = readFileSync("src/c2c-worker.ts", "utf8").includes("const USE_MESH_QUERY_ORDER = true;");
  await page.route("**/__mesh-cache-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    const prelude = `Object.defineProperty(navigator,"hardwareConcurrency",{value:5});
      const NativeWorker=Worker; let intercepted=false;
      globalThis.Worker=class extends NativeWorker { postMessage(request,options) {
        if (${JSON.stringify(fault)}!=="none" && !intercepted && request.kind==="mesh-prepared") {
          intercepted=true; self.postMessage({meshStarted:true});
          if (${JSON.stringify(fault)}==="loss") { this.dispatchEvent(new Event("error")); return; }
          setTimeout(()=>super.postMessage(request,options),100); return;
        }
        super.postMessage(request,options);
      }};\n`;
    await route.fulfill({ response, body: prelude + await response.text() });
  });
  await page.goto("/__mesh-cache-test");
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n));
  return page.evaluate(async ({ workerPath, fault, orderEnabled }) => {
    const worker = new Worker(workerPath, { type: "module" }); let seq = 0, started!: () => void;
    const start = new Promise<void>(resolve => { started = resolve; }), pending = new Map<number, any>();
    worker.onmessage = ({data}) => {
      if (data.meshStarted) { started(); return; }
      if (!data.response) return; const p=pending.get(data.seq); if(!p)return;pending.delete(data.seq);
      data.response.ok?p.resolve(data.response.value):p.reject(new Error(data.response.error));
    };
    worker.onerror=e=>{for(const p of pending.values())p.reject(new Error(e.message));pending.clear();};
    const call=(req:any):Promise<any>=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});worker.postMessage({seq:id,req});});
    const count=200319, header=new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
    const bytes=new Uint8Array(header.length+count*24);bytes.set(header);const view=new DataView(bytes.buffer,header.length);
    for(let i=0;i<count;i++) [368000+(i%701)*0.01,3955000+Math.floor(i/701)*0.01,40+(i%5-2)*0.001].forEach((v,a)=>view.setFloat64(i*24+a*8,v,true));
    const file=new File([bytes],"queries.ply");
    const meshFile=new File([`v 368000 3955000 40\nv 368008 3955000 40\nv 368000 3955008 40\nf 1 2 3\nf 3 2 1\n`],"surface.obj");
    const load=()=>call({kind:"load",file,maxPoints:count}), loadMesh=()=>call({kind:"load",file:meshFile,maxPoints:count});
    const same=(a:ArrayBufferView,b:ArrayBufferView)=>{const x=new Uint8Array(a.buffer,a.byteOffset,a.byteLength),y=new Uint8Array(b.buffer,b.byteOffset,b.byteLength);return x.length===y.length&&x.every((v,i)=>v===y[i]);};
    try {
      await call({kind:"cpu-prepare"});let cloud=await load(),mesh=await loadMesh();
      const state=()=>call({kind:"mesh-cache-state"});
      const distance=(meshCache=true,signed=true)=>call({kind:"c2c",compared:cloud.id,reference:mesh.id,signed,meshCache});
      const expected=await distance(false);
      if(fault!=="none") {
        const work=distance().then(value=>({value,error:""}),error=>({value:null,error:error.message}));await start;
        if(fault==="edit")await call({kind:"transform",id:cloud.id,matrix:[1,0,0,0,0,1,0,0,0,0,1,0.01,0,0,0,1]});
        if(fault==="remove-query")await call({kind:"remove",id:cloud.id});
        if(fault==="remove-mesh")await call({kind:"remove",id:mesh.id});
        const result=await work;
        if(fault==="loss") {
          if(result.error||result.value.meshCached||!result.value.meshCacheFallback||!same(expected.distances,result.value.distances))throw new Error("Mesh lane failure did not recover exactly");
        }else if(!result.error.includes("source changed"))throw new Error("Stale C2M result was published");
        if(fault==="remove-query")cloud=await load();if(fault==="remove-mesh")mesh=await loadMesh();
        const retry=await distance();if(!retry.meshCached||retry.meshCacheFallback)throw new Error("Mesh cache retry failed");
        if(retry.meshQueryOrderHit||retry.meshQueryOrderBytes!==(orderEnabled?count*28:0))throw new Error("Changed/lost sources reused an old query order");
        const control=await distance(false);if(!same(retry.distances,control.distances))throw new Error("Retry/control bits differ");
        return {fault,recovered:true};
      }
      const first=await distance();if(!first.meshCached||first.meshCacheReused||!same(expected.distances,first.distances))throw new Error("Cold prepared C2M changed");
      if(first.meshQueryOrderHit||first.meshQueryOrderBytes!==(orderEnabled?count*28:0)||first.meshQueryOrderEntries!==(orderEnabled?first.meshSlices:0))throw new Error("Cold query-order payload differs");
      const initial=await state();if(!initial||initial.pending||initial.bytes>initial.reservation||initial.reservedTotal>initial.budget)throw new Error("Invalid mesh budget state");
      const both=await Promise.all([distance(),distance()]);
      if(both.some(r=>!r.meshCacheReused||!same(expected.distances,r.distances)))throw new Error("Overlapping mesh queries changed");
      if(both.some(r=>r.meshQueryOrderHit!==orderEnabled||r.meshQueryOrderBytes!==(orderEnabled?count*28:0)))throw new Error("Warm overlapping query orders missed");
      const unsigned=await distance(true,false), fresh=await distance(false,false);
      if(unsigned.meshQueryOrderHit!==orderEnabled)throw new Error("Sign switch lost query order");
      if(!same(unsigned.distances,fresh.distances)||JSON.stringify(unsigned.stats)!==JSON.stringify(fresh.stats)||await state()!==null)throw new Error("Signed/cache control differs");
      const noOrder=await call({kind:"c2c",compared:cloud.id,reference:mesh.id,signed:false,meshOrderCache:false});
      if(noOrder.meshQueryOrderHit||noOrder.meshQueryOrderBytes||!same(unsigned.distances,noOrder.distances))throw new Error("Disabled order control differs");
      await distance();const before=await state();const second=await loadMesh();mesh=second;
      await distance();if((await state()).job===before.job)throw new Error("Different meshes reused a BVH");
      await call({kind:"remove",id:mesh.id});if(await state()!==null)throw new Error("Removed mesh retained preparation");
      return {fault,matched:true};
    } finally {worker.terminate();}
  },{workerPath,fault,orderEnabled});
}

test("C2M retains exact mesh preparation across signs and overlapping batches",async({page})=>{
  expect(await scenario(page,"none")).toEqual({fault:"none",matched:true});
});
for(const fault of ["edit","remove-query","remove-mesh","loss"] as const)test(`C2M handles ${fault} and safely retries`,async({page})=>{
  expect(await scenario(page,fault)).toEqual({fault,recovered:true});
});
