import { expect, test } from "@playwright/test";
import { readdirSync } from "node:fs";

test("M3C2 fused projection preserves serial, copied, prepared and cached outputs", async ({ page }) => {
  await page.route("**/__m3c2-projection", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__m3c2-projection");
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n));
  const result = await page.evaluate(async workerPath => {
    const worker = new Worker(workerPath, { type: "module" }), waiting = new Map<number, any>(); let seq = 0;
    worker.onmessage = ({ data }) => {
      if (!data.response) return; const p = waiting.get(data.seq); if (!p) return; waiting.delete(data.seq);
      data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = e => { for (const p of waiting.values()) p.reject(new Error(e.message)); waiting.clear(); };
    const call = (req: any): Promise<any> => new Promise((resolve, reject) => { const id = ++seq; waiting.set(id, { resolve, reject }); worker.postMessage({ seq: id, req }); });
    const count = 40013;
    const fixture = (lift: number) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + count*24); bytes.set(header); const v = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < count; i++) {
        const x = i%201*0.1, y = Math.floor(i/201)*0.1;
        [368000+x,3955000+y,40+x*0.2+Math.sin(y)*0.01+lift].forEach((p,a) => v.setFloat64(i*24+a*8,p,true));
      }
      return new File([bytes],"plane.ply");
    };
    const same = (a: ArrayBufferView,b: ArrayBufferView) => {
      const x = new Uint8Array(a.buffer,a.byteOffset,a.byteLength), y = new Uint8Array(b.buffer,b.byteOffset,b.byteLength);
      return x.length === y.length && x.every((v,i)=>v===y[i]);
    };
    try {
      await call({ kind:"cpu-prepare" });
      const reference = await call({ kind:"load",file:fixture(0),maxPoints:count }), compared = await call({ kind:"load",file:fixture(0.07),maxPoints:count });
      const compute = (options: any) => call({ kind:"m3c2",reference:reference.id,compared:compared.id,normalRadius:0.4,projectionRadius:0.25,maxDepth:0.5,coreSpacing:0,...options });
      const control = await compute({ parallel:false,fusedProjection:false });
      const ply = (r: any) => call({kind:"export",id:r.cloud.id,format:"ply"});
      const bytes = await ply(control); let checked = 0;
      for (const expectedHit of [false,false,true]) {
        const r = await compute({});
        if (r.fusedProjection !== expectedHit || r.normalCacheHit !== expectedHit || !same(bytes,await ply(r))) throw new Error("Automatic projection policy changed first/capture/reuse calls");
        await call({kind:"remove",id:r.cloud.id});
      }
      for (const options of [
        { parallel:false,sharedSearch:false },
        { preparedTrees:false,mainWorker:false,sharedSearch:true },
        { preparedTrees:true,cacheNormals:false,sharedSearch:true },
        { preparedTrees:true,cacheNormals:true,sharedSearch:true },
      ]) {
        for (const fusedProjection of [false,true,undefined]) {
          const r = await compute({...options,fusedProjection});
          if (r.fusedProjection !== (fusedProjection ?? !!options.cacheNormals) || r.cloud.count !== count ||
            ["distance","lod95","significant"].some(k=>!same(control[k],r[k])) || !same(bytes,await ply(r))) throw new Error("M3C2 projection changed result/path");
          if (options.cacheNormals && fusedProjection !== false && !r.normalCacheHit) throw new Error("Projection control invalidated fitted normals");
          await call({kind:"remove",id:r.cloud.id}); checked++;
        }
      }
      const both = await Promise.all([compute({fusedProjection:false,cacheNormals:true}),compute({fusedProjection:true,cacheNormals:true})]);
      for (const r of both) if (!r.normalCacheHit || !same(bytes,await ply(r))) throw new Error("Overlapping projection modes changed cached results");
      return {checked,concurrent:both.length};
    } finally { worker.terminate(); }
  }, workerPath);
  expect(result).toEqual({ checked:12,concurrent:2 });
});
