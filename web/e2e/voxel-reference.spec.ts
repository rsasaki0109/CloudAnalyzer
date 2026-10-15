import { expect, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";

test("voxel reference policy preserves bypasses, invalidation and bounded fallback", async () => {
  let created = 0, freed = 0, bytes = 100, fail = false, raw = 0;
  class Reference {
    retainedBytes = bytes; rootsReused = 0; rootsComputed = 7;
    constructor() { created++; }
    compute() {
      if (fail) { fail = false; throw new Error("reference compute failed"); }
      const result = new Float64Array([1,2,7]); this.rootsReused = this.rootsComputed ? 0 : 7; this.rootsComputed = 0; return result;
    }
    free() { freed++; }
  }
  (globalThis as any).__voxelHooks = { VoxelReference:Reference,mapQuality:()=>{raw++;return new Float64Array([1,2,7]);} };
  const source = stripTypeScriptTypes(readFileSync("src/map-voxel-cache.ts","utf8"),{mode:"transform"}).replace(/^import\b[\s\S]*?;\s*/gm,"");
  const prefix = "const {VoxelReference,mapQuality}=globalThis.__voxelHooks;\n";
  const m = await import("data:text/javascript;base64,"+Buffer.from(prefix+source).toString("base64"));
  const ref = {length:1000}, query = {length:1000};
  const score = (compared=query,reference=ref,voxel=1,simd=false,policy?:boolean,final=false) => m.snapshotVoxelScores(compared,reference,voxel,5,simd,final,policy);
  try {
    expect(score().reference.cached).toBe(false); expect(created).toBe(0);
    expect(score().reference.reused).toBe(false); expect(created).toBe(1);
    expect(score().reference.reused).toBe(true);
    expect(score(query,ref,1,false,false).reference.cached).toBe(false); expect(freed).toBe(0);
    m.invalidateVoxelScores(query); expect(score().reference.reused).toBe(true);
    m.invalidateVoxelScores(ref); expect(freed).toBe(1); expect(score().reference.cached).toBe(false);
    expect(score(query,ref,1,false,true,true).reference.cached).toBe(true);
    const before = created; expect(score(query,ref,1,false,true,true).reused).toBe(true); expect(created).toBe(before);
    expect(score({},ref,1,false,true).reference.reused).toBe(true);
    expect(score({},ref,0.5,false,true).reference.reused).toBe(false); expect(freed).toBe(2);
    expect(score({},ref,0.5,true,true).reference.reused).toBe(false); expect(freed).toBe(3);
    fail=true; expect(score({},ref,0.5,true,true).reference.fallback).toContain("compute failed"); expect(freed).toBe(4);
    expect(score({},ref,0.5,true,true).reference.cached).toBe(true);
    bytes=129*1024*1024;
    expect(score({},{length:1000},1,false,true).reference.fallback).toContain("retained allowance");
    const allocations=created;
    expect(score({},{length:1_050_000},1,false,true).reference.fallback).toContain("preparation allowance"); expect(created).toBe(allocations);
    expect(score({},{length:1_000_000_000},1,false,true).reference.cached).toBe(false); expect(created).toBe(allocations);
    expect(raw).toBeGreaterThan(0);
  } finally { m.invalidateVoxelScores(ref); delete (globalThis as any).__voxelHooks; }
});

test("voxel reference reuse follows actual clouds and retains scalar/SIMD score semantics", async ({page}) => {
  await page.route("**/__voxel-reference",route=>route.fulfill({contentType:"text/html",body:"<!doctype html>"}));
  await page.goto("/__voxel-reference");
  const workerPath="/assets/"+readdirSync("dist/assets").find(n=>/^worker-.*\.js$/.test(n));
  const result=await page.evaluate(async workerPath=>{
    const worker=new Worker(workerPath,{type:"module"}),pending=new Map<number,any>();let seq=0;
    worker.onmessage=({data})=>{if(!data.response)return;const p=pending.get(data.seq);if(!p)return;pending.delete(data.seq);data.response.ok?p.resolve(data.response.value):p.reject(new Error(data.response.error));};
    worker.onerror=e=>{for(const p of pending.values())p.reject(new Error(e.message));pending.clear();};
    const call=(req:any):Promise<any>=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});worker.postMessage({seq:id,req});});
    const count=200319;
    const fixture=(lift:number)=>{
      const header=new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`),bytes=new Uint8Array(header.length+count*24);bytes.set(header);const d=new DataView(bytes.buffer,header.length);
      for(let i=0;i<count;i++){const x=i%701*0.05,y=Math.floor(i/701)*0.05;[368000+x,3955000+y,40+Math.sin(x)*0.01+lift].forEach((v,a)=>d.setFloat64(i*24+a*8,v,true));}
      return new File([bytes],"points.ply");
    };
    const same=(a:any,b:any)=>{if(a.voxels!==b.voxels||["awd","scs"].some(k=>!Object.is(a[k],b[k])&&Math.abs(a[k]-b[k])>1e-12*Math.max(1,Math.abs(a[k]),Math.abs(b[k]))))throw new Error("Voxel metrics changed");};
    try {
      await call({kind:"cpu-prepare"});
      let reference=await call({kind:"load",file:fixture(0),maxPoints:count}),compared=await call({kind:"load",file:fixture(0.02),maxPoints:count});
      const score=(policy?:boolean,voxel=1,simd=false,final=false)=>call({kind:"map-quality",compared:compared.id,reference:reference.id,voxel,minPoints:5,voxelSimd:simd,voxelCache:final,voxelReferenceCache:policy});
      const first=await score();if(first.voxelReference.cached)throw new Error("First auto call prepared reference");
      const capture=await score(),warm=await score();same(first,capture);same(first,warm);
      if(capture.voxelReference.reused||!capture.voxelReference.cached||!warm.voxelReference.reused||warm.voxelReference.rootsComputed||warm.voxelReference.rootsReused!==warm.voxels||warm.voxelReference.bytes>128*1024*1024)throw new Error("Invalid reference preparation/reuse");
      const second=await call({kind:"load",file:fixture(0.04),maxPoints:count});await call({kind:"remove",id:compared.id});compared=second;
      const changed=await score();same(changed,await score(false));if(!changed.voxelReference.reused)throw new Error("Compared replacement lost reference");
      const matrix=[1,0,0,0,0,1,0,0,0,0,1,0.01,0,0,0,1];await call({kind:"transform",id:compared.id,matrix});
      const moved=await score();same(moved,await score(false));if(!moved.voxelReference.reused)throw new Error("Compared edit lost reference");
      for(const voxel of [0.5,1])for(const simd of [false,true]){const cold=await score(true,voxel,simd),again=await score(true,voxel,simd);same(cold,again);same(cold,await score(false,voxel,simd));if(cold.voxelReference.reused||!again.voxelReference.reused)throw new Error("Settings reused wrong reference");}
      await call({kind:"transform",id:reference.id,matrix});
      const refMoved=await score();same(refMoved,await score(false));if(refMoved.voxelReference.cached)throw new Error("Reference edit retained stale statistics");
      await score(true);await call({kind:"remove",id:reference.id});reference=await call({kind:"load",file:fixture(0),maxPoints:count});
      const replaced=await score();same(replaced,await score(false));if(replaced.voxelReference.cached)throw new Error("New reference reused removed source");
      const cached=await score(true,1,false,true),hit=await score(true,1,false,true);same(cached,hit);if(!hit.voxelCacheHit||hit.voxelReference!==undefined)throw new Error("Final score cache did not skip reference work");
      const evaluate=(policy:boolean)=>call({kind:"map-evaluate",reference:reference.id,compared:compared.id,threshold:0.15,voxel:1,minPoints:5,voxelCache:false,voxelReferenceCache:policy});
      const a=await evaluate(true),b=await evaluate(false);same(a.quality,b.quality);
      const x=new Uint8Array(a.distances.buffer),y=new Uint8Array(b.distances.buffer);if(x.length!==y.length||!x.every((v,i)=>v===y[i])||Object.keys(a.stats).some(k=>!Object.is(a.stats[k],b.stats[k]))||!a.voxelReference.reused||b.voxelReference.cached)throw new Error("Combined Map quality changed");
      return {matched:true,roots:warm.voxelReference.rootsReused};
    }finally{worker.terminate();}
  },workerPath);
  expect(result.matched).toBe(true);expect(result.roots).toBeGreaterThan(0);
});
