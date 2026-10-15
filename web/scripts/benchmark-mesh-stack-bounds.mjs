// Whole-RPC retained C2M with cached traversal bounds, against archived retained CPU.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { readFileSync, readdirSync, existsSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { cpus, platform, release, totalmem } from "node:os";

const args=process.argv.slice(2), allowed=["--output","--reference","--compared","--input-manifest","--baseline-dist","--repeats"];
for(let i=0;i<args.length;i+=2)if(!allowed.includes(args[i])||!args[i+1]||args[i+1].startsWith("--")||args.indexOf(args[i])!==i)throw new Error(`Invalid or repeated option: ${args[i]}`);
const value=(key,fallback)=>args.includes(key)?args[args.indexOf(key)+1]:fallback;
const output=value("--output"),manifestPath=value("--input-manifest"),baselinePath=value("--baseline-dist"),repeats=Number(value("--repeats","10")),warmupRounds=5;
if(!output||!manifestPath||!baselinePath||!value("--reference")||!value("--compared"))throw new Error("Supply --baseline-dist DIST --reference FILE --compared FILE --input-manifest JSON --output NEW.json [--repeats 10]");
if(existsSync(output))throw new Error(`Report already exists: ${output}`);
if(!Number.isInteger(repeats)||repeats<1||repeats>100)throw new Error("Invalid repeats");
const hash=b=>createHash("sha256").update(b).digest("hex"),manifestBytes=readFileSync(manifestPath),inputManifest=JSON.parse(manifestBytes);
const inputData={},inputs={};
for(const role of ["reference","compared"]){
  const path=value(`--${role}`),bytes=readFileSync(path);inputData[role]=bytes;
  inputs[role]={path,name:basename(path),bytes:bytes.length,sha256:hash(bytes)};
  const expected=inputManifest.inputs?.[role];
  if(!expected||expected.sha256!==inputs[role].sha256||expected.bytes!==bytes.length||!Number.isInteger(expected.points)||expected.points<200_000||expected.points>1_000_000)throw new Error(`Expected full measured ${role} input matching its manifest`);
}
const sourcePath=resolve("web",inputManifest.source.path);
if(hash(readFileSync(sourcePath))!==inputManifest.source.sha256)throw new Error("Source differs from manifest");
const builds={};
for(const [name,root] of [["baseline",resolve(baselinePath)],["current",fileURLToPath(new URL("../dist/",import.meta.url))]]){
  const assets=resolve(root,"assets"),files=readdirSync(assets).filter(n=>/^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n)),worker=files.find(n=>/^worker-/.test(n));
  if(!worker||files.length!==3)throw new Error(`Expected three production artifacts in ${name}`);
  builds[name]={root,worker,artifactHashes:Object.fromEntries(files.map(n=>[n,hash(readFileSync(resolve(assets,n)))]))};
}
if(JSON.stringify(builds.baseline.artifactHashes)===JSON.stringify(builds.current.artifactHashes))throw new Error("Expected an archived production oracle");
const scriptPath=fileURLToPath(import.meta.url),scriptSha256=hash(readFileSync(scriptPath));
if(!readFileSync("rust/crates/ca-core/src/distance.rs","utf8").includes("const USE_MESH_STACK_BOUNDS: bool = true;"))throw new Error("Enable cached mesh traversal bounds and rebuild before measuring");
const sourceFiles=["web/src/mesh-preparation.ts","web/src/worker.ts","web/src/c2c-worker.ts","web/src/pool.ts","web/src/protocol.ts","rust/crates/ca-core/src/mesh.rs","rust/crates/ca-core/src/distance.rs","rust/crates/ca-core/src/lib.rs","rust/crates/ca-core/src/delaunay.rs","rust/crates/ca-core/src/io/write.rs","rust/crates/ca-wasm/src/lib.rs"];
const sourceHashes=Object.fromEntries(sourceFiles.map(p=>[p,hash(readFileSync(resolve(p)))]));
const server=createServer((req,res)=>{
  const path=new URL(req.url,"http://localhost").pathname;
  if(path==="/"){res.setHeader("Content-Type","text/html");res.end("<!doctype html>");return;}
  if(path==="/__reference"||path==="/__compared"){res.setHeader("Content-Type","application/octet-stream");res.end(inputData[path.slice(3)]);return;}
  const mode=path.split("/")[1],build=builds[mode];if(!build){res.writeHead(404).end();return;}
  const target=resolve(build.root,"."+path.slice(mode.length+1));
  if(!target.startsWith(build.root.endsWith(sep)?build.root:build.root+sep)){res.writeHead(403).end();return;}
  try{res.setHeader("Content-Type",({".js":"text/javascript",".wasm":"application/wasm"})[extname(target)]||"application/octet-stream");res.end(readFileSync(target));}catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));let browser;
try{
  browser=await chromium.launch({channel:"chromium",headless:true});const page=await browser.newPage();
  await page.exposeFunction("benchmarkProgress",message=>console.log(message));await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report=await page.evaluate(async({builds,inputs,inputManifest,repeats,warmupRounds})=>{
    const files={};for(const role of ["reference","compared"])files[role]=new File([await(await fetch(`/__${role}`)).arrayBuffer()],inputs[role].name);
    const digest=async a=>[...new Uint8Array(await crypto.subtle.digest("SHA-256",a))].map(v=>v.toString(16).padStart(2,"0")).join("");
    const same=(a,b)=>{const x=new Uint8Array(a.buffer,a.byteOffset,a.byteLength),y=new Uint8Array(b.buffer,b.byteOffset,b.byteLength);return x.length===y.length&&x.every((v,i)=>v===y[i]);};
    const context=async mode=>{
      const worker=new Worker(`/${mode}/assets/${builds[mode].worker}`,{type:"module"});let seq=0;const pending=new Map();
      worker.onmessage=({data})=>{if(!data.response)return;const p=pending.get(data.seq);if(!p)return;pending.delete(data.seq);clearTimeout(p.timer);data.response.ok?p.resolve(data.response.value):p.reject(new Error(data.response.error));};
      worker.onerror=e=>{for(const p of pending.values()){clearTimeout(p.timer);p.reject(new Error(e.message));}pending.clear();};
      const call=req=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(new Error(`Timeout ${req.kind}`));},180_000);pending.set(id,{resolve,reject,timer});worker.postMessage({seq:id,req});});
      try{
        await call({kind:"cpu-prepare"});const clouds={};
        for(const role of ["reference","compared"]){clouds[role]=await call({kind:"load",file:files[role],maxPoints:2_000_000});const c=clouds[role];if(c.kind!=="cloud"||c.count!==inputManifest.inputs[role].points||c.filePoints!==c.count||c.keepEvery!==1)throw new Error("Input point selection changed");}
        const built=await call({kind:"mesh",id:clouds.reference.id,maxEdge:null});
        if(!built.mesh.triangles||built.voxel!==0)throw new Error("Expected full-input measured TIN without voxel thinning");
        const geometry=await call({kind:"export",id:built.mesh.id,format:"ply"});
        const mesh={vertices:built.mesh.count,triangles:built.mesh.triangles,points:built.points,voxel:built.voxel,maxEdge:built.maxEdge,removed:built.removed,positionsSha256:await digest(built.mesh.positions),indicesSha256:await digest(built.mesh.indices),geometryPlyBytes:geometry.byteLength,geometryPlySha256:await digest(geometry)};
        return{worker,call,clouds,mesh,meshId:built.mesh.id};
      }catch(error){worker.terminate();throw error;}
    };
    const contexts=[];
    try {
      const archived=await context("baseline");contexts.push(archived);
      const request=ctx=>({kind:"c2c",compared:ctx.clouds.compared.id,reference:ctx.meshId,signed:false,meshCache:true,meshGpu:false});
      const oracle=await archived.call(request(archived));
      if(!oracle.meshCached||oracle.meshCacheFallback||oracle.meshCompute)throw new Error("Expected retained archived CPU oracle");
      const oracleBytes=await archived.call({kind:"export",id:archived.clouds.compared.id,format:"ply",scalar:{name:"C2M",values:oracle.distances}});
      const distanceSha256=await digest(oracle.distances),plySha256=await digest(oracleBytes);archived.worker.terminate();
      const ctxs={baseline:await context("baseline"),bounded:await context("current")};contexts.push(...Object.values(ctxs));
      for(const ctx of Object.values(ctxs))if(JSON.stringify(ctx.mesh)!==JSON.stringify(archived.mesh))throw new Error("Production TIN changed");
      const compute=async mode=>{
        const ctx=ctxs[mode],start=performance.now(),r=await ctx.call(request(ctx)),elapsedMillis=performance.now()-start;
        const cache=await ctx.call({kind:"mesh-cache-state"});
        if(r.kind!=="c2m"||r.signed||!r.meshCached||r.meshCacheFallback||r.meshCompute||r.workers!==oracle.workers||r.meshSlices!==oracle.meshSlices||
          !cache||cache.pending||cache.bytes!==r.meshCacheBytes||cache.bytes>cache.reservation||cache.reservedTotal>cache.budget)throw new Error("Unexpected retained C2M path/layout/budget");
        if(!same(oracle.distances,r.distances)||Object.keys(oracle.stats).some(k=>!Object.is(oracle.stats[k],r.stats[k])))throw new Error("C2M bits/summaries changed");
        const bytes=await ctx.call({kind:"export",id:ctx.clouds.compared.id,format:"ply",scalar:{name:"C2M",values:r.distances}});
        if(!same(oracleBytes,bytes))throw new Error("Complete C2M scalar PLY changed");
        return{elapsedMillis,workerMillis:r.millis,workers:r.workers,slices:r.meshSlices,reused:r.meshCacheReused,retainedBytes:r.meshCacheBytes,cache,stats:r.stats,resultParity:true};
      };
      const firstUse={baseline:await compute("baseline"),bounded:await compute("bounded")};
      if(firstUse.baseline.reused||firstUse.bounded.reused)throw new Error("Unexpected cold reuse");
      const checkPair=values=>{if(values.baseline.retainedBytes!==values.bounded.retainedBytes||values.baseline.cache.reservation!==values.bounded.cache.reservation)throw new Error("Retained geometry allowance changed");};checkPair(firstUse);
      const warmups=[],trials=[];
      for(let i=-warmupRounds;i<repeats;i++){
        const order=(i+warmupRounds)%2?["bounded","baseline"]:["baseline","bounded"],values={};for(const mode of order)values[mode]=await compute(mode);
        if(!values.baseline.reused||!values.bounded.reused)throw new Error("Expected warm retained geometry");checkPair(values);
        (i<0?warmups:trials).push({order,...values});
        await globalThis.benchmarkProgress('C2M '+(i<0?'warm-up':'trial')+' '+(i<0?i+warmupRounds+1:i+1)+': baseline '+values.baseline.elapsedMillis.toFixed(1)+', bounded '+values.bounded.elapsedMillis.toFixed(1)+' ms; '+values.bounded.workers+' retained BVHs / '+values.bounded.slices+' slices; parity');
      }
      return{hardwareConcurrency:navigator.hardwareConcurrency,userAgent:navigator.userAgent,mesh:archived.mesh,points:archived.clouds.compared.count,
        oracleWorkers:oracle.workers,oracleSlices:oracle.meshSlices,oracleStats:oracle.stats,distanceSha256,plySha256,firstUse,warmups,trials};
    }finally{for(const ctx of contexts)ctx.worker.terminate();}
  },{builds:Object.fromEntries(Object.entries(builds).map(([k,b])=>[k,{worker:b.worker}])),inputs,inputManifest,repeats,warmupRounds});
  for(const [name,build] of Object.entries(builds))for(const [file,expected]of Object.entries(build.artifactHashes))if(hash(readFileSync(resolve(build.root,"assets",file)))!==expected)throw new Error(`Changed ${name} artifact`);
  for(const [p,expected]of Object.entries(sourceHashes))if(hash(readFileSync(resolve(p)))!==expected)throw new Error(`Changed source ${p}`);
  for(const role of ["reference","compared"])if(hash(readFileSync(inputs[role].path))!==inputs[role].sha256)throw new Error("Input changed during timing");
  if(hash(readFileSync(manifestPath))!==hash(manifestBytes)||hash(readFileSync(sourcePath))!==inputManifest.source.sha256||hash(readFileSync(scriptPath))!==scriptSha256)throw new Error("Provenance/script changed during timing");
  const median=a=>{a=[...a].sort((x,y)=>x-y);return a.length%2?a[a.length>>1]:(a[a.length/2-1]+a[a.length/2])/2;};
  const reductions=report.trials.map(t=>100*(1-t.bounded.elapsedMillis/t.baseline.elapsedMillis));
  Object.assign(report,{schemaVersion:1,comparison:"c2m-cached-stack-bounds",createdAt:new Date().toISOString(),builds,scriptSha256,sourceHashes,inputs,inputManifest,inputManifestSha256:hash(manifestBytes),repeats,warmupRounds,browserVersion:browser.version(),
    commit:execFileSync("git",["rev-parse","HEAD"],{encoding:"utf8",windowsHide:true}).trim(),workingTreeDirty:!!execFileSync("git",["status","--porcelain"],{encoding:"utf8",windowsHide:true}).trim(),
    machine:{platform:platform(),release:release(),cpu:cpus()[0]?.model,logicalCpus:cpus().length,memoryBytes:totalmem()},
    summary:{baselineMedianMillis:median(report.trials.map(t=>t.baseline.elapsedMillis)),boundedMedianMillis:median(report.trials.map(t=>t.bounded.elapsedMillis)),medianPairedReductionPercent:median(reductions),fasterTrials:reductions.filter(x=>x>0).length,trials:repeats},
    note:"Full unchanged measured source-index splits, not independent surveys/accuracy ground truth. Full-input production 2.5D TIN records XY duplicates and automatic edge rejection; no new coordinates, voxel thinning or input resampling. An independent archived retained-CPU worker supplies f32 distance, exact f64 summary and full scalar PLY oracles, and terminates before timing. Separate archived/current production contexts both use adopted 1 GiB retained mesh/BVH policy, identical vector capacities, resident worker count and original logical query slices. Every batch resets Morton order, guessed triangle and traversal stack. Candidate stack entries retain child f64 box distances already computed for ordering, avoiding their duplicate recomputation when popped; pruning occurs on pop with updated best distance. Child ordering, triangle tests/strict insertion and distance arithmetic retain preceding behavior. No new persistent geometry, query/result cache, public request or GPU dispatch. Temporary stack entries grow from one usize to (usize,f64); cache bytes and reservations remain identical and do not describe peak process memory. One excluded first-use pair and five alternating warm-up pairs precede ten alternating timed pairs. Whole unsigned C2M RPC includes copies/transfers, query ordering/search, join/summaries/returned arrays. Loading/production meshing/pool startup/first observations/warmups/export/parity/cleanup excluded. Timed geometry-cache misses/fallback, layout/budget mismatch, or output differences fail. Source, three-artifact sets, input/manifest/original PCD/script hashes checked before/after. No builds/tests/scans/edits alongside timing. Clocks/power/scheduling/GC/other apps/code layout/retained pages uncontrolled; no isolated search, signed/native/GPU speedup claim. No GPU model identifiers collected."});
  writeFileSync(output,JSON.stringify(report,null,2)+"\n",{flag:"wx"});console.log(`Saved ${output}`);
}finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
