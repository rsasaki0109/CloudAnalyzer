// Changing compared-cloud identities with a fixed measured voxel reference.
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
const sourcePath=resolve(inputManifest.source.path);
if(hash(readFileSync(sourcePath))!==inputManifest.source.sha256)throw new Error("Source differs from manifest");
const builds={};
for(const [name,root] of [["baseline",resolve(baselinePath)],["current",fileURLToPath(new URL("../dist/",import.meta.url))]]){
  const assets=resolve(root,"assets"),files=readdirSync(assets).filter(n=>/^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n)),worker=files.find(n=>/^worker-/.test(n));
  if(!worker||files.length!==3)throw new Error(`Expected three production artifacts in ${name}`);
  builds[name]={root,worker,artifactHashes:Object.fromEntries(files.map(n=>[n,hash(readFileSync(resolve(assets,n)))]))};
}
if(JSON.stringify(builds.baseline.artifactHashes)===JSON.stringify(builds.current.artifactHashes))throw new Error("Expected an archived production oracle");
const scriptPath=fileURLToPath(import.meta.url),scriptSha256=hash(readFileSync(scriptPath));
const sourceFiles=["web/src/map-voxel-cache.ts","web/src/map-quality-workers.ts","web/src/worker.ts","web/src/c2c-worker.ts","web/src/pool.ts","web/src/protocol.ts","rust/crates/ca-core/src/map_quality.rs","rust/crates/ca-core/src/icp.rs","rust/crates/ca-core/src/kdtree.rs","rust/crates/ca-core/src/distance.rs","rust/crates/ca-wasm/src/lib.rs"];
const sourceHashes=Object.fromEntries(sourceFiles.map(p=>[p,hash(readFileSync(resolve("..",p)))]));
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
try {
  browser=await chromium.launch({channel:"chromium",headless:true});const page=await browser.newPage();
  await page.exposeFunction("benchmarkProgress",message=>console.log(message));await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report=await page.evaluate(async({builds,inputs,inputManifest,repeats,warmupRounds})=>{
    const params={voxel:1,minPoints:5,threshold:0.5},files={};
    for(const role of ["reference","compared"])files[role]=new File([await(await fetch(`/__${role}`)).arrayBuffer()],inputs[role].name);
    const digest=async a=>[...new Uint8Array(await crypto.subtle.digest("SHA-256",a))].map(v=>v.toString(16).padStart(2,"0")).join("");
    const same=(a,b)=>{const x=new Uint8Array(a.buffer,a.byteOffset,a.byteLength),y=new Uint8Array(b.buffer,b.byteOffset,b.byteLength);return x.length===y.length&&x.every((v,i)=>v===y[i]);};
    const close=(a,b)=>Object.is(a,b)||Math.abs(a-b)<=1e-12*Math.max(1,Math.abs(a),Math.abs(b));
    const context=async mode=>{
      const worker=new Worker(`/${mode}/assets/${builds[mode].worker}`,{type:"module"});let seq=0;const pending=new Map();
      worker.onmessage=({data})=>{if(!data.response)return;const p=pending.get(data.seq);if(!p)return;pending.delete(data.seq);clearTimeout(p.timer);data.response.ok?p.resolve(data.response.value):p.reject(new Error(data.response.error));};
      worker.onerror=e=>{for(const p of pending.values()){clearTimeout(p.timer);p.reject(new Error(e.message));}pending.clear();};
      const call=req=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(new Error(`Timeout ${req.kind}`));},180_000);pending.set(id,{resolve,reject,timer});worker.postMessage({seq:id,req});});
      const load=async role=>{
        const r=await call({kind:"load",file:files[role],maxPoints:2_000_000});
        if(r.kind!=="cloud"||r.count!==inputManifest.inputs[role].points||r.filePoints!==r.count||r.keepEvery!==1)throw new Error("Measured input selection changed");
        return r;
      };
      try{await call({kind:"cpu-prepare"});return{worker,call,load,reference:await load("reference")};}catch(error){worker.terminate();throw error;}
    };
    const request=(ctx,compared,method,mode)=>({kind:method,compared:compared.id,reference:ctx.reference.id,...params,voxelSimd:false,voxelCache:false,
      voxelReferenceCache:mode==="fresh"?false:undefined});
    const cases=[];
    for(const method of ["map-quality","map-evaluate"]){
      const baseline=await context("baseline");let oracle,oraclePly;
      try{
        const compared=await baseline.load("compared");oracle=await baseline.call(request(baseline,compared,method,"fresh"));
        if(oracle.voxelCacheHit)throw new Error("Archived oracle used final score cache");
        if(method==="map-evaluate")oraclePly=await baseline.call({kind:"export",id:compared.id,format:"ply",scalar:{name:"MapDistance",values:oracle.distances}});
      }finally{baseline.worker.terminate();}
      const oracleScores=method==="map-quality"?{awd:oracle.awd,scs:oracle.scs,voxels:oracle.voxels}:oracle.quality;
      const oracleHashes=method==="map-evaluate"?{distance:await digest(oracle.distances),ply:await digest(oraclePly)}:undefined;
      const ctx=await context("current");let previousId=-1;
      try{
        const compute=async(mode,warm=true)=>{
          // New identity for the same original measured coordinates: final
          // score and bidirectional pair caches cannot satisfy this request.
          const compared=await ctx.load("compared");if(compared.id<=previousId)throw new Error("Compared identity did not change");previousId=compared.id;
          try{
            const start=performance.now(),r=await ctx.call(request(ctx,compared,method,mode)),elapsedMillis=performance.now()-start;
            const scores=method==="map-quality"?r:r.quality,prep=r.voxelReference;
            if(r.voxelCacheHit||!prep||prep.fallback||prep.cached!==(mode==="cached")||prep.reused!==(warm&&mode==="cached"))throw new Error("Unexpected reference/final score cache path: "+JSON.stringify(prep));
            if(mode==="cached"&&(prep.bytes>128*1024*1024||prep.rootsComputed!==(warm?0:scores.voxels)||prep.rootsReused!==(warm?scores.voxels:0)))throw new Error("Unexpected root reuse or allowance");
            if(scores.voxels!==oracleScores.voxels||!close(scores.awd,oracleScores.awd)||!close(scores.scs,oracleScores.scs))throw new Error("Voxel scores changed beyond original reduction allowance");
            if(method==="map-evaluate"){
              if(r.workers!==oracle.workers||!r.cpuCached||r.cpuCacheReused||r.cpuCacheInputBytes!==oracle.cpuCacheInputBytes||!same(r.distances,oracle.distances)||Object.keys(oracle.stats).some(k=>!Object.is(r.stats[k],oracle.stats[k])))throw new Error("Nearest preparation/layout/distance/statistics changed");
              for(const k of ["threshold","accuracy","precision","completeness","f1","chamfer"])if(!Object.is(scores[k],oracleScores[k]))throw new Error("Map metric changed: "+k);
              const ply=await ctx.call({kind:"export",id:compared.id,format:"ply",scalar:{name:"MapDistance",values:r.distances}});if(!same(ply,oraclePly))throw new Error("Full scalar PLY changed");
            }
            return{elapsedMillis,workerMillis:r.millis,workers:r.workers,comparedId:compared.id,voxelReference:prep,voxelCacheHit:r.voxelCacheHit,
              cpuCacheReused:r.cpuCacheReused,cpuCacheInputBytes:r.cpuCacheInputBytes,scores:{awd:scores.awd,scs:scores.scs,voxels:scores.voxels},resultParity:true};
          }finally{await ctx.call({kind:"remove",id:compared.id});}
        };
        const firstUse={fresh:await compute("fresh",false),cached:await compute("cached",false)},warmups=[],trials=[];
        for(let i=-warmupRounds;i<repeats;i++){
          const order=(i+warmupRounds)%2?["cached","fresh"]:["fresh","cached"],values={};for(const mode of order)values[mode]=await compute(mode);
          (i<0?warmups:trials).push({order,...values});
          await globalThis.benchmarkProgress(`${method} ${i<0?"warm-up":"trial"} ${i<0?i+warmupRounds+1:i+1}: fresh ${values.fresh.elapsedMillis.toFixed(1)}, reference ${values.cached.elapsedMillis.toFixed(1)} ms; ${values.cached.voxelReference.rootsReused} roots reused; parity`);
        }
        cases.push({method,oracleScores,oracleHashes,oracleWorkers:oracle.workers,firstUse,warmups,trials});
      }finally{ctx.worker.terminate();}
    }
    return{params,hardwareConcurrency:navigator.hardwareConcurrency,userAgent:navigator.userAgent,cases};
  },{builds:Object.fromEntries(Object.entries(builds).map(([k,b])=>[k,{worker:b.worker}])),inputs,inputManifest,repeats,warmupRounds});
  for(const[name,b]of Object.entries(builds))for(const[file,expected]of Object.entries(b.artifactHashes))if(hash(readFileSync(resolve(b.root,"assets",file)))!==expected)throw new Error(`Changed ${name} artifact`);
  for(const[p,expected]of Object.entries(sourceHashes))if(hash(readFileSync(resolve("..",p)))!==expected)throw new Error(`Changed source ${p}`);
  for(const role of ["reference","compared"])if(hash(readFileSync(inputs[role].path))!==inputs[role].sha256)throw new Error("Input changed during timing");
  if(hash(readFileSync(manifestPath))!==hash(manifestBytes)||hash(readFileSync(sourcePath))!==inputManifest.source.sha256||hash(readFileSync(scriptPath))!==scriptSha256)throw new Error("Provenance/script changed during timing");
  const median=a=>{a=[...a].sort((x,y)=>x-y);return a.length%2?a[a.length>>1]:(a[a.length/2-1]+a[a.length/2])/2;};
  for(const c of report.cases){const reductions=c.trials.map(t=>100*(1-t.cached.elapsedMillis/t.fresh.elapsedMillis));c.summary={freshMedianMillis:median(c.trials.map(t=>t.fresh.elapsedMillis)),referenceMedianMillis:median(c.trials.map(t=>t.cached.elapsedMillis)),medianPairedReductionPercent:median(reductions),fasterTrials:reductions.filter(x=>x>0).length,trials:repeats};}
  Object.assign(report,{schemaVersion:1,comparison:"fixed-voxel-reference-changing-compared-identity",createdAt:new Date().toISOString(),builds,scriptSha256,sourceHashes,inputs,inputManifest,inputManifestSha256:hash(manifestBytes),repeats,warmupRounds,browserVersion:browser.version(),
    commit:execFileSync("git",["rev-parse","HEAD"],{encoding:"utf8",windowsHide:true}).trim(),workingTreeDirty:!!execFileSync("git",["status","--porcelain"],{encoding:"utf8",windowsHide:true}).trim(),
    machine:{platform:platform(),release:release(),cpu:cpus()[0]?.model,logicalCpus:cpus().length,memoryBytes:totalmem()},
    note:"Full unchanged measured source-index splits, not independent surveys/accuracy ground truth. A new compared Cloud identity is loaded from the same original measured file before EVERY calculation, with no added/transformed/thinned coordinates; this models resubmitting/replacing a compared cloud against a fixed reference. It does not measure different acquisition geometry or final score-cache hits. Final score reuse is disabled. Current fresh and automatic reference controls share one context per method; cached Gaussian/root memory remains resident during bypass controls. Reference preparation is deferred until a matching reference is encountered again. Timed automatic calls require retained reference reuse, zero new roots and one reused root per matched voxel under the 128 MiB retained payload allowance. A changing compared identity also prevents whole bidirectional nearest-pair reuse in combined Map quality: both modes require fresh exact C2C preparation, unchanged lanes/input allowances, identical f32 distance bits/f64 statistics/non-voxel metrics and complete scalar PLY. Archived production outputs supply oracles. Gaussian/individual Wasserstein f64 arithmetic remains unchanged; AWD/SCS permit only the original 1e-12 relative/absolute randomized HashMap reduction allowance. Whole RPC includes Gaussian building, exact Wasserstein/SCS, optional nearest preparation/search, joins/summary and returned arrays. Compared/reference loading, pool startup, first observations, five alternating warm-up pairs, parity/export/removal excluded. First fresh observation builds both maps; first automatic observation prepares the reference and roots, so these single observations are not cold medians. Source/artifact/input/source/manifest/script hashes checked before/after. No builds/tests/scans alongside timing; clocks, power, other applications, scheduling, GC, HashMap seeds and retained WASM pages uncontrolled. Retained allowance estimates hash payload/capacity with slack, not temporary allocations, driver/allocator overhead or peak process memory. No GPU model identifiers recorded."});
  writeFileSync(output,JSON.stringify(report,null,2)+"\n",{flag:"wx"});console.log(`Saved ${output}`);
}finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
