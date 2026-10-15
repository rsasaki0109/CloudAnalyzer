// Compare indexed/slot-only SOR queries with identical warmed preparations.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2), allowed = ["--output", "--input", "--input-manifest", "--baseline-dist", "--repeats"];
for (let i=0;i<args.length;i+=2) if (!allowed.includes(args[i]) || !args[i+1] || args[i+1].startsWith("--") || args.indexOf(args[i])!==i)
  throw new Error(`Invalid or repeated option: ${args[i]}`);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key)+1] : fallback;
const output=value("--output"), inputPath=value("--input"), manifestPath=value("--input-manifest"), baselineRoot=value("--baseline-dist");
if (!output || !inputPath || !manifestPath || !baselineRoot) throw new Error("Supply --baseline-dist DIST --input COMPARED.ply --input-manifest JSON --output NEWREPORT.json [--repeats 10]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const repeats=Number(value("--repeats","10")), warmupRounds=5;
if (!Number.isInteger(repeats) || repeats<1 || repeats>100) throw new Error("Invalid repeats");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const inputBytes=readFileSync(inputPath), manifestBytes=readFileSync(manifestPath), inputManifest=JSON.parse(manifestBytes);
const input={path:inputPath,name:basename(inputPath),bytes:inputBytes.length,sha256:digest(inputBytes)};
if (inputManifest.inputs?.compared?.sha256!==input.sha256 || inputManifest.inputs.compared.bytes!==input.bytes ||
  !Number.isInteger(inputManifest.inputs.compared.points) || inputManifest.inputs.compared.points<1 || inputManifest.inputs.compared.points>2_000_000) throw new Error("Expected full measured input matching its manifest, at most two million points");
const originalPath=resolve('web',inputManifest.source?.path??''),originalBytes=readFileSync(originalPath);
if(originalBytes.length!==inputManifest.source?.bytes||digest(originalBytes)!==inputManifest.source.sha256) throw new Error('Original measured source differs from manifest');
if(!/const USE_SOR_TREE_SLOTS:\s*bool\s*=\s*true\s*;/.test(readFileSync('rust/crates/ca-core/src/filter.rs','utf8')))
  throw new Error('Set USE_SOR_TREE_SLOTS = true and rebuild matching production WASM/app');
const builds={};
for (const [mode,root] of [["baseline",resolve(baselineRoot)],["optimized",fileURLToPath(new URL("../dist/",import.meta.url))]]) {
  const names=readdirSync(resolve(root,"assets")).filter(n=>/^(worker-.*\.js|c2c-worker-.*\.js|ca_wasm.*\.wasm)$/.test(n));
  const worker=names.find(n=>/^worker-/.test(n)); if (!worker || names.length!==3) throw new Error(`Expected three production artifacts in ${mode}`);
  builds[mode]={root,worker,artifactHashes:Object.fromEntries(names.map(n=>[n,digest(readFileSync(resolve(root,"assets",n)))]))};
}
if (JSON.stringify(builds.baseline.artifactHashes)===JSON.stringify(builds.optimized.artifactHashes)) throw new Error("Expected a distinct archived pre-change build for the output oracle");
const scriptSha256=digest(readFileSync(fileURLToPath(import.meta.url)));
const sourceFiles=["rust/crates/ca-core/src/kdtree.rs","web/src/knn-preparation.ts","web/src/worker.ts","web/src/c2c-worker.ts","web/src/pool.ts","rust/crates/ca-core/src/filter.rs","rust/crates/ca-core/src/normals.rs","rust/crates/ca-wasm/src/lib.rs","web/src/knn-join.ts","web/src/protocol.ts"];
const sourceHashes=Object.fromEntries(sourceFiles.map(path=>[path,digest(readFileSync(path))]));
const server=createServer((req,res)=>{
  const path=new URL(req.url,"http://localhost").pathname;
  if(path==="/"){res.setHeader("Content-Type","text/html");res.end("<!doctype html>");return;}
  if(path==="/__input"){res.setHeader("Content-Type","application/octet-stream");res.end(inputBytes);return;}
  const mode=path.split("/")[1], build=builds[mode]; if(!build){res.writeHead(404).end();return;}
  const target=resolve(build.root,"."+path.slice(mode.length+1));
  if(!target.startsWith(build.root.endsWith(sep)?build.root:build.root+sep)){res.writeHead(403).end();return;}
  try{res.setHeader("Content-Type",({".js":"text/javascript",".wasm":"application/wasm"})[extname(target)]||"application/octet-stream");res.end(readFileSync(target));}
  catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
let browser;
try {
  browser=await chromium.launch({channel:"chromium",headless:true}); const page=await browser.newPage();
  await page.exposeFunction("benchmarkProgress",message=>console.log(message)); await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report=await page.evaluate(async({builds,input,points,repeats,warmupRounds})=>{
    const file=new File([await(await fetch("/__input")).arrayBuffer()],input.name);
    const same=(a,b)=>{const x=new Uint8Array(a.buffer,a.byteOffset,a.byteLength),y=new Uint8Array(b.buffer,b.byteOffset,b.byteLength);return x.length===y.length&&x.every((v,i)=>v===y[i]);};
    const context=async mode=>{
      const build=mode==="slots"?"optimized":"baseline",worker=new Worker(`/${build}/assets/${builds[build].worker}`,{type:"module"}); let seq=0; const waiting=new Map();
      worker.onmessage=({data})=>{if(!data.response)return;const p=waiting.get(data.seq);if(!p)return;waiting.delete(data.seq);clearTimeout(p.timer);data.response.ok?p.resolve(data.response.value):p.reject(new Error(data.response.error));};
      worker.onerror=e=>{for(const p of waiting.values()){clearTimeout(p.timer);p.reject(new Error(e.message));}waiting.clear();};
      const call=req=>new Promise((resolve,reject)=>{const id=++seq,timer=setTimeout(()=>{waiting.delete(id);reject(new Error(`Timeout: ${req.kind}`));},180_000);waiting.set(id,{resolve,reject,timer});worker.postMessage({seq:id,req});});
      try{await call({kind:"cpu-prepare"});const cloud=await call({kind:"load",file,maxPoints:2_000_000});
        if(cloud.kind!=="cloud"||cloud.count!==points||cloud.filePoints!==points||cloud.keepEvery!==1)throw new Error("Input was thinned or changed");
        return{worker,call,cloud};}catch(e){worker.terminate();throw e;}
    };
    const cases=[]; let description;
    for(const method of ["sor"]){
      const contexts={};
      try{
        for(const mode of ["baseline","indexed","slots"])contexts[mode]=await context(mode);
        description={points,bounds:contexts.indexed.cloud.bounds,keepEvery:1};
        const compute=async mode=>{
          const {cloud,call}=contexts[mode],knnCache=true;
          const request=method==="normals"?{kind:method,id:cloud.id,k:24,orientation:"up",knnCache,knnBounds:false}:{kind:"filter",id:cloud.id,op:"sor",a:24,b:1.0,knnCache,knnBounds:false};
          const t=performance.now(),result=await call(request),elapsedMillis=performance.now()-t;
          const state=mode==="baseline"?null:await call({kind:"knn-cache-state"});
          if(mode!=="baseline"&&(!state||state.pending||state.points!==points||!(state.bytes>0)||state.bytes>128*1024*1024))throw new Error("Expected bounded live preparation");
          return{result,elapsedMillis,state};
        };
        const oracle=await compute("baseline");
        const expected=method==="normals"?oracle.result:await contexts.baseline.call({kind:"export",id:oracle.result.id,format:"ply"});
        const oraclePoints=method==="normals"?oracle.result.length/3:oracle.result.count;
        contexts.baseline.worker.terminate();
        const verify=async values=>{
          const a=values.indexed.result,b=values.slots.result;
          if(method==="normals"){
            if(!same(a,b)||!same(a,expected))throw new Error("Normal f32 bits changed versus indexed/slots/oracle controls");
            return{resultParity:true,archivedParity:true,outputPoints:a.length/3};
          }
          const x=await contexts.indexed.call({kind:"export",id:a.id,format:"ply"}),y=await contexts.slots.call({kind:"export",id:b.id,format:"ply"});
          if(a.count!==b.count||a.count!==oraclePoints||!same(x,y)||!same(x,expected))throw new Error("SOR kept points/PLY changed versus indexed/slots/oracle controls");
          return{resultParity:true,archivedParity:true,outputPoints:a.count,exportedBytes:x.length};
        };
        const cleanup=async values=>{if(method==="sor")for(const mode of["indexed","slots"])await contexts[mode].call({kind:"remove",id:values[mode].result.id});};
        const setup={indexed:await compute("indexed"),slots:await compute("slots")}; const firstParity=await verify(setup);
        if(setup.slots.state.reuses!==0||setup.indexed.state.reuses!==0)throw new Error("First preparation use was already warm");
        const cacheJob=setup.slots.state.job,baselineJob=setup.indexed.state.job;await cleanup(setup);
        const warmups=[],trials=[];
        for(let i=-warmupRounds;i<repeats;i++){
          const order=Math.abs(i)%2?["slots","indexed"]:["indexed","slots"],values={};
          for(const mode of order)values[mode]=await compute(mode);
          if(values.slots.state.job!==cacheJob||values.slots.state.reuses!==i+warmupRounds+1||values.indexed.state.job!==baselineJob||values.indexed.state.reuses!==i+warmupRounds+1||values.indexed.state.bytes!==values.slots.state.bytes||values.indexed.state.parts!==values.slots.state.parts)throw new Error("Preparation was rebuilt or reuse count changed");
          const parity=await verify(values);
          (i<0?warmups:trials).push({order,indexedMillis:values.indexed.elapsedMillis,slotsMillis:values.slots.elapsedMillis,...parity,cache:values.slots.state,baselineCache:values.indexed.state});
          await cleanup(values);
          await globalThis.benchmarkProgress(`${method} ${i<0?"warm-up":"trial"} ${i<0?i+warmupRounds+1:i+1}: indexed ${Math.round(values.indexed.elapsedMillis)}/slots ${Math.round(values.slots.elapsedMillis)} ms; indexed/slots/oracle parity`);
        }
        cases.push({method,excludedFirstUse:{indexedMillis:setup.indexed.elapsedMillis,slotsMillis:setup.slots.elapsedMillis,cache:setup.slots.state,baselineCache:setup.indexed.state,...firstParity},archivedOracle:{points:oraclePoints,bytes:expected.byteLength},warmups,trials});
      }finally{for(const c of Object.values(contexts))c.worker.terminate();}
    }
    return{hardwareConcurrency:navigator.hardwareConcurrency,userAgent:navigator.userAgent,...description,cases};
  },{builds:Object.fromEntries(Object.entries(builds).map(([m,b])=>[m,{worker:b.worker}])),input,points:inputManifest.inputs.compared.points,repeats,warmupRounds});
  for(const[path,hash]of [[inputPath,input.sha256],[originalPath,digest(originalBytes)],[manifestPath,digest(manifestBytes)],[fileURLToPath(import.meta.url),scriptSha256],...Object.entries(sourceHashes)])
    if(digest(readFileSync(path))!==hash)throw new Error(`Input/source/script changed during timing: ${path}`);
  for(const b of Object.values(builds))for(const[name,hash]of Object.entries(b.artifactHashes))if(digest(readFileSync(resolve(b.root,"assets",name)))!==hash)throw new Error("Production artifact changed during timing");
  Object.assign(report,{schemaVersion:1,comparison:"cpu-sor-tree-slots",createdAt:new Date().toISOString(),
    commit:execFileSync("git",["rev-parse","HEAD"],{encoding:"utf8",windowsHide:true}).trim(),workingTreeDirty:!!execFileSync("git",["status","--porcelain"],{encoding:"utf8",windowsHide:true}).trim(),
    builds,scriptSha256,sourceHashes,settings:{k:24,ratio:1,knnCache:true,knnBounds:false},browserVersion:browser.version(),machine:{platform:platform(),release:release(),cpu:cpus()[0]?.model,logicalCpus:cpus().length,memoryBytes:totalmem()},
    input,inputManifest,inputManifestSha256:digest(manifestBytes),repeats,warmupRounds,
    note:"Two independent production workers compare the archived index-normalized SOR queries with distance-only queries that leave tree slots in caller scratch. Both reuse unchanged full-cloud partitions/trees under the existing 128 MiB idle payload budget, with exact incrementing reuse counters, stable job IDs, equal lanes and capacities. The adopted sorted-prefix JavaScript join is retained by both builds. A third archived worker supplies an excluded full output oracle, then terminates before timing. Every pair checks kept count and all filtered PLY bytes against both controls and that oracle. Full unchanged measured input, k=24, ratio=1, knnCache:true, knnBounds:false. No candidate hints or GPU work added. One excluded first-use pair, five excluded alternating warm-up pairs, ten alternating timed pairs unless overridden. Whole RPC includes local/cross-part searches, region checks, messages, joining, filtering and output indexing. Startup/load, diagnostics, export/validation and removal excluded. Tree traversal, distance arithmetic, ties, order, square-root/reduction and threshold are unchanged; only the unconsumed original-index gather is skipped. Normal estimation and other indexed APIs retain normalization. Input/manifest/original measured source/ten implementation sources/script/both builds hashed before/after; no builds/tests/scans during timing. CPU clocks/power/GC/other apps uncontrolled; even-odd source subsets are not independent surveys or filter ground truth. No GPU metadata collected."});
  writeFileSync(output,JSON.stringify(report,null,2)+"\n",{flag:"wx"});console.log(`Saved ${output}`);
}finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
