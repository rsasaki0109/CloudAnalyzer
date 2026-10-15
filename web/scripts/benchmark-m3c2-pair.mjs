// Paired fresh M3C2 normal fits against archived individual production.
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
if(!readFileSync("rust/crates/ca-wasm/src/m3c2.rs","utf8").includes("const USE_M3C2_PAIR_EIGEN: bool = true;"))throw new Error("Enable paired fresh M3C2 normals and rebuild before measuring");
const sourceFiles=["rust/crates/ca-core/src/normal_pair.rs","rust/crates/ca-core/src/normals.rs","rust/crates/ca-core/src/icp.rs","rust/crates/ca-core/src/lib.rs","rust/crates/ca-core/src/map_quality.rs","web/src/m3c2-workers.ts","web/src/m3c2-preparation.ts","web/src/worker.ts","web/src/c2c-worker.ts","web/src/pool.ts","web/src/protocol.ts","rust/crates/ca-core/src/m3c2.rs","rust/crates/ca-core/src/kdtree.rs","rust/crates/ca-wasm/src/m3c2.rs","rust/crates/ca-wasm/src/lib.rs"];
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
try {
  browser = await chromium.launch({ channel: "chromium", headless: true });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report = await page.evaluate(async ({ builds, inputs, inputManifest, repeats, warmupRounds }) => {
    const params = { normalRadius: 0.4, projectionRadius: 0.25, maxDepth: 0.5, coreSpacing: 0.25 };
    const files = {};
    for (const role of ["reference", "compared"]) files[role] = new File([await (await fetch(`/__${role}`)).arrayBuffer()], inputs[role].name);
    const digest = async a => [...new Uint8Array(await crypto.subtle.digest("SHA-256", a))].map(v => v.toString(16).padStart(2,"0")).join("");
    const same = (a,b) => {
      const x = new Uint8Array(a.buffer,a.byteOffset,a.byteLength), y = new Uint8Array(b.buffer,b.byteOffset,b.byteLength);
      return x.length === y.length && x.every((v,i)=>v===y[i]);
    };
    const context = async mode => {
      const worker = new Worker(`/${mode}/assets/${builds[mode].worker}`, { type:"module" }); let seq = 0; const pending = new Map();
      worker.onmessage = ({ data }) => {
        if (!data.response) return; const p = pending.get(data.seq); if (!p) return; pending.delete(data.seq); clearTimeout(p.timer);
        data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
      };
      worker.onerror = e => { for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new Error(e.message)); } pending.clear(); };
      const call = req => new Promise((resolve,reject) => {
        const id = ++seq, timer = setTimeout(()=>{pending.delete(id);reject(new Error(`Timeout ${req.kind}`));},180_000);
        pending.set(id,{resolve,reject,timer});worker.postMessage({seq:id,req});
      });
      try {
        await call({kind:"cpu-prepare"}); const clouds = {};
        for (const role of ["reference","compared"]) {
          clouds[role] = await call({kind:"load",file:files[role],maxPoints:2_000_000}); const c = clouds[role];
          if (c.kind !== "cloud" || c.count !== inputManifest.inputs[role].points || c.filePoints !== c.count || c.keepEvery !== 1) throw new Error("Input point selection changed");
        }
        return {worker,call,clouds};
      } catch (error) { worker.terminate(); throw error; }
    };
    const request = (ctx,cacheNormals,fusedProjection) => ({ kind:"m3c2",reference:ctx.clouds.reference.id,compared:ctx.clouds.compared.id,...params,
      radiusBounds:false,sharedSearch:true,mainWorker:true,preparedTrees:true,interleaved:true,cacheNormals,fusedProjection });
    const baseline = await context("baseline"); let oracle, oracleBytes;
    try {
      oracle = await baseline.call(request(baseline,false,false));
      if (!oracle.preparedTrees || !oracle.mainWorker || oracle.workers < 2 || oracle.normalCacheHit) throw new Error("Expected archived prepared M3C2 oracle");
      oracleBytes = await baseline.call({kind:"export",id:oracle.cloud.id,format:"ply"});
    } finally { baseline.worker.terminate(); }
    const oracleHashes = Object.fromEntries(await Promise.all(["distance","lod95","significant"].map(async k=>[k,await digest(oracle[k])])));
    oracleHashes.ply = await digest(oracleBytes);
    const cases = [];
    for (const cacheNormals of [false,true]) {
      const ctxs = { single:await context("baseline"), paired:await context("current") }, name = cacheNormals ? "reused-normals" : "fresh-normals";
      try {
        const compute = async (mode,warm=true) => {
          const ctx=ctxs[mode], fusedProjection = cacheNormals ? undefined : false;
          const start = performance.now(), r = await ctx.call(request(ctx,cacheNormals,fusedProjection)), elapsedMillis = performance.now()-start;
          if (!r.preparedTrees || !r.mainWorker || r.workers !== oracle.workers || r.radiusBounds || !r.sharedSearch || r.fusedProjection !== (warm && cacheNormals) ||
            r.normalCacheHit !== (warm && cacheNormals) || r.normalCacheBytes > 64*1024*1024 || !cacheNormals && r.normalCacheBytes || warm && !r.treeCacheHit) throw new Error("Unexpected M3C2 control/cache/layout");
          if (r.cloud.count !== oracle.cloud.count || ["distance","lod95","significant"].some(k=>!same(r[k],oracle[k]))) throw new Error("M3C2 result bits changed");
          const bytes = await ctx.call({kind:"export",id:r.cloud.id,format:"ply"});
          if (!same(bytes,oracleBytes)) throw new Error("M3C2 complete PLY changed");
          await ctx.call({kind:"remove",id:r.cloud.id});
          return {elapsedMillis,workerMillis:r.millis,workers:r.workers,treeCacheHit:r.treeCacheHit,normalCacheHit:r.normalCacheHit,normalCacheBytes:r.normalCacheBytes,fusedProjection:r.fusedProjection,resultParity:true};
        };
        const firstUse = {single:await compute("single",false),paired:await compute("paired",false)};
        if (firstUse.single.treeCacheHit || firstUse.paired.treeCacheHit) throw new Error("Unexpected cold tree reuse");
        const checkPair=values=>{if(values.single.normalCacheBytes!==values.paired.normalCacheBytes)throw new Error("Normal allowance differs between builds");};checkPair(firstUse);
        const warmups=[],trials=[];
        for (let i=-warmupRounds;i<repeats;i++) {
          const order = (i+warmupRounds)%2 ? ["paired","single"] : ["single","paired"], values={};
          for (const mode of order) values[mode] = await compute(mode);
          checkPair(values);(i<0?warmups:trials).push({order,...values});
          await globalThis.benchmarkProgress(`M3C2 ${name} ${i<0?"warm-up":"trial"} ${i<0?i+warmupRounds+1:i+1}: single ${values.single.elapsedMillis.toFixed(1)}, paired ${values.paired.elapsedMillis.toFixed(1)} ms; parity`);
        }
        cases.push({name,cacheNormals,firstUse,warmups,trials});
      } finally { for(const ctx of Object.values(ctxs))ctx.worker.terminate(); }
    }
    return {params,hardwareConcurrency:navigator.hardwareConcurrency,userAgent:navigator.userAgent,workers:oracle.workers,cores:oracle.cloud.count,
      finiteDistances:oracle.distance.filter(Number.isFinite).length,oracleHashes,cases};
  },{builds:Object.fromEntries(Object.entries(builds).map(([k,b])=>[k,{worker:b.worker}])),inputs,inputManifest,repeats,warmupRounds});
  for (const [name,b] of Object.entries(builds)) for (const [file,expected] of Object.entries(b.artifactHashes)) if (hash(readFileSync(resolve(b.root,"assets",file)))!==expected) throw new Error(`Changed ${name} artifact`);
  for (const [p,expected] of Object.entries(sourceHashes)) if (hash(readFileSync(resolve(p)))!==expected) throw new Error(`Changed source ${p}`);
  for (const role of ["reference","compared"]) if (hash(readFileSync(inputs[role].path))!==inputs[role].sha256) throw new Error("Input changed during timing");
  if (hash(readFileSync(manifestPath))!==hash(manifestBytes) || hash(readFileSync(sourcePath))!==inputManifest.source.sha256 || hash(readFileSync(scriptPath))!==scriptSha256) throw new Error("Provenance/script changed during timing");
  const median = a => { a=[...a].sort((x,y)=>x-y);return a.length%2?a[a.length>>1]:(a[a.length/2-1]+a[a.length/2])/2; };
  for (const c of report.cases) {
    const reductions = c.trials.map(t=>100*(1-t.paired.elapsedMillis/t.single.elapsedMillis));
    c.summary = {singleMedianMillis:median(c.trials.map(t=>t.single.elapsedMillis)),pairedMedianMillis:median(c.trials.map(t=>t.paired.elapsedMillis)),medianPairedReductionPercent:median(reductions),fasterTrials:reductions.filter(x=>x>0).length,trials:repeats};
  }
  Object.assign(report,{schemaVersion:1,comparison:"paired-fresh-m3c2-eigen",createdAt:new Date().toISOString(),builds,scriptSha256,sourceHashes,inputs,inputManifest,inputManifestSha256:hash(manifestBytes),repeats,warmupRounds,browserVersion:browser.version(),
    commit:execFileSync("git",["rev-parse","HEAD"],{encoding:"utf8",windowsHide:true}).trim(),workingTreeDirty:!!execFileSync("git",["status","--porcelain"],{encoding:"utf8",windowsHide:true}).trim(),
    machine:{platform:platform(),release:release(),cpu:cpus()[0]?.model,logicalCpus:cpus().length,memoryBytes:totalmem()},
    note:"Full unchanged measured source-index split inputs; no generated/transformed/thinned coordinates. Original-point core representatives at spacing 0.25 m, not independent surveys/change ground truth. Normal radius 0.4, projection radius 0.25, half-depth 0.5; bounds off, shared reference search on, prepared trees and interleaving on, same lane count and 64 MiB normal allowance. Archived individual and current paired production use independent workers per fitting case. Fresh fitting disables normal reuse and projection fusion in BOTH builds; cached fitting enables normal retention and both builds automatically fuse projection on matching normal-cache hits. Two excluded first observations build each context trees and optionally capture normals, then five alternating warm-up pairs precede ten alternating timed pairs. Candidate pairs two independent fresh normal eigensolves only in SIMD-enabled prepared WASM, while retaining both cores reference hit lists and original projection arithmetic/order. Cached normals, odd tails, unsafe matrices and unsupported targets keep individual kernels. Additional per-slice hit buffers and scheduling/allocation/code-layout effects are included. Cached timing is a separate control without new pair fitting. Independent archived worker supplies full distance/LoD/significance/PLY oracle and terminates before timing; both timed results require exact output bits and PLY bytes against it. Whole RPC includes core planning, bindings/transfers, radius queries, moments/eigen fitting, projection, join, attributes/LOD and result arrays. Startup/loading/first observations/warmups/export/parity/removal excluded. Source, artifacts, inputs, manifest, original source and script checked before/after. No builds/tests/scans alongside timing. Clocks/power/scheduling/GC/other applications/WASM pages uncontrolled. No GPU dispatch/model identifiers collected."});
  writeFileSync(output,JSON.stringify(report,null,2)+"\n",{flag:"wx"});console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve=>server.close(resolve)); }
