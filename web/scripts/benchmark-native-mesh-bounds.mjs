// Run from the repository root. Two archived native binaries alternate serially.
import { spawn, execFileSync } from "node:child_process";
import { createInterface } from "node:readline";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { cpus, totalmem, platform, release } from "node:os";
import { fileURLToPath } from "node:url";

const args=process.argv.slice(2),allowed=["--baseline-root","--candidate-root","--reference","--compared","--input-manifest","--output","--threads","--repeats"];
for(let i=0;i<args.length;i+=2)if(!allowed.includes(args[i])||!args[i+1]||args[i+1].startsWith("--")||args.indexOf(args[i])!==i)throw new Error(`Invalid or repeated option: ${args[i]}`);
const value=(k,f)=>args.includes(k)?args[args.indexOf(k)+1]:f;
const output=value("--output"),manifestPath=value("--input-manifest"),repeats=Number(value("--repeats","10")),threads=Number(value("--threads",String(cpus().length))),warmupRounds=5;
for(const k of ["--baseline-root","--candidate-root","--reference","--compared","--input-manifest","--output"])if(!value(k))throw new Error("Supply --baseline-root DIR --candidate-root DIR --reference PLY --compared PLY --input-manifest JSON --output NEW.json [--threads 12] [--repeats 10]");
if(existsSync(output))throw new Error(`Report already exists: ${output}`);
if(!Number.isInteger(repeats)||repeats<1||repeats>100||!Number.isInteger(threads)||threads<1||threads>64)throw new Error("Invalid repeats/threads");
const hash=bytes=>createHash("sha256").update(bytes).digest("hex"),hashFile=p=>hash(readFileSync(p));
const manifestBytes=readFileSync(manifestPath),inputManifest=JSON.parse(manifestBytes),inputs={};
for(const role of ["reference","compared"]){
  const path=resolve(value(`--${role}`)),bytes=readFileSync(path),e=inputManifest.inputs?.[role];
  if(!e||hash(bytes)!==e.sha256||bytes.length!==e.bytes||!Number.isInteger(e.points)||e.points<200_000||e.points>1_000_000)throw new Error(`Expected full measured ${role} input matching its manifest`);
  inputs[role]={path,bytes:bytes.length,sha256:hash(bytes),points:e.points};
}
const original=resolve("web",inputManifest.source.path);
if(hashFile(original)!==inputManifest.source.sha256||readFileSync(original).length!==inputManifest.source.bytes)throw new Error("Original source differs from manifest");
const sourceFiles=["rust/crates/ca-core/src/distance.rs","rust/crates/ca-core/src/mesh.rs","rust/crates/ca-core/src/lib.rs","rust/crates/ca-core/src/delaunay.rs","rust/crates/ca-core/src/io/write.rs","rust/crates/ca-core/src/io/ply.rs","rust/crates/ca-core/src/io/mod.rs","rust/crates/ca-core/examples/bench_c2m_source.rs","rust/crates/ca-py/src/lib.rs","rust/crates/ca-core/Cargo.toml","rust/Cargo.toml","rust/Cargo.lock"];
const builds={};
for(const mode of ["baseline","bounded"]){
  const root=resolve(value(mode==="baseline"?"--baseline-root":"--candidate-root")),exe=resolve(root,"bench_c2m_source.exe");
  builds[mode]={root,exe,exeSha256:hashFile(exe),compiler:readFileSync(resolve(root,"compiler.txt"),"utf8"),sourceHashes:Object.fromEntries(sourceFiles.map(p=>[p,hashFile(resolve(root,p))]))};
}
const distancePath=sourceFiles[0],before=readFileSync(resolve(builds.baseline.root,distancePath),"utf8"),after=readFileSync(resolve(builds.bounded.root,distancePath),"utf8");
if(!before.includes("const USE_PAR_MESH_STACK_BOUNDS: bool = false;")||before.replace("const USE_PAR_MESH_STACK_BOUNDS: bool = false;","const USE_PAR_MESH_STACK_BOUNDS: bool = true;")!==after)throw new Error("Expected only native cached-bound flag to differ");
for(const text of [before,after])if(!text.includes("const USE_PAR_MESH_STACK_REUSE: bool = true;"))throw new Error("Keep adopted native stack reuse enabled");
for(const p of sourceFiles.slice(1))if(builds.baseline.sourceHashes[p]!==builds.bounded.sourceHashes[p])throw new Error(`Changed unrelated source ${p}`);
if(builds.baseline.exeSha256===builds.bounded.exeSha256)throw new Error("Expected distinct archived executables");
if(builds.baseline.compiler!==builds.bounded.compiler)throw new Error("Changed compiler");
const scriptPath=fileURLToPath(import.meta.url),scriptSha256=hashFile(scriptPath);
const scratch=resolve("notes",`native-mesh-bounds-run-${randomUUID()}`);mkdirSync(scratch);
const children=[];
function start(mode,label){
  const dir=resolve(scratch,label),queue=[],waiters=[];
  const child=spawn(builds[mode].exe,[inputs.reference.path,inputs.compared.path,dir,String(threads)],{stdio:["pipe","pipe","pipe"],windowsHide:true});children.push(child);
  let failure,stderr="";
  const fail=e=>{failure=e;for(const w of waiters.splice(0)){clearTimeout(w.timer);w.reject(e);}};
  child.stderr.on("data",b=>{stderr=(stderr+b.toString()).slice(-4000);});child.on("error",fail);
  child.on("exit",code=>{if(code!==0)fail(new Error(`${label} exited ${code}: ${stderr}`));else fail(new Error(`${label} finished`));});
  createInterface({input:child.stdout}).on("line",line=>{
    let v;try{v=JSON.parse(line);}catch{fail(new Error(`Invalid protocol from ${label}: ${line}`));return;}
    const w=waiters.shift();if(w){clearTimeout(w.timer);w.resolve(v);}else queue.push(v);
  });
  const next=()=>failure?Promise.reject(failure):queue.length?Promise.resolve(queue.shift()):new Promise((resolve,reject)=>{
    const w={resolve,reject,timer:null};w.timer=setTimeout(()=>{fail(new Error(`Timeout ${label}: ${stderr}`));child.kill();},180_000);waiters.push(w);
  });
  return {dir,child,next,async close(){const closed=new Promise(r=>child.once("exit",r));child.stdin.end("quit\n");await closed;}};
}
const sameReady=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const resultProof=(ctx,r)=>{
  if(r.kind!=="result"||r.signed!==false||r.count!==inputs.compared.points||!Number.isFinite(r.elapsedMillis)||r.elapsedMillis<=0||r.statisticsBits?.length!==6)throw new Error("Unexpected native C2M result");
  const raw=readFileSync(resolve(ctx.dir,"distances-f64.bin")),ply=readFileSync(resolve(ctx.dir,"distances.ply"));
  if(raw.length!==r.count*8)throw new Error("Incomplete raw f64 output");
  return {distanceBytes:raw.length,distanceSha256:hash(raw),scalarPlyBytes:ply.length,scalarPlySha256:hash(ply),count:r.count,statisticsBits:r.statisticsBits};
};
async function compute(ctx){const result=ctx.next();ctx.child.stdin.write("unsigned\n");const r=await result;return{r,proof:resultProof(ctx,r)};}
const median=xs=>{const a=[...xs].sort((x,y)=>x-y);return(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
let report;
try{
  // Independent preceding-binary oracle exits before either timed context starts.
  const oracle=start("baseline","oracle"),ready=await oracle.next();
  if(ready.kind!=="ready"||ready.threads!==threads||ready.chunkSize!==4096||ready.referencePoints!==inputs.reference.points||ready.queryPoints!==inputs.compared.points||!ready.triangles)throw new Error("Changed input/layout");
  const oracleGeometry=readFileSync(resolve(oracle.dir,"geometry.ply")),geometry={bytes:oracleGeometry.length,sha256:hash(oracleGeometry)};
  const oracleResult=await compute(oracle);await oracle.close();
  const contexts={};
  for(const mode of ["baseline","bounded"]){
    const ctx=start(mode,mode);contexts[mode]=ctx;
    if(!sameReady(ready,await ctx.next())||!oracleGeometry.equals(readFileSync(resolve(ctx.dir,"geometry.ply"))))throw new Error("Changed native TIN or thread layout");
  }
  const run=async mode=>{
    const {r,proof}=await compute(contexts[mode]);
    if(JSON.stringify(proof)!==JSON.stringify(oracleResult.proof))throw new Error(`Changed full output or exact summaries in ${mode}`);
    return{elapsedMillis:r.elapsedMillis,...proof,resultParity:true};
  };
  const firstUse={baseline:await run("baseline"),bounded:await run("bounded")},warmups=[],trials=[];
  for(let i=-warmupRounds;i<repeats;i++){
    const order=(i+warmupRounds)%2?["bounded","baseline"]:["baseline","bounded"],values={};for(const mode of order)values[mode]=await run(mode);
    (i<0?warmups:trials).push({order,...values});
    console.log(`Native C2M ${i<0?"warm-up":"trial"} ${i<0?i+warmupRounds+1:i+1}: baseline ${values.baseline.elapsedMillis.toFixed(1)}, bounded ${values.bounded.elapsedMillis.toFixed(1)} ms; full f64/summary/PLY parity`);
  }
  for(const ctx of Object.values(contexts))await ctx.close();
  const reductions=trials.map(t=>100*(1-t.bounded.elapsedMillis/t.baseline.elapsedMillis));
  report={schemaVersion:1,comparison:"native-c2m-chunk-cached-bounds",createdAt:new Date().toISOString(),builds,scriptSha256,inputs,inputManifest,inputManifestSha256:hash(manifestBytes),threads,repeats,warmupRounds,mesh:ready,geometry,oracle:oracleResult.proof,firstUse,warmups,trials,
    summary:{baselineMedianMillis:median(trials.map(t=>t.baseline.elapsedMillis)),boundedMedianMillis:median(trials.map(t=>t.bounded.elapsedMillis)),medianPairedReductionPercent:median(reductions),fasterTrials:reductions.filter(x=>x>0).length,trials:repeats},
    commit:execFileSync("git",["rev-parse","HEAD"],{encoding:"utf8",windowsHide:true}).trim(),workingTreeDirty:!!execFileSync("git",["status","--porcelain"],{encoding:"utf8",windowsHide:true}).trim(),
    machine:{platform:platform(),release:release(),cpu:cpus()[0]?.model,logicalCpus:cpus().length,memoryBytes:totalmem()},
    note:"Full unchanged measured-source XYZ, not independent surveys or accuracy ground truth. Native PLY input order is preserved; it is distinct from browser LOD indexing. Full-input production Delaunay TIN uses automatic edge rejection, with no extra thinning or transformed/generated coordinates. Independent archived native baseline supplies complete f64 distance bits, f64 statistics bits and scalar PLY oracle, then exits before timing. Separate archived baseline/candidate contexts share identical TIN, 4096-point original Morton chunks and fixed Rayon thread count. Each timed whole native cloud_to_mesh_par includes fresh BVH build, query ordering, parallel searches, result join and BVH destruction. No retained geometry/BVH across calls. Both builds already reuse per-chunk traversal scratch. Candidate stack entries retain each child f64 box distance computed for traversal ordering, avoiding its recomputation on pop; pop pruning still uses the current best hit. Scratch/guess reset at chunk/call boundaries, and missing-hit recovery stays intact. Near-child ordering, triangle arithmetic, strict insertion and signed behavior remain unchanged. Stack entries grow from one usize to (usize,f64); no new persistent geometry, query or result cache is retained. This is not a peak process memory guarantee. Loading/meshing/thread-pool construction, first use, five alternating warmups, f64/summary/PLY serialization/verification and cleanup are excluded. Ten alternating timed pairs are retained without outlier removal. No builds/tests/scans/edits alongside timing. Source, executable, script, input/manifest/original PCD hashes are checked before/after. Clocks/power/scheduling/other apps/allocator/code layout/pages uncontrolled. No Python conversion, browser RPC, WASM, GPU or isolated traversal speedup claim. No GPU model identifiers collected."};
  for(const b of Object.values(builds)){
    if(hashFile(b.exe)!==b.exeSha256)throw new Error("Executable changed during timing");
    for(const [p,h] of Object.entries(b.sourceHashes))if(hashFile(resolve(b.root,p))!==h)throw new Error(`Archived source changed ${p}`);
  }
  for(const input of Object.values(inputs))if(hashFile(input.path)!==input.sha256||readFileSync(input.path).length!==input.bytes)throw new Error("Input changed during timing");
  if(hashFile(original)!==inputManifest.source.sha256||hashFile(manifestPath)!==hash(manifestBytes)||hashFile(scriptPath)!==scriptSha256)throw new Error("Original/manifest/script changed during timing");
  writeFileSync(output,JSON.stringify(report,null,2)+"\n",{flag:"wx"});console.log(`Saved ${output}`);
}finally{for(const child of children)if(child.exitCode===null)child.kill();}
