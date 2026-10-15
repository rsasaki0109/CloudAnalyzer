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
for(const mode of ["baseline","clamped"]){
  const root=resolve(value(mode==="baseline"?"--baseline-root":"--candidate-root")),exe=resolve(root,"bench_c2m_source.exe");
  builds[mode]={root,exe,exeSha256:hashFile(exe),compiler:readFileSync(resolve(root,"compiler.txt"),"utf8"),sourceHashes:Object.fromEntries(sourceFiles.map(p=>[p,hashFile(resolve(root,p))]))};
}
const distancePath=sourceFiles[0],before=readFileSync(resolve(builds.baseline.root,distancePath),"utf8"),after=readFileSync(resolve(builds.clamped.root,distancePath),"utf8");
if(!before.includes("const USE_PAR_MESH_BOUND_PAIR_CLAMP: bool = false;")||before.replace("const USE_PAR_MESH_BOUND_PAIR_CLAMP: bool = false;","const USE_PAR_MESH_BOUND_PAIR_CLAMP: bool = true;")!==after)throw new Error("Expected only native bound-pair-clamp flag to differ");
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[0]),"utf8");for(const flag of ["USE_PAR_MESH_STACK_REUSE","USE_PAR_MESH_STACK_BOUNDS"])if(!text.includes(`const ${flag}: bool = true;`))throw new Error("Keep adopted native traversal enabled");}
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[1]),"utf8");for(const flag of ["USE_MESH_PAR_PREPARATION","USE_MESH_BOUND_MERGE","USE_MESH_PAR_BUILD"])if(!text.includes(`const ${flag}: bool = true;`))throw new Error("Keep adopted mesh preparation, bounds and builder enabled");}
for(const p of sourceFiles.filter(p=>p!==distancePath))if(builds.baseline.sourceHashes[p]!==builds.clamped.sourceHashes[p])throw new Error(`Changed unrelated source ${p}`);
if(builds.baseline.exeSha256===builds.clamped.exeSha256)throw new Error("Expected distinct archived executables");
if(builds.baseline.compiler!==builds.clamped.compiler)throw new Error("Changed compiler");
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[0]),"utf8");if(!text.includes("const USE_PAR_MESH_SKIP_GUESS: bool = false;"))throw new Error("Keep rejected seed-triangle experiment disabled");}
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[0]),"utf8");if(!text.includes("const USE_PAR_MESH_MORTON_PREPARATION: bool = false;"))throw new Error("Keep rejected Morton experiment disabled");}
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[0]),"utf8");if(!text.includes("const USE_PAR_MESH_COMPACT_RESULTS: bool = true;"))throw new Error("Keep adopted compact results enabled");}
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[0]),"utf8");if(!text.includes("const USE_PAR_MESH_PAR_RESULTS: bool = false;"))throw new Error("Keep rejected parallel restoration disabled");}
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[0]),"utf8");if(!text.includes("const USE_PAR_MESH_EAGER_PRUNE: bool = false;"))throw new Error("Keep rejected push-time pruning disabled");}
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[0]),"utf8");if(!text.includes("const USE_PAR_MESH_TRIANGLE_SEED: bool = false;"))throw new Error("Keep rejected triangle seeds disabled");}
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[1]),"utf8");if(!text.includes("const USE_MESH_PAR_CENTROID_BOUNDS: bool = false;"))throw new Error("Keep rejected centroid scans disabled");}
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[0]),"utf8");if(!text.includes("const USE_PAR_MESH_INDEXED_CORNERS: bool = false;"))throw new Error("Keep rejected original-vertex access disabled");}
for(const mode of ["baseline","clamped"]){const text=readFileSync(resolve(builds[mode].root,sourceFiles[0]),"utf8");if(!text.includes("const USE_PAR_MESH_BOUND_PAIR: bool = false;"))throw new Error("Keep masked sibling-bound pairing disabled");}
const scriptPath=fileURLToPath(import.meta.url),scriptSha256=hashFile(scriptPath);
const scratch=resolve("notes",`native-mesh-bound-pair-clamp-run-${randomUUID()}`);mkdirSync(scratch);
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
  return {dir,child,next,async close(){if(child.exitCode!==null){if(child.exitCode!==0)throw new Error(`Failed process ${label}`);return;}const closed=new Promise((resolve,reject)=>child.once("exit",code=>code===0?resolve():reject(new Error(`Failed process ${label}`))));child.stdin.end("quit\n");await closed;}};
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
  for(const mode of ["baseline","clamped"]){
    const ctx=start(mode,mode);contexts[mode]=ctx;
    if(!sameReady(ready,await ctx.next())||!oracleGeometry.equals(readFileSync(resolve(ctx.dir,"geometry.ply"))))throw new Error("Changed native TIN or thread layout");
  }
  const run=async mode=>{
    const {r,proof}=await compute(contexts[mode]);
    if(JSON.stringify(proof)!==JSON.stringify(oracleResult.proof))throw new Error(`Changed full output or exact summaries in ${mode}`);
    return{elapsedMillis:r.elapsedMillis,...proof,resultParity:true};
  };
  const firstUse={baseline:await run("baseline"),clamped:await run("clamped")},warmups=[],trials=[];
  for(let i=-warmupRounds;i<repeats;i++){
    const order=(i+warmupRounds)%2?["clamped","baseline"]:["baseline","clamped"],values={};for(const mode of order)values[mode]=await run(mode);
    (i<0?warmups:trials).push({order,...values});
    console.log(`Native C2M ${i<0?"warm-up":"trial"} ${i<0?i+warmupRounds+1:i+1}: baseline ${values.baseline.elapsedMillis.toFixed(1)}, clamped ${values.clamped.elapsedMillis.toFixed(1)} ms; full f64/summary/PLY parity`);
  }
  for(const ctx of Object.values(contexts))await ctx.close();
  const reductions=trials.map(t=>100*(1-t.clamped.elapsedMillis/t.baseline.elapsedMillis));
  report={schemaVersion:1,comparison:"native-c2m-bound-pair-clamp",createdAt:new Date().toISOString(),builds,scriptSha256,inputs,inputManifest,inputManifestSha256:hash(manifestBytes),threads,repeats,warmupRounds,mesh:ready,geometry,oracle:oracleResult.proof,firstUse,warmups,trials,
    summary:{baselineMedianMillis:median(trials.map(t=>t.baseline.elapsedMillis)),clampedMedianMillis:median(trials.map(t=>t.clamped.elapsedMillis)),medianPairedReductionPercent:median(reductions),fasterTrials:reductions.filter(x=>x>0).length,trials:repeats},
    commit:execFileSync("git",["rev-parse","HEAD"],{encoding:"utf8",windowsHide:true}).trim(),workingTreeDirty:!!execFileSync("git",["status","--porcelain"],{encoding:"utf8",windowsHide:true}).trim(),
    machine:{platform:platform(),release:release(),cpu:cpus()[0]?.model,logicalCpus:cpus().length,memoryBytes:totalmem()},
    note:"Full unchanged measured-source XYZ, not independent surveys or accuracy ground truth. Native PLY order differs from browser LOD indexing. Production full-input Delaunay TIN uses the same automatic edge rejection and duplicate-XY handling, with no extra thinning or generated/transformed coordinates. An independent archived native baseline supplies complete f64 distance bits, six statistics bits and scalar PLY oracle, then exits before timed contexts start. Separate baseline/candidate contexts retain identical TIN and fixed Rayon pools; only one computes at a time. Whole native cloud_to_mesh_par includes fresh BVH construction, original serial Morton preparation/sorting, parallel queries in the original 4096-point chunks, compact result allocation and serial scatter, BVH destruction and pool installation. Both builds keep adopted stack reuse, cached bounds, indexed centroid preparation, bottom-up bounds, parallel BVH construction and compact query results enabled, with original-vertex access, duplicate-seed skipping, Morton, parallel restoration, push-time pruning, triangle-seed, centroid-bound and preceding masked sibling-bound experiments disabled. Candidate pairs left/right child box lower bounds in two independent f64 SSE2 lanes on native x86_64. SSE2 is part of that target baseline; other targets retain two scalar calls. MAXPD(below,0) and MAXPD(above,0) each map NaN/negative/zero to positive zero; a third MAXPD selects their nonnegative maximum. This folds NaN handling into clamping rather than using a separate unordered mask. The final square and initial zero addition erase intermediate zero-sign differences. Each lane retains original differences and 0+dx*dx+dy*dy+dz*dz reduction order without FMA, horizontal regrouping or precision conversion. Intrinsics operate on register values without pointers or unchecked memory access. Root bounds, nodes/order/corners/capacities, near-first traversal, pop-time pruning, duplicate seed leaf tests, strict hit insertion, triangle projection/sign/missing-hit arithmetic, chunk-local warm states and result order remain unchanged. Selection is native parallel C2M only; ordinary serial/prepared/WASM/hinted queries, GPU exports and native C2C retain their preceding path. There is no new storage or persistent cache. Baseline retains the preceding scalar sibling calculations, with both sibling SIMD flags disabled; this comparison is not against the preceding masked candidate. Loading, meshing, pool construction, first use, five alternating warmups, serialization, statistics/parity and cleanup are excluded. Ten alternating timed pairs are retained without outlier removal. No builds/tests/scans/edits run alongside timing. Sources/executables/compiler/script/inputs/manifests/original PCD are checked before/after. Clocks, power, scheduling, other applications, allocator, code layout and retained pages are uncontrolled. No isolated box-distance, retained-mesh query, Python conversion, browser, WASM or GPU speedup claim. No GPU model identifiers collected."};
  for(const b of Object.values(builds)){
    if(hashFile(b.exe)!==b.exeSha256||readFileSync(resolve(b.root,"compiler.txt"),"utf8")!==b.compiler)throw new Error("Executable/compiler changed during timing");
    for(const [p,h] of Object.entries(b.sourceHashes))if(hashFile(resolve(b.root,p))!==h)throw new Error(`Archived source changed ${p}`);
  }
  for(const input of Object.values(inputs))if(hashFile(input.path)!==input.sha256||readFileSync(input.path).length!==input.bytes)throw new Error("Input changed during timing");
  if(hashFile(original)!==inputManifest.source.sha256||hashFile(manifestPath)!==hash(manifestBytes)||hashFile(scriptPath)!==scriptSha256)throw new Error("Original/manifest/script changed during timing");
  writeFileSync(output,JSON.stringify(report,null,2)+"\n",{flag:"wx"});console.log(`Saved ${output}`);
}finally{for(const child of children)if(child.exitCode===null)child.kill();}
