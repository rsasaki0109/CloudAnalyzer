// Native stage diagnostics. Run from the repository root with frozen builds.
import {spawn,execFileSync} from 'node:child_process';
import {createInterface} from 'node:readline';
import {readFileSync,writeFileSync,existsSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {cpus,totalmem,platform,release} from 'node:os';
const args=process.argv.slice(2),allowed=['--baseline-root','--profile-root','--reference','--compared','--input-manifest','--output','--threads','--repeats'];
for(let i=0;i<args.length;i+=2)if(!allowed.includes(args[i])||!args[i+1]||args[i+1].startsWith('--')||args.indexOf(args[i])!==i)throw new Error('Invalid or repeated option');
const value=(k,d)=>args.includes(k)?args[args.indexOf(k)+1]:d;
for(const k of allowed.slice(0,6))if(!value(k))throw new Error('Supply baseline/profile roots, reference/compared PLY, input manifest and new output');
const output=value('--output'),threads=Number(value('--threads','12')),repeats=Number(value('--repeats','10')),warmupRounds=5;
if(existsSync(output))throw new Error('Report already exists');
if(!Number.isInteger(threads)||threads<1||threads>64||!Number.isInteger(repeats)||repeats<1||repeats>100)throw new Error('Invalid repeats/threads');
const bytesHash=b=>createHash('sha256').update(b).digest('hex'),hash=p=>bytesHash(readFileSync(p));
const manifestPath=value('--input-manifest'),manifestBytes=readFileSync(manifestPath),inputManifest=JSON.parse(manifestBytes),inputs={};
for(const role of ['reference','compared']){
 const path=resolve(value('--'+role)),b=readFileSync(path),entry=inputManifest.inputs?.[role];
 if(!entry||b.length!==entry.bytes||bytesHash(b)!==entry.sha256||!Number.isInteger(entry.points)||entry.points<200000||entry.points>1000000)throw new Error('Expected full measured-source input');
 inputs[role]={path,bytes:b.length,sha256:bytesHash(b),points:entry.points};
}
const original=resolve('web',inputManifest.source.path);
if(hash(original)!==inputManifest.source.sha256||readFileSync(original).length!==inputManifest.source.bytes)throw new Error('Original source changed');
const sourceFiles=['rust/crates/ca-core/src/distance.rs','rust/crates/ca-core/src/mesh.rs','rust/crates/ca-core/src/lib.rs','rust/crates/ca-core/src/delaunay.rs','rust/crates/ca-core/src/io/write.rs','rust/crates/ca-core/src/io/ply.rs','rust/crates/ca-core/src/io/mod.rs','rust/crates/ca-core/examples/bench_c2m_source.rs','rust/crates/ca-py/src/lib.rs','rust/crates/ca-core/Cargo.toml','rust/Cargo.toml','rust/Cargo.lock'];
const builds={};
for(const mode of ['baseline','profile']){
 const root=resolve(value(mode==='baseline'?'--baseline-root':'--profile-root')),exe=resolve(root,mode==='baseline'?'bench_c2m_source.exe':'profile_c2m_stages.exe');
 builds[mode]={root,exe,exeSha256:hash(exe),compiler:readFileSync(resolve(root,'compiler.txt'),'utf8'),sourceHashes:Object.fromEntries(sourceFiles.map(p=>[p,hash(resolve(root,p))]))};
}
for(const p of sourceFiles)if(builds.baseline.sourceHashes[p]!==builds.profile.sourceHashes[p])throw new Error('Production source changed '+p);
if(builds.baseline.compiler!==builds.profile.compiler)throw new Error('Compiler changed');
const examplePath='rust/crates/ca-core/examples/profile_c2m_stages.rs',exampleSha256=hash(resolve(builds.profile.root,examplePath));
for(const b of Object.values(builds)){
 const d=readFileSync(resolve(b.root,sourceFiles[0]),'utf8'),m=readFileSync(resolve(b.root,sourceFiles[1]),'utf8');
 for(const flag of ['USE_PAR_MESH_STACK_REUSE','USE_PAR_MESH_STACK_BOUNDS','USE_PAR_MESH_COMPACT_RESULTS'])if(!d.includes(`const ${flag}: bool = true;`))throw new Error('Adopted native policy disabled');
 for(const flag of ['USE_PAR_MESH_INDEXED_CORNERS','USE_PAR_MESH_TRIANGLE_SEED','USE_PAR_MESH_EAGER_PRUNE','USE_PAR_MESH_SKIP_GUESS','USE_PAR_MESH_MORTON_PREPARATION','USE_PAR_MESH_PAR_RESULTS'])if(!d.includes(`const ${flag}: bool = false;`))throw new Error('Rejected native experiment enabled');
 for(const flag of ['USE_MESH_PAR_PREPARATION','USE_MESH_BOUND_MERGE','USE_MESH_PAR_BUILD'])if(!m.includes(`const ${flag}: bool = true;`))throw new Error('Adopted builder disabled');
 if(!m.includes('const USE_MESH_PAR_CENTROID_BOUNDS: bool = false;'))throw new Error('Rejected centroid scan enabled');
}
const scriptPath=fileURLToPath(import.meta.url),scriptSha256=hash(scriptPath),scratch=resolve('notes','native-mesh-stage-profile-'+randomUUID());mkdirSync(scratch);
const children=[];
function start(mode,label){
 const dir=resolve(scratch,label),queue=[],waiters=[];let failure,stderr='';
 const child=spawn(builds[mode].exe,[inputs.reference.path,inputs.compared.path,dir,String(threads)],{stdio:['pipe','pipe','pipe'],windowsHide:true});children.push(child);
 const fail=e=>{failure=e;for(const w of waiters.splice(0)){clearTimeout(w.timer);w.reject(e);}};
 child.stderr.on('data',b=>stderr=(stderr+b.toString()).slice(-4000));child.on('error',fail);
 child.on('exit',code=>fail(new Error(label+' exited '+code+': '+stderr)));
 createInterface({input:child.stdout}).on('line',line=>{let result;try{result=JSON.parse(line);}catch{fail(new Error('Invalid protocol'));return;}const w=waiters.shift();if(w){clearTimeout(w.timer);w.resolve(result);}else queue.push(result);});
 const next=()=>failure?Promise.reject(failure):queue.length?Promise.resolve(queue.shift()):new Promise((resolve,reject)=>{const w={resolve,reject,timer:null};w.timer=setTimeout(()=>{fail(new Error('Timeout '+label));child.kill();},180000);waiters.push(w);});
 return{child,dir,next,async close(){
  if(child.exitCode!==null){if(child.exitCode!==0)throw new Error('Failed process '+label);return;}
  const done=new Promise((resolve,reject)=>child.once('exit',code=>code===0?resolve():reject(new Error('Failed process '+label))));
  child.stdin.end('quit\n');await done;
 }};
}
function proof(ctx,r){
 if(r.kind!=='result'||r.signed!==false||r.count!==inputs.compared.points||!Number.isFinite(r.elapsedMillis)||r.elapsedMillis<=0||r.statisticsBits?.length!==6)throw new Error('Unexpected output');
 const raw=readFileSync(resolve(ctx.dir,'distances-f64.bin')),ply=readFileSync(resolve(ctx.dir,'distances.ply'));
 if(raw.length!==r.count*8)throw new Error('Incomplete f64 output');
 return{distanceBytes:raw.length,distanceSha256:bytesHash(raw),scalarPlyBytes:ply.length,scalarPlySha256:bytesHash(ply),count:r.count,statisticsBits:r.statisticsBits};
}
async function compute(ctx,mode){const pending=ctx.next();ctx.child.stdin.write(mode+'\n');const r=await pending;return{r,proof:proof(ctx,r)};}
const median=xs=>{const a=[...xs].sort((x,y)=>x-y);return(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
const stageNames=['bvhBuild','mortonPreparation','parallelQueries','allocationScatter','orderBvhDestruction'];
try{
 const oracleCtx=start('baseline','oracle'),ready=await oracleCtx.next();
 if(ready.kind!=='ready'||ready.threads!==threads||ready.chunkSize!==4096||ready.referencePoints!==inputs.reference.points||ready.queryPoints!==inputs.compared.points)throw new Error('Changed input/layout');
 const geometryBytes=readFileSync(resolve(oracleCtx.dir,'geometry.ply')),geometry={bytes:geometryBytes.length,sha256:bytesHash(geometryBytes)};
 const oracle=await compute(oracleCtx,'unsigned');await oracleCtx.close();
 const ctx=start('profile','profile');
 if(JSON.stringify(ready)!==JSON.stringify(await ctx.next())||!geometryBytes.equals(readFileSync(resolve(ctx.dir,'geometry.ply'))))throw new Error('Changed TIN/layout');
 const run=async mode=>{
  const {r,proof:p}=await compute(ctx,mode);
  if(r.mode!==mode||JSON.stringify(p)!==JSON.stringify(oracle.proof))throw new Error('Changed complete output or statistics');
  if(mode==='stages'){
   if(r.stageMillis?.length!==5||r.stageMillis.some(x=>!Number.isFinite(x)||x<0)||r.stageMillis.reduce((a,b)=>a+b,0)>r.elapsedMillis+1e-6)throw new Error('Invalid stage clocks');
  }else if(r.stageMillis!==null)throw new Error('Whole call contains stage clocks');
  const stageMillis=mode==='stages'?Object.fromEntries(stageNames.map((name,i)=>[name,r.stageMillis[i]])):null;
  return{elapsedMillis:r.elapsedMillis,stageMillis,unattributedMillis:stageMillis?r.elapsedMillis-Object.values(stageMillis).reduce((a,b)=>a+b,0):null,...p,resultParity:true};
 };
 const firstUse={whole:await run('whole'),stages:await run('stages')},warmups=[],trials=[];
 for(let i=-warmupRounds;i<repeats;i++){
  const order=(i+warmupRounds)%2?['stages','whole']:['whole','stages'],row={order};for(const mode of order)row[mode]=await run(mode);
  (i<0?warmups:trials).push(row);
  console.log(`Native stage ${i<0?'warm-up':'trial'} ${i<0?i+warmupRounds+1:i+1}: whole ${row.whole.elapsedMillis.toFixed(1)}, staged ${row.stages.elapsedMillis.toFixed(1)} ms; complete parity`);
 }
 await ctx.close();
 const summary={wholeMedianMillis:median(trials.map(t=>t.whole.elapsedMillis)),stagedMedianMillis:median(trials.map(t=>t.stages.elapsedMillis)),medianPairedStagedOverheadPercent:median(trials.map(t=>100*(t.stages.elapsedMillis/t.whole.elapsedMillis-1))),stageMedianMillis:Object.fromEntries(stageNames.map(n=>[n,median(trials.map(t=>t.stages.stageMillis[n]))])),stageMedianSharePercent:Object.fromEntries(stageNames.map(n=>[n,median(trials.map(t=>100*t.stages.stageMillis[n]/t.stages.elapsedMillis))])),unattributedMedianMillis:median(trials.map(t=>t.stages.unattributedMillis))};
 const report={schemaVersion:1,comparison:'native-c2m-stage-profile',createdAt:new Date().toISOString(),builds,examplePath,exampleSha256,scriptSha256,inputs,inputManifest,inputManifestSha256:bytesHash(manifestBytes),threads,repeats,warmupRounds,mesh:ready,geometry,oracle:oracle.proof,firstUse,warmups,trials,summary,commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8',windowsHide:true}).trim(),workingTreeDirty:!!execFileSync('git',['status','--porcelain'],{encoding:'utf8',windowsHide:true}).trim(),machine:{platform:platform(),release:release(),cpu:cpus()[0]?.model,logicalCpus:cpus().length,memoryBytes:totalmem()},note:'Diagnostic source-inclusion harness recompiles unchanged production distance/mesh modules under example-local symbols. Whole and staged modes alternate in one process/pool on the same unchanged measured-source TIN. Staged mode retains original serial Morton preparation, 4096-point indexed query chunks, chunk-local previous-hit/cached stack, exact unsigned f64 arithmetic and compact serial scatter. Full distances, six statistics and scalar PLY match an independent archived production oracle on every call. Original mesh vectors move without coordinate copies/transforms/reordering. BVH construction, Morton bounds/keys/sort/index extraction, parallel query/collection including chunk scratch destruction, final allocation/serial scatter including result-chunk destruction, and explicit order/BVH destruction are measured separately. Remaining pool-installation/timer/dispatch costs are unattributed. Loading, TIN/pool construction, first use, five warmups, serialization/statistics/parity and cleanup are excluded. Ten alternating pairs and all outliers are retained. Stage timers, source-included symbol/code layout, closure state and compilation can change timing; output parity establishes correctness, not timing equivalence to the archived binary. Diagnostic percentages are not an optimization or a production speedup claim. Shares are medians of per-trial fractions and need not sum to 100%. Production source/configuration fingerprints remain unchanged. Native-only profiling; no browser/WASM/GPU/Python conversion speedup claim. No GPU model identifiers collected. Clocks/power/scheduling/other applications/allocator/retained pages are uncontrolled. No builds/tests/scans/edits run alongside timing.'};
 for(const b of Object.values(builds)){if(hash(b.exe)!==b.exeSha256||readFileSync(resolve(b.root,'compiler.txt'),'utf8')!==b.compiler)throw new Error('Build changed');for(const[p,h]of Object.entries(b.sourceHashes))if(hash(resolve(b.root,p))!==h)throw new Error('Source changed');}
 for(const input of Object.values(inputs))if(hash(input.path)!==input.sha256)throw new Error('Input changed');
 if(hash(original)!==inputManifest.source.sha256||hash(manifestPath)!==bytesHash(manifestBytes)||hash(scriptPath)!==scriptSha256||hash(resolve(builds.profile.root,examplePath))!==exampleSha256)throw new Error('Fingerprint changed');
 writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(summary));console.log('Saved '+output);
}finally{for(const child of children)if(child.exitCode===null)child.kill();}
