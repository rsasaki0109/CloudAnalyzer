// Run from the repository root. Frozen native fresh/retained C2M contexts.
import {spawn,execFileSync} from 'node:child_process';
import {createInterface} from 'node:readline';
import {readFileSync,writeFileSync,existsSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {cpus,totalmem,platform,release} from 'node:os';

const args=process.argv.slice(2),allowed=['--oracle-root','--native-root','--reference','--compared','--input-manifest','--output','--threads','--repeats'];
for(let i=0;i<args.length;i+=2)if(!allowed.includes(args[i])||!args[i+1]||args[i+1].startsWith('--')||args.indexOf(args[i])!==i)throw new Error('Invalid or repeated option');
const value=(k,d)=>args.includes(k)?args[args.indexOf(k)+1]:d;
for(const k of allowed.slice(0,6))if(!value(k))throw new Error('Supply oracle/native roots, reference/compared PLY, manifest and new output');
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
const extraFiles=['rust/crates/ca-core/examples/bench_c2m_retained.rs','rust/crates/ca-core/tests/mesh_preparation.rs','rust/crates/ca-core/tests/mesh_parallel_stack.rs'];
const builds={};
for(const mode of ['oracle','native']){
 const root=resolve(value(mode==='oracle'?'--oracle-root':'--native-root')),exe=resolve(root,mode==='oracle'?'bench_c2m_source.exe':'bench_c2m_retained.exe');
 builds[mode]={root,exe,exeSha256:hash(exe),compiler:readFileSync(resolve(root,'compiler.txt'),'utf8'),sourceHashes:Object.fromEntries(sourceFiles.map(p=>[p,hash(resolve(root,p))]))};
}
const nativeFiles=Object.fromEntries(extraFiles.map(p=>[p,hash(resolve(builds.native.root,p))]));
if(builds.oracle.compiler!==builds.native.compiler)throw new Error('Compiler changed');
for(const p of sourceFiles.slice(2))if(builds.oracle.sourceHashes[p]!==builds.native.sourceHashes[p])throw new Error('Unrelated source changed '+p);
for(const b of Object.values(builds)){
 const d=readFileSync(resolve(b.root,sourceFiles[0]),'utf8'),m=readFileSync(resolve(b.root,sourceFiles[1]),'utf8');
 for(const flag of ['USE_PAR_MESH_STACK_REUSE','USE_PAR_MESH_STACK_BOUNDS','USE_PAR_MESH_COMPACT_RESULTS','USE_PAR_MESH_BOUND_PAIR_CLAMP'])if(!d.includes(`const ${flag}: bool = true;`))throw new Error('Adopted native policy disabled');
 for(const flag of ['USE_PAR_MESH_INDEXED_CORNERS','USE_PAR_MESH_TRIANGLE_SEED','USE_PAR_MESH_EAGER_PRUNE','USE_PAR_MESH_SKIP_GUESS','USE_PAR_MESH_MORTON_PREPARATION','USE_PAR_MESH_PAR_RESULTS','USE_PAR_MESH_BOUND_PAIR','USE_PAR_MESH_PAIRED_PROJECTION'])if(!d.includes(`const ${flag}: bool = false;`))throw new Error('Rejected native experiment enabled');
 for(const flag of ['USE_MESH_PAR_PREPARATION','USE_MESH_BOUND_MERGE','USE_MESH_PAR_BUILD'])if(!m.includes(`const ${flag}: bool = true;`))throw new Error('Adopted builder disabled');
 if(!m.includes('const USE_MESH_PAR_CENTROID_BOUNDS: bool = false;'))throw new Error('Rejected centroid scan enabled');
}
const scriptPath=fileURLToPath(import.meta.url),scriptSha256=hash(scriptPath),scratch=resolve('notes','native-mesh-retained-par-'+randomUUID());mkdirSync(scratch);
const children=[];
function start(mode,label){
 const dir=resolve(scratch,label),queue=[],waiters=[],build=builds[mode==='oracle'?'oracle':'native'];let failure,stderr='';
 const argv=[inputs.reference.path,inputs.compared.path,dir,String(threads)];if(mode!=='oracle')argv.push(mode);
 const child=spawn(build.exe,argv,{stdio:['pipe','pipe','pipe'],windowsHide:true});children.push(child);
 const fail=e=>{failure=e;for(const w of waiters.splice(0)){clearTimeout(w.timer);w.reject(e);}};
 child.stderr.on('data',b=>stderr=(stderr+b.toString()).slice(-4000));child.on('error',fail);
 child.on('exit',code=>fail(new Error(label+' exited '+code+': '+stderr)));
 createInterface({input:child.stdout}).on('line',line=>{let result;try{result=JSON.parse(line);}catch{fail(new Error('Invalid protocol'));return;}const w=waiters.shift();if(w){clearTimeout(w.timer);w.resolve(result);}else queue.push(result);});
 const next=()=>failure?Promise.reject(failure):queue.length?Promise.resolve(queue.shift()):new Promise((resolve,reject)=>{const w={resolve,reject,timer:null};w.timer=setTimeout(()=>{fail(new Error('Timeout '+label));child.kill();},180000);waiters.push(w);});
 return{child,dir,next,async command(command){const result=next();child.stdin.write(command+'\n');return result;},async close(){
  if(child.exitCode!==null){if(child.exitCode!==0)throw new Error('Failed process '+label);return;}
  const done=new Promise((resolve,reject)=>child.once('exit',code=>code===0?resolve():reject(new Error('Failed process '+label))));child.stdin.end('quit\n');await done;
 }};
}
function proof(ctx,r){
 if(r.kind!=='result'||r.signed!==false||r.count!==inputs.compared.points||!Number.isFinite(r.elapsedMillis)||r.elapsedMillis<=0||r.statisticsBits?.length!==6)throw new Error('Unexpected output');
 const raw=readFileSync(resolve(ctx.dir,'distances-f64.bin')),ply=readFileSync(resolve(ctx.dir,'distances.ply'));
 if(raw.length!==r.count*8)throw new Error('Incomplete f64 output');
 return{distanceBytes:raw.length,distanceSha256:bytesHash(raw),scalarPlyBytes:ply.length,scalarPlySha256:bytesHash(ply),count:r.count,statisticsBits:r.statisticsBits};
}
const median=xs=>{const a=[...xs].sort((x,y)=>x-y);return(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
try{
 const oracleCtx=start('oracle','oracle'),ready=await oracleCtx.next();
 if(ready.kind!=='ready'||ready.threads!==threads||ready.chunkSize!==4096||ready.referencePoints!==inputs.reference.points||ready.queryPoints!==inputs.compared.points)throw new Error('Changed input/layout');
 const geometryBytes=readFileSync(resolve(oracleCtx.dir,'geometry.ply')),geometry={bytes:geometryBytes.length,sha256:bytesHash(geometryBytes)};
 const oracle=proof(oracleCtx,await oracleCtx.command('unsigned'));await oracleCtx.close();
 const contexts={},preparation={};
 for(const mode of ['fresh','retained']){
  const ctx=start(mode,mode);contexts[mode]=ctx;
  if(JSON.stringify(ready)!==JSON.stringify(await ctx.next())||!geometryBytes.equals(readFileSync(resolve(ctx.dir,'geometry.ply'))))throw new Error('Changed TIN/layout');
  const p=await ctx.command('prepare');
  if(p.kind!=='preparation'||p.policy!==mode||!Number.isFinite(p.elapsedMillis)||p.elapsedMillis<0||!Number.isSafeInteger(p.sourceGeometryBytes)||p.sourceGeometryBytes<=0||!Number.isSafeInteger(p.retainedBytes)||!Number.isSafeInteger(p.addedBvhBytes)||p.addedBvhBytes!==p.retainedBytes-p.sourceGeometryBytes||p.addedBvhBytes<0)throw new Error('Invalid preparation');
  if(mode==='fresh'&&(p.elapsedMillis!==0||p.addedBvhBytes!==0)||mode==='retained'&&(p.elapsedMillis<=0||p.addedBvhBytes<=0))throw new Error('Changed retention policy');
  preparation[mode]=p;
 }
 if(preparation.fresh.sourceGeometryBytes!==preparation.retained.sourceGeometryBytes)throw new Error('Changed geometry capacity');
 const run=async mode=>{const ctx=contexts[mode],r=await ctx.command('unsigned'),p=proof(ctx,r);if(JSON.stringify(p)!==JSON.stringify(oracle))throw new Error('Changed complete output/statistics');return{elapsedMillis:r.elapsedMillis,...p,resultParity:true};};
 const firstUse={fresh:await run('fresh'),retained:await run('retained')},warmups=[],trials=[];
 for(let i=-warmupRounds;i<repeats;i++){
  const order=(i+warmupRounds)%2?['retained','fresh']:['fresh','retained'],row={order};for(const mode of order)row[mode]=await run(mode);
  (i<0?warmups:trials).push(row);
  console.log(`Native retained ${i<0?'warm-up':'trial'} ${i<0?i+warmupRounds+1:i+1}: fresh ${row.fresh.elapsedMillis.toFixed(1)}, retained ${row.retained.elapsedMillis.toFixed(1)} ms; full parity`);
 }
 for(const ctx of Object.values(contexts))await ctx.close();
 const reductions=trials.map(t=>100*(1-t.retained.elapsedMillis/t.fresh.elapsedMillis)),saved=median(trials.map(t=>t.fresh.elapsedMillis-t.retained.elapsedMillis));
 const summary={freshMedianMillis:median(trials.map(t=>t.fresh.elapsedMillis)),retainedMedianMillis:median(trials.map(t=>t.retained.elapsedMillis)),medianPairedReductionPercent:median(reductions),medianPairedSavedMillis:saved,fasterTrials:reductions.filter(x=>x>0).length,trials:repeats,setupMillis:preparation.retained.elapsedMillis,estimatedSetupAmortizationQueries:saved>0?Math.ceil(preparation.retained.elapsedMillis/saved):null,firstObservedTotalMillis:{fresh:firstUse.fresh.elapsedMillis,retained:preparation.retained.elapsedMillis+firstUse.retained.elapsedMillis}};
 const report={schemaVersion:1,comparison:'native-c2m-retained-par',createdAt:new Date().toISOString(),builds,nativeFiles,scriptSha256,inputs,inputManifest,inputManifestSha256:bytesHash(manifestBytes),threads,repeats,warmupRounds,mesh:ready,geometry,oracle,preparation,firstUse,warmups,trials,summary,commit:execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8',windowsHide:true}).trim(),workingTreeDirty:!!execFileSync('git',['status','--porcelain'],{encoding:'utf8',windowsHide:true}).trim(),machine:{platform:platform(),release:release(),cpu:cpus()[0]?.model,logicalCpus:cpus().length,memoryBytes:totalmem()},note:'Repeated queries on the same unchanged full measured-source TIN; not independent surveys or accuracy ground truth. No transformed/generated coordinates or extra thinning. Native PLY order differs from browser LOD order. An independent preceding archived production oracle exits before current native contexts start. Fresh and retained modes use the same frozen native executable in separate contexts; only one computes at a time. Retained construction moves original mesh vectors into PreparedMesh without cloning/reordering/coordinate conversion, validates faces once and builds the existing copied-corner BVH in the fixed pool. Preparation time and logical capacity bytes are recorded separately. Retained query calls borrow immutable storage and use the shared native parallel query policy; fresh calls build/drop BVHs inside each call. Both retain original serial Morton preparation/sorting, 4096-point chunks, chunk-local warm states, copied corners, clamped sibling SIMD, scalar triangle projection, compact f64 results and serial scatter. Query order and scratch are rebuilt each call; no cross-query cache or global retention is added. Retained ownership releases geometry/BVH when the object is dropped. Hot whole-call timing includes pool installation, Morton work, parallel queries, result allocation/scatter and call-local destruction; fresh additionally includes BVH build/destruction. Constructor time and final retained-object destruction are outside hot clocks. Source geometry remains owned in both contexts. Logical retained bytes include mesh and BVH vector capacities, exclude headers/allocator/pages, and do not establish process RSS or net peak memory reduction. Loading, meshing, pool creation, first observations, five warmups, serialization/statistics/parity and cleanup are excluded. All ten alternating timed pairs and outliers are retained. Setup amortization is an estimate from one observed constructor cost divided by median paired hot-call savings, excludes loading/meshing/pool/conversion/final destruction, and is not a cold-process or single-call guarantee. First-observed totals record setup plus first query, with fixed execution order and uncontrolled allocator/pages. Sources/executables/compiler/example/tests/script/inputs/manifests/original PCD are checked before/after. No builds/tests/scans/edits run alongside timing. Clocks/power/scheduling/other applications/code layout/allocator/retained pages are uncontrolled. Rust native API opt-in; stateless/Python/browser/WASM/GPU API selection is unchanged. No Python conversion, browser/WASM/GPU speedup claim. No GPU model identifiers collected.'};
 for(const b of Object.values(builds)){if(hash(b.exe)!==b.exeSha256||readFileSync(resolve(b.root,'compiler.txt'),'utf8')!==b.compiler)throw new Error('Build changed');for(const[p,h]of Object.entries(b.sourceHashes))if(hash(resolve(b.root,p))!==h)throw new Error('Source changed');}
 for(const[p,h]of Object.entries(nativeFiles))if(hash(resolve(builds.native.root,p))!==h)throw new Error('Native example/test changed');
 for(const input of Object.values(inputs))if(hash(input.path)!==input.sha256)throw new Error('Input changed');
 if(hash(original)!==inputManifest.source.sha256||hash(manifestPath)!==bytesHash(manifestBytes)||hash(scriptPath)!==scriptSha256)throw new Error('Provenance/script changed');
 writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(summary));console.log('Saved '+output);
}finally{for(const child of children)if(child.exitCode===null)child.kill();}
