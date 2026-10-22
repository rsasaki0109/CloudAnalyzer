// Constructor screening in a dedicated Worker using two matched production WASM builds.
import {chromium} from '@playwright/test';
import {createServer} from 'node:http';
import {createHash} from 'node:crypto';
import {existsSync,readFileSync,writeFileSync} from 'node:fs';
import {resolve,basename} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {collectionSource,transformCollectionSource} from './kdtree-collection-source.mjs';

const args=process.argv.slice(2),allowed=['--output','--build-info','--reference','--compared','--input-manifest','--warmups','--repeats','--first'];
for(let i=0;i<args.length;i+=2)if(!allowed.includes(args[i])||!args[i+1]||args[i+1].startsWith('--')||args.indexOf(args[i])!==i)throw new Error('Invalid option: '+args[i]);
const option=(n,f)=>args.includes(n)?args[args.indexOf(n)+1]:f;
const output=option('--output'),buildFile=option('--build-info'),reference=option('--reference'),compared=option('--compared'),first=option('--first','baseline'),warmups=Number(option('--warmups','6')),repeats=Number(option('--repeats','12'));
if(!output||!buildFile||!reference||!compared||existsSync(output)||!['baseline','unzip'].includes(first)||!Number.isInteger(warmups)||warmups<0||warmups>12||!Number.isInteger(repeats)||repeats<1||repeats>24)throw new Error('Supply NEW output, build, inputs and valid trial settings');
const root=resolve(fileURLToPath(new URL('../../',import.meta.url))),hash=b=>createHash('sha256').update(b).digest('hex'),normal=n=>hash(readFileSync(resolve(root,n),'utf8').replaceAll('\r\n','\n'));
const buildBytes=readFileSync(buildFile),build=JSON.parse(buildBytes);
function verifyBuild(){
 if(build.purpose!=='isolated-kdtree-collection-build'||!build.productionWorkspaceUnchanged||!build.offline)throw new Error('Wrong isolated WASM build');
 for(const[n,h]of Object.entries({...build.rustInputs,...build.helpers}))if(normal(n)!==h)throw new Error('Build source changed: '+n);
 const original=readFileSync(resolve(root,collectionSource),'utf8');
 for(const mode of ['baseline','unzip']){
  const v=build.variants[mode],expected=hash(transformCollectionSource(original,mode));
  if(v.sourceSha256!==expected||normal(v.sourceArchive)!==expected)throw new Error('Wrong transformed source');
  for(const[n,h]of Object.entries(v.normalizedRustInputs))if(h!==(n===collectionSource?expected:build.rustInputs[n]))throw new Error('Variant source mismatch: '+n);
  if(Object.keys(v.normalizedRustInputs).sort().join('\0')!==Object.keys(build.rustInputs).sort().join('\0'))throw new Error('Incomplete variant source set');
  for(const[n,h]of Object.entries(v.artifacts))if(hash(readFileSync(resolve(root,n)))!==h)throw new Error('Variant artifact changed: '+n);
 }
}
verifyBuild();
const inputs=Object.fromEntries([['reference',reference],['compared',compared]].map(([role,path])=>{const b=readFileSync(path);return[role,{name:basename(path),path:resolve(path),bytes:b.length,sha256:hash(b)}];}));
const specFile=option('--input-manifest'),specBytes=specFile?readFileSync(specFile):undefined,inputManifest=specBytes?JSON.parse(specBytes):undefined;
if(inputManifest)for(const[role,v]of Object.entries(inputs)){const e=inputManifest.inputs[role];if(v.sha256!==e.sha256||v.bytes!==e.bytes)throw new Error('Input specification mismatch');v.points=e.points;}
const harness='web/scripts/benchmark-kdtree-collection.mjs',harnessHash=normal(harness);
const server=createServer((req,res)=>{
 res.setHeader('Cache-Control','no-store');const url=new URL(req.url,'http://localhost');
 if(url.pathname==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>WASM KD tree collection comparison</title>');return;}
 for(const[role,v]of Object.entries(inputs))if(url.pathname==='/__input/'+role){const b=readFileSync(v.path);if(hash(b)!==v.sha256){res.writeHead(409).end();return;}res.end(b);return;}
 const match=/^\/__pkg\/(baseline|unzip)\/(ca_wasm\.js|ca_wasm_bg\.wasm)$/.exec(url.pathname);
 if(!match){res.writeHead(404).end();return;}
 const path=resolve(root,build.variants[match[1]].packageDirectory,match[2]);
 if(!Object.hasOwn(build.variants[match[1]].artifacts,relativeName(path))){res.writeHead(403).end();return;}
 res.setHeader('Content-Type',match[2].endsWith('.wasm')?'application/wasm':'text/javascript');res.end(readFileSync(path));
});
function relativeName(path){return path.slice(root.length+1).replaceAll('\\','/');}

async function workerMain({base,inputs,warmups,repeats,first}){
 const modes=['baseline','unzip'],modules={},instances={},clouds={},counts={created:{baseline:0,unzip:0},freed:{baseline:0,unzip:0},jobs:0,jobReleases:0};let gpuRequests=0;
 if(navigator.gpu)Object.defineProperty(navigator.gpu,'requestAdapter',{value:()=>{gpuRequests++;throw new Error('Constructor screening must not request GPU adapters');},configurable:true});
 const digest=async a=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new Uint8Array(a.buffer,a.byteOffset,a.byteLength).slice()))).map(v=>v.toString(16).padStart(2,'0')).join('');
 const bitsEqual=(a,b)=>{if(a.length!==b.length)return false;const left=new BigUint64Array(a.buffer,a.byteOffset,a.length),right=new BigUint64Array(b.buffer,b.byteOffset,b.length);return left.every((v,i)=>v===right[i]);};
 const files={};for(const role of ['reference','compared']){const r=await fetch(base+'/__input/'+role);if(!r.ok)throw new Error('Input fetch failed');files[role]=new Uint8Array(await r.arrayBuffer());}
 function create(mode){const value=new modules[mode].GpuC2cReference(clouds[mode].reference);counts.created[mode]++;return value;}
 function release(mode,value){value.free();counts.freed[mode]++;}
 async function validate(mode){
  const wasm=modules[mode];let index=create(mode),job,buffers;
  try{
   const snapshot=index.snapshot();buffers=index.candidateBuffers();const points=buffers.takePoints(),nodes=buffers.takeNodes();buffers.free();buffers=undefined;
   if(points.length%4||nodes.length%5||(inputs.reference.points&&points.length/4!==inputs.reference.points))throw new Error('Reference points changed');
   job=new wasm.GpuC2cJob(clouds[mode].compared,index);counts.jobs++;release(mode,index);index=undefined;
   if(inputs.compared.points&&job.length!==inputs.compared.points)throw new Error('Compared points changed');
   job.useBounds(true);job.useSimd(true);
   const distances=new Float64Array(job.length);
   for(let start=0;start<job.length;start+=65536){const n=Math.min(65536,job.length-start);distances.set(job.complete(start,new Uint32Array(n).fill(0xffffffff)),start);}
   const exact=job.queriesF64(0,job.length),copy=job.queriesF64WithCopy(0,Math.min(257,job.length),true),original=job.queriesF64WithCopy(0,Math.min(257,job.length),false);
   if(!bitsEqual(copy,original))throw new Error('Query export bits changed');copy.fill(NaN);
   if(!bitsEqual(job.queriesF64WithCopy(0,Math.min(257,job.length),true),original))throw new Error('Query export borrowed memory');
   let hintChecks=0;
   for(const bounds of [false,true])for(const simd of [false,true])for(const hint of [0xffffffff,0,points.length/4-1]){
    job.useBounds(bounds);job.useSimd(simd);const n=Math.min(257,job.length),value=job.complete(0,new Uint32Array(n).fill(hint));if(!bitsEqual(value,distances.subarray(0,n)))throw new Error('Hint/bounds/SIMD changed exact distances');hintChecks++;
   }
   let rangeChecks=0;for(const f of [()=>job.queriesF64(job.length,1),()=>job.complete(job.length,new Uint32Array(1))]){let rejected=false;try{f();}catch{rejected=true;}if(!rejected)throw new Error('Invalid range accepted');rangeChecks++;}
   const result={referencePoints:points.length/4,comparedPoints:job.length,nodes:nodes.length/5,snapshotBytes:snapshot.length,snapshotSha256:await digest(snapshot),gpuPointBitsSha256:await digest(points),gpuNodeBitsSha256:await digest(nodes),queryF64BitsSha256:await digest(exact),distanceF64BitsSha256:await digest(distances),displayedF32Sha256:await digest(Float32Array.from(distances)),hintChecks,rangeChecks,independentQueryExport:true,referenceReleasedBeforeVerification:true};
   return result;
  }finally{buffers?.free();if(job){job.free();counts.jobReleases++;}if(index)release(mode,index);}
 }
 function operation(mode){const Constructor=modules[mode].GpuC2cReference,cloud=clouds[mode].reference,start=performance.now(),index=new Constructor(cloud),constructorMillis=performance.now()-start;counts.created[mode]++;release(mode,index);return{constructorMillis,wasmLinearMemoryBytes:instances[mode].memory.buffer.byteLength};}
 let result;
 try{
  for(const mode of modes){const wasm=await import(base+'/__pkg/'+mode+'/ca_wasm.js');modules[mode]=wasm;instances[mode]=await wasm.default();if(typeof SharedArrayBuffer!=='undefined'&&instances[mode].memory.buffer instanceof SharedArrayBuffer)throw new Error('Screen requires ordinary non-shared WASM memory');clouds[mode]={};for(const role of ['reference','compared'])clouds[mode][role]=wasm.Cloud.parse(inputs[role].name,files[role]);}
  const initial={};for(const mode of modes)initial[mode]=await validate(mode);
  if(JSON.stringify(initial.baseline)!==JSON.stringify(initial.unzip))throw new Error('Matched WASM builds changed tree/export/query/distance bits');
  const warming=[],trials=[];
  for(let i=0;i<warmups+repeats;i++){
   const firstMode=i%2?(first==='baseline'?'unzip':'baseline'):first,order=firstMode==='baseline'?['baseline','unzip']:['unzip','baseline'],results={};
   for(const mode of order)results[mode]=operation(mode);
   (i<warmups?warming:trials).push({order,results});
  }
  const final={};for(const mode of modes)final[mode]=await validate(mode);
  if(JSON.stringify(initial)!==JSON.stringify(final))throw new Error('WASM validation changed during screening');
  if(gpuRequests||counts.jobs!==counts.jobReleases||modes.some(mode=>counts.created[mode]!==counts.freed[mode]))throw new Error('Leaked ownership or unexpected GPU request');
  result={workerScope:true,isolated:self.crossOriginIsolated,gpuRequests,wasmSharedMemory:false,validation:initial,validationUnchanged:true,distanceBitsExact:true,snapshotBitsExact:true,gpuExportBitsExact:true,queryBitsExact:true,ownershipChecksPassed:true,warming,trials,counts};
 }finally{for(const mode of modes)for(const cloud of Object.values(clouds[mode]??{}))cloud.free();}
 return{...result,sourceCloudsReleased:true,liveReferences:0,liveJobs:counts.jobs-counts.jobReleases};
}

const report={schemaVersion:1,wasmKdTreeCollectionScreen:true,createdUtc:new Date().toISOString(),purpose:inputManifest&&warmups===6&&repeats===12?'constructor-screen':'compatibility-smoke',warmups,repeats,first,inputs,inputManifest,inputManifestSha256:specBytes?hash(specBytes):undefined,build,buildInfoSha256:hash(buildBytes),sources:{[harness]:harnessHash},commit:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8',windowsHide:true}).trim(),
 scope:'Dedicated Worker wall clock around the complete ordinary exported GpuC2cReference constructor, including median partition, point/index collection and derived certificates/bounds. Parsed LOD-ordered source clouds are retained. Reference free runs after the constructor clock. No GPU, verification pool, complete C2C RPC, or application-load timing is measured.',
 note:'Two matched independent stable/SIMD WASM modules share a Worker and hold separate ordinary memories. Source, snapshot, rounded export, query and complete exact result identity are checked before and after timing. All tails remain. Linear memory bytes are committed WebAssembly memory, not live/peak heap usage. This screen can reject an ineffective candidate; it cannot establish complete application acceleration. All earlier budgets remain closed.'};
let browser;
try{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));browser=await chromium.launch({channel:'chromium',headless:true});report.browserVersion=browser.version();const page=await browser.newPage();const base=`http://127.0.0.1:${server.address().port}`;await page.goto(base);
 const workerSource=`self.onmessage=async({data})=>{try{self.postMessage({ok:true,value:await (${workerMain.toString()})(data)});}catch(error){self.postMessage({ok:false,error:String(error),stack:error.stack});}};`;
 report.measurement=await page.evaluate(({workerSource,settings})=>new Promise((resolve,reject)=>{const url=URL.createObjectURL(new Blob([workerSource],{type:'text/javascript'})),worker=new Worker(url,{type:'module'}),timer=setTimeout(()=>finish(new Error('Constructor screen timed out')),120000);function finish(error,value){clearTimeout(timer);worker.terminate();URL.revokeObjectURL(url);error?reject(error):resolve(value);}worker.onerror=e=>finish(new Error(e.message||'Worker error'));worker.onmessageerror=()=>finish(new Error('Worker message error'));worker.onmessage=({data})=>data.ok?finish(undefined,data.value):finish(new Error(data.error+'\n'+data.stack));worker.postMessage(settings);}),{workerSource,settings:{base,inputs,warmups,repeats,first}});
 verifyBuild();for(const input of Object.values(inputs))if(hash(readFileSync(input.path))!==input.sha256)throw new Error('Input changed');if(hash(readFileSync(buildFile))!==hash(buildBytes)||normal(harness)!==harnessHash||specFile&&hash(readFileSync(specFile))!==hash(specBytes))throw new Error('Build, harness or input specification changed');report.filesUnchangedDuringMeasurement=true;report.status='passed';
}catch(error){report.status='failed';report.error=String(error);process.exitCode=1;}
finally{await browser?.close();await new Promise(resolve=>server.close(resolve));writeFileSync(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({status:report.status,output,error:report.error}));}
