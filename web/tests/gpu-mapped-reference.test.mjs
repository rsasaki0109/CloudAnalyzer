import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {stripTypeScriptTypes} from 'node:module';

const load=async name=>import('data:text/javascript;base64,'+Buffer.from(stripTypeScriptTypes(readFileSync(new URL('../../'+name,import.meta.url),'utf8'),{mode:'transform'})+'\nexport{SHADER,PACKED_SHADER};').toString('base64'));
const current=await load('web/src/gpu-candidates.ts');
const prior=await load('benchmarks/wasm-reference-preparation/executed-source/pr346/web/src/gpu-candidates.ts');
globalThis.GPUBufferUsage={STORAGE:128,COPY_DST:8};

function fixture(){
 const live=new Set(),created=[],writes=[];let failure,scopeError,scopes=0,pops=0;
 const fake={features:new Set(),lost:new Promise(()=>{}),limits:{maxBufferSize:1e8,maxStorageBufferBindingSize:1e8},queue:{writeBuffer(buffer,offset,data){if(failure==='queue'){failure=undefined;throw new Error('injected queue');}writes.push(new Uint8Array(data.buffer,data.byteOffset,data.byteLength).slice());buffer.bytes.set(writes.at(-1),offset);}},
  pushErrorScope(){scopes++;},async popErrorScope(){scopes--;pops++;if(scopeError){scopeError=false;return {message:'injected scope'};}return null;},
  createBuffer(desc){if(failure==='allocation'){failure=undefined;throw new Error('injected allocation');}const bytes=new Uint8Array(desc.size);let mapped=desc.mappedAtCreation;
   const buffer={size:desc.size,bytes,desc,destroyed:false,unmaps:0,getMappedRange(){if(failure==='mapping'){failure=undefined;throw new Error('injected mapping');}if(failure==='copy'){failure=undefined;return new ArrayBuffer(desc.size-4);}assert.equal(mapped,true);return bytes.buffer;},unmap(){this.unmaps++;if(failure==='unmap'){failure=undefined;throw new Error('injected unmap');}mapped=false;},destroy(){assert.equal(this.destroyed,false,'buffer destroyed twice');this.destroyed=true;live.delete(this);}};
   live.add(buffer);created.push(buffer);return buffer;
  }};
 const padded=new Float32Array(16).fill(-123);const points=padded.subarray(4,12);points.set([1,-0,3,0,4,5,6,0]);
 const raw=new Uint32Array(15).fill(0xffffffff),nodes=raw.subarray(5,10);nodes.set([0,2,0,255,0]);
 return {runtime:new current.GpuCandidateDevice(fake),live,created,writes,points,nodes,fail:v=>{failure=v;},scopeFail:()=>{scopeError=true;},assertDrained(){assert.equal(scopes,0);assert.ok(pops>=3);}};
}

test('mapped reference changes no ordinary or optional GPU shader bytes',()=>{
 for(const n of [20,32])assert.equal(current.SHADER(n),prior.SHADER(n));
 assert.equal(current.PACKED_SHADER,prior.PACKED_SHADER);
});

test('queued defaults and mapped references own exactly the requested source bytes',async()=>{
 for(const mapped of [false,true])for(const packed of [false,true]){
  const f=fixture(),expectedPoints=new Uint8Array(f.points.buffer,f.points.byteOffset,f.points.byteLength).slice();
  const expectedNodes=packed?current.packGpuNodePairs(f.nodes,2):f.nodes.slice();
  const search=!mapped&&!packed?await current.GpuCandidateSearch.create(f.points,f.nodes,f.runtime):await current.GpuCandidateSearch.create(f.points,f.nodes,f.runtime,false,false,packed,mapped);
  assert.deepEqual(f.created[0].bytes,expectedPoints);assert.deepEqual(f.created[1].bytes,new Uint8Array(expectedNodes.buffer));
  assert.equal(f.created.length,2);assert.equal(f.writes.length,mapped?0:2);assert.ok(f.created.every(b=>b.desc.mappedAtCreation===mapped&&b.unmaps===Number(mapped)));
  f.points.fill(NaN);f.nodes.fill(0);assert.deepEqual(f.created[0].bytes,expectedPoints);
  search.dispose();assert.equal(f.live.size,0);f.assertDrained();
 }
});

test('mapped initialization drains scopes and destroys every partial allocation',async()=>{
 for(const failure of ['allocation','mapping','copy','unmap','scope']){
  const f=fixture();failure==='scope'?f.scopeFail():f.fail(failure);
  await assert.rejects(current.GpuCandidateSearch.create(f.points,f.nodes,f.runtime,false,false,false,true),failure==='copy'?RangeError:/injected/);
  assert.equal(f.live.size,0);f.assertDrained();assert.equal(f.runtime.usable,true);
  const retry=await current.GpuCandidateSearch.create(f.points,f.nodes,f.runtime,false,false,true,true);retry.dispose();assert.equal(f.live.size,0);
 }
});

test('lazy mapped node retries share preparation and release buffers even before assignment',async()=>{
 for(const initialPacked of [false,true]){
  const f=fixture(),search=await current.GpuCandidateSearch.create(f.points,f.nodes,f.runtime,false,false,initialPacked,true);
  let exports=0;const source=()=>{exports++;return f.nodes;};
  for(const failure of ['allocation','mapping','copy','unmap','scope']){
   failure==='scope'?f.scopeFail():f.fail(failure);
   const results=await Promise.allSettled([search.prepareNodes(!initialPacked,source),search.prepareNodes(!initialPacked,source)]);
   assert.ok(results.every(r=>r.status==='rejected'));assert.equal(f.live.size,2);assert.equal(search.nodeRepresentationReady(!initialPacked),false);f.assertDrained();
  }
  await Promise.all([search.prepareNodes(!initialPacked,source),search.prepareNodes(!initialPacked,source),search.prepareNodes(!initialPacked,source)]);
  await search.prepareNodes(!initialPacked,()=>{throw new Error('cached representation re-exported');});
  assert.equal(exports,6);assert.equal(f.live.size,3);assert.equal(search.retainedNodeBytes,28);assert.equal(f.writes.length,0);
  search.dispose();assert.equal(f.live.size,0);f.assertDrained();
 }
});
