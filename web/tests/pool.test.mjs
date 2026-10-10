import assert from 'node:assert/strict';
import { after, test } from 'node:test';

const workerDescriptor=Object.getOwnPropertyDescriptor(globalThis,'Worker');
const navigatorDescriptor=Object.getOwnPropertyDescriptor(globalThis,'navigator');
const created=[];
class PoolWorker {
  requests=[];
  listeners=new Map();
  terminated=false;
  constructor() { created.push(this); }
  addEventListener(type,handler) { const list=this.listeners.get(type) ?? new Set(); list.add(handler); this.listeners.set(type,list); }
  removeEventListener(type,handler) { this.listeners.get(type)?.delete(handler); }
  postMessage(request) { this.requests.push(request); }
  terminate() { this.terminated=true; }
  respond(request,memory=65536) {
    for (const handler of [...(this.listeners.get('message') ?? [])]) handler({data:{seq:request.seq,ok:true,value:new Float64Array([5]),memory}});
  }
}
Object.defineProperty(globalThis,'Worker',{value:PoolWorker,configurable:true});
Object.defineProperty(globalThis,'navigator',{value:{hardwareConcurrency:4},configurable:true});
const {warmUpPool,releasePool,runOn,poolMemory}=await import('../src/pool.ts');
after(() => {
  if (workerDescriptor) Object.defineProperty(globalThis,'Worker',workerDescriptor); else delete globalThis.Worker;
  if (navigatorDescriptor) Object.defineProperty(globalThis,'navigator',navigatorDescriptor); else delete globalThis.navigator;
});
const cloud=()=>({kind:'cloud',reference:new Float64Array([0,0,0]),queries:new Float64Array([1,0,0])});

test('release cancels background warmup, settles its promises and recreates workers lazily', {timeout:2000}, async () => {
  const warming=warmUpPool(), old=created.slice();
  assert.equal(old.length,3);
  releasePool(); await warming;
  assert.ok(old.every(worker=>worker.terminated));
  assert.equal(poolMemory(),0);
  for (const worker of old) worker.respond(worker.requests[0]);
  assert.equal(poolMemory(),0); // Late replies cannot resurrect freed worker memory.
  const job=runOn(0,cloud()), next=created.at(-1);
  assert.ok(!old.includes(next)); assert.equal(next.terminated,false);
  next.respond(next.requests[0]);
  assert.deepEqual(await job,new Float64Array([5]));
  assert.equal(poolMemory(),65536);
  releasePool(); assert.equal(next.terminated,true); assert.equal(poolMemory(),0);
});

test('release protects real computations even while background warmup is pending', async () => {
  const first=created.length, warming=warmUpPool(), workers=created.slice(first);
  const job=runOn(0,cloud());
  assert.throws(releasePool,/active operation/);
  assert.ok(workers.every(worker=>!worker.terminated));
  for (const worker of workers) worker.respond(worker.requests.find(request=>request.kind==='warm-up'));
  await warming;
  assert.throws(releasePool,/active operation/);
  workers[0].respond(workers[0].requests.find(request=>request.kind==='cloud'));
  await job; releasePool();
  assert.ok(workers.every(worker=>worker.terminated));
});
