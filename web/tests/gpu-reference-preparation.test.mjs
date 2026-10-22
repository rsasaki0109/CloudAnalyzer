import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {stripTypeScriptTypes} from 'node:module';
import {preparationPrelude,preparationTargets,instrumentPreparationSource,PREPARATION_EVENT_LIMIT} from '../scripts/gpu-reference-preparation-probe.mjs';
const text=n=>readFileSync(new URL('../../'+n,import.meta.url),'utf8').replaceAll('\r\n','\n');
const load=async source=>import('data:text/javascript;base64,'+Buffer.from(stripTypeScriptTypes(source,{mode:'transform'})).toString('base64'));
const baseline=await load(text('web/src/gpu-candidates.ts')+'\nexport{SHADER,PACKED_SHADER};');
const instrumented=await load(instrumentPreparationSource('web/src/gpu-candidates.ts',text('web/src/gpu-candidates.ts'))+'\nexport{SHADER,PACKED_SHADER};');
const helpers=await load(preparationPrelude+'\nexport{__caMark as mark,__caScope as scope,__caPreparationSnapshot as snapshot};');

test('preparation probe preserves every ordinary and optional WGSL shader byte',()=>{
 for(const capacity of [20,32])assert.equal(instrumented.SHADER(capacity),baseline.SHADER(capacity));
 assert.equal(instrumented.PACKED_SHADER,baseline.PACKED_SHADER);
 for(const n of preparationTargets){const source=text(n),changed=instrumentPreparationSource(n,source);assert.notEqual(source,changed);assert.ok(changed.startsWith(preparationPrelude));}
 assert.throws(()=>instrumentPreparationSource('other.ts',''),/Unexpected/);
 assert.throws(()=>instrumentPreparationSource('web/src/gpu-c2c.ts',text('web/src/gpu-c2c.ts').replace('new GpuC2cReference(reference)','changedReference(reference)')),/target changed/);
});

test('disabled preparation logging reads no clock and enabled records stay bounded',()=>{
 const descriptor=Object.getOwnPropertyDescriptor(performance,'now');let calls=0;
 Object.defineProperty(performance,'now',{configurable:true,value:()=>++calls});
 try{
  globalThis.__caPreparationTrace=undefined;helpers.mark('off');assert.equal(helpers.scope(),undefined);assert.equal(helpers.snapshot(),undefined);assert.equal(calls,0);
  globalThis.__caPreparationTrace={events:[],omitted:0,nextScope:0};assert.equal(helpers.scope(),0);assert.equal(helpers.scope(),1);
  for(let i=0;i<PREPARATION_EVENT_LIMIT+5;i++)helpers.mark('bounded',i);
  const snapshot=helpers.snapshot();assert.equal(snapshot.events.length,256);assert.equal(snapshot.omitted,5);assert.equal(calls,256);
  globalThis.__caPreparationTrace.events.length=0;assert.equal(snapshot.events.length,256);
 }finally{delete globalThis.__caPreparationTrace;if(descriptor)Object.defineProperty(performance,'now',descriptor);else delete performance.now;}
});

test('instrumented device scopes serialize and drain on work and validation failures',async()=>{
 let active=0,maximum=0,pops=0,failValidation=false;
 const fake={features:new Set(),lost:new Promise(()=>{}),pushErrorScope(){maximum=Math.max(maximum,++active);},async popErrorScope(){pops++;active--;if(failValidation){failValidation=false;return {message:'injected validation'};}return null;}};
 globalThis.__caPreparationTrace={events:[],omitted:0,nextScope:0};
 try{
  const runtime=new instrumented.GpuCandidateDevice(fake);let release;const held=new Promise(resolve=>{release=resolve;});
  const first=runtime.run(async()=>{await held;throw new Error('injected work');});const second=runtime.run(async()=>42);
  assert.equal(active,3);release();await assert.rejects(first,/injected work/);assert.equal(await second,42);
  failValidation=true;await assert.rejects(runtime.run(async()=>43),/injected validation/);
  assert.equal(active,0);assert.equal(pops,9);assert.equal(maximum,3);
  for(const id of [0,1,2])assert.deepEqual(globalThis.__caPreparationTrace.events.filter(e=>e.id===id).map(e=>e.phase),['device-scope-enter','scope-pop-start','scope-pop-end']);
 }finally{delete globalThis.__caPreparationTrace;}
});
