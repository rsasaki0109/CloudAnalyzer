import { test } from 'node:test';
import assert from 'node:assert/strict';
import { visitVolumesCsv } from '../src/volume-visits.ts';
test('visit CSV retains separate baseline-relative rows, missing support and safe labels',()=>{
  const base={name:'A',bounds:[0,0,1,1],added:2,removed:1,addedArea:1,removedArea:1,matchedCells:2,totalCells:4,outsideRegionCells:3};
  const csv=visitVolumesCsv([{...base,visit:'=SUM(1)',visitIndex:1},{...base,visit:'later, "two"',visitIndex:2,matchedCells:0}],'m');
  assert.ok(csv.includes(`1,"'=SUM(1)","A",0,0,1,1,"m",true,2,1,1,1,1,2,2,3`));
  assert.ok(csv.includes('2,"later, ""two""","A",0,0,1,1,"m",false,,,,,,0,4,3'));
  assert.equal(csv.trim().split('\n').length,3);
  assert.equal(visitVolumesCsv([],'m').trim().split('\n').length,1);
});

test('visit CSV records support mode and mask identity without treating empty support as zero change',()=>{
  const row={name:'A',bounds:[0,0,1,1],added:0,removed:0,addedArea:0,removedArea:0,matchedCells:0,totalCells:4,outsideRegionCells:3,visit:'later',visitIndex:1};
  const csv=visitVolumesCsv([row],'m',{mode:'common_observed',sha256:'a'.repeat(64)});
  assert.ok(csv.includes('support_mode,common_mask_sha256'));
  assert.ok(csv.includes('false,,,,,,0,4,3,common_observed,'+'a'.repeat(64)));
});

test('explicit visit order survives unrelated additions, drops deselected IDs and appends new selections without sorting names',async()=>{
  const {reconcileVisitOrder}=await import('../src/volume-visits.ts');
  const previous=[3,1,2];assert.deepEqual(reconcileVisitOrder(previous,[1,2,3]),[3,1,2]);
  assert.deepEqual(reconcileVisitOrder(previous,[1,3,4]),[3,1,4]);assert.deepEqual(previous,[3,1,2]);
  assert.deepEqual(reconcileVisitOrder([],[]),[]);
  assert.throws(()=>reconcileVisitOrder([1,1],[1]));assert.throws(()=>reconcileVisitOrder([],[NaN]));
});
test('interval CSV retains direct pair identities, safe labels and common-mask linkage with empty support quantities blank',async()=>{
  const {intervalVolumesCsv}=await import('../src/volume-visits.ts');
  const row={name:'A',bounds:[0,0,1,1],added:3,removed:2,addedArea:1,removedArea:1,matchedCells:2,totalCells:3,outsideRegionCells:0,
    intervalIndex:1,fromVisitIndex:1,toVisitIndex:2,fromId:8,toId:9,fromVisit:'  =earlier',toVisit:'later, "two"'};
  const csv=intervalVolumesCsv([row,{...row,name:'Empty',matchedCells:0}],'m',{mode:'common_observed',sha256:'b'.repeat(64)});
  assert.ok(csv.includes('1,1,8,"\'  =earlier",2,9,"later, ""two""","A",0,0,1,1,"m",true,3,2,1,1,1,2,1,0,common_observed,'+'b'.repeat(64)));
  assert.ok(csv.includes('"Empty",0,0,1,1,"m",false,,,,,,0,3,0,common_observed,'));
});
