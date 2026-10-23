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
