import { test } from 'node:test';
import assert from 'node:assert/strict';
import { validateVolumeGrid, sharedVolumeGrid, volumeCellsCsv, VOLUME_CSV_CELL_LIMIT } from '../src/surface-volume.ts';

test('volume grid budget follows the existing union grid and nearest-center edge rule',()=>{
  assert.doesNotThrow(()=>validateVolumeGrid([[0,0,0,999,999,1]],1));
  assert.throws(()=>validateVolumeGrid([[0,0,0,1000,999,1]],1),/1,000,000/);
  assert.throws(()=>validateVolumeGrid([[0,0,0,0,0,0],[2000,2000,0,2000,2000,1]],1),/1,000,000/);
  assert.doesNotThrow(()=>validateVolumeGrid([[1,2,3,1,2,3]],1));
});
test('volume rejects invalid or unsafe cell sizes and undefined surfaces before allocation',()=>{
  for(const cell of [0,-1,NaN,Infinity,1e200,1e-200,1e-30])assert.throws(()=>validateVolumeGrid([[0,0,0,1,1,1]],cell));
  for(const bounds of [[],[[0,0,0,Infinity,1,1]],[[2,0,0,1,1,1]]])assert.throws(()=>validateVolumeGrid(bounds,1));
});
test('volume CSV retains signed f32 differences and shifted world after-cell coordinates without modifying inputs',()=>{
  const p=Float32Array.of(1,2,3,4,5,6),d=Float32Array.of(.5,-1),before=p.slice();
  const csv=volumeCellsCsv(p,[100,200,300],d).trim().split('\n');
  assert.deepEqual(csv,['cell_index,x,y,after_height,after_minus_before','0,101,202,303,0.5','1,104,205,306,-1']);
  assert.deepEqual(p,before);assert.deepEqual(d,Float32Array.of(.5,-1));
});
test('volume CSV fails closed on oversized or undefined cells and still exports an empty header',()=>{
  assert.equal(volumeCellsCsv(new Float32Array(),[0,0,0],new Float32Array()).trim().split('\n').length,1);
  assert.throws(()=>volumeCellsCsv(Float32Array.of(1),[0,0,0],Float32Array.of(1)),/matching/);
  assert.throws(()=>volumeCellsCsv(Float32Array.of(NaN,0,0),[0,0,0],Float32Array.of(1)),/undefined/);
  assert.throws(()=>volumeCellsCsv(Float32Array.of(0,0,0),[0,0,0],Float32Array.of(Infinity)),/undefined/);
  const n=VOLUME_CSV_CELL_LIMIT+1;
  assert.throws(()=>volumeCellsCsv(new Float32Array(n*3),[0,0,0],new Float32Array(n)),/100,000/);
});

test('shared visit grid uses all selected extents and exports boundary origin and dimensions',()=>{
  assert.deepEqual(sharedVolumeGrid([[0,0,10,2,1,10],[-.6,0,10,1,1,10]],1),[-1.1,-.5,4,2]);
});

test('height screening accepts zero and explicit finite thresholds and ignores unused invalid values',async()=>{
 const {readHeightThreshold}=await import('../src/surface-volume.ts');
 assert.equal(readHeightThreshold(true,'0'),0);assert.equal(readHeightThreshold(true,' 0.5 '),.5);
 assert.equal(readHeightThreshold(false,''),undefined);assert.equal(readHeightThreshold(false,'bad'),undefined);
 for(const value of ['', ' ', '-1', 'NaN', 'Infinity'])assert.throws(()=>readHeightThreshold(true,value),/nonnegative/);
});
test('height screening CSV distinguishes empty support from within-band coverage and preserves full screened differences',async()=>{
 const {heightReviewCsv}=await import('../src/surface-volume.ts');
 const h={threshold:.5,riseCells:1,fallCells:1,withinCells:3,riseArea:4,fallArea:4,riseVolume:8,fallVolume:12};
 assert.ok(heightReviewCsv(h,5,'m').includes('0.5,"m",true,1,1,3,4,4,8,12,-4'));
 const empty={...h,riseCells:0,fallCells:0,withinCells:0,riseArea:0,fallArea:0,riseVolume:0,fallVolume:0};
 assert.ok(heightReviewCsv(empty,0,'m').includes('0.5,"m",false,0,0,0,,,,,'));
 assert.ok(heightReviewCsv({...empty,withinCells:5},5,'m').includes('0.5,"m",true,0,0,5,0,0,0,0,0'));
});
