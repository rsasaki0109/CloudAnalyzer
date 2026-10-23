import { test } from 'node:test';
import assert from 'node:assert/strict';
import { locateDistanceRegions, inspectDistanceCells, distanceCellsCsv, distanceRegionsCsv, REGION_INPUT_BYTES } from '../src/distance-regions.ts';
const xyz=points=>Float32Array.from(points.flatMap(p=>[p[0],p[1],p[2]??0]));
test('regions rank actual exceedances, preserve world bounds and count unknown rows separately',()=>{
  const p=xyz([[0,0],[.25,0],[2,2,1],[2.5,2,1],[10,0,2],[NaN,0],[0,0,NaN],[5,5]]);
  const d=Float32Array.from([0,.3,.5,.75,2,5,8,NaN]),a=p.slice(),b=d.slice();
  const r=locateDistanceRegions(p,d,[100,200,300],.3,1,1);
  assert.equal(r.measured,5);assert.equal(r.invalidPositions,2);assert.equal(r.unmeasured,1);assert.equal(r.exceeding,3);
  assert.equal(r.regions.length,2);assert.deepEqual(r.regions[0],{rank:1,points:1,maxDistance:2,meanDistance:2,bounds:[110,200,302,110,200,302]});
  assert.equal(r.regions[1].points,2);assert.equal(r.regions[1].meanDistance,.625);
  assert.deepEqual(p,a);assert.deepEqual(d,b);
});
test('eight-neighbor cell connectivity and minimum support keep omitted points visible in accounting',()=>{
  const r=locateDistanceRegions(xyz([[0,0],[1.25,1.25],[2.25,2.25],[9,0]]),Float32Array.from([1,1,1,2]),[0,0,0],.3,1,2);
  assert.equal(r.components,2);assert.equal(r.eligibleRegions,1);assert.equal(r.regions[0].points,3);assert.equal(r.smallRegionPoints,1);
  assert.equal(r.regions.reduce((n,g)=>n+g.points,0)+r.smallRegionPoints+r.unlistedEligiblePoints,r.exceeding);
});
test('top-50 list cap retains complete counts and deterministic ranking of ties',()=>{
  const p=Array.from({length:60},(_,i)=>[i*4,0]),d=new Float32Array(60).fill(1);
  const r=locateDistanceRegions(xyz(p),d,[0,0,0],0,1,1);
  assert.equal(r.eligibleRegions,60);assert.equal(r.regions.length,50);assert.equal(r.unlistedEligiblePoints,10);
  assert.equal(r.regions[49].bounds[0],196);
  const reversed=locateDistanceRegions(xyz(p.toReversed()),d,[0,0,0],0,1,1);assert.deepEqual(r.regions,reversed.regions);
  const small=locateDistanceRegions(xyz(p),d,[0,0,0],0,1,2);assert.equal(small.smallRegionPoints,60);assert.equal(small.regions.length,0);
});
test('the threshold has the precision of stored f32 distances and is strictly exceeded',()=>{
  const r=locateDistanceRegions(xyz([[0,0],[5,0]]),Float32Array.from([.3,.300001]),[0,0,0],.3,1,1);
  assert.equal(r.effectiveThreshold,Math.fround(.3));assert.equal(r.exceeding,1);
  const none=locateDistanceRegions(xyz([[0,0]]),Float32Array.of(0),[0,0,0],0,1,1);assert.equal(none.regions.length,0);
  const unknown=locateDistanceRegions(xyz([[0,0],[1,0]]),Float32Array.from([NaN,Infinity]),[0,0,0],0,1,1);
  assert.equal(unknown.measured,0);assert.equal(unknown.unmeasured,2);assert.equal(unknown.exceeding,0);
});
test('invalid settings, signed distances, mismatched rows and unsafe coordinates fail',()=>{
  const p=xyz([[0,0]]),d=Float32Array.of(1);
  for(const args of [[-1,1,1],[NaN,1,1],[Infinity,1,1],[1e100,1,1],[0,0,1],[0,1,0],[0,1,1.5]])assert.throws(()=>locateDistanceRegions(p,d,[0,0,0],...args));
  assert.throws(()=>locateDistanceRegions(p,Float32Array.of(-1),[0,0,0],0,1,1),/Unsigned/);
  assert.throws(()=>locateDistanceRegions(p,new Float32Array(),[0,0,0],0,1,1),/matching/);
  assert.throws(()=>locateDistanceRegions(p,d,[1e30,0,0],0,1,1),/coordinate frame/);
  assert.throws(()=>locateDistanceRegions(xyz([[0,0],[1,0]]),Float32Array.of(1,1),[0,0,0],0,1e-30,1),/extent/);
});
test('work limits reject instead of truncating cells or silently sampling input',()=>{
  const n=50001,p=xyz(Array.from({length:n},(_,i)=>[i*3,0]));
  assert.throws(()=>locateDistanceRegions(p,new Float32Array(n).fill(1),[0,0,0],0,1,1),/50,000/);
  const count=Math.floor(REGION_INPUT_BYTES/16)+1;
  assert.throws(()=>locateDistanceRegions(new Float32Array(count*3),new Float32Array(count),[0,0,0],0,1,1),/64 MiB/);
});
test('CSV contains exactly the ranked, listed world-coordinate regions and a valid empty header',()=>{
  const r=locateDistanceRegions(xyz([[1,2,3]]),Float32Array.of(2),[10,20,30],0,1,1);
  const csv=distanceRegionsCsv(r).trim().split('\n');assert.equal(csv.length,2);assert.equal(csv[1],'1,1,2,2,11,22,33,11,22,33');
  r.regions=[];assert.equal(distanceRegionsCsv(r).trim().split('\n').length,1);
});

test('inspection selects exact component membership even with another component inside its bounding box',()=>{
  // A perimeter surrounds a separate center cell without connecting to it.
  const ring=[];for(let x=0;x<=6;x++)for(let y=0;y<=6;y++)if(x===0||x===6||y===0||y===6)ring.push([x,y]);
  const p=xyz([...ring,[3,3],[0,0,1],[0,0,2]]),d=Float32Array.from([...ring.map(()=>2),9,.1,NaN]);
  const before=p.slice(),dist=d.slice();const r=inspectDistanceCells(p,d,[100,200,300],.3,1,1,2,.5);
  assert.equal(r.regionPoints,24);assert.equal(r.occupiedCells,24);assert.equal(r.unlistedPoints,0);
  assert.equal(r.cells.reduce((n,c)=>n+c.points,0),24);assert.ok(r.cells.every(c=>c.maxDistance===2));
  assert.deepEqual(r.gridOrigin,[100,200]);assert.deepEqual(r.cells[0].bounds,[100,200,300,100,200,300]);
  assert.deepEqual(p,before);assert.deepEqual(d,dist);
});

test('inspection preserves all region points, lists top 50 cells and does not filter low support cells',()=>{
  const p=xyz(Array.from({length:61},(_,i)=>[i,0])),d=Float32Array.from({length:61},(_,i)=>i+1);
  const r=inspectDistanceCells(p,d,[0,0,0],0,100,60,1,.5);
  assert.equal(r.regionPoints,61);assert.equal(r.occupiedCells,61);assert.equal(r.cells.length,50);assert.equal(r.unlistedPoints,11);
  assert.equal(r.cells[0].maxDistance,61);assert.equal(r.cells[49].maxDistance,12);
  assert.equal(r.cells.reduce((n,c)=>n+c.points,0)+r.unlistedPoints,r.regionPoints);
  const reverse=inspectDistanceCells(xyz(Array.from({length:61},(_,i)=>[60-i,0])),d.toReversed(),[0,0,0],0,100,60,1,.5);
  assert.deepEqual(reverse,r);assert.equal(distanceCellsCsv(r).trim().split('\n').length,51);
  assert.equal(distanceCellsCsv(r).split('\n')[1],'1,1,120,0,1,61,61,60,0,0,60,0,0');
});

test('inspection rejects absent selections, unsafe grids and over-budget fine cells without truncation',()=>{
  const p=xyz([[0,0],[1,0]]),d=Float32Array.of(1,1);
  for(const [rank,size] of [[0,1],[51,1],[1.5,1],[1,0],[1,2],[1,NaN]])assert.throws(()=>inspectDistanceCells(p,d,[0,0,0],0,1,1,rank,size));
  assert.throws(()=>inspectDistanceCells(p,d,[0,0,0],0,1,1,2,.5),/no longer/);
  assert.throws(()=>inspectDistanceCells(p,d,[0,0,0],0,1,1,1,1e-30),/extent/);
  assert.throws(()=>inspectDistanceCells(p,d,[1e16,0,0],0,10,1,1,.1),/extent/);
  const n=50001;
  assert.throws(()=>inspectDistanceCells(xyz(Array.from({length:n},(_,i)=>[i,0])),new Float32Array(n).fill(1),[0,0,0],0,100000,1,1,.5),/50,000/);
});
