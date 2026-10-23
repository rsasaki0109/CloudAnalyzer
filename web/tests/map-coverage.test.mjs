import assert from 'node:assert/strict';
import { test } from 'node:test';
import { analyzeCoverage, MAX_COVERAGE_INPUT_BYTES } from '../src/map-coverage.ts';
const xyz=points=>Float32Array.from(points.flatMap(p=>[p[0],p[1],p[2]??0]));
const ring=()=>Array.from({length:25},(_,i)=>[i%5+.25,Math.floor(i/5)+.25,0]).filter(([x,y])=>x===.25||x===4.25||y===.25||y===4.25);
test('all loaded points contribute to exact occupied counts and user-defined sparse density',()=>{
  const raw=xyz([[0,0],[0,0],[1,0],[1,1],[1,1],[1,1]]),before=raw.slice();
  const g=analyzeCoverage(raw,[500000,4000000,30],1,3);
  assert.deepEqual(g.counts,Uint32Array.from([2,1,0,3]));assert.equal(g.sparse,2);assert.equal(g.occupied,3);
  assert.equal(g.empty,1);assert.equal(g.enclosedEmpty,0);assert.equal(g.finitePoints,6);
  assert.deepEqual(g.density,{min:1,median:2,max:3,mean:2});assert.equal(g.minX,500000);assert.equal(g.minY,4000000);
  assert.deepEqual(raw,before);
});
test('closed observed ring has nine enclosed empty cells, never an automatic defect decision',()=>{
  const g=analyzeCoverage(xyz(ring().flatMap(p=>[p,p])),[0,0,0],1,3);
  assert.equal(g.nx,5);assert.equal(g.ny,5);assert.equal(g.occupied,16);assert.equal(g.enclosedEmpty,9);
  assert.equal(g.sparse,16);assert.equal(g.counts.reduce((a,b)=>a+b),32);
  assert.equal(g.categories.filter(v=>v===3).length,9);
});
test('eight-neighbor diagonal access to the boundary prevents false enclosed-gap flags',()=>{
  const points=ring().filter(([x,y])=>x!==.25||y!==.25);
  const g=analyzeCoverage(xyz(points),[0,0,0],1,1);assert.equal(g.enclosedEmpty,0);
  assert.equal(g.categories.filter(v=>v===0).length,10);
});
test('XY projection combines elevations, excludes invalid XYZ and normalizes density by area',()=>{
  const g=analyzeCoverage(xyz([[0,0,-4],[0,0,20],[NaN,0,0],[0,0,Infinity]]),[0,0,100],2,2);
  assert.equal(g.points,4);assert.equal(g.finitePoints,2);assert.equal(g.invalidPoints,2);
  assert.equal(g.occupied,1);assert.equal(g.sparse,0);assert.equal(g.minZ,96);assert.equal(g.maxZ,120);
  assert.equal(g.density.median,.5);
});
test('grid and input limits reject oversized work instead of sampling silently',()=>{
  assert.throws(()=>analyzeCoverage(xyz([[0,0],[1000,1000]]),[0,0,0],1,1),/262,144/);
  assert.throws(()=>analyzeCoverage(xyz([[0,0],[3000,0]]),[0,0,0],1,1),/2,048/);
  assert.throws(()=>analyzeCoverage(new Float32Array(Math.floor(MAX_COVERAGE_INPUT_BYTES/12)*3+3),[0,0,0],1,1),/64 MiB/);
});
test('bad parameters, empty/nonfinite sources and unrepresentable coordinates reject',()=>{
  for(const cell of [0,-1,NaN,Infinity])assert.throws(()=>analyzeCoverage(xyz([[0,0]]),[0,0,0],cell,1));
  for(const n of [0,1.5,NaN,1000001])assert.throws(()=>analyzeCoverage(xyz([[0,0]]),[0,0,0],1,n));
  assert.throws(()=>analyzeCoverage(new Float32Array(),[0,0,0],1,1));
  assert.throws(()=>analyzeCoverage(xyz([[NaN,0]]),[0,0,0],1,1),/finite XYZ/);
  assert.throws(()=>analyzeCoverage(xyz([[0,0]]),[1e20,0,0],1,1),/coordinate frame/);
  assert.throws(()=>analyzeCoverage(xyz([[0,0]]),[0,0,0],1e-160,1),/Density/);
});
