import { test } from 'node:test';
import assert from 'node:assert/strict';
import { compareSections, validateSection, SECTION_INPUT_BYTES, SECTION_POINT_LIMIT } from '../src/distance-section.ts';
const cloud = (points,shift=[0,0,0])=>({positions:Float32Array.from(points.flat()),shift});
const settings = {axis:'x',center:[0,0],length:4,width:2};
test('section preserves original world heights and separate counts for two shifted clouds',()=>{
  const a=cloud([[-2,0,1],[0,0,2],[2,0,3]],[100,200,300]),b=cloud([[2,0,10]],[98,200,300]);
  const before=a.positions.slice(),r=compareSections(a,b,{...settings,center:[100,200]});
  assert.deepEqual(Array.from(r.source.along),[0,2,4]);assert.deepEqual(Array.from(r.source.positions),[98,200,301,100,200,302,102,200,303]);
  assert.equal(r.source.selectedPoints,3);assert.equal(r.source.zMin,301);assert.equal(r.source.zMax,303);
  assert.equal(r.reference.selectedPoints,1);assert.equal(r.reference.zMin,310);assert.deepEqual(a.positions,before);
});
test('section includes band and end boundaries, excludes outside points, and swaps XY axes',()=>{
  const a=cloud([[-2,-1,0],[2,1,0],[0,1.01,0],[2.01,0,0],[0,1.5,0]]);
  const x=compareSections(a,cloud([]),settings);assert.equal(x.source.selectedPoints,2);assert.equal(x.reference.selectedPoints,0);assert.equal(x.reference.zMin,null);
  const y=compareSections(a,cloud([]),{...settings,axis:'y'});assert.equal(y.source.selectedPoints,2);
  assert.deepEqual(Array.from(y.source.positions),[0,Math.fround(1.01),0,0,1.5,0]);
});
test('section counts nonfinite rows without treating unknown positions as a missing surface',()=>{
  const r=compareSections(cloud([[NaN,0,0],[0,Infinity,0],[0,0,NaN],[100,100,0],[0,0,5]]),cloud([]),settings);
  assert.equal(r.source.points,5);assert.equal(r.source.invalidPoints,3);assert.equal(r.source.finitePoints,2);assert.equal(r.source.selectedPoints,1);
  assert.equal(r.source.finitePoints+r.source.invalidPoints,r.source.points);
});
test('section rejects malformed dimensions, coordinates and XYZ layouts',()=>{
  for(const s of [{...settings,axis:'z'},{...settings,length:0},{...settings,width:NaN},{...settings,center:[Infinity,0]},
    {...settings,center:[1e30,0]},{...settings,center:[1.5e308,0],length:1.5e308}])assert.throws(()=>validateSection(s));
  assert.throws(()=>compareSections({positions:Float32Array.of(1),shift:[0,0,0]},cloud([]),settings),/XYZ/);
  assert.throws(()=>compareSections(cloud([], [0,NaN,0]),cloud([]),settings),/shift/);
});
test('section input and output budgets reject excess without silently sampling',()=>{
  const a={positions:new Float32Array(SECTION_POINT_LIMIT*3),shift:[0,0,0]};
  assert.equal(compareSections(a,cloud([]),settings).source.selectedPoints,SECTION_POINT_LIMIT);
  assert.throws(()=>compareSections({positions:new Float32Array((SECTION_POINT_LIMIT+1)*3),shift:[0,0,0]},cloud([]),settings),/50,000/);
  const n=Math.floor(SECTION_INPUT_BYTES/24)+1;
  assert.throws(()=>compareSections({positions:new Float32Array(n*3),shift:[0,0,0]}, {positions:new Float32Array(n*3),shift:[0,0,0]},settings),/64 MiB/);
});
