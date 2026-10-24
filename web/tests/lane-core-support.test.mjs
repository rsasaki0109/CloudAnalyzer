import assert from 'node:assert/strict';
import test from 'node:test';
import { readCandidateCores, laneCoreSupport, computeLaneCoreEvidence } from '../src/lane-core-support.ts';
const lane=(id,z=0)=>({id,left:[[0,2,z],[10,2,z]],right:[[0,-2,z],[10,-2,z]]});

test('native core selection keeps float64 coordinates, original indices and bounded block reads',()=>{
  const count=9000, labels=new Float32Array(count).fill(NaN);labels[1]=1;labels[4095]=1;labels[8193]=1;
  const reads=[];
  const source={length:count,attribute:()=>labels,positionsRange:(a,b)=>{
    reads.push([a,b]);return Float64Array.from(Array.from({length:(b-a)*3},(_,i)=>300000000+(a+Math.floor(i/3))*.001+(i%3)*.0001));
  }};
  const cores=readCandidateCores(source,1,3);
  assert.deepEqual([...cores.indices],[1,4095,8193]);
  assert.deepEqual(reads,[[0,4096],[8192,9000]]);
  assert.equal(cores.positions[0],300000000.001);
  assert.notEqual(cores.positions[0],Math.fround(cores.positions[0]));
  assert.throws(()=>readCandidateCores(source,1,2),/count/);
  assert.throws(()=>readCandidateCores(source,1,3,[0,0,0,1,1,1]),/bounds/);
  assert.throws(()=>readCandidateCores({...source,length:2000001,attribute:()=>{throw Error('must not allocate labels');}},1,3),/limit/);
});

test('individual cores reject an empty-space box link and distinguish stacked roads',()=>{
  const positions=new Float64Array([1,0,0,5,0,.5,5,0,8,4,5,0]);
  const lower=laneCoreSupport(lane(1),positions,1),upper=laneCoreSupport(lane(2,8),positions,1);
  assert.equal(lower.status,'complete');assert.equal(lower.supported,2);assert.deepEqual([...lower.mask],[1,1,0,0]);
  assert.equal(upper.supported,1);assert.deepEqual([...upper.mask],[0,0,1,0]);
  const boxOnly=laneCoreSupport(lane(3),new Float64Array([1,3,0,2,-3,0]),1);
  assert.equal(boxOnly.supported,0);assert.equal(boxOnly.status,'complete');
});

test('local barycentric Z, boundary points and shared triangle seams match the independent protocol',()=>{
  const slope={id:1,left:[[0,2,0],[10,2,10]],right:[[0,-2,0],[10,-2,10]]};
  const result=laneCoreSupport(slope,new Float64Array([1,0,9,5,0,5,10,2,11,10,2,11.01]),1);
  assert.equal(result.supported,2);assert.deepEqual([...result.mask],[0,1,1,0]);
  assert.equal(laneCoreSupport(lane(2),new Float64Array([5,0,0]),0).supported,1);
  const origin=300000000;
  const precise={id:3,left:[[origin,1,0],[origin+.2,1,0]],right:[[origin,-1,0],[origin+.2,-1,0]]};
  assert.equal(laneCoreSupport(precise,new Float64Array([origin+.1,0,0]),0).supported,1);
});

test('missing labels, invalid geometry and work limits stay unavailable, never zero',()=>{
  assert.throws(()=>readCandidateCores({length:8,attribute:()=>null,positionsRange:()=>new Float64Array()},1,8),/labels/);
  const points=new Float64Array([1,0,0]);
  for(const row of [laneCoreSupport(lane(1),points,1,1),laneCoreSupport(lane(1),new Float64Array([NaN,0,0]),1),
    laneCoreSupport({id:2,left:[[0,0,0],[0,0,1]],right:[[0,0,0],[0,0,1]]},points,1)]){
    assert.equal(row.status,'unavailable');assert.equal(row.supported,null);assert.equal(row.mask,null);assert.ok(row.reason);
  }
  const cores={positions:points,indices:new Uint32Array([0]),sourceCount:1};
  assert.equal(computeLaneCoreEvidence(cores,[lane(1),lane(2,8)],1).rows.length,2);
  assert.throws(()=>computeLaneCoreEvidence(cores,Array.from({length:51},(_,i)=>lane(i)),1),/50 lanes/);
});
