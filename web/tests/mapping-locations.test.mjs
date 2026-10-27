import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {parseMappingLocations,locateMappingInterval} from '../src/mapping-locations.ts';
const digest='a'.repeat(64);
const source=()=>({schema:'cloudanalyzer.corridor_proposals.v1',coordinate_frame:'local_slam_metres',candidates:[{id:1,from_m:10,to_m:20,sections:[
  {station_m:10,left:[0,2,1],right:[0,-2,1]},{station_m:15,left:[5,3,2],right:[5,-3,2]},{station_m:20,left:[10,2,3],right:[10,-2,3]},
]}]});
test('clips source stations with explicit interpolation and retains interior geometry',()=>{
 const p=parseMappingLocations(source(),digest),r=locateMappingInterval(p,digest,1,12.5,17.5);
 assert.deepEqual(r.left,[[2.5,2.5,1.5],[5,3,2],[7.5,2.5,2.5]]);
 assert.deepEqual(r.right,[[2.5,-2.5,1.5],[5,-3,2],[7.5,-2.5,2.5]]);
 for(const [hash,id,a,b] of [[null,1,10,20],['b'.repeat(64),1,10,20],[digest,2,10,20],[digest,1,9,20],[digest,1,10,21],[digest,1,20,10]])assert.throws(()=>locateMappingInterval(p,hash,id,a,b));
});
test('rejects malformed, duplicate, incomplete or unsupported source geometry',()=>{
 for(const mutate of [v=>v.coordinate_frame='unknown',v=>v.candidates.push(v.candidates[0]),v=>v.candidates[0].sections[1].station_m=10,v=>v.candidates[0].from_m=9,v=>v.candidates[0].sections[0].left[0]=NaN]){
  const v=source();mutate(v);const p=parseMappingLocations(v,digest);assert.ok(p.unavailable);assert.equal(p.candidates.size,0);
 }
});
test('NCLT real source links only to the recorded matching proposal hash',()=>{
 const raw=readFileSync(new URL('../../benchmarks/vector-map/nclt-motion-pairs/april/corridor-proposals-trajectory.json',import.meta.url));
 const hash=createHash('sha256').update(raw).digest('hex'),p=parseMappingLocations(JSON.parse(raw),hash);
 assert.equal(p.unavailable,null);
 const run=JSON.parse(readFileSync(new URL('../../benchmarks/vector-map/nclt-motion-pairs/april/run.json',import.meta.url)));
 const step=run.history[2],d=step.action.decisions[0];
 assert.equal(step.proposal_sha256,hash);
 assert.equal(locateMappingInterval(p,step.proposal_sha256,d.candidate_id,d.from_m,d.to_m).left.length,2);
});
