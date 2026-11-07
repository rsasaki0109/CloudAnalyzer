import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseReviewArchive, archiveCsv, archiveHtml } from '../src/candidate-review-archive.ts';
import { locateDistanceRegions } from '../src/distance-regions.ts';
import { emptyDecision, editDecision, reviewCounts } from '../src/candidate-review.ts';
function fixture() {
  const analysis=locateDistanceRegions(Float32Array.of(0,0,1,10,0,1),Float32Array.of(1,1),[0,0,0],.3,2,1);
  const source={id:2,name:'<img src=x onerror=alert(1)>',loadedPoints:2,filePoints:4,keepEvery:2,preview:true,copcLevels:null};
  const reference={...source,id:1,name:'reference',filePoints:2,keepEvery:1,preview:false};
  const candidates=analysis.regions.map((r,i)=>({...r,...(i?emptyDecision():editDecision('deferred','=Needs\n"inspection"','2026-10-24T01:00:00.000Z'))}));
  const comparison={direction:'forward',source,reference,analysis,reusedDistances:false,review:{analyzedAt:'2026-10-24T00:00:00.000Z',counts:reviewCounts(candidates),candidates}};
  return {schema:'cloudanalyzer.distance_candidate_reviews.v1',comparisons:[comparison]};
}
test('archive parser accepts actual grouping output and retains independent directions without mutating source',()=>{
  const value=fixture(),reverse=structuredClone(value.comparisons[0]);reverse.direction='reverse';[reverse.source,reverse.reference]=[reverse.reference,reverse.source];
  value.comparisons.push(reverse);const parsed=parseReviewArchive(value);
  assert.equal(parsed.comparisons.length,2);assert.equal(parsed.comparisons[0].source.preview,true);
  parsed.comparisons[0].review.candidates[0].note='changed';assert.equal(value.comparisons[0].review.candidates[0].note,'=Needs\n"inspection"');
});
test('archive rejects inconsistent schema, directions, geometry, settings, counts and decision timestamps',()=>{
  const mutations=[
    v=>v.schema='other',v=>v.comparisons=[],v=>v.comparisons.push(structuredClone(v.comparisons[0])),
    v=>v.comparisons[0].direction='reverse',v=>v.comparisons[0].source.loadedPoints=3,
    v=>v.comparisons[0].source.filePoints=1,v=>v.comparisons[0].source.preview='yes',
    v=>v.comparisons[0].analysis.measured=99,v=>v.comparisons[0].analysis.effectiveThreshold=.3,
    v=>v.comparisons[0].analysis.regions[0].bounds[0]=10,v=>v.comparisons[0].review.candidates[0].points=9,
    v=>v.comparisons[0].review.counts.deferred=0,v=>v.comparisons[0].review.candidates[0].updatedAt=null,
    v=>v.comparisons[0].review.candidates[0].note='x'.repeat(2001),v=>v.comparisons[0].review.analyzedAt='yesterday',
    v=>v.comparisons[0].review.candidates[1].status='reviewed',
  ];
  for(const mutate of mutations){const value=fixture();mutate(value);assert.throws(()=>parseReviewArchive(value));}
  const value=fixture(),reverse=structuredClone(value.comparisons[0]);reverse.direction='reverse';[reverse.source,reverse.reference]=[reverse.reference,reverse.source];reverse.analysis.cell=3;value.comparisons.push(reverse);assert.throws(()=>parseReviewArchive(value));
});
test('archive handles empty results and limited candidate lists with explicit unlisted support',()=>{
  for(const count of [0,53]){
    const value=fixture(),c=value.comparisons[0];
    const n=count||2;c.analysis=locateDistanceRegions(new Float32Array(Array.from({length:n},(_,i)=>[i*4,0,0]).flat()),new Float32Array(n).fill(count?1:0),[0,0,0],.3,1,1);
    c.source.loadedPoints=n;c.source.filePoints=n;c.review.candidates=c.analysis.regions.map(r=>({...r,...emptyDecision()}));c.review.counts=reviewCounts(c.review.candidates);
    const parsed=parseReviewArchive(value);assert.equal(parsed.comparisons[0].review.candidates.length,count?50:0);
    assert.equal(parsed.comparisons[0].analysis.unlistedEligiblePoints,count?3:0);
  }
});
test('HTML keeps imported markup literal and CSV protects formula comments with multiline quotes',()=>{
  const archive=parseReviewArchive(fixture()),html=archiveHtml(archive),csv=archiveCsv(archive);
  assert.ok(html.startsWith('<!doctype html>'));assert.ok(html.includes('&lt;img src=x onerror=alert(1)&gt;'));assert.ok(!html.includes('<img src=x'));
  assert.ok(html.includes('Historical file only'));assert.ok(html.includes('keep every 2, preview true'));
  assert.ok(csv.includes(`"'=Needs\n""inspection"""`));assert.equal(archive.comparisons[0].review.candidates[0].note,'=Needs\n"inspection"');
});
