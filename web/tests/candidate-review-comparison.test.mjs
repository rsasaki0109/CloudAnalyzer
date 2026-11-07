import { test } from 'node:test';
import assert from 'node:assert/strict';
import { compareReviewArchives, reviewComparisonCsv, reviewComparisonHtml, reviewComparisonSummary } from '../src/candidate-review-comparison.ts';
import { parseReviewArchive } from '../src/candidate-review-archive.ts';
import { locateDistanceRegions } from '../src/distance-regions.ts';
import { emptyDecision, editDecision, reviewCounts } from '../src/candidate-review.ts';
function fixture(count=4,hasChanges=true){
 const analysis=locateDistanceRegions(new Float32Array(Array.from({length:count},(_,i)=>[i*4,0,0]).flat()),new Float32Array(count).fill(hasChanges?1:0),[0,0,0],.3,1,1);
 const source={id:2,name:'map',loadedPoints:count,filePoints:count+5,keepEvery:2,preview:true,copcLevels:null};
 const c={direction:'forward',source,reference:{...source,id:1,name:'truth'},analysis,reusedDistances:false,review:{analyzedAt:'2026-10-24T00:00:00.000Z',counts:null,candidates:analysis.regions.map(r=>({...r,...emptyDecision()}))}};
 c.review.counts=reviewCounts(c.review.candidates);return {schema:'cloudanalyzer.distance_candidate_reviews.v1',comparisons:[c]};
}
function parsed(value){for(const c of value.comparisons)c.review.counts=reviewCounts(c.review.candidates);return parseReviewArchive(value);}
function both(value){const reverse=structuredClone(value.comparisons[0]);reverse.direction='reverse';[reverse.source,reverse.reference]=[reverse.reference,reverse.source];value.comparisons.push(reverse);return value;}
test('same archived diagnosis permits independent status, note, timestamp-only and unchanged rows, without mutating sources',()=>{
 const raw=fixture(),a=parsed(raw),bRaw=structuredClone(raw),c=bRaw.comparisons[0];
 Object.assign(c.review.candidates[0],editDecision('reviewed','checked','2026-10-24T01:00:00.000Z'));
 Object.assign(c.review.candidates[1],editDecision('unreviewed','comment only','2026-10-24T01:00:00.000Z'));
 Object.assign(c.review.candidates[2],editDecision('unreviewed','','2026-10-24T01:00:00.000Z'));
 const b=parsed(bRaw),before=structuredClone({a,b}),result=compareReviewArchives(a,b,'a.json','b.json'),d=result.directions[0];
 assert.equal(d.compatible,true);assert.deepEqual(d.rows.map(r=>r.changed),[['status','note','updatedAt'],['note','updatedAt'],['updatedAt'],[]]);
 assert.match(reviewComparisonSummary(d),/1 status changes, 2 comment changes, 1 timestamp-only changes, 1 unchanged/);
 assert.deepEqual({a,b},before);result.a.archive.comparisons[0].review.candidates[0].note='mutated';assert.deepEqual({a,b},before);
 const reverse=compareReviewArchives(b,a,'b','a');assert.equal(reverse.directions[0].rows[0].a.status,'reviewed');assert.equal(reverse.directions[0].rows[0].b.status,'unreviewed');
});
test('identical rank and filenames never pair candidates across changed analysis identity or geometry',()=>{
 const a=parsed(fixture());
 const changes=[c=>c.source.id=9,c=>c.source.name='other',c=>c.source.filePoints++,c=>c.source.keepEvery=1,c=>c.source.preview=false,c=>c.source.copcLevels=1,c=>c.reference.name='changed reference',
 c=>c.review.analyzedAt='2026-10-24T00:00:00.001Z',c=>c.reusedDistances=true,c=>c.analysis.cell=2,
 c=>{c.analysis.regions[0].bounds[0]++;c.analysis.regions[0].bounds[3]++;c.review.candidates[0].bounds=[...c.analysis.regions[0].bounds];},
 c=>{c.analysis.points++;c.analysis.unmeasured++;c.source.loadedPoints++;}];
 for(const change of changes){const raw=fixture();change(raw.comparisons[0]);const result=compareReviewArchives(a,parsed(raw),'same.json','same.json'),d=result.directions[0];assert.equal(d.compatible,false);assert.ok(d.reasons.length);assert.deepEqual(d.rows,[]);assert.match(reviewComparisonSummary(d),/No candidate transitions inferred/);}
});
test('directions are matched independently, missing directions and empty lists have no invented transitions, unlisted stay excluded',()=>{
 const a=parsed(both(fixture())),b=parsed(fixture());
 const result=compareReviewArchives(a,b,'A','B');assert.equal(result.directions[0].compatible,true);assert.equal(result.directions[1].compatible,false);assert.equal(result.directions[1].b,null);assert.deepEqual(result.directions[1].rows,[]);
 const raw=both(fixture());raw.comparisons[1].review.analyzedAt='2026-10-24T01:00:00.000Z';const oneMismatch=compareReviewArchives(a,parsed(raw),'A','B');assert.deepEqual(oneMismatch.directions.map(d=>d.compatible),[true,false]);
 for(const [count,changes,listed] of [[2,false,0],[53,true,50]]){const archive=parsed(fixture(count,changes)),d=compareReviewArchives(archive,archive,'A','B').directions[0];assert.equal(d.compatible,true);assert.equal(d.rows.length,listed);}
 const csv=reviewComparisonCsv(result);assert.match(csv,/reverse,false,"Direction is absent from one archive",/);assert.equal(csv.trimEnd().split('\n').at(-1).split(',').length,22);
});
test('comparison exports retain both complete records, formula-safe multiline comments and literal markup',()=>{
 const raw=fixture(),a=parsed(raw);Object.assign(raw.comparisons[0].review.candidates[0],editDecision('deferred',' \t=check, "x"\n<img src=x onerror=alert(1)>','2026-10-24T01:00:00.000Z'));
 const result=compareReviewArchives(a,parsed(raw),' =a.json','b</pre><script>alert(1)</script>.json');
 const csv=reviewComparisonCsv(result),html=reviewComparisonHtml(result);
 assert.ok(csv.includes('"\' =a.json"'));assert.ok(csv.includes('"\' \t=check, ""x""\n<img src=x onerror=alert(1)>"'));
 assert.ok(html.includes('&lt;img src=x'));assert.ok(!/<script|<img|<iframe/i.test(html));assert.ok(html.includes("default-src 'none'"));
 const json=html.match(/<pre id="comparison-json">([\s\S]*?)<\/pre>/)[1].replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&quot;','"').replaceAll('&#39;',"'").replaceAll('&amp;','&');assert.deepEqual(JSON.parse(json),result);
 assert.equal(result.a.archive.comparisons[0].review.candidates[0].status,'unreviewed');assert.equal(result.b.archive.comparisons[0].review.candidates[0].status,'deferred');
});
