import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startHeightCellReview, updateHeightCellReview, heightCellReviewSnapshot, heightCellReviewsCsv } from '../src/height-cell-review.ts';
const time = '2026-10-24T00:00:00.000Z';
function fixture() {
 const c = {index:27,x:1000000.25,y:2000000.5,before:0,after:1.0000000000000002,difference:1.0000000000000002};
 const review = {threshold:1,riseCells:51,fallCells:0,withinCells:4,unlistedCells:50,cellLimit:50,cells:[c]};
 const context = {label:'  =Survey, "A"',before:{id:1,name:'before'},after:{id:2,name:'after'},settings:{cell:2,height:'mean',fillEmpty:false,coordinateUnit:'m',region:[1,2,3,4]},support:{mode:'common_observed',sha256:'a'.repeat(64)}};
 return {review,context};
}
test('review snapshots retain copied bounded f64 evidence, support and independent comparison decisions',()=>{
 const {review,context} = fixture(), a=startHeightCellReview(review,context,time), b=startHeightCellReview(review,context,time);
 review.cells[0].after=99;context.settings.region[0]=99;context.before.name='changed';
 updateHeightCellReview(a,0,'reviewed','目視\n<img src=x>',time);
 const snapshot=heightCellReviewSnapshot([a,b]);assert.equal(snapshot.schema,'cloudanalyzer.height_cell_reviews.v1');
 const [first,second]=snapshot.comparisons;assert.equal(first.cells[0].after,1.0000000000000002);assert.deepEqual(first.settings.region,[1,2,3,4]);assert.equal(first.before.name,'before');
 assert.deepEqual(first.counts,{unreviewed:0,reviewed:1,deferred:0});assert.equal(first.screening.unlistedCells,50);assert.equal(first.cells.length,1);
 assert.equal(second.cells[0].status,'unreviewed');assert.equal(second.cells[0].updatedAt,null);assert.equal(first.cells[0].note,'目視\n<img src=x>');
 first.settings.region[0]=88;first.cells[0].note='mutated';assert.equal(a.context.settings.region[0],1);assert.equal(a.decisions[0].note,'目視\n<img src=x>');
});
test('rejected edits cannot replace valid decisions and only listed ranks can be edited',()=>{
 const {review,context}=fixture(),session=startHeightCellReview(review,context,time);const saved=updateHeightCellReview(session,0,'deferred','valid',time);
 for (const rank of [-1,1,.5,NaN]) assert.throws(()=>updateHeightCellReview(session,rank,'reviewed','',time),/listed/);
 assert.throws(()=>updateHeightCellReview(session,0,'reviewed','x'.repeat(2001),time),/2,000/);
 assert.throws(()=>updateHeightCellReview(session,0,'accepted','',time),/state/);
 assert.deepEqual(session.decisions[0],saved);
});
test('review CSV preserves f64 cell identity, bounded coverage, timestamps and formula-safe multiline notes',()=>{
 const {review,context}=fixture(),session=startHeightCellReview(review,context,time);updateHeightCellReview(session,0,'deferred',' \t+SUM(1)\n"再確認"',time);
 const csv=heightCellReviewsCsv([session]);assert.ok(csv.includes('"\'  =Survey, ""A"""'));assert.ok(csv.includes('"\' \t+SUM(1)\n""再確認"""'));
 assert.ok(csv.includes('1,27,1000000.25,2000000.5,0,1.0000000000000002,1.0000000000000002,deferred'));
 assert.ok(csv.includes(time));assert.equal(csv.endsWith(',50\n'),true);assert.equal(heightCellReviewsCsv([]).trimEnd().split('\n').length,1);
});
