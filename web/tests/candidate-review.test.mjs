import { test } from 'node:test';
import assert from 'node:assert/strict';
import { emptyDecision, editDecision, reviewCounts, candidateReviewsCsv, REVIEW_NOTE_LIMIT } from '../src/candidate-review.ts';
const time='2026-10-23T12:00:00.000Z';
test('review edits retain exact human evidence, explicit pending state and immutable defaults',()=>{
  const first=emptyDecision(),second=emptyDecision();
  const edited=editDecision('deferred','Needs another pass\n座標を確認',time);
  assert.deepEqual(first,{status:'unreviewed',note:'',updatedAt:null});assert.notEqual(first,second);
  assert.deepEqual(edited,{status:'deferred',note:'Needs another pass\n座標を確認',updatedAt:time});
  assert.deepEqual(reviewCounts([first,edited,editDecision('reviewed','',time)]),{unreviewed:1,reviewed:1,deferred:1});
  assert.deepEqual(reviewCounts([]),{unreviewed:0,reviewed:0,deferred:0});
});
test('invalid review state, oversized comments, hidden controls and ambiguous dates fail without truncation',()=>{
  assert.throws(()=>editDecision('confirmed','',time),/state/);
  assert.throws(()=>editDecision('reviewed','x'.repeat(REVIEW_NOTE_LIMIT+1),time),/2,000/);
  assert.throws(()=>editDecision('reviewed','hidden\0value',time),/control/);
  for(const stamp of ['never','2026-10-23','2026-10-23T12:00:00Z'])assert.throws(()=>editDecision('reviewed','',stamp),/timestamp/);
  assert.equal(editDecision('reviewed','x'.repeat(REVIEW_NOTE_LIMIT),time).note.length,REVIEW_NOTE_LIMIT);
});
test('review CSV keeps directions and runtime identities distinct and escapes comments and spreadsheet labels',()=>{
  const row={direction:'reverse',sourceId:2,source:'=map',referenceId:1,reference:'truth',rank:1,points:9,maxDistance:1,meanDistance:.5,bounds:[0,1,2,3,4,5],...editDecision('deferred','  +SUM(1)\n"inspect"',time)};
  const csv=candidateReviewsCsv([row,{...row,direction:'forward',...emptyDecision()}]);
  assert.ok(csv.includes(`reverse,2,"'=map",1,"truth",1,deferred,"'  +SUM(1)\n""inspect""",${time},9,1,0.5,0,1,2,3,4,5`));
  assert.ok(csv.includes('forward,2,"\'=map",1,"truth",1,unreviewed,"",,9'));
  assert.equal(candidateReviewsCsv([]).trim().split('\n').length,1);
});
