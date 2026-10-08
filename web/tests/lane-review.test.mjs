import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LaneReviews, parseReviews, reviewCsv } from '../src/lane-review.ts';

test('edits invalidate affected reviews while retaining notes and previous decisions', () => {
  const r = new LaneReviews();
  r.save(1,'shared boundary A','reviewed','Check curb');
  r.save(2,'shared boundary A','deferred','Wait for survey');
  r.save(3,'other road','reviewed','OK');
  r.reconcile(new Map([[1,'shared boundary B'],[2,'shared boundary B'],[3,'other road']]));
  assert.equal(r.get(1).status,'unreviewed');
  assert.equal(r.get(1).previousStatus,'reviewed');
  assert.equal(r.get(1).notes,'Check curb');
  assert.equal(r.get(2).status,'unreviewed');
  assert.equal(r.get(3).status,'reviewed');
});
test('no-op updates preserve reviews, deleted IDs cannot inherit reviews, source changes stale all', () => {
  const r = new LaneReviews(); r.save(7,'same','needs-fix','bad width');
  r.reconcile(new Map([[7,'same']])); assert.equal(r.get(7).status,'needs-fix');
  const saved = r.snapshot(); const restored = new LaneReviews();
  restored.restore(saved,new Map([[7,'same']])); assert.equal(restored.get(7).notes,'bad width');
  restored.invalidateSource(); assert.equal(restored.get(7).status,'unreviewed');
  assert.match(restored.get(7).staleReason,/source changed/);
  r.reconcile(new Map()); r.reconcile(new Map([[7,'same']])); assert.equal(r.get(7),undefined);
  assert.throws(() => parseReviews([...saved,...saved]));
});

test('review rows include untouched live lanes and retain stale decisions without geometry', () => {
  const r=new LaneReviews();
  r.save(9,'large geometry signature','reviewed','確認した縁石');
  r.save(30,'removed lane','needs-fix','do not export');
  r.reconcile(new Map([[9,'new geometry']]));
  const rows=r.rows([12,9,12]);
  assert.deepEqual(rows.map(row=>row.lane),[9,12]);
  assert.equal(rows[0].status,'unreviewed');
  assert.equal(rows[0].previousStatus,'reviewed');
  assert.equal(rows[0].notes,'確認した縁石');
  assert.match(rows[0].staleReason,/changed/);
  assert.equal(Object.hasOwn(rows[0],'signature'),false);
  assert.equal(rows[1].status,'unreviewed');
  assert.equal(rows[1].updated,'');
  rows[0].notes='changed copy';
  assert.equal(r.get(9).notes,'確認した縁石');
});

test('review CSV preserves Unicode, commas, quotes and embedded line breaks', () => {
  const output=reviewCsv([{lane:7,status:'needs-fix',notes:'縁石, "実測"\n幅を再確認',updated:'2026-10-08T09:00:00Z',previousStatus:'reviewed',staleReason:'Rule changed'}]);
  assert.equal(output,'\uFEFFlane_id,status,notes,updated_at,previous_status,stale_reason\r\n7,"needs-fix","縁石, ""実測""\n幅を再確認","2026-10-08T09:00:00Z","reviewed","Rule changed"\r\n');
  assert.equal(reviewCsv([]),'\uFEFFlane_id,status,notes,updated_at,previous_status,stale_reason\r\n');
});

test('review CSV neutralizes spreadsheet formulas in all imported text fields', () => {
  for (const notes of ['=SUM(1,2)','+1+2','-1+2','@SUM(1,2)',' \t=1+2','\tplain','\rplain','\nplain']) {
    const output=reviewCsv([{lane:1,status:'unreviewed',notes,updated:'=1+2',staleReason:'@cmd'}]);
    assert.ok(output.includes(`"'${notes.replaceAll('"','""')}"`));
    assert.ok(output.includes("\"'=1+2\""));
    assert.ok(output.includes("\"'@cmd\""));
  }
});
