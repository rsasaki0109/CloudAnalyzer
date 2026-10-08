import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LaneReviews, parseReviews } from '../src/lane-review.ts';

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
