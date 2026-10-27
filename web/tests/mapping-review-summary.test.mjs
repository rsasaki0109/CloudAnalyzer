import test from 'node:test';
import assert from 'node:assert/strict';
import { mappingReviewSummary } from '../src/mapping-review-summary.ts';
test('missing evidence never becomes a passing gate or zero failures', () => {
  const result = mappingReviewSummary({}).join('\n');
  assert.match(result, /extent met: unavailable/);
  assert.match(result, /Held traces: unavailable/);
});
test('saved failures and unresolved assumptions remain visible', () => {
  const result = mappingReviewSummary({diagnosis:{review_holds:{passes_requested_extent:false, traces_total:7, unexplained_traces_total:2, unverified:['legal_routing']}}}).join('\n');
  assert.match(result, /extent met: no/);
  assert.match(result, /Held traces: 7; unexplained: 2/);
  assert.match(result, /legal_routing/);
});
