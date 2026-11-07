import assert from 'node:assert/strict';
import test from 'node:test';
import { changeCandidates, changeDecisionCounts, comparisonDiagnostics, graphEvidence, sameGraphEvidence, setChangeDecision } from '../src/change-review.ts';

const state = () => ({ nodeIds: new Float64Array([10, 20, 30, 40]), poses: new Float64Array(64),
  edges: new Uint32Array([0, 1, 1, 2, 0, 3, 2, 3, 1, 3]), edgeKinds: new Uint8Array([1, 0, 1, 1, 1]),
  edgeErrors: new Float64Array([100, 999, 9, 64, 16]), fixed: new Uint8Array([1, 0, 0, 0]), planes: 0, planeEdges: 0, gravityEdges: 0 });

test('registration evidence excludes odometry and within-pass loops and does not hide unmeasured cores', () => {
  const d = comparisonDiagnostics(state(), 2, new Float32Array([0, .5, NaN, Infinity]));
  assert.deepEqual(d.crossPartLoops, [{ from: 10, to: 40, weightedResidual: 3 }, { from: 20, to: 40, weightedResidual: 4 }]);
  assert.equal(d.weightedRms, Math.sqrt(12.5)); assert.equal(d.weightedMax, 4);
  assert.equal(d.measured, 2); assert.equal(d.unmeasured, 2); assert.equal(d.measuredFraction, .5);
  const s = state(); s.edgeKinds.fill(0);
  const absent = comparisonDiagnostics(s, 2, new Float32Array());
  assert.equal(absent.weightedRms, null); assert.equal(absent.weightedMax, null); assert.equal(absent.measuredFraction, null);
  s.edgeKinds[2] = 1; s.edgeErrors[2] = -1;
  const invalid = comparisonDiagnostics(s, 2, new Float32Array([NaN]));
  assert.equal(invalid.crossPartLoops[0].weightedResidual, null); assert.equal(invalid.residualsAvailable, 0);
});

test('all objects retain original geometry, signs and independent decisions beyond the first page', () => {
  const flat = new Float64Array(Array.from({ length: 53 }, (_, i) => [8, i, 0, 0, i, 0, 0, i + 1, 1, 1, -.5]).flat());
  const candidates = changeCandidates(flat); assert.equal(candidates.length, 53);
  setChangeDecision(candidates[52], 'misalignment', '確認\n<img src=x>');
  assert.equal(candidates[52].rank, 53); assert.deepEqual(candidates[52].bounds, [52, 0, 0, 53, 1, 1]);
  assert.equal(candidates[52].meanChange, -.5); assert.equal(candidates[0].decision, 'unreviewed');
  assert.deepEqual(changeDecisionCounts(candidates), { unreviewed: 52, misalignment: 1, change: 0, insufficient: 0, deferred: 0 });
  assert.equal(JSON.parse(JSON.stringify(candidates))[52].note, '確認\n<img src=x>');
  const saved = structuredClone(candidates[52]);
  for (const [decision, note] of [['invented', 'x'], ['change', 'x'.repeat(2001)], ['change', '\x00']]) {
    assert.throws(() => setChangeDecision(candidates[52], decision, note)); assert.deepEqual(candidates[52], saved);
  }
  assert.deepEqual(changeDecisionCounts([]), { unreviewed: 0, change: 0, misalignment: 0, insufficient: 0, deferred: 0 });
  assert.throws(() => changeCandidates(new Float64Array(10)));
});

test('graph evidence detects in-place pose, constraint, fixation and gravity changes', () => {
  const original = state(); const frozen = graphEvidence(original);
  assert.equal(sameGraphEvidence(frozen, original), true);
  for (const mutate of [s => s.poses[3]++, s => s.edges[1]++, s => s.edgeKinds[0] = 0,
    s => s.edgeErrors[0]++, s => s.fixed[1] = 1, s => s.gravityEdges++, s => s.nodeIds[0]++]) {
    const s = graphEvidence(original); mutate(s); assert.equal(sameGraphEvidence(frozen, s), false);
  }
  original.poses[3]++; assert.equal(frozen.poses[3], 0); assert.equal(sameGraphEvidence(frozen, original), false);
});

test('filters retain original ranks and notes, including no-match and decided-with-deferrals cases', async () => {
  const { filterChangeCandidates } = await import('../src/change-review.ts');
  const candidates = changeCandidates(new Float64Array(Array.from({ length: 55 }, (_, i) => [8, i, 0, 0, i, 0, 0, i + 1, 1, 1, -.5]).flat()));
  setChangeDecision(candidates[0], 'change', 'one');
  setChangeDecision(candidates[1], 'deferred', 'two');
  setChangeDecision(candidates[54], 'insufficient', 'last');
  const original = structuredClone(candidates);
  assert.deepEqual(filterChangeCandidates(candidates, 'decided').map(c => c.rank), [1, 2, 55]);
  assert.deepEqual(filterChangeCandidates(candidates, 'deferred').map(c => c.rank), [2]);
  assert.deepEqual(filterChangeCandidates(candidates, 'unreviewed').map(c => c.rank), Array.from({ length: 52 }, (_, i) => i + 3));
  assert.deepEqual(filterChangeCandidates(candidates, 'misalignment'), []);
  assert.deepEqual(filterChangeCandidates([], 'all'), []);
  assert.deepEqual(candidates, original);
});

test('next-unreviewed navigation crosses original pages, wraps once, and excludes the current record', async () => {
  const { nextUnreviewedCandidate } = await import('../src/change-review.ts');
  const candidates = changeCandidates(new Float64Array(Array.from({ length: 55 }, (_, i) => [8, i, 0, 0, i, 0, 0, i + 1, 1, 1, -.5]).flat()));
  for (const candidate of candidates) if (![1, 55].includes(candidate.rank)) setChangeDecision(candidate, 'deferred', 'later');
  assert.equal(nextUnreviewedCandidate(candidates, 1)?.rank, 55);
  assert.equal(nextUnreviewedCandidate(candidates, 55)?.rank, 1);
  setChangeDecision(candidates[0], 'change', 'checked');
  assert.equal(nextUnreviewedCandidate(candidates, 55), null);
  assert.equal(nextUnreviewedCandidate(candidates, 2)?.rank, 55);
  setChangeDecision(candidates[54], 'change', 'checked');
  assert.equal(nextUnreviewedCandidate(candidates, 2), null);
  assert.equal(nextUnreviewedCandidate([], 1), null);
});
