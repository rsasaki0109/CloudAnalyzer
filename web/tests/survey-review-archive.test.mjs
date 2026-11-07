import assert from 'node:assert/strict';
import test from 'node:test';
import { changeCandidates, changeDecisionCounts, comparisonDiagnostics, setChangeDecision } from '../src/change-review.ts';
import { parseSurveyReview, resumeSurveyDecisions, surveyReviewIdentity } from '../src/survey-review-archive.ts';
const time = '2026-10-24T00:00:00.000Z';
function graph() {
  return { nodeIds: new Float64Array([10, 20, 30, 40]), poses: new Float64Array(64), edges: new Uint32Array([0, 3]),
    edgeKinds: new Uint8Array([1]), edgeErrors: new Float64Array([9]), fixed: new Uint8Array([1, 0, 0, 0]), planes: 0, planeEdges: 0, gravityEdges: 0 };
}
function fixture() {
  const candidates = changeCandidates(new Float64Array([8, .5, .5, .5, 0, 0, 0, 1, 1, 1, -.5]));
  return { schema: 'cloudanalyzer.survey_change_review.v1', scope: 'Human judgments', analyzedAt: time, exportedAt: time,
    staleReason: null, identity: null,
    context: { graph: 'survey', sessions: [{ name: 'a', first: 0, count: 2 }, { name: 'b', first: 2, count: 2 }], splitNode: 30,
      firstNodes: [10, 20], secondNodes: [30, 40], mapVoxel: .3, reach: 50,
      m3c2: { normalRadius: 1, projectionRadius: .5, maxDepth: 2, coreSpacing: .5 }, clustering: { minChange: .3, link: 1, minPoints: 8 } },
    sources: { before: { id: 1, name: 'before', points: 40 }, after: { id: 2, name: 'after', points: 40 }, difference: { id: 3, name: 'difference', points: 20 } },
    diagnostics: comparisonDiagnostics(graph(), 2, new Float32Array(Array.from({ length: 20 }, (_, i) => i < 16 ? .5 : NaN))),
    candidates, counts: changeDecisionCounts(candidates) };
}
const bytes = id => Promise.resolve(new Uint8Array([id, 42, 0, 255]));
function pack(value) { return { ...value, counts: changeDecisionCounts(value.candidates), exportedAt: time }; }

test('old files remain readable history and strict parsing rejects inconsistent evidence or judgments', () => {
  const old = fixture(); delete old.identity;
  const parsed = parseSurveyReview(old); assert.equal(parsed.identity, null); assert.equal(parsed.candidates[0].meanChange, -.5);
  for (const mutate of [v => v.schema = 'other', v => v.diagnostics.measured++, v => v.diagnostics.weightedRms++,
    v => v.candidates[0].bounds.reverse(), v => v.candidates[0].rank++, v => v.candidates[0].centroid[0] = 5,
    v => v.candidates[0].points = 17, v => v.candidates[0].decision = 'invented', v => v.candidates[0].note = 'x'.repeat(2001),
    v => v.candidates[0].note = '\x00', v => v.candidates[0].updatedAt = 'today', v => v.counts.change++,
    v => v.sources.after.id = 1, v => v.context.secondNodes[0] = 10, v => v.identity = { algorithm: 'sha256-ply-graph-v1', digest: 'no' }]) {
    const value = fixture(); mutate(value); assert.throws(() => parseSurveyReview(value));
  }
  assert.throws(() => resumeSurveyDecisions(parsed, parsed), /no content identity/);
});

test('resumption matches complete contents across session IDs and JSON ordering, preserving notes and times', async () => {
  const raw = fixture(); raw.identity = await surveyReviewIdentity(raw, graph(), bytes);
  setChangeDecision(raw.candidates[0], 'misalignment', '確認\n<img src=x onerror=alert(1)>');
  const saved = parseSurveyReview(pack(raw));
  const currentRaw = fixture();
  for (const source of Object.values(currentRaw.sources)) source.id += 100;
  currentRaw.identity = await surveyReviewIdentity(currentRaw, graph(), id => bytes(id - 100));
  currentRaw.analyzedAt = '2026-10-24T01:00:00.000Z';
  const current = parseSurveyReview(currentRaw);
  const before = structuredClone(current), archiveBefore = structuredClone(saved);
  const decisions = resumeSurveyDecisions(current, saved);
  assert.equal(decisions[0].decision, 'misalignment'); assert.equal(decisions[0].note, saved.candidates[0].note);
  assert.equal(decisions[0].updatedAt, saved.candidates[0].updatedAt);
  assert.deepEqual(current, before); assert.deepEqual(saved, archiveBefore);
  decisions[0].note = 'edited'; assert.deepEqual(saved, archiveBefore);
  const reordered = parseSurveyReview({ ...pack(saved), context: Object.fromEntries(Object.entries(saved.context).reverse()) });
  assert.deepEqual(resumeSurveyDecisions(current, reordered), resumeSurveyDecisions(current, saved));
});

test('changed bytes, graph, parameters and candidate evidence never inherit decisions by rank or filename', async () => {
  const raw = fixture(); raw.identity = await surveyReviewIdentity(raw, graph(), bytes);
  const saved = parseSurveyReview(raw);
  const changedCloud = await surveyReviewIdentity(raw, graph(), id => Promise.resolve(new Uint8Array([id, 43, 0, 255])));
  assert.notEqual(changedCloud.digest, saved.identity.digest);
  const s = graph(); s.poses[3] = .001;
  assert.notEqual((await surveyReviewIdentity(raw, s, bytes)).digest, saved.identity.digest);
  for (const mutate of [v => v.identity.digest = changedCloud.digest, v => v.context.mapVoxel = .4,
    v => v.candidates[0].meanChange = -.6, v => v.sources.before.name = 'renamed',
    v => v.staleReason = 'Graph changed', v => v.identity = null]) {
    const current = structuredClone(saved); mutate(current);
    const before = structuredClone(current); assert.throws(() => resumeSurveyDecisions(current, saved)); assert.deepEqual(current, before);
  }
  const staleSaved = structuredClone(saved); staleSaved.staleReason = 'Graph closed';
  assert.throws(() => resumeSurveyDecisions(saved, staleSaved), /Historical/);
});

test('empty reviews retain null unavailable residuals and cannot manufacture decisions', async () => {
  const value = fixture(); value.candidates = []; value.counts = changeDecisionCounts([]);
  value.diagnostics.crossPartLoops = []; value.diagnostics.residualsAvailable = 0;
  value.diagnostics.weightedRms = value.diagnostics.weightedMax = null;
  value.identity = await surveyReviewIdentity(value, graph(), bytes);
  const parsed = parseSurveyReview(value); assert.deepEqual(resumeSurveyDecisions(parsed, parsed), []);
});

test('identity reads bytes past the first chunk and binds all three clouds and graph constraints', async () => {
  const review = fixture(), base = new Uint8Array(1024 * 1024 + 1);
  const original = await surveyReviewIdentity(review, graph(), () => Promise.resolve(base));
  for (const target of [1, 2, 3]) {
    const changed = base.slice(); changed[changed.length - 1] = 1;
    const identity = await surveyReviewIdentity(review, graph(), id => Promise.resolve(id === target ? changed : base));
    assert.notEqual(identity.digest, original.digest);
  }
  for (const mutate of [s => s.edgeErrors[0]++, s => s.fixed[1] = 1, s => s.gravityEdges++, s => s.planeEdges++, s => s.planes++]) {
    const s = graph(); mutate(s);
    assert.notEqual((await surveyReviewIdentity(review, s, () => Promise.resolve(base))).digest, original.digest);
  }
  setChangeDecision(review.candidates[0], 'deferred', 'Review later');
  assert.equal((await surveyReviewIdentity(review, graph(), () => Promise.resolve(base))).digest, original.digest);
});
