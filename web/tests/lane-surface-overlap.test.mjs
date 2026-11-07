import assert from 'node:assert/strict';
import test from 'node:test';
import { overlappingSurveyLanes, resampleLaneCurve } from '../src/lane-surface-overlap.ts';
const lane = (id, z = 0, y = 0) => ({ id, left: [[0, y + 2, z], [10, y + 2, z]], right: [[0, y - 2, z], [10, y - 2, z]] });

test('box inside a wide lane is detected, without associating parallel or stacked roads', () => {
  const lanes = [lane(1), lane(2, 0, 10), lane(3, 8)];
  const before = structuredClone(lanes);
  assert.deepEqual(overlappingSurveyLanes(lanes, [4, -.2, 0, 5, .2, .5], 0, 1).ids, [1]);
  assert.deepEqual(overlappingSurveyLanes(lanes, [4, -.2, 0, 5, .2, .5], 0, 8).ids, [1, 3]);
  assert.deepEqual(overlappingSurveyLanes(lanes, [4, 2.5, 0, 5, 3, 0], 0, 0).ids, []);
  assert.deepEqual(overlappingSurveyLanes(lanes, [4, 2.5, 0, 5, 3, 0], .5, 0).ids, [1]);
  assert.deepEqual(lanes, before);
});

test('surface tests reject broad-box false positives and check height locally on slopes', () => {
  const bend = { id: 1, left: [[0, 1, 0], [9, 1, 0], [9, 10, 0]], right: [[0, -1, 0], [11, -1, 0], [11, 10, 0]] };
  assert.deepEqual(overlappingSurveyLanes([bend], [1, 8, 0, 2, 9, 0], 0, 0).ids, []);
  const slope = { id: 2, left: [[0, 2, 0], [10, 2, 10]], right: [[0, -2, 0], [10, -2, 10]] };
  assert.deepEqual(overlappingSurveyLanes([slope], [0, -.2, 9, 1, .2, 10], 0, 0).ids, []);
  assert.deepEqual(overlappingSurveyLanes([slope], [8, -.2, 8, 9, .2, 9], 0, 0).ids, [2]);
});

test('resampling uses XY station and preserves local Z, matching the viewer with unequal boundary counts', () => {
  assert.deepEqual(resampleLaneCurve([[0, 0, 0], [10, 0, 20]], 3), [[0, 0, 0], [5, 0, 10], [10, 0, 20]]);
  const unequal = lane(1); unequal.left.splice(1, 0, [5, 2, 0]);
  assert.deepEqual(overlappingSurveyLanes([unequal], [5, -.5, 0, 6, .5, 0], 0, 0).ids, [1]);
  assert.deepEqual(resampleLaneCurve([], 2), []);
});

test('invalid geometry and bounded work are explicit unchecked counts, never silent no-overlap results', () => {
  const bad = lane(2); bad.left[0][2] = NaN;
  const collapsed = { id: 3, left: [[0, 0, 0], [10, 0, 0]], right: [[0, 0, 0], [10, 0, 0]] };
  const bounds = [4, -.1, 0, 5, .1, 0];
  const result = overlappingSurveyLanes([lane(1), bad, collapsed], bounds, 0, 0);
  assert.deepEqual(result.ids, [1]); assert.equal(result.invalid, 2); assert.equal(result.checked, 1);
  const capped = overlappingSurveyLanes([lane(1), lane(2)], bounds, 0, 0, { lanes: 2, vertices: 8, triangles: 2 });
  assert.equal(capped.checked, 1); assert.equal(capped.limited, 1);
  const noVertices = overlappingSurveyLanes([lane(1)], bounds, 0, 0, { lanes: 1, vertices: 0, triangles: 2 });
  assert.equal(noVertices.limited, 1); assert.equal(noVertices.checked, 0);
  assert.equal(overlappingSurveyLanes([lane(1)], bounds, 0, 0, { lanes: 0, vertices: 8, triangles: 2 }).limited, 1);
  for (const [b, xy, z] of [[bounds, -1, 0], [bounds, 0, 11], [[1, 1, 1, 0, 0, 0], 0, 0], [bounds, NaN, 0]]) {
    assert.throws(() => overlappingSurveyLanes([lane(1)], b, xy, z));
  }
});
