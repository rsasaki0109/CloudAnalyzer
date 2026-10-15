import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseRoadBuildFailure } from '../src/app/road-build-failure.ts';

const failure = () => ({
  message: 'road boundaries have ambiguous travel directions',
  diagnostic: {
    code: 'ambiguous_boundary_direction', location: [500001, 4000000, 12],
    reference: [[500000, 4000000, 12], [500010, 4000000, 13]],
    boundaries: [0, 1, 2].map(y => [[500000, 4000000 + y, 12], [500010, 4000000 + y, 13]]),
    context_length: 10, forward_lanes: 1, backward_lanes: 1, lane_width: 3.5, segment_length: 50,
  },
});

test('worker error preserves rejected survey geometry and settings', () => {
  const value = failure();
  assert.deepEqual(parseRoadBuildFailure(JSON.stringify(value)), value);
  assert.equal(parseRoadBuildFailure('Choose a point cloud'), null);
  assert.equal(parseRoadBuildFailure('{broken JSON'), null);
});

test('invalid diagnostics cannot create a misleading or unbounded preview', () => {
  for (const change of [
    d => { d.code = 'unknown'; },
    d => { d.location[0] = null; },
    d => { d.reference = [[0, 0, 0]]; },
    d => { d.backward_lanes = 0; }, // Boundary count no longer matches lanes.
    d => { d.lane_width = -1; },
    d => { d.boundaries[0] = Array.from({length: 257}, () => [0, 0, 0]); },
    d => {
      d.forward_lanes = 16; d.backward_lanes = 0;
      d.boundaries = Array.from({length: 17}, () => Array.from({length: 128}, () => [0, 0, 0]));
    },
  ]) {
    const value = failure(); change(value.diagnostic);
    assert.equal(parseRoadBuildFailure(JSON.stringify(value)), null);
  }
  assert.equal(parseRoadBuildFailure(' '.repeat(300001)), null);
});
