import assert from 'node:assert/strict';
import test from 'node:test';
import { parseMappingTrajectory, TRAJECTORY_DISPLAY, TRAJECTORY_POSES } from '../src/mapping-trajectory.ts';
const pose = (x,y=0,z=2) => `1 0 0 ${x} 0 1 0 ${y} 0 0 1 ${z}`;
test('translations preserve original coordinates and full-source bounds', () => {
  const result = parseMappingTrajectory(`# KITTI\n${pose(636800.125,851200.25)}\r\n${pose(636810.5,851220.75,4)}\n`);
  assert.equal(result.unavailable,null);
  assert.deepEqual(result.positions,[[636800.125,851200.25,2],[636810.5,851220.75,4]]);
  assert.deepEqual(result.bounds,[636800.125,851200.25,2,636810.5,851220.75,4]);
});
test('bounded display preserves endpoints, counts and bounds across every original pose', () => {
  const rows = Array.from({length:11000},(_,i)=>pose(i,i===100?9000:0));
  const result = parseMappingTrajectory(rows.join('\n'));
  assert.equal(result.positions.length,TRAJECTORY_DISPLAY);
  assert.equal(result.count,11000);
  assert.deepEqual(result.positions[0],[0,0,2]);
  assert.deepEqual(result.positions.at(-1),[10999,0,2]);
  assert.equal(result.bounds[4],9000);
  rows[10998] = 'NaN ' + rows[10998].slice(2);
  assert.ok(parseMappingTrajectory(rows.join('\n')).unavailable);
});
test('invalid, oversized and unsupported trajectories produce no partial display', () => {
  for (const value of ['',pose(1), '0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1',
    `${pose(0)}\n${pose(Infinity)}`, `${pose(0)}\n${pose(1).replace(/^1 /,'2 ')}`,
    `${pose(0)}\n${pose(1).replace(/^1 /,'-1 ')}`, `${pose(0)}\n${pose(1e13)}`,
    Array(TRAJECTORY_POSES+1).fill(pose(0)).join('\n')]) {
    const result=parseMappingTrajectory(value);
    assert.ok(result.unavailable);assert.deepEqual(result.positions,[]);assert.equal(result.bounds,null);
  }
});
