import assert from 'node:assert/strict';
import { test } from 'node:test';
import { crosswalkTriangles, signalTriangles } from '../src/app/vectormap-geometry.ts';

const area = (positions) => {
  let sum = 0;
  for (let i = 0; i < positions.length; i += 9) {
    sum += Math.abs((positions[i+3] - positions[i]) * (positions[i+7] - positions[i+1]) -
      (positions[i+4] - positions[i+1]) * (positions[i+6] - positions[i])) / 2;
  }
  return sum;
};
const inside = (x, y, polygon) => {
  let hit = false;
  for (let i = 0, j = polygon.length-1; i < polygon.length; j = i++) {
    const a = polygon[i], b = polygon[j];
    if ((a[1] > y) !== (b[1] > y) && x < (b[0]-a[0]) * (y-a[1]) / (b[1]-a[1]) + a[0]) hit = !hit;
  }
  return hit;
};

test('a concave crossing has no stripes outside its outline and keeps surveyed slopes', () => {
  const polygon = [[0,0],[12,0],[12,4],[8,4],[8,2],[4,2],[4,4],[0,4]];
  const origin = [3800, 73700, 20];
  const input = polygon.map(([x,y]) => [origin[0]+x, origin[1]+y, origin[2]+x*0.02+y*0.03]);
  const original = structuredClone(input);
  const positions = crosswalkTriangles(input, origin);
  assert.ok(positions.length > 0);
  assert.ok(Math.abs(area(positions) - 20) < 1e-8); // Half of the 40 m² concave outline.
  for (let i = 0; i < positions.length; i += 3) {
    const [x,y,z] = positions.slice(i,i+3);
    assert.ok(Math.abs(z - (x*0.02+y*0.03)) < 1e-10);
  }
  // Interior samples in each triangle also rule out triangles bridging the notch.
  for (let i = 0; i < positions.length; i += 9) {
    const p = positions.slice(i,i+9);
    for (const [a,b,c] of [[1/3,1/3,1/3],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]]) {
      const x = a*p[0]+b*p[3]+c*p[6], y = a*p[1]+b*p[4]+c*p[7];
      if (Math.abs((p[3]-p[0])*(p[7]-p[1])-(p[4]-p[1])*(p[6]-p[0])) > 1e-10) assert.ok(inside(x,y,polygon));
    }
  }
  assert.deepEqual(input, original);
  assert.deepEqual(crosswalkTriangles([...input, input[0]], origin), positions);
  assert.deepEqual(crosswalkTriangles([...input].reverse(), origin).length, positions.length);
});

test('crosswalk bands retain centimetre details at large survey coordinates', () => {
  const near = [[0,0,0],[8.125,0,0.125],[8.125,4,0.25],[0,4,0.125]];
  const origin = [500000, 4000000, 100];
  const far = near.map(p => p.map((v,k) => v+origin[k]));
  assert.deepEqual(crosswalkTriangles(far,origin),crosswalkTriangles(near,[0,0,0]));
});

test('empty, collapsed, non-finite and excessive crossing extents do not make faces', () => {
  for (const p of [[],[[0,0,0]],[[0,0,0],[1,0,0],[2,0,0]],[[0,0,0],[1,NaN,0],[0,1,0]],
    [[0,0,0],[1e9,0,0],[1e9,4,0],[0,4,0]]]) assert.deepEqual(crosswalkTriangles(p,[0,0,0]),[]);
});

test('signal face uses only the stored polyline and known positive height', () => {
  const origin = [3820,73780,24];
  const bottom = [[3822,73783,24.75],[3823.5,73783.2,24.8]];
  const positions = signalTriangles(bottom,0.45,origin);
  assert.equal(positions.length,18);
  assert.ok(Math.abs(Math.max(...positions.filter((_,i)=>i%3===2)) - 1.25) < 1e-12);
  assert.ok(Math.abs(Math.min(...positions.filter((_,i)=>i%3===2)) - 0.75) < 1e-12);
  for (const height of [null,NaN,Infinity,0,-1]) assert.deepEqual(signalTriangles(bottom,height,origin),[]);
  assert.deepEqual(signalTriangles([],0.45,origin),[]);
});
