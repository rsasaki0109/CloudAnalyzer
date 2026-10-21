import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
const source = stripTypeScriptTypes(readFileSync(new URL('../src/gpu-candidates.ts', import.meta.url), 'utf8'), { mode: 'transform' });
const { packGpuNodeBoxes } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

test('GPU boxes enclose every descendant point, including repeated-axis splits and signed zero', () => {
  const points = new Float32Array([-4, 2, -0, 0, -1, -3, 0, 0, 3, 7, 5, 0, 3, 7, 5, 0]);
  // Preorder: root, its left branch, two leaves, then its right leaf.
  const nodes = new Uint32Array([0,4,4,0,0, 0,2,3,0,0, 0,1,0,255,0, 1,2,0,255,0, 2,4,0,255,0]);
  const before = [points.slice(), nodes.slice()], boxes = packGpuNodeBoxes(points, nodes);
  assert.equal(boxes.byteLength, nodes.length / 5 * 24);
  for (let node = 0; node < nodes.length / 5; node++) for (let axis = 0; axis < 3; axis++) {
    const values = Array.from({ length: nodes[node * 5 + 1] - nodes[node * 5] }, (_, i) => points[(nodes[node * 5] + i) * 4 + axis]);
    assert.ok(Object.is(boxes[node * 6 + axis], Math.min(...values)));
    assert.ok(Object.is(boxes[node * 6 + 3 + axis], Math.max(...values)));
  }
  assert.deepEqual(points, before[0]); assert.deepEqual(nodes, before[1]);
});

test('GPU boxes leave nonfinite axes unbounded and reject malformed trees before upload', () => {
  const points = new Float32Array([NaN, 2, Infinity, 0, 1, -3, 5, 0]);
  assert.deepEqual(Array.from(packGpuNodeBoxes(points, new Uint32Array([0,2,0,255,0]))), [-Infinity,-3,-Infinity,Infinity,2,Infinity]);
  const good = new Float32Array(8);
  for (const [p, n] of [
    [new Float32Array(), new Uint32Array([0,2,0,255,0])], [new Float32Array(7), new Uint32Array([0,2,0,255,0])],
    [good, new Uint32Array(4)], [good, new Uint32Array([1,2,0,255,0])],
    [good, new Uint32Array([0,2,0,3,0])], [good, new Uint32Array([0,2,0,0,0])],
    [good, new Uint32Array([0,2,2,0,0, 0,2,0,255,0, 1,2,0,255,0])],
    [good, new Uint32Array([0,2,2,0,0, 0,0,0,255,0, 0,2,0,255,0])],
  ]) assert.throws(() => packGpuNodeBoxes(p, n), /GPU box/);
});
