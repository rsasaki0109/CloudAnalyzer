import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
const source = stripTypeScriptTypes(readFileSync(new URL('../src/gpu-candidates.ts', import.meta.url), 'utf8'), { mode: 'transform' });
const { packGpuParentLinks } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

test('GPU parents preserve preorder ancestry and own their exported links', () => {
  const nodes = new Uint32Array([0,4,4,0,0, 0,2,3,1,0, 0,1,0,255,0, 1,2,0,255,0, 2,4,0,255,0]);
  const before = nodes.slice(), parents = packGpuParentLinks(nodes, 4);
  assert.deepEqual(parents, new Uint32Array([0xffffffff,0,1,1,0]));
  assert.equal(parents.byteLength, nodes.length / 5 * 4);
  assert.deepEqual(nodes, before);
  nodes.fill(0); assert.equal(parents[3], 1);
  assert.deepEqual(packGpuParentLinks(new Uint32Array([0,4,0,255,0]), 4), new Uint32Array([0xffffffff]));
});

test('GPU parents reject disconnected, duplicated and malformed trees before upload', () => {
  for (const [nodes, count] of [
    [[], 1], [[0,1,0,255], 1], [[1,2,0,255,0], 2], [[0,2,0,255,0], 3],
    [[0,2,0,255,0], 0], [[0,2,0,255,0], NaN], [[0,2,0,255,0], 2.5],
    [[0,2,0,3,0], 2], [[0,2,0,0,0], 2],
    [[0,2,2,0,0, 0,2,0,255,0, 1,2,0,255,0], 2],
    [[0,2,2,0,0, 0,0,0,255,0, 0,2,0,255,0], 2],
    [[0,2,0,255,0, 0,1,0,255,0], 2],
    [[0,4,3,0,0, 0,2,3,1,0, 0,1,0,255,0, 2,4,0,255,0], 4],
  ]) assert.throws(() => packGpuParentLinks(new Uint32Array(nodes), count), /GPU parent/);
});
