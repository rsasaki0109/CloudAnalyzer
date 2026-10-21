import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';

const source = stripTypeScriptTypes(readFileSync(new URL('../src/gpu-candidates.ts', import.meta.url), 'utf8'), { mode: 'transform' });
const { packGpuNodePairs } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

test('packed GPU nodes preserve axis, range and every split bit independently', () => {
  for (const axis of [0, 1, 2]) for (const bits of [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0xffffffff, 1, 0x7f7fffff]) {
    const nodes = new Uint32Array([0,2,2,axis,bits, 0,1,0,255,0, 1,2,0,255,0]);
    const before = nodes.slice(), packed = packGpuNodePairs(nodes, 2);
    assert.equal(packed[0] >>> 30, axis); assert.equal(packed[0] & 0x3fffffff, 2); assert.equal(packed[1], bits);
    assert.deepEqual(Array.from(packed.slice(2)), [0xc0000000,1,0xc0000001,2]);
    assert.equal(packed.byteLength, nodes.length / 5 * 8); assert.deepEqual(nodes, before);
    nodes.fill(0); assert.equal(packed[1], bits);
  }
  for (const count of [1, 16, 2_000_000])
    assert.deepEqual(Array.from(packGpuNodePairs(new Uint32Array([0,count,0,255,0]), count)), [0xc0000000,count]);
});

test('packed GPU nodes reject disconnected, duplicated and malformed trees', () => {
  for (const [nodes, count] of [
    [[],1], [[0,2,0,255],2], [[1,2,0,255,0],2], [[0,2,0,255,0],3],
    [[0,2,0,255,0],0], [[0,2,0,255,0],NaN], [[0,2,0,255,0],2.5], [[0,2,0,255,0],2_000_001],
    [[0,2,0,3,0],2], [[0,2,1,0,0,0,1,0,255,0,1,2,0,255,0],2],
    [[0,2,3,0,0,0,1,0,255,0,1,2,0,255,0],2], [[0,2,0,255,0,0,1,0,255,0],2],
    [[0,4,3,0,0, 0,2,3,1,0, 0,1,0,255,0, 2,4,0,255,0],4],
  ]) assert.throws(() => packGpuNodePairs(new Uint32Array(nodes), count), /[Pp]acked GPU/);
});
