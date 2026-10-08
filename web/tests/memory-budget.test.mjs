import assert from 'node:assert/strict';
import { test } from 'node:test';
import { bufferBytes, geometryBytes, nativeCloudEstimate, trimHistory } from '../src/memory-budget.ts';

test('retained memory deduplicates backing buffers across fields, maps, views and cycles', () => {
  const a = new ArrayBuffer(1000), b = new ArrayBuffer(20);
  const entry = {points: new Float32Array(a), fields: new Map([['field',new Uint8Array(a,0,20)]]), file:new File([new Uint8Array(4096)],'external.ply')};
  entry.self = entry;
  assert.equal(bufferBytes([entry, new Uint8Array(b)]),1020);
});
test('history is trimmed by both step and byte limits, including oversized single entries', () => {
  const list = [new Uint8Array(10),new Uint8Array(20),new Uint8Array(30)];
  assert.equal(trimHistory(list,2,40,bufferBytes).length,2);
  assert.equal(list.length,1); assert.equal(list[0].length,30);
  assert.equal(trimHistory(list,20,20,bufferBytes).length,1);
  assert.equal(list.length,0);
  const shared = new ArrayBuffer(32), multiple = [new Uint8Array(shared),new Uint16Array(shared)];
  assert.equal(trimHistory(multiple,10,32,bufferBytes).length,0);
  assert.equal(trimHistory(multiple,0,32,bufferBytes).length,2);
});

test('cloud history accounts for native full-density data as well as transferred display buffers', () => {
  const cloud = {count:1000000, triangles:0, colors:null, normals:null, classification:null, scalarNames:['intensity'], lodNodes:new Float64Array(100)};
  assert.equal(nativeCloudEstimate(cloud),72001600);
  assert.equal(nativeCloudEstimate({...cloud, normals:new Float32Array(0)},2),112001600);
});

test('GPU estimate uses uploaded view sizes, deduplicating shared interleaved attributes', () => {
  const buffer = new ArrayBuffer(10000), shared = {}, separate = {};
  const attributes = [{owner:shared,array:new Float32Array(buffer,0,10)}, {owner:shared,array:new Float32Array(buffer,0,10)}, {owner:separate,array:new Uint8Array(buffer,40,5)}];
  assert.equal(geometryBytes(attributes),45);
  assert.equal(bufferBytes(attributes),10000);
});
