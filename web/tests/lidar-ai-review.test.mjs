import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readAiScanReview, parseAiScanReport } from '../src/lidar-ai-review.ts';
import { aiFixture, sha } from './lidar-ai-fixture.mjs';
const read = (f, signal = new AbortController().signal) => readAiScanReview(f.files(), signal);
test('source-order split preserves exact XYZI bytes and every unclassified return', async () => {
  const f = aiFixture(), result = await read(f);
  assert.deepEqual(Buffer.from(await result.retained.arrayBuffer()),f.retained);
  assert.deepEqual(Buffer.from(await result.candidates.arrayBuffer()),f.candidates);
  assert.equal(result.report.unclassified,3);
  assert.equal(sha(await result.source.bytes()),sha(f.raw));
});
test('wrong same-sized source, malformed mask and changed record order reject before loading', async () => {
  let f = aiFixture(); f.raw[0] ^= 1; await assert.rejects(read(f),/Source SHA256 differs/);
  f = aiFixture(); f.mask[0] = 2; await assert.rejects(read(f),/0\/1/);
  f = aiFixture(); f.mask[0] = 0; await assert.rejects(read(f),/mask count/);
  f = aiFixture(); f.mask[0] = 0; f.mask[2] = 1; await assert.rejects(read(f),/record order or output SHA256/);
});
test('even self-consistent hashes cannot select occluded or out-of-view returns', async () => {
  for (const id of [1,4,5]) {
    const mask = Array(9).fill(0); mask[id] = 1;
    await assert.rejects(read(aiFixture({mask})),/unclassified or occluded/);
  }
});
test('NaN payloads stay byte-exact in retained records, never candidates', async () => {
  const f = aiFixture({points:[[10,0,0,.5],[NaN,0,0,.25]],mask:[1,0],
    projection:{projected:1,occluded:0,invalid:1,outside_view:0,outside_range:0}});
  // Preserve a noncanonical NaN payload rather than reserializing it.
  f.raw.writeUInt32LE(0x7fc01234,16); f.retained.writeUInt32LE(0x7fc01234,0);
  f.report.input_sha256 = sha(f.raw); f.report.retained_sha256 = sha(f.retained);
  assert.deepEqual(Buffer.from(await (await read(f)).retained.arrayBuffer()),f.retained);
});
test('zero candidates and zero retained records produce valid empty downloads', async () => {
  const none=aiFixture({mask:Array(9).fill(0)}); assert.equal((await read(none)).candidates.size,0);
  const all=aiFixture({points:[[10,0,0,.5],[0,10,0,.3]],mask:[1,1],carCount:2,
    projection:{projected:2,occluded:0,invalid:0,outside_view:0,outside_range:0}});
  assert.equal((await read(all)).retained.size,0);
});
test('unsupported profiles, model identities, malformed counts and score metadata reject', () => {
  const mutations = [r=>r.sensor_profile.height=32,r=>r.model='other',r=>r.onnx_sha256='0'.repeat(64),
    r=>r.selection.threshold=NaN,r=>r.selection.target_classes=['unlabeled'],r=>r.selection.retained++,
    r=>r.projection.occluded++,r=>r.selection.class_counts[9]++,r=>r.class_names.reverse(),r=>r.weights_sha256='bad'];
  for (const mutate of mutations) { const f=aiFixture(); mutate(f.report); assert.throws(()=>parseAiScanReport(f.report)); }
});
test('file budgets, missing files and cancellation return no partial review', async () => {
  const f=aiFixture(), controller=new AbortController(); controller.abort();
  await assert.rejects(read(f,controller.signal),{name:'AbortError'});
  await assert.rejects(readAiScanReview(f.files().slice(1),new AbortController().signal),/exactly/);
  const files=f.files(); files[0]=new File([Buffer.alloc(150001*16)],'scan.bin');
  await assert.rejects(readAiScanReview(files,new AbortController().signal),/limits/);
  const mid=new AbortController(), source=f.files()[0], original=source.arrayBuffer.bind(source);
  source.arrayBuffer=async()=>{const bytes=await original();mid.abort();return bytes;};
  await assert.rejects(readAiScanReview([source,...f.files().slice(1)],mid.signal),{name:'AbortError'});
});
