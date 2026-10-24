import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readCompletionScanReview } from '../src/lidar-completion-review.ts';
import { readAiScanReview } from '../src/lidar-ai-review.ts';
import { completionFixture } from './lidar-completion-fixture.mjs';
import { aiFixture, sha } from './lidar-ai-fixture.mjs';
const read = (f, signal = new AbortController().signal) => readCompletionScanReview(f.files(),signal);
test('completion keeps exact completed, added and original baseline source records separate',async()=>{
  const f=completionFixture(),r=await read(f);
  assert.deepEqual(Buffer.from(await r.retained.arrayBuffer()),f.retained);
  assert.deepEqual(Buffer.from(await r.candidates.arrayBuffer()),f.candidates);
  assert.deepEqual(Buffer.from(await r.completion.added.arrayBuffer()),f.addedRecords);
  assert.deepEqual(Buffer.from(await r.completion.baselineCandidates.arrayBuffer()),f.baseline.candidates);
  assert.deepEqual(Buffer.from(await r.completion.baselineRetained.arrayBuffer()),f.baseline.retained);
  assert.equal(r.completion.addedCount,1);assert.equal(r.completion.removed,3);
  assert.equal(sha(Buffer.from(await r.source.arrayBuffer())),sha(f.baseline.raw));
  await assert.rejects(readAiScanReview(f.files(),new AbortController().signal),/exactly/);
});
test('completion rejects substituted baseline reports, masks, selection and model bindings',async()=>{
  for(const mutate of [f=>f.report.base.report_sha256='0'.repeat(64),f=>f.report.base.candidate_mask_sha256='0'.repeat(64),
    f=>f.report.base.threshold=.8,f=>f.report.base.target_classes=['truck'],f=>f.report.base.weights_sha256='0'.repeat(64),
    f=>f.report.input_sha256='0'.repeat(64),f=>f.report.base.probabilities_sha256='bad',
    f=>f.baseline.report.note='same outputs, different report bytes',f=>f.report.retained_sha256='0'.repeat(64)]) {
    const f=completionFixture();mutate(f);await assert.rejects(read(f));
  }
});
test('even rehashed masks cannot remove baseline decisions or add owners, invalid, or distant returns',async()=>{
  let f=completionFixture();f.completed[0]=0;f.sync();await assert.rejects(read(f),/counts do not balance/);
  f=completionFixture();f.added[0]=1;f.sync();await assert.rejects(read(f),/counts do not balance|disjoint/);
  f=completionFixture();f.added[1]=0;f.completed[1]=0;f.added[3]=1;f.completed[3]=1;f.sync();
  await assert.rejects(read(f),/eligible collision/);
  const base=aiFixture({points:[[10,0,0,.5],[10.2,0,0,.5],[10,.05,0,.5],[0,0,10,.5]],
    mask:[1,0,1,0],carCount:2,projection:{projected:2,occluded:1,invalid:0,outside_view:1,outside_range:0}});
  await assert.rejects(read(completionFixture({baseline:base})),/direct-owner radius/);
  f=completionFixture();f.added[1]=2;f.sync();await assert.rejects(read(f));
});
test('unsupported policy, unbalanced counts, missing/duplicate files and size budgets reject',async()=>{
  for(const mutate of [f=>f.report.policy.radius_m=.2,f=>f.report.policy.seed_score=.99,f=>f.report.counts.added++,
    f=>f.report.counts.retained++,f=>f.report.counts.support_rejected=150000,f=>f.report.source_modified=true]) {
    const f=completionFixture();mutate(f);await assert.rejects(read(f));
  }
  const f=completionFixture(),signal=new AbortController().signal;
  await assert.rejects(readCompletionScanReview(f.files().slice(1),signal),/exactly/);
  const duplicate=f.files();duplicate[5]=duplicate[4];await assert.rejects(readCompletionScanReview(duplicate,signal),/exactly/);
  const oversized=f.files();oversized[3]=new File([' '.repeat(65537)],'completion-report.json');
  await assert.rejects(readCompletionScanReview(oversized,signal),/limits/);
});
test('zero added points keep valid empty added download and original completed records',async()=>{
  const f=completionFixture({added:[0,0,0,0]}),r=await read(f);
  assert.equal(r.completion.added.size,0);assert.equal(r.completion.addedCount,0);
  assert.deepEqual(Buffer.from(await r.candidates.arrayBuffer()),f.baseline.candidates);
});
test('cancellation before and during completion verification returns no partial review',async()=>{
  const f=completionFixture(),early=new AbortController();early.abort();
  await assert.rejects(read(f,early.signal),{name:'AbortError'});
  const controller=new AbortController(),files=f.files(),original=files[4].arrayBuffer.bind(files[4]);
  files[4].arrayBuffer=async()=>{const bytes=await original();controller.abort();return bytes;};
  await assert.rejects(readCompletionScanReview(files,controller.signal),{name:'AbortError'});
});
