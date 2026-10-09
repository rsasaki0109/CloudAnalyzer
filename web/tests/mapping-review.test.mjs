import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readMappingReview, parseSavedAudits, REVIEW_LIMIT } from '../src/mapping-review.ts';
import { fixture, packageFixture, zip, quality } from './mapping-review-fixture.mjs';
const read = (buffer, signal = new AbortController().signal) => readMappingReview(new File([buffer], 'review.zip'), signal);

for (const method of [0,8]) test(`verifies complete maps/evidence with method ${method} and ZIP64 local headers`, async () => {
  const result=await read(packageFixture(undefined,{method}));
  assert.equal(result.verifiedFiles,10);assert.equal(result.members.size,3);
  assert.deepEqual(parseSavedAudits(fixture().evidence).map(a=>a.report.low_support_lanes),[[7],[7],[],[]]);
  assert.equal(result.review.diagnosis.extent.passes_requested_extent,false);
  assert.match(await result.members.get(result.roles.map).text(),/element vertex 4/);
});
test('changed bytes in a member not displayed still reject the entire package', async () => {
  const bytes=packageFixture(data=>{data.entries.find(([name])=>name===data.manifest.roles.graph)[1]=Buffer.from('GRAPH');});
  await assert.rejects(read(bytes),/hash differs/);
});
test('unlisted, missing and duplicate members reject the package', async () => {
  await assert.rejects(read(packageFixture(d=>d.entries.push(['files/extra.json',Buffer.from('{}')]))),/members differ/);
  await assert.rejects(read(packageFixture(d=>d.entries.pop())),/members differ/);
  await assert.rejects(read(packageFixture(d=>d.entries.push(d.entries[0]))),/Duplicate/);
});
test('unsafe names, symlinks and encryption are rejected', async () => {
  await assert.rejects(read(packageFixture(d=>d.entries.push(['../escape',Buffer.from('bad')]))),/Unsafe/);
  await assert.rejects(read(packageFixture(undefined,{mode:0o120777})),/nonregular/);
  await assert.rejects(read(packageFixture(undefined,{flags:1})),/encrypted/);
});
test('changed role or delivered identity does not substitute another member', async () => {
  await assert.rejects(read(packageFixture(d=>d.manifest.review.artifacts.map={...d.manifest.review.artifacts.map,sha256:'0'.repeat(64)})),/identity/);
  await assert.rejects(read(packageFixture(d=>delete d.manifest.roles.hd_map)),/missing maps/);
});
test('ZIP limits and actual decompressed bytes are bounded before display', async () => {
  const excessive=Buffer.alloc(22); excessive.writeUInt32LE(0x06054b50,0);excessive.writeUInt16LE(130,8);excessive.writeUInt16LE(130,10);
  await assert.rejects(read(excessive),/excessive/);
  const bomb=zip([['manifest.json',Buffer.alloc(2000000)]],{declaredSize:1});
  await assert.rejects(read(bomb),/byte limit/);
  const declared=zip([['manifest.json',Buffer.from('{}')]],{declaredSize:REVIEW_LIMIT+1});
  await assert.rejects(read(declared),/browser limits/);
});
test('cancellation stops verification without delivering a partial pair', async () => {
  const controller=new AbortController();controller.abort();
  await assert.rejects(read(packageFixture(),controller.signal),{name:'AbortError'});
});
test('malformed or nonfinite saved locations cannot reach the viewer', () => {
  for(const mutate of [q=>q.problems[0].points[0][0]=Infinity,q=>q.problems[0].lane=99,q=>q.lanes[0].left.fraction=2,q=>q.problems_limited=undefined]){
    const q=quality();mutate(q);const audits=fixture().evidence;audits.editable.quality=q;
    assert.throws(()=>parseSavedAudits(audits),/saved|Saved/);
  }
});
