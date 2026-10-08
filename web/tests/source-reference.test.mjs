import assert from 'node:assert/strict';
import { test } from 'node:test';
import { referenceFile, matchesFile, parseSource } from '../src/source-reference.ts';

test('same names and lengths do not allow altered source content', async () => {
  const original = new File(['point cloud A'], 'map.ply');
  const different = new File(['point cloud B'], 'map.ply');
  const saved = await referenceFile(original);
  assert.equal(await matchesFile(new File(['point cloud A'], 'map.ply'), saved), true);
  assert.equal(await matchesFile(different, saved), false);
});

test('fingerprinting covers the interior of files across chunk boundaries', async () => {
  const data = new Uint8Array(9 * 1024 * 1024);
  const saved = await referenceFile(new File([data], 'large.las'));
  data[4 * 1024 * 1024] = 1;
  assert.equal(await matchesFile(new File([data], 'large.las'), saved), false);
});

test('cancellation stops fingerprinting and does not cache a partial identity', async () => {
  const file = new File(['cloud'], 'map.ply');
  const controller = new AbortController(); controller.abort();
  await assert.rejects(referenceFile(file, controller.signal), { name: 'AbortError' });
  assert.equal((await referenceFile(file)).kind, 'file');
});

test('remote references require strong HTTP content identities', () => {
  assert.throws(() => parseSource({kind:'http',name:'cloud.laz',url:'https://example.com/cloud.laz',size:10,etag:'W/"weak"'}));
  assert.equal(parseSource({kind:'http',name:'cloud.laz',url:'https://example.com/cloud.laz',size:10,etag:'"strong"'}).kind, 'http');
});
