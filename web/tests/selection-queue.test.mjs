import assert from 'node:assert/strict';
import { test } from 'node:test';
import { SelectionQueue } from '../src/selection-queue.ts';

const deferred = () => {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
};

test('Enter waits for both picks and keeps click order when replies arrive backwards', async () => {
  const queue = new SelectionQueue(), first = deferred(), second = deferred(), points = [];
  queue.enqueue(() => first.promise, p => points.push(p));
  queue.enqueue(() => second.promise, p => points.push(p));
  let road;
  const finished = queue.finish(async () => { road = [...points]; });
  second.resolve('B');
  await Promise.resolve();
  assert.equal(road, undefined);
  first.resolve('A');
  await finished;
  assert.deepEqual(road, ['A', 'B']);
});

test('Escape invalidates pending picks and Enter without blocking a new sketch', async () => {
  const queue = new SelectionQueue(), pending = deferred(), points = [];
  const pick = queue.enqueue(() => pending.promise, p => points.push(p));
  let abandoned = false;
  const finish = queue.finish(async () => { abandoned = true; });
  queue.cancel();
  await queue.enqueue(async () => 'new', p => points.push(p));
  pending.resolve('old');
  await Promise.all([pick, finish]);
  assert.equal(abandoned, false);
  assert.deepEqual(points, ['new']);
});

test('repeated Enter builds once and does not accept clicks during completion', async () => {
  const queue = new SelectionQueue(), pending = deferred();
  let builds = 0, picks = 0;
  const finish = queue.finish(async () => { builds++; await pending.promise; });
  await Promise.resolve();
  await queue.finish(async () => { builds++; });
  await queue.enqueue(async () => 1, () => { picks++; });
  pending.resolve();
  await finish;
  assert.equal(builds, 1);
  assert.equal(picks, 0);
});

test('Backspace is applied after a pending point', async () => {
  const queue = new SelectionQueue(), pending = deferred(), points = [];
  queue.enqueue(() => pending.promise, p => points.push(p));
  const backspace = queue.enqueue(async () => null, () => points.pop());
  pending.resolve('A');
  await backspace;
  assert.deepEqual(points, []);
});
