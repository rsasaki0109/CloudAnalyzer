import test from 'node:test';
import assert from 'node:assert/strict';
import { verifierModes, verifierTrialOrder } from '../experiments/shared-gpu-c2c/trial-order.mjs';

test('shared GPU comparison balances positions and every directed predecessor pair', () => {
  const rows = Array.from({ length: 4 }, (_, i) => verifierTrialOrder(i));
  const pairs = new Map();
  for (const row of rows) {
    assert.deepEqual([...row].sort(), [...verifierModes].sort());
    for (let j = 1; j < row.length; j++) {
      const key = `${row[j - 1]}:${row[j]}`;
      pairs.set(key, (pairs.get(key) ?? 0) + 1);
    }
  }
  for (let position = 0; position < 4; position++) assert.equal(new Set(rows.map(row => row[position])).size, 4);
  assert.equal(pairs.size, 12);
  assert.ok([...pairs.values()].every(count => count === 1));
  assert.deepEqual(verifierTrialOrder(4), rows[0]);
  const changed = verifierTrialOrder(0); changed.reverse();
  assert.deepEqual(verifierTrialOrder(0), rows[0]);
});

test('shared GPU comparison rejects invalid schedule indices', () => {
  for (const value of [-1, 0.5, NaN, Infinity, '0']) assert.throws(() => verifierTrialOrder(value));
});
