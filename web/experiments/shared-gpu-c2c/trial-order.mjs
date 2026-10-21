// Williams design: each four-row block balances positions and predecessors.
export const verifierModes = Object.freeze(['pool', 'pool-staged', 'shared-serial', 'shared']);
const rows = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];
export function verifierTrialOrder(index) {
  if (!Number.isSafeInteger(index) || index < 0) throw new Error('Expected a nonnegative trial index');
  return rows[index % rows.length].map(i => verifierModes[i]);
}
