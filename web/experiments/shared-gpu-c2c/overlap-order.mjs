// Six-row Williams design; the four-mode schedule remains available unchanged.
export const overlapModes = Object.freeze(['pool', 'pool-staged', 'pool-staged-overlap', 'shared-serial', 'shared', 'shared-overlap']);
const first = [0, 1, 5, 2, 4, 3];
export function overlapTrialOrder(index) {
  if (!Number.isSafeInteger(index) || index < 0) throw new Error('Expected a nonnegative trial index');
  return first.map(i => overlapModes[(i + index % 6) % 6]);
}
