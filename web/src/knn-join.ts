import type { SorLocalResult } from "./c2c-worker";

/** Scatter partition-local normals without allocating a view for each point. */
export function scatterNormalParts(parts: readonly Float32Array[], indices: readonly Uint32Array[], count: number): Float32Array {
  const out = new Float32Array(count * 3);
  const bits = new Uint32Array(out.buffer);
  for (let j = 0; j < parts.length; j++) {
    const sourcePart = parts[j], idx = indices[j];
    // Raw lanes also preserve signed zero and every NaN payload.
    const part = new Uint32Array(sourcePart.buffer, sourcePart.byteOffset, sourcePart.length);
    for (let i = 0; i < idx.length; i++) {
      const source = i * 3, target = idx[i] * 3;
      bits[target] = part[source];
      bits[target + 1] = part[source + 1];
      bits[target + 2] = part[source + 2];
    }
  }
  return out;
}

export interface SorQueries {
  means: Float64Array;
  /** Transferred to the corresponding lane; finishSorMeans does not read them. */
  queries: Float64Array[];
  /** Global open-query index to the row returned by each lane; -1 means absent. */
  rows: (Int32Array | null)[];
  openIndex: Uint32Array;
}

/** Pack boundary queries directly, retaining row numbers rather than per-point objects. */
export function makeSorQueries(local: readonly SorLocalResult[], indices: readonly Uint32Array[], count: number): SorQueries {
  const parts = local.length;
  const openCount = local.reduce((n, part) => n + part.open.length, 0);
  const counts = new Uint32Array(parts);
  for (const part of local) for (const reach of part.reach) {
    for (let j = 0; j < parts; j++) if ((reach >>> j) & 1) counts[j]++;
  }
  const queries = Array.from(counts, n => new Float64Array(n * 3));
  const rows = Array.from(counts, n => n ? new Int32Array(openCount).fill(-1) : null);
  const next = new Uint32Array(parts);
  const means = new Float64Array(count);
  const openIndex = new Uint32Array(openCount);
  let q = 0;
  for (let i = 0; i < parts; i++) {
    const part = local[i], idx = indices[i];
    for (let j = 0; j < part.means.length; j++) if (!Number.isNaN(part.means[j])) means[idx[j]] = part.means[j];
    for (let o = 0; o < part.open.length; o++, q++) {
      openIndex[q] = idx[part.open[o]];
      const source = o * 3;
      for (let j = 0; j < parts; j++) if ((part.reach[o] >>> j) & 1) {
        const row = next[j]++, target = row * 3;
        rows[j]![q] = row;
        const query = queries[j];
        query[target] = part.openPoints[source];
        query[target + 1] = part.openPoints[source + 1];
        query[target + 2] = part.openPoints[source + 2];
      }
    }
  }
  return { means, queries, rows, openIndex };
}

const USE_PREFIX_SOR_MERGE = true;

/** Merge only the required sorted prefix; unusual rows retain the full sort. */
export function finishSorMeans(local: readonly SorLocalResult[], answers: readonly (Float64Array | null)[], packed: SorQueries, k: number, prefix = USE_PREFIX_SOR_MERGE): Float64Array {
  const width = k + 1;
  const first = prefix && Number.isSafeInteger(k) && k > 0 ? new Float64Array(width) : null;
  const second = first ? new Float64Array(width) : null;
  // A buffer for each actual candidate-list count avoids padding that could
  // change infinity/NaN semantics, and avoids a new array/view per open point.
  const scratch: (Float64Array | undefined)[] = new Array(local.length + 2);
  let q = 0;
  for (const part of local) for (let o = 0; o < part.open.length; o++, q++) {
    const merged = first && second ? mergeSorPrefix(part.candidates, o * width, answers, packed.rows, q, width, first, second) : null;
    if (merged) {
      let sum = 0;
      for (let j = 1; j <= k; j++) sum += Math.sqrt(merged[j]);
      packed.means[packed.openIndex[q]] = sum / k;
      continue;
    }
    let lists = 1;
    for (let j = 0; j < answers.length; j++) if (answers[j] && (packed.rows[j]?.[q] ?? -1) >= 0) lists++;
    const all = scratch[lists] ??= new Float64Array(lists * width);
    let at = 0;
    const own = o * width;
    for (let i = 0; i < width; i++) all[at++] = part.candidates[own + i];
    for (let j = 0; j < answers.length; j++) {
      const answer = answers[j], row = packed.rows[j]?.[q] ?? -1;
      if (!answer || row < 0) continue;
      const source = row * width;
      for (let i = 0; i < width; i++) all[at++] = answer[source + i];
    }
    all.sort();
    let sum = 0;
    for (let j = 1; j <= k; j++) sum += Math.sqrt(all[j]);
    packed.means[packed.openIndex[q]] = sum / k;
  }
  return packed.means;
}

/** Kernel rows are ascending nonnegative distances, with optional infinity padding. */
function orderedSorRow(values: Float64Array, start: number, width: number): boolean {
  let previous = -Infinity;
  for (let i = 0; i < width; i++) {
    const value = values[start + i];
    // Retain the native typed-array sort for NaNs, negative/signed-zero or
    // malformed/unsorted rows, including any historical NaN payload behavior.
    if (!(value >= 0) || value < previous || (value === 0 && Object.is(value, -0))) return false;
    previous = value;
  }
  return true;
}

function mergeSorPrefix(own: Float64Array, start: number, answers: readonly (Float64Array | null)[], rows: readonly (Int32Array | null)[], q: number, width: number, first: Float64Array, second: Float64Array): Float64Array | null {
  if (!orderedSorRow(own, start, width)) return null;
  for (let i = 0; i < width; i++) first[i] = own[start + i];
  let current = first, next = second;
  for (let j = 0; j < answers.length; j++) {
    const answer = answers[j], row = rows[j]?.[q] ?? -1;
    if (!answer || row < 0) continue;
    const offset = row * width;
    if (!orderedSorRow(answer, offset, width)) return null;
    let a = 0, b = 0;
    for (let i = 0; i < width; i++) {
      // Both cursors consume at most width values. Equal distances retain
      // the earlier list; all admitted equal values have identical f64 bits.
      if (current[a] <= answer[offset + b]) next[i] = current[a++];
      else next[i] = answer[offset + b++];
    }
    const swap = current; current = next; next = swap;
  }
  return current;
}
