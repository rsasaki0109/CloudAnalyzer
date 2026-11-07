// Level-of-detail index builds, split across the worker pool for large clouds.

import { Cloud } from "../wasm/ca_wasm.js";
import { poolSize, runSlices } from "../pool";

/** Clouds at least this large build their octree on the worker pool. */
const MIN_PARALLEL_INDEX = 1_000_000;
const BUCKETS = 64;
/**
 * Build the octree on the worker pool, in the three steps of
 * `Cloud.indexCube`: sort slices of the cloud by bucket, build each bucket
 * (largest first, so the slowest job starts early), then place them.
 */
export async function buildIndex(cloud: Cloud): Promise<number> {
  const n = cloud.length;
  if (n < MIN_PARALLEL_INDEX || poolSize() < 2) {
    cloud.buildIndex();
    return 1;
  }
  const cube = new Float64Array(cloud.indexCube());
  const parts = poolSize();
  const per = Math.ceil(n / parts);
  const chunks = await runSlices(parts, (k) => ({
    kind: "bucket-chunk" as const,
    positions: cloud.positionsRange(k * per, (k + 1) * per),
    colors: cloud.colorsRange(k * per, (k + 1) * per) ?? null,
    cube: cube.slice(),
  }));
  const hasColors = chunks[0].colors !== null;
  // Where bucket b starts within chunk k.
  const starts = chunks.map((c) => {
    const s = new Uint32Array(BUCKETS + 1);
    for (let b = 0; b < BUCKETS; b++) s[b + 1] = s[b] + c.counts[b];
    return s;
  });
  const sizes = new Uint32Array(BUCKETS);
  for (const c of chunks) for (let b = 0; b < BUCKETS; b++) sizes[b] += c.counts[b];
  const keys = Array.from({ length: BUCKETS }, (_, b) => b)
    .filter((b) => sizes[b] > 0)
    .sort((a, b) => sizes[b] - sizes[a]);
  // Cloud index of each bucket's points, in bucket order.
  const origins = new Map<number, Uint32Array>();
  const results = await runSlices(keys.length, (i) => {
    const b = keys[i];
    const positions = new Float64Array(sizes[b] * 3);
    const colors = hasColors ? new Uint8Array(sizes[b] * 3) : null;
    const origin = new Uint32Array(sizes[b]);
    let at = 0;
    chunks.forEach((c, k) => {
      const [from, to] = [starts[k][b], starts[k][b + 1]];
      positions.set(c.positions.subarray(from * 3, to * 3), at * 3);
      if (colors && c.colors) colors.set(c.colors.subarray(from * 3, to * 3), at * 3);
      for (let j = from; j < to; j++) origin[at + j - from] = c.order[j] + k * per;
      at += to - from;
    });
    origins.set(b, origin);
    return { kind: "bucket" as const, positions, colors, cube: cube.slice(), key: b };
  });
  const rootKept = new Uint32Array(BUCKETS);
  const level1Kept = new Uint32Array(BUCKETS);
  results.forEach((r, i) => {
    rootKept[keys[i]] = r.counts[0];
    level1Kept[keys[i]] = r.counts[1];
  });
  cloud.beginBuckets(cube, sizes, rootKept, level1Kept);
  results.forEach((r, i) => {
    const origin = origins.get(keys[i])!;
    const order = new Uint32Array(r.order.length);
    for (let j = 0; j < order.length; j++) order[j] = origin[r.order[j]];
    cloud.putBucket(keys[i], r.positions, r.colors ?? undefined, order, r.nodes);
  });
  cloud.finishBuckets();
  return Math.min(poolSize(), keys.length);
}
