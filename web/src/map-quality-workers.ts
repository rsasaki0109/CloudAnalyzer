import { cloudToCloud, planCloudToCloud, summarizeDistances, type Cloud, type C2cResult } from "./wasm/ca_wasm.js";
import { snapshotVoxelScores } from "./map-voxel-cache";
import { eachSlice, MIN_PARALLEL_QUERIES, poolSize, runSlices } from "./pool";
import type { Slice } from "./c2c-worker";
import type { MapQuality, VoxelReferenceCompute } from "./protocol";
import { findCpuPreparation, prepareCpuSlices } from "./cpu-preparation";

// Scheduling estimate for input coordinate/index copies and joined output.
// Count the actual spatially pruned references; this is not a total heap cap.
const COPY_BUDGET = 256 * 1024 * 1024;
type Job = { direction: number; indices: Uint32Array | null; slice?: Extract<Slice, { kind: "cloud" }> };

function partsFor(counts: number[], available: number): number[] {
  if (available < 2 || 40 * (counts[0] + counts[1]) > COPY_BUDGET) return [0, 0];
  const parts = counts.map(n => n >= MIN_PARALLEL_QUERIES ? 1 : 0);
  while (parts[0] + parts[1] < available) {
    const next = counts.map((n, i) => parts[i] ? n / (parts[i] + 1) : 0);
    const i = next[0] >= next[1] ? 0 : 1;
    if (next[i] < MIN_PARALLEL_QUERIES) break;
    parts[i]++;
  }
  return parts;
}

/** Retains the UI's original float32 inlier arithmetic and traversal order. */
function inliers(distances: Float32Array, threshold: number) {
  let [n, sum, sum2] = [0, 0, 0];
  for (const d of distances) {
    if (!(d <= threshold)) continue;
    n++; sum += d; sum2 += d * d;
  }
  return { share: distances.length ? n / distances.length : 0,
    mean: n ? sum / n : Number.NaN, rms: n ? Math.sqrt(sum2 / n) : Number.NaN };
}

/** Both directions share one bounded queue. All clouds are read before the
 * first yield; the existing voxel kernel runs while the pool computes C2C. */
export async function evaluateMapQuality(compared: Cloud, reference: Cloud,
  threshold: number, voxel: number, minPoints: number, streamSummaries = true, cpuCache = true, cpuPipeline = true, voxelSimd = false, voxelCache = true, voxelReferenceCache?: boolean,
): Promise<{ result: C2cResult; distances: Float32Array; quality: MapQuality; workers: number;
  cached: boolean; reused: boolean; inputBytes?: number; voxelCacheHit: boolean; voxelReference?: VoxelReferenceCompute }> {
  const clouds = [compared, reference];
  const counts = clouds.map(c => c.length);
  const available = poolSize();
  const parts = partsFor(counts, available);
  const directions: Job[][] = [[], []];
  const layout = parts.join(",");
  let prepared = cpuCache ? findCpuPreparation("map-quality", compared, reference, layout) : undefined;
  const reused = !!prepared;
  const results: (C2cResult | undefined)[] = [];
  let coloring: Float32Array | undefined;
  let closed = false;
  try {
    if (prepared) {
      // Reconstruct the same alternating lane layout without reading coordinates.
      let lane = 0;
      for (let k = 0; k < Math.max(...parts); k++) {
        for (let d = 0; d < 2; d++) if (k < parts[d]) {
          directions[d].push({ direction: d, indices: prepared.indices[lane++] });
        }
      }
    }
    for (let d = 0; d < 2; d++) {
      if (!parts[d]) {
        results[d] = cloudToCloud(clouds[d], clouds[1 - d]);
      } else if (prepared) {
        continue;
      } else if (parts[d] === 1) {
        directions[d].push({ direction: d, indices: null, slice: { kind: "cloud",
          queries: clouds[d].rawPositions(), reference: clouds[1 - d].rawPositions() } });
      } else {
        const plan = planCloudToCloud(clouds[d], clouds[1 - d], parts[d]);
        try {
          for (let k = 0; k < plan.length; k++) {
            directions[d].push({ direction: d, indices: plan.takeQueryIndices(k),
              slice: { kind: "cloud", queries: plan.takeQueries(k), reference: plan.takeReference(k) } });
          }
        } finally { plan.free(); }
      }
    }
    const copiedBytes = (prepared?.inputBytes ?? directions.flat().reduce((n, job) => n + job.slice!.queries.byteLength +
      job.slice!.reference.byteLength + (job.indices?.byteLength ?? 0), 0)) +
      counts.reduce((n, count, d) => n + (parts[d] ? count * 12 : 0), 0);
    if (copiedBytes > COPY_BUDGET) {
      // Planning can briefly exceed this estimate. Release the owned slices
      // before dispatch rather than multiplying large references across heaps.
      for (const direction of directions) direction.length = 0;
      for (let d = 0; d < 2; d++) {
        if (parts[d]) results[d] = cloudToCloud(clouds[d], clouds[1 - d]);
        parts[d] = 0;
      }
    }
    // Alternate directions so even the first dispatch wave includes both.
    const jobs: Job[] = [];
    for (let k = 0; k < Math.max(directions[0].length, directions[1].length); k++) {
      for (const direction of directions) if (direction[k]) jobs.push(direction[k]);
    }
    if (cpuCache && jobs.length && !prepared) {
      prepared = prepareCpuSlices("map-quality", compared, reference, layout,
        jobs.map(job => ({ ...job.slice!, indices: job.indices })));
      // Coordinate buffers have transferred; only output indices stay here.
      for (const job of jobs) job.slice = undefined;
    }
    const joined: (Float64Array | undefined)[] = [];
    const matches: (ReturnType<typeof inliers> | undefined)[] = [];
    const remaining = directions.map(direction => direction.length);
    const finish = (d: number) => {
      const distances = joined[d];
      if (distances && streamSummaries && d === 1) {
        // Reverse distances only feed inlier arithmetic: round them exactly
        // as before, without constructing unused statistics or sorting them.
        matches[d] = inliers(new Float32Array(distances), threshold);
      } else {
        if (distances) results[d] = summarizeDistances(distances);
        const values = results[d]!.distances();
        matches[d] = inliers(values, threshold);
        if (d === 0) coloring = values;
      }
      joined[d] = undefined;
    };
    const consume = (k: number, values: Float64Array) => {
      const job = jobs[k], d = job.direction;
      const expected = job.indices?.length ?? counts[d];
      if (values.length !== expected) throw new Error("Incomplete map quality worker result");
      const distances = joined[d] ??= new Float64Array(counts[d]);
      if (!job.indices) distances.set(values);
      else for (let i = 0; i < expected; i++) distances[job.indices[i]] = values[i];
      if (!prepared) job.indices = null;
    };
    const each = (done: (k: number, values: Float64Array) => void) => prepared
      ? prepared.run(done, undefined, cpuPipeline)
      : eachSlice<Extract<Slice, { kind: "cloud" }>>(jobs.length, k => jobs[k].slice!, done);
    const work = streamSummaries
      ? each((k, values) => {
        // Another lane may still finish after a failed job has been closed.
        if (closed) return;
        consume(k, values);
        if (--remaining[jobs[k].direction] === 0) finish(jobs[k].direction);
      })
      : (prepared ? (async () => {
        const values: Float64Array[] = [];
        // Join before the lease releases indices of a concurrently retired pair.
        let remaining = jobs.length;
        await prepared.run((k, part) => {
          values[k] = part;
          if (--remaining === 0 && !closed) {
            values.forEach((part, k) => consume(k, part));
            for (let d = 0; d < 2; d++) if (parts[d]) finish(d);
          }
        }, undefined, cpuPipeline);
      })() : runSlices(jobs.length, k => jobs[k].slice!).then(values => {
        if (closed) return;
        values.forEach((part, k) => consume(k, part));
        for (let d = 0; d < 2; d++) if (parts[d]) finish(d);
      }));
    // Attach both handlers before the synchronous voxel computation, including
    // when that kernel fails. In-flight slices own all of their coordinates.
    const pending = work.then(() => ({ ok: true as const }), error => {
      closed = true;
      return { ok: false as const, error };
    });
    const voxelSnapshot = snapshotVoxelScores(compared, reference, voxel, minPoints, voxelSimd, voxelCache, voxelReferenceCache);
    const { awd, scs, voxels } = voxelSnapshot.scores;
    const outcome = await pending;
    if (!outcome.ok) throw outcome.error;
    for (let d = 0; d < 2; d++) if (!matches[d]) finish(d);
    const [a, b] = matches as ReturnType<typeof inliers>[];
    const quality: MapQuality = { threshold, accuracy: a.rms, precision: a.share,
      completeness: b.share, f1: a.share + b.share > 0 ? 2 * a.share * b.share / (a.share + b.share) : 0,
      chamfer: (a.mean + b.mean) / 2, awd, scs, voxels };
    const result = results[0]!;
    results[0] = undefined; // Caller owns the forward result; reverse is temporary.
    return { result, distances: coloring!, quality, workers: Math.max(1, Math.min(available, jobs.length)),
      cached: !!prepared, reused, inputBytes: prepared?.inputBytes, voxelCacheHit: voxelSnapshot.reused, voxelReference: voxelSnapshot.reference };
  } catch (error) {
    prepared?.retire();
    throw error;
  } finally { closed = true; for (const result of results) result?.free(); }
}
