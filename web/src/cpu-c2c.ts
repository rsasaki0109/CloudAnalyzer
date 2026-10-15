import { planCloudToCloud, type Cloud } from "./wasm/ca_wasm.js";
import { MIN_PARALLEL_QUERIES, poolSize } from "./pool";
import { findCpuPreparation, prepareCpuSlices } from "./cpu-preparation";

export { invalidateCpuPreparation as invalidateCpuC2c } from "./cpu-preparation";

/** Reuse preparation; compute distances and summaries afresh on every call. */
export async function cachedCloudToCloud(compared: Cloud, reference: Cloud, simd?: boolean,
  stream = true, pipeline = false): Promise<{ distances: Float64Array; workers: number; reused: boolean; inputBytes: number } | null> {
  const count = compared.length;
  const parts = Math.min(poolSize(), Math.floor(count / MIN_PARALLEL_QUERIES));
  if (parts < 2) return null;
  let job = findCpuPreparation("c2c", compared, reference, String(parts));
  const reused = !!job;
  if (!job) {
    // Capture every coordinate and output index before yielding to mutations.
    const plan = planCloudToCloud(compared, reference, parts);
    try {
      const inputs = Array.from({ length: plan.length }, (_, lane) => ({
        reference: plan.takeReference(lane), queries: plan.takeQueries(lane), indices: plan.takeQueryIndices(lane),
      }));
      job = prepareCpuSlices("c2c", compared, reference, String(parts), inputs);
    } finally { plan.free(); }
  }
  let distances: Float64Array | undefined;
  const consume = (lane: number, part: Float64Array) => {
    const indices = job.indices[lane]!;
    distances ??= new Float64Array(count);
    for (let i = 0; i < indices.length; i++) distances[indices[i]] = part[i];
  };
  if (stream) await job.run(consume, simd, pipeline);
  else {
    const collected: Float64Array[] = [];
    // Join inside the lease, before retiring a concurrently replaced group.
    let remaining = job.workers;
    await job.run((lane, part) => {
      collected[lane] = part;
      if (--remaining === 0) collected.forEach((values, k) => consume(k, values));
    }, simd, pipeline);
  }
  return { distances: distances ?? new Float64Array(count), workers: job.workers, reused, inputBytes: job.inputBytes };
}
