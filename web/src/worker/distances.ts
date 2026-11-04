// Cloud-to-cloud and cloud-to-mesh distances split across the worker pool.

import { Cloud, Mesh, planCloudToCloud } from "../wasm/ca_wasm.js";
import { acquireMesh, trackMeshSources } from "../mesh-preparation";
import type { Slice } from "../c2c-worker";
import { eachSlice, MIN_PARALLEL_QUERIES, poolSize, runOn, runSlices } from "../pool";
import type { C2cOutput } from "../protocol";

/**
 * Split a C2C job into spatially compact parts and run them on the worker
 * pool. Returns null when the job is too small for the split to pay off.
 */
export async function parallelCloudToCloud(
  compared: Cloud,
  reference: Cloud,
  simd?: boolean,
  stream = true,
): Promise<{ distances: Float64Array; workers: number } | null> {
  const count = compared.length;
  const parts = Math.min(poolSize(), Math.floor(count / MIN_PARALLEL_QUERIES));
  if (parts < 2) return null;
  const plan = planCloudToCloud(compared, reference, parts);
  const slices: Extract<Slice, { kind: "cloud" }>[] = Array.from({ length: plan.length }, (_, k) => ({
    kind: "cloud",
    reference: plan.takeReference(k),
    queries: plan.takeQueries(k),
    simd,
  }));
  const indices: (Uint32Array | undefined)[] = Array.from({ length: plan.length }, (_, k) => plan.takeQueryIndices(k));
  plan.free();
  let distances: Float64Array | undefined;
  let closed = false;
  const consume = (k: number, part: Float64Array) => {
    if (closed) return;
    const idx = indices[k]!;
    if (part.length !== idx.length) throw new Error("Incomplete C2C worker result");
    distances ??= new Float64Array(count);
    for (let i = 0; i < idx.length; i++) distances[idx[i]] = part[i];
    indices[k] = undefined;
  };
  try {
    if (stream) await eachSlice<Extract<Slice, { kind: "cloud" }>>(slices.length, k => slices[k], consume);
    else (await runSlices(slices.length, k => slices[k])).forEach((part, k) => consume(k, part));
    return { distances: distances ?? new Float64Array(count), workers: slices.length };
  } finally { closed = true; }
}
/**
 * C2M on the worker pool: each worker gets the whole mesh and a contiguous
 * slice of the compared points (octree order keeps slices spatially compact).
 */
export async function parallelCloudToMesh(
  compared: Cloud,
  mesh: Mesh,
  signed: boolean,
  cache = true,
  gpu = false,
  orderCache = true,
): Promise<{ distances: Float64Array; workers: number; cached: boolean; reused: boolean; slices: number; retainedBytes?: number; queryOrder?: { hit: boolean; bytes: number; entries: number }; fallback?: string; compute?: C2cOutput["meshCompute"] } | null> {
  const parts = Math.min(poolSize(), Math.floor(compared.length / MIN_PARALLEL_QUERIES));
  if (parts < 2) return null;
  const guard = trackMeshSources(mesh, compared);
  try {
    const count = compared.length, positions = compared.rawPositions(), per = Math.ceil(count / parts);
    const queries = (k: number) => positions.slice(k * per * 3, Math.min(count, (k + 1) * per) * 3);
    const join = (results: Float64Array[]) => {
      guard.assertSource();
      const distances = new Float64Array(count);
      results.forEach((part, k) => {
        if (part.length !== Math.min(per, count - k * per)) throw new Error("Incomplete C2M worker result");
        distances.set(part, k * per);
      });
      return distances;
    };
    let fallback: string | undefined;
    const compute: C2cOutput["meshCompute"] = gpu ? {backend:"cpu",prepareMillis:0,gpuMillis:0,verifyMillis:0,batches:0,completeRows:0,referenceReused:false,gpuBytes:0} : undefined;
    let lease;
    try {
      lease = acquireMesh(mesh, parts, cache);
      if (lease) {
        let hints: Uint32Array | undefined;
        if (compute) {
          try {
            const result = await lease.gpuHints(positions); guard.assertSource();
            hints=result.hints;
            compute.backend="webgpu";compute.prepareMillis=result.prepareMillis;compute.gpuMillis=result.gpuMillis;
            compute.batches=result.batches;compute.completeRows=result.completeRows;compute.referenceReused=result.referenceReused;compute.gpuBytes=result.gpuBytes;
          } catch (error) {guard.assertSource();compute.fallback=error instanceof Error?error.message:String(error);}
        }
        const verifyStart=performance.now();
        const distances = join(await lease.run(parts, queries, signed, hints ? k=>hints!.slice(k*per,Math.min(count,(k+1)*per)) : undefined, orderCache));
        if(compute)compute.verifyMillis=performance.now()-verifyStart;
        return { distances, workers: lease.workers, cached: true, reused: lease.reused, slices: parts, retainedBytes: lease.bytes(), queryOrder: lease.queryOrderState(), compute };
      }
    } catch (error) {
      guard.assertSource();
      lease?.retire();
      fallback = error instanceof Error ? error.message : String(error);
      if(compute){compute.backend="cpu";compute.fallback=fallback;}
    } finally { lease?.release(); }
    guard.assertSource();
    const vertices = mesh.rawVertices(), indices = mesh.indices();
    if(compute){compute.backend="cpu";compute.fallback??="Retained mesh preparation is unavailable";}
    // Drain all fresh slices too, so a failed lane cannot publish late data.
    const pending = Array.from({ length: parts }, (_, k) => runOn(k, { kind: "mesh" as const, vertices, indices, queries: queries(k), signed }));
    const results = await Promise.allSettled(pending);
    guard.assertSource();
    const failed = results.find(r => r.status === "rejected");
    if (failed?.status === "rejected") throw failed.reason;
    return { distances: join(results.map(r => (r as PromiseFulfilledResult<Float64Array>).value)),
      workers: parts, cached: false, reused: false, slices: parts, fallback, compute };
  } finally { guard.release(); }
}
