import { IcpPrepared, registerIcp, c2cSimdEnabled, type Cloud } from "./wasm/ca_wasm.js";

const BUDGET = 128 * 1024 * 1024;
let current: { source: Cloud; prepared: IcpPrepared; bytes: number } | undefined;

export function invalidateIcpReference(cloud: Cloud): void {
  if (current?.source !== cloud) return;
  current.prepared.free();
  current = undefined;
}

/** Synchronous jobs own their reference; only one idle reference is retained. */
export function cachedRegisterIcp(moving: Cloud, reference: Cloud, maxIterations: number,
  overlap: number, matchCentroids: boolean, pointToPlane: boolean, cache = true, simd = false, temporal = false) {
  const reused = cache && current?.source === reference;
  if (!reused && current) invalidateIcpReference(current.source);
  // Conservative point/tree/index/node/normal capacity estimate before allocation.
  if (cache && !current && reference.length * 96 + 16_384 <= BUDGET) {
    const prepared = new IcpPrepared(reference);
    const bytes = prepared.retainedBytes;
    if (bytes <= BUDGET) current = { source: reference, prepared, bytes };
    else prepared.free();
  }
  try {
    const outcome = current?.prepared.compute(moving, maxIterations, overlap, matchCentroids, pointToPlane, simd, temporal) ??
      registerIcp(moving, reference, maxIterations, overlap, matchCentroids, pointToPlane, simd, temporal);
    return { outcome, referenceReused: !!reused, referenceCached: !!current, referenceBytes: current?.bytes,
      correspondenceSimd: simd && c2cSimdEnabled(), correspondenceTemporal: outcome.correspondenceTemporal,
      temporalQueries: outcome.temporalQueries, temporalScratchBytes: outcome.temporalScratchBytes };
  } catch (error) {
    if (current) invalidateIcpReference(current.source);
    throw error;
  }
}
