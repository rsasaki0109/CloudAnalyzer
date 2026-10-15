import { mapQuality, VoxelReference, type Cloud } from "./wasm/ca_wasm.js";
import type { VoxelReferenceCompute } from "./protocol";

interface Scores { awd: number; scs: number; voxels: number }
let current: { compared: Cloud; reference: Cloud; voxel: number; minPoints: number;
  simd: boolean; scores: Readonly<Scores> } | undefined;
const REFERENCE_BUDGET = 128 * 1024 * 1024;
type Key = { source: Cloud; voxel: number; minPoints: number; simd: boolean };
let seen: Key | undefined;
let referenceCache: (Key & { prepared: VoxelReference; bytes: number }) | undefined;

function matches(key: Key | undefined, source: Cloud, voxel: number, minPoints: number, simd: boolean) {
  return !!key && key.source === source && Object.is(key.voxel, voxel) && Object.is(key.minPoints, minPoints) && key.simd === simd;
}

function releaseReference() { referenceCache?.prepared.free(); referenceCache = undefined; }

function referenceScores(compared: Cloud, reference: Cloud, voxel: number, minPoints: number, simd: boolean,
  cache: boolean | undefined): { values: Float64Array; reference: VoxelReferenceCompute } {
  const reused = matches(referenceCache, reference, voxel, minPoints, simd);
  const repeat = matches(seen, reference, voxel, minPoints, simd);
  seen = { source: reference, voxel, minPoints, simd };
  const enabled = cache ?? (reused || repeat);
  if (enabled && referenceCache && !reused) releaseReference();
  let fallback: string | undefined;
  if (enabled && !referenceCache) {
    // At most n/min eligible voxels; allow Gaussian/root slots and 27 u32
    // neighbours, with two payload capacities plus headers.
    const estimate = Math.ceil(reference.length / Math.max(2, minPoints)) * 640 + 4096;
    if (!Number.isFinite(voxel) || voxel <= 0 || !Number.isSafeInteger(minPoints) || minPoints < 0 || estimate > REFERENCE_BUDGET) {
      fallback = "Voxel reference exceeds preparation allowance or has invalid parameters";
    } else {
      let prepared: VoxelReference | undefined;
      try {
        prepared = new VoxelReference(reference, voxel, minPoints, simd);
        const bytes = prepared.retainedBytes;
        if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > REFERENCE_BUDGET) throw new Error("Voxel reference exceeds retained allowance");
        referenceCache = { source: reference, voxel, minPoints, simd, prepared, bytes };
      } catch (error) { prepared?.free(); fallback = error instanceof Error ? error.message : String(error); }
    }
  }
  if (enabled && referenceCache) {
    try {
      const values = referenceCache.prepared.compute(compared);
      return { values, reference: { cached: true, reused, bytes: referenceCache.bytes,
        neighborVoxels: referenceCache.prepared.neighborVoxels, neighborsUsed: referenceCache.prepared.neighborsUsed,
        rootsReused: referenceCache.prepared.rootsReused, rootsComputed: referenceCache.prepared.rootsComputed } };
    } catch (error) { releaseReference(); fallback = error instanceof Error ? error.message : String(error); }
  }
  return { values: mapQuality(compared, reference, voxel, minPoints, simd), reference: { cached: false, reused: false, fallback } };
}

export function invalidateVoxelScores(cloud: Cloud): void {
  if (current?.compared === cloud || current?.reference === cloud) current = undefined;
  if (referenceCache?.source === cloud) releaseReference();
  if (seen?.source === cloud) seen = undefined;
}

/** Three scalar scores for one immutable pair/settings; no Gaussian or distance arrays. */
export function snapshotVoxelScores(compared: Cloud, reference: Cloud, voxel: number,
  minPoints: number, simd = false, cache = true, referenceCachePolicy?: boolean): { scores: Readonly<Scores>; reused: boolean; reference?: VoxelReferenceCompute } {
  if (cache && current?.compared === compared && current.reference === reference &&
    Object.is(current.voxel, voxel) && Object.is(current.minPoints, minPoints) && current.simd === simd)
    return { scores: current.scores, reused: true };
  const result = referenceScores(compared, reference, voxel, minPoints, simd, referenceCachePolicy);
  const [awd, scs, voxels] = result.values;
  const scores = { awd, scs, voxels };
  if (cache) current = { compared, reference, voxel, minPoints, simd, scores };
  return { scores, reused: false, reference: result.reference };
}
