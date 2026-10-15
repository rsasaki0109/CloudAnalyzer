import { M3c2Plan, m3c2CloudDistances, type Cloud } from "./wasm/ca_wasm.js";
import { poolSize, runOn } from "./pool";
import { prepareM3c2 } from "./m3c2-preparation";
import { planM3c2Cores } from "./m3c2-core-selection";

const MIN_CORES_PER_WORKER = 10_000;
const MAX_M3C2_WORKERS = 4;
// Spread expensive spatial regions across lanes while retaining local scans.
const CORE_BLOCK = 1024;
// Full inputs are copied to each WASM instance. Allow for JS/WASM copies,
// owned positions and both trees; this is an estimate, not a heap limit.
const COPY_BUDGET = 256 * 1024 * 1024;
const BYTES_PER_INPUT_POINT = 128;
// The analysis lane borrows its resident Cloud inputs: only tree construction
// storage is added. 80 allows for partition items, owned tree points/indices
// and node capacity growth. Core/result/source storage remains excluded from
// this scheduling estimate, as for the existing pool-only policy.
const MAIN_BYTES_PER_INPUT_POINT = 80;
// An additional bounded intermediate cache, across all lanes of one pair.
const NORMAL_CACHE_BUDGET = 64 * 1024 * 1024;
const NORMAL_BYTES_PER_CORE = 56; // f64 core triple + tagged optional f64 normal

// Exact trees retain <=34 bytes/point (coordinates, indices, nodes). 36
// allows headers/capacity slack. Imports allow 104 for JS + WASM snapshot
// copies + decoded trees + validation bitmap. Exporting a fresh pair while
// earlier lanes import allows 108 on the analysis lane. This is the same
// input-storage scheduling estimate, excluding cores/results/resident heaps.
function preparedBytes(points: number, pools: number): number {
  return points * Math.max(MAIN_BYTES_PER_INPUT_POINT, 108 + (pools - 1) * 104, 36 + pools * 104);
}

/** Null for small/memory-heavy jobs; otherwise all inputs are snapshotted
 * synchronously before the first yield, including core attributes. */
export function parallelM3c2(compared: Cloud, reference: Cloud, normalRadius: number,
  projectionRadius: number, maxDepth: number, coreSpacing: number, interleaved = true, radiusBounds = false, mainWorker = true,
  preparedTrees = true, cacheNormals: boolean | undefined = undefined, sharedSearch = true, fusedProjection?: boolean,
  cacheCoreSelection = true,
): Promise<{ cloud: Cloud; workers: number; mainWorker: boolean; preparedTrees: boolean; treeCacheHit: boolean;
  normalCacheHit: boolean; normalCacheBytes: number; fusedProjection: boolean;
  coreSelectionCacheHit: boolean; coreSelectionCacheBytes: number }> | null {
  const limit = Math.min(poolSize(), MAX_M3C2_WORKERS);
  const inputs = compared.length + reference.length;
  const poolBytes = inputs * BYTES_PER_INPUT_POINT;
  const available = Math.min(limit, Math.floor(COPY_BUDGET / poolBytes));
  let withMain = mainWorker ? Math.min(limit,
    1 + Math.floor((COPY_BUDGET - inputs * MAIN_BYTES_PER_INPUT_POINT) / poolBytes)) : available;
  const prepared = preparedTrees && mainWorker;
  if (prepared) {
    withMain = 1;
    for (let pools = 1; pools < limit; pools++) {
      if (preparedBytes(inputs, pools) <= COPY_BUDGET) withMain = pools + 1;
    }
  }
  const eligible = prepared ? withMain : Math.max(available, withMain);
  if (eligible < 2 || compared.length < MIN_CORES_PER_WORKER * 2) return null;
  const selected = planM3c2Cores(compared, coreSpacing, cacheCoreSelection);
  let plan: M3c2Plan | undefined = selected.plan;
  const parts = Math.min(eligible, Math.floor(plan.length / MIN_CORES_PER_WORKER));
  if (parts < 2) { plan.free(); return null; }
  // Prepared jobs reuse the analysis tree; the original control involves
  // that lane only when it adds capacity under the same budget.
  const mainPart = prepared || parts > available ? parts - 1 : undefined;
  let input: Float64Array, referencePoints: Float64Array;
  let cached: ReturnType<typeof prepareM3c2> | undefined;
  try {
    if (prepared) cached = prepareM3c2(compared, reference, parts - 1);
    else {
      input = compared.rawPositions();
      referencePoints = reference.rawPositions();
    }
  } catch (error) { plan.free(); throw error; }
  const count = plan.length;
  const fitSignature = JSON.stringify([normalRadius, coreSpacing, radiusBounds, interleaved]);
  const repeatFit = cached?.preparation.normalFitSignature === fitSignature;
  const per = Math.ceil(count / parts);
  const full = Math.floor(count / (parts * CORE_BLOCK)) * parts * CORE_BLOCK;
  const tail = count - full;
  return (async () => {
    try {
      const cores = (k: number) => interleaved ? plan!.interleavedPoints(k, parts, CORE_BLOCK)
        : plan!.points(k * per, Math.min(per, count - k * per));
      let results: Float64Array[];
      let normalCacheHit = false, normalCacheBytes = 0;
      let fused = fusedProjection ?? false;
      if (cached) {
        const capacity = Math.floor(NORMAL_CACHE_BUDGET / (NORMAL_BYTES_PER_CORE * parts));
        const maxCachedCores = per <= capacity ? capacity : 0;
        const useNormalCache = cacheNormals ?? (repeatFit && maxCachedCores > 0);
        const outputs = await cached.preparation.compute(cores, normalRadius, projectionRadius, maxDepth,
          radiusBounds, useNormalCache, maxCachedCores, sharedSearch, fusedProjection);
        normalCacheHit = useNormalCache && outputs.every(part => part.normalCacheHit);
        fused = outputs.every(part => part.fusedProjection);
        normalCacheBytes = outputs.reduce((sum, part) => sum + part.normalCacheBytes, 0);
        if (!Number.isSafeInteger(normalCacheBytes) || normalCacheBytes < 0 || normalCacheBytes > NORMAL_CACHE_BUDGET)
          throw new Error("M3C2 normal cache exceeded its budget");
        results = outputs.map(part => part.distances);
      } else {
        const pending = Promise.allSettled(Array.from({ length: parts - Number(mainPart !== undefined) }, async (_, k) => runOn(k, {
          kind: "m3c2" as const,
          reference: referencePoints,
          compared: input,
          core: cores(k),
          normalRadius, projectionRadius, maxDepth, radiusBounds, sharedSearch, fusedProjection,
        }))).then(outcomes => {
          const failed = outcomes.find(outcome => outcome.status === "rejected");
          if (failed?.status === "rejected") throw failed.reason;
          return outcomes.map(outcome => (outcome as PromiseFulfilledResult<Float64Array>).value);
        });
        // Attach the rejection handler before synchronous work. Drain the pool
        // response even if the main lane fails; never publish a partial cloud.
        const settled = pending.then(values => ({ ok: true as const, values }), error => ({ ok: false as const, error }));
        let mainValues: Float64Array | undefined;
        let mainFailed = false, mainError: unknown;
        if (mainPart !== undefined) {
          try {
            // No await before this call: original inputs remain alive and
            // unchanged until their borrowed computation finishes. Pool lanes
            // and the result plan already own snapshots for the later yield.
            mainValues = m3c2CloudDistances(compared, reference, cores(mainPart),
              normalRadius, projectionRadius, maxDepth, radiusBounds, sharedSearch, fusedProjection);
          } catch (error) { mainFailed = true; mainError = error; }
        }
        const outcome = await settled;
        if (!outcome.ok) throw outcome.error;
        if (mainFailed) throw mainError;
        results = outcome.values;
        if (mainValues) results.push(mainValues);
      }
      const joined = new Float64Array(count * 2);
      results.forEach((part, k) => {
        if (!interleaved) {
          if (part.length !== Math.min(per, count - k * per) * 2)
            throw new Error("Incomplete M3C2 worker result");
          joined.set(part, k * per * 2);
          return;
        }
        const tailCount = Math.floor(tail / parts) + (k < tail % parts ? 1 : 0);
        if (part.length !== (full / parts + tailCount) * 2)
          throw new Error("Incomplete M3C2 worker result");
        let offset = 0;
        for (let first = k * CORE_BLOCK; first < full; first += parts * CORE_BLOCK) {
          const length = CORE_BLOCK * 2;
          joined.set(part.subarray(offset, offset + length), first * 2);
          offset += length;
        }
        const tailFirst = full + k * Math.floor(tail / parts) + Math.min(k, tail % parts);
        joined.set(part.subarray(offset), tailFirst * 2);
      });
      const finished = plan!;
      plan = undefined; // finish consumes its WASM object, including on error.
      const cloud = finished.finish(joined);
      if (cached) cached.preparation.normalFitSignature = fitSignature;
      return { cloud, workers: parts, mainWorker: mainPart !== undefined,
        preparedTrees: prepared, treeCacheHit: cached?.hit ?? false, normalCacheHit, normalCacheBytes, fusedProjection: fused,
        coreSelectionCacheHit: selected.hit, coreSelectionCacheBytes: selected.bytes };
    } catch (error) { cached?.preparation.retire(); throw error; }
    finally { plan?.free(); }
  })();
}
