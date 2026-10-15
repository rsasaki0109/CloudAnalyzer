import { M3c2Prepared, type Cloud } from "./wasm/ca_wasm.js";
import { releaseOn, runOn, runOnGeneration, workerGeneration } from "./pool";
import type { M3c2PartResult } from "./c2c-worker";

let nextJob = 0;
let current: M3c2Preparation | undefined;

/** One idle immutable pair plus leases for computations already in flight. */
export class M3c2Preparation {
  readonly id = ++nextJob;
  readonly main: M3c2Prepared;
  normalFitSignature?: string;
  private readonly ready: Promise<unknown>[] = [];
  private readonly generations: (number | undefined)[] = [];
  private users = 0;
  private retired = false;
  private disposed = false;

  constructor(readonly compared: Cloud, readonly reference: Cloud, readonly poolWorkers: number) {
    this.main = new M3c2Prepared(compared, reference);
    try {
      for (let lane = 0; lane < poolWorkers; lane++) {
        // Export one pair at a time and transfer ownership immediately. No
        // JS raw inputs or retained snapshot copies accompany these trees.
        const ready = runOn(lane, { kind: "m3c2-prepare", job: this.id,
          reference: this.main.referenceSnapshot(), compared: this.main.comparedSnapshot(),
          referenceCount: reference.length, comparedCount: compared.length });
        this.ready.push(ready);
        this.generations.push(workerGeneration(lane));
        void ready.catch(() => {});
      }
    } catch (error) { this.retire(); throw error; }
  }

  live(): boolean {
    return !this.retired && this.generations.every((generation, lane) =>
      generation !== undefined && generation === workerGeneration(lane));
  }

  retire(): void {
    this.retired = true;
    if (current === this) current = undefined;
    this.release();
  }

  private release(): void {
    if (!this.retired || this.users || this.disposed) return;
    this.disposed = true;
    this.main.free();
    for (let lane = 0; lane < this.ready.length; lane++) {
      // Queued after preparation/search on the same instance. A replacement
      // has no such job; never start a worker merely to release old state.
      void releaseOn(lane, { kind: "m3c2-release", job: this.id }).catch(() => {});
    }
  }

  /** Queue all pool searches before synchronous main work, then drain both
   * preparation and computation responses before dropping any lease. */
  async compute(cores: (lane: number) => Float64Array, normalRadius: number,
    projectionRadius: number, maxDepth: number, radiusBounds: boolean,
    cacheNormals = false, maxCachedCores = 0, sharedSearch = true, fusedProjection?: boolean): Promise<M3c2PartResult[]> {
    // A fixed per-instance allowance also bounds overlapping core layouts;
    // cache replacement on one lane cannot borrow space from another lane.
    maxCachedCores = Math.min(maxCachedCores, Math.floor(64 * 1024 * 1024 / (56 * (this.poolWorkers + 1))));
    this.users++;
    try {
      const pending = Promise.allSettled(this.ready.map(async (ready, lane) => {
        const search = (async () => runOnGeneration(lane, this.generations[lane], { kind: "m3c2-prepared",
          job: this.id, core: cores(lane), normalRadius, projectionRadius, maxDepth, radiusBounds, cacheNormals, maxCachedCores, sharedSearch, fusedProjection }))();
        const outcomes = await Promise.allSettled([ready, search]);
        const failed = outcomes.find(result => result.status === "rejected");
        if (failed?.status === "rejected") throw failed.reason;
        return (outcomes[1] as PromiseFulfilledResult<M3c2PartResult>).value;
      }));
      let mainValues: M3c2PartResult | undefined, mainError: unknown, mainFailed = false;
      try {
        const distances = cacheNormals ? this.main.computeCached(cores(this.poolWorkers), normalRadius,
          projectionRadius, maxDepth, radiusBounds, maxCachedCores, sharedSearch, fusedProjection) : this.main.compute(cores(this.poolWorkers), normalRadius,
          projectionRadius, maxDepth, radiusBounds, sharedSearch, fusedProjection);
        // Capture metadata before yielding: another request may replace the cache.
        mainValues = { distances, normalCacheHit: cacheNormals ? this.main.normalCacheHit : false,
          fusedProjection: fusedProjection ?? (cacheNormals && this.main.normalCacheHit),
          normalCacheBytes: this.main.normalCacheBytes ?? 0 };
      }
      catch (error) { mainFailed = true; mainError = error; }
      const outcomes = await pending;
      const failed = outcomes.find(result => result.status === "rejected");
      if (failed?.status === "rejected") throw failed.reason;
      if (mainFailed) throw mainError;
      return [...outcomes.map(result => (result as PromiseFulfilledResult<M3c2PartResult>).value), mainValues!];
    } catch (error) { this.retire(); throw error; }
    finally { this.users--; this.release(); }
  }
}

export function prepareM3c2(compared: Cloud, reference: Cloud, poolWorkers: number): {
  preparation: M3c2Preparation; hit: boolean;
} {
  if (current?.compared === compared && current.reference === reference &&
    current.poolWorkers === poolWorkers && current.live()) return { preparation: current, hit: true };
  current?.retire();
  current = new M3c2Preparation(compared, reference, poolWorkers);
  return { preparation: current, hit: false };
}

/** Coordinates/order changed, or either source is being removed. */
export function invalidateM3c2Preparation(cloud: Cloud): void {
  if (current?.compared === cloud || current?.reference === cloud) current.retire();
}
