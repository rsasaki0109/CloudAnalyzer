import type { GpuC2cJob, GpuC2cReference } from "./wasm/ca_wasm.js";
import { releaseOn, runOn, runOnGeneration, workerGeneration } from "./pool";

let nextReference = 0;
let nextQueryJob = 0;

/** One job's exact lane queries, released independently of cached references. */
export class GpuVerifierQueries {
  readonly id = ++nextQueryJob;
  private readonly usedLanes = new Set<number>();
  bytes = 0;

  constructor(private readonly referenceId: number, private readonly batchSize: number,
    private readonly workers: number, private readonly poolWorkers: number,
    private readonly generation: (lane: number) => number | undefined,
    private readonly invalidate: (lane: number) => void) {}

  async prepare(job: GpuC2cJob): Promise<void> {
    const resident = this.poolWorkers < this.workers;
    const lanes = Math.min(this.workers, this.batchSize, job.length) - Number(resident);
    const results = await Promise.allSettled(Array.from({ length: lanes }, async (_, lane) => {
      const queries = job.workerQueries(lane, this.workers, this.batchSize, resident);
      this.bytes += queries.byteLength;
      this.usedLanes.add(lane);
      return runOnGeneration(lane, this.generation(lane),
        { kind: "gpu-query-prepare", referenceId: this.referenceId, queryJob: this.id, queries });
    }));
    results.forEach((r, lane) => { if (r.status === "rejected") this.invalidate(lane); });
    const failed = results.find(r => r.status === "rejected");
    if (failed?.status === "rejected") throw failed.reason;
  }

  first(first: number, lane: number): number {
    const lanes = Math.min(this.workers, this.batchSize);
    const count = Math.floor((lane + 1) * this.batchSize / lanes) - Math.floor(lane * this.batchSize / lanes);
    return Math.floor(first / this.batchSize) * count;
  }

  dispose(): void {
    for (const lane of this.usedLanes) {
      void releaseOn(lane, { kind: "gpu-query-release", queryJob: this.id })
        .then(alive => { if (!alive) this.invalidate(lane); })
        .catch(() => this.invalidate(lane));
    }
    this.usedLanes.clear();
  }
}

/** Reference-owned caches, released only after all snapshot jobs finish. */
export class GpuVerifiers {
  private readonly id = ++nextReference;
  private readonly lanes = new Map<number, Promise<unknown>>();
  private readonly generations = new Map<number, number | undefined>();
  private readonly usedLanes = new Set<number>();
  private bytes = 0;

  constructor(private readonly snapshot = true, private readonly reuseBounds = true) {}

  get referenceBytes(): number { return this.bytes; }

  queries(batchSize: number, workers: number, mainWorker = false): GpuVerifierQueries {
    return new GpuVerifierQueries(this.id, batchSize, workers, workers - Number(mainWorker),
      lane => this.generations.get(lane), lane => this.lanes.delete(lane));
  }

  reused(workers: number): boolean {
    return Array.from({ length: workers }, (_, lane) => this.lanes.has(lane) &&
      this.generations.get(lane) !== undefined && this.generations.get(lane) === workerGeneration(lane)).every(Boolean);
  }

  async prepare(index: GpuC2cReference, workers: number): Promise<void> {
    let reference: Float64Array | Uint8Array | undefined;
    const pending = Array.from({ length: workers }, (_, lane) => {
      let ready = this.lanes.get(lane);
      if (this.generations.get(lane) !== workerGeneration(lane)) ready = undefined;
      if (!ready) {
        reference ??= this.snapshot ? index.snapshot() : index.xyz();
        this.bytes = reference.byteLength;
        const snapshot = reference.slice();
        ready = (async () => runOn(lane, { kind: "gpu-verify-prepare", referenceId: this.id, reference: snapshot, reuseBounds: this.reuseBounds }))();
        this.lanes.set(lane, ready);
        this.generations.set(lane, workerGeneration(lane));
        this.usedLanes.add(lane);
      }
      return ready;
    });
    // Drain every lane before allowing a failed job to release its snapshot.
    const results = await Promise.allSettled(pending);
    results.forEach((r, lane) => { if (r.status === "rejected") this.lanes.delete(lane); });
    const failed = results.find(r => r.status === "rejected");
    if (failed?.status === "rejected") throw failed.reason;
  }

  async complete(job: GpuC2cJob, first: number, candidates: Uint32Array, workers: number,
    bounds: boolean, simd: boolean, distances: Float64Array, queries?: GpuVerifierQueries,
    mainWorker = false, sliceQueryCopy = true): Promise<number> {
    const lanes = Math.min(workers, candidates.length);
    const poolLanes = lanes - Number(mainWorker);
    // Post pool work before the resident synchronous lane so exact searches
    // overlap. Catch every lane immediately, and drain all on either failure.
    const pending = Array.from({ length: lanes }, async (_, lane) => {
      const start = Math.floor(lane * candidates.length / lanes);
      const end = Math.floor((lane + 1) * candidates.length / lanes);
      if (lane === poolLanes && mainWorker) {
        const verified = job.verify(first + start, candidates.subarray(start, end));
        try { return { distances: verified.takeDistances(), certified: verified.certified }; }
        finally { verified.free(); }
      }
      return runOnGeneration(lane, this.generations.get(lane), { kind: "gpu-verify", referenceId: this.id,
        ...(queries ? { queryJob: queries.id, first: queries.first(first, lane) } :
          { queries: job.queriesF64WithCopy(first + start, end - start, sliceQueryCopy) }),
        candidates: candidates.slice(start, end), bounds, simd });
    });
    const results = await Promise.allSettled(pending);
    results.forEach((r, lane) => { if (r.status === "rejected" && lane < poolLanes) this.lanes.delete(lane); });
    const failed = results.find(r => r.status === "rejected");
    if (failed?.status === "rejected") throw failed.reason;
    let certified = 0;
    for (let lane = 0; lane < lanes; lane++) {
      const result = results[lane];
      if (result.status !== "fulfilled") continue;
      const start = Math.floor(lane * candidates.length / lanes);
      const end = Math.floor((lane + 1) * candidates.length / lanes);
      if (result.value.distances.length !== end - start) throw new Error("Incomplete GPU verification batch");
      distances.set(result.value.distances, first + start);
      certified += result.value.certified;
    }
    return certified;
  }

  dispose(): void {
    for (const lane of this.usedLanes) {
      void releaseOn(lane, {
        kind: "gpu-verify-release", referenceId: this.id,
      }).catch(() => {});
    }
    this.lanes.clear();
    this.generations.clear();
    this.usedLanes.clear();
  }
}
