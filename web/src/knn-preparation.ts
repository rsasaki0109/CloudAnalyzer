import { planSor, type Cloud } from "./wasm/ca_wasm.js";
import { releaseOn, runOn, runOnGeneration, workerGeneration } from "./pool";
import type { Slice, SliceResult } from "./c2c-worker";
import { GpuKnnBlocks, knnGpuDevice } from "./gpu-knn-blocks";
import { GpuKnnCandidates } from "./gpu-knn-candidates";

const BUDGET = 128 * 1024 * 1024;
let nextJob = 0;
let current: KnnPreparation | undefined;
const active = new Set<KnnPreparation>();

async function drain<T>(jobs: Promise<T>[]): Promise<T[]> {
  const results = await Promise.allSettled(jobs);
  const failed = results.find(r => r.status === "rejected");
  if (failed?.status === "rejected") throw failed.reason;
  return results.map(r => (r as PromiseFulfilledResult<T>).value);
}

/** One immutable partition/tree group, protected until every active job drains. */
class KnnPreparation {
  readonly id = ++nextJob;
  readonly indices: Uint32Array[];
  readonly regions: Float64Array;
  readonly ready: Promise<void>;
  private readonly generations: (number | undefined)[] = [];
  private users = 0;
  private retired = false;
  private disposed = false;
  private changed = false;
  private baseBytes = 0;
  private hintsReady?: Promise<void>;
  private gpuReady?: Promise<GpuKnnBlocks[]>;
  private gpuParts?: GpuKnnBlocks[];
  private slotReady?: Promise<GpuKnnCandidates[]>;
  private slotParts?: GpuKnnCandidates[];
  pending = true;
  bytes = 0;
  reuses = 0;

  constructor(readonly source: Cloud, readonly points: number, readonly requestedParts: number) {
    const plan = planSor(source, requestedParts);
    const preparing: Promise<number>[] = [];
    active.add(this);
    try {
      this.indices = Array.from({ length: plan.length }, (_, j) => plan.indices(j));
      this.regions = plan.regions();
      this.bytes = this.baseBytes = this.regions.byteLength + this.indices.reduce((n, v) => n + v.byteLength, 0);
      // Capture every part before the first yield. All worker-owned coordinates
      // survive source edits/removal; no raw JS point copies are retained.
      for (let lane = 0; lane < plan.length; lane++) {
        const job = runOn(lane, { kind: "knn-prepare", job: this.id, points: plan.points(source, lane) });
        preparing.push(job);
        this.generations.push(workerGeneration(lane));
        void job.catch(() => {});
      }
      this.ready = drain(preparing).then(bytes => {
        if (bytes.some(n => !Number.isSafeInteger(n) || n < 0)) throw new Error("Invalid neighbourhood preparation size");
        this.bytes += bytes.reduce((a, b) => a + b, 0);
        this.pending = false;
        if (this.bytes > BUDGET) this.retire();
      }).catch(error => { this.retire(); throw error; });
      void this.ready.catch(() => {});
    } catch (error) { this.retire(); throw error; }
    finally { plan.free(); }
  }

  live(): boolean {
    return !this.retired && this.generations.every((g, lane) => g !== undefined && g === workerGeneration(lane));
  }

  assertSource(): void {
    if (this.changed) throw new Error("Point coordinates changed during neighbourhood calculation");
  }

  sourceValid(): boolean { return !this.changed; }

  invalidate(): void { this.changed = true; this.retire(); }

  retire(): void {
    this.retired = true;
    if (current === this) current = undefined;
    this.dispose();
  }

  private dispose(): void {
    if (!this.retired || this.users || this.disposed) return;
    this.disposed = true;
    active.delete(this);
    for (const part of this.gpuParts ?? []) part.dispose();
    for (const part of this.slotParts ?? []) part.dispose();
    for (let lane = 0; lane < this.generations.length; lane++) {
      if (this.generations[lane] !== workerGeneration(lane)) continue;
      void releaseOn(lane, { kind: "knn-release", job: this.id }).catch(() => {});
    }
  }

  async prepareHints(): Promise<void> {
    await this.ready; this.assertSource();
    if (!this.hintsReady) {
      this.hintsReady = drain(this.generations.map((g, lane) => runOnGeneration(lane, g, { kind: "knn-hint-prepare", job: this.id }))).then(bytes => {
        if (bytes.some(n => !Number.isSafeInteger(n) || n < 0)) throw new Error("Invalid hint preparation size");
        this.bytes = this.baseBytes + bytes.reduce((a, b) => a + b, 0);
        if (this.bytes > BUDGET) this.retire();
      }).catch(error => { this.retire(); throw error; });
      void this.hintsReady.catch(() => {});
    }
    await this.hintsReady; this.assertSource();
  }

  async gpuHints(k: number, slots = false): Promise<KnnGpuHints> {
    const start = performance.now(), reused = !!(slots ? this.slotParts : this.gpuParts);
    await this.prepareHints();
    let ready: Promise<(GpuKnnBlocks | GpuKnnCandidates)[]> | undefined = slots ? this.slotReady : this.gpuReady;
    if (!ready) {
      ready = (async () => {
        const runtime = await knnGpuDevice(); this.assertSource();
        const snapshots = await drain(this.generations.map((g, lane) => runOnGeneration(lane, g, { kind: "knn-gpu-buffers", job: this.id, slots })));
        const created: (GpuKnnBlocks | GpuKnnCandidates)[] = [];
        try {
          const parts = await drain(snapshots.map(async (s, lane) => {
            if (s.queries.length / 4 !== this.indices[lane].length) throw new Error("GPU neighbourhood query count changed");
            const part = slots ? await GpuKnnCandidates.create(runtime, s.nodes, s.queries, s.points!) :
              await GpuKnnBlocks.create(runtime, s.nodes, s.queries); created.push(part); return part;
          }));
          if (slots) this.slotParts = parts as GpuKnnCandidates[];
          else this.gpuParts = parts as GpuKnnBlocks[];
          return parts;
        } catch (error) { for (const part of created) part.dispose(); throw error; }
      })().catch(error => { this.retire(); throw error; });
      if (slots) this.slotReady = ready as Promise<GpuKnnCandidates[]>;
      else this.gpuReady = ready as Promise<GpuKnnBlocks[]>;
      void ready.catch(() => {});
    }
    const parts = await ready; this.assertSource();
    const prepareMillis = performance.now() - start, gpuStart = performance.now();
    try {
      const rows = await drain(parts.map(part => part.find(k))); this.assertSource();
      return { hints: rows.map(r => r.hints), prepareMillis, gpuMillis: performance.now() - gpuStart,
        batches: rows.reduce((n, r) => n + r.batches, 0), referenceReused: reused, gpuBytes: parts.reduce((n, p) => n + p.bytes, 0),
        completeRows: slots ? rows.reduce((n, r) => n + ("completeRows" in r && typeof r.completeRows === "number" ? r.completeRows : 0), 0) : undefined };
    } catch (error) { this.retire(); throw error; }
  }

  acquire(hit: boolean): KnnLease {
    this.users++;
    if (hit) this.reuses++;
    let released = false;
    return {
      indices: this.indices, regions: this.regions,
      sourceValid: () => this.sourceValid(), assertSource: () => this.assertSource(),
      retire: () => this.retire(),
      prepareHints: () => this.prepareHints(), gpuHints: (k, slots) => this.gpuHints(k, slots),
      run: async make => {
        await this.ready;
        this.assertSource();
        try {
          return await drain(this.generations.map((g, lane) => runOnGeneration(lane, g, make(this.id, lane))));
        } catch (error) { this.retire(); throw error; }
      },
      release: () => { if (!released) { released = true; this.users--; this.dispose(); } },
    };
  }
}

type PreparedSlice = Extract<Slice, { kind: "knn-local" | "knn-within" | "knn-normals" }>;
export interface KnnLease {
  indices: Uint32Array[];
  regions: Float64Array;
  sourceValid(): boolean;
  assertSource(): void;
  retire(): void;
  prepareHints(): Promise<void>;
  gpuHints(k: number, slots?: boolean): Promise<KnnGpuHints>;
  run<S extends PreparedSlice>(make: (job: number, lane: number) => S): Promise<SliceResult<S>[]>;
  release(): void;
}

export interface KnnGpuHints {
  hints: Uint32Array[]; prepareMillis: number; gpuMillis: number; batches: number;
  referenceReused: boolean; gpuBytes: number;
  completeRows?: number;
}

/** Conservative eligibility before allocating; actual retained capacities checked after preparation. */
export function acquireKnn(source: Cloud, parts: number, cache = true): KnnLease | undefined {
  const points = source.length;
  if (!cache || 100 * points + 1024 * 1024 > BUDGET) { current?.retire(); return undefined; }
  if (current?.source === source && current.requestedParts === parts && current.live()) return current.acquire(true);
  current?.retire();
  const preparation = new KnnPreparation(source, points, parts);
  current = preparation;
  return preparation.acquire(false);
}

/** Invalidate leased groups too, including a source whose idle entry was replaced. */
export function invalidateKnnPreparation(source: Cloud): void {
  for (const preparation of active) if (preparation.source === source) preparation.invalidate();
}

/** Internal diagnostics for repeatability and budget/lifecycle tests. */
export function knnCacheState(): { job: number; points: number; parts: number; bytes: number; pending: boolean; reuses: number } | null {
  if (!current?.live()) return null;
  return { job: current.id, points: current.points, parts: current.indices.length, bytes: current.bytes, pending: current.pending, reuses: current.reuses };
}
