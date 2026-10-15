import type { Cloud, Mesh } from "./wasm/ca_wasm.js";
import { releaseOn, runOn, runOnGeneration, workerGeneration } from "./pool";
import { GpuMeshReference, meshGpuDevice } from "./gpu-mesh-candidates";

const BUDGET = 1024 * 1024 * 1024;
let nextJob = 0;
let current: MeshPreparation | undefined;
const active = new Set<MeshPreparation>();
const jobs = new Set<{ mesh: Mesh; cloud: Cloud; changed: boolean }>();

function releaseQueryOrders(): void {
  // The pool has at most eight workers. Never start a lane to release scratch.
  for (let lane = 0; lane < 8; lane++) if (workerGeneration(lane) !== undefined)
    void releaseOn(lane, { kind: "mesh-order-release" }).catch(() => {});
}

async function drain<T>(pending: Promise<T>[]): Promise<T[]> {
  const results = await Promise.allSettled(pending);
  const failed = results.find(r => r.status === "rejected");
  if (failed?.status === "rejected") throw failed.reason;
  return results.map(r => (r as PromiseFulfilledResult<T>).value);
}

/** Guard query/source identities while immutable pool snapshots are in flight. */
export function trackMeshSources(mesh: Mesh, cloud: Cloud) {
  const job = { mesh, cloud, changed: false }; jobs.add(job);
  return {
    assertSource: () => { if (job.changed) throw new Error("C2M source changed during calculation"); },
    sourceValid: () => !job.changed,
    release: () => { jobs.delete(job); if (job.changed) releaseQueryOrders(); },
  };
}

/** One idle mesh group, with bounded reservations across overlapping leases. */
class MeshPreparation {
  readonly id = ++nextJob;
  readonly ready: Promise<void>;
  private generations: (number | undefined)[] = [];
  private users = 0;
  private retired = false;
  private disposed = false;
  private changed = false;
  private gpuReady?: Promise<GpuMeshReference>;
  private gpu?: GpuMeshReference;
  private gpuTail: Promise<void> = Promise.resolve();
  pending = true;
  bytes = 0;
  reuses = 0;

  constructor(readonly mesh: Mesh, readonly workers: number, readonly reservation: number) {
    const pending: Promise<number>[] = [];
    active.add(this);
    try {
      const vertices = mesh.rawVertices(), indices = mesh.indices();
      for (let lane = 0; lane < workers; lane++) {
        const ready = runOn(lane, { kind: "mesh-prepare", job: this.id, vertices, indices });
        pending.push(ready); this.generations.push(workerGeneration(lane)); void ready.catch(() => {});
      }
      this.ready = drain(pending).then(bytes => {
        if (bytes.some(n => !Number.isSafeInteger(n) || n < 0)) throw new Error("Invalid retained mesh size");
        this.bytes = bytes.reduce((a, b) => a + b, 0);
        this.pending = false;
        if (this.bytes > this.reservation) throw new Error("Retained mesh exceeds reserved memory");
      }).catch(error => { this.retire(); throw error; });
      void this.ready.catch(() => {});
    } catch (error) { this.retire(); throw error; }
  }

  live(): boolean {
    return !this.retired && this.generations.every((g, lane) => g !== undefined && g === workerGeneration(lane));
  }

  invalidate(): void { this.changed = true; this.retire(); }
  retire(): void { this.retired = true; if (current === this) current = undefined; this.dispose(); }

  private dispose(): void {
    if (!this.retired || this.users || this.disposed) return;
    this.disposed = true; active.delete(this);
    this.gpu?.dispose();
    for (let lane = 0; lane < this.generations.length; lane++) {
      if (this.generations[lane] !== workerGeneration(lane)) continue;
      void releaseOn(lane, { kind: "mesh-release", job: this.id }).catch(() => {});
    }
  }

  private gpuHints(points: Float64Array): ReturnType<MeshLease["gpuHints"]> {
    // Serialize the complete find/cleanup, not just individual device batches.
    // An overlapping failure must not destroy a reference still being read.
    this.users++;
    const pending = this.gpuTail.then(async () => {
      const start = performance.now(), reused = !!this.gpu;
      await this.ready;
      if (this.changed) throw new Error("C2M source changed during calculation");
      try {
        if (!this.gpuReady) {
          this.gpuReady = (async () => {
            const runtime = await meshGpuDevice();
            const s = await runOnGeneration(0, this.generations[0], { kind: "mesh-gpu-buffers", job: this.id });
            this.gpu = await GpuMeshReference.create(runtime, s.origin, s.nodes, s.triangles);
            return this.gpu;
          })();
          void this.gpuReady.catch(() => {});
        }
        const ref = await this.gpuReady;
        if (this.changed) throw new Error("C2M source changed during calculation");
        const prepareMillis = performance.now() - start, gpuStart = performance.now();
        const result = await ref.find(points);
        if (this.changed) throw new Error("C2M source changed during calculation");
        return { ...result, prepareMillis, gpuMillis: performance.now() - gpuStart, referenceReused: reused, gpuBytes: ref.bytes };
      } catch (error) {
        this.gpu?.dispose(); this.gpu = undefined; this.gpuReady = undefined; throw error;
      }
    }).finally(() => { this.users--; this.dispose(); });
    this.gpuTail = pending.then(() => {}, () => {});
    return pending;
  }

  acquire(hit: boolean): MeshLease {
    this.users++; if (hit) this.reuses++;
    let released = false;
    let orderState = { hit: false, bytes: 0, entries: 0 };
    return {
      workers: this.workers, reused: hit, bytes: () => this.bytes,
      retire: () => this.retire(),
      gpuHints: points => this.gpuHints(points),
      queryOrderState: () => orderState,
      run: async (count, queries, signed, hints, orderCache = true) => {
        await this.ready;
        if (this.changed) throw new Error("C2M source changed during calculation");
        try {
          // Keep the original logical slices and reset each slice's warm guess.
          // Fewer resident BVHs must not change signed equal-distance ties.
          const results = await drain(Array.from({ length: count }, (_, part) => {
            const lane = part % this.workers;
            return runOnGeneration(lane, this.generations[lane], { kind: "mesh-prepared", job: this.id, queries: queries(part), signed, hints:hints?.(part), orderCache });
          }));
          if (this.changed) throw new Error("C2M source changed during calculation");
          const bytes = new Array(this.workers).fill(0), entries = new Array(this.workers).fill(0);
          results.forEach((result, part) => {
            if (!Number.isSafeInteger(result.queryOrderBytes) || result.queryOrderBytes < 0 || result.queryOrderBytes > 8 * 1024 * 1024 ||
              !Number.isSafeInteger(result.queryOrderEntries) || result.queryOrderEntries < 0 || result.queryOrderEntries > 8)
              throw new Error("Invalid C2M query-order allowance");
            const lane = part % this.workers;
            bytes[lane] = Math.max(bytes[lane], result.queryOrderBytes);
            entries[lane] = Math.max(entries[lane], result.queryOrderEntries);
          });
          orderState = { hit: results.every(result => result.queryOrderHit), bytes: bytes.reduce((a, b) => a + b, 0), entries: entries.reduce((a, b) => a + b, 0) };
          return results.map(result => result.distances);
        } catch (error) { releaseQueryOrders(); this.retire(); throw error; }
      },
      release: () => { if (!released) { released = true; this.users--; this.dispose(); } },
    };
  }
}

export interface MeshLease {
  workers: number;
  reused: boolean;
  bytes(): number;
  retire(): void;
  gpuHints(points: Float64Array): Promise<{ hints: Uint32Array; batches: number; completeRows: number; prepareMillis: number; gpuMillis: number; referenceReused: boolean; gpuBytes: number }>;
  queryOrderState(): { hit: boolean; bytes: number; entries: number };
  run(count: number, queries: (part: number) => Float64Array, signed: boolean, hints?: (part: number) => Uint32Array, orderCache?: boolean): Promise<Float64Array[]>;
  release(): void;
}

/** 56-byte nodes may grow to t+2 capacity; corners/order/triangles use 88t.
 * WASM input vectors have exact lengths. Actual retained capacities checked. */
export function acquireMesh(mesh: Mesh, parts: number, cache = true): MeshLease | undefined {
  if (!cache) { current?.retire(); return; }
  const estimate = 24 * mesh.vertexCount + 144 * mesh.triangleCount + 4096;
  const desired = Math.min(parts, Math.floor(BUDGET / estimate));
  if (desired < 1 || !Number.isSafeInteger(estimate)) { current?.retire(); return; }
  if (current?.mesh === mesh && current.workers === desired && current.live()) return current.acquire(true);
  current?.retire();
  const available = BUDGET - [...active].reduce((n, p) => n + p.reservation, 0);
  const workers = Math.min(desired, Math.floor(available / estimate));
  if (workers < 1) return;
  current = new MeshPreparation(mesh, workers, workers * estimate);
  return current.acquire(false);
}

/** Query edits reject active jobs; removing a mesh also retires leased BVHs. */
export function invalidateMeshSources(source: Cloud | Mesh): void {
  for (const job of jobs) if (job.mesh === source || job.cloud === source) job.changed = true;
  for (const preparation of active) if (preparation.mesh === source) preparation.invalidate();
  releaseQueryOrders();
}

/** Internal diagnostics, not an estimate of process/peak memory. */
export function meshCacheState() {
  if (!current?.live()) return null;
  return { job: current.id, workers: current.workers, bytes: current.bytes, pending: current.pending, reuses: current.reuses,
    reservation: current.reservation, reservedTotal: [...active].reduce((n, p) => n + p.reservation, 0), budget: BUDGET };
}
