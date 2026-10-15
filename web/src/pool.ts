// A pool of WASM workers for data-parallel jobs.

import type { Slice, SliceRequest, SliceResponse, SliceResult } from "./c2c-worker";

/** Below this many queries, splitting a distance job costs more than it saves. */
export const MIN_PARALLEL_QUERIES = 100_000;
const MAX_WORKERS = 8;

const workers: (Worker | undefined)[] = [];
const generations: number[] = [];
let seq = 0;

export function poolSize(): number {
  return Math.max(1, Math.min(MAX_WORKERS, (navigator.hardwareConcurrency || 2) - 1));
}

function worker(i: number): Worker {
  if (!workers[i]) {
    const created = new Worker(new URL("./c2c-worker.ts", import.meta.url), { type: "module" });
    workers[i] = created;
    generations[i] = (generations[i] ?? 0) + 1;
    // Cached WASM state can disappear while no slice is awaiting a reply.
    const lost = () => { if (workers[i] === created) workers[i] = undefined; created.terminate(); };
    created.addEventListener("error", lost);
    created.addEventListener("messageerror", lost);
  }
  return workers[i];
}

/** Identify a live WASM instance; cached preparations never survive replacement. */
export function workerGeneration(lane: number): number | undefined {
  return workers[lane] ? generations[lane] : undefined;
}

function runSlice<S extends Slice>(w: Worker, slice: S): Promise<SliceResult<S>> {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      w.removeEventListener("message", onMessage);
      w.removeEventListener("error", onError);
      w.removeEventListener("messageerror", onError);
    };
    const onError = (event: Event) => {
      cleanup();
      const lane = workers.indexOf(w);
      if (lane >= 0) workers[lane] = undefined;
      w.terminate();
      reject(new Error(event instanceof ErrorEvent ? event.message : "Pool worker message failed"));
    };
    const onMessage = (event: MessageEvent<SliceResponse>) => {
      if (event.data.seq !== id) return;
      cleanup();
      if (event.data.ok) resolve(event.data.value as SliceResult<S>);
      else reject(new Error(event.data.error));
    };
    w.addEventListener("message", onMessage);
    w.addEventListener("error", onError);
    w.addEventListener("messageerror", onError);
    const request: SliceRequest = { ...slice, seq: id };
    // Per-slice buffers are transferred; a mesh shared by all slices is copied.
    const transfer: Transferable[] = [];
    if (slice.kind === "cloud" || slice.kind === "cloud-prepare") transfer.push(slice.queries.buffer, slice.reference.buffer);
    if (slice.kind === "gpu-verify-prepare") transfer.push(slice.reference.buffer);
    if (slice.kind === "gpu-query-prepare") transfer.push(slice.queries.buffer);
    if (slice.kind === "gpu-verify") {
      if ("queries" in slice) transfer.push(slice.queries.buffer);
      transfer.push(slice.candidates.buffer);
    }
    if (slice.kind === "mesh" || slice.kind === "mesh-prepared") transfer.push(slice.queries.buffer);
    if (slice.kind === "mesh-prepared" && slice.hints) transfer.push(slice.hints.buffer);
    // Full M3C2 inputs are shared by slices and copied; only cores transfer.
    if (slice.kind === "m3c2") transfer.push(slice.core.buffer);
    if (slice.kind === "m3c2-prepare") transfer.push(slice.reference.buffer, slice.compared.buffer);
    if (slice.kind === "m3c2-prepared") transfer.push(slice.core.buffer);
    if (slice.kind === "sor-local" || slice.kind === "normals" || slice.kind === "knn-prepare") transfer.push(slice.points.buffer);
    if ((slice.kind === "knn-local" || slice.kind === "knn-normals") && slice.gpuHints) transfer.push(slice.gpuHints.buffer);
    if (slice.kind === "sor-within" || slice.kind === "knn-within") transfer.push(slice.queries.buffer);
    if (slice.kind === "dynamic") transfer.push(slice.context.buffer);
    if (slice.kind === "loop") transfer.push(slice.from.buffer, slice.to.buffer);
    if (slice.kind === "odom-map") transfer.push(slice.placed.buffer, slice.origin.buffer);
    if (slice.kind === "odom-source") transfer.push(slice.source.buffer);
    if (slice.kind === "odom-equations") transfer.push(slice.total.buffer, slice.terms.buffer);
    if (slice.kind === "copc-nodes") transfer.push(slice.head.buffer, slice.nodes.buffer);
    if (slice.kind === "las-chunks") transfer.push(slice.head.buffer, slice.chunks.buffer);
    if (slice.kind === "bucket-chunk" || slice.kind === "bucket") {
      transfer.push(slice.positions.buffer);
      if (slice.colors) transfer.push(slice.colors.buffer);
    }
    try { w.postMessage(request, { transfer }); }
    catch (error) { cleanup(); reject(error); }
  });
}

/** Run a slice on a given worker, e.g. one holding state from an earlier slice. */
export function runOn<S extends Slice>(lane: number, slice: S): Promise<SliceResult<S>> {
  return runSlice(worker(lane), slice);
}

/** Follow prepared state on this exact live instance; never spawn a replacement. */
export function runOnGeneration<S extends Slice>(lane: number, generation: number | undefined,
  slice: S): Promise<SliceResult<S>> {
  const existing = workers[lane];
  if (!existing || generation === undefined || generations[lane] !== generation)
    return Promise.reject(new Error("CPU C2C worker preparation was lost"));
  return runSlice(existing, slice);
}

/** Destroy cached state only on a live lane; terminated WASM state is already gone. */
export async function releaseOn(lane: number,
  slice: Extract<Slice, { kind: "gpu-query-release" | "gpu-verify-release" | "cloud-release" | "m3c2-release" | "knn-release" | "mesh-release" | "mesh-order-release" }>): Promise<boolean> {
  const existing = workers[lane];
  if (!existing) return false;
  await runSlice(existing, slice);
  return true;
}

/** Start every pool worker and get its kernels optimized, while idle. */
export function warmUpPool(): Promise<unknown> {
  const n = poolSize();
  return n < 2 ? Promise.resolve() : Promise.all(Array.from({ length: n }, (_, i) => runSlice(worker(i), { kind: "warm-up" })));
}

/**
 * Run slices on the pool, each idle worker taking the next slice, so uneven
 * slices still keep every worker busy. Resolves to results in slice order.
 * `make(i)` creates slice `i` just before it is sent, so large buffers are
 * not all copied out at once.
 */
export async function runSlices<S extends Slice>(
  count: number,
  make: (i: number) => S,
  /** Receives the worker ("lane") that ran each slice, for follow-ups with `runOn`. */
  lanesUsed?: number[],
): Promise<SliceResult<S>[]> {
  const results: SliceResult<S>[] = new Array(count);
  await eachSlice(count, make, (i, r) => (results[i] = r), lanesUsed);
  return results;
}

/**
 * Like {@link runSlices}, but hands each result to `done` as it arrives
 * instead of keeping them all. The first error (also one thrown by `done`)
 * stops handing out slices.
 */
export async function eachSlice<S extends Slice>(
  count: number,
  make: (i: number) => S,
  done: (i: number, result: SliceResult<S>) => void,
  lanesUsed?: number[],
): Promise<void> {
  let next = 0;
  let failed = false;
  const lanes = Array.from({ length: Math.min(poolSize(), count) }, async (_, lane) => {
    try {
      while (!failed && next < count) {
        const i = next++;
        if (lanesUsed) lanesUsed[i] = lane;
        done(i, await runSlice(worker(lane), make(i)));
      }
    } catch (err) {
      failed = true;
      throw err;
    }
  });
  await Promise.all(lanes);
}

let nextLane = 0;

/** Run one slice on the pool, taking the workers in turn. */
export function runAny<S extends Slice>(slice: S): Promise<SliceResult<S>> {
  nextLane = (nextLane + 1) % poolSize();
  return runSlice(worker(nextLane), slice);
}
