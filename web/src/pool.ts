// A pool of WASM workers for data-parallel jobs.

import type { Slice, SliceRequest, SliceResponse, SliceResult } from "./c2c-worker";

/** Below this many queries, splitting a distance job costs more than it saves. */
export const MIN_PARALLEL_QUERIES = 100_000;
const MAX_WORKERS = 8;

const workers: Worker[] = [];
let seq = 0;

export function poolSize(): number {
  return Math.max(1, Math.min(MAX_WORKERS, (navigator.hardwareConcurrency || 2) - 1));
}

function worker(i: number): Worker {
  workers[i] ??= new Worker(new URL("./c2c-worker.ts", import.meta.url), { type: "module" });
  return workers[i];
}

function runSlice<S extends Slice>(w: Worker, slice: S): Promise<SliceResult<S>> {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    const onMessage = (event: MessageEvent<SliceResponse>) => {
      if (event.data.seq !== id) return;
      w.removeEventListener("message", onMessage);
      if (event.data.ok) resolve(event.data.value as SliceResult<S>);
      else reject(new Error(event.data.error));
    };
    w.addEventListener("message", onMessage);
    const request: SliceRequest = { ...slice, seq: id };
    // Per-slice buffers are transferred; a mesh shared by all slices is copied.
    const transfer: Transferable[] = [];
    if (slice.kind === "cloud") transfer.push(slice.queries.buffer, slice.reference.buffer);
    if (slice.kind === "mesh") transfer.push(slice.queries.buffer);
    if (slice.kind === "sor-local" || slice.kind === "normals") transfer.push(slice.points.buffer);
    if (slice.kind === "sor-within") transfer.push(slice.queries.buffer);
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
    w.postMessage(request, { transfer });
  });
}

/** Run a slice on a given worker, e.g. one holding state from an earlier slice. */
export function runOn<S extends Slice>(lane: number, slice: S): Promise<SliceResult<S>> {
  return runSlice(worker(lane), slice);
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
