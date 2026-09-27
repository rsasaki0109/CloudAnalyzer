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
    if (slice.kind === "octree") {
      transfer.push(slice.positions.buffer);
      if (slice.colors) transfer.push(slice.colors.buffer);
    }
    w.postMessage(request, { transfer });
  });
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
): Promise<SliceResult<S>[]> {
  const results: SliceResult<S>[] = new Array(count);
  let next = 0;
  const lanes = Array.from({ length: Math.min(poolSize(), count) }, async (_, lane) => {
    while (next < count) {
      const i = next++;
      results[i] = await runSlice(worker(lane), make(i));
    }
  });
  await Promise.all(lanes);
  return results;
}
