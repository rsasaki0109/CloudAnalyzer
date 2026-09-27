// A pool of WASM workers for data-parallel nearest-neighbour queries.

import type { Slice, SliceRequest, SliceResponse } from "./c2c-worker";

/** Below this many queries, splitting the job costs more than it saves. */
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

function runSlice(w: Worker, slice: Slice): Promise<Float64Array> {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    const onMessage = (event: MessageEvent<SliceResponse>) => {
      if (event.data.seq !== id) return;
      w.removeEventListener("message", onMessage);
      if (event.data.ok) resolve(event.data.distances);
      else reject(new Error(event.data.error));
    };
    w.addEventListener("message", onMessage);
    const request: SliceRequest = { ...slice, seq: id };
    // Per-slice buffers are transferred; a mesh shared by all slices is copied.
    const transfer: Transferable[] = [slice.queries.buffer];
    if (slice.kind === "cloud") transfer.push(slice.reference.buffer);
    w.postMessage(request, { transfer });
  });
}

/** Run each slice on its own pool worker; resolves to per-slice distances. */
export function runSlices(slices: Slice[]): Promise<Float64Array[]> {
  return Promise.all(slices.map((s, i) => runSlice(worker(i % poolSize()), s)));
}
