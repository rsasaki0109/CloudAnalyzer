// Pool worker: nearest-neighbour distances for one slice of the compared cloud.
// Each pool worker owns a separate WASM instance, so no SharedArrayBuffer
// (and no COOP/COEP headers) is needed.

import init, { nearestDistances } from "./wasm/ca_wasm.js";

export interface SliceRequest {
  seq: number;
  reference: Float64Array;
  queries: Float64Array;
}

export type SliceResponse =
  | { seq: number; ok: true; distances: Float64Array }
  | { seq: number; ok: false; error: string };

const ready = init();

self.onmessage = async (event: MessageEvent<SliceRequest>) => {
  const { seq, reference, queries } = event.data;
  try {
    await ready;
    const distances = nearestDistances(reference, queries);
    const response: SliceResponse = { seq, ok: true, distances };
    self.postMessage(response, { transfer: [distances.buffer] });
  } catch (err) {
    const response: SliceResponse = {
      seq,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(response);
  }
};
