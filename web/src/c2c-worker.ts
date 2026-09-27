// Pool worker: distances for one slice of the compared cloud, against either
// a reference cloud (C2C) or a mesh (C2M). Each pool worker owns a separate
// WASM instance, so no SharedArrayBuffer (and no COOP/COEP headers) is needed.

import init, { meshDistances, nearestDistances } from "./wasm/ca_wasm.js";

export type Slice =
  | { kind: "cloud"; reference: Float64Array; queries: Float64Array }
  | { kind: "mesh"; vertices: Float64Array; indices: Uint32Array; queries: Float64Array; signed: boolean };

export type SliceRequest = Slice & { seq: number };

export type SliceResponse =
  | { seq: number; ok: true; distances: Float64Array }
  | { seq: number; ok: false; error: string };

const ready = init();

self.onmessage = async (event: MessageEvent<SliceRequest>) => {
  const request = event.data;
  try {
    await ready;
    const distances =
      request.kind === "cloud"
        ? nearestDistances(request.reference, request.queries)
        : meshDistances(request.vertices, request.indices, request.queries, request.signed);
    const response: SliceResponse = { seq: request.seq, ok: true, distances };
    self.postMessage(response, { transfer: [distances.buffer] });
  } catch (err) {
    const response: SliceResponse = {
      seq: request.seq,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(response);
  }
};
