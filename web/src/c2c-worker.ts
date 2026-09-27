// Pool worker: one slice of a data-parallel job. Distances for part of the
// compared cloud (against a reference cloud or a mesh), or one subtree of an
// octree index build. Each pool worker owns a separate WASM instance, so no
// SharedArrayBuffer (and no COOP/COEP headers) is needed.

import init, { buildSubtree, meshDistances, nearestDistances } from "./wasm/ca_wasm.js";

export type Slice =
  | { kind: "cloud"; reference: Float64Array; queries: Float64Array }
  | { kind: "mesh"; vertices: Float64Array; indices: Uint32Array; queries: Float64Array; signed: boolean }
  | { kind: "octree"; positions: Float64Array; colors: Uint8Array | null; job: Float64Array };

export interface SubtreeResult {
  positions: Float64Array;
  colors: Uint8Array | null;
  nodes: Float64Array;
  /** Permutation applied to the slice, for reordering attributes. */
  order: Uint32Array;
}

/** What each slice kind produces. */
export type SliceResult<S extends Slice> = S extends { kind: "octree" } ? SubtreeResult : Float64Array;

export type SliceRequest = Slice & { seq: number };

export type SliceResponse =
  | { seq: number; ok: true; value: Float64Array | SubtreeResult }
  | { seq: number; ok: false; error: string };

const ready = init();

function run(request: SliceRequest): { value: Float64Array | SubtreeResult; transfer: Transferable[] } {
  switch (request.kind) {
    case "cloud": {
      const d = nearestDistances(request.reference, request.queries);
      return { value: d, transfer: [d.buffer] };
    }
    case "mesh": {
      const d = meshDistances(request.vertices, request.indices, request.queries, request.signed);
      return { value: d, transfer: [d.buffer] };
    }
    case "octree": {
      const subtree = buildSubtree(request.positions, request.colors, request.job);
      const value: SubtreeResult = {
        positions: subtree.positions(),
        colors: subtree.colors() ?? null,
        nodes: subtree.nodes(),
        order: subtree.order(),
      };
      subtree.free();
      const transfer: Transferable[] = [value.positions.buffer, value.nodes.buffer, value.order.buffer];
      if (value.colors) transfer.push(value.colors.buffer);
      return { value, transfer };
    }
  }
}

self.onmessage = async (event: MessageEvent<SliceRequest>) => {
  const request = event.data;
  try {
    await ready;
    const { value, transfer } = run(request);
    const response: SliceResponse = { seq: request.seq, ok: true, value };
    self.postMessage(response, { transfer });
  } catch (err) {
    const response: SliceResponse = {
      seq: request.seq,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(response);
  }
};
