// Pool worker: one slice of a data-parallel job. Distances for part of the
// compared cloud (against a reference cloud or a mesh), or one step of a
// parallel octree index build. Each pool worker owns a separate WASM instance, so no
// SharedArrayBuffer (and no COOP/COEP headers) is needed.

import { type ByteSource, readRange } from "./bytes";
import init, {
  bucketChunk,
  decodeCopcNodes,
  buildBucket,
  meshDistances,
  nearestDistances,
  normalsOf,
  type Reordered,
  SorPart,
  warmUp,
} from "./wasm/ca_wasm.js";

export type Slice =
  | { kind: "cloud"; reference: Float64Array; queries: Float64Array }
  | { kind: "mesh"; vertices: Float64Array; indices: Uint32Array; queries: Float64Array; signed: boolean }
  /** Index step 1: sort a slice of the cloud by bucket. */
  | { kind: "bucket-chunk"; positions: Float64Array; colors: Uint8Array | null; cube: Float64Array }
  /** Index step 2: build one bucket. */
  | { kind: "bucket"; positions: Float64Array; colors: Uint8Array | null; cube: Float64Array; key: number }
  /**
   * SOR step 1: the statistic of one part, where its own points settle it.
   * The part stays indexed on this worker under `job` for step 2.
   */
  | { kind: "sor-local"; job: number; points: Float64Array; k: number; own: number; regions: Float64Array }
  /** SOR step 2 on this worker's part of `job`: nearest squared distances from queries. */
  | { kind: "sor-within"; job: number; queries: Float64Array; k: number }
  /** Normals of one part of a cloud, from its own points. */
  | { kind: "normals"; points: Float64Array; k: number; orientation: Float64Array }
  /** Fetch and decode COPC nodes (`offset, size, points` triples). */
  | { kind: "copc-nodes"; source: ByteSource; head: Uint8Array; nodes: Float64Array }
  /** Drop this worker's part of `job`. */
  | { kind: "sor-release"; job: number }
  /** Run every kernel once so the browser optimizes them (see `warmUp`). */
  | { kind: "warm-up" };

/** Reordered points; see `Reordered` in the WASM API for `counts`/`nodes`. */
export interface ReorderedResult {
  positions: Float64Array;
  colors: Uint8Array | null;
  /** `order[i]` is the input index of the point now at `i`. */
  order: Uint32Array;
  counts: Uint32Array;
  nodes: Float64Array;
}

export interface CopcNodesResult {
  positions: Float64Array;
  /** 16-bit RGB, or null. */
  colors: Uint16Array | null;
  intensity: Float32Array;
  classification: Uint8Array;
}

export interface SorLocalResult {
  means: Float64Array;
  open: Uint32Array;
  openPoints: Float64Array;
  candidates: Float64Array;
  /** Per open point, the bit mask of the other parts to ask. */
  reach: Uint32Array;
}

/** What each slice kind produces. */
export type SliceResult<S extends Slice> = S extends { kind: "bucket-chunk" | "bucket" }
  ? ReorderedResult
  : S extends { kind: "sor-local" }
    ? SorLocalResult
    : S extends { kind: "copc-nodes" }
      ? CopcNodesResult
    : S extends { kind: "normals" }
      ? Float32Array
      : Float64Array;

export type SliceRequest = Slice & { seq: number };

export type SliceResponse =
  | { seq: number; ok: true; value: Value }
  | { seq: number; ok: false; error: string };

const ready = init();
/** SOR parts indexed in step 1, by job, until released. */
const sorParts = new Map<number, SorPart>();

function unpack(r: Reordered): { value: ReorderedResult; transfer: Transferable[] } {
  const value: ReorderedResult = {
    positions: r.positions(),
    colors: r.colors() ?? null,
    order: r.order(),
    counts: r.counts(),
    nodes: r.nodes(),
  };
  r.free();
  const transfer: Transferable[] = [value.positions.buffer, value.order.buffer, value.nodes.buffer];
  if (value.colors) transfer.push(value.colors.buffer);
  return { value, transfer };
}

type Value = Float64Array | Float32Array | ReorderedResult | SorLocalResult | CopcNodesResult;

async function run(request: SliceRequest): Promise<{ value: Value; transfer: Transferable[] }> {
  switch (request.kind) {
    case "cloud": {
      const d = nearestDistances(request.reference, request.queries);
      return { value: d, transfer: [d.buffer] };
    }
    case "mesh": {
      const d = meshDistances(request.vertices, request.indices, request.queries, request.signed);
      return { value: d, transfer: [d.buffer] };
    }
    case "bucket-chunk":
      return unpack(bucketChunk(request.positions, request.colors, request.cube));
    case "bucket":
      return unpack(buildBucket(request.positions, request.colors, request.cube, request.key));
    case "sor-local": {
      sorParts.get(request.job)?.free();
      const part = new SorPart(request.points);
      sorParts.set(request.job, part);
      const r = part.local(request.k, request.own, request.regions);
      const value: SorLocalResult = {
        means: r.means(),
        open: r.open(),
        openPoints: r.openPoints(),
        candidates: r.candidates(),
        reach: r.reach(),
      };
      r.free();
      return {
        value,
        transfer: [
          value.means.buffer,
          value.open.buffer,
          value.openPoints.buffer,
          value.candidates.buffer,
          value.reach.buffer,
        ],
      };
    }
    case "sor-within": {
      const part = sorParts.get(request.job);
      if (!part) throw new Error("SOR part is gone");
      const d = part.within(request.queries, request.k);
      return { value: d, transfer: [d.buffer] };
    }
    case "normals": {
      const n = normalsOf(request.points, request.k, request.orientation);
      return { value: n, transfer: [n.buffer] };
    }
    case "copc-nodes": {
      const count = request.nodes.length / 3;
      const offsets = Array.from({ length: count }, (_, k) => request.nodes[k * 3]);
      const sizes = Uint32Array.from({ length: count }, (_, k) => request.nodes[k * 3 + 1]);
      const counts = Uint32Array.from({ length: count }, (_, k) => request.nodes[k * 3 + 2]);
      // One request per run of nodes lying (nearly) back to back in the file.
      const runs: { start: number; end: number }[] = [];
      for (const k of offsets.map((_, k) => k).sort((a, b) => offsets[a] - offsets[b])) {
        const last = runs.at(-1);
        const end = offsets[k] + sizes[k];
        if (last && offsets[k] <= last.end + (64 << 10)) last.end = Math.max(last.end, end);
        else runs.push({ start: offsets[k], end });
      }
      const data = await Promise.all(runs.map((r) => readRange(request.source, r.start, r.end - r.start)));
      const joined = new Uint8Array(sizes.reduce((a, b) => a + b, 0));
      let at = 0;
      for (let k = 0; k < count; k++) {
        const r = runs.findIndex((run) => offsets[k] >= run.start && offsets[k] + sizes[k] <= run.end);
        const from = offsets[k] - runs[r].start;
        joined.set(data[r].subarray(from, from + sizes[k]), at);
        at += sizes[k];
      }
      const decoded = decodeCopcNodes(request.head, joined, sizes, counts);
      const value: CopcNodesResult = {
        positions: decoded.positions(),
        colors: decoded.colors() ?? null,
        intensity: decoded.intensity(),
        classification: decoded.classification(),
      };
      decoded.free();
      const transfer: Transferable[] = [value.positions.buffer, value.intensity.buffer, value.classification.buffer];
      if (value.colors) transfer.push(value.colors.buffer);
      return { value, transfer };
    }
    case "sor-release":
      sorParts.get(request.job)?.free();
      sorParts.delete(request.job);
      return { value: new Float64Array(0), transfer: [] };
    case "warm-up":
      warmUp();
      return { value: new Float64Array(0), transfer: [] };
  }
}

self.onmessage = async (event: MessageEvent<SliceRequest>) => {
  const request = event.data;
  try {
    await ready;
    const { value, transfer } = await run(request);
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
