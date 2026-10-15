// Pool worker: one slice of a data-parallel job. Distances for part of the
// compared cloud (against a reference cloud or a mesh), or one step of a
// parallel octree index build. Each pool worker owns a separate WASM instance, so no
// SharedArrayBuffer (and no COOP/COEP headers) is needed.

import { type ByteSource, MAX_RANGE_BYTES, readRange } from "./bytes";
import init, {
  bucketChunk,
  C2cPrepared,
  GpuC2cReference,
  GpuC2cVerifier,
  OdometryMap,
  decodeCopcNodes,
  decodeLasChunks,
  buildBucket,
  dynamicScans,
  meshDistances,
  MeshPrepared,
  MeshQueryOrder,
  m3c2Distances,
  M3c2Prepared,
  nearestDistances,
  normalsOf,
  type Reordered,
  registerScans,
  SorPart,
  warmUp,
} from "./wasm/ca_wasm.js";

export type Slice =
  | { kind: "cloud"; reference: Float64Array; queries: Float64Array; simd?: boolean }
  | { kind: "cloud-prepare"; job: number; reference: Float64Array; queries: Float64Array }
  | { kind: "cloud-prepared"; job: number; simd?: boolean }
  | { kind: "cloud-release"; job: number }
  | { kind: "gpu-verify-prepare"; referenceId: number; reference: Float64Array | Uint8Array; reuseBounds?: boolean }
  | ({ kind: "gpu-verify"; referenceId: number; candidates: Uint32Array; bounds: boolean; simd: boolean } &
      ({ queries: Float64Array } | { queryJob: number; first: number }))
  | { kind: "gpu-query-prepare"; referenceId: number; queryJob: number; queries: Float64Array }
  | { kind: "gpu-query-release"; queryJob: number }
  | { kind: "gpu-verify-release"; referenceId: number }
  | { kind: "mesh"; vertices: Float64Array; indices: Uint32Array; queries: Float64Array; signed: boolean }
  | { kind: "mesh-prepare"; job: number; vertices: Float64Array; indices: Uint32Array }
  | { kind: "mesh-prepared"; job: number; queries: Float64Array; signed: boolean; hints?: Uint32Array; orderCache?: boolean }
  | { kind: "mesh-order-release" }
  | { kind: "mesh-gpu-buffers"; job: number }
  | { kind: "mesh-release"; job: number }
  /** Core points only are split; both full inputs retain their original order. */
  | { kind: "m3c2"; reference: Float64Array; compared: Float64Array; core: Float64Array;
      normalRadius: number; projectionRadius: number; maxDepth: number; radiusBounds?: boolean; sharedSearch?: boolean; fusedProjection?: boolean }
  | { kind: "m3c2-prepare"; job: number; reference: Uint8Array; compared: Uint8Array;
      referenceCount: number; comparedCount: number }
  | { kind: "m3c2-prepared"; job: number; core: Float64Array;
      normalRadius: number; projectionRadius: number; maxDepth: number; radiusBounds?: boolean; cacheNormals?: boolean; maxCachedCores?: number; sharedSearch?: boolean; fusedProjection?: boolean }
  | { kind: "m3c2-release"; job: number }
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
  /** Immutable part shared by repeated normal/SOR calculations. */
  | { kind: "knn-prepare"; job: number; points: Float64Array }
  | { kind: "knn-hint-prepare"; job: number }
  | { kind: "knn-gpu-buffers"; job: number; slots?: boolean }
  | { kind: "knn-local"; job: number; k: number; own: number; regions: Float64Array; bounds?: boolean; gpuHints?: Uint32Array; cpuHints?: boolean; hintSlots?: boolean }
  | { kind: "knn-within"; job: number; queries: Float64Array; k: number; bounds?: boolean }
  | { kind: "knn-normals"; job: number; k: number; orientation: Float64Array; bounds?: boolean; gpuHints?: Uint32Array; cpuHints?: boolean; hintSlots?: boolean }
  | { kind: "knn-release"; job: number }
  /** Normals of one part of a cloud, from its own points. */
  | { kind: "normals"; points: Float64Array; k: number; orientation: Float64Array }
  /** Fetch and decode COPC nodes (`offset, size, points` triples). */
  | { kind: "copc-nodes"; source: ByteSource; head: Uint8Array; nodes: Float64Array }
  /**
   * Decode consecutive LAS/LAZ chunks (`offset, size, count, first` each),
   * keeping every `keepEvery`-th point of the file; `head` is the file's
   * start up to its point data.
   */
  | { kind: "las-chunks"; source: ByteSource; head: Uint8Array; chunks: Float64Array; keepEvery: number }
  /** Drop this worker's part of `job`. */
  | { kind: "sor-release"; job: number }
  /**
   * Register a candidate loop's scans (interleaved xyz, each in its node's
   * frame) from `guess`, as `PoseGraphSession.registerLoop` does.
   */
  | {
      kind: "loop";
      from: Float64Array;
      to: Float64Array;
      guess: number[];
      maxIterations: number;
      overlap: number;
      inlierDistance: number;
      retryBelow: number;
      retryHeadings: number;
    }
  /**
   * Find the dynamic points of `count` scans from `first` of `context`
   * (`PoseGraphSession.dynamicContext`, holding `window` scans more on either side).
   */
  | { kind: "dynamic"; context: Float64Array; first: number; count: number; window: number; margin: number; votes: number }
  /** Run every kernel once so the browser optimizes them (see `warmUp`). */
  | { kind: "warm-up" }
  /** Odometry on the pool: this worker keeps a copy of the local map. */
  | { kind: "odom-reset"; voxel: number; maxPoints: number }
  /** Points the last scan added to the map, and where the sensor is now. */
  | { kind: "odom-map"; placed: Float64Array; origin: Float64Array; range: number }
  /** This worker's share of a scan's source points, for the steps to come. */
  | { kind: "odom-source"; source: Float64Array }
  /** The normal equations of one step over that share: 42 numbers. */
  | { kind: "odom-equations"; total: Float64Array; terms: Float64Array };

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

export interface LasChunksResult extends CopcNodesResult {
  /** `min, max` corners per chunk, over all of its points. */
  bounds: Float64Array;
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
  : S extends { kind: "m3c2-prepared" }
    ? M3c2PartResult
  : S extends { kind: "mesh-prepared" }
    ? MeshPartResult
  : S extends { kind: "gpu-verify" }
    ? GpuVerifyResult
  : S extends { kind: "knn-prepare" | "knn-hint-prepare" | "mesh-prepare" }
    ? number
  : S extends { kind: "knn-gpu-buffers" }
    ? { nodes: Uint32Array; queries: Float32Array; points?: Float32Array }
  : S extends { kind: "mesh-gpu-buffers" }
    ? MeshGpuSnapshot
  : S extends { kind: "sor-local" | "knn-local" }
    ? SorLocalResult
    : S extends { kind: "copc-nodes" }
      ? CopcNodesResult
    : S extends { kind: "las-chunks" }
      ? LasChunksResult
    : S extends { kind: "normals" | "knn-normals" }
      ? Float32Array
    : S extends { kind: "dynamic" }
      ? Uint8Array
      : Float64Array;

export type SliceRequest = Slice & { seq: number };

export type SliceResponse =
  | { seq: number; ok: true; value: Value; memory: number }
  | { seq: number; ok: false; error: string; memory: number };

const ready = init();
/** SOR parts indexed in step 1, by job, until released. */
const sorParts = new Map<number, SorPart>();
// One idle GPU reference plus any snapshots still leased by running jobs.
const gpuReferences = new Map<number, GpuC2cReference>();
const gpuQueries = new Map<number, GpuC2cVerifier>();
const cloudJobs = new Map<number, C2cPrepared>();
const m3c2Jobs = new Map<number, M3c2Prepared>();
// Whole C2M gains were small/mixed, with repeated regressions on one source.
const USE_MESH_QUERY_ORDER = false;
let meshQueryOrder: MeshQueryOrder | undefined;
export interface MeshPartResult { distances: Float64Array; queryOrderHit: boolean; queryOrderBytes: number; queryOrderEntries: number }
export interface GpuVerifyResult { distances: Float64Array; certified: number }
export interface M3c2PartResult { distances: Float64Array; normalCacheHit: boolean; normalCacheBytes: number; fusedProjection: boolean }
/** This worker's copy of the odometry's local map, and its share of the scan being registered. */
let odometryMap: OdometryMap | null = null;
let odometrySource: Float64Array = new Float64Array(0);

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

const knnParts = new Map<number, SorPart>();
const meshParts = new Map<number, MeshPrepared>();

function sorLocalResult(part: SorPart, k: number, own: number, regions: Float64Array, bounds?: boolean,
  gpuHints?: Uint32Array, cpuHints = false, hintSlots = false): { value: SorLocalResult; transfer: Transferable[] } {
  const r = gpuHints ? hintSlots ? part.localCandidates(k, own, regions, gpuHints) : part.localHinted(k, own, regions, gpuHints) : cpuHints ? part.localCpuHints(k, own, regions) :
    bounds === undefined ? part.local(k, own, regions) : part.localWithBounds(k, own, regions, bounds);
  try {
    const value: SorLocalResult = { means: r.means(), open: r.open(), openPoints: r.openPoints(), candidates: r.candidates(), reach: r.reach() };
    return { value, transfer: [value.means.buffer, value.open.buffer, value.openPoints.buffer, value.candidates.buffer, value.reach.buffer] };
  } finally { r.free(); }
}

export interface MeshGpuSnapshot { origin: Float64Array; nodes: Uint32Array; triangles: Float32Array }
type Value = number | Float64Array | Float32Array | Uint8Array | ReorderedResult | SorLocalResult | CopcNodesResult | LasChunksResult | GpuVerifyResult | M3c2PartResult | MeshPartResult | MeshGpuSnapshot | { nodes: Uint32Array; queries: Float32Array; points?: Float32Array };

async function run(request: SliceRequest): Promise<{ value: Value; transfer: Transferable[] }> {
  switch (request.kind) {
    case "cloud-prepare": {
      if (cloudJobs.has(request.job)) throw new Error("Duplicate CPU C2C job");
      cloudJobs.set(request.job, new C2cPrepared(request.reference, request.queries));
      return { value: new Float64Array(0), transfer: [] };
    }
    case "cloud-prepared": {
      const job = cloudJobs.get(request.job);
      if (!job) throw new Error("CPU C2C preparation is unavailable");
      const value = job.distances(request.simd);
      return { value, transfer: [value.buffer] };
    }
    case "cloud-release":
      cloudJobs.get(request.job)?.free();
      cloudJobs.delete(request.job);
      return { value: new Float64Array(0), transfer: [] };
    case "gpu-verify-prepare": {
      if (!gpuReferences.has(request.referenceId))
        gpuReferences.set(request.referenceId, request.reference instanceof Uint8Array
          ? GpuC2cReference.fromSnapshot(request.reference, request.reuseBounds) : GpuC2cReference.fromXyz(request.reference));
      return { value: new Float64Array(0), transfer: [] };
    }
    case "gpu-verify": {
      let verified;
      if ("queries" in request) {
        const reference = gpuReferences.get(request.referenceId);
        if (!reference) throw new Error("GPU verification reference is unavailable");
        verified = reference.verify(request.queries, request.candidates, request.bounds, request.simd);
      } else {
        const job = gpuQueries.get(request.queryJob);
        if (!job) throw new Error("GPU verification queries are unavailable");
        verified = job.verify(request.first, request.candidates, request.bounds, request.simd);
      }
      try {
        const value = { distances: verified.takeDistances(), certified: verified.certified };
        return { value, transfer: [value.distances.buffer] };
      } finally { verified.free(); }
    }
    case "gpu-query-prepare": {
      const reference = gpuReferences.get(request.referenceId);
      if (!reference) throw new Error("GPU verification reference is unavailable");
      if (gpuQueries.has(request.queryJob)) throw new Error("Duplicate GPU query job");
      gpuQueries.set(request.queryJob, new GpuC2cVerifier(reference, request.queries));
      return { value: new Float64Array(0), transfer: [] };
    }
    case "gpu-query-release":
      gpuQueries.get(request.queryJob)?.free();
      gpuQueries.delete(request.queryJob);
      return { value: new Float64Array(0), transfer: [] };
    case "gpu-verify-release":
      gpuReferences.get(request.referenceId)?.free();
      gpuReferences.delete(request.referenceId);
      return { value: new Float64Array(0), transfer: [] };
    case "cloud": {
      const d = nearestDistances(request.reference, request.queries, request.simd);
      return { value: d, transfer: [d.buffer] };
    }
    case "mesh": {
      const d = meshDistances(request.vertices, request.indices, request.queries, request.signed);
      return { value: d, transfer: [d.buffer] };
    }
    case "mesh-prepare": {
      if (meshParts.has(request.job)) throw new Error("Duplicate mesh preparation");
      const part = new MeshPrepared(request.vertices, request.indices);
      meshParts.set(request.job, part);
      return { value: part.retainedBytes, transfer: [] };
    }
    case "mesh-prepared": {
      const part = meshParts.get(request.job);
      if (!part) throw new Error("Mesh preparation is gone");
      const reuse = USE_MESH_QUERY_ORDER && request.orderCache !== false && !request.hints;
      if (reuse) meshQueryOrder ??= new MeshQueryOrder(8 * 1024 * 1024);
      else meshQueryOrder?.clear();
      const d = request.hints ? part.distancesHinted(request.queries,request.signed,request.hints)
        : reuse ? meshQueryOrder!.distances(part, request.queries, request.signed) : part.distances(request.queries, request.signed);
      return { value: { distances: d, queryOrderHit: reuse && meshQueryOrder!.hit,
        queryOrderBytes: meshQueryOrder?.retainedBytes ?? 0, queryOrderEntries: meshQueryOrder?.entryCount ?? 0 }, transfer: [d.buffer] };
    }
    case "mesh-order-release":
      meshQueryOrder?.clear();
      return { value: new Float64Array(0), transfer: [] };
    case "mesh-gpu-buffers": {
      const part = meshParts.get(request.job);
      if (!part) throw new Error("Mesh preparation is gone");
      const buffers = part.gpuBuffers();
      try {
        const origin = buffers.origin(), nodes = buffers.takeNodes(), triangles = buffers.takeTriangles();
        return { value: {origin,nodes,triangles}, transfer: [origin.buffer,nodes.buffer,triangles.buffer] };
      } finally { buffers.free(); }
    }
    case "mesh-release":
      meshParts.get(request.job)?.free(); meshParts.delete(request.job);
      return { value: new Float64Array(0), transfer: [] };
    case "m3c2": {
      const d = m3c2Distances(request.reference, request.compared, request.core,
        request.normalRadius, request.projectionRadius, request.maxDepth, request.radiusBounds, request.sharedSearch, request.fusedProjection);
      return { value: d, transfer: [d.buffer] };
    }
    case "bucket-chunk":
      return unpack(bucketChunk(request.positions, request.colors, request.cube));
    case "m3c2-prepare": {
      if (m3c2Jobs.has(request.job)) throw new Error("Duplicate M3C2 preparation");
      m3c2Jobs.set(request.job, M3c2Prepared.fromSnapshots(request.reference, request.compared,
        request.referenceCount, request.comparedCount));
      return { value: new Float64Array(0), transfer: [] };
    }
    case "m3c2-prepared": {
      const job = m3c2Jobs.get(request.job);
      if (!job) throw new Error("M3C2 preparation was lost");
      const d = request.cacheNormals ? job.computeCached(request.core, request.normalRadius, request.projectionRadius,
        request.maxDepth, request.radiusBounds, request.maxCachedCores ?? 0, request.sharedSearch, request.fusedProjection) : job.compute(request.core,
        request.normalRadius, request.projectionRadius, request.maxDepth, request.radiusBounds, request.sharedSearch, request.fusedProjection);
      return { value: { distances: d, normalCacheHit: request.cacheNormals ? job.normalCacheHit : false,
        fusedProjection: request.fusedProjection ?? (!!request.cacheNormals && job.normalCacheHit),
        normalCacheBytes: job.normalCacheBytes }, transfer: [d.buffer] };
    }
    case "m3c2-release":
      m3c2Jobs.get(request.job)?.free();
      m3c2Jobs.delete(request.job);
      return { value: new Float64Array(0), transfer: [] };
    case "bucket":
      return unpack(buildBucket(request.positions, request.colors, request.cube, request.key));
    case "sor-local": {
      sorParts.get(request.job)?.free();
      const part = new SorPart(request.points);
      sorParts.set(request.job, part);
      return sorLocalResult(part, request.k, request.own, request.regions);
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
    case "knn-prepare": {
      if (knnParts.has(request.job)) throw new Error("Duplicate neighbourhood preparation");
      const part = SorPart.prepared(request.points);
      knnParts.set(request.job, part);
      return { value: part.retainedBytes, transfer: [] };
    }
    case "knn-local": {
      const part = knnParts.get(request.job);
      if (!part) throw new Error("Neighbourhood preparation is gone");
      return sorLocalResult(part, request.k, request.own, request.regions, request.bounds ?? false, request.gpuHints, request.cpuHints, request.hintSlots);
    }
    case "knn-within": {
      const part = knnParts.get(request.job);
      if (!part) throw new Error("Neighbourhood preparation is gone");
      const d = part.withinWithBounds(request.queries, request.k, request.bounds ?? false);
      return { value: d, transfer: [d.buffer] };
    }
    case "knn-normals": {
      const part = knnParts.get(request.job);
      if (!part) throw new Error("Neighbourhood preparation is gone");
      const n = request.gpuHints ? request.hintSlots ? part.normalsCandidates(request.k, request.orientation, request.gpuHints) : part.normalsHinted(request.k, request.orientation, request.gpuHints) :
        request.cpuHints ? part.normalsCpuHints(request.k, request.orientation) : part.normalsWithBounds(request.k, request.orientation, request.bounds ?? false);
      return { value: n, transfer: [n.buffer] };
    }
    case "knn-hint-prepare": {
      const part = knnParts.get(request.job);
      if (!part) throw new Error("Neighbourhood preparation is gone");
      return { value: part.prepareHintBounds(), transfer: [] };
    }
    case "knn-gpu-buffers": {
      const part = knnParts.get(request.job);
      if (!part) throw new Error("Neighbourhood preparation is gone");
      const buffers = request.slots ? part.gpuCandidateBuffers() : part.gpuHintBuffers();
      try {
        const nodes = buffers.takeNodes(), queries = buffers.takeQueries();
        if (request.slots) {
          const points = buffers.takePoints();
          return { value: { nodes, queries, points }, transfer: [nodes.buffer, queries.buffer, points.buffer] };
        }
        return { value: { nodes, queries }, transfer: [nodes.buffer, queries.buffer] };
      } finally { buffers.free(); }
    }
    case "knn-release":
      knnParts.get(request.job)?.free();
      knnParts.delete(request.job);
      return { value: new Float64Array(0), transfer: [] };
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
        if (last && offsets[k] <= last.end + (64 << 10) && end - last.start <= MAX_RANGE_BYTES) last.end = Math.max(last.end, end);
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
    case "las-chunks": {
      const { chunks } = request;
      const n = chunks.length / 4;
      // Chunks follow each other in the file: one read for all.
      const start = chunks[0];
      const bytes = await readRange(request.source, start, chunks[(n - 1) * 4] + chunks[(n - 1) * 4 + 1] - start);
      const decoded = decodeLasChunks(request.head, bytes, chunks, request.keepEvery);
      const value: LasChunksResult = {
        positions: decoded.positions(),
        colors: decoded.colors() ?? null,
        intensity: decoded.intensity(),
        classification: decoded.classification(),
        bounds: decoded.bounds(),
      };
      decoded.free();
      const transfer: Transferable[] = [
        value.positions.buffer,
        value.intensity.buffer,
        value.classification.buffer,
        value.bounds.buffer,
      ];
      if (value.colors) transfer.push(value.colors.buffer);
      return { value, transfer };
    }
    case "loop": {
      const r = registerScans(
        request.from,
        request.to,
        new Float64Array(request.guess),
        request.maxIterations,
        request.overlap,
        request.inlierDistance,
        request.retryBelow,
        request.retryHeadings,
      );
      return { value: r, transfer: [r.buffer] };
    }
    case "dynamic": {
      const r = dynamicScans(request.context, request.first, request.count, request.window, request.margin, request.votes);
      return { value: r, transfer: [r.buffer] };
    }
    case "sor-release":
      sorParts.get(request.job)?.free();
      sorParts.delete(request.job);
      return { value: new Float64Array(0), transfer: [] };
    case "warm-up":
      warmUp();
      return { value: new Float64Array(0), transfer: [] };
    case "odom-reset":
      odometryMap?.free();
      odometryMap = new OdometryMap(request.voxel, request.maxPoints);
      odometrySource = new Float64Array(0);
      return { value: new Float64Array(0), transfer: [] };
    case "odom-map":
      odometryMap?.update(request.placed, request.origin, request.range);
      return { value: new Float64Array(0), transfer: [] };
    case "odom-source":
      odometrySource = request.source;
      return { value: new Float64Array(0), transfer: [] };
    case "odom-equations": {
      if (!odometryMap) throw new Error("the odometry map was not set up on this worker");
      const r = odometryMap.equations(odometrySource, request.total, request.terms) as Float64Array<ArrayBuffer>;
      return { value: r, transfer: [r.buffer] };
    }
  }
}

self.onmessage = async (event: MessageEvent<SliceRequest>) => {
  const request = event.data;
  try {
    await ready;
    const { value, transfer } = await run(request);
    const response: SliceResponse = { seq: request.seq, ok: true, value, memory: (await ready).memory.buffer.byteLength };
    self.postMessage(response, { transfer });
  } catch (err) {
    const response: SliceResponse = {
      seq: request.seq,
      ok: false,
      memory: await ready.then(w => w.memory.buffer.byteLength, () => 0),
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(response);
  }
};
