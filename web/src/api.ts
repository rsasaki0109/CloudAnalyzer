// Promise-based client for the WASM worker.

import type {
  C2cOutput,
  DetailChunk,
  ExportFormat,
  FilterOp,
  IcpOutput,
  LoadedCloud,
  M3c2Output,
  MeshFormat,
  MeshOutput,
  PoseFormat,
  PoseGraphFound,
  PoseGraphLoop,
  PoseGraphOpened,
  PoseGraphOptimized,
  PoseGraphState,
  ProfileOutput,
  RemovedEdge,
  Progress,
  RasterGrid,
  RasterOutput,
  Request,
  SegmentOutput,
  TrajectoryEvaluation,
  TrajectoryPoses,
  UiMessage,
  Vec3,
  VolumeOutput,
  WorkerMessage,
} from "./protocol";

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const pending = new Map<
  number,
  { resolve: (v: unknown) => void; reject: (e: Error) => void; progress?: (p: Progress) => void }
>();
let seq = 0;

/** Called with the size of the worker's WASM memory after every request. */
let onMemory: (bytes: number) => void = () => {};
export function setMemoryListener(listener: (bytes: number) => void): void {
  onMemory = listener;
}

worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  const entry = pending.get(message.seq);
  if (!entry) return;
  if ("progress" in message) {
    entry.progress?.(message.progress);
    return;
  }
  pending.delete(message.seq);
  onMemory(message.memory);
  if (message.response.ok) entry.resolve(message.response.value);
  else entry.reject(new Error(message.response.error));
};

function call<T>(
  req: Request,
  transfer: Transferable[] = [],
  progress?: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<T> {
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject, progress });
    const message: UiMessage = { seq: id, req };
    worker.postMessage(message, { transfer });
    signal?.addEventListener("abort", () => worker.postMessage({ cancel: id } satisfies UiMessage), { once: true });
  });
}

/** Load a file; `signal` stops it between steps (rejecting with `CANCELLED`). */
export function loadCloud(
  file: File,
  maxPoints: number,
  progress?: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<LoadedCloud> {
  return call({ kind: "load", file, maxPoints }, [], progress, signal);
}

/**
 * Read a remote file of `size` bytes with range requests: COPC down to the
 * levels that fit `maxPoints`, other LAS/LAZ chunk by chunk (as for a local file).
 */
export function loadUrl(
  url: string,
  name: string,
  size: number,
  maxPoints: number,
  progress?: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<LoadedCloud> {
  return call({ kind: "load-url", url, name, size, maxPoints }, [], progress, signal);
}

/** Every point of one chunk of a thinned LAS/LAZ cloud's file, relative to `shift`. */
export function readDetail(id: number, chunk: number, shift: Vec3): Promise<DetailChunk> {
  return call({ kind: "detail", id, chunk, shift });
}

/** A trajectory file's poses (TUM, KITTI or CSV), or null when the file is not a trajectory. */
export function loadTrajectory(file: File): Promise<TrajectoryPoses | null> {
  return call({ kind: "trajectory", file });
}

/** ATE / RPE of an estimated trajectory against a reference. */
export function evaluateTrajectory(
  params: Omit<Extract<Request, { kind: "trajectory-eval" }>, "kind">,
): Promise<TrajectoryEvaluation> {
  return call({ kind: "trajectory-eval", ...params });
}

/** C2C, or C2M when `reference` is a mesh (`signed` then applies). */
export function cloudToCloud(compared: number, reference: number, signed = true): Promise<C2cOutput> {
  return call({ kind: "c2c", compared, reference, signed });
}

export function removeCloud(id: number): Promise<void> {
  return call({ kind: "remove", id });
}

/** Exact (unshifted, f64) coordinates of a point in octree order. */
export function pointAt(id: number, index: number): Promise<[number, number, number]> {
  return call({ kind: "point", id, index });
}

export function registerIcp(params: Omit<Extract<Request, { kind: "icp" }>, "kind">): Promise<IcpOutput> {
  return call({ kind: "icp", ...params });
}

/** Apply a row-major 4x4 rigid transform to a cloud; returns it re-described. */
export function transformCloud(id: number, matrix: number[]): Promise<LoadedCloud> {
  return call({ kind: "transform", id, matrix });
}

/** Serialize a cloud (octree order) as binary PLY, LAS, LAZ, CSV or E57. */
export function exportCloud(
  id: number,
  format: ExportFormat,
  scalar?: { name: string; values: Float32Array },
): Promise<Uint8Array> {
  // The scalar is copied, not transferred, so the UI keeps its distances.
  return call({ kind: "export", id, format, scalar });
}

/** Serialize a mesh as binary PLY or OBJ. */
export function exportMesh(id: number, format: MeshFormat): Promise<Uint8Array> {
  return call({ kind: "export", id, format });
}

/** 2.5D Delaunay mesh of a cloud (see `Cloud.meshDelaunay`), as a new mesh. */
export function meshCloud(id: number, maxEdge: number | null): Promise<MeshOutput> {
  return call({ kind: "mesh", id, maxEdge });
}

/** Extract the points inside (or outside) a box, in original coordinates, as a new cloud. */
export function cropCloud(id: number, min: Vec3, max: Vec3, inside: boolean): Promise<LoadedCloud> {
  return call({ kind: "crop", id, min, max, inside });
}

/** Split a cloud by a screen-space lasso: the selected points, the rest, or both (in that order). */
export function segmentCloud(params: Omit<Extract<Request, { kind: "segment" }>, "kind">): Promise<LoadedCloud[]> {
  return call({ kind: "segment", ...params });
}

/** A scalar field of a cloud ("Z" for heights), one value per point in octree order. */
export function fieldValues(id: number, name: string): Promise<Float32Array> {
  return call({ kind: "field", id, name });
}

/** Store a computed field on a cloud so exports carry it (the values are copied). */
export function setField(id: number, name: string, values: Float32Array): Promise<void> {
  const copy = values.slice();
  return call({ kind: "set-field", id, name, values: copy }, [copy.buffer]);
}

/** A new cloud with the points whose field value is within (or outside) [lo, hi]. */
export function filterByField(
  id: number,
  values: Float32Array,
  lo: number,
  hi: number,
  inside: boolean,
): Promise<LoadedCloud> {
  const copy = values.slice();
  return call({ kind: "filter-field", id, values: copy, lo, hi, inside }, [copy.buffer]);
}

/** The rigid transform (row-major 4x4) taking picked points onto their partners, with the residuals. */
export async function alignPairs(
  moving: number[],
  reference: number[],
): Promise<{ matrix: number[]; rms: number; residuals: number[] }> {
  const values: Float64Array = await call({ kind: "align-pairs", moving, reference });
  return { matrix: [...values.subarray(0, 16)], rms: values[16], residuals: [...values.subarray(17)] };
}

/** A filtered copy of a cloud as a new cloud (see `Cloud.filter`). */
export function filterCloud(id: number, op: FilterOp, a: number, b = 0): Promise<LoadedCloud> {
  return call({ kind: "filter", id, op, a, b });
}

/** Cut/fill volume between two surfaces (clouds, meshes or constant heights). */
export function computeVolume(params: Omit<Extract<Request, { kind: "volume" }>, "kind">): Promise<VolumeOutput> {
  return call({ kind: "volume", ...params });
}

/** Rasterize a cloud into a height grid (DEM / DSM); its cells come back as a new cloud. */
export function rasterizeCloud(params: Omit<Extract<Request, { kind: "rasterize" }>, "kind">): Promise<RasterOutput> {
  return call({ kind: "rasterize", ...params });
}

/** A raster as a single-band Float32 GeoTIFF. */
export function rasterGeotiff(raster: RasterGrid): Promise<Uint8Array> {
  // The heights are copied, not transferred, so the UI keeps them.
  return call({ kind: "geotiff", raster });
}

/** Ground extraction (Cloth Simulation Filter) as a new cloud. */
export function extractGround(params: Omit<Extract<Request, { kind: "ground" }>, "kind">): Promise<LoadedCloud> {
  return call({ kind: "ground", ...params });
}

/** M3C2 change between two clouds, at core points from `compared`. */
export function computeM3c2(params: Omit<Extract<Request, { kind: "m3c2" }>, "kind">): Promise<M3c2Output> {
  return call({ kind: "m3c2", ...params });
}

/** Cross-section of a cloud along a polyline (x, y pairs, original coordinates). */
export function profileCloud(params: Omit<Extract<Request, { kind: "profile" }>, "kind">): Promise<ProfileOutput> {
  return call({ kind: "profile", ...params });
}

/** One cloud from several; `fills` colors clouds without RGB when others have it. */
export function mergeClouds(ids: number[], fills: Vec3[]): Promise<LoadedCloud> {
  return call({ kind: "merge", ids, fills });
}

/** One new cloud per class code (or per merged source). */
export function splitCloud(id: number, by: "classification" | "source"): Promise<LoadedCloud[]> {
  return call({ kind: "split", id, by });
}

/** Estimate normals (stored on the cloud); resolves to them, interleaved in octree order. */
export function estimateNormals(id: number, k: number, orientation: "up" | "outward"): Promise<Float32Array> {
  return call({ kind: "normals", id, k, orientation });
}

/** RANSAC shapes or Euclidean clusters of a cloud, as new clouds with a table. */
export function findShapes(params: Omit<Extract<Request, { kind: "shapes" }>, "kind">): Promise<SegmentOutput> {
  return call({ kind: "shapes", ...params });
}

/** Open a pose graph with its scans (replacing any open one). */
export function openPoseGraph(
  params: Omit<Extract<Request, { kind: "pg-open" }>, "kind">,
  progress?: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<PoseGraphOpened> {
  return call({ kind: "pg-open", ...params }, [], progress, signal);
}

/** Add a loop edge between two nodes, measured by ICP of their scans. */
export function addPoseGraphLoop(params: Omit<Extract<Request, { kind: "pg-loop" }>, "kind">): Promise<PoseGraphLoop> {
  return call({ kind: "pg-loop", ...params });
}

/** Find, verify and add loops automatically, then optimise. */
export function findPoseGraphLoops(
  params: Omit<Extract<Request, { kind: "pg-find-loops" }>, "kind">,
  progress?: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<PoseGraphFound> {
  return call({ kind: "pg-find-loops", ...params }, [], progress, signal);
}

export function optimizePoseGraph(loopKernel: number): Promise<PoseGraphOptimized> {
  return call({ kind: "pg-optimize", loopKernel });
}

/** Remove edges and, unless `loopKernel` is null, optimise. */
export function removePoseGraphEdges(
  indices: number[],
  loopKernel: number | null,
): Promise<{ state: PoseGraphState; removed: RemovedEdge[] }> {
  return call({ kind: "pg-remove-edges", indices, loopKernel });
}

/** Put removed edges back. */
export function insertPoseGraphEdges(edges: RemovedEdge[]): Promise<PoseGraphState> {
  return call({ kind: "pg-insert-edges", edges });
}

export function setPoseGraphPoses(poses: Float64Array): Promise<PoseGraphState> {
  return call({ kind: "pg-set-poses", poses });
}

/** The graph (g2o) or its poses (KITTI, TUM) as text. */
export function exportPoseGraph(format: PoseFormat): Promise<string> {
  return call({ kind: "pg-export", format });
}

/** Every scan at its optimised pose, as a new cloud. */
export function poseGraphMap(voxel: number): Promise<LoadedCloud> {
  return call({ kind: "pg-map", voxel });
}

export function closePoseGraph(): Promise<void> {
  return call({ kind: "pg-close" });
}
