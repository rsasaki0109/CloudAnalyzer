// Promise-based client for the WASM worker.

import type {
  C2cOutput,
  IcpOutput,
  LoadedCloud,
  M3c2Output,
  ProfileOutput,
  Request,
  Vec3,
  VolumeOutput,
  WorkerMessage,
} from "./protocol";

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const pending = new Map<
  number,
  { resolve: (v: unknown) => void; reject: (e: Error) => void; progress?: (note: string) => void }
>();
let seq = 0;

worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  const entry = pending.get(message.seq);
  if (!entry) return;
  if ("progress" in message) {
    entry.progress?.(message.progress);
    return;
  }
  pending.delete(message.seq);
  if (message.response.ok) entry.resolve(message.response.value);
  else entry.reject(new Error(message.response.error));
};

function call<T>(req: Request, transfer: Transferable[] = [], progress?: (note: string) => void): Promise<T> {
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject, progress });
    worker.postMessage({ seq: id, req }, { transfer });
  });
}

export function loadCloud(
  file: File,
  maxPoints: number,
  progress?: (note: string) => void,
): Promise<LoadedCloud> {
  return call({ kind: "load", file, maxPoints }, [], progress);
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

/** Serialize a cloud (octree order) as binary PLY or CSV. */
export function exportCloud(
  id: number,
  format: "ply" | "csv",
  scalar?: { name: string; values: Float32Array },
): Promise<Uint8Array> {
  // The scalar is copied, not transferred, so the UI keeps its distances.
  return call({ kind: "export", id, format, scalar });
}

/** Extract the points inside (or outside) a box, in original coordinates, as a new cloud. */
export function cropCloud(id: number, min: Vec3, max: Vec3, inside: boolean): Promise<LoadedCloud> {
  return call({ kind: "crop", id, min, max, inside });
}

/** A filtered copy of a cloud as a new cloud (see `Cloud.filter`). */
export function filterCloud(id: number, op: "voxel" | "random" | "sor", a: number, b = 0): Promise<LoadedCloud> {
  return call({ kind: "filter", id, op, a, b });
}

/** Cut/fill volume between two surfaces (clouds, meshes or constant heights). */
export function computeVolume(params: Omit<Extract<Request, { kind: "volume" }>, "kind">): Promise<VolumeOutput> {
  return call({ kind: "volume", ...params });
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
