// Messages exchanged between the UI thread and the WASM worker.

export type Vec3 = [number, number, number];

export interface LoadedCloud {
  /** Point clouds get octree LOD; meshes are drawn as triangles. */
  kind: "cloud" | "mesh";
  id: number;
  name: string;
  /** Points, or vertices for a mesh. */
  count: number;
  /** Triangle count (0 for a point cloud). */
  triangles: number;
  /** Triangle vertex indices for a mesh, null for a point cloud. */
  indices: Uint32Array | null;
  /** Interleaved xyz in octree order, already shifted by the session's global shift. */
  positions: Float32Array;
  /** Interleaved rgb in octree order, or null when the file carries no colors. */
  colors: Uint8Array | null;
  /** Per-point intensity (octree order), or null. */
  intensity: Float32Array | null;
  /** Per-point ASPRS class codes (octree order), or null. */
  classification: Uint8Array | null;
  /** Interleaved unit normals (octree order), or null. */
  normals: Float32Array | null;
  /** For a merged cloud, the names of the clouds its `source` attribute refers to. */
  sources: string[] | null;
  /** Octree node table, see `Cloud.lodNodes()`. */
  lodNodes: Float64Array;
  lodGrid: number;
  /** 1 when every point was loaded; n when only every n-th point was kept. */
  keepEvery: number;
  /** Points in the file (before thinning). */
  filePoints: number;
  /** Milliseconds spent in each loading stage inside the worker. */
  timings: { parse: number; index: number; prepare: number; workers?: number };
  /** [minX, minY, minZ, maxX, maxY, maxZ] in original coordinates. */
  bounds: number[];
  shift: Vec3;
}

export interface C2cStats {
  count: number;
  min: number;
  max: number;
  mean: number;
  rms: number;
  stdDev: number;
  median: number;
}

export interface C2cOutput {
  /** Cloud-to-cloud, cloud-to-mesh, or a cut/fill height difference. */
  kind: "c2c" | "c2m" | "volume" | "m3c2";
  /** Negative distances mean "behind the mesh" (C2M only). */
  signed: boolean;
  /** Per-point distances of the compared cloud (octree order, like its positions). */
  distances: Float32Array;
  stats: C2cStats;
  millis: number;
  /** Number of WASM workers the computation was split across. */
  workers: number;
}

export interface IcpOutput {
  /** The moving cloud after the transform (new point order and octree). */
  cloud: LoadedCloud;
  /** Row-major 4x4 transform applied to the moving cloud. */
  matrix: number[];
  rmsInitial: number;
  rmsFinal: number;
  iterations: number;
  converged: boolean;
  millis: number;
}

/** A volume surface: a loaded cloud or mesh, or a horizontal plane. */
export type VolumeSide = { id: number } | { z: number };

export interface VolumeOutput {
  added: number;
  removed: number;
  addedArea: number;
  removedArea: number;
  matchedCells: number;
  totalCells: number;
  cell: number;
  /** One point per compared cell, at the "after" height. */
  cells: LoadedCloud | null;
  /** after − before per cell, in the cells' order. */
  difference: Float32Array | null;
  millis: number;
}

export interface M3c2Output {
  /** Core points, carrying the results as attributes. */
  cloud: LoadedCloud;
  /** Per core point, in the cloud's order; NaN where undefined. */
  distance: Float32Array;
  lod95: Float32Array;
  significant: Float32Array;
  millis: number;
}

export interface ProfileOutput {
  /** Distance of each point along the line. */
  along: Float64Array;
  /** Interleaved xyz in original coordinates. */
  positions: Float64Array;
  /** Points in the band before thinning to `maxPoints`. */
  total: number;
}

export type Request =
  | {
      kind: "load";
      /** Read by the worker in slices, so large files are never held whole. */
      file: File;
      /** Thin to at most this many points (every n-th point is kept). */
      maxPoints: number;
    }
  | { kind: "c2c"; compared: number; reference: number; signed: boolean }
  | { kind: "remove"; id: number }
  | { kind: "point"; id: number; index: number }
  | {
      kind: "icp";
      moving: number;
      reference: number;
      maxIterations: number;
      overlap: number;
      matchCentroids: boolean;
      pointToPlane: boolean;
    }
  | { kind: "transform"; id: number; matrix: number[] }
  | {
      kind: "volume";
      before: VolumeSide;
      after: VolumeSide;
      cell: number;
      height: "mean" | "min" | "max";
      fillEmpty: boolean;
    }
  | {
      kind: "filter";
      id: number;
      op: "voxel" | "random" | "sor";
      /** voxel: edge length; random: point count; sor: neighbours. */
      a: number;
      /** sor: standard-deviation threshold. */
      b: number;
    }
  | {
      kind: "m3c2";
      compared: number;
      reference: number;
      normalRadius: number;
      projectionRadius: number;
      maxDepth: number;
      /** Core points: one per voxel of this size (0 = every compared point). */
      coreSpacing: number;
    }
  | { kind: "merge"; ids: number[]; fills: Vec3[] }
  | { kind: "normals"; id: number; k: number; orientation: "up" | "outward" }
  | { kind: "split"; id: number; by: "classification" | "source" }
  | { kind: "profile"; id: number; line: number[]; halfWidth: number; maxPoints: number }
  | {
      kind: "ground";
      id: number;
      clothResolution: number;
      classThreshold: number;
      rigidness: "flat" | "relief" | "steep";
      output: "classified" | "ground" | "objects";
    }
  | {
      kind: "crop";
      id: number;
      /** Box corners in original (unshifted) coordinates. */
      min: Vec3;
      max: Vec3;
      inside: boolean;
    }
  | {
      kind: "export";
      id: number;
      format: "ply" | "csv";
      /** Optional scalar field, one value per point in the cloud's order. */
      scalar?: { name: string; values: Float32Array };
    };

export type Response =
  | { ok: true; value: unknown }
  | { ok: false; error: string };

/** How far a request is: a note, and the done fraction when it is known. */
export interface Progress {
  note: string;
  fraction?: number;
}

/**
 * Worker -> UI messages: a final response (with the size of the worker's
 * WASM memory), or progress on a request.
 */
export type WorkerMessage =
  | { seq: number; response: Response; memory: number }
  | { seq: number; progress: Progress };

/** UI -> worker messages: a request, or asking to stop one (loads stop between steps). */
export type UiMessage = { seq: number; req: Request } | { cancel: number };

/** Error message of a request stopped by `cancel`. */
export const CANCELLED = "Cancelled";
