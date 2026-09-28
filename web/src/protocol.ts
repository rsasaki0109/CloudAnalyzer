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
  /** Gaussian-splat opacity, 0..1 (octree order), or null for other clouds. */
  opacity: Float32Array | null;
  /** For a merged cloud, the names of the clouds its `source` attribute refers to. */
  sources: string[] | null;
  /** Octree node table, see `Cloud.lodNodes()`. */
  lodNodes: Float64Array;
  lodGrid: number;
  /** 1 when every point was loaded; n when only every n-th point was kept. */
  keepEvery: number;
  /** Points in the file (before thinning). */
  filePoints: number;
  /** For a COPC file read to a budget: how many octree levels were read. */
  copcLevels: number | null;
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
  /** Cloud-to-cloud, cloud-to-mesh, a cut/fill height difference, or raster heights. */
  kind: "c2c" | "c2m" | "volume" | "m3c2" | "raster";
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

/** A height grid (DEM / DSM). */
export interface RasterGrid {
  nx: number;
  ny: number;
  /** Lower-left corner of the grid, original coordinates. */
  minX: number;
  minY: number;
  cell: number;
  /** Per-cell heights, row-major from the lowest y; NaN where empty. */
  heights: Float32Array;
}

export interface RasterOutput extends RasterGrid {
  /** Cells that had points, before filling. */
  populatedCells: number;
  /** One point per non-empty cell at its height. */
  cells: LoadedCloud;
  /** The height of each cell point, in the cells' order. */
  cellHeights: Float32Array;
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

export interface MeshOutput {
  mesh: LoadedCloud;
  /** Points triangulated (fewer than the cloud's when it was thinned). */
  points: number;
  /** Voxel size the cloud was thinned with, 0 when it was not. */
  voxel: number;
  /** Longest horizontal edge kept (Infinity when all were kept). */
  maxEdge: number;
  /** Triangles dropped for a longer edge. */
  removed: number;
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

/** File formats a cloud can be saved in. */
export type ExportFormat = "ply" | "las" | "laz" | "csv" | "e57";

/** File formats a mesh can be saved in. */
export type MeshFormat = "ply" | "obj";

/** What `segment` looks for. */
export type SegmentMethod = "plane" | "sphere" | "cylinder" | "cluster";

/** One shape, cluster or the rest, as a row of the result table. */
export interface Segment {
  kind: "plane" | "sphere" | "cylinder" | "cluster" | "other clusters" | "rest";
  count: number;
  /**
   * Original coordinates. Plane: unit normal, d (n · p + d = 0); sphere:
   * centre, radius; cylinder: axis midpoint, unit axis, radius, length;
   * clusters: centroid, extent.
   */
  params: number[];
  /** RMS distance to the shape (NaN if not a shape). */
  rms: number;
  color: Vec3;
}

export interface SegmentOutput {
  /** One segmented cloud (its `source` is the segment), or a cloud per segment; none if nothing was found. */
  clouds: LoadedCloud[];
  segments: Segment[];
  /** Shapes or clusters found. */
  found: number;
  /** Normals estimated for the source cloud (interleaved, octree order), or null if it had them. */
  normals: Float32Array | null;
  millis: number;
}

/** Timed poses from a trajectory file, in original coordinates. */
export interface TrajectoryPoses {
  format: "tum" | "kitti" | "csv";
  /** Seconds; KITTI poses get their index. */
  timestamps: Float64Array;
  /** Interleaved xyz. */
  positions: Float64Array;
  /** Interleaved unit quaternions [x, y, z, w], or null without orientations. */
  orientations: Float64Array | null;
}

/** Summary of an error series (`std` is the population deviation, as in the Python CLI). */
export interface ErrorStats {
  count: number;
  rmse: number;
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
}

export type TrajectoryAlignment = "none" | "origin" | "se3" | "sim3";

export interface TrajectoryEvaluation {
  /** Times of the matched reference poses. */
  timestamps: Float64Array;
  /** Matched estimate after alignment, and the reference, interleaved xyz. */
  estimate: Float64Array;
  reference: Float64Array;
  /** Position error per matched pose. */
  ate: Float64Array;
  /** Degrees, when both trajectories have orientations. */
  ateRotation: Float64Array | null;
  stats: {
    ate: ErrorStats;
    ateRotation: ErrorStats | null;
    /** Null when no pose pair is `delta` apart. */
    rpe: ErrorStats | null;
    rpeRotation: ErrorStats | null;
    /** Translation error in % of the distance travelled (metre deltas). */
    rpePercent: ErrorStats | null;
  };
  /** Row-major 4x4 transform taking the estimate onto the reference (scale included). */
  matrix: number[];
  scale: number;
  endpointDrift: number;
  referenceLength: number;
  estimateLength: number;
}

export type Request =
  | { kind: "trajectory"; file: File }
  | {
      kind: "trajectory-eval";
      estimate: TrajectoryPoses;
      reference: TrajectoryPoses;
      maxTimeDelta: number;
      alignment: TrajectoryAlignment;
      delta: number;
      deltaUnit: "frames" | "m";
    }
  | {
      kind: "load";
      /** Read by the worker in slices, so large files are never held whole. */
      file: File;
      /** Thin to at most this many points (every n-th point is kept; for COPC, whole levels). */
      maxPoints: number;
    }
  | {
      /** A remote COPC file, read node by node with range requests. */
      kind: "load-copc";
      url: string;
      name: string;
      /** Read octree levels while their points fit this. */
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
      kind: "rasterize";
      id: number;
      cell: number;
      height: "mean" | "min" | "max" | "percentile";
      /** 0-100, for `height: "percentile"`. */
      percentile: number;
      fillEmpty: boolean;
      /** Only points of this class code. */
      class: number | null;
    }
  | { kind: "geotiff"; raster: RasterGrid }
  | {
      kind: "filter";
      id: number;
      op: "voxel" | "random" | "sor" | "splat";
      /** voxel: edge length; random: point count; sor: neighbours; splat: minimum opacity. */
      a: number;
      /** sor: standard-deviation threshold; splat: maximum size. */
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
  | {
      kind: "shapes";
      id: number;
      method: SegmentMethod;
      /** Shapes: largest distance to the shape; clusters: linking distance. */
      distance: number;
      /** Fewest points of a shape or cluster. */
      minPoints: number;
      /** Shapes only. */
      maxShapes: number;
      /** A cloud per segment instead of one segmented cloud. */
      split: boolean;
    }
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
      kind: "segment";
      id: number;
      /** Row-major 4x4 matrix from original coordinates to clip space. */
      matrix: number[];
      /** Lasso vertices in normalized device coordinates, as x, y pairs. */
      polygon: number[];
      /** Only points in this box (original coordinates) are selected. */
      clip: { min: Vec3; max: Vec3 } | null;
      hiddenClasses: number[];
      keep: "inside" | "outside" | "both";
    }
  | {
      kind: "align-pairs";
      /** Picked points as flat xyz triples (original coordinates), pair i in both. */
      moving: number[];
      reference: number[];
    }
  | {
      /** 2.5D Delaunay mesh of a cloud's XY positions. */
      kind: "mesh";
      id: number;
      /** Drop triangles with a longer horizontal edge; null: automatic, 0: keep all. */
      maxEdge: number | null;
    }
  | {
      kind: "export";
      id: number;
      /** A cloud's format, or a mesh's. */
      format: ExportFormat | MeshFormat;
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
