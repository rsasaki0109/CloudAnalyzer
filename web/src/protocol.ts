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
  /** Interleaved xyz in octree order, minus `shift`. */
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
  /** Float attributes usable as scalar fields (values fetched on demand). */
  scalarNames: string[];
  /** Octree node table, see `Cloud.lodNodes()`. */
  lodNodes: Float64Array;
  lodGrid: number;
  /** 1 when every point was loaded; n when only every n-th point was kept. */
  keepEvery: number;
  /** Points in the file (before thinning). */
  filePoints: number;
  /** For a COPC file read to a budget: how many octree levels were read. */
  copcLevels: number | null;
  /** Original COPC source is still valid for a bounded full-density box. */
  copcBoxAvailable?: boolean;
  copcBox?: { sourcePoints: number; sourceReadBytes: number; nodes: number };
  /**
   * For a thinned LAS/LAZ file: its chunks, which the worker can decode at
   * full density on demand (see `DetailChunk`), as `minX, minY, minZ, maxX,
   * maxY, maxZ, points` per chunk in original coordinates. Null otherwise.
   */
  detailChunks: Float64Array | null;
  /** Milliseconds spent in each loading stage inside the worker. */
  timings: { parse: number; index: number; prepare: number; workers?: number };
  /** [minX, minY, minZ, maxX, maxY, maxZ] in original coordinates. */
  bounds: number[];
  /**
   * This item's own offset (its suggested shift), so that float32 positions
   * stay precise whatever else is open; the UI places it in the scene.
   */
  shift: Vec3;
}

/** Every point of one chunk of a thinned cloud's file. */
export interface DetailChunk {
  /** Interleaved xyz minus the requested shift. */
  positions: Float32Array;
  /** Interleaved rgb narrowed like the cloud's colors, or null. */
  colors: Uint8Array | null;
  intensity: Float32Array;
  classification: Uint8Array;
  /**
   * The points come in slices along the chunk's longest side, so a chunk
   * shaped like a long strip (airborne files are in flight-line order) is
   * drawn only where it is in view: `count, minX, minY, minZ, maxX, maxY,
   * maxZ` per slice, relative to the shift, in point order.
   */
  pieces: Float64Array;
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

/** A map against its ground truth, after MapEval. Distances in metres. */
export interface MapQuality {
  /** Inlier distance: nearest points closer than this count as matched. */
  threshold: number;
  /** RMS distance from the map's matched points to the truth. */
  accuracy: number;
  /** Share of the map's points near the truth. */
  precision: number;
  /** Share of the truth's points near the map. */
  completeness: number;
  f1: number;
  /** Mean matched distance, averaged over both directions. */
  chamfer: number;
  /** Average Wasserstein distance between matching voxel Gaussians. */
  awd: number;
  /** Spatial consistency: spread of that distance among neighbouring voxels. */
  scs: number;
  voxels: number;
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
  /** Set by the Map quality method. */
  quality?: MapQuality;
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

/** A pose graph as the worker holds it (see `PoseGraphSession`). */
export interface PoseGraphState {
  /** Node ids from the file (frame numbers for a trajectory). */
  nodeIds: Float64Array;
  /** Row-major 4x4 pose per node, original coordinates. */
  poses: Float64Array;
  /** `from, to` node indices per edge. */
  edges: Uint32Array;
  /** 0 odometry, 1 loop, per edge. */
  edgeKinds: Uint8Array;
  /** Squared (information-weighted) error per edge. */
  edgeErrors: Float64Array;
  /** 1 per node the optimiser holds in place. */
  fixed: Uint8Array;
  /** Plane landmarks (e.g. the floor), and node-to-plane edges. */
  planes: number;
  planeEdges: number;
  /** Nodes tied to gravity (an IMU's up direction). */
  gravityEdges: number;
}

/** A floor tied to the keyframes that see it (see `pg-floor`). */
export interface PoseGraphFloor {
  state: PoseGraphState;
  /** The new planes (indices `first` on), for undo. */
  planes: { first: number; count: number };
  /** Keyframes whose scan shows the floor. */
  tied: number;
  /** The up axis used, in scan coordinates. */
  up: number[];
  optimized: { initialCost: number; finalCost: number; iterations: number };
}

export interface PoseGraphOpened extends PoseGraphState {
  name: string;
  /** Per node: a thinned copy of its scan in the node's frame, or null. */
  scans: (Float32Array | null)[];
  /** Points kept for registration, over all scans. */
  scanPoints: number;
  /** Scan files that matched no node. */
  unmatched: string[];
  /** When the poses came from odometry: what it went through, and per node the IMU's up direction (three each, NaN where none). */
  odometry: {
    frames: number;
    pathLength: number;
    seconds: number;
    topic: string | null;
    imuTopic: string | null;
    ups: Float64Array | null;
  } | null;
}

export interface PoseGraphLoop {
  state: PoseGraphState;
  edge: number;
  rmsInitial: number;
  rmsFinal: number;
  iterations: number;
  converged: boolean;
  /** Fraction of B's points within the inlier distance of A's once registered. */
  fitness: number;
  /** The registration that counted came from the same-place retry. */
  retried: boolean;
}

/** A second graph joined to the open one (see `pg-merge`). */
export interface PoseGraphMerged {
  state: PoseGraphState;
  /** The joined graph's file name. */
  name: string;
  /** Display copies of its scans, for its nodes from `offset` on. */
  scans: (Float32Array | null)[];
  scanPoints: number;
  unmatched: string[];
  /** Index of its first node in the joined graph. */
  offset: number;
  /** Overlap and ICP RMS of the registration that placed it. */
  fitness: number;
  rms: number;
  optimized: { initialCost: number; finalCost: number; iterations: number };
  /** When its poses came from odometry (a bag): as for `PoseGraphOpened`, its nodes counted from 0. */
  odometry: PoseGraphOpened["odometry"];
}

/** A pose graph's files and how to read them (see `pg-open`). */
export interface PoseGraphFiles {
  /** A g2o graph or a trajectory, with `scans`; a ROS bag (.bag, .mcap) alone; or null for scans without poses. */
  graph: File | null;
  scans: File[];
  /** For a bag, or scans without poses: odometry, keeping a keyframe every `keyframeSpacing` metres of travel. */
  odometry: { minRange: number; maxRange: number; keyframeSpacing: number; deskew: boolean };
  /** Voxel size scans are thinned to (0 keeps every point). */
  voxel: number;
  /** Points per scan sent back for display. */
  displayPoints: number;
  /** Row-major 4x4 from scan to pose frame, or null. */
  extrinsic: number[] | null;
  /** Odometry standard deviations for a trajectory (metres, degrees). */
  sigmaT: number;
  sigmaRDeg: number;
}

export interface PoseGraphSource {
  files: PoseGraphFiles;
  first: number;
  nodeIds: number[];
}

export interface PoseGraphProject {
  snapshot: string;
  sources: PoseGraphSource[];
}

/** Loops found automatically (see `pg-find-loops`). */
export interface PoseGraphFound {
  state: PoseGraphState;
  candidates: number;
  /** Accepted loops, as node indices, with their fitness and whether the same-place retry found them. */
  added: { from: number; to: number; fitness: number; retried: boolean }[];
  /** Their edge indices, for undo. */
  edges: number[];
  /** Candidates that overlapped but were placed further from where the graph has them than drift allows. */
  implausible: number;
  /** Graph cost before and after the optimisation that followed, if any loop was added. */
  optimized: { initialCost: number; finalCost: number; iterations: number } | null;
}

export interface PoseGraphOptimized {
  state: PoseGraphState;
  initialCost: number;
  finalCost: number;
  iterations: number;
  converged: boolean;
  millis: number;
}

export type PoseFormat = "g2o" | "kitti" | "tum";

/** A removed edge, to put back on undo (see `PoseGraphSession.edgeData`). */
export interface RemovedEdge {
  index: number;
  data: Float64Array;
}

/** A ROS 1 bag or an MCAP file, by name. */
export const isBag = (name: string) => /\.(bag|mcap)$/i.test(name);

/** Filters that keep a subset of a cloud's points (see `Cloud.filter`). */
export type FilterOp = "voxel" | "random" | "spatial" | "octree" | "sor" | "splat";

/** What a "vm" request does. */
export type VectorMapOp = "history-budget" | "history-clear" | "check-project" | "open" | "apply" | "quality" | "feature-edit" | "relations-edit" | "relations-preview" | "relations-adopt" | "feature-discover" | "feature-confirm" | "build" | "junction-preview" | "junction-connect" | "signal-preview" | "signal-add" | "crosswalk-preview" | "crosswalk-add" | "undo" | "clear" | "view" | "validate" | "export" | "nearest" | "json";

export type Request =
  | { kind: "memory-stats" }
  | { kind: "release-pool" }
  | { kind: "pg-project-save" }
  | { kind: "pg-project-open"; project: PoseGraphProject; name: string }
  | { kind: "copc-box"; id: number; min: Vec3; max: Vec3; maxPoints: number }
  | ({
      /** Open a pose graph (g2o, or a TUM / KITTI trajectory as an odometry chain) with a scan per node. */
      kind: "pg-open";
    } & PoseGraphFiles)
  | {
      /**
       * Join a second graph: its node `nodeB` (an id; null for its first
       * node) stands near the open graph's node `nodeA` (an index). Their
       * scans are registered with a yaw search, which places the second
       * graph; the registration becomes a loop edge; then optimise.
       */
      kind: "pg-merge";
      files: PoseGraphFiles;
      nodeA: number;
      nodeB: number | null;
      yawSteps: number;
      maxIterations: number;
      overlap: number;
      inlierDistance: number;
      minFitness: number;
      sigmaT: number;
      sigmaRDeg: number;
      loopKernel: number;
    }
  | {
      /** Register node `to`'s scan onto node `from`'s and add a loop edge. */
      kind: "pg-loop";
      from: number;
      to: number;
      maxIterations: number;
      overlap: number;
      pointToPlane: boolean;
      /** For the reported fitness (metres). */
      inlierDistance: number;
      /** Below half overlap, register again as if both scans were taken at the same place, from this many headings (0: no retry). */
      retryHeadings: number;
      sigmaT: number;
      sigmaRDeg: number;
    }
  | {
      /**
       * ICP of node `to`'s scan onto node `from`'s from `guess` (`to` in
       * `from`'s frame, row-major 4x4), without changing the graph.
       */
      kind: "pg-register";
      from: number;
      to: number;
      guess: number[];
      maxIterations: number;
      overlap: number;
      inlierDistance: number;
    }
  | {
      /** Add a loop edge measuring `to` in `from`'s frame as `matrix` (row-major 4x4). */
      kind: "pg-add-edge";
      from: number;
      to: number;
      matrix: number[];
      sigmaT: number;
      sigmaRDeg: number;
    }
  | {
      /**
       * Find loops: node pairs close in space but far apart along the path,
       * each registered with ICP and kept when its fitness is high enough;
       * then optimise.
       */
      kind: "pg-find-loops";
      maxDistance: number;
      /** Odometry drift (a fraction): the search, and the check of a loop, widen by this share of the path between its nodes. */
      drift: number;
      minTravel: number;
      spacing: number;
      /** 0..1: fraction of points within `inlierDistance` after registration. */
      minFitness: number;
      inlierDistance: number;
      /** Below `minFitness`, register again as if both scans were taken at the same place, from this many headings (0: no retry). */
      retryHeadings: number;
      maxIterations: number;
      overlap: number;
      sigmaT: number;
      sigmaRDeg: number;
      loopKernel: number;
    }
  | {
      /** Find the floor under each keyframe, tie them to one floor plane, optimise. */
      kind: "pg-floor";
      /** Up in scan coordinates, or null to pick +z, -y or +y automatically. */
      up: number[] | null;
      maxTiltDeg: number;
      /** Largest distance of a floor point from the plane (metres). */
      threshold: number;
      minPoints: number;
      sigmaAngleDeg: number;
      sigmaOffset: number;
      loopKernel: number;
    }
  | { kind: "pg-remove-plane"; index: number }
  | {
      /** Tie nodes to the up direction each measured (in its frame, e.g. from an IMU), then optimise. */
      kind: "pg-gravity";
      nodes: number[];
      /** Three per node. */
      ups: Float64Array;
      sigmaDeg: number;
      /** Estimate the IMU's rotation into the scans' frame from the drive first. */
      calibrate: boolean;
      loopKernel: number;
    }
  | { kind: "pg-clear-gravity" }
  | {
      /** Put a node at `pose` (row-major 4x4); with `carry`, the nodes after it move along. */
      kind: "pg-set-node-pose";
      index: number;
      pose: number[];
      carry: boolean;
    }
  | { kind: "pg-set-fixed"; index: number; fixed: boolean }
  | { kind: "pg-optimize"; /** Huber threshold for loops, 0 for none. */ loopKernel: number }
  | {
      /** Remove edges (the removed ones come back for undo), then optimise if `loopKernel` is given. */
      kind: "pg-remove-edges";
      indices: number[];
      /** Huber threshold for the optimisation (0 for none), or null to skip it. */
      loopKernel: number | null;
    }
  | { kind: "pg-insert-edges"; /** In ascending index order. */ edges: RemovedEdge[] }
  | { kind: "pg-set-poses"; poses: Float64Array }
  | { kind: "pg-export"; format: PoseFormat }
  | {
      kind: "pg-map";
      /** Voxel size of the map, 0 for none. */
      voxel: number;
      /** The scans at their poses as loaded, instead of now. */
      initial: boolean;
      /** Give each point a `correction` field: how far it moved from its place as loaded. */
      correction: boolean;
      /** Only these nodes (indices; all when empty). */
      nodes: number[];
      /** Added to the cloud's name, e.g. which part it is. */
      label: string;
      /** 0 every point, 1 the static ones, 2 the dynamic ones (after `pg-dynamic`). */
      part: 0 | 1 | 2;
    }
  | {
      /** Find the dynamic points of every scan by visibility (see `ca_core::dynamic`). */
      kind: "pg-dynamic";
      window: number;
      margin: number;
      votes: number;
    }
  | { kind: "pg-close" }
  | {
      /**
       * The vector map (see `VectorMapSession`): open a Lanelet2 / IR file,
       * apply vectormap commands (a JSON list), undo, clear, or read its
       * view, issues, export or the lane nearest to a point. Answers JSON text.
       */
      kind: "vm";
      op: VectorMapOp;
      pairs?: [number, number][];
      id?: number;
      positions?: Float64Array;
      name?: string;
      text?: string;
      x?: number;
      y?: number;
      autoware?: boolean;
    }
  | {
      /** An M3C2 result's significant changes as objects (see `Cloud.changedObjects`). */
      kind: "change-objects";
      id: number;
      /** Smallest change counted (metres). */
      minChange: number;
      /** Significant core points closer than this are one object (metres). */
      link: number;
      minPoints: number;
    }
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
      /**
       * A remote file read with range requests: COPC node by node, other
       * LAS/LAZ chunk by chunk; anything else is downloaded whole.
       */
      kind: "load-url";
      url: string;
      name: string;
      /** File size in bytes. */
      size: number;
      etag?: string;
      /** As for `load`. */
      maxPoints: number;
    }
  | {
      /** One chunk of a thinned LAS/LAZ cloud's file at full density (see `LoadedCloud.detailChunks`). */
      kind: "detail";
      id: number;
      chunk: number;
      /** Positions come back relative to this (the cloud's `shift`). */
      shift: Vec3;
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
      op: FilterOp;
      /**
       * voxel: edge length; random: point count; spatial: minimum distance;
       * octree: level; sor: neighbours; splat: minimum opacity.
       */
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
  | {
      /** Voxel Gaussian scores of `compared` against the ground-truth map `reference` (see `ca_core::map_quality`). */
      kind: "map-quality";
      compared: number;
      reference: number;
      voxel: number;
      minPoints: number;
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
  | { kind: "field"; id: number; name: string }
  | { kind: "set-field"; id: number; name: string; values: Float32Array }
  | {
      kind: "filter-field";
      id: number;
      /** One value per point (octree order); NaN never matches. */
      values: Float32Array;
      lo: number;
      hi: number;
      /** Keep the points within [lo, hi] (true) or the others. */
      inside: boolean;
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
  | { seq: number; response: Response; memory: number; poolMemory: number }
  | { seq: number; progress: Progress };

/** UI -> worker messages: a request, or asking to stop one (loads stop between steps). */
export type UiMessage = { seq: number; req: Request } | { cancel: number };

/** Error message of a request stopped by `cancel`. */
export const CANCELLED = "Cancelled";
