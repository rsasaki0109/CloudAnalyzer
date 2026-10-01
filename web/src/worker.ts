// Runs the Rust/WASM core off the UI thread. All clouds and meshes live here
// so that analyses can use full f64 coordinates without copying them back and
// forth.

import init, {
  alignPairs,
  announcedPoints,
  BagFile,
  Cloud,
  CloudMerger,
  CopcReader,
  CopcBoxReader,
  cloudToCloud,
  cloudToMesh,
  computeM3c2,
  mapQuality,
  computeVolume,
  e57ScanNames,
  evaluateTrajectory,
  LasReader,
  LidarOdometry,
  Mesh,
  StreamLoader,
  planCloudToCloud,
  planSor,
  PoseGraphSession,
  rasterGeotiff,
  registerIcp,
  registerScans,
  summarizeDistances,
  TrajectoryData,
  upsAt,
  VectorMapSession,
  VolumeSurface,
  warmUp,
} from "./wasm/ca_wasm.js";
import { type ByteSource, readRange } from "./bytes";
import type { LasChunksResult, Slice } from "./c2c-worker";
import { CANCELLED, isBag } from "./protocol";
import { eachSlice, MIN_PARALLEL_QUERIES, poolSize, runAny, runOn, runSlices, warmUpPool } from "./pool";
import type {
  C2cOutput,
  DetailChunk,
  IcpOutput,
  LoadedCloud,
  M3c2Output,
  MeshOutput,
  PoseGraphFound,
  PoseGraphFiles,
  PoseGraphFloor,
  PoseGraphMerged,
  PoseGraphOpened,
  PoseGraphState,
  ProfileOutput,
  Progress,
  RasterOutput,
  Segment,
  SegmentOutput,
  TrajectoryEvaluation,
  TrajectoryPoses,
  UiMessage,
  Request,
  Response,
  Vec3,
  VolumeOutput,
  VolumeSide,
  WorkerMessage,
} from "./protocol";

type Item =
  | {
      kind: "cloud";
      cloud: Cloud;
      name: string;
      keepEvery?: number;
      filePoints?: number;
      /** Names of the merged clouds, by `source` value. */
      sources?: string[];
      copcLevels?: number;
      detail?: Detail;
      copcSource?: CopcSource;
      copcBox?: LoadedCloud["copcBox"];
    }
  | { kind: "mesh"; mesh: Mesh; name: string };

/**
 * Where a thinned LAS/LAZ cloud's file can be read again at full density,
 * chunk by chunk (see `LasReader`). Dropped once the points are moved.
 */
interface Detail {
  source: ByteSource;
  /** The file up to its point data (what `decodeLasChunks` needs). */
  head: Uint8Array;
  /** `offset, size, count, first` per chunk. */
  chunks: Float64Array;
  /** `min, max` corners per chunk. */
  bounds: Float64Array;
  /** How the file's 16-bit colors were narrowed to 8 bits. */
  colorShift: number;
}

interface CopcSource { source: ByteSource; size: number; head: Uint8Array }

const ready = init().then((wasm) => {
  // Optimize the heavy kernels here and on the pool before the first file
  // arrives (see `warmUp`); requests wait for this one.
  warmUp();
  void warmUpPool();
  return wasm;
});
/** Larger clouds are voxel-thinned before meshing, to bound time and memory. */
const MESH_MAX_POINTS = 5_000_000;
const items = new Map<number, Item>();
let nextId = 1;

/**
 * Split a C2C job into spatially compact parts and run them on the worker
 * pool. Returns null when the job is too small for the split to pay off.
 */
async function parallelCloudToCloud(
  compared: Cloud,
  reference: Cloud,
): Promise<{ distances: Float64Array; workers: number } | null> {
  const parts = Math.min(poolSize(), Math.floor(compared.length / MIN_PARALLEL_QUERIES));
  if (parts < 2) return null;
  const plan = planCloudToCloud(compared, reference, parts);
  const slices: Extract<Slice, { kind: "cloud" }>[] = Array.from({ length: plan.length }, (_, k) => ({
    kind: "cloud",
    reference: plan.takeReference(k),
    queries: plan.takeQueries(k),
  }));
  const indices = Array.from({ length: plan.length }, (_, k) => plan.takeQueryIndices(k));
  plan.free();
  const results = await runSlices(slices.length, (k) => slices[k]);
  const distances = new Float64Array(compared.length);
  results.forEach((part, k) => {
    const idx = indices[k];
    for (let i = 0; i < idx.length; i++) distances[idx[i]] = part[i];
  });
  return { distances, workers: results.length };
}

/**
 * C2M on the worker pool: each worker gets the whole mesh and a contiguous
 * slice of the compared points (octree order keeps slices spatially compact).
 */
async function parallelCloudToMesh(
  compared: Cloud,
  mesh: Mesh,
  signed: boolean,
): Promise<{ distances: Float64Array; workers: number } | null> {
  const parts = Math.min(poolSize(), Math.floor(compared.length / MIN_PARALLEL_QUERIES));
  if (parts < 2) return null;
  const positions = compared.rawPositions();
  const vertices = mesh.rawVertices();
  const indices = mesh.indices();
  const per = Math.ceil(compared.length / parts);
  const results = await runSlices(parts, (k) => ({
    kind: "mesh" as const,
    vertices,
    indices,
    queries: positions.slice(k * per * 3, Math.min(compared.length, (k + 1) * per) * 3),
    signed,
  }));
  const distances = new Float64Array(compared.length);
  results.forEach((part, k) => distances.set(part, k * per));
  return { distances, workers: parts };
}

let sorJobs = 0;

/**
 * SOR on the worker pool (see `planSor`): each part settles most of its
 * points on its own; points whose neighbourhood reaches into other parts
 * are completed by the workers still holding those parts. Returns the
 * statistic per point, identical to the single-threaded one, or null when
 * the job is too small to split.
 */
async function parallelSorMeans(cloud: Cloud, k: number): Promise<Float64Array | null> {
  const n = cloud.length;
  // One part per worker: each keeps its part indexed for step 2.
  const parts = Math.min(poolSize(), Math.floor(n / MIN_PARALLEL_QUERIES));
  if (parts < 2 || n <= k) return null;
  const job = ++sorJobs;
  const plan = planSor(cloud, parts);
  const m = plan.length;
  const regions = plan.regions();
  const indices = Array.from({ length: m }, (_, j) => plan.indices(j));
  const lanes: number[] = [];
  try {
    const local = await runSlices(
      m,
      (j) => ({ kind: "sor-local" as const, job, points: plan.points(cloud, j), k, own: j, regions: regions.slice() }),
      lanes,
    );
    const means = new Float64Array(n);
    // Candidate lists per open point, and the queries each part must answer.
    const lists: Float64Array[][] = [];
    const openIndex: number[] = [];
    const asked: { query: number; at: [number, number, number] }[][] = Array.from({ length: m }, () => []);
    local.forEach((part, i) => {
      const idx = indices[i];
      for (let j = 0; j < part.means.length; j++) if (!Number.isNaN(part.means[j])) means[idx[j]] = part.means[j];
      for (let o = 0; o < part.open.length; o++) {
        const q = lists.length;
        lists.push([part.candidates.subarray(o * (k + 1), (o + 1) * (k + 1))]);
        openIndex.push(idx[part.open[o]]);
        const [x, y, z] = part.openPoints.subarray(o * 3, o * 3 + 3);
        for (let j = 0; j < m; j++) if ((part.reach[o] >>> j) & 1) asked[j].push({ query: q, at: [x, y, z] });
      }
    });
    const answers = await Promise.all(
      asked.map((a, j) =>
        a.length === 0
          ? null
          : runOn(lanes[j], {
              kind: "sor-within" as const,
              job,
              queries: Float64Array.from(a.flatMap((x) => x.at)),
              k,
            }),
      ),
    );
    answers.forEach((d, j) => {
      if (d) asked[j].forEach((a, row) => lists[a.query].push(d.subarray(row * (k + 1), (row + 1) * (k + 1))));
    });
    // Same arithmetic as the core: the k + 1 smallest, skip the point itself.
    lists.forEach((candidates, q) => {
      const all = new Float64Array(candidates.reduce((s, c) => s + c.length, 0));
      let at = 0;
      for (const c of candidates) {
        all.set(c, at);
        at += c.length;
      }
      all.sort();
      let sum = 0;
      for (let j = 1; j <= k; j++) sum += Math.sqrt(all[j]);
      means[openIndex[q]] = sum / k;
    });
    return means;
  } finally {
    plan.free();
    await Promise.all([...new Set(lanes)].map((lane) => runOn(lane, { kind: "sor-release" as const, job })));
  }
}

/**
 * Normals on the worker pool: the cloud is split along its octree and each
 * part is done from its own points (near a part's border, neighbours from
 * across it are not used). Returns false when the cloud is too small to split.
 */
async function parallelNormals(cloud: Cloud, k: number, orientation: "up" | "outward"): Promise<boolean> {
  const n = cloud.length;
  const parts = Math.min(poolSize(), Math.floor(n / MIN_PARALLEL_QUERIES));
  if (parts < 2) return false;
  const plan = planSor(cloud, parts);
  try {
    const how = new Float64Array(cloud.normalsOrientation(orientation));
    const indices = Array.from({ length: plan.length }, (_, j) => plan.indices(j));
    const results = await runSlices(plan.length, (j) => ({
      kind: "normals" as const,
      points: plan.points(cloud, j),
      k,
      orientation: how.slice(),
    }));
    const normals = new Float32Array(n * 3);
    results.forEach((part, j) => {
      const idx = indices[j];
      for (let i = 0; i < idx.length; i++) normals.set(part.subarray(i * 3, i * 3 + 3), idx[i] * 3);
    });
    cloud.setNormals(normals);
    return true;
  } finally {
    plan.free();
  }
}

/** Bytes read per slice when streaming a file. */
const STREAM_CHUNK = 16 << 20;

type Loaded =
  | { kind: "mesh"; mesh: Mesh }
  | {
      kind: "cloud";
      cloud: Cloud;
      keepEvery: number;
      filePoints: number;
      copcLevels?: number;
      /** Scan names of a multi-scan E57, by `source` value. */
      sources?: string[];
      detail?: Detail;
      copcSource?: CopcSource;
    };

/** COPC nodes decoded per pool job. */
const COPC_JOB_POINTS = 500_000;

/**
 * Read a COPC file: every octree level while the points so far fit
 * `maxPoints` (at least the root level), so the density stays even. The
 * nodes are fetched and decoded on the worker pool.
 */
async function readCopc(
  source: ByteSource,
  size: number,
  maxPoints: number,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<Loaded> {
  progress("reading the COPC header");
  let head = await readRange(source, 0, Math.min(size, 1 << 16));
  const needed = CopcReader.headerLength(head) ?? head.length;
  if (!Number.isSafeInteger(needed) || needed > size || needed > 8 * 1024 * 1024)
    throw new Error("COPC header exceeds the 8 MiB metadata limit or file size");
  head = needed > head.length ? await readRange(source, 0, needed) : head.slice(0, needed);
  const reader = CopcReader.open(head);
  try {
    let level = 0;
    let chosen = 0;
    for (;;) {
      progress(`reading the hierarchy (level ${level})`);
      const pages = reader.pagesFor(level);
      const ranges = Array.from({ length: pages.length / 2 }, (_, k) => [pages[2 * k], pages[2 * k + 1]]);
      const bytes = await Promise.all(ranges.map(([o, s]) => readRange(source, o, s)));
      ranges.forEach(([o], k) => reader.addPage(o, bytes[k]));
      check();
      const points = reader.levelPoints(level);
      if (points === 0 || (level > 0 && chosen + points > maxPoints)) break;
      chosen += points;
      level++;
      if (!reader.deeperThan(level - 1)) break;
    }
    const found = reader.nodesTo(level - 1);
    // In file order, so each job reads a few contiguous ranges.
    const order = Array.from({ length: found.length / 3 }, (_, k) => k).sort((a, b) => found[3 * a] - found[3 * b]);
    const nodes = order.flatMap((k) => [found[3 * k], found[3 * k + 1], found[3 * k + 2]]);
    // Group nodes into jobs of about COPC_JOB_POINTS points.
    const jobs: number[][] = [];
    let current: number[] = [];
    let inJob = 0;
    for (let k = 0; k < nodes.length; k += 3) {
      current.push(nodes[k], nodes[k + 1], nodes[k + 2]);
      inJob += nodes[k + 2];
      if (inJob >= COPC_JOB_POINTS) {
        jobs.push(current);
        [current, inJob] = [[], 0];
      }
    }
    if (current.length) jobs.push(current);
    progress(`reading ${chosen.toLocaleString()} points (levels 0–${level - 1})`);
    const results = await runSlices(jobs.length, (j) => ({
      kind: "copc-nodes" as const,
      source,
      head: head.slice(),
      nodes: Float64Array.from(jobs[j]),
    }));
    check();
    for (const r of results) reader.addDecoded(r.positions, r.colors ?? undefined, r.intensity, r.classification);
    const filePoints = reader.totalPoints;
    const cloud = reader.finish();
    return { kind: "cloud", cloud, keepEvery: 1, filePoints, copcLevels: level, copcSource: { source, size, head } };
  } catch (err) {
    reader.free();
    throw err;
  }
}

/** Read one node at a time into a capped, full-density working cloud. */
async function readFullDensityBox(
  req: Extract<Request, { kind: "copc-box" }>,
  progress: (note: string, fraction?: number) => void, check: () => void, signal: AbortSignal,
): Promise<{ value: LoadedCloud; transfer: Transferable[] }> {
  const item = items.get(req.id);
  if (item?.kind !== "cloud" || !item.copcSource) throw new Error("no original COPC source for this cloud");
  const snapshot = item.copcSource;
  if ("url" in snapshot.source && !snapshot.source.etag)
    throw new Error("full-density HTTP selection requires an exposed strong ETag; open the local COPC file instead");
  if (!Number.isSafeInteger(req.maxPoints) || req.maxPoints < 1 || req.maxPoints > 1_000_000)
    throw new Error("selection point limit must be in 1..1000000");
  const valid = () => {
    check();
    const current = items.get(req.id);
    if (current?.kind !== "cloud" || current.copcSource !== snapshot)
      throw new Error("original COPC cloud was removed or moved during selection");
  };
  valid();
  let reader: CopcBoxReader | undefined = CopcBoxReader.open(snapshot.head, snapshot.size,
    new Float64Array([...req.min, ...req.max]), req.maxPoints);
  let cloud: Cloud | undefined;
  let sourceReadBytes = 0;
  let nodes = 0;
  const start = performance.now();
  try {
    for (;;) {
      valid();
      const next = reader.nextItem();
      if (!next.length) break;
      const [kind, offset, size] = next;
      const bytes = await readRange(snapshot.source, offset, size, signal);
      valid();
      if (bytes.length !== size) throw new Error("original COPC range is truncated");
      sourceReadBytes += bytes.length;
      if (kind === 0) reader.supplyPage(offset, bytes);
      else { reader.supplyNode(offset, bytes); nodes++; }
      progress(`full-density box: ${reader.selectedPoints.toLocaleString()} points, ${nodes} nodes`);
    }
    // finish consumes its WASM object, including on errors.
    const finished = reader;
    reader = undefined;
    cloud = finished.finish();
    valid();
    const parse = performance.now() - start;
    progress(`indexing ${cloud.length.toLocaleString()} full-density points`);
    const indexStart = performance.now();
    await buildIndex(cloud);
    valid();
    const id = nextId++;
    items.set(id, { kind: "cloud", cloud, name: `${item.name.replace(/\.[^.]+$/, "")}_full-density-box`,
      copcBox: { sourcePoints: item.filePoints ?? 0, sourceReadBytes, nodes } });
    try {
      const output = describe(id, { parse, index: performance.now() - indexStart });
      cloud = undefined;
      return output;
    } catch (error) {
      items.delete(id);
      throw error;
    }
  } finally {
    reader?.free();
    cloud?.free();
  }
}

/** LAS/LAZ chunks decoded per pool job. */
const LAS_JOB_POINTS = 500_000;

/**
 * Read a plain LAS/LAZ file chunk by chunk on the worker pool (see
 * `LasReader`), keeping every n-th point above `maxPoints`. Records each
 * chunk's bounds on the way, so a thinned cloud can later show any part of
 * the file at full density. Returns null when the file cannot be read in
 * chunks (not LAS, or LAZ without a chunk table).
 */
async function readLas(
  source: ByteSource,
  size: number,
  head: Uint8Array,
  maxPoints: number,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<Loaded | null> {
  const reader = LasReader.open(head, size);
  if (!reader) return null;
  let chunks: Float64Array;
  try {
    for (let need = reader.needs(); need.length; need = reader.needs()) {
      reader.supply(await readRange(source, need[0], need[1]));
    }
    chunks = reader.chunks();
  } catch {
    // E.g. LAZ whose writer left no chunk table: read it whole.
    reader.free();
    return null;
  }
  try {
    const filePoints = reader.totalPoints;
    const keepEvery = filePoints > maxPoints ? Math.ceil(filePoints / maxPoints) : 1;
    const n = chunks.length / 4;
    // Jobs of consecutive chunks, so each reads one range.
    const jobs: [number, number][] = [];
    for (let k = 0, points = 0; k < n; k++) {
      if (points === 0) jobs.push([k, k + 1]);
      else jobs[jobs.length - 1][1] = k + 1;
      points += chunks[k * 4 + 2];
      if (points >= LAS_JOB_POINTS) points = 0;
    }
    const dataHead = head.slice(0, reader.dataOffset);
    const bounds = new Float64Array(n * 6);
    // Results arrive out of order; add them to the cloud in file order.
    const waiting = new Map<number, LasChunksResult>();
    let added = 0;
    let read = 0;
    const note = keepEvery > 1 ? ` (keeping 1 in ${keepEvery})` : "";
    progress(`reading 0%${note}`, 0);
    await eachSlice(
      jobs.length,
      (j) => ({
        kind: "las-chunks" as const,
        source,
        head: dataHead.slice(),
        chunks: chunks.slice(jobs[j][0] * 4, jobs[j][1] * 4),
        keepEvery,
      }),
      (j, result) => {
        check();
        bounds.set(result.bounds, jobs[j][0] * 6);
        waiting.set(j, result);
        for (let r = waiting.get(added); r; r = waiting.get(added)) {
          reader.addDecoded(r.positions, r.colors ?? undefined, r.intensity, r.classification);
          waiting.delete(added++);
        }
        for (let k = jobs[j][0]; k < jobs[j][1]; k++) read += chunks[k * 4 + 2];
        const fraction = read / Math.max(1, filePoints);
        progress(`reading ${Math.round(fraction * 100)}%${note}`, fraction);
      },
    );
    const detail =
      keepEvery > 1 ? { source, head: dataHead, chunks, bounds, colorShift: reader.colorShift } : undefined;
    const cloud = reader.finish();
    return { kind: "cloud", cloud, keepEvery, filePoints, detail };
  } catch (err) {
    reader.free();
    throw err;
  }
}

/**
 * Read a file: plain LAS/LAZ chunk by chunk on the pool and other
 * fixed-record formats (binary PLY/PCD) slice by slice, so they are never
 * held whole; read anything else (meshes, text, E57) at once. Clouds larger
 * than `maxPoints` keep every n-th point.
 */
async function readPoints(
  source: ByteSource,
  name: string,
  size: number,
  maxPoints: number,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<Loaded> {
  const thinning = (points: number) => (points > maxPoints ? Math.ceil(points / maxPoints) : 1);
  progress("reading header");
  let head = await readRange(source, 0, Math.min(size, 1 << 16));
  const want = LasReader.headerLength(head) ?? StreamLoader.headerLength(name, head);
  if (size > head.length && (want === undefined || want > head.length)) {
    head = await readRange(source, 0, Math.min(size, Math.max(1 << 22, want ?? 0)));
  }
  if (CopcReader.isCopc(head)) return readCopc(source, size, maxPoints, progress, check);
  const las = await readLas(source, size, head, maxPoints, progress, check);
  if (las) return las;
  const headerLength = StreamLoader.headerLength(name, head);
  const loader = headerLength !== undefined ? StreamLoader.open(name, head.subarray(0, headerLength)) : undefined;
  if (loader) {
    const filePoints = loader.totalPoints;
    const keepEvery = thinning(filePoints);
    loader.setKeepEvery(keepEvery);
    const start = loader.dataOffset;
    try {
      for (let at = start; at < size; at += STREAM_CHUNK) {
        const fraction = (at - start) / Math.max(1, size - start);
        progress(
          `reading ${Math.round(fraction * 100)}%${keepEvery > 1 ? ` (keeping 1 in ${keepEvery})` : ""}`,
          fraction,
        );
        const chunk = await readRange(source, at, Math.min(STREAM_CHUNK, size - at));
        check();
        loader.push(chunk);
      }
    } catch (err) {
      loader.free();
      throw err;
    }
    const cloud = loader.finish();
    loader.free();
    return { kind: "cloud", cloud, keepEvery, filePoints };
  }
  progress("reading");
  const bytes = await readRange(source, 0, size);
  check();
  progress("parsing");
  const mesh = Mesh.parse(name, bytes);
  if (mesh) return { kind: "mesh", mesh };
  const announced = announcedPoints(name, bytes);
  const keepEvery = announced === undefined ? 1 : thinning(announced);
  const cloud = Cloud.parseThinned(name, bytes, keepEvery);
  // Each E57 scan is a `source`, so the cloud splits back into its scans.
  let sources: string[] | undefined;
  if (/\.e57$/i.test(name) && cloud.splitValues("source")) {
    const base = name.replace(/\.[^.]+$/, "");
    sources = e57ScanNames(bytes).map((scan, k) => scan || `${base}_scan${k + 1}`);
  }
  return { kind: "cloud", cloud, keepEvery, filePoints: announced ?? cloud.length, sources };
}

/** Clouds at least this large build their octree on the worker pool. */
const MIN_PARALLEL_INDEX = 1_000_000;
const BUCKETS = 64;

/**
 * Build the octree on the worker pool, in the three steps of
 * `Cloud.indexCube`: sort slices of the cloud by bucket, build each bucket
 * (largest first, so the slowest job starts early), then place them.
 */
async function buildIndex(cloud: Cloud): Promise<number> {
  const n = cloud.length;
  if (n < MIN_PARALLEL_INDEX || poolSize() < 2) {
    cloud.buildIndex();
    return 1;
  }
  const cube = new Float64Array(cloud.indexCube());
  const parts = poolSize();
  const per = Math.ceil(n / parts);
  const chunks = await runSlices(parts, (k) => ({
    kind: "bucket-chunk" as const,
    positions: cloud.positionsRange(k * per, (k + 1) * per),
    colors: cloud.colorsRange(k * per, (k + 1) * per) ?? null,
    cube: cube.slice(),
  }));
  const hasColors = chunks[0].colors !== null;
  // Where bucket b starts within chunk k.
  const starts = chunks.map((c) => {
    const s = new Uint32Array(BUCKETS + 1);
    for (let b = 0; b < BUCKETS; b++) s[b + 1] = s[b] + c.counts[b];
    return s;
  });
  const sizes = new Uint32Array(BUCKETS);
  for (const c of chunks) for (let b = 0; b < BUCKETS; b++) sizes[b] += c.counts[b];
  const keys = Array.from({ length: BUCKETS }, (_, b) => b)
    .filter((b) => sizes[b] > 0)
    .sort((a, b) => sizes[b] - sizes[a]);
  // Cloud index of each bucket's points, in bucket order.
  const origins = new Map<number, Uint32Array>();
  const results = await runSlices(keys.length, (i) => {
    const b = keys[i];
    const positions = new Float64Array(sizes[b] * 3);
    const colors = hasColors ? new Uint8Array(sizes[b] * 3) : null;
    const origin = new Uint32Array(sizes[b]);
    let at = 0;
    chunks.forEach((c, k) => {
      const [from, to] = [starts[k][b], starts[k][b + 1]];
      positions.set(c.positions.subarray(from * 3, to * 3), at * 3);
      if (colors && c.colors) colors.set(c.colors.subarray(from * 3, to * 3), at * 3);
      for (let j = from; j < to; j++) origin[at + j - from] = c.order[j] + k * per;
      at += to - from;
    });
    origins.set(b, origin);
    return { kind: "bucket" as const, positions, colors, cube: cube.slice(), key: b };
  });
  const rootKept = new Uint32Array(BUCKETS);
  const level1Kept = new Uint32Array(BUCKETS);
  results.forEach((r, i) => {
    rootKept[keys[i]] = r.counts[0];
    level1Kept[keys[i]] = r.counts[1];
  });
  cloud.beginBuckets(cube, sizes, rootKept, level1Kept);
  results.forEach((r, i) => {
    const origin = origins.get(keys[i])!;
    const order = new Uint32Array(r.order.length);
    for (let j = 0; j < order.length; j++) order[j] = origin[r.order[j]];
    cloud.putBucket(keys[i], r.positions, r.colors ?? undefined, order, r.nodes);
  });
  cloud.finishBuckets();
  return Math.min(poolSize(), keys.length);
}

/** Everything the UI needs to draw an item; buffers are listed for transfer. */
function describe(
  id: number,
  timings: Omit<LoadedCloud["timings"], "prepare"> = { parse: 0, index: 0 },
): { value: LoadedCloud; transfer: Transferable[] } {
  const start = performance.now();
  const item = items.get(id)!;
  // Each item gets its own shift: one shared shift taken from the first file
  // would leave a UTM cloud opened after a local one in float32 metres.
  const shift = Array.from((item.kind === "mesh" ? item.mesh : item.cloud).suggestedShift()) as Vec3;
  const s = new Float64Array(shift);
  if (item.kind === "mesh") {
    const positions = item.mesh.positions(s);
    const indices = item.mesh.indices();
    const value: LoadedCloud = {
      kind: "mesh",
      id,
      name: item.name,
      count: item.mesh.vertexCount,
      triangles: item.mesh.triangleCount,
      positions,
      indices,
      colors: null,
      intensity: null,
      classification: null,
      normals: null,
      opacity: null,
      sources: null,
      scalarNames: [],
      bounds: Array.from(item.mesh.bounds()),
      shift,
      lodNodes: new Float64Array(),
      lodGrid: 0,
      keepEvery: 1,
      filePoints: item.mesh.vertexCount,
      copcLevels: null,
      detailChunks: null,
      timings: { ...timings, prepare: performance.now() - start },
    };
    return { value, transfer: [positions.buffer, indices.buffer] };
  }
  const { cloud } = item;
  const positions = cloud.positions(s);
  const colors = cloud.colors() ?? null;
  const intensity = cloud.intensity() ?? null;
  const classification = cloud.classification() ?? null;
  const normals = cloud.normals() ?? null;
  const opacity = cloud.attribute("opacity") ?? null;
  const lodNodes = cloud.lodNodes();
  let detailChunks: Float64Array | null = null;
  if (item.detail) {
    const { bounds, chunks } = item.detail;
    const n = chunks.length / 4;
    detailChunks = new Float64Array(n * 7);
    for (let k = 0; k < n; k++) {
      detailChunks.set(bounds.subarray(k * 6, k * 6 + 6), k * 7);
      detailChunks[k * 7 + 6] = chunks[k * 4 + 2];
    }
  }
  const value: LoadedCloud = {
    kind: "cloud",
    id,
    name: item.name,
    count: cloud.length,
    triangles: 0,
    positions,
    indices: null,
    colors,
    intensity,
    classification,
    normals,
    opacity,
    sources: item.sources ?? null,
    scalarNames: cloud.scalarNames(),
    bounds: Array.from(cloud.bounds()),
    shift,
    lodNodes,
    lodGrid: cloud.lodGrid,
    keepEvery: item.keepEvery ?? 1,
    filePoints: item.filePoints ?? cloud.length,
    copcLevels: item.copcLevels ?? null,
    copcBoxAvailable: Boolean(item.copcSource),
    copcBox: item.copcBox,
    detailChunks,
    timings: { ...timings, prepare: performance.now() - start },
  };
  const transfer: Transferable[] = [positions.buffer, lodNodes.buffer];
  if (detailChunks) transfer.push(detailChunks.buffer);
  for (const buffer of [colors, intensity, classification, normals, opacity]) if (buffer) transfer.push(buffer.buffer);
  return { value, transfer };
}

function getCloud(id: number): Cloud {
  const item = items.get(id);
  if (!item) throw new Error("cloud not found");
  if (item.kind !== "cloud") throw new Error(`${item.name} is a mesh, not a point cloud`);
  return item.cloud;
}

function output(
  result: ReturnType<typeof cloudToCloud>,
  start: number,
  workers: number,
  kind: C2cOutput["kind"],
  signed: boolean,
): { value: C2cOutput; transfer: Transferable[] } {
  const distances = result.distances();
  const value: C2cOutput = {
    kind,
    signed,
    distances,
    stats: {
      count: result.count,
      min: result.min,
      max: result.max,
      mean: result.mean,
      rms: result.rms,
      stdDev: result.stdDev,
      median: result.median,
    },
    millis: performance.now() - start,
    workers,
  };
  result.free();
  return { value, transfer: [distances.buffer] };
}

/** One new cloud per class code or `source` value of a cloud. */
async function splitItem(
  item: Extract<Item, { kind: "cloud" }>,
  by: "classification" | "source",
): Promise<{ value: LoadedCloud[]; transfer: Transferable[] }> {
  const values = item.cloud.splitValues(by);
  if (!values) throw new Error(`${item.name} has no ${by} to split by`);
  const base = item.name.replace(/\.[^.]+$/, "");
  const out: LoadedCloud[] = [];
  const transfer: Transferable[] = [];
  for (const v of values) {
    const t = performance.now();
    const part = item.cloud.splitPart(by, v);
    await buildIndex(part);
    const id = nextId++;
    const name = by === "source" ? (item.sources?.[v] ?? `${base}_part${v}`) : `${base}_class${v}`;
    items.set(id, { kind: "cloud", cloud: part, name });
    const described = describe(id, { parse: 0, index: performance.now() - t });
    out.push(described.value);
    transfer.push(...described.transfer);
  }
  return { value: out, transfer };
}

const SEGMENT_KINDS: Segment["kind"][] = ["plane", "sphere", "cylinder", "cluster", "other clusters", "rest"];

/**
 * Shapes (RANSAC) or clusters of a cloud: one cloud whose `source` is the
 * segment of each point, or split into a cloud per segment.
 */
async function shapes(
  req: Extract<Request, { kind: "shapes" }>,
): Promise<{ value: SegmentOutput; transfer: Transferable[] }> {
  const start = performance.now();
  const source = items.get(req.id)!;
  const cloud = getCloud(req.id);
  let normals: Float32Array | null = null;
  if (req.method !== "cluster" && !cloud.normals()) {
    // Shapes are sampled with normals: estimate them on the pool and keep them.
    if (!(await parallelNormals(cloud, 12, "up"))) cloud.estimateNormals(12, "up");
    normals = cloud.normals()!;
  }
  const result =
    req.method === "cluster"
      ? cloud.clusters(req.distance, req.minPoints, !req.split)
      : cloud.detectShapes(req.method, req.distance, req.minPoints, req.maxShapes, !req.split);
  const kinds = result.kinds();
  const counts = result.counts();
  const params = result.params();
  const rms = result.rms();
  const colors = result.colors();
  const segments: Segment[] = Array.from(kinds, (k, i) => ({
    kind: SEGMENT_KINDS[k],
    count: counts[i],
    params: Array.from(params.subarray(i * 8, i * 8 + 8)),
    rms: rms[i],
    color: Array.from(colors.subarray(i * 3, i * 3 + 3)) as Vec3,
  }));
  const found = result.found;
  const segmented = result.takeCloud();
  result.free();
  const base = source.name.replace(/\.[^.]+$/, "");
  const names = segments.map((s, i) =>
    s.kind === "rest"
      ? `${base}_${req.method === "cluster" ? "noise" : "rest"}`
      : s.kind === "other clusters"
        ? `${base}_clusters${i + 1}+`
        : `${base}_${s.kind}${i + 1}`,
  );
  const value: SegmentOutput = { clouds: [], segments, found, normals, millis: 0 };
  const transfer: Transferable[] = normals ? [normals.buffer] : [];
  if (found === 0) {
    segmented.free();
  } else {
    const item = { kind: "cloud" as const, cloud: segmented, name: `${base}_${req.method}s`, sources: names };
    if (req.split) {
      const parts = await splitItem(item, "source");
      segmented.free();
      value.clouds = parts.value;
      transfer.push(...parts.transfer);
    } else {
      await buildIndex(segmented);
      const id = nextId++;
      items.set(id, item);
      const described = describe(id);
      value.clouds = [described.value];
      transfer.push(...described.transfer);
    }
  }
  value.millis = performance.now() - start;
  return { value, transfer };
}

/** A trajectory file's poses, or null when the file is not a trajectory. */
async function readTrajectory(file: File): Promise<TrajectoryPoses | null> {
  // TUM and CSV share extensions with point clouds: judge by the first line.
  const format = TrajectoryData.detect(file.name, await file.slice(0, 4096).text());
  if (!format) return null;
  const data = TrajectoryData.parse(await file.text(), format);
  const poses: TrajectoryPoses = {
    format: format as TrajectoryPoses["format"],
    timestamps: data.timestamps(),
    positions: data.positions(),
    orientations: data.orientations() ?? null,
  };
  data.free();
  return poses;
}

function evaluate(req: Extract<Request, { kind: "trajectory-eval" }>): TrajectoryEvaluation {
  const data = (p: TrajectoryPoses) => new TrajectoryData(p.timestamps, p.positions, p.orientations ?? new Float64Array());
  const [estimate, reference] = [data(req.estimate), data(req.reference)];
  try {
    const out = evaluateTrajectory(estimate, reference, req.maxTimeDelta, req.alignment, req.delta, req.deltaUnit);
    const stats = (name: string) => {
      const s = out.stats(name);
      if (!s) return null;
      const [count, rmse, mean, median, std, min, max] = s;
      return { count, rmse, mean, median, std, min, max };
    };
    const ateRotation = out.values("ate_rotation");
    const value: TrajectoryEvaluation = {
      timestamps: out.timestamps(),
      estimate: out.estimate(),
      reference: out.reference(),
      ate: out.values("ate")!,
      ateRotation: ateRotation ?? null,
      stats: {
        ate: stats("ate")!,
        ateRotation: stats("ate_rotation"),
        rpe: stats("rpe"),
        rpeRotation: stats("rpe_rotation"),
        rpePercent: stats("rpe_percent"),
      },
      matrix: Array.from(out.matrix()),
      scale: out.scale,
      endpointDrift: out.endpointDrift,
      referenceLength: out.referenceLength,
      estimateLength: out.estimateLength,
    };
    out.free();
    return value;
  } finally {
    estimate.free();
    reference.free();
  }
}

/** The file no longer matches a cloud whose points moved. */
function dropDetail(id: number): void {
  const item = items.get(id);
  if (item?.kind === "cloud") { item.detail = undefined; item.copcSource = undefined; }
}

/** Decode one chunk of a thinned cloud's file at full density, on the pool. */
async function readDetail(
  req: Extract<Request, { kind: "detail" }>,
): Promise<{ value: DetailChunk; transfer: Transferable[] }> {
  const item = items.get(req.id);
  const detail = item?.kind === "cloud" ? item.detail : undefined;
  if (!detail) throw new Error("no full-density file for this cloud");
  const k = req.chunk;
  const r = await runAny({
    kind: "las-chunks",
    source: detail.source,
    head: detail.head.slice(),
    chunks: detail.chunks.slice(k * 4, k * 4 + 4),
    keepEvery: 1,
  });
  const n = r.intensity.length;
  const [sx, sy, sz] = req.shift;
  const bounds = detail.bounds.subarray(k * 6, k * 6 + 6);
  // Slices along the longer horizontal side, about as many as it is longer
  // than the shorter one: a flight-line strip becomes squarish pieces.
  const axis = bounds[3] - bounds[0] >= bounds[4] - bounds[1] ? 0 : 1;
  const long = bounds[3 + axis] - bounds[axis];
  const short = bounds[4 - axis] - bounds[1 - axis];
  const slices = Math.max(1, Math.min(MAX_SLICES, Math.round(long / Math.max(short, long / MAX_SLICES, 1e-9))));
  const slice = (i: number) => Math.min(slices - 1, Math.floor(((r.positions[i * 3 + axis] - bounds[axis]) / long) * slices) || 0);
  const starts = new Uint32Array(slices + 1);
  for (let i = 0; i < n; i++) starts[slice(i) + 1]++;
  for (let s = 0; s < slices; s++) starts[s + 1] += starts[s];
  const next = starts.slice(0, slices);
  const positions = new Float32Array(n * 3);
  const colors = r.colors ? new Uint8Array(n * 3) : null;
  const intensity = new Float32Array(n);
  const classification = new Uint8Array(n);
  const pieces = new Float64Array(slices * 7);
  for (let s = 0; s < slices; s++) pieces.set([0, Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity], s * 7);
  for (let i = 0; i < n; i++) {
    const s = slice(i);
    const j = next[s]++;
    const xyz = [r.positions[i * 3] - sx, r.positions[i * 3 + 1] - sy, r.positions[i * 3 + 2] - sz];
    positions.set(xyz, j * 3);
    const piece = s * 7;
    pieces[piece]++;
    for (let a = 0; a < 3; a++) {
      pieces[piece + 1 + a] = Math.min(pieces[piece + 1 + a], xyz[a]);
      pieces[piece + 4 + a] = Math.max(pieces[piece + 4 + a], xyz[a]);
    }
    if (colors && r.colors) {
      for (let c = 0; c < 3; c++) colors[j * 3 + c] = r.colors[i * 3 + c] >> detail.colorShift;
    }
    intensity[j] = r.intensity[i];
    classification[j] = r.classification[i];
  }
  const value: DetailChunk = {
    positions,
    colors,
    intensity,
    classification,
    pieces: pieces.filter((_, i) => pieces[i - (i % 7)] > 0),
  };
  const transfer: Transferable[] = [positions.buffer, intensity.buffer, classification.buffer, value.pieces.buffer];
  if (colors) transfer.push(colors.buffer);
  return { value, transfer };
}

/** Most slices a full-density chunk is cut into (see `readDetail`). */
const MAX_SLICES = 16;

/** The open pose graph, if any (one at a time). */
let poseGraph: { session: PoseGraphSession; name: string } | null = null;

/** The vector map, made on first use. */
let vectorMap: VectorMapSession | null = null;

/** A "vm" request's answer: JSON text, or for edits `{result, view}` with the view after the edit. */
function vectorMapRequest(req: Extract<Request, { kind: "vm" }>): string {
  const map = (vectorMap ??= new VectorMapSession());
  const withView = (result: string) => `{"result":${result},"view":${map.view()},"undo":${map.undoDepth}}`;
  switch (req.op) {
    case "open":
      return withView(map.open(req.name ?? "", req.text ?? ""));
    case "apply":
      return withView(map.apply(req.text ?? "[]"));
    case "build":
      if (req.id === undefined || !req.positions) throw new Error("Choose a point cloud and trajectory.");
      return withView(map.buildFromTrajectory(getCloud(req.id), req.positions, req.text ?? "{}"));
    case "junction-preview":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return map.previewJunctions(getCloud(req.id), req.text ?? "{}");
    case "signal-preview":
    case "signal-add":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return withView(map.measureSignal(getCloud(req.id), req.text ?? "{}", req.op === "signal-preview"));
    case "junction-connect":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return withView(map.connectJunctions(getCloud(req.id), req.text ?? "{}", JSON.stringify(req.pairs ?? null)));
    case "undo":
      return withView(String(map.undo()));
    case "clear":
      map.clear();
      return withView("null");
    case "view":
      return withView("null");
    case "validate":
      return map.validate(req.autoware ?? true);
    case "export":
      return map.exportLanelet2(req.autoware ?? true);
    case "nearest":
      return map.nearestLane(req.x ?? 0, req.y ?? 0);
    case "json":
      return map.toJson();
  }
}

function openGraph(): PoseGraphSession {
  if (!poseGraph) throw new Error("no pose graph is open");
  return poseGraph.session;
}

function graphState(session: PoseGraphSession): PoseGraphState {
  return {
    nodeIds: session.nodeIds(),
    poses: session.poses(),
    edges: session.edgeEnds(),
    edgeKinds: session.edgeKinds(),
    edgeErrors: session.edgeErrors(),
    fixed: session.fixedNodes(),
    planes: session.planeCount,
    planeEdges: session.planeEdgeCount,
    gravityEdges: session.gravityEdgeCount,
  };
}

const stateTransfer = (s: PoseGraphState): Transferable[] =>
  [s.nodeIds, s.poses, s.edges, s.edgeKinds, s.edgeErrors, s.fixed].map((a) => a.buffer);

/** The last run of digits in a file's base name (`000123.pcd` -> 123). */
function frameNumber(name: string): number | null {
  const digits = name.replace(/\.[^.]+$/, "").match(/\d+(?!.*\d)/);
  return digits ? Number(digits[0]) : null;
}

/**
 * Node index per scan file: by the number in the file name when that names
 * a node id for most files, else in name order when there is one scan per
 * node. Null entries match no node.
 */
function matchScans(files: File[], nodeIds: Float64Array): (number | null)[] {
  const byId = new Map<number, number>();
  nodeIds.forEach((id, i) => byId.set(id, i));
  const numbered = files.map((f) => {
    const n = frameNumber(f.name);
    return n === null ? undefined : byId.get(n);
  });
  const hits = numbered.filter((i) => i !== undefined).length;
  if (hits > 0 && hits >= files.length / 2) return numbered.map((i) => i ?? null);
  if (files.length === nodeIds.length) {
    const order = files.map((_, k) => k).sort((a, b) => files[a].name.localeCompare(files[b].name, undefined, { numeric: true }));
    const out: (number | null)[] = new Array(files.length).fill(null);
    order.forEach((k, i) => (out[k] = i));
    return out;
  }
  throw new Error(
    `cannot match ${files.length} scans to ${nodeIds.length} poses: name the scans by frame number (e.g. 000042.pcd) or give one scan per pose`,
  );
}

interface LoadedGraph {
  session: PoseGraphSession;
  scans: (Float32Array | null)[];
  scanPoints: number;
  unmatched: string[];
  odometry: PoseGraphOpened["odometry"];
}

/** A scan without a pose: from a bag's messages or a file. */
interface Frame {
  id: number;
  stamp: number;
  cloud: Cloud;
}

const POINT_CLOUD = "sensor_msgs/PointCloud2";
const IMU = "sensor_msgs/Imu";

/**
 * A bag's scans (its PointCloud2 topic with the most messages), and its
 * IMU's messages (seven numbers each, see `BagMessage.imu`) into `imu` on
 * the way. The Rust core reads the bag, a slice of the file at a time.
 */
async function* bagFrames(
  file: File,
  imu: number[],
  topics: { scans: string | null; imu: string | null },
  progress: (note: string, fraction?: number) => void,
): AsyncGenerator<Frame> {
  progress(`reading ${file.name}`);
  const reader = new FileReaderSync();
  const bag = new BagFile(file.size, (at: number, length: number) => new Uint8Array(reader.readAsArrayBuffer(file.slice(at, at + length))));
  try {
    const names = bag.topicNames();
    const kinds = bag.topicKinds();
    const counts = bag.topicCounts();
    const busiest = (kind: string) => {
      let best: { topic: string; count: number } | null = null;
      names.forEach((topic, k) => {
        if (kinds[k] === kind && (!best || counts[k] > best.count)) best = { topic, count: counts[k] };
      });
      return best as { topic: string; count: number } | null;
    };
    const scans = busiest(POINT_CLOUD);
    if (!scans) throw new Error(`${file.name} has no sensor_msgs/PointCloud2 messages`);
    const imuTopic = busiest(IMU);
    topics.scans = scans.topic;
    topics.imu = imuTopic?.topic ?? null;
    bag.start([scans.topic, ...(imuTopic ? [imuTopic.topic] : [])]);
    let id = 0;
    for (let m = bag.next(); m; m = bag.next()) {
      try {
        if (m.topic !== scans.topic) {
          imu.push(...m.imu());
          continue;
        }
        progress(`${scans.topic}: scan ${id + 1} of ${scans.count}`, bag.progress());
        const cloud = m.scan();
        yield { id: id++, stamp: m.stamp(), cloud };
      } finally {
        m.free();
      }
    }
  } finally {
    bag.free();
  }
}

/** Scan files in name order, numbered by their names where they have numbers. */
async function* fileFrames(files: File[], progress: (note: string, fraction?: number) => void): AsyncGenerator<Frame> {
  const sorted = files.slice().sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const reads = new Map<number, Promise<ArrayBuffer>>();
  for (const [k, file] of sorted.entries()) {
    for (let j = k; j < Math.min(sorted.length, k + READ_AHEAD); j++) {
      if (!reads.has(j)) reads.set(j, sorted[j].arrayBuffer());
    }
    progress(`scan ${k + 1} of ${sorted.length}`, k / sorted.length);
    const bytes = new Uint8Array(await reads.get(k)!);
    reads.delete(k);
    yield { id: frameNumber(file.name) ?? k, stamp: k, cloud: Cloud.parse(file.name, bytes) };
  }
}

/**
 * Registering a scan with the odometry, the normal equations of each step
 * summed over the worker pool when there is one: each worker keeps a copy
 * of the local map and takes a share of the scan's points. With one core,
 * or no pool, the odometry registers on its own.
 */
async function pooledRegistration(odometry: LidarOdometry): Promise<(cloud: Cloud) => Promise<Float64Array>> {
  const lanes = poolSize();
  if (lanes < 2) return async (cloud) => odometry.register(cloud);
  const all = Array.from({ length: lanes }, (_, lane) => lane);
  await Promise.all(all.map((lane) => runOn(lane, { kind: "odom-reset", voxel: odometry.mapVoxel, maxPoints: odometry.mapPoints })));
  const range = odometry.maxRange;
  return async (cloud) => {
    const registration = odometry.begin(cloud);
    try {
      if (registration.needsAlignment) {
        const source = registration.source();
        const points = source.length / 3;
        const per = Math.ceil(points / lanes);
        await Promise.all(
          all.map((lane) => runOn(lane, { kind: "odom-source", source: source.slice(3 * lane * per, 3 * Math.min(points, (lane + 1) * per)) })),
        );
        const terms = registration.terms();
        for (;;) {
          const total = registration.total();
          const parts = await Promise.all(all.map((lane) => runOn(lane, { kind: "odom-equations", total: total.slice(), terms: terms.slice() })));
          const sum = new Float64Array(42);
          for (const part of parts) for (let k = 0; k < 42; k++) sum[k] += part[k];
          if (odometry.step(registration, sum)) break;
        }
      }
      const pose = odometry.finish(registration);
      const placed = odometry.placed();
      const origin = new Float64Array([pose[3], pose[7], pose[11]]);
      await Promise.all(all.map((lane) => runOn(lane, { kind: "odom-map", placed: placed.slice(), origin: origin.slice(), range })));
      return pose;
    } catch (err) {
      registration.free?.();
      throw err;
    }
  };
}

/**
 * A graph for scans without poses: odometry registers each onto the ones
 * before; a keyframe every so many metres of travel (and the last scan)
 * becomes a node with its scan.
 */
async function odometryGraph(
  req: PoseGraphFiles,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<LoadedGraph> {
  const started = performance.now();
  const o = req.odometry;
  const imu: number[] = [];
  const topics = { scans: null as string | null, imu: null as string | null };
  const frames = req.graph ? bagFrames(req.graph, imu, topics, progress) : fileFrames(req.scans, progress);
  const odometry = new LidarOdometry(o.minRange, o.maxRange, o.deskew);
  const register = await pooledRegistration(odometry);
  const session = PoseGraphSession.empty();
  const scans: Float32Array[] = [];
  const stamps: number[] = [];
  const none = new Float64Array(0);
  let scanPoints = 0;
  let count = 0;
  let pathLength = 0;
  let travel = 0;
  let last: Float64Array | null = null;
  let held: (Frame & { pose: Float64Array }) | null = null;
  const keep = (frame: Frame, pose: Float64Array) => {
    const node = session.addNode(pose, frame.id, req.sigmaT, req.sigmaRDeg);
    scanPoints += session.setScan(node, frame.cloud, req.voxel, none);
    scans.push(session.scanPositions(node, req.displayPoints));
    stamps.push(frame.stamp);
  };
  try {
    for await (const frame of frames) {
      check();
      let pose: Float64Array;
      try {
        pose = await register(frame.cloud);
      } catch (err) {
        frame.cloud.free();
        throw err;
      }
      if (last) {
        const step = Math.hypot(pose[3] - last[3], pose[7] - last[7], pose[11] - last[11]);
        travel += step;
        pathLength += step;
      }
      last = pose;
      count++;
      held?.cloud.free();
      held = null;
      if (count === 1 || travel >= o.keyframeSpacing) {
        try {
          keep(frame, pose);
        } finally {
          frame.cloud.free();
        }
        travel = 0;
      } else {
        // The last scan is always a keyframe: hold this one until the next.
        held = { ...frame, pose };
      }
    }
    if (held) keep(held, held.pose);
    if (count === 0) throw new Error(req.graph ? `${req.graph.name} has no scans` : "no scans");
  } catch (err) {
    session.free();
    throw err;
  } finally {
    held?.cloud.free();
    odometry.free();
  }
  const ups = imu.length ? upsAt(new Float64Array(imu), new Float64Array(stamps), 0.5) : null;
  return {
    session,
    scans,
    scanPoints,
    unmatched: [],
    odometry: {
      frames: count,
      pathLength,
      seconds: (performance.now() - started) / 1000,
      topic: topics.scans,
      imuTopic: topics.imu,
      ups,
    },
  };
}

/**
 * Metres of possible drift between a candidate loop's nodes from which a
 * failed registration is retried as if both scans were taken at the same place.
 */
const RETRY_MIN_DRIFT = 5;

/** Fewest keyframes one pool worker judges for dynamic points: fewer would mostly send neighbours. */
const DYNAMIC_MIN_PART = 20;

/** Scan files read at once while loading a pose graph. */
const READ_AHEAD = 8;

/** Read a graph and its scans into a new session (freed again on failure). */
async function loadGraph(
  req: PoseGraphFiles,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<LoadedGraph> {
  if (!req.graph || isBag(req.graph.name)) return odometryGraph(req, progress, check);
  const text = await req.graph.text();
  let session: PoseGraphSession;
  if (/\.g2o$/i.test(req.graph.name)) {
    session = PoseGraphSession.fromG2o(text);
  } else {
    const format = TrajectoryData.detect(req.graph.name, text.slice(0, 4096));
    if (!format) throw new Error(`${req.graph.name} is neither a g2o file nor a TUM / KITTI trajectory`);
    const trajectory = TrajectoryData.parse(text, format);
    try {
      session = PoseGraphSession.fromTrajectory(trajectory, req.sigmaT, req.sigmaRDeg);
    } finally {
      trajectory.free();
    }
  }
  try {
    const nodeIds = session.nodeIds();
    const nodes = matchScans(req.scans, nodeIds);
    const extrinsic = new Float64Array(req.extrinsic ?? []);
    const scans: (Float32Array | null)[] = new Array(nodeIds.length).fill(null);
    const unmatched: string[] = [];
    let scanPoints = 0;
    // Reading a file is slow next to parsing it: keep a few reads in flight.
    const reads = new Map<number, Promise<ArrayBuffer>>();
    const readAhead = (from: number) => {
      for (let j = from; j < Math.min(req.scans.length, from + READ_AHEAD); j++) {
        if (nodes[j] !== null && !reads.has(j)) reads.set(j, req.scans[j].arrayBuffer());
      }
    };
    for (const [k, file] of req.scans.entries()) {
      check();
      readAhead(k);
      progress(`scan ${k + 1} of ${req.scans.length}`, k / req.scans.length);
      const node = nodes[k];
      if (node === null) {
        unmatched.push(file.name);
        continue;
      }
      const bytes = new Uint8Array(await reads.get(k)!);
      reads.delete(k);
      const cloud = Cloud.parse(file.name, bytes);
      try {
        scanPoints += session.setScan(node, cloud, req.voxel, extrinsic);
      } finally {
        cloud.free();
      }
      scans[node] = session.scanPositions(node, req.displayPoints);
    }
    return { session, scans, scanPoints, unmatched, odometry: null };
  } catch (err) {
    session.free();
    throw err;
  }
}

async function openPoseGraph(
  req: PoseGraphFiles,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<{ value: PoseGraphOpened; transfer: Transferable[] }> {
  const { session, scans, scanPoints, unmatched, odometry } = await loadGraph(req, progress, check);
  poseGraph?.session.free();
  const name = req.graph?.name ?? "scans";
  poseGraph = { session, name: name.replace(/\.[^.]+$/, "") };
  const value: PoseGraphOpened = { ...graphState(session), name, scans, scanPoints, unmatched, odometry };
  const transfer = stateTransfer(value);
  for (const scan of scans) if (scan) transfer.push(scan.buffer);
  if (odometry?.ups) transfer.push(odometry.ups.buffer);
  return { value, transfer };
}

async function mergePoseGraph(
  req: Extract<Request, { kind: "pg-merge" }>,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<{ value: PoseGraphMerged; transfer: Transferable[] }> {
  const session = openGraph();
  const other = await loadGraph(req.files, progress, check);
  try {
    const ids = other.session.nodeIds();
    const b = req.nodeB === null ? 0 : ids.indexOf(req.nodeB);
    if (b < 0) throw new Error(`${req.files.graph?.name ?? "the scans"} has no node ${req.nodeB}`);
    progress("registering the two graphs");
    const [fitness, rms, , offset] = session.merge(
      other.session,
      req.nodeA,
      b,
      req.yawSteps,
      req.maxIterations,
      req.overlap,
      req.inlierDistance,
      req.minFitness,
      req.sigmaT,
      req.sigmaRDeg,
    );
    progress("optimising");
    const [initialCost, finalCost, iterations] = session.optimize(req.loopKernel);
    const state = graphState(session);
    const value: PoseGraphMerged = {
      state,
      name: req.files.graph?.name ?? "scans",
      scans: other.scans,
      scanPoints: other.scanPoints,
      unmatched: other.unmatched,
      odometry: other.odometry,
      offset,
      fitness,
      rms,
      optimized: { initialCost, finalCost, iterations },
    };
    const transfer = stateTransfer(state);
    for (const scan of other.scans) if (scan) transfer.push(scan.buffer);
    if (other.odometry?.ups) transfer.push(other.odometry.ups.buffer);
    return { value, transfer };
  } finally {
    other.session.free();
  }
}

async function handle(
  req: Request,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
  signal: AbortSignal,
): Promise<{ value: unknown; transfer: Transferable[] }> {
  await ready;
  switch (req.kind) {
    case "copc-box":
      return readFullDensityBox(req, progress, check, signal);
    case "pg-open":
      return openPoseGraph(req, progress, check);
    case "pg-merge":
      return mergePoseGraph(req, progress, check);
    case "pg-loop": {
      const session = openGraph();
      const r = session.registerLoop(
        req.from,
        req.to,
        req.maxIterations,
        req.overlap,
        req.pointToPlane,
        req.inlierDistance,
        0.5,
        req.retryHeadings,
      );
      const edge = session.addLoopEdge(req.from, req.to, r.subarray(5, 21), req.sigmaT, req.sigmaRDeg);
      const state = graphState(session);
      const [rmsInitial, rmsFinal, iterations, converged, fitness] = r;
      const value = {
        state,
        edge,
        rmsInitial,
        rmsFinal,
        iterations,
        converged: converged === 1,
        fitness,
        retried: r[21] === 1,
      };
      return { value, transfer: stateTransfer(state) };
    }
    case "pg-register": {
      const session = openGraph();
      const r = registerScans(
        session.scanXyz(req.from),
        session.scanXyz(req.to),
        new Float64Array(req.guess),
        req.maxIterations,
        req.overlap,
        req.inlierDistance,
        0,
        0,
      );
      const [rmsInitial, rmsFinal, , converged, fitness] = r;
      const value = { matrix: Array.from(r.subarray(5, 21)), rmsInitial, rmsFinal, fitness, converged: converged === 1 };
      return { value, transfer: [] };
    }
    case "pg-add-edge": {
      const session = openGraph();
      const edge = session.addLoopEdge(req.from, req.to, new Float64Array(req.matrix), req.sigmaT, req.sigmaRDeg);
      const state = graphState(session);
      return { value: { state, edge }, transfer: stateTransfer(state) };
    }
    case "pg-find-loops": {
      const session = openGraph();
      const found = session.loopCandidates(req.maxDistance, req.drift, req.minTravel, req.spacing);
      const candidates = found.length / 3;
      const added: PoseGraphFound["added"] = [];
      const edges: number[] = [];
      let implausible = 0;
      // Register the candidates on the pool (each needs only its two scans),
      // then keep those that overlap and sit where drift allows, in order.
      const retryFor = (travel: number) =>
        // Only a long way round can leave the graph's guess too far off for ICP;
        // along a short stretch, "the same place" would only match a corridor to itself.
        req.drift * travel >= RETRY_MIN_DRIFT ? req.retryHeadings : 0;
      const results: (Float64Array | null)[] = new Array(candidates).fill(null);
      let checked = 0;
      const report = () =>
        progress(`checked ${checked} of ${candidates} candidates`, checked / Math.max(1, candidates));
      if (poolSize() >= 2 && candidates > 1) {
        await eachSlice(
          candidates,
          (k) => ({
            kind: "loop" as const,
            from: session.scanXyz(found[3 * k]),
            to: session.scanXyz(found[3 * k + 1]),
            guess: Array.from(session.relativePose(found[3 * k], found[3 * k + 1])),
            maxIterations: req.maxIterations,
            overlap: req.overlap,
            inlierDistance: req.inlierDistance,
            retryBelow: req.minFitness,
            retryHeadings: retryFor(found[3 * k + 2]),
          }),
          (k, r) => {
            check();
            results[k] = r;
            checked++;
            report();
          },
        ).catch((err) => {
          // A node without a scan or too few matching points only rules out its pair.
          if (err instanceof Error && err.message === CANCELLED) throw err;
        });
      }
      for (let k = 0; k < candidates; k++) {
        if (results[k]) continue;
        check();
        try {
          results[k] = session.registerLoop(
            found[3 * k],
            found[3 * k + 1],
            req.maxIterations,
            req.overlap,
            true,
            req.inlierDistance,
            req.minFitness,
            retryFor(found[3 * k + 2]),
          );
        } catch {
          // a node without a scan, or too few matching points
        }
        checked++;
        report();
      }
      for (let k = 0; k < candidates; k++) {
        const r = results[k];
        const [from, to, travel] = [found[3 * k], found[3 * k + 1], found[3 * k + 2]];
        if (!r || !(r[4] >= req.minFitness)) continue;
        if (r[22] > req.maxDistance + req.drift * travel) {
          implausible++;
          continue;
        }
        edges.push(session.addLoopEdge(from, to, r.subarray(5, 21), req.sigmaT, req.sigmaRDeg));
        added.push({ from, to, fitness: r[4], retried: r[21] === 1 });
      }
      let optimized: PoseGraphFound["optimized"] = null;
      if (added.length) {
        progress("optimising");
        const [initialCost, finalCost, iterations] = session.optimize(req.loopKernel);
        optimized = { initialCost, finalCost, iterations };
      }
      const state = graphState(session);
      const value: PoseGraphFound = { state, candidates, added, edges, implausible, optimized };
      return { value, transfer: stateTransfer(state) };
    }
    case "pg-floor": {
      const session = openGraph();
      const [first, count, tied, ...up] = session.addFloor(
        new Float64Array(req.up ?? []),
        req.maxTiltDeg,
        req.threshold,
        req.minPoints,
        req.sigmaAngleDeg,
        req.sigmaOffset,
      );
      const [initialCost, finalCost, iterations] = session.optimize(req.loopKernel);
      const state = graphState(session);
      const value: PoseGraphFloor = {
        state,
        planes: { first, count },
        tied,
        up,
        optimized: { initialCost, finalCost, iterations },
      };
      return { value, transfer: stateTransfer(state) };
    }
    case "pg-set-node-pose": {
      const session = openGraph();
      session.setNodePose(req.index, new Float64Array(req.pose), req.carry);
      const state = graphState(session);
      return { value: state, transfer: stateTransfer(state) };
    }
    case "pg-set-fixed": {
      const session = openGraph();
      session.setFixed(req.index, req.fixed);
      const state = graphState(session);
      return { value: state, transfer: stateTransfer(state) };
    }
    case "pg-gravity": {
      const session = openGraph();
      const [tied, spread, calibratedSpread, sigmaDeg] = session.setGravity(
        new Uint32Array(req.nodes),
        req.ups,
        req.sigmaDeg,
        req.calibrate,
      );
      const [initialCost, finalCost, iterations] = session.optimize(req.loopKernel);
      const state = graphState(session);
      const value = { state, tied, spread, calibratedSpread, sigmaDeg, initialCost, finalCost, iterations };
      return { value, transfer: stateTransfer(state) };
    }
    case "pg-clear-gravity": {
      const session = openGraph();
      session.clearGravity();
      const state = graphState(session);
      return { value: state, transfer: stateTransfer(state) };
    }
    case "pg-remove-plane": {
      const session = openGraph();
      session.removePlane(req.index);
      const state = graphState(session);
      return { value: state, transfer: stateTransfer(state) };
    }
    case "pg-optimize": {
      const session = openGraph();
      const start = performance.now();
      const [initialCost, finalCost, iterations, converged] = session.optimize(req.loopKernel);
      const millis = performance.now() - start;
      const state = graphState(session);
      const value = { state, initialCost, finalCost, iterations, converged: converged === 1, millis };
      return { value, transfer: stateTransfer(state) };
    }
    case "pg-remove-edges": {
      const session = openGraph();
      const removed = [...new Set(req.indices)]
        .sort((a, b) => b - a)
        .map((index) => {
          const data = session.edgeData(index);
          session.removeEdge(index);
          return { index, data };
        })
        .reverse();
      if (req.loopKernel !== null) session.optimize(req.loopKernel);
      const state = graphState(session);
      return { value: { state, removed }, transfer: stateTransfer(state) };
    }
    case "pg-insert-edges": {
      const session = openGraph();
      for (const edge of req.edges) session.insertEdge(edge.index, edge.data);
      const state = graphState(session);
      return { value: state, transfer: stateTransfer(state) };
    }
    case "pg-set-poses": {
      const session = openGraph();
      session.setPoses(req.poses);
      const state = graphState(session);
      return { value: state, transfer: stateTransfer(state) };
    }
    case "pg-export":
      return { value: openGraph().export(req.format), transfer: [] };
    case "pg-map": {
      const t = performance.now();
      // Thinned as it is assembled: a joined drive's whole map would not fit.
      const cloud = openGraph().map(req.initial, req.correction, new Uint32Array(req.nodes), req.part, req.voxel);
      await buildIndex(cloud);
      const id = nextId++;
      const part = req.label ? `_${req.label}` : "";
      items.set(id, { kind: "cloud", cloud, name: `${poseGraph!.name}_${req.initial ? "map_start" : "map"}${part}` });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "change-objects": {
      const objects = getCloud(req.id).changedObjects(req.minChange, req.link, req.minPoints);
      return { value: objects, transfer: [objects.buffer] };
    }
    case "pg-dynamic": {
      const start = performance.now();
      const session = openGraph();
      const n = session.nodeCount;
      const lanes = poolSize();
      if (lanes < 2 || n < 2 * DYNAMIC_MIN_PART) {
        const [dynamic, total] = session.detectDynamic(req.window, req.margin, req.votes);
        return { value: { dynamic, total, millis: performance.now() - start }, transfer: [] };
      }
      // Parts of the drive on the pool: each needs its own scans and `window` more on either side.
      const size = Math.max(DYNAMIC_MIN_PART, Math.ceil(n / (3 * lanes)));
      const parts = Math.ceil(n / size);
      let dynamic = 0;
      let total = 0;
      let done = 0;
      await eachSlice(
        parts,
        (k) => {
          const first = k * size;
          const lo = Math.max(0, first - req.window);
          return {
            kind: "dynamic" as const,
            context: session.dynamicContext(lo, Math.min(n, first + size + req.window)),
            first: first - lo,
            count: Math.min(size, n - first),
            window: req.window,
            margin: req.margin,
            votes: req.votes,
          };
        },
        (k, flags) => {
          check();
          dynamic += session.setDynamic(k * size, flags);
          total += flags.length;
          progress(`judged ${++done} of ${parts} parts of the drive`, done / parts);
        },
      );
      return { value: { dynamic, total, millis: performance.now() - start }, transfer: [] };
    }
    case "vm":
      return { value: vectorMapRequest(req), transfer: [] };
    case "pg-close":
      poseGraph?.session.free();
      poseGraph = null;
      return { value: null, transfer: [] };
    case "trajectory":
      return { value: await readTrajectory(req.file), transfer: [] };
    case "trajectory-eval":
      return { value: evaluate(req), transfer: [] };
    case "load":
    case "load-url": {
      const { maxPoints } = req;
      const name = req.kind === "load" ? req.file.name : req.name;
      let t = performance.now();
      const loaded =
        req.kind === "load"
          ? await readPoints({ file: req.file }, name, req.file.size, maxPoints, progress, check)
          : await readPoints({ url: req.url, total: req.size, etag: req.etag }, name, req.size, maxPoints, progress, check);
      const parse = performance.now() - t;
      try {
        check();
      } catch (err) {
        (loaded.kind === "mesh" ? loaded.mesh : loaded.cloud).free();
        throw err;
      }
      if (loaded.kind === "mesh") {
        const id = nextId++;
        items.set(id, { kind: "mesh", mesh: loaded.mesh, name });
        return describe(id, { parse, index: 0 });
      }
      const { cloud } = loaded;
      progress(`indexing ${cloud.length.toLocaleString()} points`);
      t = performance.now();
      const workers = await buildIndex(cloud);
      const index = performance.now() - t;
      try {
        check();
      } catch (err) {
        cloud.free();
        throw err;
      }
      const id = nextId++;
      items.set(id, {
        kind: "cloud",
        cloud,
        name,
        keepEvery: loaded.keepEvery,
        filePoints: loaded.filePoints,
        copcLevels: loaded.copcLevels,
        sources: loaded.sources,
        detail: loaded.detail,
        copcSource: loaded.copcSource,
      });
      progress("preparing for display");
      return describe(id, { parse, index, workers });
    }
    case "c2c": {
      const compared = getCloud(req.compared);
      const reference = items.get(req.reference);
      if (!reference) throw new Error("reference not found");
      const start = performance.now();
      if (reference.kind === "mesh") {
        const parallel = await parallelCloudToMesh(compared, reference.mesh, req.signed);
        const result = parallel
          ? summarizeDistances(parallel.distances)
          : cloudToMesh(compared, reference.mesh, req.signed);
        return output(result, start, parallel?.workers ?? 1, "c2m", req.signed);
      }
      const parallel = await parallelCloudToCloud(compared, reference.cloud);
      const result = parallel
        ? summarizeDistances(parallel.distances)
        : cloudToCloud(compared, reference.cloud);
      return output(result, start, parallel?.workers ?? 1, "c2c", false);
    }
    case "detail":
      return readDetail(req);
    case "point": {
      const item = items.get(req.id);
      const xyz = item?.kind === "cloud" ? item.cloud.point(req.index) : undefined;
      if (!xyz) throw new Error("point not found");
      return { value: Array.from(xyz), transfer: [] };
    }
    case "icp": {
      const moving = getCloud(req.moving);
      const start = performance.now();
      const outcome = registerIcp(
        moving,
        getCloud(req.reference),
        req.maxIterations,
        req.overlap,
        req.matchCentroids,
        req.pointToPlane,
      );
      const matrix = Array.from(outcome.matrix());
      moving.transform(new Float64Array(matrix));
      dropDetail(req.moving);
      const described = describe(req.moving);
      const value: IcpOutput = {
        cloud: described.value,
        matrix,
        rmsInitial: outcome.rmsInitial,
        rmsFinal: outcome.rmsFinal,
        iterations: outcome.iterations,
        converged: outcome.converged,
        millis: performance.now() - start,
      };
      outcome.free();
      return { value, transfer: described.transfer };
    }
    case "transform": {
      getCloud(req.id).transform(new Float64Array(req.matrix));
      dropDetail(req.id);
      return describe(req.id);
    }
    case "volume": {
      const start = performance.now();
      const surface = (side: VolumeSide) => {
        if ("z" in side) return VolumeSurface.constant(side.z);
        const item = items.get(side.id);
        if (!item) throw new Error("surface not found");
        return item.kind === "mesh" ? VolumeSurface.fromMesh(item.mesh) : VolumeSurface.fromCloud(item.cloud);
      };
      const label = (side: VolumeSide) => ("z" in side ? `z${side.z}` : items.get(side.id)!.name.replace(/\.[^.]+$/, ""));
      const before = surface(req.before);
      const after = surface(req.after);
      const out = computeVolume(before, after, req.cell, req.height, req.fillEmpty);
      before.free();
      after.free();
      const value: VolumeOutput = {
        added: out.added,
        removed: out.removed,
        addedArea: out.addedArea,
        removedArea: out.removedArea,
        matchedCells: out.matchedCells,
        totalCells: out.totalCells,
        cell: out.cell,
        cells: null,
        difference: null,
        millis: 0,
      };
      const transfer: Transferable[] = [];
      if (out.matchedCells > 0) {
        const cells = out.takeCells();
        const id = nextId++;
        items.set(id, { kind: "cloud", cloud: cells, name: `volume_${label(req.before)}_${label(req.after)}` });
        const described = describe(id);
        value.cells = described.value;
        value.difference = cells.attribute("height_difference") ?? null;
        transfer.push(...described.transfer);
        if (value.difference) transfer.push(value.difference.buffer);
      }
      out.free();
      value.millis = performance.now() - start;
      return { value, transfer };
    }
    case "rasterize": {
      const start = performance.now();
      const source = items.get(req.id);
      const out = getCloud(req.id).rasterize(
        req.cell,
        req.height,
        req.percentile,
        req.fillEmpty,
        req.class ?? undefined,
      );
      const heights = out.takeHeights();
      const id = nextId++;
      const base = source!.name.replace(/\.[^.]+$/, "");
      const cells = out.takeCells();
      items.set(id, { kind: "cloud", cloud: cells, name: `${base}_raster${req.cell}` });
      const described = describe(id);
      const cellHeights = cells.attribute("height")!;
      const value: RasterOutput = {
        nx: out.nx,
        ny: out.ny,
        minX: out.minX,
        minY: out.minY,
        cell: out.cell,
        heights,
        populatedCells: out.populatedCells,
        cells: described.value,
        cellHeights,
        millis: 0,
      };
      out.free();
      value.millis = performance.now() - start;
      return { value, transfer: [...described.transfer, heights.buffer, cellHeights.buffer] };
    }
    case "geotiff": {
      const r = req.raster;
      const bytes = rasterGeotiff(r.heights, r.nx, r.ny, r.minX, r.minY, r.cell);
      return { value: bytes, transfer: [bytes.buffer] };
    }
    case "filter": {
      const source = items.get(req.id);
      const t = performance.now();
      const cloud = getCloud(req.id);
      const k = Math.max(1, Math.floor(req.a));
      const means = req.op === "sor" ? await parallelSorMeans(cloud, k) : null;
      const filtered = means ? cloud.filterSor(means, req.b) : cloud.filter(req.op, req.a, req.b);
      await buildIndex(filtered);
      const id = nextId++;
      const base = source!.name.replace(/\.[^.]+$/, "");
      const suffix = {
        voxel: `voxel${req.a}`,
        random: `random${req.a}`,
        spatial: `space${req.a}`,
        octree: `octree${req.a}`,
        sor: "sor",
        splat: "clean",
      }[req.op];
      items.set(id, { kind: "cloud", cloud: filtered, name: `${base}_${suffix}` });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "normals": {
      const cloud = getCloud(req.id);
      const k = Math.max(3, Math.floor(req.k));
      if (!(await parallelNormals(cloud, k, req.orientation))) cloud.estimateNormals(k, req.orientation);
      const normals = cloud.normals()!;
      return { value: normals, transfer: [normals.buffer] };
    }
    case "merge": {
      const t = performance.now();
      const merger = new CloudMerger();
      const names: string[] = [];
      req.ids.forEach((id, k) => {
        merger.add(getCloud(id), ...req.fills[k]);
        names.push(items.get(id)!.name);
      });
      const merged = merger.finish();
      await buildIndex(merged);
      const id = nextId++;
      items.set(id, { kind: "cloud", cloud: merged, name: `merged_${names.length}`, sources: names });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "split": {
      const item = items.get(req.id);
      if (item?.kind !== "cloud") throw new Error("not a point cloud");
      return splitItem(item, req.by);
    }
    case "shapes":
      return shapes(req);
    case "map-quality": {
      const [awd, scs, voxels] = mapQuality(getCloud(req.compared), getCloud(req.reference), req.voxel, req.minPoints);
      return { value: { awd, scs, voxels }, transfer: [] };
    }
    case "m3c2": {
      const start = performance.now();
      const cloud = computeM3c2(
        getCloud(req.compared),
        getCloud(req.reference),
        req.normalRadius,
        req.projectionRadius,
        req.maxDepth,
        req.coreSpacing,
      );
      const id = nextId++;
      const base = items.get(req.compared)!.name.replace(/\.[^.]+$/, "");
      items.set(id, { kind: "cloud", cloud, name: `${base}_m3c2` });
      const described = describe(id);
      const distance = cloud.attribute("m3c2_distance")!;
      const lod95 = cloud.attribute("lod95")!;
      const significant = cloud.attribute("significant")!;
      const value: M3c2Output = {
        cloud: described.value,
        distance,
        lod95,
        significant,
        millis: performance.now() - start,
      };
      return {
        value,
        transfer: [...described.transfer, distance.buffer, lod95.buffer, significant.buffer],
      };
    }
    case "profile": {
      const hits = getCloud(req.id).profile(new Float64Array(req.line), req.halfWidth, req.maxPoints);
      const value: ProfileOutput = { along: hits.along(), positions: hits.positions(), total: hits.total };
      hits.free();
      return { value, transfer: [value.along.buffer, value.positions.buffer] };
    }
    case "ground": {
      const source = items.get(req.id);
      const t = performance.now();
      const cloud = getCloud(req.id).extractGround(req.clothResolution, req.classThreshold, req.rigidness, req.output);
      const id = nextId++;
      const base = source!.name.replace(/\.[^.]+$/, "");
      const suffix = { classified: "csf", ground: "ground", objects: "objects" }[req.output];
      items.set(id, { kind: "cloud", cloud, name: `${base}_${suffix}` });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "crop": {
      const source = items.get(req.id);
      const cropped = getCloud(req.id).crop(new Float64Array(req.min), new Float64Array(req.max), req.inside);
      const id = nextId++;
      const base = source!.name.replace(/\.[^.]+$/, "");
      items.set(id, { kind: "cloud", cloud: cropped, name: `${base}_${req.inside ? "crop" : "rest"}` });
      return describe(id);
    }
    case "segment": {
      const source = items.get(req.id);
      const t = performance.now();
      const cloud = getCloud(req.id);
      const clip = req.clip ? new Float64Array([...req.clip.min, ...req.clip.max]) : new Float64Array();
      const mask = cloud.lassoMask(
        new Float64Array(req.matrix),
        new Float64Array(req.polygon),
        clip,
        new Uint8Array(req.hiddenClasses),
      );
      const selected = mask.reduce((n, v) => n + v, 0);
      if (req.keep !== "outside" && selected === 0) throw new Error("the lasso selected no points");
      if (req.keep !== "inside" && selected === mask.length) throw new Error("the lasso selected every point");
      const base = source!.name.replace(/\.[^.]+$/, "");
      const parts: [number, string][] = [];
      if (req.keep !== "outside") parts.push([1, "segmented"]);
      if (req.keep !== "inside") parts.push([0, "remaining"]);
      const out: LoadedCloud[] = [];
      const transfer: Transferable[] = [];
      for (const [value, suffix] of parts) {
        const part = cloud.selectMask(mask, value);
        await buildIndex(part);
        const id = nextId++;
        items.set(id, { kind: "cloud", cloud: part, name: `${base}_${suffix}` });
        const described = describe(id, { parse: 0, index: performance.now() - t });
        out.push(described.value);
        transfer.push(...described.transfer);
      }
      return { value: out, transfer };
    }
    case "field": {
      const cloud = getCloud(req.id);
      const values = req.name === "Z" ? cloud.heights() : cloud.attribute(req.name);
      if (!values) throw new Error(`no field "${req.name}"`);
      return { value: values, transfer: [values.buffer] };
    }
    case "set-field": {
      getCloud(req.id).setAttribute(req.name, req.values);
      return { value: null, transfer: [] };
    }
    case "filter-field": {
      const source = items.get(req.id);
      const t = performance.now();
      const { values, lo, hi, inside } = req;
      const mask = new Uint8Array(values.length);
      for (let i = 0; i < values.length; i++) mask[i] = +(values[i] >= lo && values[i] <= hi);
      const part = getCloud(req.id).selectMask(mask, inside ? 1 : 0);
      await buildIndex(part);
      const id = nextId++;
      const base = source!.name.replace(/\.[^.]+$/, "");
      items.set(id, { kind: "cloud", cloud: part, name: `${base}_${inside ? "in" : "out"}` });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "align-pairs": {
      const values = alignPairs(new Float64Array(req.moving), new Float64Array(req.reference));
      return { value: values, transfer: [values.buffer] };
    }
    case "mesh": {
      const start = performance.now();
      const out = getCloud(req.id).meshDelaunay(req.maxEdge ?? undefined, MESH_MAX_POINTS);
      const id = nextId++;
      const base = items.get(req.id)!.name.replace(/\.[^.]+$/, "");
      items.set(id, { kind: "mesh", mesh: out.takeMesh(), name: `${base}_mesh` });
      const described = describe(id);
      const value: MeshOutput = {
        mesh: described.value,
        points: out.points,
        voxel: out.voxel,
        maxEdge: out.maxEdge,
        removed: out.removed,
        millis: performance.now() - start,
      };
      out.free();
      return { value, transfer: described.transfer };
    }
    case "export": {
      const item = items.get(req.id);
      const bytes =
        item?.kind === "mesh"
          ? item.mesh.export(req.format)
          : getCloud(req.id).export(req.format, req.scalar?.name, req.scalar?.values);
      return { value: bytes, transfer: [bytes.buffer] };
    }
    case "remove": {
      const item = items.get(req.id);
      if (item?.kind === "cloud") item.cloud.free();
      if (item?.kind === "mesh") item.mesh.free();
      items.delete(req.id);
      return { value: null, transfer: [] };
    }
  }
}

/** Requests asked to stop; they check between steps. */
const cancelled = new Set<number>();
const controllers = new Map<number, AbortController>();

self.onmessage = async (event: MessageEvent<UiMessage>) => {
  if ("cancel" in event.data) {
    if (controllers.has(event.data.cancel)) {
      cancelled.add(event.data.cancel);
      controllers.get(event.data.cancel)!.abort();
    }
    return;
  }
  const { seq, req } = event.data;
  const controller = new AbortController();
  controllers.set(seq, controller);
  let response: Response;
  let transfer: Transferable[] = [];
  const progress = (note: string, fraction?: number) => {
    const update: Progress = { note, fraction };
    const message: WorkerMessage = { seq, progress: update };
    self.postMessage(message);
  };
  const check = () => {
    if (cancelled.has(seq)) throw new Error(CANCELLED);
  };
  try {
    const out = await handle(req, progress, check, controller.signal);
    response = { ok: true, value: out.value };
    transfer = out.transfer;
  } catch (err) {
    let error = err instanceof Error ? err.message : String(err);
    if (cancelled.has(seq)) error = CANCELLED;
    if (/memory|allocation|unreachable/i.test(error)) {
      error = `out of memory (${error}); lower "Max points" and load the file again`;
    }
    response = { ok: false, error };
  }
  cancelled.delete(seq);
  controllers.delete(seq);
  const memory = (await ready).memory.buffer.byteLength;
  const message: WorkerMessage = { seq, response, memory };
  self.postMessage(message, { transfer });
};
