// Runs the Rust/WASM core off the UI thread. All clouds and meshes live here
// so that analyses can use full f64 coordinates without copying them back and
// forth.

import init, {
  announcedPoints,
  Cloud,
  CloudMerger,
  CopcReader,
  cloudToCloud,
  cloudToMesh,
  computeM3c2,
  computeVolume,
  Mesh,
  StreamLoader,
  planCloudToCloud,
  planSor,
  rasterGeotiff,
  registerIcp,
  summarizeDistances,
  VolumeSurface,
  warmUp,
} from "./wasm/ca_wasm.js";
import { type ByteSource, readRange } from "./bytes";
import type { Slice } from "./c2c-worker";
import { CANCELLED } from "./protocol";
import { MIN_PARALLEL_QUERIES, poolSize, runOn, runSlices, warmUpPool } from "./pool";
import type {
  C2cOutput,
  IcpOutput,
  LoadedCloud,
  M3c2Output,
  ProfileOutput,
  Progress,
  RasterOutput,
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
    }
  | { kind: "mesh"; mesh: Mesh; name: string };

const ready = init().then((wasm) => {
  // Optimize the heavy kernels here and on the pool before the first file
  // arrives (see `warmUp`); requests wait for this one.
  warmUp();
  void warmUpPool();
  return wasm;
});
const items = new Map<number, Item>();
let nextId = 1;
// Like CloudCompare's global shift: chosen from the first file, shared by all.
let shift: Vec3 | null = null;

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
  | { kind: "cloud"; cloud: Cloud; keepEvery: number; filePoints: number; copcLevels?: number };

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
    return { kind: "cloud", cloud, keepEvery: 1, filePoints, copcLevels: level };
  } catch (err) {
    reader.free();
    throw err;
  }
}

/**
 * Read a file: stream fixed-record formats (LAS, binary PLY/PCD) slice by
 * slice so they are never held whole; read anything else (LAZ, meshes, text)
 * at once. Clouds larger than `maxPoints` keep every n-th point.
 */
async function readPoints(
  file: File,
  maxPoints: number,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<Loaded> {
  const name = file.name;
  const thinning = (points: number) => (points > maxPoints ? Math.ceil(points / maxPoints) : 1);
  progress("reading header");
  let head = new Uint8Array(await file.slice(0, 1 << 16).arrayBuffer());
  let headerLength = StreamLoader.headerLength(name, head);
  if (headerLength === undefined && file.size > head.length) {
    head = new Uint8Array(await file.slice(0, 1 << 22).arrayBuffer());
    headerLength = StreamLoader.headerLength(name, head);
  }
  if (CopcReader.isCopc(head)) return readCopc({ file }, file.size, maxPoints, progress, check);
  const loader = headerLength !== undefined ? StreamLoader.open(name, head.subarray(0, headerLength)) : undefined;
  if (loader) {
    const filePoints = loader.totalPoints;
    const keepEvery = thinning(filePoints);
    loader.setKeepEvery(keepEvery);
    const start = loader.dataOffset;
    try {
      for (let at = start; at < file.size; at += STREAM_CHUNK) {
        const fraction = (at - start) / Math.max(1, file.size - start);
        progress(
          `reading ${Math.round(fraction * 100)}%${keepEvery > 1 ? ` (keeping 1 in ${keepEvery})` : ""}`,
          fraction,
        );
        const chunk = new Uint8Array(await file.slice(at, at + STREAM_CHUNK).arrayBuffer());
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
  const bytes = new Uint8Array(await file.arrayBuffer());
  check();
  progress("parsing");
  const mesh = Mesh.parse(name, bytes);
  if (mesh) return { kind: "mesh", mesh };
  const announced = announcedPoints(name, bytes);
  const keepEvery = announced === undefined ? 1 : thinning(announced);
  const cloud = Cloud.parseThinned(name, bytes, keepEvery);
  return { kind: "cloud", cloud, keepEvery, filePoints: announced ?? cloud.length };
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
  const s = new Float64Array(shift!);
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
      sources: null,
      bounds: Array.from(item.mesh.bounds()),
      shift: shift!,
      lodNodes: new Float64Array(),
      lodGrid: 0,
      keepEvery: 1,
      filePoints: item.mesh.vertexCount,
      copcLevels: null,
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
  const lodNodes = cloud.lodNodes();
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
    sources: item.sources ?? null,
    bounds: Array.from(cloud.bounds()),
    shift: shift!,
    lodNodes,
    lodGrid: cloud.lodGrid,
    keepEvery: item.keepEvery ?? 1,
    filePoints: item.filePoints ?? cloud.length,
    copcLevels: item.copcLevels ?? null,
    timings: { ...timings, prepare: performance.now() - start },
  };
  const transfer: Transferable[] = [positions.buffer, lodNodes.buffer];
  for (const buffer of [colors, intensity, classification, normals]) if (buffer) transfer.push(buffer.buffer);
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

async function handle(
  req: Request,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<{ value: unknown; transfer: Transferable[] }> {
  await ready;
  switch (req.kind) {
    case "load":
    case "load-copc": {
      const { maxPoints } = req;
      const name = req.kind === "load" ? req.file.name : req.name;
      let t = performance.now();
      const loaded =
        req.kind === "load"
          ? await readPoints(req.file, maxPoints, progress, check)
          : await readCopc({ url: req.url }, Number.POSITIVE_INFINITY, maxPoints, progress, check);
      const parse = performance.now() - t;
      try {
        check();
      } catch (err) {
        (loaded.kind === "mesh" ? loaded.mesh : loaded.cloud).free();
        throw err;
      }
      if (loaded.kind === "mesh") {
        shift ??= Array.from(loaded.mesh.suggestedShift()) as Vec3;
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
      shift ??= Array.from(cloud.suggestedShift()) as Vec3;
      const id = nextId++;
      items.set(id, {
        kind: "cloud",
        cloud,
        name,
        keepEvery: loaded.keepEvery,
        filePoints: loaded.filePoints,
        copcLevels: loaded.copcLevels,
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
      const suffix = { voxel: `voxel${req.a}`, random: `random${req.a}`, sor: "sor" }[req.op];
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
      const values = item.cloud.splitValues(req.by);
      if (!values) throw new Error(`${item.name} has no ${req.by} to split by`);
      const base = item.name.replace(/\.[^.]+$/, "");
      const out: LoadedCloud[] = [];
      const transfer: Transferable[] = [];
      for (const v of values) {
        const t = performance.now();
        const part = item.cloud.splitPart(req.by, v);
        await buildIndex(part);
        const id = nextId++;
        const name =
          req.by === "source"
            ? (item.sources?.[v] ?? `${base}_part${v}`)
            : `${base}_class${v}`;
        items.set(id, { kind: "cloud", cloud: part, name });
        const described = describe(id, { parse: 0, index: performance.now() - t });
        out.push(described.value);
        transfer.push(...described.transfer);
      }
      return { value: out, transfer };
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
    case "export": {
      const bytes = getCloud(req.id).export(req.format, req.scalar?.name, req.scalar?.values);
      return { value: bytes, transfer: [bytes.buffer] };
    }
    case "remove": {
      const item = items.get(req.id);
      if (item?.kind === "cloud") item.cloud.free();
      if (item?.kind === "mesh") item.mesh.free();
      items.delete(req.id);
      if (items.size === 0) shift = null;
      return { value: null, transfer: [] };
    }
  }
}

/** Requests asked to stop; they check between steps. */
const cancelled = new Set<number>();

self.onmessage = async (event: MessageEvent<UiMessage>) => {
  if ("cancel" in event.data) {
    cancelled.add(event.data.cancel);
    return;
  }
  const { seq, req } = event.data;
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
    const out = await handle(req, progress, check);
    response = { ok: true, value: out.value };
    transfer = out.transfer;
  } catch (err) {
    let error = err instanceof Error ? err.message : String(err);
    if (/memory|allocation|unreachable/i.test(error)) {
      error = `out of memory (${error}); lower "Max points" and load the file again`;
    }
    response = { ok: false, error };
  }
  cancelled.delete(seq);
  const memory = (await ready).memory.buffer.byteLength;
  const message: WorkerMessage = { seq, response, memory };
  self.postMessage(message, { transfer });
};
