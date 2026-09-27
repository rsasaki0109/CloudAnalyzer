// Runs the Rust/WASM core off the UI thread. All clouds and meshes live here
// so that analyses can use full f64 coordinates without copying them back and
// forth.

import init, {
  announcedPoints,
  Cloud,
  cloudToCloud,
  cloudToMesh,
  computeM3c2,
  computeVolume,
  Mesh,
  StreamLoader,
  planCloudToCloud,
  registerIcp,
  summarizeDistances,
  VolumeSurface,
} from "./wasm/ca_wasm.js";
import type { Slice } from "./c2c-worker";
import { MIN_PARALLEL_QUERIES, poolSize, runSlices } from "./pool";
import type {
  C2cOutput,
  IcpOutput,
  LoadedCloud,
  M3c2Output,
  Request,
  Response,
  Vec3,
  VolumeOutput,
  VolumeSide,
  WorkerMessage,
} from "./protocol";

type Item =
  | { kind: "cloud"; cloud: Cloud; name: string; keepEvery?: number; filePoints?: number }
  | { kind: "mesh"; mesh: Mesh; name: string };

const ready = init();
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

/** Bytes read per slice when streaming a file. */
const STREAM_CHUNK = 16 << 20;

type Loaded =
  | { kind: "mesh"; mesh: Mesh }
  | { kind: "cloud"; cloud: Cloud; keepEvery: number; filePoints: number };

/**
 * Read a file: stream fixed-record formats (LAS, binary PLY/PCD) slice by
 * slice so they are never held whole; read anything else (LAZ, meshes, text)
 * at once. Clouds larger than `maxPoints` keep every n-th point.
 */
async function readPoints(file: File, maxPoints: number, progress: (note: string) => void): Promise<Loaded> {
  const name = file.name;
  const thinning = (points: number) => (points > maxPoints ? Math.ceil(points / maxPoints) : 1);
  progress("reading header");
  let head = new Uint8Array(await file.slice(0, 1 << 16).arrayBuffer());
  let headerLength = StreamLoader.headerLength(name, head);
  if (headerLength === undefined && file.size > head.length) {
    head = new Uint8Array(await file.slice(0, 1 << 22).arrayBuffer());
    headerLength = StreamLoader.headerLength(name, head);
  }
  const loader = headerLength !== undefined ? StreamLoader.open(name, head.subarray(0, headerLength)) : undefined;
  if (loader) {
    const filePoints = loader.totalPoints;
    const keepEvery = thinning(filePoints);
    loader.setKeepEvery(keepEvery);
    const start = loader.dataOffset;
    for (let at = start; at < file.size; at += STREAM_CHUNK) {
      const percent = Math.round(((at - start) / Math.max(1, file.size - start)) * 100);
      progress(`reading ${percent}%${keepEvery > 1 ? ` (keeping 1 in ${keepEvery})` : ""}`);
      loader.push(new Uint8Array(await file.slice(at, at + STREAM_CHUNK).arrayBuffer()));
    }
    const cloud = loader.finish();
    loader.free();
    return { kind: "cloud", cloud, keepEvery, filePoints };
  }
  progress("reading");
  const bytes = new Uint8Array(await file.arrayBuffer());
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
/** The top levels built here; each node below becomes a pool job (up to 8^2). */
const INDEX_SPLIT_LEVEL = 2;

/**
 * Build the octree: the top levels here, the subtrees below them on the
 * worker pool (largest first, so the slowest job starts early).
 */
async function buildIndex(cloud: Cloud): Promise<number> {
  if (cloud.length < MIN_PARALLEL_INDEX || poolSize() < 2) {
    cloud.buildIndex();
    return 1;
  }
  const jobs = cloud.startIndex(INDEX_SPLIT_LEVEL);
  const count = jobs.length / 7;
  const order = Array.from({ length: count }, (_, k) => k).sort(
    (a, b) => jobs[b * 7 + 1] - jobs[b * 7] - (jobs[a * 7 + 1] - jobs[a * 7]),
  );
  const results = await runSlices(count, (i) => {
    const k = order[i];
    return {
      kind: "octree" as const,
      positions: cloud.subtreePositions(k),
      colors: cloud.subtreeColors(k) ?? null,
      job: jobs.slice(k * 7, k * 7 + 7),
    };
  });
  results.forEach((subtree, i) =>
    cloud.finishSubtree(order[i], subtree.positions, subtree.colors, subtree.nodes, subtree.order),
  );
  cloud.endIndex();
  return Math.min(poolSize(), count);
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
      bounds: Array.from(item.mesh.bounds()),
      shift: shift!,
      lodNodes: new Float64Array(),
      lodGrid: 0,
      keepEvery: 1,
      filePoints: item.mesh.vertexCount,
      timings: { ...timings, prepare: performance.now() - start },
    };
    return { value, transfer: [positions.buffer, indices.buffer] };
  }
  const { cloud } = item;
  const positions = cloud.positions(s);
  const colors = cloud.colors() ?? null;
  const intensity = cloud.intensity() ?? null;
  const classification = cloud.classification() ?? null;
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
    bounds: Array.from(cloud.bounds()),
    shift: shift!,
    lodNodes,
    lodGrid: cloud.lodGrid,
    keepEvery: item.keepEvery ?? 1,
    filePoints: item.filePoints ?? cloud.length,
    timings: { ...timings, prepare: performance.now() - start },
  };
  const transfer: Transferable[] = [positions.buffer, lodNodes.buffer];
  for (const buffer of [colors, intensity, classification]) if (buffer) transfer.push(buffer.buffer);
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
  progress: (note: string) => void,
): Promise<{ value: unknown; transfer: Transferable[] }> {
  await ready;
  switch (req.kind) {
    case "load": {
      const { file, maxPoints } = req;
      const name = file.name;
      let t = performance.now();
      const loaded = await readPoints(file, maxPoints, progress);
      const parse = performance.now() - t;
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
      shift ??= Array.from(cloud.suggestedShift()) as Vec3;
      const id = nextId++;
      items.set(id, { kind: "cloud", cloud, name, keepEvery: loaded.keepEvery, filePoints: loaded.filePoints });
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
    case "filter": {
      const source = items.get(req.id);
      const t = performance.now();
      const filtered = getCloud(req.id).filter(req.op, req.a, req.b);
      const id = nextId++;
      const base = source!.name.replace(/\.[^.]+$/, "");
      const suffix = { voxel: `voxel${req.a}`, random: `random${req.a}`, sor: "sor" }[req.op];
      items.set(id, { kind: "cloud", cloud: filtered, name: `${base}_${suffix}` });
      return describe(id, { parse: 0, index: performance.now() - t });
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

self.onmessage = async (event: MessageEvent<{ seq: number; req: Request }>) => {
  const { seq, req } = event.data;
  let response: Response;
  let transfer: Transferable[] = [];
  const progress = (note: string) => {
    const message: WorkerMessage = { seq, progress: note };
    self.postMessage(message);
  };
  try {
    const out = await handle(req, progress);
    response = { ok: true, value: out.value };
    transfer = out.transfer;
  } catch (err) {
    let error = err instanceof Error ? err.message : String(err);
    if (/memory|allocation|unreachable/i.test(error)) {
      error = `out of memory (${error}); lower "Max points" and load the file again`;
    }
    response = { ok: false, error };
  }
  self.postMessage({ seq, response }, { transfer });
};
