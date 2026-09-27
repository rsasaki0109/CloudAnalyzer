// Runs the Rust/WASM core off the UI thread. All clouds and meshes live here
// so that analyses can use full f64 coordinates without copying them back and
// forth.

import init, {
  Cloud,
  cloudToCloud,
  cloudToMesh,
  Mesh,
  planCloudToCloud,
  registerIcp,
  summarizeDistances,
} from "./wasm/ca_wasm.js";
import type { Slice } from "./c2c-worker";
import { MIN_PARALLEL_QUERIES, poolSize, runSlices } from "./pool";
import type { C2cOutput, IcpOutput, LoadedCloud, Request, Response, Vec3, WorkerMessage } from "./protocol";

type Item = { kind: "cloud"; cloud: Cloud; name: string } | { kind: "mesh"; mesh: Mesh; name: string };

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
  const slices: Slice[] = Array.from({ length: plan.length }, (_, k) => ({
    kind: "cloud",
    reference: plan.takeReference(k),
    queries: plan.takeQueries(k),
  }));
  const indices = Array.from({ length: plan.length }, (_, k) => plan.takeQueryIndices(k));
  plan.free();
  const results = await runSlices(slices);
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
  const slices: Slice[] = Array.from({ length: parts }, (_, k) => ({
    kind: "mesh",
    vertices,
    indices,
    queries: positions.slice(k * per * 3, Math.min(compared.length, (k + 1) * per) * 3),
    signed,
  }));
  const results = await runSlices(slices);
  const distances = new Float64Array(compared.length);
  results.forEach((part, k) => distances.set(part, k * per));
  return { distances, workers: parts };
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
      bounds: Array.from(item.mesh.bounds()),
      shift: shift!,
      lodNodes: new Float64Array(),
      lodGrid: 0,
      timings: { ...timings, prepare: performance.now() - start },
    };
    return { value, transfer: [positions.buffer, indices.buffer] };
  }
  const { cloud } = item;
  const positions = cloud.positions(s);
  const colors = cloud.colors() ?? null;
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
    bounds: Array.from(cloud.bounds()),
    shift: shift!,
    lodNodes,
    lodGrid: cloud.lodGrid,
    timings: { ...timings, prepare: performance.now() - start },
  };
  const transfer: Transferable[] = [positions.buffer, lodNodes.buffer];
  if (colors) transfer.push(colors.buffer);
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
      progress("parsing");
      let t = performance.now();
      const bytes = new Uint8Array(req.bytes);
      const mesh = Mesh.parse(req.name, bytes);
      if (mesh) {
        const parse = performance.now() - t;
        shift ??= Array.from(mesh.suggestedShift()) as Vec3;
        const id = nextId++;
        items.set(id, { kind: "mesh", mesh, name: req.name });
        return describe(id, { parse, index: 0 });
      }
      const cloud = Cloud.parse(req.name, bytes);
      const parse = performance.now() - t;
      progress(`indexing ${cloud.length.toLocaleString()} points`);
      t = performance.now();
      cloud.buildIndex();
      const index = performance.now() - t;
      shift ??= Array.from(cloud.suggestedShift()) as Vec3;
      const id = nextId++;
      items.set(id, { kind: "cloud", cloud, name: req.name });
      progress("preparing for display");
      return describe(id, { parse, index });
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
    response = { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
  self.postMessage({ seq, response }, { transfer });
};
