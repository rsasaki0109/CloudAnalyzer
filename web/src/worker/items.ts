// Loaded items (clouds, meshes, full-density details) and the replies describing them.

import { Cloud, cloudToCloud, Mesh } from "../wasm/ca_wasm.js";
import type { ByteSource } from "../bytes";
import { invalidateGpuReference } from "../gpu-c2c";
import { invalidateCpuC2c } from "../cpu-c2c";
import { invalidateM3c2Preparation } from "../m3c2-preparation";
import { invalidateM3c2CoreSelection } from "../m3c2-core-selection";
import { invalidateIcpReference } from "../icp-preparation";
import { invalidateVoxelScores } from "../map-voxel-cache";
import { invalidateMeshSources } from "../mesh-preparation";
import { invalidateKnnPreparation } from "../knn-preparation";
import { runAny } from "../pool";
import type { C2cOutput, DetailChunk, LoadedCloud, Request, Vec3 } from "../protocol";
import { buildIndex } from "./build-index";

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
export interface Detail {
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
export interface CopcSource { source: ByteSource; size: number; head: Uint8Array }

export const items = new Map<number, Item>();
let nextId = 1;

export function newId(): number {
  return nextId++;
}

/** Everything the UI needs to draw an item; buffers are listed for transfer. */
export function describe(
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
export function getCloud(id: number): Cloud {
  const item = items.get(id);
  if (!item) throw new Error("cloud not found");
  if (item.kind !== "cloud") throw new Error(`${item.name} is a mesh, not a point cloud`);
  return item.cloud;
}
export function output(
  result: ReturnType<typeof cloudToCloud>,
  start: number,
  workers: number,
  kind: C2cOutput["kind"],
  signed: boolean,
  ownedDistances?: Float32Array,
): { value: C2cOutput; transfer: Transferable[] } {
  const distances = ownedDistances ?? result.distances();
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
export async function splitItem(
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
    const id = newId();
    const name = by === "source" ? (item.sources?.[v] ?? `${base}_part${v}`) : `${base}_class${v}`;
    items.set(id, { kind: "cloud", cloud: part, name });
    const described = describe(id, { parse: 0, index: performance.now() - t });
    out.push(described.value);
    transfer.push(...described.transfer);
  }
  return { value: out, transfer };
}

/** The file no longer matches a cloud whose points moved. */
export function dropDetail(id: number): void {
  const item = items.get(id);
  if (item?.kind === "cloud") {
    invalidateCpuC2c(item.cloud);
    invalidateM3c2Preparation(item.cloud);
    invalidateM3c2CoreSelection(item.cloud);
    invalidateGpuReference(item.cloud);
    invalidateIcpReference(item.cloud);
    invalidateVoxelScores(item.cloud);
    invalidateKnnPreparation(item.cloud);
    invalidateMeshSources(item.cloud);
    item.detail = undefined;
    item.copcSource = undefined;
  }
}
/** Decode one chunk of a thinned cloud's file at full density, on the pool. */
export async function readDetail(
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
