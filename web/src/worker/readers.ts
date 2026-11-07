// Reading point files: streamed formats, COPC and LAS/LAZ (chunked across the pool).

import {
  announcedPoints,
  Cloud,
  CopcReader,
  CopcBoxReader,
  e57ScanNames,
  LasReader,
  Mesh,
  StreamLoader,
} from "../wasm/ca_wasm.js";
import { type ByteSource, readRange } from "../bytes";
import type { LasChunksResult } from "../c2c-worker";
import { eachSlice, runSlices } from "../pool";
import type { LoadedCloud, Request } from "../protocol";
import { type CopcSource, type Detail, describe, items, newId } from "./items";
import { buildIndex } from "./build-index";

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
export async function readFullDensityBox(
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
    const id = newId();
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
export async function readPoints(
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
