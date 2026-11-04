// Pose graphs: scans and bags, LiDAR odometry, and the open native session.

import { BagFile, Cloud, LidarOdometry, PoseGraphSession, TrajectoryData, upsAt } from "../wasm/ca_wasm.js";
import {
  isBag,
  type PoseGraphFiles,
  type PoseGraphMerged,
  type PoseGraphOpened,
  type PoseGraphProject,
  type PoseGraphState,
  type PoseGraphSource,
  type Request,
} from "../protocol";
import { poolSize, runOn } from "../pool";

/** The open pose graph, if any (one at a time). */
export type NativeGraph = { session: PoseGraphSession; name: string; sources: PoseGraphSource[] };
export let poseGraph: NativeGraph | null = null;

export function setPoseGraph(graph: NativeGraph | null): void {
  poseGraph = graph;
}

export async function prepareGraphProject(saved: PoseGraphProject, name: string, progress: (note:string,fraction?:number)=>void, check:()=>void) {
  const session = PoseGraphSession.fromSnapshot(saved.snapshot);
  try {
    const covered = new Set<number>();
    for (const source of saved.sources) {
      check();
      const loaded = await loadGraph(source.files, progress, check);
      try {
        const ids = Array.from(loaded.session.nodeIds());
        if (ids.length !== source.nodeIds.length || ids.some((id, i) => id !== source.nodeIds[i])) throw new Error("Source keyframes do not match the saved project");
        if (!Number.isSafeInteger(source.first) || source.first < 0 || source.first + ids.length > session.nodeCount) throw new Error("Invalid project scan range");
        for (let i = 0; i < ids.length; i++) {
          const node = source.first + i;
          if (covered.has(node)) throw new Error("Overlapping project scan ranges");
          covered.add(node);
        }
        session.attachSourceScans(loaded.session, source.first);
      } finally { loaded.session.free(); }
    }
    if (covered.size !== session.nodeCount) throw new Error("Project is missing original scan sources");
    check();
    const scans = Array.from({ length: session.nodeCount }, (_, i) => {
      const source = saved.sources.find(s => i >= s.first && i < s.first + s.nodeIds.length);
      return source ? session.scanPositions(i, source.files.displayPoints) : null;
    });
    const value: PoseGraphOpened = { ...graphState(session), name: name, scans, scanPoints: session.scanPointCount(), unmatched: [], odometry: null };
    const native = { session, name, sources: saved.sources };
    return { native, value, transfer: [...stateTransfer(value), ...scans.filter((s): s is Float32Array => s !== null).map(s => s.buffer)] };
  } catch (error) { session.free(); throw error; }
}

export function openGraph(): PoseGraphSession {
  if (!poseGraph) throw new Error("no pose graph is open");
  return poseGraph.session;
}
export function graphState(session: PoseGraphSession): PoseGraphState {
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
export const stateTransfer = (s: PoseGraphState): Transferable[] =>
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
export const RETRY_MIN_DRIFT = 5;
/** Fewest keyframes one pool worker judges for dynamic points: fewer would mostly send neighbours. */
export const DYNAMIC_MIN_PART = 20;
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
export async function openPoseGraph(
  req: PoseGraphFiles,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<{ value: PoseGraphOpened; transfer: Transferable[] }> {
  const { session, scans, scanPoints, unmatched, odometry } = await loadGraph(req, progress, check);
  poseGraph?.session.free();
  const name = req.graph?.name ?? "scans";
  poseGraph = { session, name: name.replace(/\.[^.]+$/, ""), sources: [{ files: req, first: 0, nodeIds: Array.from(session.nodeIds()) }] };
  const value: PoseGraphOpened = { ...graphState(session), name, scans, scanPoints, unmatched, odometry };
  const transfer = stateTransfer(value);
  for (const scan of scans) if (scan) transfer.push(scan.buffer);
  if (odometry?.ups) transfer.push(odometry.ups.buffer);
  return { value, transfer };
}
export async function mergePoseGraph(
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
    poseGraph!.sources.push({ files: req.files, first: offset, nodeIds: Array.from(ids) });
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
