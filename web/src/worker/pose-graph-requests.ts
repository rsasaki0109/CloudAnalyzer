// Pose graph requests: opening, editing, optimizing and exporting the open graph.

import { registerScans } from "../wasm/ca_wasm.js";
import { CANCELLED, type PoseGraphFound, type PoseGraphFloor, type Request } from "../protocol";
import { eachSlice, poolSize } from "../pool";
import { describe, items, newId } from "./items";
import { buildIndex } from "./build-index";
import {
  DYNAMIC_MIN_PART,
  RETRY_MIN_DRIFT,
  graphState,
  mergePoseGraph,
  openGraph,
  openPoseGraph,
  poseGraph,
  prepareGraphProject,
  setPoseGraph,
  stateTransfer,
} from "./pose-graph";

export const POSE_GRAPH_KINDS = [
  "pg-open",
  "pg-project-save",
  "pg-project-open",
  "pg-merge",
  "pg-loop",
  "pg-register",
  "pg-add-edge",
  "pg-find-loops",
  "pg-floor",
  "pg-set-node-pose",
  "pg-set-fixed",
  "pg-gravity",
  "pg-clear-gravity",
  "pg-remove-plane",
  "pg-optimize",
  "pg-remove-edges",
  "pg-insert-edges",
  "pg-set-poses",
  "pg-export",
  "pg-map",
  "pg-dynamic",
  "pg-close",
] as const;
export type PoseGraphRequest = Extract<Request, { kind: (typeof POSE_GRAPH_KINDS)[number] }>;
const kinds: ReadonlySet<string> = new Set(POSE_GRAPH_KINDS);
export const isPoseGraphRequest = (req: Request): req is PoseGraphRequest => kinds.has(req.kind);

export async function poseGraphRequest(
  req: PoseGraphRequest,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<{ value: unknown; transfer: Transferable[] }> {
  switch (req.kind) {
    case "pg-open":
      return openPoseGraph(req, progress, check);
    case "pg-project-save":
      return { value: { snapshot: openGraph().toSnapshot(), sources: poseGraph!.sources }, transfer: [] };
    case "pg-project-open": {
      const result = await prepareGraphProject(req.project, req.name, progress, check);
      poseGraph?.session.free();
      setPoseGraph(result.native);
      return {value:result.value,transfer:result.transfer};
    }
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
      const id = newId();
      const part = req.label ? `_${req.label}` : "";
      items.set(id, { kind: "cloud", cloud, name: `${poseGraph!.name}_${req.initial ? "map_start" : "map"}${part}` });
      return describe(id, { parse: 0, index: performance.now() - t });
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
    case "pg-close":
      poseGraph?.session.free();
      setPoseGraph(null);
      return { value: null, transfer: [] };
  }
}
