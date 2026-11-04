// Runs the Rust/WASM core off the UI thread. All clouds and meshes live here
// so that analyses can use full f64 coordinates without copying them back and
// forth.

import init, {
  alignPairs,
  Cloud,
  CloudMerger,
  cloudToCloud,
  c2cSimdEnabled,
  cloudToMesh,
  computeM3c2,
  computeVolume,
  ObservedVolumeSupport,
  rasterGeotiff,
  registerScans,
  summarizeDistances,
  VectorMapSession,
  VolumeSurface,
  warmUp,
} from "./wasm/ca_wasm.js";
import { gpuCloudToCloud, invalidateGpuReference, prepareGpuC2c } from "./gpu-c2c";
import { cachedCloudToCloud, invalidateCpuC2c } from "./cpu-c2c";
import { parallelM3c2 } from "./m3c2-workers";
import { invalidateM3c2Preparation } from "./m3c2-preparation";
import { invalidateM3c2CoreSelection } from "./m3c2-core-selection";
import { evaluateMapQuality } from "./map-quality-workers";
import { cachedRegisterIcp, invalidateIcpReference } from "./icp-preparation";
import { snapshotVoxelScores, invalidateVoxelScores } from "./map-voxel-cache";
import { invalidateMeshSources, meshCacheState } from "./mesh-preparation";
import { invalidateKnnPreparation, knnCacheState } from "./knn-preparation";
import {
  CANCELLED,
  type IcpOutput,
  type LoadedCloud,
  type M3c2Output,
  type MeshOutput,
  type PoseGraphFound,
  type PoseGraphFloor,
  type ProfileOutput,
  type Progress,
  type RasterOutput,
  type UiMessage,
  type Request,
  type Response,
  type VolumeOutput,
  type VolumeSide,
  type WorkerMessage,
} from "./protocol";
import { parseFilterRecipe, recipeParameters } from "./filter-recipe";
import { computeLaneCoreEvidence, readCandidateCores } from "./lane-core-support";
import { poolMemory, releasePool, eachSlice, poolSize, warmUpPool } from "./pool";
import { describe, dropDetail, getCloud, items, newId, output, readDetail, splitItem } from "./worker/items";
import { buildIndex } from "./worker/build-index";
import { parallelCloudToCloud, parallelCloudToMesh } from "./worker/distances";
import { lastKnnHintState, parallelNormals, parallelSorMeans } from "./worker/knn";
import { readFullDensityBox, readPoints } from "./worker/readers";
import { shapes } from "./worker/shapes";
import { evaluate, readTrajectory } from "./worker/trajectory";
import {
  DYNAMIC_MIN_PART,
  type NativeGraph,
  RETRY_MIN_DRIFT,
  graphState,
  mergePoseGraph,
  openGraph,
  openPoseGraph,
  poseGraph,
  prepareGraphProject,
  setPoseGraph,
  stateTransfer,
} from "./worker/pose-graph";

let poolPreparation: Promise<unknown> = Promise.resolve();
const ready = init().then((wasm) => {
  // Optimize the heavy kernels here and on the pool before the first file
  // arrives (see `warmUp`); requests wait for this one.
  warmUp();
  poolPreparation = warmUpPool();
  void poolPreparation.catch(error => console.warn("Could not warm the worker pool", error));
  return wasm;
});
/** Larger clouds are voxel-thinned before meshing, to bound time and memory. */
const MESH_MAX_POINTS = 5_000_000;

/** The vector map, made on first use. */
let vectorMap: VectorMapSession | null = null;

let workspacePreparing = false;
/** The current native voxel key packs 21 bits per relative axis. */
function requireVoxelRange(cloud:Cloud,size:number):void {
  const bounds=cloud.bounds(),extent=Math.max(bounds[3]-bounds[0],bounds[4]-bounds[1],bounds[5]-bounds[2]);
  const minimum=extent/((1<<21)-2);
  if(size<minimum)throw new Error("Voxel size is too small for this cloud's extent. Use at least "+(minimum*1.00001).toPrecision(6)+" in the cloud's units");
}
let workspaceSequence = 0;
let preparedWorkspace: {token:number; map:VectorMapSession; graph:NativeGraph | null; ids:number[]} | null = null;
function releasePreparedWorkspace(): void {
  if (!preparedWorkspace) return;
  for (const id of preparedWorkspace.ids) {
    const item = items.get(id);
    if (item?.kind === "cloud") item.cloud.free();
    if (item?.kind === "mesh") item.mesh.free();
    items.delete(id);
  }
  preparedWorkspace.map.free();
  preparedWorkspace.graph?.session.free();
  preparedWorkspace = null;
}
/** A "vm" request's answer: JSON text, or for edits `{result, view}` with the view after the edit. */
function vectorMapRequest(req: Extract<Request, { kind: "vm" }>): string {
  const map = (vectorMap ??= new VectorMapSession());
  const withView = (result: string) => `{"result":${result},"view":${map.view()},"undo":${map.undoDepth}}`;
  switch (req.op) {
    case "open":
      return withView(map.open(req.name ?? "", req.text ?? ""));
    case "apply":
      return withView(map.apply(req.text ?? "[]"));
    case "relations-preview":
      return map.previewRelations(BigInt(req.id ?? 0));
    case "relations-adopt":
      return withView(map.adoptRelation(req.text ?? "{}"));
    case "relations-edit":
      return withView(map.editRelations(req.text ?? "{}"));
    case "feature-edit":
      return withView(map.editFeatureGeometry(req.text ?? "{}"));
    case "feature-discover":
    case "feature-confirm": {
      if(req.id === undefined) throw new Error("Choose a point cloud.");
      const request=JSON.parse(req.text ?? "{}");
      return req.op === "feature-discover" ? map.discoverFeatures(getCloud(req.id), JSON.stringify(request.options ?? {})) : withView(map.confirmFeatures(getCloud(req.id),JSON.stringify(request.options ?? {}),JSON.stringify(request.confirmations ?? [])));
    }
    case "build":
      if (req.id === undefined || !req.positions) throw new Error("Choose a point cloud and trajectory.");
      return withView(map.buildFromTrajectory(getCloud(req.id), req.positions, req.text ?? "{}"));
    case "quality":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return map.auditQuality(getCloud(req.id));
    case "junction-preview":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return map.previewJunctions(getCloud(req.id), req.text ?? "{}");
    case "signal-preview":
    case "signal-add":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return withView(map.measureSignal(getCloud(req.id), req.text ?? "{}", req.op === "signal-preview"));
    case "crosswalk-preview":
    case "crosswalk-add":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return withView(map.measureCrosswalk(getCloud(req.id), req.text ?? "{}", req.op === "crosswalk-preview"));
    case "junction-connect":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return withView(map.connectJunctions(getCloud(req.id), req.text ?? "{}", JSON.stringify(req.pairs ?? null)));
    case "undo":
      return withView(String(map.undo()));
    case "clear":
      map.clear();
      return withView("null");
    case "history-budget": {
      const policy = JSON.parse(req.text ?? "{}");
      map.setHistoryBudget(policy.steps, policy.bytes);
      return withView("null");
    }
    case "history-clear":
      map.clearHistory(); return withView("null");
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
    case "check-project": {
      const candidate = new VectorMapSession();
      try { candidate.open("project-map.json", req.text ?? ""); }
      finally { candidate.free(); }
      return "null";
    }
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
    case "workspace-prepare": {
      if (workspacePreparing || preparedWorkspace) throw new Error("A workspace import is already staged");
      if (req.clouds.length > 127) throw new Error("Workspace import exceeds cloud limit");
      workspacePreparing = true;
      try {
        const staged = {token:++workspaceSequence, map:new VectorMapSession(), graph:null as NativeGraph | null, ids:[] as number[]};
        preparedWorkspace = staged;
        const result = staged.map.open("project-map.json",req.mapText);
        const map = `{"result":${result},"view":${staged.map.view()},"undo":0}`;
        const graph = req.graph ? await prepareGraphProject(req.graph,req.graphName,progress,check) : null;
        staged.graph = graph?.native ?? null;
        const clouds:LoadedCloud[] = [], transfer:Transferable[] = [...(graph?.transfer ?? [])];
        for (const source of req.clouds) {
          check();
          progress(`Staging ${source.name}`);
          const loaded = await handle({kind:"load",file:source.file,displayName:source.name,maxPoints:Infinity},(note,fraction)=>progress(`${source.name}: ${note}`,fraction),check,signal);
          const cloud = loaded.value as LoadedCloud;
          staged.ids.push(cloud.id); clouds.push(cloud);transfer.push(...loaded.transfer);
        }
        check();
        return {value:{token:staged.token,clouds,graph:graph?.value ?? null,map},transfer};
      } catch(error) { releasePreparedWorkspace(); throw error; }
      finally {workspacePreparing = false;}
    }
    case "workspace-commit": {
      if (workspacePreparing || preparedWorkspace?.token !== req.token) throw new Error("Workspace stage is unavailable");
      const staged = preparedWorkspace;
      vectorMap?.free(); poseGraph?.session.free();
      vectorMap = staged.map; setPoseGraph(staged.graph);
      preparedWorkspace = null;
      return {value:null,transfer:[]};
    }
    case "workspace-discard": {
      if (workspacePreparing) throw new Error("Workspace preparation is still running");
      if (preparedWorkspace?.token === req.token) releasePreparedWorkspace();
      return {value:null,transfer:[]};
    }
    case "memory-stats":
      return {value: {main: (await ready).memory.buffer.byteLength, pool: poolMemory(), mapHistory: vectorMap?.historyBytes() ?? 0, mapSteps: vectorMap?.undoDepth ?? 0}, transfer: []};
    case "release-pool":
      if (controllers.size > 1) throw new Error("Finish the active operation before releasing memory");
      releasePool(); return {value: null, transfer: []};
    case "copc-box":
      return readFullDensityBox(req, progress, check, signal);
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
      setPoseGraph(null);
      return { value: null, transfer: [] };
    case "trajectory":
      return { value: await readTrajectory(req.file), transfer: [] };
    case "trajectory-eval":
      return { value: evaluate(req), transfer: [] };
    case "load":
    case "load-url": {
      const { maxPoints } = req;
      const name = req.kind === "load" ? req.file.name : req.name;
      const displayName = req.kind === "load" ? req.displayName ?? name : name;
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
        const id = newId();
        items.set(id, { kind: "mesh", mesh: loaded.mesh, name: displayName });
        try { return describe(id, { parse, index: 0 }); }
        catch(error) {loaded.mesh.free();items.delete(id);throw error;}
      }
      const { cloud } = loaded;
      progress(`indexing ${cloud.length.toLocaleString()} points`);
      t = performance.now();
      let workers: number;
      try { workers = await buildIndex(cloud); check(); }
      catch(error) { cloud.free(); throw error; }
      const index = performance.now() - t;
      const id = newId();
      items.set(id, {
        kind: "cloud",
        cloud,
        name: displayName,
        keepEvery: loaded.keepEvery,
        filePoints: loaded.filePoints,
        copcLevels: loaded.copcLevels,
        sources: loaded.sources,
        detail: loaded.detail,
        copcSource: loaded.copcSource,
      });
      progress("preparing for display");
      try { return describe(id, { parse, index, workers }); }
      catch(error) {cloud.free();items.delete(id);throw error;}
    }
    case "gpu-prepare":
      await prepareGpuC2c();
      return { value: undefined, transfer: [] };
    case "cpu-prepare":
      await poolPreparation;
      return { value: undefined, transfer: [] };
    case "knn-cache-state":
      return { value: knnCacheState(), transfer: [] };
    case "knn-hint-state":
      return { value: lastKnnHintState, transfer: [] };
    case "mesh-cache-state":
      return { value: meshCacheState(), transfer: [] };
    case "c2c": {
      const compared = getCloud(req.compared);
      const reference = items.get(req.reference);
      if (!reference) throw new Error("reference not found");
      const start = performance.now();
      if (reference.kind === "mesh") {
        const parallel = await parallelCloudToMesh(compared, reference.mesh, req.signed, req.meshCache, req.meshGpu, req.meshOrderCache);
        const result = parallel
          ? summarizeDistances(parallel.distances, req.summarySort)
          : cloudToMesh(compared, reference.mesh, req.signed, req.summarySort);
        const out = output(result, start, parallel?.workers ?? 1, "c2m", req.signed);
        out.value.summarySort = req.summarySort === true;
        out.value.meshCached = parallel?.cached ?? false;
        out.value.meshCacheReused = parallel?.reused ?? false;
        out.value.meshCacheBytes = parallel?.retainedBytes;
        out.value.meshQueryOrderHit = parallel?.queryOrder?.hit ?? false;
        out.value.meshQueryOrderBytes = parallel?.queryOrder?.bytes ?? 0;
        out.value.meshQueryOrderEntries = parallel?.queryOrder?.entries ?? 0;
        out.value.meshSlices = parallel?.slices ?? 1;
        out.value.meshCacheFallback = parallel?.fallback;
        out.value.meshCompute = parallel?.compute;
        return out;
      }
      let fallback: string | undefined;
      if (req.backend === "webgpu") {
        try {
          const gpu = await gpuCloudToCloud(compared, reference.cloud, req.gpuPrefetch,
            req.gpuVerifyBounds, req.gpuReuseDevice, req.gpuVerifySimd, req.gpuBatchSize, req.gpuLeafOnly,
            req.gpuWorkgroupSize, req.gpuVerifyWorkers, req.gpuVerifySnapshot, req.gpuStageQueries, req.gpuMainWorker, req.gpuReuseImportBounds, req.gpuBufferedReadback, req.gpuQuerySlice, req.gpuQueryOwned, req.gpuInheritedBounds, req.gpuDiagnostics, req.gpuBoxBounds, req.gpuParentTraversal, req.gpuCompactStack, req.gpuPackedNodes, req.gpuMappedReference);
          const out = output(summarizeDistances(gpu.distances, req.summarySort), start, gpu.compute.verifyWorkers ?? 1, "c2c", false);
          out.value.compute = gpu.compute;
          out.value.summarySort = req.summarySort === true;
          return out;
        } catch (error) {
          fallback = error instanceof Error ? error.message : String(error);
        }
      }
      const parallel = req.cpuCache !== false
        ? await cachedCloudToCloud(compared, reference.cloud, req.cpuNearestSimd, req.cpuStream, req.cpuPipeline)
        : await parallelCloudToCloud(compared, reference.cloud, req.cpuNearestSimd, req.cpuStream);
      const result = parallel
        ? summarizeDistances(parallel.distances, req.summarySort)
        : cloudToCloud(compared, reference.cloud, req.cpuNearestSimd, req.summarySort);
      const out = output(result, start, parallel?.workers ?? 1, "c2c", false);
      out.value.summarySort = req.summarySort === true;
      out.value.cpuSimd = req.cpuNearestSimd === true && c2cSimdEnabled();
      out.value.cpuStreaming = !!parallel && req.cpuStream !== false;
      out.value.cpuCached = !!parallel && req.cpuCache !== false;
      out.value.cpuCacheReused = !!parallel && "reused" in parallel && parallel.reused === true;
      out.value.cpuPipelined = out.value.cpuCached && req.cpuPipeline === true;
      if (parallel && "inputBytes" in parallel && typeof parallel.inputBytes === "number")
        out.value.cpuCacheInputBytes = parallel.inputBytes;
      if (fallback) out.value.compute = { backend: "cpu", fallback, prepareMillis: 0,
        gpuMillis: 0, verifyMillis: 0, batches: 0, batchSize: 0, referenceReused: false, certified: 0,
        deviceReused: false, prefetch: false, verifyBounds: false, verifySimd: false, leafOnly: false, workgroupSize: 0 };
      return out;
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
      const { outcome, referenceReused, referenceCached, referenceBytes, correspondenceSimd,
        correspondenceTemporal, temporalQueries, temporalScratchBytes } = cachedRegisterIcp(
        moving,
        getCloud(req.reference),
        req.maxIterations,
        req.overlap,
        req.matchCentroids,
        req.pointToPlane,
        req.icpCache,
        req.icpSimd,
        req.icpTemporal,
      );
      const matrix = Array.from(outcome.matrix());
      moving.transform(new Float64Array(matrix));
      dropDetail(req.moving);
      const described = describe(req.moving);
      const value: IcpOutput = {
        referenceReused, referenceCached, referenceBytes, correspondenceSimd,
        correspondenceTemporal, temporalQueries, temporalScratchBytes,
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
    case "volume-support": {
      if (req.ids.length < 2 || req.ids.length > 9 || new Set(req.ids).size !== req.ids.length) throw new Error("Observed support requires a baseline and 1–8 different point clouds");
      const support = new ObservedVolumeSupport(req.cell, req.height, new Float64Array(req.sharedGrid));
      try {
        for (const id of req.ids) {
          const surface = VolumeSurface.fromCloud(getCloud(id));
          try { support.intersect(surface); } finally { surface.free(); }
        }
        const mask = support.takeMask();
        return { value: mask, transfer: [mask.buffer] };
      } finally { support.free(); }
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
      let out: ReturnType<typeof computeVolume>;
      try {
        out = computeVolume(before, after, req.cell, req.height, req.fillEmpty, req.region ? new Float64Array(req.region) : undefined, req.summaryOnly ?? false, req.sharedGrid ? new Float64Array(req.sharedGrid) : undefined, req.supportMask, req.heightThreshold);
      } finally { before.free(); after.free(); }
      const value: VolumeOutput = {
        added: out.added,
        removed: out.removed,
        addedArea: out.addedArea,
        removedArea: out.removedArea,
        matchedCells: out.matchedCells,
        totalCells: out.totalCells,
        outsideRegionCells: out.outsideRegionCells,
        cell: out.cell,
        cells: null,
        difference: null,
        millis: 0,
      };
      const transfer: Transferable[] = [];
      const review = out.takeHeightReview();
      if (review) {
        try { value.heightReview = { threshold: review.threshold, riseCells: review.riseCells, fallCells: review.fallCells,
          withinCells: review.withinCells, riseArea: review.riseArea, fallArea: review.fallArea,
          riseVolume: review.riseVolume, fallVolume: review.fallVolume };
          const packed = review.takeCells(), cells = [];
          for (let i = 0; i < packed.length; i += 6) cells.push({ index: packed[i], x: packed[i + 1], y: packed[i + 2], before: packed[i + 3], after: packed[i + 4], difference: packed[i + 5] });
          value.heightReview.cells = cells; value.heightReview.cellLimit = review.cellLimit;
          value.heightReview.unlistedCells = review.riseCells + review.fallCells - cells.length;
        }
        finally { review.free(); }
      }
      if (out.matchedCells > 0 && !req.summaryOnly) {
        const cells = out.takeCells();
        const id = newId();
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
      const id = newId();
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
      if(req.op==="voxel")requireVoxelRange(cloud,req.a);
      const k = Math.max(1, Math.floor(req.a));
      const means = req.op === "sor" ? await parallelSorMeans(cloud, k, req.knnCache, req.knnBounds, req.knnHint) : null;
      const filtered = means ? cloud.filterSor(means, req.b) : cloud.filter(req.op, req.a, req.b);
      await buildIndex(filtered);
      const id = newId();
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
    case "filter-batch": {
      const recipe=parseFilterRecipe(req.recipe);
      if(req.sources.length<1||req.sources.length>16||new Set(req.sources.map(s=>s.id)).size!==req.sources.length)
        throw new Error("Choose 1–16 different point clouds");
      if(req.sources.some(s=>!s.name||s.name.length>1024)||new Set(req.sources.map(s=>s.name)).size!==req.sources.length)
        throw new Error("Batch output names must be unique");
      if(req.sources.reduce((sum,source)=>sum+getCloud(source.id).length,0)>4000000)
        throw new Error("Recipe batches support at most 4 million loaded points");
      const owned=new Set<Cloud>(),registered:number[]=[],finals:{cloud:Cloud;name:string;millis:number;sources:string[]|null}[]=[];
      try {
        for(const [sourceIndex,source] of req.sources.entries()){
          let current:Cloud|null=null;const start=performance.now(),sourceItem=items.get(source.id);
          const sourceNames=sourceItem?.kind==="cloud"?sourceItem.sources?.slice()??null:null;
          for(const [stepIndex,step] of recipe.steps.entries()){
            progress(`Cloud ${sourceIndex+1}/${req.sources.length}, step ${stepIndex+1}/${recipe.steps.length}: ${step.op}`,(sourceIndex*recipe.steps.length+stepIndex)/(req.sources.length*recipe.steps.length));
            await new Promise(resolve=>setTimeout(resolve,0));check();
            const input:Cloud=current??getCloud(source.id),{op,a,b}=recipeParameters(step);
            if(op==="voxel")requireVoxelRange(input,a);
            const means:Float64Array|null=op==="sor"?await parallelSorMeans(input,a):null;check();
            const filtered:Cloud=means?input.filterSor(means,b):input.filter(op,a,b);
            owned.add(filtered);
            await buildIndex(filtered);check();
            if(current){owned.delete(current);current.free();}
            current=filtered;
          }
          finals.push({cloud:current!,name:source.name,millis:performance.now()-start,sources:sourceNames});
        }
        check();const value:LoadedCloud[]=[],transfer:Transferable[]=[];
        for(const final of finals){
          const id=newId();registered.push(id);items.set(id,{kind:"cloud",cloud:final.cloud,name:final.name,sources:final.sources??undefined});
          const output=describe(id,{parse:0,index:final.millis});value.push(output.value);transfer.push(...output.transfer);
        }
        owned.clear();
        return {value,transfer};
      } catch(error){
        for(const id of registered)items.delete(id);
        for(const cloud of owned)cloud.free();
        throw error;
      }
    }
    case "normals": {
      const cloud = getCloud(req.id);
      const k = Math.max(3, Math.floor(req.k));
      if (!(await parallelNormals(cloud, k, req.orientation, req.knnCache, req.knnBounds, req.knnHint))) cloud.estimateNormals(k, req.orientation);
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
      const id = newId();
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
    case "map-evaluate": {
      const start = performance.now();
      const { result, distances, quality, workers, cached, reused, inputBytes, voxelCacheHit, voxelReference } = await evaluateMapQuality(getCloud(req.compared),
        getCloud(req.reference), req.threshold, req.voxel, req.minPoints, req.streamSummaries, req.cpuCache, req.cpuPipeline, req.voxelSimd, req.voxelCache, req.voxelReferenceCache);
      const out = output(result, start, workers, "c2c", false,
        req.streamSummaries === false ? undefined : distances);
      out.value.quality = quality;
      out.value.voxelCacheHit = voxelCacheHit;
      out.value.voxelReference = voxelReference;
      out.value.cpuCached = cached;
      out.value.cpuCacheReused = reused;
      out.value.cpuPipelined = cached && req.cpuPipeline !== false;
      if (inputBytes !== undefined) out.value.cpuCacheInputBytes = inputBytes;
      return out;
    }
    case "map-quality": {
      const { scores, reused, reference } = snapshotVoxelScores(getCloud(req.compared), getCloud(req.reference), req.voxel, req.minPoints, req.voxelSimd, req.voxelCache, req.voxelReferenceCache);
      return { value: { ...scores, voxelCacheHit: reused, voxelReference: reference }, transfer: [] };
    }
    case "m3c2": {
      const start = performance.now();
      const compared = getCloud(req.compared), reference = getCloud(req.reference);
      const base = items.get(req.compared)!.name.replace(/\.[^.]+$/, "");
      const pending = req.parallel === false ? null : parallelM3c2(compared, reference,
        req.normalRadius, req.projectionRadius, req.maxDepth, req.coreSpacing, req.interleaved, req.radiusBounds, req.mainWorker, req.preparedTrees, req.cacheNormals, req.sharedSearch, req.fusedProjection, req.cacheCoreSelection);
      const result = pending ? await pending : { cloud: computeM3c2(compared, reference,
        req.normalRadius, req.projectionRadius, req.maxDepth, req.coreSpacing, req.radiusBounds, req.sharedSearch, req.fusedProjection), workers: 1, mainWorker: false,
        preparedTrees: false, treeCacheHit: false, normalCacheHit: false, normalCacheBytes: 0, fusedProjection: req.fusedProjection ?? false,
        coreSelectionCacheHit: false, coreSelectionCacheBytes: 0 };
      const { cloud, workers } = result;
      const id = newId();
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
        workers,
        radiusBounds: req.radiusBounds ?? false,
        sharedSearch: req.sharedSearch ?? true,
        fusedProjection: result.fusedProjection,
        mainWorker: result.mainWorker,
        preparedTrees: result.preparedTrees,
        treeCacheHit: result.treeCacheHit,
        normalCacheHit: result.normalCacheHit,
        normalCacheBytes: result.normalCacheBytes,
        coreSelectionCacheHit: result.coreSelectionCacheHit,
        coreSelectionCacheBytes: result.coreSelectionCacheBytes,
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
      const id = newId();
      const base = source!.name.replace(/\.[^.]+$/, "");
      const suffix = { classified: "csf", ground: "ground", objects: "objects" }[req.output];
      items.set(id, { kind: "cloud", cloud, name: `${base}_${suffix}` });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "crop": {
      const source = items.get(req.id);
      const cropped = getCloud(req.id).crop(new Float64Array(req.min), new Float64Array(req.max), req.inside);
      const id = newId();
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
        const id = newId();
        items.set(id, { kind: "cloud", cloud: part, name: `${base}_${suffix}` });
        const described = describe(id, { parse: 0, index: performance.now() - t });
        out.push(described.value);
        transfer.push(...described.transfer);
      }
      return { value: out, transfer };
    }
    case "change-core-support": {
      const cores = readCandidateCores(getCloud(req.id), req.rank, req.expectedPoints, req.bounds);
      const value = computeLaneCoreEvidence(cores, req.lanes, req.heightMargin);
      const transfer: Transferable[] = [value.positions.buffer, value.indices.buffer];
      for (const row of value.rows) if (row.mask) transfer.push(row.mask.buffer);
      return { value, transfer };
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
      const id = newId();
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
      const id = newId();
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
    case "discard-cloud":
    case "remove": {
      const item = items.get(req.id);
      if (item?.kind === "cloud") {
        invalidateCpuC2c(item.cloud);
        invalidateM3c2Preparation(item.cloud);
        invalidateM3c2CoreSelection(item.cloud);
        invalidateGpuReference(item.cloud);
        invalidateIcpReference(item.cloud);
        invalidateVoxelScores(item.cloud);
        invalidateKnnPreparation(item.cloud);
        invalidateMeshSources(item.cloud);
        item.cloud.free();
      }
      if (item?.kind === "mesh") { invalidateMeshSources(item.mesh); item.mesh.free(); }
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
  const message: WorkerMessage = { seq, response, memory, poolMemory: poolMemory() };
  self.postMessage(message, { transfer });
};
