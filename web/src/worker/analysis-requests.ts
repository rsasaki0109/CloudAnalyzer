// Analysis requests: distances, registration, volumes, rasters, map quality, change detection and trajectories.

import {
  alignPairs,
  cloudToCloud,
  c2cSimdEnabled,
  cloudToMesh,
  computeM3c2,
  computeVolume,
  ObservedVolumeSupport,
  rasterGeotiff,
  summarizeDistances,
  VolumeSurface,
} from "../wasm/ca_wasm.js";
import { gpuCloudToCloud, prepareGpuC2c } from "../gpu-c2c";
import { cachedCloudToCloud } from "../cpu-c2c";
import { parallelM3c2 } from "../m3c2-workers";
import { evaluateMapQuality } from "../map-quality-workers";
import { cachedRegisterIcp } from "../icp-preparation";
import { snapshotVoxelScores } from "../map-voxel-cache";
import { meshCacheState } from "../mesh-preparation";
import { knnCacheState } from "../knn-preparation";
import type {
  IcpOutput,
  M3c2Output,
  ProfileOutput,
  RasterOutput,
  Request,
  VolumeOutput,
  VolumeSide,
} from "../protocol";
import { computeLaneCoreEvidence, readCandidateCores } from "../lane-core-support";
import { describe, dropDetail, getCloud, items, newId, output } from "./items";
import { parallelCloudToCloud, parallelCloudToMesh } from "./distances";
import { lastKnnHintState } from "./knn";
import { evaluate, readTrajectory } from "./trajectory";

export const ANALYSIS_KINDS = [
  "trajectory",
  "trajectory-eval",
  "gpu-prepare",
  "knn-cache-state",
  "knn-hint-state",
  "mesh-cache-state",
  "c2c",
  "icp",
  "volume-support",
  "volume",
  "rasterize",
  "geotiff",
  "map-evaluate",
  "map-quality",
  "m3c2",
  "profile",
  "change-objects",
  "change-core-support",
  "align-pairs",
] as const;
export type AnalysisRequest = Extract<Request, { kind: (typeof ANALYSIS_KINDS)[number] }>;
const kinds: ReadonlySet<string> = new Set(ANALYSIS_KINDS);
export const isAnalysisRequest = (req: Request): req is AnalysisRequest => kinds.has(req.kind);

export async function analysisRequest(
  req: AnalysisRequest,
): Promise<{ value: unknown; transfer: Transferable[] }> {
  switch (req.kind) {
    case "change-objects": {
      const objects = getCloud(req.id).changedObjects(req.minChange, req.link, req.minPoints);
      return { value: objects, transfer: [objects.buffer] };
    }
    case "trajectory":
      return { value: await readTrajectory(req.file), transfer: [] };
    case "trajectory-eval":
      return { value: evaluate(req), transfer: [] };
    case "gpu-prepare":
      await prepareGpuC2c();
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
    case "change-core-support": {
      const cores = readCandidateCores(getCloud(req.id), req.rank, req.expectedPoints, req.bounds);
      const value = computeLaneCoreEvidence(cores, req.lanes, req.heightMargin);
      const transfer: Transferable[] = [value.positions.buffer, value.indices.buffer];
      for (const row of value.rows) if (row.mask) transfer.push(row.mask.buffer);
      return { value, transfer };
    }
    case "align-pairs": {
      const values = alignPairs(new Float64Array(req.moving), new Float64Array(req.reference));
      return { value: values, transfer: [values.buffer] };
    }
  }
}
