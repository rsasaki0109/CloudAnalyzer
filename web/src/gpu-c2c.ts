import { GpuC2cJob, GpuC2cReference, type Cloud } from "./wasm/ca_wasm.js";
import { GpuCandidateDevice, GpuCandidateSearch, validateGpuWorkgroupSize } from "./gpu-candidates";
import type { C2cOutput } from "./protocol";
import { GpuVerifiers, type GpuVerifierQueries } from "./gpu-verifiers";
import { poolSize } from "./pool";
import { GPU_DIAGNOSTIC_BATCH_LIMIT, timeGpuVerification, type GpuC2cDiagnostics, type GpuBatchTiming, type GpuLaneTiming } from "./gpu-diagnostics";

const DEFAULT_GPU_INHERITED_BOUNDS = false;
const DEFAULT_GPU_BOX_BOUNDS = false;
const DEFAULT_GPU_PARENT_TRAVERSAL = false;

interface SearchEntry { ready: Promise<GpuCandidateSearch>; gpu?: GpuCandidateSearch }
interface ReferenceEntry {
  cloud: Cloud;
  index: GpuC2cReference;
  searches: Map<boolean, SearchEntry>;
  users: number;
  retired: boolean;
  disposed: boolean;
  reuseDevice: boolean;
  verifiers?: Map<string, GpuVerifiers>;
}

// Keep at most one idle reference. Retired entries remain alive only while
// snapshot jobs use them; mutations must not destroy an in-flight readback.
let current: ReferenceEntry | undefined;

interface DeviceEntry { ready: Promise<GpuCandidateDevice>; device?: GpuCandidateDevice }
// At most two devices: ordinary requests never negotiate timestamp-query.
const sharedDevices = new Map<boolean, DeviceEntry>();

function acquireDevice(inheritedBounds = DEFAULT_GPU_INHERITED_BOUNDS, diagnostics = false): Promise<GpuCandidateDevice> {
  let sharedDevice = sharedDevices.get(diagnostics);
  if (sharedDevice?.device && !sharedDevice.device.usable) {
    sharedDevice.device.dispose();
    sharedDevices.delete(diagnostics);
    sharedDevice = undefined;
  }
  if (!sharedDevice) {
    const entry: DeviceEntry = { ready: GpuCandidateDevice.create(false, inheritedBounds, diagnostics) };
    sharedDevice = entry;
    sharedDevices.set(diagnostics, entry);
    entry.ready = entry.ready.then(device => { entry.device = device; return device; });
    // Handle failure immediately, even if synchronous snapshot preparation
    // later throws. A failed preparation must not prevent a future retry.
    void entry.ready.catch(() => { if (sharedDevices.get(diagnostics) === entry) sharedDevices.delete(diagnostics); });
  }
  return sharedDevice.ready;
}

/** Start device/pipeline preparation when the user selects WebGPU. */
export async function prepareGpuC2c(): Promise<void> { await acquireDevice(); }

function release(entry: ReferenceEntry): void {
  if (!entry.retired || entry.users || entry.disposed) return;
  entry.disposed = true;
  entry.index.free();
  for (const search of entry.searches.values()) search.gpu?.dispose();
  for (const verifier of entry.verifiers?.values() ?? []) verifier.dispose();
}

function retire(entry: ReferenceEntry): void {
  entry.retired = true;
  if (current === entry) current = undefined;
  release(entry);
}

/** Call after coordinate edits, and before freeing a source cloud. */
export function invalidateGpuReference(cloud: Cloud): void {
  if (current?.cloud === cloud) retire(current);
}

function completeHere(job: GpuC2cJob, first: number, candidates: Uint32Array, distances: Float64Array,
  timings?: GpuLaneTiming[]): number {
  if (!timings) { distances.set(job.complete(first, candidates), first); return 0; }
  const started = performance.now();
  const result = timeGpuVerification(() => job.verify(first, candidates));
  distances.set(result.distances, first);
  timings.push({ first, count: candidates.length, lane: 0, resident: true, queryExportMillis: 0,
    roundTripMillis: performance.now() - started, ...result.timing });
  return result.certified;
}

/** Experimental hybrid: GPU hints, authoritative f64 verification per query. */
export async function gpuCloudToCloud(compared: Cloud, reference: Cloud, prefetch = true,
  verifyBounds = true, reuseDevice = true, verifySimd = true, requestedBatchSize?: number, leafOnly = false,
  workgroupSize = 64, requestedVerifyWorkers = 1, verifySnapshot = true, stageQueries = false,
  mainWorker = false, reuseImportBounds = true, bufferedReadback = false, sliceQueryCopy = false,
  ownedQueryCopy = false, inheritedBounds = DEFAULT_GPU_INHERITED_BOUNDS, diagnosticsEnabled = false,
  boxBounds = DEFAULT_GPU_BOX_BOUNDS, parentTraversal = DEFAULT_GPU_PARENT_TRAVERSAL): Promise<{
  distances: Float64Array;
  compute: NonNullable<C2cOutput["compute"]>;
}> {
  if (!globalThis.navigator?.gpu) throw new Error("WebGPU is unavailable in this browser");
  validateGpuWorkgroupSize(workgroupSize);
  if (!Number.isSafeInteger(requestedVerifyWorkers) || requestedVerifyWorkers < 1 || requestedVerifyWorkers > 8)
    throw new Error("GPU verification workers must be an integer in 1..8");
  if (requestedBatchSize !== undefined && (!Number.isSafeInteger(requestedBatchSize) ||
    requestedBatchSize < 1 || requestedBatchSize > 262_144))
    throw new Error("WebGPU C2C batch size must be an integer in 1..262144");
  const start = performance.now();
  const diagnostics: GpuC2cDiagnostics | undefined = diagnosticsEnabled ? {
    timestampQuery: "unavailable", gpu: [], verification: [], gpuBatchesOmitted: 0, verificationBatchesOmitted: 0,
  } : undefined;
  if (!compared.length || !reference.length) throw new Error("WebGPU C2C requires nonempty clouds");
  if (compared.length > 2_000_000 || reference.length > 2_000_000) {
    throw new Error("WebGPU C2C currently supports up to 2,000,000 points per cloud");
  }
  let verifyWorkers = Math.min(requestedVerifyWorkers, poolSize());
  const resident = mainWorker && verifyWorkers > 1;
  const poolWorkers = verifyWorkers - Number(resident);
  // Parallel verification amortizes lane messages with larger batches once
  // there are at least four such batches. Serial/small jobs keep finer overlap.
  // Capture this before source mutations can run.
  const wantedBatchSize = requestedBatchSize ?? (compared.length >= 1_000_000 ||
    verifyWorkers >= 4 && compared.length >= 524_288 ? 131_072 : 65_536);
  const cachedSearch = current?.searches.get(diagnosticsEnabled)?.gpu;
  if (cachedSearch && !cachedSearch.usable) retire(current!);
  const referenceReused = current?.cloud === reference && current.reuseDevice === reuseDevice;
  const deviceReused = referenceReused && !!cachedSearch?.usable || reuseDevice && !!sharedDevices.get(diagnosticsEnabled)?.device?.usable;
  // Start native adapter discovery before synchronous tree/query preparation.
  // The snapshots are still owned before the first yield to other requests.
  const deviceReady = reuseDevice ? acquireDevice(inheritedBounds, diagnosticsEnabled) : undefined;
  if (!referenceReused) {
    if (current) retire(current);
    current = { cloud: reference, index: new GpuC2cReference(reference), users: 0,
      retired: false, disposed: false, reuseDevice, searches: new Map() };
  }
  const entry = current!;
  entry.users++;
  let job: GpuC2cJob | undefined;
  let gpu: GpuCandidateSearch | undefined;
  let fallback: string | undefined;
  let gpuMillis = 0;
  let gpuWaitMillis = 0;
  let verifyMillis = 0;
  let verifyPrepareMillis = 0;
  let verifyQueryPrepareMillis = 0;
  let boxBoundsPrepareMillis = 0;
  let boxBoundsReused = false;
  let parentTraversalPrepareMillis = 0;
  let parentTraversalReused = false;
  let parentSource: Uint32Array | undefined;
  let queryJob: GpuVerifierQueries | undefined;
  let verifyReferenceReused = false;
  let verifyFallback: string | undefined;
  let certified = 0;
  let batches = 0;
  type BatchResult = { ok: true; candidates: Uint32Array<ArrayBuffer> } | { ok: false; error: string };
  const pending: { started: number; result: Promise<BatchResult> }[] = [];
  const readbackAhead = bufferedReadback && prefetch;
  try {
    // Snapshot the queries synchronously, before GPU setup or a queued dispatch
    // yields to other worker requests that may transform/remove either cloud.
    job = new GpuC2cJob(compared, entry.index);
    job.useBounds(verifyBounds);
    job.useSimd(verifySimd);
    // Import independent exact trees once per immutable reference. Catch at
    // submission so a rejected lane cannot become an unhandled rejection.
    let verifiers: GpuVerifiers | undefined;
    let verificationReady: Promise<void> | undefined;
    if (verifyWorkers > 1) {
      entry.verifiers ??= new Map();
      const key = `${verifySnapshot}:${verifySnapshot && reuseImportBounds}`;
      verifiers = entry.verifiers.get(key);
      if (!verifiers) {
        verifiers = new GpuVerifiers(verifySnapshot, verifySnapshot && reuseImportBounds);
        entry.verifiers.set(key, verifiers);
      }
      verifyReferenceReused = verifiers.reused(poolWorkers);
      const started = performance.now();
      verificationReady = verifiers.prepare(entry.index, poolWorkers).catch(error => {
        verifyFallback = error instanceof Error ? error.message : String(error);
        verifiers = undefined;
        verifyWorkers = 1;
      }).finally(() => { verifyPrepareMillis = performance.now() - started; });
    }
    let search = entry.searches.get(diagnosticsEnabled);
    if (!search) {
      const packed = entry.index.candidateBuffers();
      try {
        const points = packed.takePoints(), nodes = packed.takeNodes();
        if (parentTraversal && !leafOnly) parentSource = nodes;
        const prepared: SearchEntry = { ready: (async () => GpuCandidateSearch.create(points, nodes,
          deviceReady ? await deviceReady : undefined, inheritedBounds, diagnosticsEnabled))() };
        prepared.ready = prepared.ready.then(gpu => {
          prepared.gpu = gpu;
          if (entry.disposed) gpu.dispose();
          return gpu;
        });
        entry.searches.set(diagnosticsEnabled, prepared);
        search = prepared;
      } finally { packed.free(); }
    }
    const prepareMillis = performance.now() - start;
    let t = performance.now();
    let deviceBatchSize = 65_536;
    try {
      gpu = await search!.ready;
      if (diagnostics) diagnostics.timestampQuery = gpu.timestampQueries ? "enabled" : "unsupported";
      deviceBatchSize = gpu.batchLimit(workgroupSize);
      if (!leafOnly && boxBounds && parentTraversal) throw new Error("Choose one GPU traversal variant");
      if (parentTraversal && !leafOnly) {
        parentTraversalReused = gpu.parentTraversalReady;
        const started = performance.now();
        try {
          await gpu.prepareParentTraversal(() => {
            if (parentSource) { const nodes = parentSource; parentSource = undefined; return nodes; }
            const packed = entry.index.candidateBuffers();
            try { return packed.takeNodes(); } finally { packed.free(); }
          });
        } finally { parentSource = undefined; parentTraversalPrepareMillis = performance.now() - started; }
      }
      if (boxBounds && !leafOnly) {
        boxBoundsReused = gpu.boxBoundsReady;
        const started = performance.now();
        try {
          await gpu.prepareBoxBounds(() => {
            const packed = entry.index.candidateBuffers();
            try { return { points: packed.takePoints(), nodes: packed.takeNodes() }; }
            finally { packed.free(); }
          });
        } finally { boxBoundsPrepareMillis = performance.now() - started; }
      }
    } catch (error) {
      fallback = error instanceof Error ? error.message : String(error);
      gpu = undefined;
      retire(entry);
    }
    const deviceWaitMillis = performance.now() - t;
    gpuMillis += deviceWaitMillis;
    gpuWaitMillis += deviceWaitMillis;
    const distances = new Float64Array(job.length);
    // A per-job limit permits controlled comparisons without rebuilding the
    // reference or device. Hardware limits still bound every dispatch/buffer.
    const batchSize = Math.min(wantedBatchSize, deviceBatchSize);
    const dispatch = (first: number) => {
      const started = performance.now();
      const count = Math.min(batchSize, job!.length - first);
      let timing: GpuBatchTiming | undefined;
      if (diagnostics) {
        if (diagnostics.gpu.length < GPU_DIAGNOSTIC_BATCH_LIMIT) {
          timing = { first, count, queryExportMillis: 0 };
          diagnostics.gpu.push(timing);
        } else diagnostics.gpuBatchesOmitted++;
      }
      const exportStarted = timing ? performance.now() : 0;
      const queries = job!.queries(first, count);
      if (timing) timing.queryExportMillis = performance.now() - exportStarted;
      // Handle rejection immediately, including while CPU verification runs.
      const result = gpu!.find(queries, leafOnly, workgroupSize, readbackAhead, inheritedBounds, timing, boxBounds, parentTraversal).then<BatchResult, BatchResult>(
        candidates => ({ ok: true, candidates }),
        error => ({ ok: false, error: error instanceof Error ? error.message || error.name : String(error) }),
      );
      return { started, result };
    };
    if (gpu) {
      pending.push(dispatch(0));
      if (readbackAhead && batchSize < job.length) pending.push(dispatch(batchSize));
    }
    await verificationReady;
    if (verifiers && stageQueries) {
      const started = performance.now();
      queryJob = verifiers.queries(batchSize, verifyWorkers, resident);
      try { await queryJob.prepare(job); }
      catch (error) {
        verifyFallback = error instanceof Error ? error.message : String(error);
        verifiers = undefined;
        verifyWorkers = 1;
      } finally { verifyQueryPrepareMillis = performance.now() - started; }
    }
    for (let first = 0; first < job.length; first += batchSize) {
      const count = Math.min(batchSize, job.length - first);
      let candidates = new Uint32Array(count).fill(0xffffffff);
      if (gpu && !pending.length) pending.push(dispatch(first));
      if (pending.length) {
        const next = pending.shift()!;
        const waitStarted = performance.now();
        const batch = await next.result;
        gpuWaitMillis += performance.now() - waitStarted;
        gpuMillis += performance.now() - next.started;
        if (batch.ok) {
          candidates = batch.candidates;
          batches++;
        } else {
          fallback = batch.error;
          await Promise.all(pending.splice(0).map(batch => batch.result));
          gpu!.dispose();
          gpu = undefined;
          retire(entry);
        }
        // One ahead overlaps CPU verification. With two slots, queue another
        // submission while the next readback is still in flight, in job order.
        const ahead = first + count + pending.length * batchSize;
        if (gpu && prefetch && ahead < job.length) pending.push(dispatch(ahead));
      }
      t = performance.now();
      const laneTimings = diagnostics && first / batchSize < GPU_DIAGNOSTIC_BATCH_LIMIT ? diagnostics.verification : undefined;
      if (diagnostics && !laneTimings) diagnostics.verificationBatchesOmitted++;
      if (verifiers) {
        try {
          certified += await verifiers.complete(job, first, candidates, verifyWorkers,
            verifyBounds, verifySimd, distances, queryJob, resident, sliceQueryCopy, ownedQueryCopy, laneTimings);
        } catch (error) {
          // All submitted lanes have settled. Recompute the entire batch from
          // this job's immutable snapshot, including any partially joined lane.
          verifyFallback = error instanceof Error ? error.message : String(error);
          verifiers = undefined;
          verifyWorkers = 1;
          certified += completeHere(job, first, candidates, distances, laneTimings);
        }
      } else certified += completeHere(job, first, candidates, distances, laneTimings);
      verifyMillis += performance.now() - t;
    }
    return { distances, compute: { backend: fallback ? "cpu" : "webgpu", fallback,
      prepareMillis, gpuMillis, gpuWaitMillis, verifyMillis, batches, batchSize, referenceReused, deviceReused,
      inheritedBounds: !leafOnly && !parentTraversal && inheritedBounds,
      boxBounds: !leafOnly && boxBounds, boxBoundsPrepareMillis, boxBoundsReused,
      boxBoundsBytes: gpu?.boxBoundsBytes ?? 0,
      parentTraversal: !leafOnly && parentTraversal, parentTraversalPrepareMillis, parentTraversalReused,
      parentTraversalBytes: gpu?.parentTraversalBytes ?? 0,
      certified: certified + job.certified, prefetch, bufferedReadback: readbackAhead,
      verifyBounds, verifySimd: job.simdEnabled, leafOnly, workgroupSize,
      verifyWorkers, verifyMainWorker: resident && verifyWorkers > 1,
      verifyPoolWorkers: verifyWorkers > 1 ? poolWorkers : 0,
      verifyPrepareMillis, verifyReferenceReused, verifyFallback, verifySnapshot,
      verifyReuseImportBounds: verifySnapshot && reuseImportBounds,
      verifyReferenceBytes: verifiers?.referenceBytes,
      verifyStagedQueries: !!queryJob && verifyWorkers > 1, verifyQueryPrepareMillis,
      querySliceCopy: sliceQueryCopy && !ownedQueryCopy, queryOwnedCopy: ownedQueryCopy,
      verifyQueryBytes: queryJob?.bytes, ...(diagnostics ? { diagnostics } : {}) } };
  } finally {
    await Promise.all(pending.map(batch => batch.result));
    queryJob?.dispose();
    job?.free();
    entry.users--;
    // An unexpected failure before GPU initialization should not retain a
    // partially prepared reference. Successful entries stay for reuse.
    if (!entry.searches.size) retire(entry);
    release(entry);
  }
}
