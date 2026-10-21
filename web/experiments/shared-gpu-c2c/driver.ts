// Measurement-only driver: imports the production candidate search and verifier pool.
import init, { Cloud, GpuC2cJob, GpuC2cReference, summarizeDistances } from "../../src/wasm/ca_wasm.js";
import { GpuCandidateDevice, GpuCandidateSearch } from "../../src/gpu-candidates";
import { GpuVerifiers, type GpuVerifierQueries } from "../../src/gpu-verifiers";
import { poolMemory, poolSize, releaseOn, warmUpPool } from "../../src/pool";

type Mode = "pool" | "pool-staged" | "pool-staged-overlap" | "shared-serial" | "shared" | "shared-overlap";
type Pipeline = "fixed" | "gpu";
interface Memory { committedBytes: number; sharedMemory: boolean; snapshotBytes: number; queryBytes: number; referenceGeneration: number; queryGeneration: number }
interface Verification extends Memory { distances: Float64Array; certified: number; millis: number; threads: number }
interface SharedInit extends Memory { threads: number; indices: number[]; isolated: boolean }
interface FileInput { name: string; bytes: Uint8Array }
const modes: Mode[] = ["pool", "pool-staged", "pool-staged-overlap", "shared-serial", "shared", "shared-overlap"];

function rpc(url: string) {
  const worker = new Worker(url, { type: "module" });
  let seq = 0;
  const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  worker.onmessage = ({ data }) => {
    const waiting = pending.get(data.seq);
    if (!waiting) return;
    clearTimeout(waiting.timer); pending.delete(data.seq);
    if (data.ok) waiting.resolve(data.value); else waiting.reject(new Error(data.error));
  };
  const fail = (error: Error) => { for (const waiting of pending.values()) { clearTimeout(waiting.timer); waiting.reject(error); } pending.clear(); };
  worker.onerror = event => fail(new Error(event.message || "Shared worker failed"));
  worker.onmessageerror = () => fail(new Error("Shared worker message failed"));
  return {
    call<T>(req: unknown, transfer: Transferable[] = []): Promise<T> {
      const id = ++seq;
      return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(() => { pending.delete(id); reject(new Error("Shared verifier timed out")); }, 60_000);
        pending.set(id, { resolve: value => resolve(value as T), reject, timer });
        try { worker.postMessage({ seq: id, req }, { transfer }); }
        catch (error) { clearTimeout(timer); pending.delete(id); reject(error); }
      });
    },
    close() { fail(new Error("Shared verifier closed")); worker.terminate(); },
  };
}

let main: Awaited<ReturnType<typeof init>>;
let threads = 0, batchSize = 0;
let shared: ReturnType<typeof rpc>;
let runtime: GpuCandidateDevice | undefined, gpu: GpuCandidateSearch | undefined;
let index: GpuC2cReference | undefined, job: GpuC2cJob | undefined;
let verifiers: GpuVerifiers | undefined, staged: GpuVerifierQueries | undefined;
let hints: Uint32Array | undefined;
let queryCloud: Cloud | undefined, freshJobs = false, jobSequence = 0;
let gpuRequested = false;
const memory = () => ({ mainCommittedBytes: main?.memory.buffer.byteLength ?? 0, poolCommittedBytes: poolMemory() });
const verifyShared = (first: number, candidates: Uint32Array, parallel: boolean) => {
  const copy = candidates.slice();
  return shared.call<Verification>({ kind: "verify", first, candidates: copy, parallel, bounds: true, simd: true }, [copy.buffer]);
};

function output(exact: Float64Array, certified: number, includeExact: boolean, extra = {}) {
  const summary = summarizeDistances(exact);
  try {
    const distances = summary.distances();
    return { distances, stats: { count: summary.count, min: summary.min, max: summary.max, mean: summary.mean,
      rms: summary.rms, stdDev: summary.stdDev, median: summary.median }, certified,
      ...(includeExact ? { exact } : {}), ...memory(), ...extra };
  } finally { summary.free(); }
}

async function prepare(reference: FileInput, compared: FileInput, requestedBatch: number, retainQueries: boolean) {
  if (job) throw new Error("Already prepared");
  const start = performance.now();
  const r = Cloud.parse(reference.name, reference.bytes), q = Cloud.parse(compared.name, compared.bytes);
  const referenceCount = r.length, queryCount = q.length;
  if (!referenceCount || referenceCount > 2_000_000 || !queryCount || queryCount > 2_000_000) {
    r.free(); q.free(); throw new Error("Invalid cloud counts");
  }
  try { index = new GpuC2cReference(r); job = new GpuC2cJob(q, index); }
  finally { r.free(); if (retainQueries) queryCloud = q; else q.free(); }
  batchSize = Math.min(requestedBatch, runtime?.batchLimit(64) ?? 262_144);
  const commonPrepareMillis = performance.now() - start;
  let t = performance.now();
  verifiers = new GpuVerifiers(true, true);
  await verifiers.prepare(index!, threads);
  const poolReferenceMillis = performance.now() - t;
  t = performance.now();
  staged = verifiers.queries(batchSize, threads);
  await staged.prepare(job!);
  const poolQueryMillis = performance.now() - t;
  t = performance.now();
  const snapshot = index!.snapshot(), queries = job!.queriesF64(0, job!.length);
  const snapshotBytes = snapshot.byteLength, queryBytes = queries.byteLength;
  const sharedMemory = await shared.call<Memory>({ kind: "prepare", snapshot, queries }, [snapshot.buffer, queries.buffer]);
  const sharedPrepareMillis = performance.now() - t;
  t = performance.now();
  if (runtime) {
    const buffers = index!.candidateBuffers();
    try { gpu = await GpuCandidateSearch.create(buffers.takePoints(), buffers.takeNodes(), runtime); }
    finally { buffers.free(); }
  }
  const gpuPrepareMillis = performance.now() - t;
  t = performance.now();
  hints = new Uint32Array(job!.length).fill(0xffffffff);
  if (gpu) for (let first = 0; first < job!.length; first += batchSize)
    hints.set(await gpu.find(job!.queries(first, Math.min(batchSize, job!.length - first))), first);
  return { referenceCount, queryCount, batchSize, commonPrepareMillis, poolReferenceMillis, poolQueryMillis,
    sharedPrepareMillis, gpuPrepareMillis, hintCaptureMillis: performance.now() - t,
    poolWorkers: threads, poolReferenceReused: verifiers.reused(threads),
    poolSnapshotTransferredBytes: verifiers.referenceBytes * threads, poolStagedQueryBytes: staged.bytes,
    sharedSnapshotTransferredBytes: snapshotBytes, sharedQueryBytes: queryBytes, sharedMemory,
    candidateSource: gpu ? "physical-webgpu" : "full-search-sentinel", ...memory() };
}

async function releaseStaged(queries: GpuVerifierQueries) {
  // Use the production release primitive and wait for every acknowledgement.
  // dispose() deliberately schedules releases without waiting in the application.
  const results = await Promise.allSettled(Array.from({ length: threads }, (_, lane) =>
    releaseOn(lane, { kind: "gpu-query-release", queryJob: queries.id })));
  if (results.some(result => result.status === "rejected" || !result.value))
    throw new Error("Staged query release was not acknowledged by every lane");
}

async function beginFreshJobs() {
  if (!job || !queryCloud || !staged || freshJobs) throw new Error("Retain parsed queries before starting fresh jobs");
  await releaseStaged(staged); staged = undefined;
  const released = await shared.call<Memory>({ kind: "release-queries" });
  if (released.queryBytes || released.referenceGeneration !== 1 || !released.snapshotBytes)
    throw new Error("Expected an empty query job with the reference still cached");
  try {
    await verifyShared(0, new Uint32Array([0]), true);
    throw new Error("Released queries were still available");
  } catch (error) { if (!String(error).includes("No prepared queries")) throw error; }
  job.free(); job = undefined; freshJobs = true;
  return { freshJobs, cachedReferenceGeneration: released.referenceGeneration, initialQueriesReleased: true,
    releasedQueryVerificationRejected: true };
}

async function run(mode: Mode, pipeline: Pipeline, includeExact: boolean) {
  if (!modes.includes(mode) || !["fixed", "gpu"].includes(pipeline)) throw new Error("Unknown comparison mode");
  if ((!job && !freshJobs) || !index || !verifiers || !hints) throw new Error("Prepare the clouds first");
  if (pipeline === "gpu" && !gpu) throw new Error("GPU pipeline not available");
  if (mode.endsWith("-overlap") && !freshJobs) throw new Error("Overlap comparison requires fresh query jobs");
  const verifier = mode.replace(/-overlap$/, ""), overlap = pipeline === "gpu" && mode.endsWith("-overlap");
  const started = performance.now(), sequence = ++jobSequence;
  const activeJob = freshJobs ? new GpuC2cJob(queryCloud!, index) : job!;
  const queryJobMillis = performance.now() - started;
  let activeStaged = freshJobs ? undefined : staged;
  let queryStageMillis = 0, queryReleaseMillis = 0, stagedQueryBytes = 0;
  let certified = 0, verifyMillis = 0, gpuWaitMillis = 0, sharedMemory: Memory | undefined;
  let sharedPrepared = false, queryReleaseAcknowledged = !freshJobs, result;
  let firstGpuDispatchMillis: number | undefined, gpuDispatchCount = 0;
  let queryStageStartedMillis = 0, queryStageFinishedMillis = 0;
  type Batch = { ok: true; candidates: Uint32Array } | { ok: false; error: unknown };
  const dispatch = (first: number) => {
    if (first === 0) firstGpuDispatchMillis = performance.now() - started;
    gpuDispatchCount++;
    // Catch immediately: query staging may still be awaiting a different worker.
    return gpu!.find(activeJob.queries(first, Math.min(batchSize, activeJob.length - first)))
      .then<Batch, Batch>(candidates => ({ ok: true, candidates }), error => ({ ok: false, error }));
  };
  let pending: Promise<Batch> | undefined;
  try {
    if (overlap) pending = dispatch(0);
    const stageStart = performance.now();
    queryStageStartedMillis = stageStart - started;
    if (freshJobs && verifier === "pool-staged") {
      activeStaged = verifiers.queries(batchSize, threads);
      await activeStaged.prepare(activeJob); stagedQueryBytes = activeStaged.bytes;
    } else if (freshJobs && verifier.startsWith("shared")) {
      const queries = activeJob.queriesF64(0, activeJob.length); stagedQueryBytes = queries.byteLength;
      sharedMemory = await shared.call<Memory>({ kind: "queries", queries }, [queries.buffer]);
      sharedPrepared = true;
      if (sharedMemory.referenceGeneration !== 1) throw new Error("Shared reference unexpectedly rebuilt");
    }
    queryStageMillis = performance.now() - stageStart;
    queryStageFinishedMillis = performance.now() - started;
    const distances = new Float64Array(activeJob.length);
    if (pipeline === "gpu" && !pending) pending = dispatch(0);
    for (let first = 0; first < activeJob.length; first += batchSize) {
      const count = Math.min(batchSize, activeJob.length - first);
      let candidates = hints.subarray(first, first + count);
      if (pending) {
        const t = performance.now(), batch = await pending;
        gpuWaitMillis += performance.now() - t; pending = undefined;
        if (!batch.ok) throw batch.error;
        candidates = batch.candidates;
        if (first + count < activeJob.length) pending = dispatch(first + count);
      }
      const t = performance.now();
      if (verifier.startsWith("pool")) {
        certified += await verifiers.complete(activeJob, first, candidates, threads, true, true, distances,
          verifier === "pool-staged" ? activeStaged : undefined);
      } else {
        const verified = await verifyShared(first, candidates, verifier === "shared");
        if (verified.distances.length !== count) throw new Error("Incomplete shared verification batch");
        distances.set(verified.distances, first); certified += verified.certified; sharedMemory = verified;
      }
      verifyMillis += performance.now() - t;
    }
    result = output(distances, certified, includeExact, { verifyMillis, gpuWaitMillis,
      poolReferenceReused: verifiers.reused(threads), ...(sharedMemory ? { sharedCommittedBytes: sharedMemory.committedBytes,
        sharedReferenceGeneration: sharedMemory.referenceGeneration, sharedQueryGeneration: sharedMemory.queryGeneration } : {}) });
  } finally {
    // A staging failure must not free owner queries while the early GPU batch is pending.
    if (pending) await pending;
    if (freshJobs) {
      const releaseStart = performance.now();
      try {
        if (activeStaged) await releaseStaged(activeStaged);
        if (sharedPrepared) {
          const released = await shared.call<Memory>({ kind: "release-queries" });
          if (released.queryBytes || released.referenceGeneration !== 1 || !released.snapshotBytes)
            throw new Error("Shared query release lost the cached reference");
        }
        queryReleaseAcknowledged = true;
      } finally { activeJob.free(); queryReleaseMillis = performance.now() - releaseStart; }
    }
  }
  return { ...result, millis: performance.now() - started, freshJob: freshJobs, jobSequence: sequence,
    queryJobMillis, queryStageMillis, queryReleaseMillis, stagedQueryBytes, queryReleaseAcknowledged,
    firstGpuDispatchMillis, gpuDispatchCount, queryStageStartedMillis, queryStageFinishedMillis,
    firstGpuDispatchedBeforeStaging: overlap && firstGpuDispatchMillis !== undefined && firstGpuDispatchMillis <= queryStageStartedMillis };
}

async function expected() {
  if (!job || !hints) throw new Error("Prepare the clouds first");
  const distances = new Float64Array(job.length);
  let certified = 0;
  for (let first = 0; first < job.length; first += batchSize) {
    const count = Math.min(batchSize, job.length - first);
    const full = job.verify(first, new Uint32Array(count).fill(0xffffffff));
    const seeded = job.verify(first, hints.subarray(first, first + count));
    try {
      const a = full.takeDistances(), b = seeded.takeDistances();
      const bitsA = new BigUint64Array(a.buffer), bitsB = new BigUint64Array(b.buffer);
      if (bitsA.some((v, i) => v !== bitsB[i])) throw new Error("GPU candidate verification differs from full search");
      distances.set(a, first); certified += seeded.certified;
    } finally { full.free(); seeded.free(); }
  }
  return output(distances, certified, true, { hints: hints.slice() });
}

async function guards() {
  if (!job) throw new Error("Prepare the clouds first");
  const reject = async (req: unknown, message: string) => {
    try { await shared.call(req); } catch (error) { if (String(error).includes(message)) return; throw error; }
    throw new Error(`Expected rejection: ${message}`);
  };
  await reject({ kind: "prepare", snapshot: new Uint8Array([1]), queries: new Float64Array([NaN, 0, 0]) }, "finite");
  await reject({ kind: "prepare", snapshot: new Uint8Array([1]), queries: new Float64Array([0, 0, 0]) }, "snapshot");
  await reject({ kind: "queries", queries: new Float64Array([NaN, 0, 0]) }, "finite");
  await reject({ kind: "queries", queries: new Float64Array(0) }, "triples");
  await reject({ kind: "verify", first: job.length, candidates: new Uint32Array([0]), parallel: true, bounds: true, simd: true }, "range");
  await reject({ kind: "verify", first: -1, candidates: new Uint32Array([0]), parallel: true, bounds: true, simd: true }, "request");
  return { rejectedNonfiniteReplacement: true, rejectedInvalidSnapshot: true, rejectedRange: true, rejectedNegativeRange: true, rejectedInvalidQueryReplacement: true };
}

type Request = { kind: "init"; threads: number; sharedWorkerUrl: string; gpu: boolean } |
  { kind: "prepare"; reference: FileInput; compared: FileInput; batchSize: number; retainQueries?: boolean } |
  { kind: "run"; mode: Mode; pipeline: Pipeline; exact?: boolean } |
  { kind: "expected" | "snapshot" | "guards" | "begin-fresh-jobs" | "finish" };
async function handle(req: Request) {
  if (req.kind === "init") {
    if (threads) throw new Error("Already initialized");
    if (!Number.isInteger(req.threads) || req.threads < 1 || req.threads > 8) throw new Error("Invalid thread count");
    threads = req.threads; gpuRequested = req.gpu;
    // Measurement-only cap, before any production pool worker is created.
    Object.defineProperty(navigator, "hardwareConcurrency", { value: threads + 1 });
    const start = performance.now(); main = await init();
    const wasmInitMillis = performance.now() - start;
    let t = performance.now(); await warmUpPool(); const poolInitMillis = performance.now() - t;
    shared = rpc(req.sharedWorkerUrl); t = performance.now();
    const sharedInit = await shared.call<SharedInit>({ kind: "init", build: "threads", threads });
    const sharedInitMillis = performance.now() - t;
    if (!sharedInit.sharedMemory || sharedInit.indices.some((v, i) => v !== i) || sharedInit.indices.length !== threads)
      throw new Error("Not every shared lane responded");
    t = performance.now(); if (gpuRequested) runtime = await GpuCandidateDevice.create(true);
    return { wasmInitMillis, poolInitMillis, sharedInitMillis, sharedInit, gpuInitMillis: performance.now() - t,
      physicalGpuRequired: gpuRequested, effectivePoolSize: poolSize(), ...memory() };
  }
  if (!main) throw new Error("Initialize the driver first");
  if (req.kind === "prepare") {
    if (!Number.isInteger(req.batchSize) || req.batchSize < 1 || req.batchSize > 262_144) throw new Error("Invalid batch size");
    return prepare(req.reference, req.compared, req.batchSize, req.retainQueries ?? false);
  }
  if (req.kind === "run") return run(req.mode, req.pipeline, req.exact ?? false);
  if (req.kind === "begin-fresh-jobs") return beginFreshJobs();
  if (req.kind === "expected") return expected();
  if (req.kind === "snapshot") {
    if (!index || !job) throw new Error("Prepare the clouds first");
    return { snapshot: index.snapshot(), queries: job.queriesF64(0, job.length) };
  }
  if (req.kind === "guards") return guards();
  if (req.kind === "finish") {
    const released = await shared.call<Memory>({ kind: "release" });
    if (released.snapshotBytes || released.queryBytes) throw new Error("Shared snapshot was not released");
    try { await verifyShared(0, new Uint32Array([0]), true); throw new Error("Expected released snapshot rejection"); }
    catch (error) { if (!String(error).includes("No prepared snapshot")) throw error; }
    shared.close(); staged?.dispose(); verifiers?.dispose(); gpu?.dispose(); runtime?.dispose(); job?.free(); index?.free(); queryCloud?.free();
    job = undefined; index = undefined; hints = undefined; queryCloud = undefined;
    return { releaseVerified: true, memory: released };
  }
  throw new Error("Unknown request");
}

let queue = Promise.resolve();
self.onmessage = ({ data }: MessageEvent<{ seq: number; req: Request }>) => {
  queue = queue.then(async () => {
    try {
      const value = await handle(data.req);
      const transfer: Transferable[] = [];
      for (const item of Object.values(value)) if (ArrayBuffer.isView(item)) transfer.push(item.buffer as ArrayBuffer);
      self.postMessage({ seq: data.seq, ok: true, value }, { transfer });
    } catch (error) { self.postMessage({ seq: data.seq, ok: false, error: error instanceof Error ? error.message : String(error) }); }
  });
};
