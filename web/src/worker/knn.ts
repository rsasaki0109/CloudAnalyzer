// Neighbour jobs on the worker pool: statistical outlier removal and normals.

import { Cloud, planSor } from "../wasm/ca_wasm.js";
import { finishSorMeans, makeSorQueries, scatterNormalParts } from "../knn-join";
import { acquireKnn, type KnnLease } from "../knn-preparation";
import { MIN_PARALLEL_QUERIES, poolSize, runOn, runSlices } from "../pool";

let sorJobs = 0;
type KnnHintMode = "cpu" | "webgpu" | "webgpu-knn";
interface KnnHintState {
  requested: KnnHintMode; backend: "cpu" | "cpu-hints" | "webgpu" | "webgpu-knn"; points: number; k: number; workers: number;
  prepareMillis: number; gpuMillis: number; verifyMillis: number; batches: number; referenceReused: boolean; gpuBytes: number; fallback?: string;
  completeRows?: number;
}
export let lastKnnHintState: KnnHintState | null = null;
function beginKnnHints(mode: KnnHintMode | undefined, points: number, k: number): KnnHintState | undefined {
  lastKnnHintState = null;
  if (mode === undefined) return;
  if (mode !== "cpu" && mode !== "webgpu" && mode !== "webgpu-knn") throw new Error("Invalid neighbourhood hint mode");
  const state: KnnHintState = { requested: mode, backend: "cpu", points, k, workers: 1,
    prepareMillis: 0, gpuMillis: 0, verifyMillis: 0, batches: 0, referenceReused: false, gpuBytes: 0 };
  lastKnnHintState = state; return state;
}
async function prepareKnnHints(lease: KnnLease, state: KnnHintState | undefined): Promise<Uint32Array[] | undefined> {
  if (!state) return;
  if (state.k > 32) throw new Error("Neighbourhood hint prototype supports k up to 32");
  state.workers = lease.indices.length;
  if (state.requested === "cpu") {
    const start = performance.now(); await lease.prepareHints();
    state.prepareMillis = performance.now() - start; state.backend = "cpu-hints"; return;
  }
  const gpu = await lease.gpuHints(state.k, state.requested === "webgpu-knn");
  state.prepareMillis = gpu.prepareMillis; state.gpuMillis = gpu.gpuMillis;
  state.batches = gpu.batches; state.referenceReused = gpu.referenceReused; state.gpuBytes = gpu.gpuBytes; state.backend = state.requested;
  state.completeRows = gpu.completeRows;
  return gpu.hints;
}
function failedKnnHints(state: KnnHintState | undefined, error: unknown): void {
  if (state) { state.backend = "cpu"; state.fallback ??= error instanceof Error ? error.message : String(error); }
}
// Combined ancestral bounds remain opt-in: measured whole-RPC gains were unstable.
const DEFAULT_KNN_BOUNDS = false;
/**
 * SOR on the worker pool (see `planSor`): each part settles most of its
 * points on its own; points whose neighbourhood reaches into other parts
 * are completed by the workers still holding those parts. Returns the
 * statistic per point, identical to the single-threaded one, or null when
 * the job is too small to split.
 */
export async function parallelSorMeans(cloud: Cloud, k: number, cache = true, bounds = DEFAULT_KNN_BOUNDS, hintMode?: KnnHintMode): Promise<Float64Array | null> {
  const n = cloud.length;
  const hintState = beginKnnHints(hintMode, n, k + 1);
  // One part per worker: each keeps its part indexed for step 2.
  const parts = Math.min(poolSize(), Math.floor(n / MIN_PARALLEL_QUERIES));
  if (parts < 2 || n <= k) { failedKnnHints(hintState, "Neighbourhood job uses the serial CPU path"); return null; }
  const lease = acquireKnn(cloud, parts, cache);
  if (lease) {
    try {
      const hints = await prepareKnnHints(lease, hintState), verifyStart = performance.now();
      const local = await lease.run((job, own) => ({ kind: "knn-local" as const, job, own, k, bounds, regions: lease.regions.slice(),
        gpuHints: hints?.[own], cpuHints: hintState?.backend === "cpu-hints", hintSlots: hintState?.backend === "webgpu-knn" }));
      const packed = makeSorQueries(local, lease.indices, n);
      const answers = await lease.run((job, lane) => ({ kind: "knn-within" as const, job, k, bounds, queries: packed.queries[lane] }));
      lease.assertSource();
      const means = finishSorMeans(local, answers, packed, k);
      if (hintState) hintState.verifyMillis = performance.now() - verifyStart;
      return means;
    } catch (error) {
      failedKnnHints(hintState, error);
      lease.retire();
      if (!lease.sourceValid()) throw error;
      // A lost preparation recovers through the existing fresh partition path.
    } finally { lease.release(); }
  }
  failedKnnHints(hintState, "Neighbourhood preparation is unavailable or over budget");
  const job = ++sorJobs;
  const plan = planSor(cloud, parts);
  const m = plan.length;
  const regions = plan.regions();
  const indices = Array.from({ length: m }, (_, j) => plan.indices(j));
  const lanes: number[] = [];
  try {
    const local = await runSlices(
      m,
      (j) => ({ kind: "sor-local" as const, job, points: plan.points(cloud, j), k, own: j, regions: regions.slice() }),
      lanes,
    );
    const packed = makeSorQueries(local, indices, n);
    const answers = await Promise.all(
      packed.queries.map((queries, j) =>
        queries.length === 0
          ? null
          : runOn(lanes[j], {
              kind: "sor-within" as const,
              job,
              queries,
              k,
            }),
      ),
    );
    return finishSorMeans(local, answers, packed, k);
  } finally {
    plan.free();
    await Promise.all([...new Set(lanes)].map((lane) => runOn(lane, { kind: "sor-release" as const, job })));
  }
}
/**
 * Normals on the worker pool: the cloud is split along its octree and each
 * part is done from its own points (near a part's border, neighbours from
 * across it are not used). Returns false when the cloud is too small to split.
 */
export async function parallelNormals(cloud: Cloud, k: number, orientation: "up" | "outward", cache = true, bounds = DEFAULT_KNN_BOUNDS, hintMode?: KnnHintMode): Promise<boolean> {
  const n = cloud.length;
  const hintState = beginKnnHints(hintMode, n, k);
  const parts = Math.min(poolSize(), Math.floor(n / MIN_PARALLEL_QUERIES));
  if (parts < 2) { failedKnnHints(hintState, "Neighbourhood job uses the serial CPU path"); return false; }
  const how = new Float64Array(cloud.normalsOrientation(orientation));
  const lease = acquireKnn(cloud, parts, cache);
  if (lease) {
    try {
      const hints = await prepareKnnHints(lease, hintState), verifyStart = performance.now();
      const results = await lease.run((job, lane) => ({ kind: "knn-normals" as const, job, k, bounds, orientation: how.slice(),
        gpuHints: hints?.[lane], cpuHints: hintState?.backend === "cpu-hints", hintSlots: hintState?.backend === "webgpu-knn" }));
      lease.assertSource();
      cloud.setNormals(scatterNormalParts(results, lease.indices, n));
      if (hintState) hintState.verifyMillis = performance.now() - verifyStart;
      return true;
    } catch (error) {
      failedKnnHints(hintState, error);
      lease.retire();
      if (!lease.sourceValid()) throw error;
    } finally { lease.release(); }
  }
  failedKnnHints(hintState, "Neighbourhood preparation is unavailable or over budget");
  const plan = planSor(cloud, parts);
  try {
    const how = new Float64Array(cloud.normalsOrientation(orientation));
    const indices = Array.from({ length: plan.length }, (_, j) => plan.indices(j));
    const results = await runSlices(plan.length, (j) => ({
      kind: "normals" as const,
      points: plan.points(cloud, j),
      k,
      orientation: how.slice(),
    }));
    cloud.setNormals(scatterNormalParts(results, indices, n));
    return true;
  } finally {
    plan.free();
  }
}
