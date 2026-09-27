// Runs the Rust/WASM core off the UI thread. All clouds live here so that
// analyses can use full f64 coordinates without copying them back and forth.

import init, { Cloud, cloudToCloud, planCloudToCloud, summarizeDistances } from "./wasm/ca_wasm.js";
import { MIN_PARALLEL_QUERIES, poolSize, runSlices } from "./pool";
import type { C2cOutput, LoadedCloud, Request, Response, Vec3 } from "./protocol";

const ready = init();
const clouds = new Map<number, Cloud>();
let nextId = 1;
// Like CloudCompare's global shift: chosen from the first cloud, shared by all.
let shift: Vec3 | null = null;

/**
 * Split the job into spatially compact parts and run them on the worker pool.
 * Returns null when the job is too small for the split to pay off.
 */
async function parallelCloudToCloud(
  compared: Cloud,
  reference: Cloud,
): Promise<{ distances: Float64Array; workers: number } | null> {
  const parts = Math.min(poolSize(), Math.floor(compared.length / MIN_PARALLEL_QUERIES));
  if (parts < 2) return null;
  const plan = planCloudToCloud(compared, reference, parts);
  const slices = Array.from({ length: plan.length }, (_, k) => ({
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

async function handle(req: Request): Promise<{ value: unknown; transfer: Transferable[] }> {
  await ready;
  switch (req.kind) {
    case "load": {
      const cloud = Cloud.parse(req.name, new Uint8Array(req.bytes));
      shift ??= Array.from(cloud.suggestedShift()) as Vec3;
      const positions = cloud.positions(new Float64Array(shift));
      const colors = cloud.colors() ?? null;
      const lodNodes = cloud.lodNodes();
      const id = nextId++;
      clouds.set(id, cloud);
      const value: LoadedCloud = {
        id,
        name: req.name,
        count: cloud.length,
        positions,
        colors,
        bounds: Array.from(cloud.bounds()),
        shift,
        lodNodes,
        lodGrid: cloud.lodGrid,
      };
      const transfer: Transferable[] = [positions.buffer, lodNodes.buffer];
      if (colors) transfer.push(colors.buffer);
      return { value, transfer };
    }
    case "c2c": {
      const compared = clouds.get(req.compared);
      const reference = clouds.get(req.reference);
      if (!compared || !reference) throw new Error("cloud not found");
      const start = performance.now();
      const parallel = await parallelCloudToCloud(compared, reference);
      const result = parallel
        ? summarizeDistances(parallel.distances)
        : cloudToCloud(compared, reference);
      const distances = result.distances();
      const value: C2cOutput = {
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
        workers: parallel?.workers ?? 1,
      };
      result.free();
      return { value, transfer: [distances.buffer] };
    }
    case "remove": {
      clouds.get(req.id)?.free();
      clouds.delete(req.id);
      if (clouds.size === 0) shift = null;
      return { value: null, transfer: [] };
    }
  }
}

self.onmessage = async (event: MessageEvent<{ seq: number; req: Request }>) => {
  const { seq, req } = event.data;
  let response: Response;
  let transfer: Transferable[] = [];
  try {
    const out = await handle(req);
    response = { ok: true, value: out.value };
    transfer = out.transfer;
  } catch (err) {
    response = { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
  self.postMessage({ seq, response }, { transfer });
};
