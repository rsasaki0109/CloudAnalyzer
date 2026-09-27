// Runs the Rust/WASM core off the UI thread. All clouds live here so that
// analyses can use full f64 coordinates without copying them back and forth.

import init, { Cloud, cloudToCloud } from "./wasm/ca_wasm.js";
import type { C2cOutput, LoadedCloud, Request, Response, Vec3 } from "./protocol";

const ready = init();
const clouds = new Map<number, Cloud>();
let nextId = 1;
// Like CloudCompare's global shift: chosen from the first cloud, shared by all.
let shift: Vec3 | null = null;

async function handle(req: Request): Promise<{ value: unknown; transfer: Transferable[] }> {
  await ready;
  switch (req.kind) {
    case "load": {
      const cloud = Cloud.parse(req.name, new Uint8Array(req.bytes));
      shift ??= Array.from(cloud.suggestedShift()) as Vec3;
      const positions = cloud.positions(new Float64Array(shift));
      const colors = cloud.colors() ?? null;
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
      };
      const transfer: Transferable[] = [positions.buffer];
      if (colors) transfer.push(colors.buffer);
      return { value, transfer };
    }
    case "c2c": {
      const compared = clouds.get(req.compared);
      const reference = clouds.get(req.reference);
      if (!compared || !reference) throw new Error("cloud not found");
      const start = performance.now();
      const result = cloudToCloud(compared, reference);
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
