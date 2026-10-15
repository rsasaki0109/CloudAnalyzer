import { test, type Page } from "@playwright/test";
import { readdirSync } from "node:fs";
import type { C2cOutput, LoadedCloud, Request } from "../src/protocol";

export async function summaryScenario(page: Page, mode: "cpu" | "webgpu" | "mesh") {
  await page.route("**/__summary-worker", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__summary-worker");
  if (mode === "webgpu") test.skip(!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), "No WebGPU adapter");
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    await route.fulfill({ response, body: 'Object.defineProperty(navigator, "hardwareConcurrency", { value: 5 });\n' + await response.text() });
  });
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n))!;
  return page.evaluate(async ({ workerPath, mode }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const pending = waiting.get(data.seq)!; waiting.delete(data.seq);
      data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const p of waiting.values()) p.reject(new Error(event.message)); waiting.clear(); };
    const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
      const id = ++seq; waiting.set(id, { resolve: value => resolve(value as T), reject });
      worker.postMessage({ seq: id, req });
    });
    const fixture = (n: number, lift: number) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 24); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < n; i++) {
        const x = i % 401, y = Math.floor(i / 401);
        [368_000 + x * 0.1, 3_955_000 + y * 0.1, 40 + Math.sin(x / 20) * 0.1 + lift]
          .forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    const mesh = new File(["v 368000 3955000 40\nv 368100 3955000 40\nv 368100 3955100 40\nv 368000 3955100 40\nf 1 2 3 4\n"], "plane.obj");
    const cases = [];
    try {
      await call({ kind: "cpu-prepare" });
      for (const n of mode === "cpu" ? [1013, 1014, 200_013, 200_014] : mode === "mesh" ? [1014, 200_013] : [200_013, 200_014]) {
        const reference = await call<LoadedCloud>({ kind: "load", file: mode === "mesh" ? mesh : fixture(n + 4, 0), maxPoints: n + 4 });
        const compared = await call<LoadedCloud>({ kind: "load", file: fixture(n, 0.000_009), maxPoints: n });
        const c2c = (backend: "cpu" | "webgpu", summarySort?: boolean) => call<C2cOutput>({
          kind: "c2c", reference: reference.id, compared: compared.id, signed: mode === "mesh", backend, summarySort });
        const baseline = await c2c("cpu", true);
        const reused: boolean[] = [];
        for (const flag of [true, undefined, false, true]) {
          const result = await c2c(mode === "webgpu" ? "webgpu" : "cpu", flag);
          const a = new Uint32Array(baseline.distances.buffer), b = new Uint32Array(result.distances.buffer);
          if (a.length !== n || b.length !== n || a.some((v, i) => v !== b[i]) ||
              !Object.keys(baseline.stats).every(k => Object.is(baseline.stats[k as keyof typeof baseline.stats], result.stats[k as keyof typeof result.stats])))
            throw new Error("Selected median changed distances/statistics");
          if (result.summarySort !== (flag === true) || result.kind !== (mode === "mesh" ? "c2m" : "c2c"))
            throw new Error("Median controls did not propagate");
          if (mode === "webgpu") {
            if (result.compute?.backend !== "webgpu" || result.compute.batches !== 4 || result.workers !== 1)
              throw new Error("Expected partial GPU batches");
            reused.push(result.compute.referenceReused);
          } else if (result.workers !== baseline.workers || result.workers !== (n >= 200_000 ? 2 : 1)) {
            throw new Error("Median control changed CPU workers");
          }
        }
        cases.push({ n, workers: baseline.workers, reused, signed: baseline.signed,
          crossesPlane: mode !== "mesh" || baseline.stats.min < 0 && baseline.stats.max > 0 });
        await call({ kind: "remove", id: compared.id });
        await call({ kind: "remove", id: reference.id });
      }
      return cases;
    } finally { worker.terminate(); }
  }, { workerPath, mode });
}
