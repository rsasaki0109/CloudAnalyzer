import { expect, type Page, test } from "@playwright/test";
import { readdirSync } from "node:fs";
import type { C2cOutput, LoadedCloud, Request } from "../src/protocol";

async function scenario(page: Page, failure?: "cloud-prepare" | "cloud-prepared" | "cloud-release") {
  await page.route("**/__c2c-cache", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__c2c-cache");
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    const prelude = `Object.defineProperty(navigator, "hardwareConcurrency", { value: 5 });
      const NativeWorker = Worker, lanes = [];
      Worker = class extends NativeWorker { constructor(...args) { super(...args); lanes.push(this); } };
      self.addEventListener("message", event => {
        if (event.data.killIdleCpuLane) {
          event.stopImmediatePropagation();
          lanes[0].dispatchEvent(new ErrorEvent("error", { message: "Injected idle worker loss" }));
          self.postMessage({ idleCpuLaneLost: true });
        }
      });
      const post = Worker.prototype.postMessage;
      let failOnce = ${JSON.stringify(failure ?? "")};
      Worker.prototype.postMessage = function(request, ...args) {
        if (request.kind === failOnce) {
          failOnce = "";
          this.dispatchEvent(new ErrorEvent("error", { message: "Injected CPU C2C worker loss" }));
          return;
        }
        return post.call(this, request, ...args);
      };\n`;
    await route.fulfill({ response, body: prelude + await response.text() });
  });
  const workerPath = "/assets/" + readdirSync("dist/assets").find(name => /^worker-.*\.js$/.test(name))!;
  return page.evaluate(async ({ workerPath, failure }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    let idleLost!: () => void;
    const idleLoss = new Promise<void>(resolve => { idleLost = resolve; });
    const waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (data.idleCpuLaneLost) { idleLost(); return; }
      if (!data.response) return;
      const pending = waiting.get(data.seq)!;
      waiting.delete(data.seq);
      data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const pending of waiting.values()) pending.reject(new Error(event.message)); waiting.clear(); };
    const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
      const id = ++seq;
      waiting.set(id, { resolve: value => resolve(value as T), reject });
      worker.postMessage({ seq: id, req });
    });
    const fixture = (count: number, lift: number) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + count * 24); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < count; i++) {
        const x = i % 401, y = Math.floor(i / 401);
        [368_000 + (x - x % 2) * 0.1, 3_955_000 + y * 0.1, 40 + Math.sin(x / 20) * 0.1 + lift]
          .forEach((value, axis) => view.setFloat64(i * 24 + axis * 8, value, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    const equal = (a: C2cOutput, b: C2cOutput) => {
      const first = new Uint32Array(a.distances.buffer), second = new Uint32Array(b.distances.buffer);
      return first.length === second.length && first.every((bits, i) => bits === second[i]) &&
        Object.keys(a.stats).every(key => Object.is(a.stats[key as keyof typeof a.stats], b.stats[key as keyof typeof b.stats]));
    };
    const matrix = [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 0.1, 0, 0, 0, 1];
    try {
      await call({ kind: "cpu-prepare" });
      const reference = await call<LoadedCloud>({ kind: "load", file: fixture(300_017, 0), maxPoints: 300_017 });
      const compared = await call<LoadedCloud>({ kind: "load", file: fixture(300_013, 9e-6), maxPoints: 300_013 });
      const c2c = (cpuCache?: boolean, reverse = false, cpuStream = true, cpuNearestSimd = false, cpuPipeline?: boolean) =>
        call<C2cOutput>({ kind: "c2c", reference: reverse ? compared.id : reference.id,
          compared: reverse ? reference.id : compared.id, signed: false, backend: "cpu", cpuCache, cpuStream, cpuNearestSimd, cpuPipeline });
      const baseline = await c2c(false);
      if (failure === "cloud-prepare" || failure === "cloud-prepared") {
        let rejected = false;
        try { await c2c(); } catch (error) { rejected = String(error).includes("Injected CPU C2C worker loss"); }
        const retry = await c2c(), warm = await c2c();
        return { rejected, parity: equal(baseline, retry) && equal(retry, warm),
          cached: retry.cpuCached, cold: retry.cpuCacheReused === false, warm: warm.cpuCacheReused, workers: retry.workers };
      }
      const cold = await c2c(), warm = await c2c(true), collected = await c2c(true, false, false, true);
      const queued = await c2c(true, false, true, false, true);
      if (!equal(baseline, cold) || !equal(cold, warm) || !equal(cold, collected) || !equal(cold, queued)) throw new Error("Cached CPU output changed");
      let idleRecovered = false;
      if (!failure) {
        worker.postMessage({ killIdleCpuLane: true }); await idleLoss;
        const rebuilt = await c2c(), repeated = await c2c();
        idleRecovered = equal(baseline, rebuilt) && equal(rebuilt, repeated) &&
          rebuilt.cpuCacheReused === false && repeated.cpuCacheReused === true;
      }
      const reverseBaseline = await c2c(false, true);
      if (failure === "cloud-release") {
        const reverse = await c2c(true, true), repeated = await c2c(true, true);
        return { parity: equal(reverseBaseline, reverse) && equal(reverse, repeated),
          cached: reverse.cpuCached, cold: reverse.cpuCacheReused === false, warm: repeated.cpuCacheReused, workers: reverse.workers };
      }
      // Reuse and replacement jobs own independent snapshots before mutations yield.
      const first = c2c(true), second = c2c(true, true);
      for (const id of [compared.id, reference.id]) {
        await call({ kind: "transform", id, matrix });
        await call({ kind: "remove", id });
      }
      const concurrent = await Promise.all([first, second]);
      const ref = await call<LoadedCloud>({ kind: "load", file: fixture(200_017, 0), maxPoints: 200_017 });
      const comp = await call<LoadedCloud>({ kind: "load", file: fixture(200_013, 9e-6), maxPoints: 200_013 });
      const compute = (cpuCache?: boolean) => call<C2cOutput>({ kind: "c2c", reference: ref.id, compared: comp.id, signed: false, backend: "cpu", cpuCache });
      const before = await compute(), beforeWarm = await compute();
      await call({ kind: "transform", id: ref.id, matrix });
      const refEdited = await compute(), refControl = await compute(false);
      await call({ kind: "transform", id: comp.id, matrix });
      const bothEdited = await compute(), bothControl = await compute(false), bothWarm = await compute();
      return { parity: equal(baseline, concurrent[0]) && equal(reverseBaseline, concurrent[1]) &&
          equal(before, beforeWarm) && equal(refEdited, refControl) && equal(bothEdited, bothControl) && equal(bothEdited, bothWarm),
        changed: !equal(before, refEdited),
        cold: cold.cpuCacheReused === false && before.cpuCacheReused === false &&
          refEdited.cpuCacheReused === false && bothEdited.cpuCacheReused === false,
        warm: warm.cpuCacheReused && beforeWarm.cpuCacheReused && bothWarm.cpuCacheReused,
        cached: cold.cpuCached, pipelined: cold.cpuPipelined, explicitPipeline: queued.cpuPipelined,
        streaming: warm.cpuStreaming, scalar: !warm.cpuSimd, vector: collected.cpuSimd, idleRecovered,
        workers: [cold.workers, before.workers] };
    } finally { worker.terminate(); }
  }, { workerPath, failure });
}

test("CPU C2C caches preparation, preserves controls and invalidates both edited sources", async ({ page }) => {
  expect(await scenario(page)).toMatchObject({ parity: true, changed: true, cold: true, warm: true,
    cached: true, pipelined: false, explicitPipeline: true, streaming: true, scalar: true, vector: true, idleRecovered: true, workers: [3, 2] });
});

for (const failure of ["cloud-prepare", "cloud-prepared"] as const) {
  test(`CPU C2C rebuilds after ${failure} worker loss`, async ({ page }) => {
    expect(await scenario(page, failure)).toMatchObject({ rejected: true, parity: true,
      cached: true, cold: true, warm: true, workers: 3 });
  });
}

test("CPU C2C cleanup avoids preparing on a worker lost during release", async ({ page }) => {
  expect(await scenario(page, "cloud-release")).toMatchObject({ parity: true,
    cached: true, cold: true, warm: true, workers: 3 });
});
