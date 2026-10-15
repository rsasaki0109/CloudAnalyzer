import { expect, type Page, test } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import type { C2cOutput, LoadedCloud, Request } from "../src/protocol";

async function scenario(page: Page, failure?: "cloud-prepare" | "cloud-prepared" | "cloud-release" | "short") {
  await page.route("**/__map-cache-test", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__map-cache-test");
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    const prelude = `Object.defineProperty(navigator, "hardwareConcurrency", { value: 5 });
      const NativeWorker = Worker, lanes = [];
      let failOnce = ${JSON.stringify(failure ?? "")}, hold = false;
      Worker = class extends NativeWorker {
        held = new Set(); pending = [];
        constructor(...args) {
          super(...args); lanes.push(this);
          this.addEventListener("message", event => {
            if (this.held.has(event.data.seq)) {
              event.stopImmediatePropagation(); this.pending.push(event.data);
              self.postMessage({ mapReplyHeld: true }); return;
            }
            if (event.data.seq === this.shortened && event.data.ok) {
              event.data.value = event.data.value.subarray(0, event.data.value.length - 1);
            }
          });
        }
        postMessage(request, ...args) {
          if (request.kind === failOnce) {
            failOnce = "";
            this.dispatchEvent(new ErrorEvent("error", { message: "Injected Map quality worker loss" })); return;
          }
          if (request.kind === "cloud-prepared") {
            if (hold) this.held.add(request.seq);
            if (failOnce === "short") { failOnce = ""; this.shortened = request.seq; }
          }
          return super.postMessage(request, ...args);
        }
        release() {
          this.held.clear();
          for (const data of this.pending) this.dispatchEvent(new MessageEvent("message", { data }));
          this.pending = [];
        }
      };
      self.addEventListener("message", event => {
        const action = event.data.mapCacheControl;
        if (!action) return;
        event.stopImmediatePropagation();
        if (action === "idle-loss") lanes[0].dispatchEvent(new ErrorEvent("error", { message: "Injected idle loss" }));
        if (action === "hold") hold = true;
        if (action === "release") { hold = false; for (const lane of lanes) lane.release(); }
        self.postMessage({ mapCacheControlled: action });
      });\n`;
    await route.fulfill({ response, body: prelude + await response.text() });
  });
  const workerPath = "/assets/" + readdirSync("dist/assets").find(name => /^worker-.*\.js$/.test(name))!;
  return page.evaluate(async ({ workerPath, failure }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    const controls = new Map<string, () => void>();
    let held!: () => void;
    const heldReply = new Promise<void>(resolve => { held = resolve; });
    worker.onmessage = ({ data }) => {
      if (data.mapReplyHeld) { held(); return; }
      if (data.mapCacheControlled) { controls.get(data.mapCacheControlled)!(); controls.delete(data.mapCacheControlled); return; }
      if (!data.response) return;
      const pending = waiting.get(data.seq)!;
      waiting.delete(data.seq);
      data.response.ok ? pending.resolve(data.response.value) : pending.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const pending of waiting.values()) pending.reject(new Error(event.message)); waiting.clear(); };
    const control = (action: string) => new Promise<void>(resolve => {
      controls.set(action, resolve); worker.postMessage({ mapCacheControl: action });
    });
    const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
      const id = ++seq;
      waiting.set(id, { resolve: value => resolve(value as T), reject }); worker.postMessage({ seq: id, req });
    });
    const fixture = (count: number, lift: number) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + count * 24); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < count; i++) {
        const x = i % 401, y = Math.floor(i / 401);
        [368_000 + (x - x % 2) * 0.1, 3_955_000 + y * 0.1, 40 + Math.sin(x / 20) * 0.1 + (x < 200 ? lift : lift * 4)]
          .forEach((value, axis) => view.setFloat64(i * 24 + axis * 8, value, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    const equal = (a: C2cOutput, b: C2cOutput) => {
      const first = new Uint32Array(a.distances.buffer), second = new Uint32Array(b.distances.buffer);
      return first.length === second.length && first.every((bits, i) => bits === second[i]) &&
        Object.keys(a.stats).every(key => Object.is(a.stats[key as keyof typeof a.stats], b.stats[key as keyof typeof b.stats])) &&
        (!a.quality || !b.quality || Object.keys(a.quality).every(key => {
          const x = a.quality![key as keyof typeof a.quality], y = b.quality![key as keyof typeof b.quality];
          return Object.is(x, y) || ["awd", "scs"].includes(key) && Math.abs(x - y) <= 1e-12 * Math.max(1, Math.abs(x), Math.abs(y));
        }));
    };
    const matrix = [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 0.1, 0, 0, 0, 1];
    try {
      await call({ kind: "cpu-prepare" });
      const reference = await call<LoadedCloud>({ kind: "load", file: fixture(200_017, 0), maxPoints: 200_017 });
      const compared = await call<LoadedCloud>({ kind: "load", file: fixture(300_013, 0.07), maxPoints: 300_013 });
      const params = { reference: reference.id, compared: compared.id, threshold: 0.15, voxel: 1, minPoints: 10 };
      const evaluate = (cpuCache?: boolean, reverse = false, streamSummaries = true, changed = false, cpuPipeline?: boolean) =>
        call<C2cOutput>({ kind: "map-evaluate", ...params, cpuCache, streamSummaries, cpuPipeline,
          reference: reverse ? compared.id : reference.id, compared: reverse ? reference.id : compared.id,
          ...(changed ? { threshold: 0.001, voxel: 2, minPoints: 2 } : {}) });
      const baseline = await evaluate(false);
      if (failure === "cloud-prepare" || failure === "cloud-prepared" || failure === "short") {
        let rejected = false;
        try { await evaluate(); } catch (error) { rejected = String(error).includes(failure === "short" ? "Incomplete C2C" : "Injected Map quality"); }
        const retry = await evaluate(), repeated = await evaluate();
        return { rejected, parity: equal(baseline, retry) && equal(retry, repeated),
          cold: retry.cpuCacheReused === false, warm: repeated.cpuCacheReused === true, cached: retry.cpuCached, workers: retry.workers };
      }
      const cold = await evaluate(), warm = await evaluate(), collected = await evaluate(true, false, false);
      const acknowledgement = await evaluate(true, false, true, false, false);
      const changed = await evaluate(true, false, true, true), changedControl = await evaluate(false, false, true, true);
      if (!equal(baseline, cold) || !equal(cold, warm) || !equal(cold, collected) || !equal(cold, acknowledgement) || !equal(changed, changedControl)) throw new Error("Map cache output changed");
      const nearest = (cpuCache?: boolean) => call<C2cOutput>({ kind: "c2c", reference: reference.id, compared: compared.id, signed: false, cpuCache });
      const firstC2c = await nearest(), warmC2c = await nearest();
      if (failure === "cloud-release") {
        const retry = await evaluate(), repeated = await evaluate();
        return { parity: equal(baseline, retry) && equal(retry, repeated) && equal(firstC2c, warmC2c),
          cold: retry.cpuCacheReused === false, warm: repeated.cpuCacheReused === true, cached: retry.cpuCached, workers: retry.workers };
      }
      // Uncached Map quality controls leave the idle C2C preparation intact.
      await evaluate(false);
      const preservedC2c = await nearest(), afterSwitch = await evaluate(), afterSwitchWarm = await evaluate();
      await control("idle-loss");
      const rebuilt = await evaluate(), rebuiltWarm = await evaluate();
      await call({ kind: "transform", id: reference.id, matrix });
      const editedReference = await evaluate(), referenceControl = await evaluate(false);
      await call({ kind: "transform", id: compared.id, matrix });
      const editedBoth = await evaluate(), bothControl = await evaluate(false), bothWarm = await evaluate();
      const reverseControl = await evaluate(false, true);
      // Hold actual replies so both warm and replacement groups remain leased
      // through editing/removal; no source Cloud can be read after the await.
      await control("hold");
      const first = evaluate(), second = evaluate(true, true, false);
      await heldReply;
      for (const id of [compared.id, reference.id]) {
        await call({ kind: "transform", id, matrix }); await call({ kind: "remove", id });
      }
      await control("release");
      const pending = await Promise.all([first, second]);
      const ref = await call<LoadedCloud>({ kind: "load", file: fixture(100_017, 0), maxPoints: 100_017 });
      const comp = await call<LoadedCloud>({ kind: "load", file: fixture(100_013, 0.07), maxPoints: 100_013 });
      const reload = (cpuCache?: boolean) => call<C2cOutput>({ kind: "map-evaluate", ...params, reference: ref.id, compared: comp.id, cpuCache });
      const reloaded = await reload(), reloadControl = await reload(false), reloadedWarm = await reload();
      return { parity: equal(firstC2c, warmC2c) && equal(warmC2c, preservedC2c) && equal(baseline, afterSwitch) && equal(afterSwitch, afterSwitchWarm) &&
          equal(baseline, rebuilt) && equal(rebuilt, rebuiltWarm) && equal(editedReference, referenceControl) && equal(editedBoth, bothControl) &&
          equal(bothWarm, pending[0]) && equal(reverseControl, pending[1]) && equal(reloaded, reloadControl) && equal(reloaded, reloadedWarm),
        cold: [cold, firstC2c, afterSwitch, rebuilt, editedReference, editedBoth, reloaded].every(result => result.cpuCacheReused === false),
        warm: [warm, collected, changed, warmC2c, preservedC2c, afterSwitchWarm, rebuiltWarm, bothWarm, reloadedWarm].every(result => result.cpuCacheReused === true),
        parametersApplied: changed.quality!.threshold === 0.001 && changed.quality!.precision !== warm.quality!.precision,
        sourcesChanged: !equal(baseline, editedReference), cached: cold.cpuCached,
        pipelined: cold.cpuPipelined, waitingControl: acknowledgement.cpuPipelined === false,
        bytesRetained: cold.cpuCacheInputBytes! >= (compared.count + reference.count) * 24,
        workers: [cold.workers, reloaded.workers] };
    } finally { worker.terminate(); }
  }, { workerPath, failure });
}

test("Map quality reuses both directions, applies changed parameters and replaces C2C preparation", async ({ page }) => {
  expect(await scenario(page)).toMatchObject({ parity: true, cold: true, warm: true, parametersApplied: true,
    sourcesChanged: true, cached: true, pipelined: true, waitingControl: true, bytesRetained: true, workers: [4, 2] });
});

for (const failure of ["cloud-prepare", "cloud-prepared", "short"] as const) {
  test(`Map quality rebuilds after ${failure} failure`, async ({ page }) => {
    expect(await scenario(page, failure)).toMatchObject({ rejected: true, parity: true, cold: true, warm: true, cached: true, workers: 4 });
  });
}

test("Map quality and C2C recover when switching loses a release worker", async ({ page }) => {
  expect(await scenario(page, "cloud-release")).toMatchObject({ parity: true, cold: true, warm: true, cached: true, workers: 4 });
});

test("cold and warm CPU preparations dispatch before voxel work; the acknowledgement control waits", async ({ page }) => {
  const { stripTypeScriptTypes } = await import("node:module");
  const source = readFileSync("src/cpu-preparation.ts", "utf8").replace(/import .*?;\r?\n/g, "").replace(/\bexport /g, "");
  const result = await page.evaluate(async code => {
    const events: string[] = [];
    const make = new Function("releaseOn", "runOn", "runOnGeneration", "workerGeneration", code + "\nreturn prepareCpuSlices;");
    const submit = async (lane: number, request: { kind: string }) => {
      events.push(`${request.kind}:${lane}`);
      return request.kind === "cloud-prepared" ? new Float64Array([0.07]) : new Float64Array();
    };
    const prepare = make(async () => true, submit, (lane: number, _generation: number, request: { kind: string }) => submit(lane, request), () => 1);
    const cloud = { length: 2 };
    const inputs = () => [0, 1].map(() => ({
      reference: new Float64Array(3), queries: new Float64Array(3), indices: null,
    }));
    const job = prepare("map-quality", cloud, cloud, "1,1", inputs());
    const cold = job.run((lane: number) => events.push(`done:${lane}`));
    events.push("voxel");
    await cold;
    const first = [...events]; events.length = 0;
    const warm = job.run((lane: number) => events.push(`done:${lane}`));
    events.push("voxel");
    await warm;
    const second = [...events]; events.length = 0;
    const control = prepare("map-quality", cloud, cloud, "1,1", inputs());
    const gated = control.run((lane: number) => events.push(`done:${lane}`), undefined, false);
    events.push("voxel");
    await gated;
    return { cold: first, warm: second, control: events };
  }, stripTypeScriptTypes(source, { mode: "transform" }));
  expect(result.cold).toEqual(["cloud-prepare:0", "cloud-prepare:1", "cloud-prepared:0", "cloud-prepared:1", "voxel", "done:0", "done:1"]);
  expect(result.warm).toEqual(["cloud-prepared:0", "cloud-prepared:1", "voxel", "done:0", "done:1"]);
  expect(result.control).toEqual(["cloud-prepare:0", "cloud-prepare:1", "voxel", "cloud-prepared:0", "cloud-prepared:1", "done:0", "done:1"]);
});

test("a new WASM worker processes preparation and queued searches in order before initialization", async ({ page }) => {
  await page.route("**/__cold-cpu-worker", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__cold-cpu-worker");
  const workerPath = "/assets/" + readdirSync("dist/assets").find(name => /^c2c-worker-.*\.js$/.test(name))!;
  const result = await page.evaluate(async workerPath => {
    const worker = new Worker(workerPath, { type: "module" });
    const responses: { seq: number; ok: boolean; value?: Float64Array; error?: string }[] = [];
    try {
      const complete = new Promise<void>((resolve, reject) => {
        worker.onmessage = ({ data }) => { responses.push(data); if (responses.length === 4) resolve(); };
        worker.onerror = event => reject(new Error(event.message));
      });
      const reference = new Float64Array([0, 0, 0, 2, 0, 0]), queries = new Float64Array([1, 0, 0, 3, 0, 0, -1, 0, 0]);
      worker.postMessage({ seq: 1, kind: "cloud-prepare", job: 7, reference, queries }, { transfer: [reference.buffer, queries.buffer] });
      worker.postMessage({ seq: 2, kind: "cloud-prepared", job: 7, simd: false });
      worker.postMessage({ seq: 3, kind: "cloud-prepared", job: 7, simd: true });
      worker.postMessage({ seq: 4, kind: "cloud-release", job: 7 });
      await complete;
      return responses.map(response => ({ seq: response.seq, ok: response.ok, values: Array.from(response.value ?? []), error: response.error }));
    } finally { worker.terminate(); }
  }, workerPath);
  expect(result).toEqual([
    { seq: 1, ok: true, values: [], error: undefined },
    { seq: 2, ok: true, values: [1, 1, 1], error: undefined },
    { seq: 3, ok: true, values: [1, 1, 1], error: undefined },
    { seq: 4, ok: true, values: [], error: undefined },
  ]);
});

test("a failed preparation acknowledgement drains queued searches before releasing state", async ({ page }) => {
  const { stripTypeScriptTypes } = await import("node:module");
  const source = readFileSync("src/cpu-preparation.ts", "utf8").replace(/import .*?;\r?\n/g, "").replace(/\bexport /g, "");
  const result = await page.evaluate(async code => {
    let resolveSearch!: (values: Float64Array) => void;
    const held = new Promise<Float64Array>(resolve => { resolveSearch = resolve; });
    const released: number[] = [], searched: number[] = [];
    let writes = 0, settled = false, rejected = false;
    const make = new Function("releaseOn", "runOn", "runOnGeneration", "workerGeneration", code + "\nreturn prepareCpuSlices;");
    const prepare = make(async (lane: number) => { released.push(lane); return true; },
      async (lane: number) => { if (!lane) throw new Error("Rejected preparation acknowledgement"); },
      async (lane: number) => { searched.push(lane); return lane ? new Float64Array([0.07]) : held; }, () => 1);
    const cloud = { length: 2 };
    const job = prepare("map-quality", cloud, cloud, "1,1", [0, 1].map(() => ({
      reference: new Float64Array(3), queries: new Float64Array(3), indices: null,
    })));
    const pending = job.run(() => { writes++; }).catch((error: unknown) => {
      rejected = String(error).includes("Rejected preparation acknowledgement");
    }).finally(() => { settled = true; });
    // Let the rejected ack and other lane's successful response be handled;
    // the queued search remains explicitly gated, not a timed workload.
    await new Promise(resolve => setTimeout(resolve, 0));
    const before = { settled, writes, released: [...released], searched: [...searched] };
    resolveSearch(new Float64Array([0.07]));
    await pending;
    return { before, rejected, settled, writes, released, indicesReleased: job.indices.length === 0 };
  }, stripTypeScriptTypes(source, { mode: "transform" }));
  expect(result).toEqual({ before: { settled: false, writes: 0, released: [], searched: [0, 1] },
    rejected: true, settled: true, writes: 0, released: [0, 1], indicesReleased: true });
});
