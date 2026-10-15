import { expect, type Page, test } from "@playwright/test";
import { readdirSync } from "node:fs";
import type { C2cOutput, LoadedCloud, Request } from "../src/protocol";

async function scenario(page: Page, controlled = false, fail = false, cache = false) {
  await page.route("**/__c2c-streaming", route => route.fulfill({ contentType: "text/html", body: "<!doctype html>" }));
  await page.goto("/__c2c-streaming");
  await page.route(/\/assets\/worker-[^/]+\.js$/, async route => {
    const response = await route.fetch();
    const prelude = `Object.defineProperty(navigator, "hardwareConcurrency", { value: 5 });
      let active = false, sent = 0, failOnce = ${fail}, writes = 0;
      const lanes = [];
      self.addEventListener("message", event => {
        if (event.data.releaseC2c) {
          event.stopImmediatePropagation();
          for (const lane of lanes) lane.postMessage({ releaseC2c: true });
        } else if (event.data.probeC2c) {
          event.stopImmediatePropagation(); self.postMessage({ c2cWrites: writes });
        } else if (event.data.req?.kind === "c2c") {
          active = ${controlled} && event.data.req.cpuStream !== false; sent = 0; writes = 0;
        }
      });
      const NativeWorker = Worker;
      Worker = class extends NativeWorker {
        held = -1;
        shortened = -1;
        released = false;
        pending;
        constructor(...args) {
          super(...args); lanes.push(this);
          this.addEventListener("message", event => {
            if (event.data.seq === this.shortened && event.data.ok) {
              event.data.value = event.data.value.subarray(0, event.data.value.length - 1);
              self.postMessage({ shortenedC2c: true });
            }
            if (event.data.seq === this.held && !this.released) {
              event.stopImmediatePropagation(); this.pending = event.data;
              self.postMessage({ heldC2c: true }); return;
            }
            if (event.data.seq === this.held) self.postMessage({ lateC2c: true });
          });
        }
        postMessage(message, ...args) {
          if (message.releaseC2c) {
            this.released = true;
            if (this.pending) {
              const data = this.pending; this.pending = undefined;
              this.dispatchEvent(new MessageEvent("message", { data }));
            }
            return;
          }
          if (message.kind === "cloud" || message.kind === "cloud-prepared") {
            sent++;
            this.held = active && sent === 2 ? message.seq : -1;
            this.shortened = active && failOnce && sent === 1 ? message.seq : -1;
            if (this.shortened !== -1) failOnce = false;
            this.released = false;
          }
          return super.postMessage(message, ...args);
        }
      };
      const NativeArray = Float64Array;
      Float64Array = new Proxy(NativeArray, { construct(target, args) {
        const values = Reflect.construct(target, args);
        if (!active || args[0] !== 200013) return values;
        return new Proxy(values, {
          get(target, key) { const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value; },
          set(target, key, value) {
            if (typeof key === "string" && /^\\d+$/.test(key)) {
              if (++writes === 1) self.postMessage({ joinedC2c: true });
            }
            return Reflect.set(target, key, value, target);
          },
        });
      } });\n`;
    await route.fulfill({ response, body: prelude + await response.text() });
  });
  const workerPath = "/assets/" + readdirSync("dist/assets").find(n => /^worker-.*\.js$/.test(n))!;
  return page.evaluate(async ({ workerPath, controlled, fail, cache }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    let announce: () => void, announceHeld: () => void, announceLate: () => void,
      announceShortened: () => void, answerProbe: (n: number) => void;
    const joined = new Promise<void>(resolve => { announce = resolve; });
    const held = new Promise<void>(resolve => { announceHeld = resolve; });
    const late = new Promise<void>(resolve => { announceLate = resolve; });
    const shortened = new Promise<void>(resolve => { announceShortened = resolve; });
    const probed = new Promise<number>(resolve => { answerProbe = resolve; });
    const waiting = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
    worker.onmessage = ({ data }) => {
      if (data.joinedC2c) { announce(); return; }
      if (data.heldC2c) { announceHeld(); return; }
      if (data.lateC2c) { announceLate(); return; }
      if (data.shortenedC2c) { announceShortened(); return; }
      if (data.c2cWrites !== undefined) { answerProbe(data.c2cWrites); return; }
      if (!data.response) return;
      const p = waiting.get(data.seq)!;
      waiting.delete(data.seq);
      data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = event => { for (const p of waiting.values()) p.reject(new Error(event.message)); waiting.clear(); };
    const call = <T>(req: Request) => new Promise<T>((resolve, reject) => {
      const id = ++seq;
      waiting.set(id, { resolve: value => resolve(value as T), reject });
      worker.postMessage({ seq: id, req });
    });
    const fixture = (n: number, lift: number) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 24); bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      for (let i = 0; i < n; i++) {
        const x = i % 401, y = Math.floor(i / 401);
        [368_000 + (x - x % 2) * 0.1, 3_955_000 + y * 0.1, 40 + Math.sin(x / 20) * 0.1 + lift]
          .forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    try {
      await call({ kind: "cpu-prepare" });
      const reference = await call<LoadedCloud>({ kind: "load", file: fixture(200_017, 0), maxPoints: 200_017 });
      const compared = await call<LoadedCloud>({ kind: "load", file: fixture(200_013, 9e-6), maxPoints: 200_013 });
      const c2c = (cpuStream?: boolean, cpuCache = cache) => call<C2cOutput>({ kind: "c2c", reference: reference.id, compared: compared.id,
        signed: false, backend: "cpu", cpuStream, cpuCache });
      const baseline = await c2c(false, false);
      let output: C2cOutput, rejected = false, writes: number | undefined, failure: string | undefined;
      if (fail) {
        const pending = c2c(true).catch(error => { failure = String(error); rejected = failure.includes("Incomplete C2C worker result"); });
        if (cache) { await Promise.all([held, shortened]); worker.postMessage({ releaseC2c: true }); }
        await pending;
        if (!cache) worker.postMessage({ releaseC2c: true });
        await late;
        worker.postMessage({ probeC2c: true }); writes = await probed;
        output = await c2c(false);
      } else {
        const pending = c2c(controlled ? true : undefined);
        if (controlled) {
          await Promise.all([joined, held]); // Joining proceeds while another result is withheld.
          for (const id of [compared.id, reference.id]) {
            await call({ kind: "transform", id, matrix: [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 0.1, 0, 0, 0, 1] });
            await call({ kind: "remove", id });
          }
          worker.postMessage({ releaseC2c: true });
        }
        output = await pending;
      }
      const a = new Uint32Array(baseline.distances.buffer), b = new Uint32Array(output.distances.buffer);
      return { parity: a.length === 200_013 && b.length === a.length && a.every((v, i) => v === b[i]),
        statsEqual: Object.keys(baseline.stats).every(k => Object.is(baseline.stats[k as keyof typeof baseline.stats], output.stats[k as keyof typeof output.stats])),
        workers: output.workers, streaming: output.cpuStreaming, simd: output.cpuSimd,
        cached: output.cpuCached, reused: output.cpuCacheReused, rejected, writes, failure };
    } finally { worker.terminate(); }
  }, { workerPath, controlled, fail, cache });
}

test("progressive CPU C2C joining matches the collected control exactly", async ({ page }) => {
  expect(await scenario(page)).toMatchObject({ parity: true, statsEqual: true, workers: 2, streaming: true, simd: false });
});

test("CPU C2C joins an early slice while a later slice is held and sources are removed", async ({ page }) => {
  expect(await scenario(page, true)).toMatchObject({ parity: true, statsEqual: true, workers: 2, streaming: true });
});

test("incomplete CPU C2C slices reject, ignore late results and recover on the same pool", async ({ page }) => {
  const result = await scenario(page, true, true);
  expect(result, JSON.stringify(result)).toMatchObject({ parity: true, statsEqual: true, workers: 2, streaming: false, rejected: true, writes: 0 });
});

test("cached CPU C2C joins exactly like uncached collected verification", async ({ page }) => {
  expect(await scenario(page, false, false, true)).toMatchObject({ parity: true, statsEqual: true,
    workers: 2, streaming: true, cached: true, reused: false });
});

test("cached CPU C2C keeps a held snapshot alive through source edits/removal", async ({ page }) => {
  expect(await scenario(page, true, false, true)).toMatchObject({ parity: true, statsEqual: true,
    workers: 2, streaming: true, cached: true });
});

test("incomplete cached CPU C2C drains held results and rebuilds preparation", async ({ page }) => {
  const result = await scenario(page, true, true, true);
  expect(result, JSON.stringify(result)).toMatchObject({ parity: true, statsEqual: true,
    workers: 2, streaming: false, cached: true, reused: false, rejected: true, writes: 0 });
});
