// Serial vs automatic CPU/WASM workers, through the production worker.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output");
if (!output) throw new Error("Usage: node scripts/benchmark-m3c2.mjs --output REPORT.json [--sizes 10000,40000,100000,200000] [--repeats 3] [--core-spacing 0] [--compare-layouts] [--fixture grid|mixed-density]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const sizes = value("--sizes", "10000,40000,100000,200000").split(",").map(Number);
const repeats = Number(value("--repeats", "3"));
const coreSpacing = Number(value("--core-spacing", "0"));
const compareLayouts = args.includes("--compare-layouts");
const fixtureKind = value("--fixture", "grid");
if (!sizes.every(n => Number.isInteger(n) && n > 0 && n <= 2_000_000) ||
    !Number.isInteger(repeats) || repeats < 1 || !Number.isFinite(coreSpacing) || coreSpacing < 0 ||
    !["grid", "mixed-density"].includes(fixtureKind))
  throw new Error("Invalid sizes, repeats or core spacing");
const root = fileURLToPath(new URL("../dist/", import.meta.url));
const assets = resolve(root, "assets");
const workerName = readdirSync(assets).find(n => /^worker-.*\.js$/.test(n));
if (!workerName) throw new Error("Build the production app first");
const server = createServer((req, res) => {
  if (req.url === "/__m3c2-benchmark") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  const path = resolve(root, "." + new URL(req.url, "http://localhost").pathname);
  if (!path.startsWith(root.endsWith(sep) ? root : root + sep)) { res.writeHead(403).end(); return; }
  try {
    res.setHeader("Content-Type", ({ ".js": "text/javascript", ".wasm": "application/wasm" })[extname(path)] || "application/octet-stream");
    res.end(readFileSync(path));
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ channel: "chromium", headless: true });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/__m3c2-benchmark`);
  const report = await page.evaluate(async ({ workerPath, sizes, repeats, coreSpacing, compareLayouts, fixtureKind }) => {
    const worker = new Worker(workerPath, { type: "module" });
    let seq = 0;
    const waiting = new Map();
    worker.onmessage = ({ data }) => {
      if (!data.response) return;
      const p = waiting.get(data.seq);
      if (!p) return;
      waiting.delete(data.seq); clearTimeout(p.timer);
      data.response.ok ? p.resolve(data.response.value) : p.reject(new Error(data.response.error));
    };
    worker.onerror = event => {
      for (const p of waiting.values()) { clearTimeout(p.timer); p.reject(new Error(event.message)); }
      waiting.clear();
    };
    const call = req => new Promise((resolve, reject) => {
      const id = ++seq;
      const timer = setTimeout(() => { waiting.delete(id); reject(new Error(`Worker timeout: ${req.kind}`)); }, 60_000);
      waiting.set(id, { resolve, reject, timer });
      worker.postMessage({ seq: id, req });
    });
    const fixture = (n, lift) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 24);
      bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      const width = Math.ceil(Math.sqrt(n));
      for (let i = 0; i < n; i++) {
        let x = i % width * 0.1, y = Math.floor(i / width) * 0.1, phase = (i % width) / 20;
        if (fixtureKind === "mixed-density") {
          const denseCount = Math.floor(n / 4), dense = i < denseCount;
          const local = dense ? i : i - denseCount;
          const columns = Math.ceil(Math.sqrt(dense ? denseCount : n - denseCount));
          const step = dense ? 0.035 : 0.14;
          x = (dense ? 0 : 100) + local % columns * step;
          y = Math.floor(local / columns) * step;
          phase = x / 2;
        }
        [368_000 + x, 3_955_000 + y, 40 + Math.sin(phase) * 0.1 + lift]
          .forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], lift ? "compared.ply" : "reference.ply");
    };
    const parity = (serial, parallel) => {
      if (serial.cloud.count !== parallel.cloud.count) throw new Error("Core count mismatch");
      let mismatches = 0;
      for (const key of ["distance", "lod95", "significant"]) {
        const a = new Uint32Array(serial[key].buffer), b = new Uint32Array(parallel[key].buffer);
        if (a.length !== b.length) throw new Error("Missing core results");
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) mismatches++;
      }
      if (mismatches) throw new Error(`M3C2 bit parity failed: ${mismatches}`);
      return mismatches;
    };
    const exportResult = result => call({ kind: "export", id: result.cloud.id, format: "ply" });
    const cleanup = result => call({ kind: "remove", id: result.cloud.id });
    const cases = [];
    try {
      for (const n of sizes) {
        const reference = await call({ kind: "load", file: fixture(n, 0), maxPoints: n });
        const compared = await call({ kind: "load", file: fixture(n, 0.07), maxPoints: n });
        if (reference.count !== n || compared.count !== n) throw new Error("Unexpected thinning");
        const compute = (parallel, interleaved = true) => call({ kind: "m3c2", sharedSearch: false, compared: compared.id, reference: reference.id,
          normalRadius: 0.4, projectionRadius: 0.25, maxDepth: 0.5, coreSpacing, parallel, interleaved, preparedTrees: false });
        // Each mode runs once before measurement to compile its WASM kernels.
        const coldSerial = await compute(false), coldParallel = await compute(true);
        const coldContiguous = compareLayouts ? await compute(true, false) : undefined;
        parity(coldSerial, coldParallel);
        const a = await exportResult(coldSerial), b = await exportResult(coldParallel);
        const exportsEqual = a.length === b.length && a.every((v, i) => v === b[i]);
        if (!exportsEqual) throw new Error("M3C2 export parity failed");
        if (coldContiguous) {
          parity(coldSerial, coldContiguous);
          const c = await exportResult(coldContiguous);
          if (c.length !== b.length || !c.every((v, i) => v === b[i])) throw new Error("Contiguous export parity failed");
          if (coldContiguous.workers !== coldParallel.workers) throw new Error("Layout changed worker count");
          await cleanup(coldContiguous);
        }
        await cleanup(coldSerial); await cleanup(coldParallel);
        const trials = [];
        for (let i = 0; i < repeats; i++) {
          let serial, parallel, contiguous, order;
          if (compareLayouts) {
            const modes = ["serial", "contiguous", "interleaved"];
            order = modes.slice(i % 3).concat(modes.slice(0, i % 3));
            const results = {};
            for (const mode of order) results[mode] = await compute(mode !== "serial", mode !== "contiguous");
            ({ serial, contiguous, interleaved: parallel } = results);
          } else {
            order = i % 2 ? "parallel-first" : "serial-first";
            if (i % 2) { parallel = await compute(true); serial = await compute(false); }
            else { serial = await compute(false); parallel = await compute(true); }
          }
          const trial = { order, serialMillis: serial.millis,
            parallelMillis: parallel.millis, workers: parallel.workers, cores: parallel.cloud.count,
            mismatches: parity(serial, parallel) };
          if (contiguous) {
            if (contiguous.workers !== parallel.workers) throw new Error("Layout changed worker count");
            trial.contiguousMillis = contiguous.millis;
            trial.contiguousMismatches = parity(serial, contiguous);
            await cleanup(contiguous);
          }
          trials.push(trial);
          await cleanup(serial); await cleanup(parallel);
        }
        cases.push({ pointsPerCloud: n, cores: coldSerial.cloud.count, workers: coldParallel.workers,
          coldSerialMillis: coldSerial.millis, coldParallelMillis: coldParallel.millis,
          coldContiguousMillis: coldContiguous?.millis, exportsEqual, trials });
        await globalThis.benchmarkProgress(`${fixtureKind}, ${n} points: serial ${trials.map(t => Math.round(t.serialMillis))}; interleaved ${trials.map(t => Math.round(t.parallelMillis))}${compareLayouts ? `; contiguous ${trials.map(t => Math.round(t.contiguousMillis))}` : ""} ms; exact parity`);
        await call({ kind: "remove", id: reference.id }); await call({ kind: "remove", id: compared.id });
      }
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, cases };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, sizes, repeats, coreSpacing, compareLayouts, fixtureKind });
  const hash = name => createHash("sha256").update(readFileSync(resolve(assets, name))).digest("hex");
  Object.assign(report, { schemaVersion: 2, compareLayouts, coreBlock: 1024, coreTail: "balanced", createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    workerSha256: hash(workerName), poolWorkerSha256: hash(readdirSync(assets).find(n => /^c2c-worker-.*\.js$/.test(n))),
    wasmSha256: hash(readdirSync(assets).find(n => /^ca_wasm.*\.wasm$/.test(n))), browserVersion: browser.version(),
    machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
    params: { normalRadius: 0.4, projectionRadius: 0.25, maxDepth: 0.5, coreSpacing },
    fixture: fixtureKind === "grid" ? "UTM wavy grid, 0.1m spacing, 0.07m vertical offset"
      : "UTM wavy grids separated by 100m in x; first quarter of points at 0.035m spacing, remaining points at 0.14m spacing; 0.07m vertical offset",
    note: "Synthetic data. Requested CPU modes run sequentially. Three modes rotate when comparing layouts, otherwise serial and automatic modes alternate. Every mode is warmed before trials. Timings include core selection, input snapshots/copies, per-worker tree builds, computation, assembly and result indexing; file load/export and UI rendering are excluded. Automatic mode uses serial for small or memory-heavy jobs. Exact result attributes are checked each trial; full PLY bytes checked on warm-up. Other applications and CPU power/clocks are uncontrolled." });
  if (compareLayouts) report.note += " Contiguous and interleaved modes use identical complete input clouds, worker counts, tree kernels and per-core arithmetic. Interleaved mode packs 1024-core blocks cyclically across workers, balances the final partial cycle so lane counts differ by at most one core, and restores results to original core order. Every contiguous result also passes bit parity.";
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
  for (const c of report.cases) console.log(`${c.pointsPerCloud} points, ${c.cores} cores, ${c.workers} workers: serial ${c.trials.map(t => Math.round(t.serialMillis))} ms; auto ${c.trials.map(t => Math.round(t.parallelMillis))} ms; exact parity`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
