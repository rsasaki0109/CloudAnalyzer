// Previous three-RPC workflow vs combined CPU evaluation, using production workers.
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
if (!output) throw new Error("Usage: node scripts/benchmark-map-quality.mjs --output REPORT.json [--compare-summaries] [--sizes 10000,100000,200000,500000,1000000] [--repeats 6]");
if (existsSync(output)) throw new Error(`Report already exists: ${output}`);
const sizes = value("--sizes", "10000,100000,200000,500000,1000000").split(",").map(Number);
const repeats = Number(value("--repeats", "6"));
const compareSummaries = args.includes("--compare-summaries");
if (!sizes.every(n => Number.isInteger(n) && n > 0 && n <= 2_000_000) || !Number.isInteger(repeats) || repeats < 1)
  throw new Error("Invalid sizes or repeats");
const root = fileURLToPath(new URL("../dist/", import.meta.url));
const assets = resolve(root, "assets");
const workerName = readdirSync(assets).find(n => /^worker-.*\.js$/.test(n));
if (!workerName) throw new Error("Build the production app first");
const server = createServer((req, res) => {
  if (req.url === "/__quality-benchmark") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
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
  await page.goto(`http://127.0.0.1:${server.address().port}/__quality-benchmark`);
  const report = await page.evaluate(async ({ workerPath, sizes, repeats, compareSummaries }) => {
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
    const fixture = (n, changed) => {
      const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
      const bytes = new Uint8Array(header.length + n * 24);
      bytes.set(header);
      const view = new DataView(bytes.buffer, header.length);
      const width = Math.ceil(Math.sqrt(n));
      for (let i = 0; i < n; i++) {
        const x = i % width, y = Math.floor(i / width);
        [368_000 + x * 0.1, 3_955_000 + y * 0.1,
          40 + Math.sin(x / 20) * 0.1 + (changed ? x < width / 2 ? 0.07 : 1.1 : 0)]
          .forEach((v, a) => view.setFloat64(i * 24 + a * 8, v, true));
      }
      return new File([bytes], changed ? "compared.ply" : "reference.ply");
    };
    const inliers = (ds, threshold) => {
      let n = 0, sum = 0, squares = 0;
      for (const d of ds) if (d <= threshold) { n++; sum += d; squares += d * d; }
      return { share: n / ds.length, mean: n ? sum / n : NaN, rms: n ? Math.sqrt(squares / n) : NaN };
    };
    const parity = (previous, combined) => {
      const a = new Uint32Array(previous.distances.buffer), b = new Uint32Array(combined.distances.buffer);
      if (a.length !== b.length) throw new Error("Missing map distances");
      let mismatches = 0;
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) mismatches++;
      if (mismatches) throw new Error(`Distance bit parity failed: ${mismatches}`);
      for (const key of Object.keys(previous.stats)) {
        if (!Object.is(previous.stats[key], combined.stats[key])) throw new Error(`Stats mismatch: ${key}`);
      }
      let maxVoxelDifference = 0;
      for (const key of Object.keys(previous.quality)) {
        const a = previous.quality[key], b = combined.quality[key];
        if (Object.is(a, b)) continue;
        if (["awd", "scs"].includes(key)) {
          const delta = Math.abs(a - b);
          if (delta <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b))) { maxVoxelDifference = Math.max(maxVoxelDifference, delta); continue; }
        }
        throw new Error(`Quality mismatch: ${key}: ${a} vs ${b}`);
      }
      return { distanceBitMismatches: mismatches, statsExact: true, inlierMetricsExact: true, maxVoxelDifference };
    };
    const cases = [];
    try {
      for (const n of sizes) {
        await globalThis.benchmarkProgress(`Measuring ${n} points per cloud...`);
        const reference = await call({ kind: "load", file: fixture(n, false), maxPoints: n });
        const compared = await call({ kind: "load", file: fixture(n, true), maxPoints: n });
        if (reference.count !== n || compared.count !== n) throw new Error("Unexpected thinning");
        const params = { compared: compared.id, reference: reference.id, threshold: 0.15, voxel: 1, minPoints: 10 };
        const previous = async () => {
          const start = performance.now();
          const ours = await call({ kind: "c2c", compared: compared.id, reference: reference.id, signed: false, cpuCache: false });
          const theirs = await call({ kind: "c2c", compared: reference.id, reference: compared.id, signed: false, cpuCache: false });
          const voxels = await call({ kind: "map-quality", voxelCache: false, ...params });
          const a = inliers(ours.distances, params.threshold), b = inliers(theirs.distances, params.threshold);
          return { ...ours, quality: { threshold: params.threshold, accuracy: a.rms, precision: a.share,
            completeness: b.share, f1: a.share + b.share ? 2 * a.share * b.share / (a.share + b.share) : 0,
            chamfer: (a.mean + b.mean) / 2, ...voxels }, elapsedMillis: performance.now() - start };
        };
        const combined = async (streamSummaries = true) => {
          const start = performance.now();
          const result = await call({ kind: "map-evaluate", voxelCache: false, ...params, streamSummaries, cpuCache: false });
          return { ...result, elapsedMillis: performance.now() - start };
        };
        const coldPrevious = await previous(), coldCombined = await combined();
        const coldParity = parity(coldPrevious, coldCombined);
        const coldCollected = compareSummaries ? await combined(false) : undefined;
        if (coldCollected) parity(coldCollected, coldCombined);
        const trials = [];
        for (let i = 0; i < repeats; i++) {
          let a, b;
          let collected;
          let order;
          if (compareSummaries) {
            const modes = ["previous", "collected", "streamed"];
            order = modes.slice(i % 3).concat(modes.slice(0, i % 3));
            const results = {};
            for (const mode of order) results[mode] = mode === "previous" ? await previous() : await combined(mode === "streamed");
            a = results.previous; b = results.streamed; collected = results.collected;
          } else {
            order = i % 2 ? ["streamed", "previous"] : ["previous", "streamed"];
            if (i % 2) { b = await combined(); a = await previous(); }
            else { a = await previous(); b = await combined(); }
          }
          const trial = { order, previousMillis: a.elapsedMillis,
            combinedMillis: b.elapsedMillis, previousWorkersPerDirection: a.workers, combinedPoolWorkers: b.workers,
            ...parity(a, b) };
          if (collected) {
            if (collected.workers !== b.workers) throw new Error("Summary mode changed pool scheduling");
            trial.collectedMillis = collected.elapsedMillis;
            trial.collectedParity = parity(collected, b);
          }
          trials.push(trial);
        }
        cases.push({ pointsPerCloud: n, coldPreviousMillis: coldPrevious.elapsedMillis,
          coldCombinedMillis: coldCombined.elapsedMillis, coldCollectedMillis: coldCollected?.elapsedMillis, coldParity, trials });
        await globalThis.benchmarkProgress(`Completed ${n} points: previous ${trials.map(t => Math.round(t.previousMillis))}; combined ${trials.map(t => Math.round(t.combinedMillis))}${compareSummaries ? `; collect-all control ${trials.map(t => Math.round(t.collectedMillis))}` : ""} ms`);
        await call({ kind: "remove", id: reference.id }); await call({ kind: "remove", id: compared.id });
      }
      return { hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent, cases };
    } finally { worker.terminate(); }
  }, { workerPath: `/assets/${workerName}`, sizes, repeats, compareSummaries });
  const hash = name => createHash("sha256").update(readFileSync(resolve(assets, name))).digest("hex");
  Object.assign(report, { schemaVersion: 2, compareSummaries, createdAt: new Date().toISOString(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim(),
    workingTreeDirty: !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim(),
    workerSha256: hash(workerName), poolWorkerSha256: hash(readdirSync(assets).find(n => /^c2c-worker-.*\.js$/.test(n))),
    wasmSha256: hash(readdirSync(assets).find(n => /^ca_wasm.*\.wasm$/.test(n))), browserVersion: browser.version(),
    machine: { platform: platform(), release: release(), cpu: cpus()[0]?.model, logicalCpus: cpus().length, memoryBytes: totalmem() },
    params: { threshold: 0.15, voxel: 1, minPoints: 10 },
    fixture: "UTM wavy grid, 0.1m spacing; half shifted 0.07m, half shifted 1.1m",
    note: "Synthetic data. Requested CPU workflows run sequentially; three modes rotate when comparing summaries, otherwise two modes alternate. Each mode is warmed once before trials. Previous workflow retains its existing per-direction worker splitting. Wall timings include all worker RPCs, snapshots/copies, tree builds, both exact C2C searches, voxel scores, summaries and inlier metrics. File load and viewer rendering are excluded. Distances are bit-exact and statistics/inlier metrics exact every trial; existing voxel HashMap reductions allow 1e-12 relative/absolute roundoff. Combined workers count pool lanes (1 means native serial); the main WASM worker computes voxel scores concurrently. Other applications and CPU power/clocks are uncontrolled." });
  if (compareSummaries) report.note += " --compare-summaries rotates all three modes; collectedMillis retains the collect-all assembly, both distance summaries and an extra forward coloring copy, while combinedMillis streams slice assembly and forward summaries, omits unused reverse statistics and reuses the forward coloring array. Both combined modes use identical input snapshots, pool scheduling and voxel kernels. Every control result also passes parity.";
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
  for (const c of report.cases) console.log(`${c.pointsPerCloud} points: previous ${c.trials.map(t => Math.round(t.previousMillis))} ms; combined ${c.trials.map(t => Math.round(t.combinedMillis))} ms; exact distances/stats/inliers`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
