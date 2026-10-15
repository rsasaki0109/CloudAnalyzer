// Isolated production-WASM summary stage, including input copy and coloring allocation.
import { chromium } from "@playwright/test";
import { createServer } from "node:http";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { cpus, platform, release, totalmem } from "node:os";

const args = process.argv.slice(2);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const output = value("--output"), repeats = Number(value("--repeats", "10"));
const sizes = value("--sizes", "100000,100001,1000000,1000001,2000000,2000001").split(",").map(Number);
const fixtures = value("--fixtures", "constant,random,ordered").split(",");
if (!output || existsSync(output)) throw new Error("Supply a new --output report path");
if (!Number.isInteger(repeats) || repeats < 1 || !sizes.every(n => Number.isInteger(n) && n > 0 && n <= 10_000_000) ||
    !fixtures.every(f => ["constant", "random", "ordered"].includes(f))) throw new Error("Invalid repeats, sizes or fixtures");
const root = fileURLToPath(new URL("../dist/", import.meta.url));
const assets = resolve(root, "assets");
const wasmName = readdirSync(assets).find(n => /^ca_wasm.*\.wasm$/.test(n));
if (!wasmName) throw new Error("Build the production app first");
const gluePath = fileURLToPath(new URL("../src/wasm/ca_wasm.js", import.meta.url));
const server = createServer((req, res) => {
  if (req.url === "/__summary") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html>"); return; }
  if (req.url === "/__summary-wasm.js") { res.setHeader("Content-Type", "text/javascript"); res.end(readFileSync(gluePath)); return; }
  const path = resolve(root, "." + new URL(req.url, "http://localhost").pathname);
  if (!path.startsWith(root.endsWith(sep) ? root : root + sep)) { res.writeHead(403).end(); return; }
  try {
    res.setHeader("Content-Type", ({ ".wasm": "application/wasm", ".js": "text/javascript" })[extname(path)] || "application/octet-stream");
    res.end(readFileSync(path));
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ channel: "chromium", headless: true });
  const page = await browser.newPage();
  await page.exposeFunction("benchmarkProgress", message => console.log(message));
  await page.goto(`http://127.0.0.1:${server.address().port}/__summary`);
  const report = await page.evaluate(async ({ sizes, fixtures, repeats, wasmPath }) => {
    const { default: init, summarizeDistances } = await import("/__summary-wasm.js");
    await init({ module_or_path: wasmPath });
    const fields = ["count", "min", "max", "mean", "rms", "stdDev", "median"];
    const cases = [], warmupRounds = 5;
    for (const fixture of fixtures) for (const n of sizes) {
      let state = 0x5eed;
      const distances = Float64Array.from({ length: n }, () => {
        state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
        return fixture === "constant" ? 0.05000000000000426 : 0.01 + (state >>> 0) / 0x1_0000_0000;
      });
      if (fixture === "ordered") distances.sort(); // Fixture preparation is excluded.
      const measure = sorted => {
        const start = performance.now();
        const result = summarizeDistances(distances, sorted);
        return { millis: performance.now() - start, result };
      };
      const parity = (a, b) => {
        if (fields.some(f => !Object.is(a.result[f], b.result[f]))) throw new Error("Summary statistics changed");
        const first = new Uint32Array(a.result.distances().buffer), second = new Uint32Array(b.result.distances().buffer);
        if (first.length !== n || second.length !== n || first.some((v, i) => v !== second[i])) throw new Error("Summary coloring changed");
      };
      await globalThis.benchmarkProgress(`Summary ${fixture}, ${n} distances...`);
      for (let i = 0; i < warmupRounds; i++) {
        const a = measure(true), b = measure(false);
        try { parity(a, b); } finally { a.result.free(); b.result.free(); }
      }
      const trials = [];
      for (let i = 0; i < repeats; i++) {
        const order = i % 2 ? ["selected", "sorted"] : ["sorted", "selected"], results = {};
        try {
          for (const mode of order) results[mode] = measure(mode === "sorted");
          parity(results.sorted, results.selected);
          trials.push({ order, sortedMillis: results.sorted.millis, selectedMillis: results.selected.millis,
            statsEqual: true, mismatches: 0 });
        } finally { for (const value of Object.values(results)) value.result.free(); }
      }
      cases.push({ fixture, distances: n, trials });
      await globalThis.benchmarkProgress(`Completed: sort ${trials.map(t => Math.round(t.sortedMillis))}; select ${trials.map(t => Math.round(t.selectedMillis))} ms; exact parity`);
    }
    return { cases, warmupRounds, userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency };
  }, { sizes, fixtures, repeats, wasmPath: `/assets/${wasmName}` });
  report.schemaVersion = 1; report.createdAt = new Date().toISOString();
  report.browserVersion = browser.version();
  report.commit = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", windowsHide: true }).trim();
  report.workingTreeDirty = !!execFileSync("git", ["status", "--porcelain"], { encoding: "utf8", windowsHide: true }).trim();
  const hash = path => createHash("sha256").update(readFileSync(path)).digest("hex");
  report.wasmSha256 = hash(resolve(assets, wasmName)); report.glueSha256 = hash(gluePath);
  report.machine = { platform: platform(), release: release(), cpu: cpus()[0]?.model,
    logicalCpus: cpus().length, memoryBytes: totalmem() };
  report.note = "Isolated summarizeDistances with the production SIMD WASM and generated glue, no worker pool, rendering or GPU. Constant nonzero, deterministic varied positive distances, or the same varied values in ascending order. Preparation/fixture sorting is excluded. Times include JS-to-WASM f64 input copy, original-order min/max/sum/squared-sum, median, Rust f32 coloring allocation and the WASM call return. Statistic getters, f32 cloning/readback, exact parity checks and frees run outside timing. Five full-sort/selection warm-up pairs precede ten (or requested) alternating timed pairs. Nonzero uniform input skips the optimized median scratch copy; confirmed ordered inputs read their original middle values after endpoint checks and a complete total-order scan; other inputs select by total order. All pairs must match f64 statistics and f32 coloring bits. CPU clocks, power and other applications are uncontrolled. These stage timings are not end-to-end C2C speedups.";
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(`Saved ${output}`);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
