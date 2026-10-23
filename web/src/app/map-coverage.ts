/** Coverage inspection is read-only, bounded, cancellable and scoped to loaded points. */
import * as THREE from "three";
import { projectChanged } from "../project-change";
import { workerBusy } from "../api";
import { MAX_COVERAGE_INPUT_BYTES, type CoverageGrid } from "../map-coverage";
import { $, download, errorText, fmt, setStatus } from "./dom";
import { addReportSection } from "./report";
import { clouds, distanceChanged, entries, listChanged, pointsInvalidated, toRender, viewer, type Entry } from "./state";
import { endTask, startTask, taskActive } from "./tasks";

const source = $<HTMLSelectElement>("coverage-source"), reference = $<HTMLSelectElement>("coverage-reference");
const cell = $<HTMLInputElement>("coverage-cell"), minimum = $<HTMLInputElement>("coverage-min");
const run = $<HTMLButtonElement>("coverage-run"), compare = $<HTMLButtonElement>("coverage-compare");
const canvas = $<HTMLCanvasElement>("coverage-grid");
let running = false, generation = 0;
let last: { entry: Entry; cloud: Entry["cloud"]; grid: CoverageGrid } | null = null;
function invalidate(): void {
  const hadResult = last !== null;
  generation++; last = null; $("coverage-result").hidden = true;
  $("coverage-cell-info").textContent = "";
  if (hadResult) distanceChanged.emit();
}
function render(): void {
  const all = clouds(), current = all.some(e => String(e.cloud.id) === source.value) ? source.value : String(all.at(-1)?.cloud.id ?? "");
  source.replaceChildren(...all.map(e => new Option(e.cloud.name, String(e.cloud.id)))); source.value = current;
  const others = all.filter(e => String(e.cloud.id) !== current), previous = reference.value;
  reference.replaceChildren(new Option("Choose an aligned reference", ""), ...others.map(e => new Option(e.cloud.name, String(e.cloud.id))));
  reference.value = others.some(e => String(e.cloud.id) === previous) ? previous : "";
  const entry = entries.get(Number(current));
  if (last && (last.entry !== entry || last.cloud !== entry?.cloud)) invalidate();
  if (cell.dataset.for !== current) {
    invalidate(); cell.dataset.for = current;
    if (entry) cell.value = String(Math.max(entry.cloud.bounds[3] - entry.cloud.bounds[0], entry.cloud.bounds[4] - entry.cloud.bounds[1]) / 128 || 1);
  }
  $("coverage-scope").textContent = !entry ? "Open or build a point cloud to analyze."
    : `${entry.cloud.count.toLocaleString()} loaded points. ` +
      (entry.cloud.keepEvery > 1 || entry.cloud.count < entry.cloud.filePoints || entry.cloud.copcLevels !== null || entry.origin.displayPreview
        ? "Sampled/partial data: this measures only loaded points, not the full source map."
        : "Analysis uses the loaded cloud, not an expected survey footprint.");
  run.disabled = !entry || running; compare.disabled = !entry || !reference.value || running;
  source.disabled = running || !entry; reference.disabled = running || !others.length;
  cell.disabled = minimum.disabled = running || !entry;
  $<HTMLButtonElement>("pg-map-coverage").disabled = running || !all.length;
}
listChanged.add(render);
pointsInvalidated.add(id => { if (id === Number(source.value)) invalidate(); });
source.onchange = () => { invalidate(); render(); };
reference.onchange = render;
cell.oninput = minimum.oninput = invalidate;

$("pg-map-coverage").onclick = () => {
  const latest = clouds().at(-1); if (!latest) return;
  source.value = String(latest.cloud.id); render(); $("coverage-panel").scrollIntoView({ block: "start" });
};

function compute(entry: Entry, signal: AbortSignal): Promise<CoverageGrid> {
  if (entry.cloud.positions.byteLength > MAX_COVERAGE_INPUT_BYTES) throw new Error("Coverage input exceeds 64 MiB; crop a smaller area");
  const points = entry.cloud.positions.slice();
  const worker = new Worker(new URL("../map-coverage-worker.ts", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    const cleanup = () => { worker.terminate(); signal.removeEventListener("abort", abort); };
    const abort = () => { cleanup(); reject(new DOMException("Cancelled", "AbortError")); };
    worker.onmessage = event => { cleanup(); event.data.error ? reject(new Error(event.data.error)) : resolve(event.data.grid); };
    worker.onerror = () => { cleanup(); reject(new Error("Coverage worker failed")); };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) { abort(); return; }
    try { worker.postMessage({ positions: points, shift: entry.cloud.shift, cell: Number(cell.value), minPoints: Number(minimum.value) }, [points.buffer]); }
    catch (error) { cleanup(); reject(error); }
  });
}
function showGrid(grid: CoverageGrid): void {
  canvas.width = grid.nx; canvas.height = grid.ny;
  const scale = Math.min(250 / grid.nx, 230 / grid.ny);
  canvas.style.width = `${Math.max(1, grid.nx * scale)}px`; canvas.style.height = `${Math.max(1, grid.ny * scale)}px`;
  const ctx = canvas.getContext("2d")!, image = ctx.createImageData(grid.nx, grid.ny);
  const colors = [[40, 45, 54], [70, 165, 225], [255, 183, 55], [255, 85, 135]];
  for (let y = 0; y < grid.ny; y++) for (let x = 0; x < grid.nx; x++) {
    const offset = ((grid.ny - 1 - y) * grid.nx + x) * 4, color = colors[grid.categories[y * grid.nx + x]];
    image.data.set([...color, 255], offset);
  }
  ctx.putImageData(image, 0, 0);
  $("coverage-summary").textContent = `${grid.finitePoints.toLocaleString()} finite points · ${grid.occupied.toLocaleString()} occupied cells · ${grid.sparse.toLocaleString()} sparse (<${grid.minPoints} points/cell) · ${grid.enclosedEmpty.toLocaleString()} enclosed empty cells. Median occupied-cell density ${fmt(grid.density.median)} points/unit². ${grid.invalidPoints.toLocaleString()} invalid points excluded.`;
  $("coverage-result").hidden = false;
}
run.onclick = async () => {
  if (running || taskActive() || workerBusy()) return setStatus("Finish the current operation before analyzing coverage", true);
  const entry = entries.get(Number(source.value)); if (!entry) return;
  invalidate(); const token = generation, cloud = entry.cloud, signal = startTask(); running = true; render();
  try {
    setStatus(`Analyzing loaded XY coverage of ${cloud.name}…`);
    const grid = await compute(entry, signal); signal.throwIfAborted();
    if (token !== generation || entries.get(cloud.id) !== entry || entry.cloud !== cloud) throw new Error("Source changed during coverage analysis; run it again");
    last = { entry, cloud, grid }; showGrid(grid); distanceChanged.emit(); setStatus("Coverage ready. Click a grid cell to inspect it; enclosed empty cells are potential gaps, not confirmed defects.");
  } catch (error) { setStatus("Coverage analysis failed: " + (signal.aborted ? "Cancelled" : errorText(error)), true); }
  finally { running = false; endTask(signal); render(); }
};
canvas.onclick = event => {
  if (!last || running || taskActive()) return;
  const { grid } = last, rect = canvas.getBoundingClientRect();
  const x = Math.min(grid.nx - 1, Math.max(0, Math.floor((event.clientX - rect.left) / rect.width * grid.nx)));
  const y = grid.ny - 1 - Math.min(grid.ny - 1, Math.max(0, Math.floor((event.clientY - rect.top) / rect.height * grid.ny)));
  const minX = grid.minX + x * grid.cell, minY = grid.minY + y * grid.cell, i = y * grid.nx + x;
  viewer.frameBox(new THREE.Box3(toRender([minX, minY, grid.minZ]), toRender([minX + grid.cell, minY + grid.cell, grid.maxZ])).expandByScalar(grid.cell));
  projectChanged();
  $("coverage-cell-info").textContent = `Cell (${x}, ${y}): ${grid.counts[i]} points; ${["exterior-connected empty", "occupied", "sparse", "enclosed empty — inspect source/occlusion"][grid.categories[i]]}.`;
};
compare.onclick = () => {
  if (running || taskActive() || workerBusy()) return setStatus("Finish the current operation before preparing a comparison", true);
  if (!entries.has(Number(source.value)) || !entries.has(Number(reference.value)) || source.value === reference.value) return;
  const select = (id: string, value: string) => { const el = $<HTMLSelectElement>(id); el.value = value; el.dispatchEvent(new Event("change")); };
  select("c2c-compared", source.value); select("c2c-reference", reference.value); select("distance-method", "quality");
  $("c2c-panel").scrollIntoView({ block: "start" });
  setStatus("Map quality comparison prepared. Check alignment, distance threshold and voxel settings, then Compute. C2C/M3C2 are available in Method.");
};
function summary() {
  if (!last) return null;
  const { counts: _counts, categories: _categories, ...grid } = last.grid;
  return { schema: "cloudanalyzer.map_coverage.v1", source: last.cloud.name, loadedPoints: last.cloud.count,
    sourceFilePoints: last.cloud.filePoints, keepEvery: last.cloud.keepEvery,
    sourcePreview: last.entry.origin.displayPreview === true, copcLevels: last.cloud.copcLevels, ...grid,
    scope: "XY projection of loaded points; sparse threshold is user-defined; enclosed empty uses eight-neighbor exterior connectivity, not proof of missing survey data" };
}
$("coverage-download").onclick = () => { const result = summary(); if (result) download(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }), "map-coverage.json"); };
addReportSection("map-coverage", () => last ? { title: `Coverage of ${last.cloud.name}`, metrics: {
  occupied: { label: "Occupied XY cells", value: last.grid.occupied }, sparse: { label: "Sparse XY cells", value: last.grid.sparse },
  enclosed: { label: "Enclosed empty XY cells (potential gaps)", value: last.grid.enclosedEmpty },
  density: { label: "Median occupied-cell density", value: last.grid.density.median, unit: "points/unit²" },
}, notes: [`Loaded points only; cell ${last.grid.cell}, sparse below ${last.grid.minPoints} points/cell. Empty areas can be unsurveyed or occluded.`] } : null);
render();
