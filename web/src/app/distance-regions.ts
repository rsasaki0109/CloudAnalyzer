/** Bound each diagnosis to both cloud snapshots; never infer reference identity from names. */
import * as THREE from "three";
import { cloudToCloud, workerBusy } from "../api";
import { REGION_INPUT_BYTES, distanceRegionsCsv, validateRegionSettings, type DistanceRegions } from "../distance-regions";
import { projectChanged } from "../project-change";
import { refreshColors } from "./colors";
import { $, download, errorText, fmt, setStatus } from "./dom";
import { renderList } from "./entries";
import { addReportSection } from "./report";
import { display, distanceChanged, entries, listChanged, pointsInvalidated, toRender, viewer, type Entry } from "./state";
import { endTask, startTask, taskActive } from "./tasks";

const source = $<HTMLSelectElement>("coverage-source"), reference = $<HTMLSelectElement>("coverage-reference");
const threshold = $<HTMLInputElement>("regions-threshold"), cell = $<HTMLInputElement>("regions-cell"), minimum = $<HTMLInputElement>("regions-min");
const button = $<HTMLButtonElement>("regions-run");
type Snapshot = { entry: Entry; cloud: Entry["cloud"]; name: string };
type Cache = { source: Snapshot; reference: Snapshot; output: NonNullable<Entry["c2c"]> };
let cache: Cache | null = null, last: { pair: Cache; result: DistanceRegions; reused: boolean } | null = null;
let running = false, generation = 0;
function valid(pair: Cache): boolean {
  return [pair.source, pair.reference].every(s => entries.get(s.cloud.id) === s.entry && s.entry.cloud === s.cloud) &&
    source.value === String(pair.source.cloud.id) && reference.value === String(pair.reference.cloud.id);
}
function clear(drop = false): void {
  const hadResult = last !== null; generation++; last = null; if (drop) cache = null;
  $("regions-result").hidden = true; $("regions-list").replaceChildren();
  if (hadResult) distanceChanged.emit();
}
function render(): void {
  if (cache && !valid(cache)) clear(true);
  if (last && last.pair.source.entry.c2c !== last.pair.output) clear();
  const a = entries.get(Number(source.value)), b = entries.get(Number(reference.value));
  button.disabled = running || !a || !b || a === b || a.cloud.kind !== "cloud" || b.cloud.kind !== "cloud";
  threshold.disabled = cell.disabled = minimum.disabled = running;
}
listChanged.add(render); distanceChanged.add(render);
for (const select of [source, reference]) select.addEventListener("change", () => { clear(true); render(); });
pointsInvalidated.add(id => {
  if ([Number(source.value), Number(reference.value), cache?.source.cloud.id, cache?.reference.cloud.id].includes(id)) clear(true);
});
for (const input of [threshold, cell, minimum]) input.oninput = () => clear();
function group(pair: Cache, signal: AbortSignal, settings: number[]): Promise<DistanceRegions> {
  const positions = pair.source.cloud.positions.slice(), distances = pair.output.distances.slice();
  const worker = new Worker(new URL("../distance-regions-worker.ts", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    const cleanup = () => { worker.terminate(); signal.removeEventListener("abort", abort); };
    const abort = () => { cleanup(); reject(new DOMException("Cancelled", "AbortError")); };
    worker.onmessage = event => { cleanup(); event.data.error ? reject(new Error(event.data.error)) : resolve(event.data.result); };
    worker.onerror = () => { cleanup(); reject(new Error("Discrepancy grouping worker failed")); };
    signal.addEventListener("abort", abort, { once: true }); if (signal.aborted) { abort(); return; }
    try { worker.postMessage({ positions, distances, shift: pair.source.cloud.shift, threshold: settings[0], cell: settings[1], minPoints: settings[2] }, [positions.buffer, distances.buffer]); }
    catch (error) { cleanup(); reject(error); }
  });
}
function show(result: DistanceRegions, pair: Cache, reused: boolean): void {
  $("regions-summary").textContent = `${pair.source.name} → ${pair.reference.name}: ${result.exceeding.toLocaleString()} of ${result.measured.toLocaleString()} measured points exceed ${fmt(result.threshold)}. ${result.regions.length} of ${result.eligibleRegions.toLocaleString()} eligible regions shown; ${result.smallRegionPoints.toLocaleString()} points in below-minimum regions, ${result.unlistedEligiblePoints.toLocaleString()} in unlisted eligible regions. ${result.unmeasured.toLocaleString()} unmeasured; ${result.invalidPositions.toLocaleString()} invalid positions. ${reused ? "Reused the same distance snapshot." : "Computed unsigned CPU C2C distances."}`;
  for (const region of result.regions) {
    const row = document.createElement("li"), focus = document.createElement("button");
    focus.textContent = `Region ${region.rank}: ${region.points.toLocaleString()} points · max ${fmt(region.maxDistance)} · mean ${fmt(region.meanDistance)}`;
    focus.onclick = () => {
      if (!last || last.result !== result || !valid(pair) || running || taskActive()) return;
      viewer.frameBox(new THREE.Box3(toRender(region.bounds.slice(0, 3)), toRender(region.bounds.slice(3, 6))).expandByScalar(result.cell * .5));
      projectChanged();
    };
    row.append(focus); $("regions-list").append(row);
  }
  $("regions-result").hidden = false;
}
button.onclick = async () => {
  if (running || taskActive() || workerBusy()) return setStatus("Finish the current operation before locating discrepancies", true);
  const a = entries.get(Number(source.value)), b = entries.get(Number(reference.value));
  if (!a || !b || a === b || a.cloud.kind !== "cloud" || b.cloud.kind !== "cloud") return;
  const settings = [Number(threshold.value), Number(cell.value), Number(minimum.value)];
  try {
    validateRegionSettings(...settings as [number, number, number]);
    if (a.cloud.positions.byteLength / 3 * 4 > REGION_INPUT_BYTES || b.cloud.positions.byteLength > REGION_INPUT_BYTES) {
      throw new Error("Discrepancy input exceeds 64 MiB; crop a smaller source/reference area");
    }
  } catch (error) { return setStatus(errorText(error), true); }
  clear(); const token = generation, signal = startTask(); running = true; render();
  let phase = "distance";
  const cancelled = () => { if (phase === "distance") setStatus("Cancel requested; the current C2C computation finishes before diagnostics stop."); };
  signal.addEventListener("abort", cancelled);
  try {
    let pair = cache && valid(cache) ? cache : null; const reused = pair !== null;
    if (!pair) {
      const sourceSnapshot: Snapshot = { entry: a, cloud: a.cloud, name: a.cloud.name };
      const referenceSnapshot: Snapshot = { entry: b, cloud: b.cloud, name: b.cloud.name };
      setStatus("Computing unsigned source → reference distances…");
      const output = await cloudToCloud(a.cloud.id, b.cloud.id, false, "cpu"); signal.throwIfAborted();
      pair = { source: sourceSnapshot, reference: referenceSnapshot, output: { ...output, referenceName: referenceSnapshot.name } };
    }
    if (!valid(pair) || token !== generation) throw new Error("Comparison inputs changed; run the diagnosis again");
    phase = "group"; setStatus("Grouping points above the distance threshold…");
    const result = await group(pair, signal, settings); signal.throwIfAborted();
    if (!valid(pair) || token !== generation) throw new Error("Comparison inputs/settings changed; run the diagnosis again");
    cache = pair; last = { pair, result, reused };
    a.c2c = pair.output; a.mode = "c2c"; display.activeC2c = a.cloud.id; display.range = { lo: 0, hi: settings[0] || 1 };
    refreshColors(a); show(result, pair, reused); renderList(); distanceChanged.emit();
    setStatus("Distance regions ready. Inspect candidates for alignment, scene changes or missing reference coverage; these are not confirmed registration defects.");
  } catch (error) { setStatus("Could not locate distance regions: " + (signal.aborted ? "Cancelled" : errorText(error)), true); }
  finally { signal.removeEventListener("abort", cancelled); running = false; endTask(signal); render(); }
};
const describe = (s: Snapshot) => ({ id: s.cloud.id, name: s.name, loadedPoints: s.cloud.count, filePoints: s.cloud.filePoints,
  keepEvery: s.cloud.keepEvery, preview: s.entry.origin.displayPreview === true, copcLevels: s.cloud.copcLevels });
$("regions-json").onclick = () => {
  if (!last || !valid(last.pair)) return;
  download(new Blob([JSON.stringify({ schema: "cloudanalyzer.distance_regions.v1", source: describe(last.pair.source), reference: describe(last.pair.reference),
    method: "unsigned CPU nearest-neighbor source-to-reference", scope: "Loaded points; XY eight-neighbor connected cells, ranked by maximum distance then point count; top 50 eligible regions only; not a registration-defect classifier",
    reusedDistances: last.reused, ...last.result }, null, 2)], { type: "application/json" }), "distance-regions.json");
};
$("regions-csv").onclick = () => { if (last && valid(last.pair)) download(new Blob([distanceRegionsCsv(last.result)], { type: "text/csv" }), "distance-regions.csv"); };
addReportSection("distance-regions", () => last && valid(last.pair) ? { title: `Distance regions: ${last.pair.source.name} → ${last.pair.reference.name}`,
  metrics: { exceeding: { label: "Source points above threshold", value: last.result.exceeding }, eligible: { label: "Eligible distance regions", value: last.result.eligibleRegions },
    listed: { label: "Listed distance regions", value: last.result.regions.length }, unmeasured: { label: "Unmeasured source distances", value: last.result.unmeasured } },
  notes: [`Loaded source → reference points only; threshold ${last.result.threshold}, grouping cell ${last.result.cell}, minimum ${last.result.minPoints} exceeding points.`,
    "Possible scene/alignment/coverage differences, not confirmed defects; reverse comparison requires swapping source and reference."] } : null);
render();
