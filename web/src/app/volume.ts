/** 2.5D cut / fill volume panel. */
import * as THREE from "three";

import { prepareVolumeSupport, computeVolume, removeCloud, workerBusy } from "../api";
import type { VolumeOutput, HeightReview } from "../protocol";
import { screenedCellBounds, screenedCellsCsv } from "../height-review-cells";
import { validateVolumeGrid, sharedVolumeGrid, type VolumeGrid, volumeCellsCsv, heightReviewScope, readHeightThreshold, heightReviewCsv } from "../surface-volume";
import { validateExpectedOverlap, type ExpectedOverlap } from "../distance-regions";
import { captureNamedAreas, replaceNamedAreas, onNamedAreasChanged, parseNamedAreas, parseAreaFile,
  areaVolumesCsv, type NamedArea, type AreaVolumeRow } from "../named-areas";
import { MAX_VOLUME_VISITS, reconcileVisitOrder, intervalVolumeScope, intervalVolumesCsv, type IntervalVolumeRow, visitVolumeScope, commonVolumeScope, visitVolumesCsv, type VisitVolumeRow, type VisitSupport, type VisitSupportMode } from "../volume-visits";
import { volumeVisitReport, type VolumeVisitSnapshot } from "../volume-visit-report";
import { projectChanged } from "../project-change";
import { finiteStats, showSigned } from "./distance";
import { $, download, errorText, fillTable, fmt, roundUp, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { addReportSection } from "./report";
import { distanceChanged, entries, isMesh, listChanged, pointsInvalidated, toRender, viewer, type Entry } from "./state";
import { endTask, startTask, showProgress, taskActive } from "./tasks";

const volumeBefore = $<HTMLSelectElement>("volume-before");
const volumeAfter = $<HTMLSelectElement>("volume-after");
const volumeCell = $<HTMLInputElement>("volume-cell");
const volumeRun = $<HTMLButtonElement>("volume-run");
const coordinateUnit = $<HTMLSelectElement>("volume-unit");
const CONSTANT = "constant";
const visitSelect = $<HTMLSelectElement>("volume-visits");
const visitSupport = $<HTMLSelectElement>("volume-visits-support");
const intervalEnabled = $<HTMLInputElement>("volume-visits-intervals");
let visitOrder: number[] = [];
function syncVisitOrder(): void {
  visitOrder = reconcileVisitOrder(visitOrder, [...visitSelect.selectedOptions].map(option => Number(option.value)));
}
function renderVisitOrder(): void {
  const list = $("volume-visit-order"); list.replaceChildren();
  if (visitOrder.length > MAX_VOLUME_VISITS) { list.textContent = "Choose at most 8 visits before ordering them."; return; }
  visitOrder.forEach((id, index) => {
    const li = document.createElement("li"), name = document.createElement("span");
    name.textContent = `${index + 1}. ${entries.get(id)?.cloud.name ?? "Unavailable"}`;
    li.append(name);
    for (const [offset, label] of [[-1, "Earlier"], [1, "Later"]] as const) {
      const button = document.createElement("button"); button.textContent = label;
      button.setAttribute("aria-label", `Move visit ${index + 1} ${label.toLowerCase()}`);
      button.disabled = running || index + offset < 0 || index + offset >= visitOrder.length;
      button.onclick = () => {
        if (running || taskActive()) return;
        const target = index + offset;
        [visitOrder[index], visitOrder[target]] = [visitOrder[target], visitOrder[index]];
        clearResult(); updateForm();
        list.children[target]?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus();
      };
      li.append(button);
    }
    list.append(li);
  });
}
type Input = { side: { id: number } | { z: number }; name: string; entry?: Entry; cloud?: Entry["cloud"] };
type Settings = { cell: number; height: "mean" | "min" | "max"; fillEmpty: boolean; coordinateUnit: string; region?: ExpectedOverlap; areaName?: string; heightThreshold?: number };
const areaEnabled = $<HTMLInputElement>("volume-area-enabled");
const areaNames = ["min-x", "min-y", "max-x", "max-y"];
const areaInputs = areaNames.map(id => $<HTMLInputElement>(`volume-area-${id}`));
const areaName = $<HTMLInputElement>("volume-area-name"), savedAreas = $<HTMLSelectElement>("volume-areas");
function readRegion(): ExpectedOverlap | undefined {
  if (!areaEnabled.checked) return undefined;
  if (areaInputs.some(input => !input.value.trim())) throw new Error("Enter all four volume review bounds in world coordinates");
  return validateExpectedOverlap(areaInputs.map(input => Number(input.value)) as unknown as ExpectedOverlap)!;
}
function scope(settings: Settings): string {
  return settings.region
    ? `${settings.areaName ? settings.areaName + ": " : ""}World XY rectangle [${settings.region.join(", ")}], cell centers and edges included; whole cell areas, no boundary clipping. Grid and interpolation use full loaded surfaces.`
    : "Full loaded surfaces on the union XY grid; distance-review bounds are independent unless copied into the volume review area.";
}
let last: { out: VolumeOutput; before: string; after: string; inputs: Input[]; settings: Settings; output: Entry | null } | null = null;
let batch: { rows: AreaVolumeRow[]; inputs: Input[]; settings: Settings } | null = null;
type CompletedVisits = { rows: VisitVolumeRow[]; inputs: Input[]; settings: Settings; grid: VolumeGrid; support: VisitSupport; intervals: IntervalVolumeRow[] | null };
let visits: CompletedVisits | null = null;
let running = false, generation = 0, activeInputs: Input[] = [];
let cellInspection: { review: HeightReview; settings: Settings; label: string; token: number } | null = null;
function clearCellInspection(): void {
  cellInspection = null; $("height-cell-inspection").hidden = true; $("height-cell-list").replaceChildren();
}
$("height-cell-close").onclick = clearCellInspection;
function inspectCells(review: HeightReview, settings: Settings, label: string): void {
  if (running || taskActive()) return;
  const token = generation;
  const selection = { review, settings, label, token }; cellInspection = selection;
  $("height-cell-context").textContent = `${label}; strict ±${review.threshold} ${settings.coordinateUnit}; cell ${settings.cell}, ${settings.height}, interpolation ${settings.fillEmpty ? "enabled" : "disabled"}. ${review.cells?.length ?? 0} listed of ${review.riseCells + review.fallCells} exceedance cells; ${review.unlistedCells ?? "unknown"} unlisted. Largest absolute original height differences first, ties by row-major grid index; maximum ${review.cellLimit ?? 50}. Grid centers, not source-point matches. Full aggregates include unlisted cells. Camera focus does not isolate or identify physical changes.`;
  $("height-cell-list").replaceChildren(...(review.cells ?? []).map((c, index) => {
    const li = document.createElement("li"), text = document.createElement("p"), button = document.createElement("button");
    li.style.display = "block"; text.style.margin = "4px 0"; text.style.overflowWrap = "anywhere";
    text.textContent = `${index + 1}. ${c.difference > 0 ? "Rise" : "Fall"}; grid index ${c.index}; world XY [${c.x}, ${c.y}]; Before ${c.before}; After ${c.after}; difference ${c.difference} ${settings.coordinateUnit}.`;
    button.textContent = "Focus cell"; button.setAttribute("aria-label", `Focus screened cell ${index + 1}`);
    button.onclick = () => {
      if (cellInspection !== selection || token !== generation || running || taskActive()) return;
      try {
        const b = screenedCellBounds(c, settings.cell);
        viewer.frameBox(new THREE.Box3(toRender(b.slice(0, 3)), toRender(b.slice(3, 6)))); projectChanged();
        setStatus(`Focused screened grid cell ${c.index}: ${label}. Full square footprint and Before/After height span; other points remain visible.`);
      } catch (error) { setStatus(errorText(error), true); }
    };
    li.append(text, button); return li;
  }));
  $("height-cell-inspection").hidden = false;
  $("height-cell-inspection").scrollIntoView({ block: "nearest" });
}
$("height-cell-csv").onclick = () => {
  if (!cellInspection || cellInspection.token !== generation || running || taskActive()) return;
  const { review, settings, label } = cellInspection;
  download(new Blob([screenedCellsCsv(review, settings.cell, settings.coordinateUnit, label)], { type: "text/csv" }), "listed-height-screening-cells.csv");
};
function validInput(input: Input): boolean {
  return !input.entry || entries.get(input.cloud!.id) === input.entry && input.entry.cloud === input.cloud;
}
function validResult(result: NonNullable<typeof last>): boolean {
  return result.inputs.every(validInput) && (!result.output || entries.get(result.output.cloud.id) === result.output && result.output.cloud === result.out.cells);
}
function clearResult(): void {
  clearCellInspection();
  for (const id of ["volume-areas-screening", "volume-visits-screening", "volume-intervals-screening"]) {
    $(id).replaceChildren(); $(id).hidden = true;
  }
  const had = last !== null || batch !== null || visits !== null; generation++; last = null; batch = null; visits = null;
  $("volume-result").hidden = $("volume-exports").hidden = true;
  $("height-review-result").hidden = true;
  $("volume-areas-result").hidden = $("volume-visits-result").hidden = $("volume-intervals-result").hidden = true;
  if (had) distanceChanged.emit();
}
pointsInvalidated.add(id => {
  if ([...activeInputs, ...(last?.inputs ?? []), ...(batch?.inputs ?? []), ...(visits?.inputs ?? [])].some(input => input.cloud?.id === id) || last?.out.cells?.id === id) clearResult();
});

addReportSection("volume", () =>
  last && validResult(last)
    ? {
        title: `${last.settings.areaName ? last.settings.areaName + ": " : ""}Volume ${last.after} vs ${last.before}`,
        metrics: {
          fill: { label: "Fill (added)", value: last.out.added, unit: `${last.settings.coordinateUnit}³` },
          cut: { label: "Cut (removed)", value: last.out.removed, unit: `${last.settings.coordinateUnit}³` },
          net: { label: "Net", value: last.out.added - last.out.removed, unit: `${last.settings.coordinateUnit}³` },
          fill_area: { label: "Fill area", value: last.out.addedArea, unit: `${last.settings.coordinateUnit}²` },
          cut_area: { label: "Cut area", value: last.out.removedArea, unit: `${last.settings.coordinateUnit}²` },
          uncompared: { label: "Grid cells without both surface heights", value: last.out.totalCells - last.out.matchedCells },
          outside_region: { label: "Cells outside review area", value: last.out.outsideRegionCells },
          ...(last.out.totalCells ? { coverage: {
            label: "Cells compared",
            value: (100 * last.out.matchedCells) / last.out.totalCells,
            unit: "%",
          } } : {}),
        },
        notes: [`Cell size ${last.out.cell}; height ${last.settings.height}; interpolation ${last.settings.fillEmpty ? "enabled" : "disabled"}; coordinate unit ${last.settings.coordinateUnit} (no conversion).`,
          `${scope(last.settings)} Only cells with both heights contribute; undefined cells are not zero change. ${last.out.outsideRegionCells} grid cells outside the volume review area. ${last.out.totalCells ? "" : "No grid-cell centers in this review area; coverage is unavailable."} ${last.out.matchedCells ? "" : "No compared cells: zero totals do not establish zero surface change."}`,
          "2.5D surface difference, not closed-solid volume. Use aligned, comparable ground surfaces for earthwork."],
      }
    : null,
);

function renderSelects(): void {
  const ids = [...entries.keys()].map(String);
  const option = (id: string) => {
    const entry = entries.get(Number(id))!;
    return new Option(isMesh(entry) ? `${entry.cloud.name} (mesh)` : entry.cloud.name, id);
  };
  let [before, after] = [volumeBefore.value, volumeAfter.value];
  const valid = (v: string) => v === CONSTANT || ids.includes(v);
  // Keep a pair the user picked; otherwise follow the defaults as files arrive.
  const picked = volumeBefore.dataset.picked === "1";
  if (!picked || !valid(before) || !valid(after) || before === after) {
    // Default: the first item as "before", the newest as "after"; a single
    // cloud is compared with a constant height.
    before = ids.length > 1 ? ids[0] : CONSTANT;
    after = ids.at(-1) ?? "";
  }
  for (const [select, value] of [
    [volumeBefore, before],
    [volumeAfter, after],
  ] as const) {
    select.replaceChildren(...ids.map(option), new Option("Constant height…", CONSTANT));
    select.value = value;
  }
  const selected = new Set([...visitSelect.selectedOptions].map(option => option.value));
  visitSelect.replaceChildren(...ids.map(option));
  for (const option of visitSelect.options) option.selected = selected.has(option.value);
  syncVisitOrder(); updateForm();
}
listChanged.add(() => { if (last && !validResult(last) || batch && !batch.inputs.every(validInput) || visits && !visits.inputs.every(validInput)) clearResult(); renderSelects(); });

function updateForm(): void {
  $("volume-before-z-row").hidden = volumeBefore.value !== CONSTANT;
  $("volume-after-z-row").hidden = volumeAfter.value !== CONSTANT;
  const both = volumeBefore.value === CONSTANT && volumeAfter.value === CONSTANT;
  volumeRun.disabled = running || entries.size === 0 || both || volumeBefore.value === volumeAfter.value;
  for (const input of [volumeBefore, volumeAfter, volumeCell, coordinateUnit,
    $("volume-height"), $("volume-fill"), $("volume-before-z"), $("volume-after-z"), areaEnabled,
    ...areaInputs, areaName, savedAreas, $("height-review-enabled"), $("height-review-batch-enabled"), $("height-review-threshold"), $("volume-area-save"), $("volume-area-apply"), $("volume-area-remove"),
    $("volume-areas-import"), $("volume-areas-export"), $("volume-use-distance-area"), $("volume-send-distance-area")] as (HTMLInputElement | HTMLSelectElement | HTMLButtonElement)[]) input.disabled = running;
  $<HTMLButtonElement>("volume-areas-run").disabled = volumeRun.disabled || !captureNamedAreas().length;
  $<HTMLInputElement>("height-review-threshold").disabled = running || !($<HTMLInputElement>("height-review-enabled").checked || $<HTMLInputElement>("height-review-batch-enabled").checked);
  visitSelect.disabled = visitSupport.disabled = running;
  if (visitSupport.value !== "common_observed" || visitOrder.length < 2) intervalEnabled.checked = false;
  intervalEnabled.disabled = running || visitSupport.value !== "common_observed" || visitOrder.length < 2;
  renderVisitOrder();
  $<HTMLButtonElement>("volume-visits-mask").disabled = running || !visits?.support.mask;
  $<HTMLButtonElement>("volume-visits-run").disabled = running || !captureNamedAreas().length || !visitSelect.selectedOptions.length;
  for (const id of ["volume-area-apply", "volume-area-remove"]) $<HTMLButtonElement>(id).disabled = running || !savedAreas.value;
  $("volume-area-bounds").hidden = !areaEnabled.checked;
  for (const input of areaInputs) input.disabled = running || !areaEnabled.checked;
  // Default cell: about 1/200 of the larger horizontal extent.
  const source = [volumeAfter.value, volumeBefore.value]
    .map((v) => entries.get(Number(v)))
    .find((e) => e !== undefined);
  if (source && volumeCell.dataset.for !== String(source.cloud.id)) {
    const b = source.cloud.bounds;
    volumeCell.value = String(roundUp(Math.max(b[3] - b[0], b[4] - b[1]) / 200 || 1));
    volumeCell.dataset.for = String(source.cloud.id);
  }
}
volumeBefore.onchange = volumeAfter.onchange = () => {
  clearResult();
  volumeBefore.dataset.picked = "1";
  updateForm();
};

for (const id of ["volume-cell", "volume-before-z", "volume-after-z", "volume-height", "volume-fill", "volume-unit"]) {
  $(id).addEventListener(id === "volume-height" || id === "volume-fill" || id === "volume-unit" ? "change" : "input", clearResult);
}
areaEnabled.onchange = () => { clearResult(); updateForm(); };
for (const input of areaInputs) input.oninput = clearResult;
areaName.oninput = clearResult;
$("height-review-enabled").onchange = $("height-review-batch-enabled").onchange = () => { clearResult(); updateForm(); };
$("height-review-threshold").oninput = clearResult;
$("volume-use-distance-area").onclick = () => {
  if (running || taskActive()) return;
  try {
    if (!$<HTMLInputElement>("regions-overlap-enabled").checked) throw new Error("Enable and define a distance review rectangle first");
    const values = areaNames.map(id => $<HTMLInputElement>(`regions-overlap-${id}`).value);
    if (values.some(value => !value.trim())) throw new Error("Enter all four distance review bounds");
    validateExpectedOverlap(values.map(Number) as unknown as ExpectedOverlap);
    areaInputs.forEach((input, i) => { input.value = values[i]; });
    areaEnabled.checked = true; clearResult(); updateForm();
    setStatus("Distance review bounds copied to volume. Review cell size and compute volume; cells are selected by center with full square areas.");
  } catch (error) { setStatus(errorText(error), true); }
};
$("volume-send-distance-area").onclick = () => {
  if (running || taskActive()) return;
  try {
    const region = readRegion();
    if (!region) throw new Error("Enable and define a volume review rectangle first");
    const enabled = $<HTMLInputElement>("regions-overlap-enabled"); enabled.checked = true;
    enabled.dispatchEvent(new Event("change"));
    areaNames.forEach((id, i) => {
      const input = $<HTMLInputElement>(`regions-overlap-${id}`); input.value = String(region[i]); input.dispatchEvent(new Event("input"));
    });
    setStatus("Volume review bounds copied to distance review. Recompute distance regions to apply them; reference searches still use the full loaded reference.");
  } catch (error) { setStatus(errorText(error), true); }
};
function snapshot(select: HTMLSelectElement, z: HTMLInputElement): Input {
  if (select.value === CONSTANT) {
    if (!z.value.trim() || !Number.isFinite(Number(z.value))) throw new Error("Enter a finite constant surface height");
    return { side: { z: Number(z.value) }, name: `Constant height ${z.value}` };
  }
  const entry = entries.get(Number(select.value));
  if (!entry) throw new Error("Choose two available surfaces");
  return { side: { id: entry.cloud.id }, name: entry.cloud.name, entry, cloud: entry.cloud };
}

visitSupport.onchange = () => { if (visitSupport.value !== "common_observed") intervalEnabled.checked = false; clearResult(); updateForm(); };
intervalEnabled.onchange = () => { clearResult(); updateForm(); };
visitSelect.onchange = () => { syncVisitOrder(); if (visitOrder.length < 2) intervalEnabled.checked = false; clearResult(); updateForm(); };

function unit(value: number, power: 2 | 3): string {
  return `${fmt(value)} ${last?.settings.coordinateUnit ?? coordinateUnit.value}${power === 3 ? "³" : "²"}`;
}

volumeRun.onclick = async () => {
  if (running || taskActive() || workerBusy()) return setStatus("Finish the current operation before computing volume", true);
  const settings: Settings = { cell: Number(volumeCell.value), height: $<HTMLSelectElement>("volume-height").value as Settings["height"],
    fillEmpty: $<HTMLInputElement>("volume-fill").checked, coordinateUnit: coordinateUnit.value };
  let inputs: Input[];
  try {
    const heightThreshold = readHeightThreshold($<HTMLInputElement>("height-review-enabled").checked, $<HTMLInputElement>("height-review-threshold").value);
    if (heightThreshold !== undefined) settings.heightThreshold = heightThreshold;
    const region = readRegion();
    if (region) settings.region = region;
    if (region && areaName.value.trim()) settings.areaName = parseNamedAreas([{ name: areaName.value, bounds: region }])[0].name;
    inputs = [snapshot(volumeBefore, $<HTMLInputElement>("volume-before-z")), snapshot(volumeAfter, $<HTMLInputElement>("volume-after-z"))];
    if (volumeBefore.value === volumeAfter.value) throw new Error("Choose different before and after surfaces");
    validateVolumeGrid(inputs.filter(input => input.cloud).map(input => input.cloud!.bounds), settings.cell);
    if (!["mean", "min", "max"].includes(settings.height) || !["m", "ft", "unit"].includes(settings.coordinateUnit)) throw new Error("Choose valid height and coordinate-unit settings");
  } catch (error) { return setStatus(errorText(error), true); }
  clearResult(); const token = generation, signal = startTask(); running = true; activeInputs = inputs;
  volumeBefore.dataset.picked = "1"; updateForm();
  const cancelled = () => setStatus("Cancel requested; the current volume computation finishes before its result is discarded.");
  signal.addEventListener("abort", cancelled);
  let out: VolumeOutput | null = null, adopted = false;
  setStatus("Computing volume…");
  try {
    out = await computeVolume({
      before: inputs[0].side, after: inputs[1].side,
      cell: settings.cell, height: settings.height, fillEmpty: settings.fillEmpty,
      region: settings.region,
      heightThreshold: settings.heightThreshold,
    });
    signal.throwIfAborted();
    if (token !== generation || !inputs.every(validInput)) throw new Error("Volume inputs/settings changed; compute it again");
    if (![out.added, out.removed, out.addedArea, out.removedArea].every(Number.isFinite)) throw new Error("Volume cannot be represented at this coordinate scale");
    let output: Entry | null = null;
    if (out.cells && out.difference) {
      output = addEntry(out.cells); adopted = true;
      record({ label: "the volume", added: [output] });
      showSigned(output, { kind: "volume", signed: true, distances: out.difference, stats: finiteStats(out.difference),
        millis: out.millis, workers: 1, referenceName: inputs[0].name });
    }
    const coverage = out.totalCells ? (100 * out.matchedCells) / out.totalCells : 0;
    last = {
      out,
      before: inputs[0].name, after: inputs[1].name, inputs, settings, output,
    };
    $("volume-result").hidden = false;
    fillTable($("volume-stats"), [
      ["Fill (added)", unit(out.added, 3)],
      ["Cut (removed)", unit(out.removed, 3)],
      ["Net", unit(out.added - out.removed, 3)],
      ["Fill area", unit(out.addedArea, 2)],
      ["Cut area", unit(out.removedArea, 2)],
      ["Cells compared", `${out.matchedCells.toLocaleString()} of ${out.totalCells.toLocaleString()} (${out.totalCells ? coverage.toFixed(1) + " %" : "unavailable"})`],
      ["Uncompared cells", (out.totalCells - out.matchedCells).toLocaleString()],
      ["Cells outside review area", out.outsideRegionCells.toLocaleString()],
      ["Cell size", fmt(out.cell)],
    ]);
    $("volume-exports").hidden = false;
    $("height-review-result").hidden = !out.heightReview;
    if (out.heightReview) {
      $<HTMLButtonElement>("height-review-inspect").disabled = !out.heightReview.cells?.length;
      const h = out.heightReview, available = out.matchedCells > 0;
      fillTable($("height-review-stats"), [
        ["Strict height threshold", `${fmt(h.threshold)} ${settings.coordinateUnit}`],
        ["Rise / fall / within-band cells", `${h.riseCells} / ${h.fallCells} / ${h.withinCells}`],
        ["Rise area", available ? unit(h.riseArea, 2) : "Unavailable"],
        ["Fall area", available ? unit(h.fallArea, 2) : "Unavailable"],
        ["Rise volume (full differences)", available ? unit(h.riseVolume, 3) : "Unavailable"],
        ["Fall volume (full differences)", available ? unit(h.fallVolume, 3) : "Unavailable"],
        ["Screened net", available ? unit(h.riseVolume - h.fallVolume, 3) : "Unavailable"],
      ]);
      $("height-review-scope").textContent = heightReviewScope;
    }
    $<HTMLButtonElement>("volume-isolate").disabled = !output;
    $<HTMLButtonElement>("volume-mesh").disabled = !output;
    $("volume-scope").textContent = `${inputs[0].name} → ${inputs[1].name}; ${settings.height} height per cell; interpolation ${settings.fillEmpty ? "enabled" : "disabled"}. ${scope(settings)} Blue = cut, red = fill; undefined cells are not zero change. ${out.matchedCells ? "" : "No compared cells: zero totals do not establish zero surface change."}`;
    distanceChanged.emit();
    setStatus(
      `Volume: fill ${unit(out.added, 3)}, cut ${unit(out.removed, 3)}, net ${unit(out.added - out.removed, 3)} ` +
        `(${Math.round(out.millis)} ms)`,
    );
  } catch (err) {
    if (out?.cells && !adopted) await removeCloud(out.cells.id);
    setStatus(`Volume failed: ${signal.aborted ? "Cancelled" : errorText(err)}`, true);
  } finally {
    signal.removeEventListener("abort", cancelled); activeInputs = []; running = false; endTask(signal);
    updateForm();
  }
};

$("volume-isolate").onclick = () => {
  if (!last?.output || !validResult(last) || running || taskActive()) return;
  for (const input of last.inputs) if (input.entry) {
    input.entry.visible = false; viewer.setVisible(input.entry.cloud.id, false);
  }
  last.output.visible = true; viewer.setVisible(last.output.cloud.id, true);
  renderList(); projectChanged();
  setStatus("Height-difference cells shown; input surfaces hidden. Use their cloud-list checkboxes to show them again.");
};

$("volume-mesh").onclick = () => {
  if (!last?.output || !validResult(last) || running || taskActive() || workerBusy()) return;
  const select = $<HTMLSelectElement>("mesh-cloud"); select.value = String(last.output.cloud.id); select.dispatchEvent(new Event("change"));
  $("mesh-panel").scrollIntoView({ block: "start" });
  setStatus("Meshing prepared from compared-cell centers at After height. Review max edge length, then Mesh. This is an After surface; signed change colors are not transferred to mesh vertices.");
};

$("volume-json").onclick = () => {
  if (!last || !validResult(last)) return;
  const describe = (input: Input) => input.cloud ? { id: input.cloud.id, name: input.name, kind: input.cloud.kind,
    loadedPoints: input.cloud.count, filePoints: input.cloud.filePoints, keepEvery: input.cloud.keepEvery,
    preview: input.entry!.origin.displayPreview === true, copcLevels: input.cloud.copcLevels } : { constantHeight: "z" in input.side ? input.side.z : null };
  const out = last.out;
  download(new Blob([JSON.stringify({ schema: "cloudanalyzer.surface_volume.v1", before: describe(last.inputs[0]), after: describe(last.inputs[1]),
    settings: last.settings, comparedCellCloud: last.output ? { id: last.output.cloud.id, name: last.output.cloud.name, points: last.output.cloud.count, geometry: "Compared cell centers at After height" } : null,
    method: "2.5D grid height difference: after minus before; no registration or coordinate-unit conversion",
    scope: `${scope(last.settings)} Only cells with both heights contribute. Empty cells are undefined unless interpolation is explicitly enabled. Not closed-solid volume.`,
    reviewArea: last.settings.region ? { name: last.settings.areaName ?? null, bounds: last.settings.region, selection: "Cell centers inside closed world XY rectangle; full square cell area, no fractional clipping", outsideCells: out.outsideRegionCells } : null,
    result: { added: out.added, removed: out.removed, net: out.added - out.removed, addedArea: out.addedArea, removedArea: out.removedArea,
      matchedCells: out.matchedCells, totalCells: out.totalCells, uncomparedCells: out.totalCells - out.matchedCells, cell: out.cell,
      ...(out.heightReview ? { heightReview: { ...out.heightReview, comparisonAvailable: out.matchedCells > 0, scope: heightReviewScope } } : {}) },
    cellExport: "Compared cell centers at after height; CSV XYZ and signed height differences come from the displayed f32 records. Totals use the existing f64 engine and may differ from a sum of rounded CSV values." }, null, 2)], { type: "application/json" }), "surface-volume.json");
};
$("volume-csv").onclick = () => {
  if (!last || !validResult(last)) return;
  try {
    const out = last.out;
    const csv = volumeCellsCsv(out.cells?.positions ?? new Float32Array(), out.cells?.shift ?? [0, 0, 0], out.difference ?? new Float32Array());
    download(new Blob([csv], { type: "text/csv" }), "surface-volume-cells.csv");
  } catch (error) { setStatus(errorText(error), true); }
};
$("height-review-csv").onclick = () => {
  if (last?.out.heightReview && validResult(last)) download(new Blob([heightReviewCsv(last.out.heightReview, last.out.matchedCells, last.settings.coordinateUnit)], { type: "text/csv" }), "height-change-screening.csv");
};
$("height-review-inspect").onclick = () => {
  if (last?.out.heightReview && validResult(last)) inspectCells(last.out.heightReview, last.settings, `${last.before} → ${last.after}${last.settings.areaName ? " / " + last.settings.areaName : ""}`);
};
addReportSection("height-change-screening", () => {
  if (!last?.out.heightReview || !validResult(last)) return null;
  const h = last.out.heightReview, label = last.settings.coordinateUnit;
  return { title: `Height screening: ${last.before} → ${last.after}`, metrics: {
    threshold: { label: "User height threshold (strict)", value: h.threshold, unit: label },
    rise_cells: { label: "Rise cells", value: h.riseCells }, fall_cells: { label: "Fall cells", value: h.fallCells },
    within_cells: { label: "Within-band cells (not proof of zero change)", value: h.withinCells },
    ...(last.out.matchedCells ? {
      rise_area: { label: "Screened rise area", value: h.riseArea, unit: `${label}²` },
      fall_area: { label: "Screened fall area", value: h.fallArea, unit: `${label}²` },
      rise_volume: { label: "Screened rise volume", value: h.riseVolume, unit: `${label}³` },
      fall_volume: { label: "Screened fall volume", value: h.fallVolume, unit: `${label}³` },
      screened_net: { label: "Screened net", value: h.riseVolume - h.fallVolume, unit: `${label}³` },
    } : {}),
  }, notes: [heightReviewScope, `${scope(last.settings)} Height ${last.settings.height}; interpolation ${last.settings.fillEmpty}. Compared cells ${last.out.matchedCells}; uncompared ${last.out.totalCells - last.out.matchedCells}; outside ${last.out.outsideRegionCells}.`,
    ...(last.out.matchedCells ? [] : ["No compared cells; screened quantities unavailable."])] };
});

function renderAreas(): void {
  const selected = savedAreas.value;
  savedAreas.replaceChildren(...captureNamedAreas().map(area => new Option(area.name, area.name)));
  if ([...savedAreas.options].some(option => option.value === selected)) savedAreas.value = selected;
  updateForm();
}
onNamedAreasChanged(() => { clearResult(); renderAreas(); });
savedAreas.onchange = updateForm;
$("volume-area-save").onclick = () => {
  if (running || taskActive()) return;
  try {
    const bounds = readRegion(); if (!bounds) throw new Error("Enable a volume review rectangle before saving an area");
    const area = parseNamedAreas([{ name: areaName.value, bounds }])[0];
    const areas = captureNamedAreas(), index = areas.findIndex(saved => saved.name === area.name);
    if (index < 0) areas.push(area); else areas[index] = area;
    replaceNamedAreas(areas); savedAreas.value = area.name;
    setStatus(`Saved area ${area.name}. Apply it for single-area analysis or compute all saved areas.`);
  } catch (error) { setStatus(errorText(error), true); }
};
$("volume-area-apply").onclick = () => {
  if (running || taskActive()) return;
  const area = captureNamedAreas().find(area => area.name === savedAreas.value); if (!area) return;
  areaEnabled.checked = true; areaName.value = area.name;
  areaInputs.forEach((input, i) => { input.value = String(area.bounds[i]); });
  clearResult(); updateForm(); setStatus(`Applied ${area.name}; review inputs and settings, then compute. No analysis was run.`);
};
$("volume-area-remove").onclick = () => {
  if (!running && !taskActive()) replaceNamedAreas(captureNamedAreas().filter(area => area.name !== savedAreas.value));
};
$("volume-areas-export").onclick = () => download(new Blob([JSON.stringify({
  schema: "cloudanalyzer.named_review_areas.v1", coordinates: "Original world XY; frame/units must match inputs; no coordinate conversion",
  areas: captureNamedAreas(),
}, null, 2)], { type: "application/json" }), "review-areas.json");
const areaFile = $<HTMLInputElement>("volume-areas-file");
$("volume-areas-import").onclick = () => { if (!running && !taskActive()) areaFile.click(); };
areaFile.onchange = async () => {
  const file = areaFile.files?.[0]; areaFile.value = ""; if (!file || running || taskActive()) return;
  const token = generation;
  try {
    if (file.size > 65536) throw new Error("Area definitions exceed the 64 KiB limit");
    const areas = parseAreaFile(JSON.parse(await file.text()));
    if (token !== generation || running || taskActive()) throw new Error("Work changed while reading areas; import the file again");
    replaceNamedAreas(areas); setStatus(`Loaded ${areas.length} area definitions. Check coordinate frame and units before applying them.`);
  } catch (error) { setStatus(errorText(error), true); }
};
function describeInput(input: Input) {
  return input.cloud ? { id: input.cloud.id, name: input.name, kind: input.cloud.kind,
    loadedPoints: input.cloud.count, filePoints: input.cloud.filePoints, keepEvery: input.cloud.keepEvery,
    preview: input.entry!.origin.displayPreview === true, copcLevels: input.cloud.copcLevels }
    : { constantHeight: "z" in input.side ? input.side.z : null };
}
const batchScope = "Independent named rectangles on the same loaded union grid. Closed cell-center selection with full square cell areas; no fractional clipping. Full loaded surfaces determine grid, heights and interpolation. Outside and undefined-inside cells are distinct. Overlapping areas may count the same cells; do not sum rows. Counts do not establish coverage beyond the loaded grid. 2.5D height volume, not closed-solid volume or survey accuracy.";
function renderScreening(id: string, data: readonly { label: string; row: AreaVolumeRow }[], settings: Settings): void {
  const root = $(id); root.replaceChildren(); root.hidden = settings.heightThreshold === undefined;
  if (root.hidden) return;
  const title = document.createElement("h3"), note = document.createElement("p");
  title.textContent = `Height screening: strict ±${settings.heightThreshold} ${settings.coordinateUnit}`;
  note.className = "hint small"; note.textContent = "Full differences in screened cells; equality stays within-band. User criterion, not calibrated uncertainty. Totals remain separate. No compared cells means unavailable; overlapping areas and gross interval amounts must not be summed.";
  const wrap = document.createElement("div"); wrap.className = "review-table-wrap"; wrap.tabIndex = 0;
  wrap.setAttribute("aria-label", "Height screening summaries; scroll horizontally for quantities");
  const table = document.createElement("table"); table.className = "stats review-table area-volume-table";
  const head = table.createTHead().insertRow();
  for (const text of ["Comparison / area", "Rise cells", "Fall cells", "Within cells", "Rise area", "Fall area", "Rise volume", "Fall volume", "Screened net", "Cells"]) {
    const th = document.createElement("th"); th.textContent = text; th.scope = "col"; head.append(th);
  }
  const body = table.createTBody();
  for (const { label, row } of data) {
    const h = row.heightReview!;
    const tr = body.insertRow();
    for (const value of [label, h.riseCells, h.fallCells, h.withinCells,
      ...[h.riseArea, h.fallArea, h.riseVolume, h.fallVolume, h.riseVolume - h.fallVolume].map((v, i) => row.matchedCells ? `${fmt(v)} ${settings.coordinateUnit}${i < 2 ? "²" : "³"}` : "Unavailable")]) tr.insertCell().textContent = String(value);
    const token = generation, button = document.createElement("button");
    button.textContent = "Inspect cells"; button.setAttribute("aria-label", `Inspect screened cells: ${label}`); button.disabled = !h.cells?.length;
    button.onclick = () => { if (token === generation) inspectCells(h, settings, label); };
    tr.insertCell().append(button);
  }
  wrap.append(table); root.append(title, note, wrap);
}
$("volume-areas-run").onclick = async () => {
  if (running || taskActive() || workerBusy()) return setStatus("Finish the current operation before computing area volumes", true);
  const areas = captureNamedAreas(); if (!areas.length) return;
  const settings: Settings = { cell: Number(volumeCell.value), height: $<HTMLSelectElement>("volume-height").value as Settings["height"],
    fillEmpty: $<HTMLInputElement>("volume-fill").checked, coordinateUnit: coordinateUnit.value };
  let inputs: Input[];
  try {
    const threshold = readHeightThreshold($<HTMLInputElement>("height-review-batch-enabled").checked, $<HTMLInputElement>("height-review-threshold").value);
    if (threshold !== undefined) settings.heightThreshold = threshold;
    inputs = [snapshot(volumeBefore, $<HTMLInputElement>("volume-before-z")), snapshot(volumeAfter, $<HTMLInputElement>("volume-after-z"))];
    if (volumeBefore.value === volumeAfter.value) throw new Error("Choose different before and after surfaces");
    validateVolumeGrid(inputs.filter(input => input.cloud).map(input => input.cloud!.bounds), settings.cell);
    if (!["mean", "min", "max"].includes(settings.height) || !["m", "ft", "unit"].includes(settings.coordinateUnit)) throw new Error("Choose valid height and coordinate-unit settings");
  } catch (error) { return setStatus(errorText(error), true); }
  clearResult(); const token = generation, signal = startTask(); running = true; activeInputs = inputs;
  volumeBefore.dataset.picked = "1"; updateForm();
  try {
    const rows: AreaVolumeRow[] = [];
    for (const area of areas) {
      signal.throwIfAborted();
      showProgress({ note: `Area ${rows.length + 1}/${areas.length}: ${area.name}`, fraction: rows.length / areas.length });
      const out = await computeVolume({ before: inputs[0].side, after: inputs[1].side, ...settings, region: area.bounds, summaryOnly: true });
      if (out.cells) { await removeCloud(out.cells.id); throw new Error("Area summary unexpectedly created a point cloud"); }
      signal.throwIfAborted();
      if (token !== generation || !inputs.every(validInput)) throw new Error("Area inputs/settings changed; compute the summaries again");
      if (![out.added, out.removed, out.addedArea, out.removedArea].every(Number.isFinite)) throw new Error("Area volume cannot be represented at this coordinate scale");
      const { added, removed, addedArea, removedArea, matchedCells, totalCells, outsideRegionCells } = out;
      if (settings.heightThreshold !== undefined && !out.heightReview) throw new Error("Height screening summary missing");
      rows.push({ ...area, added, removed, addedArea, removedArea, matchedCells, totalCells, outsideRegionCells, ...(out.heightReview ? { heightReview: out.heightReview } : {}) });
    }
    batch = { rows, inputs, settings };
    renderScreening("volume-areas-screening", rows.map(row => ({ label: row.name, row })), settings);
    $("volume-areas-rows").replaceChildren(...rows.map(row => {
      const tr = document.createElement("tr");
      for (const value of [row.name, ...[row.added, row.removed, row.added - row.removed].map(v => row.matchedCells ? `${fmt(v)} ${settings.coordinateUnit}³` : "Unavailable"), row.matchedCells, row.totalCells - row.matchedCells, row.outsideRegionCells]) {
        const td = document.createElement("td"); td.textContent = String(value); tr.append(td);
      }
      return tr;
    }));
    $("volume-areas-scope").textContent = `${inputs[0].name} → ${inputs[1].name}; ${rows.length} areas, cell ${settings.cell}, ${settings.height}, interpolation ${settings.fillEmpty ? "enabled" : "disabled"}. 2.5D cut/fill; whole cells selected by center. Overlapping rows must not be summed. Unavailable means no compared cells. Scroll the table for coverage counts.`;
    $("volume-areas-result").hidden = false; distanceChanged.emit();
    setStatus(`Computed ${rows.length} named area summaries. Independent rows; overlapping areas must not be summed.`);
  } catch (error) {
    setStatus(`Area volumes failed: ${signal.aborted ? "Cancelled; no partial summaries published" : errorText(error)}`, true);
  } finally { activeInputs = []; running = false; endTask(signal); updateForm(); }
};
$("volume-areas-json").onclick = () => {
  if (!batch || !batch.inputs.every(validInput)) return;
  download(new Blob([JSON.stringify({ schema: "cloudanalyzer.named_area_volumes.v1", before: describeInput(batch.inputs[0]), after: describeInput(batch.inputs[1]),
    settings: batch.settings, scope: batchScope, ...(batch.settings.heightThreshold !== undefined ? { screeningScope: heightReviewScope } : {}), areas: batch.rows.map(row => ({ ...row, net: row.added - row.removed,
      uncomparedCells: row.totalCells - row.matchedCells, comparisonAvailable: row.matchedCells > 0 })) }, null, 2)], { type: "application/json" }), "area-volumes.json");
};
$("volume-areas-csv").onclick = () => {
  if (batch && batch.inputs.every(validInput)) download(new Blob([areaVolumesCsv(batch.rows, batch.settings.coordinateUnit)], { type: "text/csv" }), "area-volumes.csv");
};
addReportSection("volume-areas", () => {
  if (!batch || !batch.inputs.every(validInput)) return null;
  const unit = `${batch.settings.coordinateUnit}³`;
  return { title: `Named area volumes: ${batch.inputs[0].name} → ${batch.inputs[1].name}`,
    metrics: Object.fromEntries(batch.rows.flatMap(row => [
      [`area_${encodeURIComponent(row.name)}_compared`, { label: `${row.name}: compared cells`, value: row.matchedCells }],
      [`area_${encodeURIComponent(row.name)}_missing`, { label: `${row.name}: uncompared cells`, value: row.totalCells - row.matchedCells }],
      ...(row.matchedCells ? [
        [`area_${encodeURIComponent(row.name)}_fill`, { label: `${row.name}: fill`, value: row.added, unit }],
        [`area_${encodeURIComponent(row.name)}_cut`, { label: `${row.name}: cut`, value: row.removed, unit }],
        [`area_${encodeURIComponent(row.name)}_net`, { label: `${row.name}: net`, value: row.added - row.removed, unit }],
      ] : []),
    ])), notes: [batchScope, ...batch.rows.map(row => `${row.name}: XY [${row.bounds.join(", ")}]; ${row.outsideRegionCells} outside cells. ${row.matchedCells ? "" : "No compared cells; quantities unavailable."}`)] };
});
renderAreas();

addReportSection("volume-areas-screening", () => {
  if (!batch || batch.settings.heightThreshold === undefined || !batch.inputs.every(validInput)) return null;
  const unit = batch.settings.coordinateUnit;
  return { title: "Named area height screening", metrics: Object.fromEntries(batch.rows.flatMap(row => {
    const h = row.heightReview!, prefix = `area_${encodeURIComponent(row.name)}_screen_`;
    return [
      [prefix + "threshold", { label: `${row.name}: user height threshold`, value: h.threshold, unit }],
      ...(["riseCells", "fallCells", "withinCells"] as const).map(key => [prefix + key, { label: `${row.name}: ${key}`, value: h[key] }]),
      ...(row.matchedCells ? (["riseArea", "fallArea", "riseVolume", "fallVolume"] as const).map(key => [prefix + key, { label: `${row.name}: ${key}`, value: h[key], unit: unit + (key.endsWith("Area") ? "²" : "³") }]) : []),
    ];
  })), notes: [heightReviewScope, ...batch.rows.map(row => `${row.name}: XY [${row.bounds.join(", ")}]; ${row.matchedCells} compared; ${row.totalCells - row.matchedCells} uncompared; ${row.outsideRegionCells} outside. ${row.matchedCells ? "" : "No support: quantities unavailable."}`)] };
});

// Stage summary-only rows serially: bounded worker memory and no partial publication.
$("volume-visits-run").onclick = async () => {
  if (running || taskActive() || workerBusy()) return setStatus("Finish the current operation before comparing visits", true);
  const areas = captureNamedAreas();
  syncVisitOrder();
  const options = new Map([...visitSelect.selectedOptions].map(option => [Number(option.value), option]));
  const selected = visitOrder.map(id => options.get(id)!);
  const includeIntervals = intervalEnabled.checked;
  const support: VisitSupport = { mode: visitSupport.value as VisitSupportMode };
  const settings: Settings = { cell: Number(volumeCell.value), height: $<HTMLSelectElement>("volume-height").value as Settings["height"],
    fillEmpty: $<HTMLInputElement>("volume-fill").checked, coordinateUnit: coordinateUnit.value };
  let inputs: Input[], grid: VolumeGrid;
  try {
    const threshold = readHeightThreshold($<HTMLInputElement>("height-review-batch-enabled").checked, $<HTMLInputElement>("height-review-threshold").value);
    if (threshold !== undefined) settings.heightThreshold = threshold;
    if (!areas.length || !selected.length || selected.length > MAX_VOLUME_VISITS) throw new Error("Choose 1–8 visit surfaces and save at least one area");
    inputs = [snapshot(volumeBefore, $<HTMLInputElement>("volume-before-z")), ...selected.map(option => {
      const entry = entries.get(Number(option.value));
      if (!entry) throw new Error("A selected visit is no longer available");
      return { side: { id: entry.cloud.id }, name: entry.cloud.name, entry, cloud: entry.cloud };
    })];
    if (includeIntervals && (selected.length < 2 || support.mode !== "common_observed")) throw new Error("Direct intervals require at least two visits and common observed cells");
    if (!["independent", "common_observed"].includes(support.mode)) throw new Error("Choose a valid visit support mode");
    if (support.mode === "common_observed" && (settings.fillEmpty || inputs.some(input => !input.cloud || input.cloud.kind !== "cloud"))) throw new Error("Common observed cells require point clouds for the baseline and visits, with interpolation disabled");
    if (selected.some(option => option.value === volumeBefore.value)) throw new Error("The baseline cannot also be a visit");
    grid = sharedVolumeGrid(inputs.filter(input => input.cloud).map(input => input.cloud!.bounds), settings.cell);
    if (!["mean", "min", "max"].includes(settings.height) || !["m", "ft", "unit"].includes(settings.coordinateUnit)) throw new Error("Choose valid height and coordinate-unit settings");
  } catch (error) { return setStatus(errorText(error), true); }
  clearResult(); const token = generation, signal = startTask(); running = true; activeInputs = inputs;
  volumeBefore.dataset.picked = "1"; updateForm();
  const cancelled = () => setStatus("Cancel requested; the current support preparation or comparison finishes before all staged visit rows are discarded.");
  signal.addEventListener("abort", cancelled);
  try {
    const rows: VisitVolumeRow[] = [], intervals: IntervalVolumeRow[] | null = includeIntervals ? [] : null;
    const jobs = (selected.length + (includeIntervals ? selected.length - 1 : 0)) * areas.length;
    let completed = 0;
    if (support.mode === "common_observed") {
      showProgress({ note: "Intersecting raw observed cells across baseline and all visits", fraction: 0 });
      const mask = await prepareVolumeSupport({ ids: inputs.map(input => input.cloud!.id), cell: settings.cell, height: settings.height, sharedGrid: grid });
      signal.throwIfAborted();
      if (token !== generation || !inputs.every(validInput)) throw new Error("Visit inputs/settings changed; compare again");
      if (mask.length !== grid[2] * grid[3] || mask.some(v => v !== 0 && v !== 1)) throw new Error("Invalid observed support mask");
      support.mask = mask; support.commonGridCells = mask.reduce((sum, v) => sum + v, 0);
      if (!Number.isFinite(support.commonGridCells * settings.cell ** 2)) throw new Error("Common observed area cannot be represented at this coordinate scale");
      const digest = await crypto.subtle.digest("SHA-256", mask.slice().buffer);
      signal.throwIfAborted();
      if (token !== generation || !inputs.every(validInput)) throw new Error("Visit inputs/settings changed; compare again");
      support.sha256 = Array.from(new Uint8Array(digest), v => v.toString(16).padStart(2, "0")).join("");
    }
    async function summarize(before: Input, after: Input, area: NamedArea, label: string): Promise<AreaVolumeRow> {
      signal.throwIfAborted();
      showProgress({ note: `${label} / ${area.name}`, fraction: completed / jobs });
      const out = await computeVolume({ before: before.side, after: after.side, ...settings, region: area.bounds, sharedGrid: grid, supportMask: support.mask, summaryOnly: true });
      if (out.cells) { await removeCloud(out.cells.id); throw new Error("Visit summary unexpectedly created a point cloud"); }
      signal.throwIfAborted();
      if (token !== generation || !inputs.every(validInput)) throw new Error("Visit inputs/settings changed; compare again");
      if (![out.added, out.removed, out.addedArea, out.removedArea].every(Number.isFinite)) throw new Error("Visit volume cannot be represented at this coordinate scale");
      const { added, removed, addedArea, removedArea, matchedCells, totalCells, outsideRegionCells } = out;
      completed++;
      if (settings.heightThreshold !== undefined && !out.heightReview) throw new Error("Height screening summary missing");
      return { ...area, added, removed, addedArea, removedArea, matchedCells, totalCells, outsideRegionCells, ...(out.heightReview ? { heightReview: out.heightReview } : {}) };
    }
    for (const [index, input] of inputs.slice(1).entries()) for (const area of areas) {
      rows.push({ ...await summarize(inputs[0], input, area, `Baseline → ${input.name}`), visit: input.name, visitIndex: index + 1 });
    }
    if (intervals) for (let index = 1; index < inputs.length - 1; index++) for (const area of areas) {
      const from = inputs[index], to = inputs[index + 1];
      intervals.push({ ...await summarize(from, to, area, `Interval ${index}: ${from.name} → ${to.name}`), intervalIndex: index,
        fromVisitIndex: index, toVisitIndex: index + 1, fromId: from.cloud!.id, toId: to.cloud!.id, fromVisit: from.name, toVisit: to.name });
    }
    visits = { rows, inputs, settings, grid, support, intervals };
    const renderRows = (id: string, data: readonly { label: string; row: AreaVolumeRow }[]) => $(id).replaceChildren(...data.map(({ label, row }) => {
      const tr = document.createElement("tr");
      for (const value of [label, ...[row.added, row.removed, row.added - row.removed].map(v => row.matchedCells ? `${fmt(v)} ${settings.coordinateUnit}³` : "Unavailable"), row.matchedCells, row.totalCells - row.matchedCells, row.outsideRegionCells]) {
        const td = document.createElement("td"); td.textContent = String(value); tr.append(td);
      }
      return tr;
    }));
    renderRows("volume-visits-rows", rows.map(row => ({ label: `${row.visitIndex}. ${row.visit} / ${row.name}`, row })));
    renderScreening("volume-visits-screening", rows.map(row => ({ label: `${row.visitIndex}. ${row.visit} / ${row.name}`, row })), settings);
    $("volume-intervals-result").hidden = intervals === null;
    if (intervals) {
      renderScreening("volume-intervals-screening", intervals.map(row => ({ label: `${row.fromVisitIndex}. ${row.fromVisit} → ${row.toVisitIndex}. ${row.toVisit} / ${row.name}`, row })), settings);
      renderRows("volume-intervals-rows", intervals.map(row => ({ label: `${row.fromVisitIndex}. ${row.fromVisit} → ${row.toVisitIndex}. ${row.toVisit} / ${row.name}`, row })));
      $("volume-intervals-scope").textContent = "Direct comparisons in the displayed visit order on the same common observed cells, including the baseline's footprint. Dates are not inferred. 2.5D cut/fill; overlapping areas and gross interval volumes must not be summed as unique earthwork. No compared cells means unavailable.";
    }
    const common = support.mode === "common_observed";
    $("volume-visits-scope").textContent = `Baseline: ${inputs[0].name}; ${selected.length} visits × ${areas.length} areas; cell ${settings.cell} ${settings.coordinateUnit}, height ${settings.height}, interpolation ${settings.fillEmpty ? "enabled" : "disabled"}. ` +
      (common ? `Same raw observed cells in all inputs: ${support.commonGridCells} of ${grid[2] * grid[3]} union-grid cells; each row uses the part inside its area. Uncompared includes cells missing in any input.` : "Shared grid; missing support varies by visit: do not subtract totals for between-visit quantities.") +
      " 2.5D volumes relative to baseline. Unavailable means no compared cells. Overlapping areas must not be summed. Scroll for coverage counts.";
    $("volume-visits-result").hidden = false;
    setStatus(`Compared ${selected.length} visits across ${areas.length} areas with a fixed baseline. ${common ? "Same raw observed support in every row of an area." : "No between-visit increments inferred."}${intervals ? ` Computed ${selected.length - 1} direct intervals.` : ""}`);
  } catch (error) { setStatus(`Visit comparison failed: ${signal.aborted ? "Cancelled; no partial summaries published" : errorText(error)}`, true); }
  finally { signal.removeEventListener("abort", cancelled); activeInputs = []; running = false; endTask(signal); updateForm(); }
};
function visitSnapshot(visits: CompletedVisits): VolumeVisitSnapshot {
  return { schema: "cloudanalyzer.area_volume_visits.v1", baseline: describeInput(visits.inputs[0]),
    visits: visits.inputs.slice(1).map((input, i) => ({ visitIndex: i + 1, ...describeInput(input) })),
    ordering: { policy: "User-controlled visit list, initially loaded order; no acquisition dates inferred", cloudIds: visits.inputs.slice(1).map(input => input.cloud!.id) },
    ...(visits.intervals ? { intervals: { scope: intervalVolumeScope, rows: visits.intervals.map(row => ({ ...row, net: row.added - row.removed,
      uncomparedCells: row.totalCells - row.matchedCells, comparisonAvailable: row.matchedCells > 0 })) } } : {}),
    settings: visits.settings, ...(visits.settings.heightThreshold !== undefined ? { screeningScope: heightReviewScope } : {}), sharedGrid: { min: visits.grid.slice(0, 2), nx: visits.grid[2], ny: visits.grid[3], cell: visits.settings.cell, origin: "Lower-left boundary; centers offset by half a cell" },
    support: { mode: visits.support.mode, ...(visits.support.mask ? { commonGridCells: visits.support.commonGridCells,
      commonGridArea: visits.support.commonGridCells! * visits.settings.cell ** 2, mask: { file: "common-observed-cells.bin", sha256: visits.support.sha256,
        byteLength: visits.support.mask.length, encoding: "uint8 0/1, row-major j*nx+i on sharedGrid; intersect with each named area's closed cell-center bounds" } } : {}) },
    scope: visits.support.mode === "common_observed" ? commonVolumeScope : visitVolumeScope, rows: visits.rows.map(row => ({ ...row, net: row.added - row.removed,
      uncomparedCells: row.totalCells - row.matchedCells, comparisonAvailable: row.matchedCells > 0 })) };
}
$("volume-visits-json").onclick = () => {
  if (!visits || !visits.inputs.every(validInput)) return;
  download(new Blob([JSON.stringify(visitSnapshot(visits), null, 2)], { type: "application/json" }), "area-volume-visits.json");
};
$("volume-visits-report").onclick = () => {
  if (!visits || !visits.inputs.every(validInput)) return;
  download(new Blob([volumeVisitReport(visitSnapshot(visits))], { type: "text/html" }), "survey-volume-report.html");
};
$("volume-visits-csv").onclick = () => {
  if (visits && visits.inputs.every(validInput)) download(new Blob([visitVolumesCsv(visits.rows, visits.settings.coordinateUnit, visits.support)], { type: "text/csv" }), "area-volume-visits.csv");
};

$("volume-visits-mask").onclick = () => {
  if (visits?.support.mask && visits.inputs.every(validInput)) download(new Blob([visits.support.mask.slice().buffer], { type: "application/octet-stream" }), "common-observed-cells.bin");
};

$("volume-intervals-csv").onclick = () => {
  if (visits?.intervals && visits.inputs.every(validInput)) download(new Blob([intervalVolumesCsv(visits.intervals, visits.settings.coordinateUnit, visits.support)], { type: "text/csv" }), "direct-visit-intervals.csv");
};
