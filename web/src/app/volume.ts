/** 2.5D cut / fill volume panel. */

import { computeVolume, removeCloud, workerBusy } from "../api";
import type { VolumeOutput } from "../protocol";
import { validateVolumeGrid, volumeCellsCsv } from "../surface-volume";
import { projectChanged } from "../project-change";
import { finiteStats, showSigned } from "./distance";
import { $, download, errorText, fillTable, fmt, roundUp, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { addReportSection } from "./report";
import { distanceChanged, entries, isMesh, listChanged, pointsInvalidated, viewer, type Entry } from "./state";
import { endTask, startTask, taskActive } from "./tasks";

const volumeBefore = $<HTMLSelectElement>("volume-before");
const volumeAfter = $<HTMLSelectElement>("volume-after");
const volumeCell = $<HTMLInputElement>("volume-cell");
const volumeRun = $<HTMLButtonElement>("volume-run");
const coordinateUnit = $<HTMLSelectElement>("volume-unit");
const CONSTANT = "constant";
type Input = { side: { id: number } | { z: number }; name: string; entry?: Entry; cloud?: Entry["cloud"] };
type Settings = { cell: number; height: "mean" | "min" | "max"; fillEmpty: boolean; coordinateUnit: string };
let last: { out: VolumeOutput; before: string; after: string; inputs: Input[]; settings: Settings; output: Entry | null } | null = null;
let running = false, generation = 0, activeInputs: Input[] = [];
function validInput(input: Input): boolean {
  return !input.entry || entries.get(input.cloud!.id) === input.entry && input.entry.cloud === input.cloud;
}
function validResult(result: NonNullable<typeof last>): boolean {
  return result.inputs.every(validInput) && (!result.output || entries.get(result.output.cloud.id) === result.output && result.output.cloud === result.out.cells);
}
function clearResult(): void {
  const had = last !== null; generation++; last = null;
  $("volume-result").hidden = $("volume-exports").hidden = true;
  if (had) distanceChanged.emit();
}
pointsInvalidated.add(id => {
  if ([...activeInputs, ...(last?.inputs ?? [])].some(input => input.cloud?.id === id) || last?.out.cells?.id === id) clearResult();
});

addReportSection("volume", () =>
  last && validResult(last)
    ? {
        title: `Volume ${last.after} vs ${last.before}`,
        metrics: {
          fill: { label: "Fill (added)", value: last.out.added, unit: `${last.settings.coordinateUnit}³` },
          cut: { label: "Cut (removed)", value: last.out.removed, unit: `${last.settings.coordinateUnit}³` },
          net: { label: "Net", value: last.out.added - last.out.removed, unit: `${last.settings.coordinateUnit}³` },
          fill_area: { label: "Fill area", value: last.out.addedArea, unit: `${last.settings.coordinateUnit}²` },
          cut_area: { label: "Cut area", value: last.out.removedArea, unit: `${last.settings.coordinateUnit}²` },
          uncompared: { label: "Grid cells without both surface heights", value: last.out.totalCells - last.out.matchedCells },
          coverage: {
            label: "Cells compared",
            value: last.out.totalCells ? (100 * last.out.matchedCells) / last.out.totalCells : 0,
            unit: "%",
          },
        },
        notes: [`Cell size ${last.out.cell}; height ${last.settings.height}; interpolation ${last.settings.fillEmpty ? "enabled" : "disabled"}; coordinate unit ${last.settings.coordinateUnit} (no conversion).`,
          "Full loaded surfaces on the union XY grid; expected overlap bounds in distance review do not clip this volume. Only cells with both heights contribute; undefined cells are not zero change.",
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
  updateForm();
}
listChanged.add(() => { if (last && !validResult(last)) clearResult(); renderSelects(); });

function updateForm(): void {
  $("volume-before-z-row").hidden = volumeBefore.value !== CONSTANT;
  $("volume-after-z-row").hidden = volumeAfter.value !== CONSTANT;
  const both = volumeBefore.value === CONSTANT && volumeAfter.value === CONSTANT;
  volumeRun.disabled = running || entries.size === 0 || both || volumeBefore.value === volumeAfter.value;
  for (const input of [volumeBefore, volumeAfter, volumeCell, coordinateUnit,
    $("volume-height"), $("volume-fill"), $("volume-before-z"), $("volume-after-z")] as (HTMLInputElement | HTMLSelectElement)[]) input.disabled = running;
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
function snapshot(select: HTMLSelectElement, z: HTMLInputElement): Input {
  if (select.value === CONSTANT) {
    if (!z.value.trim() || !Number.isFinite(Number(z.value))) throw new Error("Enter a finite constant surface height");
    return { side: { z: Number(z.value) }, name: `Constant height ${z.value}` };
  }
  const entry = entries.get(Number(select.value));
  if (!entry) throw new Error("Choose two available surfaces");
  return { side: { id: entry.cloud.id }, name: entry.cloud.name, entry, cloud: entry.cloud };
}

function unit(value: number, power: 2 | 3): string {
  return `${fmt(value)} ${last?.settings.coordinateUnit ?? coordinateUnit.value}${power === 3 ? "³" : "²"}`;
}

volumeRun.onclick = async () => {
  if (running || taskActive() || workerBusy()) return setStatus("Finish the current operation before computing volume", true);
  const settings: Settings = { cell: Number(volumeCell.value), height: $<HTMLSelectElement>("volume-height").value as Settings["height"],
    fillEmpty: $<HTMLInputElement>("volume-fill").checked, coordinateUnit: coordinateUnit.value };
  let inputs: Input[];
  try {
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
      ["Cells compared", `${out.matchedCells.toLocaleString()} of ${out.totalCells.toLocaleString()} (${coverage.toFixed(1)} %)`],
      ["Uncompared cells", (out.totalCells - out.matchedCells).toLocaleString()],
      ["Cell size", fmt(out.cell)],
    ]);
    $("volume-exports").hidden = false;
    $<HTMLButtonElement>("volume-isolate").disabled = !output;
    $<HTMLButtonElement>("volume-mesh").disabled = !output;
    $("volume-scope").textContent = `${inputs[0].name} → ${inputs[1].name}; ${settings.height} height per cell; interpolation ${settings.fillEmpty ? "enabled" : "disabled"}. Full loaded surfaces; distance-review overlap bounds do not clip this volume. Blue = cut, red = fill; undefined cells are not zero change.`;
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
    scope: "Full loaded surfaces on their union XY grid; only cells with both heights contribute. Distance-review overlap bounds do not clip this volume. Empty cells are undefined unless interpolation is explicitly enabled. Not closed-solid volume.",
    result: { added: out.added, removed: out.removed, net: out.added - out.removed, addedArea: out.addedArea, removedArea: out.removedArea,
      matchedCells: out.matchedCells, totalCells: out.totalCells, uncomparedCells: out.totalCells - out.matchedCells, cell: out.cell },
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
