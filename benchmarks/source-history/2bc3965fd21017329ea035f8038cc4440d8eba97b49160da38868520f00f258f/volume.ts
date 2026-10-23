/** 2.5D cut / fill volume panel. */

import { computeVolume } from "../api";
import type { VolumeOutput } from "../protocol";
import { finiteStats, showSigned } from "./distance";
import { $, errorText, fillTable, fmt, roundUp, setStatus } from "./dom";
import { addEntry } from "./entries";
import { record } from "./history";
import { addReportSection } from "./report";
import { entries, isMesh, listChanged } from "./state";

const volumeBefore = $<HTMLSelectElement>("volume-before");
const volumeAfter = $<HTMLSelectElement>("volume-after");
const volumeCell = $<HTMLInputElement>("volume-cell");
const volumeRun = $<HTMLButtonElement>("volume-run");
const CONSTANT = "constant";
let last: { out: VolumeOutput; before: string; after: string } | null = null;

addReportSection("volume", () =>
  last
    ? {
        title: `Volume ${last.after} vs ${last.before}`,
        metrics: {
          fill: { label: "Fill (added)", value: last.out.added, unit: "m³" },
          cut: { label: "Cut (removed)", value: last.out.removed, unit: "m³" },
          net: { label: "Net", value: last.out.added - last.out.removed, unit: "m³" },
          fill_area: { label: "Fill area", value: last.out.addedArea, unit: "m²" },
          cut_area: { label: "Cut area", value: last.out.removedArea, unit: "m²" },
          coverage: {
            label: "Cells compared",
            value: last.out.totalCells ? (100 * last.out.matchedCells) / last.out.totalCells : 0,
            unit: "%",
          },
        },
        notes: [`Cell size ${fmt(last.out.cell)}`],
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
listChanged.add(renderSelects);

function updateForm(): void {
  $("volume-before-z-row").hidden = volumeBefore.value !== CONSTANT;
  $("volume-after-z-row").hidden = volumeAfter.value !== CONSTANT;
  const both = volumeBefore.value === CONSTANT && volumeAfter.value === CONSTANT;
  volumeRun.disabled = entries.size === 0 || both || volumeBefore.value === volumeAfter.value;
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
  volumeBefore.dataset.picked = "1";
  updateForm();
};

function side(select: HTMLSelectElement, z: HTMLInputElement): { id: number } | { z: number } {
  return select.value === CONSTANT ? { z: Number(z.value) } : { id: Number(select.value) };
}

function unit(value: number, power: 2 | 3): string {
  return `${fmt(value)} ${power === 3 ? "m³" : "m²"}`;
}

volumeRun.onclick = async () => {
  const cell = Number(volumeCell.value);
  if (!(cell > 0)) {
    setStatus("Enter a positive cell size", true);
    return;
  }
  volumeRun.disabled = true;
  setStatus("Computing volume…");
  try {
    const out = await computeVolume({
      before: side(volumeBefore, $<HTMLInputElement>("volume-before-z")),
      after: side(volumeAfter, $<HTMLInputElement>("volume-after-z")),
      cell,
      height: $<HTMLSelectElement>("volume-height").value as "mean" | "min" | "max",
      fillEmpty: $<HTMLInputElement>("volume-fill").checked,
    });
    const coverage = out.totalCells ? (100 * out.matchedCells) / out.totalCells : 0;
    last = {
      out,
      before: volumeBefore.selectedOptions[0]?.text ?? "before",
      after: volumeAfter.selectedOptions[0]?.text ?? "after",
    };
    $("volume-result").hidden = false;
    fillTable($("volume-stats"), [
      ["Fill (added)", unit(out.added, 3)],
      ["Cut (removed)", unit(out.removed, 3)],
      ["Net", unit(out.added - out.removed, 3)],
      ["Fill area", unit(out.addedArea, 2)],
      ["Cut area", unit(out.removedArea, 2)],
      ["Cells compared", `${out.matchedCells.toLocaleString()} of ${out.totalCells.toLocaleString()} (${coverage.toFixed(1)} %)`],
      ["Cell size", fmt(out.cell)],
    ]);
    if (out.cells && out.difference) {
      const cells = addEntry(out.cells);
      record({ label: "the volume", added: [cells] });
      // Blue = cut, red = fill.
      showSigned(cells, {
        kind: "volume",
        signed: true,
        distances: out.difference,
        stats: finiteStats(out.difference),
        millis: out.millis,
        workers: 1,
        referenceName: volumeBefore.selectedOptions[0]?.text ?? "before",
      });
    }
    setStatus(
      `Volume: fill ${unit(out.added, 3)}, cut ${unit(out.removed, 3)}, net ${unit(out.added - out.removed, 3)} ` +
        `(${Math.round(out.millis)} ms)`,
    );
  } catch (err) {
    setStatus(`Volume failed: ${errorText(err)}`, true);
  } finally {
    updateForm();
  }
};
