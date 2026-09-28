/** ICP registration panel. */

import { registerIcp, transformCloud } from "../api";
import { $, errorText, fillTable, fmt, setStatus } from "./dom";
import { replaceCloud } from "./entries";
import { clouds, entries, listChanged } from "./state";

const icpMoving = $<HTMLSelectElement>("icp-moving");
const icpReference = $<HTMLSelectElement>("icp-reference");
const icpButton = $<HTMLButtonElement>("icp-run");
let lastIcp: number | null = null;

interface IcpSummary {
  cloudId: number;
  rmsInitial: number;
  rmsFinal: number;
  iterations: number;
  converged: boolean;
  millis: number;
}
let icpSummary: IcpSummary | null = null;

function renderSelects(): void {
  const ids = clouds().map((e) => String(e.cloud.id));
  let [moving, reference] = [icpMoving.value, icpReference.value];
  if (!(ids.includes(moving) && ids.includes(reference) && moving !== reference)) {
    // Default: align the newest cloud to the first one.
    [moving, reference] = [ids.at(-1) ?? "", ids[0] ?? ""];
  }
  for (const [select, value] of [
    [icpMoving, moving],
    [icpReference, reference],
  ] as const) {
    select.replaceChildren(...ids.map((id) => new Option(entries.get(Number(id))!.cloud.name, id)));
    select.value = value;
  }
  updateButton();
  renderResult();
}
listChanged.add(renderSelects);

function updateButton(): void {
  icpButton.disabled = !icpMoving.value || !icpReference.value || icpMoving.value === icpReference.value;
}
icpMoving.onchange = icpReference.onchange = updateButton;

/** Inverse of a row-major 4x4 rigid transform. */
export function invertRigid(m: number[]): number[] {
  const r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const t = [m[3], m[7], m[11]];
  const rt = (i: number, j: number) => r[j * 3 + i]; // transpose
  const ti = [0, 1, 2].map((i) => -(rt(i, 0) * t[0] + rt(i, 1) * t[1] + rt(i, 2) * t[2]));
  return [
    rt(0, 0), rt(0, 1), rt(0, 2), ti[0],
    rt(1, 0), rt(1, 1), rt(1, 2), ti[1],
    rt(2, 0), rt(2, 1), rt(2, 2), ti[2],
    0, 0, 0, 1,
  ];
}

/** A row-major 4x4 matrix as four lines of text. */
export function matrixText(matrix: number[]): string {
  return [0, 1, 2, 3].map((r) => matrix.slice(r * 4, r * 4 + 4).map((v) => v.toFixed(9)).join(" ")).join("\n");
}

function renderResult(): void {
  const entry = lastIcp !== null ? entries.get(lastIcp) : undefined;
  const matrix = entry?.transforms.at(-1);
  $("icp-result").hidden = !entry || !matrix;
  if (!entry || !matrix) return;
  const rows: [string, string][] = [["Aligned", entry.cloud.name]];
  if (icpSummary?.cloudId === entry.cloud.id) {
    rows.push(
      ["RMS before", fmt(icpSummary.rmsInitial)],
      ["RMS after", fmt(icpSummary.rmsFinal)],
      ["Iterations", `${icpSummary.iterations}${icpSummary.converged ? "" : " (not converged)"}`],
    );
  }
  fillTable($("icp-stats"), rows);
  $<HTMLTextAreaElement>("icp-matrix").value = matrixText(matrix);
}

icpButton.onclick = async () => {
  const moving = entries.get(Number(icpMoving.value));
  const reference = entries.get(Number(icpReference.value));
  if (!moving || !reference) return;
  icpButton.disabled = true;
  setStatus(`Aligning ${moving.cloud.name} to ${reference.cloud.name}…`);
  try {
    const out = await registerIcp({
      moving: moving.cloud.id,
      reference: reference.cloud.id,
      maxIterations: Math.max(1, Number($<HTMLInputElement>("icp-iterations").value) || 50),
      overlap: Math.min(100, Math.max(10, Number($<HTMLInputElement>("icp-overlap").value) || 100)) / 100,
      matchCentroids: $<HTMLInputElement>("icp-centroids").checked,
      pointToPlane: $<HTMLSelectElement>("icp-metric").value === "plane",
    });
    moving.transforms.push(out.matrix);
    lastIcp = moving.cloud.id;
    icpSummary = { cloudId: moving.cloud.id, ...out };
    replaceCloud(moving, out.cloud);
    setStatus(
      `Aligned in ${out.iterations} iterations (${Math.round(out.millis)} ms): RMS ${fmt(out.rmsInitial)} → ${fmt(out.rmsFinal)}` +
        (out.converged ? "" : " — not converged, try more iterations"),
    );
  } catch (err) {
    setStatus(`ICP failed: ${errorText(err)}`, true);
  } finally {
    updateButton();
  }
};

$<HTMLButtonElement>("icp-undo").onclick = async () => {
  const entry = lastIcp !== null ? entries.get(lastIcp) : undefined;
  const matrix = entry?.transforms.pop();
  if (!entry || !matrix) return;
  try {
    const cloud = await transformCloud(entry.cloud.id, invertRigid(matrix));
    icpSummary = null;
    replaceCloud(entry, cloud);
    setStatus(`Undid the alignment of ${entry.cloud.name}`);
  } catch (err) {
    entry.transforms.push(matrix);
    setStatus(`Undo failed: ${errorText(err)}`, true);
  }
};
