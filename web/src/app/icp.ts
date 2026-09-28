/** ICP registration panel. */

import { registerIcp } from "../api";
import { $, errorText, fillTable, fmt, setStatus } from "./dom";
import { replaceCloud } from "./entries";
import { lastStep, record, undo } from "./history";
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
    record({ label: "the ICP alignment", moved: { entry: moving, matrix: out.matrix } });
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

/** Undo the last alignment of the shown cloud, if it is the last step. */
$<HTMLButtonElement>("icp-undo").onclick = async () => {
  const entry = lastIcp !== null ? entries.get(lastIcp) : undefined;
  if (!entry || lastStep()?.moved?.entry !== entry) {
    setStatus("Other steps came after this alignment; undo those first (Ctrl+Z)", true);
    return;
  }
  icpSummary = null;
  await undo();
};
