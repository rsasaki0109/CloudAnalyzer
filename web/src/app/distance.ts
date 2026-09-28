/** Distance panel (C2C, C2M, M3C2) and the colorbar of the shown distances. */

import { cloudToCloud, computeM3c2 } from "../api";
import { RAMPS, gradientCss, lut, quantile, type RampName } from "../colormap";
import type { C2cOutput } from "../protocol";
import { refreshColors } from "./colors";
import { $, errorText, fillTable, fmt, setStatus } from "./dom";
import { addEntry, renderList, saveCloud } from "./entries";
import { record } from "./history";
import { display, distanceChanged, type Entry, entries, isMesh, listChanged } from "./state";

const comparedSelect = $<HTMLSelectElement>("c2c-compared");
const referenceSelect = $<HTMLSelectElement>("c2c-reference");
export const runButton = $<HTMLButtonElement>("c2c-run");
const methodSelect = $<HTMLSelectElement>("distance-method");

function renderSelects(): void {
  // Only point clouds can be compared; the reference may also be a mesh.
  const clouds = [...entries.values()].filter((e) => !isMesh(e)).map((e) => String(e.cloud.id));
  const all = [...entries.keys()].map(String);
  let [compared, reference] = [comparedSelect.value, referenceSelect.value];
  const valid = clouds.includes(compared) && all.includes(reference) && compared !== reference;
  if (!valid) {
    // Default: the newest cloud against a mesh if there is one, else the first cloud.
    compared = clouds.at(-1) ?? "";
    const mesh = [...entries.values()].find(isMesh);
    reference = mesh ? String(mesh.cloud.id) : (all.find((id) => id !== compared) ?? "");
  }
  const option = (id: string) => {
    const entry = entries.get(Number(id))!;
    return new Option(isMesh(entry) ? `${entry.cloud.name} (mesh)` : entry.cloud.name, id);
  };
  comparedSelect.replaceChildren(...clouds.map(option));
  comparedSelect.value = compared;
  referenceSelect.replaceChildren(...all.map(option));
  referenceSelect.value = reference;
  updateRunButton();
}
listChanged.add(renderSelects);

function updateRunButton(): void {
  const reference = entries.get(Number(referenceSelect.value));
  const m3c2 = methodSelect.value === "m3c2";
  runButton.disabled =
    !comparedSelect.value ||
    !referenceSelect.value ||
    comparedSelect.value === referenceSelect.value ||
    (m3c2 && !!reference && isMesh(reference));
  $("c2m-signed-row").hidden = m3c2 || !reference || !isMesh(reference);
  $("m3c2-options").hidden = !m3c2;
}
methodSelect.onchange = updateRunButton;
comparedSelect.onchange = referenceSelect.onchange = updateRunButton;

/** Summary statistics over the finite values only. */
export function finiteStats(values: Float32Array): C2cOutput["stats"] {
  const finite = values.filter((v) => Number.isFinite(v));
  let [lo, hi, sum, sum2] = [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 0, 0];
  for (const v of finite) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
    sum += v;
    sum2 += v * v;
  }
  const n = finite.length || 1;
  const mean = sum / n;
  return {
    count: finite.length,
    min: finite.length ? lo : 0,
    max: finite.length ? hi : 0,
    mean,
    rms: Math.sqrt(sum2 / n),
    stdDev: Math.sqrt(Math.max(0, sum2 / n - mean * mean)),
    median: finite.length ? quantile(finite, 0.5) : 0,
  };
}

/** Show signed values on `entry` with a symmetric range on a diverging ramp (blue < 0 < red). */
export function showSigned(entry: Entry, result: NonNullable<Entry["c2c"]>): void {
  entry.c2c = result;
  entry.mode = "c2c";
  display.activeC2c = entry.cloud.id;
  const m = Math.max(Math.abs(result.stats.min), Math.abs(result.stats.max)) || 1;
  display.range = { lo: -m, hi: m };
  display.ramp = "Blue > White > Red";
  refreshColors(entry);
  renderList();
  distanceChanged.emit();
}

async function runM3c2(compared: Entry, reference: Entry): Promise<void> {
  const num = (id: string) => Number($<HTMLInputElement>(id).value);
  setStatus(`Computing M3C2: ${reference.cloud.name} → ${compared.cloud.name}…`);
  const out = await computeM3c2({
    compared: compared.cloud.id,
    reference: reference.cloud.id,
    normalRadius: num("m3c2-normal"),
    projectionRadius: num("m3c2-projection"),
    maxDepth: num("m3c2-depth"),
    coreSpacing: num("m3c2-core"),
  });
  const entry = addEntry(out.cloud);
  const stats = finiteStats(out.distance);
  let significant = 0;
  for (let i = 0; i < out.significant.length; i++) if (out.significant[i] > 0) significant++;
  record({ label: "M3C2", added: [entry], hide: [compared] });
  showSigned(entry, {
    kind: "m3c2",
    signed: true,
    distances: out.distance,
    stats,
    millis: out.millis,
    workers: 1,
    referenceName: reference.cloud.name,
  });
  const n = out.distance.length;
  setStatus(
    `M3C2 at ${n.toLocaleString()} core points in ${Math.round(out.millis)} ms: ` +
      `${stats.count.toLocaleString()} measured, ${significant.toLocaleString()} significant ` +
      `(${n ? ((100 * significant) / n).toFixed(1) : 0} %), mean ${fmt(stats.mean)}`,
  );
}

runButton.onclick = async () => {
  const compared = entries.get(Number(comparedSelect.value));
  const reference = entries.get(Number(referenceSelect.value));
  if (!compared || !reference) return;
  runButton.disabled = true;
  try {
    if (methodSelect.value === "m3c2") await runM3c2(compared, reference);
    else await runNearest(compared, reference, $<HTMLInputElement>("c2m-signed").checked);
  } catch (err) {
    setStatus(`M3C2 failed: ${errorText(err)}`, true);
  } finally {
    updateRunButton();
  }
};

/** C2C (or C2M against a mesh) from `compared` to `reference`, shown on `compared`. */
export async function runNearest(compared: Entry, reference: Entry, signed: boolean): Promise<void> {
  const kind = isMesh(reference) ? "C2M" : "C2C";
  setStatus(`Computing ${kind} distance: ${compared.cloud.name} → ${reference.cloud.name}…`);
  try {
    const result = await cloudToCloud(compared.cloud.id, reference.cloud.id, signed);
    compared.c2c = { ...result, referenceName: reference.cloud.name };
    compared.mode = "c2c";
    display.activeC2c = compared.cloud.id;
    display.range = null;
    refreshColors(compared);
    renderList();
    distanceChanged.emit();
    setStatus(
      `${kind} distance computed for ${result.stats.count.toLocaleString()} points in ${Math.round(result.millis)} ms` +
        (result.workers > 1 ? ` on ${result.workers} workers` : ""),
    );
  } catch (err) {
    setStatus(`${kind} failed: ${errorText(err)}`, true);
  }
}

// ---------------------------------------------------------------- colorbar

const rampSelect = $<HTMLSelectElement>("ramp");
for (const name of Object.keys(RAMPS)) rampSelect.add(new Option(name, name));
rampSelect.onchange = () => {
  display.ramp = rampSelect.value as RampName;
  applyRange();
};

const minInput = $<HTMLInputElement>("range-min");
const maxInput = $<HTMLInputElement>("range-max");
minInput.onchange = maxInput.onchange = () => {
  const lo = Number(minInput.value);
  const hi = Number(maxInput.value);
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
    display.range = { lo, hi };
    applyRange();
  }
};

const activeEntry = () => (display.activeC2c !== null ? entries.get(display.activeC2c) : undefined);

for (const format of ["ply", "las", "laz", "csv"] as const) {
  $<HTMLButtonElement>(`export-${format}`).onclick = () => {
    const entry = activeEntry();
    if (entry) void saveCloud(entry, format);
  };
}

$<HTMLButtonElement>("range-reset").onclick = () => {
  display.range = null;
  applyRange();
};

/** Recolor the shown distances after the ramp or range changed. */
export function applyRange(): void {
  const entry = activeEntry();
  if (entry) refreshColors(entry);
  distanceChanged.emit();
}

function renderResult(): void {
  const entry = activeEntry();
  const c2c = entry?.mode === "c2c" ? entry.c2c : undefined;
  $("c2c-result").hidden = !c2c;
  $("colorbar").hidden = !c2c;
  if (!entry || !c2c) return;

  const { stats } = c2c;
  fillTable($("c2c-stats"), [
    ["Points", stats.count.toLocaleString()],
    ["Mean", fmt(stats.mean)],
    ["Std. dev.", fmt(stats.stdDev)],
    ["RMS", fmt(stats.rms)],
    ["Median", fmt(stats.median)],
    ["Min", fmt(stats.min)],
    ["Max", fmt(stats.max)],
  ]);

  const { lo, hi } = display.range ?? { lo: stats.min, hi: stats.max };
  minInput.value = fmt(lo);
  maxInput.value = fmt(hi);
  rampSelect.value = display.ramp;
  $("colorbar-title").textContent =
    c2c.kind === "volume"
      ? `Height difference (after − before) · ${entry.cloud.name}`
      : c2c.kind === "m3c2"
        ? `M3C2 distance · ${entry.cloud.name}`
        : c2c.kind === "raster"
          ? `Height · ${entry.cloud.name}`
          : `${c2c.kind === "c2m" ? (c2c.signed ? "Signed C2M" : "C2M") : "C2C"} distance · ${entry.cloud.name}`;
  $("colorbar-ramp").style.background = gradientCss(display.ramp);
  $("colorbar-max").textContent = fmt(hi);
  $("colorbar-mid").textContent = fmt((lo + hi) / 2);
  $("colorbar-min").textContent = fmt(lo);
}
distanceChanged.add(renderResult);

/** Draw the colorbar's ramp into an exported image. */
export function rampColor(t: number): string {
  const table = lut(display.ramp);
  const k = Math.round(Math.min(1, Math.max(0, t)) * 255) * 3;
  return `rgb(${table[k]} ${table[k + 1]} ${table[k + 2]})`;
}
