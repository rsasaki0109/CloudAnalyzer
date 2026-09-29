/** Distance panel (C2C, C2M, M3C2) and the colorbar of the shown distances. */

import { cloudToCloud, computeM3c2, mapVoxelScores } from "../api";
import { RAMPS, gradientCss, lut, quantile, type RampName } from "../colormap";
import type { C2cOutput, MapQuality } from "../protocol";
import { refreshColors } from "./colors";
import { $, errorText, fillTable, fmt, setStatus } from "./dom";
import { addEntry, renderList, saveCloud } from "./entries";
import { record } from "./history";
import { addReportSection, type ReportSection } from "./report";
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
  const quality = methodSelect.value === "quality";
  runButton.disabled =
    !comparedSelect.value ||
    !referenceSelect.value ||
    comparedSelect.value === referenceSelect.value ||
    ((m3c2 || quality) && !!reference && isMesh(reference));
  $("c2m-signed-row").hidden = m3c2 || quality || !reference || !isMesh(reference);
  $("m3c2-options").hidden = !m3c2;
  $("quality-options").hidden = !quality;
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

/** M3C2 from `reference` to `compared` with the panel's settings, shown on the core points. */
export async function runM3c2(compared: Entry, reference: Entry): Promise<Entry> {
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
  return entry;
}

runButton.onclick = async () => {
  const compared = entries.get(Number(comparedSelect.value));
  const reference = entries.get(Number(referenceSelect.value));
  if (!compared || !reference) return;
  runButton.disabled = true;
  try {
    if (methodSelect.value === "m3c2") await runM3c2(compared, reference);
    else if (methodSelect.value === "quality") await runMapQuality(compared, reference);
    else await runNearest(compared, reference, $<HTMLInputElement>("c2m-signed").checked);
  } catch (err) {
    setStatus(`M3C2 failed: ${errorText(err)}`, true);
  } finally {
    updateRunButton();
  }
};

/** Mean and RMS of the distances up to `threshold`, and their share. */
function inliers(distances: Float32Array, threshold: number): { share: number; mean: number; rms: number } {
  let [n, sum, sum2] = [0, 0, 0];
  for (const d of distances) {
    if (!(d <= threshold)) continue;
    n++;
    sum += d;
    sum2 += d * d;
  }
  return {
    share: distances.length ? n / distances.length : 0,
    mean: n ? sum / n : Number.NaN,
    rms: n ? Math.sqrt(sum2 / n) : Number.NaN,
  };
}

/**
 * A map (`compared`) against its ground truth (`reference`), after MapEval:
 * accuracy, completeness and Chamfer from nearest distances both ways, and
 * AWD / SCS from voxel Gaussians. The map is coloured by its distance.
 */
export async function runMapQuality(compared: Entry, reference: Entry): Promise<MapQuality> {
  const num = (id: string) => Number($<HTMLInputElement>(id).value);
  const threshold = Math.max(1e-6, num("quality-threshold") || 0.3);
  const voxel = Math.max(1e-3, num("quality-voxel") || 1);
  const minPoints = Math.max(3, Math.round(num("quality-min") || 10));
  setStatus(`Map quality: ${compared.cloud.name} against ${reference.cloud.name}…`);
  const start = performance.now();
  const ours = await cloudToCloud(compared.cloud.id, reference.cloud.id, false);
  const theirs = await cloudToCloud(reference.cloud.id, compared.cloud.id, false);
  const voxels = await mapVoxelScores({
    compared: compared.cloud.id,
    reference: reference.cloud.id,
    voxel,
    minPoints,
  });
  const [a, b] = [inliers(ours.distances, threshold), inliers(theirs.distances, threshold)];
  const f1 = a.share + b.share > 0 ? (2 * a.share * b.share) / (a.share + b.share) : 0;
  const quality: MapQuality = {
    threshold,
    accuracy: a.rms,
    precision: a.share,
    completeness: b.share,
    f1,
    chamfer: (a.mean + b.mean) / 2,
    ...voxels,
  };
  compared.c2c = {
    ...ours,
    quality,
    millis: performance.now() - start,
    referenceName: reference.cloud.name,
  };
  compared.mode = "c2c";
  display.activeC2c = compared.cloud.id;
  display.range = { lo: 0, hi: threshold };
  refreshColors(compared);
  renderList();
  distanceChanged.emit();
  const pct = (v: number) => `${(100 * v).toFixed(1)} %`;
  setStatus(
    `Map quality within ${fmt(threshold)} m: accuracy ${fmt(quality.accuracy)} m, completeness ${pct(quality.completeness)}, ` +
      `F1 ${quality.f1.toFixed(3)}, Chamfer ${fmt(quality.chamfer)} m, AWD ${fmt(quality.awd)} m, SCS ${fmt(quality.scs)} m ` +
      `(${voxels.voxels.toLocaleString()} voxels)`,
  );
  return quality;
}

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
  // A scalar field shows the colorbar too; its statistics are in the Scalar fields panel.
  const scalar = entry?.mode === "scalar" ? entry.field : undefined;
  $("c2c-result").hidden = !c2c;
  $("colorbar").hidden = !c2c && !scalar;
  if (entry && scalar) {
    renderColorbar(`${scalar.name} · ${entry.cloud.name}`, scalar.stats);
    return;
  }
  if (!entry || !c2c) return;

  const { stats, quality } = c2c;
  const pct = (v: number) => `${(100 * v).toFixed(1)} %`;
  fillTable($("c2c-stats"), [
    ...(quality
      ? ([
          [`Accuracy (≤ ${fmt(quality.threshold)})`, fmt(quality.accuracy)],
          ["Precision", pct(quality.precision)],
          ["Completeness", pct(quality.completeness)],
          ["F1", quality.f1.toFixed(3)],
          ["Chamfer", fmt(quality.chamfer)],
          ["AWD", fmt(quality.awd)],
          ["SCS", fmt(quality.scs)],
        ] as [string, string][])
      : []),
    ["Points", stats.count.toLocaleString()],
    ["Mean", fmt(stats.mean)],
    ["Std. dev.", fmt(stats.stdDev)],
    ["RMS", fmt(stats.rms)],
    ["Median", fmt(stats.median)],
    ["Min", fmt(stats.min)],
    ["Max", fmt(stats.max)],
  ]);

  renderColorbar(
    c2c.quality
      ? `Distance to ${c2c.referenceName} · ${entry.cloud.name}`
      : c2c.kind === "volume"
        ? `Height difference (after − before) · ${entry.cloud.name}`
        : c2c.kind === "m3c2"
          ? `M3C2 distance · ${entry.cloud.name}`
          : c2c.kind === "raster"
            ? `Height · ${entry.cloud.name}`
            : `${c2c.kind === "c2m" ? (c2c.signed ? "Signed C2M" : "C2M") : "C2C"} distance · ${entry.cloud.name}`,
    stats,
  );
}

function renderColorbar(title: string, stats: { min: number; max: number }): void {
  const { lo, hi } = display.range ?? { lo: stats.min, hi: stats.max };
  minInput.value = fmt(lo);
  maxInput.value = fmt(hi);
  rampSelect.value = display.ramp;
  $("colorbar-title").textContent = title;
  $("colorbar-ramp").style.background = gradientCss(display.ramp);
  $("colorbar-max").textContent = fmt(hi);
  $("colorbar-mid").textContent = fmt((lo + hi) / 2);
  $("colorbar-min").textContent = fmt(lo);
}
distanceChanged.add(renderResult);

/** 95th percentile of |d|, cached per result (a quantile of millions of values is not free). */
const p95Cache = new WeakMap<Float32Array, number>();
function p95(distances: Float32Array): number {
  let v = p95Cache.get(distances);
  if (v === undefined) {
    v = quantile(distances.filter(Number.isFinite).map(Math.abs), 0.95);
    p95Cache.set(distances, v);
  }
  return v;
}

addReportSection("distance", () => {
  const out = new Map<string, ReportSection>();
  for (const entry of entries.values()) {
    const c2c = entry.c2c;
    if (!c2c || c2c.kind === "volume" || c2c.kind === "raster") continue;
    const kind = c2c.quality
      ? "Map quality"
      : c2c.kind === "m3c2"
        ? "M3C2"
        : c2c.kind === "c2m"
          ? c2c.signed
            ? "Signed C2M"
            : "C2M"
          : "C2C";
    const { stats, quality } = c2c;
    out.set(entry.cloud.name, {
      title: `${kind} ${entry.cloud.name} → ${c2c.referenceName}`,
      metrics: {
        count: { label: "Points", value: stats.count },
        mean: { label: "Mean", value: stats.mean },
        median: { label: "Median", value: stats.median },
        rms: { label: "RMS", value: stats.rms },
        std: { label: "Std. dev.", value: stats.stdDev },
        min: { label: "Min", value: stats.min },
        max: { label: "Max", value: stats.max },
        p95: { label: "95th percentile |d|", value: p95(c2c.distances) },
        ...(quality && {
          accuracy: {
            label: `Accuracy (RMS ≤ ${quality.threshold} m)`,
            value: quality.accuracy,
          },
          precision: { label: "Precision", value: quality.precision },
          completeness: { label: "Completeness", value: quality.completeness },
          f1: { label: "F1", value: quality.f1 },
          chamfer: { label: "Chamfer", value: quality.chamfer },
          awd: { label: "AWD", value: quality.awd },
          scs: { label: "SCS", value: quality.scs },
        }),
      },
    });
  }
  return out;
});

/** Draw the colorbar's ramp into an exported image. */
export function rampColor(t: number): string {
  const table = lut(display.ramp);
  const k = Math.round(Math.min(1, Math.max(0, t)) * 255) * 3;
  return `rgb(${table[k]} ${table[k + 1]} ${table[k + 2]})`;
}
