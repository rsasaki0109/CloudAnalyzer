/**
 * Scalar fields: every per-point value of a cloud (intensity, distances,
 * M3C2 results, splat opacity, calculator results, Z…) can color the cloud,
 * be summarized in a histogram, cut a range out, or feed the calculator.
 */

import { fieldValues, filterByField, setField } from "../api";
import { lut, quantile } from "../colormap";
import { refreshColors } from "./colors";
import { finiteStats } from "./distance";
import { $, errorText, fillTable, fmt, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { evaluate, fieldsOf, parse } from "./expr";
import { record } from "./history";
import { fillCloudSelect } from "./processing";
import { clouds, display, distanceChanged, type Entry, entries, listChanged } from "./state";

const cloudSelect = $<HTMLSelectElement>("field-cloud");
const fieldSelect = $<HTMLSelectElement>("field-name");
const histogram = $("field-histogram").querySelector("canvas")!;
const loInput = $<HTMLInputElement>("field-lo");
const hiInput = $<HTMLInputElement>("field-hi");
const BINS = 64;

/** The name a computed distance goes by as a field. */
export function distanceField(entry: Entry): string | null {
  const c2c = entry.c2c;
  if (!c2c) return null;
  return { c2c: "C2C distance", c2m: "C2M distance", m3c2: "M3C2 distance", volume: "Height difference", raster: "Height" }[
    c2c.kind
  ];
}

/** Names of the fields a cloud has, the distance result first. */
export function fieldNames(entry: Entry): string[] {
  const names = new Set<string>();
  const distance = distanceField(entry);
  if (distance) names.add(distance);
  for (const name of entry.cloud.scalarNames) names.add(name);
  for (const name of entry.fields.keys()) names.add(name);
  names.add("Z");
  return [...names];
}

/** A field's values, fetched from the worker the first time. */
export async function field(entry: Entry, name: string): Promise<Float32Array> {
  if (name === distanceField(entry) && entry.c2c) return entry.c2c.distances;
  let values = entry.fields.get(name);
  if (!values) {
    values = await fieldValues(entry.cloud.id, name);
    entry.fields.set(name, values);
  }
  return values;
}

/** Color a cloud by one of its fields, with the colorbar's ramp and range. */
export async function colorByField(entry: Entry, name: string): Promise<void> {
  const values = await field(entry, name);
  entry.field = { name, values, stats: finiteStats(values) };
  entry.mode = "scalar";
  display.activeC2c = entry.cloud.id;
  display.range = null;
  refreshColors(entry);
  renderList();
  distanceChanged.emit();
}

// ---------------------------------------------------------------- panel

const selected = () => entries.get(Number(cloudSelect.value));

function renderFields(): void {
  const entry = selected();
  const previous = fieldSelect.value;
  const names = entry ? fieldNames(entry) : [];
  fieldSelect.replaceChildren(...names.map((n) => new Option(n, n)));
  if (names.includes(previous)) fieldSelect.value = previous;
  for (const id of ["field-color", "field-keep", "field-remove", "field-calc"]) {
    $<HTMLButtonElement>(id).disabled = !entry;
  }
  void renderStats();
}

listChanged.add(() => {
  fillCloudSelect(cloudSelect, clouds());
  renderFields();
});
cloudSelect.onchange = renderFields;
fieldSelect.onchange = () => {
  loInput.value = hiInput.value = "";
  void renderStats();
};

let statsFor = "";

async function renderStats(): Promise<void> {
  const entry = selected();
  const name = fieldSelect.value;
  $("field-result").hidden = !entry || !name;
  if (!entry || !name) return;
  const key = `${entry.cloud.id}:${name}`;
  statsFor = key;
  let values: Float32Array;
  try {
    values = await field(entry, name);
  } catch (err) {
    setStatus(`${name}: ${errorText(err)}`, true);
    return;
  }
  if (statsFor !== key) return;
  const finite = values.filter(Number.isFinite);
  const stats = finiteStats(values);
  fillTable($("field-stats"), [
    ["Values", `${stats.count.toLocaleString()}${stats.count < values.length ? ` (${(values.length - stats.count).toLocaleString()} missing)` : ""}`],
    ["Mean", fmt(stats.mean)],
    ["Std. dev.", fmt(stats.stdDev)],
    ["Min", fmt(stats.min)],
    ["5 %", fmt(quantile(finite, 0.05))],
    ["Median", fmt(stats.median)],
    ["95 %", fmt(quantile(finite, 0.95))],
    ["Max", fmt(stats.max)],
  ]);
  if (!loInput.value) loInput.value = fmt(stats.min);
  if (!hiInput.value) hiInput.value = fmt(stats.max);
  drawHistogram(finite, stats.min, stats.max);
}

function drawHistogram(values: Float32Array, min: number, max: number): void {
  const dpr = window.devicePixelRatio || 1;
  const [w, h] = [histogram.clientWidth, histogram.clientHeight];
  histogram.width = Math.round(w * dpr);
  histogram.height = Math.round(h * dpr);
  const ctx = histogram.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const span = max - min || 1;
  const counts = new Uint32Array(BINS);
  for (const v of values) counts[Math.min(BINS - 1, Math.floor(((v - min) / span) * BINS))]++;
  const peak = Math.max(1, ...counts);
  const [lo, hi] = [Number(loInput.value), Number(hiInput.value)];
  const table = lut(display.ramp);
  const barW = w / BINS;
  for (let b = 0; b < BINS; b++) {
    const center = min + ((b + 0.5) / BINS) * span;
    const inside = !(center < lo) && !(center > hi);
    const k = Math.round((b / (BINS - 1)) * 255) * 3;
    ctx.fillStyle = inside ? `rgb(${table[k]} ${table[k + 1]} ${table[k + 2]})` : "rgb(139 147 165 / 0.35)";
    const barH = Math.max(counts[b] ? 1 : 0, (counts[b] / peak) * (h - 2));
    ctx.fillRect(b * barW + 0.5, h - barH, Math.max(1, barW - 1), barH);
  }
}

loInput.onchange = hiInput.onchange = () => void renderStats();
new ResizeObserver(() => void renderStats()).observe(histogram);

$<HTMLButtonElement>("field-color").onclick = async () => {
  const entry = selected();
  if (!entry || !fieldSelect.value) return;
  try {
    await colorByField(entry, fieldSelect.value);
  } catch (err) {
    setStatus(`${fieldSelect.value}: ${errorText(err)}`, true);
  }
};

async function cut(inside: boolean): Promise<void> {
  const entry = selected();
  const name = fieldSelect.value;
  const [lo, hi] = [Number(loInput.value), Number(hiInput.value)];
  if (!entry || !name) return;
  if (!(lo <= hi)) {
    setStatus("Enter a range with min ≤ max", true);
    return;
  }
  try {
    const values = await field(entry, name);
    const cloud = await filterByField(entry.cloud.id, values, lo, hi, inside);
    const added = addEntry(cloud);
    record({ label: `the ${name} filter`, added: [added], hide: [entry] });
    renderList();
    setStatus(
      `${cloud.name}: ${cloud.count.toLocaleString()} of ${entry.cloud.count.toLocaleString()} points with ` +
        `${name} ${inside ? "within" : "outside"} ${fmt(lo)} … ${fmt(hi)}`,
    );
  } catch (err) {
    setStatus(`Filter failed: ${errorText(err)}`, true);
  }
}
$<HTMLButtonElement>("field-keep").onclick = () => void cut(true);
$<HTMLButtonElement>("field-remove").onclick = () => void cut(false);

$<HTMLButtonElement>("field-calc").onclick = async () => {
  const entry = selected();
  const name = $<HTMLInputElement>("field-calc-name").value.trim() || "result";
  const text = $<HTMLInputElement>("field-expr").value;
  if (!entry) return;
  try {
    const node = parse(text);
    const inputs = new Map<string, Float32Array>();
    const known = fieldNames(entry);
    for (const used of fieldsOf(node)) {
      const match = known.find((k) => k === used) ?? known.find((k) => k.toLowerCase() === used.toLowerCase());
      if (!match) throw new Error(`no field "${used}" (have: ${known.join(", ")})`);
      inputs.set(used, await field(entry, match));
    }
    const values = evaluate(node, inputs, entry.cloud.count);
    // Stored in the worker too, so exports carry it.
    await setField(entry.cloud.id, name, values);
    entry.fields.set(name, values);
    if (!entry.cloud.scalarNames.includes(name)) entry.cloud.scalarNames.push(name);
    renderFields();
    fieldSelect.value = name;
    await colorByField(entry, name);
    setStatus(`Computed ${name} = ${text} for ${entry.cloud.count.toLocaleString()} points`);
  } catch (err) {
    setStatus(`Calculator: ${errorText(err)}`, true);
  }
};
