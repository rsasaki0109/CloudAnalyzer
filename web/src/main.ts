import * as THREE from "three";
import { cloudToCloud, loadCloud, pointAt, registerIcp, removeCloud, transformCloud } from "./api";
import { RAMPS, colorize, gradientCss, lut, type RampName } from "./colormap";
import { type LodNode, parseNodes } from "./lod";
import type { C2cOutput, LoadedCloud } from "./protocol";
import { Viewer } from "./viewer";

type ColorMode = "rgb" | "solid" | "c2c";

interface Entry {
  cloud: LoadedCloud;
  nodes: LodNode[];
  solid: [number, number, number];
  mode: ColorMode;
  visible: boolean;
  c2c?: C2cOutput & { referenceName: string };
  /** Transforms applied by ICP, newest last, for undo. */
  transforms: number[][];
}

const SOLID_COLORS: [number, number, number][] = [
  [235, 235, 235],
  [255, 176, 0],
  [79, 195, 247],
  [255, 110, 156],
  [156, 204, 101],
  [186, 104, 200],
];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const viewer = new Viewer($("viewport"));
viewer.setBackground(getComputedStyle(document.documentElement).getPropertyValue("--viewport").trim());

const entries = new Map<number, Entry>();
let rampName: RampName = "Blue > Green > Yellow > Red";
let range: { lo: number; hi: number } | null = null;
let activeC2c: number | null = null;

// ---------------------------------------------------------------- status

function setStatus(message: string, isError = false): void {
  const el = $("status");
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function fmt(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e4 || a < 1e-3 ? v.toExponential(3) : v.toPrecision(5).replace(/\.?0+$/, "");
}

// ---------------------------------------------------------------- colors

function solidColors(entry: Entry): Uint8Array {
  const out = new Uint8Array(entry.cloud.count * 3);
  for (let i = 0; i < out.length; i += 3) out.set(entry.solid, i);
  return out;
}

function colorsFor(entry: Entry): Uint8Array {
  if (entry.mode === "rgb" && entry.cloud.colors) return entry.cloud.colors;
  if (entry.mode === "c2c" && entry.c2c) {
    const { lo, hi } = range ?? { lo: entry.c2c.stats.min, hi: entry.c2c.stats.max };
    return colorize(entry.c2c.distances, lo, hi, lut(rampName));
  }
  return solidColors(entry);
}

function refreshColors(entry: Entry): void {
  if (entry.cloud.kind === "mesh") viewer.setMeshColor(entry.cloud.id, entry.solid);
  else viewer.setColors(entry.cloud.id, colorsFor(entry));
}

const isMesh = (entry: Entry) => entry.cloud.kind === "mesh";

/** Label for a distance result, e.g. "C2M distance → part.stl". */
function distanceLabel(c2c: Entry["c2c"]): string {
  if (!c2c) return "Distance";
  return `${c2c.kind === "c2m" ? "C2M" : "C2C"} distance → ${c2c.referenceName}`;
}

// ---------------------------------------------------------------- cloud list

function renderList(): void {
  const list = $("cloud-list");
  list.replaceChildren();
  $("empty-hint").hidden = entries.size > 0;
  for (const entry of entries.values()) {
    const { cloud } = entry;
    const li = document.createElement("li");

    const visible = document.createElement("input");
    visible.type = "checkbox";
    visible.checked = entry.visible;
    visible.title = "Show / hide";
    visible.onchange = () => {
      entry.visible = visible.checked;
      viewer.setVisible(cloud.id, entry.visible);
    };

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `rgb(${entry.solid.join(" ")})`;

    const name = document.createElement("span");
    name.className = "name";
    name.title = cloud.name;
    name.textContent = cloud.name;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent =
      cloud.kind === "mesh"
        ? `${cloud.triangles.toLocaleString()} triangles · mesh`
        : `${cloud.count.toLocaleString()} points${cloud.colors ? " · RGB" : ""}`;
    name.append(meta);

    const remove = document.createElement("button");
    remove.className = "remove";
    remove.textContent = "✕";
    remove.title = "Remove";
    remove.onclick = () => void removeEntry(cloud.id);

    const mode = document.createElement("select");
    mode.title = "Color by";
    const options: [ColorMode, string, boolean][] = [
      ["rgb", "RGB", cloud.colors !== null],
      ["solid", "Solid color", true],
      ["c2c", distanceLabel(entry.c2c), !!entry.c2c],
    ];
    for (const [value, label, enabled] of options) {
      if (!enabled) continue;
      mode.add(new Option(label, value, false, value === entry.mode));
    }
    mode.onchange = () => {
      entry.mode = mode.value as ColorMode;
      refreshColors(entry);
      if (entry.mode === "c2c") activeC2c = cloud.id;
      else if (activeC2c === cloud.id) activeC2c = null;
      renderC2cResult();
    };

    // Meshes are drawn in their solid color only.
    if (isMesh(entry)) li.append(visible, swatch, name, remove);
    else li.append(visible, swatch, name, remove, mode);
    list.append(li);
  }
  renderC2cSelects();
  renderIcpSelects();
}

async function removeEntry(id: number): Promise<void> {
  entries.delete(id);
  viewer.remove(id);
  forgetPoints(id);
  await removeCloud(id);
  if (activeC2c === id) activeC2c = null;
  // Distances computed against the removed cloud are no longer meaningful to keep around.
  for (const entry of entries.values()) {
    if (entry.c2c && !findByName(entry.c2c.referenceName)) {
      entry.c2c = undefined;
      if (entry.mode === "c2c") {
        entry.mode = entry.cloud.colors ? "rgb" : "solid";
        refreshColors(entry);
      }
    }
  }
  if (entries.size === 0) $("shift").textContent = "";
  renderList();
  renderC2cResult();
}

function findByName(name: string): Entry | undefined {
  return [...entries.values()].find((e) => e.cloud.name === name);
}

// ---------------------------------------------------------------- loading

async function loadFiles(files: Iterable<File>): Promise<void> {
  for (const file of files) {
    const mb = (file.size / 1e6).toFixed(file.size >= 1e7 ? 0 : 1);
    setStatus(`Loading ${file.name} (${mb} MB): reading…`);
    const start = performance.now();
    try {
      const bytes = await file.arrayBuffer();
      const read = performance.now() - start;
      const cloud = await loadCloud(file.name, bytes, (note) =>
        setStatus(`Loading ${file.name} (${mb} MB): ${note}…`),
      );
      const entry: Entry = {
        cloud,
        nodes: parseNodes(cloud.lodNodes, cloud.lodGrid, cloud.shift),
        solid: SOLID_COLORS[entries.size % SOLID_COLORS.length],
        mode: cloud.colors ? "rgb" : "solid",
        visible: true,
        transforms: [],
      };
      entries.set(cloud.id, entry);
      if (cloud.kind === "mesh") viewer.addMesh(cloud.id, cloud.positions, cloud.indices!, entry.solid);
      else viewer.add(cloud.id, cloud.positions, colorsFor(entry), entry.nodes);
      if (entries.size === 1) viewer.fit();
      const [sx, sy, sz] = cloud.shift;
      $("shift").textContent =
        sx || sy || sz ? `Global shift: (${-sx}, ${-sy}, ${-sz})` : "";
      const { parse, index, prepare } = cloud.timings;
      const s = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
      const size =
        cloud.kind === "mesh"
          ? `${cloud.triangles.toLocaleString()} triangles`
          : `${cloud.count.toLocaleString()} points`;
      setStatus(
        `Loaded ${file.name}: ${size} in ${s(performance.now() - start)} ` +
          `(read ${s(read)} · parse ${s(parse)} · index ${s(index)} · prepare ${s(prepare)})`,
      );
    } catch (err) {
      setStatus(`${file.name}: ${err instanceof Error ? err.message : err}`, true);
    }
    renderList();
  }
}

$<HTMLButtonElement>("load-sample").onclick = async () => {
  const names = ["lidar_reference.pcd", "lidar_candidate.pcd"];
  try {
    setStatus("Downloading sample…");
    const files = await Promise.all(
      names.map(async (name) => {
        const response = await fetch(`${import.meta.env.BASE_URL}samples/${name}`);
        if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
        return new File([await response.blob()], name);
      }),
    );
    await loadFiles(files);
    if (entries.size >= 2) runButton.click();
  } catch (err) {
    setStatus(`Sample: ${err instanceof Error ? err.message : err}`, true);
  }
};

$<HTMLButtonElement>("open").onclick = () => $<HTMLInputElement>("file-input").click();
$<HTMLInputElement>("file-input").onchange = (e) => {
  const input = e.target as HTMLInputElement;
  if (input.files) void loadFiles([...input.files]);
  input.value = "";
};

const overlay = $("drop-overlay");
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (!e.dataTransfer?.types.includes("Files")) return;
  dragDepth++;
  overlay.hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) overlay.hidden = true;
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  overlay.hidden = true;
  if (e.dataTransfer?.files.length) void loadFiles([...e.dataTransfer.files]);
});

// ---------------------------------------------------------------- view controls

$<HTMLButtonElement>("fit").onclick = () => viewer.fit();
const VIEWS: Record<string, [number, number, number]> = {
  top: [0, -1e-3, 1],
  front: [0, -1, 0],
  side: [1, 0, 0],
  iso: [1, -1, 1],
};
for (const button of document.querySelectorAll<HTMLButtonElement>("[data-view]")) {
  button.onclick = () => {
    const [x, y, z] = VIEWS[button.dataset.view!];
    viewer.view({ x, y, z });
  };
}
$<HTMLInputElement>("point-size").oninput = (e) =>
  viewer.setPointSize(Number((e.target as HTMLInputElement).value));
$<HTMLSelectElement>("point-budget").onchange = (e) =>
  viewer.setPointBudget(Number((e.target as HTMLSelectElement).value));

function compact(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);
}
viewer.onDrawn = (points) => {
  const total = [...entries.values()].reduce(
    (sum, e) => sum + (e.visible && !isMesh(e) ? e.cloud.count : 0),
    0,
  );
  $("drawn").textContent = total ? `Drawing ${compact(points)} of ${compact(total)} points` : "";
};
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === "f" || e.key === "F") viewer.fit();
});

// ---------------------------------------------------------------- C2C

const comparedSelect = $<HTMLSelectElement>("c2c-compared");
const referenceSelect = $<HTMLSelectElement>("c2c-reference");
const runButton = $<HTMLButtonElement>("c2c-run");

function renderC2cSelects(): void {
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

function updateRunButton(): void {
  runButton.disabled =
    !comparedSelect.value || !referenceSelect.value || comparedSelect.value === referenceSelect.value;
  const reference = entries.get(Number(referenceSelect.value));
  $("c2m-signed-row").hidden = !reference || !isMesh(reference);
}
comparedSelect.onchange = referenceSelect.onchange = updateRunButton;

runButton.onclick = async () => {
  const compared = entries.get(Number(comparedSelect.value));
  const reference = entries.get(Number(referenceSelect.value));
  if (!compared || !reference) return;
  runButton.disabled = true;
  const kind = isMesh(reference) ? "C2M" : "C2C";
  setStatus(`Computing ${kind} distance: ${compared.cloud.name} → ${reference.cloud.name}…`);
  try {
    const result = await cloudToCloud(
      compared.cloud.id,
      reference.cloud.id,
      $<HTMLInputElement>("c2m-signed").checked,
    );
    compared.c2c = { ...result, referenceName: reference.cloud.name };
    compared.mode = "c2c";
    activeC2c = compared.cloud.id;
    range = null;
    refreshColors(compared);
    renderList();
    renderC2cResult();
    renderPickPanel();
    setStatus(
      `${kind} distance computed for ${result.stats.count.toLocaleString()} points in ${Math.round(result.millis)} ms` +
        (result.workers > 1 ? ` on ${result.workers} workers` : ""),
    );
  } catch (err) {
    setStatus(`${kind} failed: ${err instanceof Error ? err.message : err}`, true);
  } finally {
    updateRunButton();
  }
};

const rampSelect = $<HTMLSelectElement>("ramp");
for (const name of Object.keys(RAMPS)) rampSelect.add(new Option(name, name));
rampSelect.onchange = () => {
  rampName = rampSelect.value as RampName;
  applyRange();
};

const minInput = $<HTMLInputElement>("range-min");
const maxInput = $<HTMLInputElement>("range-max");
minInput.onchange = maxInput.onchange = () => {
  const lo = Number(minInput.value);
  const hi = Number(maxInput.value);
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
    range = { lo, hi };
    applyRange();
  }
};
$<HTMLButtonElement>("range-reset").onclick = () => {
  range = null;
  applyRange();
};

function applyRange(): void {
  const entry = activeC2c !== null ? entries.get(activeC2c) : undefined;
  if (entry) refreshColors(entry);
  renderC2cResult();
}

function renderC2cResult(): void {
  const entry = activeC2c !== null ? entries.get(activeC2c) : undefined;
  const c2c = entry?.mode === "c2c" ? entry.c2c : undefined;
  $("c2c-result").hidden = !c2c;
  $("colorbar").hidden = !c2c;
  if (!entry || !c2c) return;

  const { stats } = c2c;
  const rows: [string, string][] = [
    ["Points", stats.count.toLocaleString()],
    ["Mean", fmt(stats.mean)],
    ["Std. dev.", fmt(stats.stdDev)],
    ["RMS", fmt(stats.rms)],
    ["Median", fmt(stats.median)],
    ["Min", fmt(stats.min)],
    ["Max", fmt(stats.max)],
  ];
  $("c2c-stats").replaceChildren(
    ...rows.map(([k, v]) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = k;
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(th, td);
      return tr;
    }),
  );

  const { lo, hi } = range ?? { lo: stats.min, hi: stats.max };
  minInput.value = fmt(lo);
  maxInput.value = fmt(hi);
  rampSelect.value = rampName;
  $("colorbar-title").textContent = `${c2c.kind === "c2m" ? (c2c.signed ? "Signed C2M" : "C2M") : "C2C"} distance · ${entry.cloud.name}`;
  $("colorbar-ramp").style.background = gradientCss(rampName);
  $("colorbar-max").textContent = fmt(hi);
  $("colorbar-mid").textContent = fmt((lo + hi) / 2);
  $("colorbar-min").textContent = fmt(lo);
}

// ---------------------------------------------------------------- ICP

const icpMoving = $<HTMLSelectElement>("icp-moving");
const icpReference = $<HTMLSelectElement>("icp-reference");
const icpButton = $<HTMLButtonElement>("icp-run");
let lastIcp: number | null = null;

function renderIcpSelects(): void {
  const ids = [...entries.values()].filter((e) => !isMesh(e)).map((e) => String(e.cloud.id));
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
  updateIcpButton();
  renderIcpResult();
}

function updateIcpButton(): void {
  icpButton.disabled = !icpMoving.value || !icpReference.value || icpMoving.value === icpReference.value;
}
icpMoving.onchange = icpReference.onchange = updateIcpButton;

/** Swap in a cloud whose points moved (and were reordered) in the worker. */
function replaceCloud(entry: Entry, cloud: LoadedCloud): void {
  const name = entry.cloud.name;
  entry.cloud = cloud;
  entry.nodes = parseNodes(cloud.lodNodes, cloud.lodGrid, cloud.shift);
  // Distances involving the moved cloud no longer describe the data.
  for (const other of entries.values()) {
    if (other.c2c && (other === entry || other.c2c.referenceName === name)) {
      other.c2c = undefined;
      if (other.mode === "c2c") other.mode = other.cloud.colors ? "rgb" : "solid";
      if (activeC2c === other.cloud.id) activeC2c = null;
      if (other !== entry) refreshColors(other);
    }
  }
  viewer.remove(cloud.id);
  viewer.add(cloud.id, cloud.positions, colorsFor(entry), entry.nodes);
  viewer.setVisible(cloud.id, entry.visible);
  forgetPoints(cloud.id);
  renderList();
  renderC2cResult();
}

/** Inverse of a row-major 4x4 rigid transform. */
function invertRigid(m: number[]): number[] {
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

interface IcpSummary {
  cloudId: number;
  rmsInitial: number;
  rmsFinal: number;
  iterations: number;
  converged: boolean;
  millis: number;
}
let icpSummary: IcpSummary | null = null;

function renderIcpResult(): void {
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
  $("icp-stats").replaceChildren(
    ...rows.map(([k, v]) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = k;
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(th, td);
      return tr;
    }),
  );
  $<HTMLTextAreaElement>("icp-matrix").value = [0, 1, 2, 3]
    .map((r) => matrix.slice(r * 4, r * 4 + 4).map((v) => v.toFixed(9)).join(" "))
    .join("\n");
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
    setStatus(`ICP failed: ${err instanceof Error ? err.message : err}`, true);
  } finally {
    updateIcpButton();
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
    setStatus(`Undo failed: ${err instanceof Error ? err.message : err}`, true);
  }
};

// ---------------------------------------------------------------- picking & measuring

interface PickedPoint {
  cloudId: number;
  index: number;
  /** Render (shifted) position, for drawing. */
  render: THREE.Vector3;
  /** Exact original coordinates. */
  exact: [number, number, number];
}

interface Measurement {
  a: PickedPoint;
  b: PickedPoint;
  label: HTMLSpanElement;
}

const PICK_COLOR = "#ffd54f";
const MEASURE_COLOR = "#4fc3f7";
let picked: PickedPoint | null = null;
let measuring = false;
let pending: PickedPoint | null = null;
const measurements: Measurement[] = [];
const measureButton = $<HTMLButtonElement>("measure");

function coord(v: number): string {
  return v.toFixed(Math.abs(v) >= 1e5 ? 3 : 4);
}

function distance(a: PickedPoint, b: PickedPoint): { d: number; delta: number[] } {
  const delta = a.exact.map((v, i) => b.exact[i] - v);
  return { d: Math.hypot(...delta), delta };
}

function refreshAnnotations(): void {
  const markers: { position: THREE.Vector3; color: string }[] = [];
  if (picked) markers.push({ position: picked.render, color: PICK_COLOR });
  for (const m of measurements) {
    markers.push({ position: m.a.render, color: MEASURE_COLOR }, { position: m.b.render, color: MEASURE_COLOR });
  }
  if (pending) markers.push({ position: pending.render, color: MEASURE_COLOR });
  viewer.setAnnotations(
    markers,
    measurements.map((m) => [m.a.render, m.b.render]),
  );
}

function renderPickPanel(): void {
  $("pick-panel").hidden = !picked;
  if (!picked) return;
  const entry = entries.get(picked.cloudId);
  const rows: [string, string][] = [
    ["Cloud", entry?.cloud.name ?? "?"],
    ["X", coord(picked.exact[0])],
    ["Y", coord(picked.exact[1])],
    ["Z", coord(picked.exact[2])],
  ];
  const rgb = entry?.cloud.colors?.subarray(picked.index * 3, picked.index * 3 + 3);
  if (rgb) rows.push(["RGB", Array.from(rgb).join(", ")]);
  if (entry?.c2c) rows.push([`C2C → ${entry.c2c.referenceName}`, fmt(entry.c2c.distances[picked.index])]);
  $("pick-info").replaceChildren(
    ...rows.map(([k, v]) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = k;
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(th, td);
      return tr;
    }),
  );
}

function renderMeasurements(): void {
  $("measure-panel").hidden = !measuring && measurements.length === 0;
  $("measure-hint").textContent = measuring
    ? pending
      ? "Click the second point. Esc cancels."
      : "Click the first point."
    : "";
  $("measure-clear").hidden = measurements.length === 0;
  $("measure-list").replaceChildren(
    ...measurements.map((m, i) => {
      const { d, delta } = distance(m.a, m.b);
      const li = document.createElement("li");
      const value = document.createElement("span");
      value.className = "distance";
      value.textContent = fmt(d);
      const remove = document.createElement("button");
      remove.className = "remove";
      remove.textContent = "✕";
      remove.title = "Remove";
      remove.onclick = () => {
        measurements.splice(i, 1);
        m.label.remove();
        refreshAnnotations();
        renderMeasurements();
      };
      const deltas = document.createElement("span");
      deltas.className = "delta";
      deltas.textContent = `ΔX ${fmt(delta[0])}  ΔY ${fmt(delta[1])}  ΔZ ${fmt(delta[2])}`;
      li.append(remove, value, deltas);
      return li;
    }),
  );
}

function setMeasuring(on: boolean): void {
  measuring = on;
  pending = null;
  measureButton.setAttribute("aria-pressed", String(on));
  $("viewport").classList.toggle("measuring", on);
  refreshAnnotations();
  renderMeasurements();
}

/** Drop picks and measurements that refer to a removed cloud. */
function forgetPoints(cloudId: number): void {
  if (picked?.cloudId === cloudId) picked = null;
  if (pending?.cloudId === cloudId) pending = null;
  for (let i = measurements.length - 1; i >= 0; i--) {
    const m = measurements[i];
    if (m.a.cloudId === cloudId || m.b.cloudId === cloudId) {
      m.label.remove();
      measurements.splice(i, 1);
    }
  }
  refreshAnnotations();
  renderPickPanel();
  renderMeasurements();
}

viewer.onClick = async (x, y) => {
  const hit = viewer.pick(x, y);
  if (!hit) {
    if (!measuring && picked) {
      picked = null;
      refreshAnnotations();
      renderPickPanel();
    }
    return;
  }
  let exact: [number, number, number];
  try {
    exact = await pointAt(hit.cloudId, hit.index);
  } catch {
    return; // the cloud was removed while we were asking
  }
  const point: PickedPoint = { cloudId: hit.cloudId, index: hit.index, render: hit.position, exact };
  if (measuring) {
    if (!pending) {
      pending = point;
    } else {
      const label = document.createElement("span");
      $("labels").append(label);
      measurements.push({ a: pending, b: point, label });
      pending = null;
      const { d } = distance(measurements.at(-1)!.a, point);
      setStatus(`Distance: ${fmt(d)}`);
    }
    renderMeasurements();
  } else {
    picked = point;
    renderPickPanel();
  }
  refreshAnnotations();
};

// Keep distance labels at the middle of their segments.
viewer.onAfterRender = () => {
  for (const m of measurements) {
    const mid = m.a.render.clone().add(m.b.render).multiplyScalar(0.5);
    const at = viewer.project(mid);
    m.label.hidden = !at;
    if (!at) continue;
    m.label.textContent = fmt(distance(m.a, m.b).d);
    m.label.style.left = `${at.x}px`;
    m.label.style.top = `${at.y}px`;
  }
};

measureButton.onclick = () => setMeasuring(!measuring);
$<HTMLButtonElement>("measure-clear").onclick = () => {
  for (const m of measurements) m.label.remove();
  measurements.length = 0;
  refreshAnnotations();
  renderMeasurements();
};
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === "m" || e.key === "M") setMeasuring(!measuring);
  if (e.key === "Escape") {
    if (pending) {
      pending = null;
      refreshAnnotations();
      renderMeasurements();
    } else if (measuring) {
      setMeasuring(false);
    } else if (picked) {
      picked = null;
      refreshAnnotations();
      renderPickPanel();
    }
  }
});

renderList();
