import * as THREE from "three";
import {
  cloudToCloud,
  computeM3c2,
  computeVolume,
  cropCloud,
  exportCloud,
  extractGround,
  filterCloud,
  estimateNormals,
  loadCloud,
  mergeClouds,
  pointAt,
  profileCloud,
  setMemoryListener,
  registerIcp,
  removeCloud,
  splitCloud,
  transformCloud,
} from "./api";
import {
  RAMPS,
  classColor,
  className,
  colorize,
  gradientCss,
  lut,
  quantile,
  type RampName,
  toRgba,
} from "./colormap";
import { type LodNode, parseNodes } from "./lod";
import { CANCELLED, type C2cOutput, type LoadedCloud, type Progress } from "./protocol";
import { decodeSession, encodeSession, nameFromUrl, parseSession, type Session } from "./session";
import { Viewer } from "./viewer";

type ColorMode = "rgb" | "solid" | "intensity" | "classification" | "c2c" | "normal" | "shade";

/** Where a cloud came from: sessions can restore file and URL clouds. */
type Origin = { kind: "file" } | { kind: "url"; url: string } | { kind: "derived" };

interface Entry {
  cloud: LoadedCloud;
  nodes: LodNode[];
  solid: [number, number, number];
  mode: ColorMode;
  visible: boolean;
  c2c?: C2cOutput & { referenceName: string };
  /** Transforms applied by ICP, newest last, for undo. */
  transforms: number[][];
  origin: Origin;
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

// ---------------------------------------------------------------- tasks

/** A cancellable long operation shown with a progress bar in the status bar. */
let task: AbortController | null = null;

function startTask(): AbortSignal {
  task = new AbortController();
  $("task").hidden = false;
  showProgress({ note: "" });
  return task.signal;
}

function showProgress(p: Progress): void {
  const bar = $("progress-bar");
  const known = p.fraction !== undefined;
  bar.parentElement!.classList.toggle("indeterminate", !known);
  bar.style.width = known ? `${Math.round(p.fraction! * 100)}%` : "";
}

/** Hide the progress bar, unless a newer task has taken it over. */
function endTask(signal: AbortSignal): void {
  if (task?.signal !== signal) return;
  task = null;
  $("task").hidden = true;
}

$<HTMLButtonElement>("task-cancel").onclick = () => task?.abort();

setMemoryListener((bytes) => {
  $("memory").textContent = bytes >= 1e9 ? `WASM ${(bytes / 1e9).toFixed(2)} GB` : `WASM ${Math.round(bytes / 1e6)} MB`;
});

// ---------------------------------------------------------------- colors

function solidColors(entry: Entry): Uint8Array {
  const out = new Uint8Array(entry.cloud.count * 4);
  const [r, g, b] = entry.solid;
  for (let i = 0; i < out.length; i += 4) {
    out[i] = r;
    out[i + 1] = g;
    out[i + 2] = b;
    out[i + 3] = 255;
  }
  return out;
}

function classificationColors(classes: Uint8Array): Uint8Array {
  const table = new Uint8Array(256 * 3);
  for (let c = 0; c < 256; c++) table.set(classColor(c), c * 3);
  const out = new Uint8Array(classes.length * 4);
  for (let i = 0; i < classes.length; i++) {
    const t = classes[i] * 3;
    out[i * 4] = table[t];
    out[i * 4 + 1] = table[t + 1];
    out[i * 4 + 2] = table[t + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

/**
 * Colors from normals: their direction as RGB, or a grey hillshade lit
 * from the north-west at 45 degrees. Points without a normal are grey.
 */
function normalColors(normals: Float32Array, shade: boolean): Uint8Array {
  const out = new Uint8Array((normals.length / 3) * 4);
  const light = [-0.5, 0.5, Math.SQRT1_2];
  for (let i = 0; i < normals.length / 3; i++) {
    const [x, y, z] = [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]];
    const o = i * 4;
    out[o + 3] = 255;
    if (x === 0 && y === 0 && z === 0) {
      out.fill(128, o, o + 3);
    } else if (shade) {
      const lit = Math.max(0, x * light[0] + y * light[1] + z * light[2]);
      out.fill(Math.round(35 + 220 * lit), o, o + 3);
    } else {
      out[o] = Math.round((x + 1) * 127.5);
      out[o + 1] = Math.round((y + 1) * 127.5);
      out[o + 2] = Math.round((z + 1) * 127.5);
    }
  }
  return out;
}

/** Interleaved rgba for the entry's color mode, with hidden classes at alpha 0. */
function colorsFor(entry: Entry): Uint8Array {
  const { cloud } = entry;
  let out: Uint8Array;
  if (entry.mode === "rgb" && cloud.colors) {
    out = toRgba(cloud.colors);
  } else if (entry.mode === "intensity" && cloud.intensity) {
    // Stretch between the 2nd and 98th percentile so a few bright returns
    // do not wash everything out.
    const lo = quantile(cloud.intensity, 0.02);
    const hi = quantile(cloud.intensity, 0.98);
    out = colorize(cloud.intensity, lo, hi > lo ? hi : lo + 1, lut("Grey"));
  } else if (entry.mode === "classification" && cloud.classification) {
    out = classificationColors(cloud.classification);
  } else if ((entry.mode === "normal" || entry.mode === "shade") && cloud.normals) {
    out = normalColors(cloud.normals, entry.mode === "shade");
  } else if (entry.mode === "c2c" && entry.c2c) {
    const { lo, hi } = range ?? { lo: entry.c2c.stats.min, hi: entry.c2c.stats.max };
    out = colorize(entry.c2c.distances, lo, hi, lut(rampName));
  } else {
    out = solidColors(entry);
  }
  if (cloud.classification && hiddenClasses.size > 0) {
    const classes = cloud.classification;
    for (let i = 0; i < classes.length; i++) if (hiddenClasses.has(classes[i])) out[i * 4 + 3] = 0;
  }
  return out;
}

/** The color mode a newly loaded cloud starts with. */
function defaultMode(cloud: LoadedCloud): ColorMode {
  if (cloud.colors) return "rgb";
  if (cloud.intensity) return "intensity";
  return "solid";
}

function refreshColors(entry: Entry): void {
  if (entry.cloud.kind === "mesh") viewer.setMeshColor(entry.cloud.id, entry.solid);
  else viewer.setColors(entry.cloud.id, colorsFor(entry));
}

const isMesh = (entry: Entry) => entry.cloud.kind === "mesh";

/** Label for a distance result, e.g. "C2M distance → part.stl". */
function distanceLabel(c2c: Entry["c2c"]): string {
  if (!c2c) return "Distance";
  if (c2c.kind === "volume") return `Height difference vs ${c2c.referenceName}`;
  if (c2c.kind === "m3c2") return `M3C2 distance from ${c2c.referenceName}`;
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
        : `${cloud.count.toLocaleString()} points${cloud.keepEvery > 1 ? ` (1 in ${cloud.keepEvery})` : ""}` +
          `${cloud.colors ? " · RGB" : ""}`;
    name.append(meta);

    const remove = document.createElement("button");
    remove.className = "remove";
    remove.textContent = "✕";
    remove.title = "Remove";
    remove.onclick = () => void removeEntry(cloud.id);

    const actions = document.createElement("span");
    actions.className = "actions";
    if (!isMesh(entry)) {
      const save = document.createElement("button");
      save.className = "icon";
      save.textContent = "⤓";
      save.title = entry.c2c ? "Save as PLY with distances" : "Save as PLY";
      save.onclick = () => void saveCloud(entry, "ply");
      actions.append(save);
    }
    actions.append(remove);

    const mode = document.createElement("select");
    mode.title = "Color by";
    const options: [ColorMode, string, boolean][] = [
      ["rgb", "RGB", cloud.colors !== null],
      ["intensity", "Intensity", cloud.intensity !== null],
      ["classification", "Classification", cloud.classification !== null],
      ["solid", "Solid color", true],
      ["c2c", distanceLabel(entry.c2c), !!entry.c2c],
      ["normal", "Normals", cloud.normals !== null],
      ["shade", "Hillshade", cloud.normals !== null],
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
    if (isMesh(entry)) li.append(visible, swatch, name, actions);
    else li.append(visible, swatch, name, actions, mode);
    list.append(li);
  }
  renderC2cSelects();
  renderIcpSelects();
  renderClasses();
  renderFilterSelect();
  renderMergeSplit();
  renderNormalsSelect();
  renderVolumeSelects();
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
        entry.mode = defaultMode(entry.cloud);
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

// ---------------------------------------------------------------- export

/** Download a cloud as PLY or CSV, including its distances if computed. */
async function saveCloud(entry: Entry, format: "ply" | "csv"): Promise<void> {
  const { cloud, c2c } = entry;
  const kind = c2c?.kind.toUpperCase();
  // M3C2 results already carry m3c2_distance / lod95 / significant attributes.
  const scalar =
    c2c && kind && c2c.kind !== "m3c2" ? { name: `${kind}_distance`, values: c2c.distances } : undefined;
  const base = cloud.name.replace(/\.[^.]+$/, "");
  const filename = `${base}${kind ? `_${kind}` : ""}.${format}`;
  setStatus(`Saving ${filename}…`);
  try {
    const bytes = await exportCloud(cloud.id, format, scalar);
    const url = URL.createObjectURL(new Blob([bytes as BlobPart]));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    // Give the browser a moment to start the download before revoking.
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    setStatus(`Saved ${filename} (${(bytes.byteLength / 1e6).toFixed(1)} MB)`);
  } catch (err) {
    setStatus(`Save failed: ${err instanceof Error ? err.message : err}`, true);
  }
}

// ---------------------------------------------------------------- loading

/** Register a loaded cloud or mesh and draw it. */
function addEntry(cloud: LoadedCloud, origin: Origin = { kind: "derived" }): Entry {
  const entry: Entry = {
    cloud,
    nodes: parseNodes(cloud.lodNodes, cloud.lodGrid, cloud.shift),
    solid: SOLID_COLORS[entries.size % SOLID_COLORS.length],
    mode: defaultMode(cloud),
    visible: true,
    transforms: [],
    origin,
  };
  entries.set(cloud.id, entry);
  // On a phone, get the sheet out of the way once there is something to see.
  if (entries.size === 1 && narrow.matches) setPanelsOpen(false);
  if (cloud.kind === "mesh") viewer.addMesh(cloud.id, cloud.positions, cloud.indices!, entry.solid);
  else viewer.add(cloud.id, cloud.positions, colorsFor(entry), entry.nodes);
  return entry;
}

/**
 * Load point clouds and meshes; session files (.json) among them are applied
 * once the others are in. `origins` tells where each file came from.
 */
async function loadFiles(files: File[], origins?: Origin[]): Promise<void> {
  const sessions: File[] = [];
  const signal = startTask();
  for (const [i, file] of files.entries()) {
    if (signal.aborted) break;
    if (/\.json$/i.test(file.name)) {
      sessions.push(file);
      continue;
    }
    const mb = (file.size / 1e6).toFixed(file.size >= 1e7 ? 0 : 1);
    setStatus(`Loading ${file.name} (${mb} MB): reading…`);
    const start = performance.now();
    try {
      const maxPoints = Number($<HTMLSelectElement>("max-points").value) || Number.POSITIVE_INFINITY;
      const cloud = await loadCloud(
        file,
        maxPoints,
        (p) => {
          showProgress(p);
          setStatus(`Loading ${file.name} (${mb} MB): ${p.note}…`);
        },
        signal,
      );
      addEntry(cloud, origins?.[i] ?? { kind: "file" });
      if (entries.size === 1) viewer.fit();
      const [sx, sy, sz] = cloud.shift;
      $("shift").textContent =
        sx || sy || sz ? `Global shift: (${-sx}, ${-sy}, ${-sz})` : "";
      const { parse, index, prepare, workers } = cloud.timings;
      const s = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
      const size =
        cloud.kind === "mesh"
          ? `${cloud.triangles.toLocaleString()} triangles`
          : cloud.keepEvery > 1
            ? `${cloud.count.toLocaleString()} of ${cloud.filePoints.toLocaleString()} points (1 in ${cloud.keepEvery})`
            : `${cloud.count.toLocaleString()} points`;
      setStatus(
        `Loaded ${file.name}: ${size} in ${s(performance.now() - start)} ` +
          `(read ${s(parse)} · index ${s(index)}` +
          `${workers && workers > 1 ? ` on ${workers} workers` : ""} · prepare ${s(prepare)})`,
      );
    } catch (err) {
      if (err instanceof Error && err.message === CANCELLED) {
        const rest = files.length - i - 1;
        setStatus(`Stopped loading ${file.name}${rest > 0 ? ` (and ${rest} more)` : ""}`);
      } else {
        setStatus(`${file.name}: ${err instanceof Error ? err.message : err}`, true);
      }
    }
    renderList();
  }
  endTask(signal);
  if (signal.aborted) return;
  for (const file of sessions) {
    try {
      await applySession(parseSession(JSON.parse(await file.text())));
    } catch (err) {
      setStatus(`${file.name}: ${err instanceof Error ? err.message : err}`, true);
    }
  }
  // Files a pending session was waiting for.
  if (sessions.length === 0 && pendingSession && !applyingSession) await restorePending();
}

/** Download clouds from URLs (the server must allow cross-origin requests) and load them. */
async function loadUrls(urls: string[]): Promise<void> {
  const files: File[] = [];
  const origins: Origin[] = [];
  const failed: string[] = [];
  const signal = startTask();
  for (const raw of urls) {
    const url = new URL(raw, location.href).href;
    const name = nameFromUrl(url);
    setStatus(`Downloading ${name}…`);
    try {
      const response = await fetch(url, { signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      files.push(new File([await readWithProgress(response, name)], name));
      origins.push({ kind: "url", url });
    } catch (err) {
      if (signal.aborted) {
        endTask(signal);
        setStatus(`Stopped downloading ${name}`);
        return;
      }
      failed.push(`${name} (${err instanceof Error ? err.message : err})`);
    }
  }
  endTask(signal);
  await loadFiles(files, origins);
  if (failed.length) {
    setStatus(`Could not download ${failed.join(", ")}; the server must allow cross-origin requests`, true);
  }
}

/** The body of a download, reporting progress when its size is known. */
async function readWithProgress(response: Response, name: string): Promise<Blob> {
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return response.blob();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    showProgress({ note: "", fraction: Math.min(1, received / total) });
    setStatus(`Downloading ${name}: ${Math.round((received / total) * 100)}% of ${(total / 1e6).toFixed(1)} MB…`);
  }
  return new Blob(chunks as BlobPart[]);
}

// ---------------------------------------------------------------- demos

/** Set a select and let its listeners react, as if the user had picked it. */
function choose(id: string, value: string): void {
  const select = $<HTMLSelectElement>(id);
  select.value = value;
  select.dispatchEvent(new Event("change"));
}

const idOf = (name: string) => String([...entries.values()].find((e) => e.cloud.name === name)?.cloud.id ?? "");

function fill(values: Record<string, string>): void {
  for (const [id, value] of Object.entries(values)) $<HTMLInputElement>(id).value = value;
}

/** Hide input clouds so a demo shows its result alone. */
function hide(...names: string[]): void {
  for (const e of entries.values()) {
    if (!names.includes(e.cloud.name)) continue;
    e.visible = false;
    viewer.setVisible(e.cloud.id, false);
  }
  renderList();
}

/** Sample data and the analysis each demo runs on it (see `scripts/make-samples.mjs`). */
const DEMOS: Record<string, { files: string[]; run: () => void }> = {
  c2c: {
    files: ["lidar_reference.pcd", "lidar_candidate.pcd"],
    run: () => runButton.click(),
  },
  volume: {
    files: ["stockpile_before.ply", "stockpile_after.ply"],
    run: () => {
      choose("volume-before", idOf("stockpile_before.ply"));
      choose("volume-after", idOf("stockpile_after.ply"));
      fill({ "volume-cell": "0.4" });
      $<HTMLButtonElement>("volume-run").click();
      hide("stockpile_before.ply", "stockpile_after.ply");
    },
  },
  ground: {
    files: ["town.ply"],
    run: () => {
      choose("filter-cloud", idOf("town.ply"));
      choose("filter-op", "ground");
      fill({ "csf-resolution": "1", "csf-threshold": "0.3" });
      choose("csf-rigidness", "relief");
      choose("csf-output", "classified");
      $<HTMLButtonElement>("filter-run").click();
    },
  },
  m3c2: {
    files: ["slope_before.ply", "slope_after.ply"],
    run: () => {
      choose("distance-method", "m3c2");
      choose("c2c-compared", idOf("slope_after.ply"));
      choose("c2c-reference", idOf("slope_before.ply"));
      fill({ "m3c2-normal": "1", "m3c2-projection": "0.5", "m3c2-depth": "2", "m3c2-core": "0.3" });
      runButton.click();
      hide("slope_before.ply");
    },
  },
};

/** Load a demo's sample files, then run its analysis. */
async function runDemo(name: string): Promise<void> {
  const demo = DEMOS[name];
  if (!demo) {
    setStatus(`Unknown demo "${name}" (try ${Object.keys(DEMOS).join(", ")})`, true);
    return;
  }
  const missing = demo.files.filter((f) => !idOf(f));
  await loadUrls(missing.map((f) => `${import.meta.env.BASE_URL}samples/${f}`));
  if (!demo.files.every((f) => idOf(f))) return;
  if (name !== "c2c") {
    // The synthetic samples are dense grids: bigger points close the gaps.
    $<HTMLInputElement>("point-size").value = "5";
    viewer.setPointSize(5);
  }
  demo.run();
}

$<HTMLButtonElement>("load-sample").onclick = () => void runDemo("c2c");
for (const button of document.querySelectorAll<HTMLButtonElement>("[data-demo]")) {
  button.onclick = () => void runDemo(button.dataset.demo!);
}

$<HTMLButtonElement>("url-open").onclick = () => {
  const url = $<HTMLInputElement>("url-input").value.trim();
  if (url) void loadUrls([url]);
};

$<HTMLButtonElement>("open").onclick = () => $<HTMLInputElement>("file-input").click();
$<HTMLButtonElement>("session-open").onclick = () => $<HTMLInputElement>("file-input").click();
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
const edlToggle = $<HTMLInputElement>("edl");
const edlStrength = $<HTMLInputElement>("edl-strength");
edlToggle.onchange = edlStrength.oninput = () => {
  viewer.setEdl(edlToggle.checked, Number(edlStrength.value));
  edlStrength.disabled = !edlToggle.checked;
};
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
  const reference = entries.get(Number(referenceSelect.value));
  const m3c2 = $<HTMLSelectElement>("distance-method").value === "m3c2";
  runButton.disabled =
    !comparedSelect.value ||
    !referenceSelect.value ||
    comparedSelect.value === referenceSelect.value ||
    (m3c2 && !!reference && isMesh(reference));
  $("c2m-signed-row").hidden = m3c2 || !reference || !isMesh(reference);
  $("m3c2-options").hidden = !m3c2;
}
$<HTMLSelectElement>("distance-method").onchange = updateRunButton;
comparedSelect.onchange = referenceSelect.onchange = updateRunButton;

/** Summary statistics over the finite values only. */
function finiteStats(values: Float32Array): C2cOutput["stats"] {
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
  entry.c2c = {
    kind: "m3c2",
    signed: true,
    distances: out.distance,
    stats,
    millis: out.millis,
    workers: 1,
    referenceName: reference.cloud.name,
  };
  entry.mode = "c2c";
  activeC2c = entry.cloud.id;
  const m = Math.max(Math.abs(stats.min), Math.abs(stats.max)) || 1;
  range = { lo: -m, hi: m };
  rampName = "Blue > White > Red";
  refreshColors(entry);
  compared.visible = false;
  viewer.setVisible(compared.cloud.id, false);
  renderList();
  renderC2cResult();
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
  if ($<HTMLSelectElement>("distance-method").value === "m3c2") {
    runButton.disabled = true;
    try {
      await runM3c2(compared, reference);
    } catch (err) {
      setStatus(`M3C2 failed: ${err instanceof Error ? err.message : err}`, true);
    } finally {
      updateRunButton();
    }
    return;
  }
  runButton.disabled = true;
  try {
    await runNearest(compared, reference, $<HTMLInputElement>("c2m-signed").checked);
  } finally {
    updateRunButton();
  }
};

/** C2C (or C2M against a mesh) from `compared` to `reference`, shown on `compared`. */
async function runNearest(compared: Entry, reference: Entry, signed: boolean): Promise<void> {
  const kind = isMesh(reference) ? "C2M" : "C2C";
  setStatus(`Computing ${kind} distance: ${compared.cloud.name} → ${reference.cloud.name}…`);
  try {
    const result = await cloudToCloud(compared.cloud.id, reference.cloud.id, signed);
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
  }
}

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
for (const format of ["ply", "csv"] as const) {
  $<HTMLButtonElement>(`export-${format}`).onclick = () => {
    const entry = activeC2c !== null ? entries.get(activeC2c) : undefined;
    if (entry) void saveCloud(entry, format);
  };
}

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
  $("colorbar-title").textContent =
    c2c.kind === "volume"
      ? `Height difference (after − before) · ${entry.cloud.name}`
      : c2c.kind === "m3c2"
        ? `M3C2 distance · ${entry.cloud.name}`
      : `${c2c.kind === "c2m" ? (c2c.signed ? "Signed C2M" : "C2M") : "C2C"} distance · ${entry.cloud.name}`;
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
      if (other.mode === "c2c") other.mode = defaultMode(other.cloud);
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

// ---------------------------------------------------------------- classes

/** ASPRS class codes currently hidden in every cloud. */
const hiddenClasses = new Set<number>();

function renderClasses(): void {
  const counts = new Map<number, number>();
  for (const entry of entries.values()) {
    const classes = entry.cloud.classification;
    if (!classes) continue;
    const local = new Uint32Array(256);
    for (let i = 0; i < classes.length; i++) local[classes[i]]++;
    local.forEach((n, code) => n && counts.set(code, (counts.get(code) ?? 0) + n));
  }
  $("class-panel").hidden = counts.size === 0;
  $("class-list").replaceChildren(
    ...[...counts.entries()]
      .sort(([a], [b]) => a - b)
      .map(([code, n]) => {
        const li = document.createElement("li");
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = !hiddenClasses.has(code);
        box.onchange = () => {
          if (box.checked) hiddenClasses.delete(code);
          else hiddenClasses.add(code);
          for (const entry of entries.values()) if (entry.cloud.classification) refreshColors(entry);
        };
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = `rgb(${classColor(code).join(" ")})`;
        const label = document.createElement("span");
        label.textContent = `${code} · ${className(code)}`;
        const count = document.createElement("span");
        count.className = "meta";
        count.textContent = n.toLocaleString();
        const row = document.createElement("label");
        row.append(box, swatch, label, count);
        li.append(row);
        return li;
      }),
  );
}

for (const [id, show] of [
  ["class-all", true],
  ["class-none", false],
] as const) {
  $<HTMLButtonElement>(id).onclick = () => {
    hiddenClasses.clear();
    if (!show) for (let c = 0; c < 256; c++) hiddenClasses.add(c);
    for (const entry of entries.values()) if (entry.cloud.classification) refreshColors(entry);
    renderClasses();
  };
}

// ---------------------------------------------------------------- volume

const volumeBefore = $<HTMLSelectElement>("volume-before");
const volumeAfter = $<HTMLSelectElement>("volume-after");
const volumeCell = $<HTMLInputElement>("volume-cell");
const volumeRun = $<HTMLButtonElement>("volume-run");
const CONSTANT = "constant";

function renderVolumeSelects(): void {
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
  updateVolumeForm();
}

function updateVolumeForm(): void {
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
    const raw = Math.max(b[3] - b[0], b[4] - b[1]) / 200 || 1;
    const pow = 10 ** Math.floor(Math.log10(raw));
    volumeCell.value = String(Number((Math.ceil(raw / pow) * pow).toPrecision(2)));
    volumeCell.dataset.for = String(source.cloud.id);
  }
}
volumeBefore.onchange = volumeAfter.onchange = () => {
  volumeBefore.dataset.picked = "1";
  updateVolumeForm();
};

function volumeSide(select: HTMLSelectElement, z: HTMLInputElement): { id: number } | { z: number } {
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
      before: volumeSide(volumeBefore, $<HTMLInputElement>("volume-before-z")),
      after: volumeSide(volumeAfter, $<HTMLInputElement>("volume-after-z")),
      cell,
      height: $<HTMLSelectElement>("volume-height").value as "mean" | "min" | "max",
      fillEmpty: $<HTMLInputElement>("volume-fill").checked,
    });
    const coverage = out.totalCells ? (100 * out.matchedCells) / out.totalCells : 0;
    const rows: [string, string][] = [
      ["Fill (added)", unit(out.added, 3)],
      ["Cut (removed)", unit(out.removed, 3)],
      ["Net", unit(out.added - out.removed, 3)],
      ["Fill area", unit(out.addedArea, 2)],
      ["Cut area", unit(out.removedArea, 2)],
      ["Cells compared", `${out.matchedCells.toLocaleString()} of ${out.totalCells.toLocaleString()} (${coverage.toFixed(1)} %)`],
      ["Cell size", fmt(out.cell)],
    ];
    $("volume-result").hidden = false;
    $("volume-stats").replaceChildren(
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
    if (out.cells && out.difference) {
      const entry = addEntry(out.cells);
      let lo = Number.POSITIVE_INFINITY;
      let hi = Number.NEGATIVE_INFINITY;
      let sum = 0;
      let sum2 = 0;
      for (const d of out.difference) {
        lo = Math.min(lo, d);
        hi = Math.max(hi, d);
        sum += d;
        sum2 += d * d;
      }
      const n = out.difference.length;
      const mean = sum / n;
      entry.c2c = {
        kind: "volume",
        signed: true,
        distances: out.difference,
        stats: {
          count: n,
          min: lo,
          max: hi,
          mean,
          rms: Math.sqrt(sum2 / n),
          stdDev: Math.sqrt(Math.max(0, sum2 / n - mean * mean)),
          median: quantile(out.difference, 0.5),
        },
        millis: out.millis,
        workers: 1,
        referenceName: volumeBefore.selectedOptions[0]?.text ?? "before",
      };
      entry.mode = "c2c";
      activeC2c = entry.cloud.id;
      // Symmetric range on a diverging ramp: blue = cut, red = fill.
      const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
      range = { lo: -m, hi: m };
      rampName = "Blue > White > Red";
      refreshColors(entry);
      renderList();
      renderC2cResult();
    }
    setStatus(
      `Volume: fill ${unit(out.added, 3)}, cut ${unit(out.removed, 3)}, net ${unit(out.added - out.removed, 3)} ` +
        `(${Math.round(out.millis)} ms)`,
    );
  } catch (err) {
    setStatus(`Volume failed: ${err instanceof Error ? err.message : err}`, true);
  } finally {
    updateVolumeForm();
  }
};

// ---------------------------------------------------------------- filters

const filterCloudSelect = $<HTMLSelectElement>("filter-cloud");
const filterOp = $<HTMLSelectElement>("filter-op");
const filterRun = $<HTMLButtonElement>("filter-run");
const voxelInput = $<HTMLInputElement>("filter-voxel");

function renderFilterSelect(): void {
  const clouds = [...entries.values()].filter((e) => !isMesh(e));
  const previous = filterCloudSelect.value;
  filterCloudSelect.replaceChildren(...clouds.map((e) => new Option(e.cloud.name, String(e.cloud.id))));
  filterCloudSelect.value = clouds.some((e) => String(e.cloud.id) === previous)
    ? previous
    : String(clouds.at(-1)?.cloud.id ?? "");
  filterRun.disabled = clouds.length === 0;
  suggestVoxel();
}

/** Default voxel: about 1/200 of the cloud's largest extent, rounded. */
function suggestVoxel(): void {
  const entry = entries.get(Number(filterCloudSelect.value));
  if (!entry || voxelInput.dataset.cloud === filterCloudSelect.value) return;
  const b = entry.cloud.bounds;
  const extent = Math.max(b[3] - b[0], b[4] - b[1], b[5] - b[2]);
  const raw = extent / 200;
  const pow = 10 ** Math.floor(Math.log10(raw || 1));
  voxelInput.value = String(Number((Math.ceil(raw / pow) * pow).toPrecision(2)));
  voxelInput.dataset.cloud = filterCloudSelect.value;
}

filterCloudSelect.onchange = suggestVoxel;
filterOp.onchange = () => {
  for (const group of document.querySelectorAll<HTMLElement>("#filter-panel [data-op]")) {
    group.hidden = group.dataset.op !== filterOp.value;
  }
};

filterRun.onclick = async () => {
  const entry = entries.get(Number(filterCloudSelect.value));
  if (!entry) return;
  if (filterOp.value === "ground") {
    await runGround(entry);
    return;
  }
  const op = filterOp.value as "voxel" | "random" | "sor";
  let a = 0;
  let b = 0;
  if (op === "voxel") a = Number(voxelInput.value);
  if (op === "random") a = Math.round((entry.cloud.count * Number($<HTMLInputElement>("filter-percent").value)) / 100);
  if (op === "sor") {
    a = Number($<HTMLInputElement>("filter-k").value);
    b = Number($<HTMLInputElement>("filter-ratio").value);
  }
  if (!(a > 0)) {
    setStatus("Enter a positive value", true);
    return;
  }
  filterRun.disabled = true;
  setStatus(`Filtering ${entry.cloud.name}…`);
  try {
    const cloud = await filterCloud(entry.cloud.id, op, a, b);
    addEntry(cloud);
    entry.visible = false;
    viewer.setVisible(entry.cloud.id, false);
    renderList();
    const removed = entry.cloud.count - cloud.count;
    setStatus(
      `${cloud.name}: kept ${cloud.count.toLocaleString()} of ${entry.cloud.count.toLocaleString()} points ` +
        `(${removed.toLocaleString()} removed) in ${Math.round(cloud.timings.index)} ms`,
    );
  } catch (err) {
    setStatus(`Filter failed: ${err instanceof Error ? err.message : err}`, true);
  } finally {
    filterRun.disabled = false;
  }
};

async function runGround(entry: Entry): Promise<void> {
  const clothResolution = Number($<HTMLInputElement>("csf-resolution").value);
  const classThreshold = Number($<HTMLInputElement>("csf-threshold").value);
  if (!(clothResolution > 0) || !(classThreshold > 0)) {
    setStatus("Enter a positive cloth resolution and threshold", true);
    return;
  }
  const output = $<HTMLSelectElement>("csf-output").value as "classified" | "ground" | "objects";
  filterRun.disabled = true;
  setStatus(`Extracting ground from ${entry.cloud.name}…`);
  try {
    const cloud = await extractGround({
      id: entry.cloud.id,
      clothResolution,
      classThreshold,
      rigidness: $<HTMLSelectElement>("csf-rigidness").value as "flat" | "relief" | "steep",
      output,
    });
    const added = addEntry(cloud);
    if (output === "classified") {
      added.mode = "classification";
      refreshColors(added);
    }
    entry.visible = false;
    viewer.setVisible(entry.cloud.id, false);
    renderList();
    let detail = `${cloud.count.toLocaleString()} points`;
    if (output === "classified" && cloud.classification) {
      const ground = cloud.classification.reduce((n, c) => n + (c === 2 ? 1 : 0), 0);
      detail = `${ground.toLocaleString()} of ${cloud.count.toLocaleString()} points are ground`;
    }
    setStatus(`${cloud.name}: ${detail} (${Math.round(cloud.timings.index)} ms)`);
  } catch (err) {
    setStatus(`Ground extraction failed: ${err instanceof Error ? err.message : err}`, true);
  } finally {
    filterRun.disabled = false;
  }
}

// ---------------------------------------------------------------- merge / split

const mergeRun = $<HTMLButtonElement>("merge-run");
const splitCloudSelect = $<HTMLSelectElement>("split-cloud");
const splitBy = $<HTMLSelectElement>("split-by");
const splitRun = $<HTMLButtonElement>("split-run");

function renderMergeSplit(): void {
  const clouds = [...entries.values()].filter((e) => !isMesh(e));
  mergeRun.disabled = clouds.filter((e) => e.visible).length < 2;
  const splittable = clouds.filter((e) => e.cloud.classification || e.cloud.sources);
  const previous = splitCloudSelect.value;
  splitCloudSelect.replaceChildren(...splittable.map((e) => new Option(e.cloud.name, String(e.cloud.id))));
  splitCloudSelect.value = splittable.some((e) => String(e.cloud.id) === previous)
    ? previous
    : String(splittable.at(-1)?.cloud.id ?? "");
  updateSplit();
}

function updateSplit(): void {
  const entry = entries.get(Number(splitCloudSelect.value));
  const can = { classification: !!entry?.cloud.classification, source: !!entry?.cloud.sources };
  for (const option of splitBy.options) option.disabled = !can[option.value as keyof typeof can];
  if (splitBy.selectedOptions[0]?.disabled) {
    splitBy.value = [...splitBy.options].find((o) => !o.disabled)?.value ?? splitBy.value;
  }
  splitRun.disabled = !entry || !can[splitBy.value as keyof typeof can];
}
splitCloudSelect.onchange = splitBy.onchange = updateSplit;

mergeRun.onclick = async () => {
  const sources = [...entries.values()].filter((e) => e.visible && !isMesh(e));
  if (sources.length < 2) return;
  mergeRun.disabled = true;
  setStatus(`Merging ${sources.length} clouds…`);
  try {
    const cloud = await mergeClouds(
      sources.map((e) => e.cloud.id),
      sources.map((e) => e.solid),
    );
    addEntry(cloud);
    for (const e of sources) {
      e.visible = false;
      viewer.setVisible(e.cloud.id, false);
    }
    renderList();
    setStatus(
      `${cloud.name}: ${cloud.count.toLocaleString()} points from ${sources.map((e) => e.cloud.name).join(", ")} ` +
        `(${Math.round(cloud.timings.index)} ms)`,
    );
  } catch (err) {
    setStatus(`Merge failed: ${err instanceof Error ? err.message : err}`, true);
  } finally {
    renderMergeSplit();
  }
};

splitRun.onclick = async () => {
  const entry = entries.get(Number(splitCloudSelect.value));
  if (!entry) return;
  const by = splitBy.value as "classification" | "source";
  splitRun.disabled = true;
  setStatus(`Splitting ${entry.cloud.name} by ${by}…`);
  try {
    const parts = await splitCloud(entry.cloud.id, by);
    for (const cloud of parts) {
      const added = addEntry(cloud);
      if (by === "classification") {
        const code = cloud.classification?.[0] ?? 0;
        added.solid = classColor(code);
      }
      refreshColors(added);
    }
    entry.visible = false;
    viewer.setVisible(entry.cloud.id, false);
    renderList();
    const names = parts.map((c) =>
      by === "classification" ? `${className(c.classification?.[0] ?? 0)} (${c.count.toLocaleString()})` : c.name,
    );
    setStatus(`Split ${entry.cloud.name} into ${parts.length} clouds: ${names.join(", ")}`);
  } catch (err) {
    setStatus(`Split failed: ${err instanceof Error ? err.message : err}`, true);
  } finally {
    updateSplit();
  }
};

// ---------------------------------------------------------------- normals

const normalsCloud = $<HTMLSelectElement>("normals-cloud");
const normalsRun = $<HTMLButtonElement>("normals-run");

function renderNormalsSelect(): void {
  const clouds = [...entries.values()].filter((e) => !isMesh(e));
  const previous = normalsCloud.value;
  normalsCloud.replaceChildren(...clouds.map((e) => new Option(e.cloud.name, String(e.cloud.id))));
  normalsCloud.value = clouds.some((e) => String(e.cloud.id) === previous)
    ? previous
    : String(clouds.at(-1)?.cloud.id ?? "");
  normalsRun.disabled = clouds.length === 0;
}

normalsRun.onclick = async () => {
  const entry = entries.get(Number(normalsCloud.value));
  if (!entry) return;
  normalsRun.disabled = true;
  setStatus(`Estimating normals of ${entry.cloud.name}…`);
  const start = performance.now();
  try {
    const k = Math.max(3, Number($<HTMLInputElement>("normals-k").value) || 12);
    const orientation = $<HTMLSelectElement>("normals-orient").value as "up" | "outward";
    entry.cloud.normals = await estimateNormals(entry.cloud.id, k, orientation);
    entry.mode = "shade";
    refreshColors(entry);
    renderList();
    setStatus(
      `Normals of ${entry.cloud.count.toLocaleString()} points in ${Math.round(performance.now() - start)} ms ` +
        `(${k} neighbours, facing ${orientation === "up" ? "up" : "away from the centre"})`,
    );
  } catch (err) {
    setStatus(`Normals failed: ${err instanceof Error ? err.message : err}`, true);
  } finally {
    normalsRun.disabled = false;
  }
};

// ---------------------------------------------------------------- clipping box

const clipEnabled = $<HTMLInputElement>("clip-enabled");
/** Box the sliders span (render coordinates), captured when clipping starts. */
let clipExtent = new THREE.Box3();
const SLIDER_MAX = 1000;
const clipRows = [...document.querySelectorAll<HTMLDivElement>(".clip-axis")];
const slider = (axis: number, end: "min" | "max") =>
  clipRows[axis].querySelector<HTMLInputElement>(`input[data-end="${end}"]`)!;

function globalShift(): [number, number, number] {
  return [...entries.values()][0]?.cloud.shift ?? [0, 0, 0];
}

/** The clipping box described by the sliders, in render coordinates. */
function clipBoxFromSliders(): THREE.Box3 {
  const min = new THREE.Vector3();
  const max = new THREE.Vector3();
  for (let axis = 0; axis < 3; axis++) {
    let lo = Number(slider(axis, "min").value) / SLIDER_MAX;
    let hi = Number(slider(axis, "max").value) / SLIDER_MAX;
    if (lo > hi) [lo, hi] = [hi, lo];
    const a = clipExtent.min.getComponent(axis);
    const size = clipExtent.max.getComponent(axis) - a;
    min.setComponent(axis, a + lo * size);
    max.setComponent(axis, a + hi * size);
  }
  return new THREE.Box3(min, max);
}

function applyClip(): void {
  if (!clipEnabled.checked) {
    viewer.setClipBox(null);
    return;
  }
  const box = clipBoxFromSliders();
  viewer.setClipBox(box);
  const shift = globalShift();
  clipRows.forEach((row, axis) => {
    const lo = box.min.getComponent(axis) + shift[axis];
    const hi = box.max.getComponent(axis) + shift[axis];
    row.querySelector(".clip-values")!.textContent = `${fmt(lo)} … ${fmt(hi)}`;
  });
}

function resetClip(): void {
  clipExtent = viewer.contentBounds();
  if (clipExtent.isEmpty()) clipExtent.set(new THREE.Vector3(), new THREE.Vector3(1, 1, 1));
  // A little margin so points on the bounds are not clipped by rounding.
  clipExtent.expandByScalar(Math.max(1e-6, clipExtent.getSize(new THREE.Vector3()).length() * 1e-6));
  for (let axis = 0; axis < 3; axis++) {
    slider(axis, "min").value = "0";
    slider(axis, "max").value = String(SLIDER_MAX);
  }
  applyClip();
}

clipEnabled.onchange = () => {
  $("clip-controls").hidden = !clipEnabled.checked;
  if (clipEnabled.checked) resetClip();
  else applyClip();
};
for (const input of document.querySelectorAll<HTMLInputElement>(".clip-axis input")) {
  input.oninput = applyClip;
}
$<HTMLButtonElement>("clip-reset").onclick = resetClip;

/** A thin slab across `axis` through the middle of the current box. */
for (const button of document.querySelectorAll<HTMLButtonElement>("[data-slice]")) {
  button.onclick = () => {
    const axis = Number(button.dataset.slice);
    for (let other = 0; other < 3; other++) {
      slider(other, "min").value = "0";
      slider(other, "max").value = String(SLIDER_MAX);
    }
    const half = 10; // 2% of the extent
    slider(axis, "min").value = String(SLIDER_MAX / 2 - half);
    slider(axis, "max").value = String(SLIDER_MAX / 2 + half);
    applyClip();
    // Look straight at the section.
    const direction = [0, 0, 0];
    direction[axis] = 1;
    viewer.view({ x: direction[0], y: direction[1] - (axis === 2 ? 1e-3 : 0), z: direction[2] });
    viewer.fit();
  };
}

$<HTMLButtonElement>("clip-crop").onclick = async () => {
  const sources = [...entries.values()].filter((e) => e.visible && !isMesh(e));
  if (!clipEnabled.checked || sources.length === 0) return;
  const box = clipBoxFromSliders();
  const shift = globalShift();
  const min = [0, 1, 2].map((a) => box.min.getComponent(a) + shift[a]) as [number, number, number];
  const max = [0, 1, 2].map((a) => box.max.getComponent(a) + shift[a]) as [number, number, number];
  const created: string[] = [];
  for (const source of sources) {
    try {
      const cloud = await cropCloud(source.cloud.id, min, max, true);
      addEntry(cloud);
      created.push(`${cloud.name} (${cloud.count.toLocaleString()} points)`);
      source.visible = false;
      viewer.setVisible(source.cloud.id, false);
    } catch (err) {
      created.push(`${source.cloud.name}: ${err instanceof Error ? err.message : err}`);
    }
  }
  clipEnabled.checked = false;
  $("clip-controls").hidden = true;
  applyClip();
  renderList();
  viewer.fit();
  setStatus(`Cropped: ${created.join(", ")}`);
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
  if (entry?.c2c) rows.push([distanceLabel(entry.c2c), fmt(entry.c2c.distances[picked.index])]);
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
    const [x, y, z] = point.exact.map(coord);
    setStatus(`Picked ${entries.get(point.cloudId)?.cloud.name ?? "point"}: ${x}, ${y}, ${z}`);
  }
  refreshAnnotations();
};

// Double click / double tap: orbit around (and centre) the point under it.
viewer.onDoubleClick = (x, y) => {
  const hit = viewer.pick(x, y);
  if (hit) viewer.centerOn(hit.position);
};

// ---------------------------------------------------------------- narrow screens

// Below 720 px the sidebar is a bottom sheet toggled from the toolbar.
const narrow = window.matchMedia("(max-width: 720px)");
const panelsButton = $<HTMLButtonElement>("panels");
function setPanelsOpen(open: boolean): void {
  document.body.classList.toggle("panels-open", open);
  panelsButton.setAttribute("aria-pressed", String(open));
}
panelsButton.onclick = () => setPanelsOpen(!document.body.classList.contains("panels-open"));
// Start with the sheet open so the Open / sample hints are visible.
setPanelsOpen(narrow.matches);

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

// ---------------------------------------------------------------- profile

/** Profile polyline in original XY coordinates. */
let profileLine: [number, number][] = [];
/** Render-space height the line is drawn (and clicked) at. */
let profileZ = 0;
let profileDrawing = false;
interface ProfileSeries {
  name: string;
  color: string;
  along: Float64Array;
  positions: Float64Array;
  total: number;
}
let profileSeries: ProfileSeries[] = [];
const PROFILE_MAX_POINTS = 200_000;
const profileWidth = $<HTMLInputElement>("profile-width");
const profileDraw = $<HTMLButtonElement>("profile-draw");
const profilePlot = $("profile-plot");
const profileCanvas = profilePlot.querySelector("canvas")!;

const profileHalfWidth = () => Math.max(0, Number(profileWidth.value) || 0) / 2;

function profileLength(): number {
  let length = 0;
  for (let i = 1; i < profileLine.length; i++) {
    length += Math.hypot(profileLine[i][0] - profileLine[i - 1][0], profileLine[i][1] - profileLine[i - 1][1]);
  }
  return length;
}

/** `v` rounded to one significant digit, e.g. 0.37 -> 0.4. */
function roundNicely(v: number): number {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return Math.round(v / p) * p;
}

function drawProfileLine(): void {
  const shift = globalShift();
  viewer.setProfile(
    profileLine.map(([x, y]) => new THREE.Vector3(x - shift[0], y - shift[1], profileZ)),
    profileHalfWidth(),
  );
}

function renderProfileHint(): void {
  $("profile-hint").textContent = profileDrawing
    ? "Click points on the view; double-click or Enter to finish, Esc to cancel."
    : profileLine.length >= 2
      ? `${profileLine.length} vertices, ${fmt(profileLength())} long.`
      : "Draw a line across the clouds; looking from above works best.";
}

function setProfileDrawing(on: boolean): void {
  profileDrawing = on;
  profileDraw.setAttribute("aria-pressed", String(on));
  profileDraw.textContent = on ? "Finish line" : "Draw line";
  $("viewport").classList.toggle("measuring", on || measuring);
  renderProfileHint();
}

function clearProfile(): void {
  profileLine = [];
  profileSeries = [];
  setProfileDrawing(false);
  drawProfileLine();
  renderProfilePlot();
}

profileDraw.onclick = () => {
  if (profileDrawing) {
    void finishProfile();
    return;
  }
  if (measuring) setMeasuring(false);
  profileLine = [];
  profileSeries = [];
  renderProfilePlot();
  profileZ = viewer.getCamera().target.z;
  if (!(Number(profileWidth.value) > 0)) {
    const size = viewer.contentBounds().getSize(new THREE.Vector3());
    profileWidth.value = String(roundNicely(Math.hypot(size.x, size.y) / 100));
  }
  drawProfileLine();
  setProfileDrawing(true);
};
$<HTMLButtonElement>("profile-clear").onclick = clearProfile;
$<HTMLButtonElement>("profile-close").onclick = clearProfile;
profileWidth.onchange = () => {
  drawProfileLine();
  if (!profileDrawing && profileLine.length >= 2) void computeProfile();
};

async function finishProfile(): Promise<void> {
  setProfileDrawing(false);
  if (profileLine.length < 2) {
    clearProfile();
    return;
  }
  await computeProfile();
}

/** Cut every visible cloud along the line and plot the result. */
async function computeProfile(): Promise<void> {
  const halfWidth = profileHalfWidth();
  const sources = [...entries.values()].filter((e) => e.visible && !isMesh(e));
  if (profileLine.length < 2 || sources.length === 0) return;
  if (!(halfWidth > 0)) {
    setStatus("Profile: enter a positive width", true);
    return;
  }
  setStatus("Computing the profile…");
  try {
    const series: ProfileSeries[] = [];
    for (const entry of sources) {
      const out = await profileCloud({
        id: entry.cloud.id,
        line: profileLine.flat(),
        halfWidth,
        maxPoints: PROFILE_MAX_POINTS,
      });
      series.push({ name: entry.cloud.name, color: `rgb(${entry.solid.join(" ")})`, ...out });
    }
    profileSeries = series;
    renderProfilePlot();
    renderProfileHint();
    const total = series.reduce((sum, s) => sum + s.total, 0);
    $<HTMLButtonElement>("profile-csv").disabled = total === 0;
    setStatus(
      `Profile: ${total.toLocaleString()} points from ${series.length} ${series.length === 1 ? "cloud" : "clouds"} ` +
        `within ${fmt(halfWidth * 2)} of a ${fmt(profileLength())} line`,
    );
  } catch (err) {
    setStatus(`Profile failed: ${err instanceof Error ? err.message : err}`, true);
  }
}

/** Round tick positions covering `[lo, hi]`, about `count` of them. */
function ticks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? 10 * p;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

/** Data window and pixel frame of the last plot, for the cursor readout. */
let plotFrame: { x0: number; x1: number; y0: number; y1: number; left: number; top: number; w: number; h: number } | null =
  null;

function renderProfilePlot(): void {
  profilePlot.hidden = profileSeries.length === 0;
  $<HTMLButtonElement>("profile-csv").disabled = profileSeries.every((s) => s.along.length === 0);
  $("profile-legend").replaceChildren(
    ...profileSeries.map((s) => {
      const item = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.background = s.color;
      item.append(swatch, `${s.name} (${s.total.toLocaleString()})`);
      return item;
    }),
  );
  if (profilePlot.hidden) {
    plotFrame = null;
    return;
  }
  const dpr = window.devicePixelRatio || 1;
  const cw = profileCanvas.clientWidth;
  const ch = profileCanvas.clientHeight;
  profileCanvas.width = Math.round(cw * dpr);
  profileCanvas.height = Math.round(ch * dpr);
  const ctx = profileCanvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  let [x0, x1] = [0, profileLength()];
  let [y0, y1] = [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (const s of profileSeries) {
    for (let i = 2; i < s.positions.length; i += 3) {
      y0 = Math.min(y0, s.positions[i]);
      y1 = Math.max(y1, s.positions[i]);
    }
  }
  if (!(y1 >= y0)) [y0, y1] = [0, 1];
  const pad = Math.max((y1 - y0) * 0.05, 1e-6);
  [y0, y1] = [y0 - pad, y1 + pad];
  const left = 64;
  const top = 8;
  const w = Math.max(10, cw - left - 12);
  const h = Math.max(10, ch - top - 22);
  if ($<HTMLInputElement>("profile-equal").checked) {
    // Same units per pixel on both axes: widen whichever range is short.
    const scale = Math.max((x1 - x0) / w, (y1 - y0) / h);
    const [cx, cy] = [(x0 + x1) / 2, (y0 + y1) / 2];
    [x0, x1] = [cx - (scale * w) / 2, cx + (scale * w) / 2];
    [y0, y1] = [cy - (scale * h) / 2, cy + (scale * h) / 2];
  }
  const px = (d: number) => left + ((d - x0) / (x1 - x0)) * w;
  const py = (z: number) => top + (1 - (z - y0) / (y1 - y0)) * h;

  ctx.font = "11px system-ui, sans-serif";
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgb(255 255 255 / 0.08)";
  ctx.fillStyle = "#8b93a5";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of ticks(x0, x1, Math.max(2, Math.floor(w / 90)))) {
    ctx.beginPath();
    ctx.moveTo(px(t) + 0.5, top);
    ctx.lineTo(px(t) + 0.5, top + h);
    ctx.stroke();
    ctx.fillText(fmt(t), px(t), top + h + 4);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of ticks(y0, y1, Math.max(2, Math.floor(h / 40)))) {
    ctx.beginPath();
    ctx.moveTo(left, py(t) + 0.5);
    ctx.lineTo(left + w, py(t) + 0.5);
    ctx.stroke();
    ctx.fillText(fmt(t), left - 6, py(t));
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, w, h);
  ctx.clip();
  for (const s of profileSeries) {
    ctx.fillStyle = s.color;
    for (let i = 0; i < s.along.length; i++) {
      ctx.fillRect(px(s.along[i]) - 1, py(s.positions[i * 3 + 2]) - 1, 2, 2);
    }
  }
  ctx.restore();
  plotFrame = { x0, x1, y0, y1, left, top, w, h };
}

$<HTMLInputElement>("profile-equal").onchange = renderProfilePlot;
new ResizeObserver(() => {
  if (!profilePlot.hidden) renderProfilePlot();
}).observe(profileCanvas);

profileCanvas.onmousemove = (e) => {
  const f = plotFrame;
  if (!f) return;
  const rect = profileCanvas.getBoundingClientRect();
  const [mx, my] = [e.clientX - rect.left - f.left, e.clientY - rect.top - f.top];
  const inside = mx >= 0 && my >= 0 && mx <= f.w && my <= f.h;
  $("profile-readout").textContent = inside
    ? `distance ${fmt(f.x0 + (mx / f.w) * (f.x1 - f.x0))} · z ${fmt(f.y1 - (my / f.h) * (f.y1 - f.y0))}`
    : "";
};
profileCanvas.onmouseleave = () => {
  $("profile-readout").textContent = "";
};

$<HTMLButtonElement>("profile-csv").onclick = () => {
  const quote = (s: string) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  const rows = ["cloud,distance,x,y,z"];
  for (const s of profileSeries) {
    const name = quote(s.name);
    for (let i = 0; i < s.along.length; i++) {
      const p = s.positions;
      rows.push(`${name},${s.along[i]},${p[i * 3]},${p[i * 3 + 1]},${p[i * 3 + 2]}`);
    }
  }
  const url = URL.createObjectURL(new Blob([`${rows.join("\n")}\n`], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "profile.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  setStatus(`Saved profile.csv (${(rows.length - 1).toLocaleString()} points)`);
};

// Clicks go to the line while drawing it.
const pickOnClick = viewer.onClick;
viewer.onClick = (x, y) => {
  if (!profileDrawing) return pickOnClick(x, y);
  const p = viewer.groundPoint(x, y, profileZ);
  if (!p) return;
  const shift = globalShift();
  profileLine.push([p.x + shift[0], p.y + shift[1]]);
  drawProfileLine();
  renderProfileHint();
};
const centerOnDoubleClick = viewer.onDoubleClick;
viewer.onDoubleClick = (x, y) => {
  if (!profileDrawing) return centerOnDoubleClick(x, y);
  // The second click of the double click added a duplicate vertex.
  profileLine.pop();
  void finishProfile();
};
window.addEventListener("keydown", (e) => {
  if (!profileDrawing || e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === "Enter") void finishProfile();
  if (e.key === "Escape") clearProfile();
});
renderProfileHint();

// ---------------------------------------------------------------- sessions

/** A session waiting for some of its clouds to be opened. */
let pendingSession: Session | null = null;
/** Clouds of the pending session already restored. */
const restored = new Set<number>();
let applyingSession = false;

const pointSizeInput = $<HTMLInputElement>("point-size");
const pointBudgetSelect = $<HTMLSelectElement>("point-budget");

function captureSession(): Session {
  const shift = globalShift();
  const toOriginal = (v: THREE.Vector3) => [v.x + shift[0], v.y + shift[1], v.z + shift[2]] as [number, number, number];
  const camera = viewer.getCamera();
  let clip: Session["clip"] = null;
  if (clipEnabled.checked) {
    const box = clipBoxFromSliders();
    clip = { min: toOriginal(box.min), max: toOriginal(box.max) };
  }
  return {
    app: "CloudAnalyzer Web",
    version: 1,
    camera: entries.size ? { position: toOriginal(camera.position), target: toOriginal(camera.target) } : undefined,
    pointSize: Number(pointSizeInput.value),
    edl: edlToggle.checked,
    edlStrength: Number(edlStrength.value),
    pointBudget: Number(pointBudgetSelect.value),
    ramp: rampName,
    range,
    hiddenClasses: [...hiddenClasses],
    clip,
    profile: profileLine.length >= 2 ? { line: profileLine, halfWidth: profileHalfWidth() } : null,
    clouds: [...entries.values()]
      .filter((e) => e.origin.kind !== "derived")
      .map((e) => ({
        name: e.cloud.name,
        url: e.origin.kind === "url" ? e.origin.url : undefined,
        visible: e.visible,
        mode: e.mode,
        solid: e.solid,
        transforms: e.transforms,
        distance:
          e.c2c && (e.c2c.kind === "c2c" || e.c2c.kind === "c2m")
            ? { reference: e.c2c.referenceName, signed: e.c2c.signed }
            : undefined,
      })),
  };
}

/** Apply a session: settings now, URL clouds after downloading them, file clouds as they are opened. */
async function applySession(session: Session): Promise<void> {
  pendingSession = session;
  restored.clear();
  pointSizeInput.value = String(session.pointSize);
  viewer.setPointSize(session.pointSize);
  edlToggle.checked = session.edl;
  edlStrength.value = String(session.edlStrength);
  viewer.setEdl(session.edl, session.edlStrength);
  edlStrength.disabled = !session.edl;
  if ([...pointBudgetSelect.options].some((o) => Number(o.value) === session.pointBudget)) {
    pointBudgetSelect.value = String(session.pointBudget);
    viewer.setPointBudget(session.pointBudget);
  }
  const loaded = new Set([...entries.values()].map((e) => e.cloud.name));
  const urls = session.clouds.filter((c) => c.url && !loaded.has(c.name)).map((c) => c.url!);
  if (urls.length) {
    applyingSession = true;
    try {
      await loadUrls(urls);
    } finally {
      applyingSession = false;
    }
  }
  await restorePending();
}

/** Restore what the pending session can with the clouds open now. */
async function restorePending(): Promise<void> {
  const session = pendingSession;
  if (!session) return;
  const byName = (name: string, sources = true) =>
    [...entries.values()].find((e) => e.cloud.name === name && (!sources || e.origin.kind !== "derived"));
  const missing: string[] = [];
  for (const saved of session.clouds) {
    const entry = byName(saved.name);
    if (!entry) {
      missing.push(saved.name);
      continue;
    }
    if (restored.has(entry.cloud.id)) continue;
    restored.add(entry.cloud.id);
    if (entry.transforms.length === 0) {
      for (const matrix of saved.transforms) {
        replaceCloud(entry, await transformCloud(entry.cloud.id, matrix));
        entry.transforms.push(matrix);
      }
    }
    entry.visible = saved.visible;
    viewer.setVisible(entry.cloud.id, saved.visible);
    entry.solid = saved.solid;
    const available: Record<ColorMode, boolean> = {
      rgb: entry.cloud.colors !== null,
      intensity: entry.cloud.intensity !== null,
      classification: entry.cloud.classification !== null,
      solid: true,
      c2c: false,
      normal: entry.cloud.normals !== null,
      shade: entry.cloud.normals !== null,
    };
    if (available[saved.mode]) entry.mode = saved.mode;
    refreshColors(entry);
  }
  for (const saved of session.clouds) {
    const entry = byName(saved.name);
    const reference = saved.distance ? byName(saved.distance.reference, false) : undefined;
    if (entry && reference && saved.distance && !entry.c2c) await runNearest(entry, reference, saved.distance.signed);
  }
  if (session.ramp in RAMPS) {
    rampName = session.ramp as RampName;
    rampSelect.value = rampName;
  }
  range = session.range;
  applyRange();
  hiddenClasses.clear();
  for (const c of session.hiddenClasses) hiddenClasses.add(c);
  renderClasses();
  for (const entry of entries.values()) refreshColors(entry);
  const shift = globalShift();
  const toRender = (v: number[]) => new THREE.Vector3(v[0] - shift[0], v[1] - shift[1], v[2] - shift[2]);
  if (session.clip && entries.size) {
    clipEnabled.checked = true;
    $("clip-controls").hidden = false;
    resetClip();
    const [lo, hi] = [toRender(session.clip.min), toRender(session.clip.max)];
    for (let axis = 0; axis < 3; axis++) {
      const a = clipExtent.min.getComponent(axis);
      const size = clipExtent.max.getComponent(axis) - a || 1;
      const at = (v: number) => String(Math.round(Math.min(1, Math.max(0, (v - a) / size)) * SLIDER_MAX));
      slider(axis, "min").value = at(lo.getComponent(axis));
      slider(axis, "max").value = at(hi.getComponent(axis));
    }
    applyClip();
  }
  if (session.camera && entries.size) viewer.setCamera(toRender(session.camera.position), toRender(session.camera.target));
  if (session.profile && entries.size) {
    profileLine = session.profile.line;
    profileWidth.value = String(session.profile.halfWidth * 2);
    profileZ = viewer.getCamera().target.z;
    drawProfileLine();
    await computeProfile();
  }
  renderList();
  if (missing.length) {
    setStatus(`Session: open ${missing.join(", ")} to finish restoring it`);
  } else {
    pendingSession = null;
    restored.clear();
    setStatus(`Session restored (${session.clouds.length} ${session.clouds.length === 1 ? "cloud" : "clouds"})`);
  }
}

$<HTMLButtonElement>("session-save").onclick = () => {
  const json = JSON.stringify(captureSession(), null, 2);
  const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "session.cloudanalyzer.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  setStatus("Saved the session; open it together with the same files to restore it");
};

$<HTMLButtonElement>("share").onclick = async () => {
  const session = captureSession();
  const link = `${location.origin}${location.pathname}#session=${encodeSession(session)}`;
  const field = $<HTMLInputElement>("share-link");
  field.value = link;
  field.hidden = false;
  field.select();
  let copied = false;
  try {
    await navigator.clipboard.writeText(link);
    copied = true;
  } catch {
    // Not allowed here: the link stays selected in the field.
  }
  const local = session.clouds.filter((c) => !c.url).map((c) => c.name);
  setStatus(
    `${copied ? "Link copied" : "Link ready"}` +
      (local.length
        ? `; ${local.join(", ")} ${local.length === 1 ? "is a local file" : "are local files"}, not in the link — ` +
          "whoever opens it is asked to open them"
        : ""),
  );
};

/** `#session=…` restores a shared view, `?demo=…` runs a demo, `?url=…` (repeatable) opens clouds. */
async function startFromLink(): Promise<void> {
  const encoded = new URLSearchParams(location.hash.slice(1)).get("session");
  const params = new URLSearchParams(location.search);
  const urls = params.getAll("url");
  const demo = params.get("demo");
  try {
    if (encoded) await applySession(decodeSession(encoded));
    else if (demo) await runDemo(demo);
    else if (urls.length) await loadUrls(urls);
  } catch (err) {
    setStatus(`Link: ${err instanceof Error ? err.message : err}`, true);
  }
}
void startFromLink();
