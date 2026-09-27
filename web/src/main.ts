import { cloudToCloud, loadCloud, removeCloud } from "./api";
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
  viewer.setColors(entry.cloud.id, colorsFor(entry));
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
    meta.textContent = `${cloud.count.toLocaleString()} points${cloud.colors ? " · RGB" : ""}`;
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
      ["c2c", entry.c2c ? `C2C distance → ${entry.c2c.referenceName}` : "C2C distance", !!entry.c2c],
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

    li.append(visible, swatch, name, remove, mode);
    list.append(li);
  }
  renderC2cSelects();
}

async function removeEntry(id: number): Promise<void> {
  entries.delete(id);
  viewer.remove(id);
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
    setStatus(`Loading ${file.name}…`);
    const start = performance.now();
    try {
      const cloud = await loadCloud(file.name, await file.arrayBuffer());
      const entry: Entry = {
        cloud,
        nodes: parseNodes(cloud.lodNodes, cloud.lodGrid, cloud.shift),
        solid: SOLID_COLORS[entries.size % SOLID_COLORS.length],
        mode: cloud.colors ? "rgb" : "solid",
        visible: true,
      };
      entries.set(cloud.id, entry);
      viewer.add(cloud.id, cloud.positions, colorsFor(entry), entry.nodes);
      if (entries.size === 1) viewer.fit();
      const [sx, sy, sz] = cloud.shift;
      $("shift").textContent =
        sx || sy || sz ? `Global shift: (${-sx}, ${-sy}, ${-sz})` : "";
      setStatus(
        `Loaded ${file.name}: ${cloud.count.toLocaleString()} points in ${Math.round(performance.now() - start)} ms`,
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
  const total = [...entries.values()].reduce((sum, e) => sum + (e.visible ? e.cloud.count : 0), 0);
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
  const ids = [...entries.keys()].map(String);
  let [compared, reference] = [comparedSelect.value, referenceSelect.value];
  const valid = ids.includes(compared) && ids.includes(reference) && compared !== reference;
  if (!valid) {
    // Default: the newest cloud is compared against the first one.
    [compared, reference] = [ids.at(-1) ?? "", ids[0] ?? ""];
  }
  for (const [select, value] of [
    [comparedSelect, compared],
    [referenceSelect, reference],
  ] as const) {
    select.replaceChildren(...ids.map((id) => new Option(entries.get(Number(id))!.cloud.name, id)));
    select.value = value;
  }
  updateRunButton();
}

function updateRunButton(): void {
  runButton.disabled = entries.size < 2 || comparedSelect.value === referenceSelect.value;
}
comparedSelect.onchange = referenceSelect.onchange = updateRunButton;

runButton.onclick = async () => {
  const compared = entries.get(Number(comparedSelect.value));
  const reference = entries.get(Number(referenceSelect.value));
  if (!compared || !reference) return;
  runButton.disabled = true;
  setStatus(`Computing C2C distance: ${compared.cloud.name} → ${reference.cloud.name}…`);
  try {
    const result = await cloudToCloud(compared.cloud.id, reference.cloud.id);
    compared.c2c = { ...result, referenceName: reference.cloud.name };
    compared.mode = "c2c";
    activeC2c = compared.cloud.id;
    range = null;
    refreshColors(compared);
    renderList();
    renderC2cResult();
    setStatus(
      `C2C distance computed for ${result.stats.count.toLocaleString()} points in ${Math.round(result.millis)} ms` +
        (result.workers > 1 ? ` on ${result.workers} workers` : ""),
    );
  } catch (err) {
    setStatus(`C2C failed: ${err instanceof Error ? err.message : err}`, true);
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
  $("colorbar-title").textContent = `C2C distance · ${entry.cloud.name}`;
  $("colorbar-ramp").style.background = gradientCss(rampName);
  $("colorbar-max").textContent = fmt(hi);
  $("colorbar-mid").textContent = fmt((lo + hi) / 2);
  $("colorbar-min").textContent = fmt(lo);
}

renderList();
