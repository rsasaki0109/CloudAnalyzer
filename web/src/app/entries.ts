/** The cloud list: adding, replacing, removing and saving clouds and meshes. */

import { exportCloud, exportMesh, removeCloud } from "../api";
import { parseNodes } from "../lod";
import type { ExportFormat, LoadedCloud, MeshFormat, Vec3 } from "../protocol";
import { colorsFor, defaultMode, distanceLabel, refreshColors } from "./colors";
import { $, download, errorText, removeButton, setStatus } from "./dom";
import { colorByField, distanceField, fieldNames } from "./scalars";
import { narrow, setPanelsOpen } from "./layout";
import {
  type ColorMode,
  display,
  distanceChanged,
  type Entry,
  entries,
  findByName,
  globalShift,
  isMesh,
  listChanged,
  type Origin,
  pointsInvalidated,
  putEntry,
  toRender,
  viewer,
} from "./state";

const SOLID_COLORS: Vec3[] = [
  [235, 235, 235],
  [255, 176, 0],
  [79, 195, 247],
  [255, 110, 156],
  [156, 204, 101],
  [186, 104, 200],
];

/** Register a loaded cloud or mesh and draw it. */
export function addEntry(cloud: LoadedCloud, origin: Origin = { kind: "derived" }): Entry {
  const entry: Entry = {
    cloud,
    nodes: [],
    solid: SOLID_COLORS[entries.size % SOLID_COLORS.length],
    mode: defaultMode(cloud),
    visible: true,
    transforms: [],
    fields: new Map(),
    origin,
  };
  putEntry(entry);
  // On a phone, get the sheet out of the way once there is something to see.
  if (entries.size === 1 && narrow.matches) setPanelsOpen(false);
  drawEntry(entry);
  return entry;
}

/** Draw a listed entry's cloud or mesh at its place in render coordinates. */
export function drawEntry(entry: Entry): void {
  const { cloud } = entry;
  entry.nodes = parseNodes(cloud.lodNodes, cloud.lodGrid, globalShift());
  const offset = toRender(cloud.shift);
  if (cloud.kind === "mesh") viewer.addMesh(cloud.id, cloud.positions, cloud.indices!, entry.solid, offset);
  else viewer.add(cloud.id, cloud.positions, colorsFor(entry), entry.nodes, offset);
  viewer.setVisible(cloud.id, entry.visible);
}

/** Drop a distance result, falling back to the default colors. */
function clearDistance(entry: Entry): void {
  entry.c2c = undefined;
  if (entry.mode === "c2c") entry.mode = defaultMode(entry.cloud);
  if (display.activeC2c === entry.cloud.id) display.activeC2c = null;
}

export async function removeEntry(id: number): Promise<void> {
  entries.delete(id);
  viewer.remove(id);
  pointsInvalidated.emit(id);
  await removeCloud(id);
  if (display.activeC2c === id) display.activeC2c = null;
  // Distances computed against the removed cloud are no longer meaningful to keep around.
  for (const entry of entries.values()) {
    if (entry.c2c && !findByName(entry.c2c.referenceName)) {
      const colored = entry.mode === "c2c";
      clearDistance(entry);
      if (colored) refreshColors(entry);
    }
  }
  if (entries.size === 0) $("shift").textContent = "";
  renderList();
  distanceChanged.emit();
}

/** Swap in a cloud whose points moved (and were reordered) in the worker. */
export function replaceCloud(entry: Entry, cloud: LoadedCloud): void {
  const name = entry.cloud.name;
  entry.cloud = cloud;
  // Fields follow the old point order: fetch them again when needed.
  entry.fields.clear();
  if (entry.field) {
    entry.field = undefined;
    if (entry.mode === "scalar") entry.mode = defaultMode(cloud);
  }
  // Distances involving the moved cloud no longer describe the data.
  for (const other of entries.values()) {
    if (other.c2c && (other === entry || other.c2c.referenceName === name)) {
      clearDistance(other);
      if (other !== entry) refreshColors(other);
    }
  }
  viewer.remove(cloud.id);
  drawEntry(entry);
  pointsInvalidated.emit(cloud.id);
  renderList();
  distanceChanged.emit();
}

/** Offered by a cloud's ⤓ button, with what each keeps. */
const SAVE_FORMATS: [ExportFormat, string][] = [
  ["ply", "Binary PLY with every attribute"],
  ["las", "LAS 1.4: intensity, classes, colors; other fields as extra bytes"],
  ["laz", "Compressed LAS"],
  ["csv", "Text table"],
  ["e57", "E57: XYZ, intensity, colors"],
];

/** Offered by a mesh's ⤓ button. */
const MESH_FORMATS: [MeshFormat, string][] = [
  ["ply", "Binary PLY"],
  ["obj", "Wavefront OBJ (text)"],
];

/** Download a mesh. */
async function saveMesh(entry: Entry, format: MeshFormat): Promise<void> {
  const filename = `${entry.cloud.name.replace(/\.[^.]+$/, "")}.${format}`;
  setStatus(`Saving ${filename}…`);
  try {
    const bytes = await exportMesh(entry.cloud.id, format);
    download(bytes, filename);
    setStatus(`Saved ${filename} (${(bytes.byteLength / 1e6).toFixed(1)} MB)`);
  } catch (err) {
    setStatus(`Save failed: ${errorText(err)}`, true);
  }
}

/** Download a cloud, including its distances if computed (not in E57). */
export async function saveCloud(entry: Entry, format: ExportFormat): Promise<void> {
  const { cloud } = entry;
  const c2c = format === "e57" ? undefined : entry.c2c;
  const kind = c2c?.kind.toUpperCase();
  // M3C2 results already carry m3c2_distance / lod95 / significant
  // attributes, and raster cells their height.
  const scalar =
    c2c && kind && c2c.kind !== "m3c2" && c2c.kind !== "raster"
      ? { name: `${kind}_distance`, values: c2c.distances }
      : undefined;
  const base = cloud.name.replace(/\.[^.]+$/, "");
  const filename = `${base}${kind ? `_${kind}` : ""}.${format}`;
  setStatus(`Saving ${filename}…`);
  try {
    const bytes = await exportCloud(cloud.id, format, scalar);
    download(bytes, filename);
    setStatus(`Saved ${filename} (${(bytes.byteLength / 1e6).toFixed(1)} MB)`);
  } catch (err) {
    setStatus(`Save failed: ${errorText(err)}`, true);
  }
}

export function renderList(): void {
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

    const remove = removeButton(() => void removeEntry(cloud.id));

    const actions = document.createElement("span");
    actions.className = "actions";
    // ⤓ opens a row of format buttons under the cloud.
    const formats = document.createElement("span");
    formats.className = "save-formats";
    formats.hidden = true;
    const save = document.createElement("button");
    save.className = "icon";
    save.textContent = "⤓";
    save.title = entry.c2c ? "Save with distances…" : "Save as…";
    save.ariaExpanded = "false";
    save.onclick = () => {
      formats.hidden = !formats.hidden;
      save.ariaExpanded = String(!formats.hidden);
    };
    const choices: [string, string, () => Promise<void>][] = isMesh(entry)
      ? MESH_FORMATS.map(([format, title]) => [format, title, () => saveMesh(entry, format)])
      : SAVE_FORMATS.map(([format, title]) => [format, title, () => saveCloud(entry, format)]);
    for (const [format, title, run] of choices) {
      const button = document.createElement("button");
      button.textContent = format.toUpperCase();
      button.title = title;
      button.onclick = () => {
        formats.hidden = true;
        save.ariaExpanded = "false";
        void run();
      };
      formats.append(button);
    }
    actions.append(save, remove);

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
      ["opacity", "Opacity", cloud.opacity !== null],
    ];
    for (const [value, label, enabled] of options) {
      if (!enabled) continue;
      mode.add(new Option(label, value, false, value === entry.mode));
    }
    // Other scalar fields (the distance and intensity have their own entries above).
    const fields = fieldNames(entry).filter((f) => f !== distanceField(entry) && f !== "intensity" && f !== "opacity");
    if (fields.length) {
      const group = document.createElement("optgroup");
      group.label = "Scalar field";
      for (const f of fields) {
        group.append(new Option(f, `field:${f}`, false, entry.mode === "scalar" && entry.field?.name === f));
      }
      mode.add(group);
    }
    mode.onchange = () => {
      if (mode.value.startsWith("field:")) {
        void colorByField(entry, mode.value.slice(6)).catch((err) => setStatus(errorText(err), true));
        return;
      }
      entry.mode = mode.value as ColorMode;
      refreshColors(entry);
      if (entry.mode === "c2c") display.activeC2c = cloud.id;
      else if (display.activeC2c === cloud.id) display.activeC2c = null;
      distanceChanged.emit();
    };

    // Meshes are drawn in their solid color only.
    if (isMesh(entry)) li.append(visible, swatch, name, actions, formats);
    else li.append(visible, swatch, name, actions, mode, formats);
    list.append(li);
  }
  listChanged.emit();
}
