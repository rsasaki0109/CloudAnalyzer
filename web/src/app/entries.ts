/** The cloud list: adding, replacing, removing and saving clouds. */

import { exportCloud, removeCloud } from "../api";
import { parseNodes } from "../lod";
import type { LoadedCloud, Vec3 } from "../protocol";
import { colorsFor, defaultMode, distanceLabel, refreshColors } from "./colors";
import { $, download, errorText, removeButton, setStatus } from "./dom";
import { narrow, setPanelsOpen } from "./layout";
import {
  type ColorMode,
  display,
  distanceChanged,
  type Entry,
  entries,
  findByName,
  isMesh,
  listChanged,
  type Origin,
  pointsInvalidated,
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
  entry.nodes = parseNodes(cloud.lodNodes, cloud.lodGrid, cloud.shift);
  // Distances involving the moved cloud no longer describe the data.
  for (const other of entries.values()) {
    if (other.c2c && (other === entry || other.c2c.referenceName === name)) {
      clearDistance(other);
      if (other !== entry) refreshColors(other);
    }
  }
  viewer.remove(cloud.id);
  viewer.add(cloud.id, cloud.positions, colorsFor(entry), entry.nodes);
  viewer.setVisible(cloud.id, entry.visible);
  pointsInvalidated.emit(cloud.id);
  renderList();
  distanceChanged.emit();
}

/** Download a cloud as PLY or CSV, including its distances if computed. */
export async function saveCloud(entry: Entry, format: "ply" | "csv"): Promise<void> {
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
      if (entry.mode === "c2c") display.activeC2c = cloud.id;
      else if (display.activeC2c === cloud.id) display.activeC2c = null;
      distanceChanged.emit();
    };

    // Meshes are drawn in their solid color only.
    if (isMesh(entry)) li.append(visible, swatch, name, actions);
    else li.append(visible, swatch, name, actions, mode);
    list.append(li);
  }
  listChanged.emit();
}
