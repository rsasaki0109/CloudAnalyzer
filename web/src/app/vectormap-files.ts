/** Opening, saving, clearing and undoing the map. */

import { onHistoryPolicy, historyPolicy } from "../memory-budget";
import * as THREE from "three";
import { vectorMap } from "../api";
import { $, download, errorText, setStatus } from "./dom";
import { entries, globalShift, viewer } from "./state";
import { setTool } from "./tools";
import {
  busy,
  type Edited,
  exportButton,
  type Issue,
  type MapView,
  plural,
  renderHint,
  reviews,
  setBoundaryVertex,
  setSelected,
  setUndoDepth,
  takeView,
  undoButton,
  undoDepth,
  view,
} from "./vectormap";

/** Open a Lanelet2 map (.osm) or vectormap IR (.json) over the clouds. */
export function captureMapProject(): Promise<string> {
  if (busy) return Promise.reject(new Error("Finish the map operation before saving the project"));
  return vectorMap<unknown>("json").then(value => JSON.stringify(value));
}

export async function openVectorMap(name: string, text: string): Promise<MapView> {
  return takePreparedMap(name, await vectorMap<Edited>("open", { name, text }));
}

/** Native map has already been validated and committed with its workspace. */
export function takePreparedMap(name: string, value: unknown): MapView {
  const opened = value as Edited;
  setTool(null);
  const issues = opened.result as Issue[];
  $("vm-import-notes").hidden = issues.length === 0;
  $("vm-import-notes").setAttribute("open", "");
  $("vm-import-issues").replaceChildren(...issues.map((issue) => {
    const li = document.createElement("li");
    li.className = issue.severity;
    li.textContent = issue.message;
    li.title = issue.code;
    return li;
  }));
  setSelected(null);
  setBoundaryVertex(null);
  reviews.clear();
  takeView(opened);
  if (view.lanes.length > 0 && entries.size === 0) frameMap();
  const problems = issues.filter((i) => i.severity !== "info").length;
  setStatus(
    `Opened ${name}: ${view.lanes.length} lanes${problems ? `, ${problems} load issues (see Import notes)` : ""}.`,
  );
  return view;
}

const fileInput = $<HTMLInputElement>("vm-file");
$("vm-open").onclick = () => fileInput.click();
fileInput.onchange = async () => {
  const file = fileInput.files?.[0];
  fileInput.value = "";
  if (!file) return;
  try {
    await openVectorMap(file.name, await file.text());
  } catch (err) {
    setStatus(`Could not open ${file.name}: ${errorText(err)}`);
  }
};

function frameMap(): void {
  const shift = globalShift();
  const box = new THREE.Box3();
  for (const b of view.boundaries) {
    for (const p of b.points) box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
  }
  for (const s of view.signals) for (const p of s.points) {
    box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2] + (s.height ?? 0)));
  }
  for (const c of view.crosswalks) for (const p of c.outline) {
    box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
  }
  viewer.frameBox(box);
}
$("vm-fit").onclick = frameMap;
$("vm-plan").onclick = () => { viewer.view({ x: 0, y: 0, z: 1 }); frameMap(); };
$("vm-iso").onclick = () => { viewer.view({ x: 0.65, y: -0.9, z: 1.2 }); frameMap(); };

undoButton.onclick = async () => {
  if (busy) return;
  takeView(await vectorMap<Edited>("undo"));
  setStatus("Undone.");
};
$("vm-clear").onclick = async () => {
  if (busy) return;
  setTool(null);
  if (view.lanes.length === 0 && view.boundaries.length === 0) return;
  setSelected(null);
  setBoundaryVertex(null);
  reviews.clear();
  takeView(await vectorMap<Edited>("clear"));
  setStatus("Map cleared (Undo brings it back).");
};
exportButton.onclick = async () => {
  if (busy || exportButton.disabled) return;
  exportButton.disabled = true;
  try {
    const exported = await vectorMap<{ osm: string; projectorInfo: string; issues: Issue[] }>("export", { autoware: true });
    download(new Blob([exported.osm], { type: "application/xml" }), "lanelet2_map.osm");
    download(new Blob([exported.projectorInfo], { type: "text/yaml" }), "map_projector_info.yaml");
    const issues = exported.issues.filter(issue => issue.severity !== "info");
    const counts = `${plural(issues.filter(i => i.severity === "error").length, "error")}, ${plural(issues.filter(i => i.severity === "warning").length, "warning")}`;
    $("vm-export-summary").textContent = `Last saved Lanelet2: ${counts}${issues.length > 20 ? " (first 20 shown)" : ""}.`;
    $("vm-export-issues").replaceChildren(...issues.slice(0, 20).map(issue => {
      const li = document.createElement("li");
      li.className = issue.severity;
      li.textContent = issue.message;
      li.title = issue.code;
      return li;
    }));
    $("vm-export-report").hidden = issues.length === 0;
    setStatus(`Saved lanelet2_map.osm and map_projector_info.yaml (${exported.projectorInfo.split("\n")[0].replace("projector_type: ", "")} projector). Export: ${counts}; also review the map validation and source coverage.`);
  } catch (error) {
    setStatus(`Could not export Lanelet2: ${errorText(error)}`);
  } finally {
    exportButton.disabled = busy || view.lanes.length === 0;
  }
};

renderHint();
void vectorMap<Edited>("view").then(edited => takeView(edited, false));

export async function clearMapHistory(): Promise<void> { takeView(await vectorMap<Edited>("history-clear")); }
onHistoryPolicy(policy => { void vectorMap<Edited>("history-budget", {text: JSON.stringify(policy)}).then(edited => { setUndoDepth(edited.undo); undoButton.disabled = busy || undoDepth === 0; }); });
void vectorMap<Edited>("history-budget", {text: JSON.stringify(historyPolicy())}).then(edited => { setUndoDepth(edited.undo); undoButton.disabled = busy || undoDepth === 0; });
