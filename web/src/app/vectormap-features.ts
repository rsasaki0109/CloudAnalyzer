/** Editing crosswalks, traffic lights and stop lines. */

import * as THREE from "three";
import { vectorMap } from "../api";
import { $, errorText, setStatus } from "./dom";
import { globalShift, viewer } from "./state";
import { setTool, toggleTool, type Tool } from "./tools";
import {
  busy,
  clearCrosswalkPreview,
  clearJunctionPreview,
  clearSignalPreview,
  draw,
  type Edited,
  editingFeature,
  featureDrag,
  hint,
  junctionInputs,
  renderHint,
  setBusy,
  setEditingFeature,
  setFeatureDrag,
  setView,
  takeView,
  view,
  type XYZ,
} from "./vectormap";

function featureSelection(): { kind: "crosswalk" | "signal" | "stop_line"; id: number } | null {
  const [kind, id] = $<HTMLSelectElement>("vm-feature").value.split(":");
  return kind === "crosswalk" || kind === "signal" || kind === "stop_line" ? { kind, id: Number(id) } : null;
}
export function featurePoints(): XYZ[] {
  const s = featureSelection();
  return s?.kind === "crosswalk" ? view.crosswalks.find(c => c.id === s.id)?.outline ?? [] : s?.kind === "stop_line" ? view.stopLines.find(c => c.id === s.id)?.points ?? [] : view.signals.find(c => c.id === s?.id)?.points ?? [];
}
export function featureInputs(): void {
  const s = featureSelection();
  const valid = !!s && featurePoints().length >= 2 && (s.kind !== "crosswalk" || view.crosswalks.find(c => c.id === s.id)?.editable === true);
  for (const el of $("vm-feature-editor").querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button")) el.disabled = busy || !!featureDrag || (!valid && el.id !== "vm-feature");
}
function renderFeatureVertex(): void {
  const p = featurePoints()[Number($<HTMLSelectElement>("vm-feature-vertex").value)];
  for (const [i, axis] of ["x", "y", "z"].entries()) $<HTMLInputElement>(`vm-feature-${axis}`).value = p ? String(p[i]) : "";
}
export function renderFeatureFields(): void {
  const s = featureSelection();
  const p = featurePoints();
  const select = $<HTMLSelectElement>("vm-feature-vertex");
  const index = select.value;
  select.replaceChildren(...p.map((_, i) => new Option(`Vertex ${i + 1}`, String(i))));
  if (Number(index) < p.length) select.value = index || "0";
  const signal = view.signals.find(c => s?.kind === "signal" && c.id === s.id);
  $("vm-feature-height-row").hidden = !signal;
  $<HTMLInputElement>("vm-feature-height").value = signal?.height == null ? "" : String(signal.height);
  const crossing = view.crosswalks.find(c => s?.kind === "crosswalk" && c.id === s.id);
  const stop = view.stopLines.find(c => s?.kind === "stop_line" && c.id === s.id);
  const source = signal?.geometrySource ?? crossing?.geometrySource ?? stop?.geometrySource ?? "imported_or_manual";
  const label = source.startsWith("point_cloud_brightness_stripes") ? "Measured ground-paint bands" : source.startsWith("point_cloud_brightness_bar") ? "Measured ground-paint bar" : source.startsWith("point_cloud_box_fit") ? "Measured point-cloud housing" : "Imported or manually drawn geometry";
  $("vm-feature-source").textContent = s ? `${label}${source.endsWith("_user_edited") ? "; manually edited" : ""}. ${crossing && !crossing.editable ? "Separate polygon/edge geometry requires a dedicated editor." : "Review lane associations and legal meaning after edits."}` : "Choose a feature explicitly before editing.";
  renderFeatureVertex(); featureInputs();
}
export function renderFeatures(): void {
  const select = $<HTMLSelectElement>("vm-feature");
  const old = select.value;
  select.replaceChildren(new Option("Choose a crossing or signal", ""),
    ...view.crosswalks.map(c => new Option(`Crossing ${c.id}`, `crosswalk:${c.id}`)),
    ...view.stopLines.map(c => new Option(`Stop line ${c.id}`, `stop_line:${c.id}`)),
    ...view.signals.map(c => new Option(`Signal ${c.id}`, `signal:${c.id}`)));
  select.value = [...select.options].some(o => o.value === old) ? old : "";
  if (!select.value && editingFeature) setTool(null);
  renderFeatureFields();
}
export const featureSelect = $<HTMLSelectElement>("vm-feature");
featureSelect.onchange = () => { if (editingFeature) setTool(null); renderFeatureFields(); draw(); };
$<HTMLSelectElement>("vm-feature-vertex").onchange = renderFeatureVertex;
$("vm-feature-focus").onclick = () => {
  const shift = globalShift(); const box = new THREE.Box3();
  const s = featureSelection();
  const height = s?.kind === "signal" ? view.signals.find(c => c.id === s.id)?.height ?? 0 : 0;
  for (const p of featurePoints()) { box.expandByPoint(new THREE.Vector3(p[0]-shift[0], p[1]-shift[1], p[2]-shift[2])); box.expandByPoint(new THREE.Vector3(p[0]-shift[0], p[1]-shift[1], p[2]-shift[2]+height)); }
  if (!box.isEmpty()) viewer.frameBox(box.expandByScalar(2));
};
async function editFeature(points: XYZ[], height?: number): Promise<void> {
  const s = featureSelection(); if (busy || !s) return;
  setBusy(true); junctionInputs();
  try {
    const edited = await vectorMap<Edited>("feature-edit", { text: JSON.stringify({ ...s, points, height }) });
    takeView(edited);
    const report = edited.result as { changed: boolean; warnings: string[] };
    setStatus(report.changed ? `Feature ${s.id} edited. ${report.warnings.join(" ")}` : "Geometry unchanged; no Undo step added.");
  } catch (err) { setStatus(`Feature edit failed: ${errorText(err)}`); draw(); }
  finally { setBusy(false); junctionInputs(); }
}
$("vm-feature-apply").onclick = () => {
  if (busy) return;
  const values = ["x", "y", "z"].map(axis => $<HTMLInputElement>(`vm-feature-${axis}`).value.trim());
  const h = $<HTMLInputElement>("vm-feature-height").value.trim();
  if (values.some(v => !v || !Number.isFinite(Number(v))) || (h && !Number.isFinite(Number(h)))) return setStatus("Enter finite metre coordinates and a valid housing height.");
  const p = structuredClone(featurePoints());
  const index = Number($<HTMLSelectElement>("vm-feature-vertex").value);
  if (!p[index]) return;
  p[index] = values.map(Number) as XYZ;
  void editFeature(p, featureSelection()?.kind === "signal" && h ? Number(h) : undefined);
};
const featureTool: Tool = {
  click() {},
  pointerDown(x, y) {
    if (busy || !featureSelection()) return false;
    const rect = $("viewport").querySelector(":scope > canvas")!.getBoundingClientRect();
    const shift = globalShift(); let best = 10; let hit = -1;
    for (const [i, p] of featurePoints().entries()) {
      const screen = viewer.project(new THREE.Vector3(p[0]-shift[0], p[1]-shift[1], p[2]-shift[2]));
      if (!screen) continue;
      const distance = Math.hypot(screen.x+rect.left-x, screen.y+rect.top-y);
      if (distance < best) { best = distance; hit = i; }
    }
    if (hit < 0) return false;
    const point = [...featurePoints()[hit]] as XYZ;
    const ground = viewer.groundPoint(x,y,point[2]-shift[2]); if (!ground) return false;
    setFeatureDrag({ before: structuredClone(view), index: hit, point, start:[x,y], moved:false, offset:[point[0]-shift[0]-ground.x,point[1]-shift[1]-ground.y] });
    $<HTMLSelectElement>("vm-feature-vertex").value = String(hit); renderFeatureVertex(); featureInputs();
    return true;
  },
  pointerMove(x,y) {
    const d = featureDrag; if (!d || (!d.moved && Math.hypot(x-d.start[0],y-d.start[1]) < 2)) return;
    const shift = globalShift(); const p = viewer.groundPoint(x,y,d.point[2]-shift[2]); if (!p) return;
    const next: XYZ = [p.x+shift[0]+d.offset[0],p.y+shift[1]+d.offset[1],d.point[2]];
    featurePoints()[d.index] = next; d.moved = Math.hypot(next[0]-d.point[0],next[1]-d.point[1]) > 1e-4;
    renderFeatureVertex(); draw();
  },
  async pointerUp(x,y) {
    featureTool.pointerMove!(x,y); const d = featureDrag; if (!d) return;
    const points = structuredClone(featurePoints()); setFeatureDrag(null); setView(d.before); renderFeatureFields(); draw();
    if (d.moved) await editFeature(points);
  },
  pointerCancel() { if (featureDrag) { setView(featureDrag.before); setFeatureDrag(null); renderFeatureFields(); draw(); } },
  enter() { setEditingFeature(true); clearJunctionPreview(); clearSignalPreview(); clearCrosswalkPreview(); $("vm-feature-drag").setAttribute("aria-pressed","true"); hint.textContent = "Drag a yellow feature vertex; Z is kept. Escape cancels. Review observed paint, lamps and lane associations after editing."; draw(); },
  exit() { featureTool.pointerCancel!(); setEditingFeature(false); $("vm-feature-drag").setAttribute("aria-pressed","false"); renderHint(); draw(); },
};
$("vm-feature-drag").onclick = () => { if (!busy && featureSelection()) toggleTool(featureTool); };
