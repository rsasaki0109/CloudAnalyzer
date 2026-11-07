/** Traffic light measurement from the cloud. */

import * as THREE from "three";
import { cropCloud, removeCloud, vectorMap } from "../api";
import { addEntry, renderList } from "./entries";
import { refreshColors } from "./colors";
import { record } from "./history";
import { $, errorText, fmt, setStatus } from "./dom";
import { entries, globalShift, viewer } from "./state";
import { setTool } from "./tools";
import {
  busy,
  draw,
  type Edited,
  featureSelect,
  junctionInputs,
  renderFeatureFields,
  selected,
  setBusy,
  type SignalReport,
  takeView,
  type XYZ,
} from "./vectormap";

export let signalPreview: SignalReport | null = null;
let signalSnapshot: string | null = null;
let signalRevision = 0;

export function signalInputs(): void {
  for (const el of $("vm-signal-preview").closest("details")!.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button")) el.disabled = busy;
  $<HTMLButtonElement>("vm-signal-add").disabled = busy || !signalPreview || !signalSnapshot;
  $<HTMLButtonElement>("vm-signal-inspect").disabled = busy || !signalPreview || !signalSnapshot;
}
export function clearSignalPreview(): void {
  signalRevision++;
  signalPreview = null;
  signalSnapshot = null;
  $("vm-signal-report").textContent = "";
  signalInputs();
}
function signalRequest(): { id: number; text: string } {
  const id = Number($<HTMLSelectElement>("vm-signal-cloud").value);
  if (!entries.has(id)) throw new Error("Choose a point cloud.");
  const read = (edge: string) => ["x", "y", "z"].map((axis) => {
    const value = $<HTMLInputElement>(`vm-signal-${edge}-${axis}`).value;
    if (!value.trim()) throw new Error("Enter all six box coordinates, or use the isolated cloud's bounds.");
    return Number(value);
  });
  const lanes = $<HTMLInputElement>("vm-signal-lanes").value.split(",").map((s) => Number(s.trim()));
  if (!lanes.every((id) => Number.isSafeInteger(id) && id > 0)) throw new Error("Enter confirmed controlled lane IDs.");
  return { id, text: JSON.stringify({ min: read("min"), max: read("max"), lanes, kind: $<HTMLSelectElement>("vm-signal-kind").value }) };
}
for (const input of $("vm-signal-preview").closest("details")!.querySelectorAll("input,select")) input.addEventListener("input", () => { clearSignalPreview(); draw(); });
$("vm-signal-bounds").onclick = () => {
  if (busy) return;
  const cloud = entries.get(Number($<HTMLSelectElement>("vm-signal-cloud").value))?.cloud;
  if (!cloud) return setStatus("Choose a point cloud.");
  ["min", "max"].forEach((edge, j) => ["x", "y", "z"].forEach((axis, i) => {
    const padding = cloud.bounds[i] === cloud.bounds[i + 3] ? 0.001 : 0;
    $<HTMLInputElement>(`vm-signal-${edge}-${axis}`).value = String(cloud.bounds[j * 3 + i] + (j ? padding : -padding));
  }));
  clearSignalPreview(); draw();
};
$("vm-signal-selected").onclick = () => {
  if (busy) return;
  if (selected === null) return setStatus("Select the controlled lane first.");
  $<HTMLInputElement>("vm-signal-lanes").value = String(selected);
  clearSignalPreview(); draw();
};
async function measureSignal(preview: boolean): Promise<void> {
  if (busy) return;
  const revision = signalRevision;
  try {
    const request = signalRequest();
    if (!preview && JSON.stringify(request) !== signalSnapshot) throw new Error("Measure and review these inputs before adding.");
    setTool(null);
    setBusy(true);
    junctionInputs();
    const edited = await vectorMap<Edited>(preview ? "signal-preview" : "signal-add", request);
    if (revision !== signalRevision) return;
    takeView(edited);
    const report = edited.result as SignalReport;
    if (!preview && (report.added || report.reused)) { featureSelect.value = `signal:${report.added ?? report.reused}`; renderFeatureFields(); }
    if (preview) {
      signalPreview = report;
      signalSnapshot = JSON.stringify(request);
      const shift = globalShift();
      const box = new THREE.Box3();
      for (const p of report.geometry) {
        box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
        box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] + report.height - shift[2]));
      }
      viewer.frameBox(box.expandByScalar(2));
    }
    $("vm-signal-report").textContent = `${report.points} points; width ${fmt(report.width)} m, height ${fmt(report.height)} m; thickness ${fmt(report.thickness)} m; plane RMS ${fmt(report.plane_rms)} m. ${report.warnings.join(" ")}`;
    setStatus(preview ? "Signal geometry measured. Confirm the cyan outline, object kind and controlled lanes before adding." : report.added ? `Measured signal ${report.added} added. Undo removes this signal and its reviewed lane assignment.` : `Signal ${report.reused} already measured; no changes.`);
    draw();
  } catch (err) {
    if (revision !== signalRevision) return;
    clearSignalPreview(); draw();
    setStatus(`Could not measure signal: ${errorText(err)}`);
  } finally { setBusy(false); junctionInputs(); }
}
$("vm-signal-preview").onclick = () => { void measureSignal(true); };
$("vm-signal-add").onclick = () => { void measureSignal(false); };
$("vm-signal-inspect").onclick = async () => {
  if (busy || !signalPreview || !signalSnapshot) return;
  const revision = signalRevision;
  setBusy(true); junctionInputs();
  try {
    const request = signalRequest();
    if (JSON.stringify(request) !== signalSnapshot) throw new Error("Measure these inputs first.");
    const options = JSON.parse(request.text) as { min: XYZ; max: XYZ };
    const cloud = await cropCloud(request.id, options.min, options.max, true);
    if (revision !== signalRevision || !entries.has(request.id)) {
      await removeCloud(cloud.id);
      throw new Error("Point cloud changed; measure again before inspecting.");
    }
    const source = entries.get(request.id)!;
    const added = addEntry(cloud);
    added.mode = "solid";
    refreshColors(added);
    record({ label: "the signal box", added: [added], hide: [source] });
    renderList();
    $<HTMLSelectElement>("vm-signal-cloud").value = String(cloud.id);
    setStatus(`${cloud.count} box points copied for inspection. The source is hidden; cloud Undo restores it. Measure this isolated cloud again before adding.`);
  } catch (err) { setStatus(`Could not inspect signal points: ${errorText(err)}`); }
  finally { setBusy(false); junctionInputs(); }
};
