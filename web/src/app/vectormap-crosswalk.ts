/** Crosswalk measurement from painted stripes. */

import * as THREE from "three";
import { cropCloud, removeCloud, vectorMap } from "../api";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { $, errorText, fmt, setStatus } from "./dom";
import { entries, globalShift, viewer } from "./state";
import { setTool } from "./tools";
import {
  busy,
  type CrosswalkReport,
  draw,
  type Edited,
  featureSelect,
  junctionInputs,
  renderFeatureFields,
  selected,
  setBusy,
  takeView,
  type XYZ,
} from "./vectormap";

export let crosswalkPreview: CrosswalkReport | null = null;
let crosswalkSnapshot: string | null = null;
let crosswalkRevision = 0;
export let crosswalkCandidate: number | null = null;

export function crosswalkInputs(): void {
  for (const el of $("vm-crosswalk-measure").querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button")) el.disabled = busy;
  $<HTMLButtonElement>("vm-crosswalk-add").disabled = busy || !crosswalkPreview || !crosswalkSnapshot || crosswalkCandidate === null || !$<HTMLInputElement>("vm-crosswalk-lanes").value.trim();
  $<HTMLButtonElement>("vm-crosswalk-inspect").disabled = busy || !crosswalkPreview || !crosswalkSnapshot;
}
export function clearCrosswalkPreview(): void {
  crosswalkRevision++;
  crosswalkPreview = null; crosswalkSnapshot = null; crosswalkCandidate = null;
  $("vm-crosswalk-candidates").replaceChildren();
  $("vm-crosswalk-report").textContent = "";
  crosswalkInputs();
}
function crosswalkRequest(): { id: number; text: string } {
  const id = Number($<HTMLSelectElement>("vm-crosswalk-cloud").value);
  if (!entries.has(id)) throw new Error("Choose a point cloud.");
  const read = (edge: string) => ["x", "y", "z"].map((axis) => {
    const value = $<HTMLInputElement>(`vm-crosswalk-${edge}-${axis}`).value;
    if (!value.trim()) throw new Error("Enter all six original-coordinate box bounds.");
    const n = Number(value);
    if (!Number.isFinite(n)) throw new Error("Box bounds must be finite.");
    return n;
  });
  const input = $<HTMLInputElement>("vm-crosswalk-lanes").value.trim();
  const lanes = input ? input.split(",").map((s) => Number(s.trim())) : [];
  if (!lanes.every((id) => Number.isSafeInteger(id) && id > 0) || new Set(lanes).size !== lanes.length) throw new Error("Enter distinct confirmed crossing lane IDs.");
  return { id, text: JSON.stringify({ min: read("min"), max: read("max"), lanes, brightness_fraction: Number($<HTMLInputElement>("vm-crosswalk-brightness").value) / 100 }) };
}
for (const input of $("vm-crosswalk-measure").querySelectorAll("input,select")) input.addEventListener("input", () => { clearCrosswalkPreview(); draw(); });
$("vm-crosswalk-bounds").onclick = () => {
  if (busy) return;
  const cloud = entries.get(Number($<HTMLSelectElement>("vm-crosswalk-cloud").value))?.cloud;
  if (!cloud) return setStatus("Choose a point cloud.");
  ["min", "max"].forEach((edge, j) => ["x", "y", "z"].forEach((axis, i) => {
    const padding = cloud.bounds[i] === cloud.bounds[i + 3] ? 0.001 : 0;
    $<HTMLInputElement>(`vm-crosswalk-${edge}-${axis}`).value = String(cloud.bounds[j * 3 + i] + (j ? padding : -padding));
  }));
  clearCrosswalkPreview(); draw();
};
$("vm-crosswalk-selected").onclick = () => {
  if (busy) return;
  if (selected === null) return setStatus("Select a crossing lane first.");
  $<HTMLInputElement>("vm-crosswalk-lanes").value = String(selected);
  clearCrosswalkPreview(); draw();
};
async function measureCrosswalk(preview: boolean): Promise<void> {
  if (busy) return;
  const revision = crosswalkRevision;
  try {
    const request = crosswalkRequest();
    if (!preview && (JSON.stringify(request) !== crosswalkSnapshot || crosswalkCandidate === null)) throw new Error("Preview and select a reviewed paint candidate first.");
    const candidate = crosswalkCandidate;
    const sent = preview ? request : { ...request, text: JSON.stringify({ ...JSON.parse(request.text), candidate }) };
    setTool(null); setBusy(true); junctionInputs();
    const edited = await vectorMap<Edited>(preview ? "crosswalk-preview" : "crosswalk-add", sent);
    if (revision !== crosswalkRevision) return;
    takeView(edited);
    const report = edited.result as CrosswalkReport;
    if (!preview && (report.added || report.reused)) { featureSelect.value = `crosswalk:${report.added ?? report.reused}`; renderFeatureFields(); }
    if (preview) {
      crosswalkPreview = report; crosswalkSnapshot = JSON.stringify(request);
      report.candidates.forEach((c, index) => {
        const label = document.createElement("label");
        label.style.display = "block";
        const radio = document.createElement("input");
        radio.type = "radio"; radio.name = "crosswalk-candidate"; radio.value = String(index);
        radio.onchange = () => {
          if (busy) return;
          crosswalkCandidate = index;
          const shift = globalShift(); const box = new THREE.Box3();
          for (const p of c.outline) box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
          viewer.frameBox(box.expandByScalar(2)); crosswalkInputs(); draw();
        };
        label.append(radio, ` Candidate ${index + 1}: ${c.stripe_count} bands, ${fmt(c.width)} × ${fmt(c.length)} m, ${c.angle_degrees}°; rank ${fmt(c.score)}`);
        $("vm-crosswalk-candidates").append(label);
      });
    }
    $("vm-crosswalk-report").textContent = `${report.points} box points; ${report.ground_points} ground; ${report.brightness_source}; ground RMS ${fmt(report.plane_rms)} m. ${report.warnings.join(" ")}`;
    setStatus(preview ? `${report.candidates.length} paint candidates. Select one to inspect measured bands, then confirm the crossing and lane IDs.` : report.added ? `Measured crossing ${report.added} added. Undo removes the crossing and its lane assignment.` : `Crossing ${report.reused} already measured; no changes.`);
    draw();
  } catch (err) {
    if (revision !== crosswalkRevision) return;
    clearCrosswalkPreview(); draw(); setStatus(`Could not measure crossing: ${errorText(err)}`);
  } finally { setBusy(false); junctionInputs(); }
}
$("vm-crosswalk-preview").onclick = () => { void measureCrosswalk(true); };
$("vm-crosswalk-add").onclick = () => { void measureCrosswalk(false); };
$("vm-crosswalk-inspect").onclick = async () => {
  if (busy || !crosswalkPreview || !crosswalkSnapshot) return;
  const revision = crosswalkRevision;
  setBusy(true); junctionInputs();
  try {
    const request = crosswalkRequest();
    if (JSON.stringify(request) !== crosswalkSnapshot) throw new Error("Preview these inputs first.");
    const options = JSON.parse(request.text) as { min: XYZ; max: XYZ };
    const cloud = await cropCloud(request.id, options.min, options.max, true);
    if (revision !== crosswalkRevision || !entries.has(request.id)) {
      await removeCloud(cloud.id);
      throw new Error("Point cloud changed; preview again before inspecting.");
    }
    const source = entries.get(request.id)!;
    const added = addEntry(cloud);
    record({ label: "the paint box", added: [added], hide: [source] });
    renderList();
    $<HTMLSelectElement>("vm-crosswalk-cloud").value = String(cloud.id);
    setStatus(`${cloud.count} paint-box points copied with their attributes; source hidden. Preview the isolated cloud again before adding. Cloud Undo restores the source. Disable EDL if sparse points appear dark.`);
  } catch (err) { setStatus(`Could not inspect paint points: ${errorText(err)}`); }
  finally { setBusy(false); junctionInputs(); }
};
