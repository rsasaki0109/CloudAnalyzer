/** Junction preview and apply: turning lanes and stop lines for chosen candidates. */

import * as THREE from "three";
import { vectorMap } from "../api";
import { $, errorText, fmt, setStatus } from "./dom";
import { entries, globalShift, viewer } from "./state";
import { setTool } from "./tools";
import {
  boundaryInputs,
  busy,
  crosswalkInputs,
  discoveryInputs,
  draw,
  type Edited,
  featureInputs,
  type JunctionReport,
  relationInputs,
  setBusy,
  signalInputs,
  sourceProblemInputs,
  takeView,
  view,
} from "./vectormap";

export let junctionPreview: JunctionReport | null = null;
export const junctionSelection = new Set<number>();
let junctionSnapshot: { id: number; text: string } | null = null;
let junctionRevision = 0;

export const junctionCloud = $<HTMLSelectElement>("vm-junction-cloud");
const junctionGap = $<HTMLInputElement>("vm-junction-gap");
const junctionSupport = $<HTMLInputElement>("vm-junction-support");
const junctionBoundaries = $<HTMLInputElement>("vm-junction-boundaries");
const junctionPreviewButton = $<HTMLButtonElement>("vm-junction-preview");
const junctionApplyButton = $<HTMLButtonElement>("vm-junction-apply");
const junctionAll = $<HTMLButtonElement>("vm-junction-all");
const junctionNone = $<HTMLButtonElement>("vm-junction-none");
const junctionList = $("vm-junction-candidates");
export function junctionInputs(): void {
  $<HTMLSelectElement>("vm-quality-cloud").disabled = busy;
  const qualityCloud = $<HTMLSelectElement>("vm-quality-cloud").value;
  $<HTMLButtonElement>("vm-quality-check").disabled = busy || !qualityCloud || !view.lanes.length || entries.get(Number(qualityCloud))?.origin.displayPreview === true;
  sourceProblemInputs();
  discoveryInputs();
  featureInputs();
  boundaryInputs();
  relationInputs();
  signalInputs();
  crosswalkInputs();
  junctionCloud.disabled = junctionGap.disabled = junctionSupport.disabled = junctionBoundaries.disabled = busy;
  junctionPreviewButton.disabled = busy || !junctionCloud.value || !view.lanes.length;
  junctionApplyButton.disabled = busy || !junctionPreview || !junctionSelection.size;
  junctionAll.disabled = junctionNone.disabled = busy || !junctionPreview?.candidates.length;
}
export function clearJunctionPreview(): void {
  junctionRevision++;
  junctionPreview = null;
  junctionSnapshot = null;
  junctionSelection.clear();
  junctionList.replaceChildren();
  $("vm-junction-report").textContent = "Preview connections against the selected point cloud.";
  junctionInputs();
}
function renderJunctionSelection(): void {
  if (!junctionPreview) return;
  $("vm-junction-report").textContent = `${junctionPreview.candidates.length} ground-supported candidates; ${junctionSelection.size} selected. ` +
    junctionPreview.warnings.join(" ");
  junctionList.querySelectorAll<HTMLInputElement>("input").forEach((input) => {
    input.checked = junctionSelection.has(Number(input.value));
  });
  junctionInputs();
  draw();
}
for (const input of [junctionCloud, junctionGap, junctionSupport, junctionBoundaries]) input.onchange = () => {
  clearJunctionPreview();
  draw();
};
junctionAll.onclick = () => {
  if (busy) return;
  junctionPreview?.candidates.forEach((_, index) => junctionSelection.add(index));
  renderJunctionSelection();
};
junctionNone.onclick = () => {
  if (busy) return;
  junctionSelection.clear();
  renderJunctionSelection();
};
junctionPreviewButton.onclick = async () => {
  if (busy || !junctionCloud.value) return;
  setTool(null);
  clearJunctionPreview();
  const snapshot = {
    id: Number(junctionCloud.value),
    text: JSON.stringify({ max_gap: Number(junctionGap.value), min_ground_support: Number(junctionSupport.value) / 100, check_boundary_support: junctionBoundaries.checked }),
  };
  const revision = junctionRevision;
  setBusy(true);
  junctionInputs();
  setStatus("Finding ground-supported junction drafts…");
  try {
    const proposal = await vectorMap<JunctionReport>("junction-preview", snapshot);
    if (revision !== junctionRevision) return;
    junctionPreview = proposal;
    junctionSnapshot = snapshot;
    junctionPreview.candidates.forEach((candidate, index) => {
      junctionSelection.add(index);
      const label = document.createElement("label");
      label.style.display = "block";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = String(index);
      checkbox.checked = true;
      checkbox.onchange = () => {
        if (busy) { checkbox.checked = junctionSelection.has(index); return; }
        if (checkbox.checked) junctionSelection.add(index); else junctionSelection.delete(index);
        renderJunctionSelection();
      };
      const boundaries = candidate.boundary_support ? `, boundaries ${candidate.boundary_support.map(s => `${Math.round(s * 100)}%`).join(" / ")}` : "";
      label.append(checkbox, ` ${candidate.from} → ${candidate.to}: ${fmt(candidate.gap)} m, ${Math.round(candidate.turn_degrees)}°, ground ${Math.round(candidate.ground_support * 100)}%${boundaries}${candidate.ambiguous ? "; shared branch" : ""}`);
      const focus = document.createElement("button");
      focus.textContent = "Show";
      focus.onclick = () => {
        const shift = globalShift();
        const box = new THREE.Box3();
        for (const p of [...candidate.left, ...candidate.right]) box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
        box.expandByScalar(5);
        viewer.frameBox(box);
      };
      label.append(focus);
      junctionList.append(label);
    });
    renderJunctionSelection();
    setStatus("Cyan connections are draft candidates. Review permitted turns, lane width and clearance before adding them.");
  } catch (err) {
    clearJunctionPreview();
    setStatus(`Could not preview junctions: ${errorText(err)}`);
  } finally {
    setBusy(false);
    junctionInputs();
  }
};
junctionApplyButton.onclick = async () => {
  if (busy || !junctionPreview || !junctionSnapshot || !junctionSelection.size) return;
  const pairs: [number, number][] = [...junctionSelection].map((index) => {
    const candidate = junctionPreview!.candidates[index];
    return [candidate.from, candidate.to];
  });
  setBusy(true);
  junctionInputs();
  try {
    const edited = await vectorMap<Edited>("junction-connect", { ...junctionSnapshot, pairs });
    const report = edited.result as JunctionReport;
    takeView(edited);
    setStatus(`${report.added.length} draft connections added. Review traffic rules and clearance before export. Undo removes this batch.`);
  } catch (err) {
    setStatus(`Could not add junctions: ${errorText(err)}`);
  } finally {
    setBusy(false);
    junctionInputs();
  }
};
