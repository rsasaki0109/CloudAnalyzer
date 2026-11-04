/** Source quality: lane centres and boundaries checked against the clouds. */

import * as THREE from "three";
import { vectorMap } from "../api";
import { $, errorText, fmt, setStatus } from "./dom";
import { entries, globalShift, viewer } from "./state";
import { setTool } from "./tools";
import {
  activeBoundary,
  busy,
  chooseBoundaryVertex,
  draw,
  junctionInputs,
  laneById,
  renderReviews,
  reviewSourceChanged,
  selectLane,
  setActiveBoundary,
  setBusy,
  setSourceProblems,
  setSourceProblemSelected,
  type SourceCurveSupport,
  sourceProblems,
  sourceProblemSelected,
  type SourceQualityReport,
  vertexTool,
  view,
} from "./vectormap";

export let lowSupport = new Set<number>();

let qualityRevision = 0;
export function clearQuality(): void {
  qualityRevision++;
  lowSupport = new Set();
  setSourceProblems([]); setSourceProblemSelected(null);
  $("vm-quality-locations").hidden = true;
  $<HTMLSelectElement>("vm-quality-problem").replaceChildren(new Option("Choose a problem interval", ""));
  sourceProblemInputs();
  $("vm-quality-report").textContent = "Source coverage has not been checked for the current map and cloud. Run Check source coverage after editing.";
  $("vm-quality-lanes").replaceChildren();
  renderReviews();
}
$("vm-quality-cloud").onchange = () => { clearQuality(); reviewSourceChanged(); draw(); junctionInputs(); };
$("vm-quality-check").onclick = async () => {
  if (busy) return;
  if (entries.get(Number($<HTMLSelectElement>("vm-quality-cloud").value))?.origin.displayPreview) return setStatus("Load the original full point map before running a new source check.", true);
  clearQuality(); draw(); const revision = qualityRevision;
  setBusy(true); junctionInputs(); setStatus("Checking lane centres and boundaries against source points…");
  try {
    const report = await vectorMap<SourceQualityReport>("quality", {id: Number($<HTMLSelectElement>("vm-quality-cloud").value)});
    if (revision !== qualityRevision) return;
    renderSourceQuality(report);
    setStatus(`Source coverage checked: ${report.low_support_lanes.length} lanes need review. The map is unchanged.`);
  } catch (err) { setStatus(`Could not check source coverage: ${errorText(err)}`); }
  finally { setBusy(false); junctionInputs(); }
};

function renderSourceQuality(report: SourceQualityReport, prefix = ""): void {
  $("vm-quality-report").textContent = prefix + `${report.lanes.length} lanes checked; ${report.low_support_lanes.length} need source review; ${report.omitted_lanes.length} omitted; ${report.malformed_lanes.length} malformed. ${report.limited ? "Coverage check limited. " : ""}` + report.warnings.join(" ");
  lowSupport = new Set([...report.low_support_lanes, ...report.omitted_lanes, ...report.malformed_lanes]);
  setSourceProblems(report.problems);
  $("vm-quality-locations").hidden = !sourceProblems.length && !report.problems_limited;
  $("vm-quality-location-summary").textContent = `${sourceProblems.length} problem intervals shown at checked sample locations.` +
    (report.problems_limited ? " Location preview limited; further failed samples are not displayed. Coverage figures include all checked samples." : "");
  $<HTMLSelectElement>("vm-quality-problem").replaceChildren(new Option("Choose a problem interval", ""), ...sourceProblems.map((p, i) => {
    const curve = p.curve === "center" ? "centre" : `${p.curve} boundary`;
    const reason = p.reason === "insufficient_returns" ? "insufficient returns" : "height disagreement";
    return new Option(`Lane ${p.lane}, ${curve}: ${fmt(p.from_m)}–${fmt(p.to_m)} m, ${reason}`, String(i));
  }));
  renderReviews();
  const percentage = (s: SourceCurveSupport) => `${Math.round(s.fraction*100)}%${s.start_supported && s.end_supported ? "" : " (end support missing)"}`;
  for (const lane of report.lanes.filter(l => l.needs_review)) {
    const button = document.createElement("button"); button.textContent = `Lane ${lane.lane}: centre ${percentage(lane.center)}, left ${percentage(lane.left)}, right ${percentage(lane.right)}`;
    button.onclick = () => {
      const index = sourceProblems.findIndex(p => p.lane === lane.lane);
      if (index >= 0) chooseSourceProblem(index); else selectLane(lane.lane, true);
    }; $("vm-quality-lanes").append(button);
  }
  draw();
}

/** Frozen audit display; callers must invalidate it after map/source changes. */
export function showSavedSourceQuality(report: SourceQualityReport, label: string, cloud: number): void {
  clearQuality();
  $<HTMLSelectElement>("vm-quality-cloud").value = String(cloud);
  renderSourceQuality(report, `Saved ${label} audit of the exported map. `);
  junctionInputs();
  $("vm-quality").setAttribute("open", "");
}

export function sourceProblemInputs(): void {
  const p = sourceProblemSelected === null ? undefined : sourceProblems[sourceProblemSelected];
  $("vm-quality-problem-detail").textContent = p
    ? `Lane ${p.lane}, ${p.curve === "center" ? "centre" : `${p.curve} boundary`}: ${p.reason === "insufficient_returns" ? "insufficient returns" : "height disagreement"}. ${p.points.length} failed samples at ${fmt(p.from_m)}–${fmt(p.to_m)} m along travel.`
    : "Choose an interval to inspect its source evidence.";
  $<HTMLSelectElement>("vm-quality-problem").disabled = busy || !sourceProblems.length;
  $<HTMLButtonElement>("vm-quality-next").disabled = busy || !sourceProblems.length;
  for (const id of ["vm-quality-focus", "vm-quality-edit"]) $<HTMLButtonElement>(id).disabled = busy || sourceProblemSelected === null;
}
function focusSourceProblem(forEditing = false): void {
  const p = sourceProblemSelected === null ? undefined : sourceProblems[sourceProblemSelected];
  if (!p) return;
  $<HTMLInputElement>("vm-quality-show").checked = true;
  const shift = globalShift(), box = new THREE.Box3();
  for (const point of p.points) box.expandByPoint(new THREE.Vector3(point[0] - shift[0], point[1] - shift[1], point[2] - shift[2]));
  if (forEditing) {
    const lane = laneById(p.lane), target = box.getCenter(new THREE.Vector3());
    const ids = lane ? p.curve === "center" ? [lane.leftRef.id, lane.rightRef.id] : [p.curve === "left" ? lane.leftRef.id : lane.rightRef.id] : [];
    for (const b of view.boundaries) if (ids.includes(b.id)) {
      let nearest: THREE.Vector3 | null = null, distance = Infinity;
      for (const point of b.points) {
        const v = new THREE.Vector3(point[0] - shift[0], point[1] - shift[1], point[2] - shift[2]), d = v.distanceToSquared(target);
        if (d < distance) { nearest = v; distance = d; }
      }
      if (nearest) box.expandByPoint(nearest);
    }
  }
  viewer.frameBox(box.expandByScalar(3)); draw();
}
function chooseSourceProblem(index: number | null): void {
  setSourceProblemSelected(index);
  $<HTMLSelectElement>("vm-quality-problem").value = index === null ? "" : String(index);
  const p = index === null ? undefined : sourceProblems[index];
  if (p) { selectLane(p.lane); focusSourceProblem(); } else draw();
  sourceProblemInputs();
}
$("vm-quality-show").onchange = () => draw();
$("vm-quality-problem").onchange = () => {
  const value = $<HTMLSelectElement>("vm-quality-problem").value;
  chooseSourceProblem(value === "" ? null : Number(value));
};
$("vm-quality-next").onclick = () => { if (!busy && sourceProblems.length) chooseSourceProblem(((sourceProblemSelected ?? -1) + 1) % sourceProblems.length); };
$("vm-quality-focus").onclick = () => focusSourceProblem();
$("vm-quality-edit").onclick = () => {
  const p = sourceProblemSelected === null ? undefined : sourceProblems[sourceProblemSelected];
  if (busy || !p) return;
  const lane = laneById(p.lane); if (!lane) return;
  setTool(vertexTool);
  setActiveBoundary(p.curve === "center" ? null : p.curve === "left" ? lane.leftRef.id : lane.rightRef.id);
  if (activeBoundary !== null) {
    const boundary = view.boundaries.find(b => b.id === activeBoundary);
    const middle = p.points[Math.floor(p.points.length / 2)];
    if (boundary && middle) {
      let nearest = 0, distance = Infinity;
      for (const [i, point] of boundary.points.entries()) {
        const d = Math.hypot(point[0] - middle[0], point[1] - middle[1], point[2] - middle[2]);
        if (d < distance) { nearest = i; distance = d; }
      }
      chooseBoundaryVertex(boundary.id, nearest, true);
    }
  } else chooseBoundaryVertex(null, 0, true);
  focusSourceProblem(true);
};
