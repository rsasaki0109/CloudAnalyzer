/** Road building from a cloud and trajectory, and the report of a failed build. */

import * as THREE from "three";
import { vectorMap } from "../api";
import { $, errorText, fmt, setStatus } from "./dom";
import { clouds, entries, globalShift, listChanged, pointsInvalidated, viewer } from "./state";
import { inputTrajectories, trajectoryChanged } from "./trajectory";
import { setTool } from "./tools";
import { parseRoadBuildFailure, type RoadBuildFailure } from "./road-build-failure";
import {
  type BuildReport,
  busy,
  clearCrosswalkPreview,
  clearDiscovery,
  clearJunctionPreview,
  clearQuality,
  clearSignalPreview,
  discoveryPreview,
  draw,
  type Edited,
  finishRoad,
  junctionCloud,
  junctionInputs,
  type PaintBudgetInfo,
  reviewSourceChanged,
  runDiscovery,
  setBusy,
  sketch,
  takeView,
  view,
  type XYZ,
} from "./vectormap";

export let buildFailure: {failure: RoadBuildFailure; cloud: number; trajectory: number | null; path: XYZ[] | null} | null = null;
export let buildFailureRevision = 0;
export let buildSketchRevision = 0;

const buildButton = $<HTMLButtonElement>("vm-build");
export const cloudInput = $<HTMLSelectElement>("vm-cloud");
const trajectoryInput = $<HTMLSelectElement>("vm-trajectory");

export function buildInputs(): void {
  const fill = (select: HTMLSelectElement, items: { id: number; name: string }[]) => {
    const value = select.value;
    select.replaceChildren(...items.map((item) => new Option(item.name, String(item.id))));
    if (items.some((item) => String(item.id) === value)) select.value = value;
  };
  fill(cloudInput, clouds().map((entry) => entry.cloud));
  fill(junctionCloud, clouds().map((entry) => entry.cloud));
  fill($<HTMLSelectElement>("vm-quality-cloud"), clouds().map((entry) => entry.cloud));
  fill($<HTMLSelectElement>("vm-signal-cloud"), clouds().map((entry) => entry.cloud));
  fill($<HTMLSelectElement>("vm-crosswalk-cloud"), clouds().map((entry) => entry.cloud));
  fill($<HTMLSelectElement>("vm-discovery-cloud"), clouds().map((entry) => entry.cloud));
  fill(trajectoryInput, inputTrajectories());
  buildButton.disabled = busy || !cloudInput.value || !trajectoryInput.value;
  $<HTMLButtonElement>("vm-failure-retry").disabled = busy || !buildFailure;
  junctionInputs();
}
listChanged.add(() => { clearBuildFailure(); clearQuality(); clearDiscovery(); clearJunctionPreview(); clearSignalPreview(); clearCrosswalkPreview(); buildInputs(); draw(); });
pointsInvalidated.add((id) => { if (id === buildFailure?.cloud || id === Number(cloudInput.value)) clearBuildFailure(); if (id === Number($<HTMLSelectElement>("vm-quality-cloud").value)) reviewSourceChanged(); clearQuality(); clearDiscovery(); clearJunctionPreview(); clearSignalPreview(); clearCrosswalkPreview(); draw(); });
trajectoryChanged.add(() => { clearBuildFailure(); buildInputs(); draw(); });
cloudInput.addEventListener("change", () => { clearBuildFailure(); draw(); });
trajectoryInput.addEventListener("change", () => { clearBuildFailure(); draw(); });
buildInputs();

export function clearBuildFailure(): void {
  buildFailureRevision++;
  buildSketchRevision++;
  buildFailure = null;
  $("vm-build-failure").hidden = true;
  $("vm-failure-stale").hidden = true;
}

function focusBuildFailure(): void {
  if (!buildFailure) return;
  const d = buildFailure.failure.diagnostic, shift = globalShift(), box = new THREE.Box3();
  for (const p of [d.location, ...d.reference, ...d.boundaries.flat()]) box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
  viewer.frameBox(box.expandByScalar(5));
}

export function showBuildFailure(err: unknown, revision: number, cloud: number, trajectory: number | null, path: XYZ[] | null = null): string {
  const text = errorText(err), failure = parseRoadBuildFailure(text);
  if (!failure) return text;
  if (revision !== buildFailureRevision || !entries.has(cloud)) return failure.message;
  buildFailure = {failure, cloud, trajectory, path};
  const d = failure.diagnostic;
  buildButton.closest('details')!.open = true;
  $("vm-build-failure").hidden = false;
  $("vm-failure-reason").textContent = `Boundary travel direction is inconsistent in this ${fmt(d.context_length)} m section. Review the red preview against the source points.`;
  $("vm-failure-settings").textContent = `Rejected settings: ${d.forward_lanes} forward / ${d.backward_lanes} backward lanes, ${fmt(d.lane_width)} m per lane, ${fmt(d.segment_length)} m pieces.`;
  draw(); focusBuildFailure();
  return failure.message;
}

$("vm-failure-focus").onclick = focusBuildFailure;
$("vm-failure-options").onclick = () => {
  $<HTMLDetailsElement>("vm-road-options").open = true;
  const input = $<HTMLInputElement>("vm-forward"); input.focus(); input.select();
};
$("vm-failure-dismiss").onclick = () => { clearBuildFailure(); draw(); };
$("vm-failure-retry").onclick = () => {
  if (busy || !buildFailure) return;
  if (buildFailure.trajectory !== null) buildButton.click();
  else if (JSON.stringify(sketch) === JSON.stringify(buildFailure.path)) void finishRoad();
  else { clearBuildFailure(); draw(); setStatus("Trace the path again before rebuilding."); }
};
for (const panel of [$("vm-road-options"), buildButton.parentElement!]) {
  panel.addEventListener("input", event => {
    if ((event.target as HTMLElement).closest("#vm-build-failure") || event.target === cloudInput || event.target === trajectoryInput) return;
    buildFailureRevision++;
    if (buildFailure) $("vm-failure-stale").hidden = false;
  });
}
export function roadBuildOptions(): object {
  return {
    forward_lanes: Number($<HTMLInputElement>("vm-forward").value),
    backward_lanes: Number($<HTMLInputElement>("vm-backward").value),
    left_hand_traffic: $<HTMLSelectElement>("vm-traffic").value === "left",
    lane_width: Number($<HTMLInputElement>("vm-width").value),
    search_margin: Number($<HTMLInputElement>("vm-search-margin").value),
    speed_limit: Number($<HTMLInputElement>("vm-speed").value),
    segment_length: Number($<HTMLInputElement>("vm-segment").value),
    anchor_width_prior: $<HTMLInputElement>("vm-anchor-prior").checked,
    physical_anchors_only: $<HTMLInputElement>("vm-physical-anchors").checked,
    align_trace_to_curbs: $<HTMLInputElement>("vm-align-curbs").checked,
    infer_lane_edges: $<HTMLInputElement>("vm-lane-edges").checked,
    fit_paint_divider: $<HTMLInputElement>("vm-paint-divider").checked,
    fit_paint_corridor: $<HTMLInputElement>("vm-paint-corridor").checked,
    ...($<HTMLSelectElement>("vm-paint-channel").value === "intensity" ? {paint_channel:"intensity"} : {}),
    track_boundaries: $<HTMLInputElement>("vm-track-boundaries").checked,
    fit_boundaries: $<HTMLInputElement>("vm-fit-boundaries").checked,
    fit_source_surface: $<HTMLInputElement>("vm-source-surface").checked,
    verify_curb_profiles: $<HTMLInputElement>("vm-verify-curbs").checked,
    merge_repeated_passes: $<HTMLInputElement>("vm-merge-passes").checked,
  };
}

export function checkRoadSearchMargin(): boolean {
  const width = Number($<HTMLInputElement>("vm-width").value), input = $<HTMLInputElement>("vm-search-margin"), margin = Number(input.value);
  // Leave unrelated invalid options to the native validator.
  if (width >= 1.5 && width <= 6 && (!Number.isFinite(margin) || margin < 0 || margin > width * 0.45)) {
    $<HTMLDetailsElement>("vm-road-options").open = true;
    input.focus(); input.select();
    setStatus(`Boundary search margin must be between 0 and ${fmt(width * 0.45)} m for a ${fmt(width)} m lane. Adjust it before rebuilding.`);
    return false;
  }
  return true;
}
function paintBudgetText(report: PaintBudgetInfo): string {
  if (!report.budget_stage) return "";
  const names: Record<string,string> = {
    roi_points: "paint region", bright_candidates: "bright candidates",
    contrast_neighbours: "nearby paint contrast", trace_ground_neighbours: "ground near the trace",
    paint_points: "paint candidates", component_neighbours: "paint connections",
    boundary_ground_neighbours: "ground at a boundary",
  };
  const q = report.budget_query;
  return `Search limit at ${names[report.budget_stage] ?? report.budget_stage.replaceAll("_"," ")}${q ? `: ${q.candidate_points} potential points for a ${fmt(q.radius_m)} m radius (limit ${q.limit})` : ""}. `;
}
function paintCandidateText(report: PaintBudgetInfo): string {
  const d = report.candidate_diagnostics;
  if (!d) return "";
  const reasons: [string,number][] = [
    ["local support missing", d.local_ground_missing], ["local height mismatch", d.local_height_mismatch],
    ["trace support missing", d.trace_ground_missing], ["trace height mismatch", d.trace_height_mismatch],
    ["flank support missing", d.flank_support_missing], ["flank contrast insufficient", d.flank_contrast_insufficient],
  ];
  const classified = d.accepted + reasons.reduce((sum,[,count]) => sum + count, 0);
  return `Bright candidate checks (${d.complete ? "complete scan" : "incomplete scan"}): ${d.accepted}/${d.bright_candidates} accepted for component checks. ` +
    `Rejected: ${reasons.map(([name,count]) => `${name} ${count}`).join(", ")}. ` +
    (!d.complete ? `${d.bright_candidates - classified} pending when the scan stopped; later candidates unexamined. ` : "") +
    "These checks do not classify a return as road paint. ";
}
export function renderBuildReport(report: BuildReport): void {
  $("vm-build-report").textContent = `${report.roads} road stretches, ${report.lanes} lanes, ${fmt(report.generated_length)} m. ` +
    `Added ${fmt(report.added_length)} m; reused ${fmt(report.reused_length)} m of existing lanes. ` +
    `Measured sources before fitting, left to right: ${report.observed_fraction.map(f => `${Math.round(f*100)}%`).join(", ")}. ` +
    `Tracking changed ${report.tracked_vertices} sources; fitted ${report.fitted_vertices} vertices (maximum XY movement ${fmt(report.maximum_fit_displacement)} m). ` +
    (report.trace_alignment ? `Trace alignment ${report.trace_alignment.applied ? "applied" : "held"}: XY shift (${fmt(report.trace_alignment.shift_xy[0])}, ${fmt(report.trace_alignment.shift_xy[1])}) m; paired curbs ${report.trace_alignment.curb_pair_sections}/${report.trace_alignment.sampled_sections} sections. ${report.trace_alignment.reason}. ` : "") +
    (report.lane_edge_inference ? `Outer lane-edge inference ${report.lane_edge_inference.applied ? "applied" : "held"}${report.lane_edge_inference.limited ? " (scan limit reached)" : ""}: ${report.lane_edge_inference.reason}. Configured width ${fmt(report.lane_edge_inference.configured_lane_width_m)} m. ` + report.lane_edge_inference.sides.map(s => `${s.boundary_slot === 0 ? "Left" : "Right"}: ${s.applied ? `${s.inferred_vertices_before_trimming} inferred vertices before footprint trimming; maximum movement ${fmt(s.maximum_movement_m)} m` : s.reason}. `).join("") : "") +
    (report.paint_divider ? `Interior paint correction ${report.paint_divider.applied ? "applied" : "held"}${report.paint_divider.limited ? " (scan limit reached)" : ""}: ${report.paint_divider.reason}. ${report.paint_divider.source_channel === "intensity" ? "Source: retained intensity. " : ""}${paintBudgetText(report.paint_divider)}${paintCandidateText(report.paint_divider)}Curb pairs: ${report.paint_divider.curb_pair_sections}/${report.paint_divider.sampled_sections}. ` + (report.paint_divider.applied && report.paint_divider.track ? `Maximum divider movement ${fmt(report.paint_divider.maximum_divider_movement_m)} m. Paint lengths before footprint trimming (observed / interpolated / extended): ${fmt(report.paint_divider.track.observed_length_m)} / ${fmt(report.paint_divider.track.interpolated_length_m)} / ${fmt(report.paint_divider.track.extrapolated_length_m)} m. ` : "") : "") +
    (report.paint_corridor ? `White paint fit ${report.paint_corridor.applied ? "applied" : "held"}${report.paint_corridor.limited ? " (scan limit reached)" : ""}: ${report.paint_corridor.reason}. ${report.paint_corridor.source_channel === "intensity" ? `Source: retained intensity; ROI normalization P10/P99.9 ${report.paint_corridor.intensity_range?.map(fmt).join(" / ") ?? "unavailable"}. ` : ""}${paintBudgetText(report.paint_corridor)}${paintCandidateText(report.paint_corridor)}` + (report.paint_corridor.applied ? `Measured widths: ${report.paint_corridor.measured_lane_widths_m.map(fmt).join(", ")} m. Source track lengths before footprint trimming (observed / interpolated / extended), left to right: ${report.paint_corridor.tracks.map(t => `${fmt(t.observed_length_m)} / ${fmt(t.interpolated_length_m)} / ${fmt(t.extrapolated_length_m)} m`).join("; ")}. ` : "") : "") +
    `Coverage-edge anchor candidates ignored: ${report.coverage_edge_anchor_candidates_ignored}. ` +
    (report.surface_fit ? `Source footprint: ${fmt(report.surface_fit.deferred_length_m)} m deferred; inferred lane widths ${fmt(report.surface_fit.minimum_lane_width_m ?? 0)}–${fmt(report.surface_fit.maximum_lane_width_m ?? 0)} m. ` : "") +report.warnings.join(" ");
}
buildButton.onclick = async () => {
  if (busy) return;
  const trajectory = inputTrajectories().find((t) => String(t.id) === trajectoryInput.value);
  if (!trajectory || !cloudInput.value) return;
  if (!checkRoadSearchMargin()) return;
  const cloud = Number(cloudInput.value);
  clearBuildFailure(); draw();
  setBusy(true);
  buildInputs();
  setTool(null);
  const revision = buildFailureRevision;
  setStatus("Building draft roads from the point cloud and trajectory…");
  try {
    const options = roadBuildOptions();
    const edited = await vectorMap<Edited>("build", {
      id: cloud, positions: trajectory.poses.positions, text: JSON.stringify(options),
    });
    takeView(edited);
    const report = edited.result as BuildReport;
    renderBuildReport(report);
    setStatus(report.lanes ? "Draft roads added. Review the boundaries, lane directions and junctions before export." : "Existing lanes matched; no new geometry added. Review the report before export.");
    if ($<HTMLInputElement>("vm-discover-after-build").checked && view.lanes.length) {
      $<HTMLSelectElement>("vm-discovery-cloud").value = cloudInput.value;
      const built = report.lanes ? "Draft roads added." : "Existing lanes matched; no new geometry added.";
      try { await runDiscovery(); setStatus(`${built} Equipment search found ${discoveryPreview?.candidates.length ?? 0} unconfirmed proposals; inspect classification and lane associations.`); }
      catch (err) { setStatus(`${built} Equipment search failed: ${errorText(err)}`); }
    }
  } catch (err) {
    setStatus(`Could not build draft roads: ${showBuildFailure(err, revision, cloud, trajectory.id)}`);
  } finally {
    setBusy(false);
    buildInputs();
  }
};
