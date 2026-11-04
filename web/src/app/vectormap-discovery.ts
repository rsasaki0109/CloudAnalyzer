/** Discovery of crosswalk and traffic-light candidates along the map. */

import * as THREE from "three";
import { cropCloud, removeCloud, vectorMap } from "../api";
import { addEntry, renderList } from "./entries";
import { refreshColors } from "./colors";
import { record } from "./history";
import { $, errorText, fmt, setStatus } from "./dom";
import { entries, globalShift, viewer } from "./state";
import {
  busy,
  type DiscoveryCandidate,
  type DiscoveryReport,
  draw,
  type Edited,
  featureSelect,
  junctionInputs,
  renderFeatureFields,
  selected,
  setBusy,
  takeView,
  view,
} from "./vectormap";

export let discoveryPreview: DiscoveryReport | null = null;
let discoverySnapshot: { id: number; text: string } | null = null;
let discoveryRevision = 0;

export let discoveryCandidate: number | null = null;
export const discardedCandidates = new Set<number>();
export const confirmedCandidates = new Set<number>();

export function discoveryInputs(): void {
  const c = discoveryPreview?.candidates.find(c => c.id === discoveryCandidate);
  for (const el of $("vm-discovery").querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button")) el.disabled = busy;
  for (const radio of $("vm-discovery-candidates").querySelectorAll<HTMLInputElement>("input")) radio.disabled = busy || confirmedCandidates.has(Number(radio.value));
  $<HTMLButtonElement>("vm-discovery-search").disabled = busy || (!$<HTMLSelectElement>("vm-discovery-scope").value.includes("ground_surface") && !view.lanes.length) || !$<HTMLSelectElement>("vm-discovery-cloud").value;
  for (const id of ["focus", "inspect", "reject", "selected"]) $<HTMLButtonElement>(`vm-discovery-${id}`).disabled = busy || !c || confirmedCandidates.has(c.id);
  $<HTMLButtonElement>("vm-discovery-add").disabled = busy || !c || confirmedCandidates.has(c.id) || !$<HTMLSelectElement>("vm-discovery-kind").value || !$<HTMLInputElement>("vm-discovery-lanes").value.trim();
}
export function clearDiscovery(): void {
  discoveryRevision++; discoveryPreview = null; discoverySnapshot = null; discoveryCandidate = null;
  discardedCandidates.clear(); confirmedCandidates.clear();
  $("vm-discovery-candidates").replaceChildren(); $("vm-discovery-report").textContent = "";
  $<HTMLSelectElement>("vm-discovery-kind").replaceChildren(new Option("Identify the object first", ""));
  $<HTMLInputElement>("vm-discovery-lanes").value = "";
  discoveryInputs();
}
function discoveryRequest(): { id: number; text: string } {
  const id = Number($<HTMLSelectElement>("vm-discovery-cloud").value);
  if (!entries.has(id)) throw new Error("Choose the original point cloud.");
  return { id, text: JSON.stringify({ options: { scope: $<HTMLSelectElement>("vm-discovery-scope").value, corridor_radius: Number($<HTMLInputElement>("vm-discovery-radius").value), brightness_fraction: Number($<HTMLInputElement>("vm-discovery-brightness").value)/100 } }) };
}
function focusDiscovery(): void {
  const c = discoveryPreview?.candidates.find(c => c.id === discoveryCandidate); if (!c) return;
  const shift = globalShift();
  viewer.frameBox(new THREE.Box3(new THREE.Vector3(...c.min).sub(new THREE.Vector3(...shift)), new THREE.Vector3(...c.max).sub(new THREE.Vector3(...shift))).expandByScalar(2)); draw();
}
function renderDiscovery(): void {
  $("vm-discovery-candidates").replaceChildren();
  if (!discoveryPreview) return;
  for (const c of discoveryPreview.candidates) {
    if (discardedCandidates.has(c.id)) continue;
    const label = document.createElement("label"); const radio = document.createElement("input");
    radio.type = "radio"; radio.name = "discovered-feature"; radio.value = String(c.id); radio.checked = discoveryCandidate === c.id; radio.disabled = busy || confirmedCandidates.has(c.id);
    const e = c.evidence;
    const description = e.kind === "repeated_paint" ? `${e.measurement.stripe_count} repeated paint bands` : e.kind === "bright_bar" ? `${e.transverse_to_road ? "transverse paint" : "bright paint bar"} (${fmt(e.width)} m; ${e.points} points)` : `elevated panel (${fmt(e.width)} × ${fmt(e.height)} m; ${e.points} points)`;
    label.append(radio, `#${c.id + 1} ${description}${confirmedCandidates.has(c.id) ? " — added" : " — unconfirmed"}. Nearby lane IDs: ${c.nearby_lanes.join(", ") || "none"}.`);
    radio.onchange = () => chooseDiscovery(c);
    $("vm-discovery-candidates").append(label);
  }
  const r = discoveryPreview;
  $("vm-discovery-report").textContent = `${r.candidates.length} proposals shown${r.limited ? ` of ${r.detected_candidates} detected (preview limited)` : ""}; ${r.corridor_points} search points from ${r.source_points}; ${r.windows} windows (${r.unsupported_windows} unsupported). ${discardedCandidates.size} discarded; ${confirmedCandidates.size} added. ${r.warnings.join(" ")}`;
  discoveryInputs(); draw();
}
function chooseDiscovery(c: DiscoveryCandidate): void {
  discoveryCandidate = c.id;
  const select = $<HTMLSelectElement>("vm-discovery-kind"); select.replaceChildren(new Option("Identify the object first", ""));
  if (c.evidence.kind === "repeated_paint") select.add(new Option("Confirmed crosswalk", "crosswalk"));
  else if (c.evidence.kind === "bright_bar") select.add(new Option("Confirmed stop-line marking", "stop_line"));
  else { select.add(new Option("Confirmed vehicle signal", "vehicle_signal")); select.add(new Option("Confirmed pedestrian signal", "pedestrian_signal")); }
  $<HTMLInputElement>("vm-discovery-lanes").value = ""; focusDiscovery(); discoveryInputs();
}
export async function runDiscovery(): Promise<void> {
  clearDiscovery(); const revision = discoveryRevision; const request = discoveryRequest();
  setStatus("Searching road equipment around lane geometry…");
  const report = await vectorMap<DiscoveryReport>("feature-discover", request);
  if (revision !== discoveryRevision || !entries.has(request.id)) throw new Error("Point cloud or map changed; search again.");
  discoveryPreview = report; discoverySnapshot = request;
  $("vm-discovery").setAttribute("open", ""); renderDiscovery();
}
for (const id of ["cloud", "scope", "radius", "brightness"]) $("vm-discovery-" + id).addEventListener("input", () => { clearDiscovery(); draw(); });
for (const id of ["kind", "lanes"]) $("vm-discovery-" + id).addEventListener("input", discoveryInputs);
$("vm-discovery-show").addEventListener("input", draw);
$("vm-discovery-search").onclick = async () => {
  if (busy) return; setBusy(true); junctionInputs();
  try { await runDiscovery(); setStatus(`Found ${discoveryPreview?.candidates.length ?? 0} unconfirmed equipment proposals. Inspect the points before adding.`); }
  catch (err) { clearDiscovery(); draw(); setStatus(`Equipment search failed: ${errorText(err)}`); }
  finally { setBusy(false); junctionInputs(); renderDiscovery(); }
};
$("vm-discovery-focus").onclick = focusDiscovery;
$("vm-discovery-reject").onclick = () => { if (discoveryCandidate !== null) discardedCandidates.add(discoveryCandidate); discoveryCandidate = null; $<HTMLSelectElement>("vm-discovery-kind").value = ""; $<HTMLInputElement>("vm-discovery-lanes").value = ""; renderDiscovery(); };
$("vm-discovery-selected").onclick = () => { if (selected !== null) $<HTMLInputElement>("vm-discovery-lanes").value = String(selected); discoveryInputs(); };
function discoveryState() { return { report: discoveryPreview, snapshot: discoverySnapshot, discarded: [...discardedCandidates], confirmed: [...confirmedCandidates] }; }
function restoreDiscovery(saved: ReturnType<typeof discoveryState>): void {
  discoveryPreview = saved.report; discoverySnapshot = saved.snapshot;
  discardedCandidates.clear(); saved.discarded.forEach(c => discardedCandidates.add(c));
  confirmedCandidates.clear(); saved.confirmed.forEach(c => confirmedCandidates.add(c));
  discoveryCandidate = null; renderDiscovery();
}
$("vm-discovery-add").onclick = async () => {
  if (busy || discoveryCandidate === null || !discoverySnapshot) return;
  const c = discoveryPreview?.candidates.find(c => c.id === discoveryCandidate); if (!c) return;
  setBusy(true); junctionInputs();
  try {
    const request = discoveryRequest();
    if (JSON.stringify(request) !== JSON.stringify(discoverySnapshot)) throw new Error("Search these inputs again before confirming.");
    const classification = $<HTMLSelectElement>("vm-discovery-kind").value;
    const lanes = $<HTMLInputElement>("vm-discovery-lanes").value.split(",").map(v => Number(v.trim()));
    if (!classification || !lanes.length || lanes.some(id => !Number.isSafeInteger(id) || id <= 0)) throw new Error("Confirm the object type and positive lane IDs explicitly.");
    const saved = discoveryState();
    const edited = await vectorMap<Edited>("feature-confirm", { id: request.id, text: JSON.stringify({ ...JSON.parse(request.text), confirmations: [{ candidate: c.id, key: c.key, classification, lanes }] }) });
    takeView(edited); saved.confirmed.push(c.id); restoreDiscovery(saved);
    const added = (edited.result as { classification: string; id: number; reused: boolean }[])[0];
    featureSelect.value = `${classification === "vehicle_signal" || classification === "pedestrian_signal" ? "signal" : classification}:${added.id}`; renderFeatureFields();
    setStatus(`Reviewed ${classification.replaceAll("_", " ")} ${added.id} ${added.reused ? "already exists; no changes" : "added from point-cloud evidence"}. Geometry and lane assignments can be undone together.`);
  } catch (err) { setStatus(`Could not add equipment: ${errorText(err)}`); }
  finally { setBusy(false); junctionInputs(); renderDiscovery(); }
};
$("vm-discovery-inspect").onclick = async () => {
  if (busy || !discoverySnapshot) return; const c = discoveryPreview?.candidates.find(c => c.id === discoveryCandidate); if (!c) return;
  const revision = discoveryRevision; const request = discoverySnapshot; const saved = discoveryState();
  setBusy(true); junctionInputs();
  try {
    const cloud = await cropCloud(request.id, c.min, c.max, true);
    if (revision !== discoveryRevision || !entries.has(request.id)) { await removeCloud(cloud.id); throw new Error("Source changed; search again."); }
    const source = entries.get(request.id)!; const added = addEntry(cloud);
    if (c.evidence.kind === "elevated_panel") { added.mode = "solid"; refreshColors(added); }
    record({ label: "automatic equipment proposal points", added: [added], hide: [source] }); renderList();
    $<HTMLSelectElement>("vm-discovery-cloud").value = String(request.id); restoreDiscovery(saved); chooseDiscovery(c); renderDiscovery();
    setStatus(`${cloud.count} original points isolated with attributes; source hidden. Confirm the object or discard it. Cloud Undo restores the source.`);
  } catch (err) { setStatus(`Could not inspect equipment points: ${errorText(err)}`); }
  finally { setBusy(false); junctionInputs(); renderDiscovery(); }
};
