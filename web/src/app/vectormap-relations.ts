/** Regulatory relations: which lanes a traffic light or stop line governs. */

import * as THREE from "three";
import { vectorMap } from "../api";
import { $, errorText, fmt, setStatus } from "./dom";
import { globalShift, viewer } from "./state";
import {
  busy,
  draw,
  type Edited,
  type EquipmentRule,
  featureDrag,
  junctionInputs,
  type RelationCandidate,
  type RelationProposal,
  setBusy,
  takeView,
  view,
  type XYZ,
} from "./vectormap";

let relationPreview: RelationProposal | null = null;
let relationCandidate: string | null = null;
let relationRevision = 0;

export function chosenRelationCandidate(): RelationCandidate | undefined {
  return relationPreview && relationPreview.rule_id === relationSelection()?.id ? relationPreview.candidates.find(c=>c.key===relationCandidate) : undefined;
}
export function clearRelationPreview(): void {
  relationRevision++; relationPreview = null; relationCandidate = null;
  $("vm-relation-candidates").replaceChildren(); $("vm-relation-proposal-report").textContent = "";
  relationInputs();
}
export function relationCandidateGeometry(c: RelationCandidate): XYZ[][] {
  const lines: XYZ[][] = [];
  const rule = relationSelection();
  for (const h of view.signals) if(rule?.signals.includes(h.id)) {
    const [a,b]=[h.points[0],h.points.at(-1)];
    if(a&&b) {const hgt=h.height??0;lines.push([a,b,[b[0],b[1],b[2]+hgt],[a[0],a[1],a[2]+hgt],a]);}
  }
  for (const x of view.crosswalks) if(c.controlled_crosswalks.includes(x.id) && x.outline.length) lines.push([...x.outline,x.outline[0]]);
  for (const x of view.stopLines) if(c.stop_lines.includes(x.id)) lines.push(x.points);
  for (const x of view.lanes) if(c.lanes.includes(x.id)) lines.push(x.center);
  return lines;
}
function renderRelationPreview(): void {
  $("vm-relation-candidates").replaceChildren();
  const p=relationPreview; if(!p) return;
  $("vm-relation-proposal-report").textContent = `${p.eligible_count} supported draft candidate${p.eligible_count===1?"":"s"}; ${p.candidates.length-p.eligible_count} nearby alternatives held.${p.ambiguous?" Multiple candidates remain; inspect each before choosing.":""}${p.limited?" Preview incomplete; adoption disabled.":""} ${p.warnings.join(" ")}`;
  for (const c of p.candidates) {
    const label=document.createElement("label"); const radio=document.createElement("input");
    radio.type="radio"; radio.name="relation-candidate"; radio.value=c.key; radio.checked=c.key===relationCandidate;
    const context=c.road_context === "same_lane" ? "same road lanes" : c.road_context === "connected_lanes" ? "connected road lanes" : "road context unavailable";
    label.append(radio, `${c.target_kind.replaceAll("_"," ")} ${c.target_id}: ${c.eligible?"draft candidate":"held"}${c.already_linked?" (already linked)":""}; ${c.distance_m===null?"distance unavailable":`${fmt(c.distance_m)} m`}; ${c.axis_degrees===null?"orientation unavailable":`${fmt(c.axis_degrees)}° axis difference`}; ${context}. ${c.reasons.join(" ")}`);
    radio.onchange=()=>{relationCandidate=c.key;relationInputs();focusRelationCandidate();draw();};
    $("vm-relation-candidates").append(label);
  }
  relationInputs(); draw();
}
function focusRelationCandidate(): void {
  const c=chosenRelationCandidate();if(!c)return;
  const box=new THREE.Box3();const shift=new THREE.Vector3(...globalShift());
  for(const points of relationCandidateGeometry(c)) for(const p of points) box.expandByPoint(new THREE.Vector3(...p).sub(shift));
  if(!box.isEmpty()) viewer.frameBox(box.expandByScalar(2));
}
$("vm-relation-candidate-focus").onclick=focusRelationCandidate;
$("vm-relation-preview").onclick=async()=>{
  const rule=relationSelection();if(busy||!rule)return;
  clearRelationPreview();const revision=relationRevision;setBusy(true);junctionInputs();
  try {
    const proposal=await vectorMap<RelationProposal>("relations-preview",{id:rule.id});
    if(revision!==relationRevision||relationSelection()?.id!==rule.id)throw new Error("Map or equipment changed; preview again.");
    relationPreview=proposal;renderRelationPreview();
    setStatus(`Rule ${rule.id}: ${proposal.eligible_count} supported draft targets. Inspect and select a candidate explicitly; no map changes.`);
  } catch(err){clearRelationPreview();setStatus(`Target preview failed: ${errorText(err)}`);}
  finally {setBusy(false);junctionInputs();}
};
$("vm-relation-adopt").onclick=async()=>{
  const p=relationPreview;const c=chosenRelationCandidate();if(busy||!p||!c?.eligible||p.limited)return;
  setBusy(true);junctionInputs();
  try {
    const edited=await vectorMap<Edited>("relations-adopt",{text:JSON.stringify({rule_id:p.rule_id,map_snapshot:p.map_snapshot,candidate_key:c.key})});
    takeView(edited);const result=edited.result as {changed:boolean;warnings:string[]};
    setStatus(result.changed?`Rule ${p.rule_id} reviewed candidate adopted. ${result.warnings.join(" ")}`:"Associations unchanged; no Undo step added.");
  }catch(err){clearRelationPreview();draw();setStatus(`Candidate adoption failed: ${errorText(err)}`);}
  finally{setBusy(false);junctionInputs();}
};

export function relationSelection(): EquipmentRule | undefined {
  return view.regulatoryElements?.find(r => String(r.id) === $<HTMLSelectElement>("vm-relation").value);
}
export function relationInputs(): void {
  const r = relationSelection();
  for (const el of $("vm-relations-editor").querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button")) el.disabled = busy || !!featureDrag || (!r && el.id !== "vm-relation") || r?.kind === "mixed";
  $<HTMLSelectElement>("vm-relation").disabled = busy || !!featureDrag;
  $<HTMLInputElement>("vm-relation-stops").readOnly = r?.kind === "stop_line";
  $<HTMLButtonElement>("vm-relation-preview").disabled = busy || !!featureDrag || !(r?.kind === "vehicle" || r?.kind === "pedestrian");
  $<HTMLButtonElement>("vm-relation-candidate-focus").disabled = busy || !!featureDrag || !chosenRelationCandidate();
  $<HTMLButtonElement>("vm-relation-adopt").disabled = busy || !!featureDrag || !chosenRelationCandidate()?.eligible || !relationPreview?.map_snapshot || !!relationPreview?.limited;

}
function renderRelationFields(): void {
  const r = relationSelection(); const pedestrian = r?.kind === "pedestrian";
  $("vm-relation-lanes-row").hidden = pedestrian;
  $("vm-relation-crosswalks-row").hidden = !pedestrian;
  $("vm-relation-stops-row").hidden = pedestrian;
  $<HTMLInputElement>("vm-relation-lanes").value = pedestrian ? "" : r?.lanes.join(",") ?? "";
  $<HTMLInputElement>("vm-relation-crosswalks").value = r?.controlled_crosswalks.join(",") ?? "";
  $<HTMLInputElement>("vm-relation-stops").value = pedestrian ? "" : r?.stop_lines.join(",") ?? "";
  $("vm-relation-current").textContent = r ? `Rule ${r.id} (${r.kind}); ${r.review_source}. Current lanes: ${r.lanes.join(",") || "none"}; crosswalks: ${r.controlled_crosswalks.join(",") || "none"}; stops: ${r.stop_lines.join(",") || "none"}.${pedestrian && r.lanes.length ? " Legacy vehicle-lane references require review; Apply replaces them with the chosen crosswalk targets." : ""}` : "Choose an equipment rule explicitly.";
  $("vm-relation-targets").textContent = r ? pedestrian ? `Available crossings: ${view.crosswalks.map(c => c.id).join(", ")}. No crossing is selected automatically.` : `Available stop lines: ${view.stopLines.map(c => c.id).join(", ") || "none"}. Review a marking's lane context before attaching a vehicle signal.` : "";
  relationInputs();
}
export function renderRelations(): void {
  const select = $<HTMLSelectElement>("vm-relation"); const old = select.value;
  select.replaceChildren(new Option("Choose an equipment rule", ""), ...(view.regulatoryElements ?? []).map(r => new Option(`Rule ${r.id}: ${r.kind}${r.signals.length ? ` signal ${r.signals.join(",")}` : r.crosswalk ? ` crossing ${r.crosswalk}` : ` stop ${r.stop_lines.join(",")}`}`, String(r.id))));
  select.value = [...select.options].some(o => o.value === old) ? old : "";
  renderRelationFields();
}
$<HTMLSelectElement>("vm-relation").onchange = () => { clearRelationPreview(); renderRelationFields(); draw(); };
$("vm-relation-focus").onclick = () => {
  const r = relationSelection(); if (!r) return;
  const box = new THREE.Box3(); const shift = globalShift();
  const add = (points: XYZ[], height = 0) => { for (const p of points) { box.expandByPoint(new THREE.Vector3(p[0]-shift[0],p[1]-shift[1],p[2]-shift[2])); box.expandByPoint(new THREE.Vector3(p[0]-shift[0],p[1]-shift[1],p[2]-shift[2]+height)); } };
  for (const c of view.signals) if (r.signals.includes(c.id)) add(c.points,c.height ?? 0);
  for (const c of view.crosswalks) if (r.controlled_crosswalks.includes(c.id) || r.crosswalk === c.id) add(c.outline);
  for (const c of view.stopLines) if (r.stop_lines.includes(c.id)) add(c.points);
  for (const lane of view.lanes) if (r.lanes.includes(lane.id)) add(lane.center);
  if (!box.isEmpty()) viewer.frameBox(box.expandByScalar(2));
};
function relationIds(id: string): number[] {
  const text = $<HTMLInputElement>(id).value.trim();
  if (!text) return [];
  const parts = text.split(/[\s,]+/);
  if (parts.some(p => !/^\d+$/.test(p) || !Number.isSafeInteger(Number(p)) || Number(p) <= 0)) throw new Error("Enter positive integer IDs separated by commas or spaces.");
  return parts.map(Number);
}
$("vm-relation-apply").onclick = async () => {
  const r = relationSelection(); if (busy || !r) return;
  setBusy(true); junctionInputs();
  try {
    const edited = await vectorMap<Edited>("relations-edit", { text:JSON.stringify({rule_id:r.id,lanes:r.kind === "pedestrian" ? [] : relationIds("vm-relation-lanes"),controlled_crosswalks:r.kind === "pedestrian" ? relationIds("vm-relation-crosswalks") : [],stop_lines:r.kind === "pedestrian" ? [] : relationIds("vm-relation-stops")}) });
    takeView(edited); const report = edited.result as {changed:boolean;warnings:string[]};
    setStatus(report.changed ? `Rule ${r.id} associations updated. ${report.warnings.join(" ")}` : "Associations unchanged; no Undo step added.");
  } catch (err) { setStatus(`Association edit failed: ${errorText(err)}`); }
  finally { setBusy(false); junctionInputs(); }
};
