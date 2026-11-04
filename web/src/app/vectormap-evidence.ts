/** Build evidence: boundary sources and saved road edges, shown and inspected on demand. */

import * as THREE from "three";
import { $ } from "./dom";
import { globalShift, viewer } from "./state";
import { toggleTool, type Tool } from "./tools";
import {
  type BoundarySource,
  dashedPairs,
  display,
  drag,
  type EvidenceProfile,
  group,
  hint,
  local,
  materials,
  renderHint,
  type RoadEdgeSnapshot,
  segments,
  view,
  type XYZ,
} from "./vectormap";

// Build evidence is session-only, not map geometry or exported lane-marking semantics.
const sourceLabels: Record<BoundarySource,string> = {rgb_paint:"Selected RGB paint source",intensity:"Selected intensity source",curb:"Selected curb source",support_edge:"Coverage-limit candidate",width_prior:"Inferred width prior"};
const sourceColors: Record<BoundarySource,number> = {rgb_paint:0x31e4df,intensity:0xffe066,curb:0xffad58,support_edge:0x8b98a5,width_prior:0xb69cff};
const sourceMaterials = Object.fromEntries(Object.entries(sourceColors).map(([key,color])=>[key,new THREE.PointsMaterial({color,size:6,sizeAttenuation:false,depthTest:false})])) as Record<BoundarySource,THREE.PointsMaterial>;
const evidenceProfiles = () => drag?.moved ? [] : view.boundaryEvidence?.profiles ?? [];
const savedEdges = () => drag?.moved ? [] : view.boundaryEvidence?.roadEdges ?? [];
export function drawBoundaryEvidence(): void {
  const markers: Record<BoundarySource,number[]> = {rgb_paint:[],intensity:[],curb:[],support_edge:[],width_prior:[]};
  const lines:number[]=[];
  if(display.evidence) for(const p of evidenceProfiles()) {
    for(const id of p.boundary_ids) { const b=view.boundaries.find(b=>b.id===id);if(b)dashedPairs(b.points,lines,.8,.5); }
    p.evidence.forEach((label,i)=>markers[label].push(...local(label==="width_prior" ? p.geometry[i] : p.source_before_fitting[i])));
  }
  segments(lines,materials.inferred,4);
  const edges:number[]=[];
  if(display.roadEdges) for(const p of savedEdges())dashedPairs(p.geometry,edges,.12,.35);
  segments(edges,materials.savedEdge,3);
  for(const key of Object.keys(markers) as BoundarySource[]) {
    if(!markers[key].length)continue;
    const geometry=new THREE.BufferGeometry();geometry.setAttribute("position",new THREE.Float32BufferAttribute(markers[key],3));
    const dots=new THREE.Points(geometry,sourceMaterials[key]);dots.renderOrder=5;group.add(dots);
  }
}
export function renderBoundaryEvidence(): void {
  const profiles=evidenceProfiles(),edges=savedEdges();
  const counts:Record<BoundarySource,number>={rgb_paint:0,intensity:0,curb:0,support_edge:0,width_prior:0};
  for(const p of profiles)for(const e of p.evidence)counts[e]++;
  $("vm-evidence-summary").textContent=profiles.length ? `${profiles.length} build profiles: ${counts.rgb_paint} RGB paint, ${counts.intensity} intensity, ${counts.curb} curb, ${counts.support_edge} coverage-limit and ${counts.width_prior} inferred vertices. ${edges.length} saved road-edge drafts before trimming. ${view.boundaryEvidence?.limited ? "Partial snapshot: evidence budget reached. " : ""}Selected sources precede fitting; connecting lines are inferred. This is a build snapshot, not a live source audit.` : `No current build snapshot: imported/manual or edited boundaries have no observed-source label. ${view.boundaryEvidence?.limited ? "Evidence budget reached." : ""}`;
  const select=$<HTMLSelectElement>("vm-evidence-profile");select.replaceChildren();
  profiles.forEach((p,i)=>select.add(new Option(`Boundary ${p.boundary_ids.join(", ")}`,`b:${i}`)));
  edges.forEach((_,i)=>select.add(new Option(`Saved road edge ${i+1} (before trimming)`,`e:${i}`)));
  select.disabled=profiles.length+edges.length===0;
  $<HTMLButtonElement>("vm-evidence-inspect").disabled=select.disabled;
  renderEvidenceVertices();$("vm-evidence-detail").textContent="Choose a source vertex or inspect a visible dot / connector.";
}
function renderEvidenceVertices(): void {
  const [type,index]=$<HTMLSelectElement>("vm-evidence-profile").value.split(":");
  const p=type==="e" ? savedEdges()[Number(index)] : evidenceProfiles()[Number(index)];
  const select=$<HTMLSelectElement>("vm-evidence-vertex");select.replaceChildren();
  p?.evidence.forEach((label,i)=>select.add(new Option(`${i+1}: ${sourceLabels[label]}`,String(i))));select.disabled=!p;
}
function inspectEvidence(saved:boolean,profile:number,vertex:number,connector=false): void {
  const p=saved ? savedEdges()[profile] : evidenceProfiles()[profile];if(!p)return;
  $<HTMLSelectElement>("vm-evidence-profile").value=`${saved ? "e" : "b"}:${profile}`;renderEvidenceVertices();
  $<HTMLSelectElement>("vm-evidence-vertex").value=String(vertex);
  const source=p.source_before_fitting[vertex],fitted=p.geometry[vertex],label=p.evidence[vertex];
  $("vm-evidence-detail").textContent=`${saved ? "Saved road-edge draft BEFORE trimming; not an exported lane boundary. " : "Generated boundary source snapshot. "}${connector ? "Inferred connector, not continuously observed paint. Nearest endpoint: " : ""}${sourceLabels[label]}. Selected source XYZ (${source.map(v=>v.toFixed(3)).join(", ")}); fitted draft before map splitting XYZ (${fitted.map(v=>v.toFixed(3)).join(", ")}); XY movement ${Math.hypot(fitted[0]-source[0],fitted[1]-source[1]).toFixed(3)} m. ${label==="width_prior" ? "Position inferred; no observed outer paint or legal lane-width certification." : "Evidence labels the selected source before fitting, not a surveyed or legal boundary."}`;
}
$("vm-evidence-profile").onchange=()=>{renderEvidenceVertices();$("vm-evidence-vertex").dispatchEvent(new Event("change"));};
$("vm-evidence-vertex").onchange=()=>{const [type,index]=$<HTMLSelectElement>("vm-evidence-profile").value.split(":");inspectEvidence(type==="e",Number(index),Number($<HTMLSelectElement>("vm-evidence-vertex").value));};
const evidenceTool:Tool={
  click(x,y){
    const rect=$("viewport").querySelector(":scope > canvas")!.getBoundingClientRect(),shift=globalShift();
    const project=(p:XYZ)=>viewer.project(new THREE.Vector3(p[0]-shift[0],p[1]-shift[1],p[2]-shift[2]));
    let hit:{saved:boolean;profile:number;vertex:number;connector:boolean}|null=null,best=12;
    const collections:[boolean,(EvidenceProfile|RoadEdgeSnapshot)[]][]=[[false,display.evidence ? evidenceProfiles():[]],[true,display.roadEdges ? savedEdges():[]]];
    for(const [saved,rows] of collections)if(!saved)rows.forEach((p,profile)=>p.evidence.forEach((label,vertex)=>{
      const point=project(saved||label==="width_prior" ? p.geometry[vertex]:p.source_before_fitting[vertex]);if(!point)return;
      const distance=Math.hypot(point.x+rect.left-x,point.y+rect.top-y);if(distance<best){best=distance;hit={saved,profile,vertex,connector:false};}
    }));
    if(!hit)for(const [saved,rows] of collections)rows.forEach((p,profile)=>{
      const curves=saved ? [p.geometry] : (p as EvidenceProfile).boundary_ids.flatMap(id=>{const b=view.boundaries.find(b=>b.id===id);return b?[b.points]:[];});
      for(const curve of curves)for(let i=1;i<curve.length;i++){
        const a=project(curve[i-1]),b=project(curve[i]);if(!a||!b)continue;
        const dx=b.x-a.x,dy=b.y-a.y,den=dx*dx+dy*dy;if(den===0)continue;
        const t=Math.max(0,Math.min(1,((x-rect.left-a.x)*dx+(y-rect.top-a.y)*dy)/den));
        const distance=Math.hypot(a.x+t*dx+rect.left-x,a.y+t*dy+rect.top-y);
        if(distance<best){
          const q=curve[t<.5 ? i-1:i];let vertex=0,nearest=Infinity;
          p.geometry.forEach((v,j)=>{const d=Math.hypot(v[0]-q[0],v[1]-q[1]);if(d<nearest){nearest=d;vertex=j;}});
          best=distance;hit={saved,profile,vertex,connector:true};
        }
      }
    });
    if(hit){const h=hit as {saved:boolean;profile:number;vertex:number;connector:boolean};inspectEvidence(h.saved,h.profile,h.vertex,h.connector);}
    else $("vm-evidence-detail").textContent="No visible evidence dot or connector here. Enable build evidence / saved road edges, or choose a vertex from the list.";
  },
  enter(){ $("vm-evidence-inspect").setAttribute("aria-pressed","true");hint.textContent="Click a source dot or an inferred connector to inspect its build evidence."; },
  exit(){ $("vm-evidence-inspect").setAttribute("aria-pressed","false");renderHint(); }
};
$("vm-evidence-inspect").onclick=()=>toggleTool(evidenceTool);
