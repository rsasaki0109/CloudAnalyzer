/** Cell-scoped pair inspection; independent from the manual all-visible-cloud profile. */
import * as THREE from "three";
import { workerBusy } from "../api";
import { SECTION_INPUT_BYTES, validateSection, type SectionSeries, type SectionSettings } from "../distance-section";
import type { ExpectedOverlap } from "../distance-regions";
import { projectChanged } from "../project-change";
import { $, download, errorText, fmt, setStatus } from "./dom";
import { addReportSection } from "./report";
import { distanceChanged, entries, listChanged, pointsInvalidated, toRender, viewer, type Entry } from "./state";
import { endTask, startTask, taskActive } from "./tasks";
import "./distance-section.css";

interface Snapshot { entry: Entry; cloud: Entry["cloud"]; name: string }
interface Context {
  source: Snapshot; reference: Snapshot; regionRank: number; cellRank: number;
  bounds: readonly number[]; overlap: ExpectedOverlap | null; isCurrent: () => boolean;
}
type Result = { source: SectionSeries; reference: SectionSeries };
const axis = $<HTMLSelectElement>("regions-section-axis"), length = $<HTMLInputElement>("regions-section-length"), width = $<HTMLInputElement>("regions-section-width");
const panel = $("regions-section-plot"), canvas = $<HTMLCanvasElement>("regions-section-canvas");
const equal = $<HTMLInputElement>("regions-section-equal");
let context: Context | null = null, last: { context: Context; settings: SectionSettings; result: Result } | null = null;
let generation = 0, running = false, cancelPending: (() => void) | null = null;
let overlay: THREE.LineSegments | null = null;
function valid(c: Context): boolean {
  return c.isCurrent() && [c.source, c.reference].every(s => entries.get(s.cloud.id) === s.entry && s.entry.cloud === s.cloud);
}
export function clearDistanceSection(): void {
  const had = last !== null, pending = cancelPending !== null; generation++; last = null; context = null; cancelPending?.(); panel.hidden = true;
  if (overlay) { viewer.scene.remove(overlay); overlay.geometry.dispose(); (overlay.material as THREE.Material).dispose(); overlay = null; }
  if (had) distanceChanged.emit();
  if (had || pending) setStatus("Section comparison cleared; compare a cell to inspect the current inputs/settings.");
}
for (const input of [axis, length, width]) input.addEventListener(input === axis ? "change" : "input", clearDistanceSection);
$("regions-section-close").onclick = clearDistanceSection;
listChanged.add(() => { if (context && !valid(context)) clearDistanceSection(); });
pointsInvalidated.add(id => { if (context && [context.source.cloud.id, context.reference.cloud.id].includes(id)) clearDistanceSection(); });
function compute(c: Context, settings: SectionSettings, signal: AbortSignal): Promise<Result> {
  const a = c.source.cloud.positions.slice(), b = c.reference.cloud.positions.slice();
  const worker = new Worker(new URL("../distance-section-worker.ts", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    const cleanup = () => { worker.terminate(); signal.removeEventListener("abort", abort); cancelPending = null; };
    const abort = () => { cleanup(); reject(new DOMException("Cancelled", "AbortError")); };
    cancelPending = abort;
    worker.onmessage = e => { cleanup(); e.data.error ? reject(new Error(e.data.error)) : resolve(e.data.result); };
    worker.onerror = () => { cleanup(); reject(new Error("Section comparison worker failed")); };
    signal.addEventListener("abort", abort, { once: true }); if (signal.aborted) { abort(); return; }
    try { worker.postMessage({ source: { positions: a, shift: c.source.cloud.shift }, reference: { positions: b, shift: c.reference.cloud.shift }, settings }, [a.buffer, b.buffer]); }
    catch (error) { cleanup(); reject(error); }
  });
}
export async function inspectDistanceSection(c: Context): Promise<void> {
  if (running || taskActive() || workerBusy() || !valid(c)) return;
  const settings: SectionSettings = { axis: axis.value as "x" | "y", center: [c.bounds[0] + (c.bounds[3] - c.bounds[0]) / 2, c.bounds[1] + (c.bounds[4] - c.bounds[1]) / 2],
    length: Number(length.value), width: Number(width.value) };
  try {
    validateSection(settings);
    if (c.source.cloud.positions.byteLength + c.reference.cloud.positions.byteLength > SECTION_INPUT_BYTES) throw new Error("Section input exceeds 64 MiB combined XYZ; crop a smaller pair");
  } catch (error) { setStatus(errorText(error), true); return; }
  clearDistanceSection(); context = c; const token = generation, signal = startTask(); running = true;
  axis.disabled = length.disabled = width.disabled = true;
  try {
    setStatus("Comparing source/reference section using all loaded points within the band…");
    const result = await compute(c, settings, signal); signal.throwIfAborted();
    if (token !== generation || !valid(c)) throw new Error("Section inputs/settings changed; compare the cell again");
    last = { context: c, settings, result }; panel.hidden = false;
    $("regions-section-title").textContent = `Region ${c.regionRank}, cell ${c.cellRank} · ${settings.axis.toUpperCase()} section · length ${fmt(settings.length)}, width ${fmt(settings.width)}`;
    $("regions-section-summary").textContent = `Orange: ${c.source.name} (${result.source.selectedPoints.toLocaleString()}); blue: ${c.reference.name} (${result.reference.selectedPoints.toLocaleString()}). Invalid loaded positions: ${result.source.invalidPoints}/${result.reference.invalidPoints}. All band points shown, including points below the distance threshold. Empty series indicate no loaded points in this band, not confirmed missing map data. Axes use cloud units; automatic height scale unless 1:1 is checked.`;
    draw(); drawBand(c, settings, result); distanceChanged.emit();
    setStatus("Section comparison ready; original points from the selected pair only. No nearest-point height correspondence is inferred.");
  } catch (error) { if (token === generation) { clearDistanceSection(); setStatus("Section comparison failed: " + (signal.aborted ? "Cancelled" : errorText(error)), true); } }
  finally { running = false; axis.disabled = length.disabled = width.disabled = false; endTask(signal); }
}
function drawBand(c: Context, s: SectionSettings, result: Result): void {
  const dx = (s.axis === "x" ? s.length : s.width) / 2, dy = (s.axis === "y" ? s.length : s.width) / 2;
  const [x, y] = s.center, z = c.bounds[2] + (c.bounds[5] - c.bounds[2]) / 2;
  const corners = [[x-dx,y-dy,z],[x+dx,y-dy,z],[x+dx,y+dy,z],[x-dx,y+dy,z]];
  const vertices = corners.flatMap((p, i) => [toRender(p), toRender(corners[(i+1)%4])]);
  vertices.push(toRender(s.axis === "x" ? [x-dx,y,z] : [x,y-dy,z]), toRender(s.axis === "x" ? [x+dx,y,z] : [x,y+dy,z]));
  overlay = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(vertices), new THREE.LineBasicMaterial({ color: "#e1e6ee", depthTest: false, depthWrite: false, transparent: true, opacity: .8 }));
  overlay.renderOrder = 999; viewer.scene.add(overlay);
  const zs = [result.source.zMin, result.source.zMax, result.reference.zMin, result.reference.zMax].filter((v): v is number => v !== null);
  viewer.frameBox(new THREE.Box3(toRender([x-dx,y-dy,zs.length ? Math.min(...zs) : c.bounds[2]]), toRender([x+dx,y+dy,zs.length ? Math.max(...zs) : c.bounds[5]])).expandByScalar(s.width));
  projectChanged();
}
function draw(): void {
  if (!last || !valid(last.context) || panel.hidden) return;
  const cw = canvas.clientWidth, ch = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
  const ctx = canvas.getContext("2d")!; ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,cw,ch);
  const left = 70, top = 12, w = Math.max(10,cw-left-15), h = Math.max(10,ch-top-35);
  const zs = [last.result.source.zMin,last.result.source.zMax,last.result.reference.zMin,last.result.reference.zMax].filter((v):v is number=>v!==null);
  let x0 = 0, x1 = last.settings.length, y0 = zs.length ? Math.min(...zs) : 0, y1 = zs.length ? Math.max(...zs) : 1;
  const pad = Math.max((y1-y0)*.08,.001); y0 -= pad; y1 += pad;
  if (equal.checked) {
    const scale = Math.max((x1-x0)/w,(y1-y0)/h), cx = (x0+x1)/2, cy = (y0+y1)/2;
    x0=cx-scale*w/2; x1=cx+scale*w/2; y0=cy-scale*h/2; y1=cy+scale*h/2;
  }
  const px = (v:number)=>left+(v-x0)/(x1-x0)*w, py=(v:number)=>top+(y1-v)/(y1-y0)*h;
  ctx.font="11px system-ui"; ctx.fillStyle="#a8b3c5"; ctx.strokeStyle="#475060";
  ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,top+h);ctx.lineTo(left+w,top+h);ctx.stroke();
  ctx.textAlign="center";
  for(const t of [x0,(x0+x1)/2,x1])ctx.fillText(fmt(t),px(t),top+h+14);
  ctx.fillText(`Along ${last.settings.axis.toUpperCase()} (cloud units)`,left+w/2,ch-3);
  ctx.textAlign="right";for(const t of [y0,(y0+y1)/2,y1])ctx.fillText(fmt(t),left-6,py(t)+4);
  ctx.save();ctx.translate(14,top+h/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillText("Height Z",0,0);ctx.restore();
  ctx.save();ctx.beginPath();ctx.rect(left,top,w,h);ctx.clip();ctx.globalAlpha=.7;
  for(const [s,color] of [[last.result.reference,"#58c4ff"],[last.result.source,"#ffbd59"]] as const){
    ctx.fillStyle=color;for(let i=0;i<s.along.length;i++)ctx.fillRect(px(s.along[i])-1.5,py(s.positions[i*3+2])-1.5,3,3);
  }
  ctx.restore();
}
equal.onchange=draw; new ResizeObserver(draw).observe(canvas);
const describe = (s:Snapshot) => ({ id:s.cloud.id,name:s.name,loadedPoints:s.cloud.count,filePoints:s.cloud.filePoints,keepEvery:s.cloud.keepEvery,
  preview:s.entry.origin.displayPreview===true,copcLevels:s.cloud.copcLevels });
$("regions-section-json").onclick=()=>{
  if(!last||!valid(last.context))return;
  const summarize=(s:SectionSeries)=>({points:s.points,finitePoints:s.finitePoints,invalidPoints:s.invalidPoints,selectedPoints:s.selectedPoints,zMin:s.zMin,zMax:s.zMax});
  const c=last.context;
  download(new Blob([JSON.stringify({schema:"cloudanalyzer.distance_section.v1",source:{...describe(c.source),...summarize(last.result.source)},
    reference:{...describe(c.reference),...summarize(last.result.reference)},regionRank:c.regionRank,cellRank:c.cellRank,cellBounds:c.bounds,comparisonOverlap:c.overlap,settings:last.settings,
    scope:"All finite loaded points inside a closed straight XY band, including non-exceeding points; only the selected pair, irrespective of visibility. The context band is not clipped to comparisonOverlap. No sampling or nearest-point height correspondence; not a registration or missing-map accuracy score."},null,2)],{type:"application/json"}),"distance-section-summary.json");
};
$("regions-section-csv").onclick=()=>{
  if(!last||!valid(last.context))return;
  const rows=["cloud_role,cloud_id,distance_along,x,y,z"];
  for(const role of ["source","reference"] as const){const s=last.result[role],id=last.context[role].cloud.id;
    for(let i=0;i<s.along.length;i++)rows.push([role,id,s.along[i],...s.positions.subarray(i*3,i*3+3)].join(","));}
  download(new Blob([rows.join("\n")+"\n"],{type:"text/csv"}),"distance-section-points.csv");
};
addReportSection("distance-section",()=>last&&valid(last.context)?{title:`Section: ${last.context.source.name} / ${last.context.reference.name}`,
  metrics:{sourcePoints:{label:"Source points in section",value:last.result.source.selectedPoints},referencePoints:{label:"Reference points in section",value:last.result.reference.selectedPoints}},
  notes:[`Region ${last.context.regionRank}, cell ${last.context.cellRank}; ${last.settings.axis.toUpperCase()} length ${last.settings.length}, width ${last.settings.width}; all loaded band points, not just threshold exceedances.`,
    last.context.overlap ? `Parent comparison overlap XY [${last.context.overlap.join(", ")}]; the context band is not clipped to it.` : "Parent comparison uses the full loaded query cloud.",
    "Raw heights with no nearest-point correspondence. Empty bands, different sampling or different surfaces do not establish missing-map or registration accuracy."]}:null);
