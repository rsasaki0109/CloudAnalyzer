/** Shared map state and the three.js scene: lanes, boundaries, regulatory elements, previews and labels. */

import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { $ } from "./dom";
import { globalShift, listChanged, viewer } from "./state";
import { appendDashedPairs, crosswalkTriangles, signalTriangles } from "./vectormap-geometry";
import type { RoadBuildFailure } from "./road-build-failure";
import { resampleLaneCurve as resample } from "../lane-surface-overlap";
import {
  boundaryVertexPoint,
  chosenRelationCandidate,
  confirmedCandidates,
  crosswalkCandidate,
  crosswalkPreview,
  discardedCandidates,
  discoveryCandidate,
  discoveryPreview,
  drawBoundaryEvidence,
  featurePoints,
  junctionPreview,
  junctionSelection,
  type LaneView,
  type MapView,
  relationCandidateGeometry,
  type RelationProposal,
  relationSelection,
  signalPreview,
  type SourceProblem,
  type XYZ,
} from "./vectormap";

/** Dashes of dashed lane lines (metres). */
const DASH = 3;
const GAP = 3;
/** Direction chevrons every this many metres along a lane. */
const ARROW_EVERY = 12;
/** Clicks this far outside every lane (metres) hit none. */
export const LANE_REACH = 1.0;

export const hint = $("vm-hint");
export const status = $("vm-status");
export const issueList = $<HTMLOListElement>("vm-issues");
export const laneBox = $("vm-lane");
export const undoButton = $<HTMLButtonElement>("vm-undo");
export const exportButton = $<HTMLButtonElement>("vm-export");

export let view: MapView = { lanes: [], boundaries: [], stopLines: [], crosswalks: [], signals: [], georeferenced: false };
export function setView(value: typeof view): void { view = value; }
export let undoDepth = 0;
export function setUndoDepth(value: typeof undoDepth): void { undoDepth = value; }
export let selected: number | null = null;
export function setSelected(value: typeof selected): void { selected = value; }
export let busy = false;
export function setBusy(value: typeof busy): void { busy = value; }
export let buildFailure: {failure: RoadBuildFailure; cloud: number; trajectory: number | null; path: XYZ[] | null} | null = null;
export function setBuildFailure(value: typeof buildFailure): void { buildFailure = value; }

export let sourceProblems: SourceProblem[] = [];
export function setSourceProblems(value: typeof sourceProblems): void { sourceProblems = value; }
export let sourceProblemSelected: number | null = null;
export function setSourceProblemSelected(value: typeof sourceProblemSelected): void { sourceProblemSelected = value; }

export let relationPreview: RelationProposal | null = null;
export function setRelationPreview(value: typeof relationPreview): void { relationPreview = value; }
export let relationCandidate: string | null = null;
export function setRelationCandidate(value: typeof relationCandidate): void { relationCandidate = value; }
export let relationRevision = 0;
export function setRelationRevision(value: typeof relationRevision): void { relationRevision = value; }

/** Points clicked so far while drawing a road (original coordinates). */
export let sketch: XYZ[] = [];
export function setSketch(value: typeof sketch): void { sketch = value; }

export const group = new THREE.Group();
group.name = "vector-map";
viewer.overlay.add(group);
/** Everything is drawn relative to this (original coordinates), for float32 precision. */
let origin: XYZ = [0, 0, 0];

export const materials = {
  solid: viewer.lineMaterial({ color: 0xf5f5f5, linewidth: 2, depthTest: false }),
  dashed: viewer.lineMaterial({ color: 0xe0e0e0, linewidth: 2, depthTest: false }),
  edge: viewer.lineMaterial({ color: 0xffa726, linewidth: 2.5, depthTest: false }),
  virtual: viewer.lineMaterial({ color: 0x5d899c, linewidth: 1, depthTest: false }),
  arrow: viewer.lineMaterial({ color: 0x8de8ff, linewidth: 2.5, depthTest: false }),
  stop: viewer.lineMaterial({ color: 0xff6375, linewidth: 5, depthTest: false }),
  signal: viewer.lineMaterial({ color: 0xffd166, linewidth: 3, depthTest: false }),
  crosswalk: viewer.lineMaterial({ color: 0xffffff, linewidth: 1.5, depthTest: false }),
  sketch: viewer.lineMaterial({ color: 0xffeb3b, linewidth: 3, depthTest: false }),
  proposal: viewer.lineMaterial({ color: 0x00e5ff, linewidth: 3, depthTest: false }),
  selectedRoute: viewer.lineMaterial({ color: 0xffeb3b, linewidth: 4, depthTest: false }),
  inferred: viewer.lineMaterial({ color: 0xb69cff, linewidth: 2.5, depthTest: false }),
  savedEdge: viewer.lineMaterial({ color: 0xffad58, linewidth: 2.5, depthTest: false }),
  incomingRoute: viewer.lineMaterial({ color: 0xa894fa, linewidth: 3, depthTest: false }),
  outgoingRoute: viewer.lineMaterial({ color: 0x48dfaf, linewidth: 3, depthTest: false }),
  failure: viewer.lineMaterial({ color: 0xff5364, linewidth: 4, depthTest: false }),
  sourceMissing: viewer.lineMaterial({ color: 0xff8a3d, linewidth: 4, depthTest: false }),
  sourceHeight: viewer.lineMaterial({ color: 0xd98fff, linewidth: 4, depthTest: false }),
};
const sourceDots = {
  insufficient_returns: new THREE.PointsMaterial({color: 0xff8a3d, size: 7, sizeAttenuation: false, depthTest: false}),
  height_mismatch: new THREE.PointsMaterial({color: 0xd98fff, size: 7, sizeAttenuation: false, depthTest: false}),
};
const laneFill = new THREE.MeshBasicMaterial({
  color: 0x2485bf,
  transparent: true,
  opacity: 0.28,
  side: THREE.DoubleSide,
  depthTest: false,
  depthWrite: false,
});
const selectedFill = new THREE.MeshBasicMaterial({
  color: 0xffeb3b,
  transparent: true,
  opacity: 0.4,
  side: THREE.DoubleSide,
  depthTest: false,
  depthWrite: false,
});
const turnFill = laneFill.clone();
turnFill.color.setHex(0x18b8a6);
turnFill.opacity = 0.18;
const predecessorFill = laneFill.clone();
predecessorFill.color.setHex(0xa894fa);
predecessorFill.opacity = 0.42;
const successorFill = laneFill.clone();
successorFill.color.setHex(0x48dfaf);
successorFill.opacity = 0.42;
const walkFill = new THREE.MeshBasicMaterial({ color: 0xf3f5f0, transparent: true, side: THREE.DoubleSide, depthTest: false, depthWrite: false });
const signalFill = walkFill.clone();
signalFill.color.setHex(0xffd166);
const proposalFill = turnFill.clone();
proposalFill.color.setHex(0x00e5ff);
proposalFill.opacity = 0.22;
export const display = { evidence: false, roadEdges: false, surfaces: true, directions: true, markings: true, virtual: false, regulations: true, labels: true };
const labelLayer = $("vm-labels");
const legend = $("vm-legend");
const mapLabels: { element: HTMLElement; point: XYZ; priority: number }[] = [];
const vertexMaterial = new THREE.PointsMaterial({ color: 0xffeb3b, size: 7, sizeAttenuation: false, transparent: true, depthTest: false });
let editingVertices = false;
export function setEditingVertices(value: typeof editingVertices): void { editingVertices = value; }
export let editingFeature = false;
export function setEditingFeature(value: typeof editingFeature): void { editingFeature = value; }
export let featureDrag: { before: MapView; index: number; point: XYZ; offset: [number, number]; start: [number, number]; moved: boolean } | null = null;
export function setFeatureDrag(value: typeof featureDrag): void { featureDrag = value; }
export let activeBoundary: number | null = null;
export function setActiveBoundary(value: typeof activeBoundary): void { activeBoundary = value; }
export let boundaryVertex: { boundary: number; index: number } | null = null;
export function setBoundaryVertex(value: typeof boundaryVertex): void { boundaryVertex = value; }
export let drag: {
  before: MapView; boundary: number; index: number; point: XYZ;
  offset: [number, number]; start: [number, number]; moved: boolean;
} | null = null;
export function setDrag(value: typeof drag): void { drag = value; }

function clearGroup(): void {
  for (const child of [...group.children]) {
    group.remove(child);
    if (child instanceof LineSegments2 || child instanceof THREE.Mesh || child instanceof THREE.Points) child.geometry.dispose();
  }
}

export const local = (p: XYZ): XYZ => [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];

export function segments(pairs: number[], material: THREE.Material, order = 1): void {
  if (pairs.length === 0) return;
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(pairs);
  const lines = new LineSegments2(geometry, material as never);
  lines.renderOrder = order;
  group.add(lines);
}

/** A polyline as segment pairs (local coordinates). */
function polylinePairs(points: XYZ[], out: number[]): void {
  for (let i = 1; i < points.length; i++) out.push(...local(points[i - 1]), ...local(points[i]));
}

let dashDisplayLimited = false;
/** A polyline cut into bounded display dashes. */
export function dashedPairs(points: XYZ[], out: number[], dash = DASH, gap = GAP): void {
  if (!appendDashedPairs(points, origin, out, dash, gap)) dashDisplayLimited = true;
}

/** The area between a lane's boundaries as a triangle strip. */
function laneMesh(lane: Pick<LaneView, "left" | "right">, material: THREE.Material): THREE.Mesh {
  const n = Math.max(lane.left.length, lane.right.length, 2);
  const left = resample(lane.left, n);
  const right = resample(lane.right, n);
  const positions: number[] = [];
  for (let i = 1; i < Math.min(left.length, right.length); i++) {
    const [l0, l1, r0, r1] = [left[i - 1], left[i], right[i - 1], right[i]].map(local);
    positions.push(...l0, ...r0, ...l1, ...l1, ...r0, ...r1);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 0;
  return mesh;
}

/** Chevrons pointing along a lane's centreline. */
function arrowPairs(lane: LaneView, out: number[]): void {
  const c = lane.center;
  if (c.length < 2) return;
  let length = 0;
  for (let i = 1; i < c.length; i++) length += Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]);
  const count = Math.max(1, Math.floor(length / ARROW_EVERY));
  const samples = resample(c, Math.max(2, count * 8 + 1));
  for (let k = 0; k < count; k++) {
    const i = Math.round(((k + 0.5) / count) * (samples.length - 1));
    const [p, q] = [samples[Math.max(0, i - 1)], samples[Math.min(samples.length - 1, i + 1)]];
    const dx = q[0] - p[0];
    const dy = q[1] - p[1];
    const d = Math.hypot(dx, dy) || 1;
    const [ux, uy] = [dx / d, dy / d];
    const tip: XYZ = samples[i];
    const size = 0.9;
    for (const side of [1, -1]) {
      const back: XYZ = [tip[0] - ux * size - side * uy * size * 0.7, tip[1] - uy * size + side * ux * size * 0.7, tip[2]];
      out.push(...local(back), ...local(tip));
    }
  }
}

export function draw(): void {
  dashDisplayLimited = false;
  clearGroup();
  const shift = globalShift();
  const first = view.boundaries[0]?.points[0] ?? sketch[0] ?? buildFailure?.failure.diagnostic.reference[0];
  if (first) origin = [first[0], first[1], first[2]];
  group.position.set(origin[0] - shift[0], origin[1] - shift[1], origin[2] - shift[2]);
  if (buildFailure) {
    const d = buildFailure.failure.diagnostic;
    const failed: number[] = [];
    for (const line of d.boundaries) polylinePairs(line, failed);
    segments(failed, materials.failure, 6);
    const marker: number[] = [];
    const [x, y, z] = d.location;
    polylinePairs([[x - 1, y - 1, z], [x + 1, y + 1, z]], marker);
    polylinePairs([[x - 1, y + 1, z], [x + 1, y - 1, z]], marker);
    segments(marker, materials.sketch, 7);
  }
  const focused = view.lanes.find((l) => l.id === selected);
  if (display.surfaces) for (const lane of view.lanes) {
    const material = lane.id === selected ? selectedFill : focused?.successors.includes(lane.id) ? successorFill :
      focused?.predecessors.includes(lane.id) ? predecessorFill : lane.turn && lane.turn !== "straight" ? turnFill : laneFill;
    group.add(laneMesh(lane, material));
  }
  const annotated = new Set(display.evidence && !drag?.moved ? view.boundaryEvidence?.profiles.flatMap(p => p.boundary_ids) : []);
  const unknownEvidence: number[] = [];
  const byKind: Record<"solid" | "dashed" | "edge" | "virtual", number[]> = { solid: [], dashed: [], edge: [], virtual: [] };
  if (display.markings) for (const b of view.boundaries) {
    if (annotated.has(b.id)) continue;
    if (display.evidence) { dashedPairs(b.points,unknownEvidence,.8,.5); continue; }
    const type = b.kind.type;
    if (type === "lane_marking" && b.kind.pattern === "dashed") dashedPairs(b.points, byKind.dashed);
    else if (type === "lane_marking") polylinePairs(b.points, byKind.solid);
    else if (type === "curb" || type === "road_edge") polylinePairs(b.points, byKind.edge);
    else if (display.virtual) polylinePairs(b.points, byKind.virtual);
  }
  for (const kind of ["solid", "dashed", "edge", "virtual"] as const) segments(byKind[kind], materials[kind]);
  segments(unknownEvidence,materials.virtual,3);
  drawBoundaryEvidence();
  drawSourceProblems();
  $("vm-evidence-render-limit").hidden = !dashDisplayLimited;
  const arrows: number[] = [];
  if (display.directions) for (const lane of view.lanes) arrowPairs(lane, arrows);
  segments(arrows, materials.arrow, 2);
  if (focused && display.directions) {
    for (const [ids, material] of [
      [[focused.id], materials.selectedRoute], [focused.predecessors, materials.incomingRoute],
      [focused.successors, materials.outgoingRoute],
    ] as const) {
      const route: number[] = [];
      for (const lane of view.lanes) if (ids.includes(lane.id)) polylinePairs(lane.center, route);
      segments(route, material, 2);
    }
  }
  const stops: number[] = [];
  if (display.regulations) for (const s of view.stopLines) polylinePairs(s.points, stops);
  segments(stops, materials.stop, 3);
  const walks: number[] = [];
  const stripes: number[] = [];
  if (display.regulations) for (const c of view.crosswalks) {
    if (c.outline.length < 3) continue;
    polylinePairs([...c.outline, c.outline[0]], walks);
    for (const value of crosswalkTriangles(c.outline, origin, c.paintBands)) stripes.push(value);
  }
  triangles(stripes, walkFill, 3);
  segments(walks, materials.crosswalk, 3);
  const signals: number[] = [];
  const faces: number[] = [];
  if (display.regulations) for (const s of view.signals) {
    polylinePairs(s.points, signals);
    faces.push(...signalTriangles(s.points, s.height, origin));
    const h = s.height;
    if (h === null || !Number.isFinite(h) || h <= 0) continue;
    polylinePairs(
      s.points.map(([x, y, z]) => [x, y, z + h] as XYZ),
      signals,
    );
    for (const p of [s.points[0], s.points.at(-1)]) {
      if (p) polylinePairs([p, [p[0], p[1], p[2] + h]], signals);
    }
  }
  triangles(faces, signalFill, 3);
  segments(signals, materials.signal, 3);
  if (signalPreview) {
    const [a, b] = signalPreview.geometry;
    const raised = ([x, y, z]: XYZ): XYZ => [x, y, z + signalPreview!.height];
    const outline: number[] = [];
    polylinePairs([a, b, raised(b), raised(a), a], outline);
    segments(outline, materials.proposal, 4);
  }
  if (crosswalkPreview && crosswalkCandidate !== null) {
    const candidate = crosswalkPreview.candidates[crosswalkCandidate];
    const outline: number[] = [];
    for (const p of [candidate.outline, ...candidate.stripes]) polylinePairs([...p, p[0]], outline);
    segments(outline, materials.proposal, 4);
  }
  if (discoveryPreview && $<HTMLInputElement>("vm-discovery-show").checked) {
    const chosen: number[] = []; const other: number[] = [];
    for (const c of discoveryPreview.candidates) {
      if (discardedCandidates.has(c.id) || confirmedCandidates.has(c.id)) continue;
      const target = c.id === discoveryCandidate ? chosen : other;
      const e = c.evidence;
      if (e.kind === "repeated_paint") for (const p of [e.measurement.outline, ...e.measurement.stripes]) polylinePairs([...p, p[0]], target);
      else if (e.kind === "bright_bar") polylinePairs(e.geometry, target);
      else { const [a,b] = e.geometry; const raised = (p: XYZ): XYZ => [p[0], p[1], p[2] + e.height]; polylinePairs([a,b,raised(b),raised(a),a], target); }
    }
    segments(other, materials.virtual, 4); segments(chosen, materials.proposal, 4);
  }
  const relation = chosenRelationCandidate();
  if (relation) {
    const highlighted: number[] = [];
    for (const points of relationCandidateGeometry(relation)) polylinePairs(points, highlighted);
    segments(highlighted, materials.proposal, 5);
    const source = view.signals.find(h => relationSelection()?.signals.includes(h.id));
    const target = relation.target_kind === "crosswalk" ? view.crosswalks.find(c=>c.id===relation.target_id)?.outline : view.stopLines.find(c=>c.id===relation.target_id)?.points;
    if (source?.points.length && target?.length) {
      const average = (points: XYZ[]): XYZ => [0,1,2].map(i=>points.reduce((sum,p)=>sum+p[i],0)/points.length) as XYZ;
      const link: number[]=[]; dashedPairs([average(source.points),average(target)],link);
      segments(link, materials.proposal, 5);
    }
  }
  if (junctionPreview) {
    const chosen: number[] = [];
    const other: number[] = [];
    junctionPreview.candidates.forEach((candidate, index) => {
      if (junctionSelection.has(index)) {
        group.add(laneMesh(candidate, proposalFill));
        polylinePairs(candidate.left, chosen);
        polylinePairs(candidate.right, chosen);
        dashedPairs(candidate.center, chosen);
      } else polylinePairs(candidate.center, other);
    });
    segments(other, materials.virtual, 3);
    segments(chosen, materials.proposal, 4);
  }
  if (sketch.length > 0) {
    const pairs: number[] = [];
    polylinePairs(sketch, pairs);
    if (sketch.length === 1) pairs.push(...local(sketch[0]), ...local([sketch[0][0] + 0.3, sketch[0][1], sketch[0][2]]));
    segments(pairs, materials.sketch, 4);
  }
  if (editingVertices || editingFeature) {
    const points = editingFeature ? featurePoints().flatMap(local) : view.boundaries.flatMap((b) => b.points.flatMap(local));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
    const dots = new THREE.Points(geometry, vertexMaterial);
    dots.renderOrder = 5;
    group.add(dots);
    const boundary = view.boundaries.find((b) => b.id === activeBoundary);
    if (boundary) {
      const pairs: number[] = [];
      polylinePairs(boundary.points, pairs);
      segments(pairs, materials.sketch, 4);
    }
  }
  const chosenVertex = boundaryVertexPoint();
  if (chosenVertex && !editingFeature) {
    const [x, y, z] = chosenVertex, marker: number[] = [];
    polylinePairs([[x - .4, y, z], [x + .4, y, z]], marker);
    polylinePairs([[x, y - .4, z], [x, y + .4, z]], marker);
    segments(marker, materials.sketch, 6);
  }
  buildMapLabels();
  legend.hidden = view.lanes.length === 0;
  $("vm-evidence-legend").hidden = !display.evidence && !display.roadEdges;
  $("vm-route-legend").hidden = !focused || !display.surfaces;
  viewer.requestRender();
}

function drawSourceProblems(): void {
  if (!$<HTMLInputElement>("vm-quality-show").checked || drag?.moved) return;
  for (const reason of ["insufficient_returns", "height_mismatch"] as const) {
    const pairs: number[] = [], points: number[] = [];
    for (const p of sourceProblems) if (p.reason === reason) {
      polylinePairs(p.points, pairs);
      for (const point of p.points) points.push(...local(point));
    }
    segments(pairs, reason === "insufficient_returns" ? materials.sourceMissing : materials.sourceHeight, 6);
    if (points.length) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
      const dots = new THREE.Points(geometry, sourceDots[reason]); dots.renderOrder = 6; group.add(dots);
    }
  }
  const selected = sourceProblemSelected === null ? undefined : sourceProblems[sourceProblemSelected];
  if (selected) {
    const pairs: number[] = []; polylinePairs(selected.points, pairs);
    const [x, y, z] = selected.points[0];
    polylinePairs([[x-.4, y-.4, z], [x+.4, y+.4, z]], pairs);
    polylinePairs([[x-.4, y+.4, z], [x+.4, y-.4, z]], pairs);
    segments(pairs, materials.sketch, 7);
  }
}

function triangles(positions: number[], material: THREE.Material, order: number): void {
  if (!positions.length) return;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = order;
  group.add(mesh);
}

function buildMapLabels(): void {
  labelLayer.replaceChildren();
  mapLabels.length = 0;
  if (!display.labels) return;
  const add = (text: string, point: XYZ, kind: string, priority = 1) => {
    const element = document.createElement("span");
    element.className = `vm-map-label ${kind}`;
    element.textContent = text;
    element.hidden = true;
    labelLayer.append(element);
    mapLabels.push({ element, point, priority });
  };
  const lane = view.lanes.find((l) => l.id === selected);
  if (lane?.center.length) {
    const center = resample(lane.center, 3)[1];
    add(`Lane ${lane.id}${lane.turn ? ` · ${lane.turn}` : ""}${lane.speedLimit === null ? "" : ` · ${lane.speedLimit} km/h`}`, center, "selected", 0);
  }
  if (!display.regulations) return;
  for (const signal of view.signals) {
    if (!signal.points.length) continue;
    const mid = resample(signal.points, 3)[1];
    add(`Signal ${signal.id}`, [mid[0], mid[1], mid[2] + (signal.height ?? 0)], "signal");
  }
  for (const walk of view.crosswalks) {
    if (!walk.outline.length) continue;
    const center = walk.outline.reduce((a, p) => a.map((v, k) => v + p[k] / walk.outline.length) as XYZ, [0, 0, 0] as XYZ);
    add(`Crosswalk ${walk.id}`, center, "crosswalk", 2);
  }
}

viewer.afterRenderListeners.add(() => {
  const shift = globalShift();
  const viewport = $("viewport");
  const occupied: { x: number; y: number; width: number; height: number }[] = [];
  const sorted = mapLabels.map((label) => ({ ...label, at: viewer.project(new THREE.Vector3(
    label.point[0] - shift[0], label.point[1] - shift[1], label.point[2] - shift[2],
  )) })).sort((a, b) => a.priority - b.priority ||
    (a.at ? Math.hypot(a.at.x - viewport.clientWidth / 2, a.at.y - viewport.clientHeight / 2) : Infinity) -
    (b.at ? Math.hypot(b.at.x - viewport.clientWidth / 2, b.at.y - viewport.clientHeight / 2) : Infinity));
  for (const { element, at } of sorted) {
    element.hidden = true;
    if (!at || occupied.length >= 16) continue;
    // Measure before hiding to use actual text widths, not an assumed label size.
    element.hidden = false;
    const width = element.offsetWidth, height = element.offsetHeight;
    const x = at.x - width / 2, y = at.y - height - 10;
    if (x < 8 || y < 8 || x + width > viewport.clientWidth - 8 || y + height > viewport.clientHeight - 64 ||
      occupied.some((b) => x < b.x + b.width + 6 && x + width + 6 > b.x && y < b.y + b.height + 6 && y + height + 6 > b.y)) {
      element.hidden = true;
      continue;
    }
    element.style.left = `${x}px`;
    element.style.top = `${y}px`;
    occupied.push({ x, y, width, height });
  }
});

for (const key of Object.keys(display) as (keyof typeof display)[]) {
  $<HTMLInputElement>(`vm-show-${key}`).onchange = (event) => {
    display[key] = (event.target as HTMLInputElement).checked;
    draw();
  };
}
$<HTMLInputElement>("vm-context").oninput = (event) => viewer.setCloudBrightness(Number((event.target as HTMLInputElement).value) / 100);

listChanged.add(() => draw());
