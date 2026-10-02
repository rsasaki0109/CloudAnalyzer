/**
 * Vector map panel: roads drawn over the clouds become lanes (both
 * directions, shared boundaries, connected pieces), joined through
 * junctions with turning lanes and given stop lines, traffic lights and
 * crosswalks; the map is checked for Autoware and saved as
 * lanelet2_map.osm with map_projector_info.yaml. The map lives in the
 * worker (vectormap-rs); this module draws its view and turns clicks into
 * vectormap commands.
 */

import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { cropCloud, removeCloud, vectorMap } from "../api";
import { addEntry, renderList } from "./entries";
import { refreshColors } from "./colors";
import { record } from "./history";
import { $, download, errorText, fmt, setStatus } from "./dom";
import { clouds, entries, globalShift, listChanged, pointsInvalidated, viewer } from "./state";
import { inputTrajectories, trajectoryChanged } from "./trajectory";
import { activeTool, pickPoint, setTool, toggleTool, type Tool } from "./tools";
import { crosswalkTriangles, signalTriangles } from "./vectormap-geometry";

type XYZ = [number, number, number];

interface Neighbor {
  lane: number;
  direction?: "same" | "opposite";
}

interface LaneView {
  id: number;
  kind: string;
  left: XYZ[];
  right: XYZ[];
  leftRef: { id: number; reversed: boolean };
  rightRef: { id: number; reversed: boolean };
  center: XYZ[];
  successors: number[];
  predecessors: number[];
  leftNeighbor: Neighbor | null;
  rightNeighbor: Neighbor | null;
  speedLimit: number | null;
  turn: string | null;
  oneWay: boolean;
}

interface BoundaryView {
  id: number;
  kind: { type: string; pattern?: string };
  points: XYZ[];
}

export interface MapView {
  lanes: LaneView[];
  boundaries: BoundaryView[];
  stopLines: { id: number; points: XYZ[]; geometrySource?: string }[];
  crosswalks: { id: number; outline: XYZ[]; paintBands?: XYZ[][] | null; editable?: boolean; geometrySource?: string }[];
  signals: { id: number; points: XYZ[]; height: number | null; geometrySource?: string }[];
  georeferenced: boolean;
}

interface Issue {
  severity: "info" | "warning" | "error";
  code: string;
  message: string;
  entity?: { kind: string; id: number };
}

interface Nearest {
  lane: number;
  distance: number;
  station: number;
  inside: boolean;
}

interface JunctionCandidate {
  from: number;
  to: number;
  gap: number;
  turn_degrees: number;
  ground_support: number;
  boundary_support?: [number, number];
  ambiguous: boolean;
  center: XYZ[];
  left: XYZ[];
  right: XYZ[];
}
interface JunctionReport {
  candidates: JunctionCandidate[];
  added: number[];
  warnings: string[];
}
interface SignalReport {
  geometry: XYZ[]; height: number; width: number; thickness: number; points: number;
  plane_rms: number; added: number | null; reused: number | null; warnings: string[];
}
interface CrosswalkReport {
  candidates: { outline: XYZ[]; left_edge: XYZ[]; right_edge: XYZ[]; stripes: XYZ[][]; stripe_count: number; width: number; length: number; angle_degrees: number; score: number }[];
  points: number; ground_points: number; plane_rms: number; brightness_source: string;
  local_profile_seeds: number; local_profiles_limited: boolean; component_bands: number; component_bands_limited: boolean;
  added: number | null; reused: number | null; warnings: string[];
}
type DiscoveryEvidence =
  | { kind: "repeated_paint"; measurement: CrosswalkReport["candidates"][number] }
  | { kind: "bright_bar"; transverse_to_road: boolean; geometry: XYZ[]; width: number; thickness: number; points: number }
  | { kind: "elevated_panel"; geometry: XYZ[]; height: number; width: number; thickness: number; points: number; plane_rms: number };
interface DiscoveryCandidate { id: number; key: string; min: XYZ; max: XYZ; nearby_lanes: number[]; evidence: DiscoveryEvidence }
interface DiscoveryReport { candidates: DiscoveryCandidate[]; detected_candidates: number; limited: boolean; source_points: number; corridor_points: number; windows: number; unsupported_windows: number; paint_refinement_windows: number; paint_refinement_unsupported_windows: number; paint_refinement_limited: boolean; warnings: string[] }

/** Dashes of dashed lane lines (metres). */
const DASH = 3;
const GAP = 3;
/** Direction chevrons every this many metres along a lane. */
const ARROW_EVERY = 12;
/** Clicks this far outside every lane (metres) hit none. */
const LANE_REACH = 1.0;

const hint = $("vm-hint");
const status = $("vm-status");
const issueList = $<HTMLOListElement>("vm-issues");
const laneBox = $("vm-lane");
const undoButton = $<HTMLButtonElement>("vm-undo");
const exportButton = $<HTMLButtonElement>("vm-export");

let view: MapView = { lanes: [], boundaries: [], stopLines: [], crosswalks: [], signals: [], georeferenced: false };
let undoDepth = 0;
let selected: number | null = null;
let busy = false;
let junctionPreview: JunctionReport | null = null;
const junctionSelection = new Set<number>();
let junctionSnapshot: { id: number; text: string } | null = null;
let junctionRevision = 0;
let signalPreview: SignalReport | null = null;
let signalSnapshot: string | null = null;
let signalRevision = 0;
let crosswalkPreview: CrosswalkReport | null = null;
let crosswalkSnapshot: string | null = null;
let crosswalkRevision = 0;
let crosswalkCandidate: number | null = null;
let discoveryPreview: DiscoveryReport | null = null;
let discoverySnapshot: { id: number; text: string } | null = null;
let discoveryRevision = 0;
let discoveryCandidate: number | null = null;
const discardedCandidates = new Set<number>();
const confirmedCandidates = new Set<number>();
/** Points clicked so far while drawing a road (original coordinates). */
let sketch: XYZ[] = [];

const group = new THREE.Group();
group.name = "vector-map";
viewer.overlay.add(group);
/** Everything is drawn relative to this (original coordinates), for float32 precision. */
let origin: XYZ = [0, 0, 0];

const materials = {
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
  incomingRoute: viewer.lineMaterial({ color: 0xa894fa, linewidth: 3, depthTest: false }),
  outgoingRoute: viewer.lineMaterial({ color: 0x48dfaf, linewidth: 3, depthTest: false }),
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
const display = { surfaces: true, directions: true, markings: true, virtual: false, regulations: true, labels: true };
const labelLayer = $("vm-labels");
const legend = $("vm-legend");
const mapLabels: { element: HTMLElement; point: XYZ; priority: number }[] = [];
const vertexMaterial = new THREE.PointsMaterial({ color: 0xffeb3b, size: 7, sizeAttenuation: false, transparent: true, depthTest: false });
let editingVertices = false;
let editingFeature = false;
let featureDrag: { before: MapView; index: number; point: XYZ; offset: [number, number]; start: [number, number]; moved: boolean } | null = null;
let activeBoundary: number | null = null;
let drag: {
  before: MapView; boundary: number; index: number; point: XYZ;
  offset: [number, number]; start: [number, number]; moved: boolean;
} | null = null;

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function clearGroup(): void {
  for (const child of [...group.children]) {
    group.remove(child);
    if (child instanceof LineSegments2 || child instanceof THREE.Mesh || child instanceof THREE.Points) child.geometry.dispose();
  }
}

const local = (p: XYZ): XYZ => [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];

function segments(pairs: number[], material: THREE.Material, order = 1): void {
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

/** A polyline cut into dashes. */
function dashedPairs(points: XYZ[], out: number[]): void {
  let along = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1];
    const b = points[i];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    let s = 0;
    while (s < length) {
      const phase = along % (DASH + GAP);
      const step = Math.min(length - s, phase < DASH ? DASH - phase : DASH + GAP - phase);
      if (phase < DASH && step > 0) {
        const t0 = s / length;
        const t1 = (s + step) / length;
        const at = (t: number): XYZ => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
        out.push(...local(at(t0)), ...local(at(t1)));
      }
      s += step;
      along += step;
      if (step <= 0) break;
    }
  }
}

/** Resample a polyline to `n` points evenly spaced along it. */
function resample(points: XYZ[], n: number): XYZ[] {
  if (points.length === 0) return [];
  if (points.length === 1) return Array.from({ length: n }, () => [...points[0]] as XYZ);
  const cumulative = [0];
  for (let i = 1; i < points.length; i++) {
    const [a, b] = [points[i - 1], points[i]];
    cumulative.push(cumulative[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  const total = cumulative[cumulative.length - 1];
  const out: XYZ[] = [];
  let j = 1;
  for (let k = 0; k < n; k++) {
    const s = (total * k) / (n - 1);
    while (j < points.length - 1 && cumulative[j] < s) j++;
    const span = cumulative[j] - cumulative[j - 1] || 1;
    const t = Math.min(1, Math.max(0, (s - cumulative[j - 1]) / span));
    const [a, b] = [points[j - 1], points[j]];
    out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
  }
  return out;
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

function draw(): void {
  clearGroup();
  const shift = globalShift();
  const first = view.boundaries[0]?.points[0] ?? sketch[0];
  if (first) origin = [first[0], first[1], first[2]];
  group.position.set(origin[0] - shift[0], origin[1] - shift[1], origin[2] - shift[2]);
  const focused = view.lanes.find((l) => l.id === selected);
  if (display.surfaces) for (const lane of view.lanes) {
    const material = lane.id === selected ? selectedFill : focused?.successors.includes(lane.id) ? successorFill :
      focused?.predecessors.includes(lane.id) ? predecessorFill : lane.turn && lane.turn !== "straight" ? turnFill : laneFill;
    group.add(laneMesh(lane, material));
  }
  const byKind: Record<"solid" | "dashed" | "edge" | "virtual", number[]> = { solid: [], dashed: [], edge: [], virtual: [] };
  if (display.markings) for (const b of view.boundaries) {
    const type = b.kind.type;
    if (type === "lane_marking" && b.kind.pattern === "dashed") dashedPairs(b.points, byKind.dashed);
    else if (type === "lane_marking") polylinePairs(b.points, byKind.solid);
    else if (type === "curb" || type === "road_edge") polylinePairs(b.points, byKind.edge);
    else if (display.virtual) polylinePairs(b.points, byKind.virtual);
  }
  for (const kind of ["solid", "dashed", "edge", "virtual"] as const) segments(byKind[kind], materials[kind]);
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
  buildMapLabels();
  legend.hidden = view.lanes.length === 0;
  $("vm-route-legend").hidden = !focused || !display.surfaces;
  viewer.requestRender();
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

// ---------------------------------------------------------------------------
// Map state
// ---------------------------------------------------------------------------

interface Edited {
  result: unknown;
  view: MapView;
  undo: number;
}

interface BuildReport {
  surface_fit?: { deferred_length_m: number; minimum_lane_width_m: number | null; maximum_lane_width_m: number | null } | null;
  roads: number;
  lanes: number;
  generated_length: number;
  added_length: number;
  reused_length: number;
  tracked_vertices: number;
  fitted_vertices: number;
  maximum_fit_displacement: number;
  observed_fraction: number[];
  warnings: string[];
}

const buildButton = $<HTMLButtonElement>("vm-build");
const cloudInput = $<HTMLSelectElement>("vm-cloud");
const trajectoryInput = $<HTMLSelectElement>("vm-trajectory");
const junctionCloud = $<HTMLSelectElement>("vm-junction-cloud");
const junctionGap = $<HTMLInputElement>("vm-junction-gap");
const junctionSupport = $<HTMLInputElement>("vm-junction-support");
const junctionBoundaries = $<HTMLInputElement>("vm-junction-boundaries");
const junctionPreviewButton = $<HTMLButtonElement>("vm-junction-preview");
const junctionApplyButton = $<HTMLButtonElement>("vm-junction-apply");
const junctionAll = $<HTMLButtonElement>("vm-junction-all");
const junctionNone = $<HTMLButtonElement>("vm-junction-none");
const junctionList = $("vm-junction-candidates");
function junctionInputs(): void {
  $<HTMLSelectElement>("vm-quality-cloud").disabled = busy;
  $<HTMLButtonElement>("vm-quality-check").disabled = busy || !$<HTMLSelectElement>("vm-quality-cloud").value || !view.lanes.length;
  discoveryInputs();
  featureInputs();
  signalInputs();
  crosswalkInputs();
  junctionCloud.disabled = junctionGap.disabled = junctionSupport.disabled = junctionBoundaries.disabled = busy;
  junctionPreviewButton.disabled = busy || !junctionCloud.value || !view.lanes.length;
  junctionApplyButton.disabled = busy || !junctionPreview || !junctionSelection.size;
  junctionAll.disabled = junctionNone.disabled = busy || !junctionPreview?.candidates.length;
}
function clearJunctionPreview(): void {
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
  busy = true;
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
    busy = false;
    junctionInputs();
  }
};
junctionApplyButton.onclick = async () => {
  if (busy || !junctionPreview || !junctionSnapshot || !junctionSelection.size) return;
  const pairs: [number, number][] = [...junctionSelection].map((index) => {
    const candidate = junctionPreview!.candidates[index];
    return [candidate.from, candidate.to];
  });
  busy = true;
  junctionInputs();
  try {
    const edited = await vectorMap<Edited>("junction-connect", { ...junctionSnapshot, pairs });
    const report = edited.result as JunctionReport;
    takeView(edited);
    setStatus(`${report.added.length} draft connections added. Review traffic rules and clearance before export. Undo removes this batch.`);
  } catch (err) {
    setStatus(`Could not add junctions: ${errorText(err)}`);
  } finally {
    busy = false;
    junctionInputs();
  }
};
function signalInputs(): void {
  for (const el of $("vm-signal-preview").closest("details")!.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button")) el.disabled = busy;
  $<HTMLButtonElement>("vm-signal-add").disabled = busy || !signalPreview || !signalSnapshot;
  $<HTMLButtonElement>("vm-signal-inspect").disabled = busy || !signalPreview || !signalSnapshot;
}
function clearSignalPreview(): void {
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
    busy = true;
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
  } finally { busy = false; junctionInputs(); }
}
$("vm-signal-preview").onclick = () => { void measureSignal(true); };
$("vm-signal-add").onclick = () => { void measureSignal(false); };
$("vm-signal-inspect").onclick = async () => {
  if (busy || !signalPreview || !signalSnapshot) return;
  const revision = signalRevision;
  busy = true; junctionInputs();
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
  finally { busy = false; junctionInputs(); }
};

function crosswalkInputs(): void {
  for (const el of $("vm-crosswalk-measure").querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button")) el.disabled = busy;
  $<HTMLButtonElement>("vm-crosswalk-add").disabled = busy || !crosswalkPreview || !crosswalkSnapshot || crosswalkCandidate === null || !$<HTMLInputElement>("vm-crosswalk-lanes").value.trim();
  $<HTMLButtonElement>("vm-crosswalk-inspect").disabled = busy || !crosswalkPreview || !crosswalkSnapshot;
}
function clearCrosswalkPreview(): void {
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
    setTool(null); busy = true; junctionInputs();
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
  } finally { busy = false; junctionInputs(); }
}
$("vm-crosswalk-preview").onclick = () => { void measureCrosswalk(true); };
$("vm-crosswalk-add").onclick = () => { void measureCrosswalk(false); };
$("vm-crosswalk-inspect").onclick = async () => {
  if (busy || !crosswalkPreview || !crosswalkSnapshot) return;
  const revision = crosswalkRevision;
  busy = true; junctionInputs();
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
  finally { busy = false; junctionInputs(); }
};

function discoveryInputs(): void {
  const c = discoveryPreview?.candidates.find(c => c.id === discoveryCandidate);
  for (const el of $("vm-discovery").querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button")) el.disabled = busy;
  for (const radio of $("vm-discovery-candidates").querySelectorAll<HTMLInputElement>("input")) radio.disabled = busy || confirmedCandidates.has(Number(radio.value));
  $<HTMLButtonElement>("vm-discovery-search").disabled = busy || (!$<HTMLSelectElement>("vm-discovery-scope").value.includes("ground_surface") && !view.lanes.length) || !$<HTMLSelectElement>("vm-discovery-cloud").value;
  for (const id of ["focus", "inspect", "reject", "selected"]) $<HTMLButtonElement>(`vm-discovery-${id}`).disabled = busy || !c || confirmedCandidates.has(c.id);
  $<HTMLButtonElement>("vm-discovery-add").disabled = busy || !c || confirmedCandidates.has(c.id) || !$<HTMLSelectElement>("vm-discovery-kind").value || !$<HTMLInputElement>("vm-discovery-lanes").value.trim();
}
function clearDiscovery(): void {
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
async function runDiscovery(): Promise<void> {
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
  if (busy) return; busy = true; junctionInputs();
  try { await runDiscovery(); setStatus(`Found ${discoveryPreview?.candidates.length ?? 0} unconfirmed equipment proposals. Inspect the points before adding.`); }
  catch (err) { clearDiscovery(); draw(); setStatus(`Equipment search failed: ${errorText(err)}`); }
  finally { busy = false; junctionInputs(); renderDiscovery(); }
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
  busy = true; junctionInputs();
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
  finally { busy = false; junctionInputs(); renderDiscovery(); }
};
$("vm-discovery-inspect").onclick = async () => {
  if (busy || !discoverySnapshot) return; const c = discoveryPreview?.candidates.find(c => c.id === discoveryCandidate); if (!c) return;
  const revision = discoveryRevision; const request = discoverySnapshot; const saved = discoveryState();
  busy = true; junctionInputs();
  try {
    const cloud = await cropCloud(request.id, c.min, c.max, true);
    if (revision !== discoveryRevision || !entries.has(request.id)) { await removeCloud(cloud.id); throw new Error("Source changed; search again."); }
    const source = entries.get(request.id)!; const added = addEntry(cloud);
    if (c.evidence.kind === "elevated_panel") { added.mode = "solid"; refreshColors(added); }
    record({ label: "automatic equipment proposal points", added: [added], hide: [source] }); renderList();
    $<HTMLSelectElement>("vm-discovery-cloud").value = String(request.id); restoreDiscovery(saved); chooseDiscovery(c); renderDiscovery();
    setStatus(`${cloud.count} original points isolated with attributes; source hidden. Confirm the object or discard it. Cloud Undo restores the source.`);
  } catch (err) { setStatus(`Could not inspect equipment points: ${errorText(err)}`); }
  finally { busy = false; junctionInputs(); renderDiscovery(); }
};

function buildInputs(): void {
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
  junctionInputs();
}
listChanged.add(() => { clearQuality(); clearDiscovery(); clearJunctionPreview(); clearSignalPreview(); clearCrosswalkPreview(); buildInputs(); draw(); });
pointsInvalidated.add(() => { clearQuality(); clearDiscovery(); clearJunctionPreview(); clearSignalPreview(); clearCrosswalkPreview(); draw(); });
trajectoryChanged.add(buildInputs);
buildInputs();
function roadBuildOptions(): object {
  return {
    forward_lanes: Number($<HTMLInputElement>("vm-forward").value),
    backward_lanes: Number($<HTMLInputElement>("vm-backward").value),
    left_hand_traffic: $<HTMLSelectElement>("vm-traffic").value === "left",
    lane_width: Number($<HTMLInputElement>("vm-width").value),
    speed_limit: Number($<HTMLInputElement>("vm-speed").value),
    segment_length: Number($<HTMLInputElement>("vm-segment").value),
    anchor_width_prior: $<HTMLInputElement>("vm-anchor-prior").checked,
    track_boundaries: $<HTMLInputElement>("vm-track-boundaries").checked,
    fit_boundaries: $<HTMLInputElement>("vm-fit-boundaries").checked,
    fit_source_surface: $<HTMLInputElement>("vm-source-surface").checked,
    verify_curb_profiles: $<HTMLInputElement>("vm-verify-curbs").checked,
    merge_repeated_passes: $<HTMLInputElement>("vm-merge-passes").checked,
  };
}
function renderBuildReport(report: BuildReport): void {
  $("vm-build-report").textContent = `${report.roads} road stretches, ${report.lanes} lanes, ${fmt(report.generated_length)} m. ` +
    `Added ${fmt(report.added_length)} m; reused ${fmt(report.reused_length)} m of existing lanes. ` +
    `Measured sources before fitting, left to right: ${report.observed_fraction.map(f => `${Math.round(f*100)}%`).join(", ")}. ` +
    `Tracking changed ${report.tracked_vertices} sources; fitted ${report.fitted_vertices} vertices (maximum XY movement ${fmt(report.maximum_fit_displacement)} m). ` +
    (report.surface_fit ? `Source footprint: ${fmt(report.surface_fit.deferred_length_m)} m deferred; inferred lane widths ${fmt(report.surface_fit.minimum_lane_width_m ?? 0)}–${fmt(report.surface_fit.maximum_lane_width_m ?? 0)} m. ` : "") +report.warnings.join(" ");
}
buildButton.onclick = async () => {
  if (busy) return;
  const trajectory = inputTrajectories().find((t) => String(t.id) === trajectoryInput.value);
  if (!trajectory || !cloudInput.value) return;
  busy = true;
  buildInputs();
  setTool(null);
  setStatus("Building draft roads from the point cloud and trajectory…");
  try {
    const options = roadBuildOptions();
    const edited = await vectorMap<Edited>("build", {
      id: Number(cloudInput.value), positions: trajectory.poses.positions, text: JSON.stringify(options),
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
    setStatus(`Could not build draft roads: ${errorText(err)}`);
  } finally {
    busy = false;
    buildInputs();
  }
};

function takeView(edited: Edited): void {
  clearQuality();
  view = edited.view;
  clearDiscovery();
  clearJunctionPreview();
  clearSignalPreview();
  clearCrosswalkPreview();
  undoDepth = edited.undo;
  renderFeatures();
  if (selected !== null && !view.lanes.some((l) => l.id === selected)) selected = null;
  draw();
  undoButton.disabled = undoDepth === 0;
  exportButton.disabled = view.lanes.length === 0;
  $<HTMLButtonElement>("vm-fit").disabled = view.boundaries.length === 0;
  $<HTMLButtonElement>("vm-plan").disabled = $<HTMLButtonElement>("vm-iso").disabled = view.boundaries.length === 0;
  renderLane();
  void renderIssues();
}

/** Run vectormap commands; failures go to the status line. Returns false if they failed. */
async function apply(commands: object[], what: string): Promise<boolean> {
  if (busy) return false;
  busy = true;
  junctionInputs();
  try {
    const edited = await vectorMap<Edited>("apply", { text: JSON.stringify(commands) });
    takeView(edited);
    const warnings = (edited.result as { warnings?: Issue[] }[]).flatMap((c) => c.warnings ?? []);
    const shown = warnings.filter((w) => w.code !== "no_op");
    setStatus(shown.length ? `${what}: ${shown.map((w) => w.message).join("; ")}` : what);
    return true;
  } catch (err) {
    setStatus(`${what} failed: ${errorText(err)}`);
    return false;
  } finally {
    busy = false;
    junctionInputs();
  }
}

const plural = (n: number, what: string) => `${n} ${what}${n === 1 ? "" : "s"}`;

async function renderIssues(): Promise<void> {
  const issues = await vectorMap<Issue[]>("validate", { autoware: true });
  const counts = { error: 0, warning: 0, info: 0 };
  for (const i of issues) counts[i.severity]++;
  const kinds = [
    [view.lanes.length, "lane"],
    [view.stopLines.length, "stop line"],
    [view.signals.length, "traffic light"],
    [view.crosswalks.length, "crosswalk"],
  ] as const;
  const parts = kinds.filter(([n]) => n > 0).map(([n, what]) => plural(n, what));
  status.textContent =
    view.lanes.length === 0
      ? "No map yet."
      : `${parts.join(", ")}. Autoware check: ${plural(counts.error, "error")}, ${plural(counts.warning, "warning")}.`;
  issueList.replaceChildren(
    ...issues
      .filter((i) => i.severity !== "info")
      .slice(0, 20)
      .map((issue) => {
        const li = document.createElement("li");
        li.className = issue.severity;
        li.textContent = issue.message;
        li.title = issue.code;
        if (issue.entity?.kind === "lane") {
          const lane = issue.entity.id;
          li.classList.add("clickable");
          li.onclick = () => selectLane(lane, true);
        }
        return li;
      }),
  );
}

function laneById(id: number): LaneView | undefined {
  return view.lanes.find((l) => l.id === id);
}

function laneLength(lane: LaneView): number {
  let length = 0;
  const c = lane.center;
  for (let i = 1; i < c.length; i++) length += Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]);
  return length;
}

interface SourceCurveSupport { fraction: number; start_supported: boolean; end_supported: boolean; insufficient_returns: number; height_mismatches: number }
interface SourceQualityReport {
  lanes: {lane: number; center: SourceCurveSupport; left: SourceCurveSupport; right: SourceCurveSupport; needs_review: boolean}[];
  low_support_lanes: number[]; omitted_lanes: number[]; malformed_lanes: number[]; limited: boolean; warnings: string[];
}
let qualityRevision = 0;
function clearQuality(): void {
  qualityRevision++;
  $("vm-quality-report").textContent = "Source coverage has not been checked for the current map and cloud.";
  $("vm-quality-lanes").replaceChildren();
}
$("vm-quality-cloud").onchange = clearQuality;
$("vm-quality-check").onclick = async () => {
  if (busy) return;
  clearQuality(); const revision = qualityRevision;
  busy = true; junctionInputs(); setStatus("Checking lane centres and boundaries against source points…");
  try {
    const report = await vectorMap<SourceQualityReport>("quality", {id: Number($<HTMLSelectElement>("vm-quality-cloud").value)});
    if (revision !== qualityRevision) return;
    $("vm-quality-report").textContent = `${report.lanes.length} lanes checked; ${report.low_support_lanes.length} need source review; ${report.omitted_lanes.length} omitted; ${report.malformed_lanes.length} malformed. ${report.limited ? "Coverage check limited. " : ""}` + report.warnings.join(" ");
    const percentage = (s: SourceCurveSupport) => `${Math.round(s.fraction*100)}%${s.start_supported && s.end_supported ? "" : " (end support missing)"}`;
    for (const lane of report.lanes.filter(l => l.needs_review)) {
      const button = document.createElement("button"); button.textContent = `Lane ${lane.lane}: centre ${percentage(lane.center)}, left ${percentage(lane.left)}, right ${percentage(lane.right)}`;
      button.onclick = () => selectLane(lane.lane, true); $("vm-quality-lanes").append(button);
    }
    setStatus(`Source coverage checked: ${report.low_support_lanes.length} lanes need review. The map is unchanged.`);
  } catch (err) { setStatus(`Could not check source coverage: ${errorText(err)}`); }
  finally { busy = false; junctionInputs(); }
};

function selectLane(id: number | null, frame = false): void {
  selected = id;
  draw();
  renderLane();
  const lane = id === null ? undefined : laneById(id);
  if (frame && lane) {
    const shift = globalShift();
    const box = new THREE.Box3();
    for (const p of [...lane.left, ...lane.right]) box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
    viewer.frameBox(box.expandByScalar(10));
  }
}

function renderLane(): void {
  const lane = selected === null ? undefined : laneById(selected);
  laneBox.hidden = !lane;
  if (!lane) return;
  const list = (ids: number[]) => (ids.length ? ids.join(", ") : "none");
  $("vm-lane-title").textContent = `Lane ${lane.id}`;
  $("vm-lane-info").textContent =
    `${lane.kind}, ${fmt(laneLength(lane))} m${lane.turn ? `, turns ${lane.turn}` : ""}. ` +
    `From ${list(lane.predecessors)}; to ${list(lane.successors)}.`;
  $<HTMLInputElement>("vm-lane-speed").value = lane.speedLimit === null ? "" : String(lane.speedLimit);
}

$<HTMLButtonElement>("vm-lane-apply").onclick = () => {
  if (selected === null) return;
  const value = $<HTMLInputElement>("vm-lane-speed").value.trim();
  const kmh = value === "" ? null : Number(value);
  if (kmh !== null && !(kmh > 0)) return setStatus("The speed limit must be positive (km/h).");
  void apply([{ op: "set_speed_limit", lanes: [selected], kmh }], `Speed limit of lane ${selected} set`);
};
$<HTMLButtonElement>("vm-lane-delete").onclick = () => {
  if (selected === null) return;
  const lane = selected;
  void apply([{ op: "remove_lane", lane }], `Lane ${lane} removed`);
};

// ---------------------------------------------------------------------------
// Clicks
// ---------------------------------------------------------------------------

/** Height used when a click hits no point: the map's, else 0. */
function fallbackZ(): number {
  const p = view.lanes[0]?.center[0] ?? sketch[sketch.length - 1];
  return p ? p[2] : 0;
}

/** Original coordinates of a click: the point under it, else the ray at the map's height. */
async function clickPoint(x: number, y: number): Promise<XYZ | null> {
  const hit = await pickPoint(x, y);
  if (hit) return [hit.exact[0], hit.exact[1], hit.exact[2]];
  const shift = globalShift();
  const g = viewer.groundPoint(x, y, fallbackZ() - shift[2]);
  return g ? [g.x + shift[0], g.y + shift[1], g.z + shift[2]] : null;
}

/** The lane under a click, with where along it. */
async function clickLane(x: number, y: number): Promise<Nearest | null> {
  const p = await clickPoint(x, y);
  if (!p || view.lanes.length === 0) return null;
  const near = await vectorMap<Nearest | null>("nearest", { x: p[0], y: p[1] });
  return near && (near.inside || near.distance < LANE_REACH) ? near : null;
}

/** The lane and its neighbors running the same way, left to right. */
function sameWayGroup(id: number): number[] {
  const out = [id];
  for (const side of ["leftNeighbor", "rightNeighbor"] as const) {
    let at = laneById(id);
    while (at) {
      const n: Neighbor | null = at[side];
      if (!n || n.direction === "opposite" || out.includes(n.lane)) break;
      if (side === "leftNeighbor") out.unshift(n.lane);
      else out.push(n.lane);
      at = laneById(n.lane);
    }
  }
  return out;
}

/** A tool acting on the lane under each click. */
function laneTool(button: string, what: string, act: (near: Nearest) => Promise<unknown>): Tool {
  const tool: Tool = {
    async click(x, y) {
      const near = await clickLane(x, y);
      if (!near) return setStatus("No lane there.");
      selectLane(near.lane);
      await act(near);
    },
    enter() {
      $(button).setAttribute("aria-pressed", "true");
      hint.textContent = what;
    },
    exit() {
      $(button).setAttribute("aria-pressed", "false");
      renderHint();
    },
  };
  $(button).onclick = () => toggleTool(tool);
  return tool;
}

// Draw road: click points along the road, double-click or Enter to build it.

function roadCommand(reference: XYZ[]): object {
  const forward = Math.max(0, Math.round(Number($<HTMLInputElement>("vm-forward").value)));
  const backward = Math.max(0, Math.round(Number($<HTMLInputElement>("vm-backward").value)));
  const width = Number($<HTMLInputElement>("vm-width").value) || 3.5;
  const leftHand = $<HTMLSelectElement>("vm-traffic").value === "left";
  const segment = Number($<HTMLInputElement>("vm-segment").value);
  const speed = Number($<HTMLInputElement>("vm-speed").value);
  const lane = (direction: string) => ({ width, direction });
  // Looking along the drawn line, left-hand traffic keeps its forward lanes on the left.
  const forwards = Array.from({ length: forward }, () => lane("forward"));
  const backwards = Array.from({ length: backward }, () => lane("backward"));
  const lanes = leftHand ? [...forwards, ...backwards] : [...backwards, ...forwards];
  const command: Record<string, unknown> = { op: "build_road", reference, lanes };
  // With both directions the line is the centre line between them.
  if (forward > 0 && backward > 0) command.left_edge = width * (leftHand ? forward : backward);
  if (segment > 0) command.segment_length = segment;
  if (speed > 0) command.speed_limit = { kmh: speed };
  return command;
}

async function finishRoad(): Promise<void> {
  if (busy) return;
  const reference = sketch;
  if (reference.length < 2) return setStatus("Click at least two points along the road.");
  const lanes = (roadCommand(reference) as { lanes: unknown[] }).lanes;
  if (lanes.length === 0) return setStatus("Give the road at least one lane.");
  sketch = [];
  if (!$<HTMLInputElement>("vm-refine-sketch").checked) {
    if (await apply([roadCommand(reference)], "Road built")) setTool(null);
    else draw();
    return;
  }
  busy = true; buildInputs();
  try {
    if (!cloudInput.value) throw new Error("Choose the point cloud for this drawn path.");
    const edited = await vectorMap<Edited>("build", { id: Number(cloudInput.value), positions: new Float64Array(reference.flat()), text: JSON.stringify(roadBuildOptions()) });
    takeView(edited); const report = edited.result as BuildReport; renderBuildReport(report); setTool(null);
    setStatus(`Road built from the point cloud and your traced path: ${report.lanes} added lanes. The path and nominal widths are operator inputs; review the boundary evidence report.`);
  } catch (err) { sketch = reference; draw(); setStatus(`Could not fit the drawn road: ${errorText(err)}`); }
  finally { busy = false; buildInputs(); }
}

const roadTool: Tool = {
  async click(x, y) {
    const p = await clickPoint(x, y);
    if (!p) return;
    sketch.push(p);
    draw();
    renderHint();
  },
  doubleClick() {
    // The second click of the double click added a duplicate vertex.
    sketch.pop();
    void finishRoad();
  },
  key(e) {
    if (e.key === "Enter") {
      void finishRoad();
      return true;
    }
    if (e.key === "Backspace" && sketch.length > 0) {
      sketch.pop();
      draw();
      renderHint();
      return true;
    }
    return false;
  },
  enter() {
    sketch = [];
    $("vm-road").setAttribute("aria-pressed", "true");
    renderHint();
  },
  exit() {
    sketch = [];
    $("vm-road").setAttribute("aria-pressed", "false");
    draw();
    renderHint();
  },
};
$("vm-road").onclick = () => toggleTool(roadTool);

// Connect: the lane to leave, then the lane to enter.
let connectFrom: number | null = null;
const connectTool = laneTool("vm-connect", "Click the lane to leave, then the lane to enter.", async (near) => {
  if (connectFrom === null || connectFrom === near.lane) {
    connectFrom = near.lane;
    hint.textContent = `From lane ${near.lane}: now click the lane to enter.`;
    return;
  }
  const from = connectFrom;
  connectFrom = null;
  hint.textContent = "Click the lane to leave, then the lane to enter.";
  await apply([{ op: "add_connector", from, to: near.lane }], `Lanes ${from} and ${near.lane} connected`);
});
const connectExit = connectTool.exit;
connectTool.exit = () => {
  connectFrom = null;
  connectExit();
};

laneTool("vm-stop", "Click a lane where its stop line goes; it spans the lanes beside it going the same way.", (near) =>
  apply(
    [{ op: "add_stop_line", lanes: sameWayGroup(near.lane), placement: { at_station: { station: near.station } } }],
    "Stop line added",
  ),
);
laneTool(
  "vm-light",
  "Click a lane: a traffic light for it and the lanes beside it, at their stop line (made at the lane end if there is none).",
  (near) => apply([{ op: "add_traffic_signal", lanes: sameWayGroup(near.lane) }], "Traffic light added"),
);
laneTool("vm-crosswalk", "Click a lane where the crosswalk crosses the road; stop lines are added before it.", (near) =>
  apply(
    [
      {
        op: "add_crosswalk",
        geometry: { across: { lane: near.lane, station: near.station, width: 4.0, margin: 0.5 } },
        stop_line_offset: 1.5,
      },
    ],
    "Crosswalk added",
  ),
);
laneTool("vm-select", "Click a lane to see it, set its speed limit or remove it.", async () => {});

function featureSelection(): { kind: "crosswalk" | "signal" | "stop_line"; id: number } | null {
  const [kind, id] = $<HTMLSelectElement>("vm-feature").value.split(":");
  return kind === "crosswalk" || kind === "signal" || kind === "stop_line" ? { kind, id: Number(id) } : null;
}
function featurePoints(): XYZ[] {
  const s = featureSelection();
  return s?.kind === "crosswalk" ? view.crosswalks.find(c => c.id === s.id)?.outline ?? [] : s?.kind === "stop_line" ? view.stopLines.find(c => c.id === s.id)?.points ?? [] : view.signals.find(c => c.id === s?.id)?.points ?? [];
}
function featureInputs(): void {
  const s = featureSelection();
  const valid = !!s && featurePoints().length >= 2 && (s.kind !== "crosswalk" || view.crosswalks.find(c => c.id === s.id)?.editable === true);
  for (const el of $("vm-feature-editor").querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input,select,button")) el.disabled = busy || !!featureDrag || (!valid && el.id !== "vm-feature");
}
function renderFeatureVertex(): void {
  const p = featurePoints()[Number($<HTMLSelectElement>("vm-feature-vertex").value)];
  for (const [i, axis] of ["x", "y", "z"].entries()) $<HTMLInputElement>(`vm-feature-${axis}`).value = p ? String(p[i]) : "";
}
function renderFeatureFields(): void {
  const s = featureSelection();
  const p = featurePoints();
  const select = $<HTMLSelectElement>("vm-feature-vertex");
  const index = select.value;
  select.replaceChildren(...p.map((_, i) => new Option(`Vertex ${i + 1}`, String(i))));
  if (Number(index) < p.length) select.value = index || "0";
  const signal = view.signals.find(c => s?.kind === "signal" && c.id === s.id);
  $("vm-feature-height-row").hidden = !signal;
  $<HTMLInputElement>("vm-feature-height").value = signal?.height == null ? "" : String(signal.height);
  const crossing = view.crosswalks.find(c => s?.kind === "crosswalk" && c.id === s.id);
  const stop = view.stopLines.find(c => s?.kind === "stop_line" && c.id === s.id);
  const source = signal?.geometrySource ?? crossing?.geometrySource ?? stop?.geometrySource ?? "imported_or_manual";
  const label = source.startsWith("point_cloud_brightness_stripes") ? "Measured ground-paint bands" : source.startsWith("point_cloud_brightness_bar") ? "Measured ground-paint bar" : source.startsWith("point_cloud_box_fit") ? "Measured point-cloud housing" : "Imported or manually drawn geometry";
  $("vm-feature-source").textContent = s ? `${label}${source.endsWith("_user_edited") ? "; manually edited" : ""}. ${crossing && !crossing.editable ? "Separate polygon/edge geometry requires a dedicated editor." : "Review lane associations and legal meaning after edits."}` : "Choose a feature explicitly before editing.";
  renderFeatureVertex(); featureInputs();
}
function renderFeatures(): void {
  const select = $<HTMLSelectElement>("vm-feature");
  const old = select.value;
  select.replaceChildren(new Option("Choose a crossing or signal", ""),
    ...view.crosswalks.map(c => new Option(`Crossing ${c.id}`, `crosswalk:${c.id}`)),
    ...view.stopLines.map(c => new Option(`Stop line ${c.id}`, `stop_line:${c.id}`)),
    ...view.signals.map(c => new Option(`Signal ${c.id}`, `signal:${c.id}`)));
  select.value = [...select.options].some(o => o.value === old) ? old : "";
  if (!select.value && editingFeature) setTool(null);
  renderFeatureFields();
}
const featureSelect = $<HTMLSelectElement>("vm-feature");
featureSelect.onchange = () => { if (editingFeature) setTool(null); renderFeatureFields(); draw(); };
$<HTMLSelectElement>("vm-feature-vertex").onchange = renderFeatureVertex;
$("vm-feature-focus").onclick = () => {
  const shift = globalShift(); const box = new THREE.Box3();
  const s = featureSelection();
  const height = s?.kind === "signal" ? view.signals.find(c => c.id === s.id)?.height ?? 0 : 0;
  for (const p of featurePoints()) { box.expandByPoint(new THREE.Vector3(p[0]-shift[0], p[1]-shift[1], p[2]-shift[2])); box.expandByPoint(new THREE.Vector3(p[0]-shift[0], p[1]-shift[1], p[2]-shift[2]+height)); }
  if (!box.isEmpty()) viewer.frameBox(box.expandByScalar(2));
};
async function editFeature(points: XYZ[], height?: number): Promise<void> {
  const s = featureSelection(); if (busy || !s) return;
  busy = true; junctionInputs();
  try {
    const edited = await vectorMap<Edited>("feature-edit", { text: JSON.stringify({ ...s, points, height }) });
    takeView(edited);
    const report = edited.result as { changed: boolean; warnings: string[] };
    setStatus(report.changed ? `Feature ${s.id} edited. ${report.warnings.join(" ")}` : "Geometry unchanged; no Undo step added.");
  } catch (err) { setStatus(`Feature edit failed: ${errorText(err)}`); draw(); }
  finally { busy = false; junctionInputs(); }
}
$("vm-feature-apply").onclick = () => {
  if (busy) return;
  const values = ["x", "y", "z"].map(axis => $<HTMLInputElement>(`vm-feature-${axis}`).value.trim());
  const h = $<HTMLInputElement>("vm-feature-height").value.trim();
  if (values.some(v => !v || !Number.isFinite(Number(v))) || (h && !Number.isFinite(Number(h)))) return setStatus("Enter finite metre coordinates and a valid housing height.");
  const p = structuredClone(featurePoints());
  const index = Number($<HTMLSelectElement>("vm-feature-vertex").value);
  if (!p[index]) return;
  p[index] = values.map(Number) as XYZ;
  void editFeature(p, featureSelection()?.kind === "signal" && h ? Number(h) : undefined);
};
const featureTool: Tool = {
  click() {},
  pointerDown(x, y) {
    if (busy || !featureSelection()) return false;
    const rect = $("viewport").querySelector(":scope > canvas")!.getBoundingClientRect();
    const shift = globalShift(); let best = 10; let hit = -1;
    for (const [i, p] of featurePoints().entries()) {
      const screen = viewer.project(new THREE.Vector3(p[0]-shift[0], p[1]-shift[1], p[2]-shift[2]));
      if (!screen) continue;
      const distance = Math.hypot(screen.x+rect.left-x, screen.y+rect.top-y);
      if (distance < best) { best = distance; hit = i; }
    }
    if (hit < 0) return false;
    const point = [...featurePoints()[hit]] as XYZ;
    const ground = viewer.groundPoint(x,y,point[2]-shift[2]); if (!ground) return false;
    featureDrag = { before: structuredClone(view), index: hit, point, start:[x,y], moved:false, offset:[point[0]-shift[0]-ground.x,point[1]-shift[1]-ground.y] };
    $<HTMLSelectElement>("vm-feature-vertex").value = String(hit); renderFeatureVertex(); featureInputs();
    return true;
  },
  pointerMove(x,y) {
    const d = featureDrag; if (!d || (!d.moved && Math.hypot(x-d.start[0],y-d.start[1]) < 2)) return;
    const shift = globalShift(); const p = viewer.groundPoint(x,y,d.point[2]-shift[2]); if (!p) return;
    const next: XYZ = [p.x+shift[0]+d.offset[0],p.y+shift[1]+d.offset[1],d.point[2]];
    featurePoints()[d.index] = next; d.moved = Math.hypot(next[0]-d.point[0],next[1]-d.point[1]) > 1e-4;
    renderFeatureVertex(); draw();
  },
  async pointerUp(x,y) {
    featureTool.pointerMove!(x,y); const d = featureDrag; if (!d) return;
    const points = structuredClone(featurePoints()); featureDrag = null; view = d.before; renderFeatureFields(); draw();
    if (d.moved) await editFeature(points);
  },
  pointerCancel() { if (featureDrag) { view = featureDrag.before; featureDrag = null; renderFeatureFields(); draw(); } },
  enter() { editingFeature = true; clearJunctionPreview(); clearSignalPreview(); clearCrosswalkPreview(); $("vm-feature-drag").setAttribute("aria-pressed","true"); hint.textContent = "Drag a yellow feature vertex; Z is kept. Escape cancels. Review observed paint, lamps and lane associations after editing."; draw(); },
  exit() { featureTool.pointerCancel!(); editingFeature = false; $("vm-feature-drag").setAttribute("aria-pressed","false"); renderHint(); draw(); },
};
$("vm-feature-drag").onclick = () => { if (!busy && featureSelection()) toggleTool(featureTool); };

/** Boundary IDs, rather than lane-side copies, keep shared/reversed sides together. */
function previewBoundary(): void {
  const byId = new Map(view.boundaries.map((b) => [b.id, b.points]));
  for (const lane of view.lanes) {
    const oriented = (ref: LaneView["leftRef"]): XYZ[] => {
      const points = byId.get(ref.id)!;
      return ref.reversed ? [...points].reverse() : points;
    };
    lane.left = oriented(lane.leftRef);
    lane.right = oriented(lane.rightRef);
    const n = Math.max(lane.left.length, lane.right.length, 2);
    const left = resample(lane.left, n);
    const right = resample(lane.right, n);
    lane.center = left.map((p, i) => p.map((v, j) => (v + right[i][j]) / 2) as XYZ);
  }
  draw();
}

const vertexTool: Tool = {
  click() {},
  pointerDown(x, y) {
    if (busy) return false;
    const rect = $("viewport").querySelector(":scope > canvas")!.getBoundingClientRect();
    const shift = globalShift();
    let hit: { boundary: number; index: number; point: XYZ } | null = null;
    let best = 10;
    for (const b of view.boundaries) {
      for (const [index, p] of b.points.entries()) {
        const screen = viewer.project(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
        if (!screen) continue;
        const distance = Math.hypot(screen.x + rect.left - x, screen.y + rect.top - y);
        if (distance < best) {
          best = distance;
          hit = { boundary: b.id, index, point: [...p] };
        }
      }
    }
    if (!hit) return false;
    const ground = viewer.groundPoint(x, y, hit.point[2] - shift[2]);
    if (!ground) return false;
    drag = {
      before: structuredClone(view), ...hit, start: [x, y], moved: false,
      offset: [hit.point[0] - shift[0] - ground.x, hit.point[1] - shift[1] - ground.y],
    };
    activeBoundary = hit.boundary;
    hint.textContent = `Boundary ${hit.boundary}, vertex ${hit.index + 1}: drag to move; Escape cancels.`;
    draw();
    return true;
  },
  pointerMove(x, y) {
    if (!drag) return;
    if (!drag.moved && Math.hypot(x - drag.start[0], y - drag.start[1]) < 2) return;
    const shift = globalShift();
    const p = viewer.groundPoint(x, y, drag.point[2] - shift[2]);
    if (!p) return;
    const next: XYZ = [p.x + shift[0] + drag.offset[0], p.y + shift[1] + drag.offset[1], drag.point[2]];
    const b = view.boundaries.find((b) => b.id === drag!.boundary)!;
    b.points[drag.index] = next;
    drag.moved = Math.hypot(next[0] - drag.point[0], next[1] - drag.point[1]) > 1e-4;
    previewBoundary();
  },
  async pointerUp(x, y) {
    vertexTool.pointerMove!(x, y);
    const finished = drag;
    if (!finished) return;
    const points = view.boundaries.find((b) => b.id === finished.boundary)!.points;
    drag = null;
    view = finished.before;
    draw();
    if (finished.moved) await apply([{ op: "set_boundary_geometry", boundary: finished.boundary, geometry: points }], `Boundary ${finished.boundary} vertex moved`);
    hint.textContent = "Drag a yellow boundary vertex. Height is kept; shared lanes update together. Escape leaves editing.";
  },
  pointerCancel() {
    if (drag) { view = drag.before; drag = null; draw(); }
  },
  enter() {
    editingVertices = true;
    clearJunctionPreview();
    $("vm-vertices").setAttribute("aria-pressed", "true");
    hint.textContent = "Drag a yellow boundary vertex. Height is kept; shared lanes update together. Escape leaves editing.";
    draw();
  },
  exit() {
    vertexTool.pointerCancel!();
    editingVertices = false;
    activeBoundary = null;
    $("vm-vertices").setAttribute("aria-pressed", "false");
    renderHint();
    draw();
  },
};
$("vm-vertices").onclick = () => toggleTool(vertexTool);

function renderHint(): void {
  if (activeTool() === roadTool) {
    hint.textContent = sketch.length
      ? `${sketch.length} point${sketch.length === 1 ? "" : "s"}; double-click or Enter to build the road, Backspace to take one back, Esc to cancel.`
      : $<HTMLInputElement>("vm-refine-sketch").checked ? "Trace the outside forward lane over the points; Enter fits ground height and boundary evidence. The path and lane widths remain your inputs." : "Click points along the road (its centre line); double-click or Enter to build it.";
    return;
  }
  if (activeTool() === null || !hint.textContent) {
    hint.textContent =
      "Draw roads over the cloud, connect them through junctions, add stop lines, traffic lights and crosswalks, and save a Lanelet2 map for Autoware.";
  }
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

/** Open a Lanelet2 map (.osm) or vectormap IR (.json) over the clouds. */
export async function openVectorMap(name: string, text: string): Promise<MapView> {
  const opened = await vectorMap<Edited>("open", { name, text });
  setTool(null);
  const issues = opened.result as Issue[];
  $("vm-import-notes").hidden = issues.length === 0;
  $("vm-import-notes").setAttribute("open", "");
  $("vm-import-issues").replaceChildren(...issues.map((issue) => {
    const li = document.createElement("li");
    li.className = issue.severity;
    li.textContent = issue.message;
    li.title = issue.code;
    return li;
  }));
  selected = null;
  takeView(opened);
  if (view.lanes.length > 0 && entries.size === 0) frameMap();
  const problems = issues.filter((i) => i.severity !== "info").length;
  setStatus(
    `Opened ${name}: ${view.lanes.length} lanes${problems ? `, ${problems} load issues (see Import notes)` : ""}.`,
  );
  return view;
}

const fileInput = $<HTMLInputElement>("vm-file");
$("vm-open").onclick = () => fileInput.click();
fileInput.onchange = async () => {
  const file = fileInput.files?.[0];
  fileInput.value = "";
  if (!file) return;
  try {
    await openVectorMap(file.name, await file.text());
  } catch (err) {
    setStatus(`Could not open ${file.name}: ${errorText(err)}`);
  }
};

function frameMap(): void {
  const shift = globalShift();
  const box = new THREE.Box3();
  for (const b of view.boundaries) {
    for (const p of b.points) box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
  }
  for (const s of view.signals) for (const p of s.points) {
    box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2] + (s.height ?? 0)));
  }
  for (const c of view.crosswalks) for (const p of c.outline) {
    box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
  }
  viewer.frameBox(box);
}
$("vm-fit").onclick = frameMap;
$("vm-plan").onclick = () => { viewer.view({ x: 0, y: 0, z: 1 }); frameMap(); };
$("vm-iso").onclick = () => { viewer.view({ x: 0.65, y: -0.9, z: 1.2 }); frameMap(); };

undoButton.onclick = async () => {
  if (busy) return;
  takeView(await vectorMap<Edited>("undo"));
  setStatus("Undone.");
};
$("vm-clear").onclick = async () => {
  if (busy) return;
  setTool(null);
  if (view.lanes.length === 0 && view.boundaries.length === 0) return;
  selected = null;
  takeView(await vectorMap<Edited>("clear"));
  setStatus("Map cleared (Undo brings it back).");
};
exportButton.onclick = async () => {
  const exported = await vectorMap<{ osm: string; projectorInfo: string; issues: Issue[] }>("export", { autoware: true });
  download(new Blob([exported.osm], { type: "application/xml" }), "lanelet2_map.osm");
  download(new Blob([exported.projectorInfo], { type: "text/yaml" }), "map_projector_info.yaml");
  setStatus(
    `Saved lanelet2_map.osm and map_projector_info.yaml (${exported.projectorInfo.split("\n")[0].replace("projector_type: ", "")} projector).`,
  );
};

renderHint();
void vectorMap<Edited>("view").then(takeView);
