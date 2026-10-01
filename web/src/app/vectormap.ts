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
import { vectorMap } from "../api";
import { $, download, errorText, fmt, setStatus } from "./dom";
import { clouds, entries, globalShift, listChanged, viewer } from "./state";
import { inputTrajectories, trajectoryChanged } from "./trajectory";
import { activeTool, pickPoint, setTool, toggleTool, type Tool } from "./tools";

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
  stopLines: { id: number; points: XYZ[] }[];
  crosswalks: { id: number; outline: XYZ[] }[];
  signals: { id: number; points: XYZ[]; height: number | null }[];
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
  virtual: viewer.lineMaterial({ color: 0x78909c, linewidth: 1, depthTest: false }),
  arrow: viewer.lineMaterial({ color: 0x4dd0e1, linewidth: 2, depthTest: false }),
  stop: viewer.lineMaterial({ color: 0xff1744, linewidth: 4, depthTest: false }),
  signal: viewer.lineMaterial({ color: 0xffea00, linewidth: 5, depthTest: false }),
  crosswalk: viewer.lineMaterial({ color: 0xffffff, linewidth: 1.5, depthTest: false }),
  sketch: viewer.lineMaterial({ color: 0xffeb3b, linewidth: 3, depthTest: false }),
};
const laneFill = new THREE.MeshBasicMaterial({
  color: 0x42a5f5,
  transparent: true,
  opacity: 0.18,
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
const vertexMaterial = new THREE.PointsMaterial({ color: 0xffeb3b, size: 7, sizeAttenuation: false, depthTest: false });
let editingVertices = false;
let activeBoundary: number | null = null;
let drag: { before: MapView; boundary: number; index: number; point: XYZ; moved: boolean } | null = null;

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
function laneMesh(lane: LaneView, material: THREE.Material): THREE.Mesh {
  const n = Math.max(lane.left.length, lane.right.length, 2);
  const left = resample(lane.left, n);
  const right = resample(lane.right, n);
  const positions: number[] = [];
  for (let i = 1; i < n; i++) {
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
  for (const lane of view.lanes) group.add(laneMesh(lane, lane.id === selected ? selectedFill : laneFill));
  const byKind: Record<"solid" | "dashed" | "edge" | "virtual", number[]> = { solid: [], dashed: [], edge: [], virtual: [] };
  for (const b of view.boundaries) {
    const type = b.kind.type;
    if (type === "lane_marking" && b.kind.pattern === "dashed") dashedPairs(b.points, byKind.dashed);
    else if (type === "lane_marking") polylinePairs(b.points, byKind.solid);
    else if (type === "curb" || type === "road_edge") polylinePairs(b.points, byKind.edge);
    else polylinePairs(b.points, byKind.virtual);
  }
  for (const kind of ["solid", "dashed", "edge", "virtual"] as const) segments(byKind[kind], materials[kind]);
  const arrows: number[] = [];
  for (const lane of view.lanes) arrowPairs(lane, arrows);
  segments(arrows, materials.arrow, 2);
  const stops: number[] = [];
  for (const s of view.stopLines) polylinePairs(s.points, stops);
  segments(stops, materials.stop, 3);
  const walks: number[] = [];
  for (const c of view.crosswalks) polylinePairs([...c.outline, c.outline[0]], walks);
  segments(walks, materials.crosswalk, 3);
  const signals: number[] = [];
  for (const s of view.signals) {
    polylinePairs(s.points, signals);
    const h = s.height ?? 0.5;
    polylinePairs(
      s.points.map(([x, y, z]) => [x, y, z + h] as XYZ),
      signals,
    );
  }
  segments(signals, materials.signal, 3);
  if (sketch.length > 0) {
    const pairs: number[] = [];
    polylinePairs(sketch, pairs);
    if (sketch.length === 1) pairs.push(...local(sketch[0]), ...local([sketch[0][0] + 0.3, sketch[0][1], sketch[0][2]]));
    segments(pairs, materials.sketch, 4);
  }
  if (editingVertices) {
    const points = view.boundaries.flatMap((b) => b.points.flatMap(local));
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
  viewer.requestRender();
}

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
  roads: number;
  lanes: number;
  generated_length: number;
  observed_fraction: number[];
  warnings: string[];
}

const buildButton = $<HTMLButtonElement>("vm-build");
const cloudInput = $<HTMLSelectElement>("vm-cloud");
const trajectoryInput = $<HTMLSelectElement>("vm-trajectory");
function buildInputs(): void {
  const fill = (select: HTMLSelectElement, items: { id: number; name: string }[]) => {
    const value = select.value;
    select.replaceChildren(...items.map((item) => new Option(item.name, String(item.id))));
    if (items.some((item) => String(item.id) === value)) select.value = value;
  };
  fill(cloudInput, clouds().map((entry) => entry.cloud));
  fill(trajectoryInput, inputTrajectories());
  buildButton.disabled = busy || !cloudInput.value || !trajectoryInput.value;
}
listChanged.add(buildInputs);
trajectoryChanged.add(buildInputs);
buildInputs();
buildButton.onclick = async () => {
  if (busy) return;
  const trajectory = inputTrajectories().find((t) => String(t.id) === trajectoryInput.value);
  if (!trajectory || !cloudInput.value) return;
  busy = true;
  buildInputs();
  setTool(null);
  setStatus("Building draft roads from the point cloud and trajectory…");
  try {
    const options = {
      forward_lanes: Number($<HTMLInputElement>("vm-forward").value),
      backward_lanes: Number($<HTMLInputElement>("vm-backward").value),
      left_hand_traffic: $<HTMLSelectElement>("vm-traffic").value === "left",
      lane_width: Number($<HTMLInputElement>("vm-width").value),
      speed_limit: Number($<HTMLInputElement>("vm-speed").value),
      segment_length: Number($<HTMLInputElement>("vm-segment").value),
      anchor_width_prior: $<HTMLInputElement>("vm-anchor-prior").checked,
    };
    const edited = await vectorMap<Edited>("build", {
      id: Number(cloudInput.value), positions: trajectory.poses.positions, text: JSON.stringify(options),
    });
    takeView(edited);
    const report = edited.result as BuildReport;
    $("vm-build-report").textContent =
      `${report.roads} road stretches, ${report.lanes} lanes, ${fmt(report.generated_length)} m. ` +
      `Measured boundary vertices, left to right: ${report.observed_fraction.map((f) => `${Math.round(f * 100)}%`).join(", ")}. ` +
      report.warnings.join(" ");
    setStatus("Draft roads added. Review the boundaries, lane directions and junctions before export.");
  } catch (err) {
    setStatus(`Could not build draft roads: ${errorText(err)}`);
  } finally {
    busy = false;
    buildInputs();
  }
};

function takeView(edited: Edited): void {
  view = edited.view;
  undoDepth = edited.undo;
  if (selected !== null && !view.lanes.some((l) => l.id === selected)) selected = null;
  draw();
  undoButton.disabled = undoDepth === 0;
  exportButton.disabled = view.lanes.length === 0;
  $<HTMLButtonElement>("vm-fit").disabled = view.boundaries.length === 0;
  renderLane();
  void renderIssues();
}

/** Run vectormap commands; failures go to the status line. Returns false if they failed. */
async function apply(commands: object[], what: string): Promise<boolean> {
  if (busy) return false;
  busy = true;
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
  const reference = sketch;
  if (reference.length < 2) return setStatus("Click at least two points along the road.");
  const lanes = (roadCommand(reference) as { lanes: unknown[] }).lanes;
  if (lanes.length === 0) return setStatus("Give the road at least one lane.");
  sketch = [];
  if (await apply([roadCommand(reference)], "Road built")) setTool(null);
  else draw();
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
    drag = { before: structuredClone(view), ...hit, moved: false };
    activeBoundary = hit.boundary;
    hint.textContent = `Boundary ${hit.boundary}, vertex ${hit.index + 1}: drag to move; Escape cancels.`;
    draw();
    return true;
  },
  pointerMove(x, y) {
    if (!drag) return;
    const shift = globalShift();
    const p = viewer.groundPoint(x, y, drag.point[2] - shift[2]);
    if (!p) return;
    const next: XYZ = [p.x + shift[0], p.y + shift[1], drag.point[2]];
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
      : "Click points along the road (its centre line); double-click or Enter to build it.";
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
  viewer.frameBox(box);
}
$("vm-fit").onclick = frameMap;

undoButton.onclick = async () => {
  if (busy) return;
  takeView(await vectorMap<Edited>("undo"));
  setStatus("Undone.");
};
$("vm-clear").onclick = async () => {
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
