/**
 * Pose graph panel, in the spirit of interactive_slam: a g2o graph or a
 * TUM / KITTI trajectory (as an odometry chain) with a scan per node. Each
 * scan is drawn at its node's pose, so optimising only moves matrices. Pick
 * two nodes to close a loop: ICP registers their scans from the current
 * relative pose, the result becomes a loop edge and the graph is optimised.
 * Edges can be colored by their error, and the loops listed worst first to
 * remove wrong ones. The scans at their final poses become an ordinary cloud
 * for the other tools.
 */

import * as THREE from "three";
import {
  addPoseGraphFloor,
  addPoseGraphLoop,
  closePoseGraph,
  exportPoseGraph,
  findPoseGraphLoops,
  mergePoseGraph,
  openPoseGraph,
  optimizePoseGraph,
  insertPoseGraphEdges,
  poseGraphMap,
  removePoseGraphEdges,
  removePoseGraphPlane,
  setPoseGraphFixed,
  setPoseGraphNodePose,
  setPoseGraphPoses,
} from "../api";
import { colorize, gradientCss, lut } from "../colormap";
import { CANCELLED, type PoseFormat, type PoseGraphFiles, type PoseGraphState, type Progress, type RemovedEdge } from "../protocol";
import { $, download, errorText, fillTable, fmt, removeButton, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { display, distanceChanged, globalShift, listChanged, viewer } from "./state";
import { endTask, showProgress, startTask } from "./tasks";
import { setTool, toggleTool, type Tool } from "./tools";

interface Graph {
  name: string;
  state: PoseGraphState;
  scans: (Float32Array | null)[];
  scanPoints: number;
}

/** Undo information: the poses before the step, and the edges it added or removed. */
interface Step {
  poses: Float64Array;
  /** Loop edges the step added (the last ones of the graph). */
  added?: number[];
  removed?: RemovedEdge[];
  /** A plane the step added (with its edges). */
  plane?: number;
  /** A node whose fixed flag the step flipped. */
  flipped?: number;
}

/** Scan files (the poses are g2o, TUM or KITTI text). */
const SCAN_FILE = /\.(pcd|ply|bin|las|laz|xyz|pts)$/i;
const POSE_FILE = /\.(g2o|tum|kitti|txt|csv)$/i;
/** Text files found next to KITTI scans that are not poses. */
const NOT_POSES = /^(calib|times)\.txt$/i;
/** Draw at most this many scan points overall. */
const DISPLAY_BUDGET = 6_000_000;
const PICK_RADIUS_PX = 12;
/** Loops listed at most. */
const LIST_LIMIT = 100;

const HUES = 12;
const scanMaterials = Array.from(
  { length: HUES },
  (_, k) =>
    new THREE.PointsMaterial({
      size: 3,
      sizeAttenuation: false,
      color: new THREE.Color().setHSL(k / HUES, 0.65, 0.6),
    }),
);
/** The color ramp as a 256 x 1 texture. */
function rampTexture(): THREE.DataTexture {
  const rgb = lut(display.ramp);
  const rgba = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) rgba.set([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255], i * 4);
  const texture = new THREE.DataTexture(rgba, 256, 1);
  texture.needsUpdate = true;
  return texture;
}

/** Scans colored by height (render z) in the color ramp, like iridescence's "rainbow" mode. */
const heightMaterial = new THREE.ShaderMaterial({
  uniforms: { lo: { value: 0 }, hi: { value: 1 }, size: { value: 3 }, ramp: { value: rampTexture() } },
  vertexShader: `
    uniform float lo;
    uniform float hi;
    uniform float size;
    varying float t;
    void main() {
      float z = (modelMatrix * vec4(position, 1.0)).z;
      t = clamp((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size;
    }`,
  fragmentShader: `
    uniform sampler2D ramp;
    varying float t;
    void main() {
      gl_FragColor = vec4(texture2D(ramp, vec2(t, 0.5)).rgb, 1.0);
    }`,
});
// The graph is drawn over the scans, so it stays visible when they pile up.
const nodeMaterial = new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, vertexColors: true, depthTest: false });
const NODE_COLOR = new THREE.Color(0xffffff);
/** Nodes the optimiser holds in place. */
const FIXED_COLOR = new THREE.Color(0xff5252);
const selectedMaterial = new THREE.PointsMaterial({
  size: 12,
  sizeAttenuation: false,
  vertexColors: true,
  depthTest: false,
});
const edgeMaterial = new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false });
const ODOMETRY_COLOR = [0.45, 0.6, 0.8];
const LOOP_COLOR = [1, 0.6, 0.15];
const SELECTED_COLORS = [new THREE.Color(0xffeb3b), new THREE.Color(0x00e5ff)];

let graph: Graph | null = null;
const group = new THREE.Group();
group.name = "pose-graph";
viewer.overlay.add(group);
let scanObjects: (THREE.Points | null)[] = [];
/** Per scan, the 5th and 95th percentile of its local z (for the height colors). */
let scanHeights: ([number, number] | null)[] = [];
/** Node markers and edges are drawn relative to this (original coordinates), for float32 precision. */
let origin = new THREE.Vector3();
let drawnShift = "";
/** Node indices of the loop's ends (from the A and B fields), in that order. */
let selection: number[] = [];
const steps: Step[] = [];
/** Scans from a folder without poses, waiting for the poses file. */
let pendingScans: File[] = [];
let busy = false;

const num = (id: string) => Number($<HTMLInputElement>(id).value);

function clearGroup(): void {
  for (const child of [...group.children]) {
    group.remove(child);
    if (child instanceof THREE.Points || child instanceof THREE.LineSegments) child.geometry.dispose();
  }
  scanObjects = [];
  scanHeights = [];
}

/** 5th and 95th percentile of the z of interleaved points (from a sample). */
function heightRange(positions: Float32Array): [number, number] {
  const n = positions.length / 3;
  const step = Math.max(1, Math.floor(n / 1000));
  const z: number[] = [];
  for (let i = 0; i < n; i += step) z.push(positions[i * 3 + 2]);
  z.sort((a, b) => a - b);
  return [z[Math.floor(z.length * 0.05)], z[Math.floor(z.length * 0.95)]];
}

const showScans = () => $<HTMLInputElement>("pg-show-scans").checked;
const byHeight = () => $<HTMLSelectElement>("pg-colors").value === "height";

/** Translation of node `i`'s pose (original coordinates). */
function nodePosition(state: PoseGraphState, i: number): THREE.Vector3 {
  const m = state.poses;
  return new THREE.Vector3(m[i * 16 + 3], m[i * 16 + 7], m[i * 16 + 11]);
}

/** Render position of node `i`. */
function renderPosition(state: PoseGraphState, i: number): THREE.Vector3 {
  const [sx, sy, sz] = globalShift();
  return nodePosition(state, i).sub(new THREE.Vector3(sx, sy, sz));
}

/** Build the scene objects for a newly opened graph. */
function build(g: Graph): void {
  clearGroup();
  scanObjects = g.scans.map((positions, i) => {
    if (!positions || positions.length === 0) return null;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingSphere();
    const points = new THREE.Points(geometry, byHeight() ? heightMaterial : scanMaterials[i % HUES]);
    points.matrixAutoUpdate = false;
    points.visible = showScans();
    group.add(points);
    return points;
  });
  scanHeights = g.scans.map((positions) => (positions?.length ? heightRange(positions) : null));
  origin = g.state.poses.length ? nodePosition(g.state, 0).round() : new THREE.Vector3();
  update(g.state);
}

/** Place the scans at `state`'s poses and redraw nodes, edges and the selection. */
function update(state: PoseGraphState): void {
  if (!graph) return;
  graph.state = state;
  drawnShift = globalShift().join();
  const [sx, sy, sz] = globalShift();
  const m = state.poses;
  scanObjects.forEach((object, i) => {
    if (!object) return;
    const k = i * 16;
    object.matrix.set(
      m[k], m[k + 1], m[k + 2], m[k + 3] - sx,
      m[k + 4], m[k + 5], m[k + 6], m[k + 7] - sy,
      m[k + 8], m[k + 9], m[k + 10], m[k + 11] - sz,
      0, 0, 0, 1,
    );
    object.matrixWorldNeedsUpdate = true;
  });
  // Height colors span the scans' typical heights at their current poses.
  let [lo, hi] = [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY];
  scanHeights.forEach((range, i) => {
    if (!range) return;
    lo = Math.min(lo, m[i * 16 + 11] - sz + range[0]);
    hi = Math.max(hi, m[i * 16 + 11] - sz + range[1]);
  });
  if (lo < hi) [heightMaterial.uniforms.lo.value, heightMaterial.uniforms.hi.value] = [lo, hi];
  for (const name of ["nodes", "edges", "selected"]) {
    const old = group.getObjectByName(name) as THREE.Points | THREE.LineSegments | undefined;
    if (old) {
      group.remove(old);
      old.geometry.dispose();
    }
  }
  const n = m.length / 16;
  const local = (i: number) => nodePosition(state, i).sub(origin);
  const placed = new THREE.Vector3(origin.x - sx, origin.y - sy, origin.z - sz);

  const nodePositions = new Float32Array(n * 3);
  const nodeColors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    local(i).toArray(nodePositions, i * 3);
    (state.fixed[i] ? FIXED_COLOR : NODE_COLOR).toArray(nodeColors, i * 3);
  }
  // Dense graphs (thousands of keyframes) get smaller dots, or they hide the scans.
  nodeMaterial.size = n > 2000 ? 2 : n > 300 ? 3 : 5;
  const nodes = new THREE.Points(new THREE.BufferGeometry(), nodeMaterial);
  nodes.geometry.setAttribute("position", new THREE.BufferAttribute(nodePositions, 3));
  nodes.geometry.setAttribute("color", new THREE.BufferAttribute(nodeColors, 3));
  nodes.name = "nodes";
  nodes.renderOrder = 2;

  const e = state.edges.length / 2;
  const edgePositions = new Float32Array(e * 6);
  const edgeColors = new Float32Array(e * 6);
  const byError = errorColors() ? colorize(edgeErrors(state), 0, errorScale(state), lut(display.ramp)) : null;
  for (let k = 0; k < e; k++) {
    local(state.edges[2 * k]).toArray(edgePositions, k * 6);
    local(state.edges[2 * k + 1]).toArray(edgePositions, k * 6 + 3);
    const color = byError
      ? [byError[k * 4] / 255, byError[k * 4 + 1] / 255, byError[k * 4 + 2] / 255]
      : state.edgeKinds[k]
        ? LOOP_COLOR
        : ODOMETRY_COLOR;
    edgeColors.set(color, k * 6);
    edgeColors.set(color, k * 6 + 3);
  }
  const edges = new THREE.LineSegments(new THREE.BufferGeometry(), edgeMaterial);
  edges.geometry.setAttribute("position", new THREE.BufferAttribute(edgePositions, 3));
  edges.geometry.setAttribute("color", new THREE.BufferAttribute(edgeColors, 3));
  edges.name = "edges";
  edges.renderOrder = 1;

  const selected = new THREE.Points(new THREE.BufferGeometry(), selectedMaterial);
  const selectedPositions = new Float32Array(selection.length * 3);
  const selectedColors = new Float32Array(selection.length * 3);
  selection.forEach((i, k) => {
    local(i).toArray(selectedPositions, k * 3);
    SELECTED_COLORS[k].toArray(selectedColors, k * 3);
  });
  selected.geometry.setAttribute("position", new THREE.BufferAttribute(selectedPositions, 3));
  selected.geometry.setAttribute("color", new THREE.BufferAttribute(selectedColors, 3));
  selected.name = "selected";
  selected.renderOrder = 3;

  for (const object of [nodes, edges, selected]) {
    object.position.copy(placed);
    object.geometry.computeBoundingSphere();
    group.add(object);
  }
  viewer.requestRender();
  renderInfo();
}

const errorColors = () => $<HTMLInputElement>("pg-edge-errors").checked;

/** Per edge, the square root of its χ²: how far it disagrees with the graph, in standard deviations. */
function edgeErrors(state: PoseGraphState): Float32Array {
  return Float32Array.from(state.edgeErrors, Math.sqrt);
}

/** Top of the error color scale: the largest edge error, at least 1. */
function errorScale(state: PoseGraphState): number {
  return Math.max(1, ...edgeErrors(state));
}

/** Frame the graph's nodes. */
function fitGraph(state: PoseGraphState): void {
  const box = new THREE.Box3();
  for (let i = 0; i < state.poses.length / 16; i++) box.expandByPoint(renderPosition(state, i));
  if (box.isEmpty()) return;
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  // Scans reach past their nodes.
  const radius = Math.max(sphere.radius * 1.2, 20);
  const { position, target } = viewer.getCamera();
  const direction = position.clone().sub(target).normalize();
  if (direction.lengthSq() === 0) direction.set(0, -1, 1).normalize();
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(viewer.camera.fov / 2));
  viewer.setCamera(sphere.center.clone().addScaledVector(direction, distance), sphere.center);
}

function renderInfo(): void {
  const g = graph;
  $("pg-body").hidden = !g;
  $("pg-close").hidden = !g;
  $("pg-hint").hidden = !!g;
  if (!g) return;
  const { state } = g;
  const loops = state.edgeKinds.reduce((sum, k) => sum + k, 0);
  const errors = state.edgeErrors.reduce((sum, e) => sum + e, 0);
  const withScans = g.scans.filter((s) => s).length;
  fillTable($("pg-stats"), [
    ["Graph", g.name],
    ["Nodes", `${(state.poses.length / 16).toLocaleString()} (${withScans.toLocaleString()} with scans)`],
    ["Edges", `${(state.edgeKinds.length - loops).toLocaleString()} odometry, ${loops.toLocaleString()} loops`],
    ["Scan points", g.scanPoints.toLocaleString()],
    ...(state.planes
      ? [["Planes", `${state.planes} (${state.planeEdges.toLocaleString()} keyframe views)`] as [string, string]]
      : []),
    ["Total error (χ²)", fmt(errors)],
  ]);
  $<HTMLButtonElement>("pg-loop").disabled = busy || selection.length !== 2;
  $<HTMLButtonElement>("pg-optimize").disabled = busy;
  $<HTMLButtonElement>("pg-undo").disabled = busy || steps.length === 0;
  $<HTMLButtonElement>("pg-map").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-find").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-merge").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-floor").disabled = busy || withScans === 0;
  const a = nodeA();
  $<HTMLButtonElement>("pg-fix").textContent = a !== null && state.fixed[a] ? "Free A" : "Fix A";
  $<HTMLButtonElement>("pg-prune").disabled = busy || loops === 0;
  $("pg-legend").hidden = !errorColors();
  $("pg-legend-bar").style.background = gradientCss(display.ramp, "to right");
  $("pg-legend-max").textContent = fmt(errorScale(state));
  renderLoops(state);
}

/** The loop edges, worst first, each with a button to select its nodes and one to remove it. */
function renderLoops(state: PoseGraphState): void {
  const errors = edgeErrors(state);
  const loops = [...state.edgeKinds.keys()].filter((k) => state.edgeKinds[k]).sort((a, b) => errors[b] - errors[a]);
  const ids = state.nodeIds;
  $("pg-loops-hint").hidden = loops.length > 0;
  $("pg-loop-list").replaceChildren(
    ...loops.slice(0, LIST_LIMIT).map((k) => {
      const [from, to] = [state.edges[2 * k], state.edges[2 * k + 1]];
      const li = document.createElement("li");
      const name = document.createElement("button");
      name.className = "name link";
      name.textContent = `${ids[from]} – ${ids[to]}`;
      name.title = "Select its two nodes";
      name.onclick = () => setSelection([from, to]);
      const error = document.createElement("span");
      error.className = "meta";
      error.textContent = `error ${fmt(errors[k])}`;
      li.append(name, error, removeButton(() => void removeEdges([k], `loop ${ids[from]} – ${ids[to]}`)));
      return li;
    }),
  );
}

/** Remove edges and optimise again (one undo step). */
function removeEdges(indices: number[], what: string): Promise<void> {
  return run("Removing", async () => {
    const poses = graph!.state.poses.slice();
    const out = await removePoseGraphEdges(indices, kernel());
    steps.push({ poses, removed: out.removed });
    update(out.state);
    setStatus(`Removed ${what} and optimised`);
  });
}

$<HTMLButtonElement>("pg-prune").onclick = () => {
  if (!graph) return;
  const { state } = graph;
  const limit = num("pg-prune-limit");
  const errors = edgeErrors(state);
  const worse = [...state.edgeKinds.keys()].filter((k) => state.edgeKinds[k] && errors[k] > limit);
  if (worse.length === 0) {
    setStatus(`No loop has an error above ${fmt(limit)}`);
    return;
  }
  void removeEdges(worse, `${worse.length} loop${worse.length === 1 ? "" : "s"} with an error above ${fmt(limit)}`);
};

/** The nearest node within the pick radius of a click, or null. */
function nodeAt(clientX: number, clientY: number): number | null {
  if (!graph) return null;
  const rect = $("viewport").getBoundingClientRect();
  const [x, y] = [clientX - rect.left, clientY - rect.top];
  let best: number | null = null;
  let bestDistance = PICK_RADIUS_PX;
  for (let i = 0; i < graph.state.poses.length / 16; i++) {
    const p = viewer.project(renderPosition(graph.state, i));
    if (!p) continue;
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < bestDistance) [best, bestDistance] = [i, d];
  }
  return best;
}

const [fieldA, fieldB] = [$<HTMLInputElement>("pg-a"), $<HTMLInputElement>("pg-b")];

/** Take the loop's ends from the A and B fields (node ids). */
function readSelection(): void {
  if (!graph) return;
  const ids = graph.state.nodeIds;
  const index = (field: HTMLInputElement) => (field.value === "" ? -1 : ids.indexOf(Number(field.value)));
  selection = [index(fieldA), index(fieldB)].filter((i) => i >= 0);
  if (selection.length === 2 && selection[0] === selection[1]) selection.pop();
  update(graph.state);
}
fieldA.oninput = fieldB.oninput = () => {
  stopMoving();
  readSelection();
};

function setSelection(nodes: number[]): void {
  const ids = graph?.state.nodeIds;
  [fieldA.value, fieldB.value] = [0, 1].map((k) => (ids && nodes[k] !== undefined ? String(ids[nodes[k]]) : ""));
  readSelection();
}

const pickTool: Tool = {
  click(x, y) {
    const node = nodeAt(x, y);
    if (node === null || !graph) return;
    setSelection(selection.length === 1 && selection[0] !== node ? [selection[0], node] : [node]);
    if (selection.length === 2) setStatus("Two nodes picked: Add loop registers B's scan to A's");
  },
  enter() {
    $("pg-pick").setAttribute("aria-pressed", "true");
    setStatus("Click a node (a white dot), then a second one to close a loop between them");
  },
  exit() {
    $("pg-pick").setAttribute("aria-pressed", "false");
  },
};
$<HTMLButtonElement>("pg-pick").onclick = () => toggleTool(pickTool);

/** A row-major 4x4 from 12 or 16 numbers in free text (e.g. KITTI's `Tr: …`), or null when empty. */
function parseMatrix(text: string): number[] | null {
  const values = text.match(/[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?/g)?.map(Number) ?? [];
  if (values.length === 0) return null;
  if (values.length === 12) return [...values, 0, 0, 0, 1];
  if (values.length === 16) return values;
  throw new Error(`the scan → pose matrix needs 12 or 16 numbers, not ${values.length}`);
}

/** KITTI's velodyne-to-camera transform from a calib.txt among the files. */
async function kittiExtrinsic(files: File[]): Promise<number[] | null> {
  const calib = files.find((f) => /^calib\.txt$/i.test(f.name));
  const line = calib ? (await calib.text()).split("\n").find((l) => l.startsWith("Tr:")) : undefined;
  return line ? parseMatrix(line.slice(3)) : null;
}

/** The poses file among `files`: a g2o graph, else a trajectory, preferring names like "poses". */
function posesFile(files: File[]): File | undefined {
  const candidates = files.filter((f) => POSE_FILE.test(f.name) && !NOT_POSES.test(f.name));
  return (
    candidates.find((f) => /\.g2o$/i.test(f.name)) ??
    candidates.find((f) => /pose|traj|odom|gt/i.test(f.name)) ??
    candidates[0]
  );
}

/**
 * The poses file, scans and loading options for picked files, with a note on
 * the scan transform used; null (with a status message) when they do not
 * make a graph. Scans picked without poses wait for them when `pending`.
 */
async function graphFiles(
  files: File[],
  pending: boolean,
): Promise<{ files: PoseGraphFiles; note: string } | null> {
  let scans = files.filter((f) => SCAN_FILE.test(f.name));
  const poses = posesFile(files);
  if (!poses) {
    if (scans.length === 0) {
      setStatus("No poses file (.g2o, .txt, .tum, .kitti) or scans among the files", true);
    } else if (pending) {
      pendingScans = scans;
      setStatus(`${scans.length.toLocaleString()} scans found but no poses: now open the poses file (Open files…)`);
    } else {
      setStatus("No poses file (.g2o, .txt, .tum, .kitti) next to the scans", true);
    }
    return null;
  }
  if (scans.length === 0 && pending) scans = pendingScans;
  pendingScans = [];
  if (scans.length === 0) {
    setStatus(`No scans next to ${poses.name}: open a folder holding both, or the scans first`, true);
    return null;
  }
  let extrinsic: number[] | null;
  try {
    extrinsic = parseMatrix($<HTMLTextAreaElement>("pg-extrinsic").value);
  } catch (err) {
    setStatus(errorText(err), true);
    return null;
  }
  let note = "";
  if (!extrinsic) {
    extrinsic = await kittiExtrinsic(files);
    if (extrinsic) note = " (scans moved by calib.txt's Tr)";
  }
  const shown = (graph?.scans.length ?? 0) + scans.length;
  return {
    files: {
      graph: poses,
      scans,
      voxel: Math.max(0, num("pg-voxel") || 0),
      displayPoints: Math.max(100, Math.min(num("pg-display") || 5000, Math.floor(DISPLAY_BUDGET / shown))),
      extrinsic,
      sigmaT: num("pg-sigma-t") || 0.1,
      sigmaRDeg: num("pg-sigma-r") || 1,
    },
    note,
  };
}

async function open(picked: File[]): Promise<void> {
  // Opening replaces the graph: its scans do not count against the display budget.
  const current = graph;
  graph = null;
  const found = await graphFiles(picked, true);
  graph = current;
  if (!found) return;
  const { files, note: extrinsicNote } = found;
  const poses = files.graph;
  setTool(null);
  const signal = startTask();
  busy = true;
  setStatus(`Opening ${poses.name} with ${files.scans.length.toLocaleString()} scans…`);
  try {
    const opened = await openPoseGraph(
      files,
      (p: Progress) => {
        showProgress(p);
        setStatus(`Opening ${poses.name}: ${p.note}…`);
      },
      signal,
    );
    graph = { name: opened.name, state: opened, scans: opened.scans, scanPoints: opened.scanPoints };
    fieldA.value = fieldB.value = "";
    selection = [];
    steps.length = 0;
    build(graph);
    fitGraph(opened);
    const unmatched = opened.unmatched.length ? `; ${opened.unmatched.length} scans matched no pose` : "";
    setStatus(
      `Opened ${opened.name}: ${(opened.poses.length / 16).toLocaleString()} poses, ` +
        `${opened.scanPoints.toLocaleString()} scan points${extrinsicNote}${unmatched}`,
    );
  } catch (err) {
    const message = errorText(err);
    setStatus(message === CANCELLED ? "Opening cancelled" : `Could not open the pose graph: ${message}`, message !== CANCELLED);
  } finally {
    busy = false;
    endTask(signal);
    renderInfo();
  }
}

for (const [button, input] of [
  ["pg-open-folder", "pg-folder-input"],
  ["pg-open-files", "pg-files-input"],
] as const) {
  $<HTMLButtonElement>(button).onclick = () => $<HTMLInputElement>(input).click();
  $<HTMLInputElement>(input).onchange = (e) => {
    const target = e.target as HTMLInputElement;
    const files = [...(target.files ?? [])];
    target.value = "";
    if (files.length) void open(files);
  };
}

/** Join a second graph, placed by registering the scans of a node in each. */
async function merge(picked: File[]): Promise<void> {
  if (!graph || busy) return;
  const found = await graphFiles(picked, false);
  if (!found) return;
  const { files, note } = found;
  const ids = graph.state.nodeIds;
  const here = $<HTMLInputElement>("pg-merge-here").value;
  const nodeA = here === "" ? (selection[0] ?? 0) : ids.indexOf(Number(here));
  if (nodeA < 0) {
    setStatus(`This graph has no node ${here}`, true);
    return;
  }
  const there = $<HTMLInputElement>("pg-merge-there").value;
  setTool(null);
  await run("Joining", async () => {
    const signal = startTask();
    try {
      const merged = await mergePoseGraph(
        {
          files,
          nodeA,
          nodeB: there === "" ? null : Number(there),
          yawSteps: Math.max(1, Math.round(num("pg-merge-yaw") || 8)),
          maxIterations: Math.max(1, num("pg-icp-iterations") || 50),
          overlap: Math.min(100, Math.max(10, num("pg-icp-overlap") || 80)) / 100,
          inlierDistance: inlierDistance(),
          minFitness: Math.min(100, Math.max(0, num("pg-find-fitness"))) / 100,
          sigmaT: num("pg-loop-sigma-t") || 0.1,
          sigmaRDeg: num("pg-loop-sigma-r") || 1,
          loopKernel: kernel(),
        },
        (p: Progress) => {
          showProgress(p);
          setStatus(`Joining ${files.graph.name}: ${p.note}…`);
        },
        signal,
      );
      const g = graph!;
      g.name = `${g.name} + ${merged.name}`;
      g.scans = [...g.scans, ...merged.scans];
      g.scanPoints += merged.scanPoints;
      // Undo restores poses and edges of one graph; the join changed the graph itself.
      steps.length = 0;
      g.state = merged.state;
      setSelection([]);
      build(g);
      fitGraph(merged.state);
      const unmatched = merged.unmatched.length ? `; ${merged.unmatched.length} scans matched no pose` : "";
      setStatus(
        `Joined ${merged.name} (${merged.scans.length.toLocaleString()} poses${note}${unmatched}) at node ` +
          `${ids[nodeA]}: overlap ${Math.round(merged.fitness * 100)} %, ICP RMS ${fmt(merged.rms)}; ` +
          `χ² ${fmt(merged.optimized.initialCost)} → ${fmt(merged.optimized.finalCost)}. ` +
          "Find loops links the two further; the join cannot be undone.",
      );
    } finally {
      endTask(signal);
    }
  });
}

$<HTMLButtonElement>("pg-merge").onclick = () => $<HTMLInputElement>("pg-merge-input").click();
$<HTMLInputElement>("pg-merge-input").onchange = (e) => {
  const target = e.target as HTMLInputElement;
  const files = [...(target.files ?? [])];
  target.value = "";
  if (files.length) void merge(files);
};

/** Run a graph operation with the buttons disabled. */
async function run(label: string, action: () => Promise<void>): Promise<void> {
  if (!graph || busy) return;
  busy = true;
  renderInfo();
  try {
    await action();
  } catch (err) {
    setStatus(`${label} failed: ${errorText(err)}`, true);
  } finally {
    busy = false;
    renderInfo();
  }
}

const inlierDistance = () => Math.max(0.01, num("pg-inlier") || 0.5);

$<HTMLButtonElement>("pg-find").onclick = () =>
  run("Finding loops", async () => {
    const poses = graph!.state.poses.slice();
    const signal = startTask();
    try {
      const found = await findPoseGraphLoops(
        {
          maxDistance: Math.max(0, num("pg-find-radius")),
          minTravel: Math.max(0, num("pg-find-travel")),
          spacing: Math.max(0, num("pg-find-spacing")),
          minFitness: Math.min(100, Math.max(0, num("pg-find-fitness"))) / 100,
          inlierDistance: inlierDistance(),
          maxIterations: Math.max(1, num("pg-icp-iterations") || 50),
          overlap: Math.min(100, Math.max(10, num("pg-icp-overlap") || 80)) / 100,
          sigmaT: num("pg-loop-sigma-t") || 0.1,
          sigmaRDeg: num("pg-loop-sigma-r") || 1,
          loopKernel: kernel(),
        },
        (p: Progress) => {
          showProgress(p);
          setStatus(`Finding loops: ${p.note}…`);
        },
        signal,
      );
      if (found.added.length) steps.push({ poses, added: found.edges });
      update(found.state);
      const { optimized } = found;
      setStatus(
        found.candidates === 0
          ? "No loop candidates: raise the search radius or lower the minimum travel"
          : `Added ${found.added.length} of ${found.candidates} candidate loop${found.candidates === 1 ? "" : "s"}` +
              (optimized
                ? `; χ² ${fmt(optimized.initialCost)} → ${fmt(optimized.finalCost)} in ${optimized.iterations} iterations`
                : " (none overlapped enough)"),
      );
    } finally {
      endTask(signal);
    }
  });

const AXES: Record<string, number[] | null> = { auto: null, "+z": [0, 0, 1], "-y": [0, -1, 0], "+y": [0, 1, 0] };
const axisName = (up: number[]) =>
  Object.entries(AXES).find(([, v]) => v && v.every((x, i) => x === up[i]))?.[0].toUpperCase() ?? up.join(", ");

$<HTMLButtonElement>("pg-floor").onclick = () =>
  run("Adding the floor", async () => {
    const poses = graph!.state.poses.slice();
    setStatus("Looking for the floor under every keyframe…");
    const floor = await addPoseGraphFloor({
      up: AXES[$<HTMLSelectElement>("pg-floor-up").value] ?? null,
      maxTiltDeg: Math.max(0, num("pg-floor-tilt") || 20),
      threshold: Math.max(0.001, num("pg-floor-threshold") || 0.1),
      minPoints: Math.max(3, Math.round(num("pg-floor-points") || 200)),
      sigmaAngleDeg: num("pg-floor-sigma-r") || 0.5,
      sigmaOffset: num("pg-floor-sigma-t") || 0.05,
      loopKernel: kernel(),
    });
    steps.push({ poses, plane: floor.plane });
    update(floor.state);
    const withScans = graph!.scans.filter((s) => s).length;
    const { optimized } = floor;
    setStatus(
      `Floor found under ${floor.tied.toLocaleString()} of ${withScans.toLocaleString()} keyframes (up ${axisName(floor.up)}); ` +
        `χ² ${fmt(optimized.initialCost)} → ${fmt(optimized.finalCost)} in ${optimized.iterations} iterations`,
    );
  });

// --- Moving and fixing nodes -------------------------------------------------

/** The node a gizmo moves: Node A. */
function nodeA(): number | null {
  return selection.length ? selection[0] : null;
}

let gizmo: {
  node: number;
  pivot: THREE.Object3D;
  /** The pivot's matrix when the gizmo appeared, and the poses then. */
  start: THREE.Matrix4;
  poses: Float64Array;
  handle: ReturnType<typeof viewer.attachGizmo>;
} | null = null;

const carry = () => $<HTMLInputElement>("pg-move-carry").checked;
const gizmoMode = () => $<HTMLSelectElement>("pg-move-mode").value as "translate" | "rotate";

/** Node `i`'s pose as a render-space matrix. */
function renderMatrix(poses: Float64Array, i: number): THREE.Matrix4 {
  const [sx, sy, sz] = globalShift();
  const m = poses.subarray(i * 16, i * 16 + 16);
  return new THREE.Matrix4().set(
    m[0], m[1], m[2], m[3] - sx,
    m[4], m[5], m[6], m[7] - sy,
    m[8], m[9], m[10], m[11] - sz,
    0, 0, 0, 1,
  );
}

/** `poses` with the gizmo's motion so far applied to the moved nodes. */
function movedPoses(): Float64Array {
  const g = gizmo!;
  g.pivot.updateMatrix();
  const motion = g.pivot.matrix.clone().multiply(g.start.clone().invert());
  const [sx, sy, sz] = globalShift();
  const poses = g.poses.slice();
  const end = carry() ? poses.length / 16 : g.node + 1;
  for (let i = g.node; i < end; i++) {
    const moved = motion.clone().multiply(renderMatrix(g.poses, i));
    const e = moved.elements; // column-major
    const row = [e[0], e[4], e[8], e[12] + sx, e[1], e[5], e[9], e[13] + sy, e[2], e[6], e[10], e[14] + sz];
    poses.set(row, i * 16);
  }
  return poses;
}

function stopMoving(): void {
  if (!gizmo) return;
  gizmo.handle.detach();
  viewer.overlay.remove(gizmo.pivot);
  gizmo = null;
  $("pg-move").setAttribute("aria-pressed", "false");
}

function startMoving(): void {
  const node = nodeA();
  if (!graph || node === null) {
    setStatus("Pick or type Node A first: the node to move", true);
    return;
  }
  stopMoving();
  const pivot = new THREE.Object3D();
  renderMatrix(graph.state.poses, node).decompose(pivot.position, pivot.quaternion, pivot.scale);
  viewer.overlay.add(pivot);
  pivot.updateMatrix();
  gizmo = {
    node,
    pivot,
    start: pivot.matrix.clone(),
    poses: graph.state.poses.slice(),
    handle: viewer.attachGizmo(
      pivot,
      gizmoMode(),
      // While dragging, only the drawing moves.
      () => graph && gizmo && update({ ...graph.state, poses: movedPoses() }),
      () => void commitMove(),
    ),
  };
  $("pg-move").setAttribute("aria-pressed", "true");
  setStatus(`Drag the gizmo to move node ${graph.state.nodeIds[node]}${carry() ? " and the nodes after it" : ""}`);
}

/** Send a finished drag to the worker (one undo step per drag). */
async function commitMove(): Promise<void> {
  if (!graph || !gizmo) return;
  const g = gizmo;
  const poses = movedPoses();
  const moved = poses.subarray(g.node * 16, g.node * 16 + 16);
  const before = g.poses;
  await run("Moving the node", async () => {
    const state = await setPoseGraphNodePose(g.node, Array.from(moved), carry());
    steps.push({ poses: before });
    update(state);
    // Further drags start from here.
    g.pivot.updateMatrix();
    g.start = g.pivot.matrix.clone();
    g.poses = state.poses.slice();
    setStatus(
      `Moved node ${state.nodeIds[g.node]}${carry() ? " and the nodes after it" : ""}: ` +
        "add a loop from here, fix it, or optimise",
    );
  });
}

$<HTMLButtonElement>("pg-move").onclick = () => (gizmo ? stopMoving() : startMoving());
$<HTMLSelectElement>("pg-move-mode").onchange = () => gizmo?.handle.setMode(gizmoMode());

$<HTMLButtonElement>("pg-fix").onclick = () =>
  run("Fixing", async () => {
    const node = nodeA();
    if (node === null) {
      setStatus("Pick or type Node A first: the node to fix or free", true);
      return;
    }
    const fixed = !graph!.state.fixed[node];
    const state = await setPoseGraphFixed(node, fixed);
    steps.push({ poses: graph!.state.poses.slice(), flipped: node });
    update(state);
    setStatus(`Node ${state.nodeIds[node]} ${fixed ? "is held in place by the optimiser (red)" : "is free again"}`);
  });

const kernel = () => ($<HTMLInputElement>("pg-robust").checked ? Math.max(0, num("pg-kernel")) : 0);

async function optimize(): Promise<string> {
  const out = await optimizePoseGraph(kernel());
  update(out.state);
  return `χ² ${fmt(out.initialCost)} → ${fmt(out.finalCost)} in ${out.iterations} iterations (${Math.round(out.millis)} ms)`;
}

$<HTMLButtonElement>("pg-optimize").onclick = () =>
  run("Optimising", async () => {
    setStatus("Optimising…");
    steps.push({ poses: graph!.state.poses.slice() });
    setStatus(`Optimised: ${await optimize()}`);
  });

$<HTMLButtonElement>("pg-loop").onclick = () =>
  run("Adding the loop", async () => {
    const [from, to] = selection;
    const ids = graph!.state.nodeIds;
    const poses = graph!.state.poses.slice();
    setStatus(`Registering node ${ids[to]} to node ${ids[from]}…`);
    const loop = await addPoseGraphLoop({
      from,
      to,
      maxIterations: Math.max(1, num("pg-icp-iterations") || 50),
      overlap: Math.min(100, Math.max(10, num("pg-icp-overlap") || 80)) / 100,
      pointToPlane: true,
      inlierDistance: inlierDistance(),
      sigmaT: num("pg-loop-sigma-t") || 0.1,
      sigmaRDeg: num("pg-loop-sigma-r") || 1,
    });
    steps.push({ poses, added: [loop.edge] });
    update(loop.state);
    const icp =
      `ICP RMS ${fmt(loop.rmsInitial)} → ${fmt(loop.rmsFinal)}, overlap ${Math.round(loop.fitness * 100)} %` +
      (loop.converged ? "" : " (not converged)");
    setStatus(`Loop ${ids[from]} – ${ids[to]} added (${icp}); optimising…`);
    const optimized = await optimize();
    setSelection([]);
    setStatus(`Loop ${ids[from]} – ${ids[to]} added (${icp}); ${optimized}`);
  });

$<HTMLButtonElement>("pg-undo").onclick = () =>
  run("Undo", async () => {
    const step = steps.pop();
    if (!step) return;
    if (step.added) await removePoseGraphEdges(step.added, null);
    if (step.plane !== undefined) await removePoseGraphPlane(step.plane);
    if (step.flipped !== undefined) await setPoseGraphFixed(step.flipped, !graph!.state.fixed[step.flipped]);
    if (step.removed) await insertPoseGraphEdges(step.removed);
    update(await setPoseGraphPoses(step.poses));
    const plural = (n: number) => (n === 1 ? "" : "s");
    setStatus(
      step.added
        ? step.added.length === 1
          ? "Removed the last loop"
          : `Removed the ${step.added.length} loop${plural(step.added.length)} found`
        : step.plane !== undefined
          ? "Removed the floor"
          : step.flipped !== undefined
          ? "Undid the fix"
          : step.removed
          ? `Put back ${step.removed.length} removed edge${step.removed.length === 1 ? "" : "s"}`
          : "Undid the optimisation",
    );
  });

for (const format of ["g2o", "kitti", "tum"] as PoseFormat[]) {
  $<HTMLButtonElement>(`pg-save-${format}`).onclick = () =>
    run("Saving", async () => {
      const text = await exportPoseGraph(format);
      const base = graph!.name.replace(/\.[^.]+$/, "");
      const filename = `${base}_optimized.${format === "g2o" ? "g2o" : "txt"}`;
      download(new Blob([text], { type: "text/plain" }), filename);
      setStatus(`Saved ${filename}`);
    });
}

$<HTMLButtonElement>("pg-map").onclick = () =>
  run("Building the map", async () => {
    setStatus("Building the map cloud…");
    const cloud = await poseGraphMap(Math.max(0, num("pg-map-voxel") || 0));
    record({ label: "the map cloud", added: [addEntry(cloud)] });
    renderList();
    setStatus(`Added ${cloud.name}: ${cloud.count.toLocaleString()} points`);
  });

$<HTMLButtonElement>("pg-close").onclick = async () => {
  if (busy) return;
  setTool(null);
  stopMoving();
  await closePoseGraph();
  graph = null;
  fieldA.value = fieldB.value = "";
  selection = [];
  steps.length = 0;
  clearGroup();
  viewer.requestRender();
  renderInfo();
};

$<HTMLInputElement>("pg-show-scans").onchange = () => {
  for (const object of scanObjects) if (object) object.visible = showScans();
  viewer.requestRender();
};
$<HTMLInputElement>("pg-edge-errors").onchange = () => graph && update(graph.state);
$<HTMLSelectElement>("pg-colors").onchange = () => {
  scanObjects.forEach((object, i) => {
    if (object) object.material = byHeight() ? heightMaterial : scanMaterials[i % HUES];
  });
  viewer.requestRender();
};

// Thinned scans are sparse: draw them a pixel larger than the clouds, or
// EDL (which darkens points next to background) turns them black.
const pointSizeInput = $<HTMLInputElement>("point-size");
function applyPointSize(): void {
  const size = Number(pointSizeInput.value) + 1;
  for (const material of scanMaterials) material.size = size;
  heightMaterial.uniforms.size.value = size;
  viewer.requestRender();
}
pointSizeInput.addEventListener("input", applyPointSize);
applyPointSize();

// The color ramp may have changed.
distanceChanged.add(() => {
  heightMaterial.uniforms.ramp.value.dispose();
  heightMaterial.uniforms.ramp.value = rampTexture();
  if (graph) update(graph.state);
});

// The graph follows the clouds' global shift.
listChanged.add(() => {
  if (graph && globalShift().join() !== drawnShift) update(graph.state);
});

renderInfo();
