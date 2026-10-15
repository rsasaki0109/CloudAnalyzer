import { bufferBytes, historyPolicy, onHistoryPolicy, trimHistory } from "../memory-budget";
import { projectChanged } from "../project-change";
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
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import {
  addPoseGraphFloor,
  changedObjects,
  addPoseGraphEdge,
  addPoseGraphLoop,
  registerPoseGraphPair,
  closePoseGraph,
  detectPoseGraphDynamic,
  exportPoseGraph,
  savePoseGraphProject,
  restorePoseGraphProject,
  extractGround,
  rasterizeCloud,
  findPoseGraphLoops,
  mergePoseGraph,
  openPoseGraph,
  optimizePoseGraph,
  insertPoseGraphEdges,
  poseGraphMap,
  removePoseGraphEdges,
  removePoseGraphPlane,
  setPoseGraphGravity,
  clearPoseGraphGravity,
  setPoseGraphFixed,
  setPoseGraphNodePose,
  setPoseGraphPoses,
} from "../api";
import { colorize, gradientCss, lut } from "../colormap";
import {
  CANCELLED,
  isBag,
  type PoseFormat,
  type PoseGraphFiles,
  type PoseGraphOpened,
  type PoseGraphState,
  type PoseGraphProject,
  type Progress,
  type RemovedEdge,
} from "../protocol";
import { $, download, errorText, fillTable, fmt, removeButton, setStatus } from "./dom";
import { refreshColors } from "./colors";
import { runM3c2 } from "./distance";
import { addEntry, renderList } from "./entries";
import { showRaster } from "./raster";
import { colorByField } from "./scalars";
import { record } from "./history";
import { display, distanceChanged, entries, globalShift, hideEntry, listChanged, toRender, viewer } from "./state";
import { endTask, showProgress, startTask } from "./tasks";
import { setTool, toggleTool, type Tool } from "./tools";

export async function captureGraphProject(): Promise<{ project: PoseGraphProject; name: string; sessions: Graph["sessions"]; imu: Graph["imu"] } | null> {
  if (busy) throw new Error("Finish the pose graph operation before saving the project");
  if (!graph) return null;
  return { project: await savePoseGraphProject(), name: graph.name, sessions: graph.sessions, imu: graph.imu };
}
export function graphProjectReady(): boolean { return !busy; }

export async function restoreGraphProject(project: PoseGraphProject | null, metadata?: { name: string; sessions: Graph["sessions"]; imu: Graph["imu"] }): Promise<void> {
  if (busy) throw new Error("Finish the pose graph operation before opening a project");
  setTool(null);
  stopMoving();
  stopAligning();
  playing = false;
  if (!project || !metadata) {
    await closePoseGraph();
    graph = null;
    steps.length = 0;
    selection = [];
    clearGroup();
    renderInfo();
    return;
  }
  const signal = startTask();
  busy = true;
  try {
    const opened = await restorePoseGraphProject(project, metadata.name, showProgress, signal);
    graph = { name: metadata.name, state: opened, scans: opened.scans, scanPoints: opened.scanPoints, sessions: metadata.sessions, imu: metadata.imu };
    steps.length = 0;
    selection = [];
    fieldA.value = fieldB.value = "";
    build(graph);
  } finally { busy = false; endTask(signal); renderInfo(); }
}

/** Adopt the view of an already committed native graph without re-reading inputs. */
export function takePreparedGraph(opened: PoseGraphOpened | null, metadata?: {name:string;sessions:Graph["sessions"];imu:Graph["imu"]}): void {
  setTool(null);stopMoving();stopAligning();playing=false;
  steps.length=0;selection=[];fieldA.value=fieldB.value="";
  if (!opened || !metadata) {graph=null;clearGroup();renderInfo();return;}
  graph={name:metadata.name,state:opened,scans:opened.scans,scanPoints:opened.scanPoints,sessions:metadata.sessions,imu:metadata.imu};
  build(graph);renderInfo();
}

interface Graph {
  name: string;
  state: PoseGraphState;
  scans: (Float32Array | null)[];
  scanPoints: number;
  /** The recordings it was made of: the one opened, then each joined one, as node ranges. */
  sessions: { name: string; first: number; count: number }[];
  /** Up directions from the bags' IMUs: node indices and three numbers each. */
  imu: { nodes: number[]; ups: number[] };
}

/** The up directions odometry found in a bag's IMU, for its nodes from `offset` on. */
function bagUps(odometry: PoseGraphOpened["odometry"], offset: number): { nodes: number[]; ups: number[] } {
  const nodes: number[] = [];
  const ups: number[] = [];
  const u = odometry?.ups;
  for (let i = 0; u && i < u.length / 3; i++) {
    if (!Number.isFinite(u[3 * i])) continue;
    nodes.push(offset + i);
    ups.push(u[3 * i], u[3 * i + 1], u[3 * i + 2]);
  }
  return { nodes, ups };
}

/** Undo information: the poses before the step, and the edges it added or removed. */
interface Step {
  poses: Float64Array;
  /** Loop edges the step added (the last ones of the graph). */
  added?: number[];
  removed?: RemovedEdge[];
  /** Planes the step added (with their edges). */
  planes?: { first: number; count: number };
  /** A node whose fixed flag the step flipped. */
  flipped?: number;
  /** The step tied the nodes to gravity. */
  gravity?: boolean;
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
      #ifdef ROUND_POINTS
      if (length(gl_PointCoord - 0.5) > 0.5) discard;
      #endif
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
// 2-pixel edges and pose axes, drawn over the scans.
const edgeMaterial = viewer.lineMaterial({ vertexColors: true, linewidth: 2, depthTest: false });
for (const material of [...scanMaterials, heightMaterial, nodeMaterial]) viewer.registerPointMaterial(material);
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
function enforceGraphBudget(): void {
  const policy = historyPolicy();
  trimHistory(steps, policy.steps, policy.bytes, list => bufferBytes(list) + list.reduce((n,s) => n + JSON.stringify([s.added,s.removed,s.planes,s.flipped,s.gravity]).length * 2, 0));
  $<HTMLButtonElement>("pg-undo").disabled = busy || !steps.length;
}
function recordGraphStep(step: Step): void { steps.push(step); enforceGraphBudget(); }
export function graphHistoryMemory(): {bytes: number; steps: number; data: unknown} { return {bytes: bufferBytes(steps) + steps.reduce((n,s) => n + JSON.stringify([s.added,s.removed]).length * 2,0), steps: steps.length, data: graph ? [graph,steps] : steps}; }
export function clearGraphHistory(): boolean { if (busy) return false; steps.length = 0; enforceGraphBudget(); return true; }
onHistoryPolicy(enforceGraphBudget);

let busy = false;

const num = (id: string) => Number($<HTMLInputElement>(id).value);

function clearGroup(): void {
  stopAligning();
  for (const child of [...group.children]) {
    group.remove(child);
    if (child instanceof THREE.Points || child instanceof LineSegments2) child.geometry.dispose();
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
  projectChanged();
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
  for (const name of ["nodes", "edges", "selected", "axes"]) {
    const old = group.getObjectByName(name) as THREE.Points | LineSegments2 | undefined;
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
  const edgeGeometry = new LineSegmentsGeometry();
  edgeGeometry.setPositions(edgePositions);
  edgeGeometry.setColors(edgeColors);
  const edges = new LineSegments2(edgeGeometry, edgeMaterial);
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

  const axes = showAxes() ? poseAxes(state, local) : null;
  for (const object of [nodes, edges, selected, ...(axes ? [axes] : [])]) {
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

const showAxes = () => $<HTMLInputElement>("pg-axes").checked;
/** Pose axes drawn at most. */
const MAX_AXES = 1500;

/**
 * Each keyframe's axes (x red, y green, z blue), like iridescence's
 * coordinate frames, three times the median step between keyframes long
 * (at least 1 m); on large graphs every n-th keyframe.
 */
function poseAxes(state: PoseGraphState, local: (i: number) => THREE.Vector3): LineSegments2 {
  const m = state.poses;
  const n = m.length / 16;
  const steps: number[] = [];
  for (let i = 1; i < n; i++) steps.push(local(i).distanceTo(local(i - 1)));
  steps.sort((a, b) => a - b);
  const length = Math.max(1, 3 * (steps[Math.floor(steps.length / 2)] ?? 1));
  const every = Math.max(1, Math.ceil(n / MAX_AXES));
  const positions: number[] = [];
  const colors: number[] = [];
  const rgb = [
    [1, 0.3, 0.3],
    [0.3, 1, 0.3],
    [0.35, 0.55, 1],
  ];
  for (let i = 0; i < n; i += every) {
    const o = local(i);
    for (let axis = 0; axis < 3; axis++) {
      // Column `axis` of the rotation: that body axis in world coordinates.
      const d = new THREE.Vector3(m[i * 16 + axis], m[i * 16 + 4 + axis], m[i * 16 + 8 + axis]).multiplyScalar(length);
      positions.push(o.x, o.y, o.z, o.x + d.x, o.y + d.y, o.z + d.z);
      colors.push(...rgb[axis], ...rgb[axis]);
    }
  }
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(positions);
  geometry.setColors(colors);
  const axes = new LineSegments2(geometry, edgeMaterial);
  axes.name = "axes";
  axes.renderOrder = 1;
  return axes;
}

// --- Motion: corrections that glide into place, and odometry playback ---------------

/** A correction glides into place over this long (milliseconds); `?glide=` slows it for recordings. */
const GLIDE_MS = Number(new URLSearchParams(location.search).get("glide")) || 700;
/** Larger graphs jump: redrawing every frame would cost more than it shows. */
const GLIDE_MAX_NODES = 3000;

const nextFrame = () => new Promise<number>((resolve) => requestAnimationFrame(resolve));

/** `state`'s poses between `from` and `to` (row-major 4x4 each), `t` of the way: slerp and lerp. */
function blend(from: Float64Array, to: Float64Array, t: number): Float64Array {
  const out = new Float64Array(to.length);
  const [a, b] = [new THREE.Matrix4(), new THREE.Matrix4()];
  const [qa, qb] = [new THREE.Quaternion(), new THREE.Quaternion()];
  const [pa, pb, scale] = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  const m = new THREE.Matrix4();
  for (let k = 0; k < to.length; k += 16) {
    a.fromArray(from, k).transpose();
    b.fromArray(to, k).transpose();
    a.decompose(pa, qa, scale);
    b.decompose(pb, qb, scale);
    m.compose(pa.lerp(pb, t), qa.slerp(qb, t), new THREE.Vector3(1, 1, 1));
    const e = m.transpose().elements;
    out.set(e, k);
  }
  return out;
}

/** Show `state`, gliding the poses from where they are drawn now (see {@link update}). */
async function animateTo(state: PoseGraphState): Promise<void> {
  const from = graph?.state.poses;
  if (!graph || !from || from.length !== state.poses.length || state.poses.length / 16 > GLIDE_MAX_NODES) {
    update(state);
    return;
  }
  const start = performance.now();
  for (;;) {
    const t = Math.min(1, (performance.now() - start) / GLIDE_MS);
    if (t >= 1) break;
    // Ease out: fast first, settling at the end.
    update({ ...state, poses: blend(from, state.poses, 1 - (1 - t) ** 3) });
    await nextFrame();
  }
  update(state);
}

let playing = false;

/**
 * Replay the drive: the keyframes' scans appear one after the other at
 * their poses, the newest highlighted, with the camera following the
 * vehicle, like watching LiDAR odometry run.
 */
export async function play(): Promise<void> {
  if (!graph || playing) return;
  playing = true;
  $("pg-play").textContent = "Stop";
  const g = graph;
  const n = g.state.poses.length / 16;
  const perSecond = Math.max(1, num("pg-play-rate") || 20);
  const follow = $<HTMLInputElement>("pg-play-follow").checked;
  const materials = scanObjects.map((o) => o?.material);
  for (const o of scanObjects) if (o) o.visible = false;
  const highlight = new THREE.PointsMaterial({ size: 3, sizeAttenuation: false, color: 0xffffff });
  viewer.registerPointMaterial(highlight);
  const { position, target } = viewer.getCamera();
  const offset = position.clone().sub(target);
  const start = performance.now();
  let shown = -1;
  try {
    while (playing && shown < n - 1) {
      const due = Math.min(n - 1, Math.floor(((performance.now() - start) / 1000) * perSecond));
      for (let i = shown + 1; i <= due; i++) {
        const o = scanObjects[i];
        if (!o) continue;
        o.visible = true;
        o.material = highlight;
        const previous = scanObjects[shown];
        if (previous && shown >= 0) previous.material = materials[shown]!;
        shown = i;
      }
      if (follow && shown >= 0) {
        const at = renderPosition(g.state, shown);
        viewer.setCamera(at.clone().add(offset), at);
      }
      viewer.requestRender();
      $("pg-play-at").textContent = `${shown + 1} / ${n}`;
      await nextFrame();
    }
  } finally {
    scanObjects.forEach((o, i) => {
      if (o) {
        o.material = materials[i]!;
        o.visible = showScans();
      }
    });
    highlight.dispose();
    playing = false;
    $("pg-play").textContent = "Play";
    viewer.requestRender();
  }
}
$<HTMLButtonElement>("pg-play").onclick = () => {
  if (playing) playing = false;
  else void play();
};

/** Frame the graph's nodes. */
/** Frame the graph's nodes (with room for their scans), keeping the viewing direction. */
export function frameGraph(): void {
  if (graph) fitGraph(graph.state);
}

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
    ...(state.gravityEdges
      ? [["Gravity", `${state.gravityEdges.toLocaleString()} keyframes`] as [string, string]]
      : []),
    ["Total error (χ²)", fmt(errors)],
  ]);
  $<HTMLButtonElement>("pg-loop").disabled = busy || selection.length !== 2;
  $<HTMLButtonElement>("pg-align").disabled = busy || (!aligning && selection.length !== 2);
  for (const id of ["pg-align-icp", "pg-align-accept"]) $<HTMLButtonElement>(id).disabled = busy;
  $<HTMLButtonElement>("pg-optimize").disabled = busy;
  $<HTMLButtonElement>("pg-undo").disabled = busy || steps.length === 0;
  $<HTMLButtonElement>("pg-map").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-compare").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-dem").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-dynamic").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-parts").disabled = busy || withScans === 0;
  $<HTMLInputElement>("pg-split").placeholder = String(
    state.nodeIds[g.sessions[1]?.first ?? Math.floor(state.nodeIds.length / 2)] ?? "",
  );
  $<HTMLButtonElement>("pg-find").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-merge").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-floor").disabled = busy || withScans === 0;
  $<HTMLButtonElement>("pg-gravity").disabled = busy;
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
    recordGraphStep({ poses, removed: out.removed });
    await animateTo(out.state);
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
  stopAligning();
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
 * make a graph. A ROS bag, or scans without poses, get their poses from
 * odometry.
 */
async function graphFiles(files: File[]): Promise<{ files: PoseGraphFiles; note: string } | null> {
  const odometry = {
    minRange: Math.max(0, num("pg-odom-min-range")),
    maxRange: Math.max(1, num("pg-odom-range") || 80),
    keyframeSpacing: Math.max(0, num("pg-keyframe-spacing")),
    deskew: $<HTMLInputElement>("pg-odom-deskew").checked,
  };
  const options = (scans: File[]) => ({
    voxel: Math.max(0, num("pg-voxel") || 0),
    displayPoints: Math.max(
      100,
      Math.min(num("pg-display") || 5000, Math.floor(DISPLAY_BUDGET / ((graph?.scans.length ?? 0) + Math.max(scans.length, 1000)))),
    ),
    sigmaT: num("pg-sigma-t") || 0.05,
    sigmaRDeg: num("pg-sigma-r") || 0.25,
    odometry,
  });
  const bag = files.find((f) => isBag(f.name));
  let scans = files.filter((f) => SCAN_FILE.test(f.name));
  const poses = posesFile(files);
  if (bag || (!poses && scans.length > 0)) {
    lastExtrinsic = null;
    const note = bag ? "" : " (poses from odometry)";
    return { files: { graph: bag ?? null, scans: bag ? [] : scans, extrinsic: null, ...options(scans) }, note };
  }
  if (!poses) {
    setStatus("No poses file (.g2o, .txt, .tum, .kitti), ROS bag (.bag, .mcap) or scans among the files", true);
    return null;
  }
  if (scans.length === 0) {
    setStatus(`No scans next to ${poses.name}: open a folder holding both`, true);
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
  lastExtrinsic = extrinsic;
  const shown = (graph?.scans.length ?? 0) + scans.length;
  return {
    files: {
      graph: poses,
      scans,
      extrinsic,
      ...options(scans),
      displayPoints: Math.max(100, Math.min(num("pg-display") || 5000, Math.floor(DISPLAY_BUDGET / shown))),
    },
    note,
  };
}

export async function open(picked: File[]): Promise<void> {
  // Opening replaces the graph: its scans do not count against the display budget.
  const current = graph;
  graph = null;
  const found = await graphFiles(picked);
  graph = current;
  if (!found) return;
  const { files, note: extrinsicNote } = found;
  const title = files.graph?.name ?? "the scans";
  setTool(null);
  const signal = startTask();
  busy = true;
  setStatus(
    files.graph && !isBag(files.graph.name)
      ? `Opening ${title} with ${files.scans.length.toLocaleString()} scans…`
      : `Opening ${title}: odometry first…`,
  );
  try {
    const opened = await openPoseGraph(
      files,
      (p: Progress) => {
        showProgress(p);
        setStatus(`Opening ${title}: ${p.note}…`);
      },
      signal,
    );
    graph = {
      name: opened.name,
      state: opened,
      scans: opened.scans,
      scanPoints: opened.scanPoints,
      sessions: [{ name: opened.name, first: 0, count: opened.poses.length / 16 }],
      imu: bagUps(opened.odometry, 0),
    };
    fieldA.value = fieldB.value = "";
    selection = [];
    steps.length = 0;
    build(graph);
    fitGraph(opened);
    const unmatched = opened.unmatched.length ? `; ${opened.unmatched.length} scans matched no pose` : "";
    const o = opened.odometry;
    const odometry = o
      ? ` from odometry over ${o.frames.toLocaleString()} scans${o.topic ? ` of ${o.topic}` : ""} ` +
        `(${Math.round(o.pathLength).toLocaleString()} m in ${Math.round(o.seconds)} s)`
      : "";
    setStatus(
      `Opened ${opened.name}: ${(opened.poses.length / 16).toLocaleString()} poses${odometry}, ` +
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
  // The bag's IMU levels the graph, as the up directions of an OXTS folder would.
  if (graph?.imu.nodes.length) await tieGravity(graph.imu.nodes, graph.imu.ups, "the bag's IMU");
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
export async function merge(picked: File[]): Promise<void> {
  if (!graph || busy) return;
  const found = await graphFiles(picked);
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
  let joinedImu = false;
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
          setStatus(`Joining ${files.graph?.name ?? "the scans"}: ${p.note}…`);
        },
        signal,
      );
      const g = graph!;
      g.name = `${g.name} + ${merged.name}`;
      g.scans = [...g.scans, ...merged.scans];
      g.scanPoints += merged.scanPoints;
      g.sessions.push({ name: merged.name, first: merged.offset, count: merged.scans.length });
      const imu = bagUps(merged.odometry, merged.offset);
      g.imu = { nodes: [...g.imu.nodes, ...imu.nodes], ups: [...g.imu.ups, ...imu.ups] };
      joinedImu = imu.nodes.length > 0;
      $<HTMLInputElement>("pg-split").value = String(merged.state.nodeIds[merged.offset]);
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
  // Tie the bags' IMUs again, the joined one's too (gravity ties replace the earlier ones).
  if (joinedImu && graph) await tieGravity(graph.imu.nodes, graph.imu.ups, "the bags' IMUs");
}

for (const [button, input] of [
  ["pg-merge", "pg-merge-input"],
  ["pg-merge-files", "pg-merge-files-input"],
] as const) {
  $<HTMLButtonElement>(button).onclick = () => $<HTMLInputElement>(input).click();
  $<HTMLInputElement>(input).onchange = (e) => {
    const target = e.target as HTMLInputElement;
    const files = [...(target.files ?? [])];
    target.value = "";
    if (files.length) void merge(files);
  };
}

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
const retryHeadings = () => Math.max(0, Math.round(num("pg-retry-headings")));

/** Find, register and add loops automatically, then optimise (the panel's settings). */
export const findLoops = (): Promise<void> =>
  run("Finding loops", async () => {
    const poses = graph!.state.poses.slice();
    const signal = startTask();
    try {
      const found = await findPoseGraphLoops(
        {
          maxDistance: Math.max(0, num("pg-find-radius")),
          drift: Math.max(0, num("pg-find-drift")) / 100,
          minTravel: Math.max(0, num("pg-find-travel")),
          spacing: Math.max(0, num("pg-find-spacing")),
          minFitness: Math.min(100, Math.max(0, num("pg-find-fitness"))) / 100,
          inlierDistance: inlierDistance(),
          retryHeadings: retryHeadings(),
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
      if (found.added.length) recordGraphStep({ poses, added: found.edges });
      await animateTo(found.state);
      const { optimized } = found;
      setStatus(
        found.candidates === 0
          ? "No loop candidates: raise the search radius or lower the minimum travel"
          : `Added ${found.added.length} of ${found.candidates} candidate loop${found.candidates === 1 ? "" : "s"}` +
              (found.added.some((a) => a.retried)
                ? ` (${found.added.filter((a) => a.retried).length} registered as the same place)`
                : "") +
              (found.implausible ? `, ${found.implausible} rejected as further off than drift allows` : "") +
              (optimized
                ? `; χ² ${fmt(optimized.initialCost)} → ${fmt(optimized.finalCost)} in ${optimized.iterations} iterations`
                : " (none overlapped enough)"),
      );
    } finally {
      endTask(signal);
    }
  });
$<HTMLButtonElement>("pg-find").onclick = () => void findLoops();

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
    recordGraphStep({ poses, planes: floor.planes });
    await animateTo(floor.state);
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
    recordGraphStep({ poses: before });
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

// --- Align by hand: two scans overlaid, B lined up by hand, then ICP ---------------

let aligning: {
  from: number;
  to: number;
  /** A's pose in render space. */
  frame: THREE.Matrix4;
  /** Where B is drawn: its pose in render space, moved by the gizmo. */
  pivot: THREE.Object3D;
  objects: THREE.Points[];
  materials: THREE.PointsMaterial[];
  handle: ReturnType<typeof viewer.attachGizmo>;
  /** Which scans were shown before. */
  shown: boolean[];
} | null = null;

const ALIGN_FIELDS = ["x", "y", "z", "roll", "pitch", "yaw"] as const;

/** `m` as row-major numbers, and back. */
const rowMajor = (m: THREE.Matrix4) => m.clone().transpose().elements.slice();
const fromRowMajor = (v: ArrayLike<number>) => new THREE.Matrix4().fromArray(Array.from(v)).transpose();

/** B in A's frame, as the pivot now has it. */
function alignedGuess(): THREE.Matrix4 {
  const a = aligning!;
  a.pivot.updateMatrix();
  return a.frame.clone().invert().multiply(a.pivot.matrix);
}

/** Put B at `guess` (in A's frame) and show its numbers. */
function placeAligned(guess: THREE.Matrix4): void {
  const a = aligning!;
  a.frame.clone().multiply(guess).decompose(a.pivot.position, a.pivot.quaternion, a.pivot.scale);
  a.pivot.updateMatrix();
  showAlignedNumbers(guess);
  viewer.requestRender();
}

function showAlignedNumbers(guess: THREE.Matrix4): void {
  const p = new THREE.Vector3().setFromMatrixPosition(guess);
  const e = new THREE.Euler().setFromRotationMatrix(guess, "ZYX");
  const deg = THREE.MathUtils.radToDeg;
  const values = [p.x, p.y, p.z, deg(e.x), deg(e.y), deg(e.z)];
  ALIGN_FIELDS.forEach((f, k) => ($<HTMLInputElement>(`pg-align-${f}`).value = values[k].toFixed(k < 3 ? 3 : 2)));
}

/** The guess the number fields describe. */
function typedGuess(): THREE.Matrix4 {
  const [x, y, z, roll, pitch, yaw] = ALIGN_FIELDS.map((f) => num(`pg-align-${f}`) || 0);
  const rad = THREE.MathUtils.degToRad;
  return new THREE.Matrix4()
    .makeRotationFromEuler(new THREE.Euler(rad(roll), rad(pitch), rad(yaw), "ZYX"))
    .setPosition(x, y, z);
}

function stopAligning(): void {
  const a = aligning;
  if (!a) return;
  aligning = null;
  a.handle.detach();
  viewer.overlay.remove(a.pivot);
  for (const o of a.objects) {
    o.removeFromParent();
    o.geometry.dispose();
  }
  for (const m of a.materials) m.dispose();
  scanObjects.forEach((o, i) => {
    if (o) o.visible = a.shown[i];
  });
  $("pg-align-panel").hidden = true;
  $("pg-align").setAttribute("aria-pressed", "false");
  renderInfo();
  viewer.requestRender();
}

/** Show A's and B's scans alone, B on a gizmo at the graph's current guess. */
export function startAligning(): void {
  if (!graph || selection.length !== 2) {
    setStatus("Pick or type Node A and Node B first", true);
    return;
  }
  stopMoving();
  stopAligning();
  const [from, to] = selection;
  const { scans, state } = graph;
  const [scanA, scanB] = [scans[from], scans[to]];
  if (!scanA?.length || !scanB?.length) {
    setStatus("Both nodes need a scan to align", true);
    return;
  }
  const frame = renderMatrix(state.poses, from);
  const cloud = (positions: Float32Array, color: number) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingSphere();
    const material = new THREE.PointsMaterial({ size: Number(pointSizeInput.value) + 2, sizeAttenuation: false, color });
    viewer.registerPointMaterial(material);
    return new THREE.Points(geometry, material);
  };
  const objects = [cloud(scanA, 0xff9800), cloud(scanB, 0x00e5ff)];
  objects[0].matrixAutoUpdate = false;
  objects[0].matrix.copy(frame);
  const pivot = new THREE.Object3D();
  pivot.add(objects[1]);
  viewer.overlay.add(objects[0], pivot);
  const shown = scanObjects.map((o) => !!o?.visible);
  for (const o of scanObjects) if (o) o.visible = false;
  aligning = {
    from,
    to,
    frame,
    pivot,
    objects,
    materials: objects.map((o) => o.material as THREE.PointsMaterial),
    shown,
    handle: viewer.attachGizmo(
      pivot,
      $<HTMLSelectElement>("pg-align-mode").value as "translate" | "rotate",
      () => {
        if (aligning) showAlignedNumbers(alignedGuess());
      },
      () => undefined,
    ),
  };
  placeAligned(frame.clone().invert().multiply(renderMatrix(state.poses, to)));
  $("pg-align-panel").hidden = false;
  $("pg-align").setAttribute("aria-pressed", "true");
  $("pg-align-info").textContent = "";
  viewer.centerOn(new THREE.Vector3().setFromMatrixPosition(frame));
  renderInfo();
  const ids = state.nodeIds;
  setStatus(`Aligning node ${ids[to]} (cyan) onto node ${ids[from]} (orange): move it close, then ICP`);
}

$<HTMLButtonElement>("pg-align").onclick = () => (aligning ? stopAligning() : startAligning());
$<HTMLButtonElement>("pg-align-cancel").onclick = () => {
  stopAligning();
  setStatus("Alignment cancelled");
};
$<HTMLSelectElement>("pg-align-mode").onchange = () =>
  aligning?.handle.setMode($<HTMLSelectElement>("pg-align-mode").value as "translate" | "rotate");
for (const f of ALIGN_FIELDS) {
  $<HTMLInputElement>(`pg-align-${f}`).onchange = () => {
    if (aligning) placeAligned(typedGuess());
  };
}

/** ICP from where B is now; B moves to the result. */
export const alignWithIcp = (): Promise<void> =>
  run("ICP", async () => {
    const a = aligning;
    if (!a) return;
    const out = await registerPoseGraphPair({
      from: a.from,
      to: a.to,
      guess: rowMajor(alignedGuess()),
      maxIterations: Math.max(1, num("pg-icp-iterations") || 50),
      overlap: Math.min(100, Math.max(10, num("pg-icp-overlap") || 80)) / 100,
      inlierDistance: inlierDistance(),
    });
    if (aligning !== a) return;
    placeAligned(fromRowMajor(out.matrix));
    const text =
      `ICP RMS ${fmt(out.rmsInitial)} → ${fmt(out.rmsFinal)}, overlap ${Math.round(out.fitness * 100)} %` +
      (out.converged ? "" : " (not converged)");
    $("pg-align-info").textContent = text;
    setStatus(`${text}: add the loop if the scans line up`);
  });
$<HTMLButtonElement>("pg-align-icp").onclick = () => void alignWithIcp();

/** Add a loop edge where B sits now, then optimise. */
export const acceptAlignment = (): Promise<void> =>
  run("Adding the loop", async () => {
    const a = aligning;
    if (!a) return;
    const ids = graph!.state.nodeIds;
    const poses = graph!.state.poses.slice();
    const { state, edge } = await addPoseGraphEdge({
      from: a.from,
      to: a.to,
      matrix: rowMajor(alignedGuess()),
      sigmaT: num("pg-loop-sigma-t") || 0.1,
      sigmaRDeg: num("pg-loop-sigma-r") || 1,
    });
    stopAligning();
    recordGraphStep({ poses, added: [edge] });
    update(state);
    setStatus(`Loop ${ids[a.from]} – ${ids[a.to]} added by hand; optimising…`);
    const optimized = await optimize();
    setSelection([]);
    setStatus(`Loop ${ids[a.from]} – ${ids[a.to]} added by hand; ${optimized}`);
  });
$<HTMLButtonElement>("pg-align-accept").onclick = () => void acceptAlignment();

$<HTMLButtonElement>("pg-goto").onclick = () => {
  const node = nodeA();
  if (graph && node !== null) viewer.centerOn(renderPosition(graph.state, node));
  else setStatus("Pick or type Node A first: the node to look at", true);
};
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
    recordGraphStep({ poses: graph!.state.poses.slice(), flipped: node });
    update(state);
    setStatus(`Node ${state.nodeIds[node]} ${fixed ? "is held in place by the optimiser (red)" : "is free again"}`);
  });

// --- Gravity from an IMU ---------------------------------------------------------

/** The scan-to-pose matrix the scans were opened with (up vectors turn with it). */
let lastExtrinsic: number[] | null = null;

/** Numbers in a text. */
const numbers = (text: string) => text.trim().split(/\s+/).map(Number);

/**
 * Up directions in scan coordinates by frame number, from a KITTI OXTS
 * folder (`oxts/data/0000000042.txt`: roll and pitch are fields 4 and 5;
 * `calib_imu_to_velo.txt` turns them into the Velodyne frame) or from a
 * text file of `frame ux uy uz` lines.
 */
async function readUps(files: File[]): Promise<Map<number, [number, number, number]>> {
  const ups = new Map<number, [number, number, number]>();
  const calib = files.find((f) => /^calib_imu_to_velo\.txt$/i.test(f.name));
  let rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  if (calib) {
    const line = (await calib.text()).split("\n").find((l) => l.startsWith("R:"));
    if (line) rotation = numbers(line.slice(2));
  }
  for (const file of files) {
    if (!/\.txt$/i.test(file.name) || /^calib/i.test(file.name)) continue;
    const text = await file.text();
    const frame = /^(\d+)\.txt$/.exec(file.name);
    const values = numbers(text);
    if (frame && values.length >= 30) {
      // OXTS: the IMU is turned by Rz(yaw) Ry(pitch) Rx(roll), so up in its frame is that matrix's last row.
      const [roll, pitch] = [values[3], values[4]];
      const imu = [-Math.sin(pitch), Math.cos(pitch) * Math.sin(roll), Math.cos(pitch) * Math.cos(roll)];
      const r = rotation;
      ups.set(Number(frame[1]), [
        r[0] * imu[0] + r[1] * imu[1] + r[2] * imu[2],
        r[3] * imu[0] + r[4] * imu[1] + r[5] * imu[2],
        r[6] * imu[0] + r[7] * imu[1] + r[8] * imu[2],
      ]);
      continue;
    }
    for (const line of text.split("\n")) {
      const v = numbers(line);
      if (v.length === 4 && v.every(Number.isFinite)) ups.set(v[0], [v[1], v[2], v[3]]);
    }
  }
  return ups;
}

export async function addGravity(files: File[]): Promise<void> {
  const byFrame = await readUps(files);
  const state = graph!.state;
  const e = lastExtrinsic;
  const nodes: number[] = [];
  const ups: number[] = [];
  state.nodeIds.forEach((id, i) => {
    const up = byFrame.get(id);
    if (!up) return;
    // Into the pose frame, as the scans were.
    const [x, y, z] = up;
    nodes.push(i);
    ups.push(
      ...(e
        ? [e[0] * x + e[1] * y + e[2] * z, e[4] * x + e[5] * y + e[6] * z, e[8] * x + e[9] * y + e[10] * z]
        : up),
    );
  });
  if (nodes.length === 0) {
    setStatus("No up directions matched a node: pick an OXTS folder (or frame ux uy uz lines) named by frame", true);
    return;
  }
  await tieGravity(nodes, ups, null);
}

/** Tie `nodes` to the up directions `ups` (three each, in the scans' frame), then optimise; `source` names them in the status. */
async function tieGravity(nodes: number[], ups: number[], source: string | null): Promise<void> {
  await run("Adding gravity", async () => {
    const state = graph!.state;
    const poses = state.poses.slice();
    const out = await setPoseGraphGravity(
      nodes,
      new Float64Array(ups),
      num("pg-gravity-sigma") || 0.1,
      kernel(),
      $<HTMLInputElement>("pg-gravity-calibrate").checked,
    );
    recordGraphStep({ poses, gravity: true });
    await animateTo(out.state);
    setStatus(
      `Gravity${source ? ` from ${source}` : ""} tied to ${out.tied.toLocaleString()} of ${(state.poses.length / 16).toLocaleString()} keyframes ` +
        `(up directions spread ${out.spread.toFixed(2)}°` +
        (Number.isFinite(out.calibratedSpread) ? `, ${out.calibratedSpread.toFixed(2)}° with the IMU's mounting estimated` : "") +
        `; σ ${out.sigmaDeg.toFixed(2)}°); χ² ${fmt(out.initialCost)} → ${fmt(out.finalCost)} in ${out.iterations} iterations`,
    );
  });
}

$<HTMLButtonElement>("pg-gravity").onclick = () => $<HTMLInputElement>("pg-gravity-input").click();
$<HTMLInputElement>("pg-gravity-input").onchange = (event) => {
  const target = event.target as HTMLInputElement;
  const files = [...(target.files ?? [])];
  target.value = "";
  if (files.length) void addGravity(files);
};

const kernel = () => ($<HTMLInputElement>("pg-robust").checked ? Math.max(0, num("pg-kernel")) : 0);

async function optimize(): Promise<string> {
  const out = await optimizePoseGraph(kernel());
  await animateTo(out.state);
  return `χ² ${fmt(out.initialCost)} → ${fmt(out.finalCost)} in ${out.iterations} iterations (${Math.round(out.millis)} ms)`;
}

$<HTMLButtonElement>("pg-optimize").onclick = () =>
  run("Optimising", async () => {
    setStatus("Optimising…");
    recordGraphStep({ poses: graph!.state.poses.slice() });
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
      retryHeadings: retryHeadings(),
      sigmaT: num("pg-loop-sigma-t") || 0.1,
      sigmaRDeg: num("pg-loop-sigma-r") || 1,
    });
    recordGraphStep({ poses, added: [loop.edge] });
    update(loop.state);
    const icp =
      `ICP RMS ${fmt(loop.rmsInitial)} → ${fmt(loop.rmsFinal)}, overlap ${Math.round(loop.fitness * 100)} %` +
      (loop.retried ? ", registered as the same place" : "") +
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
    // Later planes move down as each goes: removing `first` repeatedly takes them all.
    for (let k = 0; k < (step.planes?.count ?? 0); k++) await removePoseGraphPlane(step.planes!.first);
    if (step.flipped !== undefined) await setPoseGraphFixed(step.flipped, !graph!.state.fixed[step.flipped]);
    if (step.gravity) await clearPoseGraphGravity();
    if (step.removed) await insertPoseGraphEdges(step.removed);
    await animateTo(await setPoseGraphPoses(step.poses));
    const plural = (n: number) => (n === 1 ? "" : "s");
    setStatus(
      step.added
        ? step.added.length === 1
          ? "Removed the last loop"
          : `Removed the ${step.added.length} loop${plural(step.added.length)} found`
        : step.planes
          ? "Removed the floor"
          : step.gravity
          ? "Removed the gravity ties"
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

/**
 * The map as it is now, colored by how far each point moved from its place
 * as loaded: where the corrections moved the map, and by how much.
 */
export const compareWithStart = (): Promise<void> =>
  run("Comparing", async () => {
    setStatus("Building the map with each point's correction…");
    const map = addEntry(await poseGraphMap(Math.max(0, num("pg-map-voxel") || 0), false, true));
    record({ label: "the corrected map", added: [map] });
    renderList();
    await colorByField(map, "correction");
    const values = map.field!.values;
    let [sum, max] = [0, 0];
    for (const v of values) [sum, max] = [sum + v, Math.max(max, v)];
    setStatus(
      `${map.cloud.name} is colored by how far each point moved from its place as loaded ` +
        `(mean ${fmt(sum / values.length)} m, max ${fmt(max)} m)`,
    );
  });
$<HTMLButtonElement>("pg-compare").onclick = () => void compareWithStart();

// --- From the map to the analysis tools -----------------------------------------

/**
 * The corrected map through ground extraction (CSF, ground points only) to
 * a terrain model (DEM) in the color ramp, ready to save as a GeoTIFF.
 */
export const groundAndDem = (): Promise<void> =>
  run("Making the DEM", async () => {
    const voxel = Math.max(0, num("pg-map-voxel")) || 0.2;
    const cell = Math.max(0.05, num("pg-dem-cell") || 1);
    setStatus("Building the map…");
    const map = addEntry(await poseGraphMap(voxel));
    setStatus("Extracting the ground…");
    const ground = addEntry(
      await extractGround({ id: map.cloud.id, clothResolution: 1, classThreshold: 0.3, rigidness: "relief", output: "ground" }),
    );
    record({ label: "the ground", added: [map, ground], hide: [map] });
    hideEntry(map);
    setStatus("Rasterizing the ground…");
    const out = await rasterizeCloud({
      id: ground.cloud.id,
      cell,
      height: "mean",
      percentile: 50,
      // Only cells with ground: filling would spread over the unsurveyed box.
      fillEmpty: false,
      class: null,
    });
    // Heights in a sequential ramp over the 2nd to 98th percentile, so a
    // few stray points below the road do not wash the terrain out.
    display.ramp = "Blue > Green > Yellow > Red";
    const dem = showRaster(ground, out);
    const sorted = Float32Array.from(out.cellHeights.filter(Number.isFinite)).sort();
    if (sorted.length) {
      display.range = { lo: sorted[Math.floor(sorted.length * 0.02)], hi: sorted[Math.floor(sorted.length * 0.98)] };
      refreshColors(dem);
      distanceChanged.emit();
    }
    if (showScans()) $<HTMLInputElement>("pg-show-scans").click();
    setStatus(
      `DEM of ${ground.cloud.count.toLocaleString()} ground points of ${map.cloud.count.toLocaleString()}: ` +
        `${out.nx} × ${out.ny} cells of ${fmt(cell)} m (save it from the Rasterize panel)`,
    );
  });
$<HTMLButtonElement>("pg-dem").onclick = () => void groundAndDem();

/**
 * The map without its dynamic points (cars and people that moved while
 * the scans were taken, found by visibility: other scans saw through
 * them), and those points as a second cloud in red.
 */
export const removeDynamic = (): Promise<void> =>
  run("Finding dynamic points", async () => {
    setStatus("Finding the points other scans saw through…");
    const found = await detectPoseGraphDynamic(
      Math.max(1, Math.round(num("pg-dyn-window") || 10)),
      Math.max(0.05, num("pg-dyn-margin") || 0.5),
      Math.max(1, Math.round(num("pg-dyn-votes") || 3)),
    );
    const voxel = Math.max(0, num("pg-map-voxel") || 0);
    setStatus("Building the static map…");
    const map = addEntry(await poseGraphMap(voxel, false, false, [], "static", 1));
    const moving = found.dynamic ? addEntry(await poseGraphMap(voxel, false, false, [], "dynamic", 2)) : null;
    if (moving) {
      moving.solid = [255, 64, 64];
      moving.mode = "solid";
      refreshColors(moving);
    }
    record({ label: "the static map", added: moving ? [map, moving] : [map] });
    renderList();
    if (showScans()) $<HTMLInputElement>("pg-show-scans").click();
    const share = found.total ? (100 * found.dynamic) / found.total : 0;
    setStatus(
      `${found.dynamic.toLocaleString()} dynamic points of ${found.total.toLocaleString()} (${share.toFixed(2)} %) ` +
        `left out of ${map.cloud.name}${moving ? `, shown in red as ${moving.cloud.name}` : ""} ` +
        `(${(found.millis / 1000).toFixed(1)} s)`,
    );
  });
$<HTMLButtonElement>("pg-dynamic").onclick = () => void removeDynamic();

/** Keyframes compared by Compare parts: those within this distance (metres) of the other part's path. */
const PART_REACH = 50;

/**
 * The map of the path up to a node against the map of the rest, by M3C2:
 * for two joined sessions, what changed between them; for one drive that
 * passes a place twice, how well the two passes agree.
 */
export const compareParts = (): Promise<void> =>
  run("Comparing the parts", async () => {
    const { state, sessions } = graph!;
    const n = state.poses.length / 16;
    const typed = $<HTMLInputElement>("pg-split").value;
    const split =
      typed === "" ? (sessions[1]?.first ?? Math.floor(n / 2)) : state.nodeIds.indexOf(Number(typed));
    if (split <= 0 || split >= n) {
      setStatus(`Split at a node between the first and the last (not ${typed})`, true);
      return;
    }
    const voxel = Math.max(0, num("pg-map-voxel")) || 0.3;
    // Only where the parts meet: each part's nodes near the other's path.
    // A long drive's whole map would not fit, and elsewhere there is
    // nothing to compare.
    const near = (from: number, to: number, others: [number, number]): number[] => {
      const out: number[] = [];
      for (let i = from; i < to; i++) {
        const p = nodePosition(state, i);
        for (let j = others[0]; j < others[1]; j++) {
          if (p.distanceToSquared(nodePosition(state, j)) < PART_REACH * PART_REACH) {
            out.push(i);
            break;
          }
        }
      }
      return out;
    };
    const [first, second] = [near(0, split, [split, n]), near(split, n, [0, split])];
    if (first.length === 0 || second.length === 0) {
      setStatus(`The two parts never come within ${PART_REACH} m of each other: nothing to compare`, true);
      return;
    }
    setStatus(`Building the maps where the parts meet (${first.length} and ${second.length} keyframes)…`);
    const ids = state.nodeIds;
    const before = addEntry(await poseGraphMap(voxel, false, false, first, `before_${ids[split]}`));
    const after = addEntry(await poseGraphMap(voxel, false, false, second, `from_${ids[split]}`));
    record({ label: "the two maps", added: [before, after] });
    hideEntry(before);
    renderList();
    for (const [id, value] of Object.entries({ "m3c2-normal": "1", "m3c2-projection": "0.5", "m3c2-depth": "2", "m3c2-core": "0.5" })) {
      $<HTMLInputElement>(id).value = value;
    }
    const result = await runM3c2(after, before);
    // A symmetric range at the 98th percentile of the changes (at least half
    // a metre), so a moved car stands out rather than a few extreme cores.
    const sizes = Float32Array.from(result.c2c!.distances.filter(Number.isFinite), Math.abs).sort();
    if (sizes.length) {
      const m = Math.max(0.5, sizes[Math.floor(sizes.length * 0.98)]);
      display.range = { lo: -m, hi: m };
      refreshColors(result);
      distanceChanged.emit();
    }
    if (showScans()) $<HTMLInputElement>("pg-show-scans").click();
    await listChanges(result.cloud.id);
  });
$<HTMLButtonElement>("pg-parts").onclick = () => void compareParts();

/** Smallest change (metres) of a point counted in a changed object. */
const CHANGE_MIN = 0.3;
/** Significant change points closer than this (metres) are one object. */
const CHANGE_LINK = 1;
/** Fewest core points of a listed object. */
const CHANGE_MIN_POINTS = 8;
/** Objects listed at most. */
const CHANGE_LIMIT = 50;

/**
 * The changed objects of an M3C2 result, largest first: where each is,
 * how big, and by how much it changed. Clicking one frames it in the view.
 */
async function listChanges(id: number): Promise<void> {
  const flat = await changedObjects(id, CHANGE_MIN, CHANGE_LINK, CHANGE_MIN_POINTS);
  const count = flat.length / 11;
  // The result now has a change_object field for the scalar field tools.
  const result = entries.get(id);
  if (result && !result.cloud.scalarNames.includes("change_object")) {
    result.cloud.scalarNames.push("change_object");
    listChanged.emit();
  }
  $("pg-changes").hidden = false;
  $("pg-changes-hint").textContent =
    count === 0
      ? "No significant change forms an object."
      : `${count.toLocaleString()} changed object${count === 1 ? "" : "s"} (points that changed by ${CHANGE_MIN} m or more), largest first` +
        (count > CHANGE_LIMIT ? ` (the first ${CHANGE_LIMIT} listed)` : "") +
        ". Color the result by its change_object field to see them all.";
  $("pg-change-list").replaceChildren(
    ...Array.from({ length: Math.min(count, CHANGE_LIMIT) }, (_, k) => {
      const o = flat.subarray(k * 11, k * 11 + 11);
      const size = [o[7] - o[4], o[8] - o[5], o[9] - o[6]].map((v) => fmt(Math.max(v, 0))).join(" × ");
      const li = document.createElement("li");
      const name = document.createElement("button");
      name.className = "name link";
      name.textContent = `#${k + 1} · ${size} m`;
      name.title = "Frame it in the view";
      // With some room round it, so its surroundings show what changed.
      name.onclick = () =>
        viewer.frameBox(
          new THREE.Box3(toRender([o[4], o[5], o[6]]), toRender([o[7], o[8], o[9]])).expandByScalar(
            Math.max(2, 0.5 * Math.max(o[7] - o[4], o[8] - o[5])),
          ),
        );
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = `${o[10] > 0 ? "+" : ""}${fmt(o[10])} m · ${o[0].toLocaleString()} points`;
      li.append(name, meta);
      return li;
    }),
  );
  const listed = status();
  setStatus(`${listed} · ${count.toLocaleString()} changed object${count === 1 ? "" : "s"} (see Changes)`);
}

const status = () => $("status").textContent ?? "";

$<HTMLButtonElement>("pg-close").onclick = async () => {
  if (busy) return;
  setTool(null);
  stopMoving();
  stopAligning();
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
$<HTMLInputElement>("pg-axes").onchange = () => graph && update(graph.state);
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
  // The two scans being aligned are all there is to see: larger still.
  for (const material of aligning?.materials ?? []) material.size = size + 1;
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
