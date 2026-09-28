/**
 * Pose graph panel, in the spirit of interactive_slam: a g2o graph or a
 * TUM / KITTI trajectory (as an odometry chain) with a scan per node. Each
 * scan is drawn at its node's pose, so optimising only moves matrices. Pick
 * two nodes to close a loop: ICP registers their scans from the current
 * relative pose, the result becomes a loop edge and the graph is optimised.
 * The scans at their final poses become an ordinary cloud for the other tools.
 */

import * as THREE from "three";
import {
  addPoseGraphLoop,
  closePoseGraph,
  exportPoseGraph,
  openPoseGraph,
  optimizePoseGraph,
  poseGraphMap,
  removePoseGraphEdge,
  setPoseGraphPoses,
} from "../api";
import { CANCELLED, type PoseFormat, type PoseGraphState, type Progress } from "../protocol";
import { $, download, errorText, fillTable, fmt, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { globalShift, listChanged, viewer } from "./state";
import { endTask, showProgress, startTask } from "./tasks";
import { setTool, toggleTool, type Tool } from "./tools";

interface Graph {
  name: string;
  state: PoseGraphState;
  scans: (Float32Array | null)[];
  scanPoints: number;
}

/** Undo information: the poses before the step, and the loop edge it added. */
interface Step {
  poses: Float64Array;
  addedEdge?: number;
}

/** Scan files (the poses are g2o, TUM or KITTI text). */
const SCAN_FILE = /\.(pcd|ply|bin|las|laz|xyz|pts)$/i;
const POSE_FILE = /\.(g2o|tum|kitti|txt|csv)$/i;
/** Text files found next to KITTI scans that are not poses. */
const NOT_POSES = /^(calib|times)\.txt$/i;
/** Draw at most this many scan points overall. */
const DISPLAY_BUDGET = 6_000_000;
const PICK_RADIUS_PX = 12;

const HUES = 12;
const scanMaterials = Array.from(
  { length: HUES },
  (_, k) =>
    new THREE.PointsMaterial({
      size: 2,
      sizeAttenuation: false,
      color: new THREE.Color().setHSL(k / HUES, 0.65, 0.6),
    }),
);
const nodeMaterial = new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, color: 0xffffff });
const selectedMaterial = new THREE.PointsMaterial({
  size: 12,
  sizeAttenuation: false,
  vertexColors: true,
  depthTest: false,
});
const edgeMaterial = new THREE.LineBasicMaterial({ vertexColors: true, depthWrite: false });
const ODOMETRY_COLOR = [0.45, 0.6, 0.8];
const LOOP_COLOR = [1, 0.6, 0.15];
const SELECTED_COLORS = [new THREE.Color(0xffeb3b), new THREE.Color(0x00e5ff)];

let graph: Graph | null = null;
const group = new THREE.Group();
group.name = "pose-graph";
viewer.scene.add(group);
let scanObjects: (THREE.Points | null)[] = [];
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
}

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
    const points = new THREE.Points(geometry, scanMaterials[i % HUES]);
    points.matrixAutoUpdate = false;
    group.add(points);
    return points;
  });
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
  for (let i = 0; i < n; i++) local(i).toArray(nodePositions, i * 3);
  const nodes = new THREE.Points(new THREE.BufferGeometry(), nodeMaterial);
  nodes.geometry.setAttribute("position", new THREE.BufferAttribute(nodePositions, 3));
  nodes.name = "nodes";

  const e = state.edges.length / 2;
  const edgePositions = new Float32Array(e * 6);
  const edgeColors = new Float32Array(e * 6);
  for (let k = 0; k < e; k++) {
    local(state.edges[2 * k]).toArray(edgePositions, k * 6);
    local(state.edges[2 * k + 1]).toArray(edgePositions, k * 6 + 3);
    const color = state.edgeKinds[k] ? LOOP_COLOR : ODOMETRY_COLOR;
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
  selected.renderOrder = 2;

  for (const object of [nodes, edges, selected]) {
    object.position.copy(placed);
    object.geometry.computeBoundingSphere();
    group.add(object);
  }
  viewer.requestRender();
  renderInfo();
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
    ["Total error (χ²)", fmt(errors)],
  ]);
  $<HTMLButtonElement>("pg-loop").disabled = busy || selection.length !== 2;
  $<HTMLButtonElement>("pg-optimize").disabled = busy;
  $<HTMLButtonElement>("pg-undo").disabled = busy || steps.length === 0;
  $<HTMLButtonElement>("pg-map").disabled = busy || withScans === 0;
}

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
fieldA.oninput = fieldB.oninput = readSelection;

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

async function open(files: File[]): Promise<void> {
  let scans = files.filter((f) => SCAN_FILE.test(f.name));
  const poses = posesFile(files);
  if (!poses) {
    if (scans.length === 0) {
      setStatus("No poses file (.g2o, .txt, .tum, .kitti) or scans among the files", true);
      return;
    }
    pendingScans = scans;
    setStatus(`${scans.length.toLocaleString()} scans found but no poses: now open the poses file (Open files…)`);
    return;
  }
  if (scans.length === 0) scans = pendingScans;
  pendingScans = [];
  if (scans.length === 0) {
    setStatus(`No scans next to ${poses.name}: open a folder holding both, or the scans first`, true);
    return;
  }
  let extrinsic: number[] | null;
  try {
    extrinsic = parseMatrix($<HTMLTextAreaElement>("pg-extrinsic").value);
  } catch (err) {
    setStatus(errorText(err), true);
    return;
  }
  let extrinsicNote = "";
  if (!extrinsic) {
    extrinsic = await kittiExtrinsic(files);
    if (extrinsic) extrinsicNote = " (scans moved by calib.txt's Tr)";
  }
  setTool(null);
  const signal = startTask();
  busy = true;
  setStatus(`Opening ${poses.name} with ${scans.length.toLocaleString()} scans…`);
  try {
    const opened = await openPoseGraph(
      {
        graph: poses,
        scans,
        voxel: Math.max(0, num("pg-voxel") || 0),
        displayPoints: Math.max(100, Math.min(num("pg-display") || 5000, Math.floor(DISPLAY_BUDGET / scans.length))),
        extrinsic,
        sigmaT: num("pg-sigma-t") || 0.1,
        sigmaRDeg: num("pg-sigma-r") || 1,
      },
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
      sigmaT: num("pg-loop-sigma-t") || 0.1,
      sigmaRDeg: num("pg-loop-sigma-r") || 1,
    });
    steps.push({ poses, addedEdge: loop.edge });
    update(loop.state);
    const icp =
      `ICP RMS ${fmt(loop.rmsInitial)} → ${fmt(loop.rmsFinal)}` + (loop.converged ? "" : " (not converged)");
    setStatus(`Loop ${ids[from]} – ${ids[to]} added (${icp}); optimising…`);
    const optimized = await optimize();
    setSelection([]);
    setStatus(`Loop ${ids[from]} – ${ids[to]} added (${icp}); ${optimized}`);
  });

$<HTMLButtonElement>("pg-undo").onclick = () =>
  run("Undo", async () => {
    const step = steps.pop();
    if (!step) return;
    if (step.addedEdge !== undefined) await removePoseGraphEdge(step.addedEdge);
    update(await setPoseGraphPoses(step.poses));
    setStatus(step.addedEdge !== undefined ? "Removed the last loop" : "Undid the optimisation");
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
  await closePoseGraph();
  graph = null;
  fieldA.value = fieldB.value = "";
  selection = [];
  steps.length = 0;
  clearGroup();
  viewer.requestRender();
  renderInfo();
};

// The graph follows the clouds' global shift.
listChanged.add(() => {
  if (graph && globalShift().join() !== drawnShift) update(graph.state);
});

renderInfo();
