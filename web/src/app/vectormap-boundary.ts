/** Editing lane boundaries vertex by vertex. */

import * as THREE from "three";
import { $, fmt } from "./dom";
import { globalShift, viewer } from "./state";
import { toggleTool, type Tool } from "./tools";
import { resampleLaneCurve as resample } from "../lane-surface-overlap";
import {
  apply,
  boundaryVertex,
  busy,
  clearJunctionPreview,
  draw,
  hint,
  type LaneView,
  type MapView,
  renderHint,
  setActiveBoundary,
  setBoundaryVertex,
  setView,
  view,
  type XYZ,
} from "./vectormap";

export let editingVertices = false;

export let drag: {
  before: MapView; boundary: number; index: number; point: XYZ;
  offset: [number, number]; start: [number, number]; moved: boolean;
} | null = null;

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

export function boundaryVertexPoint(): XYZ | undefined {
  return boundaryVertex ? view.boundaries.find(b => b.id === boundaryVertex!.boundary)?.points[boundaryVertex.index] : undefined;
}
export function boundaryInputs(): void {
  const p = boundaryVertexPoint(), blocked = busy || !!drag;
  const height = $<HTMLInputElement>("vm-boundary-z").valueAsNumber;
  $<HTMLSelectElement>("vm-boundary").disabled = blocked || !view.boundaries.length;
  for (const id of ["vm-boundary-vertex", "vm-boundary-z", "vm-boundary-focus"]) {
    ($<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>(id)).disabled = blocked || !p;
  }
  $<HTMLButtonElement>("vm-boundary-apply").disabled = blocked || !p || !Number.isFinite(height) || height === p[2];
  $("vm-boundary-edit-status").textContent = !p ? "" : !Number.isFinite(height)
    ? "Enter a finite height in metres."
    : height === p[2] ? "Height unchanged; no edit to apply." : `Height change: ${fmt(height - p[2])} m. X and Y will stay fixed.`;
}
function renderBoundaryVertex(): void {
  const p = boundaryVertexPoint();
  $<HTMLInputElement>("vm-boundary-z").value = p ? String(p[2]) : "";
  $("vm-boundary-position").textContent = p ? `Original coordinates: X ${p[0]} m, Y ${p[1]} m. Current height Z ${p[2]} m.` : "";
  const lanes = boundaryVertex ? view.lanes.filter(l => l.leftRef.id === boundaryVertex!.boundary || l.rightRef.id === boundaryVertex!.boundary) : [];
  $("vm-boundary-context").textContent = p
    ? `Boundary ${boundaryVertex!.boundary}, vertex ${boundaryVertex!.index + 1} in stored boundary order. ${lanes.length ? `Used by lanes ${lanes.map(l => l.id).join(", ")}; all update together, including reversed references.` : "No lane currently uses this boundary."}`
    : "Use Edit vertices to select a yellow handle, or choose a boundary here.";
  boundaryInputs();
}
export function renderBoundaryEditor(): void {
  const boundary = view.boundaries.find(b => b.id === boundaryVertex?.boundary);
  if (!boundary || !boundary.points[boundaryVertex!.index]) setBoundaryVertex(null);
  const select = $<HTMLSelectElement>("vm-boundary");
  select.replaceChildren(new Option("Choose a boundary or click a yellow vertex", ""), ...view.boundaries.map(b => new Option(`Boundary ${b.id}`, String(b.id))));
  select.value = boundaryVertex ? String(boundaryVertex.boundary) : "";
  const vertices = $<HTMLSelectElement>("vm-boundary-vertex");
  vertices.replaceChildren(...(boundaryVertex ? boundary!.points.map((_, i) => new Option(`Vertex ${i + 1}`, String(i))) : []));
  if (boundaryVertex) vertices.value = String(boundaryVertex.index);
  renderBoundaryVertex();
}
export function chooseBoundaryVertex(boundary: number | null, index = 0, open = false): void {
  setBoundaryVertex(boundary === null ? null : { boundary, index });
  setActiveBoundary(boundary);
  renderBoundaryEditor();
  if (open) $("vm-boundary-editor").setAttribute("open", "");
  draw();
}
$("vm-boundary").onchange = () => {
  const value = $<HTMLSelectElement>("vm-boundary").value;
  chooseBoundaryVertex(value ? Number(value) : null);
};
$("vm-boundary-vertex").onchange = () => {
  if (boundaryVertex) chooseBoundaryVertex(boundaryVertex.boundary, Number($<HTMLSelectElement>("vm-boundary-vertex").value));
};
$("vm-boundary-z").oninput = boundaryInputs;
$("vm-boundary-focus").onclick = () => {
  const p = boundaryVertexPoint(); if (busy || drag || !p) return;
  const shift = globalShift(), v = new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]);
  viewer.frameBox(new THREE.Box3(v.clone(), v.clone()).expandByScalar(3));
};
$("vm-boundary-apply").onclick = async () => {
  const p = boundaryVertexPoint(), selection = boundaryVertex;
  if (busy || drag || !p || !selection) return;
  const height = $<HTMLInputElement>("vm-boundary-z").valueAsNumber;
  if (!Number.isFinite(height) || height === p[2]) return;
  const geometry = view.boundaries.find(b => b.id === selection.boundary)!.points.map(point => [...point] as XYZ);
  geometry[selection.index][2] = height;
  await apply([{ op: "set_boundary_geometry", boundary: selection.boundary, geometry }], `Boundary ${selection.boundary} vertex ${selection.index + 1} height updated`);
};

export const vertexTool: Tool = {
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
    chooseBoundaryVertex(hit.boundary, hit.index, true);
    const ground = viewer.groundPoint(x, y, hit.point[2] - shift[2]);
    if (!ground) {
      hint.textContent = `Boundary ${hit.boundary}, vertex ${hit.index + 1}: edit height in the panel; use a top or oblique view to drag in XY.`;
      return true;
    }
    drag = {
      before: structuredClone(view), ...hit, start: [x, y], moved: false,
      offset: [hit.point[0] - shift[0] - ground.x, hit.point[1] - shift[1] - ground.y],
    };
    setActiveBoundary(hit.boundary);
    boundaryInputs();
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
    setView(finished.before);
    renderBoundaryVertex();
    draw();
    if (finished.moved) await apply([{ op: "set_boundary_geometry", boundary: finished.boundary, geometry: points }], `Boundary ${finished.boundary} vertex moved`);
    hint.textContent = "Select a yellow vertex to edit height, or drag in XY keeping Z. Shared lanes update together. Escape leaves editing.";
  },
  pointerCancel() {
    if (drag) { setView(drag.before); drag = null; renderBoundaryVertex(); draw(); }
  },
  enter() {
    editingVertices = true;
    clearJunctionPreview();
    $("vm-vertices").setAttribute("aria-pressed", "true");
    setActiveBoundary(boundaryVertex?.boundary ?? null);
    hint.textContent = "Select a yellow vertex to edit height, or drag in XY keeping Z. Shared lanes update together. Escape leaves editing.";
    draw();
  },
  exit() {
    vertexTool.pointerCancel!();
    editingVertices = false;
    setActiveBoundary(null);
    $("vm-vertices").setAttribute("aria-pressed", "false");
    renderHint();
    draw();
  },
};
$("vm-vertices").onclick = () => toggleTool(vertexTool);
