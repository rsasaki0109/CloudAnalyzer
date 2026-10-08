/** Picking points, measuring distances between them and labeling them. */

import { projectChanged } from "../project-change";
import type * as THREE from "three";
import type { Vec3 } from "../protocol";
import { distanceLabel } from "./colors";
import { $, coord, fillTable, fmt, removeButton, setStatus } from "./dom";
import { distanceChanged, entries, pointsInvalidated, toRender, viewer } from "./state";
import { idle, type PickedPoint, pickPoint, shortcut, type Tool, toggleTool } from "./tools";

interface Measurement {
  a: PickedPoint;
  b: PickedPoint;
  label: HTMLSpanElement;
}

interface Note {
  point: PickedPoint;
  text: string;
  label: HTMLSpanElement;
}

const PICK_COLOR = "#ffd54f";
const MEASURE_COLOR = "#4fc3f7";
const NOTE_COLOR = "#ff8a65";

let picked: PickedPoint | null = null;
let pending: PickedPoint | null = null;
const measurements: Measurement[] = [];
const notes: Note[] = [];
const measureButton = $<HTMLButtonElement>("measure");
const labelButton = $<HTMLButtonElement>("label");

/** Extra markers drawn by other panels (e.g. point pairs), refreshed with ours. */
const extraMarkers = new Map<string, () => { position: THREE.Vector3; color: string }[]>();
export function addMarkers(key: string, markers: () => { position: THREE.Vector3; color: string }[]): void {
  extraMarkers.set(key, markers);
}

function distance(a: PickedPoint, b: PickedPoint): { d: number; delta: number[] } {
  const delta = a.exact.map((v, i) => b.exact[i] - v);
  return { d: Math.hypot(...delta), delta };
}

export function refreshAnnotations(): void {
  const markers: { position: THREE.Vector3; color: string }[] = [];
  if (picked) markers.push({ position: picked.render, color: PICK_COLOR });
  for (const m of measurements) {
    markers.push({ position: m.a.render, color: MEASURE_COLOR }, { position: m.b.render, color: MEASURE_COLOR });
  }
  if (pending) markers.push({ position: pending.render, color: MEASURE_COLOR });
  for (const n of notes) markers.push({ position: n.point.render, color: NOTE_COLOR });
  for (const more of extraMarkers.values()) markers.push(...more());
  viewer.setAnnotations(
    markers,
    measurements.map((m) => [m.a.render, m.b.render]),
  );
}

function renderPickPanel(): void {
  $("pick-panel").hidden = !picked;
  if (!picked) return;
  const entry = entries.get(picked.cloudId);
  const rows: [string, string][] = [
    ["Cloud", entry?.cloud.name ?? "?"],
    ["X", coord(picked.exact[0])],
    ["Y", coord(picked.exact[1])],
    ["Z", coord(picked.exact[2])],
  ];
  const rgb = entry?.cloud.colors?.subarray(picked.index * 3, picked.index * 3 + 3);
  if (rgb) rows.push(["RGB", Array.from(rgb).join(", ")]);
  if (entry?.c2c) rows.push([distanceLabel(entry.c2c), fmt(entry.c2c.distances[picked.index])]);
  fillTable($("pick-info"), rows);
}
distanceChanged.add(renderPickPanel);

function setPicked(point: PickedPoint | null): void {
  picked = point;
  refreshAnnotations();
  renderPickPanel();
}

idle.click = async (x, y) => {
  const point = await pickPoint(x, y);
  if (!point) {
    if (picked) setPicked(null);
    return;
  }
  setPicked(point);
  const [px, py, pz] = point.exact.map(coord);
  setStatus(`Picked ${entries.get(point.cloudId)?.cloud.name ?? "point"}: ${px}, ${py}, ${pz}`);
};
idle.escape = () => {
  if (picked) setPicked(null);
};

// ---------------------------------------------------------------- measuring

function renderMeasurements(measuring = measureTool.active): void {
  $("measure-panel").hidden = !measuring && measurements.length === 0;
  $("measure-hint").textContent = measuring
    ? pending
      ? "Click the second point. Esc cancels."
      : "Click the first point."
    : "";
  $("measure-clear").hidden = measurements.length === 0;
  $("measure-list").replaceChildren(
    ...measurements.map((m, i) => {
      const { d, delta } = distance(m.a, m.b);
      const li = document.createElement("li");
      const value = document.createElement("span");
      value.className = "distance";
      value.textContent = fmt(d);
      const remove = removeButton(() => {
        measurements.splice(i, 1);
        m.label.remove();
        refreshAnnotations();
        renderMeasurements();
      });
      const deltas = document.createElement("span");
      deltas.className = "delta";
      deltas.textContent = `ΔX ${fmt(delta[0])}  ΔY ${fmt(delta[1])}  ΔZ ${fmt(delta[2])}`;
      li.append(remove, value, deltas);
      return li;
    }),
  );
}

const measureTool: Tool & { active: boolean } = {
  active: false,
  async click(x, y) {
    const point = await pickPoint(x, y);
    if (!point) return;
    if (!pending) {
      pending = point;
    } else {
      const label = document.createElement("span");
      $("labels").append(label);
      measurements.push({ a: pending, b: point, label });
      setStatus(`Distance: ${fmt(distance(pending, point).d)}`);
      pending = null;
    }
    renderMeasurements();
    refreshAnnotations();
  },
  enter() {
    measureTool.active = true;
    measureButton.setAttribute("aria-pressed", "true");
    renderMeasurements();
  },
  key(e) {
    if (e.key !== "Escape" || !pending) return false;
    pending = null;
    refreshAnnotations();
    renderMeasurements();
    return true;
  },
  exit() {
    measureTool.active = false;
    pending = null;
    measureButton.setAttribute("aria-pressed", "false");
    refreshAnnotations();
    renderMeasurements();
  },
};

measureButton.onclick = () => toggleTool(measureTool);
$<HTMLButtonElement>("measure-clear").onclick = () => {
  for (const m of measurements) m.label.remove();
  measurements.length = 0;
  refreshAnnotations();
  renderMeasurements();
};

// ---------------------------------------------------------------- labels

function addNote(point: PickedPoint, text: string): Note {
  const label = document.createElement("span");
  label.className = "note";
  $("labels").append(label);
  const note = { point, text, label };
  notes.push(note);
  projectChanged();
  return note;
}

function renderNotes(labeling = labelTool.active): void {
  $("note-panel").hidden = !labeling && notes.length === 0;
  $("note-hint").textContent = labeling ? "Click a point to label it; edit the text below." : "";
  $("note-list").replaceChildren(
    ...notes.map((note, i) => {
      const li = document.createElement("li");
      const input = document.createElement("input");
      input.type = "text";
      input.value = note.text;
      input.setAttribute("aria-label", `Label ${i + 1}`);
      input.oninput = () => {
        note.text = input.value;
        viewer.requestRender();
      };
      const remove = removeButton(() => {
        notes.splice(i, 1);
        note.label.remove();
        renderNotes();
        projectChanged();
      });
      li.append(input, remove);
      return li;
    }),
  );
  refreshAnnotations();
}

const labelTool: Tool & { active: boolean } = {
  active: false,
  async click(x, y) {
    const point = await pickPoint(x, y);
    if (!point) return;
    addNote(point, `Z ${coord(point.exact[2])}`);
    renderNotes();
    setStatus("Label added; edit its text in the Labels panel");
  },
  enter() {
    labelTool.active = true;
    labelButton.setAttribute("aria-pressed", "true");
    renderNotes();
  },
  exit() {
    labelTool.active = false;
    labelButton.setAttribute("aria-pressed", "false");
    renderNotes();
  },
};

labelButton.onclick = () => toggleTool(labelTool);
$<HTMLButtonElement>("note-clear").onclick = () => setNotes([]);

/** Replace the labels (sessions). Restored labels are not tied to a cloud. */
export function setNotes(saved: { position: Vec3; text: string }[]): void {
  for (const n of notes) n.label.remove();
  notes.length = 0;
  for (const l of saved) {
    const render = toRender(l.position);
    addNote({ cloudId: -1, index: -1, render, exact: l.position }, l.text);
  }
  renderNotes();
  projectChanged();
}

export function savedNotes(): { position: Vec3; text: string }[] {
  return notes.map((n) => ({ position: n.point.exact, text: n.text }));
}

shortcut("m", measureTool);
shortcut("l", labelTool);

// ---------------------------------------------------------------- cleanup and overlay

/** Drop picks, measurements and labels that refer to a removed cloud. */
pointsInvalidated.add((cloudId) => {
  if (picked?.cloudId === cloudId) picked = null;
  if (pending?.cloudId === cloudId) pending = null;
  for (let i = measurements.length - 1; i >= 0; i--) {
    const m = measurements[i];
    if (m.a.cloudId === cloudId || m.b.cloudId === cloudId) {
      m.label.remove();
      measurements.splice(i, 1);
    }
  }
  for (let i = notes.length - 1; i >= 0; i--) {
    if (notes[i].point.cloudId === cloudId) {
      notes[i].label.remove();
      notes.splice(i, 1);
    }
  }
  renderNotes();
  renderPickPanel();
  renderMeasurements();
});

// Keep labels on their points and distance labels at the middle of their segments.
viewer.onAfterRender = () => {
  for (const n of notes) {
    const at = viewer.project(n.point.render);
    n.label.hidden = !at || !n.text;
    if (!at) continue;
    n.label.textContent = n.text;
    n.label.style.left = `${at.x}px`;
    n.label.style.top = `${at.y}px`;
  }
  for (const m of measurements) {
    const mid = m.a.render.clone().add(m.b.render).multiplyScalar(0.5);
    const at = viewer.project(mid);
    m.label.hidden = !at;
    if (!at) continue;
    m.label.textContent = fmt(distance(m.a, m.b).d);
    m.label.style.left = `${at.x}px`;
    m.label.style.top = `${at.y}px`;
  }
};
