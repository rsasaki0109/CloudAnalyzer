/** Clipping box, sections and cropping. */

import { projectChanged } from "../project-change";
import * as THREE from "three";
import { cropCloud } from "../api";
import type { Vec3 } from "../protocol";
import { $, errorText, fmt, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import type { Entry } from "./state";
import { clouds, globalShift, viewer } from "./state";

const clipEnabled = $<HTMLInputElement>("clip-enabled");
/** Box the sliders span (render coordinates), captured when clipping starts. */
let clipExtent = new THREE.Box3();
const SLIDER_MAX = 1000;
const clipRows = [...document.querySelectorAll<HTMLDivElement>(".clip-axis")];
const slider = (axis: number, end: "min" | "max") =>
  clipRows[axis].querySelector<HTMLInputElement>(`input[data-end="${end}"]`)!;

/** The clipping box described by the sliders, in render coordinates. */
function clipBoxFromSliders(): THREE.Box3 {
  const min = new THREE.Vector3();
  const max = new THREE.Vector3();
  for (let axis = 0; axis < 3; axis++) {
    let lo = Number(slider(axis, "min").value) / SLIDER_MAX;
    let hi = Number(slider(axis, "max").value) / SLIDER_MAX;
    if (lo > hi) [lo, hi] = [hi, lo];
    const a = clipExtent.min.getComponent(axis);
    const size = clipExtent.max.getComponent(axis) - a;
    min.setComponent(axis, a + lo * size);
    max.setComponent(axis, a + hi * size);
  }
  return new THREE.Box3(min, max);
}

function applyClip(): void {
  if (!clipEnabled.checked) {
    viewer.setClipBox(null);
    return;
  }
  const box = clipBoxFromSliders();
  viewer.setClipBox(box);
  const shift = globalShift();
  clipRows.forEach((row, axis) => {
    const lo = box.min.getComponent(axis) + shift[axis];
    const hi = box.max.getComponent(axis) + shift[axis];
    row.querySelector(".clip-values")!.textContent = `${fmt(lo)} … ${fmt(hi)}`;
  });
}

function resetClip(): void {
  clipExtent = viewer.contentBounds();
  if (clipExtent.isEmpty()) clipExtent.set(new THREE.Vector3(), new THREE.Vector3(1, 1, 1));
  // A little margin so points on the bounds are not clipped by rounding.
  clipExtent.expandByScalar(Math.max(1e-6, clipExtent.getSize(new THREE.Vector3()).length() * 1e-6));
  for (let axis = 0; axis < 3; axis++) {
    slider(axis, "min").value = "0";
    slider(axis, "max").value = String(SLIDER_MAX);
  }
  applyClip();
}

function setClipEnabled(on: boolean): void {
  clipEnabled.checked = on;
  $("clip-controls").hidden = !on;
}

clipEnabled.onchange = () => {
  setClipEnabled(clipEnabled.checked);
  if (clipEnabled.checked) resetClip();
  else applyClip();
};
for (const input of document.querySelectorAll<HTMLInputElement>(".clip-axis input")) {
  input.oninput = applyClip;
}
$<HTMLButtonElement>("clip-reset").onclick = () => { resetClip(); projectChanged(); };

/** A thin slab across `axis` through the middle of the current box. */
for (const button of document.querySelectorAll<HTMLButtonElement>("[data-slice]")) {
  button.onclick = () => {
    const axis = Number(button.dataset.slice);
    for (let other = 0; other < 3; other++) {
      slider(other, "min").value = "0";
      slider(other, "max").value = String(SLIDER_MAX);
    }
    const half = 10; // 2% of the extent
    slider(axis, "min").value = String(SLIDER_MAX / 2 - half);
    slider(axis, "max").value = String(SLIDER_MAX / 2 + half);
    applyClip();
    // Look straight at the section.
    const direction = [0, 0, 0];
    direction[axis] = 1;
    viewer.view({ x: direction[0], y: direction[1] - (axis === 2 ? 1e-3 : 0), z: direction[2] });
    viewer.fit();
    projectChanged();
  };
}

/** The clipping box in original coordinates, or null when clipping is off. */
export function clipBox(): { min: Vec3; max: Vec3 } | null {
  if (!clipEnabled.checked) return null;
  const box = clipBoxFromSliders();
  const shift = globalShift();
  const original = (v: THREE.Vector3) => [0, 1, 2].map((a) => v.getComponent(a) + shift[a]) as Vec3;
  return { min: original(box.min), max: original(box.max) };
}

/** Turn clipping on with this box (original coordinates). */
export function setClipBox(clip: { min: Vec3; max: Vec3 }): void {
  setClipEnabled(true);
  resetClip();
  const shift = globalShift();
  for (let axis = 0; axis < 3; axis++) {
    const a = clipExtent.min.getComponent(axis);
    const size = clipExtent.max.getComponent(axis) - a || 1;
    const at = (v: number) => String(Math.round(Math.min(1, Math.max(0, (v - shift[axis] - a) / size)) * SLIDER_MAX));
    slider(axis, "min").value = at(clip.min[axis]);
    slider(axis, "max").value = at(clip.max[axis]);
  }
  applyClip();
}

$<HTMLButtonElement>("clip-crop").onclick = async () => {
  const sources = clouds().filter((e) => e.visible);
  const box = clipBox();
  if (!box || sources.length === 0) return;
  const created: string[] = [];
  const added: Entry[] = [];
  const cropped: Entry[] = [];
  for (const source of sources) {
    try {
      const cloud = await cropCloud(source.cloud.id, box.min, box.max, true);
      added.push(addEntry(cloud));
      cropped.push(source);
      created.push(`${cloud.name} (${cloud.count.toLocaleString()} points)`);
    } catch (err) {
      created.push(`${source.cloud.name}: ${errorText(err)}`);
    }
  }
  if (added.length) record({ label: "the crop", added, hide: cropped });
  setClipEnabled(false);
  applyClip();
  renderList();
  viewer.fit();
  setStatus(`Cropped: ${created.join(", ")}`);
};
