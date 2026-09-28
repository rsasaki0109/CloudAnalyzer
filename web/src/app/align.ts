/** Manual alignment: picked point pairs, a move / rotate gizmo, or a typed matrix. */

import * as THREE from "three";
import { alignPairs } from "../api";
import type { Vec3 } from "../protocol";
import { $, errorText, fmt, removeButton, setStatus } from "./dom";
import { moveCloud } from "./history";
import { addReportSection } from "./report";
import { addMarkers, refreshAnnotations } from "./picking";
import { clouds, entries, listChanged, pointsInvalidated, viewer } from "./state";
import { activeTool, type PickedPoint, pickPoint, setTool, shortcut, type Tool, toggleTool } from "./tools";

const MOVING_COLOR = "#ffb000";
const REFERENCE_COLOR = "#4fc3f7";

const movingSelect = $<HTMLSelectElement>("align-moving");
const referenceSelect = $<HTMLSelectElement>("align-reference");
const pickButton = $<HTMLButtonElement>("align-pick");
const runButton = $<HTMLButtonElement>("align-run");
const clearButton = $<HTMLButtonElement>("align-clear");

/** Picked pairs; the last one may still wait for its reference point. */
let pairs: { moving: PickedPoint; reference?: PickedPoint }[] = [];
/** Residuals of the last alignment, shown next to the pairs it used. */
let residuals: number[] = [];
/** While our own alignment replaces the cloud, its picks are moved rather than dropped. */
let aligning = false;
/** The last point-pair alignment, for the report. */
let lastAlignment: { name: string; reference: string; rms: number; residuals: number[] } | null = null;

addReportSection("pairs", () =>
  lastAlignment
    ? {
        title: `Point-pair alignment ${lastAlignment.name} → ${lastAlignment.reference}`,
        metrics: {
          pairs: { label: "Pairs", value: lastAlignment.residuals.length },
          rms: { label: "RMS residual", value: lastAlignment.rms },
          max: { label: "Largest residual", value: Math.max(...lastAlignment.residuals) },
        },
      }
    : null,
);

const moving = () => entries.get(Number(movingSelect.value));
const reference = () => entries.get(Number(referenceSelect.value));
const complete = () => pairs.filter((p) => p.reference) as { moving: PickedPoint; reference: PickedPoint }[];

listChanged.add(() => {
  const options = clouds();
  const ids = options.map((e) => String(e.cloud.id));
  let [m, r] = [movingSelect.value, referenceSelect.value];
  if (!(ids.includes(m) && ids.includes(r) && m !== r)) {
    // Default: move the newest cloud onto the first one, as ICP does.
    [m, r] = [ids.at(-1) ?? "", ids[0] ?? ""];
  }
  for (const [select, value] of [
    [movingSelect, m],
    [referenceSelect, r],
  ] as const) {
    select.replaceChildren(...options.map((e) => new Option(e.cloud.name, String(e.cloud.id))));
    select.value = value;
  }
  render();
});

function render(): void {
  const picking = activeTool() === pairTool;
  const done = complete().length;
  runButton.disabled = done < 3 || picking || movingSelect.value === referenceSelect.value;
  clearButton.disabled = pairs.length === 0;
  pickButton.setAttribute("aria-pressed", String(picking));
  const waiting = pairs.at(-1) && !pairs.at(-1)!.reference;
  $("align-hint").textContent = picking
    ? waiting
      ? `Now the same spot on ${reference()?.cloud.name ?? "the reference"}.`
      : `Click a point on ${moving()?.cloud.name ?? "the moved cloud"}${done >= 3 ? ", or Align" : ""}.`
    : done < 3
      ? "Pick at least three pairs of matching points, spread out and not in a line."
      : "";
  $("align-pairs").replaceChildren(
    ...pairs.map((pair, i) => {
      const li = document.createElement("li");
      const label = document.createElement("span");
      label.className = "distance";
      label.textContent = `Pair ${i + 1}`;
      const detail = document.createElement("span");
      detail.className = "delta";
      detail.textContent = !pair.reference
        ? "waiting for the reference point"
        : residuals[i] !== undefined
          ? `residual ${fmt(residuals[i])}`
          : `${fmt(pair.moving.exact.reduce((d, v, a) => d + (v - pair.reference!.exact[a]) ** 2, 0) ** 0.5)} apart`;
      li.append(
        removeButton(() => {
          pairs.splice(i, 1);
          residuals = [];
          render();
        }),
        label,
        detail,
      );
      return li;
    }),
  );
  refreshAnnotations();
}

addMarkers("align", () =>
  pairs.flatMap((p) => [
    { position: p.moving.render, color: MOVING_COLOR },
    ...(p.reference ? [{ position: p.reference.render, color: REFERENCE_COLOR }] : []),
  ]),
);

const pairTool: Tool = {
  async click(x, y) {
    const point = await pickPoint(x, y);
    if (!point) return;
    const last = pairs.at(-1);
    const wantReference = last && !last.reference;
    const want = wantReference ? reference() : moving();
    if (point.cloudId !== want?.cloud.id) {
      setStatus(`Pick on ${want?.cloud.name ?? "the selected cloud"} (hide the other cloud if it is in the way)`, true);
      return;
    }
    residuals = [];
    if (wantReference) last.reference = point;
    else pairs.push({ moving: point });
    render();
  },
  key(e) {
    if (e.key === "Backspace" && pairs.length) {
      const last = pairs.at(-1)!;
      if (last.reference) last.reference = undefined;
      else pairs.pop();
      render();
      return true;
    }
    return false;
  },
  enter: render,
  exit: render,
};

pickButton.onclick = () => toggleTool(pairTool);
shortcut("p", pairTool);
clearButton.onclick = () => {
  pairs = [];
  residuals = [];
  render();
};
movingSelect.onchange = referenceSelect.onchange = () => {
  pairs = [];
  residuals = [];
  render();
};

runButton.onclick = async () => {
  const entry = moving();
  const used = complete();
  if (!entry || used.length < 3) return;
  runButton.disabled = true;
  try {
    const out = await alignPairs(
      used.flatMap((p) => p.moving.exact),
      used.flatMap((p) => p.reference.exact),
    );
    aligning = true;
    await moveCloud(entry, out.matrix, "the point-pair alignment");
    $<HTMLTextAreaElement>("align-matrix").value = matrixText(out.matrix);
    // Show the moved picks where they are now, with their residuals.
    const m = out.matrix;
    const [sx, sy, sz] = entry.cloud.shift;
    pairs = used.map(({ moving, reference }) => {
      const [x, y, z] = moving.exact;
      const exact = [0, 1, 2].map((r) => m[r * 4] * x + m[r * 4 + 1] * y + m[r * 4 + 2] * z + m[r * 4 + 3]) as Vec3;
      const render = new THREE.Vector3(exact[0] - sx, exact[1] - sy, exact[2] - sz);
      return { moving: { ...moving, index: -1, exact, render }, reference };
    });
    residuals = out.residuals;
    lastAlignment = {
      name: entry.cloud.name,
      reference: reference()?.cloud.name ?? "reference",
      rms: out.rms,
      residuals: out.residuals,
    };
    setStatus(
      `Aligned ${entry.cloud.name} with ${used.length} pairs: RMS ${fmt(out.rms)}, ` +
        `largest residual ${fmt(Math.max(...out.residuals))}`,
    );
  } catch (err) {
    setStatus(`Alignment failed: ${errorText(err)}`, true);
  } finally {
    aligning = false;
    render();
  }
};

// A cloud that is moved or removed elsewhere (undo, ICP…) invalidates its picks.
pointsInvalidated.add((cloudId) => {
  if (aligning) return;
  const before = pairs.length;
  pairs = pairs.filter((p) => p.moving.cloudId !== cloudId && p.reference?.cloudId !== cloudId);
  if (pairs.length !== before) {
    residuals = [];
    render();
  }
});

/** Whether a row-major 4x4 matrix is a rotation plus translation (undo inverts it as one). */
function isRigid(m: number[]): boolean {
  const r = (i: number, j: number) => m[i * 4 + j];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const dot = r(0, i) * r(0, j) + r(1, i) * r(1, j) + r(2, i) * r(2, j);
      if (Math.abs(dot - (i === j ? 1 : 0)) > 1e-6) return false;
    }
  }
  const det = new THREE.Matrix3().set(r(0, 0), r(0, 1), r(0, 2), r(1, 0), r(1, 1), r(1, 2), r(2, 0), r(2, 1), r(2, 2)).determinant();
  return det > 0 && m[12] === 0 && m[13] === 0 && m[14] === 0 && m[15] === 1;
}

function matrixText(matrix: number[]): string {
  return [0, 1, 2, 3].map((r) => matrix.slice(r * 4, r * 4 + 4).map((v) => v.toFixed(9)).join(" ")).join("\n");
}

// ---------------------------------------------------------------- gizmo

const gizmoButtons = { translate: $<HTMLButtonElement>("gizmo-translate"), rotate: $<HTMLButtonElement>("gizmo-rotate") };
const gizmoApply = $<HTMLButtonElement>("gizmo-apply");
const gizmoCancel = $<HTMLButtonElement>("gizmo-cancel");
let gizmoCloud: number | null = null;

function renderGizmo(mode: "translate" | "rotate" | null): void {
  for (const [m, button] of Object.entries(gizmoButtons)) button.setAttribute("aria-pressed", String(m === mode));
  gizmoApply.disabled = gizmoCancel.disabled = mode === null;
}

function stopGizmo(): void {
  gizmoCloud = null;
  viewer.endGizmo();
  renderGizmo(null);
}

for (const [mode, button] of Object.entries(gizmoButtons) as ["translate" | "rotate", HTMLButtonElement][]) {
  button.onclick = () => {
    const entry = moving();
    if (!entry) return;
    if (gizmoCloud === entry.cloud.id) {
      viewer.setGizmoMode(mode);
    } else {
      const b = entry.cloud.bounds;
      const shift = entry.cloud.shift;
      const center = new THREE.Vector3(
        (b[0] + b[3]) / 2 - shift[0],
        (b[1] + b[4]) / 2 - shift[1],
        (b[2] + b[5]) / 2 - shift[2],
      );
      if (activeTool()) setTool(null);
      viewer.startGizmo(entry.cloud.id, center, mode);
      gizmoCloud = entry.cloud.id;
    }
    renderGizmo(mode);
  };
}

gizmoCancel.onclick = stopGizmo;
gizmoApply.onclick = async () => {
  const entry = gizmoCloud !== null ? entries.get(gizmoCloud) : undefined;
  const motion = viewer.gizmoMatrix();
  if (!entry || !motion) return stopGizmo();
  // The gizmo moved render coordinates; express the motion in original ones.
  const [sx, sy, sz] = entry.cloud.shift;
  const original = new THREE.Matrix4()
    .makeTranslation(sx, sy, sz)
    .multiply(motion)
    .multiply(new THREE.Matrix4().makeTranslation(-sx, -sy, -sz));
  const matrix = original.transpose().toArray();
  gizmoApply.disabled = true;
  try {
    // Replacing the cloud ends the gizmo and draws it at its new place.
    await moveCloud(entry, matrix, "the manual move");
    $<HTMLTextAreaElement>("align-matrix").value = matrixText(matrix);
    setStatus(`Moved ${entry.cloud.name}`);
  } catch (err) {
    setStatus(`Move failed: ${errorText(err)}`, true);
  } finally {
    stopGizmo();
  }
};

$<HTMLButtonElement>("align-apply-matrix").onclick = async () => {
  const entry = moving();
  const values = $<HTMLTextAreaElement>("align-matrix")
    .value.split(/[\s,;]+/)
    .filter(Boolean)
    .map(Number);
  if (!entry) return;
  if (values.length !== 16 || !values.every(Number.isFinite)) {
    setStatus("Enter 16 numbers: a row-major 4×4 matrix", true);
    return;
  }
  if (!isRigid(values)) {
    setStatus("Only rigid transforms (rotation and translation, bottom row 0 0 0 1) can be applied", true);
    return;
  }
  try {
    await moveCloud(entry, values, "the matrix");
    setStatus(`Applied the matrix to ${entry.cloud.name}`);
  } catch (err) {
    setStatus(`Transform failed: ${errorText(err)}`, true);
  }
};

render();
