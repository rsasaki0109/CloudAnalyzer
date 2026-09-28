/**
 * Trajectories panel: TUM / KITTI / CSV poses drawn as lines, and ATE / RPE
 * of an estimate against a reference (same definitions as the Python CLI).
 */

import { evaluateTrajectory, transformCloud } from "../api";
import { colorize, gradientCss, lut } from "../colormap";
import type { TrajectoryAlignment, TrajectoryEvaluation, TrajectoryPoses, Vec3 } from "../protocol";
import { $, download, errorText, fillTable, fmt, removeButton, setStatus } from "./dom";
import { replaceCloud } from "./entries";
import { record } from "./history";
import { addReportSection, type Metric } from "./report";
import { clouds, display, distanceChanged, entries, globalShift, listChanged, viewer } from "./state";

interface Trajectory {
  id: number;
  name: string;
  poses: TrajectoryPoses;
  color: string;
  visible: boolean;
  /** An evaluated estimate: ATE per pose, drawn in the color ramp. */
  errors?: Float64Array;
}

const COLORS = ["#ff7043", "#4fc3f7", "#ffd54f", "#81c784", "#ba68c8", "#f06292"];
const ALIGNMENT_NAMES: Record<TrajectoryAlignment, string> = {
  none: "None",
  origin: "Origin",
  se3: "SE(3)",
  sim3: "Sim(3)",
};

const trajectories = new Map<number, Trajectory>();
let nextId = 1;
/** The latest evaluation, for the CSV and applying its alignment; `id` is its result line. */
let last: { result: TrajectoryEvaluation; alignment: TrajectoryAlignment; name: string; id: number } | null = null;
/** Names of the last evaluation's inputs, for the report. */
let lastPair = { estimate: "", reference: "" };

addReportSection("trajectory", () => {
  if (!last) return null;
  const { stats } = last.result;
  const metrics: Record<string, Metric> = {
    matched: { label: "Matched poses", value: stats.ate.count },
    ate_rmse: { label: "ATE RMSE", value: stats.ate.rmse, unit: "m" },
    ate_mean: { label: "ATE mean", value: stats.ate.mean, unit: "m" },
    ate_median: { label: "ATE median", value: stats.ate.median, unit: "m" },
    ate_max: { label: "ATE max", value: stats.ate.max, unit: "m" },
    drift: { label: "Endpoint drift", value: last.result.endpointDrift, unit: "m" },
  };
  if (stats.ateRotation) metrics.ate_rot_rmse = { label: "ATE rotation RMSE", value: stats.ateRotation.rmse, unit: "°" };
  if (stats.rpe) metrics.rpe_rmse = { label: "RPE RMSE", value: stats.rpe.rmse, unit: "m" };
  if (stats.rpeRotation) metrics.rpe_rot_rmse = { label: "RPE rotation RMSE", value: stats.rpeRotation.rmse, unit: "°" };
  if (stats.rpePercent) metrics.rpe_percent = { label: "RPE", value: stats.rpePercent.rmse, unit: "%" };
  return {
    title: `Trajectory ${lastPair.estimate} vs ${lastPair.reference}`,
    metrics,
    notes: [`Alignment: ${last.alignment}${last.alignment === "sim3" ? ` (scale ${fmt(last.result.scale)})` : ""}`],
  };
});

const estimateSelect = $<HTMLSelectElement>("trajectory-estimate");
const referenceSelect = $<HTMLSelectElement>("trajectory-reference");
const runButton = $<HTMLButtonElement>("trajectory-run");
const applySelect = $<HTMLSelectElement>("trajectory-apply-cloud");
const applyButton = $<HTMLButtonElement>("trajectory-apply");

const baseName = (name: string) => name.replace(/\.[^.]+$/, "");

/**
 * Render offset: the clouds' global shift, or without clouds one that keeps
 * georeferenced trajectories precise in float32.
 */
function shift(): Vec3 {
  if (entries.size) return globalShift();
  const p = [...trajectories.values()][0]?.poses.positions;
  if (!p || Math.max(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2])) <= 1e4) return [0, 0, 0];
  return [0, 1, 2].map((a) => Math.round(p[a] / 100) * 100) as Vec3;
}
let drawnShift = "";

function draw(t: Trajectory): void {
  const [sx, sy, sz] = shift();
  const p = t.poses.positions;
  const positions = new Float32Array(p.length);
  for (let i = 0; i < p.length; i += 3) {
    positions[i] = p[i] - sx;
    positions[i + 1] = p[i + 1] - sy;
    positions[i + 2] = p[i + 2] - sz;
  }
  const colors = t.errors ? colorize(Float32Array.from(t.errors), ...errorRange(t.errors), lut(display.ramp)) : undefined;
  viewer.setLine(t.id, positions, t.color, colors);
  viewer.setLineVisible(t.id, t.visible);
}

function errorRange(errors: Float64Array): [number, number] {
  let [lo, hi] = [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (const e of errors) [lo, hi] = [Math.min(lo, e), Math.max(hi, e)];
  return [lo, hi];
}

function drawAll(): void {
  drawnShift = shift().join();
  for (const t of trajectories.values()) draw(t);
}

/** Show a loaded trajectory in the list and the view. */
export function addTrajectory(name: string, poses: TrajectoryPoses, errors?: Float64Array): Trajectory {
  const color = COLORS[trajectories.size % COLORS.length];
  const t: Trajectory = { id: nextId++, name, poses, color, visible: true, errors };
  trajectories.set(t.id, t);
  // The first trajectory can change the shift used without clouds.
  if (shift().join() !== drawnShift) drawAll();
  else draw(t);
  if (entries.size === 0 && trajectories.size === 1) viewer.fit();
  renderList();
  return t;
}

function removeTrajectory(id: number): void {
  trajectories.delete(id);
  viewer.removeLine(id);
  if (last?.id === id) {
    last = null;
    $("trajectory-result").hidden = true;
  }
  if (shift().join() !== drawnShift) drawAll();
  renderList();
}

function renderList(): void {
  $("trajectory-hint").hidden = trajectories.size > 0;
  $("trajectory-list").replaceChildren(
    ...[...trajectories.values()].map((t) => {
      const li = document.createElement("li");
      const visible = document.createElement("input");
      visible.type = "checkbox";
      visible.checked = t.visible;
      visible.title = "Show / hide";
      visible.onchange = () => {
        t.visible = visible.checked;
        viewer.setLineVisible(t.id, t.visible);
      };
      const color = document.createElement("input");
      color.type = "color";
      color.value = t.color;
      color.title = t.errors ? "Colored by ATE" : "Line color";
      color.disabled = !!t.errors;
      color.oninput = () => {
        t.color = color.value;
        draw(t);
      };
      const name = document.createElement("span");
      name.className = "name";
      name.title = t.name;
      name.textContent = t.name;
      const meta = document.createElement("span");
      meta.className = "meta";
      const n = t.poses.timestamps.length;
      meta.textContent = t.errors
        ? `${n.toLocaleString()} matched poses · colored by ATE`
        : `${n.toLocaleString()} poses · ${t.poses.format.toUpperCase()}${t.poses.orientations ? "" : " · no orientation"}`;
      name.append(meta);
      li.append(visible, color, name, removeButton(() => removeTrajectory(t.id)));
      return li;
    }),
  );
  renderSelects();
}

function renderSelects(): void {
  // Evaluation results are not offered as inputs.
  const ids = [...trajectories.values()].filter((t) => !t.errors).map((t) => String(t.id));
  let [estimate, reference] = [estimateSelect.value, referenceSelect.value];
  if (!(ids.includes(estimate) && ids.includes(reference) && estimate !== reference)) {
    // Default: the newest trajectory against the first one.
    [estimate, reference] = [ids.at(-1) ?? "", ids[0] ?? ""];
  }
  for (const [select, value] of [
    [estimateSelect, estimate],
    [referenceSelect, reference],
  ] as const) {
    select.replaceChildren(...ids.map((id) => new Option(trajectories.get(Number(id))!.name, id)));
    select.value = value;
  }
  updateButtons();
}

function updateButtons(): void {
  runButton.disabled = !estimateSelect.value || !referenceSelect.value || estimateSelect.value === referenceSelect.value;
  // Undo inverts rigid transforms only, and "none" has nothing to apply.
  const rigid = last?.alignment === "origin" || last?.alignment === "se3";
  applyButton.disabled = !rigid || !applySelect.value;
  applyButton.title = rigid
    ? "Move the cloud (e.g. the map built from the estimate) with the alignment"
    : "Only origin and SE(3) alignments can be applied to a cloud";
}
estimateSelect.onchange = referenceSelect.onchange = updateButtons;

listChanged.add(() => {
  const ids = clouds().map((e) => String(e.cloud.id));
  const current = ids.includes(applySelect.value) ? applySelect.value : (ids.at(-1) ?? "");
  applySelect.replaceChildren(...ids.map((id) => new Option(entries.get(Number(id))!.cloud.name, id)));
  applySelect.value = current;
  updateButtons();
  // Lines follow the clouds' global shift.
  if (shift().join() !== drawnShift) drawAll();
});

// The color ramp may have changed.
distanceChanged.add(() => {
  for (const t of trajectories.values()) if (t.errors) draw(t);
  if (last) renderLegend(last.result);
});

function renderLegend(result: TrajectoryEvaluation): void {
  const [lo, hi] = errorRange(result.ate);
  $("trajectory-legend-bar").style.background = gradientCss(display.ramp, "to right");
  $("trajectory-legend-min").textContent = fmt(lo);
  $("trajectory-legend-max").textContent = fmt(hi);
}

function renderResult(result: TrajectoryEvaluation, alignment: TrajectoryAlignment, referencePoses: number): void {
  const { stats } = result;
  const deg = (v: number) => `${fmt(v)}°`;
  const matched = stats.ate.count;
  const n = Number($<HTMLInputElement>("trajectory-delta").value);
  const metres = $<HTMLSelectElement>("trajectory-delta-unit").value === "m";
  const delta = metres ? `${n} m` : `${n} frame${n === 1 ? "" : "s"}`;
  const rows: [string, string][] = [
    ["Matched poses", `${matched.toLocaleString()} of ${referencePoses.toLocaleString()}`],
    ["Alignment", alignment === "sim3" ? `Sim(3), scale ${fmt(result.scale)}` : ALIGNMENT_NAMES[alignment]],
    ["ATE RMSE", fmt(stats.ate.rmse)],
    ["ATE mean", fmt(stats.ate.mean)],
    ["ATE median", fmt(stats.ate.median)],
    ["ATE std. dev.", fmt(stats.ate.std)],
    ["ATE min", fmt(stats.ate.min)],
    ["ATE max", fmt(stats.ate.max)],
  ];
  if (stats.ateRotation) rows.push(["ATE rotation RMSE", deg(stats.ateRotation.rmse)]);
  if (stats.rpe) {
    rows.push(
      ["RPE delta", `${delta} (${stats.rpe.count.toLocaleString()} pairs)`],
      ["RPE RMSE", fmt(stats.rpe.rmse)],
      ["RPE mean", fmt(stats.rpe.mean)],
      ["RPE max", fmt(stats.rpe.max)],
    );
    if (stats.rpePercent) rows.push(["RPE RMSE %", `${fmt(stats.rpePercent.rmse)} %`]);
    if (stats.rpeRotation) {
      rows.push(["RPE rotation RMSE", deg(stats.rpeRotation.rmse)], ["RPE rotation max", deg(stats.rpeRotation.max)]);
    }
  } else {
    rows.push(["RPE", `no poses ${delta} apart`]);
  }
  rows.push(
    ["Endpoint drift", fmt(result.endpointDrift)],
    ["Path length", fmt(result.estimateLength)],
    ["Reference path length", fmt(result.referenceLength)],
  );
  fillTable($("trajectory-stats"), rows);
  renderLegend(result);
  $("trajectory-result").hidden = false;
}

runButton.onclick = async () => {
  const estimate = trajectories.get(Number(estimateSelect.value));
  const reference = trajectories.get(Number(referenceSelect.value));
  if (!estimate || !reference) return;
  const alignment = $<HTMLSelectElement>("trajectory-align").value as TrajectoryAlignment;
  runButton.disabled = true;
  setStatus(`Evaluating ${estimate.name} against ${reference.name}…`);
  try {
    const result = await evaluateTrajectory({
      estimate: estimate.poses,
      reference: reference.poses,
      maxTimeDelta: Number($<HTMLInputElement>("trajectory-dt").value),
      alignment,
      delta: Number($<HTMLInputElement>("trajectory-delta").value),
      deltaUnit: $<HTMLSelectElement>("trajectory-delta-unit").value as "frames" | "m",
    });
    // Show the matched, aligned estimate colored by its error, in place of
    // the estimate and of any earlier result.
    for (const t of [...trajectories.values()]) if (t.errors) removeTrajectory(t.id);
    const name = `${baseName(estimate.name)}_ate`;
    const poses: TrajectoryPoses = { ...estimate.poses, timestamps: result.timestamps, positions: result.estimate, orientations: null };
    const shown = addTrajectory(name, poses, result.ate);
    estimate.visible = false;
    viewer.setLineVisible(estimate.id, false);
    last = { result, alignment, name, id: shown.id };
    lastPair = { estimate: estimate.name, reference: reference.name };
    renderList();
    renderResult(result, alignment, reference.poses.timestamps.length);
    setStatus(
      `ATE RMSE ${fmt(result.stats.ate.rmse)} over ${result.stats.ate.count.toLocaleString()} matched poses` +
        (result.stats.rpe ? `, RPE RMSE ${fmt(result.stats.rpe.rmse)}` : ""),
    );
  } catch (err) {
    setStatus(`Trajectory evaluation failed: ${errorText(err)}`, true);
  } finally {
    updateButtons();
  }
};

/** Per matched pose: time, aligned estimate, reference and errors. */
$<HTMLButtonElement>("trajectory-csv").onclick = () => {
  if (!last) return;
  const { timestamps, estimate, reference, ate, ateRotation } = last.result;
  const lines = [
    `timestamp,x,y,z,reference_x,reference_y,reference_z,ate${ateRotation ? ",ate_rotation_deg" : ""}`,
  ];
  for (let i = 0; i < timestamps.length; i++) {
    const values = [timestamps[i], ...estimate.subarray(i * 3, i * 3 + 3), ...reference.subarray(i * 3, i * 3 + 3), ate[i]];
    if (ateRotation) values.push(ateRotation[i]);
    lines.push(values.join(","));
  }
  const filename = `${last.name}.csv`;
  download(new Blob([`${lines.join("\n")}\n`], { type: "text/csv" }), filename);
  setStatus(`Saved ${filename} (${timestamps.length.toLocaleString()} poses)`);
};

applyButton.onclick = async () => {
  const entry = entries.get(Number(applySelect.value));
  if (!last || !entry) return;
  const matrix = last.result.matrix;
  applyButton.disabled = true;
  try {
    entry.transforms.push(matrix);
    replaceCloud(entry, await transformCloud(entry.cloud.id, matrix));
    record({ label: "the trajectory alignment", moved: { entry, matrix } });
    setStatus(`Moved ${entry.cloud.name} with the ${ALIGNMENT_NAMES[last.alignment]} alignment`);
  } catch (err) {
    entry.transforms.pop();
    setStatus(`Could not move the cloud: ${errorText(err)}`, true);
  } finally {
    updateButtons();
  }
};

renderList();
