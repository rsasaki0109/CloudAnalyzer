/** Shapes & clusters panel: RANSAC planes / cylinders / spheres and Euclidean clusters. */

import { findShapes } from "../api";
import type { Segment, SegmentMethod } from "../protocol";
import { refreshColors } from "./colors";
import { $, errorText, fmt, roundUp, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { addReportSection } from "./report";
import { fillCloudSelect } from "./processing";
import { clouds, entries, listChanged } from "./state";

let lastShapes: { title: string; found: number; assigned: number; worstRms: number | null } | null = null;

addReportSection("shapes", () =>
  lastShapes
    ? {
        title: lastShapes.title,
        metrics: {
          found: { label: "Found", value: lastShapes.found },
          assigned: { label: "Points assigned", value: lastShapes.assigned, unit: "%" },
          ...(lastShapes.worstRms !== null ? { worst_rms: { label: "Worst shape RMS", value: lastShapes.worstRms } } : {}),
        },
      }
    : null,
);

const segmentCloudSelect = $<HTMLSelectElement>("shapes-cloud");
const segmentMethod = $<HTMLSelectElement>("shapes-method");
const segmentDistance = $<HTMLInputElement>("shapes-distance");
const segmentMin = $<HTMLInputElement>("shapes-min");
const segmentRun = $<HTMLButtonElement>("shapes-run");

listChanged.add(() => {
  const options = clouds();
  fillCloudSelect(segmentCloudSelect, options);
  segmentRun.disabled = options.length === 0;
  suggestParameters();
});

/**
 * Defaults from the cloud's size: shapes within 1/200 of its largest extent
 * with 0.5 % of its points; clusters linked at three times the point spacing
 * (as if the points covered the two largest extents).
 */
function suggestParameters(): void {
  const entry = entries.get(Number(segmentCloudSelect.value));
  const key = `${segmentCloudSelect.value}:${segmentMethod.value}`;
  if (!entry || segmentDistance.dataset.for === key) return;
  const b = entry.cloud.bounds;
  const extents = [b[3] - b[0], b[4] - b[1], b[5] - b[2]].sort((x, y) => y - x);
  const n = entry.cloud.count;
  if (segmentMethod.value === "cluster") {
    const spacing = Math.sqrt((extents[0] * extents[1] || extents[0] ** 2) / n);
    segmentDistance.value = String(roundUp(3 * spacing));
    segmentMin.value = String(Math.max(10, Math.round(n / 10_000)));
  } else {
    segmentDistance.value = String(roundUp(extents[0] / 200));
    segmentMin.value = String(Math.max(50, Math.round(n / 200)));
  }
  segmentDistance.dataset.for = key;
}

segmentCloudSelect.onchange = suggestParameters;
segmentMethod.onchange = () => {
  const cluster = segmentMethod.value === "cluster";
  $("shapes-distance-label").firstChild!.textContent = cluster ? "Link distance " : "Max. distance ";
  $("shapes-max-row").hidden = cluster;
  suggestParameters();
};

// Rounded, without "-0.000".
const vec = (v: number[]) => `(${v.map((x) => x.toFixed(3).replace(/^-(0\.0+)$/, "$1")).join(", ")})`;

/** A result table row: what the segment is, and its parameters. */
function describeSegment(s: Segment): string {
  // Round-off (e.g. 1e-17 for d of a plane through the origin) reads as 0.
  const p = s.params.map((x) => (Math.abs(x) < 1e-9 ? 0 : x));
  const rms = Number.isNaN(s.rms) ? "" : ` · RMS ${fmt(s.rms)}`;
  switch (s.kind) {
    case "plane":
      return `normal ${vec(p.slice(0, 3))}, d ${fmt(p[3])}${rms}`;
    case "sphere":
      return `centre ${vec(p.slice(0, 3))}, radius ${fmt(p[3])}${rms}`;
    case "cylinder":
      return `radius ${fmt(p[6])}, length ${fmt(p[7])}, axis ${vec(p.slice(3, 6))} through ${vec(p.slice(0, 3))}${rms}`;
    case "cluster":
    case "other clusters":
      return `centroid ${vec(p.slice(0, 3))}, size ${p.slice(3, 6).map(fmt).join(" × ")}`;
    case "rest":
      return "";
  }
}

function showTable(names: string[], segments: Segment[]): void {
  $("shapes-result").hidden = segments.length === 0;
  $("shapes-stats").replaceChildren(
    ...segments.map((s, i) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = `rgb(${s.color.join(" ")})`;
      th.append(swatch, `${names[i]} (${s.count.toLocaleString()})`);
      const td = document.createElement("td");
      td.textContent = describeSegment(s);
      tr.append(th, td);
      return tr;
    }),
  );
}

segmentRun.onclick = async () => {
  const entry = entries.get(Number(segmentCloudSelect.value));
  if (!entry) return;
  const method = segmentMethod.value as SegmentMethod;
  const distance = Number(segmentDistance.value);
  const minPoints = Math.round(Number(segmentMin.value));
  const maxShapes = Math.round(Number($<HTMLInputElement>("shapes-max").value));
  if (!(distance > 0) || !(minPoints > 0) || (method !== "cluster" && !(maxShapes > 0))) {
    setStatus("Enter a positive distance, point count and number of shapes", true);
    return;
  }
  const split = $<HTMLSelectElement>("shapes-output").value === "split";
  segmentRun.disabled = true;
  const what = method === "cluster" ? "clusters" : `${method}s`;
  setStatus(`Looking for ${what} in ${entry.cloud.name}…`);
  try {
    const out = await findShapes({ id: entry.cloud.id, method, distance, minPoints, maxShapes, split });
    if (out.normals) entry.cloud.normals = out.normals;
    const kind = method === "cluster" ? "cluster" : method;
    const names = out.segments.map((s, i) =>
      s.kind === "rest"
        ? method === "cluster"
          ? "noise"
          : "rest"
        : s.kind === "other clusters"
          ? `clusters ${i + 1}+`
          : `${kind} ${i + 1}`,
    );
    showTable(names, out.segments);
    const time = `${Math.round(out.millis)} ms`;
    if (out.found === 0) {
      setStatus(`No ${what} found in ${entry.cloud.name} (${time}); try a larger distance or fewer points`, true);
      return;
    }
    const added = out.clouds.map((cloud, i) => {
      const part = addEntry(cloud);
      if (split) {
        // Keep the parts' own colors available, but show the segment colors.
        part.solid = out.segments[i].color;
        part.mode = "solid";
        refreshColors(part);
      }
      return part;
    });
    record({ label: `the ${what} search`, added, hide: [entry] });
    const shapes = out.segments.filter((s) => Number.isFinite(s.rms));
    lastShapes = {
      title: `${what[0].toUpperCase()}${what.slice(1)} in ${entry.cloud.name}`,
      found: out.found,
      assigned: (100 * (entry.cloud.count - (out.segments.at(-1)?.kind === "rest" ? out.segments.at(-1)!.count : 0))) / entry.cloud.count,
      worstRms: shapes.length ? Math.max(...shapes.map((s) => s.rms)) : null,
    };
    renderList();
    const rest = out.segments.at(-1)?.kind === "rest" ? out.segments.at(-1)!.count : 0;
    const assigned = entry.cloud.count - rest;
    setStatus(
      `${out.found.toLocaleString()} ${out.found === 1 ? kind : what} in ${entry.cloud.name}: ` +
        `${assigned.toLocaleString()} of ${entry.cloud.count.toLocaleString()} points, ` +
        `${rest.toLocaleString()} ${method === "cluster" ? "noise" : "left"} (${time})`,
    );
  } catch (err) {
    setStatus(`Shapes & clusters failed: ${errorText(err)}`, true);
  } finally {
    segmentRun.disabled = clouds().length === 0;
  }
};
