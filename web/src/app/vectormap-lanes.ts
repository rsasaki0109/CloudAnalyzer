/** Lane selection, the lane panel and click tools that pick lanes. */

import * as THREE from "three";
import { vectorMap } from "../api";
import { $, fmt, setStatus } from "./dom";
import { globalShift, viewer } from "./state";
import { pickPoint, toggleTool, type Tool } from "./tools";
import {
  apply,
  draw,
  filteredReviews,
  hint,
  LANE_REACH,
  laneBox,
  laneById,
  laneLength,
  markReviewSelection,
  type Nearest,
  type Neighbor,
  renderHint,
  renderReviews,
  REVIEW_PAGE_SIZE,
  reviewPage,
  reviews,
  selected,
  setReviewPage,
  setSelected,
  setSourceProblemSelected,
  sketch,
  sourceProblemInputs,
  sourceProblems,
  sourceProblemSelected,
  view,
  type XYZ,
} from "./vectormap";

export function selectLane(id: number | null, frame = false): void {
  setSelected(id);
  if (sourceProblemSelected !== null && sourceProblems[sourceProblemSelected]?.lane !== id) {
    setSourceProblemSelected(null);
    $<HTMLSelectElement>("vm-quality-problem").value = "";
    sourceProblemInputs();
  }
  draw();
  renderLane();
  const index = filteredReviews().findIndex(row => row.lane === id);
  if (index >= 0 && Math.floor(index / REVIEW_PAGE_SIZE) !== reviewPage) {
    setReviewPage(Math.floor(index / REVIEW_PAGE_SIZE)); renderReviews();
  } else markReviewSelection();
  const lane = id === null ? undefined : laneById(id);
  if (frame && lane) {
    const shift = globalShift();
    const box = new THREE.Box3();
    for (const p of [...lane.left, ...lane.right]) box.expandByPoint(new THREE.Vector3(p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]));
    viewer.frameBox(box.expandByScalar(10));
  }
}

export function renderLane(): void {
  const lane = selected === null ? undefined : laneById(selected);
  laneBox.hidden = !lane;
  if (!lane) return;
  const review = reviews.get(lane.id);
  $<HTMLSelectElement>("vm-review-state").value = review?.status ?? "unreviewed";
  $<HTMLTextAreaElement>("vm-review-notes").value = review?.notes ?? "";
  $("vm-review-stale").textContent = review?.staleReason ?? "";
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

/** Height used when a click hits no point: the map's, else 0. */
function fallbackZ(): number {
  const p = view.lanes[0]?.center[0] ?? sketch[sketch.length - 1];
  return p ? p[2] : 0;
}

/** Original coordinates of a click: the point under it, else the ray at the map's height. */
export async function clickPoint(x: number, y: number): Promise<XYZ | null> {
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
export function sameWayGroup(id: number): number[] {
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
export function laneTool(button: string, what: string, act: (near: Nearest) => Promise<unknown>): Tool {
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
