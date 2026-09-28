/** Lasso segmentation: draw a polygon on the view and cut the points inside it out. */

import * as THREE from "three";
import { segmentCloud } from "../api";
import { clipBox } from "./clip";
import { refreshColors } from "./colors";
import { $, errorText, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { clouds, type Entry, hiddenClasses, viewer } from "./state";
import { activeTool, setTool, shortcut, type Tool, toggleTool } from "./tools";

type Keep = "inside" | "outside" | "both";

/** Lasso vertices in client (page) pixels. */
let vertices: [number, number][] = [];
const button = $<HTMLButtonElement>("segment");
const bar = $("segment-bar");
const svg = $("lasso") as unknown as SVGSVGElement;
const outline = svg.querySelector("polygon")!;
const actions = [...bar.querySelectorAll<HTMLButtonElement>("[data-keep]")];

function render(): void {
  const rect = svg.getBoundingClientRect();
  outline.setAttribute("points", vertices.map(([x, y]) => `${x - rect.left},${y - rect.top}`).join(" "));
  for (const b of actions) b.disabled = vertices.length < 3;
  $("segment-hint").textContent =
    vertices.length < 3
      ? "Click around the points to cut out; Backspace removes the last vertex."
      : `${vertices.length} vertices. Enter keeps the inside, Delete removes it.`;
}

const segmentTool: Tool = {
  click(x, y) {
    vertices.push([x, y]);
    render();
  },
  doubleClick() {
    // The second click of the double click added a duplicate vertex.
    vertices.pop();
    render();
  },
  key(e) {
    if (e.key === "Backspace") {
      vertices.pop();
      render();
    } else if (e.key === "Enter" && vertices.length >= 3) {
      void apply("inside");
    } else if (e.key === "Delete" && vertices.length >= 3) {
      void apply("outside");
    } else {
      return false;
    }
    return true;
  },
  enter() {
    vertices = [];
    button.setAttribute("aria-pressed", "true");
    bar.hidden = false;
    svg.toggleAttribute("hidden", false);
    render();
  },
  exit() {
    vertices = [];
    button.setAttribute("aria-pressed", "false");
    bar.hidden = true;
    svg.toggleAttribute("hidden", true);
  },
};

/** Row-major matrix from a cloud's original coordinates to clip space. */
function clipFromWorld(shift: readonly number[]): number[] {
  const { camera } = viewer;
  camera.updateMatrixWorld();
  const m = new THREE.Matrix4()
    .multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
    .multiply(new THREE.Matrix4().makeTranslation(-shift[0], -shift[1], -shift[2]));
  return m.transpose().toArray();
}

/** Cut every visible cloud with the lasso, keeping the inside, the outside or both. */
async function apply(keep: Keep): Promise<void> {
  const sources = clouds().filter((e) => e.visible);
  if (vertices.length < 3 || sources.length === 0) return;
  const rect = document.querySelector("#viewport > canvas")!.getBoundingClientRect();
  const polygon = vertices.flatMap(([x, y]) => [
    ((x - rect.left) / rect.width) * 2 - 1,
    1 - ((y - rect.top) / rect.height) * 2,
  ]);
  setTool(null);
  const cut: Entry[] = [];
  const added: Entry[] = [];
  const failed: string[] = [];
  setStatus("Segmenting…");
  for (const source of sources) {
    try {
      const parts = await segmentCloud({
        id: source.cloud.id,
        matrix: clipFromWorld(source.cloud.shift),
        polygon,
        clip: clipBox(),
        hiddenClasses: [...hiddenClasses],
        keep,
      });
      for (const cloud of parts) {
        const entry = addEntry(cloud);
        entry.mode = source.mode === "c2c" ? "solid" : source.mode;
        refreshColors(entry);
        added.push(entry);
      }
      cut.push(source);
    } catch (err) {
      failed.push(`${source.cloud.name}: ${errorText(err)}`);
    }
  }
  if (cut.length) record({ label: "the lasso cut", added, hide: cut });
  renderList();
  const summary = added.map((e) => `${e.cloud.name} (${e.cloud.count.toLocaleString()})`).join(", ");
  setStatus(
    [summary && `Segmented: ${summary}`, ...failed].filter(Boolean).join("; ") || "Nothing to segment",
    failed.length > 0 && added.length === 0,
  );
}

button.onclick = () => toggleTool(segmentTool);
shortcut("s", segmentTool);
for (const b of actions) b.onclick = () => void apply(b.dataset.keep as Keep);
$<HTMLButtonElement>("segment-cancel").onclick = () => {
  if (activeTool() === segmentTool) setTool(null);
};
