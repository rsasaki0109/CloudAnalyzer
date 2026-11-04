/** Drawing roads, and the click tools that connect, split or remove lanes. */

import { vectorMap } from "../api";
import { SelectionQueue } from "../selection-queue";
import { $, setStatus } from "./dom";
import { activeTool, setTool, toggleTool, type Tool } from "./tools";
import {
  apply,
  buildFailureRevision,
  buildInputs,
  type BuildReport,
  buildSketchRevision,
  busy,
  checkRoadSearchMargin,
  clearBuildFailure,
  clickPoint,
  cloudInput,
  draw,
  type Edited,
  hint,
  laneTool,
  renderBuildReport,
  roadBuildOptions,
  sameWayGroup,
  setBusy,
  setSketch,
  showBuildFailure,
  sketch,
  takeView,
  type XYZ,
} from "./vectormap";

// Draw road: click points along the road, double-click or Enter to build it.

function roadCommand(reference: XYZ[]): object {
  const forward = Math.max(0, Math.round(Number($<HTMLInputElement>("vm-forward").value)));
  const backward = Math.max(0, Math.round(Number($<HTMLInputElement>("vm-backward").value)));
  const width = Number($<HTMLInputElement>("vm-width").value) || 3.5;
  const leftHand = $<HTMLSelectElement>("vm-traffic").value === "left";
  const segment = Number($<HTMLInputElement>("vm-segment").value);
  const speed = Number($<HTMLInputElement>("vm-speed").value);
  const lane = (direction: string) => ({ width, direction });
  // Looking along the drawn line, left-hand traffic keeps its forward lanes on the left.
  const forwards = Array.from({ length: forward }, () => lane("forward"));
  const backwards = Array.from({ length: backward }, () => lane("backward"));
  const lanes = leftHand ? [...forwards, ...backwards] : [...backwards, ...forwards];
  const command: Record<string, unknown> = { op: "build_road", reference, lanes };
  // With both directions the line is the centre line between them.
  if (forward > 0 && backward > 0) command.left_edge = width * (leftHand ? forward : backward);
  if (segment > 0) command.segment_length = segment;
  if (speed > 0) command.speed_limit = { kmh: speed };
  return command;
}

const roadSelections = new SelectionQueue();

export function finishRoad(): Promise<void> {
  return roadSelections.finish(buildRoad);
}

async function buildRoad(): Promise<void> {
  if (busy) return;
  const reference = sketch;
  if (reference.length < 2) return setStatus("Click at least two points along the road.");
  const lanes = (roadCommand(reference) as { lanes: unknown[] }).lanes;
  if (lanes.length === 0) return setStatus("Give the road at least one lane.");
  if ($<HTMLInputElement>("vm-refine-sketch").checked && !checkRoadSearchMargin()) return;
  const cloud = Number(cloudInput.value);
  clearBuildFailure(); const revision = buildFailureRevision, pathRevision = buildSketchRevision;
  setSketch([]);
  draw();
  if (!$<HTMLInputElement>("vm-refine-sketch").checked) {
    if (await apply([roadCommand(reference)], "Road built")) setTool(null);
    else draw();
    return;
  }
  setBusy(true); buildInputs();
  try {
    if (!cloudInput.value) throw new Error("Choose the point cloud for this drawn path.");
    const edited = await vectorMap<Edited>("build", { id: cloud, positions: new Float64Array(reference.flat()), text: JSON.stringify(roadBuildOptions()) });
    takeView(edited); const report = edited.result as BuildReport; renderBuildReport(report); setTool(null);
    setStatus(`Road built from the point cloud and your traced path: ${report.lanes} added lanes. The path and nominal widths are operator inputs; review the boundary evidence report.`);
  } catch (err) {
    // Setting changes invalidate the diagnostic, but preserve the traced path.
    // Source changes, new picks and leaving the tool cancel its restoration.
    if (pathRevision === buildSketchRevision) setSketch(reference);
    setStatus(`Could not fit the drawn road: ${showBuildFailure(err, revision, cloud, null, reference)}`); draw();
  }
  finally { setBusy(false); buildInputs(); }
}

const roadTool: Tool = {
  click(x, y) {
    return roadSelections.enqueue(() => clickPoint(x, y), p => {
      if (!p) return;
      clearBuildFailure();
      sketch.push(p);
      draw();
      renderHint();
    });
  },
  doubleClick() {
    // The second click of the double click added a duplicate vertex.
    void roadSelections.finish(async () => {
      sketch.pop();
      await buildRoad();
    });
  },
  key(e) {
    if (e.key === "Enter") {
      void finishRoad();
      return true;
    }
    if (e.key === "Backspace") {
      void roadSelections.enqueue(async () => null, () => {
        clearBuildFailure();
        sketch.pop();
        draw();
        renderHint();
      });
      return true;
    }
    return false;
  },
  enter() {
    roadSelections.cancel();
    clearBuildFailure();
    setSketch([]);
    $("vm-road").setAttribute("aria-pressed", "true");
    draw();
    renderHint();
  },
  exit() {
    roadSelections.cancel();
    clearBuildFailure();
    setSketch([]);
    $("vm-road").setAttribute("aria-pressed", "false");
    draw();
    renderHint();
  },
};
$("vm-road").onclick = () => toggleTool(roadTool);

// Connect: the lane to leave, then the lane to enter.
let connectFrom: number | null = null;
const connectTool = laneTool("vm-connect", "Click the lane to leave, then the lane to enter.", async (near) => {
  if (connectFrom === null || connectFrom === near.lane) {
    connectFrom = near.lane;
    hint.textContent = `From lane ${near.lane}: now click the lane to enter.`;
    return;
  }
  const from = connectFrom;
  connectFrom = null;
  hint.textContent = "Click the lane to leave, then the lane to enter.";
  await apply([{ op: "add_connector", from, to: near.lane }], `Lanes ${from} and ${near.lane} connected`);
});
const connectExit = connectTool.exit;
connectTool.exit = () => {
  connectFrom = null;
  connectExit();
};

laneTool("vm-stop", "Click a lane where its stop line goes; it spans the lanes beside it going the same way.", (near) =>
  apply(
    [{ op: "add_stop_line", lanes: sameWayGroup(near.lane), placement: { at_station: { station: near.station } } }],
    "Stop line added",
  ),
);
laneTool(
  "vm-light",
  "Click a lane: a traffic light for it and the lanes beside it, at their stop line (made at the lane end if there is none).",
  (near) => apply([{ op: "add_traffic_signal", lanes: sameWayGroup(near.lane) }], "Traffic light added"),
);
laneTool("vm-crosswalk", "Click a lane where the crosswalk crosses the road; stop lines are added before it.", (near) =>
  apply(
    [
      {
        op: "add_crosswalk",
        geometry: { across: { lane: near.lane, station: near.station, width: 4.0, margin: 0.5 } },
        stop_line_offset: 1.5,
      },
    ],
    "Crosswalk added",
  ),
);
laneTool("vm-select", "Click a lane to see it, set its speed limit or remove it.", async () => {});

export function renderHint(): void {
  if (activeTool() === roadTool) {
    hint.textContent = sketch.length
      ? `${sketch.length} point${sketch.length === 1 ? "" : "s"}; double-click or Enter to build the road, Backspace to take one back, Esc to cancel.`
      : $<HTMLInputElement>("vm-refine-sketch").checked ? "Trace the outside forward lane over the points; Enter fits ground height and boundary evidence. The path and lane widths remain your inputs." : "Click points along the road (its centre line); double-click or Enter to build it.";
    return;
  }
  if (activeTool() === null || !hint.textContent) {
    hint.textContent =
      "Draw roads over the cloud, connect them through junctions, add stop lines, traffic lights and crosswalks, and save a Lanelet2 map for Autoware.";
  }
}
