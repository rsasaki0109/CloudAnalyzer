/** Taking map edits from the worker: the view, undo depth, issues and status. */

import { projectChanged } from "../project-change";
import { vectorMap } from "../api";
import { $, errorText, setStatus } from "./dom";
import { Signal } from "./state";
import type { SurveyLane } from "../lane-surface-overlap";
import {
  busy,
  clearBuildFailure,
  clearCrosswalkPreview,
  clearDiscovery,
  clearJunctionPreview,
  clearQuality,
  clearRelationPreview,
  clearSignalPreview,
  draw,
  type Edited,
  exportButton,
  type Issue,
  issueList,
  junctionInputs,
  laneSignatures,
  type LaneView,
  renderBoundaryEditor,
  renderBoundaryEvidence,
  renderFeatures,
  renderLane,
  renderRelations,
  renderReviews,
  reviews,
  selected,
  selectLane,
  setBusy,
  setSelected,
  setUndoDepth,
  setView,
  status,
  undoButton,
  undoDepth,
  view,
} from "./vectormap";

export const vectorMapChanged = new Signal();
let surveyMapRevision = 0;
/** Read-only input for spatial review hints; the geometry remains owned by the map editor. */
export function surveyMapView(): { revision: number; lanes: readonly SurveyLane[] } {
  return { revision: surveyMapRevision, lanes: view.lanes };
}
export function focusSurveyLane(id: number): boolean {
  if (busy || !view.lanes.some(lane => lane.id === id)) return false;
  selectLane(id, true);
  return true;
}
export function takeView(edited: Edited, editing = true): void {
  clearBuildFailure();
  clearQuality();
  $("vm-export-report").hidden = true;
  $("vm-export-issues").replaceChildren();
  setView(edited.view);
  reviews.reconcile(laneSignatures());
  renderReviews();
  renderBoundaryEvidence();
  clearRelationPreview();
  clearDiscovery();
  clearJunctionPreview();
  clearSignalPreview();
  clearCrosswalkPreview();
  setUndoDepth(edited.undo);
  renderFeatures();
  renderBoundaryEditor();
  renderRelations();
  if (selected !== null && !view.lanes.some((l) => l.id === selected)) setSelected(null);
  draw();
  undoButton.disabled = undoDepth === 0;
  exportButton.disabled = view.lanes.length === 0;
  $<HTMLButtonElement>("vm-fit").disabled = view.boundaries.length === 0;
  $<HTMLButtonElement>("vm-plan").disabled = $<HTMLButtonElement>("vm-iso").disabled = view.boundaries.length === 0;
  renderLane();
  void renderIssues();
  surveyMapRevision++;
  vectorMapChanged.emit();
  if (editing) projectChanged();
}

/** Run vectormap commands; failures go to the status line. Returns false if they failed. */
export async function apply(commands: object[], what: string): Promise<boolean> {
  if (busy) return false;
  setBusy(true);
  junctionInputs();
  try {
    const edited = await vectorMap<Edited>("apply", { text: JSON.stringify(commands) });
    takeView(edited);
    const warnings = (edited.result as { warnings?: Issue[] }[]).flatMap((c) => c.warnings ?? []);
    const shown = warnings.filter((w) => w.code !== "no_op");
    setStatus(shown.length ? `${what}: ${shown.map((w) => w.message).join("; ")}` : what);
    return true;
  } catch (err) {
    setStatus(`${what} failed: ${errorText(err)}`);
    return false;
  } finally {
    setBusy(false);
    junctionInputs();
  }
}

export const plural = (n: number, what: string) => `${n} ${what}${n === 1 ? "" : "s"}`;

async function renderIssues(): Promise<void> {
  const issues = await vectorMap<Issue[]>("validate", { autoware: true });
  const counts = { error: 0, warning: 0, info: 0 };
  for (const i of issues) counts[i.severity]++;
  const kinds = [
    [view.lanes.length, "lane"],
    [view.stopLines.length, "stop line"],
    [view.signals.length, "traffic light"],
    [view.crosswalks.length, "crosswalk"],
  ] as const;
  const parts = kinds.filter(([n]) => n > 0).map(([n, what]) => plural(n, what));
  status.textContent =
    view.lanes.length === 0
      ? "No map yet."
      : `${parts.join(", ")}. Autoware check: ${plural(counts.error, "error")}, ${plural(counts.warning, "warning")}.`;
  issueList.replaceChildren(
    ...issues
      .filter((i) => i.severity !== "info")
      .slice(0, 20)
      .map((issue) => {
        const li = document.createElement("li");
        li.className = issue.severity;
        li.textContent = issue.message;
        li.title = issue.code;
        if (issue.entity?.kind === "lane") {
          const lane = issue.entity.id;
          li.classList.add("clickable");
          li.onclick = () => selectLane(lane, true);
        }
        return li;
      }),
  );
}

export function laneById(id: number): LaneView | undefined {
  return view.lanes.find((l) => l.id === id);
}

export function laneLength(lane: LaneView): number {
  let length = 0;
  const c = lane.center;
  for (let i = 1; i < c.length; i++) length += Math.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]);
  return length;
}
