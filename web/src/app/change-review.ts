import { projectChanged } from "../project-change";
/** Review one frozen two-pass comparison without moving the camera between views. */
import * as THREE from "three";
import { CHANGE_DECISIONS, CHANGE_NOTE_LIMIT, changeCandidates, changeDecisionCounts, changeReviewScope,
  comparisonDiagnostics, filterChangeCandidates, graphEvidence, nextUnreviewedCandidate, sameGraphEvidence, setChangeDecision, type ChangeFilter, type ComparisonContext, type SurveyReview } from "../change-review";
import type { PoseGraphState } from "../protocol";
import { overlappingSurveyLanes } from "../lane-surface-overlap";
import { focusSurveyLane, mapProjectReady, surveyMapView, vectorMapChanged } from "./vectormap";
import { exportCloud } from "../api";
import { captureLaneCoreSnapshot, clearLaneCoreView, requestLaneCoreView } from "./lane-core-view";
import { invalidateLaneTasks, recordSurveyLaneTask, refreshLaneTasks, setLaneTasksBusy } from "./survey-lane-tasks";
import { checkSurveyResume, parseSurveyReview, resumeSurveyDecisions, surveyReviewIdentity, SURVEY_REVIEW_BYTES } from "../survey-review-archive";
import { $, download, errorText, fmt, setStatus } from "./dom";
import { renderList } from "./entries";
import { entries, listChanged, pointsInvalidated, toRender, viewer, type Entry } from "./state";

const PAGE_SIZE = 50;
let review: SurveyReview | null = null;
let evidence: PoseGraphState | null = null;
let selected = 0;
let page = 0;
let busy = false;
const view = $<HTMLSelectElement>("pg-change-view");
const filter = $<HTMLSelectElement>("pg-change-filter");
const decision = $<HTMLSelectElement>("pg-change-decision");
const note = $<HTMLTextAreaElement>("pg-change-note");
for (const [value, label] of Object.entries(CHANGE_DECISIONS)) {
  decision.add(new Option(label, value)); filter.add(new Option(label, value));
}
note.maxLength = CHANGE_NOTE_LIMIT;

function makeReview(context: ComparisonContext, state: PoseGraphState, before: Entry, after: Entry, result: Entry, flat: Float64Array): SurveyReview {
  const source = (entry: Entry) => ({ id: entry.cloud.id, name: entry.cloud.name, points: entry.cloud.count });
  return { schema: "cloudanalyzer.survey_change_review.v1", scope: changeReviewScope,
    analyzedAt: new Date().toISOString(), staleReason: null, identity: null,
    context: structuredClone(context), sources: { before: source(before), after: source(after), difference: source(result) },
    diagnostics: comparisonDiagnostics(state, state.nodeIds.indexOf(context.splitNode), result.c2c!.distances),
    candidates: changeCandidates(flat) };
}

export async function startChangeReview(context: ComparisonContext, state: PoseGraphState, before: Entry, after: Entry, result: Entry, flat: Float64Array): Promise<void> {
  const next = makeReview(context, state, before, after, result, flat);
  const frozen = graphEvidence(state);
  const clouds = [before, after, result].map(entry => ({ entry, cloud: entry.cloud }));
  if (globalThis.crypto?.subtle) next.identity = await surveyReviewIdentity(next, frozen, id => exportCloud(id, "ply"));
  if (!sameGraphEvidence(frozen, state) || clouds.some(({ entry, cloud }) => entries.get(cloud.id) !== entry || entry.cloud !== cloud)) {
    throw new Error("Comparison inputs changed while preparing the review. Run Compare parts again.");
  }
  // Earlier comparison clouds otherwise obscure before/after on a rerun.
  if (review) for (const source of Object.values(review.sources)) {
    const entry = entries.get(source.id);
    if (entry) { entry.visible = false; viewer.setVisible(source.id, false); }
  }
  review = next;
  projectChanged();
  evidence = frozen;
  selected = page = 0;
  resetLaneAssociations();
  view.value = "difference";
  filter.value = "all";
  $("pg-changes").hidden = false;
  renderList();
  renderReview();
}

export function invalidateChangeReview(reason: string): void {
  if (!review || review.staleReason) return;
  clearLaneCoreView(true);
  review.staleReason = reason;
  invalidateLaneTasks(reason);
  evidence = null;
  renderReview();
}
export function checkChangeReviewGraph(state: PoseGraphState): void {
  if (evidence && !sameGraphEvidence(evidence, state)) invalidateChangeReview("Graph poses or constraints changed. Run Compare parts again.");
}
export function setChangeReviewBusy(value: boolean): void {
  const changed = busy !== value;
  busy = value;
  if (changed && busy) clearLaneCoreView();
  if (changed && !busy && review) renderLaneAssociations();
  renderControls();
}
function renderControls(): void {
  setLaneTasksBusy(busy);
  const locked = busy || !review || !!review.staleReason;
  view.disabled = locked;
  filter.disabled = busy || !review;
  const lanesLocked = locked || !surveyMapView().lanes.length;
  $<HTMLInputElement>("pg-change-lane-frame").disabled = lanesLocked;
  for (const id of ["pg-change-lane-xy", "pg-change-lane-z"]) $<HTMLInputElement>(id).disabled = lanesLocked;
  $<HTMLButtonElement>("pg-change-next-unreviewed").disabled = locked || !nextUnreviewedCandidate(review?.candidates ?? [], selected + 1);
  $<HTMLButtonElement>("pg-change-show-selected").disabled = busy || !review;
  decision.disabled = note.disabled = locked || !review?.candidates.length;
  $<HTMLButtonElement>("pg-change-export").disabled = busy || !review;
  $<HTMLButtonElement>("pg-change-prev").disabled = busy || page === 0;
  $<HTMLButtonElement>("pg-change-next").disabled = busy || !review || (page + 1) * PAGE_SIZE >= visibleCandidates().length;
  for (const button of $("pg-change-list").querySelectorAll("button")) button.disabled = locked;
  renderArchiveControls();
  for (const button of $("pg-change-lane-list").querySelectorAll("button")) button.disabled = locked || button.dataset.coreReady === "false";
}
function visibleCandidates() {
  return filterChangeCandidates(review?.candidates ?? [], filter.value as ChangeFilter);
}
function renderSummary(): void {
  if (!review) return;
  const counts = changeDecisionCounts(review.candidates);
  $("pg-change-summary").textContent = Object.entries(CHANGE_DECISIONS).map(([key, label]) =>
    `${counts[key as keyof typeof counts]} ${label.toLowerCase()}`).join(" · ");
}
function renderSelection(): void {
  const candidate = review?.candidates[selected];
  $("pg-change-editor").hidden = !candidate;
  if (!candidate) return;
  $("pg-change-selected").textContent = `Candidate #${candidate.rank}`;
  const shown = visibleCandidates().slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).some(item => item.rank === candidate.rank);
  $("pg-change-selected-context").textContent = shown ? "" : "Selected candidate is outside this page or filter. Its decision and note remain selected below.";
  $("pg-change-show-selected").hidden = shown;
  decision.value = candidate.decision;
  note.value = candidate.note;
  renderLaneAssociations();
  for (const button of $("pg-change-list").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.rank) === candidate.rank));
  }
}
function renderReview(): void {
  if (!review) return;
  const { candidates, diagnostics: d } = review;
  $("pg-change-stale").hidden = !review.staleReason;
  $("pg-change-stale").textContent = review.staleReason ? `Historical review: ${review.staleReason} Saved decisions can still be exported.` : "";
  $("pg-change-quality").textContent =
    `${review.context.graph} · split at node ${review.context.splitNode}. ` +
    `${d.crossPartLoops.length} cross-part loop constraints; ` +
    (d.weightedRms === null ? "weighted residual unavailable. " :
      `weighted residual √χ²: RMS ${fmt(d.weightedRms)}, max ${fmt(d.weightedMax!)} (${d.residualsAvailable} measured). `) +
    `M3C2: ${d.measured.toLocaleString()} / ${d.corePoints.toLocaleString()} core points measured; ` +
    `${d.unmeasured.toLocaleString()} unmeasured` +
    (d.corePoints ? ` (${(100 * d.unmeasured / d.corePoints).toFixed(1)} %).` : ".") +
    " Registration accuracy is unverified; inspect stable surfaces in both passes." +
    (!review.identity ? " Open over HTTPS or localhost to save resumable reviews." : "");
  $("pg-changes-hint").textContent = candidates.length === 0
    ? "No significant change forms an object. Unmeasured areas and smaller changes still need inspection."
    : `${candidates.length.toLocaleString()} changed object${candidates.length === 1 ? "" : "s"} (points that changed by ${review.context.clustering.minChange} m or more), largest first. Select a candidate and switch views to inspect its surroundings.`;
  refreshLaneTasks();
  renderCandidateList();
  renderSummary();
  renderSelection();
  renderControls();
}

function frameCandidate(candidate: NonNullable<SurveyReview>["candidates"][number]): void {
  const b = candidate.bounds;
  viewer.frameBox(new THREE.Box3(toRender(b.slice(0, 3)), toRender(b.slice(3, 6)))
    .expandByScalar(Math.max(2, 0.5 * Math.max(b[3] - b[0], b[4] - b[1]))));
}
function renderCandidateList(): void {
  const candidates = visibleCandidates();
  page = Math.min(page, Math.max(0, Math.ceil(candidates.length / PAGE_SIZE) - 1));
  $("pg-change-page").textContent = candidates.length
    ? `${page * PAGE_SIZE + 1}–${Math.min((page + 1) * PAGE_SIZE, candidates.length)} of ${candidates.length}` : "0 candidates";
  $("pg-change-filter-summary").textContent = `${candidates.length} of ${review?.candidates.length ?? 0} candidates match this filter. Exports include all candidates.`;
  $("pg-change-empty").hidden = candidates.length > 0;
  $("pg-change-list").replaceChildren(...candidates.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map(candidate => {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.className = "name link";
    button.dataset.rank = String(candidate.rank);
    const b = candidate.bounds;
    button.textContent = `#${candidate.rank} · ${[0, 1, 2].map(a => fmt(b[a + 3] - b[a])).join(" × ")} m`;
    button.title = "Select and frame this candidate";
    button.onclick = () => {
      if (busy || !review || review.staleReason) return;
      selected = candidate.rank - 1;
      renderSelection(); renderControls();
      frameCandidate(candidate);
    };
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `${candidate.meanChange > 0 ? "+" : ""}${fmt(candidate.meanChange)} m · ${candidate.points.toLocaleString()} points · ${CHANGE_DECISIONS[candidate.decision]}`;
    li.append(button, meta);
    return li;
  }));
}

view.onchange = () => {
  if (!review || busy || review.staleReason) return;
  for (const [key, source] of Object.entries(review.sources)) {
    const entry = entries.get(source.id);
    if (!entry) { invalidateChangeReview("A comparison cloud was removed."); return; }
    entry.visible = key === view.value;
    viewer.setVisible(source.id, entry.visible);
  }
  // No fit/frame call: before/after/difference share exactly the same camera.
  renderList();
};
function saveDecision(refreshList = false): void {
  if (!review || busy || review.staleReason || !review.candidates[selected]) return;
  try {
    setChangeDecision(review.candidates[selected], decision.value, note.value);
    projectChanged();
    refreshLaneTasks();
    renderSummary();
    if (refreshList) { renderCandidateList(); renderSelection(); renderControls(); }
  } catch (err) { setStatus(errorText(err), true); renderSelection(); }
}
decision.onchange = () => saveDecision(true);
note.oninput = () => saveDecision();
filter.onchange = () => {
  if (busy || !review) return;
  page = 0; renderReview();
};
$("pg-change-show-selected").onclick = () => {
  if (busy || !review) return;
  filter.value = "all"; page = Math.floor(selected / PAGE_SIZE); renderReview();
};
$("pg-change-next-unreviewed").onclick = () => {
  if (busy || !review || review.staleReason) return;
  const candidate = nextUnreviewedCandidate(review.candidates, selected + 1);
  if (!candidate) return;
  selected = candidate.rank - 1;
  filter.value = "unreviewed";
  page = Math.floor(visibleCandidates().findIndex(item => item.rank === candidate.rank) / PAGE_SIZE);
  renderReview(); frameCandidate(candidate);
};
$("pg-change-prev").onclick = () => { if (!busy && page > 0) { page--; renderReview(); } };
$("pg-change-next").onclick = () => {
  if (!busy && review && (page + 1) * PAGE_SIZE < visibleCandidates().length) { page++; renderReview(); }
};
$("pg-change-export").onclick = () => {
  if (!review || busy) return;
  download(new Blob([JSON.stringify({ ...review, exportedAt: new Date().toISOString(),
    counts: changeDecisionCounts(review.candidates) }, null, 2)], { type: "application/json" }), "survey-change-review.json");
};
pointsInvalidated.add(id => {
  if (review && Object.values(review.sources).some(source => source.id === id)) {
    invalidateChangeReview("A comparison cloud was changed or removed. Run Compare parts again.");
  }
});
listChanged.add(() => {
  if (review && Object.values(review.sources).some(source => !entries.has(source.id))) {
    invalidateChangeReview("A comparison cloud was removed. Run Compare parts again.");
  }
});


// Saved files are independent historical records until the explicit Resume action succeeds.
const archiveInput = $<HTMLInputElement>("pg-review-file");
let archive: SurveyReview | null = null;
let archivePage = 0;
let archiveGeneration = 0;
function resumeProblem(): string | null {
  if (busy) return "Finish the current operation before resuming decisions.";
  if (!archive) return "Open a saved review first.";
  if (archive.staleReason) return "Historical reviews cannot resume decisions";
  if (!archive.identity) return "This review has no content identity; it can be opened as history only";
  if (!review) return "Run Compare parts with the saved inputs to resume decisions.";
  try { checkSurveyResume(review, archive); return null; }
  catch (err) { return errorText(err); }
}
function renderArchiveControls(): void {
  const reason = resumeProblem();
  $<HTMLButtonElement>("pg-review-resume").disabled = reason !== null;
  $("pg-review-match").textContent = archive ? reason ?? "Comparison contents match. Resume replaces all current decisions with this file's decisions." : "";
  $<HTMLButtonElement>("pg-review-prev").disabled = archivePage === 0;
  $<HTMLButtonElement>("pg-review-next").disabled = !archive || (archivePage + 1) * PAGE_SIZE >= archive.candidates.length;
}
function renderArchive(): void {
  if (!archive) return;
  $("pg-review-summary").textContent = `${archive.context.graph} · split ${archive.context.splitNode} · ${archive.analyzedAt}. ` +
    `${archive.candidates.length} candidates. ` + Object.entries(changeDecisionCounts(archive.candidates))
      .map(([key, value]) => `${value} ${CHANGE_DECISIONS[key as keyof typeof CHANGE_DECISIONS].toLowerCase()}`).join(" · ") +
    (archive.staleReason ? ` Historical: ${archive.staleReason}` : "");
  $("pg-review-list").replaceChildren(...archive.candidates.slice(archivePage * PAGE_SIZE, (archivePage + 1) * PAGE_SIZE).map(candidate => {
    const li = document.createElement("li"), details = document.createElement("details"), title = document.createElement("summary");
    title.textContent = `#${candidate.rank} · ${CHANGE_DECISIONS[candidate.decision]} · ${candidate.points} points`;
    const geometry = document.createElement("p");
    geometry.className = "hint small";
    geometry.textContent = `Mean ${fmt(candidate.meanChange)} m; XYZ bounds [${candidate.bounds.join(", ")}]; updated ${candidate.updatedAt ?? "never"}`;
    const comment = document.createElement("p");
    comment.className = "archive-comment";
    comment.textContent = candidate.note || "(No note)";
    details.append(title, geometry, comment); li.append(details); return li;
  }));
  $("pg-review-page").textContent = archive.candidates.length
    ? `${archivePage * PAGE_SIZE + 1}–${Math.min((archivePage + 1) * PAGE_SIZE, archive.candidates.length)} of ${archive.candidates.length}` : "0 candidates";
  renderArchiveControls();
}
$("pg-review-open").onclick = () => archiveInput.click();
archiveInput.onchange = async () => {
  const file = archiveInput.files?.[0]; archiveInput.value = "";
  if (!file) return;
  const token = ++archiveGeneration;
  try {
    if (file.size > SURVEY_REVIEW_BYTES) throw new Error("Survey review exceeds the 16 MiB limit");
    const contents = await file.text();
    if (token !== archiveGeneration) return;
    const next = parseSurveyReview(JSON.parse(contents));
    archive = next; archivePage = 0;
    projectChanged();
    $("pg-review-name").textContent = file.name;
    $("pg-review-result").hidden = false;
    renderArchive();
    setStatus("Saved survey review opened as history; current decisions unchanged.");
  } catch (err) { if (token === archiveGeneration) setStatus(`Could not open survey review: ${errorText(err)}`, true); }
};
$("pg-review-resume").onclick = () => {
  if (busy || !review || !archive) return;
  try {
    review.candidates = resumeSurveyDecisions(review, archive);
    projectChanged();
    renderReview();
    setStatus("Saved survey decisions resumed for the matching comparison.");
  } catch (err) { setStatus(`Could not resume survey decisions: ${errorText(err)}`, true); }
};
$("pg-review-close").onclick = () => {
  archiveGeneration++; archive = null; archivePage = 0;
  $("pg-review-result").hidden = true; $("pg-review-list").replaceChildren();
  renderArchiveControls();
};
$("pg-review-prev").onclick = () => { if (archive && archivePage > 0) { archivePage--; renderArchive(); } };
$("pg-review-next").onclick = () => {
  if (archive && (archivePage + 1) * PAGE_SIZE < archive.candidates.length) { archivePage++; renderArchive(); }
};


// Map associations are transient hints, independent of saved point-cloud judgments.
const laneFrame = $<HTMLInputElement>("pg-change-lane-frame");
let confirmedMapRevision: number | null = null;
function resetLaneAssociations(): void {
  clearLaneCoreView(true);
  laneFrame.checked = false; confirmedMapRevision = null;
  $("pg-change-lane-list").replaceChildren();
}
function renderLaneAssociations(): void {
  clearLaneCoreView();
  const list = $("pg-change-lane-list"), hint = $("pg-change-lane-hint");
  list.replaceChildren();
  const candidate = review?.candidates[selected], map = surveyMapView();
  const locked = busy || !candidate || !!review?.staleReason || !map.lanes.length;
  laneFrame.disabled = locked;
  for (const id of ["pg-change-lane-xy", "pg-change-lane-z"]) $<HTMLInputElement>(id).disabled = locked;
  if (busy) { hint.textContent = "Finish the current operation before checking lane evidence."; return; }
  if (!candidate) { hint.textContent = "Select a change candidate first."; return; }
  if (review?.staleReason) { hint.textContent = "Historical comparison: rerun Compare parts before checking current map lanes."; return; }
  if (!map.lanes.length) { hint.textContent = "Open a Lanelet2 or vector-map JSON file in the Vector map panel."; return; }
  if (!laneFrame.checked || confirmedMapRevision !== map.revision) {
    hint.textContent = "Confirm a shared Cartesian metre frame before checking overlap with the current map."; return;
  }
  try {
    const xy = $<HTMLInputElement>("pg-change-lane-xy"), z = $<HTMLInputElement>("pg-change-lane-z");
    if (xy.value.trim() === "" || z.value.trim() === "") throw new Error("Enter both margins between 0 and 10 m");
    const result = overlappingSurveyLanes(map.lanes, candidate.bounds, Number(xy.value), Number(z.value));
    hint.textContent = `${result.ids.length} overlapping lane surface${result.ids.length === 1 ? "" : "s"} for candidate #${candidate.rank}; ` +
      `${result.checked} of ${result.total} lanes checked, ${result.invalid} invalid, ${result.limited} unchecked by limits. ` +
      `XY margin ${result.xyMargin} m; height margin ${result.zMargin} m.` +
      (result.ids.length > 50 ? " First 50 matches shown." : "") +
      (!result.ids.length ? " No checked surface overlaps this box; this does not establish that the map needs no update." : "");
    const bindings: {lane:number;label:HTMLSpanElement;button:HTMLButtonElement}[] = [];
    list.replaceChildren(...result.ids.slice(0, 50).map(id => {
      const li = document.createElement("li"), button = document.createElement("button");
      button.className = "name link"; button.textContent = `Inspect lane ${id}`;
      button.disabled = busy;
      button.onclick = () => {
        if (busy || !review || review.staleReason || !laneFrame.checked || confirmedMapRevision !== map.revision ||
            surveyMapView().revision !== map.revision || review.candidates[selected] !== candidate) return;
        if (!focusSurveyLane(id)) setStatus("Finish the map operation before selecting a related lane.", true);
      };
      const record = document.createElement("button");
      record.className = "link"; record.textContent = `Record lane ${id}`; record.disabled = busy;
      const commitLane = (resume: boolean) => {
        const comparison = review, lane = map.lanes.find(lane => lane.id === id);
        if (!comparison || !lane) return;
        const version = JSON.stringify(candidate);
        const current = () => review === comparison && !comparison.staleReason && surveyMapView().revision === map.revision &&
          JSON.stringify(comparison.candidates[candidate.rank - 1]) === version;
        const canCommit = () => current() && !busy && mapProjectReady() && selected === candidate.rank - 1 && laneFrame.checked &&
          confirmedMapRevision === map.revision && xy.value.trim() !== "" && z.value.trim() !== "" &&
          Number(xy.value) === result.xyMargin && Number(z.value) === result.zMargin;
        void recordSurveyLaneTask(comparison, candidate, lane, { xyMargin: result.xyMargin, zMargin: result.zMargin,
          checked: result.checked, invalid: result.invalid, limited: result.limited, total: result.total }, map.revision, canCommit, current,
          () => captureLaneCoreSnapshot(entries.get(comparison.sources.difference.id)?.cloud, candidate, lane, result.zMargin), resume);
      };
      record.onclick = () => commitLane(false);
      const resume = document.createElement("button");
      resume.className = "link"; resume.textContent = `Resume saved task for lane ${id}`; resume.disabled = busy;
      resume.onclick = () => commitLane(true);
      const support = document.createElement("span"); support.className = "meta";
      support.dataset.coreLane = String(id); support.dataset.coreStatus = "pending";
      support.textContent = "Counting original changed cores…";
      const show = document.createElement("button"); show.className = "link";
      show.textContent = `Show changed cores for lane ${id}`; show.disabled = true; show.dataset.coreReady = "false";
      bindings.push({lane:id,label:support,button:show});
      li.append(button, record, resume, support, show); return li;
    }));
    const source = review && entries.get(review.sources.difference.id);
    if (review && source && bindings.length) {
      const comparison = review;
      requestLaneCoreView({ key: JSON.stringify([source.cloud.id,candidate.rank,candidate.points,candidate.bounds,map.revision,result.xyMargin,result.zMargin,result.ids.slice(0,50)]),
        cloud:source.cloud,candidate,lanes:result.ids.slice(0,50).map(id=>map.lanes.find(lane=>lane.id===id)!),heightMargin:result.zMargin,
        current:()=>!busy && review===comparison && !comparison.staleReason && review.candidates[selected]===candidate &&
          laneFrame.checked && confirmedMapRevision===map.revision && surveyMapView().revision===map.revision &&
          Number(xy.value)===result.xyMargin && Number(z.value)===result.zMargin,
        frame:()=>frameCandidate(candidate) },bindings);
    }
  } catch (err) { hint.textContent = errorText(err); }
}
laneFrame.onchange = () => {
  if (busy || !review || review.staleReason) return;
  confirmedMapRevision = laneFrame.checked ? surveyMapView().revision : null;
  renderLaneAssociations();
};
for (const id of ["pg-change-lane-xy", "pg-change-lane-z"]) {
  $<HTMLInputElement>(id).oninput = renderLaneAssociations;
}
vectorMapChanged.add(() => { invalidateLaneTasks("The HD map changed; saved lane IDs are historical."); resetLaneAssociations(); renderLaneAssociations(); });

/** Preserve evidence; reconnect only through the existing exact-content resume gate. */
export function captureSurveyReview(): unknown | undefined {
  const saved = review ?? archive;
  return saved ? structuredClone({ ...saved, exportedAt: new Date().toISOString(), counts: changeDecisionCounts(saved.candidates) }) : undefined;
}
export function restoreSurveyReview(value: unknown | undefined): void {
  const next = value === undefined ? null : parseSurveyReview(value);
  archiveGeneration++;
  review = null; evidence = null; archive = next; archivePage = 0;
  resetLaneAssociations();
  $("pg-changes").hidden = true;
  $("pg-review-result").hidden = !archive;
  $("pg-review-list").replaceChildren();
  $("pg-review-name").textContent = "Project survey review";
  if (archive) renderArchive();
  renderArchiveControls();
}
