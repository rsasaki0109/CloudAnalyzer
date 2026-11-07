import { projectChanged } from "../project-change";
/** A bounded follow-up queue, independent of native map edits and historical file inspection. */
import { LANE_TASK_BYTES, LANE_TASK_LIMIT, LANE_TASK_STATES, laneTaskCounts, laneTaskFile, newLaneTask,
  parseLaneTaskFile, resumeLaneTask, setLaneTaskDecision, laneCoreSnapshotText, type LaneCoreSnapshot, type SurveyLaneTask } from "../survey-lane-tasks";
import type { ChangeCandidate, SurveyReview } from "../change-review";
import type { SurveyLane } from "../lane-surface-overlap";
import { captureMapProject, focusSurveyLane, surveyMapView } from "./vectormap";
import { $, download, errorText, setStatus } from "./dom";

interface Entry { task: SurveyLaneTask; valid: (() => boolean) | null }
const records: Entry[] = [];
let queueGeneration = 0;
let selected = -1, page = 0, nextId = 1, busy = false, recording = false;
let mapHash: { revision: number; digest: string | null } | null = null;
const state = $<HTMLSelectElement>("pg-lane-task-state"), note = $<HTMLTextAreaElement>("pg-lane-task-note");
for (const [value, label] of Object.entries(LANE_TASK_STATES)) state.add(new Option(label, value));
function expire(entry: Entry, reason: string): void {
  if (!entry.task.staleReason) entry.task.staleReason = reason;
  entry.valid = null;
}
export function refreshLaneTasks(): void {
  let changed = false;
  for (const entry of records) if (entry.valid && !entry.valid()) { expire(entry, "The source comparison, candidate judgment or map changed."); changed = true; }
  if (changed) renderTasks();
}
export function invalidateLaneTasks(reason: string): void {
  for (const entry of records) if (entry.valid) expire(entry, reason);
  mapHash = null;
  renderTasks();
}
export function setLaneTasksBusy(value: boolean): void { busy = value; renderTaskControls(); }
export async function recordSurveyLaneTask(review: SurveyReview, candidate: ChangeCandidate, lane: SurveyLane,
  overlap: SurveyLaneTask["overlap"], revision: number, canCommit: () => boolean, remainsCurrent: () => boolean, captureCores: () => Promise<LaneCoreSnapshot>, resume = false): Promise<void> {
  if (busy || recording) return;
  const generation = queueGeneration, sourceCurrent = canCommit;
  canCommit = () => generation === queueGeneration && sourceCurrent();
  const resumable = resume ? records.filter(entry => !entry.valid && !entry.task.staleReason &&
    entry.task.candidate.rank === candidate.rank && entry.task.lane.id === lane.id) : [];
  recording = true; renderTaskControls();
  try {
    if (!canCommit()) throw new Error("The lane overlap changed; inspect it again before recording");
    const duplicate = records.findIndex(entry => entry.valid && entry.valid() &&
      entry.task.candidate.rank === candidate.rank && entry.task.lane.id === lane.id);
    if (duplicate >= 0) {
      selected = duplicate; page = Math.floor(selected / 50); renderTasks();
      setStatus("This candidate and lane already have a current follow-up record."); return;
    }
    if (resume && resumable.length !== 1) throw new Error("Select a lane with exactly one saved task awaiting verification");
    if (!resume && records.length >= LANE_TASK_LIMIT) throw new Error("Lane task queue is limited to 1,000 records");
    setStatus(`Recording candidate #${candidate.rank}, lane ${lane.id}…`);
    if (!mapHash || mapHash.revision !== revision) {
      const text = await captureMapProject();
      const digest = globalThis.crypto?.subtle ? [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)))]
        .map(b => b.toString(16).padStart(2, "0")).join("") : null;
      if (!canCommit() || surveyMapView().revision !== revision) throw new Error("Map or comparison changed while recording; inspect the overlap again");
      mapHash = { revision, digest };
    }
    if (!canCommit()) throw new Error("Comparison changed while recording; inspect the overlap again");
    const coreSupport = await captureCores();
    if (!canCommit()) throw new Error("Map, comparison or selection changed while collecting core evidence; inspect it again");
    const task = newLaneTask(nextId, review, candidate, lane, overlap, mapHash.digest, coreSupport);
    if (resume) {
      const entry = resumable[0];
      entry.task = resumeLaneTask(entry.task, task);
      entry.valid = remainsCurrent;
      selected = records.indexOf(entry); page = Math.floor(selected / 50);
      $<HTMLDetailsElement>("pg-lane-tasks").open = true;
      projectChanged(); renderTasks();
      setStatus(`Resumed task ${entry.task.id}; saved decision and note retained.`);
      return;
    }
    // Validate the export budget before committing, reserving 256 bytes per record for the fixed historical-reason messages.
    const proposed = laneTaskFile([...records.map(entry => entry.task), task]);
    if (new Blob([JSON.stringify(proposed, null, 2)]).size + 256 * proposed.tasks.length > LANE_TASK_BYTES) throw new Error("Lane task records exceed the 16 MiB export limit");
    parseLaneTaskFile(proposed);
    records.push({ task, valid: remainsCurrent }); nextId++; projectChanged();
    selected = records.length - 1; page = Math.floor(selected / 50);
    $<HTMLDetailsElement>("pg-lane-tasks").open = true;
    renderTasks(); setStatus(`Recorded candidate #${candidate.rank}, lane ${lane.id} for human follow-up.`);
  } catch (err) { setStatus(`Could not record lane task: ${errorText(err)}`, true); }
  finally { recording = false; renderTaskControls(); }
}
function renderTaskControls(): void {
  const entry = records[selected], locked = busy || recording || !entry || !entry.valid || !!entry.task.staleReason;
  state.disabled = note.disabled = locked;
  $<HTMLButtonElement>("pg-lane-task-save").disabled = locked;
  $<HTMLButtonElement>("pg-lane-task-inspect").disabled = locked;
  $<HTMLButtonElement>("pg-lane-task-export").disabled = busy || recording || records.length === 0;
  $<HTMLButtonElement>("pg-lane-task-prev").disabled = busy || page === 0;
  $<HTMLButtonElement>("pg-lane-task-next").disabled = busy || (page + 1) * 50 >= records.length;
}
function renderTasks(): void {
  const tasks = records.map(entry => entry.task), counts = laneTaskCounts(tasks);
  $("pg-lane-task-summary").textContent = `${tasks.length} follow-up records; ${records.filter(entry => !entry.valid || !!entry.task.staleReason).length} historical. ` +
    Object.entries(counts).map(([key, value]) => `${value} ${LANE_TASK_STATES[key as keyof typeof counts].toLowerCase()}`).join(" · ");
  $("pg-lane-task-list").replaceChildren(...records.slice(page * 50, (page + 1) * 50).map((entry, i) => {
    const li = document.createElement("li"), button = document.createElement("button");
    button.className = "name link";
    button.textContent = `Task ${entry.task.id}: lane ${entry.task.lane.id}, candidate #${entry.task.candidate.rank} · ${LANE_TASK_STATES[entry.task.status]}${entry.task.staleReason ? " · historical" : ""}`;
    button.setAttribute("aria-pressed", String(selected === page * 50 + i));
    button.onclick = () => { selected = page * 50 + i; renderTasks(); };
    li.append(button); return li;
  }));
  $("pg-lane-task-page").textContent = records.length ? `${page * 50 + 1}–${Math.min((page + 1) * 50, records.length)} of ${records.length}` : "0 tasks";
  const task = records[selected]?.task;
  $("pg-lane-task-editor").hidden = !task;
  if (task) {
    $("pg-lane-task-title").textContent = `Task ${task.id} · lane ${task.lane.id} · candidate #${task.candidate.rank}`;
    $("pg-lane-task-source").textContent = `${task.comparison.graph} · comparison ${task.comparison.analyzedAt}. ` +
      `XY margin ${task.overlap.xyMargin} m; height margin ${task.overlap.zMargin} m. ` + (task.staleReason ?? (records[selected]?.valid ? "Current source snapshot." : "Saved snapshot awaiting verification. Use Resume saved task in Related map lanes after restoring the matching comparison decisions.")) + " " + laneCoreSnapshotText(task.coreSupport);
    state.value = task.status; note.value = task.note;
  }
  renderTaskControls();
}
$("pg-lane-task-save").onclick = () => {
  if (busy || recording || !records[selected]) return;
  const entry = records[selected];
  try {
    if (!entry.valid || !entry.valid()) throw new Error("Verify the saved task against the current comparison and map before editing");
    const updated = structuredClone(entry.task);
    setLaneTaskDecision(updated, state.value, note.value);
    const proposed = laneTaskFile(records.map(record => record === entry ? updated : record.task));
    if (new Blob([JSON.stringify(proposed, null, 2)]).size + 256 * proposed.tasks.length > LANE_TASK_BYTES) throw new Error("Task notes exceed the 16 MiB export limit");
    entry.task = updated;
    projectChanged();
    renderTasks(); setStatus("Lane follow-up decision saved; map and survey judgments unchanged.");
  } catch (err) { setStatus(errorText(err), true); renderTasks(); }
};
$("pg-lane-task-inspect").onclick = () => {
  const entry = records[selected];
  if (busy || recording || !entry || entry.task.staleReason) return;
  if (!entry.valid?.()) { expire(entry, "Source changed before inspecting the lane."); renderTasks(); return; }
  if (!focusSurveyLane(entry.task.lane.id)) setStatus("Finish the map operation before inspecting this lane.", true);
};
$("pg-lane-task-export").onclick = () => {
  if (busy || recording || !records.length) return;
  refreshLaneTasks();
  download(new Blob([JSON.stringify(laneTaskFile(records.map(entry => entry.task)), null, 2)], { type: "application/json" }), "survey-lane-tasks.json");
};
$("pg-lane-task-prev").onclick = () => { if (!busy && page > 0) { page--; renderTasks(); } };
$("pg-lane-task-next").onclick = () => { if (!busy && (page + 1) * 50 < records.length) { page++; renderTasks(); } };

let archiveGeneration = 0;
const fileInput = $<HTMLInputElement>("pg-lane-task-file");
$("pg-lane-task-open").onclick = () => fileInput.click();
fileInput.onchange = async () => {
  const file = fileInput.files?.[0]; fileInput.value = ""; if (!file) return;
  const token = ++archiveGeneration;
  try {
    if (file.size > LANE_TASK_BYTES) throw new Error("Lane tasks file exceeds the 16 MiB limit");
    const contents = await file.text(); if (token !== archiveGeneration) return;
    const tasks = parseLaneTaskFile(JSON.parse(contents));
    $("pg-lane-task-archive-name").textContent = `${file.name}: ${tasks.length} historical follow-up records`;
    $("pg-lane-task-archive-list").replaceChildren(...tasks.map(task => {
      const li = document.createElement("li"), details = document.createElement("details"), title = document.createElement("summary"), text = document.createElement("p");
      title.textContent = `Task ${task.id}: lane ${task.lane.id}, candidate #${task.candidate.rank} · ${LANE_TASK_STATES[task.status]}`;
      text.className = "archive-comment"; text.textContent = `${task.comparison.graph} · ${task.comparison.analyzedAt}\nMap SHA-256: ${task.mapSha256 ?? "unavailable"}; XY margin ${task.overlap.xyMargin} m, height margin ${task.overlap.zMargin} m.\n${laneCoreSnapshotText(task.coreSupport)}\nSurvey decision: ${task.candidate.decision}; ${task.candidate.note}\n${task.note || "(No task note)"}\n${task.staleReason ?? "Saved snapshot; current map identity is not established."}`;
      details.append(title, text); li.append(details); return li;
    }));
    $("pg-lane-task-archive").hidden = false;
    setStatus("Lane tasks opened as historical records; current queue and map unchanged.");
  } catch (err) { if (token === archiveGeneration) setStatus(`Could not open lane tasks: ${errorText(err)}`, true); }
};
$("pg-lane-task-archive-close").onclick = () => { archiveGeneration++; $("pg-lane-task-archive").hidden = true; $("pg-lane-task-archive-list").replaceChildren(); };
renderTasks();

export function captureSurveyLaneTasks(): unknown | undefined {
  refreshLaneTasks();
  return records.length ? structuredClone(laneTaskFile(records.map(entry => entry.task))) : undefined;
}
export function restoreSurveyLaneTasks(value: unknown | undefined): void {
  const tasks = value === undefined ? [] : parseLaneTaskFile(value);
  queueGeneration++;
  records.splice(0, records.length, ...tasks.map(task => ({ task, valid: null })));
  nextId = Math.max(0, ...tasks.map(task => task.id)) + 1;
  selected = tasks.length ? 0 : -1; page = 0; mapHash = null;
  renderTasks();
}
