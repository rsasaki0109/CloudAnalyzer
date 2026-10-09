import { bufferBytes, nativeCloudEstimate, historyPolicy, onHistoryPolicy } from "../memory-budget";
/**
 * Undo / redo of operations on the cloud list: an operation adds clouds and
 * hides its sources, or moves a cloud (ICP). Undone clouds stay in the worker
 * so redo is instant, and are freed when a new operation drops the redo steps.
 * Removing a cloud is not a step: it frees memory right away.
 */

import { removeCloud, transformCloud } from "../api";
import { $, errorText, setStatus, typing } from "./dom";
import { drawEntry, renderList, replaceCloud } from "./entries";
import { type Entry, entries, hideEntry, pointsInvalidated, putEntry, viewer } from "./state";
import { taskActive } from "./tasks";

interface Step {
  label: string;
  added: Entry[];
  /** Sources the operation hid (only those that were visible). */
  hidden: Entry[];
  /** A rigid transform applied to a cloud (row-major 4x4). */
  moved?: { entry: Entry; matrix: number[] };
}


const done: Step[] = [];
const undone: Step[] = [];
let busy = false;
export function cloudHistoryReady(): boolean { return !busy; }

const undoButton = $<HTMLButtonElement>("undo");
const redoButton = $<HTMLButtonElement>("redo");

/** Put a detached entry back in the list and the view. */
function attach(entry: Entry): void {
  putEntry(entry);
  drawEntry(entry);
}

/** Take an entry out of the list and the view, keeping its data in the worker. */
function detach(entry: Entry): void {
  entries.delete(entry.cloud.id);
  viewer.remove(entry.cloud.id);
  pointsInvalidated.emit(entry.cloud.id);
}

/** Release only detached clouds no remaining history step can resurrect. */
function release(list: Entry[]): void {
  const retained = new Set([...done, ...undone].flatMap(s => [...s.added, ...s.hidden, ...(s.moved ? [s.moved.entry] : [])]).map(e => e.cloud.id));
  for (const entry of new Set(list)) if (!entries.has(entry.cloud.id) && !retained.has(entry.cloud.id)) void removeCloud(entry.cloud.id);
}
function retainedHistoryBytes(): number {
  const data = [...done, ...undone];
  const referenced = new Map(data.flatMap(s => [...s.added, ...s.hidden, ...(s.moved ? [s.moved.entry] : [])]).map(e => [e.cloud.id, e]));
  return bufferBytes(data) + [...referenced.values()].reduce((sum,e) => sum + nativeCloudEstimate(e.cloud, e.fields.size),0);
}
function enforceBudget(): void {
  if (busy) return;
  const policy = historyPolicy();
  const removed: Step[] = [];
  while (done.length + undone.length && (done.length + undone.length > policy.steps || retainedHistoryBytes() > policy.bytes)) {
    removed.push(done.length ? done.shift()! : undone.shift()!);
  }
  release(removed.flatMap(s => [...s.added, ...s.hidden, ...(s.moved ? [s.moved.entry] : [])]));
  renderButtons();
}
export function historyMemory(): {bytes: number; steps: number; data: unknown} { return {bytes: retainedHistoryBytes(), steps: done.length + undone.length, data: [...done,...undone]}; }
export function clearCloudHistory(): boolean {
  if (busy) return false;
  const old = [...done.splice(0), ...undone.splice(0)];
  release(old.flatMap(s => [...s.added,...s.hidden])); renderButtons(); return true;
}
export function forgetCloudHistory(id: number): void {
  for (const list of [done,undone]) for (let i = list.length - 1; i >= 0; i--) {
    const step = list[i];
    if ([...step.added,...step.hidden,...(step.moved ? [step.moved.entry] : [])].some(e => e.cloud.id === id)) {
      list.splice(i,1); release(step.added);
    }
  }
  renderButtons();
}
onHistoryPolicy(enforceBudget);

/**
 * Record an operation that has been done: `added` clouds are already in the
 * list; `hide` sources are hidden now (so undo can show them again).
 */
export function record(step: {
  label: string;
  added?: Entry[];
  hide?: Entry[];
  moved?: Step["moved"];
}): void {
  const hidden = (step.hide ?? []).filter((e) => e.visible);
  for (const e of hidden) hideEntry(e);
  done.push({ label: step.label, added: step.added ?? [], hidden, moved: step.moved });
  for (const s of undone.splice(0)) release(s.added);
  enforceBudget();
  renderButtons();
}

async function move(entry: Entry, matrix: number[]): Promise<void> {
  replaceCloud(entry, await transformCloud(entry.cloud.id, matrix));
}

/** Inverse of a row-major 4x4 rigid transform. */
export function invertRigid(m: number[]): number[] {
  const r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const t = [m[3], m[7], m[11]];
  const rt = (i: number, j: number) => r[j * 3 + i]; // transpose
  const ti = [0, 1, 2].map((i) => -(rt(i, 0) * t[0] + rt(i, 1) * t[1] + rt(i, 2) * t[2]));
  return [
    rt(0, 0), rt(0, 1), rt(0, 2), ti[0],
    rt(1, 0), rt(1, 1), rt(1, 2), ti[1],
    rt(2, 0), rt(2, 1), rt(2, 2), ti[2],
    0, 0, 0, 1,
  ];
}

/** Apply a rigid transform (row-major 4x4, original coordinates) to a cloud as an undoable step. */
export async function moveCloud(entry: Entry, matrix: number[], label: string): Promise<void> {
  await move(entry, matrix);
  entry.transforms.push(matrix);
  record({ label, moved: { entry, matrix } });
}

export async function undo(): Promise<void> {
  const step = done.at(-1);
  if (!step || busy || taskActive()) return;
  busy = true;
  renderButtons();
  try {
    if (step.moved) {
      const { entry, matrix } = step.moved;
      if (entries.has(entry.cloud.id)) {
        await move(entry, invertRigid(matrix));
        entry.transforms.pop();
      }
    }
    for (const e of step.added) if (entries.has(e.cloud.id)) detach(e);
    for (const e of step.hidden) {
      if (!entries.has(e.cloud.id)) continue;
      e.visible = true;
      viewer.setVisible(e.cloud.id, true);
    }
    undone.push(done.pop()!);
    enforceBudget();
    renderList();
    setStatus(`Undid ${step.label}`);
  } catch (err) {
    setStatus(`Undo failed: ${errorText(err)}`, true);
  } finally {
    busy = false;
    enforceBudget();
    renderButtons();
  }
}

export async function redo(): Promise<void> {
  const step = undone.at(-1);
  if (!step || busy || taskActive()) return;
  busy = true;
  renderButtons();
  try {
    for (const e of step.added) attach(e);
    for (const e of step.hidden) if (entries.has(e.cloud.id)) hideEntry(e);
    if (step.moved && entries.has(step.moved.entry.cloud.id)) {
      const { entry, matrix } = step.moved;
      await move(entry, matrix);
      entry.transforms.push(matrix);
    }
    done.push(undone.pop()!);
    renderList();
    setStatus(`Redid ${step.label}`);
  } catch (err) {
    setStatus(`Redo failed: ${errorText(err)}`, true);
  } finally {
    busy = false;
    enforceBudget();
    renderButtons();
  }
}

/** The step undo would revert, if any. */
export function lastStep(): Readonly<Step> | undefined {
  return done.at(-1);
}

function renderButtons(): void {
  const [last, next] = [done.at(-1), undone.at(-1)];
  undoButton.disabled = busy || !last;
  redoButton.disabled = busy || !next;
  undoButton.title = last ? `Undo ${last.label} (Ctrl+Z)` : "Nothing to undo";
  redoButton.title = next ? `Redo ${next.label} (Ctrl+Shift+Z)` : "Nothing to redo";
}

undoButton.onclick = () => void undo();
redoButton.onclick = () => void redo();
window.addEventListener("keydown", (e) => {
  if (typing(e) || !(e.ctrlKey || e.metaKey)) return;
  const key = e.key.toLowerCase();
  if (key === "z" && !e.shiftKey) void undo();
  else if ((key === "z" && e.shiftKey) || key === "y") void redo();
  else return;
  e.preventDefault();
});
renderButtons();
