/** A cancellable long operation shown with a progress bar in the status bar. */

import { setMemoryListener } from "../api";
import type { Progress } from "../protocol";
import { $ } from "./dom";

let task: AbortController | null = null;
export function taskActive(): boolean { return task !== null; }

export function startTask(): AbortSignal {
  task = new AbortController();
  $("task").hidden = false;
  showProgress({ note: "" });
  return task.signal;
}

export function showProgress(p: Progress): void {
  const bar = $("progress-bar");
  const known = p.fraction !== undefined;
  bar.parentElement!.classList.toggle("indeterminate", !known);
  bar.style.width = known ? `${Math.round(p.fraction! * 100)}%` : "";
}

/** Hide the progress bar, unless a newer task has taken it over. */
export function endTask(signal: AbortSignal): void {
  if (task?.signal !== signal) return;
  task = null;
  $("task").hidden = true;
}

$<HTMLButtonElement>("task-cancel").onclick = () => task?.abort();

setMemoryListener((bytes,pool) => {
  $("memory").textContent = `WASM ${Math.round(bytes / 1e6)} MB + pool ${Math.round(pool / 1e6)} MB`;
});
