/** Review native inference files in the browser; the browser does not run AI. */
import * as THREE from "three";
import { discardCloud, loadCloud, workerBusy } from "../api";
import { readAiScanReview, type AiScanReview } from "../lidar-ai-review";
import { onProjectChanged, projectChanged } from "../project-change";
import type { LoadedCloud } from "../protocol";
import { refreshColors } from "./colors";
import { $, download, errorText, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { cloudHistoryReady, record, requireDerivedUndo } from "./history";
import { graphProjectReady } from "./posegraph";
import { entries, listChanged, pointsInvalidated, toRender, viewer, type Entry } from "./state";
import { endTask, showProgress, startTask, taskActive } from "./tasks";
import { mapProjectReady } from "./vectormap";

const open = $<HTMLButtonElement>("lidar-ai-open"), input = $<HTMLInputElement>("lidar-ai-files");
const view = $<HTMLSelectElement>("lidar-ai-view");
let running = false, revision = 0;
onProjectChanged(() => revision++);
let current: { review: AiScanReview; parts: { role: string; entry: Entry; cloud: LoadedCloud }[]; stale: boolean; present: boolean } | null = null;
function ready(): boolean {
  return !running && !taskActive() && cloudHistoryReady() && mapProjectReady() && graphProjectReady();
}
function update(): void {
  if (!current) return;
  if (current.parts.some(p => p.entry.cloud !== p.cloud)) current.stale = true;
  const present = current.parts.every(p => entries.get(p.cloud.id) === p.entry);
  if (present && !current.present && !current.stale) {
    const restored = current;
    // Redo's history step hides the original; restore the chosen comparison view
    // once that step has completed, including when "Original scan" was selected.
    queueMicrotask(() => { if (current === restored) showView(); });
  }
  current.present = present;
  view.disabled = current.stale || !present || running;
  $<HTMLButtonElement>("lidar-ai-focus").disabled = view.disabled || !current.parts.some(p => p.role === "candidates");
  $("lidar-ai-state").textContent = current.stale
    ? "Comparison clouds changed. Reopen the three files to review the original result. Downloads below remain the original verified records."
    : !present ? "Comparison closed or undone. Redo restores an undone comparison; reopen the files after removal."
    : "Gray: retained points. Red: AI candidates. Compare views before using the retained scan; class prediction alone does not establish motion.";
}
function showView(): void {
  update();
  if (!current || view.disabled || !ready()) return;
  const mode = view.value;
  for (const part of current.parts) {
    part.entry.visible = mode === "highlight" ? part.role !== "original" : part.role === mode;
    viewer.setVisible(part.cloud.id, part.entry.visible);
  }
  renderList();
  projectChanged();
}
view.onchange = showView;
$("lidar-ai-focus").onclick = () => {
  update();
  if (!current || view.disabled || !ready()) return;
  const cloud = current.parts.find(p => p.role === "candidates")?.cloud;
  if (!cloud) return;
  viewer.frameBox(new THREE.Box3(toRender(cloud.bounds.slice(0, 3)), toRender(cloud.bounds.slice(3, 6))).expandByScalar(2));
  projectChanged();
};
listChanged.add(update);
pointsInvalidated.add(update);
$("lidar-ai-download-retained").onclick = () => { if (current) download(current.review.retained, "retained.bin"); };
$("lidar-ai-download-candidates").onclick = () => { if (current) download(current.review.candidates, "candidates.bin"); };
open.onclick = () => {
  if (!ready()) return setStatus("Finish the current operation before opening AI results", true);
  input.click();
};
input.onchange = async () => {
  const files = [...input.files ?? []]; input.value = "";
  if (!files.length) return;
  if (!ready()) return setStatus("Finish the current operation before opening AI results", true);
  let expectedRevision = revision;
  const signal = startTask();
  running = true; open.disabled = true; update();
  const loaded: LoadedCloud[] = [];
  let committed = false;
  try {
    const waitingSince = performance.now();
    while (workerBusy()) {
      signal.throwIfAborted();
      if (performance.now() - waitingSince > 5000) throw new Error("Finish the current worker operation before opening AI results");
      await new Promise(resolve => setTimeout(resolve, 25));
    }
    setStatus("Verifying AI source, mask order and exact output records…");
    const review = await readAiScanReview(files, signal);
    // Keep invalid records in exact downloads; never pass NaNs to the octree renderer.
    if (review.report.projection.invalid) throw new Error("This browser comparison requires finite XYZI records; verify scans with invalid returns using the native verifier");
    const name = review.source.name.replace(/\.bin$/i, "");
    const names = new Set([...entries.values()].map(e => e.cloud.name));
    const plan = [
      { role: "original", file: review.source },
      { role: "retained", file: review.retained },
      { role: "candidates", file: review.candidates },
    ].filter(p => p.file.size > 0);
    for (const part of plan) {
      let displayName = `${name}_AI_${part.role}`, i = 2;
      while (names.has(displayName)) displayName = `${name}_AI_${part.role}_${i++}`;
      names.add(displayName);
      if (revision !== expectedRevision) throw new Error("Current work changed while opening AI results; choose the files again");
      // loadCloud synchronously emits its own project-change notification.
      const pending = loadCloud(part.file, Infinity, showProgress, signal, displayName);
      expectedRevision = revision;
      const cloud = await pending;
      loaded.push(cloud); signal.throwIfAborted();
      if (cloud.keepEvery !== 1 || cloud.count !== part.file.size / 16) throw new Error("AI comparison must load every original point record");
    }
    requireDerivedUndo(loaded, []);
    signal.throwIfAborted();
    if (revision !== expectedRevision || !cloudHistoryReady()) throw new Error("Current work changed while opening AI results; choose the files again");
    const parts = loaded.map((cloud, i) => {
      const role = plan[i].role;
      const entry = addEntry(cloud, role === "original" ? { kind: "file", file: review.source } : { kind: "derived" });
      entry.mode = "solid";
      entry.solid = role === "candidates" ? [255, 83, 83] : [185, 198, 212];
      refreshColors(entry);
      return { role, entry, cloud };
    });
    record({ label: "the AI scan comparison", added: parts.map(p => p.entry), hide: parts.filter(p => p.role === "original").map(p => p.entry) });
    committed = true;
    current = { review, parts, stale: false, present: true };
    view.value = "highlight";
    $("lidar-ai-result").hidden = false;
    const r = review.report;
    $("lidar-ai-summary").textContent = `${r.points.toLocaleString()} source points · ${r.removed.toLocaleString()} candidates · ${r.retained.toLocaleString()} retained. Target: ${r.targets.join(", ")}; score ≥ ${r.threshold.toFixed(3)}. ${r.unclassified.toLocaleString()} unclassified returns retained. File hashes and record order verified; detection accuracy is unmeasured.`;
    renderList(); viewer.fit(); projectChanged();
    setStatus("Opened AI scan comparison. Red marks candidates; Undo removes the comparison. Exact downloads preserve source records.");
  } catch (error) {
    setStatus("Could not open AI result: " + (signal.aborted ? "Cancelled" : errorText(error)), true);
  } finally {
    if (!committed) await Promise.all(loaded.map(cloud => discardCloud(cloud.id)));
    running = false; open.disabled = false; endTask(signal); update();
  }
};
