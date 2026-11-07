/** Lane reviews: status, notes and the review list. */

import { projectChanged } from "../project-change";
import type { ReviewDraft } from "../recovery-store";
import { LaneReviews, reviewCsv, type LaneReview, type LaneReviewRow, type ReviewStatus } from "../lane-review";
import { $, download, setStatus } from "./dom";
import { busy, lowSupport, renderLane, selected, selectLane, view } from "./vectormap";

export const reviews = new LaneReviews();

export let reviewPage = 0;
export function setReviewPage(value: typeof reviewPage): void { reviewPage = value; }
export const REVIEW_PAGE_SIZE = 25;
const reviewLabels: Record<ReviewStatus, string> = { unreviewed: "Unreviewed", reviewed: "Reviewed", "needs-fix": "Needs fixes", deferred: "Deferred" };
export function laneSignatures(): Map<number, string> {
  return new Map(view.lanes.map(lane => {
    const rules = (view.regulatoryElements ?? []).filter(r => r.lanes.includes(lane.id));
    const featureIds = new Set(rules.flatMap(r => [...r.signals, ...r.stop_lines, ...r.controlled_crosswalks, ...(r.crosswalk === null ? [] : [r.crosswalk])]));
    return [lane.id, JSON.stringify([lane, view.boundaries.filter(b => b.id === lane.leftRef.id || b.id === lane.rightRef.id), rules, [...view.signals, ...view.stopLines, ...view.crosswalks].filter(f => featureIds.has(f.id))])];
  }));
}
export function captureLaneReviews(): LaneReview[] { return reviews.snapshot(); }
export function restoreLaneReviews(records: LaneReview[]): void { reviews.restore(records, laneSignatures()); reviewPage = 0; renderReviews(); renderLane(); }
export function captureReviewDraft(): ReviewDraft | undefined {
  if (selected === null) return;
  const status = $<HTMLSelectElement>("vm-review-state").value as ReviewStatus, notes = $<HTMLTextAreaElement>("vm-review-notes").value;
  const saved = reviews.get(selected);
  if (status !== (saved?.status ?? "unreviewed") || notes !== (saved?.notes ?? "")) return {lane:selected,status,notes};
}
export function restoreReviewDraft(draft: ReviewDraft | undefined): void {
  if (!draft || !view.lanes.some(l => l.id === draft.lane)) return;
  selectLane(draft.lane,true);
  $<HTMLSelectElement>("vm-review-state").value = draft.status;
  $<HTMLTextAreaElement>("vm-review-notes").value = draft.notes;
}
export function mapProjectReady(): boolean { return !busy; }
export function filteredReviews(): LaneReviewRow[] {
  const filter = $<HTMLSelectElement>("vm-review-filter").value;
  return reviews.rows(view.lanes.map(l => l.id)).filter(row => filter === "all" || (filter === "low-support" ? lowSupport.has(row.lane) : row.status === filter));
}
export function markReviewSelection(): void {
  for (const button of $("vm-review-rows").querySelectorAll<HTMLButtonElement>("button[data-lane]")) {
    const current = Number(button.dataset.lane) === selected;
    if (current) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
    button.closest("tr")!.classList.toggle("selected", current);
  }
}
export function renderReviews(): void {
  const counts = { unreviewed: 0, reviewed: 0, "needs-fix": 0, deferred: 0 };
  for (const lane of view.lanes) counts[reviews.get(lane.id)?.status ?? "unreviewed"]++;
  $("vm-review-summary").textContent = `${counts.unreviewed} unreviewed · ${counts.reviewed} reviewed · ${counts["needs-fix"]} need fixes · ${counts.deferred} deferred`;
  const rows = filteredReviews();
  reviewPage = Math.min(reviewPage, Math.max(0, Math.ceil(rows.length / REVIEW_PAGE_SIZE) - 1));
  const start = reviewPage * REVIEW_PAGE_SIZE;
  $("vm-review-page").textContent = rows.length ? `Showing ${start + 1}–${Math.min(start + REVIEW_PAGE_SIZE, rows.length)} of ${rows.length} matching lanes (${view.lanes.length} total).` : `0 matching lanes (${view.lanes.length} total).`;
  $("vm-review-empty").textContent = rows.length ? "" : $<HTMLSelectElement>("vm-review-filter").value === "low-support" ? "No low-coverage lanes listed. Check source coverage for the current map and cloud." : "No lanes match this review filter.";
  $("vm-review-rows").replaceChildren(...rows.slice(start, start + REVIEW_PAGE_SIZE).map(row => {
    const tr = document.createElement("tr"), lane = document.createElement("td"), state = document.createElement("td"), notes = document.createElement("td");
    const button = document.createElement("button");
    button.textContent = String(row.lane); button.dataset.lane = String(row.lane); button.setAttribute("aria-label", `View lane ${row.lane}`);
    button.onclick = () => selectLane(row.lane, true); lane.append(button);
    state.textContent = reviewLabels[row.status];
    if (row.staleReason) {
      const stale = document.createElement("span"); stale.className = "stale"; stale.textContent = "Review again"; stale.title = row.staleReason; state.append(stale);
    }
    notes.textContent = row.notes.length > 160 ? row.notes.slice(0, 160) + "…" : row.notes;
    notes.title = row.notes;
    tr.append(lane, state, notes); return tr;
  }));
  $<HTMLButtonElement>("vm-review-page-prev").disabled = reviewPage === 0;
  $<HTMLButtonElement>("vm-review-page-next").disabled = start + REVIEW_PAGE_SIZE >= rows.length;
  $<HTMLButtonElement>("vm-review-next").disabled = rows.length === 0;
  $<HTMLButtonElement>("vm-review-export").disabled = view.lanes.length === 0;
  markReviewSelection();
}
export function reviewSourceChanged(): void { reviews.invalidateSource(); renderReviews(); renderLane(); projectChanged(); }
$("vm-review-filter").onchange = () => { reviewPage = 0; renderReviews(); };
$("vm-review-page-prev").onclick = () => { reviewPage--; renderReviews(); };
$("vm-review-page-next").onclick = () => { reviewPage++; renderReviews(); };
$("vm-review-export").onclick = () => {
  const rows = reviews.rows(view.lanes.map(l => l.id));
  download(new Blob([reviewCsv(rows)], { type: "text/csv;charset=utf-8" }), "lane-reviews.csv");
  setStatus(`Exported saved reviews for all ${rows.length} lanes.`);
};
$("vm-review-next").onclick = () => {
  const lanes = filteredReviews();
  if (!lanes.length) return setStatus("No lanes match the review filter.");
  const index = lanes.findIndex(l => l.lane === selected);
  selectLane(lanes[(index + 1) % lanes.length].lane, true);
};
$("vm-review-save").onclick = () => {
  if (selected === null) return;
  reviews.save(selected, laneSignatures().get(selected)!, $<HTMLSelectElement>("vm-review-state").value as ReviewStatus, $<HTMLTextAreaElement>("vm-review-notes").value);
  renderReviews(); renderLane(); setStatus(`Review saved for lane ${selected}`);
  projectChanged();
};
