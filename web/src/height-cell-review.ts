/** Human notes bound to one bounded screened-cell result, never spatially matched. */
import { emptyDecision, editDecision, reviewCounts, type CandidateDecision } from "./candidate-review.ts";
import type { HeightReview } from "./protocol.ts";

export const heightCellReviewScope = "Human inspection of listed screened 2.5D grid cells only. Reviewed means inspection completed, not confirmed physical change, cause, accepted earthwork or survey accuracy. Unlisted, within-band, missing and excluded cells remain outside this review. Comparisons/overlapping areas may describe the same ground; do not sum review counts as unique changes. Session-only records are reset by rerunning or invalidating the parent analysis; no transfer to other analyses or projects.";
export interface HeightCellReviewContext {
  label: string; before: Record<string, unknown>; after: Record<string, unknown>;
  settings: { cell: number; height: string; fillEmpty: boolean; coordinateUnit: string; region?: readonly number[]; areaName?: string; heightThreshold?: number };
  support: { mode: string; sha256?: string; commonGridCells?: number };
}
export function startHeightCellReview(review: HeightReview, context: HeightCellReviewContext, openedAt = new Date().toISOString()) {
  // Copy bounded evidence and descriptors so exported notes keep their original context.
  return { openedAt, context: structuredClone(context), screening: structuredClone(review),
    decisions: (review.cells ?? []).map(() => emptyDecision()) };
}
export type HeightCellReviewSession = ReturnType<typeof startHeightCellReview>;
export function updateHeightCellReview(session: HeightCellReviewSession, rank: number, status: string, note: string, updatedAt?: string): CandidateDecision {
  if (!Number.isInteger(rank) || rank < 0 || rank >= session.decisions.length) throw new Error("Choose a listed screened cell");
  const decision = editDecision(status, note, updatedAt);
  session.decisions[rank] = decision;
  return decision;
}
export function heightCellReviewSnapshot(sessions: readonly HeightCellReviewSession[]) {
  return { schema: "cloudanalyzer.height_cell_reviews.v1", scope: heightCellReviewScope, comparisons: sessions.map(s => {
    const { cells, ...summary } = s.screening;
    return { openedAt: s.openedAt, ...structuredClone(s.context), screening: summary, counts: reviewCounts(s.decisions),
      cells: (cells ?? []).map((cell, i) => ({ listedRank: i + 1, ...cell, ...s.decisions[i] })) };
  }) };
}
export function heightCellReviewsCsv(sessions: readonly HeightCellReviewSession[]): string {
  const quote = (s: string) => '"' + (/^\s*[=+\-@]/.test(s) ? "'" + s : s).replaceAll('"', '""') + '"';
  const rows = ["comparison_index,comparison,before,after,coordinate_unit,height_threshold,cell_size,support_mode,common_mask_sha256,opened_at,listed_rank,grid_index,x,y,before_height,after_height,difference,status,comment,updated_at,unlisted_cells"];
  for (const [index, s] of sessions.entries()) for (const [i, c] of (s.screening.cells ?? []).entries()) {
    const d = s.decisions[i], context = s.context;
    rows.push([index + 1, quote(context.label), quote(JSON.stringify(context.before)), quote(JSON.stringify(context.after)),
      quote(context.settings.coordinateUnit), s.screening.threshold, context.settings.cell, quote(context.support.mode), context.support.sha256 ?? "", s.openedAt,
      i + 1, c.index, c.x, c.y, c.before, c.after, c.difference, d.status, quote(d.note), d.updatedAt ?? "", s.screening.unlistedCells ?? ""].join(","));
  }
  return rows.join("\n") + "\n";
}
