/** Human inspection records for one immutable discrepancy diagnosis. */
export const REVIEW_NOTE_LIMIT = 2000;
export type ReviewStatus = "unreviewed" | "reviewed" | "deferred";
export interface CandidateDecision { status: ReviewStatus; note: string; updatedAt: string | null }
export const candidateReviewScope = "Human inspection of listed distance candidates only. Reviewed means inspection completed, not confirmed map change, defect or missing coverage. Directions are independent and may describe the same physical feature; do not sum them as unique changes. Unlisted/below-minimum candidates remain outside this review. Decisions are bound to this analysis snapshot and are reset by rerunning or invalidating it.";
export function emptyDecision(): CandidateDecision { return { status: "unreviewed", note: "", updatedAt: null }; }
export function editDecision(status: string, note: string, updatedAt = new Date().toISOString()): CandidateDecision {
  if (!["unreviewed", "reviewed", "deferred"].includes(status)) throw new Error("Choose a valid candidate review state");
  if (typeof note !== "string" || note.length > REVIEW_NOTE_LIMIT || /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(note)) throw new Error("Review comments allow up to 2,000 characters without control characters");
  if (!Number.isFinite(Date.parse(updatedAt)) || new Date(updatedAt).toISOString() !== updatedAt) throw new Error("Review update time must be an ISO timestamp");
  return { status: status as ReviewStatus, note, updatedAt };
}
export function reviewCounts(decisions: readonly CandidateDecision[]): Record<ReviewStatus, number> {
  return decisions.reduce((counts, decision) => { counts[decision.status]++; return counts; }, { unreviewed: 0, reviewed: 0, deferred: 0 });
}
export interface CandidateReviewCsvRow extends CandidateDecision {
  direction: "forward" | "reverse"; sourceId: number; source: string; referenceId: number; reference: string;
  rank: number; points: number; maxDistance: number; meanDistance: number; bounds: readonly number[];
}
export function candidateReviewsCsv(rows: readonly CandidateReviewCsvRow[]): string {
  const quoted = (s: string) => '"' + (/^[\s]*[=+\-@]/.test(s) ? "'" + s : s).replaceAll('"', '""') + '"';
  const lines = ["direction,source_id,source_name,reference_id,reference_name,region_rank,status,comment,updated_at,points,max_distance,mean_distance,min_x,min_y,min_z,max_x,max_y,max_z"];
  for (const r of rows) lines.push([r.direction, r.sourceId, quoted(r.source), r.referenceId, quoted(r.reference), r.rank,
    r.status, quoted(r.note), r.updatedAt ?? "", r.points, r.maxDistance, r.meanDistance, ...r.bounds].join(","));
  return lines.join("\n") + "\n";
}
