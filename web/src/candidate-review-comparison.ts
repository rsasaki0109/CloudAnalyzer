/** Compare decisions only when the complete archived diagnosis metadata agrees. */
import type { ArchiveComparison, ReviewArchive } from "./candidate-review-archive.ts";
import type { CandidateDecision } from "./candidate-review.ts";
import type { DistanceRegion } from "./distance-regions.ts";

export const reviewComparisonScope = "Read-only A → B review-record comparison; A/B are user-selected files, not inferred chronological order. Candidate correspondence requires identical direction, input descriptors, complete analysis settings/counts/geometry, analysis timestamp and distance-reuse provenance. This is agreement of archived metadata, not source-content authentication or proof of the same physical objects. No matching across different diagnoses by rank, name or spatial overlap. No decisions applied to live clouds or QA. Reviewed means inspection completed, not confirmed map change. Directions and unlisted candidates must not be summed as unique changes.";
export interface ReviewDecisionChange {
  rank: number; geometry: DistanceRegion; a: CandidateDecision; b: CandidateDecision;
  changed: ("status" | "note" | "updatedAt")[];
}
export interface ReviewDirectionComparison {
  direction: ArchiveComparison["direction"];
  compatible: boolean; reasons: string[];
  a: ArchiveComparison | null; b: ArchiveComparison | null;
  rows: ReviewDecisionChange[];
}
export interface ReviewComparison {
  schema: "cloudanalyzer.candidate_review_comparison.v1";
  comparedAt: string; scope: string;
  a: { fileName: string; archive: ReviewArchive }; b: { fileName: string; archive: ReviewArchive };
  directions: ReviewDirectionComparison[];
}
const decision = (r: CandidateDecision): CandidateDecision => ({ status: r.status, note: r.note, updatedAt: r.updatedAt });
export function compareReviewArchives(a: ReviewArchive, b: ReviewArchive, aName: string, bName: string): ReviewComparison {
  // Inputs have already passed the canonical archive parser. Copy all exported
  // records so later callers cannot alter an existing comparison through aliases.
  const copies = structuredClone({ a, b });
  const directions = (["forward", "reverse"] as const).flatMap(direction => {
    const a = copies.a.comparisons.find(c => c.direction === direction) ?? null;
    const b = copies.b.comparisons.find(c => c.direction === direction) ?? null;
    if (!a && !b) return [];
    const reasons: string[] = [];
    if (!a || !b) reasons.push("Direction is absent from one archive");
    else {
      if (JSON.stringify(a.source) !== JSON.stringify(b.source) || JSON.stringify(a.reference) !== JSON.stringify(b.reference)) reasons.push("Input descriptors differ");
      if (a.review.analyzedAt !== b.review.analyzedAt) reasons.push("Analysis timestamps differ");
      if (JSON.stringify(a.analysis) !== JSON.stringify(b.analysis)) reasons.push("Analysis settings, counts or candidate geometry differ");
      if (a.reusedDistances !== b.reusedDistances) reasons.push("Distance-reuse provenance differs");
    }
    const compatible = reasons.length === 0;
    const rows: ReviewDecisionChange[] = compatible ? a!.review.candidates.map((r, i) => {
      const next = b!.review.candidates[i], { rank, bounds, points, maxDistance, meanDistance } = r;
      return { rank, geometry: { rank, bounds: [...bounds], points, maxDistance, meanDistance }, a: decision(r), b: decision(next),
        changed: (["status", "note", "updatedAt"] as const).filter(key => r[key] !== next[key]) };
    }) : [];
    return [{ direction, compatible, reasons, a, b, rows }];
  });
  return { schema: "cloudanalyzer.candidate_review_comparison.v1", comparedAt: new Date().toISOString(), scope: reviewComparisonScope,
    a: { fileName: aName, archive: copies.a }, b: { fileName: bName, archive: copies.b }, directions };
}
export function reviewComparisonSummary(c: ReviewDirectionComparison): string {
  if (!c.compatible) return `${c.direction}: Not comparable — ${c.reasons.join("; ")}. No candidate transitions inferred.`;
  const status = c.rows.filter(r => r.changed.includes("status")).length;
  const note = c.rows.filter(r => r.changed.includes("note")).length;
  const timeOnly = c.rows.filter(r => r.changed.length === 1 && r.changed[0] === "updatedAt").length;
  const unchanged = c.rows.filter(r => !r.changed.length).length;
  return `${c.direction}: Archived analysis metadata agrees; ${c.rows.length} listed candidates, ${status} status changes, ${note} comment changes, ${timeOnly} timestamp-only changes, ${unchanged} unchanged. Status/comment counts may overlap.`;
}
export function reviewComparisonCsv(comparison: ReviewComparison): string {
  const quote = (s: string) => '"' + (/^\s*[=+\-@]/.test(s) ? "'" + s : s).replaceAll('"', '""') + '"';
  const lines = ["archive_a,archive_b,direction,comparable,reason,region_rank,changed_fields,status_a,status_b,comment_a,comment_b,updated_a,updated_b,points,max_distance,mean_distance,min_x,min_y,min_z,max_x,max_y,max_z"];
  for (const d of comparison.directions) {
    const prefix = [quote(comparison.a.fileName), quote(comparison.b.fileName), d.direction, d.compatible, quote(d.reasons.join("; "))];
    // Keep unmatched and empty directions visible without inventing candidate rows.
    if (!d.rows.length) lines.push([...prefix, ...Array<string>(17).fill("")].join(","));
    for (const r of d.rows) lines.push([...prefix, r.rank, quote(r.changed.join("; ")), r.a.status, r.b.status, quote(r.a.note), quote(r.b.note),
      r.a.updatedAt ?? "", r.b.updatedAt ?? "", r.geometry.points, r.geometry.maxDistance, r.geometry.meanDistance, ...r.geometry.bounds].join(","));
  }
  return lines.join("\n") + "\n";
}
export function reviewComparisonHtml(comparison: ReviewComparison): string {
  const escape = (s: unknown) => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>Archived review comparison</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}p,pre{white-space:pre-wrap;overflow-wrap:anywhere}li,section{border-top:1px solid #aaa;padding:.5rem 0}pre{font-size:12px}@media print{details{display:none}}</style></head><body><h1>Archived review comparison</h1><p>A: ${escape(comparison.a.fileName)}\nB: ${escape(comparison.b.fileName)}\nCompared: ${escape(comparison.comparedAt)} (comparison time, not acquisition time)</p><p>${escape(comparison.scope)}</p>` + comparison.directions.map(d => `<section><h2>${escape(reviewComparisonSummary(d))}</h2>${[d.a, d.b].map((c, i) => `<p>${i ? "B" : "A"}: ${escape(c ? `${c.source.name} → ${c.reference.name}; analysis ${c.review.analyzedAt}; ${c.review.counts.reviewed} reviewed, ${c.review.counts.deferred} deferred, ${c.review.counts.unreviewed} unreviewed; ${c.analysis.eligibleRegions - c.analysis.regions.length} eligible regions unlisted` : "Direction absent")}</p>`).join("")}<ol>${d.rows.map(r => `<li><p>Region ${r.rank}: ${escape(r.a.status)} → ${escape(r.b.status)}; changed: ${escape(r.changed.join(", ") || "none")}. ${r.geometry.points} points; world XYZ [${escape(r.geometry.bounds.join(", "))}]</p><p>A comment: ${escape(r.a.note)}\nUpdated A: ${escape(r.a.updatedAt ?? "never")}</p><p>B comment: ${escape(r.b.note)}\nUpdated B: ${escape(r.b.updatedAt ?? "never")}</p></li>`).join("")}</ol></section>`).join("") + `<details><summary>Complete comparison JSON</summary><pre id="comparison-json">${escape(JSON.stringify(comparison, null, 2))}</pre></details></body></html>`;
}
