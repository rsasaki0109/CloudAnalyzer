/** Read-only historical records. Parsing checks consistency, not source authenticity. */
import { editDecision, emptyDecision, reviewCounts, candidateReviewsCsv, candidateReviewScope, type CandidateDecision } from "./candidate-review.ts";
import { validateRegionSettings, validateExpectedOverlap, REGION_LIST_LIMIT, REGION_INPUT_BYTES, REGION_CELL_LIMIT, type DistanceRegions, type DistanceRegion } from "./distance-regions.ts";
export const REVIEW_ARCHIVE_BYTES = 2 * 1024 * 1024;
export interface ArchiveInput { id: number; name: string; loadedPoints: number; filePoints: number; keepEvery: number; preview: boolean; copcLevels: number | null }
export interface ArchiveComparison {
  direction: "forward" | "reverse"; source: ArchiveInput; reference: ArchiveInput; analysis: DistanceRegions; reusedDistances: boolean;
  review: { analyzedAt: string; counts: ReturnType<typeof reviewCounts>; candidates: (DistanceRegion & CandidateDecision)[] };
}
export interface ReviewArchive { schema: "cloudanalyzer.distance_candidate_reviews.v1"; comparisons: ArchiveComparison[] }
function fail(): never { throw new Error("Invalid or inconsistent candidate review archive"); }
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail();
  return value as Record<string, unknown>;
}
function integer(value: unknown): number { if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) fail(); return value; }
function finite(value: unknown): number { if (typeof value !== "number" || !Number.isFinite(value)) fail(); return value; }
function text(value: unknown, limit: number): string { if (typeof value !== "string" || value.length > limit) fail(); return value; }
function input(value: unknown): ArchiveInput {
  const v = object(value);
  const result = { id: integer(v.id), name: text(v.name, 1024), loadedPoints: integer(v.loadedPoints), filePoints: integer(v.filePoints),
    keepEvery: integer(v.keepEvery), preview: v.preview, copcLevels: v.copcLevels === null ? null : integer(v.copcLevels) };
  if (typeof result.preview !== "boolean" || result.keepEvery < 1 || result.filePoints < result.loadedPoints) fail();
  return result as ArchiveInput;
}
const countKeys = ["points", "measured", "invalidPositions", "unmeasured", "outsideOverlap", "exceeding", "occupiedCells", "components", "eligibleRegions", "smallRegionPoints", "unlistedEligiblePoints"] as const;
function region(value: unknown, rank: number, minimum: number, cutoff: number): DistanceRegion {
  const r = object(value), b = r.bounds;
  if (integer(r.rank) !== rank || !Array.isArray(b) || b.length !== 6) fail();
  const bounds = b.map(finite) as DistanceRegion["bounds"];
  const points = integer(r.points), maxDistance = finite(r.maxDistance), meanDistance = finite(r.meanDistance);
  if (points < minimum || maxDistance <= cutoff || meanDistance <= cutoff || meanDistance > maxDistance + Math.abs(maxDistance) * 1e-12 || bounds.slice(0, 3).some((v, i) => v > bounds[i + 3])) fail();
  return { rank, bounds, points, maxDistance, meanDistance };
}
function analysis(value: unknown): DistanceRegions {
  const v = object(value), threshold = finite(v.threshold), cell = finite(v.cell), minPoints = integer(v.minPoints);
  const effectiveThreshold = validateRegionSettings(threshold, cell, minPoints);
  if (v.effectiveThreshold !== effectiveThreshold) fail();
  const overlap = validateExpectedOverlap(v.overlap as DistanceRegions["overlap"]);
  const counts = Object.fromEntries(countKeys.map(key => [key, integer(v[key])])) as Pick<DistanceRegions, typeof countKeys[number]>;
  if (!Array.isArray(v.regions) || v.regions.length !== Math.min(counts.eligibleRegions, REGION_LIST_LIMIT)) fail();
  const regions = v.regions.map((r, i) => region(r, i + 1, minPoints, effectiveThreshold));
  const smallComponents = counts.components - counts.eligibleRegions;
  if (counts.points < 1 || counts.points > REGION_INPUT_BYTES / 16 || counts.occupiedCells > REGION_CELL_LIMIT ||
      counts.points !== counts.measured + counts.invalidPositions + counts.unmeasured + counts.outsideOverlap ||
      counts.exceeding > counts.measured || counts.occupiedCells > counts.exceeding || counts.components > counts.occupiedCells || counts.eligibleRegions > counts.components ||
      regions.reduce((sum, r) => sum + r.points, 0) + counts.smallRegionPoints + counts.unlistedEligiblePoints !== counts.exceeding ||
      counts.unlistedEligiblePoints < (counts.eligibleRegions - regions.length) * minPoints ||
      counts.smallRegionPoints < smallComponents || counts.smallRegionPoints > smallComponents * (minPoints - 1) ||
      (counts.eligibleRegions <= REGION_LIST_LIMIT && counts.unlistedEligiblePoints !== 0)) fail();
  return { threshold, effectiveThreshold, cell, minPoints, overlap, ...counts, regions };
}
export function parseReviewArchive(value: unknown): ReviewArchive {
  const archive = object(value);
  if (archive.schema !== "cloudanalyzer.distance_candidate_reviews.v1" || !Array.isArray(archive.comparisons) || archive.comparisons.length < 1 || archive.comparisons.length > 2) fail();
  const comparisons = archive.comparisons.map((value): ArchiveComparison => {
    const c = object(value), source = input(c.source), reference = input(c.reference), result = analysis(c.analysis), review = object(c.review);
    if (!["forward", "reverse"].includes(c.direction as string) || typeof c.reusedDistances !== "boolean" || source.id === reference.id || source.loadedPoints !== result.points ||
        !Array.isArray(review.candidates) || review.candidates.length !== result.regions.length) fail();
    const analyzedAt = text(review.analyzedAt, 40);
    editDecision("unreviewed", "", analyzedAt); // Reuse the canonical ISO timestamp check.
    const candidates = review.candidates.map((value, i) => {
      const r = object(value), geometry = region(r, i + 1, result.minPoints, result.effectiveThreshold), expected = result.regions[i];
      if (JSON.stringify(geometry) !== JSON.stringify(expected)) fail();
      let decision: CandidateDecision;
      if (r.updatedAt === null) {
        if (r.status !== "unreviewed" || r.note !== "") fail();
        decision = emptyDecision();
      } else decision = editDecision(text(r.status, 20), text(r.note, 2000), text(r.updatedAt, 40));
      return { ...geometry, ...decision };
    });
    const counts = reviewCounts(candidates), declared = object(review.counts);
    if (Object.entries(counts).some(([key, value]) => declared[key] !== value)) fail();
    return { direction: c.direction as ArchiveComparison["direction"], source, reference, analysis: result, reusedDistances: c.reusedDistances,
      review: { analyzedAt, counts, candidates } };
  });
  const forward = comparisons.find(c => c.direction === "forward"), reverse = comparisons.find(c => c.direction === "reverse");
  if (!forward || comparisons.length === 2 && (!reverse || JSON.stringify(forward.source) !== JSON.stringify(reverse.reference) || JSON.stringify(forward.reference) !== JSON.stringify(reverse.source) ||
      forward.analysis.threshold !== reverse.analysis.threshold || forward.analysis.cell !== reverse.analysis.cell || forward.analysis.minPoints !== reverse.analysis.minPoints || JSON.stringify(forward.analysis.overlap) !== JSON.stringify(reverse.analysis.overlap))) fail();
  return { schema: "cloudanalyzer.distance_candidate_reviews.v1", comparisons };
}
export function archiveCsv(archive: ReviewArchive): string {
  return candidateReviewsCsv(archive.comparisons.flatMap(c => c.review.candidates.map(r => ({ ...r, direction: c.direction,
    sourceId: c.source.id, source: c.source.name, referenceId: c.reference.id, reference: c.reference.name }))));
}
export const archiveScope = "Historical file only: no decisions applied to current clouds or QA. File structure and internal counts checked; source contents, reviewer identity and field accuracy are not independently verified. Units follow the original coordinates; no conversion. " + candidateReviewScope;
export function archiveSummary(c: ArchiveComparison): string {
  const a = c.analysis, counts = c.review.counts;
  return `${c.direction}: ${c.source.name} → ${c.reference.name}; analysis ${c.review.analyzedAt}. Threshold ${a.threshold} (effective f32 ${a.effectiveThreshold}), XY cell ${a.cell}, minimum ${a.minPoints}; overlap ${a.overlap ? "[" + a.overlap.join(", ") + "]" : "unrestricted"}. ${counts.reviewed} reviewed, ${counts.deferred} deferred, ${counts.unreviewed} unreviewed; ${a.eligibleRegions - a.regions.length} eligible regions unlisted, ${a.unlistedEligiblePoints} unlisted points, ${a.smallRegionPoints} below-minimum points. Loaded source #${c.source.id}: ${c.source.loadedPoints}/${c.source.filePoints}, keep every ${c.source.keepEvery}, preview ${c.source.preview}, COPC levels ${c.source.copcLevels ?? "n/a"}; reference #${c.reference.id}: ${c.reference.loadedPoints}/${c.reference.filePoints}, keep every ${c.reference.keepEvery}, preview ${c.reference.preview}, COPC levels ${c.reference.copcLevels ?? "n/a"}. Measured ${a.measured}, unmeasured ${a.unmeasured}, invalid ${a.invalidPositions}, outside overlap ${a.outsideOverlap}; exceeding ${a.exceeding}, occupied cells ${a.occupiedCells}, components ${a.components}, eligible regions ${a.eligibleRegions}; reused distances ${c.reusedDistances}.`;
}
export function archiveHtml(archive: ReviewArchive): string {
  const escape = (s: string) => s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Archived candidate reviews</title><style>body{font:16px/1.5 system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}section,li{border-top:1px solid #aaa;padding:1rem 0}p{white-space:pre-wrap;overflow-wrap:anywhere}h2{font-size:1.2rem}</style><h1>Archived candidate reviews</h1><p>${escape(archiveScope)}</p>` + archive.comparisons.map(c => `<section><h2>${escape(c.direction + ": " + c.source.name + " → " + c.reference.name)}</h2><p>${escape(archiveSummary(c))}</p><ol>${c.review.candidates.map(r => `<li><p>${escape(`Region ${r.rank}: ${r.status}; ${r.points} points; max ${r.maxDistance}; mean ${r.meanDistance}; world XYZ bounds [${r.bounds.join(", ")}]; updated ${r.updatedAt ?? "never"}`)}</p><p>${escape(r.note)}</p></li>`).join("")}</ol></section>`).join("") + "</html>";
}
