/** Evidence and human decisions for one two-pass M3C2 comparison. */
import type { PoseGraphState } from "./protocol";

export const CHANGE_DECISIONS = {
  unreviewed: "Unreviewed",
  change: "Actual change",
  misalignment: "Misalignment",
  insufficient: "Insufficient observations",
  deferred: "Deferred",
} as const;
export type ChangeDecision = keyof typeof CHANGE_DECISIONS;
export const CHANGE_NOTE_LIMIT = 2000;
export const changeReviewScope = "Human judgments about M3C2 candidates, not automatic change classifications. Weighted constraint residuals are not metres or independent registration accuracy. Finite M3C2 measurements do not establish full scene coverage; unmeasured areas and below-threshold/small changes remain outside the candidate list. Positive/negative distances follow reference normals, not added/removed object labels. This comparison is directional; newly visible surfaces may require a reverse comparison.";

export interface ChangeCandidate {
  rank: number;
  points: number;
  centroid: number[];
  bounds: number[];
  meanChange: number;
  decision: ChangeDecision;
  note: string;
  updatedAt: string | null;
}

/** Worker layout: count, centroid XYZ, minimum XYZ, maximum XYZ, mean distance. */
export function changeCandidates(flat: Float64Array): ChangeCandidate[] {
  if (flat.length % 11) throw new Error("Incomplete change-object evidence");
  const out: ChangeCandidate[] = [];
  for (let i = 0; i < flat.length; i += 11) {
    const o = Array.from(flat.subarray(i, i + 11));
    if (!o.every(Number.isFinite) || !Number.isSafeInteger(o[0]) || o[0] <= 0 ||
        [0, 1, 2].some(a => o[4 + a] > o[7 + a])) throw new Error("Invalid change-object evidence");
    out.push({ rank: i / 11 + 1, points: o[0], centroid: o.slice(1, 4), bounds: o.slice(4, 10),
      meanChange: o[10], decision: "unreviewed", note: "", updatedAt: null });
  }
  return out;
}

export function setChangeDecision(candidate: ChangeCandidate, decision: string, note: string): void {
  if (!Object.hasOwn(CHANGE_DECISIONS, decision)) throw new Error("Choose a valid change decision");
  if (note.length > CHANGE_NOTE_LIMIT || /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(note)) {
    throw new Error("Review notes allow up to 2,000 characters without control characters");
  }
  candidate.decision = decision as ChangeDecision;
  candidate.note = note;
  candidate.updatedAt = new Date().toISOString();
}

export function changeDecisionCounts(candidates: readonly ChangeCandidate[]): Record<ChangeDecision, number> {
  const counts: Record<ChangeDecision, number> = { unreviewed: 0, change: 0, misalignment: 0, insufficient: 0, deferred: 0 };
  for (const candidate of candidates) counts[candidate.decision]++;
  return counts;
}

export function comparisonDiagnostics(state: PoseGraphState, split: number, distances: Float32Array) {
  const links: { from: number; to: number; weightedResidual: number | null }[] = [];
  for (let i = 0; i < state.edgeKinds.length; i++) {
    const [a, b] = [state.edges[2 * i], state.edges[2 * i + 1]];
    if (state.edgeKinds[i] !== 1 || (a < split) === (b < split)) continue;
    const error = state.edgeErrors[i];
    links.push({ from: state.nodeIds[a], to: state.nodeIds[b],
      weightedResidual: Number.isFinite(error) && error >= 0 ? Math.sqrt(error) : null });
  }
  const valid = links.flatMap(l => l.weightedResidual === null ? [] : [l.weightedResidual]);
  let measured = 0;
  for (const d of distances) if (Number.isFinite(d)) measured++;
  return {
    crossPartLoops: links,
    residualsAvailable: valid.length,
    weightedRms: valid.length ? Math.sqrt(valid.reduce((sum, v) => sum + v * v / valid.length, 0)) : null,
    // Accumulate without spreading a potentially large graph into function arguments.
    weightedMax: valid.length ? valid.reduce((max, v) => Math.max(max, v), 0) : null,
    corePoints: distances.length, measured, unmeasured: distances.length - measured,
    measuredFraction: distances.length ? measured / distances.length : null,
  };
}

/** A copy, so a later in-place change cannot silently relabel old evidence. */
export function graphEvidence(state: PoseGraphState): PoseGraphState {
  return { ...state, nodeIds: state.nodeIds.slice(), poses: state.poses.slice(), edges: state.edges.slice(),
    edgeKinds: state.edgeKinds.slice(), edgeErrors: state.edgeErrors.slice(), fixed: state.fixed.slice() };
}
export function sameGraphEvidence(a: PoseGraphState, b: PoseGraphState): boolean {
  return (["nodeIds", "poses", "edges", "edgeKinds", "edgeErrors", "fixed"] as const).every(key =>
    a[key].length === b[key].length && a[key].every((v, i) => Object.is(v, b[key][i]))) &&
    a.planes === b.planes && a.planeEdges === b.planeEdges && a.gravityEdges === b.gravityEdges;
}

export interface ComparisonContext {
  graph: string;
  sessions: { name: string; first: number; count: number }[];
  splitNode: number;
  firstNodes: number[];
  secondNodes: number[];
  mapVoxel: number;
  reach: number;
  m3c2: { normalRadius: number; projectionRadius: number; maxDepth: number; coreSpacing: number };
  clustering: { minChange: number; link: number; minPoints: number };
}
export interface SurveyReview {
  schema: "cloudanalyzer.survey_change_review.v1";
  scope: string;
  analyzedAt: string;
  staleReason: string | null;
  identity: { algorithm: "sha256-ply-graph-v1"; digest: string } | null;
  context: ComparisonContext;
  sources: Record<"before" | "after" | "difference", { id: number; name: string; points: number }>;
  diagnostics: ReturnType<typeof comparisonDiagnostics>;
  candidates: ChangeCandidate[];
}

export type ChangeFilter = "all" | "decided" | ChangeDecision;
export function filterChangeCandidates(candidates: readonly ChangeCandidate[], filter: ChangeFilter): ChangeCandidate[] {
  return candidates.filter(candidate => filter === "all" || (filter === "decided"
    ? candidate.decision !== "unreviewed" : candidate.decision === filter));
}
/** Next unfinished record in original rank order, wrapping once and excluding the current one. */
export function nextUnreviewedCandidate(candidates: readonly ChangeCandidate[], selectedRank: number): ChangeCandidate | null {
  return candidates.find(candidate => candidate.rank > selectedRank && candidate.decision === "unreviewed")
    ?? candidates.find(candidate => candidate.rank < selectedRank && candidate.decision === "unreviewed") ?? null;
}
