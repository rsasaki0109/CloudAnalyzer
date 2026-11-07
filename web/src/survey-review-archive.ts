/** Survey review files: validate historical records and explicitly resume matching evidence. */
import { CHANGE_DECISIONS, CHANGE_NOTE_LIMIT, changeDecisionCounts, type ChangeCandidate, type SurveyReview } from "./change-review.ts";
import type { PoseGraphState } from "./protocol";

export const SURVEY_REVIEW_BYTES = 16 * 1024 * 1024;
export const SURVEY_CANDIDATE_LIMIT = 100_000;
function fail(message = "Invalid or inconsistent survey review JSON"): never { throw new Error(message); }
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail();
  return value as Record<string, unknown>;
}
function text(value: unknown, limit = 2048): string {
  if (typeof value !== "string" || value.length > limit || /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(value)) fail();
  return value;
}
function finite(value: unknown, minimum = -Infinity): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) fail();
  return value;
}
function integer(value: unknown, minimum = 0): number {
  const n = finite(value, minimum); if (!Number.isSafeInteger(n)) fail(); return n;
}
function stamp(value: unknown): string {
  const s = text(value, 40);
  if (!Number.isFinite(Date.parse(s)) || new Date(s).toISOString() !== s) fail();
  return s;
}
function array(value: unknown, max = 1_000_000): unknown[] {
  if (!Array.isArray(value) || value.length > max) fail(); return value;
}
function numbers(value: unknown, length: number): number[] {
  const a = array(value, length); if (a.length !== length) fail(); return a.map(v => finite(v));
}
function near(a: number, b: number): boolean { return Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b)); }

export function parseSurveyReview(value: unknown): SurveyReview {
  const v = object(value);
  if (v.schema !== "cloudanalyzer.survey_change_review.v1") fail("Unsupported survey review schema");
  const c = object(v.context), m = object(c.m3c2), k = object(c.clustering);
  const nodeIds = (value: unknown) => {
    const ids = array(value).map(v => integer(v, -Number.MAX_SAFE_INTEGER));
    if (new Set(ids).size !== ids.length) fail(); return ids;
  };
  const context: SurveyReview["context"] = {
    graph: text(c.graph), splitNode: integer(c.splitNode, -Number.MAX_SAFE_INTEGER),
    sessions: array(c.sessions, 100_000).map(value => {
      const s = object(value); return { name: text(s.name), first: integer(s.first), count: integer(s.count, 1) };
    }),
    firstNodes: nodeIds(c.firstNodes), secondNodes: nodeIds(c.secondNodes),
    mapVoxel: finite(c.mapVoxel, Number.MIN_VALUE), reach: finite(c.reach, Number.MIN_VALUE),
    m3c2: { normalRadius: finite(m.normalRadius, Number.MIN_VALUE), projectionRadius: finite(m.projectionRadius, Number.MIN_VALUE),
      maxDepth: finite(m.maxDepth, Number.MIN_VALUE), coreSpacing: finite(m.coreSpacing, Number.MIN_VALUE) },
    clustering: { minChange: finite(k.minChange, 0), link: finite(k.link, Number.MIN_VALUE), minPoints: integer(k.minPoints, 1) },
  };
  const first = new Set(context.firstNodes);
  if (!context.firstNodes.length || !context.secondNodes.length || context.secondNodes.some(id => first.has(id))) fail();
  const source = (value: unknown) => {
    const s = object(value); return { id: integer(s.id), name: text(s.name), points: integer(s.points) };
  };
  const sources = object(v.sources);
  const inputs = { before: source(sources.before), after: source(sources.after), difference: source(sources.difference) };
  if (new Set(Object.values(inputs).map(s => s.id)).size !== 3) fail();
  const d = object(v.diagnostics);
  const loops = array(d.crossPartLoops).map(value => {
    const l = object(value); return { from: integer(l.from, -Number.MAX_SAFE_INTEGER), to: integer(l.to, -Number.MAX_SAFE_INTEGER),
      weightedResidual: l.weightedResidual === null ? null : finite(l.weightedResidual, 0) };
  });
  const valid = loops.flatMap(l => l.weightedResidual === null ? [] : [l.weightedResidual]);
  const diagnostics: SurveyReview["diagnostics"] = {
    crossPartLoops: loops, residualsAvailable: integer(d.residualsAvailable),
    weightedRms: d.weightedRms === null ? null : finite(d.weightedRms, 0),
    weightedMax: d.weightedMax === null ? null : finite(d.weightedMax, 0),
    corePoints: integer(d.corePoints), measured: integer(d.measured), unmeasured: integer(d.unmeasured),
    measuredFraction: d.measuredFraction === null ? null : finite(d.measuredFraction, 0),
  };
  if (diagnostics.corePoints !== inputs.difference.points || diagnostics.measured + diagnostics.unmeasured !== diagnostics.corePoints ||
      diagnostics.residualsAvailable !== valid.length) fail();
  if (valid.length) {
    if (diagnostics.weightedRms === null || diagnostics.weightedMax === null ||
        !near(diagnostics.weightedRms, Math.sqrt(valid.reduce((s, n) => s + n * n / valid.length, 0))) ||
        !near(diagnostics.weightedMax, valid.reduce((max, n) => Math.max(max, n), 0))) fail();
  } else if (diagnostics.weightedRms !== null || diagnostics.weightedMax !== null) fail();
  if (diagnostics.corePoints ? diagnostics.measuredFraction === null || !near(diagnostics.measuredFraction, diagnostics.measured / diagnostics.corePoints)
    : diagnostics.measuredFraction !== null) fail();
  const candidates = array(v.candidates, SURVEY_CANDIDATE_LIMIT).map((value, index): ChangeCandidate => {
    const r = object(value), bounds = numbers(r.bounds, 6), centroid = numbers(r.centroid, 3);
    if (integer(r.rank, 1) !== index + 1 || bounds.slice(0, 3).some((lo, a) => lo > bounds[a + 3] ||
        centroid[a] < lo && !near(centroid[a], lo) || centroid[a] > bounds[a + 3] && !near(centroid[a], bounds[a + 3]))) fail();
    const decision = text(r.decision, 32);
    if (!Object.hasOwn(CHANGE_DECISIONS, decision)) fail();
    const note = text(r.note, CHANGE_NOTE_LIMIT), updatedAt = r.updatedAt === null ? null : stamp(r.updatedAt);
    if (updatedAt === null && (decision !== "unreviewed" || note !== "")) fail();
    const meanChange = finite(r.meanChange);
    if (Math.abs(meanChange) < context.clustering.minChange && !near(Math.abs(meanChange), context.clustering.minChange)) fail();
    return { rank: index + 1, points: integer(r.points, context.clustering.minPoints), bounds, centroid,
      meanChange, decision: decision as ChangeCandidate["decision"], note, updatedAt };
  });
  if (candidates.reduce((n, r) => n + r.points, 0) > diagnostics.measured ||
      candidates.some((r, i) => i > 0 && r.points > candidates[i - 1].points)) fail();
  const counts = object(v.counts);
  if (Object.entries(changeDecisionCounts(candidates)).some(([key, value]) => counts[key] !== value)) fail();
  let identity: SurveyReview["identity"] = null;
  if (v.identity !== undefined && v.identity !== null) {
    const i = object(v.identity);
    if (i.algorithm !== "sha256-ply-graph-v1" || typeof i.digest !== "string" || !/^[0-9a-f]{64}$/.test(i.digest)) fail();
    identity = { algorithm: i.algorithm, digest: i.digest };
  }
  stamp(v.exportedAt);
  return { schema: "cloudanalyzer.survey_change_review.v1", scope: text(v.scope, 8192), analyzedAt: stamp(v.analyzedAt),
    staleReason: v.staleReason === null ? null : text(v.staleReason), identity, context, sources: inputs, diagnostics, candidates };
}

/** Canonical key order prevents JSON object ordering from changing evidence comparisons. */
function canonical(value: unknown): string {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value && typeof value === "object") return "{" + Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([k, v]) => JSON.stringify(k) + ":" + canonical(v)).join(",") + "}";
  return JSON.stringify(value);
}
function analysisEvidence(review: SurveyReview) {
  return { context: review.context, diagnostics: review.diagnostics,
    sources: Object.fromEntries(Object.entries(review.sources).map(([key, s]) => [key, { name: s.name, points: s.points }])),
    candidates: review.candidates.map(({ rank, points, bounds, centroid, meanChange }) => ({ rank, points, bounds, centroid, meanChange })) };
}

/** Hash complete PLY exports sequentially, retaining only one cloud export at a time. */
export async function surveyReviewIdentity(review: SurveyReview, graph: PoseGraphState,
  readPly: (id: number) => Promise<Uint8Array>): Promise<NonNullable<SurveyReview["identity"]>> {
  const hex = (bytes: ArrayBuffer) => [...new Uint8Array(bytes)].map(b => b.toString(16).padStart(2, "0")).join("");
  const digest = async (bytes: Uint8Array) => {
    const chunks: string[] = [];
    const chunkSize = 1024 * 1024;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      chunks.push(hex(await crypto.subtle.digest("SHA-256", new Uint8Array(bytes.subarray(i, i + chunkSize)))));
    }
    return { length: bytes.length, chunks };
  };
  const clouds = [];
  for (const source of Object.values(review.sources)) clouds.push(await digest(await readPly(source.id)));
  const arrays = {} as Record<string, Awaited<ReturnType<typeof digest>>>;
  for (const key of ["nodeIds", "poses", "edges", "edgeKinds", "edgeErrors", "fixed"] as const) {
    const a = graph[key]; arrays[key] = await digest(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  }
  const root = new TextEncoder().encode(canonical({ protocol: "sha256-ply-graph-v1", chunkSize: 1024 * 1024,
    analysis: analysisEvidence(review), clouds, graph: { arrays, planes: graph.planes, planeEdges: graph.planeEdges, gravityEdges: graph.gravityEdges } }));
  return { algorithm: "sha256-ply-graph-v1", digest: hex(await crypto.subtle.digest("SHA-256", root)) };
}

export function checkSurveyResume(current: SurveyReview, saved: SurveyReview): void {
  if (current.staleReason || saved.staleReason) fail("Historical reviews cannot resume decisions");
  if (!current.identity || !saved.identity) fail("This review has no content identity; it can be opened as history only");
  if (current.identity.algorithm !== saved.identity.algorithm || current.identity.digest !== saved.identity.digest ||
      canonical(analysisEvidence(current)) !== canonical(analysisEvidence(saved))) {
    fail("Saved review does not match the current comparison contents and settings");
  }
}

export function resumeSurveyDecisions(current: SurveyReview, saved: SurveyReview): ChangeCandidate[] {
  checkSurveyResume(current, saved);
  // Return a complete replacement only after all evidence matches. Never merge by candidate rank alone.
  return current.candidates.map((candidate, i) => ({ ...candidate, bounds: [...candidate.bounds], centroid: [...candidate.centroid], decision: saved.candidates[i].decision,
    note: saved.candidates[i].note, updatedAt: saved.candidates[i].updatedAt }));
}
