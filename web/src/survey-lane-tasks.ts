/** Explicit human follow-up records. These never mutate Lanelet2 or certify a map update. */
import { CHANGE_DECISIONS, changeCandidates, type ChangeCandidate, type SurveyReview } from "./change-review.ts";
import { overlappingSurveyLanes, type SurveyLane } from "./lane-surface-overlap.ts";
export const LANE_TASK_STATES = { "needs-review": "Needs review", "update-needed": "Update needed", "no-update": "No update needed", deferred: "Deferred" } as const;
export const LANE_TASK_BYTES = 16 * 1024 * 1024;
export const LANE_TASK_LIMIT = 1000;
export const LANE_TASK_VERTICES = 100_000;
export interface LaneCoreSnapshot {
  protocol: "xy-lane-z-v1";
  status: "complete" | "unavailable";
  total: number;
  supported: number | null;
  heightMarginM: number;
  reason: string | null;
}
export interface SurveyLaneTask {
  id: number;
  createdAt: string;
  updatedAt: string;
  status: keyof typeof LANE_TASK_STATES;
  note: string;
  staleReason: string | null;
  comparison: { analyzedAt: string; graph: string; identity: SurveyReview["identity"] };
  candidate: ChangeCandidate;
  lane: { id: number; left: [number, number, number][]; right: [number, number, number][] };
  mapSha256: string | null;
  coreSupport: LaneCoreSnapshot | null;
  overlap: { xyMargin: number; zMargin: number; checked: number; invalid: number; limited: number; total: number };
}
export const laneTaskScope = "Human follow-up records for explicitly selected overlapping lanes. Status does not change map geometry, native lane reviews or survey judgments. Historical files do not bind to current lane IDs. Content hashes identify snapshots, not reviewer identity or authenticity.";
function fail(): never { throw new Error("Invalid or inconsistent survey lane tasks JSON"); }
function obj(value: unknown): Record<string, unknown> { if (!value || typeof value !== "object" || Array.isArray(value)) fail(); return value as Record<string, unknown>; }
function str(value: unknown, max = 2000): string {
  if (typeof value !== "string" || value.length > max || /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(value)) fail(); return value;
}
function num(value: unknown, min = -Infinity): number { if (typeof value !== "number" || !Number.isFinite(value) || value < min) fail(); return value; }
function int(value: unknown, min = 0): number { const n = num(value, min); if (!Number.isSafeInteger(n)) fail(); return n; }
function time(value: unknown): string { const s = str(value, 40); if (!Number.isFinite(Date.parse(s)) || new Date(s).toISOString() !== s) fail(); return s; }
function sha(value: unknown): string | null { if (value === null) return null; if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) fail(); return value; }
function arr(value: unknown): unknown[] { if (!Array.isArray(value)) fail(); return value; }
export function laneTaskCounts(tasks: readonly SurveyLaneTask[]) {
  const counts = { "needs-review": 0, "update-needed": 0, "no-update": 0, deferred: 0 };
  for (const task of tasks) counts[task.status]++;
  return counts;
}
export function setLaneTaskDecision(task: SurveyLaneTask, status: string, note: string): void {
  if (task.staleReason) throw new Error("Historical lane tasks cannot change decisions");
  if (!Object.hasOwn(LANE_TASK_STATES, status)) throw new Error("Choose a valid lane-task state");
  if (typeof note !== "string" || note.length > 2000 || /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(note)) throw new Error("Task notes allow up to 2,000 characters without control characters");
  task.status = status as SurveyLaneTask["status"]; task.note = note; task.updatedAt = new Date().toISOString();
}
export function newLaneTask(id: number, review: SurveyReview, candidate: ChangeCandidate, lane: SurveyLane,
  overlap: SurveyLaneTask["overlap"], mapSha256: string | null, coreSupport: LaneCoreSnapshot | null = null): SurveyLaneTask {
  if (review.staleReason) throw new Error("Historical comparisons cannot create current lane tasks");
  const now = new Date().toISOString();
  return { id, createdAt: now, updatedAt: now, status: "needs-review", note: "", staleReason: null,
    comparison: structuredClone({ analyzedAt: review.analyzedAt, graph: review.context.graph, identity: review.identity }),
    candidate: structuredClone(candidate), lane: { id: lane.id, left: lane.left.map(p => [...p]), right: lane.right.map(p => [...p]) },
    mapSha256, overlap: { ...overlap }, coreSupport: structuredClone(coreSupport) };
}
export function laneTaskFile(tasks: readonly SurveyLaneTask[]) {
  return { schema: "cloudanalyzer.survey_lane_tasks.v1", scope: laneTaskScope, exportedAt: new Date().toISOString(), counts: laneTaskCounts(tasks), tasks };
}
export function parseLaneTaskFile(value: unknown): SurveyLaneTask[] {
  const v = obj(value);
  if (v.schema !== "cloudanalyzer.survey_lane_tasks.v1") fail();
  time(v.exportedAt); str(v.scope, 4096);
  const rows = arr(v.tasks); if (rows.length > LANE_TASK_LIMIT) fail();
  const ids = new Set<number>(); let vertices = 0;
  const parsed = rows.map((value): SurveyLaneTask => {
    const r = obj(value), c = obj(r.candidate), l = obj(r.lane), o = obj(r.overlap), comparison = obj(r.comparison);
    const id = int(r.id, 1); if (ids.has(id)) fail(); ids.add(id);
    const leftRaw = arr(l.left), rightRaw = arr(l.right);
    vertices += leftRaw.length + rightRaw.length;
    if (vertices > LANE_TASK_VERTICES || leftRaw.length < 2 || rightRaw.length < 2) fail();
    const curve = (raw: unknown[]) => raw.map(value => { const p = arr(value); if (p.length !== 3) fail(); return p.map(v => num(v)) as [number, number, number]; });
    const left = curve(leftRaw), right = curve(rightRaw);
    const bounds = arr(c.bounds).map(v => num(v)), centroid = arr(c.centroid).map(v => num(v));
    if (bounds.length !== 6 || centroid.length !== 3 || !Object.hasOwn(CHANGE_DECISIONS, str(c.decision, 32))) fail();
    const geometry = changeCandidates(new Float64Array([int(c.points, 1), ...centroid, ...bounds, num(c.meanChange)]))[0];
    if (centroid.some((n, a) => n < bounds[a] - 1e-9 * Math.max(1, Math.abs(bounds[a])) || n > bounds[a + 3] + 1e-9 * Math.max(1, Math.abs(bounds[a + 3])))) fail();
    const candidate: ChangeCandidate = { ...geometry, rank: int(c.rank, 1), decision: str(c.decision, 32) as ChangeCandidate["decision"],
      note: str(c.note), updatedAt: c.updatedAt === null ? null : time(c.updatedAt) };
    if (candidate.updatedAt === null && (candidate.decision !== "unreviewed" || candidate.note !== "")) fail();
    const lane = { id: int(l.id), left, right };
    const overlap = { xyMargin: num(o.xyMargin, 0), zMargin: num(o.zMargin, 0), checked: int(o.checked, 1), invalid: int(o.invalid), limited: int(o.limited), total: int(o.total, 1) };
    if (overlap.checked + overlap.invalid + overlap.limited !== overlap.total ||
        !overlappingSurveyLanes([lane], bounds, overlap.xyMargin, overlap.zMargin).ids.includes(lane.id)) fail();
    let coreSupport: LaneCoreSnapshot | null = null;
    // Original v1 records had no core evidence. Absence is unknown, never zero.
    if (r.coreSupport !== undefined && r.coreSupport !== null) {
      const e = obj(r.coreSupport), total = int(e.total, 1), heightMarginM = num(e.heightMarginM, 0);
      if (e.protocol !== "xy-lane-z-v1" || total !== candidate.points || heightMarginM !== overlap.zMargin) fail();
      if (e.status === "complete") {
        const supported = int(e.supported);
        if (supported > total || e.reason !== null) fail();
        coreSupport = { protocol: e.protocol, status: e.status, total, supported, heightMarginM, reason: null };
      } else if (e.status === "unavailable") {
        const reason = str(e.reason, 1000);
        if (!reason.trim() || e.supported !== null) fail();
        coreSupport = { protocol: e.protocol, status: e.status, total, supported: null, heightMarginM, reason };
      } else fail();
    }
    let identity: SurveyReview["identity"] = null;
    if (comparison.identity !== null) {
      const i = obj(comparison.identity), digest = sha(i.digest);
      if (i.algorithm !== "sha256-ply-graph-v1" || digest === null) fail(); identity = { algorithm: i.algorithm, digest };
    }
    const status = str(r.status, 32); if (!Object.hasOwn(LANE_TASK_STATES, status)) fail();
    const createdAt = time(r.createdAt), updatedAt = time(r.updatedAt);
    if (Date.parse(updatedAt) < Date.parse(createdAt)) fail();
    return { id, createdAt, updatedAt, status: status as SurveyLaneTask["status"], note: str(r.note), staleReason: r.staleReason === null ? null : str(r.staleReason),
      comparison: { graph: str(comparison.graph), analyzedAt: time(comparison.analyzedAt), identity }, candidate, lane,
      mapSha256: sha(r.mapSha256), overlap, coreSupport };
  });
  const counts = obj(v.counts);
  if (Object.entries(laneTaskCounts(parsed)).some(([key, value]) => counts[key] !== value)) fail();
  return parsed;
}

/** Text-only rendering shared by the current queue and historical files. */
export function laneCoreSnapshotText(evidence: LaneCoreSnapshot | null): string {
  if (!evidence) return "Core support not recorded (legacy or absent evidence).";
  const settings = `Original float64 XYZ; XY lane inclusion, local Z margin ${evidence.heightMarginM} m; ${evidence.protocol}.`;
  return evidence.status === "complete"
    ? `Saved core support: ${evidence.supported} / ${evidence.total} changed cores${evidence.supported === 0 ? " · box overlap only" : ""}. ${settings}`
    : `Saved core support unavailable: ${evidence.reason}. ${settings}`;
}
