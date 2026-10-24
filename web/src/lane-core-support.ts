/** Individual changed-core evidence, in original float64 coordinates. No map-change classification. */
import { resampleLaneCurve, type SurveyLane } from "./lane-surface-overlap.ts";
export const CORE_SOURCE_LIMIT = 2_000_000;
export const CANDIDATE_CORE_LIMIT = 100_000;
export const CORE_TEST_LIMIT = 10_000_000;
export interface CandidateCores { positions: Float64Array; indices: Uint32Array; sourceCount: number }
export interface CoreSupportRow {
  lane: number;
  status: "complete" | "unavailable";
  total: number;
  supported: number | null;
  mask: Uint8Array | null;
  reason: string | null;
  tests: number;
}
export interface LaneCoreEvidence extends CandidateCores { rows: CoreSupportRow[] }
interface CoreSource {
  readonly length: number;
  attribute(name: string): Float32Array | undefined | null;
  positionsRange(start: number, end: number): Float64Array;
}
/** The label field is scanned once; source positions are read one bounded native block at a time. */
export function readCandidateCores(source: CoreSource, rank: number, expected: number, expectedBounds?: readonly number[]): CandidateCores {
  if (!Number.isSafeInteger(rank) || rank < 1 || !Number.isSafeInteger(expected) || expected < 1 || expected > CANDIDATE_CORE_LIMIT ||
      !Number.isSafeInteger(source.length) || source.length > CORE_SOURCE_LIMIT || source.length < 0) {
    throw new Error("Core evidence limit: at most 2,000,000 source cores and 100,000 cores per candidate");
  }
  const labels = source.attribute("change_object");
  if (!labels || labels.length !== source.length) throw new Error("Changed-core labels are unavailable or stale");
  const indices = new Uint32Array(expected);
  let count = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i] === rank) {
    if (count === expected) throw new Error("Changed-core count no longer matches this review");
    indices[count++] = i;
  }
  if (count !== expected) throw new Error("Changed-core count no longer matches this review");
  const positions = new Float64Array(expected * 3), blockSize = 4096;
  let cachedBlock = -1;
  let block: Float64Array = new Float64Array();
  const bounds = [Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity];
  for (let i = 0; i < count; i++) {
    const start = Math.floor(indices[i] / blockSize) * blockSize;
    if (start !== cachedBlock) {
      block = source.positionsRange(start, Math.min(start + blockSize, source.length)); cachedBlock = start;
    }
    const offset = (indices[i] - start) * 3;
    if (offset + 3 > block.length) throw new Error("Incomplete original-coordinate core block");
    for (let axis = 0; axis < 3; axis++) {
      const value = block[offset + axis];
      if (!Number.isFinite(value)) throw new Error("Changed-core coordinates are not finite");
      positions[i * 3 + axis] = value;
      bounds[axis] = Math.min(bounds[axis], value); bounds[axis+3] = Math.max(bounds[axis+3], value);
    }
  }
  if (expectedBounds && (expectedBounds.length !== 6 || !expectedBounds.every(Number.isFinite) || bounds.some((v,i)=>v!==expectedBounds[i]))) {
    throw new Error("Changed-core bounds no longer match this review");
  }
  return { positions, indices, sourceCount: source.length };
}

/** XY barycentric inclusion and local interpolated Z. XY box margins do not expand lane support. */
export function laneCoreSupport(lane: SurveyLane, positions: Float64Array, heightMargin: number, testBudget = CORE_TEST_LIMIT): CoreSupportRow {
  const total = positions.length / 3;
  const unavailable = (reason: string): CoreSupportRow => ({ lane: lane.id, status: "unavailable", total,
    supported: null, mask: null, reason, tests: 0 });
  if (positions.length % 3 || total > CANDIDATE_CORE_LIMIT || !positions.every(Number.isFinite) ||
      !Number.isFinite(heightMargin) || heightMargin < 0 || heightMargin > 10 || !Number.isSafeInteger(testBudget) || testBudget < 0) {
    return unavailable("Invalid original core coordinates, height margin or work budget");
  }
  const n = Math.max(lane.left.length, lane.right.length, 2), tests = total * 2 * (n - 1);
  if (lane.left.length < 2 || lane.right.length < 2 || lane.left.length + lane.right.length > 200_000 ||
      [...lane.left, ...lane.right].some(p => p.length !== 3 || !p.every(Number.isFinite))) return unavailable("Invalid or oversized lane geometry");
  if (tests > testBudget) return unavailable("Core/triangle test budget exceeded; evidence was not partially counted");
  const left = resampleLaneCurve(lane.left, n), right = resampleLaneCurve(lane.right, n);
  if ([...left, ...right].some(p => !p.every(Number.isFinite))) return unavailable("Lane interpolation is not finite");
  const mask = new Uint8Array(total);
  let surface = false, supported = 0;
  for (let i = 1; i < n; i++) {
    for (const [a, b, c] of [[left[i-1], right[i-1], left[i]], [left[i], right[i-1], right[i]]]) {
      const denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1]);
      if (!Number.isFinite(denominator)) return unavailable("Lane projection is not finite");
      if (Math.abs(denominator) < 1e-12) continue;
      surface = true;
      for (let k = 0; k < total; k++) {
        if (mask[k]) continue;
        const x = positions[3*k], y = positions[3*k+1], z = positions[3*k+2];
        const u = ((b[1]-c[1])*(x-c[0]) + (c[0]-b[0])*(y-c[1])) / denominator;
        const v = ((c[1]-a[1])*(x-c[0]) + (a[0]-c[0])*(y-c[1])) / denominator;
        const w = 1-u-v, surfaceZ = u*a[2] + v*b[2] + w*c[2];
        if (!Number.isFinite(u) || !Number.isFinite(v) || !Number.isFinite(w) || !Number.isFinite(surfaceZ)) return unavailable("Core projection is not finite");
        if (u >= -1e-9 && v >= -1e-9 && w >= -1e-9 && Math.abs(z-surfaceZ) <= heightMargin+1e-9) {
          mask[k] = 1; supported++;
        }
      }
    }
  }
  return surface ? { lane: lane.id, status: "complete", total, supported, mask, reason: null, tests }
    : unavailable("Lane has no non-degenerate XY surface");
}
export function computeLaneCoreEvidence(cores: CandidateCores, lanes: readonly SurveyLane[], heightMargin: number): LaneCoreEvidence {
  if (lanes.length > 50 || lanes.reduce((n,lane)=>n+lane.left.length+lane.right.length,0) > 200_000) {
    throw new Error("Core evidence accepts at most 50 lanes and 200,000 boundary vertices");
  }
  let budget = CORE_TEST_LIMIT;
  const rows = lanes.map(lane => {
    const row = laneCoreSupport(lane, cores.positions, heightMargin, budget);
    budget -= row.tests; return row;
  });
  return { ...cores, rows };
}
