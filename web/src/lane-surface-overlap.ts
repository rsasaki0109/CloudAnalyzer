/** Spatial review hints only: candidate boxes against the same boundary strip drawn by the viewer. */
import { Box3, Triangle, Vector3 } from "three";

type XYZ = [number, number, number];
export interface SurveyLane {
  id: number;
  left: readonly (readonly [number, number, number])[];
  right: readonly (readonly [number, number, number])[];
}
export const LANE_OVERLAP_LIMITS = { lanes: 100_000, vertices: 200_000, triangles: 200_000 } as const;

/** XY arc-length resampling shared with the viewer; Z is interpolated along each segment. */
export function resampleLaneCurve(points: SurveyLane["left"], n: number): XYZ[] {
  if (points.length === 0) return [];
  if (points.length === 1) return Array.from({ length: n }, () => [...points[0]] as XYZ);
  const cumulative = [0];
  for (let i = 1; i < points.length; i++) {
    const [a, b] = [points[i - 1], points[i]];
    cumulative.push(cumulative[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  const total = cumulative[cumulative.length - 1];
  const out: XYZ[] = [];
  let j = 1;
  for (let k = 0; k < n; k++) {
    const s = (total * k) / (n - 1);
    while (j < points.length - 1 && cumulative[j] < s) j++;
    const span = cumulative[j] - cumulative[j - 1] || 1;
    const t = Math.min(1, Math.max(0, (s - cumulative[j - 1]) / span));
    const [a, b] = [points[j - 1], points[j]];
    out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
  }
  return out;
}

export function overlappingSurveyLanes(lanes: readonly SurveyLane[], bounds: readonly number[], xyMargin: number, zMargin: number,
  limits: { lanes: number; vertices: number; triangles: number } = LANE_OVERLAP_LIMITS) {
  if (bounds.length !== 6 || !bounds.every(Number.isFinite) || bounds.slice(0, 3).some((v, a) => v > bounds[a + 3]) ||
      !Number.isFinite(xyMargin) || xyMargin < 0 || xyMargin > 10 || !Number.isFinite(zMargin) || zMargin < 0 || zMargin > 10 ||
      Object.values(limits).some(v => !Number.isSafeInteger(v) || v < 0)) throw new Error("Use finite candidate bounds and margins between 0 and 10 m");
  const query = new Box3(new Vector3(bounds[0] - xyMargin, bounds[1] - xyMargin, bounds[2] - zMargin),
    new Vector3(bounds[3] + xyMargin, bounds[4] + xyMargin, bounds[5] + zMargin));
  const ids: number[] = [];
  let checked = 0, invalid = 0, limited = Math.max(0, lanes.length - limits.lanes), vertices = 0, triangles = 0;
  const box = new Box3(), p = new Vector3(), triangle = new Triangle();
  for (let index = 0; index < Math.min(lanes.length, limits.lanes); index++) {
    const lane = lanes[index], count = lane.left.length + lane.right.length;
    if (count > limits.vertices - vertices) { limited++; continue; }
    vertices += count;
    if (!Number.isSafeInteger(lane.id) || lane.left.length < 2 || lane.right.length < 2 ||
        [...lane.left, ...lane.right].some(p => p.length !== 3 || !p.every(Number.isFinite))) { invalid++; continue; }
    box.makeEmpty();
    for (const curve of [lane.left, lane.right]) for (const point of curve) box.expandByPoint(p.fromArray(point));
    if (!query.intersectsBox(box)) { checked++; continue; }
    const n = Math.max(lane.left.length, lane.right.length, 2), required = 2 * (n - 1);
    if (required > limits.triangles - triangles) { limited++; continue; }
    triangles += required;
    const left = resampleLaneCurve(lane.left, n), right = resampleLaneCurve(lane.right, n);
    if ([...left, ...right].some(p => !p.every(Number.isFinite))) { invalid++; continue; }
    let hit = false, surface = false, invalidTriangle = false;
    for (let i = 1; i < n && !hit && !invalidTriangle; i++) {
      for (const [a, b, c] of [[left[i - 1], right[i - 1], left[i]], [left[i], right[i - 1], right[i]]]) {
        triangle.a.fromArray(a); triangle.b.fromArray(b); triangle.c.fromArray(c);
        const area = triangle.getArea();
        if (!Number.isFinite(area)) { invalidTriangle = true; break; }
        if (!(area > 0)) continue;
        surface = true;
        if (query.intersectsTriangle(triangle)) { hit = true; break; }
      }
    }
    if (!surface || invalidTriangle) { invalid++; continue; }
    checked++;
    if (hit) ids.push(lane.id);
  }
  return { ids, checked, invalid, limited, total: lanes.length, xyMargin, zMargin };
}
