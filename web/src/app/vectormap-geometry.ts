import { ShapeUtils, Vector2 } from "three";

export type XYZ = [number, number, number];

/** Zebra bands clipped to the surveyed outline, including its interpolated heights.
 * Work in doubles relative to a nearby origin before converting to GPU floats.
 * The bands are a display convention; they never replace the map's geometry.
 */
export function crosswalkTriangles(outline: XYZ[], origin: XYZ): number[] {
  if (outline.some((p) => !p.every(Number.isFinite))) return [];
  const points: XYZ[] = [];
  for (const p of outline) {
    const q: XYZ = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    const last = points.at(-1);
    if (!last || Math.hypot(q[0] - last[0], q[1] - last[1]) > 1e-8) points.push(q);
  }
  if (points.length > 1 && Math.hypot(points[0][0] - points.at(-1)![0], points[0][1] - points.at(-1)![1]) < 1e-8) points.pop();
  if (points.length < 3) return [];
  const contour = points.map((p) => new Vector2(p[0], p[1]));
  if (Math.abs(ShapeUtils.area(contour)) < 1e-8) return [];
  let length = 0;
  let direction = [1, 0];
  points.forEach((p, i) => {
    const q = points[(i + 1) % points.length];
    const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (d > length) { length = d; direction = [(q[0] - p[0]) / d, (q[1] - p[1]) / d]; }
  });
  const along = (p: XYZ) => p[0] * direction[0] + p[1] * direction[1];
  const stations = points.map(along);
  const min = Math.min(...stations), max = Math.max(...stations);
  const count = Math.ceil(max - min);
  // Imported extents can be enormous. Keep display work bounded without drawing
  // stripes over only an arbitrary portion of such an outline.
  if (count > 4096) return [];
  const clip = (polygon: XYZ[], station: number, lower: boolean): XYZ[] => {
    const out: XYZ[] = [];
    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i], b = polygon[(i + 1) % polygon.length];
      const da = along(a) - station, db = along(b) - station;
      const insideA = lower ? da >= 0 : da <= 0, insideB = lower ? db >= 0 : db <= 0;
      if (insideA) out.push(a);
      if (insideA !== insideB) {
        const t = da / (da - db);
        out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
      }
    }
    return out;
  };
  const positions: number[] = [];
  for (const indices of ShapeUtils.triangulateShape(contour, [])) {
    const triangle = indices.map((i) => points[i]);
    const low = Math.max(0, Math.floor(Math.min(...triangle.map(along)) - min));
    const high = Math.min(count, Math.ceil(Math.max(...triangle.map(along)) - min));
    for (let i = low; i < high; i++) {
      const band = clip(clip(triangle, min + i, true), min + i + 0.5, false);
      for (let j = 2; j < band.length; j++) positions.push(...band[0], ...band[j - 1], ...band[j]);
    }
  }
  return positions;
}

/** The actual signal face: bottom polyline plus a known height; no invented lamps or pole. */
export function signalTriangles(points: XYZ[], height: number | null, origin: XYZ): number[] {
  if (height === null || !Number.isFinite(height) || height <= 0 || points.some((p) => !p.every(Number.isFinite))) return [];
  const positions: number[] = [];
  for (let i = 1; i < points.length; i++) {
    const a: XYZ = points[i - 1].map((v, k) => v - origin[k]) as XYZ;
    const b: XYZ = points[i].map((v, k) => v - origin[k]) as XYZ;
    const c: XYZ = [a[0], a[1], a[2] + height], d: XYZ = [b[0], b[1], b[2] + height];
    positions.push(...a, ...b, ...c, ...c, ...b, ...d);
  }
  return positions;
}
