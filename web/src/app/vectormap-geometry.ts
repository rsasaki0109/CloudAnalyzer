import { ShapeUtils, Vector2 } from "three";

export type XYZ = [number, number, number];

/** Zebra bands clipped to the surveyed outline, including its interpolated heights.
 * Work in doubles relative to a nearby origin before converting to GPU floats.
 * Imported/manual outlines use decorative bands. Measured paint, when supplied,
 * retains its observed orientation and gaps. Neither changes map geometry.
 */
export function crosswalkTriangles(outline: XYZ[], origin: XYZ, measuredBands?: XYZ[][] | null): number[] {
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
  if (measuredBands != null) return measuredPaint(points, contour, measuredBands, origin);
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

/** Clip observed paint rectangles to the saved crossing outline. Triangulating
 * the outline first also supports concave user geometry. Heights come from the
 * map surface; missing bands stay missing rather than becoming regular stripes.
 */
function measuredPaint(points: XYZ[], contour: Vector2[], bands: XYZ[][], origin: XYZ): number[] {
  if (bands.length > 32 || points.length > 4096) return [];
  const faces = ShapeUtils.triangulateShape(contour, []).map((v) => v.map((i) => points[i]));
  const positions: number[] = [];
  for (const input of bands) {
    if (input.length !== 4 || input.some((p) => !p.every(Number.isFinite))) continue;
    const band = input.map((p) => p.map((v, i) => v - origin[i]) as XYZ);
    const signedArea = ShapeUtils.area(band.map((p) => new Vector2(p[0], p[1])));
    if (!Number.isFinite(signedArea) || Math.abs(signedArea) < 1e-8) continue;
    const sign = Math.sign(signedArea);
    if (band.some((a, i) => {
      const b = band[(i + 1) % 4], c = band[(i + 2) % 4];
      return Math.hypot(b[0] - a[0], b[1] - a[1]) > 60 ||
        sign * ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) < -1e-10;
    })) continue;
    for (const face of faces) {
      let polygon = face;
      for (let i = 0; i < 4 && polygon.length; i++) {
        const a = band[i], b = band[(i + 1) % 4];
        const distance = (p: XYZ) => sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]));
        const clipped: XYZ[] = [];
        for (let j = 0; j < polygon.length; j++) {
          const p = polygon[j], q = polygon[(j + 1) % polygon.length];
          const dp = distance(p), dq = distance(q);
          const inP = dp >= 0, inQ = dq >= 0;
          if (inP) clipped.push(p);
          if (inP !== inQ) {
            const t = dp / (dp - dq);
            clipped.push(p.map((v, k) => v + (q[k] - v) * t) as XYZ);
          }
        }
        polygon = clipped;
      }
      for (let j = 2; j < polygon.length; j++) positions.push(...polygon[0], ...polygon[j - 1], ...polygon[j]);
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

/** Append display dashes with a continuous station across vertices. Return false
 * when the display budget/range prevents drawing all segments. Source geometry
 * is untouched; integer dash indices avoid floating-modulo stalls at dash ends.
 */
export function appendDashedPairs(points: XYZ[], origin: XYZ, out: number[], dash: number, gap: number, limit = 20_000): boolean {
  const period = dash + gap;
  if (!Number.isFinite(period) || dash <= 0 || gap < 0 || !Number.isSafeInteger(limit) || limit < 0) return false;
  let along = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i];
    if (!a.every(Number.isFinite) || !b.every(Number.isFinite)) return false;
    const length = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    if (length === 0) continue;
    const end = along + length;
    if (!Number.isFinite(end) || !Number.isSafeInteger(Math.ceil(end / period))) return false;
    for (let n = Math.floor(along / period); n * period < end; n++) {
      const start = Math.max(along, n * period), stop = Math.min(end, n * period + dash);
      if (stop <= start) continue;
      if (out.length / 6 >= limit) return false;
      for (const station of [start, stop]) {
        const t = (station - along) / length;
        out.push(a[0] - origin[0] + (b[0] - a[0]) * t,
          a[1] - origin[1] + (b[1] - a[1]) * t,
          a[2] - origin[2] + (b[2] - a[2]) * t);
      }
    }
    along = end;
  }
  return true;
}
