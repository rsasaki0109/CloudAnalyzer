/** Exact, bounded straight XY-band sections of loaded points, without correspondence inference. */
export const SECTION_INPUT_BYTES = 64 * 1024 * 1024;
export const SECTION_POINT_LIMIT = 50_000;
export interface SectionSettings { axis: "x" | "y"; center: [number, number]; length: number; width: number }
export interface SectionSeries {
  points: number; finitePoints: number; invalidPoints: number; selectedPoints: number;
  zMin: number | null; zMax: number | null;
  along: Float64Array; positions: Float64Array;
}
export function validateSection(settings: SectionSettings): void {
  const { axis, center, length, width } = settings;
  if (!["x", "y"].includes(axis) || center.length !== 2 || !center.every(Number.isFinite) ||
      !Number.isFinite(length) || length <= 0 || !Number.isFinite(width) || width <= 0) {
    throw new Error("Choose X/Y, a finite center and positive finite section length/width");
  }
  const extents = axis === "x" ? [length, width] : [width, length];
  for (let i = 0; i < 2; i++) for (const sign of [-1, 1]) {
    const end = center[i] + sign * extents[i] / 2;
    if (!Number.isFinite(end) || end === center[i]) throw new Error("Section dimensions cannot be represented in this coordinate frame");
  }
}
function cut(positions: Float32Array, shift: readonly number[], settings: SectionSettings): SectionSeries {
  if (positions.length % 3 || shift.length !== 3 || !shift.every(Number.isFinite)) throw new Error("Matching XYZ and a finite source shift required");
  const axis = settings.axis === "x" ? 0 : 1, cross = 1 - axis;
  const center = settings.center.map((v, i) => v - shift[i]);
  if (!center.every(Number.isFinite)) throw new Error("Section center cannot be represented relative to cloud shift");
  const along: number[] = [], selected: number[] = [];
  let invalidPoints = 0, zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const world = [positions[i] + shift[0], positions[i + 1] + shift[1], positions[i + 2] + shift[2]];
    if (!world.every(Number.isFinite)) { invalidPoints++; continue; }
    const longitudinal = positions[i + axis] - center[axis], lateral = positions[i + cross] - center[cross];
    if (Math.abs(longitudinal) > settings.length / 2 || Math.abs(lateral) > settings.width / 2) continue;
    if (along.length === SECTION_POINT_LIMIT) throw new Error("Section exceeds 50,000 points in one cloud; shorten/narrow the section or crop");
    along.push(longitudinal + settings.length / 2); selected.push(...world);
    zMin = Math.min(zMin, world[2]); zMax = Math.max(zMax, world[2]);
  }
  const points = positions.length / 3;
  return { points, finitePoints: points - invalidPoints, invalidPoints, selectedPoints: along.length,
    zMin: along.length ? zMin : null, zMax: along.length ? zMax : null,
    along: Float64Array.from(along), positions: Float64Array.from(selected) };
}
export function compareSections(source: { positions: Float32Array; shift: readonly number[] },
  reference: { positions: Float32Array; shift: readonly number[] }, settings: SectionSettings): { source: SectionSeries; reference: SectionSeries } {
  validateSection(settings);
  if (source.positions.byteLength + reference.positions.byteLength > SECTION_INPUT_BYTES) throw new Error("Section input exceeds 64 MiB combined XYZ; crop a smaller pair");
  return { source: cut(source.positions, source.shift, settings), reference: cut(reference.positions, reference.shift, settings) };
}
