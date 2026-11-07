/** Display-only KITTI world-from-sensor translations from a verified mapping package. */
export const TRAJECTORY_BYTES = 16 * 1024 * 1024;
export const TRAJECTORY_POSES = 100_000;
export const TRAJECTORY_DISPLAY = 10_000;
export interface MappingTrajectory {
  positions: [number, number, number][];
  count: number;
  bounds: [number, number, number, number, number, number] | null;
  unavailable: string | null;
}
export function unavailableTrajectory(reason: string): MappingTrajectory {
  return { positions: [], count: 0, bounds: null, unavailable: reason };
}
export function parseMappingTrajectory(text: string): MappingTrajectory {
  try {
    if (new TextEncoder().encode(text).length > TRAJECTORY_BYTES) throw new Error('Trajectory exceeds the 16 MiB display limit');
    const lines: string[] = [];
    for (const match of text.matchAll(/[^\r\n]+/g)) {
      const line = match[0].trim();
      if (!line || line.startsWith('#')) continue;
      if (lines.length === TRAJECTORY_POSES) throw new Error('Trajectory display requires 2–100,000 KITTI poses');
      lines.push(line);
    }
    if (lines.length < 2 || lines.length > TRAJECTORY_POSES) throw new Error('Trajectory display requires 2–100,000 KITTI poses');
    const positions: [number, number, number][] = [];
    const bounds: [number, number, number, number, number, number] = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    const kept = Math.min(lines.length, TRAJECTORY_DISPLAY);
    let next = 0;
    for (let i = 0; i < lines.length; i++) {
      const row = lines[i].split(/\s+/).map(Number);
      if (row.length !== 12 || !row.every(Number.isFinite)) throw new Error('Expected finite KITTI 3×4 matrices');
      // Reject malformed rotations rather than presenting arbitrary matrix rows as poses.
      const r = [row.slice(0, 3), row.slice(4, 7), row.slice(8, 11)];
      for (let a = 0; a < 3; a++) for (let b = a; b < 3; b++) {
        if (Math.abs(r[a].reduce((sum, v, k) => sum + v * r[b][k], 0) - (a === b ? 1 : 0)) > 1e-3)
          throw new Error('Trajectory rotation is not orthonormal');
      }
      const det = r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]);
      if (Math.abs(det - 1) > 1e-3) throw new Error('Trajectory rotation has invalid handedness');
      const point: [number, number, number] = [row[3], row[7], row[11]];
      if (point.some(v => Math.abs(v) > 1e12)) throw new Error('Trajectory coordinates exceed the display range');
      for (let axis = 0; axis < 3; axis++) {
        bounds[axis] = Math.min(bounds[axis], point[axis]);
        bounds[axis + 3] = Math.max(bounds[axis + 3], point[axis]);
      }
      if (i === Math.floor(next * (lines.length - 1) / (kept - 1))) { positions.push(point); next++; }
    }
    return { positions, count: lines.length, bounds, unavailable: null };
  } catch (error) {
    return unavailableTrajectory(error instanceof Error ? error.message : 'Invalid trajectory');
  }
}
