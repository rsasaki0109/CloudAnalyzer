import type { XYZ } from './vectormap-geometry';

export interface RoadBuildDiagnostic {
  code: 'ambiguous_boundary_direction';
  location: XYZ;
  reference: XYZ[];
  boundaries: XYZ[][];
  context_length: number;
  forward_lanes: number;
  backward_lanes: number;
  lane_width: number;
  segment_length: number;
}
export interface RoadBuildFailure {
  message: string;
  diagnostic: RoadBuildDiagnostic;
}

/** Native structured failures travel in the existing worker error message. */
export function parseRoadBuildFailure(message: string): RoadBuildFailure | null {
  if (message.length > 300_000) return null;
  try {
    const value = JSON.parse(message);
    const d = value?.diagnostic;
    const point = (p: unknown): p is XYZ => Array.isArray(p) && p.length === 3 && p.every(v => typeof v === 'number' && Number.isFinite(v));
    const line = (p: unknown): p is XYZ[] => Array.isArray(p) && p.length >= 2 && p.length <= 256 && p.every(point);
    if (typeof value?.message !== 'string' || d?.code !== 'ambiguous_boundary_direction' || !point(d.location) || !line(d.reference) ||
        !Array.isArray(d.boundaries) || d.boundaries.length < 2 || d.boundaries.length > 17 || !d.boundaries.every(line) ||
        d.reference.length + d.boundaries.reduce((n: number, b: XYZ[]) => n + b.length, 0) > 2048 ||
        !Number.isInteger(d.forward_lanes) || d.forward_lanes < 1 || !Number.isInteger(d.backward_lanes) || d.backward_lanes < 0 ||
        d.forward_lanes + d.backward_lanes > 16 || d.boundaries.length !== d.forward_lanes + d.backward_lanes + 1 ||
        ![d.context_length, d.lane_width, d.segment_length].every(v => typeof v === 'number' && Number.isFinite(v) && v > 0)) return null;
    return value as RoadBuildFailure;
  } catch { return null; }
}
