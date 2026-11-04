/** Shapes of the map view and of the worker's vector-map reports. */

export type XYZ = [number, number, number];

export interface Neighbor {
  lane: number;
  direction?: "same" | "opposite";
}

export interface LaneView {
  id: number;
  kind: string;
  left: XYZ[];
  right: XYZ[];
  leftRef: { id: number; reversed: boolean };
  rightRef: { id: number; reversed: boolean };
  center: XYZ[];
  successors: number[];
  predecessors: number[];
  leftNeighbor: Neighbor | null;
  rightNeighbor: Neighbor | null;
  speedLimit: number | null;
  turn: string | null;
  oneWay: boolean;
}

interface BoundaryView {
  id: number;
  kind: { type: string; pattern?: string };
  points: XYZ[];
}

export interface EquipmentRule {
  id: number;
  kind: "vehicle" | "pedestrian" | "crosswalk" | "stop_line" | "mixed";
  signals: number[];
  crosswalk: number | null;
  lanes: number[];
  controlled_crosswalks: number[];
  stop_lines: number[];
  review_source: string;
}
export interface RelationCandidate {
  key: string; target_kind: string; target_id: number; lanes: number[];
  controlled_crosswalks: number[]; stop_lines: number[];
  distance_m: number | null; axis_degrees: number | null; road_context: string;
  eligible: boolean; already_linked: boolean; reasons: string[];
}
export interface RelationProposal {
  rule_id: number; kind: string; map_snapshot: string; candidates: RelationCandidate[];
  eligible_count: number; ambiguous: boolean; limited: boolean; warnings: string[];
}
export type BoundarySource = "rgb_paint" | "intensity" | "curb" | "support_edge" | "width_prior";
export interface EvidenceProfile { boundary_ids: number[]; geometry: XYZ[]; evidence: BoundarySource[]; source_before_fitting: XYZ[] }
export interface RoadEdgeSnapshot { geometry: XYZ[]; evidence: BoundarySource[]; source_before_fitting: XYZ[]; boundary_slot: number }
interface BoundaryEvidenceView { profiles: EvidenceProfile[]; roadEdges: RoadEdgeSnapshot[]; limited: boolean }

export interface MapView {
  boundaryEvidence?: BoundaryEvidenceView;
  lanes: LaneView[];
  boundaries: BoundaryView[];
  stopLines: { id: number; points: XYZ[]; geometrySource?: string }[];
  crosswalks: { id: number; outline: XYZ[]; paintBands?: XYZ[][] | null; editable?: boolean; geometrySource?: string }[];
  signals: { id: number; points: XYZ[]; height: number | null; kind?: string; geometrySource?: string }[];
  regulatoryElements?: EquipmentRule[];
  georeferenced: boolean;
}

export interface Issue {
  severity: "info" | "warning" | "error";
  code: string;
  message: string;
  entity?: { kind: string; id: number };
}

export interface Nearest {
  lane: number;
  distance: number;
  station: number;
  inside: boolean;
}

interface JunctionCandidate {
  from: number;
  to: number;
  gap: number;
  turn_degrees: number;
  ground_support: number;
  boundary_support?: [number, number];
  ambiguous: boolean;
  center: XYZ[];
  left: XYZ[];
  right: XYZ[];
}
export interface JunctionReport {
  candidates: JunctionCandidate[];
  added: number[];
  warnings: string[];
}
export interface SignalReport {
  geometry: XYZ[]; height: number; width: number; thickness: number; points: number;
  plane_rms: number; added: number | null; reused: number | null; warnings: string[];
}
export interface CrosswalkReport {
  candidates: { outline: XYZ[]; left_edge: XYZ[]; right_edge: XYZ[]; stripes: XYZ[][]; stripe_count: number; width: number; length: number; angle_degrees: number; score: number }[];
  points: number; ground_points: number; plane_rms: number; brightness_source: string;
  local_profile_seeds: number; local_profiles_limited: boolean; component_bands: number; component_bands_limited: boolean;
  added: number | null; reused: number | null; warnings: string[];
}
type DiscoveryEvidence =
  | { kind: "repeated_paint"; measurement: CrosswalkReport["candidates"][number] }
  | { kind: "bright_bar"; transverse_to_road: boolean; geometry: XYZ[]; width: number; thickness: number; points: number }
  | { kind: "elevated_panel"; geometry: XYZ[]; height: number; width: number; thickness: number; points: number; plane_rms: number };
export interface DiscoveryCandidate { id: number; key: string; min: XYZ; max: XYZ; nearby_lanes: number[]; evidence: DiscoveryEvidence }
export interface DiscoveryReport { candidates: DiscoveryCandidate[]; detected_candidates: number; limited: boolean; source_points: number; corridor_points: number; windows: number; unsupported_windows: number; paint_refinement_windows: number; paint_refinement_unsupported_windows: number; paint_refinement_limited: boolean; warnings: string[] }

export interface Edited {
  result: unknown;
  view: MapView;
  undo: number;
}

interface PaintCandidateDiagnostics {
  bright_candidates: number;
  local_ground_missing: number;
  local_height_mismatch: number;
  trace_ground_missing: number;
  trace_height_mismatch: number;
  flank_support_missing: number;
  flank_contrast_insufficient: number;
  accepted: number;
  complete: boolean;
}
export interface PaintBudgetInfo {
  candidate_diagnostics?: PaintCandidateDiagnostics;
  budget_stage?: string;
  budget_query?: {radius_m: number; candidate_points: number; limit: number};
}
export interface BuildReport {
  lane_edge_inference?: { applied: boolean; reason: string; limited: boolean; configured_lane_width_m: number; retained_road_edges: unknown[]; sides: { boundary_slot: number; applied: boolean; reason: string; inferred_vertices_before_trimming: number; maximum_movement_m: number }[] } | null;
  paint_divider?: PaintBudgetInfo & { source_channel?: "intensity"; applied: boolean; reason: string; limited: boolean; curb_pair_sections: number; sampled_sections: number; maximum_divider_movement_m: number; track: { observed_length_m: number; interpolated_length_m: number; extrapolated_length_m: number } | null } | null;
  paint_corridor?: PaintBudgetInfo & { source_channel?: "intensity"; intensity_range?: [number,number]; applied: boolean; reason: string; limited: boolean; measured_lane_widths_m: number[]; tracks: { observed_length_m: number; interpolated_length_m: number; extrapolated_length_m: number }[] } | null;
  trace_alignment?: { applied: boolean; reason: string; shift_xy: [number, number]; curb_pair_sections: number; sampled_sections: number } | null;
  surface_fit?: { deferred_length_m: number; minimum_lane_width_m: number | null; maximum_lane_width_m: number | null } | null;
  roads: number;
  lanes: number;
  generated_length: number;
  added_length: number;
  reused_length: number;
  tracked_vertices: number;
  coverage_edge_anchor_candidates_ignored: number;
  fitted_vertices: number;
  maximum_fit_displacement: number;
  observed_fraction: number[];
  warnings: string[];
}

export interface SourceCurveSupport { fraction: number; start_supported: boolean; end_supported: boolean; insufficient_returns: number; height_mismatches: number }
export interface SourceProblem {
  lane: number; curve: "center" | "left" | "right";
  reason: "insufficient_returns" | "height_mismatch";
  from_m: number; to_m: number; points: XYZ[];
}
export interface SourceQualityReport {
  lanes: {lane: number; center: SourceCurveSupport; left: SourceCurveSupport; right: SourceCurveSupport; needs_review: boolean}[];
  low_support_lanes: number[]; omitted_lanes: number[]; malformed_lanes: number[]; limited: boolean; warnings: string[];
  problems: SourceProblem[]; problems_limited: boolean;
}
