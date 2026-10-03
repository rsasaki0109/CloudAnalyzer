//! Explicit width priors inside curb candidates, never measured lane markings.
use super::{BuildOptions, BuildReport, Evidence, ExtractedRoad, paint_corridor, quantile};
use crate::PointCloud;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct RetainedRoadEdge {
    pub candidate_road: usize,
    pub boundary_slot: usize,
    pub reference: Vec<[f64; 3]>,
    pub geometry: Vec<[f64; 3]>,
    pub evidence: Vec<Evidence>,
    pub source_before_fitting: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct LaneEdgeSideReport {
    pub boundary_slot: usize,
    pub applied: bool,
    pub reason: String,
    pub candidate_vertices: usize,
    pub distant_curb_vertices: usize,
    pub longest_distant_curb_run: usize,
    pub median_observed_curb_distance_m: Option<f64>,
    pub inferred_vertices_before_trimming: usize,
    pub maximum_movement_m: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct LaneEdgeInferenceReport {
    pub applied: bool,
    pub reason: String,
    pub limited: bool,
    pub configured_lane_width_m: f64,
    pub minimum_curb_gap_m: f64,
    /// Side counters and retained curb candidates describe the incoming fit,
    /// BEFORE source-footprint trimming. These are not exported lane boundaries.
    pub sides: Vec<LaneEdgeSideReport>,
    pub retained_road_edges: Vec<RetainedRoadEdge>,
}

pub(super) fn apply(
    cloud: &PointCloud,
    line: &[[f64; 3]],
    o: &BuildOptions,
    roads: &mut [ExtractedRoad],
    build: &BuildReport,
) -> LaneEdgeInferenceReport {
    let gap = 0.5_f64.max(2.0 * o.bin_width);
    let mut report = LaneEdgeInferenceReport {
        configured_lane_width_m: o.lane_width,
        minimum_curb_gap_m: gap,
        ..Default::default()
    };
    let hold = |reason: &str, mut r: LaneEdgeInferenceReport| {
        r.reason = reason.into();
        r
    };
    let Some(divider) = build.paint_divider.as_ref().filter(|p| p.applied) else {
        return hold(
            "requires an applied source paint-divider fit guarded by physical curb pairs",
            report,
        );
    };
    if !o.verify_curb_profiles {
        return hold("selected curb profiles must be verified", report);
    }
    let scan = match paint_corridor::scan(cloud, line, o, 1) {
        Ok(scan) => scan,
        Err(scan) => {
            report.limited = scan.limited;
            return hold(&scan.reason, report);
        }
    };
    let original = roads.to_vec();
    let angle = divider.heading_correction_degrees.unwrap().to_radians();
    let tangent = [
        scan.d[0] * angle.cos() - scan.d[1] * angle.sin(),
        scan.d[0] * angle.sin() + scan.d[1] * angle.cos(),
    ];
    let normal = [-tangent[1], tangent[0]];
    for (j, sign) in [(0, 1.0), (2, -1.0)] {
        let mut side = LaneEdgeSideReport {
            boundary_slot: j,
            ..Default::default()
        };
        let mut distances = vec![];
        let mut has_paint = false;
        for road in roads.iter() {
            let mut run = 0;
            for k in 0..road.reference.len() {
                side.candidate_vertices += 1;
                let label = road.evidence[j][k];
                has_paint |= matches!(label, Evidence::RgbPaint | Evidence::Intensity);
                let source = road
                    .source_boundaries
                    .as_ref()
                    .map_or(road.boundaries[j][k], |s| s[j][k]);
                let center = road.boundaries[1][k];
                let width = sign
                    * ((source[0] - center[0]) * normal[0] + (source[1] - center[1]) * normal[1]);
                if label == Evidence::Curb && width > o.lane_width + gap {
                    distances.push(width);
                    side.distant_curb_vertices += 1;
                    run += 1;
                    side.longest_distant_curb_run = side.longest_distant_curb_run.max(run);
                } else {
                    run = 0;
                }
            }
        }
        if has_paint {
            side.reason = "outside paint/intensity observations take precedence".into();
        } else if side.distant_curb_vertices * 2 <= side.candidate_vertices
            || side.longest_distant_curb_run < 3
        {
            side.reason="distant verified curbs need a majority of candidate vertices and three consecutive observations".into();
        } else {
            side.median_observed_curb_distance_m = quantile(&mut distances, 0.5);
            let mut replacements = vec![];
            let mut reason = None;
            for (r, road) in roads.iter().enumerate() {
                for (k, &center) in road.boundaries[1].iter().enumerate() {
                    let old = road.boundaries[j][k];
                    let old_width = sign
                        * ((old[0] - center[0]) * normal[0] + (old[1] - center[1]) * normal[1]);
                    if old_width < o.lane_width - 1e-9 {
                        reason = Some(
                            "configured edge would expand beyond an existing outside candidate",
                        );
                        break;
                    }
                    let mut point = [
                        center[0] + sign * normal[0] * o.lane_width,
                        center[1] + sign * normal[1] * o.lane_width,
                        center[2],
                    ];
                    let dx = point[0] - scan.origin[0];
                    let dy = point[1] - scan.origin[1];
                    let s = dx * scan.d[0] + dy * scan.d[1];
                    let t = dx * scan.normal[0] + dy * scan.normal[1];
                    point[2] = match scan.samples.ground(&[s, t, 0.0]) {
                        Some(Some(z)) if (z - center[2]).abs() <= 0.3 => z,
                        None => {
                            report.limited = true;
                            reason = Some("lane-edge ground query budget exceeded");
                            break;
                        }
                        _ => {
                            reason = Some("configured edge lacks same-level source ground");
                            break;
                        }
                    };
                    replacements.push((r, k, point));
                }
                if reason.is_some() {
                    break;
                }
            }
            if let Some(reason) = reason {
                side.reason = reason.into();
            } else {
                // Retain the original selected road-edge candidate and its
                // sources separately before replacing a lane boundary.
                for (r, road) in roads.iter().enumerate() {
                    report.retained_road_edges.push(RetainedRoadEdge {
                        candidate_road: r,
                        boundary_slot: j,
                        reference: road.reference.clone(),
                        geometry: road.boundaries[j].clone(),
                        evidence: road.evidence[j].clone(),
                        source_before_fitting: road
                            .source_boundaries
                            .as_ref()
                            .map_or_else(|| road.boundaries[j].clone(), |s| s[j].clone()),
                    });
                }
                side.inferred_vertices_before_trimming = replacements.len();
                for (r, k, point) in replacements {
                    let road = &mut roads[r];
                    let old = road.boundaries[j][k];
                    side.maximum_movement_m = side
                        .maximum_movement_m
                        .max((old[0] - point[0]).hypot(old[1] - point[1]));
                    if road.source_boundaries.is_none() {
                        road.source_boundaries = Some(road.boundaries.clone());
                    }
                    road.boundaries[j][k] = point;
                    road.evidence[j][k] = Evidence::WidthPrior;
                    road.source_boundaries.as_mut().unwrap()[j][k] = point;
                }
                side.applied = true;
                side.reason="configured-width lane edge inferred inside distant curb candidates; no outer marking observed".into();
                report.applied = true;
            }
        }
        report.sides.push(side);
    }
    if report.limited {
        roads.clone_from_slice(&original);
        report.applied = false;
        report.retained_road_edges.clear();
        for side in &mut report.sides {
            side.applied = false;
            side.inferred_vertices_before_trimming = 0;
            side.maximum_movement_m = 0.0;
            side.reason = "stage held because a ground query exceeded its budget".into();
        }
        return hold(
            "lane-edge inference query budget exceeded; original geometry restored",
            report,
        );
    }
    report.reason=if report.applied { "explicit lane-width prior used inside distant curbs; road-edge candidates retained separately for review" } else { "no outside side met the source and width-prior guards" }.into();
    report
}

pub(super) fn warnings(build: &mut BuildReport) {
    if let Some(r) = &build.lane_edge_inference {
        build.warnings.push(if r.applied { "Outer lane edge inferred from the configured width relative to source paint, not detected outer paint. Original curb/road-edge candidates and their evidence are retained separately in the report before footprint trimming. The strip between them has no automatically certified shoulder/parking role. Review width, lane legality and directions.".into() } else { format!("Outer lane-edge inference held: {}. Existing boundary geometry retained.",r.reason) });
    }
}

#[cfg(test)]
mod tests {
    use super::super::extract;
    use super::*;

    fn scene(right: f64) -> PointCloud {
        let mut cloud = PointCloud::default();
        let mut colors = vec![];
        for ix in -20_i32..=320 {
            let x = ix as f64 * 0.1;
            for iy in -90..=90 {
                let y = iy as f64 * 0.1 + 0.03;
                let t = y + 0.02 * x;
                cloud.positions.push([
                    x,
                    y,
                    12.0 + x * 0.01 + if t < right || t > 3.2 { 0.2 } else { 0.0 },
                ]);
                colors.push(if t.abs() < 0.055 && ix.rem_euclid(80) < 40 {
                    [230; 3]
                } else {
                    [70; 3]
                });
            }
        }
        cloud.colors = Some(colors);
        cloud
    }
    fn options() -> BuildOptions {
        BuildOptions {
            infer_lane_edges: true,
            fit_paint_divider: true,
            physical_anchors_only: true,
            segment_length: 0.0,
            ..Default::default()
        }
    }
    fn trace() -> [[f64; 3]; 2] {
        [[0.0, 0.0, 100.0], [30.0, 0.0, 100.0]]
    }

    #[test]
    fn lane_prior_moves_inside_curb_without_observing_an_outer_marking() {
        let cloud = scene(-4.8);
        let original = cloud.positions.clone();
        let mut off = options();
        off.infer_lane_edges = false;
        let (before, old) = extract(&cloud, &trace(), &off).unwrap();
        assert!(old.lane_edge_inference.is_none());
        let (after, report) = extract(&cloud, &trace(), &options()).unwrap();
        let inferred = report.lane_edge_inference.unwrap();
        assert!(inferred.applied, "{inferred:?}");
        assert!(!inferred.sides[0].applied);
        assert!(inferred.sides[1].applied);
        assert!(inferred.sides[1].maximum_movement_m > 0.7);
        assert_eq!(inferred.retained_road_edges.len(), before.len());
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            assert_eq!(a.reference, b.reference);
            assert_eq!(a.boundaries[0], b.boundaries[0]);
            assert_eq!(a.boundaries[1], b.boundaries[1]);
            assert_eq!(a.evidence[1], b.evidence[1]);
            assert_eq!(inferred.retained_road_edges[i].geometry, a.boundaries[2]);
            assert_eq!(inferred.retained_road_edges[i].evidence, a.evidence[2]);
            assert!(b.evidence[2].iter().all(|&e| e == Evidence::WidthPrior));
            for (p, q) in b.boundaries[1].iter().zip(&b.boundaries[2]) {
                assert!(((p[0] - q[0]).hypot(p[1] - q[1]) - 3.5).abs() < 1e-9);
                assert!(q[2] < 12.4);
            }
        }
        assert_eq!(cloud.positions, original);
    }

    #[test]
    fn close_curbs_missing_paint_and_disabled_curb_checks_hold_existing_geometry() {
        for (cloud, o) in [
            (scene(-3.9), options()),
            (
                scene(-4.8),
                BuildOptions {
                    fit_paint_divider: false,
                    ..options()
                },
            ),
            (
                scene(-4.8),
                BuildOptions {
                    verify_curb_profiles: false,
                    ..options()
                },
            ),
        ] {
            let mut off = o.clone();
            off.infer_lane_edges = false;
            let (before, _) = extract(&cloud, &trace(), &off).unwrap();
            let (after, r) = extract(&cloud, &trace(), &o).unwrap();
            assert!(!r.lane_edge_inference.unwrap().applied);
            for (a, b) in before.iter().zip(after) {
                assert_eq!(a.boundaries, b.boundaries);
                assert_eq!(a.evidence, b.evidence);
            }
        }
    }

    #[test]
    fn an_observed_outer_marking_or_one_expanding_vertex_holds_the_whole_side() {
        let cloud = scene(-4.8);
        let mut off = options();
        off.infer_lane_edges = false;
        for marking in [true, false] {
            let (mut roads, build) = extract(&cloud, &trace(), &off).unwrap();
            if marking {
                roads[0].evidence[2][0] = Evidence::RgbPaint;
            } else {
                let center = roads[0].boundaries[1][0];
                roads[0].boundaries[2][0] = [center[0], center[1] - 2.5, center[2]];
            }
            let original = roads.clone();
            let line = super::super::resample(&trace(), 2.0).unwrap();
            let report = apply(&cloud, &line, &options(), &mut roads, &build);
            assert!(!report.applied);
            assert!(report.retained_road_edges.is_empty());
            for (a, b) in original.iter().zip(roads) {
                assert_eq!(a.boundaries, b.boundaries);
                assert_eq!(a.evidence, b.evidence);
            }
        }
    }

    #[test]
    fn rotation_right_hand_and_source_footprint_check_the_actual_inferred_edge() {
        let angle = 0.7_f64;
        let transform = |p: [f64; 3]| {
            [
                7000.0 + p[0] * angle.cos() - p[1] * angle.sin(),
                80000.0 + p[0] * angle.sin() + p[1] * angle.cos(),
                p[2],
            ]
        };
        let mut cloud = scene(-4.8);
        cloud.positions.iter_mut().for_each(|p| {
            p[1] = -p[1];
            *p = transform(*p);
        });
        let poses = trace().map(transform);
        let o = BuildOptions {
            left_hand_traffic: false,
            fit_source_surface: true,
            ..options()
        };
        let (roads, report) = extract(&cloud, &poses, &o).unwrap();
        assert!(
            report.lane_edge_inference.as_ref().unwrap().applied,
            "{report:?}"
        );
        assert!(report.surface_fit.unwrap().preserved_candidate_geometry);
        assert!(report.generated_length > 10.0);
        assert!(
            roads
                .iter()
                .all(|r| r.evidence[0].iter().all(|&e| e == Evidence::WidthPrior))
        );
    }
}
