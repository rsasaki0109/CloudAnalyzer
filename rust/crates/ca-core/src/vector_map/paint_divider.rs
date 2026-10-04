//! Source paint corrects one interior boundary; outside geometry stays intact.
use super::{
    BuildOptions, BuildReport, Evidence, ExtractedRoad, PaintTrackReport, SurfaceIndex,
    paint_corridor, quantile, trace_alignment,
};
use crate::PointCloud;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Default)]
pub struct PaintDividerReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_channel: Option<super::PaintChannel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_stage: Option<paint_corridor::PaintBudgetStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_query: Option<paint_corridor::PaintQueryLimit>,
    pub applied: bool,
    pub reason: String,
    pub limited: bool,
    pub roi_points: usize,
    pub contrasted_points: usize,
    pub eligible_tracks: usize,
    pub sampled_sections: usize,
    pub curb_pair_sections: usize,
    pub ambiguous_sections: usize,
    pub longest_supported_run: usize,
    pub heading_correction_degrees: Option<f64>,
    pub residual_p90_m: Option<f64>,
    pub maximum_divider_movement_m: f64,
    /// Source observation intervals before footprint trimming, not paint coverage.
    pub track: Option<PaintTrackReport>,
}

pub(super) fn apply(
    cloud: &PointCloud,
    line: &[[f64; 3]],
    o: &BuildOptions,
    roads: &mut [ExtractedRoad],
    build: &BuildReport,
) -> PaintDividerReport {
    let mut report = PaintDividerReport {
        source_channel: (o.paint_channel == super::PaintChannel::Intensity)
            .then_some(super::PaintChannel::Intensity),
        ..Default::default()
    };
    let hold = |reason: &str, mut r: PaintDividerReport| {
        r.reason = reason.into();
        r.maximum_divider_movement_m = 0.0;
        r
    };
    if o.forward_lanes + o.backward_lanes != 2 || roads.is_empty() {
        return hold(
            "requires an explicitly configured two-lane road with generated geometry",
            report,
        );
    }
    if build.paint_corridor.as_ref().is_some_and(|p| p.applied) {
        return hold(
            "complete paint corridor already applied; no divider correction needed",
            report,
        );
    }
    let scan = match paint_corridor::scan(cloud, line, o, 1) {
        Ok(scan) => scan,
        Err(scan) => {
            report.limited = scan.limited;
            report.roi_points = scan.roi_points;
            report.contrasted_points = scan.contrasted_points;
            report.budget_stage = scan.budget_stage;
            report.budget_query = scan.budget_query;
            return hold(&scan.reason, report);
        }
    };
    report.roi_points = scan.report.roi_points;
    report.contrasted_points = scan.report.contrasted_points;
    let mut tracks: Vec<_> = scan
        .tracks
        .iter()
        .filter(|t| paint_corridor::track_report(t, scan.length).strong)
        .collect();
    // Only strong tracks near the existing interior line are eligible. Multiple
    // eligible tracks never select a winner by proximity or partial evidence.
    tracks.retain(|t| {
        roads.iter().flat_map(|r| &r.boundaries[1]).all(|p| {
            let dx = p[0] - scan.origin[0];
            let dy = p[1] - scan.origin[1];
            let s = dx * scan.d[0] + dy * scan.d[1];
            let lateral = dx * scan.normal[0] + dy * scan.normal[1];
            (lateral - t.intercept - scan.initial_slope * s).abs()
                <= o.lane_width * 0.5 + o.search_margin
        })
    });
    report.eligible_tracks = tracks.len();
    if tracks.len() != 1 {
        return hold("need one unique strong interior paint track", report);
    }
    let track = tracks[0];
    let mean = [0, 1].map(|a| {
        track
            .ids
            .iter()
            .map(|&i| scan.paint.points[i][a])
            .sum::<f64>()
            / track.ids.len() as f64
    });
    let xx = track
        .ids
        .iter()
        .map(|&i| (scan.paint.points[i][0] - mean[0]).powi(2))
        .sum::<f64>();
    let xy = track
        .ids
        .iter()
        .map(|&i| (scan.paint.points[i][0] - mean[0]) * (scan.paint.points[i][1] - mean[1]))
        .sum::<f64>();
    let slope = xy / xx.max(1e-12);
    let intercept = quantile(
        &mut track
            .ids
            .iter()
            .map(|&i| scan.paint.points[i][1] - slope * scan.paint.points[i][0])
            .collect::<Vec<_>>(),
        0.5,
    )
    .unwrap();
    let residual = quantile(
        &mut track
            .ids
            .iter()
            .map(|&i| (scan.paint.points[i][1] - slope * scan.paint.points[i][0] - intercept).abs())
            .collect::<Vec<_>>(),
        0.9,
    )
    .unwrap();
    if slope.atan().abs() > 15_f64.to_radians() || residual > 0.2 {
        return hold("interior paint heading or residual exceeds limits", report);
    }
    let width = 2.0 * o.lane_width;
    let reach = 2.0 * width;
    let index = SurfaceIndex::new(cloud, line, reach + o.half_window);
    let mut run = 0;
    for &p in line {
        report.sampled_sections += 1;
        let bins = index.slice(p, scan.d, -reach, reach, o);
        let surface: Vec<_> = bins
            .iter()
            .map(|ids| {
                if ids.len() < o.min_bin_points {
                    None
                } else {
                    quantile(
                        &mut ids
                            .iter()
                            .map(|&i| cloud.positions[i][2])
                            .collect::<Vec<_>>(),
                        0.2,
                    )
                }
            })
            .collect();
        let lateral = |i: usize| -reach + (i as f64 + 0.5) * o.bin_width;
        let ground = quantile(
            &mut surface
                .iter()
                .enumerate()
                .filter(|(i, _)| lateral(*i).abs() < 0.8)
                .filter_map(|(_, z)| *z)
                .collect::<Vec<_>>(),
            0.5,
        );
        let s = (p[0] - scan.origin[0]) * scan.d[0] + (p[1] - scan.origin[1]) * scan.d[1];
        let paint = intercept + slope * s;
        let mut pairs = 0;
        let mut start = 0;
        for end in 1..=surface.len() {
            if end < surface.len()
                && surface[end]
                    .zip(surface[end - 1])
                    .is_some_and(|(a, b)| (a - b).abs() <= o.curb_height.min(0.08))
            {
                continue;
            }
            if let Some(base) = surface[start] {
                let right = lateral(start);
                let left = lateral(end - 1);
                let side_widths = [paint - right, left - paint];
                if left - right >= width - 1e-9
                    && left - right <= width + 2.0 * o.search_margin
                    && side_widths
                        .iter()
                        .all(|&w| (1.5..=6.0_f64.min(o.lane_width + o.search_margin)).contains(&w))
                    && ground.is_some_and(|z| (z - base).abs() <= 0.3)
                    && trace_alignment::curb(&surface, start, -1, o)
                    && trace_alignment::curb(&surface, end - 1, 1, o)
                {
                    pairs += 1;
                }
            }
            start = end;
        }
        if pairs == 1 {
            report.curb_pair_sections += 1;
            run += 1;
            report.longest_supported_run = report.longest_supported_run.max(run);
        } else {
            report.ambiguous_sections += usize::from(pairs > 1);
            run = 0;
        }
    }
    if report.ambiguous_sections > 0
        || report.curb_pair_sections * 2 <= report.sampled_sections
        || report.longest_supported_run < 3
    {
        return hold(
            "paint must lie inside physical curb pairs in a majority of sections, including three consecutive sections",
            report,
        );
    }
    // Validate EVERY replacement before mutating. Missing ground, crossed edges,
    // narrow widths or query limits leave the entire original geometry intact.
    let mut changes = Vec::new();
    for (r, road) in roads.iter().enumerate() {
        for (k, p) in road.reference.iter().enumerate() {
            let s = (p[0] - scan.origin[0]) * scan.d[0] + (p[1] - scan.origin[1]) * scan.d[1];
            let lateral = intercept + slope * s;
            let z = match scan.samples.ground(&[s, lateral, 0.0]) {
                Ok(Some(z)) => z,
                Err(query) => {
                    report.limited = true;
                    report.budget_stage =
                        Some(paint_corridor::PaintBudgetStage::BoundaryGroundNeighbours);
                    report.budget_query = Some(query);
                    return hold("divider ground query budget exceeded", report);
                }
                _ => return hold("divider lacks source ground at an output vertex", report),
            };
            let point = [
                scan.origin[0] + scan.d[0] * s + scan.normal[0] * lateral,
                scan.origin[1] + scan.d[1] * s + scan.normal[1] * lateral,
                z,
            ];
            let sides = [road.boundaries[0][k], road.boundaries[2][k]]
                .map(|q| (q[0] - point[0]) * scan.normal[0] + (q[1] - point[1]) * scan.normal[1]);
            if [sides[0], -sides[1]]
                .iter()
                .any(|&w| w < 1.5 || w > 6.0_f64.min(o.lane_width + o.search_margin))
            {
                return hold(
                    "paint correction would cross or excessively narrow/widen an existing lane",
                    report,
                );
            }
            let near = track
                .ids
                .iter()
                .copied()
                .filter(|&i| {
                    (scan.paint.points[i][0] - s).hypot(scan.paint.points[i][1] - lateral) <= 0.5
                })
                .min_by(|&a, &b| {
                    (scan.paint.points[a][0] - s)
                        .abs()
                        .total_cmp(&(scan.paint.points[b][0] - s).abs())
                });
            let (evidence, source) = near.map_or((Evidence::WidthPrior, point), |i| {
                (
                    o.paint_channel.evidence(),
                    cloud.positions[scan.paint.ids[i]],
                )
            });
            let old = road.boundaries[1][k];
            report.maximum_divider_movement_m = report
                .maximum_divider_movement_m
                .max((old[0] - point[0]).hypot(old[1] - point[1]));
            changes.push((r, k, point, evidence, source));
        }
    }
    for (r, k, point, evidence, source) in changes {
        let road = &mut roads[r];
        if road.source_boundaries.is_none() {
            road.source_boundaries = Some(road.boundaries.clone());
        }
        road.boundaries[1][k] = point;
        road.evidence[1][k] = evidence;
        road.source_boundaries.as_mut().unwrap()[1][k] = source;
    }
    let mut observed = paint_corridor::track_report(track, scan.length);
    observed.lateral_intercept_m = intercept;
    report.track = Some(observed);
    report.heading_correction_degrees = Some(slope.atan().to_degrees());
    report.residual_p90_m = Some(residual);
    report.applied = true;
    report.reason = "unique strong paint track inside paired physical curbs; only the interior boundary was corrected".into();
    report
}

pub(super) fn warnings(build: &mut BuildReport) {
    if let Some(p) = &build.paint_divider {
        let warning: String = if p.applied
            && build
                .lane_edge_inference
                .as_ref()
                .is_some_and(|r| r.applied)
        {
            "Interior boundary corrected using RGB paint guarded by source curb pairs. Separate lane-edge inference then changed an outer boundary using the configured width; that outer line is not observed paint. Nearby interior paint only is labelled observed; gaps/extensions remain inferred.".into()
        } else if p.applied {
            "Interior boundary corrected using RGB paint guarded by source curb pairs. Outside candidate geometry was retained and source-footprint checks can still trim unsupported intervals. Only nearby paint is labelled observed; gaps/extensions remain inferred. Lane counts, divider semantics and directions require review.".into()
        } else {
            format!(
                "Interior RGB paint correction held: {}. Existing geometry retained.",
                p.reason
            )
        };
        build.warnings.push(
            if p.source_channel == Some(super::PaintChannel::Intensity) {
                warning.replace("RGB", "intensity")
            } else {
                warning
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::extract;
    use super::*;

    fn scene(raised: f64, missing: bool, ambiguous: bool) -> PointCloud {
        let mut cloud = PointCloud::default();
        let mut colors = vec![];
        for ix in -20_i32..=320 {
            let x = ix as f64 * 0.1;
            for iy in -80..=80 {
                let y = iy as f64 * 0.1 + 0.03;
                let t = y + 0.02 * x;
                if missing && t > 3.8 {
                    continue;
                }
                let paint = (t.abs() < 0.055 || (ambiguous && (t - 2.0).abs() < 0.055))
                    && ix.rem_euclid(80) < 40;
                cloud.positions.push([
                    x,
                    y,
                    12.0 + x * 0.01
                        + if !(-3.9..=3.8).contains(&t) {
                            raised
                        } else {
                            0.0
                        },
                ]);
                colors.push(if paint { [230; 3] } else { [70; 3] });
            }
        }
        cloud.colors = Some(colors);
        cloud
    }
    fn trace() -> [[f64; 3]; 2] {
        [[0.0, 0.0, 100.0], [30.0, 0.0, 100.0]]
    }
    fn options() -> BuildOptions {
        BuildOptions {
            fit_paint_divider: true,
            physical_anchors_only: true,
            segment_length: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn intensity_divider_keeps_curbs_and_gap_labels_and_rejects_unpaired_sides() {
        for missing in [false, true] {
            let mut cloud = scene(0.2, missing, false);
            let values = cloud
                .colors
                .as_ref()
                .unwrap()
                .iter()
                .map(|c| f32::from(c[0]) * 256.)
                .collect();
            cloud.attributes.push(crate::Attribute {
                name: crate::INTENSITY.into(),
                values: crate::AttributeValues::F32(values),
            });
            cloud.colors = None;
            let mut o = options();
            o.paint_channel = super::super::PaintChannel::Intensity;
            let mut off = o.clone();
            off.fit_paint_divider = false;
            let (before, _) = extract(&cloud, &trace(), &off).unwrap();
            let (after, report) = extract(&cloud, &trace(), &o).unwrap();
            assert_eq!(report.paint_divider.unwrap().applied, !missing);
            for (a, b) in before.iter().zip(&after) {
                for j in [0, 2] {
                    assert_eq!(a.boundaries[j], b.boundaries[j]);
                    assert_eq!(a.evidence[j], b.evidence[j]);
                }
                if missing {
                    assert_eq!(a.boundaries, b.boundaries);
                } else {
                    assert!(b.evidence[1].contains(&Evidence::Intensity));
                    assert!(b.evidence[1].contains(&Evidence::WidthPrior));
                    assert!(!b.evidence[1].contains(&Evidence::RgbPaint));
                }
            }
        }
    }

    #[test]
    fn one_dashed_track_corrects_only_interior_and_does_not_observe_gaps() {
        let cloud = scene(0.2, false, false);
        let original = cloud.positions.clone();
        let mut off = options();
        off.fit_paint_divider = false;
        let (before, old) = extract(&cloud, &trace(), &off).unwrap();
        assert!(old.paint_divider.is_none());
        let (after, report) = extract(&cloud, &trace(), &options()).unwrap();
        let fit = report.paint_divider.unwrap();
        assert!(fit.applied, "{fit:?}");
        assert_eq!(before.len(), after.len());
        assert!(fit.maximum_divider_movement_m > 0.005);
        assert!(fit.track.unwrap().interpolated_length_m > 8.0);
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.reference, b.reference);
            for j in [0, 2] {
                assert_eq!(a.boundaries[j], b.boundaries[j]);
                assert_eq!(a.evidence[j], b.evidence[j]);
            }
            for p in &b.boundaries[1] {
                assert!((p[1] + 0.02 * p[0]).abs() < 0.06);
                assert!(p[2] < 12.4);
            }
            assert!(b.evidence[1].contains(&Evidence::RgbPaint));
            assert!(b.evidence[1].contains(&Evidence::WidthPrior));
        }
        assert_eq!(cloud.positions, original);
    }

    #[test]
    fn missing_curbs_walls_ambiguous_paint_and_wrong_count_leave_geometry_intact() {
        for cloud in [
            scene(0.2, true, false),
            scene(1.5, false, false),
            scene(0.2, false, true),
        ] {
            let mut off = options();
            off.fit_paint_divider = false;
            let (before, _) = extract(&cloud, &trace(), &off).unwrap();
            let (after, report) = extract(&cloud, &trace(), &options()).unwrap();
            assert!(!report.paint_divider.unwrap().applied);
            for (a, b) in before.iter().zip(&after) {
                assert_eq!(a.boundaries, b.boundaries);
                assert_eq!(a.evidence, b.evidence);
            }
        }
        let cloud = scene(0.2, false, false);
        let wrong = BuildOptions {
            backward_lanes: 2,
            ..options()
        };
        assert!(
            !extract(&cloud, &trace(), &wrong)
                .unwrap()
                .1
                .paint_divider
                .unwrap()
                .applied
        );
        let mut invalid = cloud.clone();
        invalid.colors.as_mut().unwrap().pop();
        assert!(extract(&invalid, &trace(), &options()).is_err());
    }

    #[test]
    fn rotated_source_and_source_footprint_keep_roles_and_check_actual_corrected_geometry() {
        let angle = 0.7_f64;
        let transform = |p: [f64; 3]| {
            [
                7000.0 + p[0] * angle.cos() - p[1] * angle.sin(),
                80000.0 + p[0] * angle.sin() + p[1] * angle.cos(),
                p[2],
            ]
        };
        let mut cloud = scene(0.2, false, false);
        cloud.positions.iter_mut().for_each(|p| *p = transform(*p));
        let trace = trace().map(|mut p| {
            p[1] = -1.4;
            transform(p)
        });
        let o = BuildOptions {
            fit_source_surface: true,
            left_hand_traffic: false,
            ..options()
        };
        let (roads, report) = extract(&cloud, &trace, &o).unwrap();
        let paint = report.paint_divider.unwrap();
        assert!(paint.applied, "{paint:?}");
        assert!(report.surface_fit.unwrap().preserved_candidate_geometry);
        assert!(roads.iter().all(|r| r.boundaries.len() == 3));
        assert!(report.generated_length > 10.0);
    }
}
