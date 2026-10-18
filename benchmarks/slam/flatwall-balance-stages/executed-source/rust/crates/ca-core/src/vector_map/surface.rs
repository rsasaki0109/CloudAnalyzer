//! Optional source-footprint drafting. A coherent low surface near the operator
//! path limits inferred lane widths; missing or occluded intervals are deferred.
//! This does not identify road semantics or infer the configured lane count.
use serde::Serialize;
use vectormap_core::Point3;

use super::junctions::Ground;
use super::{
    BuildError, BuildOptions, BuildReport, Evidence, ExtractedRoad, SurfaceIndex, fitting, length,
    quantile,
};
use crate::PointCloud;

#[derive(Debug, Clone, Serialize)]
pub struct SurfaceFitReport {
    pub preserved_candidate_geometry: bool,
    pub evaluated_path_length_m: f64,
    pub candidate_sections: usize,
    pub rejected_intervals: usize,
    pub discarded_short_stretches: usize,
    pub deferred_length_m: f64,
    pub minimum_lane_width_m: Option<f64>,
    pub maximum_lane_width_m: Option<f64>,
}

struct Footprint {
    left: f64,
    right: f64,
    first: usize,
    last: usize,
    observed_left: bool,
    observed_right: bool,
}

/// Contiguous low-surface bands reject abrupt raised objects. Select the lowest
/// sufficiently wide band within half a lane width of the path, never a narrow
/// low outlier. Heights of vehicle/roof bands must not pull the road upward.
fn footprint(surface: &[Option<f64>], low: f64, o: &BuildOptions) -> Option<Footprint> {
    let lanes = o.forward_lanes + o.backward_lanes;
    let lateral = |i: usize| low + (i as f64 + 0.5) * o.bin_width;
    let mut candidates = Vec::new();
    let mut start = 0;
    for end in 1..=surface.len() {
        if end < surface.len()
            && surface[end]
                .zip(surface[end - 1])
                .is_some_and(|(a, b)| (a - b).abs() <= o.curb_height.min(0.08))
        {
            continue;
        }
        if surface[start].is_some() {
            let right = lateral(start);
            let left = lateral(end - 1);
            let nearest = right.max(-left).max(0.0);
            if left - right >= lanes as f64 * 1.5 && nearest <= o.lane_width * 0.5 {
                let mut values: Vec<_> = surface[start..end].iter().flatten().copied().collect();
                let z = quantile(&mut values, 0.5).unwrap();
                candidates.push((z, nearest, start, end - 1, right, left));
            }
        }
        start = end;
    }
    let (_, _, first, last, right, left) = candidates.into_iter().min_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then(a.1.total_cmp(&b.1))
            .then(a.2.cmp(&b.2))
    })?;
    // Never expand the configured total width. Move the prior only as far as
    // required to fit the observed band; a narrow band reduces inferred widths.
    let width = (left - right).min(lanes as f64 * o.lane_width);
    let nominal_left = if o.left_hand_traffic {
        o.lane_width * 0.5
    } else {
        (lanes as f64 - 0.5) * o.lane_width
    };
    // width <= left-right mathematically, but adding it back can round above
    // left by an ULP. Keep the clamp interval ordered without expanding the band.
    let fitted_left = nominal_left.clamp((right + width).min(left), left);
    Some(Footprint {
        left: fitted_left,
        right: fitted_left - width,
        first,
        last,
        observed_left: (fitted_left - left).abs() < 1e-6,
        observed_right: (fitted_left - width - right).abs() < 1e-6,
    })
}

fn height(surface: &[Option<f64>], low: f64, lateral: f64, o: &BuildOptions, f: &Footprint) -> f64 {
    let index = ((lateral - low) / o.bin_width - 0.5).round() as usize;
    surface[index.clamp(f.first, f.last)].unwrap()
}

/// Check every lane centre and every shared boundary along the actual interval
/// at <=0.5m spacing, including both ends. Do not bridge an unsupported strip.
fn interval_supported(a: &[[f64; 3]], b: &[[f64; 3]], ground: &Ground<'_>) -> bool {
    let check = |a: [f64; 3], b: [f64; 3]| {
        let length = (a[0] - b[0]).hypot(a[1] - b[1]).hypot(a[2] - b[2]);
        let n = (length / 0.5).ceil().max(1.0) as usize;
        (0..=n).all(|k| {
            let p: [f64; 3] = std::array::from_fn(|j| a[j] + (b[j] - a[j]) * k as f64 / n as f64);
            ground.supports(Point3::new(p[0], p[1], p[2]))
        })
    };
    a.iter().zip(b).all(|(&a, &b)| check(a, b))
        && a.windows(2).zip(b.windows(2)).all(|(a, b)| {
            check(
                std::array::from_fn(|j| (a[0][j] + a[1][j]) * 0.5),
                std::array::from_fn(|j| (b[0][j] + b[1][j]) * 0.5),
            )
        })
}

pub(super) fn extract(
    cloud: &PointCloud,
    line: &[[f64; 3]],
    index: &SurfaceIndex<'_>,
    low: f64,
    high: f64,
    o: &BuildOptions,
    mut report: BuildReport,
) -> Result<(Vec<ExtractedRoad>, BuildReport), BuildError> {
    let nlanes = o.forward_lanes + o.backward_lanes;
    let ground = Ground::new(cloud)?;
    let mut fit = SurfaceFitReport {
        preserved_candidate_geometry: false,
        evaluated_path_length_m: length(line),
        candidate_sections: 0,
        rejected_intervals: 0,
        discarded_short_stretches: 0,
        deferred_length_m: 0.0,
        minimum_lane_width_m: None,
        maximum_lane_width_m: None,
    };
    report.unsupported_sections = 0;
    let mut sections = Vec::new();
    for (k, &p) in line.iter().enumerate() {
        let a = line[k.saturating_sub(1)];
        let b = line[(k + 1).min(line.len() - 1)];
        let norm = (b[0] - a[0]).hypot(b[1] - a[1]);
        if norm < 0.1 {
            sections.push(None);
            report.unsupported_sections += 1;
            continue;
        }
        let dir = [(b[0] - a[0]) / norm, (b[1] - a[1]) / norm];
        let bins = index.slice(p, dir, low, high, o);
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
        let Some(f) = footprint(&surface, low, o) else {
            sections.push(None);
            report.unsupported_sections += 1;
            continue;
        };
        let width = (f.left - f.right) / nlanes as f64;
        let points: Vec<[f64; 3]> = (0..=nlanes)
            .map(|j| {
                let lateral = f.left - j as f64 * width;
                [
                    p[0] - dir[1] * lateral,
                    p[1] + dir[0] * lateral,
                    height(&surface, low, lateral, o, &f),
                ]
            })
            .collect();
        // The trace can sit on a curb or beside an occluder. Its Z comes from
        // the chosen low band, never from the pose/sensor or raised trace returns.
        let z = height(&surface, low, 0.0, o, &f);
        fit.candidate_sections += 1;
        let labels = (0..=nlanes)
            .map(|j| {
                if (j == 0 && f.observed_left) || (j == nlanes && f.observed_right) {
                    Evidence::SupportEdge
                } else {
                    Evidence::WidthPrior
                }
            })
            .collect::<Vec<_>>();
        sections.push(Some(([p[0], p[1], z], points, labels)));
    }
    // Only narrow an observed band: bound changes of inferred lane width to
    // 0.15 m per metre of travel, instead of drawing sudden width jumps around
    // vehicle returns. Missing profiles interrupt the taper, not interpolate it.
    let mut widths: Vec<_> = sections
        .iter()
        .map(|s| {
            s.as_ref().map(|(_, p, _)| {
                (p[0][0] - p[nlanes][0]).hypot(p[0][1] - p[nlanes][1]) / nlanes as f64
            })
        })
        .collect();
    for reverse in [false, true] {
        for step in 1..widths.len() {
            let (a, b) = if reverse {
                (widths.len() - step, widths.len() - step - 1)
            } else {
                (step - 1, step)
            };
            if let (Some(previous), Some(current)) = (widths[a], widths[b]) {
                let distance = (line[a][0] - line[b][0]).hypot(line[a][1] - line[b][1]);
                widths[b] = Some(current.min(previous + 0.15 * distance));
            }
        }
    }
    for (section, width) in sections.iter_mut().zip(widths) {
        let (Some((_, points, labels)), Some(width)) = (section, width) else {
            continue;
        };
        let old = (points[0][0] - points[nlanes][0]).hypot(points[0][1] - points[nlanes][1])
            / nlanes as f64;
        if old - width <= 1e-6 {
            continue;
        }
        let left = points[0];
        let right = points[nlanes];
        for (j, point) in points.iter_mut().enumerate() {
            let t = 0.5 + (j as f64 / nlanes as f64 - 0.5) * width / old;
            *point = std::array::from_fn(|a| left[a] + (right[a] - left[a]) * t);
            let p = Point3::new(point[0], point[1], point[2]);
            if let Some(z) = ground.height(p).filter(|z| (z - point[2]).abs() <= 0.3) {
                point[2] = z;
            }
            labels[j] = Evidence::WidthPrior;
        }
    }
    let empty = || ExtractedRoad {
        reference: vec![],
        boundaries: vec![vec![]; nlanes + 1],
        evidence: vec![vec![]; nlanes + 1],
        source_boundaries: None,
    };
    let mut candidate_roads = Vec::new();
    let mut road = empty();
    for k in 0..sections.len() - 1 {
        let Some((a, ap, al)) = sections[k].as_ref() else {
            if road.reference.len() > 1 {
                candidate_roads.push(road);
            }
            road = empty();
            continue;
        };
        let Some((b, bp, bl)) = sections[k + 1].as_ref() else {
            if road.reference.len() > 1 {
                candidate_roads.push(road);
            }
            road = empty();
            continue;
        };
        if road.reference.is_empty() {
            road.reference.push(*a);
            for (j, &p) in ap.iter().enumerate() {
                road.boundaries[j].push(p);
                road.evidence[j].push(al[j]);
            }
        }
        road.reference.push(*b);
        for (j, &p) in bp.iter().enumerate() {
            road.boundaries[j].push(p);
            road.evidence[j].push(bl[j]);
        }
    }
    if road.reference.len() > 1 {
        candidate_roads.push(road);
    }
    let mut roads = Vec::new();
    for mut candidate in candidate_roads {
        if o.fit_boundaries {
            let original = candidate.clone();
            fitting::fit(&mut candidate);
            if candidate.boundaries.windows(2).any(|pair| {
                pair[0].iter().zip(&pair[1]).any(|(a, b)| {
                    let width = (a[0] - b[0]).hypot(a[1] - b[1]);
                    width < 1.5 - 1e-9 || width > o.lane_width + 1e-9
                })
            }) {
                candidate = original;
            }
        }
        let mut road = empty();
        for k in 0..candidate.reference.len() - 1 {
            let a: Vec<_> = candidate.boundaries.iter().map(|b| b[k]).collect();
            let b: Vec<_> = candidate.boundaries.iter().map(|b| b[k + 1]).collect();
            if !interval_supported(&a, &b, &ground) {
                fit.rejected_intervals += 1;
                if road.reference.len() > 1 {
                    roads.push(road);
                }
                road = empty();
                continue;
            }
            if road.reference.is_empty() {
                append(&mut road, &candidate, k);
            }
            append(&mut road, &candidate, k + 1);
        }
        if road.reference.len() > 1 {
            roads.push(road);
        }
    }
    finish(roads, fit, report)
}

/// Preserve the incoming shape when at least 60% of its length already has
/// source support. Only trim/split missing intervals. This avoids squeezing
/// sound multi-carriageway geometry into one low band across a raised median.
pub(super) fn refine(
    cloud: &PointCloud,
    line: &[[f64; 3]],
    index: &SurfaceIndex<'_>,
    range: [f64; 2],
    o: &BuildOptions,
    report: BuildReport,
    candidates: Vec<ExtractedRoad>,
) -> Result<(Vec<ExtractedRoad>, BuildReport), BuildError> {
    let ground = Ground::new(cloud)?;
    let mut flags = Vec::new();
    let mut supported_length = 0.0;
    let mut candidate_length = 0.0;
    for road in &candidates {
        let mut good = Vec::new();
        for k in 0..road.reference.len() - 1 {
            let a: Vec<_> = road.boundaries.iter().map(|b| b[k]).collect();
            let b: Vec<_> = road.boundaries.iter().map(|b| b[k + 1]).collect();
            let supported = interval_supported(&a, &b, &ground);
            let distance = length(&road.reference[k..=k + 1]);
            candidate_length += distance;
            if supported {
                supported_length += distance;
            }
            good.push(supported);
        }
        flags.push(good);
    }
    // A paint fit must retain its measured spacing/labels. Trim unsupported
    // intervals even below 60%; never silently replace it with width priors.
    if supported_length < candidate_length * 0.6
        && !report.paint_corridor.as_ref().is_some_and(|p| p.applied)
        && !report.paint_divider.as_ref().is_some_and(|p| p.applied)
    {
        drop(ground);
        return extract(cloud, line, index, range[0], range[1], o, report);
    }
    let nlanes = o.forward_lanes + o.backward_lanes;
    let empty = || ExtractedRoad {
        reference: vec![],
        boundaries: vec![vec![]; nlanes + 1],
        evidence: vec![vec![]; nlanes + 1],
        source_boundaries: None,
    };
    let mut roads = Vec::new();
    let mut fit = SurfaceFitReport {
        preserved_candidate_geometry: true,
        evaluated_path_length_m: length(line),
        candidate_sections: candidates.iter().map(|r| r.reference.len()).sum(),
        rejected_intervals: 0,
        discarded_short_stretches: 0,
        deferred_length_m: 0.0,
        minimum_lane_width_m: None,
        maximum_lane_width_m: None,
    };
    for (candidate, good) in candidates.iter().zip(flags) {
        let mut road = empty();
        for (k, good) in good.into_iter().enumerate() {
            if !good {
                fit.rejected_intervals += 1;
                if road.reference.len() > 1 {
                    roads.push(road);
                }
                road = empty();
                continue;
            }
            if road.reference.is_empty() {
                append(&mut road, candidate, k);
            }
            append(&mut road, candidate, k + 1);
        }
        if road.reference.len() > 1 {
            roads.push(road);
        }
    }
    finish(roads, fit, report)
}

fn finish(
    mut roads: Vec<ExtractedRoad>,
    mut fit: SurfaceFitReport,
    mut report: BuildReport,
) -> Result<(Vec<ExtractedRoad>, BuildReport), BuildError> {
    report.intensity_vertices = 0;
    report.rgb_paint_vertices = 0;
    report.curb_vertices = 0;
    report.support_edge_vertices = 0;
    report.width_prior_vertices = 0;
    report.fitted_vertices = 0;
    report.maximum_fit_displacement = 0.0;
    report.observed_fraction.fill(0.0);
    if !fit.preserved_candidate_geometry {
        report.anchored_prior_vertices = 0;
        report.tracked_vertices = 0;
    }
    report.warnings.clear();
    if report.coverage_edge_anchor_candidates_ignored > 0 {
        report.warnings.push(format!("{} pre-tracking coverage-edge anchor candidates were excluded from inferred offsets before source-footprint trimming. Coverage limits are not physical road-boundary observations.", report.coverage_edge_anchor_candidates_ignored));
    }
    let before = roads.len();
    roads.retain(|r| length(&r.reference) >= 2.0 - 1e-9);
    fit.discarded_short_stretches = before - roads.len();
    // Preserve widths/height evidence and report lost extent explicitly. A zero
    // review count after filtering is not an accuracy score or a complete road.
    report.roads = roads.len();
    report.generated_length = roads.iter().map(|r| length(&r.reference)).sum();
    fit.deferred_length_m = (fit.evaluated_path_length_m - report.generated_length).max(0.0);
    for road in &roads {
        for (j, labels) in road.evidence.iter().enumerate() {
            report.observed_fraction[j] += labels
                .iter()
                .filter(|&&e| e != Evidence::WidthPrior)
                .count() as f64;
            for &e in labels {
                match e {
                    Evidence::RgbPaint => report.rgb_paint_vertices += 1,
                    Evidence::SupportEdge => report.support_edge_vertices += 1,
                    Evidence::WidthPrior => report.width_prior_vertices += 1,
                    Evidence::Curb => report.curb_vertices += 1,
                    Evidence::Intensity => report.intensity_vertices += 1,
                }
            }
        }
        for pair in road.boundaries.windows(2) {
            for (a, b) in pair[0].iter().zip(&pair[1]) {
                let width = (a[0] - b[0]).hypot(a[1] - b[1]);
                fit.minimum_lane_width_m =
                    Some(fit.minimum_lane_width_m.map_or(width, |v| v.min(width)));
                fit.maximum_lane_width_m =
                    Some(fit.maximum_lane_width_m.map_or(width, |v| v.max(width)));
            }
        }
        if let Some(source) = &road.source_boundaries {
            for (a, b) in road
                .boundaries
                .iter()
                .flatten()
                .zip(source.iter().flatten())
            {
                let d = (a[0] - b[0]).hypot(a[1] - b[1]);
                if d > 1e-6 {
                    report.fitted_vertices += 1;
                    report.maximum_fit_displacement = report.maximum_fit_displacement.max(d);
                }
            }
        }
    }
    let total = roads.iter().map(|r| r.reference.len()).sum::<usize>();
    if total > 0 {
        report
            .observed_fraction
            .iter_mut()
            .for_each(|v| *v /= total as f64);
    }
    report.intensity_used = report.intensity_vertices > 0;
    report.warnings.push(if report.lane_edge_inference.as_ref().is_some_and(|p| p.applied) { "Source-footprint mode checked the corrected interior and inferred outer lane edge, then trimmed/split unsupported intervals. Original curb candidates are retained separately in the lane-edge report; the inferred outer line is a configured-width assumption, not observed paint.".into() } else if report.paint_divider.as_ref().is_some_and(|p| p.applied) { "Source-footprint mode retained outside candidate geometry and the corrected interior paint line, then trimmed/split unsupported intervals. Divider correction can change individual lane widths. Counts and directions remain manual.".into() } else if fit.preserved_candidate_geometry {"Source-footprint mode preserved incoming source-supported geometry and trimmed/split unsupported intervals. No lane width was automatically narrowed. This does not certify survey accuracy, road semantics or permitted turns.".into()} else {"Source-footprint mode keeps explicit lane counts but fits inferred widths to a coherent low-surface band. Coverage limits can be occlusion or scan gaps, not road edges. Internal lane lines remain width assumptions; this does not identify road semantics or permitted turns.".into()});
    report.warnings.push(format!("{:.2} m of the traced/recorded path was deferred; {} intervals failed centre/boundary source checks; {} fragments shorter than 2 m were deferred. No missing surface was filled and no reference map was used.",fit.deferred_length_m,fit.rejected_intervals,fit.discarded_short_stretches));
    report.surface_fit = Some(fit);
    super::paint_corridor::warnings(&mut report);
    super::paint_divider::warnings(&mut report);
    super::lane_edge_inference::warnings(&mut report);
    if roads.is_empty() {
        return Err(BuildError("No source-supported road intervals remain; inspect path placement, lane counts, width assumptions and occlusion. The existing map is unchanged.".into()));
    }
    Ok((roads, report))
}

fn append(road: &mut ExtractedRoad, candidate: &ExtractedRoad, k: usize) {
    road.reference.push(candidate.reference[k]);
    for (j, line) in candidate.boundaries.iter().enumerate() {
        road.boundaries[j].push(line[k]);
        road.evidence[j].push(candidate.evidence[j][k]);
    }
    if let Some(source) = &candidate.source_boundaries {
        let output = road
            .source_boundaries
            .get_or_insert_with(|| vec![vec![]; source.len()]);
        for (j, line) in source.iter().enumerate() {
            output[j].push(line[k]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{build, quality};
    use super::*;
    use vectormap_core::Map;

    fn scene() -> PointCloud {
        let mut cloud = PointCloud::default();
        for x in -10..=210 {
            for y in -45..=20 {
                let x = x as f64 * 0.1;
                let y = y as f64 * 0.1;
                let z = 2.0 + 0.01 * x + if y > 0.2 { 2.0 } else { 0.0 };
                cloud.positions.push([x, y, z]);
            }
        }
        cloud
    }
    #[test]
    fn band_width_roundoff_does_not_invert_the_fitting_interval() {
        for left_hand_traffic in [true, false] {
            let options = BuildOptions {
                bin_width: 0.2,
                left_hand_traffic,
                ..BuildOptions::default()
            };
            let low = -3.45;
            let surface = vec![Some(2.0); 23];
            let right = low + 0.5 * options.bin_width;
            let left = low + 22.5 * options.bin_width;
            // Reconstructing this mathematically identical endpoint rounds up.
            assert!(right + (left - right) > left);
            let fitted = footprint(&surface, low, &options).unwrap();
            assert!(fitted.left <= left);
            assert!(fitted.right >= right - 1e-12);
            assert!((fitted.left - fitted.right - (left - right)).abs() < 1e-12);
            assert!(fitted.observed_left && fitted.observed_right);
        }
    }
    #[test]
    fn narrow_low_band_fits_widths_without_following_raised_trace_returns() {
        let cloud = scene();
        let poses = [[0., 0., 50.], [20., 0., 50.]];
        let mut legacy = Map::new();
        let mut fitted = Map::new();
        build(&mut legacy, &cloud, &poses, &BuildOptions::default()).unwrap();
        assert!(
            !quality::audit(&legacy, &cloud)
                .unwrap()
                .low_support_lanes
                .is_empty()
        );
        let report = build(
            &mut fitted,
            &cloud,
            &poses,
            &BuildOptions {
                fit_source_surface: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            quality::audit(&fitted, &cloud)
                .unwrap()
                .low_support_lanes
                .is_empty()
        );
        assert_eq!(fitted.lanes().count(), 2);
        assert!(report.surface_fit.unwrap().maximum_lane_width_m.unwrap() < 3.5);
        assert!(
            fitted
                .boundaries()
                .flat_map(|b| &b.geometry.points)
                .all(|p| p.z < 2.3)
        );
    }
    #[test]
    fn occluded_strip_is_deferred_and_not_bridged() {
        let mut cloud = scene();
        cloud.positions.retain(|p| p[0] < 8.0 || p[0] > 12.0);
        let mut map = Map::new();
        let report = build(
            &mut map,
            &cloud,
            &[[0., 0., 50.], [20., 0., 50.]],
            &BuildOptions {
                fit_source_surface: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(report.roads >= 2);
        assert!(report.surface_fit.unwrap().deferred_length_m >= 4.0);
        assert!(
            quality::audit(&map, &cloud)
                .unwrap()
                .low_support_lanes
                .is_empty()
        );
    }
    #[test]
    fn unsupported_width_and_narrow_low_outlier_do_not_change_existing_map() {
        let mut cloud = PointCloud::default();
        for x in 0..=200 {
            for y in -4..=4 {
                cloud.positions.push([x as f64 * 0.1, y as f64 * 0.1, 2.0]);
            }
        }
        let mut map = Map::new();
        map.build_road(vectormap_core::NewRoad::new(
            vectormap_core::Polyline3::new(vec![
                Point3::new(100., 100., 2.),
                Point3::new(110., 100., 2.),
            ]),
            vec![vectormap_core::RoadLane::new(
                3.5,
                vectormap_core::LaneDirection::Forward,
            )],
        ))
        .unwrap();
        let original = map.clone();
        assert!(
            build(
                &mut map,
                &cloud,
                &[[0., 0., 30.], [20., 0., 30.]],
                &BuildOptions {
                    fit_source_surface: true,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert_eq!(map, original);
    }

    #[test]
    fn supported_multi_lane_geometry_is_preserved_across_raised_median() {
        let mut cloud = PointCloud::default();
        for x in -10..=210 {
            for y in -210..=40 {
                let y = y as f64 * 0.1;
                cloud.positions.push([
                    x as f64 * 0.1,
                    y,
                    if (-9.0..=-8.0).contains(&y) { 2.2 } else { 2.0 },
                ]);
            }
        }
        let options = BuildOptions {
            forward_lanes: 3,
            backward_lanes: 3,
            ..Default::default()
        };
        let poses = [[0., 0., 50.], [20., 0., 50.]];
        let mut before = Map::new();
        build(&mut before, &cloud, &poses, &options).unwrap();
        let mut after = Map::new();
        let report = build(
            &mut after,
            &cloud,
            &poses,
            &BuildOptions {
                fit_source_surface: true,
                ..options
            },
        )
        .unwrap();
        assert_eq!(before, after);
        assert!(report.surface_fit.unwrap().preserved_candidate_geometry);
    }

    #[test]
    fn translated_rotated_right_hand_scene_keeps_supported_heights_and_lane_counts() {
        let mut cloud = scene();
        let transform = |p: [f64; 3]| [100000.0 - p[1], 200000.0 + p[0], p[2]];
        cloud.positions.iter_mut().for_each(|p| *p = transform(*p));
        let mut map = Map::new();
        let report = build(
            &mut map,
            &cloud,
            &[transform([0., 0., 50.]), transform([20., 0., 50.])],
            &BuildOptions {
                fit_source_surface: true,
                left_hand_traffic: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(map.lanes().count(), 2);
        assert!(
            quality::audit(&map, &cloud)
                .unwrap()
                .low_support_lanes
                .is_empty()
        );
        assert!(report.surface_fit.unwrap().deferred_length_m < 1e-6);
    }
}
