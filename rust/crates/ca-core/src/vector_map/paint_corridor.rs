//! Source-only parallel paint fits. This does not infer lane legality or counts.
use super::{BuildOptions, BuildReport, Evidence, ExtractedRoad, quantile};
use crate::PointCloud;
use serde::Serialize;
use std::collections::HashMap;

const MAX_ROI: usize = 250_000;
const MAX_WHITE: usize = 50_000;
const MAX_PAINT: usize = 10_000;
const MAX_NEIGHBOURS: usize = 4096;

#[derive(Debug, Clone, Serialize)]
pub struct PaintTrackReport {
    pub lateral_intercept_m: f64,
    pub source_points: usize,
    pub strong: bool,
    /// Clipped longitudinal component intervals, not continuous observations.
    pub observed_intervals_m: Vec<[f64; 2]>,
    pub observed_length_m: f64,
    pub interpolated_length_m: f64,
    pub extrapolated_length_m: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PaintCorridorReport {
    pub applied: bool,
    pub reason: String,
    pub limited: bool,
    pub roi_points: usize,
    pub white_candidates: usize,
    pub contrasted_points: usize,
    pub longitudinal_components: usize,
    pub eligible_bundles: usize,
    pub heading_correction_degrees: Option<f64>,
    pub measured_lane_widths_m: Vec<f64>,
    pub residual_p90_m: Option<f64>,
    /// Ordered left to right, like output boundaries. Lane roles remain manual.
    pub tracks: Vec<PaintTrackReport>,
}

impl Default for PaintCorridorReport {
    fn default() -> Self {
        Self {
            applied: false,
            reason: "no uniquely supported parallel paint corridor".into(),
            limited: false,
            roi_points: 0,
            white_candidates: 0,
            contrasted_points: 0,
            longitudinal_components: 0,
            eligible_bundles: 0,
            heading_correction_degrees: None,
            measured_lane_widths_m: vec![],
            residual_p90_m: None,
            tracks: vec![],
        }
    }
}

// Coordinates relative to the straight trace: longitudinal s, lateral t, source z.
pub(super) struct Samples {
    pub(super) points: Vec<[f64; 3]>,
    pub(super) ids: Vec<usize>,
    cells: HashMap<(i64, i64), Vec<usize>>,
}
impl Samples {
    fn new(points: Vec<[f64; 3]>, ids: Vec<usize>) -> Self {
        let mut cells: HashMap<_, Vec<_>> = HashMap::new();
        for (i, p) in points.iter().enumerate() {
            cells.entry(Self::cell(p)).or_default().push(i);
        }
        Self { points, ids, cells }
    }
    fn cell(p: &[f64; 3]) -> (i64, i64) {
        ((p[0] / 0.5).floor() as i64, (p[1] / 0.5).floor() as i64)
    }
    fn nearby(&self, p: &[f64; 3], radius: f64) -> Option<Vec<usize>> {
        let (x, y) = Self::cell(p);
        let reach = (radius / 0.5).ceil() as i64;
        let mut result = Vec::new();
        let mut inspected = 0;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                if let Some(ids) = self.cells.get(&(x + dx, y + dy)) {
                    inspected += ids.len();
                    if inspected > MAX_NEIGHBOURS {
                        return None;
                    }
                    result.extend(ids.iter().copied().filter(|&i| {
                        let q = self.points[i];
                        (q[0] - p[0]).hypot(q[1] - p[1]) <= radius
                    }));
                }
            }
        }
        Some(result)
    }
    pub(super) fn ground(&self, p: &[f64; 3]) -> Option<Option<f64>> {
        let ids = self.nearby(p, 0.75)?;
        Some(if ids.len() >= 3 {
            quantile(
                &mut ids.iter().map(|&i| self.points[i][2]).collect::<Vec<_>>(),
                0.15,
            )
        } else {
            None
        })
    }
}

struct Component {
    ids: Vec<usize>,
    interval: [f64; 2],
    slope: f64,
}
pub(super) struct Track {
    pub(super) ids: Vec<usize>,
    pub(super) intervals: Vec<[f64; 2]>,
    pub(super) intercept: f64,
}

fn intervals(mut values: Vec<[f64; 2]>, length: f64) -> Vec<[f64; 2]> {
    values.iter_mut().for_each(|r| {
        r[0] = r[0].clamp(0.0, length);
        r[1] = r[1].clamp(0.0, length);
    });
    values.retain(|r| r[1] > r[0]);
    values.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let mut merged: Vec<[f64; 2]> = vec![];
    for v in values {
        if let Some(last) = merged.last_mut()
            && v[0] <= last[1]
        {
            last[1] = last[1].max(v[1]);
            continue;
        }
        merged.push(v);
    }
    merged
}

pub(super) fn track_report(t: &Track, length: f64) -> PaintTrackReport {
    let observed = intervals(t.intervals.clone(), length);
    let sum = observed.iter().map(|r| r[1] - r[0]).sum::<f64>();
    let span = observed
        .first()
        .zip(observed.last())
        .map_or(0.0, |(a, b)| b[1] - a[0]);
    PaintTrackReport {
        lateral_intercept_m: t.intercept,
        source_points: t.ids.len(),
        strong: span >= length * 0.5 && sum >= length * 0.1 && t.ids.len() >= 15,
        observed_intervals_m: observed,
        observed_length_m: sum,
        interpolated_length_m: (span - sum).max(0.0),
        extrapolated_length_m: (length - span).max(0.0),
    }
}

fn components(paint: &Samples) -> Option<Vec<Component>> {
    let mut visited = vec![false; paint.points.len()];
    let mut result = vec![];
    for seed in 0..paint.points.len() {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        let mut ids = vec![seed];
        let mut cursor = 0;
        while cursor < ids.len() {
            for id in paint.nearby(&paint.points[ids[cursor]], 0.3)? {
                if !visited[id] {
                    visited[id] = true;
                    ids.push(id);
                }
            }
            cursor += 1;
        }
        if ids.len() < 5 {
            continue;
        }
        let mean =
            [0, 1].map(|a| ids.iter().map(|&i| paint.points[i][a]).sum::<f64>() / ids.len() as f64);
        let mut xx = 0.0;
        let mut yy = 0.0;
        let mut xy = 0.0;
        for &i in &ids {
            let p = paint.points[i];
            xx += (p[0] - mean[0]).powi(2);
            yy += (p[1] - mean[1]).powi(2);
            xy += (p[0] - mean[0]) * (p[1] - mean[1]);
        }
        let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
        if angle.abs() > 15_f64.to_radians() {
            continue;
        }
        let d = [angle.cos(), angle.sin()];
        let mut long: Vec<_> = ids
            .iter()
            .map(|&i| paint.points[i][0] * d[0] + paint.points[i][1] * d[1])
            .collect();
        let mut wide: Vec<_> = ids
            .iter()
            .map(|&i| -paint.points[i][0] * d[1] + paint.points[i][1] * d[0])
            .collect();
        let extent = quantile(&mut long, 0.95).unwrap() - quantile(&mut long, 0.05).unwrap();
        let width = quantile(&mut wide, 0.95).unwrap() - quantile(&mut wide, 0.05).unwrap();
        if extent < 0.75 || width > 0.3 {
            continue;
        }
        let interval = [
            ids.iter()
                .map(|&i| paint.points[i][0])
                .fold(f64::INFINITY, f64::min),
            ids.iter()
                .map(|&i| paint.points[i][0])
                .fold(f64::NEG_INFINITY, f64::max),
        ];
        result.push(Component {
            ids,
            interval,
            slope: angle.tan(),
        });
    }
    Some(result)
}

pub(super) struct Scan {
    pub samples: Samples,
    pub paint: Samples,
    pub tracks: Vec<Track>,
    pub origin: [f64; 3],
    pub d: [f64; 2],
    pub normal: [f64; 2],
    pub length: f64,
    pub initial_slope: f64,
    pub report: PaintCorridorReport,
}

pub(super) fn scan(
    cloud: &PointCloud,
    line: &[[f64; 3]],
    o: &BuildOptions,
    minimum_parts: usize,
) -> Result<Scan, Box<PaintCorridorReport>> {
    let mut report = PaintCorridorReport::default();
    let hold = |reason: &str, mut r: PaintCorridorReport| {
        r.reason = reason.into();
        Err(Box::new(r))
    };
    let limited = |mut r: PaintCorridorReport| {
        r.limited = true;
        hold(
            "paint fit budget exceeded; no corridor selected from a partial scan",
            r,
        )
    };
    let Some(colors) = &cloud.colors else {
        return hold("source has no RGB colors", report);
    };
    let origin = line[0];
    let end = line[line.len() - 1];
    let length = (end[0] - origin[0]).hypot(end[1] - origin[1]);
    if length < 2.0 {
        return hold("straight trace must be at least two metres", report);
    }
    let d = [(end[0] - origin[0]) / length, (end[1] - origin[1]) / length];
    let normal = [-d[1], d[0]];
    if line.windows(2).any(|w| {
        let dx = w[1][0] - w[0][0];
        let dy = w[1][1] - w[0][1];
        (dx * d[0] + dy * d[1]) / dx.hypot(dy) < 5_f64.to_radians().cos()
    }) {
        return hold("trace is not straight within five degrees", report);
    }
    let count = o.forward_lanes + o.backward_lanes;
    let reach = 2.0 * count as f64 * o.lane_width;
    let brightness = |i: usize| f64::from(*colors[i].iter().min().unwrap());
    let mut points = vec![];
    let mut ids = vec![];
    let mut bright_min = f64::INFINITY;
    let mut bright_max = f64::NEG_INFINITY;
    for (id, p) in cloud.positions.iter().enumerate() {
        if !p.iter().all(|v| v.is_finite()) {
            continue;
        }
        let dx = p[0] - origin[0];
        let dy = p[1] - origin[1];
        let s = dx * d[0] + dy * d[1];
        let t = dx * normal[0] + dy * normal[1];
        if (-3.0..=length + 3.0).contains(&s) && t.abs() <= reach {
            points.push([s, t, p[2]]);
            ids.push(id);
            bright_min = bright_min.min(brightness(id));
            bright_max = bright_max.max(brightness(id));
            report.roi_points += 1;
            if points.len() > MAX_ROI {
                return limited(report);
            }
        }
    }
    if bright_max - bright_min < 40.0 {
        return hold("RGB has no usable local paint contrast", report);
    }
    let samples = Samples::new(points, ids);
    let mut paint_points = vec![];
    let mut paint_ids = vec![];
    for (i, p) in samples.points.iter().enumerate() {
        let value = brightness(samples.ids[i]);
        if value < 180.0 {
            continue;
        }
        report.white_candidates += 1;
        if report.white_candidates > MAX_WHITE {
            return limited(report);
        }
        let Some(near) = samples.nearby(p, 0.8) else {
            return limited(report);
        };
        let mut heights: Vec<_> = near
            .iter()
            .filter(|&&j| (samples.points[j][0] - p[0]).hypot(samples.points[j][1] - p[1]) <= 0.3)
            .map(|&j| samples.points[j][2])
            .collect();
        if heights.len() < 3 || (p[2] - quantile(&mut heights, 0.2).unwrap()).abs() > 0.12 {
            continue;
        }
        let Some(ground) = samples.ground(&[p[0], 0.0, 0.0]) else {
            return limited(report);
        };
        if !ground.is_some_and(|z| (p[2] - z).abs() <= 0.3) {
            continue;
        }
        let mut sides = [vec![], vec![]];
        for j in near {
            let q = samples.points[j];
            let dist = (q[0] - p[0]).hypot(q[1] - p[1]);
            if dist <= 0.3 || (q[2] - p[2]).abs() > 0.12 || (q[1] - p[1]).abs() <= 0.15 {
                continue;
            }
            sides[usize::from(q[1] > p[1])].push(brightness(samples.ids[j]));
        }
        if sides
            .iter_mut()
            .any(|side| side.len() < 3 || value - quantile(side, 0.5).unwrap() < 40.0)
        {
            continue;
        }
        paint_points.push(*p);
        paint_ids.push(samples.ids[i]);
        if paint_points.len() > MAX_PAINT {
            return limited(report);
        }
    }
    report.contrasted_points = paint_points.len();
    let paint = Samples::new(paint_points, paint_ids);
    let Some(mut parts) = components(&paint) else {
        return limited(report);
    };
    report.longitudinal_components = parts.len();
    if parts.len() < minimum_parts {
        return hold("too few narrow longitudinal paint components", report);
    }
    let mut slope_values: Vec<_> = parts.iter().map(|c| c.slope).collect();
    let initial_slope = quantile(&mut slope_values, 0.5).unwrap();
    let intercept = |c: &Component| {
        quantile(
            &mut c
                .ids
                .iter()
                .map(|&i| paint.points[i][1] - initial_slope * paint.points[i][0])
                .collect::<Vec<_>>(),
            0.5,
        )
        .unwrap()
    };
    parts.sort_by(|a, b| intercept(a).total_cmp(&intercept(b)));
    let mut tracks: Vec<Track> = vec![];
    for c in parts {
        let v = intercept(&c);
        if let Some(t) = tracks.last_mut()
            && (v - t.intercept).abs() <= 0.3
        {
            t.ids.extend(c.ids);
            t.intervals.push(c.interval);
            t.intercept = quantile(
                &mut t
                    .ids
                    .iter()
                    .map(|&i| paint.points[i][1] - initial_slope * paint.points[i][0])
                    .collect::<Vec<_>>(),
                0.5,
            )
            .unwrap();
            continue;
        }
        tracks.push(Track {
            ids: c.ids,
            intervals: vec![c.interval],
            intercept: v,
        });
    }
    Ok(Scan {
        samples,
        paint,
        tracks,
        origin,
        d,
        normal,
        length,
        initial_slope,
        report,
    })
}

/// All limits fail closed for this optional fit, leaving the original extractor
/// available. A partial scan never establishes a unique corridor.
pub(super) fn fit(
    cloud: &PointCloud,
    line: &[[f64; 3]],
    o: &BuildOptions,
) -> (Option<Vec<ExtractedRoad>>, PaintCorridorReport) {
    let Scan {
        samples,
        paint,
        mut tracks,
        origin,
        d,
        normal,
        length,
        initial_slope,
        mut report,
    } = match scan(cloud, line, o, o.forward_lanes + o.backward_lanes + 1) {
        Ok(scan) => scan,
        Err(report) => return (None, *report),
    };
    let count = o.forward_lanes + o.backward_lanes;
    let hold = |reason: &str, mut r: PaintCorridorReport| {
        r.reason = reason.into();
        (None, r)
    };
    let limited = |mut r: PaintCorridorReport| {
        r.limited = true;
        hold(
            "paint fit budget exceeded; no corridor selected from a partial scan",
            r,
        )
    };
    let mut bundles = vec![];
    for start in 0..tracks.len().saturating_sub(count) {
        let bundle = &tracks[start..=start + count];
        let widths: Vec<_> = bundle
            .windows(2)
            .map(|w| (w[1].intercept - w[0].intercept) / (1.0 + initial_slope.powi(2)).sqrt())
            .collect();
        let mean = widths.iter().sum::<f64>() / count as f64;
        if widths.iter().any(|&w| {
            w < 1.5
                || w > 6.0_f64.min(o.lane_width + o.search_margin)
                || (w - mean).abs() > 0.2 * mean
        }) || bundle[0].intercept > o.lane_width * 0.5
            || bundle[count].intercept < -o.lane_width * 0.5
            || bundle
                .iter()
                .filter(|t| track_report(t, length).strong)
                .count()
                < count
        {
            continue;
        }
        bundles.push(start);
    }
    report.eligible_bundles = bundles.len();
    if bundles.len() != 1 {
        return hold(
            "need one unambiguous bundle with at least one strong track per configured lane",
            report,
        );
    }
    let start = bundles[0];
    let selected = &mut tracks[start..=start + count];
    // Demeaning separately per line prevents sparse outer paint from setting
    // heading or lane spacing. No reference/survey geometry enters this fit.
    let mut xx = 0.0;
    let mut xy = 0.0;
    for t in selected.iter() {
        let mean = [0, 1]
            .map(|a| t.ids.iter().map(|&i| paint.points[i][a]).sum::<f64>() / t.ids.len() as f64);
        for &i in &t.ids {
            let p = paint.points[i];
            xx += (p[0] - mean[0]).powi(2);
            xy += (p[0] - mean[0]) * (p[1] - mean[1]);
        }
    }
    let slope = xy / xx.max(1e-12);
    if slope.atan().abs() > 15_f64.to_radians() {
        return hold("parallel paint fit exceeds heading limit", report);
    }
    let mut residuals = vec![];
    for t in selected.iter_mut() {
        t.intercept = quantile(
            &mut t
                .ids
                .iter()
                .map(|&i| paint.points[i][1] - slope * paint.points[i][0])
                .collect::<Vec<_>>(),
            0.5,
        )
        .unwrap();
        residuals.extend(
            t.ids
                .iter()
                .map(|&i| (paint.points[i][1] - t.intercept - slope * paint.points[i][0]).abs()),
        );
    }
    let residual = quantile(&mut residuals, 0.9).unwrap();
    if residual > 0.2 {
        return hold("parallel paint residual is too large", report);
    }
    report.residual_p90_m = Some(residual);
    report.heading_correction_degrees = Some(slope.atan().to_degrees());
    report.measured_lane_widths_m = selected
        .windows(2)
        .rev()
        .map(|w| (w[1].intercept - w[0].intercept) / (1.0 + slope.powi(2)).sqrt())
        .collect();
    let mean = report.measured_lane_widths_m.iter().sum::<f64>() / count as f64;
    if report.measured_lane_widths_m.iter().any(|&w| {
        w < 1.5 || w > 6.0_f64.min(o.lane_width + o.search_margin) || (w - mean).abs() > 0.2 * mean
    }) {
        return hold("refitted paint spacing is inconsistent", report);
    }
    report.tracks = selected
        .iter()
        .rev()
        .map(|t| track_report(t, length))
        .collect();
    let empty = || ExtractedRoad {
        reference: vec![],
        boundaries: vec![vec![]; count + 1],
        evidence: vec![vec![]; count + 1],
        source_boundaries: Some(vec![vec![]; count + 1]),
    };
    let mut road = empty();
    let mut roads = vec![];
    for p in line {
        let s = (p[0] - origin[0]) * d[0] + (p[1] - origin[1]) * d[1];
        let mut vertices = vec![];
        let Some(ground) = samples.ground(&[s, 0.0, 0.0]) else {
            return limited(report);
        };
        for t in selected.iter().rev() {
            let lateral = t.intercept + slope * s;
            let Some(z) = samples.ground(&[s, lateral, 0.0]) else {
                return limited(report);
            };
            let Some(z) = z else {
                break;
            };
            let point = [
                origin[0] + d[0] * s + normal[0] * lateral,
                origin[1] + d[1] * s + normal[1] * lateral,
                z,
            ];
            let near = t
                .ids
                .iter()
                .copied()
                .filter(|&i| (paint.points[i][0] - s).hypot(paint.points[i][1] - lateral) <= 0.5)
                .min_by(|&a, &b| {
                    (paint.points[a][0] - s)
                        .abs()
                        .total_cmp(&(paint.points[b][0] - s).abs())
                });
            let (label, source) = near.map_or((Evidence::WidthPrior, point), |i| {
                (Evidence::RgbPaint, cloud.positions[paint.ids[i]])
            });
            vertices.push((point, label, source));
        }
        if vertices.len() != count + 1 || ground.is_none() {
            if road.reference.len() >= 2 {
                roads.push(road);
            }
            road = empty();
            continue;
        }
        road.reference.push([p[0], p[1], ground.unwrap()]);
        for (j, (point, label, source)) in vertices.into_iter().enumerate() {
            road.boundaries[j].push(point);
            road.evidence[j].push(label);
            road.source_boundaries.as_mut().unwrap()[j].push(source);
        }
    }
    if road.reference.len() >= 2 {
        roads.push(road);
    }
    if roads.is_empty() {
        return hold("no continuous source-supported painted geometry", report);
    }
    report.applied = true;
    report.reason =
        "unique locally contrasted parallel paint bundle; counts and directions remain manual"
            .into();
    (Some(roads), report)
}

pub(super) fn warnings(report: &mut BuildReport) {
    if let Some(paint) = &report.paint_corridor {
        report.warnings.push(if paint.applied {
            "RGB paint set parallel boundary spacing and heading. Only nearby observed paint vertices have RGB evidence; gaps and extensions remain inferred. Track interval statistics describe the source fit before source-footprint trimming. Lane counts, boundary roles and travel directions still require review.".into()
        } else {
            format!("RGB paint corridor was held: {}. Existing extraction rules were used; this is not a negative road-marking diagnosis.", paint.reason)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::{build, extract, quality};
    use super::*;
    use vectormap_core::Map;

    fn scene(lines: &[f64], sparse_outer: bool) -> PointCloud {
        let mut cloud = PointCloud::default();
        let mut colors = vec![];
        for sx in -20_i32..=320 {
            let x = sx as f64 * 0.1;
            for sy in -90..=100 {
                let y = sy as f64 * 0.1;
                let white = lines.iter().enumerate().any(|(j, &t)| {
                    (y - (t - 0.04 * x)).abs() < 0.055
                        && if sparse_outer && j == lines.len() - 1 {
                            (4.0..=6.0).contains(&x)
                        } else {
                            sx.rem_euclid(80) < 40
                        }
                });
                cloud.positions.push([x, y, 12.0 + x * 0.02]);
                colors.push(if white { [230; 3] } else { [70; 3] });
            }
        }
        cloud.colors = Some(colors);
        cloud
    }
    fn options() -> BuildOptions {
        BuildOptions {
            fit_paint_corridor: true,
            fit_source_surface: true,
            segment_length: 0.0,
            ..Default::default()
        }
    }
    fn trace() -> [[f64; 3]; 2] {
        [[0.0, 0.0, 100.0], [30.0, 0.0, 100.0]]
    }

    #[test]
    fn dashed_source_changes_heading_and_width_without_observing_gaps_or_extensions() {
        let cloud = scene(&[-0.25, 2.75, 5.75], true);
        let original = cloud.positions.clone();
        let (roads, report) = extract(&cloud, &trace(), &options()).unwrap();
        let paint = report.paint_corridor.unwrap();
        assert!(paint.applied, "{}", paint.reason);
        assert!(!paint.tracks[0].strong);
        assert!(paint.tracks[0].extrapolated_length_m > 25.0);
        assert!(paint.tracks[1].interpolated_length_m > 8.0);
        assert!(
            (paint.heading_correction_degrees.unwrap() + 0.04_f64.atan().to_degrees()).abs() < 0.1
        );
        assert!(
            paint
                .measured_lane_widths_m
                .iter()
                .all(|w| (*w - 3.0).abs() < 0.04)
        );
        assert_eq!(roads.len(), 1);
        assert!((report.generated_length - 30.0).abs() < 1e-8);
        assert!(report.rgb_paint_vertices > 0 && report.width_prior_vertices > 0);
        assert_eq!(roads[0].evidence[0].last(), Some(&Evidence::WidthPrior));
        assert!(roads[0].boundaries.iter().flatten().all(|p| p[2] < 14.0));
        let mut map = Map::new();
        let built = build(&mut map, &cloud, &trace(), &options()).unwrap();
        assert_eq!(built.lanes, 2);
        assert!(
            quality::audit(&map, &cloud)
                .unwrap()
                .low_support_lanes
                .is_empty()
        );
        assert_eq!(cloud.positions, original);
    }

    #[test]
    fn rotation_large_coordinates_and_right_hand_roles_preserve_source_fit() {
        let mut cloud = scene(&[-0.25, 2.75, 5.75], true);
        let angle = 0.73_f64;
        let transform = |p: &mut [f64; 3]| {
            let [x, y, _] = *p;
            p[0] = 100_000.0 + x * angle.cos() - y * angle.sin();
            p[1] = -700_000.0 + x * angle.sin() + y * angle.cos();
        };
        cloud.positions.iter_mut().for_each(transform);
        let mut poses = trace();
        poses.iter_mut().for_each(transform);
        let o = BuildOptions {
            left_hand_traffic: false,
            ..options()
        };
        let (_, report) = extract(&cloud, &poses, &o).unwrap();
        assert!(report.paint_corridor.unwrap().applied);
    }

    #[test]
    fn missing_uniform_single_line_and_ambiguous_paint_do_not_move_default_geometry() {
        for kind in 0..4 {
            let mut cloud = match kind {
                2 => scene(&[2.75], false),
                3 => scene(&[-3.25, -0.25, 2.75, 5.75], false),
                _ => scene(&[-0.25, 2.75, 5.75], false),
            };
            if kind == 0 {
                cloud.colors = None;
            }
            if kind == 1 {
                cloud.colors.as_mut().unwrap().fill([255; 3]);
            }
            let (_, report) = extract(&cloud, &trace(), &options()).unwrap();
            assert!(!report.paint_corridor.unwrap().applied, "case {kind}");
            let mut a = Map::new();
            let mut b = Map::new();
            build(&mut a, &cloud, &trace(), &options()).unwrap();
            let off = BuildOptions {
                fit_paint_corridor: false,
                ..options()
            };
            build(&mut b, &cloud, &trace(), &off).unwrap();
            assert_eq!(
                serde_json::to_string(&a).unwrap(),
                serde_json::to_string(&b).unwrap()
            );
        }
        assert!(!BuildOptions::default().fit_paint_corridor);
    }

    #[test]
    fn wide_bright_surfaces_transverse_bars_other_levels_and_curves_are_held() {
        for kind in 0..4 {
            let mut cloud = scene(&[-0.25, 2.75, 5.75], false);
            for (p, color) in cloud
                .positions
                .iter_mut()
                .zip(cloud.colors.as_mut().unwrap())
            {
                if kind < 2 {
                    *color = if (kind == 0 && (p[1] - 2.75).abs() < 0.9)
                        || (kind == 1 && (p[0] - 10.0).abs() < 0.1)
                    {
                        [230; 3]
                    } else {
                        [70; 3]
                    };
                }
                if kind == 2 && *color == [230; 3] {
                    p[2] += 3.0;
                }
            }
            let poses = if kind == 3 {
                vec![[0.0, 0.0, 100.0], [10.0, 5.0, 100.0], [30.0, 0.0, 100.0]]
            } else {
                trace().to_vec()
            };
            let (_, report) = extract(&cloud, &poses, &options()).unwrap();
            assert!(!report.paint_corridor.unwrap().applied, "case {kind}");
        }
    }

    #[test]
    fn budgets_and_invalid_color_counts_are_explicit_and_atomic() {
        let mut cloud = scene(&[-0.25, 2.75, 5.75], false);
        let ids: Vec<_> = (0..cloud.len()).collect();
        let points = cloud.positions.clone();
        for _ in 0..4 {
            cloud.positions.extend(&points);
            cloud.colors.as_mut().unwrap().extend(
                ids.iter()
                    .map(|&i| if i % 2 == 0 { [70; 3] } else { [230; 3] }),
            );
        }
        let (roads, report) = fit(&cloud, &trace(), &options());
        assert!(roads.is_none() && report.limited);
        cloud.colors.as_mut().unwrap().pop();
        let mut map = Map::new();
        assert!(build(&mut map, &cloud, &trace(), &options()).is_err());
        assert_eq!(map.lanes().count(), 0);
    }
}
