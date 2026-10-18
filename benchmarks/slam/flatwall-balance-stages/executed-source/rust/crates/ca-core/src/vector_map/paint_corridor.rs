//! Source-only parallel paint fits. This does not infer lane legality or counts.
use super::{BuildOptions, BuildReport, Evidence, ExtractedRoad, quantile};
use crate::PointCloud;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const MAX_ROI: usize = 250_000;
const MAX_WHITE: usize = 50_000;
const MAX_PAINT: usize = 10_000;
const MAX_NEIGHBOURS: usize = 4096;
// Fine cells prune unrelated returns without changing the circular query or cap.
const FINE_CELL_WIDTH: f64 = 0.125;
const DENSE_CELL_THRESHOLD: usize = 128;
type Cell = (i64, i64);
type Subcells = Vec<(Cell, Vec<usize>)>;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaintBudgetStage {
    RoiPoints,
    BrightCandidates,
    ContrastNeighbours,
    TraceGroundNeighbours,
    PaintPoints,
    ComponentNeighbours,
    BoundaryGroundNeighbours,
}

#[derive(Debug, Clone, Serialize)]
pub struct PaintQueryLimit {
    /// Local trace coordinates; not a new source observation.
    pub center_st: [f64; 2],
    pub radius_m: f64,
    /// Potential points in intersecting cells, counted before visiting the cell.
    pub candidate_points: usize,
    pub limit: usize,
}

/// Select one retained channel explicitly; no automatic fallback or synthetic RGB.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaintChannel {
    #[default]
    Rgb,
    Intensity,
}
impl PaintChannel {
    pub fn is_rgb(&self) -> bool {
        *self == Self::Rgb
    }
    pub(super) fn evidence(self) -> Evidence {
        match self {
            Self::Rgb => Evidence::RgbPaint,
            Self::Intensity => Evidence::Intensity,
        }
    }
}

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

/// Exclusive outcomes of bright-return checks, before component/bundle fitting.
/// Height quantiles are local support tests, not certified ground classifications.
#[derive(Debug, Clone, Serialize, Default)]
pub struct PaintCandidateDiagnostics {
    pub bright_candidates: usize,
    pub local_ground_missing: usize,
    pub local_height_mismatch: usize,
    pub trace_ground_missing: usize,
    pub trace_height_mismatch: usize,
    pub flank_support_missing: usize,
    pub flank_contrast_insufficient: usize,
    pub accepted: usize,
    /// False on an interrupted scan; its pending candidate has no outcome.
    pub complete: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PaintCorridorReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_channel: Option<PaintChannel>,
    /// Raw ROI intensity P10/P99.9 used only for local contrast normalization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intensity_range: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_stage: Option<PaintBudgetStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_query: Option<PaintQueryLimit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_diagnostics: Option<PaintCandidateDiagnostics>,
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
            source_channel: None,
            intensity_range: None,
            budget_stage: None,
            budget_query: None,
            candidate_diagnostics: None,
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
    cells: HashMap<Cell, Vec<usize>>,
    subcells: HashMap<Cell, Subcells>,
}
impl Samples {
    fn new(points: Vec<[f64; 3]>, ids: Vec<usize>) -> Self {
        let mut cells: HashMap<Cell, Vec<usize>> = HashMap::new();
        for (i, p) in points.iter().enumerate() {
            cells.entry(Self::cell(p, 0.5)).or_default().push(i);
        }
        let mut subcells = HashMap::new();
        for (&cell, ids) in &cells {
            if ids.len() <= DENSE_CELL_THRESHOLD {
                continue;
            }
            let mut fine: HashMap<Cell, Vec<usize>> = HashMap::new();
            for &i in ids {
                fine.entry(Self::cell(&points[i], FINE_CELL_WIDTH))
                    .or_default()
                    .push(i);
            }
            let mut fine: Subcells = fine.into_iter().collect();
            fine.sort_unstable_by_key(|(cell, _)| *cell);
            subcells.insert(cell, fine);
        }
        Self {
            points,
            ids,
            cells,
            subcells,
        }
    }
    fn cell(p: &[f64; 3], width: f64) -> Cell {
        ((p[0] / width).floor() as i64, (p[1] / width).floor() as i64)
    }
    fn intersects(cell: Cell, width: f64, p: &[f64; 3], radius: f64) -> bool {
        let low = [cell.0 as f64 * width, cell.1 as f64 * width];
        let distance = [0, 1].map(|a| (low[a] - p[a]).max(p[a] - low[a] - width).max(0.0));
        let padding = 8.0 * f64::EPSILON * (p[0].abs() + p[1].abs() + radius + 1.0);
        distance[0].hypot(distance[1]) <= radius + padding
    }
    fn nearby(&self, p: &[f64; 3], radius: f64) -> Result<Vec<usize>, PaintQueryLimit> {
        let (x, y) = Self::cell(p, 0.5);
        let reach = (radius / 0.5).ceil() as i64;
        let mut cells = Vec::new();
        let mut coarse_count = 0;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                let cell = (x.saturating_add(dx), y.saturating_add(dy));
                if let Some(ids) = self.cells.get(&cell) {
                    coarse_count += ids.len();
                    cells.push((cell, ids));
                }
            }
        }
        let in_circle = |i: usize| {
            let q = self.points[i];
            (q[0] - p[0]).hypot(q[1] - p[1]) <= radius
        };
        // Check cell lengths before visiting any points. Ordinary queries use
        // exactly the old traversal; only a coarse overflow needs subdivision.
        if coarse_count <= MAX_NEIGHBOURS {
            return Ok(cells
                .into_iter()
                .flat_map(|(_, ids)| ids.iter().copied().filter(|&i| in_circle(i)))
                .collect());
        }
        let mut result = Vec::new();
        let mut inspected = 0;
        let mut visit = |ids: &[usize]| -> Result<(), PaintQueryLimit> {
            inspected += ids.len();
            if inspected > MAX_NEIGHBOURS {
                return Err(PaintQueryLimit {
                    center_st: [p[0], p[1]],
                    radius_m: radius,
                    candidate_points: inspected,
                    limit: MAX_NEIGHBOURS,
                });
            }
            result.extend(ids.iter().copied().filter(|&i| in_circle(i)));
            Ok(())
        };
        for (cell, ids) in cells {
            if !Self::intersects(cell, 0.5, p, radius) {
                continue;
            }
            if let Some(fine) = self.subcells.get(&cell) {
                for (cell, ids) in fine {
                    if Self::intersects(*cell, FINE_CELL_WIDTH, p, radius) {
                        visit(ids)?;
                    }
                }
            } else {
                visit(ids)?;
            }
        }
        // Preserve completed-query order for component sums and source ties.
        result.sort_unstable_by_key(|&i| (Self::cell(&self.points[i], 0.5), i));
        Ok(result)
    }
    pub(super) fn ground(&self, p: &[f64; 3]) -> Result<Option<f64>, PaintQueryLimit> {
        let ids = self.nearby(p, 0.75)?;
        Ok(if ids.len() >= 3 {
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

fn components(paint: &Samples) -> Result<Vec<Component>, PaintQueryLimit> {
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
    Ok(result)
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
    let mut report = PaintCorridorReport {
        source_channel: (o.paint_channel == PaintChannel::Intensity)
            .then_some(PaintChannel::Intensity),
        ..Default::default()
    };
    let hold = |reason: &str, mut r: PaintCorridorReport| {
        r.reason = reason.into();
        Err(Box::new(r))
    };
    let limited = |stage, query, mut r: PaintCorridorReport| {
        r.limited = true;
        r.budget_stage = Some(stage);
        r.budget_query = query;
        hold(
            "paint fit budget exceeded; no corridor selected from a partial scan",
            r,
        )
    };
    let colors = cloud.colors.as_ref();
    if o.paint_channel == PaintChannel::Rgb && colors.is_none() {
        return hold("source has no RGB colors", report);
    }
    if o.paint_channel == PaintChannel::Intensity && cloud.attribute(crate::INTENSITY).is_none() {
        return hold("source has no retained intensity", report);
    }
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
            report.roi_points += 1;
            if points.len() > MAX_ROI {
                return limited(PaintBudgetStage::RoiPoints, None, report);
            }
        }
    }
    // Normalize only the chosen intensity channel, using a bounded ROI and
    // robust upper tail. A single dashed divider can occupy less than 0.5%
    // of a two-lane ROI, so P99.9 retains its tail. Missing values never count
    // as dark flanks. Geometry, raw channels and RGB thresholds remain untouched.
    let range = if o.paint_channel == PaintChannel::Intensity {
        let mut values: Vec<_> = ids
            .iter()
            .filter_map(|&i| super::intensity(cloud, i))
            .collect();
        let low = quantile(&mut values, 0.1);
        let high = quantile(&mut values, 0.999);
        let Some((low, high)) = low.zip(high).filter(|(l, h)| h - l > 1e-6) else {
            return hold("intensity has no usable ROI contrast", report);
        };
        report.intensity_range = Some([low, high]);
        Some([low, high])
    } else {
        None
    };
    let brightness = |i: usize| -> Option<f64> {
        match range {
            Some([low, high]) => super::intensity(cloud, i)
                .map(|v| (255.0 * (v - low) / (high - low)).clamp(0.0, 255.0)),
            None => Some(f64::from(*colors.unwrap()[i].iter().min().unwrap())),
        }
    };
    for &id in &ids {
        if let Some(v) = brightness(id) {
            bright_min = bright_min.min(v);
            bright_max = bright_max.max(v);
        }
    }
    if bright_max - bright_min < 40.0 {
        return hold("RGB has no usable local paint contrast", report);
    }
    let samples = Samples::new(points, ids);
    report.candidate_diagnostics = Some(PaintCandidateDiagnostics::default());
    let mut paint_points = vec![];
    let mut paint_ids = vec![];
    for (i, p) in samples.points.iter().enumerate() {
        let Some(value) = brightness(samples.ids[i]) else {
            continue;
        };
        if value < 180.0 {
            continue;
        }
        report.white_candidates += 1;
        report
            .candidate_diagnostics
            .as_mut()
            .unwrap()
            .bright_candidates += 1;
        if report.white_candidates > MAX_WHITE {
            return limited(PaintBudgetStage::BrightCandidates, None, report);
        }
        let near = match samples.nearby(p, 0.8) {
            Ok(near) => near,
            Err(query) => {
                return limited(PaintBudgetStage::ContrastNeighbours, Some(query), report);
            }
        };
        let mut heights: Vec<_> = near
            .iter()
            .filter(|&&j| (samples.points[j][0] - p[0]).hypot(samples.points[j][1] - p[1]) <= 0.3)
            .map(|&j| samples.points[j][2])
            .collect();
        if heights.len() < 3 {
            report
                .candidate_diagnostics
                .as_mut()
                .unwrap()
                .local_ground_missing += 1;
            continue;
        }
        if (p[2] - quantile(&mut heights, 0.2).unwrap()).abs() > 0.12 {
            report
                .candidate_diagnostics
                .as_mut()
                .unwrap()
                .local_height_mismatch += 1;
            continue;
        }
        let ground = match samples.ground(&[p[0], 0.0, 0.0]) {
            Ok(ground) => ground,
            Err(query) => {
                return limited(PaintBudgetStage::TraceGroundNeighbours, Some(query), report);
            }
        };
        let Some(ground) = ground else {
            report
                .candidate_diagnostics
                .as_mut()
                .unwrap()
                .trace_ground_missing += 1;
            continue;
        };
        if (p[2] - ground).abs() > 0.3 {
            report
                .candidate_diagnostics
                .as_mut()
                .unwrap()
                .trace_height_mismatch += 1;
            continue;
        }
        let mut sides = [vec![], vec![]];
        for j in near {
            let q = samples.points[j];
            let dist = (q[0] - p[0]).hypot(q[1] - p[1]);
            if dist <= 0.3 || (q[2] - p[2]).abs() > 0.12 || (q[1] - p[1]).abs() <= 0.15 {
                continue;
            }
            if let Some(value) = brightness(samples.ids[j]) {
                sides[usize::from(q[1] > p[1])].push(value);
            }
        }
        if sides.iter().any(|side| side.len() < 3) {
            report
                .candidate_diagnostics
                .as_mut()
                .unwrap()
                .flank_support_missing += 1;
            continue;
        }
        if sides
            .iter_mut()
            .any(|side| value - quantile(side, 0.5).unwrap() < 40.0)
        {
            report
                .candidate_diagnostics
                .as_mut()
                .unwrap()
                .flank_contrast_insufficient += 1;
            continue;
        }
        report.candidate_diagnostics.as_mut().unwrap().accepted += 1;
        paint_points.push(*p);
        paint_ids.push(samples.ids[i]);
        if paint_points.len() > MAX_PAINT {
            return limited(PaintBudgetStage::PaintPoints, None, report);
        }
    }
    report.candidate_diagnostics.as_mut().unwrap().complete = true;
    report.contrasted_points = paint_points.len();
    let paint = Samples::new(paint_points, paint_ids);
    let mut parts = match components(&paint) {
        Ok(parts) => parts,
        Err(query) => return limited(PaintBudgetStage::ComponentNeighbours, Some(query), report),
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
    let limited = |stage, query, mut r: PaintCorridorReport| {
        r.limited = true;
        r.budget_stage = Some(stage);
        r.budget_query = Some(query);
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
        let ground = match samples.ground(&[s, 0.0, 0.0]) {
            Ok(ground) => ground,
            Err(query) => return limited(PaintBudgetStage::TraceGroundNeighbours, query, report),
        };
        for t in selected.iter().rev() {
            let lateral = t.intercept + slope * s;
            let z = match samples.ground(&[s, lateral, 0.0]) {
                Ok(z) => z,
                Err(query) => {
                    return limited(PaintBudgetStage::BoundaryGroundNeighbours, query, report);
                }
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
                (o.paint_channel.evidence(), cloud.positions[paint.ids[i]])
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
        let warning: String = if paint.applied {
            "RGB paint set parallel boundary spacing and heading. Only nearby observed paint vertices have RGB evidence; gaps and extensions remain inferred. Track interval statistics describe the source fit before source-footprint trimming. Lane counts, boundary roles and travel directions still require review.".into()
        } else {
            format!(
                "RGB paint corridor was held: {}. Existing extraction rules were used; this is not a negative road-marking diagnosis.",
                paint.reason
            )
        };
        report
            .warnings
            .push(if paint.source_channel == Some(PaintChannel::Intensity) {
                warning.replace("RGB", "intensity")
            } else {
                warning
            });
    }
}

#[cfg(test)]
mod tests {
    use super::super::{build, extract, quality};
    use super::*;
    use vectormap_core::Map;

    pub(super) fn scene(lines: &[f64], sparse_outer: bool) -> PointCloud {
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
    pub(super) fn options() -> BuildOptions {
        BuildOptions {
            fit_paint_corridor: true,
            fit_source_surface: true,
            segment_length: 0.0,
            ..Default::default()
        }
    }
    pub(super) fn trace() -> [[f64; 3]; 2] {
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
        assert!(matches!(
            report.budget_stage,
            Some(PaintBudgetStage::RoiPoints)
        ));
        cloud.colors.as_mut().unwrap().pop();
        let mut map = Map::new();
        assert!(build(&mut map, &cloud, &trace(), &options()).is_err());
        assert_eq!(map.lanes().count(), 0);
    }
}

#[cfg(test)]
mod query_tests {
    use super::*;

    #[test]
    fn circular_queries_match_brute_force_in_legacy_order_at_cell_and_radius_edges() {
        let mut points = vec![];
        for x in -35..=35 {
            for y in -35..=35 {
                points.push([x as f64 * 0.04, y as f64 * 0.04, (x - y) as f64]);
            }
        }
        for radius in [0.3, 0.75, 0.8] {
            points.extend([
                [radius, 0.0, 1.0],
                [-radius, 0.0, 1.0],
                [radius + 1e-12, 0.0, 1.0],
                [radius - 1e-12, 0.0, 1.0],
                [-0.125, -0.125, 1.0],
            ]);
        }
        let samples = Samples::new(points.clone(), (0..points.len()).collect());
        for center in [
            [0.0, 0.0, 0.0],
            [-0.125, 0.125, 0.0],
            [0.499999999, -0.500000001, 0.0],
        ] {
            for radius in [0.3, 0.75, 0.8] {
                let mut expected: Vec<_> = points
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| {
                        ((p[0] - center[0]).hypot(p[1] - center[1]) <= radius).then_some(i)
                    })
                    .collect();
                expected.sort_by_key(|&i| {
                    (
                        (points[i][0] / 0.5).floor() as i64,
                        (points[i][1] / 0.5).floor() as i64,
                        i,
                    )
                });
                assert_eq!(samples.nearby(&center, radius).unwrap(), expected);
            }
        }
    }

    #[test]
    fn unrelated_dense_cells_do_not_consume_budget_but_dense_nearby_returns_hold() {
        let mut points = vec![[0.0, 0.0, 12.0], [-0.1, 0.0, 12.0], [0.1, 0.0, 12.0]];
        points.extend((0..5000).map(|i| [1.2, 1.2, 12.0 + i as f64 * 0.01]));
        let samples = Samples::new(points.clone(), (0..points.len()).collect());
        // The old 0.5 m square visits all 5,003 points for this 0.8 m circle.
        assert_eq!(samples.nearby(&[0.0; 3], 0.8).unwrap(), vec![1, 0, 2]);
        assert_eq!(samples.ground(&[0.0; 3]).unwrap(), Some(12.0));
        points.extend((0..MAX_NEIGHBOURS + 1).map(|i| [0.1, 0.1, 12.0 + i as f64 * 0.01]));
        let samples = Samples::new(points.clone(), (0..points.len()).collect());
        let failure = samples.nearby(&[0.0; 3], 0.8).unwrap_err();
        assert_eq!(failure.limit, 4096);
        assert!(failure.candidate_points > failure.limit);
        assert_eq!(failure.radius_m, 0.8);
    }

    #[test]
    fn high_returns_outside_paint_query_preserve_known_fit_and_original_sources() {
        use super::tests::{options, scene, trace};
        let cloud = scene(&[-4.5, -1.5, 1.5], false);
        let baseline = fit(&cloud, &trace(), &options());
        assert!(baseline.1.applied);
        let mut cluttered = cloud.clone();
        for i in 0..5000 {
            cluttered
                .positions
                .push([-0.9, 2.4, 22.0 + i as f64 * 0.001]);
            cluttered.colors.as_mut().unwrap().push([70; 3]);
        }
        let original = cluttered.clone();
        let (roads, report) = fit(&cluttered, &trace(), &options());
        assert!(report.applied, "{}", report.reason);
        assert!(report.budget_stage.is_none() && report.budget_query.is_none());
        let roads = roads.unwrap();
        let baseline = baseline.0.unwrap();
        for (actual, expected) in roads.iter().zip(&baseline) {
            assert_eq!(actual.boundaries, expected.boundaries);
            assert_eq!(actual.evidence, expected.evidence);
            assert_eq!(actual.source_boundaries, expected.source_boundaries);
        }
        assert_eq!(cluttered, original);
        // A genuine dense neighbourhood still holds the whole optional fit.
        for _ in 0..MAX_NEIGHBOURS + 1 {
            cluttered.positions.push([0.0, 1.5, 12.0]);
            cluttered.colors.as_mut().unwrap().push([70; 3]);
        }
        let (roads, report) = fit(&cluttered, &trace(), &options());
        assert!(roads.is_none() && report.limited);
        assert!(matches!(
            report.budget_stage,
            Some(PaintBudgetStage::ContrastNeighbours)
        ));
        assert!(report.budget_query.unwrap().candidate_points > MAX_NEIGHBOURS);
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    fn outcomes(d: &PaintCandidateDiagnostics) -> [usize; 7] {
        [
            d.local_ground_missing,
            d.local_height_mismatch,
            d.trace_ground_missing,
            d.trace_height_mismatch,
            d.flank_support_missing,
            d.flank_contrast_insufficient,
            d.accepted,
        ]
    }

    #[test]
    fn bright_outcomes_are_exclusive_and_accepted_is_not_a_paint_track() {
        for outcome in 0..7 {
            let mut cloud = PointCloud {
                colors: Some(vec![]),
                ..Default::default()
            };
            let mut add = |p, c| {
                cloud.positions.push(p);
                cloud.colors.as_mut().unwrap().push([c; 3]);
            };
            add(
                [2.0, 3.0, if outcome == 1 { 15.0 } else { 12.0 }],
                if outcome == 5 { 210 } else { 230 },
            );
            if outcome != 0 {
                for dx in [-0.05, 0.0, 0.05] {
                    add([2.0 + dx, 3.05, 12.0], 70);
                }
            }
            if outcome != 2 {
                for dx in [-0.05, 0.0, 0.05] {
                    add([2.0 + dx, 0.0, if outcome == 3 { 10.0 } else { 12.0 }], 70);
                }
            }
            if outcome != 4 {
                for dy in [-0.4, 0.4] {
                    for dx in [-0.05, 0.0, 0.05] {
                        add(
                            [2.0 + dx, 3.0 + dy, 12.0],
                            if outcome == 5 { 175 } else { 70 },
                        );
                    }
                }
            }
            let original = cloud.clone();
            let report = scan(
                &cloud,
                &[[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]],
                &BuildOptions::default(),
                1,
            )
            .err()
            .unwrap();
            let d = report.candidate_diagnostics.unwrap();
            let mut expected = [0; 7];
            expected[outcome] = 1;
            assert_eq!(outcomes(&d), expected, "outcome {outcome}");
            assert!(d.complete);
            assert_eq!(outcomes(&d).iter().sum::<usize>(), d.bright_candidates);
            assert!(!report.applied && report.longitudinal_components == 0);
            assert_eq!(cloud, original);
        }
    }

    #[test]
    fn interrupted_scan_has_pending_candidate_and_does_not_claim_complete_counts() {
        let mut cloud = PointCloud {
            colors: Some(vec![]),
            ..Default::default()
        };
        cloud.positions.push([2.0, 3.0, 12.0]);
        cloud.colors.as_mut().unwrap().push([230; 3]);
        for _ in 0..MAX_NEIGHBOURS + 1 {
            cloud.positions.push([2.0, 3.1, 12.0]);
            cloud.colors.as_mut().unwrap().push([70; 3]);
        }
        let report = scan(
            &cloud,
            &[[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]],
            &BuildOptions::default(),
            1,
        )
        .err()
        .unwrap();
        let d = report.candidate_diagnostics.unwrap();
        assert!(!d.complete && report.limited && !report.applied);
        assert_eq!(d.bright_candidates, 1);
        assert_eq!(outcomes(&d), [0; 7]);
        assert_eq!(report.contrasted_points, 0);
        // Pre-scan holds must not expose fabricated zero counts.
        cloud.colors = None;
        let missing = scan(
            &cloud,
            &[[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]],
            &BuildOptions::default(),
            1,
        )
        .err()
        .unwrap();
        let json = serde_json::to_value(missing).unwrap();
        assert!(json.get("candidate_diagnostics").is_none());
    }
}

#[cfg(test)]
mod intensity_tests {
    use super::tests::{options, scene, trace};
    use super::*;
    use crate::{Attribute, AttributeValues, INTENSITY};
    fn from_rgb(mut cloud: PointCloud, scale: f32, offset: f32) -> PointCloud {
        let values: Vec<_> = cloud
            .colors
            .as_ref()
            .unwrap()
            .iter()
            .map(|c| offset + scale * f32::from(c[0]))
            .collect();
        cloud.attributes.push(Attribute {
            name: INTENSITY.into(),
            values: AttributeValues::F32(values),
        });
        cloud.colors = Some(vec![[255; 3]; cloud.len()]);
        cloud
    }
    #[test]
    fn explicit_intensity_fits_widths_and_heading_without_rgb_or_observing_gaps() {
        let rgb = scene(&[-0.25, 2.75, 5.75], true);
        let rgb_fit = super::super::extract(&rgb, &trace(), &options()).unwrap();
        for (scale, offset) in [(1., 0.), (256., 10000.), (0.01, 5.)] {
            let cloud = from_rgb(rgb.clone(), scale, offset);
            let original = cloud.clone();
            let (_, default) = super::super::extract(&cloud, &trace(), &options()).unwrap();
            assert!(!default.paint_corridor.unwrap().applied);
            let mut o = options();
            o.paint_channel = PaintChannel::Intensity;
            let (roads, report) = super::super::extract(&cloud, &trace(), &o).unwrap();
            let fit = report.paint_corridor.as_ref().unwrap();
            assert!(fit.applied, "{}", fit.reason);
            assert_eq!(fit.source_channel, Some(PaintChannel::Intensity));
            assert!(fit.intensity_range.unwrap()[1] > fit.intensity_range.unwrap()[0]);
            assert!(
                fit.measured_lane_widths_m
                    .iter()
                    .all(|w| (w - 3.).abs() < 0.04)
            );
            assert!(fit.tracks[0].extrapolated_length_m > 25.);
            assert_eq!(
                roads.iter().map(|r| &r.boundaries).collect::<Vec<_>>(),
                rgb_fit.0.iter().map(|r| &r.boundaries).collect::<Vec<_>>()
            );
            assert_eq!(report.rgb_paint_vertices, 0);
            assert_eq!(report.intensity_vertices, rgb_fit.1.rgb_paint_vertices);
            assert!(report.width_prior_vertices > 0);
            assert_eq!(cloud, original);
        }
    }
    #[test]
    fn missing_uniform_nonfinite_and_wide_intensity_never_invent_a_corridor() {
        let rgb = scene(&[-0.25, 2.75, 5.75], false);
        let mut cases = vec![rgb.clone()];
        for v in [65535., f32::NAN] {
            let mut cloud = rgb.clone();
            cloud.attributes.push(Attribute {
                name: INTENSITY.into(),
                values: AttributeValues::F32(vec![v; cloud.len()]),
            });
            cases.push(cloud);
        }
        let mut wide = rgb;
        wide.colors.as_mut().unwrap().fill([230; 3]);
        cases.push(from_rgb(wide, 1., 0.));
        for cloud in cases {
            let (_, before) = super::super::extract(&cloud, &trace(), &options()).unwrap();
            let mut o = options();
            o.paint_channel = PaintChannel::Intensity;
            let (_, after) = super::super::extract(&cloud, &trace(), &o).unwrap();
            assert!(!after.paint_corridor.unwrap().applied);
            assert_eq!(after.generated_length, before.generated_length);
        }
    }
}
