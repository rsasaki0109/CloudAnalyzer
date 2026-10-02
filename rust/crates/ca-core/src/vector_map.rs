//! Draft Lanelet2 geometry from a surveyed cloud and a drive's trajectory.
//!
//! Cross-sections follow the trajectory. Low surface quantiles reject above-road
//! returns; curb steps, support edges and intensity peaks refine nominal lane
//! boundaries. Missing observations retain the explicit width prior and are
//! counted separately. This is a draft for review, not a claim that every line
//! was observed. Ground elevation comes from the cloud, never from sensor poses.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use vectormap_core::{LaneDirection, Map, NewRoad, Point3, Polyline3, RoadLane, SpeedLimit};

use crate::{AttributeValues, INTENSITY, PointCloud};

pub mod crosswalks;
mod fitting;
mod integration;
pub mod junctions;
pub mod signals;

/// Parameters in metres, except speed in km/h and lane counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BuildOptions {
    pub forward_lanes: usize,
    pub backward_lanes: usize,
    /// The trajectory occupies the outside forward lane: leftmost for left-hand
    /// traffic, rightmost for right-hand traffic.
    pub left_hand_traffic: bool,
    pub lane_width: f64,
    pub speed_limit: f64,
    pub segment_length: f64,
    pub sample_spacing: f64,
    /// Radius in resampled vertices of the triangular trajectory smoother.
    pub smoothing_window: usize,
    /// Place inferred lane lines relative to nearby detected outer boundaries.
    pub anchor_width_prior: bool,
    /// Select boundary candidates as a continuous path across supported slices.
    pub track_boundaries: bool,
    /// Regularize trajectory-relative lateral deviations with bounded XY movement.
    pub fit_boundaries: bool,
    /// Reuse matching lanes already in the map and add only uncovered intervals.
    pub merge_repeated_passes: bool,
    /// Maximum horizontal discrepancy for centre and both oriented boundaries.
    pub merge_distance: f64,
    /// Half-length of each longitudinal slice.
    pub half_window: f64,
    /// How far a measured boundary can deviate from its nominal position.
    pub search_margin: f64,
    pub bin_width: f64,
    pub curb_height: f64,
    /// Require road-side support and a bounded raised surface, rejecting walls
    /// and above-road objects that also create a positive height discontinuity.
    pub verify_curb_profiles: bool,
    pub min_bin_points: usize,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            forward_lanes: 1,
            backward_lanes: 1,
            left_hand_traffic: true,
            lane_width: 3.5,
            speed_limit: 40.0,
            segment_length: 50.0,
            sample_spacing: 2.0,
            smoothing_window: 2,
            anchor_width_prior: true,
            track_boundaries: true,
            fit_boundaries: true,
            merge_repeated_passes: true,
            merge_distance: 0.5,
            half_window: 2.0,
            search_margin: 1.5,
            bin_width: 0.2,
            curb_height: 0.08,
            verify_curb_profiles: true,
            min_bin_points: 3,
        }
    }
}

/// A measurable source of boundary geometry, or the explicit width prior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    Intensity,
    Curb,
    SupportEdge,
    WidthPrior,
}

/// One continuously supported stretch of the drive, with boundaries ordered left
/// to right looking along the trajectory. Positions are in the input cloud frame.
#[derive(Debug, Clone)]
pub struct ExtractedRoad {
    pub reference: Vec<[f64; 3]>,
    pub boundaries: Vec<Vec<[f64; 3]>>,
    /// One evidence label per boundary vertex.
    pub evidence: Vec<Vec<Evidence>>,
    /// Selected source positions before geometric fitting, including labelled
    /// width priors. Evidence refers to these sources, not measured fit positions.
    pub source_boundaries: Option<Vec<Vec<[f64; 3]>>>,
}

/// What was measured, what was inferred, and where data were missing.
#[derive(Debug, Clone, Serialize)]
pub struct BuildReport {
    pub roads: usize,
    pub lanes: usize,
    pub trajectory_length: f64,
    pub generated_length: f64,
    pub sampled_sections: usize,
    pub unsupported_sections: usize,
    pub intensity_used: bool,
    pub intensity_vertices: usize,
    pub curb_vertices: usize,
    pub rejected_curb_candidates: usize,
    pub support_edge_vertices: usize,
    pub width_prior_vertices: usize,
    pub anchored_prior_vertices: usize,
    pub tracked_vertices: usize,
    pub fitted_vertices: usize,
    pub maximum_fit_displacement: f64,
    pub reused_intervals: usize,
    pub reused_length: f64,
    pub added_length: f64,
    pub joined_connections: usize,
    pub observed_fraction: Vec<f64>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct BuildError(pub String);

fn fail<T>(s: &str) -> Result<T, BuildError> {
    Err(BuildError(s.into()))
}

impl BuildOptions {
    fn validate(&self) -> Result<(), BuildError> {
        if self.forward_lanes == 0
            || self.forward_lanes > 16
            || self.backward_lanes > 16
            || self.forward_lanes + self.backward_lanes > 16
        {
            return fail("use 1 to 16 total lanes with at least one forward lane");
        }
        for (name, value, min, max) in [
            ("lane_width", self.lane_width, 1.5, 6.0),
            ("speed_limit", self.speed_limit, 0.1, 200.0),
            ("segment_length", self.segment_length, 0.0, 10_000.0),
            ("sample_spacing", self.sample_spacing, 0.5, 10.0),
            ("half_window", self.half_window, 0.5, 10.0),
            (
                "search_margin",
                self.search_margin,
                0.0,
                self.lane_width * 0.45,
            ),
            ("bin_width", self.bin_width, 0.05, 0.5),
            ("curb_height", self.curb_height, 0.03, 0.5),
            ("merge_distance", self.merge_distance, 0.05, 1.0),
        ] {
            if !value.is_finite() || !(min..=max).contains(&value) {
                return Err(BuildError(format!(
                    "{name} must be finite and between {min} and {max}"
                )));
            }
        }
        if !(1..=100).contains(&self.min_bin_points) {
            return fail("min_bin_points must be between 1 and 100");
        }
        if self.smoothing_window > 5 {
            return fail("smoothing_window must be between 0 and 5");
        }
        Ok(())
    }
}

fn length(line: &[[f64; 3]]) -> f64 {
    line.windows(2)
        .map(|p| (p[1][0] - p[0][0]).hypot(p[1][1] - p[0][1]))
        .sum()
}

fn quantile(values: &mut [f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    Some(values[((values.len() - 1) as f64 * q).round() as usize])
}

fn resample(poses: &[[f64; 3]], spacing: f64) -> Result<Vec<[f64; 3]>, BuildError> {
    if poses.len() < 2
        || poses
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 1e12)
    {
        return fail("trajectory needs at least two finite positions in metres");
    }
    let mut cleaned = vec![poses[0]];
    for &p in &poses[1..] {
        let prev = cleaned.last().unwrap();
        if (p[0] - prev[0]).hypot(p[1] - prev[1]) >= 0.1 {
            cleaned.push(p);
        }
    }
    if cleaned.len() < 2 {
        return fail("trajectory has no horizontal movement");
    }
    let total = length(&cleaned);
    if total / spacing > 100_000.0 {
        return fail("trajectory is too long; split it into shorter passes");
    }
    let mut out = Vec::new();
    let mut station = 0.0;
    let mut next = 0.0;
    for pair in cleaned.windows(2) {
        let d = (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
        while next <= station + d {
            let t = (next - station) / d;
            out.push(std::array::from_fn(|i| {
                pair[0][i] + t * (pair[1][i] - pair[0][i])
            }));
            next += spacing;
        }
        station += d;
    }
    let end = *cleaned.last().unwrap();
    if out
        .last()
        .is_none_or(|p| (p[0] - end[0]).hypot(p[1] - end[1]) > 0.1)
    {
        out.push(end);
    }
    Ok(out)
}

/// Spatial hash in XY; sensor-height errors do not exclude road returns.
struct SurfaceIndex<'a> {
    cloud: &'a PointCloud,
    bins: HashMap<(i64, i64), Vec<usize>>,
    cell: f64,
}

impl<'a> SurfaceIndex<'a> {
    fn new(cloud: &'a PointCloud, line: &[[f64; 3]], reach: f64) -> Self {
        let cell = 2.0;
        let lo = [0, 1].map(|i| line.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min) - reach);
        let hi =
            [0, 1].map(|i| line.iter().map(|p| p[i]).fold(f64::NEG_INFINITY, f64::max) + reach);
        let mut bins: HashMap<_, Vec<_>> = HashMap::new();
        for (i, p) in cloud.positions.iter().enumerate() {
            if p.iter().all(|v| v.is_finite()) && (0..2).all(|a| p[a] >= lo[a] && p[a] <= hi[a]) {
                bins.entry(((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64))
                    .or_default()
                    .push(i);
            }
        }
        Self { cloud, bins, cell }
    }

    fn slice(
        &self,
        p: [f64; 3],
        dir: [f64; 2],
        low: f64,
        high: f64,
        o: &BuildOptions,
    ) -> Vec<Vec<usize>> {
        let n = ((high - low) / o.bin_width).ceil() as usize;
        let mut out = vec![Vec::new(); n];
        let r = low.abs().max(high.abs()) + o.half_window;
        let lo = [0, 1].map(|i| ((p[i] - r) / self.cell).floor() as i64);
        let hi = [0, 1].map(|i| ((p[i] + r) / self.cell).floor() as i64);
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                if let Some(indices) = self.bins.get(&(x, y)) {
                    for &i in indices {
                        let q = self.cloud.positions[i];
                        let d = [q[0] - p[0], q[1] - p[1]];
                        let along = d[0] * dir[0] + d[1] * dir[1];
                        let lateral = -d[0] * dir[1] + d[1] * dir[0];
                        if along.abs() <= o.half_window && lateral >= low && lateral < high {
                            out[((lateral - low) / o.bin_width).floor() as usize].push(i);
                        }
                    }
                }
            }
        }
        out
    }
}

fn intensity(cloud: &PointCloud, i: usize) -> Option<f64> {
    match &cloud.attribute(INTENSITY)?.values {
        AttributeValues::F32(v) => Some(f64::from(*v.get(i)?)).filter(|v| v.is_finite()),
        AttributeValues::U8(v) => Some(f64::from(*v.get(i)?)),
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    lateral: f64,
    z: f64,
    evidence: Evidence,
}

struct Section {
    normal: [f64; 2],
    choices: Vec<Vec<Candidate>>,
}

/// A step alone is not curb evidence: walls/vehicle bodies can be much taller
/// and a low isolated return can sit between unrelated raised surfaces.
fn supported_curb(surface: &[Option<f64>], i: usize, outward: isize, o: &BuildOptions) -> bool {
    let z = surface[i].unwrap();
    let at = |delta: isize| surface[(i as isize + delta * outward) as usize];
    let max_step = o.curb_height + 0.3;
    let tolerance = o.curb_height.max(0.05);
    (1..=2).all(|k| at(k).is_some_and(|v| v - z <= max_step))
        && (-2..=-1).any(|k| at(k).is_some_and(|v| (v - z).abs() <= tolerance))
}

fn track_candidates(road: &mut ExtractedRoad, sections: &[Section], nominal: &[f64]) -> usize {
    let mut changed = 0;
    for (j, &prior) in nominal.iter().enumerate() {
        let unary = |c: Candidate| {
            let evidence = match c.evidence {
                Evidence::Intensity => 0.0,
                Evidence::Curb => 0.1,
                Evidence::SupportEdge => 0.3,
                Evidence::WidthPrior => 0.9,
            };
            evidence + 0.1 * (c.lateral - prior).powi(2)
        };
        let mut costs: Vec<_> = sections[0].choices[j].iter().map(|&c| unary(c)).collect();
        let mut parents = Vec::new();
        for k in 1..sections.len() {
            let a = road.reference[k - 1];
            let b = road.reference[k];
            let distance = (b[0] - a[0]).hypot(b[1] - a[1]).max(0.1);
            let mut next = Vec::new();
            let mut parent = Vec::new();
            for &candidate in &sections[k].choices[j] {
                let (from, cost) = sections[k - 1].choices[j]
                    .iter()
                    .enumerate()
                    .map(|(id, c)| {
                        (
                            id,
                            costs[id] + 8.0 / distance * (candidate.lateral - c.lateral).powi(2),
                        )
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                next.push(cost + unary(candidate));
                parent.push(from);
            }
            costs = next;
            parents.push(parent);
        }
        let mut choice = costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        for k in (0..sections.len()).rev() {
            let c = sections[k].choices[j][choice];
            let p = road.reference[k];
            let n = sections[k].normal;
            let point = [p[0] + n[0] * c.lateral, p[1] + n[1] * c.lateral, c.z];
            let old = road.boundaries[j][k];
            if (old[0] - point[0]).hypot(old[1] - point[1]) > 1e-6
                || (old[2] - point[2]).abs() > 1e-6
                || road.evidence[j][k] != c.evidence
            {
                changed += 1;
            }
            road.boundaries[j][k] = point;
            road.evidence[j][k] = c.evidence;
            if k > 0 {
                choice = parents[k - 1][choice];
            }
        }
    }
    changed
}

/// Extract geometry without modifying a map. Separate stretches are returned
/// when there is no observed road surface near the trajectory; gaps are not
/// bridged with invented elevations.
pub fn extract(
    cloud: &PointCloud,
    poses: &[[f64; 3]],
    o: &BuildOptions,
) -> Result<(Vec<ExtractedRoad>, BuildReport), BuildError> {
    o.validate()?;
    if cloud.is_empty() {
        return fail("the point cloud is empty");
    }
    if cloud
        .attributes
        .iter()
        .any(|a| a.values.len() != cloud.len())
    {
        return fail("cloud attributes do not match its points");
    }
    let mut line = resample(poses, o.sample_spacing)?;
    let trajectory_length = length(&line);
    if o.smoothing_window > 0 {
        let original = line.clone();
        for k in 1..line.len() - 1 {
            let start = k.saturating_sub(o.smoothing_window);
            let end = (k + o.smoothing_window).min(line.len() - 1);
            let weights: Vec<_> = (start..=end)
                .map(|j| (o.smoothing_window + 1 - k.abs_diff(j)) as f64)
                .collect();
            let sum: f64 = weights.iter().sum();
            for a in 0..2 {
                line[k][a] = (start..=end)
                    .zip(&weights)
                    .map(|(j, w)| original[j][a] * w)
                    .sum::<f64>()
                    / sum;
            }
        }
    }
    let nlanes = o.forward_lanes + o.backward_lanes;
    let left = if o.left_hand_traffic {
        o.lane_width * 0.5
    } else {
        o.lane_width * (nlanes as f64 - 0.5)
    };
    let nominal: Vec<_> = (0..=nlanes)
        .map(|j| left - j as f64 * o.lane_width)
        .collect();
    let low = nominal[nlanes] - o.search_margin - o.bin_width * 3.0;
    let high = nominal[0] + o.search_margin + o.bin_width * 3.0;
    let index = SurfaceIndex::new(cloud, &line, low.abs().max(high.abs()) + o.half_window);
    let mut report = BuildReport {
        roads: 0,
        lanes: 0,
        trajectory_length,
        generated_length: 0.0,
        sampled_sections: line.len(),
        unsupported_sections: 0,
        intensity_used: false,
        intensity_vertices: 0,
        curb_vertices: 0,
        rejected_curb_candidates: 0,
        support_edge_vertices: 0,
        width_prior_vertices: 0,
        anchored_prior_vertices: 0,
        tracked_vertices: 0,
        fitted_vertices: 0,
        maximum_fit_displacement: 0.0,
        reused_intervals: 0,
        reused_length: 0.0,
        added_length: 0.0,
        joined_connections: 0,
        observed_fraction: vec![0.0; nlanes + 1],
        warnings: Vec::new(),
    };
    let mut roads = Vec::new();
    let mut observations = Vec::new();
    let mut sections = Vec::new();
    let empty = || ExtractedRoad {
        reference: Vec::new(),
        boundaries: vec![Vec::new(); nlanes + 1],
        evidence: vec![Vec::new(); nlanes + 1],
        source_boundaries: None,
    };
    let mut road = empty();
    let mut prior = nominal.clone();
    for (k, &p) in line.iter().enumerate() {
        let a = line[k.saturating_sub(1)];
        let b = line[(k + 1).min(line.len() - 1)];
        let norm = (b[0] - a[0]).hypot(b[1] - a[1]);
        if norm < 0.1 {
            report.unsupported_sections += 1;
            continue;
        }
        let dir = [(b[0] - a[0]) / norm, (b[1] - a[1]) / norm];
        let bins = index.slice(p, dir, low, high, o);
        let lateral = |i: usize| low + (i as f64 + 0.5) * o.bin_width;
        let surface: Vec<_> = bins
            .iter()
            .map(|ids| {
                if ids.len() < o.min_bin_points {
                    return None;
                }
                quantile(
                    &mut ids
                        .iter()
                        .map(|&i| cloud.positions[i][2])
                        .collect::<Vec<_>>(),
                    0.2,
                )
            })
            .collect();
        let ground = quantile(
            &mut surface
                .iter()
                .enumerate()
                .filter(|(i, _)| lateral(*i).abs() < 0.8)
                .filter_map(|(_, v)| *v)
                .collect::<Vec<_>>(),
            0.5,
        );
        let Some(ground) = ground else {
            report.unsupported_sections += 1;
            if road.reference.len() >= 2 {
                roads.push(road);
                observations.push(sections);
            }
            road = empty();
            sections = Vec::new();
            prior = nominal.clone();
            continue;
        };
        let strength: Vec<Option<f64>> = bins
            .iter()
            .zip(&surface)
            .map(|(ids, z)| {
                let z = (*z)?;
                let mut v: Vec<_> = ids
                    .iter()
                    .filter(|&&i| (cloud.positions[i][2] - z).abs() < 0.12)
                    .filter_map(|&i| intensity(cloud, i))
                    .collect();
                quantile(&mut v, 0.75)
            })
            .collect();
        let mut values: Vec<_> = strength.iter().flatten().copied().collect();
        let q10 = quantile(&mut values, 0.1);
        let peak = quantile(&mut values, 1.0);
        let mut candidates = Vec::new();
        if let (Some(base), Some(peak)) = (q10, peak)
            && peak > base + f64::EPSILON
        {
            let high = |i: usize| {
                strength[i].is_some_and(|s| s > base + (peak - base) * 0.75)
                    && surface[i].is_some_and(|z| (z - ground).abs() < 0.5)
            };
            let mut i = 1;
            while i < bins.len() - 1 {
                if !high(i) {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < bins.len() - 1 && high(i) {
                    i += 1;
                }
                let end = i - 1;
                if (end - start + 1) as f64 * o.bin_width <= 1.0
                    && strength[start - 1].is_some_and(|s| s < peak - (peak - base) * 0.1)
                    && strength[i].is_some_and(|s| s < peak - (peak - base) * 0.1)
                {
                    candidates.push(Candidate {
                        lateral: (lateral(start) + lateral(end)) * 0.5,
                        z: surface[start].unwrap(),
                        evidence: Evidence::Intensity,
                    });
                    report.intensity_used = true;
                }
            }
        }
        for i in 2..bins.len() - 2 {
            let Some(z) = surface[i].filter(|z| (z - ground).abs() < 0.5) else {
                continue;
            };
            let outside = if lateral(i) > 0.0 { i + 1 } else { i - 1 };
            let outside2 = if lateral(i) > 0.0 { i + 2 } else { i - 2 };
            let evidence = match (surface[outside], surface[outside2]) {
                (Some(a), Some(b)) if a - z >= o.curb_height && b - z >= o.curb_height => {
                    if o.verify_curb_profiles
                        && !supported_curb(&surface, i, if lateral(i) > 0.0 { 1 } else { -1 }, o)
                    {
                        report.rejected_curb_candidates += 1;
                        None
                    } else {
                        Some(Evidence::Curb)
                    }
                }
                (None, None) => Some(Evidence::SupportEdge),
                _ => None,
            };
            if let Some(evidence) = evidence {
                candidates.push(Candidate {
                    lateral: lateral(i),
                    z,
                    evidence,
                });
            }
        }
        road.reference.push([p[0], p[1], ground]);
        let mut section = Section {
            normal: [-dir[1], dir[0]],
            choices: Vec::new(),
        };
        for j in 0..=nlanes {
            let target = prior[j] * 0.7 + nominal[j] * 0.3;
            let found = candidates
                .iter()
                .filter(|c| {
                    (c.lateral - nominal[j]).abs() <= o.search_margin
                        && (j == 0 || j == nlanes || c.evidence == Evidence::Intensity)
                })
                .min_by(|a, b| {
                    (a.lateral - target)
                        .abs()
                        .total_cmp(&(b.lateral - target).abs())
                });
            let c = found.copied().unwrap_or(Candidate {
                lateral: nominal[j],
                z: ground,
                evidence: Evidence::WidthPrior,
            });
            let mut choices: Vec<_> = candidates
                .iter()
                .copied()
                .filter(|c| {
                    (c.lateral - nominal[j]).abs() <= o.search_margin
                        && (j == 0 || j == nlanes || c.evidence == Evidence::Intensity)
                })
                .collect();
            choices.push(Candidate {
                lateral: nominal[j],
                z: ground,
                evidence: Evidence::WidthPrior,
            });
            section.choices.push(choices);
            prior[j] = c.lateral;
            road.boundaries[j].push([p[0] - dir[1] * c.lateral, p[1] + dir[0] * c.lateral, c.z]);
            road.evidence[j].push(c.evidence);
        }
        sections.push(section);
    }
    if road.reference.len() >= 2 {
        roads.push(road);
        observations.push(sections);
    }
    if roads.is_empty() {
        return fail(
            "no continuous road surface found near the trajectory; check the coordinate frame and point density",
        );
    }
    if o.track_boundaries {
        for (road, sections) in roads.iter_mut().zip(&mut observations) {
            // Retain the original robust outer-edge offset when an unstable
            // detection is replaced by an inferred point. Dropping the
            // observation must not silently reset the lane to the pose centre.
            let (_, offsets) = prior_offsets(road, &nominal);
            if o.anchor_width_prior {
                for (k, section) in sections.iter_mut().enumerate() {
                    for (j, choices) in section.choices.iter_mut().enumerate() {
                        choices.last_mut().unwrap().lateral =
                            nominal[j] + offsets[k].unwrap_or(0.0);
                    }
                }
            }
            report.tracked_vertices += track_candidates(road, sections, &nominal);
            if o.anchor_width_prior {
                report.anchored_prior_vertices += road
                    .evidence
                    .iter()
                    .flat_map(|line| line.iter().enumerate())
                    .filter(|(k, e)| {
                        **e == Evidence::WidthPrior && offsets[*k].is_some_and(|v| v.abs() > 1e-6)
                    })
                    .count();
            }
        }
    } else if o.anchor_width_prior {
        for road in &mut roads {
            report.anchored_prior_vertices += anchor_priors(road, &nominal);
        }
    }
    if o.fit_boundaries {
        for road in &mut roads {
            let (count, maximum) = fitting::fit(road);
            report.fitted_vertices += count;
            report.maximum_fit_displacement = report.maximum_fit_displacement.max(maximum);
        }
    }
    let mut totals = vec![0usize; nlanes + 1];
    for road in &roads {
        report.generated_length += length(&road.reference);
        for (j, labels) in road.evidence.iter().enumerate() {
            for &e in labels {
                totals[j] += 1;
                match e {
                    Evidence::Intensity => report.intensity_vertices += 1,
                    Evidence::Curb => report.curb_vertices += 1,
                    Evidence::SupportEdge => report.support_edge_vertices += 1,
                    Evidence::WidthPrior => report.width_prior_vertices += 1,
                }
                if e != Evidence::WidthPrior {
                    report.observed_fraction[j] += 1.0;
                }
            }
        }
    }
    for (j, &n) in totals.iter().enumerate() {
        if n > 0 {
            report.observed_fraction[j] /= n as f64;
        }
    }
    report.intensity_used = report.intensity_vertices > 0;
    report.roads = roads.len();
    if !report.intensity_used {
        report.warnings.push(
            "No usable intensity contrast: internal lane lines use the configured width prior."
                .into(),
        );
    }
    if report.width_prior_vertices > 0 {
        report.warnings.push(format!(
            "{} boundary vertices use the configured lane width rather than a detected feature.",
            report.width_prior_vertices
        ));
    }
    if report.rejected_curb_candidates > 0 {
        report.warnings.push(format!("{} curb-like height transitions lacked consistent road-side support or had excessive raised-surface height and were rejected; remaining candidates still require review.", report.rejected_curb_candidates));
    }
    if report.anchored_prior_vertices > 0 {
        report.warnings.push(format!("{} inferred boundary vertices are positioned relative to detected outer boundaries; they remain width assumptions, not observed lane lines.",report.anchored_prior_vertices));
    }
    if report.tracked_vertices > 0 {
        report.warnings.push(format!("{} vertices selected a continuous candidate path. Rejected detections can become labelled width assumptions; source counts do not establish survey accuracy.",report.tracked_vertices));
    }
    if report.fitted_vertices > 0 {
        report.warnings.push(format!("{} vertices were fitted to trajectory-relative boundary curves (maximum XY movement {:.3} m; heights unchanged). Evidence describes the selected sources before fitting, not direct measurements at the fitted positions.",report.fitted_vertices,report.maximum_fit_displacement));
    }
    if report.support_edge_vertices > 0 {
        report.warnings.push("Some boundaries follow the end of point coverage; verify that they are road edges rather than scan gaps.".into());
    }
    if report.unsupported_sections > 0 {
        report.warnings.push(format!("{} sections lack ground support and were omitted; disconnected stretches require review.",report.unsupported_sections));
    }
    report.warnings.push("Draft map: verify lane counts, travel directions, junctions and boundary geometry before use.".into());
    Ok((roads, report))
}

/// Estimate a slowly varying lateral offset from detected outside edges. A
/// five-section median rejects single-section curb/coverage outliers. Only
/// width-prior vertices move; measured candidates retain their geometry and
/// inferred vertices keep their evidence label. No reference map is consulted.
/// Missing slices are excluded from the median; up to two neighbouring slices
/// receive an attenuated anchor rather than treating missing evidence as zero.
fn prior_offsets(road: &ExtractedRoad, nominal: &[f64]) -> (Vec<[f64; 2]>, Vec<Option<f64>>) {
    let n = road.reference.len();
    let last = nominal.len() - 1;
    let normals: Vec<_> = (0..n)
        .map(|k| {
            let a = road.reference[k.saturating_sub(1)];
            let b = road.reference[(k + 1).min(n - 1)];
            let norm = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-12);
            [-(b[1] - a[1]) / norm, (b[0] - a[0]) / norm]
        })
        .collect();
    let residuals: Vec<_> = (0..n)
        .map(|k| {
            let p = road.reference[k];
            let mut sum = 0.0;
            let mut count = 0;
            for j in [0, last] {
                if road.evidence[j][k] != Evidence::WidthPrior {
                    let q = road.boundaries[j][k];
                    sum +=
                        (q[0] - p[0]) * normals[k][0] + (q[1] - p[1]) * normals[k][1] - nominal[j];
                    count += 1;
                }
            }
            (sum / f64::from(count.max(1)), count > 0)
        })
        .collect();
    let offsets = (0..n)
        .map(|k| {
            let start = k.saturating_sub(2);
            let end = (k + 2).min(n - 1);
            // Missing observations are not zero-offset measurements. Bridge
            // only this short neighbourhood, and retain the width-prior label.
            let mut values: Vec<_> = residuals[start..=end]
                .iter()
                .filter(|r| r.1)
                .map(|r| r.0)
                .collect();
            let offset = quantile(&mut values, 0.5)?;
            let nearest = (start..=end)
                .filter(|&i| residuals[i].1)
                .map(|i| k.abs_diff(i))
                .min()
                .unwrap();
            // Avoid an abrupt jump to the pose-centred prior at the edge of
            // supported observations. Missing slices gradually lose the anchor.
            Some(offset * (1.0 - nearest as f64 / 3.0))
        })
        .collect();
    (normals, offsets)
}

fn anchor_priors(road: &mut ExtractedRoad, nominal: &[f64]) -> usize {
    let (normals, offsets) = prior_offsets(road, nominal);
    let mut moved = 0;
    for (k, offset) in offsets.into_iter().enumerate() {
        let Some(offset) = offset else { continue };
        for (j, &value) in nominal.iter().enumerate() {
            if road.evidence[j][k] == Evidence::WidthPrior && offset.abs() > 1e-6 {
                road.boundaries[j][k][0] = road.reference[k][0] + normals[k][0] * (value + offset);
                road.boundaries[j][k][1] = road.reference[k][1] + normals[k][1] * (value + offset);
                moved += 1;
            }
        }
    }
    moved
}

/// Add extracted roads atomically to a map. Existing geometry and georeference
/// remain available for manual editing; any construction failure rolls back.
pub fn build(
    map: &mut Map,
    cloud: &PointCloud,
    poses: &[[f64; 3]],
    o: &BuildOptions,
) -> Result<BuildReport, BuildError> {
    let (roads, mut report) = extract(cloud, poses, o)?;
    let had_existing = map.lanes().next().is_some();
    let mut draft = map.clone();
    let forward = || RoadLane::new(o.lane_width, LaneDirection::Forward);
    let backward = || RoadLane::new(o.lane_width, LaneDirection::Backward);
    let lanes: Vec<_> = if o.left_hand_traffic {
        (0..o.forward_lanes)
            .map(|_| forward())
            .chain((0..o.backward_lanes).map(|_| backward()))
            .collect()
    } else {
        (0..o.backward_lanes)
            .map(|_| backward())
            .chain((0..o.forward_lanes).map(|_| forward()))
            .collect()
    };
    report.roads = 0;
    let mut created = Vec::new();
    for road in roads {
        let parts = if o.merge_repeated_passes {
            integration::uncovered(&draft, road, &lanes, o.merge_distance, &mut report)
        } else {
            vec![road]
        };
        for road in parts {
            let added_length = length(&road.reference);
            let polyline = |points: Vec<[f64; 3]>| {
                Polyline3::new(
                    points
                        .into_iter()
                        .map(|p| Point3::new(p[0], p[1], p[2]))
                        .collect(),
                )
            };
            let mut spec = NewRoad::new(polyline(road.reference), lanes.clone());
            spec.boundaries = Some(road.boundaries.into_iter().map(polyline).collect());
            spec.speed_limit = Some(SpeedLimit::from_kmh(o.speed_limit));
            spec.segment_length = (o.segment_length > 0.0).then_some(o.segment_length);
            let (built, _) = draft
                .build_road(spec)
                .map_err(|e| BuildError(e.to_string()))?;
            report.lanes += built.lanes.iter().map(Vec::len).sum::<usize>();
            report.roads += 1;
            report.added_length += added_length;
            created.extend(built.lanes.into_iter().flatten());
        }
    }
    if o.merge_repeated_passes {
        report.joined_connections = integration::link_touching(&mut draft, &created)?;
    }
    if report.reused_intervals > 0 {
        report.warnings.push(format!("{} intervals ({:.3} m) reuse matching existing lanes; their geometry, IDs and traffic rules were preserved. Evidence counts describe the incoming pass, not additional surveyed accuracy.",report.reused_intervals,report.reused_length));
    }
    if had_existing && o.merge_repeated_passes {
        report.warnings.push("Matching requires centre, both boundaries, travel direction and ground height to agree. Unmatched or ambiguous overlaps may remain and need review; this does not align drifting survey frames.".into());
    }
    *map = draft;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Attribute;

    #[test]
    fn tracking_rejects_an_isolated_peak_and_labels_missing_evidence_as_prior() {
        let reference: Vec<_> = (0..11).map(|k| [k as f64 * 2.0, 0.0, 2.0]).collect();
        let mut road = ExtractedRoad {
            boundaries: vec![
                reference
                    .iter()
                    .enumerate()
                    .map(|(k, p)| [p[0], if k == 5 { 1.2 } else { k as f64 * 0.02 }, p[2]])
                    .collect(),
            ],
            evidence: vec![vec![Evidence::Curb; reference.len()]],
            reference,
            source_boundaries: None,
        };
        road.evidence[0][5] = Evidence::Intensity;
        let sections: Vec<_> = (0..11)
            .map(|k| {
                let mut choices = vec![Candidate {
                    lateral: 0.0,
                    z: 2.0,
                    evidence: Evidence::WidthPrior,
                }];
                if k != 7 {
                    choices.push(Candidate {
                        lateral: k as f64 * 0.02,
                        z: 2.0,
                        evidence: Evidence::Curb,
                    });
                }
                if k == 5 {
                    choices.push(Candidate {
                        lateral: 1.2,
                        z: 2.0,
                        evidence: Evidence::Intensity,
                    });
                }
                Section {
                    normal: [0.0, 1.0],
                    choices: vec![choices],
                }
            })
            .collect();
        assert!(track_candidates(&mut road, &sections, &[0.0]) >= 2);
        for k in 0..11 {
            assert_eq!(
                road.evidence[0][k],
                if k == 7 {
                    Evidence::WidthPrior
                } else {
                    Evidence::Curb
                }
            );
            assert!(
                (road.boundaries[0][k][1] - if k == 7 { 0.0 } else { k as f64 * 0.02 }).abs()
                    < 1e-12
            );
            assert_eq!(road.boundaries[0][k][2], 2.0);
        }
    }

    fn marked_road() -> PointCloud {
        let mut cloud = PointCloud::default();
        let mut intensities = Vec::new();
        for i in 0..=300 {
            for j in 0..=120 {
                let x = i as f64 * 0.1;
                let y = j as f64 * 0.1 - 8.0;
                let z = 2.0
                    + 0.01 * x
                    + if !(-5.25..=1.75).contains(&y) {
                        0.2
                    } else {
                        0.0
                    };
                cloud.positions.push([x, y, z]);
                intensities.push(if (y + 1.75).abs() < 0.11 { 200.0 } else { 20.0 });
            }
        }
        cloud.attributes.push(Attribute {
            name: INTENSITY.into(),
            values: AttributeValues::F32(intensities),
        });
        cloud
    }

    #[test]
    fn tall_roadside_surfaces_are_not_reported_as_measured_curbs() {
        let mut cloud = marked_road();
        cloud.attributes.clear();
        for p in &mut cloud.positions {
            if !(-5.25..=1.75).contains(&p[1]) {
                p[2] = 2.0 + 0.01 * p[0] + 1.5;
            }
        }
        let poses = [[1.0, 0.0, 50.0], [29.0, 0.0, 50.0]];
        let (_, legacy) = extract(
            &cloud,
            &poses,
            &BuildOptions {
                verify_curb_profiles: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(legacy.curb_vertices > 0);
        let (roads, guarded) = extract(&cloud, &poses, &BuildOptions::default()).unwrap();
        assert_eq!(guarded.curb_vertices, 0);
        assert!(guarded.rejected_curb_candidates > 0);
        assert!(
            roads
                .iter()
                .flat_map(|r| &r.evidence)
                .flatten()
                .all(|e| *e == Evidence::WidthPrior)
        );
        // Guarded geometry still uses explicit width assumptions, never a
        // made-up detection or an above-road elevation.
        for r in roads {
            for p in r.boundaries.iter().flatten() {
                assert!((p[2] - (2.0 + 0.01 * p[0])).abs() < 0.04);
            }
        }
    }

    #[test]
    fn real_height_curbs_survive_but_an_isolated_low_return_has_no_road_side_support() {
        let valid = marked_road();
        let poses = [[1.0, 0.0, 50.0], [29.0, 0.0, 50.0]];
        let (roads, report) = extract(&valid, &poses, &BuildOptions::default()).unwrap();
        let (legacy, _) = extract(
            &valid,
            &poses,
            &BuildOptions {
                verify_curb_profiles: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(report.curb_vertices > 0);
        assert_eq!(report.rejected_curb_candidates, 0);
        assert_eq!(roads[0].boundaries, legacy[0].boundaries);
        // A low isolated bin amid raised returns is not the road/sidewalk step.
        let surface = [Some(0.3), Some(0.3), Some(0.0), Some(0.2), Some(0.2)];
        assert!(!supported_curb(&surface, 2, 1, &BuildOptions::default()));
        let reversed: Vec<_> = surface.into_iter().rev().collect();
        assert!(!supported_curb(&reversed, 2, -1, &BuildOptions::default()));
    }

    #[test]
    fn short_observation_gaps_do_not_pull_the_width_prior_to_the_pose() {
        let mut road = ExtractedRoad {
            reference: (0..11).map(|k| [k as f64 * 2.0, 0.0, 2.0]).collect(),
            boundaries: [1.75, -1.75, -5.25]
                .into_iter()
                .map(|y| (0..11).map(|k| [k as f64 * 2.0, y, 2.0]).collect())
                .collect(),
            evidence: vec![vec![Evidence::WidthPrior; 11]; 3],
            source_boundaries: None,
        };
        // Only one actual outer-edge observation; surrounding missing slices
        // used to count as zero offsets and overwhelm it in the median.
        road.boundaries[0][5][1] += 0.8;
        road.evidence[0][5] = Evidence::Curb;
        anchor_priors(&mut road, &[1.75, -1.75, -5.25]);
        for k in 0usize..11 {
            let expected = -1.75 + 0.8 * (1.0 - k.abs_diff(5) as f64 / 3.0).max(0.0);
            assert!((road.boundaries[1][k][1] - expected).abs() < 1e-12);
            assert_eq!(road.evidence[1][k], Evidence::WidthPrior);
            assert_eq!(road.boundaries[1][k][2], 2.0);
        }
        assert_eq!(road.boundaries[0][5][1], 2.55);
        assert_eq!(road.evidence[0][5], Evidence::Curb);
    }

    #[test]
    fn observations_refine_boundaries_and_ground_replaces_sensor_height() {
        let cloud = marked_road();
        let poses = [[1.0, 0.0, 50.0], [29.0, 0.0, 50.0]];
        let (roads, report) = extract(&cloud, &poses, &BuildOptions::default()).unwrap();
        assert_eq!(roads.len(), 1);
        assert!(report.curb_vertices > 0, "{report:?}");
        assert!(report.intensity_vertices > 0, "{report:?}");
        assert!(
            report.observed_fraction.iter().all(|&f| f > 0.8),
            "{report:?}"
        );
        for p in &roads[0].reference {
            assert!((p[2] - (2.0 + 0.01 * p[0])).abs() < 0.04);
        }
        for (j, b) in roads[0].boundaries.iter().enumerate() {
            let expected = 1.75 - j as f64 * 3.5;
            assert!(
                b.iter().all(|p| (p[1] - expected).abs() < 0.25),
                "{j}: {b:?}"
            );
        }
        let mut map = Map::new();
        let report = build(&mut map, &cloud, &poses, &BuildOptions::default()).unwrap();
        assert_eq!(report.lanes, 2);
        assert!(map.lanes().all(|l| l.speed_limit.is_some()));
    }

    #[test]
    fn invalid_input_and_unsupported_ground_do_not_change_the_map() {
        let mut map = Map::new();
        let before = map.clone();
        let cloud = marked_road();
        for poses in [
            vec![[0.0, 0.0, 0.0]; 2],
            vec![[0.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]],
            vec![[1000.0, 0.0, 0.0], [1100.0, 0.0, 0.0]],
        ] {
            assert!(build(&mut map, &cloud, &poses, &BuildOptions::default()).is_err());
            assert_eq!(map, before);
        }
    }

    #[test]
    fn absent_intensity_and_missing_data_are_reported() {
        let mut cloud = marked_road();
        cloud.attributes.clear();
        cloud.positions.retain(|p| p[0] < 10.0 || p[0] > 20.0);
        let (roads, report) = extract(
            &cloud,
            &[[1.0, 0.0, 0.0], [29.0, 0.0, 0.0]],
            &BuildOptions::default(),
        )
        .unwrap();
        assert_eq!(roads.len(), 2);
        assert!(report.unsupported_sections > 0);
        assert!(!report.intensity_used);
        assert!(report.width_prior_vertices > 0);
    }

    #[test]
    fn outer_evidence_corrects_an_off_center_drive_without_claiming_observed_lines() {
        let mut cloud = marked_road();
        cloud.attributes.clear();
        for p in &mut cloud.positions {
            p[1] += 0.8;
        }
        let poses = [[1.0, 0.0, 50.0], [29.0, 0.0, 50.0]];
        let disabled = BuildOptions {
            anchor_width_prior: false,
            ..Default::default()
        };
        let (baseline, _) = extract(&cloud, &poses, &disabled).unwrap();
        let (refined, report) = extract(&cloud, &poses, &BuildOptions::default()).unwrap();
        let expected = -1.75 + 0.8;
        let error = |road: &ExtractedRoad| {
            road.boundaries[1]
                .iter()
                .map(|p| (p[1] - expected).abs())
                .sum::<f64>()
                / road.reference.len() as f64
        };
        assert!(error(&baseline[0]) > 0.7);
        assert!(error(&refined[0]) < 0.25);
        assert_eq!(baseline[0].boundaries[0], refined[0].boundaries[0]);
        assert_eq!(baseline[0].boundaries[2], refined[0].boundaries[2]);
        assert!(
            refined[0].evidence[1]
                .iter()
                .all(|e| *e == Evidence::WidthPrior)
        );
        assert!(report.anchored_prior_vertices > 0);
        assert!(!report.intensity_used);
    }

    #[test]
    fn replay_reuses_geometry_ids_and_rules_without_adding_lanes() {
        let cloud = marked_road();
        let poses = [[1.0, 0.0, 50.0], [29.0, 0.0, 50.0]];
        let mut map = Map::new();
        let options = BuildOptions {
            segment_length: 10.0,
            ..Default::default()
        };
        build(&mut map, &cloud, &poses, &options).unwrap();
        let ids: Vec<_> = map.lanes().map(|l| l.id).collect();
        map.set_speed_limit(&ids, Some(SpeedLimit::from_kmh(20.0)))
            .unwrap();
        let before = map.clone();
        let report = build(&mut map, &cloud, &poses, &options).unwrap();
        assert_eq!(map, before);
        assert_eq!(report.lanes, 0);
        assert_eq!(report.roads, 0);
        assert_eq!(report.added_length, 0.0);
        assert!((report.reused_length - 28.0).abs() < 1e-6);
        assert!(report.reused_intervals > 0);
        let disabled = BuildOptions {
            merge_repeated_passes: false,
            ..options
        };
        let report = build(&mut map, &cloud, &poses, &disabled).unwrap();
        assert_eq!(report.reused_intervals, 0);
        assert_eq!(map.lanes().count(), ids.len() * 2);
    }

    #[test]
    fn overlap_adds_only_the_extension_and_links_both_travel_directions() {
        let cloud = marked_road();
        let mut map = Map::new();
        let o = BuildOptions::default();
        build(&mut map, &cloud, &[[1.0, 0.0, 50.0], [21.0, 0.0, 50.0]], &o).unwrap();
        let before: Vec<_> = map.lanes().cloned().collect();
        let report = build(
            &mut map,
            &cloud,
            &[[13.0, 0.0, 50.0], [29.0, 0.0, 50.0]],
            &o,
        )
        .unwrap();
        assert_eq!(report.roads, 1);
        assert_eq!(report.lanes, 2);
        assert!((report.reused_length - 8.0).abs() < 1e-6, "{report:?}");
        assert!((report.added_length - 8.0).abs() < 1e-6, "{report:?}");
        assert_eq!(report.joined_connections, 2);
        for old in before {
            assert_eq!(map.lane(old.id), Some(&old));
        }
        assert_eq!(map.lanes().count(), 4);
        let unchanged = map.clone();
        assert_eq!(
            build(
                &mut map,
                &cloud,
                &[[13.0, 0.0, 50.0], [29.0, 0.0, 50.0]],
                &o
            )
            .unwrap()
            .lanes,
            0
        );
        assert_eq!(map, unchanged);
    }

    #[test]
    fn opposite_direction_parallel_roads_and_other_levels_are_not_fused() {
        for (y, z, reverse) in [(0.0, 0.0, true), (10.0, 0.0, false), (0.0, 5.0, false)] {
            let mut cloud = marked_road();
            let mut map = Map::new();
            let o = BuildOptions {
                backward_lanes: 0,
                ..Default::default()
            };
            let a = [1.0, 0.0, 50.0];
            let b = [29.0, 0.0, 50.0];
            build(&mut map, &cloud, &[a, b], &o).unwrap();
            for p in &mut cloud.positions {
                p[1] += y;
                p[2] += z;
            }
            let mut poses = [[a[0], y, 50.0], [b[0], y, 50.0]];
            if reverse {
                poses.reverse();
            }
            let report = build(&mut map, &cloud, &poses, &o).unwrap();
            assert_eq!(report.reused_intervals, 0, "{report:?}");
            assert_eq!(report.lanes, 1);
            assert_eq!(report.joined_connections, 0);
            assert_eq!(map.lanes().count(), 2);
        }
    }

    #[test]
    fn repeat_matching_handles_edge_heading_and_staggered_section_cuts() {
        let reference: Vec<_> = (0..=30).map(|k| [k as f64 * 2.0, 0.0, 0.0]).collect();
        let boundaries: Vec<Vec<_>> = [3.5, 0.0, -3.5]
            .into_iter()
            .enumerate()
            .map(|(j, y)| {
                reference
                    .iter()
                    .enumerate()
                    .map(|(k, p)| {
                        let offset = if j == 1 {
                            0.0
                        } else {
                            (k as f64 * 0.9).sin() * 0.9
                        };
                        // A sharp ground-height change also exposes different
                        // arc-length interpolation after a section is split.
                        [p[0], y + offset, if k == 15 { 2.0 } else { 0.0 }]
                    })
                    .collect()
            })
            .collect();
        let road = ExtractedRoad {
            reference: reference.clone(),
            boundaries: boundaries.clone(),
            evidence: vec![vec![Evidence::Curb; reference.len()]; 3],
            source_boundaries: None,
        };
        let lanes = vec![
            RoadLane::new(3.5, LaneDirection::Backward),
            RoadLane::new(3.5, LaneDirection::Forward),
        ];
        let line = |p: Vec<[f64; 3]>| {
            Polyline3::new(
                p.into_iter()
                    .map(|p| Point3::new(p[0], p[1], p[2]))
                    .collect(),
            )
        };
        let mut spec = NewRoad::new(line(reference), lanes.clone());
        spec.boundaries = Some(boundaries.into_iter().map(line).collect());
        spec.segment_length = Some(10.0);
        let mut map = Map::new();
        map.build_road(spec).unwrap();
        let before = map.clone();
        let (_, mut report) = extract(
            &marked_road(),
            &[[1.0, 0.0, 50.0], [29.0, 0.0, 50.0]],
            &BuildOptions::default(),
        )
        .unwrap();
        let parts = integration::uncovered(&map, road, &lanes, 0.5, &mut report);
        assert!(parts.is_empty(), "unmatched geometry: {parts:?}");
        assert!((report.reused_length - 60.0).abs() < 1e-6);
        assert_eq!(map, before);
    }

    #[test]
    fn malformed_existing_edges_are_not_used_as_matching_candidates() {
        let cloud = marked_road();
        let poses = [[1.0, 0.0, 50.0], [29.0, 0.0, 50.0]];
        let options = BuildOptions::default();
        let mut map = Map::new();
        build(&mut map, &cloud, &poses, &options).unwrap();
        let mut doc = map.to_document();
        // An imported map can carry a usable explicit centre but a one-point
        // edge. Such an edge has no segment or direction to compare.
        for lane in &mut doc.lanes {
            lane.centerline = map.centerline(lane.id);
        }
        for edge in &mut doc.boundaries {
            edge.geometry.points.truncate(1);
        }
        let (mut map, _) = doc.into_map().unwrap();
        let before = map.clone();
        let report = build(&mut map, &cloud, &poses, &options).unwrap();
        assert_eq!(report.reused_intervals, 0);
        assert_eq!(report.lanes, 2);
        for edge in before.boundaries() {
            assert_eq!(map.boundary(edge.id), Some(edge));
        }
    }

    #[test]
    fn competing_unlinked_lane_geometries_are_not_silently_chosen() {
        let cloud = marked_road();
        let poses = [[1.0, 0.0, 50.0], [29.0, 0.0, 50.0]];
        let mut map = Map::new();
        let disabled = BuildOptions {
            merge_repeated_passes: false,
            ..Default::default()
        };
        build(&mut map, &cloud, &poses, &disabled).unwrap();
        build(&mut map, &cloud, &poses, &disabled).unwrap();
        let report = build(&mut map, &cloud, &poses, &BuildOptions::default()).unwrap();
        assert_eq!(report.reused_intervals, 0);
        assert_eq!(report.joined_connections, 0);
        assert!(report.warnings.iter().any(|w| w.contains("ambiguous")));
    }
}
