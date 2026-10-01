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
    /// Half-length of each longitudinal slice.
    pub half_window: f64,
    /// How far a measured boundary can deviate from its nominal position.
    pub search_margin: f64,
    pub bin_width: f64,
    pub curb_height: f64,
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
            half_window: 2.0,
            search_margin: 1.5,
            bin_width: 0.2,
            curb_height: 0.08,
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
    pub support_edge_vertices: usize,
    pub width_prior_vertices: usize,
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
        support_edge_vertices: 0,
        width_prior_vertices: 0,
        observed_fraction: vec![0.0; nlanes + 1],
        warnings: Vec::new(),
    };
    let mut roads = Vec::new();
    let empty = || ExtractedRoad {
        reference: Vec::new(),
        boundaries: vec![Vec::new(); nlanes + 1],
        evidence: vec![Vec::new(); nlanes + 1],
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
            }
            road = empty();
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
                    Some(Evidence::Curb)
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
            prior[j] = c.lateral;
            road.boundaries[j].push([p[0] - dir[1] * c.lateral, p[1] + dir[0] * c.lateral, c.z]);
            road.evidence[j].push(c.evidence);
        }
    }
    if road.reference.len() >= 2 {
        roads.push(road);
    }
    if roads.is_empty() {
        return fail(
            "no continuous road surface found near the trajectory; check the coordinate frame and point density",
        );
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
    if report.support_edge_vertices > 0 {
        report.warnings.push("Some boundaries follow the end of point coverage; verify that they are road edges rather than scan gaps.".into());
    }
    if report.unsupported_sections > 0 {
        report.warnings.push(format!("{} sections lack ground support and were omitted; disconnected stretches require review.",report.unsupported_sections));
    }
    report.warnings.push("Draft map: verify lane counts, travel directions, junctions and boundary geometry before use.".into());
    Ok((roads, report))
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
    for road in roads {
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
    }
    *map = draft;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Attribute;

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
}
