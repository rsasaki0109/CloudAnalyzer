//! Search road corridors for measured feature proposals. No surveyed feature
//! coordinates, semantic labels, or traffic priorities are detector inputs.
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use vectormap_core::{
    Attributes, CrosswalkGeometry, LaneId, Map, NewCrosswalk, NewStopLine, NewTrafficSignal,
    Point3, Polyline3, SignalKind, StopLineChoice, StopLinePlacement, StopRule,
};

use super::{BuildError, crosswalks, intensity, signals};
use crate::{PointCloud, cluster, kdtree::KdTree};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiscoveryOptions {
    /// Distance from generated lane centrelines, not a manually placed ROI.
    pub corridor_radius: f64,
    pub brightness_fraction: f64,
    pub scope: SearchScope,
}
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchScope {
    #[default]
    RoadCorridor,
    GroundSurface,
}
impl Default for DiscoveryOptions {
    fn default() -> Self {
        Self {
            corridor_radius: 12.0,
            brightness_fraction: 0.65,
            scope: SearchScope::RoadCorridor,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Evidence {
    RepeatedPaint {
        measurement: crosswalks::CrosswalkCandidate,
        options: crosswalks::CrosswalkOptions,
    },
    #[serde(rename = "bright_bar")]
    TransversePaint {
        transverse_to_road: bool,
        geometry: Vec<[f64; 3]>,
        width: f64,
        thickness: f64,
        points: usize,
        flank_points: [usize; 2],
        flank_brightness_fraction: [f64; 2],
        plane_rms: f64,
    },
    ElevatedPanel {
        geometry: Vec<[f64; 3]>,
        height: f64,
        width: f64,
        thickness: f64,
        points: usize,
        plane_rms: f64,
    },
}
#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub id: usize,
    pub key: String,
    pub min: [f64; 3],
    pub max: [f64; 3],
    /// Nearby geometry only; never a claim of crossing/control/priority.
    pub nearby_lanes: Vec<LaneId>,
    pub evidence: Evidence,
    pub review_required: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryReport {
    pub candidates: Vec<Candidate>,
    pub detected_candidates: usize,
    pub limited: bool,
    pub source_points: usize,
    pub corridor_points: usize,
    pub windows: usize,
    pub unsupported_windows: usize,
    pub warnings: Vec<String>,
}

struct Sample {
    p: [f64; 3],
    heading: [f64; 2],
    lane: LaneId,
}
fn surface_samples(cloud: &PointCloud) -> Result<Vec<Sample>, BuildError> {
    // First reject isolated low outliers with supported 0.5 m cells, then
    // seed each 8 m tile from its lower supported surfaces. These are scan
    // anchors, never invented roads or semantic ground-truth labels.
    let mut cells: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
    for (i, p) in cloud.positions.iter().enumerate() {
        if p.iter().all(|v| v.is_finite()) {
            cells
                .entry(((p[0] / 0.5).floor() as i64, (p[1] / 0.5).floor() as i64))
                .or_default()
                .push(i);
        }
    }
    let mut tiles: BTreeMap<(i64, i64), Vec<[f64; 3]>> = BTreeMap::new();
    for indices in cells.values().filter(|v| v.len() >= 3) {
        let x = indices.iter().map(|&i| cloud.positions[i][0]).sum::<f64>() / indices.len() as f64;
        let y = indices.iter().map(|&i| cloud.positions[i][1]).sum::<f64>() / indices.len() as f64;
        let z = quantile(
            &mut indices
                .iter()
                .map(|&i| cloud.positions[i][2])
                .collect::<Vec<_>>(),
            0.2,
        );
        tiles
            .entry(((x / 8.0).floor() as i64, (y / 8.0).floor() as i64))
            .or_default()
            .push([x, y, z]);
    }
    let mut result = Vec::new();
    for (cell, p) in tiles.iter().filter(|(_, p)| p.len() >= 16) {
        let z = quantile(&mut p.iter().map(|p| p[2]).collect::<Vec<_>>(), 0.05);
        result.push(Sample {
            p: [(cell.0 as f64 + 0.5) * 8.0, (cell.1 as f64 + 0.5) * 8.0, z],
            heading: [0.0, 0.0],
            lane: LaneId(0),
        });
    }
    if result.is_empty() || result.len() > 2000 {
        return Err(BuildError("whole-ground search needs supported surface tiles and at most 2000 tiles; split the source scene".into()));
    }
    Ok(result)
}
fn samples(map: &Map) -> Result<Vec<Sample>, BuildError> {
    let mut result = Vec::new();
    for lane in map.lanes() {
        if let Some(line) = map.centerline(lane.id) {
            if !line.length().is_finite()
                || line.length() > 40_000.0
                || line.points.len() > 20_000
                || line.points.iter().any(|p| {
                    ![p.x, p.y, p.z]
                        .iter()
                        .all(|v| v.is_finite() && v.abs() < 1e12)
                })
            {
                return Err(BuildError(
                    "feature search road extent is excessive or nonfinite; split the scene".into(),
                ));
            }
            let line = line.resample(2.0);
            for (i, p) in line.points.iter().enumerate() {
                let a = &line.points[i.saturating_sub(1)];
                let b = &line.points[(i + 1).min(line.points.len() - 1)];
                let d = (b.x - a.x).hypot(b.y - a.y);
                if d > 1e-6 && [p.x, p.y, p.z].iter().all(|v| v.is_finite()) {
                    result.push(Sample {
                        p: [p.x, p.y, p.z],
                        heading: [(b.x - a.x) / d, (b.y - a.y) / d],
                        lane: lane.id,
                    });
                }
                if result.len() > 20_000 {
                    return Err(BuildError("feature search supports at most 20000 road samples; split the generated scene".into()));
                }
            }
        }
    }
    if result.is_empty() {
        return Err(BuildError(
            "generate usable roads before searching for features".into(),
        ));
    }
    Ok(result)
}
fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let p = (v.len() - 1) as f64 * q;
    let i = p.floor() as usize;
    v[i] + (v[p.ceil() as usize] - v[i]) * (p - i as f64)
}
fn bounds(points: &[[f64; 3]], padding: f64) -> ([f64; 3], [f64; 3]) {
    (
        std::array::from_fn(|i| {
            points.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min) - padding
        }),
        std::array::from_fn(|i| {
            points
                .iter()
                .map(|p| p[i])
                .fold(f64::NEG_INFINITY, f64::max)
                + padding
        }),
    )
}
fn center(c: &Candidate) -> [f64; 3] {
    std::array::from_fn(|i| (c.min[i] + c.max[i]) * 0.5)
}
fn key(c: &Candidate) -> String {
    // Sub-micrometre serialization noise from Lanelet2 coordinates must not
    // invalidate an unchanged proposal. Stored geometry retains full f64.
    fn quantize(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Number(n) if n.is_f64() => {
                let x = n.as_f64().expect("float");
                *v = serde_json::json!((x * 1e6).round() / 1e6);
            }
            serde_json::Value::Array(a) => a.iter_mut().for_each(quantize),
            serde_json::Value::Object(o) => o.values_mut().for_each(quantize),
            _ => {}
        }
    }
    let mut v =
        serde_json::to_value((&c.min, &c.max, &c.evidence)).expect("finite measured evidence");
    quantize(&mut v);
    serde_json::to_string(&v).expect("measured support snapshot")
}
fn same(a: &Candidate, b: &Candidate) -> bool {
    if let (
        Evidence::TransversePaint {
            geometry: a,
            width: aw,
            thickness: at,
            ..
        },
        Evidence::TransversePaint {
            geometry: b,
            width: bw,
            thickness: bt,
            ..
        },
    ) = (&a.evidence, &b.evidence)
    {
        let axis = [(a[1][0] - a[0][0]) / aw, (a[1][1] - a[0][1]) / aw];
        let other = [(b[1][0] - b[0][0]) / bw, (b[1][1] - b[0][1]) / bw];
        if (axis[0] * other[0] + axis[1] * other[1]).abs() < 0.966 {
            return false;
        }
        let project = |p: [f64; 3]| (p[0] - a[0][0]) * axis[0] + (p[1] - a[0][1]) * axis[1];
        let across =
            |p: [f64; 3]| ((p[0] - a[0][0]) * (-axis[1]) + (p[1] - a[0][1]) * axis[0]).abs();
        let lo = project(b[0]).min(project(b[1]));
        let hi = project(b[0]).max(project(b[1]));
        return hi.min(*aw) - lo.max(0.0) > aw.min(*bw) * 0.6
            && b.iter().all(|&p| across(p) < at.max(*bt) * 0.5 + 0.15)
            && (a[0][2] - b[0][2]).abs() < 0.3;
    }
    std::mem::discriminant(&a.evidence) == std::mem::discriminant(&b.evidence) && {
        let x = center(a);
        let y = center(b);
        (x[0] - y[0]).hypot(x[1] - y[1]) < 2.0 && (x[2] - y[2]).abs() < 1.0
    }
}
fn score(c: &Candidate) -> f64 {
    match &c.evidence {
        Evidence::RepeatedPaint { measurement, .. } => measurement.score,
        Evidence::TransversePaint { points, .. } | Evidence::ElevatedPanel { points, .. } => {
            *points as f64
        }
    }
}
fn push(candidates: &mut Vec<Candidate>, c: Candidate) {
    if let Some(existing) = candidates.iter_mut().find(|x| same(x, &c)) {
        if score(&c) > score(existing) {
            *existing = c;
        }
    } else {
        candidates.push(c);
    }
}
fn nearby(route: &[Sample], p: [f64; 3], radius: f64) -> Vec<LaneId> {
    route
        .iter()
        .filter(|s| (s.p[0] - p[0]).hypot(s.p[1] - p[1]) <= radius && (s.p[2] - p[2]).abs() <= 9.0)
        .map(|s| s.lane)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Preview only. Operator confirmation is required before creating semantics.
pub fn propose(
    map: &Map,
    cloud: &PointCloud,
    o: &DiscoveryOptions,
) -> Result<DiscoveryReport, BuildError> {
    if !o.corridor_radius.is_finite()
        || !(4.0..=18.0).contains(&o.corridor_radius)
        || !o.brightness_fraction.is_finite()
        || !(0.4..=0.9).contains(&o.brightness_fraction)
    {
        return Err(BuildError(
            "feature search needs corridor radius 4–18 m and brightness fraction 0.4–0.9".into(),
        ));
    }
    if cloud
        .colors
        .as_ref()
        .is_some_and(|v| v.len() != cloud.len())
        || cloud
            .attributes
            .iter()
            .any(|a| a.values.len() != cloud.len())
    {
        return Err(BuildError(
            "point attributes must match position count".into(),
        ));
    }
    if o.scope == SearchScope::GroundSurface
        && (cloud.len() > 2_000_000
            || cloud
                .positions
                .iter()
                .flatten()
                .any(|v| v.is_finite() && v.abs() > 1e12))
    {
        return Err(BuildError("whole-ground search supports at most 2000000 source points in metre coordinates; split the source".into()));
    }
    let road_route = if map.lanes().next().is_none() {
        Vec::new()
    } else {
        samples(map)?
    };
    let route = if o.scope == SearchScope::GroundSurface {
        surface_samples(cloud)?
    } else {
        samples(map)?
    };
    let tree = KdTree::new(
        &route
            .iter()
            .map(|s| [s.p[0], s.p[1], 0.0])
            .collect::<Vec<_>>(),
    )
    .expect("nonempty bounded route");
    let mut grid: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
    let mut elevated = Vec::new();
    let mut corridor_points = 0;
    for (i, p) in cloud.positions.iter().enumerate() {
        if !p.iter().all(|v| v.is_finite()) {
            continue;
        }
        let hit = tree.nearest(&[p[0], p[1], 0.0], None);
        let height = p[2] - route[hit.index].p[2];
        if hit.distance_sq <= o.corridor_radius.powi(2) && (-1.5..=8.0).contains(&height) {
            corridor_points += 1;
            if corridor_points > 2_000_000 {
                return Err(BuildError(
                    "feature search corridor exceeds 2000000 points; split the generated scene"
                        .into(),
                ));
            }
            grid.entry(((p[0] / 8.0).floor() as i64, (p[1] / 8.0).floor() as i64))
                .or_default()
                .push(i);
            if (1.5..=8.0).contains(&height) {
                elevated.push(i);
            }
        }
    }
    let mut windows = Vec::<[f64; 3]>::new();
    for s in &route {
        if windows
            .iter()
            .all(|p| (s.p[0] - p[0]).hypot(s.p[1] - p[1]) >= 6.0 || (s.p[2] - p[2]).abs() > 1.5)
        {
            windows.push(s.p);
        }
    }
    if windows.len()
        > if o.scope == SearchScope::GroundSurface {
            2000
        } else {
            500
        }
    {
        return Err(BuildError(
            "feature search exceeds the window limit; split the source scene".into(),
        ));
    }
    let mut candidates = Vec::new();
    let mut unsupported_windows = 0;
    let mut local_profiles_limited_windows = 0;
    let rgb = cloud.colors.as_ref();
    for p in &windows {
        let radius = o.corridor_radius.min(10.0);
        let min = [p[0] - radius, p[1] - radius, p[2] - 1.5];
        let max = [p[0] + radius, p[1] + radius, p[2] + 1.5];
        let mut indices = Vec::new();
        for x in (min[0] / 8.0).floor() as i64..=(max[0] / 8.0).floor() as i64 {
            for y in (min[1] / 8.0).floor() as i64..=(max[1] / 8.0).floor() as i64 {
                if let Some(v) = grid.get(&(x, y)) {
                    indices.extend(v.iter().copied().filter(|&i| {
                        (0..3).all(|j| {
                            cloud.positions[i][j] >= min[j] && cloud.positions[i][j] <= max[j]
                        })
                    }));
                }
            }
        }
        if indices.len() > 200_000 || indices.len() < 100 {
            unsupported_windows += 1;
            continue;
        }
        let subset = cloud.select(&indices);
        let options = crosswalks::CrosswalkOptions {
            min,
            max,
            lanes: vec![],
            candidate: 0,
            brightness_fraction: o.brightness_fraction,
        };
        match crosswalks::propose(map, &subset, &options) {
            Ok(report) => {
                local_profiles_limited_windows += usize::from(report.local_profiles_limited);
                for (i, measurement) in report.candidates.into_iter().enumerate() {
                    let support: Vec<_> = measurement
                        .outline
                        .iter()
                        .chain(measurement.stripes.iter().flatten())
                        .copied()
                        .collect();
                    let (min, max) = bounds(&support, 0.1);
                    let mut options = options.clone();
                    options.candidate = i;
                    let c = Candidate {
                        id: 0,
                        key: String::new(),
                        min,
                        max,
                        nearby_lanes: nearby(
                            &road_route,
                            [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, p[2]],
                            radius,
                        ),
                        evidence: Evidence::RepeatedPaint {
                            measurement,
                            options,
                        },
                        review_required: true,
                    };
                    push(&mut candidates, c);
                }
            }
            Err(_) => unsupported_windows += 1,
        }
        // Transverse bright components. Heading rejects longitudinal lane lines.
        let ground: Vec<_> = indices
            .iter()
            .copied()
            .filter(|&i| {
                let q = cloud.positions[i];
                let hit = tree.nearest(&[q[0], q[1], 0.0], None);
                (q[2] - route[hit.index].p[2]).abs() <= 0.16
            })
            .collect();
        let original_brightness = |i: usize| {
            intensity(cloud, i).or_else(|| rgb.map(|v| *v[i].iter().min().unwrap() as f64))
        };
        let mut values: Vec<_> = ground
            .iter()
            .filter_map(|&i| original_brightness(i))
            .filter(|v| v.is_finite())
            .collect();
        if values.len() < 100 {
            continue;
        }
        let fallback =
            rgb.is_some() && quantile(&mut values, 0.999) <= quantile(&mut values, 0.1) + 1e-6;
        let brightness = |i: usize| {
            if fallback {
                rgb.map(|v| *v[i].iter().min().unwrap() as f64)
            } else {
                original_brightness(i)
            }
        };
        if fallback {
            values = ground.iter().filter_map(|&i| brightness(i)).collect();
        }
        let low = quantile(&mut values, 0.1);
        // A single narrow bar occupies far less than 5% of an automatic 20 m
        // window. Continuity and observed dark flanks reject isolated outliers.
        let high = quantile(&mut values, 0.999);
        if high - low < 1e-6 {
            continue;
        }
        let paint: Vec<_> = ground
            .iter()
            .copied()
            .filter(|&i| {
                brightness(i).is_some_and(|v| v >= low + o.brightness_fraction * (high - low))
            })
            .collect();
        let points: Vec<_> = paint.iter().map(|&i| cloud.positions[i]).collect();
        let labels = cluster::euclidean_clusters(&points, 0.25, 15);
        let mut groups = vec![Vec::new(); labels.sizes.len()];
        for (q, label) in points.iter().zip(labels.labels) {
            if label != cluster::NOISE {
                groups[label as usize].push(*q);
            }
        }
        for group in groups {
            let mean: [f64; 3] = std::array::from_fn(|i| {
                group.iter().map(|q| q[i]).sum::<f64>() / group.len() as f64
            });
            let xx = group.iter().map(|q| (q[0] - mean[0]).powi(2)).sum::<f64>();
            let yy = group.iter().map(|q| (q[1] - mean[1]).powi(2)).sum::<f64>();
            let xy = group
                .iter()
                .map(|q| (q[0] - mean[0]) * (q[1] - mean[1]))
                .sum::<f64>();
            let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
            let axis = [angle.cos(), angle.sin()];
            let normal = [-axis[1], axis[0]];
            let heading = route[tree.nearest(&[mean[0], mean[1], 0.0], None).index].heading;
            if heading != [0.0, 0.0] && (axis[0] * heading[0] + axis[1] * heading[1]).abs() > 0.5 {
                continue;
            }
            let mut along: Vec<_> = group
                .iter()
                .map(|q| (q[0] - mean[0]) * axis[0] + (q[1] - mean[1]) * axis[1])
                .collect();
            let mut across: Vec<_> = group
                .iter()
                .map(|q| (q[0] - mean[0]) * normal[0] + (q[1] - mean[1]) * normal[1])
                .collect();
            let a = quantile(&mut along, 0.02);
            let b = quantile(&mut along, 0.98);
            let thickness = quantile(&mut across, 0.95) - quantile(&mut across, 0.05);
            if !(1.5..=16.0).contains(&(b - a)) || !(0.15..=0.9).contains(&thickness) {
                continue;
            }
            let z_slope = group
                .iter()
                .map(|q| {
                    ((q[0] - mean[0]) * axis[0] + (q[1] - mean[1]) * axis[1]) * (q[2] - mean[2])
                })
                .sum::<f64>()
                / along.iter().map(|s| s * s).sum::<f64>().max(1e-12);
            let rms = (group
                .iter()
                .map(|q| {
                    let s = (q[0] - mean[0]) * axis[0] + (q[1] - mean[1]) * axis[1];
                    (q[2] - mean[2] - s * z_slope).powi(2)
                })
                .sum::<f64>()
                / group.len() as f64)
                .sqrt();
            if rms > 0.06
                || group
                    .iter()
                    .any(|q| (0..2).any(|i| q[i] < min[i] + 0.2 || q[i] > max[i] - 0.2))
            {
                continue;
            }
            let occupied: BTreeSet<_> = along
                .iter()
                .filter(|s| **s >= a && **s <= b)
                .map(|s| ((s - a) / 0.25).floor() as usize)
                .collect();
            if occupied.len() as f64 / ((b - a) / 0.25).ceil() < 0.8 {
                continue;
            }
            let mut flank_points = [0usize; 2];
            let mut flank_white = [0usize; 2];
            let offset = quantile(&mut across, 0.5);
            for &i in &ground {
                let q = cloud.positions[i];
                let s = (q[0] - mean[0]) * axis[0] + (q[1] - mean[1]) * axis[1];
                let t = (q[0] - mean[0]) * normal[0] + (q[1] - mean[1]) * normal[1] - offset;
                if s < a
                    || s > b
                    || t.abs() < thickness * 0.5 + 0.15
                    || t.abs() > thickness * 0.5 + 0.65
                    || (q[2] - mean[2] - s * z_slope).abs() > 0.08
                {
                    continue;
                }
                let side = usize::from(t >= 0.0);
                flank_points[side] += 1;
                flank_white[side] += usize::from(
                    brightness(i).is_some_and(|v| v >= low + o.brightness_fraction * (high - low)),
                );
            }
            if flank_points.iter().any(|n| *n < 12) {
                continue;
            }
            let flank_fraction =
                std::array::from_fn(|i| flank_white[i] as f64 / flank_points[i] as f64);
            if flank_fraction.iter().any(|f| *f > 0.35) {
                continue;
            }
            let geometry = vec![
                [
                    mean[0] + a * axis[0],
                    mean[1] + a * axis[1],
                    mean[2] + a * z_slope,
                ],
                [
                    mean[0] + b * axis[0],
                    mean[1] + b * axis[1],
                    mean[2] + b * z_slope,
                ],
            ];
            let (min, max) = bounds(&group, 0.05);
            let c = Candidate {
                id: 0,
                key: String::new(),
                min,
                max,
                nearby_lanes: nearby(&road_route, mean, radius),
                evidence: Evidence::TransversePaint {
                    transverse_to_road: heading != [0.0, 0.0],
                    geometry,
                    width: b - a,
                    thickness,
                    points: group.len(),
                    flank_points,
                    flank_brightness_fraction: flank_fraction,
                    plane_rms: rms,
                },
                review_required: true,
            };
            push(&mut candidates, c);
        }
    }
    // Remove narrow vertical stems using supported horizontal slices before
    // joining faces vertically. A connected pole must not absorb its head.
    let mut layers: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for &i in &elevated {
        layers
            .entry((cloud.positions[i][2] / 0.2).floor() as i64)
            .or_default()
            .push(i);
    }
    let mut faces = BTreeSet::new();
    for layer in layers.values() {
        let points: Vec<_> = layer.iter().map(|&i| cloud.positions[i]).collect();
        let labels = cluster::euclidean_clusters(&points, 0.25, 3);
        let mut groups = vec![Vec::new(); labels.sizes.len()];
        for (&i, label) in layer.iter().zip(labels.labels) {
            if label != cluster::NOISE {
                groups[label as usize].push(i);
            }
        }
        for group in groups {
            let mean: [f64; 2] = std::array::from_fn(|a| {
                group.iter().map(|&i| cloud.positions[i][a]).sum::<f64>() / group.len() as f64
            });
            let xx = group
                .iter()
                .map(|&i| (cloud.positions[i][0] - mean[0]).powi(2))
                .sum::<f64>();
            let yy = group
                .iter()
                .map(|&i| (cloud.positions[i][1] - mean[1]).powi(2))
                .sum::<f64>();
            let xy = group
                .iter()
                .map(|&i| (cloud.positions[i][0] - mean[0]) * (cloud.positions[i][1] - mean[1]))
                .sum::<f64>();
            let a = 0.5 * (2.0 * xy).atan2(xx - yy);
            let axis = [a.cos(), a.sin()];
            let normal = [-axis[1], axis[0]];
            let mut along: Vec<_> = group
                .iter()
                .map(|&i| {
                    (cloud.positions[i][0] - mean[0]) * axis[0]
                        + (cloud.positions[i][1] - mean[1]) * axis[1]
                })
                .collect();
            let mut across: Vec<_> = group
                .iter()
                .map(|&i| {
                    (cloud.positions[i][0] - mean[0]) * normal[0]
                        + (cloud.positions[i][1] - mean[1]) * normal[1]
                })
                .collect();
            let width = quantile(&mut along, 0.98) - quantile(&mut along, 0.02);
            let thickness = quantile(&mut across, 0.95) - quantile(&mut across, 0.05);
            if (0.18..=3.5).contains(&width) && thickness <= (width * 0.5).min(0.5) {
                faces.extend(group);
            }
        }
    }
    let elevated: Vec<_> = faces.into_iter().collect();
    // Connected supported faces. Thin poles and broad walls fail head fitting.
    let points: Vec<_> = elevated.iter().map(|&i| cloud.positions[i]).collect();
    let labels = cluster::euclidean_clusters(&points, 0.25, 12);
    let mut groups = vec![Vec::new(); labels.sizes.len()];
    for (i, label) in elevated.into_iter().zip(labels.labels) {
        if label != cluster::NOISE {
            groups[label as usize].push(i);
        }
    }
    for indices in groups {
        let subset = cloud.select(&indices);
        let (min, max) = bounds(&subset.positions, 0.02);
        if (0..3).any(|i| max[i] - min[i] > 4.0) {
            continue;
        }
        let c: [f64; 3] = std::array::from_fn(|i| (min[i] + max[i]) * 0.5);
        // Objects cut by the corridor/height selection are incomplete. Trees,
        // walls and poles must not acquire a head shape by clipping them.
        if subset.positions.iter().any(|p| {
            let hit = tree.nearest(&[p[0], p[1], 0.0], None);
            let h = p[2] - route[hit.index].p[2];
            h < 1.65 || h > 7.85 || hit.distance_sq > (o.corridor_radius - 0.15).powi(2)
        }) {
            continue;
        }
        let heading = route[tree.nearest(&[c[0], c[1], 0.0], None).index].heading;
        if let Ok(m) = signals::measure_geometry(&subset, min, max, heading) {
            if m.width < 0.18
                || m.height > 1.2
                || m.plane_rms > (m.width * 0.14).min(0.2)
                || m.thickness > (m.width * 0.45).min(0.5)
            {
                continue;
            }
            let edge = [
                m.geometry[1][0] - m.geometry[0][0],
                m.geometry[1][1] - m.geometry[0][1],
            ];
            if heading != [0.0, 0.0]
                && ((edge[0] * heading[0] + edge[1] * heading[1]) / m.width).abs() > 0.7
            {
                continue;
            }
            // A real face needs distributed support, rather than a few leaves
            // or points along the edge of a narrow pole.
            let axis = [edge[0] / m.width, edge[1] / m.width];
            let nx = (m.width / 0.15).ceil() as usize;
            let nz = (m.height / 0.15).ceil() as usize;
            let occupied: BTreeSet<_> = subset
                .positions
                .iter()
                .filter_map(|p| {
                    let s =
                        (p[0] - m.geometry[0][0]) * axis[0] + (p[1] - m.geometry[0][1]) * axis[1];
                    let z = p[2] - m.geometry[0][2];
                    (s >= 0.0 && s <= m.width && z >= 0.0 && z <= m.height)
                        .then(|| ((s / 0.15).floor() as usize, (z / 0.15).floor() as usize))
                })
                .collect();
            if occupied.len() as f64 / ((nx * nz) as f64) < 0.55 {
                continue;
            }
            candidates.push(Candidate {
                id: 0,
                key: String::new(),
                min,
                max,
                nearby_lanes: nearby(&road_route, c, o.corridor_radius),
                evidence: Evidence::ElevatedPanel {
                    geometry: m.geometry,
                    height: m.height,
                    width: m.width,
                    thickness: m.thickness,
                    points: m.points,
                    plane_rms: m.plane_rms,
                },
                review_required: true,
            });
        }
    }
    // Do not offer the observed crosswalk bands a second time as stop lines.
    let walks: Vec<_> = candidates
        .iter()
        .filter(|c| matches!(c.evidence, Evidence::RepeatedPaint { .. }))
        .map(|c| (c.min, c.max))
        .collect();
    candidates.retain(|c| {
        !matches!(c.evidence, Evidence::TransversePaint { .. })
            || !walks.iter().any(|(min, max)| {
                let p = center(c);
                (0..2).all(|i| p[i] >= min[i] - 0.3 && p[i] <= max[i] + 0.3)
            })
    });
    let detected_candidates = candidates.len();
    let mut warnings=vec!["Proposals are geometric evidence, not automatic semantic labels. Repeated paint may be road markings, bright bars may be crossing bands or lane paint, and elevated panels may be signs or other objects. Confirm classification and lane association; no priority, lamp state, or stop-line relationship is inferred.".into()];
    // Bound the preview without hiding truncation or letting common paint bars
    // crowd out all panel/crossing proposals. Ranking is not a probability.
    candidates.sort_by(|a, b| {
        score(b)
            .total_cmp(&score(a))
            .then(a.min[0].total_cmp(&b.min[0]))
    });
    let mut counts = [0usize; 3];
    if local_profiles_limited_windows > 0 {
        warnings.push(format!("Local paint profiles reached the 64 source-seed limit in {local_profiles_limited_windows} windows; smaller components may be omitted."));
    }
    candidates.retain(|c| {
        let k = match c.evidence {
            Evidence::RepeatedPaint { .. } => 0,
            Evidence::TransversePaint { .. } => 1,
            Evidence::ElevatedPanel { .. } => 2,
        };
        counts[k] += 1;
        counts[k] <= 64
    });
    let limited = candidates.len() != detected_candidates;
    if limited {
        warnings.push(format!("Preview limited to the 64 highest-support proposals per evidence type ({detected_candidates} total detected). Crop the source to inspect omitted candidates; rankings are uncalibrated."));
    }
    if o.scope == SearchScope::GroundSurface {
        warnings.push("Whole-ground scan anchors come from lower supported surfaces, which may include roofs or other levels. No road direction is assumed for a paint bar. Review the surface and object identity over the original points.".into());
    }
    if cloud.colors.is_none() && cloud.attribute(crate::INTENSITY).is_none() {
        warnings.push("No retained RGB or intensity: paint proposals are unavailable; geometry-only panels may still be found.".into());
    }
    candidates.sort_by(|a, b| {
        a.min[0]
            .total_cmp(&b.min[0])
            .then(a.min[1].total_cmp(&b.min[1]))
            .then(a.min[2].total_cmp(&b.min[2]))
    });
    for (i, c) in candidates.iter_mut().enumerate() {
        c.id = i;
        c.key = key(c);
    }
    Ok(DiscoveryReport {
        candidates,
        detected_candidates,
        limited,
        source_points: cloud.len(),
        corridor_points,
        windows: windows.len(),
        unsupported_windows,
        warnings,
    })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Classification {
    Crosswalk,
    StopLine,
    VehicleSignal,
    PedestrianSignal,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Confirmation {
    pub candidate: usize,
    pub key: String,
    pub classification: Classification,
    /// Operator-confirmed lanes; nearby suggestions are never selected implicitly.
    pub lanes: Vec<LaneId>,
}
#[derive(Debug, Serialize)]
pub struct Addition {
    pub classification: Classification,
    pub id: u64,
    pub reused: bool,
}
fn line(p: &[[f64; 3]]) -> Polyline3 {
    Polyline3::new(p.iter().map(|p| Point3::new(p[0], p[1], p[2])).collect())
}
fn provenance(attrs: &mut Attributes, key: &str, geometry: &str) {
    attrs.insert("cloudanalyzer_discovery", key);
    attrs.insert("cloudanalyzer_geometry_source", geometry);
    attrs.insert(
        "cloudanalyzer_classification_source",
        "user_confirmed_automatic_proposal",
    );
    attrs.insert("cloudanalyzer_review_required", "yes");
}
/// Recheck source support and publish all confirmations atomically. A replay
/// retains subsequent geometry edits and makes no new IDs or Undo entries.
pub fn add(
    map: &mut Map,
    cloud: &PointCloud,
    o: &DiscoveryOptions,
    confirmations: &[Confirmation],
) -> Result<Vec<Addition>, BuildError> {
    if confirmations.is_empty() || confirmations.len() > 200 {
        return Err(BuildError(
            "confirm 1–200 feature proposals explicitly".into(),
        ));
    }
    let report = propose(map, cloud, o)?;
    let mut seen = BTreeSet::new();
    for c in confirmations {
        if !seen.insert(c.candidate)
            || c.lanes.is_empty()
            || c.lanes.iter().copied().collect::<BTreeSet<_>>().len() != c.lanes.len()
            || c.lanes.iter().any(|id| map.lane(*id).is_none())
        {
            return Err(BuildError(
                "confirm distinct proposals and distinct existing lane IDs explicitly".into(),
            ));
        }
        let candidate = report
            .candidates
            .get(c.candidate)
            .ok_or_else(|| BuildError("feature proposal no longer exists; search again".into()))?;
        if candidate.key != c.key {
            return Err(BuildError(
                "feature evidence changed; search and review again".into(),
            ));
        }
        if !matches!(
            (&candidate.evidence, c.classification),
            (Evidence::RepeatedPaint { .. }, Classification::Crosswalk)
                | (Evidence::TransversePaint { .. }, Classification::StopLine)
                | (
                    Evidence::ElevatedPanel { .. },
                    Classification::VehicleSignal | Classification::PedestrianSignal
                )
        ) {
            return Err(BuildError(
                "confirmed classification must match the measured proposal type".into(),
            ));
        }
    }
    let mut draft = map.clone();
    let mut additions = Vec::new();
    for confirmation in confirmations {
        let c = &report.candidates[confirmation.candidate];
        let association = serde_json::to_string(&(
            c.key.clone(),
            confirmation.classification,
            &confirmation.lanes,
        ))
        .map_err(|e| BuildError(e.to_string()))?;
        let stored = |a: &Attributes| {
            a.get("cloudanalyzer_discovery")
                .or_else(|| a.get_prefixed("lanelet2", "cloudanalyzer_discovery"))
                .is_some_and(|v| v == association)
        };
        let existing = match confirmation.classification {
            Classification::Crosswalk => draft
                .crosswalks()
                .find(|c| stored(&c.attributes))
                .map(|c| c.id.0),
            Classification::StopLine => draft
                .stop_lines()
                .find(|c| stored(&c.attributes))
                .map(|c| c.id.0),
            _ => draft
                .traffic_signals()
                .find(|c| stored(&c.attributes))
                .map(|c| c.id.0),
        };
        if let Some(id) = existing {
            additions.push(Addition {
                classification: confirmation.classification,
                id,
                reused: true,
            });
            continue;
        }
        let rules: BTreeSet<_> = draft.regulatory_elements().map(|r| r.id).collect();
        let id = match &c.evidence {
            Evidence::RepeatedPaint { measurement, .. } => {
                let p = &measurement.outline;
                let id = draft
                    .add_crosswalk(NewCrosswalk {
                        geometry: CrosswalkGeometry::Edges {
                            left_edge: line(&p[..2]),
                            right_edge: line(&[p[3], p[2]]),
                        },
                        crossing_lanes: Some(confirmation.lanes.clone()),
                        stop_line_offset: None,
                    })
                    .map_err(|e| BuildError(e.to_string()))?
                    .0;
                let a = &mut draft.crosswalk_mut(id).expect("new crossing").attributes;
                provenance(a, &association, "point_cloud_brightness_stripes");
                a.insert(
                    "cloudanalyzer_paint_bands",
                    serde_json::to_string(&measurement.stripes)
                        .map_err(|e| BuildError(e.to_string()))?,
                );
                id.0
            }
            Evidence::TransversePaint { geometry, .. } => {
                let id = draft
                    .add_stop_line(NewStopLine {
                        lanes: confirmation.lanes.clone(),
                        placement: StopLinePlacement::Geometry {
                            geometry: line(geometry),
                        },
                        rule: StopRule::Marking,
                    })
                    .map_err(|e| BuildError(e.to_string()))?
                    .0;
                provenance(
                    &mut draft.stop_line_mut(id).expect("new stop line").attributes,
                    &association,
                    "point_cloud_brightness_bar",
                );
                id.0
            }
            Evidence::ElevatedPanel {
                geometry, height, ..
            } => {
                let mut geometry = geometry.clone();
                let center = center(c);
                let lane = draft.centerline(confirmation.lanes[0]).ok_or_else(|| {
                    BuildError("confirmed signal lane has no usable centreline".into())
                })?;
                if lane.points.len() < 2 {
                    return Err(BuildError("confirmed signal lane has no heading".into()));
                }
                if let Some(pair) = lane.points.windows(2).min_by(|a, b| {
                    let distance = |p: &[Point3]| (p[0].x - center[0]).hypot(p[0].y - center[1]);
                    distance(a).total_cmp(&distance(b))
                }) {
                    let h = [pair[1].x - pair[0].x, pair[1].y - pair[0].y];
                    let edge = [
                        geometry[1][0] - geometry[0][0],
                        geometry[1][1] - geometry[0][1],
                    ];
                    if edge[0] * h[1] - edge[1] * h[0] < 0.0 {
                        geometry.reverse();
                    }
                }
                let id = draft
                    .add_traffic_signal(NewTrafficSignal {
                        lanes: confirmation.lanes.clone(),
                        stop_line: StopLineChoice::None,
                        geometry: Some(line(&geometry)),
                        height: Some(*height),
                        kind: if confirmation.classification == Classification::VehicleSignal {
                            SignalKind::Vehicle
                        } else {
                            SignalKind::Pedestrian
                        },
                        bulbs: Some(vec![]),
                        group: None,
                    })
                    .map_err(|e| BuildError(e.to_string()))?
                    .0;
                provenance(
                    &mut draft.traffic_signal_mut(id).expect("new signal").attributes,
                    &association,
                    "point_cloud_box_fit",
                );
                id.0
            }
        };
        let added_rules: Vec<_> = draft
            .regulatory_elements()
            .filter(|r| !rules.contains(&r.id))
            .map(|r| r.id)
            .collect();
        for id in added_rules {
            let a = &mut draft
                .regulatory_element_mut(id)
                .expect("new rule")
                .attributes;
            a.insert("cloudanalyzer_lanes_source", "user_selected");
            a.insert("cloudanalyzer_review_required", "yes");
        }
        additions.push(Addition {
            classification: confirmation.classification,
            id,
            reused: false,
        });
    }
    *map = draft;
    Ok(additions)
}
