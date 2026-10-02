//! Ground-paint pattern proposals inside a user-selected box. Repeated bright
//! bands are evidence for a candidate, not semantic classification or priority.
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use vectormap_core::{
    CrosswalkGeometry, CrosswalkId, LaneId, Map, NewCrosswalk, Point3, Polyline3,
};

use super::{BuildError, intensity};
use crate::PointCloud;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrosswalkOptions {
    /// Original-coordinate box around road paint, excluding buildings/kerbs.
    pub min: [f64; 3],
    pub max: [f64; 3],
    /// Explicitly confirmed crossing lanes. May be empty for preview only.
    #[serde(default)]
    pub lanes: Vec<LaneId>,
    #[serde(default)]
    pub candidate: usize,
    /// Fraction between ground brightness P10 and P95; lower values include dim
    /// paint and more background. This is a user-controlled photometric threshold.
    #[serde(default = "default_brightness_fraction")]
    pub brightness_fraction: f64,
}
fn default_brightness_fraction() -> f64 {
    0.75
}

#[derive(Debug, Clone, Serialize)]
pub struct CrosswalkCandidate {
    pub outline: Vec<[f64; 3]>,
    /// Measured bright-band extents for inspection, not stored invented paint.
    pub stripes: Vec<Vec<[f64; 3]>>,
    pub stripe_count: usize,
    pub width: f64,
    pub length: f64,
    pub angle_degrees: f64,
    /// Uncalibrated ranking statistic, not a probability.
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrosswalkReport {
    pub candidates: Vec<CrosswalkCandidate>,
    pub points: usize,
    pub ground_points: usize,
    pub plane_rms: f64,
    pub brightness_source: &'static str,
    pub added: Option<CrosswalkId>,
    pub reused: Option<CrosswalkId>,
    pub classification_source: &'static str,
    pub lanes_source: &'static str,
    pub warnings: Vec<String>,
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let p = (v.len() - 1) as f64 * q;
    let i = p.floor() as usize;
    v[i] + (v[p.ceil() as usize] - v[i]) * (p - i as f64)
}

fn plane(points: &[[f64; 3]]) -> Option<[f64; 3]> {
    let mut a = [[0.0; 4]; 3];
    for p in points {
        let row = [p[0], p[1], 1.0];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += row[i] * row[j];
            }
            a[i][3] += row[i] * p[2];
        }
    }
    for i in 0..3 {
        let pivot = (i..3).max_by(|&j, &k| a[j][i].abs().total_cmp(&a[k][i].abs()))?;
        a.swap(i, pivot);
        if a[i][i].abs() < 1e-10 {
            return None;
        }
        let d = a[i][i];
        for v in &mut a[i][i..] {
            *v /= d;
        }
        let pivot_row = a[i];
        for (j, row) in a.iter_mut().enumerate() {
            if j != i {
                let m = row[i];
                for (v, p) in row[i..].iter_mut().zip(&pivot_row[i..]) {
                    *v -= m * p;
                }
            }
        }
    }
    Some([a[0][3], a[1][3], a[2][3]])
}

/// Read-only, deterministic proposal search using retained RGB/intensity.
pub fn propose(
    map: &Map,
    cloud: &PointCloud,
    o: &CrosswalkOptions,
) -> Result<CrosswalkReport, BuildError> {
    if !o.brightness_fraction.is_finite() || !(0.4..=0.9).contains(&o.brightness_fraction) {
        return Err(BuildError(
            "brightness fraction must be finite and between 0.4 and 0.9".into(),
        ));
    }
    if (0..3).any(|i| {
        !o.min[i].is_finite()
            || !o.max[i].is_finite()
            || o.min[i] >= o.max[i]
            || o.max[i] - o.min[i] > if i == 2 { 5.0 } else { 40.0 }
    }) {
        return Err(BuildError(
            "crosswalk box must have finite increasing bounds, XY at most 40 m and Z at most 5 m"
                .into(),
        ));
    }
    let lanes: BTreeSet<_> = o.lanes.iter().copied().collect();
    if lanes.len() != o.lanes.len() || lanes.iter().any(|id| map.lane(*id).is_none()) {
        return Err(BuildError(
            "crossing lanes must be distinct existing lanes".into(),
        ));
    }
    if cloud
        .colors
        .as_ref()
        .is_some_and(|v| v.len() != cloud.positions.len())
    {
        return Err(BuildError("RGB count does not match positions".into()));
    }
    if cloud
        .attribute(crate::INTENSITY)
        .is_some_and(|v| v.values.len() != cloud.positions.len())
    {
        return Err(BuildError(
            "intensity count does not match positions".into(),
        ));
    }
    let mut points = Vec::new();
    for (i, p) in cloud.positions.iter().enumerate() {
        if (0..3).all(|j| p[j].is_finite() && p[j] >= o.min[j] && p[j] <= o.max[j]) {
            points.push((std::array::from_fn::<_, 3, _>(|j| p[j] - o.min[j]), i));
            if points.len() > 200_000 {
                return Err(BuildError(
                    "crosswalk box exceeds 200000 points; select a smaller box".into(),
                ));
            }
        }
    }
    if points.len() < 100 {
        return Err(BuildError(
            "crosswalk box needs at least 100 finite points; load full-density road paint".into(),
        ));
    }
    let mut cells: BTreeMap<(i32, i32), Vec<[f64; 3]>> = BTreeMap::new();
    for &(p, _) in &points {
        cells
            .entry(((p[0] / 0.25).floor() as i32, (p[1] / 0.25).floor() as i32))
            .or_default()
            .push(p);
    }
    let surface: Vec<_> = cells
        .values()
        .filter(|v| v.len() >= 3)
        .map(|v| {
            [
                v.iter().map(|p| p[0]).sum::<f64>() / v.len() as f64,
                v.iter().map(|p| p[1]).sum::<f64>() / v.len() as f64,
                quantile(&mut v.iter().map(|p| p[2]).collect::<Vec<_>>(), 0.2),
            ]
        })
        .collect();
    if surface.len() < 25 {
        return Err(BuildError(
            "road plane needs at least 25 supported surface cells".into(),
        ));
    }
    let mut selected = surface.clone();
    let mut fit = [0.0; 3];
    for _ in 0..5 {
        fit = plane(&selected)
            .ok_or_else(|| BuildError("road surface has no usable plane".into()))?;
        let residual: Vec<_> = surface
            .iter()
            .map(|p| p[2] - fit[0] * p[0] - fit[1] * p[1] - fit[2])
            .collect();
        let low = quantile(&mut residual.clone(), 0.1);
        let high = quantile(&mut residual.clone(), 0.8);
        selected = surface
            .iter()
            .zip(residual)
            .filter(|(_, r)| *r >= low && *r <= high)
            .map(|(p, _)| *p)
            .collect();
    }
    if fit[0].hypot(fit[1]) > 0.3 {
        return Err(BuildError(
            "selected surface is too steep for road-paint measurement".into(),
        ));
    }
    let ground: Vec<_> = points
        .iter()
        .filter(|(p, _)| (p[2] - fit[0] * p[0] - fit[1] * p[1] - fit[2]).abs() <= 0.08)
        .copied()
        .collect();
    if ground.len() < 100 {
        return Err(BuildError(
            "road plane has too few supported ground returns".into(),
        ));
    }
    let rms = (ground
        .iter()
        .map(|(p, _)| (p[2] - fit[0] * p[0] - fit[1] * p[1] - fit[2]).powi(2))
        .sum::<f64>()
        / ground.len() as f64)
        .sqrt();
    let mut brightness: Vec<_> = ground.iter().map(|(_, i)| intensity(cloud, *i)).collect();
    let usable = brightness.iter().all(Option::is_some)
        && (cloud.colors.is_none() || {
            let mut v: Vec<_> = brightness.iter().flatten().copied().collect();
            quantile(&mut v, 0.95) > quantile(&mut v, 0.1)
        });
    let source = if usable {
        "intensity"
    } else if let Some(colors) = &cloud.colors {
        brightness = ground
            .iter()
            .map(|(_, i)| Some(f64::from(*colors[*i].iter().min().unwrap())))
            .collect();
        "rgb_min_channel"
    } else {
        return Err(BuildError(
            "paint measurement needs retained intensity or RGB; XYZ alone cannot reveal stripes"
                .into(),
        ));
    };
    let mut values: Vec<_> = brightness.into_iter().flatten().collect();
    let base = quantile(&mut values.clone(), 0.1);
    let peak = quantile(&mut values.clone(), 0.95);
    let mut report = CrosswalkReport {
        candidates: vec![], points: points.len(), ground_points: ground.len(), plane_rms: rms,
        brightness_source: source, added: None, reused: None,
        classification_source: "unconfirmed_paint_pattern", lanes_source: if o.lanes.is_empty() { "unassigned" } else { "user_selected" },
        warnings: vec!["Repeated bright ground bands are draft candidates, not an object classifier. Missing paint can shorten the measured outline. Confirm paint extents, crossing lanes and legal priority; stop lines are not inferred. Scores are uncalibrated ranking statistics.".into()],
    };
    if peak <= base + 1e-6 {
        report
            .warnings
            .push("No usable ground-brightness contrast.".into());
        return Ok(report);
    }
    // Reconstruct in source order; sorting was only used on clones for quantiles.
    values = ground
        .iter()
        .map(|(_, i)| {
            if usable {
                intensity(cloud, *i).unwrap()
            } else {
                f64::from(*cloud.colors.as_ref().unwrap()[*i].iter().min().unwrap())
            }
        })
        .collect();
    let bright: Vec<_> = values
        .iter()
        .map(|v| *v > base + o.brightness_fraction * (peak - base))
        .collect();
    for angle in (0..180).step_by(2) {
        let a = (angle as f64).to_radians();
        let axis = [a.cos(), a.sin()];
        let normal = [-axis[1], axis[0]];
        let s: Vec<_> = ground
            .iter()
            .map(|(p, _)| p[0] * axis[0] + p[1] * axis[1])
            .collect();
        let low = s.iter().copied().fold(f64::INFINITY, f64::min);
        let bins: Vec<_> = s
            .iter()
            .map(|v| ((v - low) / 0.1).floor() as usize)
            .collect();
        let mut counts = vec![0usize; bins.iter().max().unwrap() + 1];
        let mut whites = vec![0usize; counts.len()];
        for (&i, &b) in bins.iter().zip(&bright) {
            counts[i] += 1;
            whites[i] += usize::from(b);
        }
        let ratio: Vec<_> = counts
            .iter()
            .zip(whites)
            .map(|(&n, b)| if n >= 5 { b as f64 / n as f64 } else { 0.0 })
            .collect();
        let mut supported: Vec<_> = ratio
            .iter()
            .zip(&counts)
            .filter(|(_, n)| **n >= 5)
            .map(|(r, _)| *r)
            .collect();
        if supported.is_empty() {
            continue;
        }
        let threshold = (quantile(&mut supported, 0.9) * 0.7).max(0.15);
        let mut runs = Vec::new();
        let mut i = 0;
        while i < ratio.len() {
            if ratio[i] <= threshold {
                i += 1;
                continue;
            }
            let start = i;
            while i < ratio.len() && ratio[i] > threshold {
                i += 1;
            }
            if (3..=9).contains(&(i - start)) {
                runs.push((start, i));
            }
        }
        let mut chains = Vec::new();
        let mut chain = Vec::new();
        for run in runs {
            if let Some(&(_, end)) = chain.last()
                && !(2..=12).contains(&(run.0 - end))
            {
                chains.push(std::mem::take(&mut chain));
            }
            chain.push(run);
        }
        chains.push(chain);
        for chain in chains.into_iter().filter(|c| c.len() >= 3) {
            let mut extents = Vec::new();
            for &(start, end) in &chain {
                let mut t: Vec<_> = ground
                    .iter()
                    .zip(&bins)
                    .zip(&bright)
                    .filter(|((_, bin), b)| **b && **bin >= start && **bin < end)
                    .map(|(((p, _), _), _)| p[0] * normal[0] + p[1] * normal[1])
                    .collect();
                if t.len() < 12 {
                    break;
                }
                let left = quantile(&mut t, 0.05);
                let right = quantile(&mut t, 0.95);
                if right - left < 1.5 {
                    break;
                }
                // A transverse stripe must be continuous across its measured
                // extent. Oblique aliases through separated paint bars fail
                // this coverage check even when their 1D profile is periodic.
                let bin_count = ((right - left) / 0.25).ceil() as usize;
                let occupied: BTreeSet<_> = t
                    .iter()
                    .filter(|v| **v >= left && **v <= right)
                    .map(|v| (((v - left) / 0.25).floor() as usize).min(bin_count - 1))
                    .collect();
                if occupied.len() as f64 / (bin_count as f64) < 0.8 {
                    break;
                }
                extents.push((left, right));
            }
            if extents.len() != chain.len() {
                continue;
            }
            let mut centers: Vec<_> = extents.iter().map(|(l, r)| (l + r) / 2.0).collect();
            let center = quantile(&mut centers, 0.5);
            if centers.iter().any(|v| (*v - center).abs() > 1.0) {
                continue;
            }
            let t0 = quantile(&mut extents.iter().map(|v| v.0).collect::<Vec<_>>(), 0.5);
            let t1 = quantile(&mut extents.iter().map(|v| v.1).collect::<Vec<_>>(), 0.5);
            // Missing returns are not dark paint. Require contrasting *observed*
            // gaps over the same transverse footprint as both neighbouring
            // bands; a regularly sampled reflective surface must not acquire
            // stripes merely because some 0.1 m profile bins have no points.
            let local_fraction = |start: usize, end: usize, left: f64, right: f64| {
                let mut n = 0usize;
                let mut white = 0usize;
                for (((p, _), bin), b) in ground.iter().zip(&bins).zip(&bright) {
                    let t = p[0] * normal[0] + p[1] * normal[1];
                    if *bin >= start && *bin < end && t >= left && t <= right {
                        n += 1;
                        white += usize::from(*b);
                    }
                }
                (n, if n > 0 { white as f64 / n as f64 } else { 0.0 })
            };
            if chain.iter().enumerate().any(|(i, &(start, end))| {
                let (n, f) = local_fraction(start, end, extents[i].0, extents[i].1);
                n < 12 || f < 0.4
            }) {
                continue;
            }
            if chain.windows(2).enumerate().any(|(i, pair)| {
                let left = extents[i].0.max(extents[i + 1].0);
                let right = extents[i].1.min(extents[i + 1].1);
                if right - left < 1.5 {
                    return true;
                }
                let (n, gap) = local_fraction(pair[0].1, pair[1].0, left, right);
                let band = local_fraction(pair[0].0, pair[0].1, left, right)
                    .1
                    .min(local_fraction(pair[1].0, pair[1].1, left, right).1);
                n < 12 || gap > 0.35 || gap > band * 0.45
            }) {
                continue;
            }
            let s0 = low + chain[0].0 as f64 * 0.1;
            let s1 = low + chain.last().unwrap().1 as f64 * 0.1;
            if s1 - s0 > 8.0 || t1 - t0 > 35.0 {
                continue;
            }
            let at = |s: f64, t: f64| {
                let x = axis[0] * s + normal[0] * t;
                let y = axis[1] * s + normal[1] * t;
                [
                    o.min[0] + x,
                    o.min[1] + y,
                    o.min[2] + fit[0] * x + fit[1] * y + fit[2],
                ]
            };
            let rectangle = |s0, s1, t0, t1| vec![at(s0, t0), at(s1, t0), at(s1, t1), at(s0, t1)];
            let score = chain
                .iter()
                .map(|&(l, r)| ratio[l..r].iter().sum::<f64>() / (r - l) as f64)
                .sum::<f64>()
                * chain.len() as f64;
            report.candidates.push(CrosswalkCandidate {
                outline: rectangle(s0, s1, t0, t1),
                stripes: chain
                    .iter()
                    .zip(&extents)
                    .map(|(&(l, r), &(t0, t1))| {
                        rectangle(low + l as f64 * 0.1, low + r as f64 * 0.1, t0, t1)
                    })
                    .collect(),
                stripe_count: chain.len(),
                width: s1 - s0,
                length: t1 - t0,
                angle_degrees: angle as f64,
                score,
            });
        }
    }
    report.candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.angle_degrees.total_cmp(&b.angle_degrees))
    });
    let mut distinct: Vec<CrosswalkCandidate> = Vec::new();
    for candidate in std::mem::take(&mut report.candidates) {
        let center = |c: &CrosswalkCandidate| {
            [
                c.outline.iter().map(|p| p[0]).sum::<f64>() / 4.0,
                c.outline.iter().map(|p| p[1]).sum::<f64>() / 4.0,
            ]
        };
        let c = center(&candidate);
        if distinct.iter().any(|v| {
            let d = center(v);
            let a = (v.angle_degrees - candidate.angle_degrees).abs();
            (c[0] - d[0]).hypot(c[1] - d[1]) < 2.0 && a.min(180.0 - a) < 12.0
        }) {
            continue;
        }
        distinct.push(candidate);
        if distinct.len() == 4 {
            break;
        }
    }
    report.candidates = distinct;
    if report.candidates.is_empty() {
        report
            .warnings
            .push("No supported repeated paint bands found; no geometry was fabricated.".into());
    }
    Ok(report)
}

/// Recompute and add the explicitly confirmed candidate atomically.
pub fn add(
    map: &mut Map,
    cloud: &PointCloud,
    o: &CrosswalkOptions,
) -> Result<CrosswalkReport, BuildError> {
    let mut report = propose(map, cloud, o)?;
    if o.lanes.is_empty() {
        return Err(BuildError(
            "confirm crossing lanes before adding a crosswalk".into(),
        ));
    }
    let candidate = report
        .candidates
        .get(o.candidate)
        .ok_or_else(|| BuildError("selected crosswalk candidate does not exist".into()))?;
    report.classification_source = "user_confirmed_candidate";
    let key = format!("{:?}", o);
    if let Some(walk) = map.crosswalks().find(|w| {
        w.attributes
            .get("cloudanalyzer_measurement")
            .or_else(|| {
                w.attributes
                    .get_prefixed("lanelet2", "cloudanalyzer_measurement")
            })
            .is_some_and(|v| v == key)
    }) {
        report.reused = Some(walk.id);
        return Ok(report);
    }
    let mut draft = map.clone();
    let before: BTreeSet<_> = draft.regulatory_elements().map(|r| r.id).collect();
    let line = |a: usize, b: usize| {
        Polyline3::new(
            [candidate.outline[a], candidate.outline[b]]
                .iter()
                .map(|p| Point3::new(p[0], p[1], p[2]))
                .collect(),
        )
    };
    let (id, _) = draft
        .add_crosswalk(NewCrosswalk {
            geometry: CrosswalkGeometry::Edges {
                left_edge: line(0, 1),
                right_edge: line(3, 2),
            },
            crossing_lanes: Some(o.lanes.clone()),
            stop_line_offset: None,
        })
        .map_err(|e| BuildError(e.to_string()))?;
    let attrs = &mut draft.crosswalk_mut(id).expect("new crosswalk").attributes;
    attrs.insert("cloudanalyzer_measurement", key);
    attrs.insert(
        "cloudanalyzer_geometry_source",
        "point_cloud_brightness_stripes",
    );
    // Optional provenance/display metadata, not standard Lanelet2 paint geometry.
    // Retain observed bands so the measured orientation is not replaced by a
    // decorative pattern after confirmation or OSM reimport.
    attrs.insert(
        "cloudanalyzer_paint_bands",
        serde_json::to_string(&candidate.stripes).map_err(|e| BuildError(e.to_string()))?,
    );
    attrs.insert(
        "cloudanalyzer_classification_source",
        "user_confirmed_candidate",
    );
    attrs.insert("cloudanalyzer_review_required", "yes");
    attrs.insert(
        "cloudanalyzer_stripe_count",
        candidate.stripe_count.to_string(),
    );
    let rules: Vec<_> = draft
        .regulatory_elements()
        .filter(|r| !before.contains(&r.id))
        .map(|r| r.id)
        .collect();
    for id in rules {
        let attrs = &mut draft
            .regulatory_element_mut(id)
            .expect("new rule")
            .attributes;
        attrs.insert("cloudanalyzer_lanes_source", "user_selected");
        attrs.insert("cloudanalyzer_review_required", "yes");
    }
    report.added = Some(id);
    *map = draft;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectormap_core::{LaneDirection, NewRoad, RoadLane};

    fn fixture(angle: f64, paint: bool) -> (Map, PointCloud, CrosswalkOptions) {
        let mut map = Map::new();
        let lane = map
            .build_road(NewRoad::new(
                Polyline3::new(vec![
                    Point3::new(50000.0, 70000.0, 19.0),
                    Point3::new(50010.0, 70000.0, 19.0),
                ]),
                vec![RoadLane::new(3.5, LaneDirection::Forward)],
            ))
            .unwrap()
            .0
            .lanes[0][0];
        let mut cloud = PointCloud {
            colors: Some(vec![]),
            ..Default::default()
        };
        let (sin, cos) = angle.to_radians().sin_cos();
        for i in 0..=200 {
            for j in 0..=160 {
                let s = -5.0 + i as f64 * 0.05;
                let t = -4.0 + j as f64 * 0.05;
                let x = s * cos - t * sin;
                let y = s * sin + t * cos;
                cloud
                    .positions
                    .push([50005.0 + x, 70000.0 + y, 19.0 + 0.03 * x - 0.02 * y]);
                let white =
                    paint && (-2.0..2.0).contains(&s) && (s + 2.0) % 1.0 < 0.5 && t.abs() <= 3.0;
                cloud
                    .colors
                    .as_mut()
                    .unwrap()
                    .push([if white { 210 } else { 70 }; 3]);
            }
        }
        (
            map,
            cloud,
            CrosswalkOptions {
                min: [49998.0, 69993.0, 18.5],
                max: [50012.0, 70007.0, 19.5],
                lanes: vec![lane],
                candidate: 0,
                brightness_fraction: default_brightness_fraction(),
            },
        )
    }

    #[test]
    fn rotated_sloped_paint_preview_add_replay_and_rules_are_atomic() {
        let (mut map, cloud, o) = fixture(28.0, true);
        let original = map.clone();
        let report = propose(&map, &cloud, &o).unwrap();
        assert_eq!(map, original);
        let c = &report.candidates[0];
        assert!((c.angle_degrees - 28.0).abs() <= 2.0, "{c:?}");
        assert_eq!(c.stripe_count, 4);
        assert!((c.width - 3.5).abs() < 0.3);
        assert!((c.length - 6.0).abs() < 0.7);
        for p in &c.outline {
            assert!((p[2] - 19.0 - 0.03 * (p[0] - 50005.0) + 0.02 * (p[1] - 70000.0)).abs() < 0.01);
        }
        let result = add(&mut map, &cloud, &o).unwrap();
        let walk = map.crosswalk(result.added.unwrap()).unwrap();
        assert_eq!(
            walk.attributes.get("cloudanalyzer_geometry_source"),
            Some("point_cloud_brightness_stripes")
        );
        assert_eq!(map.stop_lines().count(), 0);
        assert_eq!(map.traffic_signals().count(), 0);
        // Existing lane geometry remains; explicit crossing adds a rule reference.
        for b in original.boundaries() {
            assert_eq!(map.boundary(b.id), Some(b));
        }
        let added = map.clone();
        assert_eq!(add(&mut map, &cloud, &o).unwrap().reused, result.added);
        assert_eq!(map, added);
    }

    #[test]
    fn blank_single_line_and_above_ground_paint_do_not_make_crosswalks() {
        let (map, mut cloud, o) = fixture(0.0, false);
        assert!(propose(&map, &cloud, &o).unwrap().candidates.is_empty());
        for (p, color) in cloud.positions.iter().zip(cloud.colors.as_mut().unwrap()) {
            if (p[0] - 50005.0).abs() < 0.25 {
                *color = [210; 3];
            }
        }
        assert!(propose(&map, &cloud, &o).unwrap().candidates.is_empty());
        // A bright repeated overhead pattern must not become ground paint.
        let (_, painted, _) = fixture(0.0, true);
        cloud.colors.as_mut().unwrap().fill([70; 3]);
        for (p, c) in painted.positions.iter().zip(painted.colors.unwrap()) {
            if c[0] == 210 {
                cloud.positions.push([p[0], p[1], p[2] + 0.3]);
                cloud.colors.as_mut().unwrap().push(c);
            }
        }
        assert!(propose(&map, &cloud, &o).unwrap().candidates.is_empty());
    }

    #[test]
    fn invalid_inputs_missing_attributes_and_unconfirmed_add_preserve_map() {
        let (mut map, mut cloud, mut o) = fixture(0.0, true);
        let before = map.clone();
        o.brightness_fraction = f64::NAN;
        assert!(add(&mut map, &cloud, &o).is_err());
        o.brightness_fraction = default_brightness_fraction();
        o.candidate = usize::MAX;
        assert!(add(&mut map, &cloud, &o).is_err());
        o.candidate = 0;
        o.lanes.clear();
        assert!(!propose(&map, &cloud, &o).unwrap().candidates.is_empty());
        assert!(add(&mut map, &cloud, &o).is_err());
        o.lanes = vec![LaneId(u64::MAX)];
        assert!(add(&mut map, &cloud, &o).is_err());
        o.lanes.clear();
        o.max[0] = f64::INFINITY;
        assert!(propose(&map, &cloud, &o).is_err());
        o.max[0] = 50012.0;
        cloud.colors = None;
        assert!(propose(&map, &cloud, &o).is_err());
        assert_eq!(map, before);
    }

    #[test]
    fn intensity_and_rgb_agree_and_bad_attribute_counts_and_caps_are_errors() {
        let (map, mut cloud, o) = fixture(28.0, true);
        let rgb = propose(&map, &cloud, &o).unwrap();
        cloud.attributes.push(crate::Attribute {
            name: crate::INTENSITY.into(),
            values: crate::AttributeValues::F32(
                cloud
                    .colors
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|c| c[0] as f32)
                    .collect(),
            ),
        });
        cloud.colors = None;
        let intensities = propose(&map, &cloud, &o).unwrap();
        assert_eq!(intensities.brightness_source, "intensity");
        assert_eq!(
            serde_json::to_string(&intensities.candidates).unwrap(),
            serde_json::to_string(&rgb.candidates).unwrap()
        );
        cloud.attributes[0].values = crate::AttributeValues::U8(vec![70]);
        assert!(propose(&map, &cloud, &o).is_err());
        cloud.attributes.clear();
        cloud.attributes.push(crate::Attribute {
            name: crate::INTENSITY.into(),
            values: crate::AttributeValues::U8(vec![70; cloud.positions.len()]),
        });
        let blank = propose(&map, &cloud, &o).unwrap();
        assert!(blank.candidates.is_empty());
        assert_eq!(blank.brightness_source, "intensity");
        cloud.attributes.clear();
        cloud.colors = Some(vec![]);
        assert!(propose(&map, &cloud, &o).is_err());
        cloud.colors = None;
        cloud.positions = vec![[50005.0, 70000.0, 19.0]; 200_001];
        assert!(propose(&map, &cloud, &o).unwrap_err().0.contains("200000"));
    }
}
