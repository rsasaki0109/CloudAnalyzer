//! Measure a user-identified signal head from points inside an explicit 3D box.
//! Shape fitting cannot classify a signal or infer its controlled lanes/lamps.

use serde::{Deserialize, Serialize};
use vectormap_core::{
    LaneId, Map, NewTrafficSignal, Point3, Polyline3, SignalId, SignalKind, StopLineChoice,
};

use super::BuildError;
use crate::PointCloud;

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * q;
    let i = position.floor() as usize;
    let j = position.ceil() as usize;
    sorted[i] + (sorted[j] - sorted[i]) * (position - i as f64)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalOptions {
    /// Original-coordinate box enclosing only the identified head, excluding its pole.
    pub min: [f64; 3],
    pub max: [f64; 3],
    /// User-confirmed controlled lanes; never inferred from proximity.
    pub lanes: Vec<LaneId>,
    #[serde(default)]
    pub kind: SignalKind,
}

#[derive(Debug, Clone, Serialize)]
pub struct SignalReport {
    pub geometry: Vec<[f64; 3]>,
    pub height: f64,
    pub width: f64,
    pub thickness: f64,
    pub points: usize,
    pub plane_rms: f64,
    pub added: Option<SignalId>,
    pub reused: Option<SignalId>,
    pub classification_source: &'static str,
    pub lanes_source: &'static str,
    pub warnings: Vec<String>,
}

/// Read-only measurement; coordinates retain f64 precision around a local origin.
pub fn measure(
    map: &Map,
    cloud: &PointCloud,
    o: &SignalOptions,
) -> Result<SignalReport, BuildError> {
    if (0..3).any(|i| {
        !o.min[i].is_finite()
            || !o.max[i].is_finite()
            || o.min[i] >= o.max[i]
            || o.max[i] - o.min[i] > 10.0
    }) {
        return Err(BuildError(
            "signal box must have finite increasing bounds, at most 10 m per axis".into(),
        ));
    }
    if o.lanes.is_empty() || o.lanes.iter().any(|id| map.lane(*id).is_none()) {
        return Err(BuildError(
            "select existing controlled lanes explicitly".into(),
        ));
    }
    let mut lanes = o.lanes.clone();
    lanes.sort();
    lanes.dedup();
    if lanes.len() != o.lanes.len() {
        return Err(BuildError("controlled lanes contain duplicates".into()));
    }
    let centerline = map
        .centerline(o.lanes[0])
        .ok_or_else(|| BuildError("first controlled lane has no usable centerline".into()))?;
    let pair = centerline
        .points
        .last_chunk::<2>()
        .ok_or_else(|| BuildError("first controlled lane has no heading".into()))?;
    let heading = [pair[1].x - pair[0].x, pair[1].y - pair[0].y];
    if heading[0].hypot(heading[1]) < 1e-6 {
        return Err(BuildError("first controlled lane has no heading".into()));
    }
    measure_geometry(cloud, o.min, o.max, heading)
}

/// Geometry-only fitting for automatic proposals. The caller must not infer
/// semantic identity or lane control from this result.
pub(super) fn measure_geometry(
    cloud: &PointCloud,
    min: [f64; 3],
    max: [f64; 3],
    heading: [f64; 2],
) -> Result<SignalReport, BuildError> {
    let points: Vec<_> = cloud
        .positions
        .iter()
        .filter(|p| (0..3).all(|i| p[i].is_finite() && p[i] >= min[i] && p[i] <= max[i]))
        .map(|p| std::array::from_fn::<_, 3, _>(|i| p[i] - min[i]))
        .take(200_001)
        .collect();
    if points.len() > 200_000 {
        return Err(BuildError(
            "signal box exceeds 200000 points; isolate a smaller head".into(),
        ));
    }
    if points.len() < 12 {
        return Err(BuildError("signal box needs at least 12 finite points; tighten the box or load full-density points".into()));
    }
    let n = points.len() as f64;
    let mean: [f64; 3] = std::array::from_fn(|i| points.iter().map(|p| p[i]).sum::<f64>() / n);
    let xx = points.iter().map(|p| (p[0] - mean[0]).powi(2)).sum::<f64>();
    let yy = points.iter().map(|p| (p[1] - mean[1]).powi(2)).sum::<f64>();
    let xy = points
        .iter()
        .map(|p| (p[0] - mean[0]) * (p[1] - mean[1]))
        .sum::<f64>();
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let mut axis = [angle.cos(), angle.sin()];
    // Left to right as seen by the explicitly selected traffic.
    if axis[0] * heading[1] - axis[1] * heading[0] < 0.0 {
        axis = [-axis[0], -axis[1]];
    }
    let normal = [-axis[1], axis[0]];
    let mut along: Vec<_> = points
        .iter()
        .map(|p| (p[0] - mean[0]) * axis[0] + (p[1] - mean[1]) * axis[1])
        .collect();
    let mut across: Vec<_> = points
        .iter()
        .map(|p| (p[0] - mean[0]) * normal[0] + (p[1] - mean[1]) * normal[1])
        .collect();
    let mut heights: Vec<_> = points.iter().map(|p| p[2]).collect();
    along.sort_by(f64::total_cmp);
    across.sort_by(f64::total_cmp);
    heights.sort_by(f64::total_cmp);
    let (a, b) = (quantile(&along, 0.02), quantile(&along, 0.98));
    let (z0, z1) = (quantile(&heights, 0.02), quantile(&heights, 0.98));
    let plane = quantile(&across, 0.5);
    let width = b - a;
    let height = z1 - z0;
    let thickness = quantile(&across, 0.95) - quantile(&across, 0.05);
    let rms = (across.iter().map(|p| (p - plane).powi(2)).sum::<f64>() / n).sqrt();
    if !(0.15..=3.5).contains(&width)
        || !(0.15..=2.5).contains(&height)
        || thickness > 0.5
        || rms > 0.2
    {
        return Err(BuildError(format!(
            "box does not isolate a compact vertical head: width {width:.3}, height {height:.3}, thickness {thickness:.3}, plane RMS {rms:.3} m; exclude poles/background"
        )));
    }
    let at = |s: f64| {
        [
            min[0] + mean[0] + axis[0] * s + normal[0] * plane,
            min[1] + mean[1] + axis[1] * s + normal[1] * plane,
            min[2] + z0,
        ]
    };
    Ok(SignalReport {
        geometry: vec![at(a), at(b)], height, width, thickness, points: points.len(), plane_rms: rms,
        added: None, reused: None, classification_source: "user_identified_box", lanes_source: "user_selected",
        warnings: vec!["Measured geometry is a draft. Shape alone cannot distinguish a signal from a sign; confirm the object, face direction and controlled lanes. Lamp colors/arrows and stop lines are not inferred.".into()],
    })
}

/// Recompute support and add atomically, without synthesizing a stop line or lamps.
pub fn add(
    map: &mut Map,
    cloud: &PointCloud,
    o: &SignalOptions,
) -> Result<SignalReport, BuildError> {
    let mut report = measure(map, cloud, o)?;
    let key = format!("{:?}", o);
    if let Some(signal) = map.traffic_signals().find(|s| {
        s.attributes
            .get("cloudanalyzer_measurement")
            .or_else(|| {
                s.attributes
                    .get_prefixed("lanelet2", "cloudanalyzer_measurement")
            })
            .is_some_and(|v| v == key)
    }) {
        report.reused = Some(signal.id);
        return Ok(report);
    }
    let mut draft = map.clone();
    let before_rules: std::collections::BTreeSet<_> =
        draft.regulatory_elements().map(|r| r.id).collect();
    let geometry = Polyline3::new(
        report
            .geometry
            .iter()
            .map(|p| Point3::new(p[0], p[1], p[2]))
            .collect(),
    );
    let (id, _) = draft
        .add_traffic_signal(NewTrafficSignal {
            lanes: o.lanes.clone(),
            stop_line: StopLineChoice::None,
            geometry: Some(geometry),
            height: Some(report.height),
            kind: o.kind,
            bulbs: Some(vec![]),
            group: None,
        })
        .map_err(|e| BuildError(e.to_string()))?;
    let signal = draft.traffic_signal_mut(id).expect("new signal");
    signal.attributes.insert("cloudanalyzer_measurement", key);
    signal
        .attributes
        .insert("cloudanalyzer_geometry_source", "point_cloud_box_fit");
    signal
        .attributes
        .insert("cloudanalyzer_review_required", "yes");
    let new_rules: Vec<_> = draft
        .regulatory_elements()
        .filter(|r| !before_rules.contains(&r.id))
        .map(|r| r.id)
        .collect();
    for id in new_rules {
        let rule = draft.regulatory_element_mut(id).expect("new rule");
        rule.attributes
            .insert("cloudanalyzer_lanes_source", "user_selected");
        rule.attributes
            .insert("cloudanalyzer_review_required", "yes");
    }
    report.added = Some(id);
    *map = draft;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectormap_core::{LaneDirection, NewRoad, NewStopLine, RoadLane, StopRule};

    fn fixture() -> (Map, PointCloud, SignalOptions) {
        let mut map = Map::new();
        let origin = 70_000.0;
        let lane = map
            .build_road(NewRoad::new(
                Polyline3::new(vec![
                    Point3::new(origin, 0.0, 19.0),
                    Point3::new(origin, 10.0, 19.0),
                ]),
                vec![RoadLane::new(3.5, LaneDirection::Forward)],
            ))
            .unwrap()
            .0
            .lanes[0][0];
        let mut cloud = PointCloud::default();
        for x in 0..=24 {
            for z in 0..=10 {
                cloud.positions.push([
                    origin - 0.6 + x as f64 * 0.05,
                    11.0,
                    24.0 + z as f64 * 0.05,
                ]);
            }
        }
        cloud.positions.push([f64::NAN, 11.0, 24.5]);
        let o = SignalOptions {
            min: [origin - 0.7, 10.9, 23.9],
            max: [origin + 0.7, 11.1, 24.6],
            lanes: vec![lane],
            kind: SignalKind::Vehicle,
        };
        (map, cloud, o)
    }

    #[test]
    fn measured_geometry_keeps_elevation_and_rules_and_replay_is_noop() {
        let (mut map, cloud, o) = fixture();
        map.add_stop_line(NewStopLine {
            lanes: o.lanes.clone(),
            placement: Default::default(),
            rule: StopRule::Marking,
        })
        .unwrap();
        let original = map.clone();
        let fit = measure(&map, &cloud, &o).unwrap();
        assert_eq!(map, original);
        assert_eq!(fit.points, 275);
        assert!((fit.width - 1.2).abs() < 0.02);
        assert!((fit.height - 0.5).abs() < 0.02);
        assert!(fit.geometry[0][0] < fit.geometry[1][0]);
        assert_eq!(fit.geometry[0][2], 24.0);
        let report = add(&mut map, &cloud, &o).unwrap();
        let signal = map.traffic_signal(report.added.unwrap()).unwrap();
        assert!(signal.bulbs.is_empty());
        assert_eq!(map.stop_lines().count(), 1);
        assert_eq!(
            signal.attributes.get("cloudanalyzer_review_required"),
            Some("yes")
        );
        for lane in original.lanes() {
            assert_eq!(map.lane(lane.id), Some(lane));
        }
        for boundary in original.boundaries() {
            assert_eq!(map.boundary(boundary.id), Some(boundary));
        }
        for rule in original.regulatory_elements() {
            assert_eq!(map.regulatory_element(rule.id), Some(rule));
        }
        let added = map.clone();
        assert_eq!(add(&mut map, &cloud, &o).unwrap().reused, report.added);
        assert_eq!(map, added);
    }

    #[test]
    fn sparse_pole_background_invalid_box_and_lanes_are_atomic_errors() {
        let (mut map, mut cloud, mut o) = fixture();
        let before = map.clone();
        o.max[0] = o.min[0];
        assert!(add(&mut map, &cloud, &o).is_err());
        o.max[0] = o.min[0] + 1.4;
        o.lanes.push(o.lanes[0]);
        assert!(add(&mut map, &cloud, &o).is_err());
        o.lanes.pop();
        cloud.positions.truncate(11);
        assert!(add(&mut map, &cloud, &o).is_err());
        cloud.positions = (0..40)
            .map(|i| [70_000.0, 11.0, 24.0 + i as f64 * 0.01])
            .collect();
        assert!(add(&mut map, &cloud, &o).is_err());
        o.min[1] = 10.0;
        o.max[1] = 12.0;
        cloud.positions.clear();
        for x in 0..=10 {
            for y in 0..=20 {
                for z in 0..=5 {
                    cloud.positions.push([
                        69_999.5 + x as f64 * 0.1,
                        10.0 + y as f64 * 0.1,
                        24.0 + z as f64 * 0.1,
                    ]);
                }
            }
        }
        assert!(add(&mut map, &cloud, &o).is_err());
        assert_eq!(map, before);
    }
}
