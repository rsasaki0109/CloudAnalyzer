//! Explicit geometry edits for existing crossings and signal heads. Observed
//! paint, lamps and lane assignments are retained; editing does not remeasure.
use serde::{Deserialize, Serialize};
use vectormap_core::{Attributes, CrosswalkId, Map, Point3, Polyline3, Rule, SignalId};

use super::BuildError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    Crosswalk,
    Signal,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureEdit {
    pub kind: FeatureKind,
    pub id: u64,
    /// Full existing vertex sequence in original coordinates, preserving count/order.
    pub points: Vec<[f64; 3]>,
    /// A manual housing height; omitted/null keeps the stored height.
    pub height: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct FeatureEditReport {
    pub kind: FeatureKind,
    pub id: u64,
    pub changed: bool,
    pub geometry_source: String,
    pub warnings: Vec<String>,
}

fn source(attrs: &Attributes) -> &str {
    attrs
        .get("cloudanalyzer_geometry_source")
        .or_else(|| attrs.get_prefixed("lanelet2", "cloudanalyzer_geometry_source"))
        .unwrap_or("imported_or_manual")
}
fn tag(attrs: &mut Attributes, key: &str, value: &str) {
    // OSM imports namespace extension tags. Replace, never emit conflicting keys.
    attrs.remove(&format!("lanelet2:{key}"));
    attrs.insert(key, value);
}
fn mark(attrs: &mut Attributes) {
    let original = source(attrs).to_string();
    let next = if original.ends_with("_user_edited") {
        original
    } else {
        format!("{original}_user_edited")
    };
    tag(attrs, "cloudanalyzer_geometry_source", &next);
    tag(attrs, "cloudanalyzer_user_edited", "yes");
    tag(attrs, "cloudanalyzer_review_required", "yes");
}
fn line(points: &[[f64; 3]]) -> Polyline3 {
    Polyline3::new(
        points
            .iter()
            .map(|p| Point3::new(p[0], p[1], p[2]))
            .collect(),
    )
}
fn coords(line: &Polyline3) -> Vec<[f64; 3]> {
    line.points.iter().map(|p| [p.x, p.y, p.z]).collect()
}

fn cross(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}
fn on(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> bool {
    (0..2).all(|i| p[i] >= a[i].min(b[i]) - 1e-8 && p[i] <= a[i].max(b[i]) + 1e-8)
}
fn intersects(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let (u, v, w, x) = (
        cross(a, b, c),
        cross(a, b, d),
        cross(c, d, a),
        cross(c, d, b),
    );
    (u * v < 0.0 && w * x < 0.0)
        || (u.abs() < 1e-8 && on(a, b, c))
        || (v.abs() < 1e-8 && on(a, b, d))
        || (w.abs() < 1e-8 && on(c, d, a))
        || (x.abs() < 1e-8 && on(c, d, b))
}
fn area(p: &[[f64; 3]]) -> f64 {
    let local: Vec<_> = p.iter().map(|q| [q[0] - p[0][0], q[1] - p[0][1]]).collect();
    (0..p.len())
        .map(|i| cross([0.0, 0.0], local[i], local[(i + 1) % p.len()]))
        .sum::<f64>()
        * 0.5
}
fn check_ring(p: &[[f64; 3]], old: &[[f64; 3]]) -> Result<(), BuildError> {
    let local: Vec<_> = p.iter().map(|q| [q[0] - p[0][0], q[1] - p[0][1]]).collect();
    if area(p).abs() < 1e-4 || area(p).signum() != area(old).signum() {
        return Err(BuildError(
            "crossing must retain a nonzero area and its vertex order".into(),
        ));
    }
    for i in 0..p.len() {
        let next = (i + 1) % p.len();
        if (local[i][0] - local[next][0]).hypot(local[i][1] - local[next][1]) < 1e-4 {
            return Err(BuildError("crossing has a collapsed edge".into()));
        }
        let after = (next + 1) % p.len();
        if cross(local[i], local[next], local[after]).abs() < 1e-8
            && (local[next][0] - local[i][0]) * (local[after][0] - local[next][0])
                + (local[next][1] - local[i][1]) * (local[after][1] - local[next][1])
                < 0.0
        {
            return Err(BuildError("crossing has overlapping adjacent edges".into()));
        }
        for j in i + 1..p.len() {
            let other = (j + 1) % p.len();
            if j == next || other == i {
                continue;
            }
            if intersects(local[i], local[next], local[j], local[other]) {
                return Err(BuildError(
                    "crossing outline intersects or touches itself".into(),
                ));
            }
        }
    }
    Ok(())
}
fn movement(old: &[[f64; 3]], new: &[[f64; 3]], limit: f64) -> Result<(), BuildError> {
    if old
        .iter()
        .zip(new)
        .any(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]).hypot(a[2] - b[2]) > limit)
    {
        return Err(BuildError(format!(
            "vertex movement exceeds {limit} m; check original metre coordinates"
        )));
    }
    Ok(())
}

/// Validate before publication. No IDs, lanes, stops, observed bands or lamps
/// are synthesized/reassigned; associated rules are marked for review.
pub fn edit(map: &mut Map, o: &FeatureEdit) -> Result<FeatureEditReport, BuildError> {
    if !(2..=256).contains(&o.points.len()) || o.points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(BuildError(
            "feature geometry needs 2–256 finite XYZ vertices".into(),
        ));
    }
    if (0..3).any(|i| {
        let min = o.points.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min);
        let max = o
            .points
            .iter()
            .map(|p| p[i])
            .fold(f64::NEG_INFINITY, f64::max);
        max - min
            > if o.kind == FeatureKind::Signal {
                20.0
            } else {
                100.0
            }
    }) {
        return Err(BuildError(
            "feature geometry extent is excessive; check metre coordinates".into(),
        ));
    }
    let mut report = FeatureEditReport {
        kind: o.kind,
        id: o.id,
        changed: false,
        geometry_source: String::new(),
        warnings: vec![],
    };
    let mut draft = map.clone();
    match o.kind {
        FeatureKind::Crosswalk => {
            if o.height.is_some() {
                return Err(BuildError(
                    "crossing heights are vertex coordinates, not a housing height".into(),
                ));
            }
            let walk = map
                .crosswalk(CrosswalkId(o.id))
                .ok_or_else(|| BuildError("crossing does not exist".into()))?;
            if walk.polygon.is_some()
                || walk.left_edge.points.len() < 2
                || walk.right_edge.points.len() < 2
            {
                return Err(BuildError("this crossing requires separate polygon/edge editing; its geometry is retained".into()));
            }
            // Match Crosswalk::outline / Polygon3::from_sides exactly:
            // the right edge followed by the reversed left edge.
            let right = walk.right_edge.points.len();
            let mut old = coords(&walk.right_edge);
            old.extend(coords(&walk.left_edge).into_iter().rev());
            if old.len() != o.points.len() {
                return Err(BuildError(
                    "retain the crossing's existing vertex count/order".into(),
                ));
            }
            movement(&old, &o.points, 100.0)?;
            check_ring(&o.points, &old)?;
            report.geometry_source = source(&walk.attributes).to_string();
            if old == o.points {
                return Ok(report);
            }
            let edited = draft.crosswalk_mut(walk.id).expect("existing crossing");
            edited.right_edge = line(&o.points[..right]);
            edited.left_edge = line(&o.points[right..].iter().rev().copied().collect::<Vec<_>>());
            mark(&mut edited.attributes);
            report.geometry_source = source(&edited.attributes).into();
            report.warnings.push("Crossing geometry was manually edited. Observed paint stays at its measured coordinates; missing paint is not filled. Review the crossing lanes and legal priority.".into());
        }
        FeatureKind::Signal => {
            let signal = map
                .traffic_signal(SignalId(o.id))
                .ok_or_else(|| BuildError("signal does not exist".into()))?;
            if signal.geometry.points.len() != o.points.len() {
                return Err(BuildError(
                    "retain the signal's existing vertex count/order".into(),
                ));
            }
            movement(&coords(&signal.geometry), &o.points, 20.0)?;
            if o.height
                .is_some_and(|h| !h.is_finite() || h <= 0.0 || h > 20.0)
            {
                return Err(BuildError(
                    "manual housing height must be finite, positive and at most 20 m".into(),
                ));
            }
            if o.points
                .windows(2)
                .any(|p| (p[0][0] - p[1][0]).hypot(p[0][1] - p[1][1]) < 1e-4)
            {
                return Err(BuildError(
                    "signal bottom edge has collapsed XY segments".into(),
                ));
            }
            let height = o.height.or(signal.height);
            report.geometry_source = source(&signal.attributes).into();
            if coords(&signal.geometry) == o.points && height == signal.height {
                return Ok(report);
            }
            let edited = draft
                .traffic_signal_mut(signal.id)
                .expect("existing signal");
            edited.geometry = line(&o.points);
            edited.height = height;
            mark(&mut edited.attributes);
            report.geometry_source = source(&edited.attributes).into();
            report.warnings.push("Signal geometry was manually edited; review the housing, face direction and controlled lanes.".into());
            if !signal.bulbs.is_empty() {
                report.warnings.push(
                    "Stored lamps remain at their original coordinates; review them separately."
                        .into(),
                );
            }
        }
    }
    let rules: Vec<_> = draft
        .regulatory_elements()
        .filter(|r| match (&r.rule, o.kind) {
            (Rule::Crosswalk { crosswalk, .. }, FeatureKind::Crosswalk) => crosswalk.0 == o.id,
            (Rule::TrafficLight { signals, .. }, FeatureKind::Signal) => {
                signals.iter().any(|s| s.0 == o.id)
            }
            _ => false,
        })
        .map(|r| r.id)
        .collect();
    for id in rules {
        tag(
            &mut draft
                .regulatory_element_mut(id)
                .expect("existing rule")
                .attributes,
            "cloudanalyzer_review_required",
            "yes",
        );
    }
    report.changed = true;
    *map = draft;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectormap_core::{
        CrosswalkGeometry, LaneDirection, NewCrosswalk, NewRoad, NewTrafficSignal, RoadLane,
    };

    fn fixture() -> (Map, FeatureEdit, FeatureEdit) {
        let mut map = Map::new();
        let lane = map
            .build_road(NewRoad::new(
                line(&[[50000.0, 70000.0, 19.0], [50010.0, 70000.0, 19.0]]),
                vec![RoadLane::new(3.5, LaneDirection::Forward)],
            ))
            .unwrap()
            .0
            .lanes[0][0];
        let walk = map
            .add_crosswalk(NewCrosswalk {
                geometry: CrosswalkGeometry::Edges {
                    left_edge: line(&[[50002.0, 69997.0, 19.0], [50002.0, 70003.0, 19.1]]),
                    right_edge: line(&[[50006.0, 69997.0, 19.0], [50006.0, 70003.0, 19.1]]),
                },
                crossing_lanes: Some(vec![lane]),
                stop_line_offset: Some(1.0),
            })
            .unwrap()
            .0;
        let signal = map
            .add_traffic_signal(NewTrafficSignal::for_lanes(vec![lane]))
            .unwrap()
            .0;
        let crossing = map.crosswalk_mut(walk).unwrap();
        crossing.attributes.insert(
            "cloudanalyzer_geometry_source",
            "point_cloud_brightness_stripes",
        );
        crossing.attributes.insert(
            "cloudanalyzer_paint_bands",
            "[[[50002,69997,19],[50002,70003,19],[50002.5,70003,19],[50002.5,69997,19]]]",
        );
        crossing
            .attributes
            .insert("cloudanalyzer_crosswalk_measurement", "observed-box-key");
        let walk = map.crosswalk(walk).unwrap();
        let crossing = FeatureEdit {
            kind: FeatureKind::Crosswalk,
            id: walk.id.0,
            points: walk
                .outline()
                .points
                .iter()
                .map(|p| [p.x, p.y, p.z])
                .collect(),
            height: None,
        };
        let head = map.traffic_signal(signal).unwrap();
        let signal = FeatureEdit {
            kind: FeatureKind::Signal,
            id: signal.0,
            points: coords(&head.geometry),
            height: head.height,
        };
        (map, crossing, signal)
    }
    fn unchanged_context(old: &Map, new: &Map) {
        assert_eq!(old.metadata(), new.metadata());
        for lane in old.lanes() {
            assert_eq!(new.lane(lane.id), Some(lane));
        }
        for b in old.boundaries() {
            assert_eq!(new.boundary(b.id), Some(b));
        }
        for s in old.stop_lines() {
            assert_eq!(new.stop_line(s.id), Some(s));
        }
        for r in old.regulatory_elements() {
            let n = new.regulatory_element(r.id).unwrap();
            assert_eq!(r.rule, n.rule);
            assert_eq!(r.lanes, n.lanes);
        }
    }
    #[test]
    fn crossing_edit_keeps_observations_ids_and_rules_and_noop_is_exact() {
        let (mut map, mut edit, _) = fixture();
        let old = map.clone();
        assert!(!super::edit(&mut map, &edit).unwrap().changed);
        assert_eq!(map, old);
        edit.points[0][0] += 0.2;
        edit.points[0][2] += 0.05;
        let report = super::edit(&mut map, &edit).unwrap();
        assert!(report.changed);
        unchanged_context(&old, &map);
        let new = map.crosswalk(CrosswalkId(edit.id)).unwrap();
        let original = old.crosswalk(new.id).unwrap();
        assert_eq!(
            new.attributes.get("cloudanalyzer_paint_bands"),
            original.attributes.get("cloudanalyzer_paint_bands")
        );
        assert_eq!(
            new.attributes.get("cloudanalyzer_crosswalk_measurement"),
            Some("observed-box-key")
        );
        assert_eq!(
            report.geometry_source,
            "point_cloud_brightness_stripes_user_edited"
        );
        assert_eq!(
            new.right_edge.points[0],
            Point3::new(edit.points[0][0], edit.points[0][1], edit.points[0][2])
        );
        let changed = map.clone();
        assert!(!super::edit(&mut map, &edit).unwrap().changed);
        assert_eq!(map, changed);
    }
    #[test]
    fn signal_edit_keeps_lamps_and_assignments_and_normalizes_imported_tags() {
        let (mut map, _, mut edit) = fixture();
        let attrs = &mut map
            .traffic_signal_mut(SignalId(edit.id))
            .unwrap()
            .attributes;
        attrs.insert(
            "lanelet2:cloudanalyzer_geometry_source",
            "point_cloud_box_fit",
        );
        attrs.insert("lanelet2:cloudanalyzer_review_required", "no");
        let old = map.clone();
        edit.points[0][2] += 0.1;
        edit.height = Some(0.75);
        let report = super::edit(&mut map, &edit).unwrap();
        unchanged_context(&old, &map);
        let new = map.traffic_signal(SignalId(edit.id)).unwrap();
        assert_eq!(new.bulbs, old.traffic_signal(new.id).unwrap().bulbs);
        assert!(!new.bulbs.is_empty());
        assert_eq!(new.height, Some(0.75));
        assert_eq!(report.warnings.len(), 2);
        assert_eq!(report.geometry_source, "point_cloud_box_fit_user_edited");
        assert!(
            new.attributes
                .get("lanelet2:cloudanalyzer_geometry_source")
                .is_none()
        );
        assert!(
            new.attributes
                .get("lanelet2:cloudanalyzer_review_required")
                .is_none()
        );
    }
    #[test]
    fn invalid_edits_are_atomic_for_crossings_and_signals() {
        let (mut map, crossing, signal) = fixture();
        let before = map.clone();
        let variants = [
            vec![[f64::NAN, 0.0, 0.0]; 4],
            vec![crossing.points[0]; 4],
            vec![
                crossing.points[0],
                crossing.points[2],
                crossing.points[1],
                crossing.points[3],
            ],
            crossing.points.iter().rev().copied().collect(),
            crossing
                .points
                .iter()
                .map(|p| [p[0] + 200.0, p[1], p[2]])
                .collect(),
            crossing.points[..3].to_vec(),
        ];
        for points in variants {
            assert!(
                super::edit(
                    &mut map,
                    &FeatureEdit {
                        kind: FeatureKind::Crosswalk,
                        id: crossing.id,
                        points,
                        height: None
                    }
                )
                .is_err()
            );
            assert_eq!(map, before);
        }
        for height in [0.0, -1.0, 21.0, f64::INFINITY] {
            assert!(
                super::edit(
                    &mut map,
                    &FeatureEdit {
                        kind: FeatureKind::Signal,
                        id: signal.id,
                        points: signal.points.clone(),
                        height: Some(height)
                    }
                )
                .is_err()
            );
            assert_eq!(map, before);
        }
        assert!(
            super::edit(
                &mut map,
                &FeatureEdit {
                    kind: FeatureKind::Signal,
                    id: signal.id,
                    points: vec![signal.points[0]; 2],
                    height: None
                }
            )
            .is_err()
        );
        assert!(
            super::edit(
                &mut map,
                &FeatureEdit {
                    kind: FeatureKind::Signal,
                    id: u64::MAX,
                    points: signal.points,
                    height: None
                }
            )
            .is_err()
        );
        assert_eq!(map, before);
    }
}
