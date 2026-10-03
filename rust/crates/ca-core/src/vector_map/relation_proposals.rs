//! Read-only geometric equipment suggestions, explicitly adopted after review.
use super::{
    BuildError,
    relations::{self, LinkEdit, LinkReport},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, hash_map::DefaultHasher};
use std::hash::Hasher;
use std::io::Write;
use vectormap_core::{LaneId, Map, Point3, Polyline3, RegulatoryElementId, Rule, SignalKind};

const MAX_TARGETS: usize = 128;
const MAX_VERTICES: usize = 200_000;
const MAX_DISTANCE: f64 = 16.;
const PEDESTRIAN_DISTANCE: f64 = 8.;
const MAX_AXIS_DEGREES: f64 = 35.;

#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub key: String,
    pub target_kind: String,
    pub target_id: u64,
    pub lanes: Vec<u64>,
    pub controlled_crosswalks: Vec<u64>,
    pub stop_lines: Vec<u64>,
    pub distance_m: Option<f64>,
    pub axis_degrees: Option<f64>,
    pub road_context: String,
    pub eligible: bool,
    pub already_linked: bool,
    pub reasons: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct ProposalReport {
    pub rule_id: u64,
    pub kind: String,
    pub map_snapshot: String,
    pub candidates: Vec<Candidate>,
    pub eligible_count: usize,
    pub ambiguous: bool,
    pub limited: bool,
    pub warnings: Vec<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Adoption {
    pub rule_id: u64,
    pub map_snapshot: String,
    pub candidate_key: String,
}

struct Checksum(DefaultHasher);
impl Write for Checksum {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.write(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn snapshot(map: &Map) -> Result<String, BuildError> {
    // Opaque edit-staleness token, not an authenticity or file-integrity digest.
    let mut writer = Checksum(DefaultHasher::new());
    serde_json::to_writer(&mut writer, map).map_err(|e| BuildError(e.to_string()))?;
    Ok(format!("{:016x}", writer.0.finish()))
}
fn finite(line: &Polyline3) -> bool {
    (2..=256).contains(&line.points.len())
        && line
            .points
            .iter()
            .all(|p| [p.x, p.y, p.z].iter().all(|v| v.is_finite()))
}
fn axis(line: &Polyline3) -> Option<[f64; 2]> {
    if !finite(line) {
        return None;
    }
    let (a, b) = (line.points.first()?, line.points.last()?);
    let d = [b.x - a.x, b.y - a.y];
    let length = d[0].hypot(d[1]);
    (length > 1e-6).then(|| [d[0] / length, d[1] / length])
}
fn midpoint(line: &Polyline3) -> Option<Point3> {
    axis(line)?;
    let (a, b) = (line.points.first()?, line.points.last()?);
    Some(Point3::new(
        (a.x + b.x) / 2.,
        (a.y + b.y) / 2.,
        (a.z + b.z) / 2.,
    ))
}
fn angle(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] * b[0] + a[1] * b[1])
        .abs()
        .clamp(0., 1.)
        .acos()
        .to_degrees()
}
fn near_context(map: &Map, p: Point3) -> Vec<LaneId> {
    let mut lanes: Vec<_> = map
        .lanes()
        .filter_map(|l| {
            let c = map.centerline(l.id)?;
            let nearest = c.nearest_point(p.xy())?;
            (nearest.distance <= 8. && (p.z - nearest.point.z).abs() <= 12.)
                .then_some((nearest.distance, l.id))
        })
        .collect();
    lanes.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    lanes.into_iter().take(3).map(|(_, id)| id).collect()
}
fn linked(map: &Map, a: LaneId, b: LaneId) -> bool {
    if map.lane(a).is_none() || map.lane(b).is_none() {
        return false;
    }
    let mut seen = BTreeSet::from([a]);
    let mut frontier = vec![a];
    for _ in 0..2 {
        let mut next = Vec::new();
        for id in frontier {
            for &id in map.successors(id).iter().chain(map.predecessors(id)) {
                if id == b {
                    return true;
                }
                if seen.insert(id) {
                    next.push(id);
                }
            }
        }
        frontier = next;
        if seen.len() > 128 {
            return false;
        }
    }
    false
}
fn context(map: &Map, from: &[LaneId], to: &[LaneId]) -> &'static str {
    if from
        .iter()
        .any(|id| map.lane(*id).is_some() && to.contains(id))
    {
        "same_lane"
    } else if from.iter().any(|&a| to.iter().any(|&b| linked(map, a, b))) {
        "connected_lanes"
    } else {
        "unavailable"
    }
}
fn rejection(candidate: &mut Candidate, reason: &str) {
    candidate.eligible = false;
    candidate.reasons.push(reason.into());
}

/// Propose targets for one existing signal rule; rejected nearby alternatives
/// remain visible. Geometry, topology, attributes and Undo are never changed.
pub fn propose(map: &Map, rule_id: u64) -> Result<ProposalReport, BuildError> {
    let rule = map
        .regulatory_element(RegulatoryElementId(rule_id))
        .ok_or_else(|| BuildError("equipment rule does not exist".into()))?;
    let Rule::TrafficLight { signals, .. } = &rule.rule else {
        return Err(BuildError("automatic targets require a signal rule".into()));
    };
    let heads: Vec<_> = signals
        .iter()
        .filter_map(|id| map.traffic_signal(*id))
        .collect();
    let pedestrian = !heads.is_empty() && heads.iter().all(|h| h.kind == SignalKind::Pedestrian);
    let vehicle = !heads.is_empty() && heads.iter().all(|h| h.kind == SignalKind::Vehicle);
    let kind = if pedestrian {
        "pedestrian"
    } else if vehicle {
        "vehicle"
    } else {
        "mixed"
    };
    let mut report = ProposalReport {
        rule_id, kind: kind.into(), map_snapshot: String::new(), candidates: vec![],
        eligible_count: 0, ambiguous: false, limited: false,
        warnings: vec!["Geometry suggests draft targets only; housing normals are unsigned and do not prove legal control, front face or phases.".into()],
    };
    let vertices = map
        .boundaries()
        .map(|b| b.geometry.points.len())
        .chain(map.crosswalks().map(|c| {
            c.left_edge
                .points
                .len()
                .saturating_add(c.right_edge.points.len())
        }))
        .chain(map.stop_lines().map(|s| s.geometry.points.len()))
        .fold(0usize, usize::saturating_add);
    if vertices > MAX_VERTICES
        || map.lane_count() > 10_000
        || heads.len() > 8
        || signals.len() != heads.len()
    {
        report.limited = true;
        report.warnings.push(
            "Map/head work budget or missing signal reference prevents complete suggestions."
                .into(),
        );
        return Ok(report);
    }
    report.map_snapshot = snapshot(map)?;
    if !(pedestrian || vehicle) || heads.iter().any(|h| axis(&h.geometry).is_none()) {
        report
            .warnings
            .push("Signal kind or housing orientation is unavailable; review manually.".into());
        return Ok(report);
    }
    let positions: Vec<_> = heads
        .iter()
        .map(|h| midpoint(&h.geometry).expect("checked head"))
        .collect();
    let nearby = near_context(map, positions[0]);
    let road_hint = if !rule.lanes.is_empty() {
        rule.lanes.clone()
    } else {
        nearby
    };
    let total = if pedestrian {
        map.crosswalk_count()
    } else {
        map.stop_line_count()
    };
    if total > MAX_TARGETS {
        report.limited = true;
        report
            .warnings
            .push("Target budget exceeded; no partial list is offered for adoption.".into());
        return Ok(report);
    }
    if pedestrian {
        for crossing in map.crosswalks() {
            let edges = [&crossing.left_edge, &crossing.right_edge];
            let endpoints: Vec<_> = edges
                .iter()
                .filter_map(|e| e.points.first().zip(e.points.last()))
                .flat_map(|(a, b)| [*a, *b])
                .collect();
            let distance = positions
                .iter()
                .map(|p| {
                    endpoints
                        .iter()
                        .map(|q| (q.x - p.x).hypot(q.y - p.y))
                        .fold(f64::INFINITY, f64::min)
                })
                .fold(0., f64::max);
            if distance.is_finite() && distance > MAX_DISTANCE {
                continue;
            }
            let mut candidate = Candidate {
                key: format!("crosswalk:{}", crossing.id.0),
                target_kind: "crosswalk".into(),
                target_id: crossing.id.0,
                lanes: vec![],
                controlled_crosswalks: vec![crossing.id.0],
                stop_lines: vec![],
                distance_m: distance.is_finite().then_some(distance),
                axis_degrees: None,
                road_context: "unavailable".into(),
                eligible: true,
                already_linked: false,
                reasons: vec![],
            };
            let walking = axis(edges[0]).zip(axis(edges[1]));
            if let Some((left, right)) = walking {
                let angle = heads
                    .iter()
                    .flat_map(|h| {
                        let a = axis(&h.geometry).expect("checked head");
                        [angle([-a[1], a[0]], left), angle([-a[1], a[0]], right)]
                    })
                    .fold(0., f64::max);
                candidate.axis_degrees = Some(angle);
                if angle > MAX_AXIS_DEGREES {
                    rejection(
                        &mut candidate,
                        "Housing normal and walking direction disagree (>35 degrees).",
                    );
                }
            } else {
                rejection(
                    &mut candidate,
                    "Walking edges are unavailable or degenerate.",
                );
            }
            if !distance.is_finite() || distance > PEDESTRIAN_DISTANCE {
                rejection(
                    &mut candidate,
                    "Crossing endpoints are outside the 8 m review radius.",
                );
            }
            if positions.iter().any(|p| {
                endpoints
                    .iter()
                    .min_by(|a, b| {
                        (a.x - p.x)
                            .hypot(a.y - p.y)
                            .total_cmp(&(b.x - p.x).hypot(b.y - p.y))
                    })
                    .is_none_or(|q| p.z - q.z < -0.5 || p.z - q.z > 6.)
            }) {
                rejection(
                    &mut candidate,
                    "Housing/crossing elevations do not support the same level.",
                );
            }
            let crossing_lanes: Vec<_> = map
                .regulatory_elements()
                .filter(|r| r.rule.crosswalk() == Some(crossing.id))
                .flat_map(|r| r.lanes.iter().copied())
                .collect();
            candidate.road_context = context(map, &road_hint, &crossing_lanes).into();
            if candidate.road_context == "unavailable" {
                rejection(
                    &mut candidate,
                    "Crossing has no shared or connected road context; review its lane associations.",
                );
            }
            candidate.reasons.push(
                "Road context is a location hint, not pedestrian control of vehicle lanes.".into(),
            );
            candidate.already_linked = rule.controlled_crosswalks == vec![crossing.id]
                && rule.lanes.is_empty()
                && rule.rule.stop_lines().is_empty();
            report.candidates.push(candidate);
        }
    } else {
        for stop in map.stop_lines() {
            let distance = positions
                .iter()
                .map(|p| {
                    stop.geometry
                        .nearest_point(p.xy())
                        .map_or(f64::INFINITY, |n| n.distance)
                })
                .fold(0., f64::max);
            if distance.is_finite() && distance > MAX_DISTANCE {
                continue;
            }
            let stop_lanes: BTreeSet<_> = map
                .regulatory_elements()
                .filter(|r| r.rule.stop_lines().contains(&stop.id))
                .flat_map(|r| r.lanes.iter().copied())
                .collect();
            // A topology connection can reach a different/opposing approach.
            // Suggest a stop for the reviewed movement, never replace that
            // movement with the target marking's lanes (or drop some lanes).
            let lanes = rule.lanes.clone();
            let stop_context: Vec<_> = stop_lanes.iter().copied().collect();
            let mut candidate = Candidate {
                key: format!("stop_line:{}", stop.id.0),
                target_kind: "stop_line".into(),
                target_id: stop.id.0,
                lanes: lanes.iter().map(|id| id.0).collect(),
                controlled_crosswalks: vec![],
                stop_lines: vec![stop.id.0],
                distance_m: distance.is_finite().then_some(distance),
                axis_degrees: None,
                road_context: context(map, &lanes, &stop_context).into(),
                eligible: true,
                already_linked: false,
                reasons: vec![],
            };
            if !finite(&stop.geometry) || !distance.is_finite() {
                rejection(
                    &mut candidate,
                    "Stop geometry is unavailable or non-finite.",
                );
            }
            if rule.lanes.is_empty() || lanes.is_empty() {
                rejection(
                    &mut candidate,
                    "Reviewed vehicle lanes and compatible stop-marking context are required.",
                );
            }
            if lanes.iter().any(|lane| !stop_lanes.contains(lane)) {
                rejection(
                    &mut candidate,
                    "Stop marking does not cover every reviewed vehicle lane; connected approaches cannot replace the controlled movement.",
                );
            }
            let mut maximum = 0f64;
            for &lane in &lanes {
                let Some(center) = map.centerline(lane) else {
                    rejection(&mut candidate, "Lane geometry is unavailable.");
                    continue;
                };
                let Some(mid) = midpoint(&stop.geometry) else {
                    rejection(&mut candidate, "Stop geometry is degenerate.");
                    continue;
                };
                let Some(n) = center.nearest_point(mid.xy()) else {
                    rejection(&mut candidate, "Lane direction is unavailable.");
                    continue;
                };
                if n.distance > 8. || (mid.z - n.point.z).abs() > 0.75 {
                    rejection(
                        &mut candidate,
                        "Stop marking is distant from its road or on another level.",
                    );
                }
                let tangent = Polyline3::new(
                    center
                        .points
                        .get(n.segment..n.segment + 2)
                        .unwrap_or(&[])
                        .to_vec(),
                );
                let Some(t) = axis(&tangent) else {
                    rejection(&mut candidate, "Lane tangent is degenerate.");
                    continue;
                };
                for (h, p) in heads.iter().zip(&positions) {
                    let a = axis(&h.geometry).expect("checked head");
                    maximum = maximum.max(angle([-a[1], a[0]], t));
                    if p.z - n.point.z < -0.5 || p.z - n.point.z > 12. {
                        rejection(
                            &mut candidate,
                            "Signal and road elevations do not support the same level.",
                        );
                    }
                }
            }
            if !lanes.is_empty() {
                candidate.axis_degrees = Some(maximum);
                if maximum > MAX_AXIS_DEGREES {
                    rejection(
                        &mut candidate,
                        "Housing normal and vehicle lane direction disagree (>35 degrees).",
                    );
                }
            }
            candidate.already_linked = rule.lanes == lanes
                && rule.rule.stop_lines() == vec![stop.id]
                && rule.controlled_crosswalks.is_empty();
            report.candidates.push(candidate);
        }
    }
    // The same atomic editor is authoritative for adoption; suggestions cannot
    // bypass target, participant or transverse-stop checks.
    for c in &mut report.candidates {
        if c.eligible {
            let mut draft = map.clone();
            if let Err(error) = relations::edit(
                &mut draft,
                &LinkEdit {
                    rule_id,
                    lanes: c.lanes.clone(),
                    controlled_crosswalks: c.controlled_crosswalks.clone(),
                    stop_lines: c.stop_lines.clone(),
                },
            ) {
                rejection(c, &error.to_string());
            }
        }
    }
    report.candidates.sort_by(|a, b| {
        b.eligible
            .cmp(&a.eligible)
            .then(
                a.distance_m
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&b.distance_m.unwrap_or(f64::INFINITY)),
            )
            .then(a.target_id.cmp(&b.target_id))
    });
    report.eligible_count = report.candidates.iter().filter(|c| c.eligible).count();
    report.ambiguous = report.eligible_count > 1;
    if report.eligible_count == 0 {
        report.warnings.push("No supported association candidate; keep this signal unresolved and review its source context.".into());
    }
    Ok(report)
}

/// Recompute against the complete current map, rejecting stale or rejected
/// candidates before the existing atomic editor. Preview choices never persist.
pub fn adopt(map: &mut Map, o: &Adoption) -> Result<LinkReport, BuildError> {
    let report = propose(map, o.rule_id)?;
    if report.limited || report.map_snapshot.is_empty() || report.map_snapshot != o.map_snapshot {
        return Err(BuildError(
            "Proposal is stale or incomplete; preview the current map again.".into(),
        ));
    }
    let c = report
        .candidates
        .iter()
        .find(|c| c.key == o.candidate_key && c.eligible)
        .ok_or_else(|| {
            BuildError("Candidate is unavailable or rejected; no association changed.".into())
        })?;
    relations::edit(
        map,
        &LinkEdit {
            rule_id: o.rule_id,
            lanes: c.lanes.clone(),
            controlled_crosswalks: c.controlled_crosswalks.clone(),
            stop_lines: c.stop_lines.clone(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectormap_core::{CrosswalkId, SignalId, StopLineId};
    fn fixture() -> Map {
        serde_json::from_value(serde_json::json!({"format":"vectormap-ir","version":1,
            "boundaries":[{"id":1,"kind":{"type":"virtual"},"geometry":[[0,2,0],[20,2,0]]},{"id":2,"kind":{"type":"virtual"},"geometry":[[0,-2,0],[20,-2,0]]}],
            "lanes":[{"id":3,"kind":"driving","left":1,"right":2}],
            "crosswalks":[{"id":10,"left_edge":[[8,-3,0],[8,3,0]],"right_edge":[[12,-3,0],[12,3,0]]},
                {"id":11,"left_edge":[[8,-4,0],[14,-4,0]],"right_edge":[[8,-6,0],[14,-6,0]]}],
            "stop_lines":[{"id":15,"geometry":[[6,-2,0],[6,2,0]]}],
            "traffic_signals":[{"id":20,"kind":"pedestrian","geometry":[[7,-5,3],[8,-5,3]],"height":0.5},
                {"id":22,"kind":"vehicle","geometry":[[6,-0.5,5],[6,0.5,5]],"height":0.5}],
            "regulatory_elements":[{"id":21,"rule":{"type":"traffic_light","signals":[20]},"lanes":[3]},
                {"id":23,"rule":{"type":"traffic_light","signals":[22]},"lanes":[3]},
                {"id":30,"rule":{"type":"crosswalk","crosswalk":10},"lanes":[3]},
                {"id":31,"rule":{"type":"crosswalk","crosswalk":11},"lanes":[3]},
                {"id":32,"rule":{"type":"stop_line","stop_line":15},"lanes":[3]}]})).unwrap()
    }
    fn choice(p: &ProposalReport, key: &str) -> Adoption {
        Adoption {
            rule_id: p.rule_id,
            map_snapshot: p.map_snapshot.clone(),
            candidate_key: key.into(),
        }
    }
    #[test]
    fn closer_wrong_orientation_is_held_and_preview_changes_nothing() {
        let mut map = fixture();
        let before = map.clone();
        let p = propose(&map, 21).unwrap();
        assert_eq!(map, before);
        assert_eq!(p.eligible_count, 1);
        assert!(!p.ambiguous);
        let good = p.candidates.iter().find(|c| c.target_id == 10).unwrap();
        let wrong = p.candidates.iter().find(|c| c.target_id == 11).unwrap();
        assert!(wrong.distance_m < good.distance_m);
        assert!(!wrong.eligible);
        assert!(
            wrong
                .reasons
                .iter()
                .any(|r| r.contains("direction disagree"))
        );
        assert!(adopt(&mut map, &choice(&p, "crosswalk:11")).is_err());
        assert_eq!(map, before);
        assert!(
            adopt(&mut map, &choice(&p, "crosswalk:10"))
                .unwrap()
                .changed
        );
        assert_eq!(
            map.lanes().collect::<Vec<_>>(),
            before.lanes().collect::<Vec<_>>()
        );
        assert_eq!(
            map.boundaries().collect::<Vec<_>>(),
            before.boundaries().collect::<Vec<_>>()
        );
        assert_eq!(
            map.crosswalks().collect::<Vec<_>>(),
            before.crosswalks().collect::<Vec<_>>()
        );
        assert_eq!(
            map.traffic_signals().collect::<Vec<_>>(),
            before.traffic_signals().collect::<Vec<_>>()
        );
        let rule = map.regulatory_element(RegulatoryElementId(21)).unwrap();
        assert!(rule.lanes.is_empty());
        assert_eq!(rule.controlled_crosswalks, vec![CrosswalkId(10)]);
        let after = map.clone();
        let fresh = propose(&map, 21).unwrap();
        assert!(
            fresh
                .candidates
                .iter()
                .any(|c| c.target_id == 10 && c.already_linked)
        );
        assert!(
            !adopt(&mut map, &choice(&fresh, "crosswalk:10"))
                .unwrap()
                .changed
        );
        assert_eq!(map, after);
    }
    #[test]
    fn ambiguity_is_reported_and_any_map_edit_invalidates_adoption_atomically() {
        let mut map = fixture();
        let c = map.crosswalk_mut(CrosswalkId(11)).unwrap();
        c.left_edge = Polyline3::new(vec![Point3::new(9., -3., 0.), Point3::new(9., 3., 0.)]);
        c.right_edge = Polyline3::new(vec![Point3::new(13., -3., 0.), Point3::new(13., 3., 0.)]);
        let before = map.clone();
        let p = propose(&map, 21).unwrap();
        assert_eq!(p.eligible_count, 2);
        assert!(p.ambiguous);
        assert_eq!(map, before);
        map.traffic_signal_mut(SignalId(22)).unwrap().height = Some(0.7);
        let edited = map.clone();
        assert!(
            adopt(&mut map, &choice(&p, "crosswalk:10"))
                .unwrap_err()
                .to_string()
                .contains("stale")
        );
        assert_eq!(map, edited);
        let fresh = propose(&map, 21).unwrap();
        assert!(adopt(&mut map, &choice(&fresh, "crosswalk:9999")).is_err());
        assert_eq!(map, edited);
    }
    #[test]
    fn crossing_nearest_endpoint_elevation_and_missing_road_context_hold_candidates() {
        let mut map = fixture();
        let c = map.crosswalk_mut(CrosswalkId(10)).unwrap();
        c.left_edge.points[0].z = 12.;
        c.right_edge.points[0].z = 12.;
        let p = propose(&map, 21).unwrap();
        assert_eq!(p.eligible_count, 0);
        assert!(
            p.candidates
                .iter()
                .find(|c| c.target_id == 10)
                .unwrap()
                .reasons
                .iter()
                .any(|r| r.contains("elevations"))
        );
        let mut map = fixture();
        map.regulatory_element_mut(RegulatoryElementId(30))
            .unwrap()
            .lanes
            .clear();
        assert_eq!(propose(&map, 21).unwrap().eligible_count, 0);
        let mut map = fixture();
        map.regulatory_element_mut(RegulatoryElementId(21))
            .unwrap()
            .lanes = vec![LaneId(9999)];
        map.regulatory_element_mut(RegulatoryElementId(30))
            .unwrap()
            .lanes = vec![LaneId(9999)];
        assert_eq!(propose(&map, 21).unwrap().eligible_count, 0);
    }
    #[test]
    fn vehicle_stop_requires_direction_transversality_elevation_and_existing_context() {
        let mut map = fixture();
        let p = propose(&map, 23).unwrap();
        assert_eq!(p.eligible_count, 1);
        assert!(
            adopt(&mut map, &choice(&p, "stop_line:15"))
                .unwrap()
                .changed
        );
        let mut map = fixture();
        map.stop_line_mut(StopLineId(15)).unwrap().geometry =
            Polyline3::new(vec![Point3::new(6., 0., 0.), Point3::new(10., 0., 0.)]);
        assert_eq!(propose(&map, 23).unwrap().eligible_count, 0);
        let mut map = fixture();
        for p in &mut map.stop_line_mut(StopLineId(15)).unwrap().geometry.points {
            p.z = 6.;
        }
        let p = propose(&map, 23).unwrap();
        assert_eq!(p.eligible_count, 0);
        assert!(
            p.candidates[0]
                .reasons
                .iter()
                .any(|r| r.contains("another level"))
        );
        let mut map = fixture();
        map.regulatory_element_mut(RegulatoryElementId(32))
            .unwrap()
            .lanes
            .clear();
        assert_eq!(propose(&map, 23).unwrap().eligible_count, 0);
        let mut map = fixture();
        for p in &mut map
            .traffic_signal_mut(SignalId(22))
            .unwrap()
            .geometry
            .points
        {
            p.z = 20.;
        }
        assert_eq!(propose(&map, 23).unwrap().eligible_count, 0);
        let mut map = fixture();
        map.traffic_signal_mut(SignalId(22)).unwrap().geometry =
            Polyline3::new(vec![Point3::new(6., 0., 5.), Point3::new(7., 0., 5.)]);
        assert_eq!(propose(&map, 23).unwrap().eligible_count, 0);
    }
    #[test]
    fn connected_other_movement_and_partial_lane_coverage_are_held_atomically() {
        let mut doc = serde_json::to_value(fixture()).unwrap();
        doc["boundaries"].as_array_mut().unwrap().extend([
            serde_json::json!({"id":40,"kind":{"type":"virtual"},"geometry":[[20,-2,0],[0,-2,0]]}),
            serde_json::json!({"id":41,"kind":{"type":"virtual"},"geometry":[[20,2,0],[0,2,0]]}),
        ]);
        doc["lanes"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"id":43,"kind":"driving","left":40,"right":41}));
        doc["topology"] = serde_json::json!([
            {"lane":3,"successors":[43]}, {"lane":43,"predecessors":[3]}
        ]);
        doc["stop_lines"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"id":16,"geometry":[[7,-2,0],[7,2,0]]}));
        doc["regulatory_elements"].as_array_mut().unwrap().push(
            serde_json::json!({"id":33,"rule":{"type":"stop_line","stop_line":16},"lanes":[43]}),
        );
        let mut map: Map = serde_json::from_value(doc).unwrap();
        assert!(linked(&map, LaneId(3), LaneId(43)));
        let before = map.clone();
        let p = propose(&map, 23).unwrap();
        let wrong = p.candidates.iter().find(|c| c.target_id == 16).unwrap();
        assert_eq!(wrong.road_context, "connected_lanes");
        assert_eq!(wrong.lanes, vec![3]);
        assert!(!wrong.eligible);
        assert!(
            wrong
                .reasons
                .iter()
                .any(|r| r.contains("every reviewed vehicle lane"))
        );
        assert_eq!(p.eligible_count, 1);
        assert!(adopt(&mut map, &choice(&p, "stop_line:16")).is_err());
        assert_eq!(map, before);
        map.regulatory_element_mut(RegulatoryElementId(23))
            .unwrap()
            .lanes
            .push(LaneId(43));
        let before = map.clone();
        let p = propose(&map, 23).unwrap();
        assert_eq!(p.eligible_count, 0);
        assert!(p.candidates.iter().all(|c| c.lanes == vec![3, 43]));
        assert!(adopt(&mut map, &choice(&p, "stop_line:15")).is_err());
        assert_eq!(map, before);
    }
    #[test]
    fn over_budget_map_never_offers_a_partial_adoptable_list() {
        let mut doc = serde_json::to_value(fixture()).unwrap();
        let prototype = doc["crosswalks"][0].clone();
        doc["crosswalks"] = serde_json::Value::Array(
            (0..129)
                .map(|i| {
                    let mut c = prototype.clone();
                    c["id"] = serde_json::json!(1000 + i);
                    c
                })
                .collect(),
        );
        let mut map: Map = serde_json::from_value(doc).unwrap();
        let before = map.clone();
        let p = propose(&map, 21).unwrap();
        assert!(p.limited);
        assert!(p.candidates.is_empty());
        assert!(adopt(&mut map, &choice(&p, "crosswalk:1000")).is_err());
        assert_eq!(map, before);
    }
    #[test]
    fn missing_kind_orientation_or_incomplete_budget_never_yields_adoptable_targets() {
        let mut map = fixture();
        map.traffic_signal_mut(SignalId(20))
            .unwrap()
            .geometry
            .points[1] = Point3::new(7., -5., 3.);
        assert_eq!(propose(&map, 21).unwrap().eligible_count, 0);
        let mut map = fixture();
        let rule = map.regulatory_element_mut(RegulatoryElementId(21)).unwrap();
        rule.rule = Rule::TrafficLight {
            signals: vec![SignalId(20), SignalId(22)],
            stop_line: None,
        };
        assert_eq!(propose(&map, 21).unwrap().kind, "mixed");
        assert_eq!(propose(&map, 21).unwrap().eligible_count, 0);
        let mut map = fixture();
        let rule = map.regulatory_element_mut(RegulatoryElementId(21)).unwrap();
        rule.rule = Rule::TrafficLight {
            signals: vec![SignalId(9999)],
            stop_line: None,
        };
        let p = propose(&map, 21).unwrap();
        assert!(p.limited);
        assert!(p.map_snapshot.is_empty());
        let before = map.clone();
        assert!(adopt(&mut map, &choice(&p, "crosswalk:10")).is_err());
        assert_eq!(map, before);
        assert!(propose(&fixture(), 32).is_err());
        assert!(propose(&fixture(), 99999).is_err());
        assert!(
            serde_json::from_str::<Adoption>(
                r#"{"rule_id":21,"map_snapshot":"x","candidate_key":"crosswalk:10","lanes":[3]}"#
            )
            .is_err()
        );
    }
}
