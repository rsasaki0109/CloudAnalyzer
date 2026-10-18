//! Operator-reviewed equipment associations; no geometry or semantic inference.
use super::BuildError;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use vectormap_core::{CrosswalkId, LaneId, Map, RegulatoryElementId, Rule, SignalKind, StopLineId};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkEdit {
    pub rule_id: u64,
    #[serde(default)]
    pub lanes: Vec<u64>,
    #[serde(default)]
    pub controlled_crosswalks: Vec<u64>,
    #[serde(default)]
    pub stop_lines: Vec<u64>,
}
#[derive(Debug, Serialize)]
pub struct LinkReport {
    pub rule_id: u64,
    pub changed: bool,
    pub warnings: Vec<String>,
}

/// Existing associations, including legacy pedestrian vehicle-lane references.
/// Reading never tags, resolves or changes the map.
pub fn inspect(map: &Map) -> Value {
    json!(map.regulatory_elements().filter_map(|r| {
        let kind = match &r.rule {
            Rule::TrafficLight { signals, .. } => {
                if !signals.is_empty() && signals.iter().all(|s| map.traffic_signal(*s).is_some_and(|s| s.kind == SignalKind::Pedestrian)) { "pedestrian" }
                else if !signals.is_empty() && signals.iter().all(|s| map.traffic_signal(*s).is_some_and(|s| s.kind == SignalKind::Vehicle)) { "vehicle" }
                else { "mixed" }
            },
            Rule::Crosswalk { .. } => "crosswalk",
            Rule::StopLine { .. } => "stop_line",
            _ => return None,
        };
        Some(json!({"id":r.id,"kind":kind,"signals":r.rule.signals(),"crosswalk":r.rule.crosswalk(),"lanes":r.lanes,"controlled_crosswalks":r.controlled_crosswalks,"stop_lines":r.rule.stop_lines(),
            "review_source":r.attributes.get("cloudanalyzer_relationships_source").or_else(||r.attributes.get_prefixed("lanelet2","cloudanalyzer_relationships_source")).unwrap_or("unreviewed_or_imported")}))
    }).collect::<Vec<_>>())
}

pub fn edit(map: &mut Map, o: &LinkEdit) -> Result<LinkReport, BuildError> {
    if [&o.lanes, &o.controlled_crosswalks, &o.stop_lines]
        .iter()
        .any(|v| v.len() > 128)
    {
        return Err(BuildError("association list exceeds 128 targets".into()));
    }
    let id = RegulatoryElementId(o.rule_id);
    let rule = map
        .regulatory_element(id)
        .ok_or_else(|| BuildError("equipment rule does not exist".into()))?;
    // Reusing a stop marking requires its reviewed road context; proximity alone
    // cannot establish control. Any selected stop must also cross every lane.
    for &stop in &o.stop_lines {
        let stop = StopLineId(stop);
        let line = map
            .stop_line(stop)
            .ok_or_else(|| BuildError("stop line does not exist".into()))?;
        let xyz: Vec<_> = line
            .geometry
            .points
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();
        for &lane in &o.lanes {
            if !super::quality::transverse_stop(map, &xyz, LaneId(lane)) {
                return Err(BuildError(format!(
                    "stop line {} is not transverse to lane {lane}",
                    stop.0
                )));
            }
            if matches!(rule.rule, Rule::TrafficLight { .. })
                && !map
                    .regulatory_elements()
                    .any(|r| r.rule.stop_lines().contains(&stop) && r.lanes.contains(&LaneId(lane)))
            {
                return Err(BuildError(format!(
                    "stop line {} has no reviewed association with lane {lane}; review its marking first",
                    stop.0
                )));
            }
        }
    }
    if !o.stop_lines.is_empty() && o.lanes.is_empty() && !matches!(rule.rule, Rule::StopLine { .. })
    {
        return Err(BuildError(
            "stop-line association requires controlled vehicle lanes".into(),
        ));
    }
    let mut draft = map.clone();
    let changes = draft
        .set_regulatory_links(
            id,
            &o.lanes.iter().copied().map(LaneId).collect::<Vec<_>>(),
            &o.controlled_crosswalks
                .iter()
                .copied()
                .map(CrosswalkId)
                .collect::<Vec<_>>(),
            &o.stop_lines
                .iter()
                .copied()
                .map(StopLineId)
                .collect::<Vec<_>>(),
        )
        .map_err(|e| BuildError(e.to_string()))?;
    let changed = !changes.modified.is_empty();
    if changed {
        let attrs = &mut draft
            .regulatory_element_mut(id)
            .expect("checked rule")
            .attributes;
        for (key, value) in [
            ("cloudanalyzer_relationships_source", "user_reviewed"),
            ("cloudanalyzer_review_required", "yes"),
        ] {
            attrs.remove(&format!("lanelet2:{key}"));
            attrs.insert(key, value);
        }
        *map = draft;
    }
    Ok(LinkReport { rule_id:o.rule_id, changed, warnings:vec!["Associations are operator-reviewed drafts. Geometry and lamps are unchanged; legal control and phases require independent evidence.".into()] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectormap_core::{
        CrosswalkGeometry, LaneDirection, NewCrosswalk, NewRoad, NewTrafficSignal, Point3,
        Polyline3, RoadLane, StopLineChoice,
    };
    fn fixture() -> (Map, LinkEdit, CrosswalkId) {
        let mut map = Map::new();
        let road = map
            .build_road(NewRoad::new(
                Polyline3::new(vec![Point3::new(0., 0., 2.), Point3::new(20., 0., 2.)]),
                vec![RoadLane::new(3.5, LaneDirection::Forward)],
            ))
            .unwrap()
            .0;
        let lane = road.lanes[0][0];
        let crossing = map
            .add_crosswalk(NewCrosswalk {
                geometry: CrosswalkGeometry::Across {
                    lane,
                    station: 10.,
                    width: 4.,
                    margin: 0.5,
                },
                crossing_lanes: Some(vec![lane]),
                stop_line_offset: Some(1.),
            })
            .unwrap()
            .0;
        let stop = map
            .regulatory_elements()
            .find_map(|r| match r.rule {
                Rule::Crosswalk { ref stop_lines, .. } => stop_lines.first().copied(),
                _ => None,
            })
            .unwrap();
        let mut spec = NewTrafficSignal::for_lanes(vec![lane]);
        spec.stop_line = StopLineChoice::None;
        spec.bulbs = Some(vec![]);
        let signal = map.add_traffic_signal(spec).unwrap().0;
        let rule = map
            .regulatory_elements()
            .find(|r| r.rule.signals().contains(&signal))
            .unwrap()
            .id;
        (
            map,
            LinkEdit {
                rule_id: rule.0,
                lanes: vec![lane.0],
                controlled_crosswalks: vec![],
                stop_lines: vec![stop.0],
            },
            crossing,
        )
    }
    #[test]
    fn reviewed_vehicle_stop_preserves_observations_and_repeated_edit_is_exact() {
        let (mut map, edit, _) = fixture();
        let before = map.clone();
        let read = inspect(&map);
        assert_eq!(map, before);
        assert!(read.as_array().unwrap().len() >= 2);
        assert!(super::edit(&mut map, &edit).unwrap().changed);
        assert_eq!(
            map.lanes().collect::<Vec<_>>(),
            before.lanes().collect::<Vec<_>>()
        );
        assert_eq!(
            map.boundaries().collect::<Vec<_>>(),
            before.boundaries().collect::<Vec<_>>()
        );
        assert_eq!(
            map.stop_lines().collect::<Vec<_>>(),
            before.stop_lines().collect::<Vec<_>>()
        );
        assert_eq!(
            map.traffic_signals().collect::<Vec<_>>(),
            before.traffic_signals().collect::<Vec<_>>()
        );
        let reviewed = map.clone();
        assert!(!super::edit(&mut map, &edit).unwrap().changed);
        assert_eq!(map, reviewed);
    }
    #[test]
    fn wrong_kind_missing_context_and_longitudinal_stop_leave_map_exact() {
        let (mut map, mut edit, crossing) = fixture();
        let before = map.clone();
        edit.controlled_crosswalks = vec![crossing.0];
        assert!(super::edit(&mut map, &edit).is_err());
        assert_eq!(map, before);
        edit.controlled_crosswalks.clear();
        edit.lanes = vec![99999];
        assert!(super::edit(&mut map, &edit).is_err());
        assert_eq!(map, before);
        edit.lanes = vec![before.lanes().next().unwrap().id.0];
        let stop = map.stop_line_mut(StopLineId(edit.stop_lines[0])).unwrap();
        stop.geometry = Polyline3::new(vec![Point3::new(5., 0., 2.), Point3::new(8., 0., 2.)]);
        let before = map.clone();
        assert!(super::edit(&mut map, &edit).is_err());
        assert_eq!(map, before);
    }
    #[test]
    fn pedestrian_control_clears_legacy_lanes_and_geometry_edit_requires_rereview() {
        let (mut map, mut edit, crossing) = fixture();
        let signal = map
            .regulatory_element(RegulatoryElementId(edit.rule_id))
            .unwrap()
            .rule
            .signals()[0];
        map.traffic_signal_mut(signal).unwrap().kind = SignalKind::Pedestrian;
        edit.lanes.clear();
        edit.stop_lines.clear();
        edit.controlled_crosswalks = vec![crossing.0];
        assert!(super::edit(&mut map, &edit).unwrap().changed);
        let c = map.crosswalk(crossing).unwrap();
        let mut points: Vec<_> = c.outline().points.iter().map(|p| [p.x, p.y, p.z]).collect();
        for p in &mut points {
            p[2] += 0.05;
        }
        super::super::feature_editing::edit(
            &mut map,
            &super::super::feature_editing::FeatureEdit {
                kind: super::super::feature_editing::FeatureKind::Crosswalk,
                id: crossing.0,
                points,
                height: None,
            },
        )
        .unwrap();
        assert_eq!(
            map.regulatory_element(RegulatoryElementId(edit.rule_id))
                .unwrap()
                .attributes
                .get("cloudanalyzer_review_required"),
            Some("yes")
        );
        assert_eq!(
            map.regulatory_element(RegulatoryElementId(edit.rule_id))
                .unwrap()
                .controlled_crosswalks,
            vec![crossing]
        );
    }
}
