use super::discovery::*;
use crate::PointCloud;
use vectormap_core::{LaneDirection, Map, NewRoad, Point3, Polyline3, RoadLane};

fn scene() -> (Map, PointCloud) {
    let mut map = Map::new();
    map.build_road(NewRoad::new(
        Polyline3::new(vec![
            Point3::new(-5.0, 0.0, 2.0),
            Point3::new(40.0, 0.0, 2.0),
        ]),
        vec![RoadLane::new(3.5, LaneDirection::Forward)],
    ))
    .unwrap();
    let mut cloud = PointCloud {
        colors: Some(vec![]),
        ..Default::default()
    };
    for x in 0..=450 {
        for y in 0..=140 {
            let x = -5.0 + x as f64 * 0.1;
            let y = -7.0 + y as f64 * 0.1;
            let bright = ((8.0..12.0).contains(&x) && (x - 8.0) % 1.0 < 0.5
                || (20.0..20.6).contains(&x))
                && y.abs() <= 3.0;
            cloud.positions.push([x, y, 2.0 + 0.01 * y]);
            cloud
                .colors
                .as_mut()
                .unwrap()
                .push([if bright { 220 } else { 60 }; 3]);
        }
    }
    // A head attached to a narrow stem. The broad face must be found without
    // assigning the pole's bottom as a fabricated housing elevation.
    for y in 0..=24 {
        for z in 0..=12 {
            cloud
                .positions
                .push([25.0, -0.6 + y as f64 * 0.05, 6.0 + z as f64 * 0.05]);
            cloud.colors.as_mut().unwrap().push([90; 3]);
        }
    }
    for z in 0..=85 {
        for x in 0..=2 {
            for y in 0..=2 {
                cloud.positions.push([
                    24.97 + x as f64 * 0.03,
                    -0.6 + y as f64 * 0.03,
                    2.0 + z as f64 * 0.05,
                ]);
                cloud.colors.as_mut().unwrap().push([90; 3]);
            }
        }
    }
    (map, cloud)
}

#[test]
fn source_only_scan_finds_paint_and_connected_head_without_feature_boxes() {
    let (map, cloud) = scene();
    let o = DiscoveryOptions::default();
    let r = propose(&map, &cloud, &o).unwrap();
    assert!(r.candidates.iter().any(|c|matches!(c.evidence,Evidence::RepeatedPaint{ref measurement,..} if measurement.stripe_count==4)));
    assert!(r.candidates.iter().any(|c|matches!(c.evidence,Evidence::TransversePaint{ref geometry,..} if (geometry[0][0]-20.3).abs()<0.3 && (geometry[0][2]-geometry[1][2]).abs()>0.04)));
    assert!(r.candidates.iter().any(|c|matches!(c.evidence,Evidence::ElevatedPanel{ref geometry,height,..} if geometry[0][2]>=5.95 && (0.4..0.8).contains(&height))));
    assert!(
        r.candidates
            .iter()
            .all(|c| c.review_required && !c.key.is_empty())
    );
    // Ordinary ground and single bars must not silently become crosswalks.
    assert!(
        r.candidates
            .iter()
            .filter(|c| matches!(c.evidence, Evidence::RepeatedPaint { .. }))
            .all(|c| c.min[0] < 13.0)
    );
}

#[test]
fn confirmation_is_atomic_explicit_and_replay_preserves_edits() {
    let (mut map, cloud) = scene();
    let o = DiscoveryOptions::default();
    let before = map.clone();
    let r = propose(&map, &cloud, &o).unwrap();
    let lane = map.lanes().next().unwrap().id;
    let mut selections: Vec<_> = r
        .candidates
        .iter()
        .map(|c| Confirmation {
            candidate: c.id,
            key: c.key.clone(),
            classification: match c.evidence {
                Evidence::RepeatedPaint { .. } => Classification::Crosswalk,
                Evidence::TransversePaint { .. } => Classification::StopLine,
                Evidence::ElevatedPanel { .. } => Classification::VehicleSignal,
            },
            lanes: vec![lane],
        })
        .collect();
    selections[0].lanes.clear();
    assert!(add(&mut map, &cloud, &o, &selections).is_err());
    assert_eq!(map, before);
    selections[0].lanes = vec![lane];
    selections.last_mut().unwrap().key = "stale".into();
    assert!(add(&mut map, &cloud, &o, &selections).is_err());
    assert_eq!(map, before);
    selections.last_mut().unwrap().key = r.candidates[selections.last().unwrap().candidate]
        .key
        .clone();
    let added = add(&mut map, &cloud, &o, &selections).unwrap();
    assert!(added.iter().all(|a| !a.reused));
    assert!(map.traffic_signals().all(|s| s.bulbs.is_empty()));
    assert!(
        map.regulatory_elements()
            .all(|r| r.rule.type_name() != "traffic_sign")
    );
    let stop = map.stop_lines().next().unwrap().id;
    let s = map.stop_line_mut(stop).unwrap();
    s.geometry.points[0].x += 0.1;
    let edited = map.clone();
    let replay = add(&mut map, &cloud, &o, &selections).unwrap();
    assert!(replay.iter().all(|a| a.reused));
    assert_eq!(map, edited);
}

#[test]
fn unpainted_or_invalid_sources_do_not_fabricate_features() {
    let (map, mut cloud) = scene();
    cloud.positions.retain(|p| p[2] < 3.0);
    cloud.colors = Some(vec![[60; 3]; cloud.len()]);
    assert!(
        propose(&map, &cloud, &Default::default())
            .unwrap()
            .candidates
            .is_empty()
    );
    assert!(propose(&Map::new(), &cloud, &Default::default()).is_err());
    cloud.colors.as_mut().unwrap().pop();
    assert!(propose(&map, &cloud, &Default::default()).is_err());
}

#[test]
fn regularly_missing_returns_are_not_dark_crosswalk_gaps() {
    let (map, mut cloud) = scene();
    let keep: Vec<_> = cloud
        .positions
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if p[2] > 3.0
                || ((8.0..12.0).contains(&p[0]) && p[1].abs() <= 3.0 && (p[0] - 8.0) % 1.0 >= 0.5)
            {
                None
            } else {
                Some(i)
            }
        })
        .collect();
    cloud = cloud.select(&keep);
    cloud.colors = Some(
        cloud
            .positions
            .iter()
            .map(|p| {
                [if (8.0..12.0).contains(&p[0]) && p[1].abs() <= 3.0 {
                    220
                } else {
                    60
                }; 3]
            })
            .collect(),
    );
    let r = propose(&map, &cloud, &Default::default()).unwrap();
    assert!(
        r.candidates
            .iter()
            .all(|c| !matches!(c.evidence, Evidence::RepeatedPaint { .. })),
        "sampling gaps must not be invented as dark paint"
    );
}
