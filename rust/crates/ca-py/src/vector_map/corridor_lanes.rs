//! Explicit lane hypotheses inside frozen source-span geometry; no road semantics inferred.
use serde::{Deserialize, Serialize};
use serde_json::json;
use vectormap_core::{
    BoundaryId, BoundaryKind, LaneDirection, LaneKind, Map, NewRoad, Polyline3, RoadLane,
    SpeedLimit,
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LaneSpec {
    direction: LaneDirection,
    kind: LaneKind,
    one_way: bool,
    fraction: f64,
    minimum_width_m: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SegmentSpec {
    center_curve_id: u64,
    left_curve_id: u64,
    right_curve_id: u64,
    lanes: Vec<LaneSpec>,
    speed_limit_kmh: Option<f64>,
}

pub(super) fn generate(geometry: &str, specifications: &str) -> Result<String, String> {
    let text = std::fs::read_to_string(geometry).map_err(|e| e.to_string())?;
    let source = vectormap_io::json::from_str(&text).map_err(|e| e.to_string())?;
    if source.map.lane_count() != 0
        || source.map.metadata().attributes.get("cloudanalyzer:draft") != Some("surface_geometry")
        || source
            .issues
            .iter()
            .any(|i| i.severity == vectormap_core::Severity::Error)
    {
        return Err("use a structurally valid source geometry draft without lanes".into());
    }
    let specs: Vec<SegmentSpec> =
        serde_json::from_str(specifications).map_err(|e| e.to_string())?;
    if specs.is_empty() || specs.len() > 256 {
        return Err("supply 1..256 segment specifications".into());
    }
    let mut out = Map::new();
    *out.metadata_mut() = source.map.metadata().clone();
    out.metadata_mut()
        .attributes
        .insert("cloudanalyzer:draft", "corridor_lane_hypothesis");
    out.metadata_mut()
        .attributes
        .insert("cloudanalyzer:boundary_policy", "source_span_hypothesis");
    let mut seen = std::collections::HashSet::new();
    let mut created = Vec::new();
    let mut vertices = 0;
    for spec in &specs {
        if !seen.insert(spec.center_curve_id) || spec.lanes.is_empty() || spec.lanes.len() > 16 {
            return Err(
                "use unique centre curves and 1..16 explicitly specified lanes per segment".into(),
            );
        }
        let curve = |id, role| -> Result<Polyline3, String> {
            let b = source
                .map
                .boundary(BoundaryId(id))
                .ok_or("unknown source curve")?;
            if b.attributes.get("cloudanalyzer:role") != Some(role) {
                return Err("source curve role does not match the specification".into());
            }
            Ok(b.geometry.clone())
        };
        let center = curve(spec.center_curve_id, "source_center")?;
        let left = curve(spec.left_curve_id, "source_left")?;
        let right = curve(spec.right_curve_id, "source_right")?;
        vertices += center.len() * (spec.lanes.len() + 1);
        if vertices > 100_000
            || center.len() < 2
            || center.len() != left.len()
            || center.len() != right.len()
        {
            return Err("source sections disagree or exceed the 100000-vertex build budget".into());
        }
        if spec.lanes.iter().any(|l| {
            !l.fraction.is_finite()
                || l.fraction <= 0.
                || l.fraction > 1.
                || !l.minimum_width_m.is_finite()
                || !(0.5..=10.).contains(&l.minimum_width_m)
        }) || (spec.lanes.iter().map(|l| l.fraction).sum::<f64>() - 1.).abs() > 1e-9
        {
            return Err("positive lane fractions must sum to 1; minimum widths must be finite within 0.5..10 metres".into());
        }
        if spec
            .speed_limit_kmh
            .is_some_and(|v| !v.is_finite() || !(0.1..=200.).contains(&v))
            || (spec.speed_limit_kmh.is_none()
                && spec.lanes.iter().any(|l| l.kind.is_vehicle_lane()))
        {
            return Err(
                "vehicle lane hypotheses require an explicit speed within 0.1..200 km/h".into(),
            );
        }
        let spans: Vec<f64> = left
            .points
            .iter()
            .zip(&right.points)
            .map(|(a, b)| (a.x - b.x).hypot(a.y - b.y))
            .collect();
        let minimum_span = spans.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum_span = spans.iter().copied().fold(0., f64::max);
        for lane in &spec.lanes {
            if minimum_span * lane.fraction + 1e-6 < lane.minimum_width_m {
                return Err(format!(
                    "source span cannot contain the requested minimum lane width on centre curve {}",
                    spec.center_curve_id
                ));
            }
        }
        let mut boundaries = vec![left.clone()];
        let mut fraction = 0.;
        for lane in spec.lanes.iter().take(spec.lanes.len() - 1) {
            fraction += lane.fraction;
            boundaries.push(Polyline3::from_xyz(
                &left
                    .points
                    .iter()
                    .zip(&right.points)
                    .map(|(a, b)| {
                        let a = [a.x, a.y, a.z];
                        let b = [b.x, b.y, b.z];
                        std::array::from_fn(|i| a[i] + (b[i] - a[i]) * fraction)
                    })
                    .collect::<Vec<_>>(),
            ));
        }
        boundaries.push(right.clone());
        let lanes: Vec<RoadLane> = spec
            .lanes
            .iter()
            .map(|l| RoadLane {
                width: None,
                direction: l.direction,
                kind: l.kind,
            })
            .collect();
        // Lanelet2 infers boundary directions from geometry on reload. Cut the
        // segment at source sections until every piece keeps them, as ordinary
        // road building does; cuts at sections keep every source vertex exact.
        let slice = |line: &Polyline3, i: usize, j: usize| -> Vec<[f64; 3]> {
            line.points[i..=j].iter().map(|p| [p.x, p.y, p.z]).collect()
        };
        // A lane's bounds must also run the same way end to end, as the validator
        // checks: on a tight turn a wide band's inner edge can fold back on itself.
        let aligned = |bounds: &[Vec<[f64; 3]>]| {
            let d = |a: &[f64; 3], b: &[f64; 3]| (a[0] - b[0]).hypot(a[1] - b[1]);
            bounds.windows(2).all(|pair| {
                let (l, r) = (&pair[0], &pair[1]);
                let (ls, le, rs, re) = (&l[0], &l[l.len() - 1], &r[0], &r[r.len() - 1]);
                d(ls, re) + d(le, rs) >= d(ls, rs) + d(le, re) + 1e-9
            })
        };
        let mut ranges = Vec::new();
        let mut pending = vec![(0, center.len() - 1)];
        while let Some((i, j)) = pending.pop() {
            let reference = slice(&center, i, j);
            let bounds: Vec<_> = boundaries.iter().map(|b| slice(b, i, j)).collect();
            if ca_core::vector_map::lanelet_directions_unambiguous(&reference, &bounds, &lanes)
                && aligned(&bounds)
            {
                ranges.push((i, j));
            } else if j - i <= 1 {
                return Err(format!(
                    "source geometry on centre curve {} has ambiguous Lanelet2 boundary directions between sections {i} and {j}; defer that interval",
                    spec.center_curve_id
                ));
            } else {
                let middle = (i + j) / 2;
                pending.push((middle, j));
                pending.push((i, middle));
            }
        }
        let mut chains = vec![Vec::new(); lanes.len()];
        let mut pieces = Vec::new();
        for &(i, j) in &ranges {
            let mut road = NewRoad::new(Polyline3::from_xyz(&slice(&center, i, j)), lanes.clone());
            road.boundaries = Some(
                boundaries
                    .iter()
                    .map(|b| Polyline3::from_xyz(&slice(b, i, j)))
                    .collect(),
            );
            road.speed_limit = spec.speed_limit_kmh.map(SpeedLimit::from_kmh);
            road.edge_kind = Some(BoundaryKind::Virtual);
            road.lane_line_kind = Some(BoundaryKind::Virtual);
            road.center_line_kind = Some(BoundaryKind::Virtual);
            road.attributes
                .insert("cloudanalyzer:assumption_status", "hypothesis");
            road.attributes
                .insert("cloudanalyzer:boundary_policy", "source_span_hypothesis");
            road.attributes.insert(
                "cloudanalyzer:source_center_curve",
                spec.center_curve_id.to_string(),
            );
            let (built, _) = out.build_road(road).map_err(|e| e.to_string())?;
            let lane_ids: Vec<_> = built.lanes.iter().map(|chain| chain[0]).collect();
            for (id, lane) in lane_ids.iter().zip(&spec.lanes) {
                out.lane_mut(*id).unwrap().one_way = lane.one_way;
            }
            let first = out.lane(lane_ids[0]).unwrap();
            let last = out.lane(*lane_ids.last().unwrap()).unwrap();
            let outer_left = match spec.lanes[0].direction {
                LaneDirection::Forward => first.left.boundary,
                LaneDirection::Backward => first.right.boundary,
            };
            let outer_right = match spec.lanes.last().unwrap().direction {
                LaneDirection::Forward => last.right.boundary,
                LaneDirection::Backward => last.left.boundary,
            };
            let retained = |id, original: &Polyline3| {
                let original = Polyline3::from_xyz(&slice(original, i, j));
                let b = &out.boundary(id).unwrap().geometry;
                b == &original || b == &original.reversed()
            };
            if !retained(outer_left, &left) || !retained(outer_right, &right) {
                return Err("road building changed the outer source curves".into());
            }
            for (chain, id) in chains.iter_mut().zip(&lane_ids) {
                chain.push(*id);
            }
            pieces.push(json!({"section_range":[i, j], "lane_ids":lane_ids,
                "outer_boundary_ids":[outer_left, outer_right]}));
        }
        for (chain, lane) in chains.iter_mut().zip(&spec.lanes) {
            if lane.direction == LaneDirection::Backward {
                chain.reverse();
            }
            for pair in chain.windows(2) {
                out.connect(pair[0], pair[1]).map_err(|e| e.to_string())?;
            }
        }
        let lane_ids: Vec<_> = chains.iter().map(|chain| chain[0]).collect();
        created.push(json!({"center_curve_id":spec.center_curve_id, "lane_ids":lane_ids,
            "lane_chains":chains, "pieces":pieces, "source_outer_curves_preserved":true,
            "lane_width_ranges_m":spec.lanes.iter().map(|l| [minimum_span*l.fraction, maximum_span*l.fraction]).collect::<Vec<_>>()}));
    }
    super::artifacts(
        &out,
        json!(source.issues),
        json!({"schema":"cloudanalyzer.corridor_lanes.v1",
        "boundary_policy":"source_span_hypothesis", "specifications":specs, "built_segments":created,
        "road_semantics_inferred":false, "complete_width_resolved":false, "deployment_ready":false,
        "warnings":["Source spans are explicit layout hypotheses, not complete road widths. Lane fractions, kind, direction, one-way use and speed remain unverified. Virtual boundaries do not claim observed markings. Separate source pieces are not connected or filled."]}),
    )
}
