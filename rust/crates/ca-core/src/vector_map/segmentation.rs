//! Split measured boundaries at their corresponding trajectory cross-sections.
//!
//! Nearest-point projection can jump to a different branch of a returning drive
//! or to the endpoint of a fitted boundary. Extraction already supplies aligned
//! vertices; interpolate every line using the same reference edge and fraction.
use super::{BuildDiagnostic, BuildError, BuildOptions, ExtractedRoad};
use vectormap_core::{
    BuiltRoad, LaneDirection, Map, NewRoad, Point3, Polyline3, RoadLane, SpeedLimit,
};

// Bound allocations and native edits even for a tiny positive segment length.
const MAX_PIECES: usize = 10_000;

fn polyline(points: Vec<[f64; 3]>) -> Polyline3 {
    Polyline3::new(
        points
            .into_iter()
            .map(|p| Point3::new(p[0], p[1], p[2]))
            .collect(),
    )
}

fn specification(
    reference: Vec<[f64; 3]>,
    boundaries: Vec<Vec<[f64; 3]>>,
    lanes: &[RoadLane],
    speed: f64,
) -> NewRoad {
    let mut spec = NewRoad::new(polyline(reference), lanes.to_vec());
    spec.boundaries = Some(boundaries.into_iter().map(polyline).collect());
    spec.speed_limit = Some(SpeedLimit::from_kmh(speed));
    spec
}

/// The caller works on a scratch map and commits only after all roads succeed.
#[cfg(test)]
fn build(
    map: &mut Map,
    road: ExtractedRoad,
    lanes: &[RoadLane],
    options: &BuildOptions,
) -> Result<(BuiltRoad, usize), BuildError> {
    build_diagnostic(map, road, lanes, options, None)
}

pub(super) fn build_diagnostic(
    map: &mut Map,
    road: ExtractedRoad,
    lanes: &[RoadLane],
    options: &BuildOptions,
    diagnostic: Option<&mut Option<BuildDiagnostic>>,
) -> Result<(BuiltRoad, usize), BuildError> {
    // Keep the native builder's 3D arc-length interpretation of piece length.
    let stations = polyline(road.reference.clone()).stations();
    let total = stations.last().copied().unwrap_or(0.0);
    let pieces = if options.segment_length > 0.0 {
        (total / options.segment_length).round().max(1.0)
    } else {
        1.0
    };
    if !total.is_finite() || !pieces.is_finite() || pieces > MAX_PIECES as f64 {
        return Err(BuildError(format!(
            "road segmentation exceeds the {MAX_PIECES}-piece limit; increase segment length"
        )));
    }
    let pieces = pieces as usize;
    if options.segment_length == 0.0 {
        return map
            .build_road(specification(
                road.reference,
                road.boundaries,
                lanes,
                options.speed_limit,
            ))
            .map(|(built, _)| (built, 0))
            .map_err(|e| BuildError(e.to_string()));
    }
    if road.reference.len() < 2
        || road.boundaries.len() != lanes.len() + 1
        || road
            .boundaries
            .iter()
            .any(|b| b.len() != road.reference.len())
    {
        return Err(BuildError(
            "road segmentation needs aligned boundary cross-sections".into(),
        ));
    }
    let mut chains = vec![Vec::with_capacity(pieces); lanes.len()];
    let mut pending: Vec<_> = (0..pieces)
        .rev()
        .map(|i| {
            let start = total * i as f64 / pieces as f64;
            let end = total * (i + 1) as f64 / pieces as f64;
            (start, end, start, end)
        })
        .collect();
    let mut extra_cuts = 0;
    while let Some((start, end, context_start, context_end)) = pending.pop() {
        let first = locate(&stations, start);
        let last = locate(&stations, end);
        let reference = slice(&road.reference, first, last);
        let boundaries: Vec<_> = road
            .boundaries
            .iter()
            .map(|line| slice(line, first, last))
            .collect();
        if !unambiguous(&reference, &boundaries, lanes) {
            if pieces + extra_cuts >= MAX_PIECES || end - start <= 0.2 {
                if let Some(diagnostic) = diagnostic {
                    let first = locate(&stations, context_start);
                    let last = locate(&stations, context_end);
                    // At most 2048 preview vertices, including all lane edges.
                    let limit = (2048 / (road.boundaries.len() + 1)).min(256);
                    *diagnostic = Some(BuildDiagnostic {
                        code: "ambiguous_boundary_direction",
                        location: std::array::from_fn(|axis| {
                            (reference[0][axis] + reference.last().unwrap()[axis]) * 0.5
                        }),
                        reference: sample(slice(&road.reference, first, last), limit),
                        boundaries: road
                            .boundaries
                            .iter()
                            .map(|line| sample(slice(line, first, last), limit))
                            .collect(),
                        context_length: context_end - context_start,
                        forward_lanes: options.forward_lanes,
                        backward_lanes: options.backward_lanes,
                        lane_width: options.lane_width,
                        segment_length: options.segment_length,
                    });
                }
                return Err(BuildError("road boundaries have ambiguous travel directions; review the trajectory and lane widths".into()));
            }
            let middle = (start + end) * 0.5;
            pending.push((middle, end, context_start, context_end));
            pending.push((start, middle, context_start, context_end));
            extra_cuts += 1;
            continue;
        }
        let (built, _) = map
            .build_road(specification(
                reference,
                boundaries,
                lanes,
                options.speed_limit,
            ))
            .map_err(|e| BuildError(e.to_string()))?;
        for (chain, column) in chains.iter_mut().zip(built.lanes) {
            chain.extend(column);
        }
    }
    for (chain, lane) in chains.iter_mut().zip(lanes) {
        if lane.direction == LaneDirection::Backward {
            chain.reverse();
        }
        for pair in chain.windows(2) {
            map.connect(pair[0], pair[1])
                .map_err(|e| BuildError(e.to_string()))?;
        }
    }
    Ok((
        BuiltRoad {
            lanes: chains,
            road: None,
        },
        extra_cuts,
    ))
}

fn sample(points: Vec<[f64; 3]>, limit: usize) -> Vec<[f64; 3]> {
    if points.len() <= limit {
        return points;
    }
    (0..limit)
        .map(|i| points[i * (points.len() - 1) / (limit - 1)])
        .collect()
}

fn locate(stations: &[f64], station: f64) -> (usize, f64) {
    if station <= 0.0 {
        return (0, 0.0);
    }
    if station >= *stations.last().unwrap() {
        return (stations.len() - 2, 1.0);
    }
    let edge = stations.partition_point(|s| *s < station).saturating_sub(1);
    (
        edge,
        (station - stations[edge]) / (stations[edge + 1] - stations[edge]),
    )
}

// Lanelet2 infers boundary direction by projecting each bound's middle vertex
// onto the other bound. Check that inference, as well as build_road's endpoint
// orientation, before storing a piece. A tight returning section may need an
// extra matched cut so that saving/reopening preserves travel direction.
fn unambiguous(reference: &[[f64; 3]], boundaries: &[Vec<[f64; 3]>], lanes: &[RoadLane]) -> bool {
    let reference = polyline(reference.to_vec());
    let boundaries: Vec<_> = boundaries.iter().cloned().map(polyline).collect();
    if !reference.is_valid() || boundaries.iter().any(|line| !line.is_valid()) {
        return false;
    }
    if boundaries.iter().any(|line| {
        match (
            reference.nearest_point(line.first().unwrap().xy()),
            reference.nearest_point(line.last().unwrap().xy()),
        ) {
            (Some(start), Some(end)) => end.station < start.station,
            _ => true,
        }
    }) {
        return false;
    }
    lanes.iter().enumerate().all(|(i, lane)| {
        let (left, right, reversed) = match lane.direction {
            LaneDirection::Forward => (&boundaries[i], &boundaries[i + 1], false),
            LaneDirection::Backward => (&boundaries[i + 1], &boundaries[i], true),
        };
        lanelet_alignment(left, right) == Some((reversed, reversed))
    })
}

fn lanelet_alignment(left: &Polyline3, right: &Polyline3) -> Option<(bool, bool)> {
    let middle = |line: &Polyline3| {
        if line.len() > 2 {
            line.points[line.len() / 2]
        } else {
            line.points[0].lerp(line.points[1], 0.5)
        }
    };
    let left_rev = left.nearest_point(middle(right).xy())?.lateral >= 0.0;
    let left_oriented = if left_rev {
        left.reversed()
    } else {
        left.clone()
    };
    let right_rev = right.nearest_point(middle(&left_oriented).xy())?.lateral <= 0.0;
    Some((left_rev, right_rev))
}

fn slice(line: &[[f64; 3]], first: (usize, f64), last: (usize, f64)) -> Vec<[f64; 3]> {
    let at = |(edge, fraction): (usize, f64)| {
        if fraction == 0.0 {
            return line[edge];
        }
        if fraction == 1.0 {
            return line[edge + 1];
        }
        std::array::from_fn(|axis| {
            line[edge][axis] + (line[edge + 1][axis] - line[edge][axis]) * fraction
        })
    };
    let mut result = vec![at(first)];
    for (i, point) in line.iter().enumerate().take(last.0 + 1).skip(first.0 + 1) {
        if i > first.0 + 1 || first.1 < 1.0 {
            result.push(*point);
        }
    }
    if last.1 > 0.0 {
        result.push(at(last));
    }
    // Repeated cross-sections add no geometry, but a zero-length edge can make
    // Lanelet2's nearest-point orientation test return a zero signed distance.
    result.dedup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_map::Evidence;

    fn road() -> ExtractedRoad {
        let reference = vec![[0., 0., 2.], [12., 0., 3.], [12., 20., 4.], [0., 20., 5.]];
        let boundaries = [1.75, -1.75, -5.25]
            .into_iter()
            .map(|offset| {
                polyline(reference.clone())
                    .offset(offset)
                    .points
                    .iter()
                    .map(|p| [p.x, p.y, p.z])
                    .collect()
            })
            .collect();
        ExtractedRoad {
            reference,
            boundaries,
            evidence: vec![vec![Evidence::WidthPrior; 4]; 3],
            source_boundaries: None,
        }
    }

    #[test]
    fn paired_cuts_keep_vertices_heights_shared_edges_and_opposite_chains() {
        let road = road();
        let lanes = [
            RoadLane::new(3.5, LaneDirection::Forward),
            RoadLane::new(3.5, LaneDirection::Backward),
        ];
        let mut map = Map::new();
        let built = build(
            &mut map,
            road.clone(),
            &lanes,
            &BuildOptions {
                segment_length: 8.,
                ..Default::default()
            },
        )
        .unwrap()
        .0;
        assert_eq!(
            built.lanes.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![6, 6]
        );
        for (column, chain) in built.lanes.iter().enumerate() {
            for pair in chain.windows(2) {
                assert_eq!(map.connection_gap(pair[0], pair[1]), Some(0.0));
                assert!(map.successors(pair[0]).contains(&pair[1]));
            }
            let ordered: Vec<_> = if column == 0 {
                chain.clone()
            } else {
                chain.iter().rev().copied().collect()
            };
            let mut geometry = Vec::new();
            for (k, id) in ordered.iter().enumerate() {
                let lane = map.lane(*id).unwrap();
                let boundary = if column == 0 {
                    lane.left.boundary
                } else {
                    lane.right.boundary
                };
                let points = &map.boundary(boundary).unwrap().geometry.points;
                geometry.extend(
                    points
                        .iter()
                        .skip(usize::from(k > 0))
                        .map(|p| [p.x, p.y, p.z]),
                );
            }
            for vertex in &road.boundaries[column] {
                assert!(
                    geometry.contains(vertex),
                    "missing {vertex:?} in {geometry:?}"
                );
            }
        }
        for (forward, backward) in built.lanes[0].iter().zip(built.lanes[1].iter().rev()) {
            assert_eq!(
                map.lane(*forward).unwrap().right.boundary,
                map.lane(*backward).unwrap().right.boundary
            );
        }
    }

    #[test]
    fn cuts_within_one_edge_interpolate_all_axes_without_duplicate_vertices() {
        let line = [[0., 2., 4.], [10., 6., 8.]];
        assert_eq!(
            slice(&line, (0, 0.2), (0, 0.6)),
            vec![[2., 2.8, 4.8], [6., 4.4, 6.4]]
        );
        assert_eq!(slice(&line, (0, 0.), (0, 1.)), line);
        let line = [[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]];
        assert_eq!(slice(&line, (0, 1.), (1, 1.)), line[1..]);
        assert_eq!(slice(&line, (0, 0.), (1, 0.)), line[..2]);
    }

    #[test]
    fn tiny_segments_fail_before_allocating_or_editing() {
        let mut map = Map::new();
        let before = map.clone();
        let lanes = [
            RoadLane::new(3.5, LaneDirection::Forward),
            RoadLane::new(3.5, LaneDirection::Backward),
        ];
        let error = build(
            &mut map,
            road(),
            &lanes,
            &BuildOptions {
                segment_length: 1e-12,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(error.0.contains("piece limit"));
        assert_eq!(map, before);
    }

    #[test]
    fn disabled_segmentation_preserves_the_native_unsplit_map() {
        let road = road();
        let lanes = [
            RoadLane::new(3.5, LaneDirection::Forward),
            RoadLane::new(3.5, LaneDirection::Backward),
        ];
        let mut expected = Map::new();
        expected
            .build_road(specification(
                road.reference.clone(),
                road.boundaries.clone(),
                &lanes,
                40.,
            ))
            .unwrap();
        let mut actual = Map::new();
        build(
            &mut actual,
            road,
            &lanes,
            &BuildOptions {
                segment_length: 0.,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn piece_count_uses_3d_distance_and_cuts_survive_duplicate_sections() {
        let mut road = road();
        road.reference = vec![[0., 0., 0.], [10., 0., 15.], [10., 0., 15.], [20., 0., 30.]];
        road.boundaries = [1.75, -1.75]
            .into_iter()
            .map(|y| road.reference.iter().map(|p| [p[0], y, p[2]]).collect())
            .collect();
        let lanes = [RoadLane::new(3.5, LaneDirection::Forward)];
        let mut map = Map::new();
        let built = build(
            &mut map,
            road,
            &lanes,
            &BuildOptions {
                segment_length: 18.,
                ..Default::default()
            },
        )
        .unwrap()
        .0;
        assert_eq!(built.lanes[0].len(), 2);
        assert_eq!(
            map.connection_gap(built.lanes[0][0], built.lanes[0][1]),
            Some(0.)
        );
        assert_eq!(
            map.boundaries().next().unwrap().geometry.last(),
            Some(Point3::new(10., 1.75, 15.))
        );
    }

    #[test]
    fn nclt_returning_section_rejects_a_folded_inner_lane_and_accepts_one_lane() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            reference: Vec<[f64; 3]>,
            boundaries: Vec<Vec<[f64; 3]>>,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/vector-map/nclt-returning-section.json"
        ))
        .unwrap();
        let road = ExtractedRoad {
            evidence: vec![
                vec![Evidence::WidthPrior; fixture.reference.len()];
                fixture.boundaries.len()
            ],
            reference: fixture.reference,
            boundaries: fixture.boundaries,
            source_boundaries: None,
        };
        let lanes = [
            RoadLane::new(3.5, LaneDirection::Forward),
            RoadLane::new(3.5, LaneDirection::Backward),
        ];
        assert!(!unambiguous(&road.reference, &road.boundaries, &lanes));
        let mut map = Map::new();
        let mut diagnostic = None;
        assert!(
            build_diagnostic(
                &mut map,
                road.clone(),
                &lanes,
                &BuildOptions::default(),
                Some(&mut diagnostic),
            )
            .unwrap_err()
            .0
            .contains("ambiguous travel directions")
        );
        let diagnostic = diagnostic.unwrap();
        assert_eq!(diagnostic.code, "ambiguous_boundary_direction");
        assert_eq!(diagnostic.reference, road.reference);
        assert_eq!(diagnostic.boundaries, road.boundaries);
        assert!(diagnostic.context_length > 40.);
        assert_eq!(
            (diagnostic.forward_lanes, diagnostic.backward_lanes),
            (1, 1)
        );
        assert_eq!(diagnostic.lane_width, 3.5);
        assert_eq!(diagnostic.segment_length, 50.);
        // Location is in the original survey frame, on the rejected reference.
        let marker = Point3::new(
            diagnostic.location[0],
            diagnostic.location[1],
            diagnostic.location[2],
        );
        assert!(
            polyline(road.reference.clone())
                .distance_to(marker.xy())
                .unwrap()
                < 1e-6
        );
        let mut map = Map::new();
        let mut road = road;
        road.boundaries.truncate(2);
        let (built, _) = build(&mut map, road, &lanes[..1], &BuildOptions::default()).unwrap();
        for chain in &built.lanes {
            for pair in chain.windows(2) {
                assert_eq!(map.connection_gap(pair[0], pair[1]), Some(0.));
            }
            for id in chain {
                let lane = map.lane(*id).unwrap();
                let left = &map.boundary(lane.left.boundary).unwrap().geometry;
                let right = &map.boundary(lane.right.boundary).unwrap().geometry;
                assert_eq!(
                    lanelet_alignment(left, right),
                    Some((lane.left.reversed, lane.right.reversed))
                );
            }
        }
    }

    #[test]
    fn diagnostic_preview_bounds_dense_survey_geometry_and_keeps_endpoints() {
        let points: Vec<_> = (0..100_000)
            .map(|i| [500_000. + i as f64 * 0.01, 4_000_000., 12.])
            .collect();
        let endpoints = (points[0], *points.last().unwrap());
        // The maximum sixteen-lane draft has seventeen boundaries plus a reference.
        let limit = 2048 / 18;
        let preview = sample(points, limit);
        assert!(preview.len() * 18 <= 2048);
        assert_eq!((preview[0], *preview.last().unwrap()), endpoints);
        assert!(preview.windows(2).all(|p| p[1][0] > p[0][0]));
    }
}
