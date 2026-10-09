//! Ground-supported connection drafts between open directed road ends.
//! Geometry cannot establish permitted turns. Branches are retained for review;
//! selected proposals are added atomically without changing existing lane geometry.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use vectormap_core::{LaneId, LaneKind, Map, NewConnector, Point3, Polyline3, Side};

use super::{BuildError, quantile};
use crate::PointCloud;

pub(super) const RADIUS: f64 = 0.75;
pub(super) const HEIGHT: f64 = 0.3;
pub(super) const LAYER_HEIGHT: f64 = 0.15;
pub(super) const LAYER_CELL: f64 = 0.2;
pub(super) const LAYER_MIN_AREA: f64 = 0.01;
// An endpoint rise alone cannot distinguish a graded street from another level.
// Retain the short-gap tolerance, then require a modest grade and observed ground
// at both ends and along the actual connector. This is not a legal road-grade test.
const MAX_GRADE: f64 = 0.12;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JunctionOptions {
    /// Maximum straight-line XY gap, in metres (1..100).
    pub max_gap: f64,
    /// Minimum fraction of connector centre samples supported by ground (0.5..1).
    pub min_ground_support: f64,
    /// Also require source support along both actual boundary curves and all ends.
    pub check_boundary_support: bool,
}

impl Default for JunctionOptions {
    fn default() -> Self {
        Self {
            max_gap: 30.0,
            min_ground_support: 0.9,
            check_boundary_support: false,
        }
    }
}

impl JunctionOptions {
    fn validate(&self) -> Result<(), BuildError> {
        if !self.max_gap.is_finite() || !(1.0..=100.0).contains(&self.max_gap) {
            return Err(BuildError(
                "max_gap must be finite and within 1..100 m".into(),
            ));
        }
        if !self.min_ground_support.is_finite() || !(0.5..=1.0).contains(&self.min_ground_support) {
            return Err(BuildError(
                "min_ground_support must be finite and within 0.5..1".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct JunctionCandidate {
    pub from: LaneId,
    pub to: LaneId,
    pub gap: f64,
    pub turn_degrees: f64,
    pub ground_support: f64,
    /// Left/right fractions checked with the source-audit protocol when enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boundary_support: Option<[f64; 2]>,
    pub samples: usize,
    /// Multiple supported choices share this incoming or outgoing road end.
    pub ambiguous: bool,
    pub center: Vec<[f64; 3]>,
    pub left: Vec<[f64; 3]>,
    pub right: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JunctionReport {
    pub candidates: Vec<JunctionCandidate>,
    pub added: Vec<LaneId>,
    pub open_exits: usize,
    pub open_entries: usize,
    pub malformed_lanes: usize,
    pub rejected_heading: usize,
    pub rejected_height: usize,
    pub unsupported_candidates: usize,
    pub rejected_geometry: usize,
    pub cloud_points: usize,
    pub traffic_rules_inferred: bool,
    pub warnings: Vec<String>,
}

struct End {
    id: LaneId,
    point: Point3,
    heading: [f64; 2],
}

fn end(line: &Polyline3, last: bool, id: LaneId) -> Option<End> {
    let pair = if last {
        line.points.last_chunk::<2>()?
    } else {
        line.points.first_chunk::<2>()?
    };
    let d = [pair[1].x - pair[0].x, pair[1].y - pair[0].y];
    let length = d[0].hypot(d[1]);
    if !length.is_finite() || length < 1e-6 || pair.iter().any(|p| !p.z.is_finite()) {
        return None;
    }
    Some(End {
        id,
        point: pair[usize::from(last)],
        heading: [d[0] / length, d[1] / length],
    })
}

fn coordinates(line: &Polyline3) -> Vec<[f64; 3]> {
    line.points.iter().map(|p| [p.x, p.y, p.z]).collect()
}

pub(super) struct Ground<'a> {
    cloud: &'a PointCloud,
    cells: HashMap<(i64, i64), Vec<usize>>,
    consensus: bool,
}

fn cell(x: f64, y: f64) -> (i64, i64) {
    ((x / RADIUS).floor() as i64, (y / RADIUS).floor() as i64)
}

impl<'a> Ground<'a> {
    pub(super) fn new(cloud: &'a PointCloud) -> Result<Self, BuildError> {
        if cloud.positions.is_empty() {
            return Err(BuildError(
                "choose a nonempty point cloud in the map's metre frame".into(),
            ));
        }
        let mut cells: HashMap<_, Vec<_>> = HashMap::new();
        for (i, p) in cloud.positions.iter().enumerate() {
            if p.iter().any(|v| !v.is_finite() || v.abs() > 1e15) {
                return Err(BuildError(
                    "point cloud contains invalid coordinates".into(),
                ));
            }
            cells.entry(cell(p[0], p[1])).or_default().push(i);
        }
        Ok(Self {
            cloud,
            cells,
            consensus: false,
        })
    }

    pub(super) fn new_consensus(cloud: &'a PointCloud) -> Result<Self, BuildError> {
        let mut ground = Self::new(cloud)?;
        ground.consensus = true;
        Ok(ground)
    }

    pub(super) fn height(&self, p: Point3) -> Option<f64> {
        if [p.x, p.y, p.z]
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 1e15)
        {
            return None;
        }
        let (x, y) = cell(p.x, p.y);
        let mut points = Vec::new();
        for a in x - 1..=x + 1 {
            for b in y - 1..=y + 1 {
                for &i in self.cells.get(&(a, b)).into_iter().flatten() {
                    let q = self.cloud.positions[i];
                    if (q[0] - p.x).powi(2) + (q[1] - p.y).powi(2) <= RADIUS * RADIUS {
                        points.push(q);
                    }
                }
            }
        }
        if points.len() < 3 {
            return None;
        }
        if self.consensus {
            lowest_layer(&mut points)
        } else {
            quantile(&mut points.iter().map(|p| p[2]).collect::<Vec<_>>(), 0.15)
        }
    }

    pub(super) fn supports(&self, p: Point3) -> bool {
        self.height(p).is_some_and(|z| (z - p.z).abs() <= HEIGHT)
    }
}

/// One vote per occupied XY cell prevents vertical return density from lifting
/// a low layer. Three non-collinear cells are required: a vertical wall or an
/// isolated low return cannot supply surface support. The lowest such layer can
/// still be the wrong physical level; this is not a semantic ground classifier.
pub(super) fn lowest_layer(points: &mut [[f64; 3]]) -> Option<f64> {
    points.sort_unstable_by(|a, b| a[2].total_cmp(&b[2]));
    let xy = |p: [f64; 3]| {
        (
            (p[0] / LAYER_CELL).floor() as i64,
            (p[1] / LAYER_CELL).floor() as i64,
        )
    };
    let mut occupied: BTreeMap<_, BTreeSet<usize>> = BTreeMap::new();
    let mut end = 0;
    for start in 0..points.len() {
        while end < points.len() && points[end][2] - points[start][2] <= LAYER_HEIGHT {
            occupied.entry(xy(points[end])).or_default().insert(end);
            end += 1;
        }
        if occupied.len() >= 3 {
            // Lowest representative of each cell in this window: duplicates in
            // one column do not change either its vote or horizontal footprint.
            let representatives: Vec<_> = occupied
                .values()
                .map(|ids| points[*ids.first().unwrap()])
                .collect();
            let axis = if spread(&representatives, 0) >= spread(&representatives, 1) {
                0
            } else {
                1
            };
            let a = representatives
                .iter()
                .min_by(|a, b| a[axis].total_cmp(&b[axis]))
                .unwrap();
            let b = representatives
                .iter()
                .max_by(|a, b| a[axis].total_cmp(&b[axis]))
                .unwrap();
            let area = representatives
                .iter()
                .map(|p| {
                    ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])).abs() * 0.5
                })
                .fold(0.0, f64::max);
            if area >= LAYER_MIN_AREA {
                return quantile(
                    &mut representatives.iter().map(|p| p[2]).collect::<Vec<_>>(),
                    0.5,
                );
            }
        }
        let key = xy(points[start]);
        let ids = occupied.get_mut(&key).unwrap();
        ids.remove(&start);
        if ids.is_empty() {
            occupied.remove(&key);
        }
    }
    None
}

fn spread(points: &[[f64; 3]], axis: usize) -> f64 {
    points
        .iter()
        .map(|p| p[axis])
        .fold(f64::NEG_INFINITY, f64::max)
        - points.iter().map(|p| p[axis]).fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod layer_tests {
    use super::*;

    #[test]
    fn low_surface_survives_dense_overhead_returns_and_isolated_outlier() {
        let mut cloud = PointCloud::default();
        cloud.positions.extend([
            [0.2, 0.2, 1.0],
            [-0.2, 0.2, 1.04],
            [0.2, -0.2, 1.02],
            [0., 0., -5.],
        ]);
        for i in 0..1000 {
            cloud.positions.push([0.2, 0.2, 3. + i as f64 * 0.001]);
        }
        let p = Point3::new(0., 0., 1.);
        assert!(Ground::new(&cloud).unwrap().height(p).unwrap() > 3.);
        assert_eq!(Ground::new_consensus(&cloud).unwrap().height(p), Some(1.02));
    }

    #[test]
    fn duplicates_vertical_walls_and_remote_returns_do_not_supply_a_surface() {
        let mut cloud = PointCloud::default();
        for x in -3..=3 {
            for z in 0..20 {
                cloud.positions.push([x as f64 * 0.2, 0., z as f64 * 0.01]);
            }
        }
        cloud.positions.push([2., 2., 0.]);
        let p = Point3::new(0., 0., 0.);
        assert!(Ground::new(&cloud).unwrap().height(p).is_some());
        assert_eq!(Ground::new_consensus(&cloud).unwrap().height(p), None);
        cloud.positions = vec![[0., 0., 0.]; 100];
        assert_eq!(Ground::new_consensus(&cloud).unwrap().height(p), None);
    }

    #[test]
    fn modest_grade_uses_cell_votes_and_a_lower_level_remains_ambiguous() {
        let mut cloud = PointCloud::default();
        for x in -2..=2 {
            for y in -2..=2 {
                cloud
                    .positions
                    .push([x as f64 * 0.2, y as f64 * 0.2, 2. + x as f64 * 0.02]);
            }
        }
        let p = Point3::new(0., 0., 2.);
        let height = Ground::new_consensus(&cloud).unwrap().height(p).unwrap();
        assert!((height - 2.).abs() < 0.05);
        // A coherent lower level must not be silently treated as the upper road.
        cloud
            .positions
            .extend([[0.2, 0.2, -1.], [-0.2, 0.2, -1.], [0.2, -0.2, -1.]]);
        let ground = Ground::new_consensus(&cloud).unwrap();
        assert_eq!(ground.height(p), Some(-1.));
        assert!(!ground.supports(p));
    }
}

/// Preview all ground-supported geometric choices, including branching junctions.
/// Only open driving lanes without turn labels participate. Existing connections
/// are authoritative. Neither topology nor permitted turns are inferred from a reference.
pub fn propose(
    map: &Map,
    cloud: &PointCloud,
    o: &JunctionOptions,
) -> Result<JunctionReport, BuildError> {
    o.validate()?;
    let ground = Ground::new(cloud)?;
    let mut report = JunctionReport {
        candidates: Vec::new(), added: Vec::new(), open_exits: 0, open_entries: 0,
        malformed_lanes: 0, rejected_heading: 0, rejected_height: 0,
        unsupported_candidates: 0, rejected_geometry: 0, cloud_points: cloud.len(),
        traffic_rules_inferred: false,
        warnings: vec!["Connection drafts follow geometry and nearby ground, not permitted turns. Review every branch, traffic rule and ground level before using the map. Existing rules are retained; no new right-of-way or signal rules are created.".into()],
    };
    let mut exits = Vec::new();
    let mut entries = Vec::new();
    for lane in map
        .lanes()
        .filter(|l| l.kind == LaneKind::Driving && l.turn_direction.is_none())
    {
        let Some(line) = map.centerline(lane.id) else {
            report.malformed_lanes += 1;
            continue;
        };
        let (Some(a), Some(b)) = (end(&line, false, lane.id), end(&line, true, lane.id)) else {
            report.malformed_lanes += 1;
            continue;
        };
        if map.predecessors(lane.id).is_empty() {
            entries.push(a);
        }
        if map.successors(lane.id).is_empty() {
            exits.push(b);
        }
    }
    report.open_entries = entries.len();
    report.open_exits = exits.len();
    for a in &exits {
        for b in &entries {
            let delta = [b.point.x - a.point.x, b.point.y - a.point.y];
            let gap = delta[0].hypot(delta[1]);
            if a.id == b.id || !(0.5..=o.max_gap).contains(&gap) {
                continue;
            }
            let rise = (a.point.z - b.point.z).abs();
            if rise > HEIGHT && rise / gap > MAX_GRADE {
                report.rejected_height += 1;
                continue;
            }
            let toward = |h: [f64; 2]| (h[0] * delta[0] + h[1] * delta[1]) / gap;
            let cross = a.heading[0] * b.heading[1] - a.heading[1] * b.heading[0];
            let dot = a.heading[0] * b.heading[0] + a.heading[1] * b.heading[1];
            let angle = cross.atan2(dot).to_degrees();
            if toward(a.heading) < 0.25 || toward(b.heading) < 0.25 || angle.abs() > 135.0 {
                report.rejected_heading += 1;
                continue;
            }
            // A supported interior alone can mask a floating end at the preview's
            // fractional support threshold. Never relax the endpoint ground test.
            if !ground.supports(a.point) || !ground.supports(b.point) {
                report.unsupported_candidates += 1;
                continue;
            }
            // Score the exact geometry produced by the shared connector builder,
            // rather than an independently approximated centre curve.
            let mut preview = map.clone();
            let Ok((id, _)) = preview.add_connector(NewConnector::new(a.id, b.id)) else {
                report.rejected_geometry += 1;
                continue;
            };
            let (Some(center), Some(left), Some(right)) = (
                preview.centerline(id),
                preview.oriented_boundary(id, Side::Left),
                preview.oriented_boundary(id, Side::Right),
            ) else {
                report.rejected_geometry += 1;
                continue;
            };
            let (support, samples, boundary_support) = if o.check_boundary_support {
                let [Some(c), Some(l), Some(r)] = [&center, &left, &right]
                    .map(|line| super::quality::checked_curve_support(&ground, line))
                else {
                    report.unsupported_candidates += 1;
                    continue;
                };
                if [&c, &l, &r].iter().any(|s| {
                    !s.start_supported || !s.end_supported || s.fraction < o.min_ground_support
                }) {
                    report.unsupported_candidates += 1;
                    continue;
                }
                (c.fraction, c.samples, Some([l.fraction, r.fraction]))
            } else {
                let samples = center.resample(0.5);
                let support = samples
                    .points
                    .iter()
                    .filter(|&&p| ground.supports(p))
                    .count() as f64
                    / samples.points.len().max(1) as f64;
                (support, samples.points.len(), None)
            };
            if support < o.min_ground_support {
                report.unsupported_candidates += 1;
                continue;
            }
            report.candidates.push(JunctionCandidate {
                from: a.id,
                to: b.id,
                gap,
                turn_degrees: angle,
                ground_support: support,
                boundary_support,
                samples,
                ambiguous: false,
                center: coordinates(&center),
                left: coordinates(&left),
                right: coordinates(&right),
            });
        }
    }
    let mut outgoing = BTreeMap::<LaneId, usize>::new();
    let mut incoming = BTreeMap::<LaneId, usize>::new();
    for c in &report.candidates {
        *outgoing.entry(c.from).or_default() += 1;
        *incoming.entry(c.to).or_default() += 1;
    }
    for c in &mut report.candidates {
        c.ambiguous = outgoing[&c.from] > 1 || incoming[&c.to] > 1;
    }
    if report.candidates.iter().any(|c| c.ambiguous) {
        report.warnings.push("Multiple supported branches share road ends. Geometry alone cannot decide which manoeuvres are allowed; preview and select the intended connections.".into());
    }
    Ok(report)
}

/// Atomically add selected candidates, or all proposals when `pairs` is absent.
/// Every selected pair must still be supported in the current map/cloud. This
/// protects against stale previews. An empty selection is a no-op.
pub fn connect(
    map: &mut Map,
    cloud: &PointCloud,
    o: &JunctionOptions,
    pairs: Option<&[[u64; 2]]>,
) -> Result<JunctionReport, BuildError> {
    let mut report = propose(map, cloud, o)?;
    let supported: BTreeSet<_> = report
        .candidates
        .iter()
        .map(|c| [c.from.0, c.to.0])
        .collect();
    let selected: BTreeSet<_> =
        pairs.map_or_else(|| supported.clone(), |p| p.iter().copied().collect());
    if !selected.is_subset(&supported) {
        return Err(BuildError(
            "selected connection is not supported by the current map and cloud; preview again"
                .into(),
        ));
    }
    let mut draft = map.clone();
    for [from, to] in selected {
        let (id, _) = draft
            .add_connector(NewConnector::new(LaneId(from), LaneId(to)))
            .map_err(|e| BuildError(e.to_string()))?;
        let lane = draft.lane_mut(id).expect("connector was created");
        lane.attributes.insert(
            "cloudanalyzer_geometry_source",
            "ground_supported_connection",
        );
        lane.attributes
            .insert("cloudanalyzer_review_required", "yes");
        report.added.push(id);
    }
    *map = draft;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectormap_core::{LaneDirection, NewRoad, RoadLane, SpeedLimit};

    fn road(map: &mut Map, a: [f64; 3], b: [f64; 3]) -> LaneId {
        let points = [a, b].map(|p| Point3::new(p[0], p[1], p[2]));
        let spec = NewRoad::new(
            Polyline3::new(points.to_vec()),
            vec![RoadLane::new(3.5, LaneDirection::Forward)],
        );
        map.build_road(spec).unwrap().0.lanes[0][0]
    }

    fn fixture() -> (Map, PointCloud, [LaneId; 3]) {
        let mut map = Map::new();
        let a = road(&mut map, [-20.0, 0.0, 2.0], [-10.0, 0.0, 2.0]);
        let b = road(&mut map, [0.0, 10.0, 2.0], [0.0, 20.0, 2.0]);
        let c = road(&mut map, [0.0, -10.0, 2.0], [0.0, -20.0, 2.0]);
        map.set_speed_limit(&[a], Some(SpeedLimit::from_kmh(20.0)))
            .unwrap();
        let mut cloud = PointCloud::default();
        for x in -110..=10 {
            for y in -110..=110 {
                cloud.positions.push([x as f64 * 0.2, y as f64 * 0.2, 2.0]);
            }
        }
        (map, cloud, [a, b, c])
    }

    #[test]
    fn supported_centre_does_not_certify_missing_boundaries_and_rejection_is_atomic() {
        let mut map = Map::new();
        let a = road(&mut map, [-20.0, 0.0, 2.0], [-10.0, 0.0, 2.0]);
        let b = road(&mut map, [0.0, 0.0, 2.0], [10.0, 0.0, 2.0]);
        let mut cloud = PointCloud::default();
        for x in -105..=55 {
            for y in -1..=1 {
                cloud.positions.push([x as f64 * 0.2, y as f64 * 0.2, 2.0]);
            }
        }
        let before = map.clone();
        let legacy = propose(&map, &cloud, &Default::default()).unwrap();
        assert_eq!(legacy.candidates.len(), 1);
        assert_eq!(legacy.candidates[0].boundary_support, None);
        let options = JunctionOptions {
            check_boundary_support: true,
            min_ground_support: 1.0,
            ..Default::default()
        };
        let checked = propose(&map, &cloud, &options).unwrap();
        assert!(checked.candidates.is_empty());
        assert!(connect(&mut map, &cloud, &options, Some(&[[a.0, b.0]])).is_err());
        assert_eq!(map, before);
        for x in -105..=55 {
            for y in -10..=10 {
                cloud.positions.push([x as f64 * 0.2, y as f64 * 0.2, 2.0]);
            }
        }
        let checked = propose(&map, &cloud, &options).unwrap();
        assert_eq!(checked.candidates.len(), 1);
        assert_eq!(checked.candidates[0].boundary_support, Some([1.0, 1.0]));
        let added = connect(&mut map, &cloud, &options, Some(&[[a.0, b.0]])).unwrap();
        let quality = super::super::quality::audit(&map, &cloud).unwrap();
        let lane = quality
            .lanes
            .iter()
            .find(|l| l.lane == added.added[0])
            .unwrap();
        assert_eq!(
            [
                lane.center.fraction,
                lane.left.fraction,
                lane.right.fraction
            ],
            [1.0; 3]
        );
    }

    #[test]
    fn boundary_interior_gap_is_rejected_even_when_all_endpoints_have_ground() {
        let mut map = Map::new();
        road(&mut map, [-20.0, 0.0, 2.0], [-10.0, 0.0, 2.0]);
        road(&mut map, [0.0, 0.0, 2.0], [10.0, 0.0, 2.0]);
        let mut cloud = PointCloud::default();
        for x in -105..=55 {
            for y in -15..=15 {
                let (x, y) = (x as f64 * 0.2, y as f64 * 0.2);
                if !((-8.0..=-2.0).contains(&x) && y > 0.9) {
                    cloud.positions.push([x, y, 2.0]);
                }
            }
        }
        let legacy = propose(&map, &cloud, &Default::default()).unwrap();
        assert_eq!(legacy.candidates.len(), 1);
        let candidate = &legacy.candidates[0];
        let ground = Ground::new(&cloud).unwrap();
        assert!(
            candidate
                .left
                .first()
                .into_iter()
                .chain(candidate.left.last())
                .all(|p| ground.supports(Point3::new(p[0], p[1], p[2])))
        );
        let options = JunctionOptions {
            check_boundary_support: true,
            min_ground_support: 1.0,
            ..Default::default()
        };
        assert!(
            propose(&map, &cloud, &options)
                .unwrap()
                .candidates
                .is_empty()
        );
    }

    #[test]
    fn branching_drafts_are_previewed_read_only_and_added_together_without_moving_lanes() {
        let (mut map, cloud, [a, b, c]) = fixture();
        let before = map.clone();
        let proposal = propose(&map, &cloud, &Default::default()).unwrap();
        assert_eq!(map, before);
        assert_eq!(proposal.candidates.len(), 2, "{proposal:?}");
        assert!(proposal.candidates.iter().all(|p| p.ambiguous));
        let report = connect(&mut map, &cloud, &Default::default(), None).unwrap();
        assert_eq!(report.added.len(), 2);
        assert_eq!(map.successors(a).len(), 2);
        assert_eq!(map.predecessors(b).len(), 1);
        assert_eq!(map.predecessors(c).len(), 1);
        for lane in before.lanes() {
            assert_eq!(map.lane(lane.id), Some(lane));
        }
        for boundary in before.boundaries() {
            assert_eq!(map.boundary(boundary.id), Some(boundary));
        }
        assert_eq!(map.metadata(), before.metadata());
        for id in report.added {
            let lane = map.lane(id).unwrap();
            assert_eq!(
                lane.attributes.get("cloudanalyzer_review_required"),
                Some("yes")
            );
            assert_eq!(lane.speed_limit.unwrap().kmh, 20.0);
            assert_eq!(map.predecessors(id), &[a]);
            assert!(matches!(map.successors(id), [to] if *to == b || *to == c));
        }
        let connected = map.clone();
        assert!(
            connect(&mut map, &cloud, &Default::default(), None)
                .unwrap()
                .added
                .is_empty()
        );
        assert_eq!(map, connected);
    }

    #[test]
    fn selection_can_exclude_branches_and_stale_or_invalid_selections_are_atomic() {
        let (mut map, cloud, [a, b, c]) = fixture();
        let before = map.clone();
        assert!(
            connect(
                &mut map,
                &cloud,
                &Default::default(),
                Some(&[[a.0, b.0], [b.0, a.0]])
            )
            .is_err()
        );
        assert_eq!(map, before);
        assert!(
            connect(&mut map, &cloud, &Default::default(), Some(&[]))
                .unwrap()
                .added
                .is_empty()
        );
        assert_eq!(map, before);
        let report = connect(
            &mut map,
            &cloud,
            &Default::default(),
            Some(&[[a.0, c.0], [a.0, c.0]]),
        )
        .unwrap();
        assert_eq!(report.added.len(), 1);
        assert!(map.predecessors(b).is_empty());
        let after = map.clone();
        assert!(connect(&mut map, &cloud, &Default::default(), Some(&[[a.0, b.0]])).is_err());
        assert_eq!(map, after);
    }

    #[test]
    fn unsupported_ground_other_levels_and_invalid_options_never_create_connections() {
        let (map, mut cloud, _) = fixture();
        for p in &mut cloud.positions {
            p[2] += 3.0;
        }
        let report = propose(&map, &cloud, &Default::default()).unwrap();
        assert!(report.candidates.is_empty());
        assert_eq!(report.unsupported_candidates, 2);
        let invalid = JunctionOptions {
            max_gap: f64::NAN,
            ..Default::default()
        };
        assert!(propose(&map, &cloud, &invalid).is_err());
        assert!(propose(&map, &PointCloud::default(), &Default::default()).is_err());
    }

    #[test]
    fn graded_street_ends_connect_only_when_both_ends_and_the_gap_have_ground() {
        let mut map = Map::new();
        let a = road(&mut map, [-20.0, 0.0, 0.8], [-10.0, 0.0, 1.4]);
        let b = road(&mut map, [0.0, 0.0, 2.0], [10.0, 0.0, 2.6]);
        let before = map.clone();
        let mut cloud = PointCloud::default();
        for x in -105..=55 {
            for y in -20..=20 {
                let x = x as f64 * 0.2;
                cloud.positions.push([x, y as f64 * 0.2, 2.0 + x * 0.06]);
            }
        }
        let preview = propose(&map, &cloud, &Default::default()).unwrap();
        assert_eq!(map, before);
        assert_eq!(preview.candidates.len(), 1, "{preview:?}");
        assert_eq!(
            (preview.candidates[0].from, preview.candidates[0].to),
            (a, b)
        );
        assert_eq!(preview.candidates[0].ground_support, 1.0);
        let connected = connect(&mut map, &cloud, &Default::default(), None).unwrap();
        assert_eq!(connected.added.len(), 1);
        assert_eq!(map.successors(a), connected.added);
        for boundary in before.boundaries() {
            assert_eq!(map.boundary(boundary.id), Some(boundary));
        }
        // A plausible endpoint rise must not bridge a void in the actual street.
        cloud.positions.retain(|p| p[0] < -8.0 || p[0] > -2.0);
        let unsupported = propose(&before, &cloud, &Default::default()).unwrap();
        assert!(unsupported.candidates.is_empty());
        assert_eq!(unsupported.unsupported_candidates, 1);
    }

    #[test]
    fn floating_end_and_excessive_grade_are_rejected_even_with_supported_interior() {
        let mut map = Map::new();
        road(&mut map, [-20.0, 0.0, 2.0], [-10.0, 0.0, 2.0]);
        road(&mut map, [0.0, 0.0, 2.35], [10.0, 0.0, 2.35]);
        let mut cloud = PointCloud::default();
        for x in -105..=55 {
            for y in -20..=20 {
                cloud.positions.push([x as f64 * 0.2, y as f64 * 0.2, 2.0]);
            }
        }
        let report = propose(&map, &cloud, &Default::default()).unwrap();
        assert!(report.candidates.is_empty());
        assert_eq!(report.unsupported_candidates, 1);
        let mut steep = Map::new();
        road(&mut steep, [-20.0, 0.0, 0.0], [-10.0, 0.0, 2.0]);
        road(&mut steep, [0.0, 0.0, 4.0], [10.0, 0.0, 6.0]);
        for p in &mut cloud.positions {
            p[2] = 4.0 + p[0] * 0.2;
        }
        let report = propose(&steep, &cloud, &Default::default()).unwrap();
        assert!(report.candidates.is_empty());
        assert!(report.rejected_height > 0);
    }

    #[test]
    fn elevated_or_wrong_way_road_ends_are_not_bridged_by_flat_ground() {
        let (_, cloud, _) = fixture();
        let mut elevated = Map::new();
        road(&mut elevated, [-20.0, 0.0, 2.0], [-10.0, 0.0, 2.0]);
        road(&mut elevated, [0.0, 10.0, 5.0], [0.0, 20.0, 5.0]);
        let report = propose(&elevated, &cloud, &Default::default()).unwrap();
        assert!(report.candidates.is_empty());
        assert!(report.rejected_height > 0);
        let mut reversed = Map::new();
        road(&mut reversed, [-20.0, 0.0, 2.0], [-10.0, 0.0, 2.0]);
        road(&mut reversed, [0.0, 20.0, 2.0], [0.0, 10.0, 2.0]);
        let report = propose(&reversed, &cloud, &Default::default()).unwrap();
        assert!(report.candidates.is_empty());
        assert!(report.rejected_heading > 0);
    }
}
