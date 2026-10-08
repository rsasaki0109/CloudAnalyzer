//! Read-only source coverage audit. Ground-like returns cannot establish road
//! semantics, obstacle clearance, surveyed accuracy or permitted manoeuvres.
use serde::Serialize;
use vectormap_core::{LaneId, LaneKind, Map, Point3, Polyline3, Side};

use super::{
    BuildError,
    junctions::{Ground, HEIGHT, RADIUS},
};
use crate::PointCloud;

const SPACING: f64 = 0.5;
const MIN_SUPPORT: f64 = 0.9;
const MAX_SAMPLES: usize = 100_000;
const MAX_PROBLEM_POINTS: usize = 4_096;
const MAX_PROBLEM_INTERVALS: usize = 256;

#[derive(Debug, Serialize)]
pub struct CurveSupport {
    pub samples: usize,
    pub supported: usize,
    pub insufficient_returns: usize,
    pub height_mismatches: usize,
    pub fraction: f64,
    pub start_supported: bool,
    pub end_supported: bool,
}

#[derive(Debug, Serialize)]
pub struct LaneQuality {
    pub lane: LaneId,
    pub center: CurveSupport,
    pub left: CurveSupport,
    pub right: CurveSupport,
    pub needs_review: bool,
}

#[derive(Debug, Serialize)]
pub struct QualityReport {
    pub lanes: Vec<LaneQuality>,
    pub low_support_lanes: Vec<LaneId>,
    pub omitted_lanes: Vec<LaneId>,
    pub malformed_lanes: Vec<LaneId>,
    pub sampled_points: usize,
    pub cloud_points: usize,
    pub sampling_step_m: f64,
    pub ground_radius_m: f64,
    pub ground_height_tolerance_m: f64,
    pub minimum_support_fraction: f64,
    pub sample_budget: usize,
    pub limited: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceCurve {
    Center,
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportProblem {
    InsufficientReturns,
    HeightMismatch,
}

/// Consecutive failed samples on one oriented lane curve, in the survey frame.
/// A single failed sample has equal from/to stations and is shown as a point.
#[derive(Debug, Serialize)]
pub struct ProblemInterval {
    pub lane: LaneId,
    pub curve: SourceCurve,
    pub reason: SupportProblem,
    pub from_m: f64,
    pub to_m: f64,
    pub points: Vec<[f64; 3]>,
    /// One local 15th-percentile source height per point; None means fewer than
    /// three returns. These are audit observations, not a verified road level.
    pub source_heights_m: Vec<Option<f64>>,
}

#[derive(Debug, Serialize)]
pub struct QualityDetails {
    #[serde(flatten)]
    pub report: QualityReport,
    pub problems: Vec<ProblemInterval>,
    /// Location preview limits are independent of the audit sampling budget.
    pub problems_limited: bool,
}

#[derive(Default)]
struct ProblemLocations {
    problems: Vec<ProblemInterval>,
    vertices: usize,
    last_sample: Option<usize>,
    limited: bool,
}

impl ProblemLocations {
    fn sample(
        &mut self,
        lane: LaneId,
        curve: SourceCurve,
        index: usize,
        station: f64,
        point: Point3,
        observation: (Option<SupportProblem>, Option<f64>),
    ) {
        let (reason, source_height) = observation;
        let Some(reason) = reason else {
            self.last_sample = None;
            return;
        };
        let continuing = self
            .problems
            .last()
            .is_some_and(|p| p.lane == lane && p.curve == curve && p.reason == reason)
            && self.last_sample.is_some_and(|last| last + 1 == index);
        if self.vertices == MAX_PROBLEM_POINTS
            || (!continuing && self.problems.len() == MAX_PROBLEM_INTERVALS)
        {
            self.limited = true;
            self.last_sample = None;
            return;
        }
        if continuing {
            let p = self.problems.last_mut().unwrap();
            p.to_m = station;
            p.points.push([point.x, point.y, point.z]);
            p.source_heights_m.push(source_height);
        } else {
            self.problems.push(ProblemInterval {
                lane,
                curve,
                reason,
                from_m: station,
                to_m: station,
                points: vec![[point.x, point.y, point.z]],
                source_heights_m: vec![source_height],
            });
        }
        self.vertices += 1;
        self.last_sample = Some(index);
    }
}

fn sample_count(line: &Polyline3) -> Option<usize> {
    if line.points.len() < 2
        || line.points.iter().any(|p| {
            [p.x, p.y, p.z]
                .iter()
                .any(|x| !x.is_finite() || x.abs() > 1e15)
        })
    {
        return None;
    }
    let length = line.length();
    if !length.is_finite() || length <= 1e-9 {
        return None;
    }
    // Check the count before asking the geometry library to allocate samples.
    let count = (length / SPACING).ceil() + 1.0;
    Some(count.min((MAX_SAMPLES + 1) as f64) as usize)
}

fn curve_support(
    ground: &Ground<'_>,
    line: &Polyline3,
    count: usize,
    mut on_sample: impl FnMut(usize, Point3, Option<SupportProblem>, Option<f64>),
) -> CurveSupport {
    let points = line.resample_count(count);
    let mut supported = 0;
    let mut insufficient_returns = 0;
    let mut height_mismatches = 0;
    for (i, &p) in points.points.iter().enumerate() {
        let source_height = ground.height(p);
        let reason = match source_height {
            None => {
                insufficient_returns += 1;
                Some(SupportProblem::InsufficientReturns)
            }
            Some(z) if (z - p.z).abs() > HEIGHT => {
                height_mismatches += 1;
                Some(SupportProblem::HeightMismatch)
            }
            Some(_) => {
                supported += 1;
                None
            }
        };
        on_sample(i, p, reason, source_height);
    }
    CurveSupport {
        samples: points.points.len(),
        supported,
        insufficient_returns,
        height_mismatches,
        fraction: supported as f64 / points.points.len() as f64,
        start_supported: ground.supports(points.points[0]),
        end_supported: ground.supports(*points.points.last().unwrap()),
    }
}

/// Use the audit's endpoint-inclusive spacing for junction support as well.
/// Reject over-budget curves rather than allocating unbounded samples.
pub(super) fn checked_curve_support(ground: &Ground<'_>, line: &Polyline3) -> Option<CurveSupport> {
    let count = sample_count(line)?;
    (count <= MAX_SAMPLES).then(|| curve_support(ground, line, count, |_, _, _, _| {}))
}

/// Check every driving lane's centre and both boundaries in the chosen source
/// frame. Sparse/occluded support is an uncertainty, not proof of a wrong road.
/// No geometry, topology, attributes or Undo state are changed.
pub fn audit(map: &Map, cloud: &PointCloud) -> Result<QualityReport, BuildError> {
    audit_inner(map, cloud, None)
}

/// Read-only audit plus a bounded preview; summary decisions are identical to audit().
pub fn audit_with_locations(map: &Map, cloud: &PointCloud) -> Result<QualityDetails, BuildError> {
    let mut locations = ProblemLocations::default();
    let report = audit_inner(map, cloud, Some(&mut locations))?;
    Ok(QualityDetails {
        report,
        problems: locations.problems,
        problems_limited: locations.limited,
    })
}

fn audit_inner(
    map: &Map,
    cloud: &PointCloud,
    mut locations: Option<&mut ProblemLocations>,
) -> Result<QualityReport, BuildError> {
    let ground = Ground::new(cloud)?;
    let mut report = QualityReport {
        lanes: vec![], low_support_lanes: vec![], omitted_lanes: vec![], malformed_lanes: vec![],
        sampled_points: 0, cloud_points: cloud.len(), sampling_step_m: SPACING,
        ground_radius_m: RADIUS, ground_height_tolerance_m: HEIGHT,
        minimum_support_fraction: MIN_SUPPORT, sample_budget: MAX_SAMPLES, limited: false,
        warnings: vec!["Source coverage is separate from structural validation. A sample needs at least three returns within 0.75 m and a 15th-percentile height within 0.3 m. Missing/occluded returns and another ground level can reduce support. Even full support does not certify road semantics, survey accuracy, obstacles or traffic rules.".into()],
    };
    for lane in map.lanes().filter(|l| l.kind == LaneKind::Driving) {
        let lines = [
            map.centerline(lane.id),
            map.oriented_boundary(lane.id, Side::Left),
            map.oriented_boundary(lane.id, Side::Right),
        ];
        let [Some(center), Some(left), Some(right)] = lines else {
            report.malformed_lanes.push(lane.id);
            continue;
        };
        let counts = [&center, &left, &right].map(sample_count);
        let [Some(c), Some(l), Some(r)] = counts else {
            report.malformed_lanes.push(lane.id);
            continue;
        };
        if report.sampled_points + c + l + r > MAX_SAMPLES {
            report.omitted_lanes.push(lane.id);
            report.limited = true;
            continue;
        }
        let mut check = |line: &Polyline3, count, curve| {
            let step = line.length() / (count - 1) as f64;
            curve_support(&ground, line, count, |i, p, reason, source_height| {
                if let Some(locations) = locations.as_deref_mut() {
                    locations.sample(
                        lane.id,
                        curve,
                        i,
                        i as f64 * step,
                        p,
                        (reason, source_height),
                    );
                }
            })
        };
        let center = check(&center, c, SourceCurve::Center);
        let left = check(&left, l, SourceCurve::Left);
        let right = check(&right, r, SourceCurve::Right);
        report.sampled_points += c + l + r;
        let needs_review = [&center, &left, &right]
            .iter()
            .any(|v| v.fraction < MIN_SUPPORT || !v.start_supported || !v.end_supported);
        if needs_review {
            report.low_support_lanes.push(lane.id);
        }
        report.lanes.push(LaneQuality {
            lane: lane.id,
            center,
            left,
            right,
            needs_review,
        });
    }
    if report.limited {
        report
            .warnings
            .push("Sampling budget reached; omitted lanes have not been checked.".into());
    }
    Ok(report)
}

/// The operator's selected lanes are authoritative for this geometric check;
/// the proposal's nearest-lane suggestion may refer to a different branch.
pub(super) fn transverse_stop(map: &Map, geometry: &[[f64; 3]], lane: LaneId) -> bool {
    let (Some(a), Some(b)) = (geometry.first(), geometry.last()) else {
        return false;
    };
    let v = [b[0] - a[0], b[1] - a[1]];
    let length = v[0].hypot(v[1]);
    if length < 1e-6 {
        return false;
    }
    let Some(center) = map.centerline(lane) else {
        return false;
    };
    let midpoint = Point3::new(
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    );
    let Some(nearest) = center.nearest_point(midpoint.xy()) else {
        return false;
    };
    let Some(pair) = center.points.get(nearest.segment..nearest.segment + 2) else {
        return false;
    };
    let h = [pair[1].x - pair[0].x, pair[1].y - pair[0].y];
    let norm = h[0].hypot(h[1]);
    norm > 1e-6
        && ((v[0] * h[0] + v[1] * h[1]) / (length * norm)).abs() <= std::f64::consts::FRAC_1_SQRT_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectormap_core::{LaneDirection, NewRoad, RoadLane};
    fn scene() -> (Map, PointCloud) {
        let mut map = Map::new();
        map.build_road(NewRoad::new(
            Polyline3::new(vec![Point3::new(0., 0., 2.), Point3::new(10., 0., 2.)]),
            vec![RoadLane::new(3.5, LaneDirection::Forward)],
        ))
        .unwrap();
        let mut cloud = PointCloud::default();
        for x in -5..=55 {
            for y in -15..=15 {
                cloud.positions.push([x as f64 * 0.2, y as f64 * 0.2, 2.]);
            }
        }
        (map, cloud)
    }
    #[test]
    fn full_surface_passes_read_only_but_center_only_support_exposes_edges() {
        let (map, mut cloud) = scene();
        let original = map.clone();
        let report = audit(&map, &cloud).unwrap();
        assert!(report.low_support_lanes.is_empty());
        assert!(!report.limited);
        cloud.positions.retain(|p| p[1].abs() < 0.25);
        let report = audit(&map, &cloud).unwrap();
        let lane = &report.lanes[0];
        assert_eq!(lane.center.fraction, 1.);
        assert_eq!(lane.left.fraction, 0.);
        assert_eq!(lane.right.fraction, 0.);
        assert!(lane.left.insufficient_returns > 0);
        assert_eq!(lane.left.height_mismatches, 0);
        assert_eq!(map, original);
    }
    #[test]
    fn missing_returns_and_wrong_level_are_distinguished() {
        let (map, mut cloud) = scene();
        for p in &mut cloud.positions {
            p[2] += 3.;
        }
        let r = audit(&map, &cloud).unwrap();
        assert_eq!(r.lanes[0].center.fraction, 0.);
        assert_eq!(r.lanes[0].center.insufficient_returns, 0);
        assert!(r.lanes[0].center.height_mismatches > 0);
        assert!(audit(&map, &PointCloud::default()).is_err());
    }
    #[test]
    fn locations_keep_survey_coordinates_reasons_and_unchanged_summary() {
        let (mut map, mut cloud) = scene();
        let shift = [500_000., 4_000_000., 10.];
        let boundaries: Vec<_> = map
            .boundaries()
            .map(|b| (b.id, b.geometry.clone()))
            .collect();
        for (id, mut line) in boundaries {
            for p in &mut line.points {
                p.x += shift[0];
                p.y += shift[1];
                p.z += shift[2];
            }
            map.set_boundary_geometry(id, line).unwrap();
        }
        cloud.positions.retain(|p| !(4. ..=6.).contains(&p[0]));
        for p in &mut cloud.positions {
            if p[0] >= 8. {
                p[2] += 2.;
            }
            for axis in 0..3 {
                p[axis] += shift[axis];
            }
        }
        let before = map.clone();
        let details = audit_with_locations(&map, &cloud).unwrap();
        assert_eq!(
            serde_json::to_value(&details.report).unwrap(),
            serde_json::to_value(audit(&map, &cloud).unwrap()).unwrap()
        );
        assert!(!details.problems_limited);
        let ground = Ground::new(&cloud).unwrap();
        for p in &details.problems {
            assert_eq!(p.points.len(), p.source_heights_m.len());
            for (xyz, observed) in p.points.iter().zip(&p.source_heights_m) {
                assert_eq!(
                    *observed,
                    ground.height(Point3::new(xyz[0], xyz[1], xyz[2]))
                );
                assert_eq!(
                    observed.is_none(),
                    p.reason == SupportProblem::InsufficientReturns
                );
            }
            assert_eq!(p.points[0][0] - shift[0], p.from_m);
            assert_eq!(p.points.last().unwrap()[0] - shift[0], p.to_m);
            for xyz in &p.points {
                let point = Point3::new(xyz[0], xyz[1], xyz[2]);
                match p.reason {
                    SupportProblem::InsufficientReturns => assert!(ground.height(point).is_none()),
                    SupportProblem::HeightMismatch => {
                        assert!((ground.height(point).unwrap() - point.z).abs() > HEIGHT)
                    }
                }
            }
            assert!(
                p.points
                    .windows(2)
                    .all(|v| v[1][0] - v[0][0] <= SPACING + 1e-8)
            );
        }
        let lane = &details.report.lanes[0];
        for (curve, support) in [
            (SourceCurve::Center, &lane.center),
            (SourceCurve::Left, &lane.left),
            (SourceCurve::Right, &lane.right),
        ] {
            for (reason, expected) in [
                (
                    SupportProblem::InsufficientReturns,
                    support.insufficient_returns,
                ),
                (SupportProblem::HeightMismatch, support.height_mismatches),
            ] {
                let shown: usize = details
                    .problems
                    .iter()
                    .filter(|p| p.curve == curve && p.reason == reason)
                    .map(|p| p.points.len())
                    .sum();
                assert!(expected > 0);
                assert_eq!(shown, expected);
            }
        }
        assert_eq!(map, before);
    }

    #[test]
    fn problem_stations_follow_backward_lane_travel() {
        let (mut map, cloud) = scene();
        let built = map
            .build_road(NewRoad::new(
                Polyline3::new(vec![Point3::new(0., 20., 2.), Point3::new(10., 20., 2.)]),
                vec![
                    RoadLane::new(3.5, LaneDirection::Forward),
                    RoadLane::new(3.5, LaneDirection::Backward),
                ],
            ))
            .unwrap()
            .0;
        let backward = built.lanes[1][0];
        let details = audit_with_locations(&map, &cloud).unwrap();
        let problems: Vec<_> = details
            .problems
            .iter()
            .filter(|p| p.lane == backward)
            .collect();
        assert_eq!(problems.len(), 3);
        for p in problems {
            assert_eq!((p.from_m, p.to_m), (0., 10.));
            assert_eq!(p.points.first().unwrap()[0], 10.);
            assert_eq!(p.points.last().unwrap()[0], 0.);
            assert!(p.points.windows(2).all(|v| v[1][0] < v[0][0]));
        }
    }

    #[test]
    fn location_limits_do_not_truncate_audit_figures_or_bridge_supported_gaps() {
        let (mut map, cloud) = scene();
        map.build_road(NewRoad::new(
            Polyline3::new(vec![Point3::new(0., 20., 2.), Point3::new(2_500., 20., 2.)]),
            vec![RoadLane::new(3.5, LaneDirection::Forward)],
        ))
        .unwrap();
        let details = audit_with_locations(&map, &cloud).unwrap();
        assert!(details.problems_limited);
        assert!(!details.report.limited);
        assert_eq!(details.report.lanes.len(), 2);
        assert_eq!(
            details
                .problems
                .iter()
                .map(|p| p.points.len())
                .sum::<usize>(),
            MAX_PROBLEM_POINTS
        );
        assert_eq!(
            details
                .problems
                .iter()
                .map(|p| p.source_heights_m.len())
                .sum::<usize>(),
            MAX_PROBLEM_POINTS
        );
        assert_eq!(
            serde_json::to_value(&details.report).unwrap(),
            serde_json::to_value(audit(&map, &cloud).unwrap()).unwrap()
        );

        let lane = details.report.lanes[0].lane;
        let mut locations = ProblemLocations::default();
        for i in 0..1_000 {
            let reason = (i % 2 == 0).then_some(SupportProblem::HeightMismatch);
            locations.sample(
                lane,
                SourceCurve::Left,
                i,
                i as f64,
                Point3::new(i as f64, 0., 2.),
                (reason, Some(0.0)),
            );
        }
        assert!(locations.limited);
        assert_eq!(locations.problems.len(), MAX_PROBLEM_INTERVALS);
        assert!(
            locations
                .problems
                .iter()
                .all(|p| p.points.len() == 1 && p.from_m == p.to_m)
        );
    }
    #[test]
    fn huge_lanes_are_omitted_before_allocating_and_report_is_not_all_clear() {
        let (mut map, cloud) = scene();
        map.build_road(NewRoad::new(
            Polyline3::new(vec![Point3::new(0., 20., 2.), Point3::new(1e8, 20., 2.)]),
            vec![RoadLane::new(3.5, LaneDirection::Forward)],
        ))
        .unwrap();
        let r = audit(&map, &cloud).unwrap();
        assert!(r.limited);
        assert_eq!(r.omitted_lanes.len(), 1);
        assert_eq!(r.lanes.len(), 1);
        assert!(r.sampled_points <= MAX_SAMPLES);
    }
    #[test]
    fn stop_marking_is_checked_against_selected_lane_direction() {
        let (map, _) = scene();
        let lane = map.lanes().next().unwrap().id;
        assert!(transverse_stop(&map, &[[5., -2., 2.], [5., 2., 2.]], lane));
        assert!(!transverse_stop(&map, &[[3., 0., 2.], [7., 0., 2.]], lane));
        assert!(transverse_stop(&map, &[[5., 2., 2.], [5., -2., 2.]], lane));
    }
}
