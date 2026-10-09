//! Lane-free proposals from contiguous low source bands along a recorded path.
//! Coverage edges are not road edges; neither surface support nor paired curb
//! profiles determine traffic semantics, clearance or the correct physical level.
use serde::{Deserialize, Serialize};
use vectormap_core::Point3;

use super::{BuildError, BuildOptions, SurfaceIndex, junctions, trace_alignment};
use crate::PointCloud;

const SPACING: f64 = 2.0;
const BIN: f64 = 0.5;
const HALF_WINDOW: f64 = 2.0;
const MIN_WIDTH: f64 = 1.0;
const STEP: f64 = 0.08;
const MAX_SECTIONS: usize = 2_048;
const QUERY_BUDGET: usize = 2_000_000;
const SUPPORT_BUDGET: usize = 100_000;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CorridorOptions {
    /// Symmetric search reach from the path, independent of lane count or width.
    pub search_radius_m: f64,
    /// Explicit geometric association, independent of lane identity or legal use.
    pub association: CorridorAssociation,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CorridorAssociation {
    #[default]
    AllSupportedBands,
    TrajectoryContaining,
}
impl Default for CorridorOptions {
    fn default() -> Self {
        Self {
            search_radius_m: 8.0,
            association: CorridorAssociation::AllSupportedBands,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeEvidence {
    CurbProfile,
    SupportGap,
    HeightDiscontinuity,
    SearchLimit,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrossSection {
    pub station_m: f64,
    pub center: [f64; 3],
    pub left: [f64; 3],
    pub right: [f64; 3],
    pub support_span_m: f64,
    pub left_evidence: EdgeEvidence,
    pub right_evidence: EdgeEvidence,
    pub intersects_trajectory: bool,
    pub path_level_supported: bool,
}
impl CrossSection {
    fn curbs(&self) -> bool {
        self.left_evidence == EdgeEvidence::CurbProfile
            && self.right_evidence == EdgeEvidence::CurbProfile
    }
}

#[derive(Debug, Serialize)]
pub struct CorridorCandidate {
    pub id: usize,
    pub from_m: f64,
    pub to_m: f64,
    pub minimum_support_span_m: f64,
    pub maximum_support_span_m: f64,
    pub paired_curb_sections: usize,
    /// Present only when every section has two curb-like physical profiles.
    /// Other spans measure point support, not a complete road or path width.
    pub curb_width_range_m: Option<[f64; 2]>,
    pub sections: Vec<CrossSection>,
    pub review_required: bool,
}

#[derive(Debug, Serialize)]
pub struct Profile {
    pub station_m: f64,
    pub trajectory: [f64; 3],
    pub heading_usable: bool,
    pub reference_ground_height_m: Option<f64>,
    pub bands: Vec<CrossSection>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalReason {
    MissingCoherentSurface,
    MissingGroundAnchor,
    MissingTrajectoryBand,
    UnmatchedBands,
    SourceGap,
    BranchingBands,
    UnstableHeading,
    QueryBudget,
    SupportBudget,
}
#[derive(Debug, Serialize)]
pub struct StationInterval {
    pub from_m: f64,
    pub to_m: f64,
    pub reason: IntervalReason,
}

#[derive(Debug, Serialize)]
pub struct CorridorProtocol {
    pub options: CorridorOptions,
    pub sample_spacing_m: f64,
    pub bin_width_m: f64,
    pub half_window_m: f64,
    pub minimum_support_span_m: f64,
    pub maximum_adjacent_height_step_m: f64,
    pub maximum_reference_level_grade: f64,
    pub ground_estimator: super::quality::GroundEstimator,
    pub interval_support_spacing_m: f64,
    pub interval_ground_radius_m: f64,
    pub interval_height_tolerance_m: f64,
    pub maximum_sections: usize,
    pub profile_query_point_budget: usize,
    pub interval_support_sample_budget: usize,
}

#[derive(Debug, Serialize)]
pub struct CorridorReport {
    pub schema: &'static str,
    pub coordinate_frame: &'static str,
    pub protocol: CorridorProtocol,
    pub trajectory_length_m: f64,
    /// Union of along-input-path intervals, not summed candidate/unique road lengths.
    pub with_candidate_station_length_m: f64,
    pub without_candidate_station_length_m: f64,
    pub trajectory_covered_station_length_m: f64,
    pub curb_bounded_station_length_m: f64,
    pub ambiguous_station_length_m: f64,
    pub sampled_sections: usize,
    pub evaluated_sections: usize,
    pub sections_with_bands: usize,
    pub multiple_band_sections: usize,
    pub profile_queried_points: usize,
    pub interval_support_samples: usize,
    pub unstable_heading_sections: usize,
    pub unanchored_sections: usize,
    pub level_mismatch_bands: usize,
    /// Observations kept in profiles but excluded by explicit path association.
    pub off_trajectory_bands: usize,
    pub limited: bool,
    pub profiles: Vec<Profile>,
    pub candidates: Vec<CorridorCandidate>,
    pub deferred_intervals: Vec<StationInterval>,
    pub ambiguous_intervals: Vec<StationInterval>,
    pub road_semantics_inferred: bool,
    pub deployment_ready: bool,
    pub warnings: Vec<String>,
}

/// Exact stations on the input's XY arc length, including its last position.
/// Sensor Z is used only for interpolation of the path; proposed Z is source Z.
fn samples(poses: &[[f64; 3]]) -> Result<Vec<(f64, [f64; 3])>, BuildError> {
    if poses.len() < 2
        || poses
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 1e12)
    {
        return Err(BuildError(
            "corridor search needs two finite trajectory positions in metres".into(),
        ));
    }
    let lengths: Vec<_> = poses
        .windows(2)
        .map(|p| (p[1][0] - p[0][0]).hypot(p[1][1] - p[0][1]))
        .collect();
    let total: f64 = lengths.iter().sum();
    if total < 1e-6 || (total / SPACING).ceil() + 1.0 > MAX_SECTIONS as f64 {
        return Err(BuildError(
            "corridor path has no XY movement or exceeds the section budget; split the recording"
                .into(),
        ));
    }
    let mut out = Vec::new();
    let mut next = 0.0;
    let mut station = 0.0;
    for (pair, d) in poses.windows(2).zip(lengths) {
        if d > 1e-9 {
            while next < station + d {
                let t = (next - station) / d;
                out.push((
                    next,
                    std::array::from_fn(|a| pair[0][a] + t * (pair[1][a] - pair[0][a])),
                ));
                next += SPACING;
            }
        }
        station += d;
    }
    out.push((total, *poses.last().unwrap()));
    Ok(out)
}

fn edges(surface: &[Option<f64>], index: usize, outward: isize, o: &BuildOptions) -> EdgeEvidence {
    if trace_alignment::curb(surface, index, outward, o) {
        return EdgeEvidence::CurbProfile;
    }
    match index
        .checked_add_signed(outward)
        .and_then(|i| surface.get(i))
    {
        None => EdgeEvidence::SearchLimit,
        Some(None) => EdgeEvidence::SupportGap,
        Some(Some(_)) => EdgeEvidence::HeightDiscontinuity,
    }
}

fn bands(
    surface: &[Option<f64>],
    sample: (f64, [f64; 3]),
    dir: [f64; 2],
    reach: f64,
    o: &BuildOptions,
    reference_ground: Option<f64>,
) -> Vec<CrossSection> {
    let (station, p) = sample;
    let lateral = |i: usize| -reach + (i as f64 + 0.5) * BIN;
    let xyz = |offset: f64, z: f64| [p[0] - dir[1] * offset, p[1] + dir[0] * offset, z];
    let mut result = Vec::new();
    let mut start = 0;
    for end in 1..=surface.len() {
        if end < surface.len()
            && surface[end]
                .zip(surface[end - 1])
                .is_some_and(|(a, b)| (a - b).abs() <= STEP)
        {
            continue;
        }
        if surface[start].is_some() && (end - start - 1) as f64 * BIN >= MIN_WIDTH {
            let right = lateral(start);
            let left = lateral(end - 1);
            let middle = (start + end - 1) / 2;
            result.push(CrossSection {
                station_m: station,
                center: xyz((left + right) * 0.5, surface[middle].unwrap()),
                left: xyz(left, surface[end - 1].unwrap()),
                right: xyz(right, surface[start].unwrap()),
                support_span_m: left - right,
                left_evidence: edges(surface, end - 1, 1, o),
                right_evidence: edges(surface, start, -1, o),
                intersects_trajectory: right <= 0.0 && left >= 0.0,
                path_level_supported: reference_ground.is_some_and(|z| {
                    (surface[middle].unwrap() - z).abs()
                        <= junctions::HEIGHT + 0.12 * ((left + right) * 0.5).abs()
                }),
            });
        }
        start = end;
    }
    result
}

/// Connect only uniquely overlapping bands with bounded vertical change. A
/// branch never silently chooses the widest/nearest band as a semantic road.
fn matching(a: &CrossSection, b: &CrossSection, station_distance: f64) -> bool {
    let edge = [b.left[0] - b.right[0], b.left[1] - b.right[1]];
    let norm = edge[0].hypot(edge[1]);
    let offset =
        ((a.center[0] - b.center[0]) * edge[0] + (a.center[1] - b.center[1]) * edge[1]) / norm;
    let overlap = (a.support_span_m + b.support_span_m) * 0.5 - offset.abs();
    overlap >= MIN_WIDTH * 0.5
        && (a.center[2] - b.center[2]).abs() <= station_distance * 0.12 + junctions::LAYER_HEIGHT
        && (a.center[0] - b.center[0]).hypot(a.center[1] - b.center[1])
            <= station_distance * 2.0 + BIN
}

fn support_sample_count(a: &CrossSection, b: &CrossSection) -> usize {
    [(a.center, b.center), (a.left, b.left), (a.right, b.right)]
        .iter()
        .map(|(a, b)| {
            let d = (b[0] - a[0]).hypot(b[1] - a[1]).hypot(b[2] - a[2]);
            (d / 0.5).ceil().max(1.0) as usize + 1
        })
        .sum()
}

fn supported(
    a: &CrossSection,
    b: &CrossSection,
    ground: &junctions::Ground<'_>,
    samples: &mut usize,
) -> bool {
    [(a.center, b.center), (a.left, b.left), (a.right, b.right)]
        .iter()
        .all(|(a, b)| {
            let d = (b[0] - a[0]).hypot(b[1] - a[1]).hypot(b[2] - a[2]);
            let steps = (d / 0.5).ceil().max(1.0) as usize;
            (0..=steps).all(|k| {
                *samples += 1;
                let p: [f64; 3] =
                    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * k as f64 / steps as f64);
                ground.supports(Point3::new(p[0], p[1], p[2]))
            })
        })
}

fn interval(out: &mut Vec<StationInterval>, from: f64, to: f64, reason: IntervalReason) {
    if let Some(last) = out
        .last_mut()
        .filter(|last| last.to_m == from && last.reason == reason)
    {
        last.to_m = to;
    } else {
        out.push(StationInterval {
            from_m: from,
            to_m: to,
            reason,
        });
    }
}

/// No lane count, lane width, direction, speed or existing map is an input.
/// All proposals require review; absent observations are not filled from priors.
pub fn propose(
    cloud: &PointCloud,
    poses: &[[f64; 3]],
    options: &CorridorOptions,
) -> Result<CorridorReport, BuildError> {
    propose_limited(cloud, poses, options, QUERY_BUDGET, SUPPORT_BUDGET)
}

fn propose_limited(
    cloud: &PointCloud,
    poses: &[[f64; 3]],
    options: &CorridorOptions,
    query_budget: usize,
    support_budget: usize,
) -> Result<CorridorReport, BuildError> {
    if !options.search_radius_m.is_finite() || !(1.0..=20.0).contains(&options.search_radius_m) {
        return Err(BuildError(
            "search_radius_m must be finite and within 1..20 metres".into(),
        ));
    }
    let samples = samples(poses)?;
    let total = samples.last().unwrap().0;
    // Round the reach outward to complete bins, and record the effective reach.
    let reach = (options.search_radius_m / BIN).ceil() * BIN;
    let o = BuildOptions {
        half_window: HALF_WINDOW,
        bin_width: BIN,
        curb_height: STEP,
        ..BuildOptions::default()
    };
    let ground = junctions::Ground::new_consensus(cloud)?;
    let line: Vec<_> = samples.iter().map(|(_, p)| *p).collect();
    let index = SurfaceIndex::new(cloud, &line, reach + HALF_WINDOW);
    let mut report = CorridorReport {
        schema: "cloudanalyzer.corridor_proposals.v1", coordinate_frame: "input_metres",
        protocol: CorridorProtocol { options: CorridorOptions { search_radius_m: reach, association: options.association }, sample_spacing_m: SPACING, bin_width_m: BIN, half_window_m: HALF_WINDOW,
            minimum_support_span_m: MIN_WIDTH, maximum_adjacent_height_step_m: STEP, maximum_reference_level_grade: 0.12, ground_estimator: super::quality::GroundEstimator::consensus(),
            interval_support_spacing_m: 0.5, interval_ground_radius_m: junctions::RADIUS, interval_height_tolerance_m: junctions::HEIGHT,
            maximum_sections: MAX_SECTIONS, profile_query_point_budget: query_budget, interval_support_sample_budget: support_budget },
        trajectory_length_m: total, with_candidate_station_length_m: 0., without_candidate_station_length_m: total,
        trajectory_covered_station_length_m: 0., curb_bounded_station_length_m: 0., ambiguous_station_length_m: 0.,
        sampled_sections: samples.len(), evaluated_sections: 0, sections_with_bands: 0, multiple_band_sections: 0,
        profile_queried_points: 0, interval_support_samples: 0, unstable_heading_sections: 0, unanchored_sections: 0, level_mismatch_bands: 0, off_trajectory_bands: 0, limited: false,
        profiles: Vec::new(), candidates: Vec::new(), deferred_intervals: Vec::new(), ambiguous_intervals: Vec::new(),
        road_semantics_inferred: false, deployment_ready: false,
        warnings: vec!["Proposals follow spatially supported low surfaces, not semantic roads or lanes. A lower level can be wrong. Coverage gaps and search limits are not physical road edges; support spans are incomplete widths unless every section has two curb-like profiles. Check geometry, branches, obstacles, legal use and georeferencing. Repeated passes are separate proposals, not unique road length. Bin-centre edges have 0.5 m quantization; full-width interiors are not certified by three longitudinal support curves.".into()],
    };
    let mut previous: Vec<CrossSection> = Vec::new();
    if options.association == CorridorAssociation::TrajectoryContaining {
        report.warnings.push("Explicit trajectory-containing association excludes off-path bands from matching but retains them in every original profile. This resolves geometric association only, not physical road branches or lane identity; unsupported path intervals remain deferred.".into());
    }
    let mut tracks: Vec<Option<usize>> = Vec::new();
    for (k, &(station, p)) in samples.iter().enumerate() {
        let a = samples[k.saturating_sub(1)].1;
        let b = samples[(k + 1).min(samples.len() - 1)].1;
        let norm = (b[0] - a[0]).hypot(b[1] - a[1]);
        let mut current = Vec::new();
        let mut observed = Vec::new();
        let reference_ground = ground.height(Point3::new(p[0], p[1], p[2]));
        if norm > 1e-6 {
            let dir = [(b[0] - a[0]) / norm, (b[1] - a[1]) / norm];
            let bins = index.slice(p, dir, -reach, reach, &o);
            let count: usize = bins.iter().map(Vec::len).sum();
            if report.profile_queried_points + count > query_budget {
                report.limited = true;
                interval(
                    &mut report.deferred_intervals,
                    samples[k.saturating_sub(1)].0,
                    total,
                    IntervalReason::QueryBudget,
                );
                break;
            }
            report.profile_queried_points += count;
            let surface: Vec<_> = bins
                .iter()
                .map(|ids| {
                    junctions::lowest_layer(
                        &mut ids.iter().map(|&i| cloud.positions[i]).collect::<Vec<_>>(),
                    )
                })
                .collect();
            observed = bands(&surface, (station, p), dir, reach, &o, reference_ground);
            current = observed
                .iter()
                .filter(|b| {
                    b.path_level_supported
                        && (options.association == CorridorAssociation::AllSupportedBands
                            || b.intersects_trajectory)
                })
                .cloned()
                .collect();
        }
        report.evaluated_sections += 1;
        report.unstable_heading_sections += usize::from(norm <= 1e-6);
        report.unanchored_sections += usize::from(reference_ground.is_none());
        if reference_ground.is_some() {
            report.level_mismatch_bands +=
                observed.iter().filter(|b| !b.path_level_supported).count();
        }
        report.off_trajectory_bands += observed
            .iter()
            .filter(|b| b.path_level_supported && !b.intersects_trajectory)
            .count();
        report.sections_with_bands += usize::from(!current.is_empty());
        report.multiple_band_sections += usize::from(current.len() > 1);
        report.profiles.push(Profile {
            station_m: station,
            trajectory: p,
            heading_usable: norm > 1e-6,
            reference_ground_height_m: reference_ground,
            bands: observed,
        });
        let mut next_tracks = vec![None; current.len()];
        if k > 0 {
            let distance = station - samples[k - 1].0;
            let pairs: Vec<_> = previous
                .iter()
                .map(|a| {
                    current
                        .iter()
                        .map(|b| matching(a, b, distance))
                        .collect::<Vec<_>>()
                })
                .collect();
            let row_counts: Vec<_> = pairs
                .iter()
                .map(|row| row.iter().filter(|&&v| v).count())
                .collect();
            let column_counts: Vec<_> = (0..current.len())
                .map(|j| pairs.iter().filter(|row| row[j]).count())
                .collect();
            let required: usize = previous
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    current
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| {
                            pairs[i][*j] && row_counts[i] == 1 && column_counts[*j] == 1
                        })
                        .map(|(_, b)| support_sample_count(a, b))
                        .sum::<usize>()
                })
                .sum();
            if report.interval_support_samples + required > support_budget {
                report.limited = true;
                interval(
                    &mut report.deferred_intervals,
                    samples[k - 1].0,
                    total,
                    IntervalReason::SupportBudget,
                );
                break;
            }
            let ambiguous = row_counts.iter().chain(&column_counts).any(|&n| n > 1);
            if ambiguous {
                report.ambiguous_station_length_m += distance;
                interval(
                    &mut report.ambiguous_intervals,
                    samples[k - 1].0,
                    station,
                    IntervalReason::BranchingBands,
                );
            }
            let mut connected = false;
            let mut on_trace = false;
            let mut curbs = false;
            let mut source_gap = false;
            for (i, a) in previous.iter().enumerate() {
                for (j, b) in current.iter().enumerate() {
                    if !pairs[i][j] || row_counts[i] != 1 || column_counts[j] != 1 {
                        continue;
                    }
                    if !supported(a, b, &ground, &mut report.interval_support_samples) {
                        source_gap = true;
                        continue;
                    }
                    let track = if let Some(id) = tracks[i] {
                        id
                    } else {
                        let id = report.candidates.len();
                        report.candidates.push(CorridorCandidate {
                            id: id + 1,
                            from_m: a.station_m,
                            to_m: b.station_m,
                            minimum_support_span_m: a.support_span_m,
                            maximum_support_span_m: a.support_span_m,
                            paired_curb_sections: usize::from(a.curbs()),
                            curb_width_range_m: None,
                            sections: vec![a.clone()],
                            review_required: true,
                        });
                        id
                    };
                    let c = &mut report.candidates[track];
                    c.to_m = b.station_m;
                    c.minimum_support_span_m = c.minimum_support_span_m.min(b.support_span_m);
                    c.maximum_support_span_m = c.maximum_support_span_m.max(b.support_span_m);
                    c.paired_curb_sections += usize::from(b.curbs());
                    c.sections.push(b.clone());
                    next_tracks[j] = Some(track);
                    connected = true;
                    on_trace |= a.intersects_trajectory && b.intersects_trajectory;
                    curbs |= a.curbs() && b.curbs();
                }
            }
            if connected {
                report.with_candidate_station_length_m += distance;
                report.trajectory_covered_station_length_m += if on_trace { distance } else { 0. };
                report.curb_bounded_station_length_m += if curbs { distance } else { 0. };
            } else {
                let reason = if !report.profiles[k - 1].heading_usable
                    || !report.profiles[k].heading_usable
                {
                    IntervalReason::UnstableHeading
                } else if report.profiles[k - 1].reference_ground_height_m.is_none()
                    || report.profiles[k].reference_ground_height_m.is_none()
                {
                    IntervalReason::MissingGroundAnchor
                } else if previous.is_empty() || current.is_empty() {
                    if options.association == CorridorAssociation::TrajectoryContaining
                        && ((previous.is_empty()
                            && report.profiles[k - 1]
                                .bands
                                .iter()
                                .any(|b| b.path_level_supported))
                            || (current.is_empty()
                                && report.profiles[k]
                                    .bands
                                    .iter()
                                    .any(|b| b.path_level_supported)))
                    {
                        IntervalReason::MissingTrajectoryBand
                    } else {
                        IntervalReason::MissingCoherentSurface
                    }
                } else if ambiguous {
                    IntervalReason::BranchingBands
                } else if source_gap {
                    IntervalReason::SourceGap
                } else {
                    IntervalReason::UnmatchedBands
                };
                interval(
                    &mut report.deferred_intervals,
                    samples[k - 1].0,
                    station,
                    reason,
                );
            }
        }
        previous = current;
        tracks = next_tracks;
    }
    report.without_candidate_station_length_m = total - report.with_candidate_station_length_m;
    for c in &mut report.candidates {
        if c.paired_curb_sections == c.sections.len() {
            c.curb_width_range_m = Some([c.minimum_support_span_m, c.maximum_support_span_m]);
        }
    }
    if report.limited {
        report.warnings.push("A profile-query or interval-support budget was reached; remaining intervals are deferred and were not fully inspected. Start a smaller recording segment; do not treat partial proposals as complete.".into());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene() -> PointCloud {
        let mut cloud = PointCloud::default();
        for x in -15..=115 {
            for y in -40..=40 {
                let y = y as f64 * 0.2;
                let z = if (-3.0..=1.0).contains(&y) { 2.0 } else { 2.15 };
                cloud.positions.push([x as f64 * 0.2, y, z]);
            }
        }
        cloud
    }
    fn poses() -> [[f64; 3]; 2] {
        [[0., 0., 99.], [20., 0., 99.]]
    }

    #[test]
    fn explicit_path_association_keeps_observations_and_resolves_off_path_branches() {
        let mut cloud = scene();
        for p in &mut cloud.positions {
            if p[0] > 10. && (-1.5..=-0.5).contains(&p[1]) {
                p[2] += 1.;
            }
        }
        let original = cloud.positions.clone();
        let all = propose(&cloud, &poses(), &CorridorOptions::default()).unwrap();
        let on_path = propose(
            &cloud,
            &poses(),
            &CorridorOptions {
                association: CorridorAssociation::TrajectoryContaining,
                ..CorridorOptions::default()
            },
        )
        .unwrap();
        assert!(all.ambiguous_station_length_m > 0.);
        assert_eq!(on_path.ambiguous_station_length_m, 0.);
        assert!(
            on_path.trajectory_covered_station_length_m > all.trajectory_covered_station_length_m
        );
        assert!(on_path.off_trajectory_bands > 0);
        assert_eq!(
            serde_json::to_value(&all.profiles).unwrap(),
            serde_json::to_value(&on_path.profiles).unwrap()
        );
        assert!(
            on_path
                .candidates
                .iter()
                .flat_map(|c| &c.sections)
                .all(|s| s.intersects_trajectory && s.path_level_supported)
        );
        assert_eq!(cloud.positions, original);
        assert!(!on_path.road_semantics_inferred && !on_path.deployment_ready);
    }

    #[test]
    fn path_association_never_borrows_an_outside_band_or_bridges_missing_source() {
        let options = CorridorOptions {
            association: CorridorAssociation::TrajectoryContaining,
            ..CorridorOptions::default()
        };
        let mut outside = scene();
        outside.positions.retain(|p| p[1] >= 0.4);
        let report = propose(&outside, &poses(), &options).unwrap();
        assert!(report.candidates.is_empty());
        assert_eq!(report.without_candidate_station_length_m, 20.);
        assert!(report.profiles.iter().any(|p| !p.bands.is_empty()));
        assert!(report.off_trajectory_bands > 0);
        assert!(
            report
                .deferred_intervals
                .iter()
                .any(|i| i.reason == IntervalReason::MissingTrajectoryBand)
        );
        let mut gap = scene();
        gap.positions.retain(|p| !(7.0..13.0).contains(&p[0]));
        let report = propose(&gap, &poses(), &options).unwrap();
        assert!(report.without_candidate_station_length_m >= 4.);
        assert!(
            report
                .candidates
                .iter()
                .all(|c| !(c.from_m < 7. && c.to_m > 13.))
        );
        assert_eq!(report.trajectory_length_m, 20.);
    }

    #[test]
    fn offset_curb_band_has_measured_geometry_without_lane_or_sensor_priors() {
        let cloud = scene();
        let original = cloud.positions.clone();
        let report = propose(&cloud, &poses(), &CorridorOptions::default()).unwrap();
        let road = report
            .candidates
            .iter()
            .find(|c| c.curb_width_range_m.is_some())
            .unwrap();
        assert_eq!(road.from_m, 0.);
        assert_eq!(road.to_m, 20.);
        assert!((road.sections[0].center[1] + 1.).abs() < BIN);
        assert!(
            road.sections
                .iter()
                .all(|s| (s.center[2] - 2.).abs() < 1e-9)
        );
        assert!(road.maximum_support_span_m <= 4.0);
        assert_eq!(report.with_candidate_station_length_m, 20.);
        assert_eq!(report.without_candidate_station_length_m, 0.);
        assert!(report.candidates.iter().all(|c| c.review_required));
        assert!(!report.road_semantics_inferred && !report.deployment_ready && !report.limited);
        assert_eq!(cloud.positions, original);
    }

    #[test]
    fn observation_and_search_edges_never_supply_a_complete_width() {
        let mut cloud = scene();
        cloud.positions.retain(|p| p[1] >= -3. && p[1] <= 1.);
        let report = propose(&cloud, &poses(), &CorridorOptions::default()).unwrap();
        assert!(!report.candidates.is_empty());
        assert!(
            report
                .candidates
                .iter()
                .all(|c| c.curb_width_range_m.is_none())
        );
        assert!(
            report
                .candidates
                .iter()
                .flat_map(|c| &c.sections)
                .all(|s| s.left_evidence == EdgeEvidence::SupportGap
                    && s.right_evidence == EdgeEvidence::SupportGap)
        );
        let report = propose(
            &scene(),
            &poses(),
            &CorridorOptions {
                search_radius_m: 1.0,
                ..CorridorOptions::default()
            },
        )
        .unwrap();
        assert!(
            report
                .candidates
                .iter()
                .all(|c| c.curb_width_range_m.is_none())
        );
        assert!(
            report
                .candidates
                .iter()
                .flat_map(|c| &c.sections)
                .all(|s| s.left_evidence == EdgeEvidence::SearchLimit
                    && s.right_evidence == EdgeEvidence::SearchLimit)
        );
    }

    #[test]
    fn elevated_or_lower_bands_remain_observations_without_becoming_path_proposals() {
        let mut cloud = scene();
        for p in &mut cloud.positions {
            if p[1] > 4.0 {
                p[2] = 5.0;
            }
            if p[1] < -4.0 {
                p[2] = -3.0;
            }
        }
        let report = propose(&cloud, &poses(), &CorridorOptions::default()).unwrap();
        assert!(report.level_mismatch_bands > 0);
        assert!(
            report
                .profiles
                .iter()
                .flat_map(|p| &p.bands)
                .any(|b| !b.path_level_supported)
        );
        assert!(
            report
                .candidates
                .iter()
                .flat_map(|c| &c.sections)
                .all(|s| s.path_level_supported && s.center[2] > 1.0 && s.center[2] < 3.0)
        );
    }

    #[test]
    fn missing_source_and_branches_break_tracks_and_keep_full_extent() {
        let mut cloud = scene();
        cloud.positions.retain(|p| !(7.0..13.0).contains(&p[0]));
        let report = propose(&cloud, &poses(), &CorridorOptions::default()).unwrap();
        assert_eq!(report.trajectory_length_m, 20.);
        assert!(report.without_candidate_station_length_m >= 4.0);
        assert!(
            report
                .candidates
                .iter()
                .all(|c| !(c.from_m < 7. && c.to_m > 13.))
        );
        let a = CrossSection {
            station_m: 0.,
            center: [0., 0., 2.],
            left: [0., 3., 2.],
            right: [0., -3., 2.],
            support_span_m: 6.,
            left_evidence: EdgeEvidence::SupportGap,
            right_evidence: EdgeEvidence::SupportGap,
            intersects_trajectory: true,
            path_level_supported: true,
        };
        let mut b = a.clone();
        b.center = [2., -1.5, 2.];
        b.left = [2., -0.5, 2.];
        b.right = [2., -2.5, 2.];
        b.support_span_m = 2.;
        let mut c = b.clone();
        c.center[1] = 1.5;
        c.left[1] = 2.5;
        c.right[1] = 0.5;
        assert!(matching(&a, &b, 2.) && matching(&a, &c, 2.));
        // Generate a split at x=10: one wide low band becomes two branches.
        let mut cloud = scene();
        for p in &mut cloud.positions {
            if p[0] > 10. && (-1.5..=-0.5).contains(&p[1]) {
                p[2] += 1.;
            }
        }
        let report = propose(&cloud, &poses(), &CorridorOptions::default()).unwrap();
        assert!(report.ambiguous_station_length_m > 0.);
        assert!(
            report
                .candidates
                .iter()
                .filter(|c| c.curb_width_range_m.is_some())
                .all(|c| c.to_m <= 12.)
        );
    }

    #[test]
    fn no_supported_band_and_resource_limits_do_not_become_passes() {
        let mut cloud = scene();
        cloud.positions.retain(|p| p[1] == 0.);
        let report = propose(&cloud, &poses(), &CorridorOptions::default()).unwrap();
        assert!(report.candidates.is_empty());
        assert_eq!(report.without_candidate_station_length_m, 20.);
        for (query, support, reason) in [
            (0, SUPPORT_BUDGET, IntervalReason::QueryBudget),
            (QUERY_BUDGET, 0, IntervalReason::SupportBudget),
        ] {
            let report = propose_limited(
                &scene(),
                &poses(),
                &CorridorOptions::default(),
                query,
                support,
            )
            .unwrap();
            assert!(report.limited && report.candidates.is_empty());
            assert_eq!(report.deferred_intervals.last().unwrap().reason, reason);
            assert_eq!(report.deferred_intervals.last().unwrap().to_m, 20.);
        }
        assert!(
            propose(
                &scene(),
                &[[0., 0., 0.], [10_000., 0., 0.]],
                &CorridorOptions::default()
            )
            .is_err()
        );
        assert!(
            propose(
                &scene(),
                &poses(),
                &CorridorOptions {
                    search_radius_m: f64::NAN,
                    ..CorridorOptions::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn exact_station_endpoint_and_turnaround_remain_visible() {
        let report = propose(
            &scene(),
            &[[0., 0., 99.], [4., 0., 99.], [0., 0., 99.]],
            &CorridorOptions::default(),
        )
        .unwrap();
        assert_eq!(report.trajectory_length_m, 8.);
        assert!(report.unstable_heading_sections > 0);
        assert!(
            report
                .deferred_intervals
                .iter()
                .any(|i| i.reason == IntervalReason::UnstableHeading)
        );
        let samples = samples(&[[0., 0., 99.], [0., 0., 100.], [2.01, 0., 200.]]).unwrap();
        assert_eq!(samples.last().unwrap().0, 2.01);
        assert_eq!(samples.last().unwrap().1, [2.01, 0., 200.]);
    }
}
