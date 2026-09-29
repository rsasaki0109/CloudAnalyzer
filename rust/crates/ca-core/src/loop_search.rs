//! Finding loops in a pose graph from its keyframes' scans: candidate pairs
//! (see [`crate::pose_graph::loop_candidates`]) registered with ICP, kept
//! when their structure overlaps and they sit where the drift allows.
//!
//! The web app runs the same steps with the registrations spread over its
//! worker pool; the command line calls [`find_loops`], parallel with the
//! `parallel` feature.

use crate::PointCloud;
use crate::icp::{IcpMetric, IcpParams, IcpResult, Rigid};
use crate::pose_graph::{self, PoseGraph};

/// ICP registers at most this many points of a loop's scans.
pub const LOOP_SAMPLE: usize = 8000;

/// Metres of possible drift between a candidate loop's nodes from which a
/// failed registration is retried as if both scans were taken at the same
/// place. Along a short stretch, "the same place" would only match a
/// corridor to itself.
pub const RETRY_MIN_DRIFT: f64 = 5.0;

#[derive(Debug, Clone, Copy)]
pub struct LoopSettings {
    pub max_iterations: usize,
    /// Share of the closest pairs kept each ICP iteration.
    pub overlap: f64,
    pub point_to_plane: bool,
    /// Points closer than this after ICP (metres) count towards the fitness.
    pub inlier_distance: f64,
    /// Below this fitness, register again as if both scans were taken at
    /// the same place, from `retry_headings` headings (0: no retry).
    pub retry_below: f64,
    pub retry_headings: usize,
}

/// One registered pair.
#[derive(Debug, Clone, Copy)]
pub struct LoopRegistration {
    /// The pose of `to` in the frame of `from`.
    pub measurement: Rigid,
    pub result: IcpResult,
    /// Share of `to`'s structure (not floor) near `from`'s after ICP.
    pub fitness: f64,
    /// Registered as the same place (see [`LoopSettings::retry_below`]).
    pub retried: bool,
    /// How far the measurement puts `to` from where the guess did (metres).
    pub discrepancy: f64,
}

/// Register `to_scan` onto `from_scan` from `guess` (the pose of `to` in
/// the frame of `from`).
pub fn register_pair(
    from_scan: &PointCloud,
    to_scan: &PointCloud,
    guess: &Rigid,
    settings: &LoopSettings,
) -> Option<LoopRegistration> {
    let params = IcpParams {
        metric: if settings.point_to_plane {
            IcpMetric::PointToPlane
        } else {
            IcpMetric::PointToPoint
        },
        max_iterations: settings.max_iterations,
        overlap: settings.overlap,
        sample: LOOP_SAMPLE,
        ..IcpParams::default()
    };
    let (mut measurement, mut result) =
        pose_graph::register_loop(from_scan, to_scan, guess, params)?;
    let inlier = settings.inlier_distance;
    let mut fitness = pose_graph::overlap_fitness(from_scan, to_scan, &measurement, inlier);
    let mut retried = false;
    let retry = (fitness < settings.retry_below && settings.retry_headings > 0)
        .then(|| {
            pose_graph::register_with_yaw_search(
                from_scan,
                to_scan,
                &Rigid::IDENTITY,
                settings.retry_headings,
                params,
                inlier,
            )
        })
        .flatten();
    if let Some((m, f, r)) = retry.filter(|&(_, f, _)| f > fitness) {
        (measurement, fitness, result, retried) = (m, f, r, true);
    }
    let d: [f64; 3] = std::array::from_fn(|k| measurement.translation[k] - guess.translation[k]);
    Some(LoopRegistration {
        measurement,
        result,
        fitness,
        retried,
        discrepancy: (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt(),
    })
}

/// The web app's defaults.
#[derive(Debug, Clone, Copy)]
pub struct FindLoopsParams {
    /// Candidate pairs at most this far apart as the graph has them (metres)…
    pub max_distance: f64,
    /// …plus this share of the path between them (odometry drift).
    pub drift: f64,
    /// …and at least this far apart along the path (metres).
    pub min_travel: f64,
    /// At most one candidate every this many metres of path.
    pub spacing: f64,
    /// Keep a loop when at least this share of the later scan overlaps.
    pub min_fitness: f64,
    pub inlier_distance: f64,
    pub retry_headings: usize,
    pub max_iterations: usize,
    pub overlap: f64,
    /// Loop edge standard deviations (metres, radians).
    pub sigma_t: f64,
    pub sigma_r: f64,
}

impl Default for FindLoopsParams {
    fn default() -> Self {
        Self {
            max_distance: 10.0,
            drift: 0.03,
            min_travel: 30.0,
            spacing: 5.0,
            min_fitness: 0.5,
            inlier_distance: 0.5,
            retry_headings: 8,
            max_iterations: 50,
            overlap: 0.8,
            sigma_t: 0.1,
            sigma_r: 1f64.to_radians(),
        }
    }
}

/// A loop [`find_loops`] added.
#[derive(Debug, Clone, Copy)]
pub struct FoundLoop {
    pub from: usize,
    pub to: usize,
    pub fitness: f64,
    pub retried: bool,
    /// Its edge's index in the graph.
    pub edge: usize,
}

#[derive(Debug, Clone, Default)]
pub struct FoundLoops {
    pub candidates: usize,
    pub added: Vec<FoundLoop>,
    /// Registered well, but further off than the drift allows: left out.
    pub implausible: usize,
}

/// Find loops and add them to `graph` as edges (without optimising):
/// every candidate pair is registered, and kept when it overlaps at least
/// `min_fitness` and lands within the drift allowance of where the graph
/// has it.
pub fn find_loops(
    graph: &mut PoseGraph,
    scans: &[Option<PointCloud>],
    params: &FindLoopsParams,
) -> FoundLoops {
    let candidates = pose_graph::loop_candidates(
        graph,
        params.max_distance,
        params.drift,
        params.min_travel,
        params.spacing,
    );
    let settings = |travel: f64| LoopSettings {
        max_iterations: params.max_iterations,
        overlap: params.overlap,
        point_to_plane: true,
        inlier_distance: params.inlier_distance,
        retry_below: params.min_fitness,
        retry_headings: if params.drift * travel >= RETRY_MIN_DRIFT {
            params.retry_headings
        } else {
            0
        },
    };
    let register = |c: &pose_graph::LoopCandidate| {
        let (from, to) = (scans.get(c.from)?.as_ref()?, scans.get(c.to)?.as_ref()?);
        register_pair(from, to, &graph.relative(c.from, c.to), &settings(c.travel))
    };
    #[cfg(feature = "parallel")]
    let registered: Vec<Option<LoopRegistration>> = {
        use rayon::prelude::*;
        candidates.par_iter().map(register).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let registered: Vec<Option<LoopRegistration>> = candidates.iter().map(register).collect();

    let information = pose_graph::isotropic_information(params.sigma_t, params.sigma_r);
    let mut found = FoundLoops {
        candidates: candidates.len(),
        ..FoundLoops::default()
    };
    for (c, r) in candidates.iter().zip(registered) {
        let Some(r) = r.filter(|r| r.fitness >= params.min_fitness) else {
            continue;
        };
        if r.discrepancy > params.max_distance + params.drift * c.travel {
            found.implausible += 1;
            continue;
        }
        let edge = graph.add_loop(c.from, c.to, r.measurement, information);
        found.added.push(FoundLoop {
            from: c.from,
            to: c.to,
            fitness: r.fitness,
            retried: r.retried,
            edge,
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yawed(x: f64, y: f64, yaw: f64) -> Rigid {
        let (s, c) = yaw.sin_cos();
        Rigid {
            rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            translation: [x, y, 0.0],
        }
    }

    /// A courtyard: walls round a 40 m square with a few pillars, 3 m high.
    fn courtyard() -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for k in 0..160 {
            let t = k as f64 * 0.5 - 20.0;
            for z in 0..6 {
                let z = z as f64 * 0.5;
                out.extend([[t, -20.0, z], [t, 20.0, z], [-20.0, t, z], [20.0, t, z]]);
            }
        }
        for (px, py) in [
            (-8.0, -9.0),
            (7.0, -6.0),
            (9.0, 8.0),
            (-6.0, 7.0),
            (0.0, 12.0),
        ] {
            for a in 0..12 {
                let a = a as f64 * std::f64::consts::TAU / 12.0;
                for z in 0..6 {
                    out.push([px + 0.6 * a.cos(), py + 0.6 * a.sin(), z as f64 * 0.5]);
                }
            }
        }
        out
    }

    #[test]
    fn a_drifted_lap_is_closed_where_it_returns() {
        // A lap round a 24 m square, 3 m steps, starting and ending at the same spot.
        let mut truth = Vec::new();
        for side in 0..4 {
            let yaw = side as f64 * std::f64::consts::FRAC_PI_2;
            for step in 0..8 {
                let d = step as f64 * 3.0 - 12.0;
                let (x, y) = match side {
                    0 => (d, -12.0),
                    1 => (12.0, d),
                    2 => (-d, 12.0),
                    _ => (-12.0, -d),
                };
                truth.push(yawed(x, y, yaw));
            }
        }
        truth.push(truth[0]);
        let world = courtyard();
        let scans: Vec<Option<PointCloud>> = truth
            .iter()
            .map(|pose| {
                let back = pose_graph::inverse(pose);
                Some(PointCloud {
                    positions: world.iter().map(|p| back.apply(p)).collect(),
                    ..PointCloud::default()
                })
            })
            .collect();
        // Odometry that turns a little too much at every step.
        let mut drifted = vec![truth[0]];
        for k in 1..truth.len() {
            let step = pose_graph::inverse(&truth[k - 1]).compose(&truth[k]);
            let step = yawed(0.0, 0.0, 0.005).compose(&step);
            drifted.push(drifted[k - 1].compose(&step));
        }
        let mut graph = PoseGraph::from_poses(
            &drifted,
            pose_graph::isotropic_information(0.1, 1f64.to_radians()),
        );
        let found = find_loops(&mut graph, &scans, &FindLoopsParams::default());
        assert!(found.candidates > 0);
        let last = truth.len() - 1;
        let closing = found
            .added
            .iter()
            .find(|l| l.from <= 1 && l.to >= last - 1)
            .unwrap_or_else(|| panic!("the return to the start is not among {:?}", found.added));
        assert!(closing.fitness > 0.8, "{closing:?}");
        // The loop measures the truth: the lap's end sits where it began.
        let measured = graph.edges[closing.edge].measurement;
        let truth_relative = pose_graph::inverse(&truth[closing.from]).compose(&truth[closing.to]);
        let d: f64 = (0..3)
            .map(|k| (measured.translation[k] - truth_relative.translation[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(d < 0.05, "{d}");
    }
}
