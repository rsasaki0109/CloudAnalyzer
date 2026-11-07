//! LiDAR odometry, after KISS-ICP (Vizzo et al., 2023): each scan, thinned,
//! is registered onto a local voxel map of the scans before it, starting
//! from where a constant velocity puts it, by point-to-point ICP with a
//! robust (Geman-McClure) kernel and correspondences within a threshold
//! learnt from how far the motion model has been off; then it joins the
//! map. Small enough to run in the browser, so a recording without poses
//! can be opened there.

use crate::PointCloud;
use crate::icp::Rigid;
use crate::odometry_linearization::RegistrationLinearization;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

#[path = "odometry_target_balance.rs"]
pub(crate) mod target_balance;

// Real fixed-state timing is opt-in and excluded from library builds.
#[cfg(all(test, feature = "parallel"))]
#[path = "odometry_profile.rs"]
mod profile;

/// A fast hash for voxel keys (FxHash's step): the map is looked up millions
/// of times a scan, where the default hasher's resistance to attacks is not needed.
#[derive(Default)]
struct VoxelHasher(u64);

impl Hasher for VoxelHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    fn write_i64(&mut self, v: i64) {
        self.write_u64(v as u64);
    }

    fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type VoxelMap = HashMap<[i64; 3], Vec<[f64; 3]>, BuildHasherDefault<VoxelHasher>>;

#[derive(Debug, Clone, Copy)]
pub struct OdometryParams {
    /// Points nearer than this or further than `max_range` (metres) are left out.
    pub min_range: f64,
    pub max_range: f64,
    /// The local map keeps at most `map_points` points per voxel of this size (metres).
    pub map_voxel: f64,
    pub map_points: usize,
    pub max_iterations: usize,
    /// The spread of the motion model's error (metres) until it has been measured.
    pub initial_sigma: f64,
    /// Motion model errors smaller than this (metres) are not counted.
    pub min_motion: f64,
    /// Undo the motion during each scan (see [`sweep_times`]), taking the
    /// last motion for it: the pose is the sensor's halfway through the scan.
    /// Off by default: on the drives tried (a Segway at 1 m/s, an ATV at
    /// 3-5 m/s with a 10 Hz sensor) it made no measurable difference.
    pub deskew: bool,
    /// Registration stops once a step shifts less than `tolerance_t` metres
    /// and turns less than `tolerance_r` radians. Looser (1 mm, 0.1 mrad)
    /// is a third faster but leaves the map less consistent: on hdl_400.bag
    /// the visibility check then took 11 % of the points for dynamic, not 5 %.
    pub tolerance_t: f64,
    pub tolerance_r: f64,
    /// Measure final correspondence support and residuals before inserting the
    /// scan. Off by default: this keeps sensor points and adds one map query pass.
    pub diagnostics: bool,
    /// How much of the motion between scans a sweep takes when the scan's
    /// [`TIME`] attribute does not say: 1 for a spinning sensor recorded at
    /// its full rate, less when frames were skipped.
    pub sweep: f64,
}

impl Default for OdometryParams {
    fn default() -> Self {
        Self {
            min_range: 1.5,
            max_range: 80.0,
            map_voxel: 1.0,
            map_points: 20,
            max_iterations: 500,
            initial_sigma: 2.0,
            min_motion: 0.1,
            deskew: false,
            sweep: 1.0,
            tolerance_t: 1e-4,
            tolerance_r: 1e-5,
            diagnostics: false,
        }
    }
}

/// Odometry state: the local map, the poses so far and the motion model's error.
pub struct Odometry {
    params: OdometryParams,
    registration_voxel: Option<f64>,
    capture_linearization: bool,
    target_balance: bool,
    map: LocalMap,
    poses: Vec<Rigid>,
    error_sum: f64,
    error_count: usize,
    /// The world points the last scan added to the map (see [`Odometry::finish`]).
    placed: Vec<[f64; 3]>,
    last_report: Option<RegistrationReport>,
    last_linearization: Option<RegistrationLinearization>,
}

/// Why ICP stopped. A small update alone does not certify a correct alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistrationTermination {
    EmptyMap,
    InsufficientPoints,
    Converged,
    IterationLimit,
    SingularSystem,
    NonFiniteSystem,
    NonFiniteUpdate,
    /// The caller finished a registration before `step` reported completion.
    Unfinished,
}

impl RegistrationTermination {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EmptyMap => "empty_map",
            Self::InsufficientPoints => "insufficient_points",
            Self::Converged => "converged",
            Self::IterationLimit => "iteration_limit",
            Self::SingularSystem => "singular_system",
            Self::NonFiniteSystem => "non_finite_system",
            Self::NonFiniteUpdate => "non_finite_update",
            Self::Unfinished => "unfinished",
        }
    }
}

/// Unweighted Euclidean distances at the returned pose, against the local map
/// before insertion. Uses ICP's own 27-voxel search and distance threshold;
/// these are diagnostics, not calibrated uncertainty or a quality gate.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CorrespondenceStats {
    pub matched_points: usize,
    /// Matched points / registration points (including unmatched points).
    pub matched_fraction: f64,
    /// None when no correspondence passed the search and threshold.
    pub rmse_m: Option<f64>,
    pub max_residual_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RegistrationReport {
    pub termination: RegistrationTermination,
    /// Successfully applied Gauss-Newton updates, excluding failed solves.
    pub iterations: usize,
    pub input_points: usize,
    /// Points after range filtering, deskew and registration thinning, even if
    /// the empty map or fewer than ten points prevented alignment.
    pub registration_points: usize,
    pub threshold_m: f64,
    /// None when diagnostics are disabled or alignment was not attempted.
    pub correspondences: Option<CorrespondenceStats>,
}

#[derive(Debug, thiserror::Error)]
pub enum PredictionError {
    #[error("relative prediction requires a previously registered scan")]
    NoPreviousScan,
    #[error("relative prediction must be a finite proper rigid transform")]
    InvalidTransform,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("registration_voxel must be finite and positive")]
pub struct RegistrationVoxelError;

/// The local map: a voxel grid keeping at most so many points per voxel,
/// with the nearest-point search and the normal equations of a
/// registration step. A copy of it can live elsewhere (a worker), fed the
/// same [`LocalMap::insert`]s and [`LocalMap::retain_within`]s.
pub struct LocalMap {
    voxel: f64,
    max_points: usize,
    map: VoxelMap,
}

/// A scan's registration in progress: made by [`Odometry::begin`], moved a
/// step at a time by [`Odometry::step`] from the normal equations (from
/// [`Odometry::equations`], or summed from copies of the map elsewhere), and
/// finished by [`Odometry::finish`].
pub struct Registration {
    /// The scan thinned for the map, in the sensor's frame.
    frame: Vec<[f64; 3]>,
    /// The scan thinned for registering, in the world at the prediction; empty when there is nothing to register onto.
    source: Vec<[f64; 3]>,
    prediction: Rigid,
    /// The correction so far, applied to `source`.
    total: Rigid,
    center: [f64; 3],
    threshold_sq: f64,
    kernel: f64,
    target_balance: bool,
    iterations: usize,
    input_points: usize,
    registration_points: usize,
    termination: Option<RegistrationTermination>,
    /// Retained only for opt-in residuals at the exact, orthonormalized pose.
    diagnostic_source: Option<Vec<[f64; 3]>>,
    /// Independent opt-in retention; never enables or changes the old report.
    linearization_source: Option<Vec<[f64; 3]>>,
}

impl Registration {
    /// Whether the scan is registered at all (the map has points and the scan enough).
    pub fn needs_alignment(&self) -> bool {
        !self.source.is_empty()
    }

    // The default H/b path can fold one 256-point chunk on the caller.
    // Balanced search also splits within a chunk. Enter the pool once for the
    // remaining iterations when repeated work is needed. Keep an existing
    // caller's Rayon pool; completed or first-iteration work stays on the caller.
    #[cfg(feature = "parallel")]
    fn needs_alignment_worker(&self) -> bool {
        self.needs_alignment()
            && (self.target_balance || self.source.len() > 256)
            && self.iterations > 0
            && self.termination.is_none()
            && rayon::current_thread_index().is_none()
    }

    /// The points registered, in the world at the prediction.
    pub fn source(&self) -> &[[f64; 3]] {
        &self.source
    }

    /// The correction found so far (applied to the source points).
    pub fn total(&self) -> &Rigid {
        &self.total
    }

    /// Where the sensor is: rotations are taken about it.
    pub fn center(&self) -> [f64; 3] {
        self.center
    }

    /// Pairs further apart than this (squared) are left out.
    pub fn threshold_sq(&self) -> f64 {
        self.threshold_sq
    }

    /// The robust kernel's scale.
    pub fn kernel(&self) -> f64 {
        self.kernel
    }

    /// None while steps are still required. A finish before completion is
    /// recorded as `unfinished` in the final report.
    pub fn termination(&self) -> Option<RegistrationTermination> {
        self.termination
    }
}

type Vec6 = [f64; 6];
type Mat6 = [[f64; 6]; 6];

/// `a x = b` by elimination with partial pivoting.
fn solve6(mut a: Mat6, mut b: Vec6) -> Result<Vec6, RegistrationTermination> {
    if !a.iter().flatten().chain(&b).all(|v| v.is_finite()) {
        return Err(RegistrationTermination::NonFiniteSystem);
    }
    for col in 0..6 {
        let pivot = (col..6)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .expect("nonempty pivot column");
        if !a[pivot][col].is_finite() {
            return Err(RegistrationTermination::NonFiniteSystem);
        }
        if a[pivot][col].abs() < 1e-12 {
            return Err(RegistrationTermination::SingularSystem);
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let top = a[col];
        for row in col + 1..6 {
            let f = a[row][col] / top[col];
            for (x, t) in a[row].iter_mut().zip(top).skip(col) {
                *x -= f * t;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 6];
    for row in (0..6).rev() {
        let s: f64 = (row + 1..6).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    if x.iter().all(|v| v.is_finite()) {
        Ok(x)
    } else {
        Err(RegistrationTermination::NonFiniteSystem)
    }
}

/// One point per voxel (the first in each).
fn thin(points: &[[f64; 3]], voxel: f64) -> Vec<[f64; 3]> {
    let mut seen: HashSet<[i64; 3], BuildHasherDefault<VoxelHasher>> = HashSet::default();
    points
        .iter()
        .copied()
        .filter(|p| seen.insert(p.map(|v| (v / voxel).floor() as i64)))
        .collect()
}

/// `r` made a rotation again (Gram-Schmidt on its rows): products of
/// rotations drift from one, and constant velocity, which multiplies by an
/// inverse taken as the transpose, would let that grow without end.
fn orthonormal(r: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let unit = |v: [f64; 3]| {
        let n = norm(&v);
        v.map(|x| x / n)
    };
    let x = unit(r[0]);
    let d = x[0] * r[1][0] + x[1] * r[1][1] + x[2] * r[1][2];
    let y = unit(std::array::from_fn(|k| r[1][k] - d * x[k]));
    let z = [
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    ];
    [x, y, z]
}

/// The name of a per-point attribute giving when in its scan each point
/// was taken (any unit; only the order and spacing count).
pub const TIME: &str = "time";

/// How far through its sweep the scan took each point, 0 to 1: from the
/// scan's [`TIME`] attribute when it has one, else from the azimuths, when
/// the points come in the order a spinning sensor takes them (one turn from
/// first to last; a scan sorted by ring, or already corrected, like
/// KITTI's, gives many turns and None).
pub fn sweep_times(scan: &PointCloud) -> Option<Vec<f32>> {
    if let Some(crate::AttributeValues::F32(t)) = scan.attribute(TIME).map(|a| &a.values) {
        let (lo, hi) = t
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        return (hi > lo).then(|| t.iter().map(|v| (v - lo) / (hi - lo)).collect());
    }
    let mut turned = Vec::with_capacity(scan.len());
    let mut total = 0.0;
    let mut last = None;
    for p in &scan.positions {
        let azimuth = p[1].atan2(p[0]);
        if let Some(prev) = last {
            let mut step = azimuth - prev;
            if step > std::f64::consts::PI {
                step -= std::f64::consts::TAU;
            } else if step < -std::f64::consts::PI {
                step += std::f64::consts::TAU;
            }
            total += step;
        }
        last = Some(azimuth);
        turned.push(total);
    }
    let turns = total.abs() / std::f64::consts::TAU;
    if !(0.8..1.2).contains(&turns) {
        return None;
    }
    Some(
        turned
            .iter()
            .map(|t| (t / total).clamp(0.0, 1.0) as f32)
            .collect(),
    )
}

/// `point` taken `fraction` of a scan after its middle, with the sensor
/// moving by `motion` over the scan, moved to where the sensor was at the
/// middle (a constant velocity: the motion scaled to the fraction).
fn deskewed(point: &[f64; 3], fraction: f64, motion: &Rigid, spin: &[f64; 3]) -> [f64; 3] {
    let part = Rigid {
        rotation: crate::pose_graph::exp_so3(&spin.map(|w| w * fraction)),
        translation: motion.translation.map(|t| t * fraction),
    };
    part.apply(point)
}

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

impl LocalMap {
    pub fn new(voxel: f64, max_points: usize) -> Self {
        LocalMap {
            voxel,
            max_points,
            map: VoxelMap::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// How many points the map holds.
    pub fn len(&self) -> usize {
        self.map.values().map(Vec::len).sum()
    }

    /// Add world points, each to its voxel while the voxel has room.
    pub fn insert(&mut self, points: &[[f64; 3]]) {
        for w in points {
            let cell = self
                .map
                .entry(w.map(|v| (v / self.voxel).floor() as i64))
                .or_default();
            if cell.len() < self.max_points {
                cell.push(*w);
            }
        }
    }

    /// Drop the voxels further than `range` from `origin`.
    pub fn retain_within(&mut self, origin: [f64; 3], range: f64) {
        self.map.retain(|_, points| {
            let q = points[0];
            norm(&[q[0] - origin[0], q[1] - origin[1], q[2] - origin[2]]) <= range
        });
    }

    /// The map point nearest `q` in its voxel and the 26 around it, with the
    /// squared distance; a voxel is skipped when it is no nearer than the best so far.
    #[inline(always)]
    fn nearest(&self, q: &[f64; 3]) -> Option<([f64; 3], f64)> {
        let v = self.voxel;
        let key = q.map(|x| (x / v).floor() as i64);
        // Each neighboring voxel uses the same six face distances. Keep the
        // original axis order when summing their squared lower bounds below.
        let gaps: [[f64; 3]; 3] = std::array::from_fn(|k| {
            let lower = q[k] - key[k] as f64 * v;
            let upper = (key[k] + 1) as f64 * v - q[k];
            [lower * lower, 0.0, upper * upper]
        });
        let mut best = ([0.0; 3], f64::INFINITY);
        let search = |cell: [i64; 3], best: &mut ([f64; 3], f64)| {
            for m in self.map.get(&cell).into_iter().flatten() {
                let d = (m[0] - q[0]).powi(2) + (m[1] - q[1]).powi(2) + (m[2] - q[2]).powi(2);
                if d < best.1 {
                    *best = (*m, d);
                }
            }
        };
        search(key, &mut best);
        for dx in -1..=1_i64 {
            let gap_x = 0.0 + gaps[0][(dx + 1) as usize];
            if gap_x >= best.1 {
                continue;
            }
            for dy in -1..=1_i64 {
                let gap_xy = gap_x + gaps[1][(dy + 1) as usize];
                if gap_xy >= best.1 {
                    continue;
                }
                for dz in -1..=1_i64 {
                    let offset = [dx, dy, dz];
                    if offset == [0, 0, 0] {
                        continue;
                    }
                    let gap = gap_xy + gaps[2][(dz + 1) as usize];
                    if gap < best.1 {
                        search(std::array::from_fn(|k| key[k] + offset[k]), &mut best);
                    }
                }
            }
        }
        best.1.is_finite().then_some(best)
    }

    fn correspondence_stats(
        &self,
        source: &[[f64; 3]],
        pose: &Rigid,
        threshold_sq: f64,
    ) -> CorrespondenceStats {
        let mut matched_points = 0;
        let mut sum_sq = 0.0;
        let mut max_sq: f64 = 0.0;
        for p in source {
            let Some((_, distance_sq)) = self.nearest(&pose.apply(p)) else {
                continue;
            };
            if distance_sq > threshold_sq {
                continue;
            }
            matched_points += 1;
            sum_sq += distance_sq;
            max_sq = max_sq.max(distance_sq);
        }
        CorrespondenceStats {
            matched_points,
            matched_fraction: matched_points as f64 / source.len() as f64,
            rmse_m: (matched_points != 0).then(|| (sum_sq / matched_points as f64).sqrt()),
            max_residual_m: (matched_points != 0).then(|| max_sq.sqrt()),
        }
    }

    fn capture_linearization(
        &self,
        source: Vec<[f64; 3]>,
        pose: Rigid,
        registration: &Registration,
        max_range: f64,
    ) -> RegistrationLinearization {
        let source_world: Vec<_> = source.iter().map(|p| pose.apply(p)).collect();
        let mut snapshot = RegistrationLinearization {
            pose,
            center_world: registration.center,
            threshold_sq: registration.threshold_sq,
            kernel: registration.kernel,
            target_balance: registration.target_balance,
            map_voxel: self.voxel,
            map_points: self.max_points,
            max_range,
            alignment_eligible: registration.needs_alignment(),
            source_sensor: source,
            source_world,
            matched_source_indices: Vec::new(),
            target_world: Vec::new(),
            squared_distances: Vec::new(),
            weights: Vec::new(),
            robust_weights: Vec::new(),
            target_multiplicity: Vec::new(),
            normal_matrix: [[0.0; 6]; 6],
            right_hand_side: [0.0; 6],
            weight_sum: 0.0,
            weighted_rms_radius_m: None,
            inserted_world: Vec::new(),
            finite: true,
        };
        snapshot.finite = snapshot
            .source_world
            .iter()
            .flatten()
            .chain(pose.rotation.iter().flatten())
            .chain(&pose.translation)
            .chain(&snapshot.center_world)
            .all(|v| v.is_finite())
            && [
                snapshot.threshold_sq,
                snapshot.kernel,
                self.voxel,
                max_range,
            ]
            .iter()
            .all(|v| v.is_finite());
        if snapshot.alignment_eligible {
            for (index, q) in snapshot.source_world.iter().enumerate() {
                // Diagnostic capture must not introduce a search on nonfinite coordinates.
                if !q.iter().all(|v| v.is_finite()) {
                    continue;
                }
                let Some((target, distance_sq)) = self.nearest(q) else {
                    continue;
                };
                if distance_sq > snapshot.threshold_sq {
                    continue;
                }
                snapshot.matched_source_indices.push(index);
                snapshot.target_world.push(target);
                snapshot.squared_distances.push(distance_sq);
            }
        }
        snapshot.accumulate();
        snapshot
    }

    /// The normal equations of one Gauss-Newton step: each point of `source`
    /// moved by `total`, paired with the map point nearest it within the
    /// threshold, weighted by a Geman-McClure kernel of scale `kernel`;
    /// rotations about `center`. Summed over chunks of points (in parallel
    /// on the cores, natively).
    pub fn normal_equations(
        &self,
        source: &[[f64; 3]],
        total: &Rigid,
        center: [f64; 3],
        threshold_sq: f64,
        kernel: f64,
    ) -> (Mat6, Vec6) {
        let chunk = |points: &[[f64; 3]]| -> (Mat6, Vec6) {
            let mut jtj = [[0.0; 6]; 6];
            let mut jtr = [0.0; 6];
            for p in points {
                let q = total.apply(p);
                let Some((m, distance_sq)) = self.nearest(&q) else {
                    continue;
                };
                if distance_sq > threshold_sq {
                    continue;
                }
                let r = [q[0] - m[0], q[1] - m[1], q[2] - m[2]];
                let w = kernel * kernel / (kernel + distance_sq).powi(2);
                // d q / d (translation, rotation) = [I, -[q - center]x] for a change on the left.
                let q = [q[0] - center[0], q[1] - center[1], q[2] - center[2]];
                let rows = [
                    [1.0, 0.0, 0.0, 0.0, q[2], -q[1]],
                    [0.0, 1.0, 0.0, -q[2], 0.0, q[0]],
                    [0.0, 0.0, 1.0, q[1], -q[0], 0.0],
                ];
                // Each Jacobian row has only three structurally nonzero entries.
                // Preserve the row order and both matrix triangles: skipping zero
                // terms avoids 81 of the 108 products per correspondence without
                // changing the accumulation order of the remaining terms.
                for ((row, r), indices) in rows.iter().zip(r).zip([[0, 4, 5], [1, 3, 5], [2, 3, 4]])
                {
                    for i in indices {
                        jtr[i] -= w * row[i] * r;
                        for j in indices {
                            jtj[i][j] += w * row[i] * row[j];
                        }
                    }
                }
            }
            (jtj, jtr)
        };
        let add = |(mut a, mut b): (Mat6, Vec6), (c, d): (Mat6, Vec6)| {
            for i in 0..6 {
                b[i] += d[i];
                for j in 0..6 {
                    a[i][j] += c[i][j];
                }
            }
            (a, b)
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            source
                .par_chunks(256)
                .map(chunk)
                .reduce(|| ([[0.0; 6]; 6], [0.0; 6]), add)
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = add;
            chunk(source)
        }
    }
}

impl Odometry {
    pub fn new(params: OdometryParams) -> Self {
        Odometry {
            params,
            registration_voxel: None,
            capture_linearization: false,
            target_balance: false,
            map: LocalMap::new(params.map_voxel, params.map_points),
            poses: Vec::new(),
            error_sum: 0.0,
            error_count: 0,
            placed: Vec::new(),
            last_report: None,
            last_linearization: None,
        }
    }

    pub fn params(&self) -> &OdometryParams {
        &self.params
    }

    pub fn poses(&self) -> &[Rigid] {
        &self.poses
    }

    pub fn map(&self) -> &LocalMap {
        &self.map
    }

    /// The world points the last finished scan added to the map.
    pub fn placed(&self) -> &[[f64; 3]] {
        &self.placed
    }

    /// Gauss-Newton steps of the last registration; zero without alignment.
    pub fn last_iterations(&self) -> usize {
        self.last_report.as_ref().map_or(0, |r| r.iterations)
    }

    /// The last finished scan, or None before the first scan. Residuals are
    /// opt-in; termination and counts are always recorded.
    pub fn last_report(&self) -> Option<&RegistrationReport> {
        self.last_report.as_ref()
    }

    /// Enable final residuals for registrations begun after this call.
    pub fn set_diagnostics(&mut self, enabled: bool) {
        self.params.diagnostics = enabled;
    }

    /// Retain a separate final-pose snapshot for registrations begun afterward.
    /// Existing pending registrations own their capture choice and source points.
    pub fn set_capture_linearization(&mut self, enabled: bool) {
        self.capture_linearization = enabled;
    }

    pub fn capture_linearization(&self) -> bool {
        self.capture_linearization
    }

    /// Experimental inverse-target-multiplicity weights, off by default.
    /// Applies to subsequent registrations; pending registrations retain their policy.
    pub fn set_target_balance(&mut self, enabled: bool) {
        self.target_balance = enabled;
    }

    pub fn target_balance(&self) -> bool {
        self.target_balance
    }

    /// Owned final-pose data captured before insertion; None before the first
    /// finish or when capture was disabled at that registration's begin.
    pub fn last_linearization(&self) -> Option<&RegistrationLinearization> {
        self.last_linearization.as_ref()
    }

    /// Explicit source-sampling voxel in metres, or None for legacy two-stage thinning.
    pub fn registration_voxel(&self) -> Option<f64> {
        self.registration_voxel
    }

    /// Sample the range-filtered, deskewed points independently of map insertion.
    /// Applies to subsequent `begin` calls; existing registrations retain their points.
    /// None restores the original two-stage path. Invalid values leave all state intact.
    pub fn set_registration_voxel(
        &mut self,
        voxel: Option<f64>,
    ) -> Result<(), RegistrationVoxelError> {
        if voxel.is_some_and(|v| !v.is_finite() || v <= 0.0) {
            return Err(RegistrationVoxelError);
        }
        self.registration_voxel = voxel;
        Ok(())
    }

    fn sigma(&self) -> f64 {
        if self.error_count == 0 {
            self.params.initial_sigma
        } else {
            (self.error_sum / self.error_count as f64).sqrt()
        }
    }

    /// Start registering the next scan (in its sensor's frame): filtered,
    /// deskewed, thinned and put where a constant velocity predicts it.
    pub fn begin(&self, scan: &PointCloud) -> Registration {
        self.begin_scan(scan, None)
    }

    /// Start with a relative pose mapping the new scan into the previous scan's
    /// frame: `prediction = last_pose * relative_prediction`. Only the ICP
    /// initial pose changes; deskew still uses the ordinary motion model.
    /// Invalid predictions leave all odometry state untouched.
    pub fn begin_with_prediction(
        &self,
        scan: &PointCloud,
        relative_prediction: &Rigid,
    ) -> Result<Registration, PredictionError> {
        if self.poses.is_empty() {
            return Err(PredictionError::NoPreviousScan);
        }
        let r = &relative_prediction.rotation;
        let determinant = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        if !r
            .iter()
            .flatten()
            .chain(&relative_prediction.translation)
            .all(|v| v.is_finite())
            || (determinant - 1.0).abs() > 1e-6
            || !determinant.is_finite()
            || (0..3).any(|i| {
                (0..3).any(|j| {
                    let dot: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                    (dot - f64::from(u8::from(i == j))).abs() > 1e-6
                })
            })
        {
            return Err(PredictionError::InvalidTransform);
        }
        Ok(self.begin_scan(scan, Some(relative_prediction)))
    }

    fn begin_scan(&self, scan: &PointCloud, relative_prediction: Option<&Rigid>) -> Registration {
        let p = self.params;
        // Constant velocity: the last motion again.
        let n = self.poses.len();
        let motion = match n {
            0 | 1 => Rigid::IDENTITY,
            _ => crate::pose_graph::inverse(&self.poses[n - 2]).compose(&self.poses[n - 1]),
        };
        let times = if p.deskew && n >= 2 {
            sweep_times(scan)
        } else {
            None
        };
        let spin = crate::pose_graph::log_so3(&motion.rotation);
        let near: Vec<[f64; 3]> = scan
            .positions
            .iter()
            .enumerate()
            .filter(|(_, q)| (p.min_range..=p.max_range).contains(&norm(q)))
            .map(|(k, q)| match &times {
                Some(t) => deskewed(q, (f64::from(t[k]) - 0.5) * p.sweep, &motion, &spin),
                None => *q,
            })
            .collect();
        // Keep insertion independent and preserve the exact legacy two-stage default.
        let frame = thin(&near, 0.5 * p.map_voxel);
        let thinned = match self.registration_voxel {
            Some(voxel) => thin(&near, voxel),
            None => thin(&frame, 1.5 * p.map_voxel),
        };
        let prediction = match n {
            0 => Rigid::IDENTITY,
            _ => self.poses[n - 1].compose(relative_prediction.unwrap_or(&motion)),
        };
        let source = if self.map.is_empty() || thinned.len() < 10 {
            Vec::new()
        } else {
            thinned.iter().map(|q| prediction.apply(q)).collect()
        };
        let registration_points = thinned.len();
        let termination = if self.map.is_empty() {
            Some(RegistrationTermination::EmptyMap)
        } else if thinned.len() < 10 {
            Some(RegistrationTermination::InsufficientPoints)
        } else {
            None
        };
        let linearization_source = self.capture_linearization.then(|| thinned.clone());
        let diagnostic_source = (p.diagnostics && !source.is_empty()).then_some(thinned);
        let sigma = self.sigma();
        Registration {
            frame,
            source,
            prediction,
            total: Rigid::IDENTITY,
            center: prediction.translation,
            // Pairs within 3 sigma, weighted by a Geman-McClure kernel of sigma / 3.
            threshold_sq: (3.0 * sigma).powi(2),
            kernel: sigma / 3.0,
            target_balance: self.target_balance,
            iterations: 0,
            input_points: scan.len(),
            registration_points,
            termination,
            diagnostic_source,
            linearization_source,
        }
    }

    /// The normal equations of the registration's next step, from this map.
    pub fn equations(&self, registration: &Registration) -> (Mat6, Vec6) {
        let r = registration;
        if r.target_balance {
            return target_balance::equations(
                &self.map,
                &r.source,
                &r.total,
                r.center,
                r.threshold_sq,
                r.kernel,
            );
        }
        self.map
            .normal_equations(&r.source, &r.total, r.center, r.threshold_sq, r.kernel)
    }

    /// Take one Gauss-Newton step from the normal equations (from
    /// [`Odometry::equations`], or summed over copies of the map). True when
    /// the registration is done: converged, or out of iterations, or with
    /// nothing to solve. Rotations are about the sensor: about the world's
    /// origin, far from it, a small turn would come with a large shift, and
    /// the steps would shrink below the tolerance only after many iterations.
    pub fn step(&self, registration: &mut Registration, (jtj, jtr): (Mat6, Vec6)) -> bool {
        let r = registration;
        if r.termination.is_some() {
            return true;
        }
        let dx = match solve6(jtj, jtr) {
            Ok(dx) => dx,
            Err(reason) => {
                r.termination = Some(reason);
                return true;
            }
        };
        // Turn about the center, then shift.
        let rotation = crate::pose_graph::exp_so3(&[dx[3], dx[4], dx[5]]);
        let turned = Rigid {
            rotation,
            translation: [0.0; 3],
        }
        .apply(&r.center);
        let step = Rigid {
            rotation,
            translation: std::array::from_fn(|k| r.center[k] - turned[k] + dx[k]),
        };
        let total = step.compose(&r.total);
        if !total
            .rotation
            .iter()
            .flatten()
            .chain(&total.translation)
            .all(|v| v.is_finite())
        {
            r.termination = Some(RegistrationTermination::NonFiniteUpdate);
            return true;
        }
        r.total = total;
        r.iterations += 1;
        // Converged: steps of under the tolerances.
        let shift = dx[..3].iter().map(|v| v * v).sum::<f64>().sqrt();
        let turn = dx[3..].iter().map(|v| v * v).sum::<f64>().sqrt();
        r.termination = if shift < self.params.tolerance_t && turn < self.params.tolerance_r {
            Some(RegistrationTermination::Converged)
        } else if r.iterations >= self.params.max_iterations {
            Some(RegistrationTermination::IterationLimit)
        } else {
            None
        };
        r.termination.is_some()
    }

    /// Finish the registration: the scan's pose, learnt from for the
    /// threshold, and the scan joins the map (see [`Odometry::placed`]).
    pub fn finish(&mut self, registration: Registration) -> Rigid {
        let p = self.params;
        let mut r = registration;
        let mut pose = r.total.compose(&r.prediction);
        pose.rotation = orthonormal(&pose.rotation);
        self.last_linearization = r.linearization_source.take().map(|source| {
            self.map
                .capture_linearization(source, pose, &r, p.max_range)
        });
        self.last_report = Some(RegistrationReport {
            termination: r.termination.unwrap_or(RegistrationTermination::Unfinished),
            iterations: r.iterations,
            input_points: r.input_points,
            registration_points: r.registration_points,
            threshold_m: r.threshold_sq.sqrt(),
            correspondences: r
                .diagnostic_source
                .as_ref()
                .map(|source| self.map.correspondence_stats(source, &pose, r.threshold_sq)),
        });
        // How far the motion model was off, as the most a point in range moved.
        let deviation = crate::pose_graph::inverse(&r.prediction).compose(&pose);
        let m = &deviation.rotation;
        let theta = ((m[0][0] + m[1][1] + m[2][2] - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos();
        let error = 2.0 * p.max_range * (theta / 2.0).sin() + norm(&deviation.translation);
        if error > p.min_motion {
            self.error_sum += error * error;
            self.error_count += 1;
        }
        // The scan joins the map; voxels out of the sensor's range leave it.
        self.placed = r.frame.iter().map(|q| pose.apply(q)).collect();
        if let Some(snapshot) = &mut self.last_linearization {
            snapshot.inserted_world = self.placed.clone();
            snapshot.finite &= snapshot
                .inserted_world
                .iter()
                .flatten()
                .all(|v| v.is_finite());
        }
        self.map.insert(&self.placed);
        self.map.retain_within(pose.translation, p.max_range);
        self.poses.push(pose);
        pose
    }

    /// Register the next scan (in its sensor's frame) and return its pose.
    pub fn register(&mut self, scan: &PointCloud) -> Rigid {
        let registration = self.begin(scan);
        self.complete(registration)
    }

    /// Register with an externally supplied relative initial pose. See
    /// [`Odometry::begin_with_prediction`] for its frame and deskew conventions.
    pub fn register_with_prediction(
        &mut self,
        scan: &PointCloud,
        relative_prediction: &Rigid,
    ) -> Result<Rigid, PredictionError> {
        let registration = self.begin_with_prediction(scan, relative_prediction)?;
        Ok(self.complete(registration))
    }

    // Keep pool-entry machinery out of small caller-side registrations.
    #[cfg(feature = "parallel")]
    #[inline(never)]
    fn align_on_worker(&self, registration: &mut Registration) {
        rayon::scope(|_| {
            loop {
                let equations = self.equations(registration);
                if self.step(registration, equations) {
                    break;
                }
            }
        });
    }

    fn complete(&mut self, mut registration: Registration) -> Rigid {
        if registration.needs_alignment() {
            loop {
                let equations = self.equations(&registration);
                if self.step(&mut registration, equations) {
                    break;
                }
                #[cfg(feature = "parallel")]
                if registration.needs_alignment_worker() {
                    self.align_on_worker(&mut registration);
                    break;
                }
            }
        }
        self.finish(registration)
    }
}

#[cfg(test)]
mod tests;
