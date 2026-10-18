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
            for dy in -1..=1_i64 {
                for dz in -1..=1_i64 {
                    let offset = [dx, dy, dz];
                    if offset == [0, 0, 0] {
                        continue;
                    }
                    // How far `q` is from that voxel.
                    let gap: f64 = (0..3).map(|k| gaps[k][(offset[k] + 1) as usize]).sum();
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
                for (row, r) in rows.iter().zip(r) {
                    for i in 0..6 {
                        jtr[i] -= w * row[i] * r;
                        for j in 0..6 {
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
mod tests {
    use super::*;

    // Exhaust all 27 cells in the same center-first order, without bounds.
    // This checks target choice and distance bits independently of pruning.
    fn ordered_neighborhood(map: &LocalMap, q: &[f64; 3]) -> Option<([f64; 3], f64)> {
        let key = q.map(|x| (x / map.voxel).floor() as i64);
        let mut cells = vec![key];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if [dx, dy, dz] != [0; 3] {
                        cells.push([key[0] + dx, key[1] + dy, key[2] + dz]);
                    }
                }
            }
        }
        let mut best = ([0.0; 3], f64::INFINITY);
        for cell in cells {
            for m in map.map.get(&cell).into_iter().flatten() {
                let distance =
                    (m[0] - q[0]).powi(2) + (m[1] - q[1]).powi(2) + (m[2] - q[2]).powi(2);
                if distance < best.1 {
                    best = (*m, distance);
                }
            }
        }
        best.1.is_finite().then_some(best)
    }

    fn assert_nearest_bits(map: &LocalMap, q: [f64; 3]) {
        let bits = |hit: Option<([f64; 3], f64)>| {
            hit.map(|(point, distance)| (point.map(f64::to_bits), distance.to_bits()))
        };
        assert_eq!(bits(map.nearest(&q)), bits(ordered_neighborhood(map, &q)));
    }

    #[test]
    fn nearest_face_bounds_match_ordered_sparse_neighborhoods() {
        for voxel in [0.1875, 0.75, 1.0, 4.0] {
            let mut map = LocalMap::new(voxel, 20);
            for x in -3..=3 {
                for y in -3..=3 {
                    for z in -3..=3 {
                        if (x + 2 * y + 3 * z) % 3 == 0 {
                            for fraction in [0.125, 0.5, 0.875] {
                                map.insert(&[[
                                    (f64::from(x) + fraction) * voxel,
                                    (f64::from(y) + fraction) * voxel,
                                    (f64::from(z) + fraction) * voxel,
                                ]]);
                            }
                        }
                    }
                }
            }
            let mut state = 1_u64;
            for _ in 0..4096 {
                let query = std::array::from_fn(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 32) as f64 / u32::MAX as f64 * 10.0 - 5.0) * voxel
                });
                assert_nearest_bits(&map, query);
            }
            assert_nearest_bits(&map, [30.0 * voxel; 3]);
            assert_nearest_bits(&LocalMap::new(voxel, 20), [0.0; 3]);
        }
    }

    #[test]
    fn nearest_face_bounds_preserve_boundaries_ties_and_signed_zero() {
        for voxel in [0.1875, 0.75, 1.0, 4.0] {
            let mut map = LocalMap::new(voxel, 20);
            map.insert(&[[-0.0, 0.0, -0.0], [0.0; 3]]);
            for x in -2..=2 {
                for y in -2..=2 {
                    for z in -2..=2 {
                        map.insert(&[[
                            f64::from(x) * voxel,
                            f64::from(y) * voxel,
                            f64::from(z) * voxel,
                        ]]);
                    }
                }
            }
            for cell in -3..=3 {
                let boundary = f64::from(cell) * voxel;
                for value in [
                    boundary.next_down(),
                    boundary,
                    boundary.next_up(),
                    boundary + 0.5 * voxel,
                ] {
                    for axis in 0..3 {
                        let mut query = [-0.0, 0.0, -0.0];
                        query[axis] = value;
                        assert_nearest_bits(&map, query);
                    }
                }
            }
            assert_eq!(
                map.nearest(&[0.0; 3]).unwrap().0.map(f64::to_bits),
                [-0.0, 0.0, -0.0].map(f64::to_bits)
            );
            assert_nearest_bits(&map, [f64::NAN, 0.0, 0.0]);
        }
    }

    fn sampling_cloud() -> PointCloud {
        PointCloud {
            positions: (0..12)
                .flat_map(|i| {
                    (0..4).map(move |j| {
                        [
                            3.0 * (i % 4) as f64 - 6.0 + 0.03125 + 0.09375 * j as f64,
                            3.0 * (i / 4) as f64 + 2.03125,
                            (i % 3) as f64 + 0.03125,
                        ]
                    })
                })
                .collect(),
            ..PointCloud::default()
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn alignment_worker_gate_uses_captured_policy_and_existing_chunk_boundary() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            ..OdometryParams::default()
        });
        let scan = sampling_cloud();
        odometry.set_target_balance(true);
        assert!(!odometry.begin(&scan).needs_alignment_worker()); // Empty map.
        odometry.register(&scan);
        let mut balanced = odometry.begin(&scan);
        odometry.set_target_balance(false);
        assert!(!balanced.needs_alignment_worker()); // First iteration stays here.
        balanced.iterations = 1;
        assert!(balanced.needs_alignment_worker()); // Choice belongs to begin.
        balanced.termination = Some(RegistrationTermination::Converged);
        assert!(!balanced.needs_alignment_worker());
        balanced.termination = None;
        balanced.source.clear();
        assert!(!balanced.needs_alignment_worker());
        let mut default = odometry.begin(&scan);
        default.iterations = 1;
        odometry.set_target_balance(true);
        for count in [0, 10, 255, 256, 257] {
            default.source.resize(count, [1.0, 2.0, 3.0]);
            assert_eq!(default.needs_alignment_worker(), count > 256);
        }
        assert!(
            !odometry
                .begin(&PointCloud::default())
                .needs_alignment_worker()
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn alignment_worker_gate_preserves_nested_local_pools() {
        let mut odometry = Odometry::new(OdometryParams::default());
        let scan = sampling_cloud();
        odometry.register(&scan);
        let mut registration = odometry.begin(&scan);
        registration.source.resize(257, [1.0, 2.0, 3.0]);
        registration.iterations = 1;
        assert!(registration.needs_alignment_worker());
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                assert_eq!(rayon::current_num_threads(), workers);
                assert!(!registration.needs_alignment_worker());
                rayon::scope(|scope| {
                    scope.spawn(|_| {
                        assert_eq!(rayon::current_num_threads(), workers);
                        assert!(!registration.needs_alignment_worker());
                    })
                });
            });
        }
    }

    #[test]
    fn automatic_alignment_preserves_stepped_state_for_both_policies_and_sizes() {
        for dense in [false, true] {
            for balanced in [false, true] {
                for predicted in [false, true] {
                    let params = OdometryParams {
                        min_range: 0.0,
                        max_iterations: 4,
                        diagnostics: true,
                        ..OdometryParams::default()
                    };
                    let mut automatic = Odometry::new(params);
                    let mut stepped = Odometry::new(params);
                    for odometry in [&mut automatic, &mut stepped] {
                        odometry.set_target_balance(balanced);
                        odometry.set_registration_voxel(Some(0.125)).unwrap();
                        odometry.set_capture_linearization(true);
                    }
                    let mut scan = if dense {
                        PointCloud {
                            positions: (0..12)
                                .flat_map(|i| {
                                    (0..12).flat_map(move |j| {
                                        (0..3).map(move |k| {
                                            [
                                                2.0 * i as f64 + 2.0,
                                                2.0 * j as f64 - 10.0,
                                                2.0 * k as f64 - 2.0 + 0.125 * (i % 3) as f64,
                                            ]
                                        })
                                    })
                                })
                                .collect(),
                            ..PointCloud::default()
                        }
                    } else {
                        sampling_cloud()
                    };
                    for frame in 0..4 {
                        let mut r = if predicted && frame > 0 {
                            stepped
                                .begin_with_prediction(&scan, &Rigid::IDENTITY)
                                .unwrap()
                        } else {
                            stepped.begin(&scan)
                        };
                        if frame > 0 {
                            assert_eq!(r.source().len() > 256, dense);
                        }
                        if r.needs_alignment() {
                            loop {
                                let equations = stepped.equations(&r);
                                if stepped.step(&mut r, equations) {
                                    break;
                                }
                            }
                        }
                        let expected = stepped.finish(r);
                        let actual = if predicted && frame > 0 {
                            automatic
                                .register_with_prediction(&scan, &Rigid::IDENTITY)
                                .unwrap()
                        } else {
                            automatic.register(&scan)
                        };
                        assert_eq!(
                            actual.to_matrix().map(f64::to_bits),
                            expected.to_matrix().map(f64::to_bits)
                        );
                        assert_eq!(automatic.last_report(), stepped.last_report());
                        assert_eq!(automatic.map.map, stepped.map.map);
                        assert_eq!(automatic.placed(), stepped.placed());
                        assert_eq!(automatic.error_sum.to_bits(), stepped.error_sum.to_bits());
                        assert_eq!(automatic.error_count, stepped.error_count);
                        let a = automatic.last_linearization().unwrap();
                        let b = stepped.last_linearization().unwrap();
                        assert_eq!(
                            a.normal_matrix.map(|row| row.map(f64::to_bits)),
                            b.normal_matrix.map(|row| row.map(f64::to_bits))
                        );
                        assert_eq!(
                            a.right_hand_side.map(f64::to_bits),
                            b.right_hand_side.map(f64::to_bits)
                        );
                        assert_eq!(a.matched_source_indices, b.matched_source_indices);
                        assert_eq!(
                            a.weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            b.weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                        assert_eq!(a.target_multiplicity, b.target_multiplicity);
                        for point in &mut scan.positions {
                            point[0] -= 0.03125;
                            point[1] += 0.015625;
                        }
                    }
                    let empty = PointCloud::default();
                    assert_eq!(automatic.register(&empty), stepped.register(&empty));
                    assert_eq!(automatic.last_report(), stepped.last_report());
                }
            }
        }
    }

    #[test]
    fn registration_sampling_is_independent_of_map_insertion() {
        let params = OdometryParams {
            min_range: 0.0,
            diagnostics: true,
            ..OdometryParams::default()
        };
        let mut legacy = Odometry::new(params);
        let mut dense = Odometry::new(params);
        dense.set_registration_voxel(Some(0.0625)).unwrap();
        let cloud = sampling_cloud();
        assert_eq!(
            legacy.register(&cloud).to_matrix(),
            dense.register(&cloud).to_matrix()
        );
        assert_eq!(legacy.placed(), dense.placed());
        assert_eq!(legacy.placed().len(), 12);
        assert_eq!(legacy.map.map, dense.map.map);
        assert_eq!(legacy.map.voxel, dense.map.voxel);
        assert_eq!(legacy.map.max_points, dense.map.max_points);
        let sparse = legacy.begin(&cloud);
        let full = dense.begin(&cloud);
        assert_eq!(sparse.source().len(), 12);
        assert_eq!(full.source(), cloud.positions);
        assert_eq!(full.frame, sparse.frame);
        assert_eq!(dense.registration_voxel(), Some(0.0625));
    }

    #[test]
    fn registration_sampling_preserves_first_source_order_and_range_filtering() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 1.0,
            ..OdometryParams::default()
        });
        let cloud = sampling_cloud();
        odometry.register(&cloud);
        odometry.set_registration_voxel(Some(0.125)).unwrap();
        let expected: Vec<_> = cloud.positions.iter().step_by(4).copied().collect();
        let mut positions = vec![[0.0; 3], [100.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]];
        for point in &expected {
            positions.push(*point);
            positions.push([point[0] + 0.03125, point[1], point[2]]);
        }
        positions.extend(expected.iter().copied());
        let scan = PointCloud {
            positions,
            ..PointCloud::default()
        };
        assert_eq!(odometry.begin(&scan).source(), expected);
    }

    #[test]
    fn registration_sampling_changes_are_atomic_and_only_affect_future_begins() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            ..OdometryParams::default()
        });
        let cloud = sampling_cloud();
        odometry.register(&cloud);
        let pending = odometry.begin(&cloud);
        odometry.set_registration_voxel(Some(0.0625)).unwrap();
        assert_eq!(pending.source().len(), 12);
        assert_eq!(odometry.begin(&cloud).source().len(), 48);
        let report = odometry.last_report().cloned();
        let placed = odometry.placed().to_vec();
        let poses: Vec<_> = odometry.poses().iter().map(Rigid::to_matrix).collect();
        for value in [0.0, -0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                odometry.set_registration_voxel(Some(value)),
                Err(RegistrationVoxelError)
            );
            assert_eq!(odometry.registration_voxel(), Some(0.0625));
            assert_eq!(odometry.last_report(), report.as_ref());
            assert_eq!(odometry.placed(), placed);
            assert_eq!(
                odometry
                    .poses()
                    .iter()
                    .map(Rigid::to_matrix)
                    .collect::<Vec<_>>(),
                poses
            );
        }
        odometry.set_registration_voxel(None).unwrap();
        assert_eq!(odometry.registration_voxel(), None);
        assert_eq!(odometry.begin(&cloud).source(), pending.source());
    }

    #[test]
    fn registration_sampling_happens_after_constant_motion_deskew() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            deskew: true,
            ..OdometryParams::default()
        });
        let mut cloud = sampling_cloud();
        odometry.register(&cloud);
        let motion = Rigid {
            translation: [0.25, 0.0, 0.0],
            ..Rigid::IDENTITY
        };
        odometry.poses.push(motion);
        odometry.set_registration_voxel(Some(0.03125)).unwrap();
        let times: Vec<f32> = (0..cloud.len()).map(|i| (i % 2) as f32).collect();
        cloud.attributes.push(crate::Attribute {
            name: TIME.into(),
            values: crate::AttributeValues::F32(times.clone()),
        });
        let predicted = Rigid {
            translation: [0.5, 0.0, 0.0],
            ..Rigid::IDENTITY
        };
        let expected: Vec<_> = cloud
            .positions
            .iter()
            .zip(times)
            .map(|(point, time)| {
                // Pure translation: undo time relative to mid-sweep, then apply the prediction.
                [
                    point[0] + (f64::from(time) - 0.5) * 0.25 + 0.5,
                    point[1],
                    point[2],
                ]
            })
            .collect();
        assert_eq!(odometry.begin(&cloud).source(), expected);
        assert_eq!(
            odometry.begin(&cloud).prediction.to_matrix(),
            predicted.to_matrix()
        );
    }

    fn diagnostic_scene() -> PointCloud {
        PointCloud {
            positions: (0..12)
                .map(|i| [3.0 * (i % 4) as f64, 3.0 * (i / 4) as f64, (i % 3) as f64])
                .collect(),
            ..PointCloud::default()
        }
    }

    #[test]
    fn linearization_capture_choice_belongs_to_begin_and_empty_results_are_owned() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            ..OdometryParams::default()
        });
        assert!(!odometry.capture_linearization());
        let scan = diagnostic_scene();
        let pending = odometry.begin(&scan);
        odometry.set_capture_linearization(true);
        odometry.finish(pending);
        assert!(odometry.last_linearization().is_none());
        let pending = odometry.begin(&PointCloud::default());
        odometry.set_capture_linearization(false);
        odometry.finish(pending);
        let snapshot = odometry.last_linearization().unwrap().clone();
        assert!(!snapshot.alignment_eligible);
        assert!(snapshot.source_sensor.is_empty());
        assert!(snapshot.matched_source_indices.is_empty());
        assert_eq!(snapshot.normal_matrix, [[0.0; 6]; 6]);
        assert_eq!(snapshot.right_hand_side, [0.0; 6]);
        assert_eq!(snapshot.weighted_rms_radius_m, None);
        assert!(snapshot.finite);
        odometry.register(&scan);
        assert!(odometry.last_linearization().is_none());
        assert!(snapshot.inserted_world.is_empty());
    }

    #[test]
    fn linearization_capture_preserves_solver_map_and_report_bits() {
        let params = OdometryParams {
            min_range: 0.0,
            diagnostics: true,
            ..OdometryParams::default()
        };
        let mut plain = Odometry::new(params);
        let mut measured = Odometry::new(params);
        measured.set_capture_linearization(true);
        let mut scan = diagnostic_scene();
        for _ in 0..5 {
            let a = plain.register(&scan);
            let b = measured.register(&scan);
            assert_eq!(
                a.to_matrix().map(f64::to_bits),
                b.to_matrix().map(f64::to_bits)
            );
            assert_eq!(plain.last_report(), measured.last_report());
            assert_eq!(plain.map.map, measured.map.map);
            assert_eq!(plain.error_sum.to_bits(), measured.error_sum.to_bits());
            assert_eq!(plain.error_count, measured.error_count);
            let snapshot = measured.last_linearization().unwrap();
            assert_eq!(snapshot.inserted_world, measured.placed());
            assert_eq!(snapshot.pose, b);
            for point in &mut scan.positions {
                point[0] -= 0.1;
            }
        }
    }

    #[test]
    fn linearization_flags_nonfinite_weights_without_repair_or_pose_feedback() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            ..OdometryParams::default()
        });
        odometry.set_capture_linearization(true);
        let scan = diagnostic_scene();
        odometry.register(&scan);
        let mut pending = odometry.begin(&scan);
        pending.kernel = 0.0; // Legacy formula has 0/0 on exact matches.
        assert_eq!(odometry.finish(pending), Rigid::IDENTITY);
        let snapshot = odometry.last_linearization().unwrap();
        assert!(!snapshot.finite);
        assert!(snapshot.weights.iter().all(|w| w.is_nan()));
        assert_eq!(snapshot.weighted_rms_radius_m, None);
        assert_eq!(snapshot.inserted_world, scan.positions);
    }

    #[test]
    fn linearization_is_recomputed_after_the_last_step_at_returned_pose() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            diagnostics: true,
            ..OdometryParams::default()
        });
        odometry.set_capture_linearization(true);
        let scan = diagnostic_scene();
        odometry.register(&scan);
        let mut pending = odometry.begin(&scan);
        let mut identity = [[0.0; 6]; 6];
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        odometry.step(
            &mut pending,
            (identity, [0.04, -0.02, 0.01, 0.01, 0.02, -0.01]),
        );
        let pose = odometry.finish(pending);
        let snapshot = odometry.last_linearization().unwrap();
        assert_eq!(
            snapshot.source_world,
            scan.positions
                .iter()
                .map(|p| pose.apply(p))
                .collect::<Vec<_>>()
        );
        assert_eq!(snapshot.matched_source_indices.len(), 12);
        assert_ne!(snapshot.normal_matrix, identity);
        assert!(snapshot.squared_distances.iter().any(|&d| d > 0.0));
        assert_eq!(snapshot.target_world, scan.positions);
        assert!(snapshot.finite);
    }

    #[test]
    fn reports_distinguish_no_alignment_and_early_finish() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            diagnostics: true,
            ..OdometryParams::default()
        });
        assert!(odometry.last_report().is_none());
        assert_eq!(odometry.last_iterations(), 0);
        odometry.register(&diagnostic_scene());
        let report = odometry.last_report().unwrap();
        assert_eq!(report.termination, RegistrationTermination::EmptyMap);
        assert_eq!(report.input_points, 12);
        assert_eq!(report.registration_points, 12);
        assert!(report.correspondences.is_none());
        let small = PointCloud {
            positions: vec![[2.0, 1.0, 0.0]],
            ..PointCloud::default()
        };
        odometry.register(&small);
        let report = odometry.last_report().unwrap();
        assert_eq!(
            report.termination,
            RegistrationTermination::InsufficientPoints
        );
        assert_eq!(report.registration_points, 1);
        assert_eq!(report.iterations, 0);
        assert!(report.correspondences.is_none());
        let registration = odometry.begin(&diagnostic_scene());
        assert_eq!(registration.termination(), None);
        odometry.finish(registration);
        assert_eq!(
            odometry.last_report().unwrap().termination,
            RegistrationTermination::Unfinished
        );
    }

    #[test]
    fn zero_matches_and_collinear_matches_are_not_convergence() {
        for collinear in [false, true] {
            let mut odometry = Odometry::new(OdometryParams {
                min_range: 0.0,
                max_range: 200.0,
                diagnostics: true,
                ..OdometryParams::default()
            });
            let mut scan = diagnostic_scene();
            if collinear {
                scan.positions = (0..12).map(|i| [3.0 * i as f64, 0.0, 0.0]).collect();
            }
            odometry.register(&scan);
            if !collinear {
                for p in &mut scan.positions {
                    p[0] += 50.0;
                }
            }
            assert_eq!(odometry.register(&scan), Rigid::IDENTITY);
            let report = odometry.last_report().unwrap();
            assert_eq!(report.termination, RegistrationTermination::SingularSystem);
            assert_eq!(report.iterations, 0);
            let stats = report.correspondences.as_ref().unwrap();
            assert_eq!(stats.matched_points, if collinear { 12 } else { 0 });
            assert_eq!(stats.rmse_m, collinear.then_some(0.0));
            assert_eq!(stats.max_residual_m, collinear.then_some(0.0));
            assert_eq!(stats.matched_fraction, if collinear { 1.0 } else { 0.0 });
            // Diagnostic only: the existing policy still inserts every scan.
            assert_eq!(odometry.placed().len(), 12);
        }
    }

    #[test]
    fn reports_convergence_limits_and_repeated_steps_without_mutation() {
        for limited in [false, true] {
            let mut odometry = Odometry::new(OdometryParams {
                min_range: 0.0,
                max_iterations: 1,
                ..OdometryParams::default()
            });
            let scan = diagnostic_scene();
            odometry.register(&scan);
            let mut registration = odometry.begin(&scan);
            let mut h = [[0.0; 6]; 6];
            for (i, row) in h.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            let dx = if limited { [0.01; 6] } else { [0.0; 6] };
            assert!(odometry.step(&mut registration, (h, dx)));
            let total = *registration.total();
            assert!(odometry.step(&mut registration, (h, [1.0; 6])));
            assert_eq!(*registration.total(), total);
            odometry.finish(registration);
            let report = odometry.last_report().unwrap();
            assert_eq!(report.iterations, 1);
            assert_eq!(
                report.termination,
                if limited {
                    RegistrationTermination::IterationLimit
                } else {
                    RegistrationTermination::Converged
                }
            );
            assert!(report.correspondences.is_none());
        }
    }

    #[test]
    fn nonfinite_systems_and_updates_preserve_the_last_finite_transform() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            ..OdometryParams::default()
        });
        let scan = diagnostic_scene();
        odometry.register(&scan);
        for case in 0..4 {
            let mut registration = odometry.begin(&scan);
            let mut h = [[0.0; 6]; 6];
            for (i, row) in h.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            let mut b = [0.0; 6];
            match case {
                0 => h[2][3] = f64::NAN,
                1 => b[1] = f64::INFINITY,
                2 => {
                    h[0][0] = f64::MAX;
                    h[0][1] = -f64::MAX;
                    h[1][0] = f64::MAX / 2.0;
                    h[1][1] = f64::MAX;
                }
                _ => b[3] = f64::MAX,
            }
            assert!(odometry.step(&mut registration, (h, b)));
            assert_eq!(*registration.total(), Rigid::IDENTITY);
            assert_eq!(registration.iterations, 0);
            assert_eq!(
                registration.termination(),
                Some(if case == 3 {
                    RegistrationTermination::NonFiniteUpdate
                } else {
                    RegistrationTermination::NonFiniteSystem
                })
            );
        }
    }

    #[test]
    fn final_residuals_match_an_independent_search_before_insertion() {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            initial_sigma: 0.1,
            max_iterations: 1,
            diagnostics: true,
            ..OdometryParams::default()
        });
        let mut scan = diagnostic_scene();
        odometry.register(&scan);
        let old_map = odometry.map.map.clone();
        for (i, p) in scan.positions.iter_mut().enumerate() {
            p[0] += 0.10 + 0.003 * i as f64;
            p[1] -= 0.02 * (i % 3) as f64;
        }
        scan.positions[0][0] += 0.6; // In the stencil, outside the threshold.
        scan.positions.push([35.0, 0.0, 0.0]); // Outside the stencil entirely.
        let registration = odometry.begin(&scan);
        let source = registration.diagnostic_source.clone().unwrap();
        let pose = odometry.complete(registration);
        let report = odometry.last_report().unwrap();
        let stats = report.correspondences.as_ref().unwrap();
        let mut distances = Vec::new();
        for p in &source {
            let q = pose.apply(p);
            let key = q.map(|v| v.floor() as i64);
            let best = old_map
                .iter()
                .filter(|(k, _)| (0..3).all(|i| (k[i] - key[i]).abs() <= 1))
                .flat_map(|(_, points)| points)
                .map(|m| (0..3).map(|i| (q[i] - m[i]).powi(2)).sum::<f64>())
                .min_by(f64::total_cmp);
            if let Some(d) = best.filter(|d| *d <= report.threshold_m.powi(2)) {
                distances.push(d);
            }
        }
        assert_eq!(stats.matched_points, 11);
        assert_eq!(stats.matched_points, distances.len());
        assert_eq!(report.registration_points, 13);
        assert_eq!(stats.matched_fraction, 11.0 / 13.0);
        let rms = (distances.iter().sum::<f64>() / distances.len() as f64).sqrt();
        assert!((stats.rmse_m.unwrap() - rms).abs() < 1e-14);
        assert!(rms > 1e-3); // Would be zero after inserting the current points.
        let maximum = distances.iter().copied().reduce(f64::max).unwrap().sqrt();
        assert!((stats.max_residual_m.unwrap() - maximum).abs() < 1e-14);
    }

    #[test]
    fn optional_diagnostics_preserve_pose_bits_and_threshold_history() {
        let params = OdometryParams {
            min_range: 0.0,
            ..OdometryParams::default()
        };
        let mut plain = Odometry::new(params);
        let mut measured = Odometry::new(OdometryParams {
            diagnostics: true,
            ..params
        });
        let mut scan = diagnostic_scene();
        for _ in 0..5 {
            let a = plain.register(&scan);
            let b = measured.register(&scan);
            assert_eq!(
                a.to_matrix().map(f64::to_bits),
                b.to_matrix().map(f64::to_bits)
            );
            assert_eq!(plain.error_sum.to_bits(), measured.error_sum.to_bits());
            assert_eq!(plain.error_count, measured.error_count);
            assert_eq!(plain.last_iterations(), measured.last_iterations());
            assert_eq!(
                plain.last_report().unwrap().termination,
                measured.last_report().unwrap().termination
            );
            for p in &mut scan.positions {
                p[0] -= 0.1;
            }
        }
    }

    #[test]
    fn relative_prediction_composes_on_the_right_without_changing_deskew() {
        let mut odometry = Odometry::new(OdometryParams {
            deskew: true,
            ..OdometryParams::default()
        });
        let empty = PointCloud::default();
        odometry.register(&empty);
        let last = Rigid {
            rotation: crate::pose_graph::exp_so3(&[0.0, 0.0, 0.4]),
            translation: [4.0, -2.0, 1.0],
        };
        odometry.register_with_prediction(&empty, &last).unwrap();
        let motion = Rigid {
            rotation: crate::pose_graph::exp_so3(&[0.1, -0.2, 0.3]),
            translation: [0.5, -0.3, 0.2],
        };
        let mut scan = PointCloud {
            positions: (0..20).map(|k| [2.0, k as f64, 1.0]).collect(),
            ..PointCloud::default()
        };
        scan.attributes.push(crate::Attribute {
            name: TIME.into(),
            values: crate::AttributeValues::F32((0..20).map(|k| k as f32).collect()),
        });
        let original = odometry.begin(&scan);
        let predicted = odometry.begin_with_prediction(&scan, &motion).unwrap();
        assert_eq!(original.frame, predicted.frame);
        assert_eq!(predicted.prediction, last.compose(&motion));
        assert_ne!(predicted.prediction, motion.compose(&last));
        let found = odometry.register_with_prediction(&empty, &motion).unwrap();
        for (a, b) in found
            .to_matrix()
            .iter()
            .zip(last.compose(&motion).to_matrix())
        {
            assert!((a - b).abs() < 1e-14);
        }
        assert_eq!(odometry.last_iterations(), 0);
    }

    #[test]
    fn invalid_predictions_cannot_change_poses_map_or_threshold_history() {
        let mut odometry = Odometry::new(OdometryParams::default());
        let empty = PointCloud::default();
        assert!(
            odometry
                .register_with_prediction(&empty, &Rigid::IDENTITY)
                .is_err()
        );
        assert!(odometry.poses().is_empty());
        odometry.register(&empty);
        for case in 0..4 {
            let mut bad = Rigid::IDENTITY;
            match case {
                0 => bad.rotation[0][0] = 2.0,
                1 => bad.rotation[0][0] = -1.0,
                2 => bad.translation[1] = f64::NAN,
                _ => bad.rotation[2][1] = f64::INFINITY,
            }
            assert!(odometry.register_with_prediction(&empty, &bad).is_err());
            assert_eq!(odometry.poses(), &[Rigid::IDENTITY]);
            assert!(odometry.map().is_empty());
            assert_eq!(odometry.error_count, 0);
            assert_eq!(odometry.last_iterations(), 0);
        }
    }

    /// Ground and boxes scattered about, in world coordinates.
    fn town() -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        let mut seed = 7u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        // Ground, at random places so that no shift matches it with itself.
        for _ in 0..100_000 {
            out.push([next() * 120.0 - 40.0, next() * 60.0 - 30.0, 0.0]);
        }
        // Boxes of 2..4 m at random places and heights.
        for _ in 0..200 {
            let (cx, cy) = (next() * 110.0 - 30.0, next() * 50.0 - 25.0);
            if cy.abs() < 4.0 {
                continue;
            }
            let (sx, sy, h) = (2.0 + 2.0 * next(), 2.0 + 2.0 * next(), 1.0 + 4.0 * next());
            let steps = |len: f64| (0..=(len / 0.2) as usize).map(move |k| k as f64 * 0.2);
            for z in steps(h) {
                for x in steps(sx) {
                    out.push([cx + x, cy, z]);
                    out.push([cx + x, cy + sy, z]);
                }
                for y in steps(sy) {
                    out.push([cx, cy + y, z]);
                    out.push([cx + sx, cy + y, z]);
                }
            }
        }
        out
    }

    /// Scans of `world` along `truth`, each taken over the move to the next
    /// pose (the sensor spinning once, so the azimuth tells the time).
    fn skewed_scans(world: &[[f64; 3]], truth: &[Rigid]) -> Vec<PointCloud> {
        truth
            .windows(2)
            .map(|w| {
                let (a, b) = (&w[0], &w[1]);
                let step = crate::pose_graph::inverse(a).compose(b);
                let spin = crate::pose_graph::log_so3(&step.rotation);
                // In a's frame, which azimuth each point is at (a full turn from -pi).
                let mut seen: Vec<([f64; 3], f64)> = world
                    .iter()
                    .map(|p| crate::pose_graph::inverse(a).apply(p))
                    .filter(|p| norm(p) < 40.0)
                    .map(|p| (p, p[1].atan2(p[0])))
                    .collect();
                seen.sort_by(|x, y| x.1.total_cmp(&y.1));
                let positions = seen
                    .iter()
                    .map(|(p, azimuth)| {
                        // Taken this far into the move: seen from there.
                        let f = (azimuth + std::f64::consts::PI) / std::f64::consts::TAU;
                        let there = Rigid {
                            rotation: crate::pose_graph::exp_so3(&spin.map(|w| w * f)),
                            translation: step.translation.map(|t| t * f),
                        };
                        crate::pose_graph::inverse(&there).apply(p)
                    })
                    .collect();
                PointCloud {
                    positions,
                    ..PointCloud::default()
                }
            })
            .collect()
    }

    #[test]
    fn a_turn_taken_while_scanning_is_undone() {
        let world = town();
        // Pulling away into a turn of 6 degrees a scan while moving 0.6 m:
        // the end of each sweep sees the world turned against its start.
        let mut u = 0.0;
        let truth: Vec<Rigid> = (0..30)
            .map(|k| {
                u += 0.1 * k.min(10) as f64;
                let yaw = 0.1 * u;
                let (s, c) = yaw.sin_cos();
                Rigid {
                    rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                    translation: [6.0 * yaw.sin(), 6.0 * (1.0 - yaw.cos()), 1.5],
                }
            })
            .collect();
        let scans = skewed_scans(&world, &truth);
        let errors: Vec<f64> = [true, false]
            .map(|deskew| {
                let mut odometry = Odometry::new(OdometryParams {
                    deskew,
                    ..OdometryParams::default()
                });
                for scan in &scans {
                    odometry.register(scan);
                }
                // Each pose stands for the middle of its scan: halfway between the two.
                let found = odometry.poses()[28];
                let first = crate::pose_graph::inverse(&truth[0]);
                let (a, b) = (first.compose(&truth[28]), first.compose(&truth[29]));
                let mid: [f64; 3] =
                    std::array::from_fn(|k| 0.5 * (a.translation[k] + b.translation[k]));
                norm(&std::array::from_fn(|k| found.translation[k] - mid[k]))
            })
            .to_vec();
        assert!(
            errors[0] < 0.15,
            "deskewed {}, skewed {}",
            errors[0],
            errors[1]
        );
        assert!(
            errors[0] < 0.5 * errors[1],
            "deskewed {}, skewed {}",
            errors[0],
            errors[1]
        );
    }

    #[test]
    fn a_drive_through_a_town_is_followed() {
        let world = town();
        let mut odometry = Odometry::new(OdometryParams::default());
        // Pulling away to 1 m a scan, turning slowly; the scans see 40 m around.
        let mut x = 0.0;
        let truth: Vec<Rigid> = (0..40)
            .map(|k| {
                x += 0.1 * k.min(10) as f64;
                let yaw: f64 = 0.004 * k as f64;
                let (s, c) = yaw.sin_cos();
                Rigid {
                    rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                    translation: [x, 0.1 * x, 1.5],
                }
            })
            .collect();
        for pose in &truth {
            let back = crate::pose_graph::inverse(pose);
            let scan = PointCloud {
                positions: world
                    .iter()
                    .map(|p| back.apply(p))
                    .filter(|p| norm(p) < 40.0)
                    .collect(),
                ..PointCloud::default()
            };
            odometry.register(&scan);
        }
        // Relative to the first pose, as odometry starts at the identity.
        let last = crate::pose_graph::inverse(&truth[0]).compose(&truth[39]);
        let found = odometry.poses()[39];
        let error = norm(&std::array::from_fn(|k| {
            found.translation[k] - last.translation[k]
        }));
        assert!(error < 0.2, "{error}");
    }
}
