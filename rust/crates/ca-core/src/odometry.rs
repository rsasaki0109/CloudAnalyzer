//! LiDAR odometry, after KISS-ICP (Vizzo et al., 2023): each scan, thinned,
//! is registered onto a local voxel map of the scans before it, starting
//! from where a constant velocity puts it, by point-to-point ICP with a
//! robust (Geman-McClure) kernel and correspondences within a threshold
//! learnt from how far the motion model has been off; then it joins the
//! map. Small enough to run in the browser, so a recording without poses
//! can be opened there.

use crate::PointCloud;
use crate::icp::Rigid;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

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
        }
    }
}

/// Odometry state: the local map, the poses so far and the motion model's error.
pub struct Odometry {
    params: OdometryParams,
    map: VoxelMap,
    poses: Vec<Rigid>,
    error_sum: f64,
    error_count: usize,
}

type Vec6 = [f64; 6];
type Mat6 = [[f64; 6]; 6];

/// `a x = b` by elimination with partial pivoting.
fn solve6(mut a: Mat6, mut b: Vec6) -> Option<Vec6> {
    for col in 0..6 {
        let pivot = (col..6).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
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
    Some(x)
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

impl Odometry {
    pub fn new(params: OdometryParams) -> Self {
        Odometry {
            params,
            map: VoxelMap::default(),
            poses: Vec::new(),
            error_sum: 0.0,
            error_count: 0,
        }
    }

    pub fn poses(&self) -> &[Rigid] {
        &self.poses
    }

    fn sigma(&self) -> f64 {
        if self.error_count == 0 {
            self.params.initial_sigma
        } else {
            (self.error_sum / self.error_count as f64).sqrt()
        }
    }

    /// The map point nearest `q` in its voxel and the 26 around it, with the
    /// squared distance; a voxel is skipped when it is no nearer than the best so far.
    fn nearest(&self, q: &[f64; 3]) -> Option<([f64; 3], f64)> {
        let v = self.params.map_voxel;
        let key = q.map(|x| (x / v).floor() as i64);
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
                    let gap: f64 = (0..3)
                        .map(|k| match offset[k] {
                            -1 => q[k] - key[k] as f64 * v,
                            1 => (key[k] + 1) as f64 * v - q[k],
                            _ => 0.0,
                        })
                        .map(|g| g * g)
                        .sum();
                    if gap < best.1 {
                        search(std::array::from_fn(|k| key[k] + offset[k]), &mut best);
                    }
                }
            }
        }
        best.1.is_finite().then_some(best)
    }

    /// The motion that best brings `source` (world points) onto the map:
    /// pairs within 3 sigma, weighted by a Geman-McClure kernel of sigma/3.
    /// Rotations are about `center` (where the sensor is): about the world's
    /// origin, far from it, a small turn would come with a large shift, and
    /// the steps would shrink below the tolerance only after many iterations.
    fn align(&self, source: &[[f64; 3]], center: [f64; 3], sigma: f64) -> Rigid {
        let threshold_sq = (3.0 * sigma).powi(2);
        let kernel = sigma / 3.0;
        let mut total = Rigid::IDENTITY;
        for _ in 0..self.params.max_iterations {
            let mut jtj = [[0.0; 6]; 6];
            let mut jtr = [0.0; 6];
            for p in source {
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
            let Some(dx) = solve6(jtj, jtr) else { break };
            // Turn about `center`, then shift.
            let rotation = crate::pose_graph::exp_so3(&[dx[3], dx[4], dx[5]]);
            let turned = Rigid {
                rotation,
                translation: [0.0; 3],
            }
            .apply(&center);
            let step = Rigid {
                rotation,
                translation: std::array::from_fn(|k| center[k] - turned[k] + dx[k]),
            };
            total = step.compose(&total);
            // Converged: steps of under a millimetre and a tenth of a milliradian.
            if dx[..3].iter().map(|v| v * v).sum::<f64>().sqrt() < self.params.tolerance_t
                && dx[3..].iter().map(|v| v * v).sum::<f64>().sqrt() < self.params.tolerance_r
            {
                break;
            }
        }
        total
    }

    /// Register the next scan (in its sensor's frame) and return its pose.
    pub fn register(&mut self, scan: &PointCloud) -> Rigid {
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
        // Half a map voxel for the map, one and a half for registering.
        let frame = thin(&near, 0.5 * p.map_voxel);
        let source = thin(&frame, 1.5 * p.map_voxel);
        let prediction = match n {
            0 => Rigid::IDENTITY,
            _ => self.poses[n - 1].compose(&motion),
        };
        let mut pose = if self.map.is_empty() || source.len() < 10 {
            prediction
        } else {
            let moved: Vec<[f64; 3]> = source.iter().map(|q| prediction.apply(q)).collect();
            self.align(&moved, prediction.translation, self.sigma())
                .compose(&prediction)
        };
        pose.rotation = orthonormal(&pose.rotation);
        // How far the motion model was off, as the most a point in range moved.
        let deviation = crate::pose_graph::inverse(&prediction).compose(&pose);
        let r = &deviation.rotation;
        let theta = ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0)
            .clamp(-1.0, 1.0)
            .acos();
        let error = 2.0 * p.max_range * (theta / 2.0).sin() + norm(&deviation.translation);
        if error > p.min_motion {
            self.error_sum += error * error;
            self.error_count += 1;
        }
        // The scan joins the map; voxels out of the sensor's range leave it.
        for q in &frame {
            let w = pose.apply(q);
            let cell = self
                .map
                .entry(w.map(|v| (v / p.map_voxel).floor() as i64))
                .or_default();
            if cell.len() < p.map_points {
                cell.push(w);
            }
        }
        let origin = pose.translation;
        self.map.retain(|_, points| {
            let q = points[0];
            norm(&[q[0] - origin[0], q[1] - origin[1], q[2] - origin[2]]) <= p.max_range
        });
        self.poses.push(pose);
        pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
