//! Trajectory evaluation: absolute (ATE) and relative (RPE) pose error of an
//! estimated trajectory against a reference.
//!
//! The definitions follow the Python package (`ca/trajectory.py`) so both
//! report the same numbers: every reference pose is matched to the estimate
//! at its timestamp (linear / slerp interpolation between the two estimate
//! poses around it when both are within `max_time_delta`, else the nearest
//! end pose if it is); the matched estimate is optionally aligned; ATE is the
//! position error per matched pose, RPE compares the displacement between
//! pose pairs (`delta` frames apart, or the first pose `delta` metres further
//! along the reference) in world coordinates, and rotation errors are
//! geodesic angles in degrees. On top of the Python modes there is Sim(3)
//! (Umeyama) alignment, KITTI pose files, and RPE deltas of several frames.

use crate::icp::rigid_fit;

/// A file layout for trajectories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// `timestamp tx ty tz [qx qy qz qw]`, whitespace separated.
    Tum,
    /// Twelve numbers per line: a row-major 3x4 pose. No timestamps: pose
    /// `i` gets time `i`, so two KITTI files match index by index.
    Kitti,
    /// Comma separated, with a header naming the columns
    /// (`timestamp,x,y,z[,qx,qy,qz,qw]` and the Python aliases) or without
    /// one (`timestamp,x,y,z`).
    Csv,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct TrajectoryError(pub String);

fn fail<T>(message: impl Into<String>) -> Result<T, TrajectoryError> {
    Err(TrajectoryError(message.into()))
}

/// Timed poses; orientations are unit quaternions `[x, y, z, w]` with
/// consecutive signs kept consistent.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    pub timestamps: Vec<f64>,
    pub positions: Vec<[f64; 3]>,
    pub orientations: Option<Vec<[f64; 4]>>,
}

// Column names the Python CSV reader accepts, in its order of preference.
const TIMESTAMP_KEYS: [&str; 4] = ["timestamp", "timestamp_sec", "time", "t"];
const POSITION_KEYS: [[&str; 3]; 2] = [["x", "y", "z"], ["x_m", "y_m", "z_m"]];
const ORIENTATION_KEYS: [[&str; 4]; 3] = [
    ["qx", "qy", "qz", "qw"],
    ["quat_x", "quat_y", "quat_z", "quat_w"],
    [
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "quaternion_w",
    ],
];

/// Non-empty lines that are not `#` comments.
fn data_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'))
}

fn numbers<'a>(
    line: usize,
    fields: impl Iterator<Item = &'a str>,
) -> Result<Vec<f64>, TrajectoryError> {
    fields
        .map(|f| {
            f.trim().parse::<f64>().map_err(|_| {
                TrajectoryError(format!("line {line}: {:?} is not a number", f.trim()))
            })
        })
        .collect()
}

/// Whether the file is a trajectory, judged by its extension and first line
/// (`head` is the start of the file). `.tum` and `.kitti` always are; `.txt`
/// only with 8 columns ending in a unit quaternion or 12 forming a pose, and
/// `.csv` only with a header naming a timestamp and x, y, z, since both
/// extensions are also used for point clouds.
pub fn detect(name: &str, head: &str) -> Option<Format> {
    let extension = name.rsplit_once('.')?.1.to_ascii_lowercase();
    let first = data_lines(head).next().map(|(_, l)| l);
    match extension.as_str() {
        "tum" => Some(Format::Tum),
        "kitti" => Some(Format::Kitti),
        "txt" => {
            let v = numbers(0, first?.split_whitespace()).ok()?;
            match v.len() {
                8 => {
                    let norm = v[4..].iter().map(|q| q * q).sum::<f64>().sqrt();
                    ((norm - 1.0).abs() < 0.01).then_some(Format::Tum)
                }
                12 => {
                    let r = |i: usize, j: usize| v[i * 4 + j];
                    let orthonormal = (0..3).all(|i| {
                        (0..3).all(|j| {
                            let dot: f64 = (0..3).map(|k| r(i, k) * r(j, k)).sum();
                            (dot - if i == j { 1.0 } else { 0.0 }).abs() < 0.01
                        })
                    });
                    orthonormal.then_some(Format::Kitti)
                }
                _ => None,
            }
        }
        "csv" => {
            let header = first?.to_ascii_lowercase();
            let keys: Vec<&str> = header.split(',').map(str::trim).collect();
            let has = |k: &str| keys.contains(&k);
            let timed = TIMESTAMP_KEYS.iter().any(|k| has(k));
            let placed = POSITION_KEYS.iter().any(|set| set.iter().all(|k| has(k)));
            (timed && placed).then_some(Format::Csv)
        }
        _ => None,
    }
}

/// Parse a trajectory file and check it as the Python loader does: at least
/// two poses, strictly increasing timestamps, non-zero quaternions.
pub fn parse(text: &str, format: Format) -> Result<Trajectory, TrajectoryError> {
    let (timestamps, positions, orientations) = match format {
        Format::Tum => parse_tum(text)?,
        Format::Kitti => parse_kitti(text)?,
        Format::Csv => parse_csv(text)?,
    };
    if timestamps.len() < 2 {
        return fail("a trajectory needs at least 2 poses");
    }
    if positions.iter().flatten().any(|v| !v.is_finite()) {
        return fail("trajectory positions must be finite");
    }
    if timestamps.windows(2).any(|w| w[1] <= w[0]) {
        return fail("trajectory timestamps must be strictly increasing");
    }
    let orientations = orientations
        .map(|mut qs| {
            for q in &mut qs {
                let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                if !norm.is_finite() || norm < 1e-12 {
                    return fail("trajectory orientations must be finite, non-zero quaternions");
                }
                *q = q.map(|v| v / norm);
            }
            // q and -q are the same rotation: keep neighbours on one side so
            // interpolation is deterministic.
            for i in 1..qs.len() {
                if dot4(&qs[i - 1], &qs[i]) < 0.0 {
                    qs[i] = qs[i].map(|v| -v);
                }
            }
            Ok(qs)
        })
        .transpose()?;
    Ok(Trajectory {
        timestamps,
        positions,
        orientations,
    })
}

type Columns = (Vec<f64>, Vec<[f64; 3]>, Option<Vec<[f64; 4]>>);

fn parse_tum(text: &str) -> Result<Columns, TrajectoryError> {
    let (mut t, mut p, mut q) = (Vec::new(), Vec::new(), Vec::new());
    let mut with_orientation = None;
    for (line, l) in data_lines(text) {
        let v = numbers(line, l.split_whitespace())?;
        if v.len() != 4 && v.len() != 8 {
            return fail(format!(
                "line {line}: TUM rows have 4 (timestamp x y z) or 8 (… qx qy qz qw) columns"
            ));
        }
        if *with_orientation.get_or_insert(v.len() == 8) != (v.len() == 8) {
            return fail(format!(
                "line {line}: rows must all include or all omit the orientation"
            ));
        }
        t.push(v[0]);
        p.push([v[1], v[2], v[3]]);
        if v.len() == 8 {
            q.push([v[4], v[5], v[6], v[7]]);
        }
    }
    Ok((t, p, with_orientation.unwrap_or(false).then_some(q)))
}

fn parse_kitti(text: &str) -> Result<Columns, TrajectoryError> {
    let (mut t, mut p, mut q) = (Vec::new(), Vec::new(), Vec::new());
    for (line, l) in data_lines(text) {
        let v = numbers(line, l.split_whitespace())?;
        if v.len() != 12 {
            return fail(format!(
                "line {line}: KITTI rows have 12 numbers (a row-major 3x4 pose)"
            ));
        }
        t.push(t.len() as f64);
        p.push([v[3], v[7], v[11]]);
        q.push(quaternion(&[
            [v[0], v[1], v[2]],
            [v[4], v[5], v[6]],
            [v[8], v[9], v[10]],
        ]));
    }
    Ok((t, p, Some(q)))
}

fn parse_csv(text: &str) -> Result<Columns, TrajectoryError> {
    let mut lines = data_lines(text).peekable();
    let Some(&(_, first)) = lines.peek() else {
        return fail("the trajectory file is empty");
    };
    let (mut t, mut p, mut q) = (Vec::new(), Vec::new(), Vec::new());
    if !first.chars().any(|c| c.is_alphabetic()) {
        for (line, l) in lines {
            let v = numbers(line, l.split(','))?;
            if v.len() < 4 {
                return fail(format!(
                    "line {line}: CSV rows need at least 4 columns: timestamp,x,y,z"
                ));
            }
            t.push(v[0]);
            p.push([v[1], v[2], v[3]]);
        }
        return Ok((t, p, None));
    }
    lines.next();
    let header: Vec<String> = first
        .split(',')
        .map(|k| k.trim().to_ascii_lowercase())
        .collect();
    let column = |k: &str| header.iter().position(|h| h == k);
    let Some(time) = TIMESTAMP_KEYS.iter().find_map(|k| column(k)) else {
        return fail("the CSV needs a timestamp, time or t column");
    };
    let Some(xyz) = POSITION_KEYS
        .iter()
        .find_map(|set| set.iter().map(|k| column(k)).collect::<Option<Vec<_>>>())
    else {
        return fail("the CSV needs x, y, z (or x_m, y_m, z_m) columns");
    };
    let quat = ORIENTATION_KEYS
        .iter()
        .find_map(|set| set.iter().map(|k| column(k)).collect::<Option<Vec<_>>>());
    for (line, l) in lines {
        let v = numbers(line, l.split(','))?;
        let at = |i: usize| {
            v.get(i)
                .copied()
                .ok_or_else(|| TrajectoryError(format!("line {line}: too few columns")))
        };
        t.push(at(time)?);
        p.push([at(xyz[0])?, at(xyz[1])?, at(xyz[2])?]);
        if let Some(c) = &quat {
            q.push([at(c[0])?, at(c[1])?, at(c[2])?, at(c[3])?]);
        }
    }
    Ok((t, p, quat.map(|_| q)))
}

// ------------------------------------------------------------- rotations

type Mat3 = [[f64; 3]; 3];

fn dot4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    (0..4).map(|i| a[i] * b[i]).sum()
}

fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn transpose(a: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

/// Rotation matrix of a unit quaternion `[x, y, z, w]`.
pub fn rotation_matrix(q: &[f64; 4]) -> Mat3 {
    let [x, y, z, w] = *q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Quaternion `[x, y, z, w]` of a rotation matrix (Shepperd's method).
pub fn quaternion(m: &Mat3) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    if trace > 0.0 {
        let s = 2.0 * (trace + 1.0).sqrt();
        [
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            s / 4.0,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        [
            s / 4.0,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
        [
            (m[0][1] + m[1][0]) / s,
            s / 4.0,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        ]
    } else {
        let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
        [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            s / 4.0,
            (m[1][0] - m[0][1]) / s,
        ]
    }
}

/// Angle of `reference^T estimate`, in degrees.
fn rotation_error(estimate: &Mat3, reference: &Mat3) -> f64 {
    let r = mul(&transpose(reference), estimate);
    let cosine = ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
}

/// Spherical interpolation of unit quaternions, linear when they are close.
fn slerp(a: &[f64; 4], b: &[f64; 4], alpha: f64) -> [f64; 4] {
    let mut b = *b;
    let mut dot = dot4(a, &b);
    if dot < 0.0 {
        b = b.map(|v| -v);
        dot = -dot;
    }
    let dot = dot.clamp(-1.0, 1.0);
    let q: [f64; 4] = if dot > 0.9995 {
        std::array::from_fn(|i| a[i] + alpha * (b[i] - a[i]))
    } else {
        let theta = dot.acos();
        let (wa, wb) = (
            ((1.0 - alpha) * theta).sin() / theta.sin(),
            (alpha * theta).sin() / theta.sin(),
        );
        std::array::from_fn(|i| wa * a[i] + wb * b[i])
    };
    let norm = dot4(&q, &q).sqrt();
    q.map(|v| v / norm)
}

// ----------------------------------------------------------- association

/// Where a sample of a series at `time` comes from: an exact or clamped pose,
/// or an interpolation between two, with its distance in time.
#[derive(Debug, Clone, Copy)]
enum Sample {
    At(usize, f64),
    Between(usize, f64, f64),
}

/// How the Python module samples `times` at `time` (`_interpolate_matches`).
fn sample(times: &[f64], time: f64, max_dt: f64) -> Option<Sample> {
    let i = times.partition_point(|&t| t < time);
    let n = times.len();
    if i < n && (times[i] - time).abs() <= 1e-9 {
        return Some(Sample::At(i, 0.0));
    }
    if 0 < i && i < n {
        let (left, right) = (time - times[i - 1], times[i] - time);
        if left <= max_dt && right <= max_dt {
            let alpha = left / (times[i] - times[i - 1]);
            return Some(Sample::Between(i - 1, alpha, left.min(right)));
        }
    }
    if i == 0 && (times[0] - time).abs() <= max_dt {
        return Some(Sample::At(0, (times[0] - time).abs()));
    }
    if i == n && (time - times[n - 1]).abs() <= max_dt {
        return Some(Sample::At(n - 1, (time - times[n - 1]).abs()));
    }
    None
}

fn position_at(positions: &[[f64; 3]], s: Sample) -> [f64; 3] {
    match s {
        Sample::At(i, _) => positions[i],
        Sample::Between(i, a, _) => {
            std::array::from_fn(|k| (1.0 - a) * positions[i][k] + a * positions[i + 1][k])
        }
    }
}

fn orientation_at(orientations: &[[f64; 4]], s: Sample) -> [f64; 4] {
    match s {
        Sample::At(i, _) => orientations[i],
        Sample::Between(i, a, _) => slerp(&orientations[i], &orientations[i + 1], a),
    }
}

// ------------------------------------------------------------- alignment

/// How the estimate is moved onto the reference before measuring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    None,
    /// Translate the first matched pose onto the reference's (Python's
    /// `--align-origin`).
    Origin,
    /// Best rigid motion (Python's `--align-rigid`).
    Se3,
    /// Best rigid motion and uniform scale (Umeyama), e.g. for monocular
    /// visual odometry whose scale is unknown.
    Sim3,
}

/// `x' = scale * R x + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Similarity {
    pub rotation: Mat3,
    pub translation: [f64; 3],
    pub scale: f64,
}

impl Similarity {
    pub const IDENTITY: Self = Self {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
        scale: 1.0,
    };

    pub fn apply(&self, p: &[f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        std::array::from_fn(|i| {
            self.scale * (r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2]) + self.translation[i]
        })
    }

    /// Row-major 4x4 homogeneous matrix (the scale is in the 3x3 block).
    pub fn to_matrix(&self) -> [f64; 16] {
        let (r, t, s) = (&self.rotation, &self.translation, self.scale);
        [
            s * r[0][0],
            s * r[0][1],
            s * r[0][2],
            t[0], //
            s * r[1][0],
            s * r[1][1],
            s * r[1][2],
            t[1], //
            s * r[2][0],
            s * r[2][1],
            s * r[2][2],
            t[2], //
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }
}

/// The similarity (or, without `with_scale`, rigid motion) that best maps
/// `source` onto `target` in the least-squares sense (Umeyama 1991). The
/// rotation is Horn's closed form, which minimises the same objective as
/// Umeyama's SVD and is always proper; the scale is then Umeyama's
/// `sum(y'·R x') / sum(|x'|²)` over the centred points.
pub fn umeyama(source: &[[f64; 3]], target: &[[f64; 3]], with_scale: bool) -> Similarity {
    let pairs = || source.iter().copied().zip(target.iter().copied());
    let rotation = rigid_fit(pairs()).rotation;
    let n = source.len() as f64;
    let mean = |ps: &[[f64; 3]]| -> [f64; 3] {
        std::array::from_fn(|a| ps.iter().map(|p| p[a]).sum::<f64>() / n)
    };
    let (ms, mt) = (mean(source), mean(target));
    let rotate = |p: &[f64; 3]| -> [f64; 3] {
        std::array::from_fn(|i| (0..3).map(|k| rotation[i][k] * p[k]).sum())
    };
    let scale = if with_scale {
        let (mut num, mut den) = (0.0, 0.0);
        for (x, y) in pairs() {
            let xc: [f64; 3] = std::array::from_fn(|a| x[a] - ms[a]);
            let rx = rotate(&xc);
            num += (0..3).map(|a| (y[a] - mt[a]) * rx[a]).sum::<f64>();
            den += (0..3).map(|a| xc[a] * xc[a]).sum::<f64>();
        }
        if den > 0.0 { num / den } else { 1.0 }
    } else {
        1.0
    };
    let rm = rotate(&ms);
    Similarity {
        rotation,
        translation: std::array::from_fn(|a| mt[a] - scale * rm[a]),
        scale,
    }
}

// --------------------------------------------------------------- metrics

/// Summary of an error series; `std` is the population deviation, as NumPy's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub count: usize,
    pub rmse: f64,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub std: f64,
}

impl Stats {
    pub fn of(values: &[f64]) -> Option<Stats> {
        if values.is_empty() {
            return None;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };
        Some(Stats {
            count: values.len(),
            rmse: (values.iter().map(|v| v * v).sum::<f64>() / n).sqrt(),
            mean,
            median,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            std: (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt(),
        })
    }
}

/// How far apart the two poses compared by RPE are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RpeDelta {
    /// This many matched poses (1: consecutive poses, as Python's RPE).
    Frames(usize),
    /// The first pose at least this far along the reference path (Python's
    /// `rpe_distances_m`, KITTI style).
    Meters(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvalParams {
    /// Seconds; the Python CLI's default is 0.05.
    pub max_time_delta: f64,
    pub alignment: Alignment,
    pub rpe_delta: RpeDelta,
}

impl Default for EvalParams {
    fn default() -> Self {
        Self {
            max_time_delta: 0.05,
            alignment: Alignment::None,
            rpe_delta: RpeDelta::Frames(1),
        }
    }
}

/// Per-pose and per-pair errors. Rotation errors are present when both
/// trajectories have orientations.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    /// Times of the matched reference poses.
    pub timestamps: Vec<f64>,
    /// The matched estimate, aligned.
    pub estimate: Vec<[f64; 3]>,
    pub reference: Vec<[f64; 3]>,
    /// Distance in time to the estimate pose(s) used for each match.
    pub time_deltas: Vec<f64>,
    /// Maps the estimate onto the reference.
    pub alignment: Similarity,
    /// Position error per matched pose.
    pub ate: Vec<f64>,
    /// Degrees.
    pub ate_rotation: Option<Vec<f64>>,
    /// The matched pose index pairs compared by RPE.
    pub rpe_pairs: Vec<(usize, usize)>,
    pub rpe_translation: Vec<f64>,
    /// Degrees.
    pub rpe_rotation: Option<Vec<f64>>,
    /// Translation error in % of the path length between the pair (metre
    /// deltas only).
    pub rpe_percent: Option<Vec<f64>>,
    /// Displacement error between the first and last matched pose.
    pub endpoint_drift: f64,
    /// Lengths of the matched paths.
    pub reference_length: f64,
    pub estimate_length: f64,
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
}

/// |(e_j - e_i) - (r_j - r_i)|.
fn displacement_error(e: &[[f64; 3]], r: &[[f64; 3]], i: usize, j: usize) -> f64 {
    (0..3)
        .map(|a| ((e[j][a] - e[i][a]) - (r[j][a] - r[i][a])).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn path_length(ps: &[[f64; 3]]) -> f64 {
    ps.windows(2).map(|w| distance(&w[0], &w[1])).sum()
}

/// Compare `estimate` with `reference` (see the module docs).
pub fn evaluate(
    estimate: &Trajectory,
    reference: &Trajectory,
    params: &EvalParams,
) -> Result<Evaluation, TrajectoryError> {
    let max_dt = params.max_time_delta;
    if max_dt.is_nan() || max_dt <= 0.0 {
        return fail("the maximum time difference must be positive");
    }
    match params.rpe_delta {
        RpeDelta::Frames(0) => return fail("the RPE delta must be at least 1 frame"),
        RpeDelta::Meters(d) if d.is_nan() || d <= 0.0 => {
            return fail("the RPE delta must be a positive distance");
        }
        _ => {}
    }

    // Match every reference pose that has estimate poses close enough.
    let mut timestamps = Vec::new();
    let mut matched = Vec::new();
    let mut ref_indices = Vec::new();
    let mut time_deltas = Vec::new();
    for (k, &time) in reference.timestamps.iter().enumerate() {
        if let Some(s) = sample(&estimate.timestamps, time, max_dt) {
            timestamps.push(time);
            ref_indices.push(k);
            time_deltas.push(match s {
                Sample::At(_, dt) | Sample::Between(_, _, dt) => dt,
            });
            matched.push(s);
        }
    }
    if timestamps.len() < 2 {
        return fail(format!(
            "fewer than 2 poses match in time (within {max_dt} s); check that both use the same clock"
        ));
    }
    let raw: Vec<[f64; 3]> = matched
        .iter()
        .map(|&s| position_at(&estimate.positions, s))
        .collect();
    let ref_positions: Vec<[f64; 3]> = ref_indices
        .iter()
        .map(|&k| reference.positions[k])
        .collect();

    let alignment = match params.alignment {
        Alignment::None => Similarity::IDENTITY,
        Alignment::Origin => Similarity {
            translation: std::array::from_fn(|a| ref_positions[0][a] - raw[0][a]),
            ..Similarity::IDENTITY
        },
        Alignment::Se3 => umeyama(&raw, &ref_positions, false),
        Alignment::Sim3 => umeyama(&raw, &ref_positions, true),
    };
    let est_positions: Vec<[f64; 3]> = raw.iter().map(|p| alignment.apply(p)).collect();

    // Rotations: the estimate's are turned by the alignment too.
    let rotations = match (&estimate.orientations, &reference.orientations) {
        (Some(eq), Some(rq)) => {
            let est: Vec<Mat3> = matched
                .iter()
                .map(|&s| {
                    mul(
                        &alignment.rotation,
                        &rotation_matrix(&orientation_at(eq, s)),
                    )
                })
                .collect();
            let refs: Vec<Mat3> = ref_indices
                .iter()
                .map(|&k| rotation_matrix(&rq[k]))
                .collect();
            Some((est, refs))
        }
        _ => None,
    };

    let ate: Vec<f64> = est_positions
        .iter()
        .zip(&ref_positions)
        .map(|(e, r)| distance(e, r))
        .collect();
    let ate_rotation = rotations
        .as_ref()
        .map(|(e, r)| e.iter().zip(r).map(|(e, r)| rotation_error(e, r)).collect());

    let n = timestamps.len();
    let rpe_pairs: Vec<(usize, usize)> = match params.rpe_delta {
        RpeDelta::Frames(d) => (0..n.saturating_sub(d)).map(|i| (i, i + d)).collect(),
        RpeDelta::Meters(d) => {
            // First pose whose distance along the reference reaches `d`.
            let mut along = vec![0.0; n];
            for i in 1..n {
                along[i] = along[i - 1] + distance(&ref_positions[i - 1], &ref_positions[i]);
            }
            (0..n.saturating_sub(1))
                .filter_map(|i| {
                    let j = along.partition_point(|&s| s < along[i] + d);
                    (j > i && j < n).then_some((i, j))
                })
                .collect()
        }
    };
    let rpe_translation = rpe_pairs
        .iter()
        .map(|&(i, j)| displacement_error(&est_positions, &ref_positions, i, j))
        .collect::<Vec<_>>();
    let rpe_rotation = rotations.as_ref().map(|(e, r)| {
        rpe_pairs
            .iter()
            .map(|&(i, j)| {
                rotation_error(
                    &mul(&transpose(&e[i]), &e[j]),
                    &mul(&transpose(&r[i]), &r[j]),
                )
            })
            .collect()
    });
    let rpe_percent = matches!(params.rpe_delta, RpeDelta::Meters(_)).then(|| {
        rpe_pairs
            .iter()
            .zip(&rpe_translation)
            .map(|(&(i, j), e)| 100.0 * e / path_length(&ref_positions[i..=j]))
            .collect()
    });

    Ok(Evaluation {
        endpoint_drift: displacement_error(&est_positions, &ref_positions, 0, n - 1),
        reference_length: path_length(&ref_positions),
        estimate_length: path_length(&est_positions),
        timestamps,
        estimate: est_positions,
        reference: ref_positions,
        time_deltas,
        alignment,
        ate,
        ate_rotation,
        rpe_pairs,
        rpe_translation,
        rpe_rotation,
        rpe_percent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotation(rx: f64, ry: f64, rz: f64) -> Mat3 {
        let (cx, sx, cy, sy, cz, sz) = (rx.cos(), rx.sin(), ry.cos(), ry.sin(), rz.cos(), rz.sin());
        let x = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
        let y = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
        let z = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
        mul(&z, &mul(&y, &x))
    }

    /// A curved 3D path at 10 Hz, heading along its tangent.
    fn path(n: usize) -> Trajectory {
        let t: Vec<f64> = (0..n).map(|i| 0.1 * i as f64).collect();
        Trajectory {
            positions: t
                .iter()
                .map(|&t| {
                    [
                        3.0 * (0.6 * t).sin(),
                        2.0 * (1.0 - (0.6 * t).cos()),
                        0.3 * t,
                    ]
                })
                .collect(),
            orientations: Some(
                t.iter()
                    .map(|&t| quaternion(&rotation(0.0, 0.1 * t.sin(), 0.5 * t)))
                    .collect(),
            ),
            timestamps: t,
        }
    }

    fn moved(t: &Trajectory, s: &Similarity) -> Trajectory {
        Trajectory {
            timestamps: t.timestamps.clone(),
            positions: t.positions.iter().map(|p| s.apply(p)).collect(),
            orientations: t.orientations.as_ref().map(|qs| {
                qs.iter()
                    .map(|q| quaternion(&mul(&s.rotation, &rotation_matrix(q))))
                    .collect()
            }),
        }
    }

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!((a - b).abs() <= tolerance, "{a} != {b}");
    }

    #[test]
    fn umeyama_recovers_rigid_and_similarity_transforms() {
        let points = path(30).positions;
        let truth = Similarity {
            rotation: rotation(0.3, -0.2, 1.1),
            translation: [100.0, -20.0, 5.0],
            scale: 2.5,
        };
        let target: Vec<_> = points.iter().map(|p| truth.apply(p)).collect();
        let found = umeyama(&points, &target, true);
        close(found.scale, 2.5, 1e-9);
        for (p, q) in points.iter().zip(&target) {
            close(distance(&found.apply(p), q), 0.0, 1e-9);
        }
        let rigid = Similarity {
            scale: 1.0,
            ..truth
        };
        let target: Vec<_> = points.iter().map(|p| rigid.apply(p)).collect();
        let found = umeyama(&points, &target, false);
        assert_eq!(found.scale, 1.0);
        for i in 0..3 {
            for j in 0..3 {
                close(found.rotation[i][j], rigid.rotation[i][j], 1e-9);
            }
            close(found.translation[i], rigid.translation[i], 1e-9);
        }
    }

    #[test]
    fn identical_trajectories_have_no_error() {
        let t = path(20);
        for alignment in [
            Alignment::None,
            Alignment::Origin,
            Alignment::Se3,
            Alignment::Sim3,
        ] {
            let params = EvalParams {
                alignment,
                ..EvalParams::default()
            };
            let e = evaluate(&t, &t, &params).unwrap();
            assert_eq!(e.ate.len(), 20);
            assert!(Stats::of(&e.ate).unwrap().max < 1e-9, "{alignment:?}");
            assert!(Stats::of(e.ate_rotation.as_ref().unwrap()).unwrap().max < 1e-5);
            assert!(Stats::of(&e.rpe_translation).unwrap().max < 1e-9);
        }
    }

    #[test]
    fn alignment_removes_a_known_transform() {
        let reference = path(25);
        let truth = Similarity {
            rotation: rotation(0.1, 0.05, 0.7),
            translation: [5.0, -3.0, 1.0],
            scale: 0.8,
        };
        let estimate = moved(&reference, &truth);
        let run = |alignment| {
            let params = EvalParams {
                alignment,
                ..EvalParams::default()
            };
            evaluate(&estimate, &reference, &params).unwrap()
        };
        assert!(Stats::of(&run(Alignment::None).ate).unwrap().rmse > 1.0);
        assert!(Stats::of(&run(Alignment::Se3).ate).unwrap().rmse > 0.01);
        let sim3 = run(Alignment::Sim3);
        assert!(Stats::of(&sim3.ate).unwrap().max < 1e-9);
        close(sim3.alignment.scale, 1.0 / 0.8, 1e-9);
        // The alignment also turns the orientations back.
        assert!(Stats::of(sim3.ate_rotation.as_ref().unwrap()).unwrap().max < 1e-5);
    }

    #[test]
    fn rpe_measures_a_steady_drift() {
        // Every step drifts 1 cm in x (and 0.2 degrees in yaw): RPE over one
        // frame is exactly that, over d frames d times that, and ATE grows.
        let reference = path(40);
        let mut estimate = reference.clone();
        for (i, (p, q)) in estimate
            .positions
            .iter_mut()
            .zip(estimate.orientations.as_mut().unwrap())
            .enumerate()
        {
            p[0] += 0.01 * i as f64;
            *q = quaternion(&mul(
                &rotation(0.0, 0.0, (0.2 * i as f64).to_radians()),
                &rotation_matrix(q),
            ));
        }
        let e = evaluate(&estimate, &reference, &EvalParams::default()).unwrap();
        let rpe = Stats::of(&e.rpe_translation).unwrap();
        assert_eq!(rpe.count, 39);
        close(rpe.min, 0.01, 1e-12);
        close(rpe.max, 0.01, 1e-12);
        let ate = Stats::of(&e.ate).unwrap();
        close(ate.max, 0.39, 1e-12);
        close(ate.min, 0.0, 1e-12);
        // A world-frame yaw drift turns each relative pose by the step.
        let rot = Stats::of(e.rpe_rotation.as_ref().unwrap()).unwrap();
        close(rot.min, 0.2, 1e-9);
        close(rot.max, 0.2, 1e-9);
        let five = EvalParams {
            rpe_delta: RpeDelta::Frames(5),
            ..EvalParams::default()
        };
        let e = evaluate(&estimate, &reference, &five).unwrap();
        assert_eq!(e.rpe_translation.len(), 35);
        close(Stats::of(&e.rpe_translation).unwrap().rmse, 0.05, 1e-12);
    }

    #[test]
    fn rpe_by_distance_pairs_poses_along_the_path() {
        // Straight line, 0.1 m per pose: 0.5 m is five poses on.
        let reference = Trajectory {
            timestamps: (0..11).map(|i| i as f64).collect(),
            positions: (0..11).map(|i| [0.1 * i as f64, 0.0, 0.0]).collect(),
            orientations: None,
        };
        let mut estimate = reference.clone();
        for (i, p) in estimate.positions.iter_mut().enumerate() {
            p[1] = 0.01 * i as f64;
        }
        let params = EvalParams {
            rpe_delta: RpeDelta::Meters(0.5),
            ..EvalParams::default()
        };
        let e = evaluate(&estimate, &reference, &params).unwrap();
        assert_eq!(e.rpe_pairs, (0..6).map(|i| (i, i + 5)).collect::<Vec<_>>());
        close(e.rpe_translation[0], 0.05, 1e-12);
        close(e.rpe_percent.unwrap()[0], 10.0, 1e-9);
        assert!(e.rpe_rotation.is_none());
    }

    #[test]
    fn association_interpolates_within_the_time_limit() {
        let reference = Trajectory {
            timestamps: vec![0.0, 1.0, 2.0, 3.0],
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            orientations: None,
        };
        // 0.0 is 0.01 s before the first pose; poses at 0.98 / 1.02 average
        // to x = 1; 2.0 has no neighbour within 0.05 s on its left; 3.0 is
        // 0.04 s after the last pose.
        let estimate = Trajectory {
            timestamps: vec![0.01, 0.98, 1.02, 2.1, 2.96],
            positions: vec![
                [0.0; 3],
                [0.9, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            orientations: None,
        };
        let e = evaluate(&estimate, &reference, &EvalParams::default()).unwrap();
        assert_eq!(e.timestamps, vec![0.0, 1.0, 3.0]);
        close(e.estimate[1][0], 1.0, 1e-12);
        close(e.time_deltas[0], 0.01, 1e-12);
        close(e.time_deltas[1], 0.02, 1e-12);
        close(e.time_deltas[2], 0.04, 1e-12);
        let far = EvalParams {
            max_time_delta: 0.001,
            ..EvalParams::default()
        };
        assert!(evaluate(&estimate, &reference, &far).is_err());
    }

    #[test]
    fn parses_tum_kitti_and_csv() {
        let tum = "# timestamp tx ty tz qx qy qz qw\n\
                   1.0 0 0 0 0 0 0 2\n\
                   2.0 1 0 0 0 0 0 -1\n";
        let t = parse(tum, Format::Tum).unwrap();
        assert_eq!(t.timestamps, vec![1.0, 2.0]);
        // Normalised, and the second flipped to the first one's side.
        assert_eq!(t.orientations.unwrap(), vec![[0.0, 0.0, 0.0, 1.0]; 2]);
        assert!(parse("1 0 0 0\n2 0 0 0 0 0 0 1\n", Format::Tum).is_err());
        assert!(parse("2 0 0 0\n1 0 0 0\n", Format::Tum).is_err());
        assert!(parse("1 0 0 0\n", Format::Tum).is_err());

        let r = rotation(0.2, -0.1, 1.3);
        let row = |x: f64| {
            format!(
                "{} {} {} {x} {} {} {} 2 {} {} {} 3\n",
                r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2]
            )
        };
        let kitti = row(1.0) + &row(4.0);
        let k = parse(&kitti, Format::Kitti).unwrap();
        assert_eq!(k.timestamps, vec![0.0, 1.0]);
        assert_eq!(k.positions[1], [4.0, 2.0, 3.0]);
        let back = rotation_matrix(&k.orientations.unwrap()[0]);
        for i in 0..3 {
            for j in 0..3 {
                close(back[i][j], r[i][j], 1e-12);
            }
        }

        let csv = "Time, X, Y, Z, qx, qy, qz, qw\n0,1,2,3,0,0,0,1\n0.5,4,5,6,0,0,0,1\n";
        let c = parse(csv, Format::Csv).unwrap();
        assert_eq!(c.positions, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert!(c.orientations.is_some());
        let plain = parse("0,1,2,3\n1,4,5,6\n", Format::Csv).unwrap();
        assert!(plain.orientations.is_none());
        assert!(parse("time,a,b,c\n0,1,2,3\n1,1,2,3\n", Format::Csv).is_err());
    }

    #[test]
    fn detects_trajectories_among_point_files() {
        let tum = "# comment\n1.0 1 2 3 0 0 0.7071068 0.7071068\n";
        assert_eq!(detect("a.txt", tum), Some(Format::Tum));
        assert_eq!(detect("a.TUM", "1 2 3 4\n"), Some(Format::Tum));
        assert_eq!(detect("poses.kitti", ""), Some(Format::Kitti));
        let kitti = "1 0 0 5 0 1 0 6 0 0 1 7\n";
        assert_eq!(detect("00.txt", kitti), Some(Format::Kitti));
        assert_eq!(
            detect("t.csv", "timestamp,x,y,z\n0,1,2,3\n"),
            Some(Format::Csv)
        );
        // Point clouds.
        assert_eq!(detect("cloud.txt", "1 2 3 255 0 0 0.5 9\n"), None);
        assert_eq!(detect("cloud.txt", "1 2 3 4\n"), None);
        assert_eq!(detect("cloud.csv", "x,y,z,intensity\n1,2,3,4\n"), None);
        assert_eq!(detect("cloud.ply", "ply\n"), None);
    }
}
