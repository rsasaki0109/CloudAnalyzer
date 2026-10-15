//! Rigid ICP registration (point-to-plane or point-to-point).
//!
//! Each iteration pairs a subsample of the moving cloud with its nearest
//! reference points and optionally keeps only the closest fraction of pairs
//! ("overlap"). Point-to-plane minimises distances along reference normals
//! (estimated by PCA of nearby reference points, on demand) with a
//! linearised 6x6 solve; point-to-point uses Horn's closed-form quaternion
//! solution. All arithmetic happens relative to the reference centroid so
//! georeferenced coordinates keep full precision.

use crate::PointCloud;
use crate::kdtree::{KdTree, Nearest, NearestRadiusHints};
use std::sync::OnceLock;

/// A rigid transform `x' = R x + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigid {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

impl Rigid {
    pub const IDENTITY: Self = Self {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };

    pub fn apply(&self, p: &[f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        std::array::from_fn(|i| {
            r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + self.translation[i]
        })
    }

    /// `self` after `first`: `x -> self(first(x))`.
    pub fn compose(&self, first: &Rigid) -> Rigid {
        let a = &self.rotation;
        let b = &first.rotation;
        let rotation = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())
        });
        Rigid {
            rotation,
            translation: self.apply(&first.translation),
        }
    }

    /// Row-major 4x4 homogeneous matrix.
    pub fn to_matrix(&self) -> [f64; 16] {
        let r = &self.rotation;
        let t = &self.translation;
        [
            r[0][0], r[0][1], r[0][2], t[0], //
            r[1][0], r[1][1], r[1][2], t[1], //
            r[2][0], r[2][1], r[2][2], t[2], //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// Read a row-major 4x4 matrix; the bottom row is ignored.
    pub fn from_matrix(m: &[f64; 16]) -> Rigid {
        Rigid {
            rotation: [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]],
            translation: [m[3], m[7], m[11]],
        }
    }

    /// Express a transform that acts on coordinates relative to `origin`
    /// (`x - origin`) as one acting on absolute coordinates.
    fn around(&self, origin: &[f64; 3]) -> Rigid {
        // x' = R (x - o) + t + o  =>  translation = t + o - R o
        let ro = Rigid {
            rotation: self.rotation,
            translation: [0.0; 3],
        }
        .apply(origin);
        Rigid {
            rotation: self.rotation,
            translation: std::array::from_fn(|i| self.translation[i] + origin[i] - ro[i]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcpMetric {
    /// Distances along reference normals; converges in far fewer iterations
    /// on surfaces sampled differently in the two clouds.
    PointToPlane,
    PointToPoint,
}

#[derive(Debug, Clone, Copy)]
pub struct IcpParams {
    pub metric: IcpMetric,
    pub max_iterations: usize,
    /// Stop when the RMS improves by less than this fraction.
    pub tolerance: f64,
    /// At most this many moving points are used per iteration.
    pub sample: usize,
    /// Fraction of closest pairs kept each iteration (1.0 keeps all).
    pub overlap: f64,
    /// Translate the moving centroid onto the reference centroid first.
    pub match_centroids: bool,
}

impl Default for IcpParams {
    fn default() -> Self {
        Self {
            metric: IcpMetric::PointToPlane,
            max_iterations: 50,
            tolerance: 1e-6,
            sample: 50_000,
            overlap: 1.0,
            match_centroids: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcpResult {
    /// Maps the moving cloud onto the reference.
    pub transform: Rigid,
    /// RMS of the kept pair distances before the first and after the last step.
    pub rms_initial: f64,
    pub rms_final: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Register `moving` onto `reference`. Returns `None` if either cloud is
/// empty or `overlap` leaves fewer than three pairs.
pub fn icp(moving: &PointCloud, reference: &PointCloud, params: IcpParams) -> Option<IcpResult> {
    icp_with_scratch(moving, reference, params, true)
}

/// Optional paired f64 correspondence arithmetic; normals and tie order are unchanged.
pub fn icp_with_simd(
    moving: &PointCloud,
    reference: &PointCloud,
    params: IcpParams,
    simd: bool,
) -> Option<IcpResult> {
    if moving.is_empty() {
        return None;
    }
    IcpPrepared::new(reference)?.compute_with_simd(moving, params, simd)
}

/// Reuse normal-estimation query/coordinate scratch; matching arithmetic is unchanged.
pub fn icp_with_scratch(
    moving: &PointCloud,
    reference: &PointCloud,
    params: IcpParams,
    reuse: bool,
) -> Option<IcpResult> {
    if moving.is_empty() {
        return None;
    }
    IcpPrepared::new(reference)?.compute_with_scratch(moving, params, reuse)
}

/// Owned immutable reference tree and reusable lazy reference normals for ICP.
pub struct IcpPrepared {
    origin: [f64; 3],
    reference: Vec<[f64; 3]>,
    tree: KdTree,
    normals: Vec<Option<Option<[f64; 3]>>>,
    radius_hints: OnceLock<NearestRadiusHints>,
    temporal_queries: usize,
    temporal_scratch_bytes: usize,
}

impl IcpPrepared {
    pub fn new(reference: &PointCloud) -> Option<Self> {
        if reference.is_empty() {
            return None;
        }
        let origin = centroid(&reference.positions);
        let local: Vec<_> = reference
            .positions
            .iter()
            .map(|p| std::array::from_fn(|i| p[i] - origin[i]))
            .collect();
        let tree = KdTree::new(&local)?;
        let normals = vec![None; local.len()];
        Some(Self {
            origin,
            reference: local,
            tree,
            normals,
            radius_hints: OnceLock::new(),
            temporal_queries: 0,
            temporal_scratch_bytes: 0,
        })
    }

    pub fn retained_bytes(&self) -> usize {
        self.reference.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.tree.storage_bytes()
            + self.normals.capacity() * std::mem::size_of::<Option<Option<[f64; 3]>>>()
    }

    /// Queries whose prior-iteration hit tightened the original seed's radius.
    pub fn temporal_queries(&self) -> usize {
        self.temporal_queries
    }

    /// Last call's temporary slot-vector capacity, released before returning.
    pub fn temporal_scratch_bytes(&self) -> usize {
        self.temporal_scratch_bytes
    }

    pub fn compute(&mut self, moving: &PointCloud, params: IcpParams) -> Option<IcpResult> {
        self.compute_with_scratch(moving, params, true)
    }

    pub fn compute_with_scratch(
        &mut self,
        moving: &PointCloud,
        params: IcpParams,
        reuse: bool,
    ) -> Option<IcpResult> {
        self.compute_with_options(moving, params, reuse, false, false)
    }

    /// SIMD only changes leaf distance arithmetic, retaining the preceding
    /// sample's guess, strict tie comparisons and overlap-selection order.
    /// Native and non-SIMD WASM targets retain scalar arithmetic.
    pub fn compute_with_simd(
        &mut self,
        moving: &PointCloud,
        params: IcpParams,
        simd: bool,
    ) -> Option<IcpResult> {
        self.compute_with_correspondence(moving, params, simd, false)
    }

    /// Recompute a safe upper radius from each sample's previous correspondence;
    /// candidates never replace the original query seed or choose tied hits.
    pub fn compute_with_correspondence(
        &mut self,
        moving: &PointCloud,
        params: IcpParams,
        simd: bool,
        temporal: bool,
    ) -> Option<IcpResult> {
        self.compute_with_options(moving, params, true, simd, temporal)
    }

    fn compute_with_options(
        &mut self,
        moving: &PointCloud,
        params: IcpParams,
        reuse: bool,
        simd: bool,
        temporal: bool,
    ) -> Option<IcpResult> {
        self.temporal_queries = 0;
        self.temporal_scratch_bytes = 0;
        if moving.is_empty() {
            return None;
        }
        let origin = self.origin;
        let local = |p: &[f64; 3]| std::array::from_fn(|i| p[i] - origin[i]);
        let reference_local = &self.reference;
        let tree = &self.tree;
        let stride = moving.len().div_ceil(params.sample.max(3));
        let samples: Vec<[f64; 3]> = moving.positions.iter().step_by(stride).map(local).collect();
        let keep = ((samples.len() as f64 * params.overlap.clamp(0.0, 1.0)).round() as usize)
            .min(samples.len());
        if keep < 3 {
            return None;
        }

        let mut current = Rigid::IDENTITY;
        if params.match_centroids {
            let (mc, rc) = (centroid(&samples), centroid(reference_local));
            current.translation = std::array::from_fn(|i| rc[i] - mc[i]);
        }

        let mut normals = Normals::new(reference_local, tree, std::mem::take(&mut self.normals));
        normals.reuse = reuse;
        let mut pairs = Vec::with_capacity(samples.len());
        let mut search = CorrespondenceSearch {
            simd,
            hints: temporal.then(|| {
                self.radius_hints
                    .get_or_init(|| NearestRadiusHints::new(tree))
            }),
            slots: if temporal {
                vec![u32::MAX; samples.len()]
            } else {
                Vec::new()
            },
            bounded: 0,
        };
        let mut rms_initial = f64::NAN;
        let mut rms_previous = f64::INFINITY;
        let mut iterations = 0;
        let mut converged = false;
        let rms = loop {
            let rms = match_pairs(
                &samples,
                &current,
                tree,
                reference_local,
                keep,
                &mut pairs,
                &mut search,
            );
            if iterations == 0 {
                rms_initial = rms;
            }
            let stalled =
                rms_previous.is_finite() && rms_previous - rms <= params.tolerance * rms_previous;
            if rms == 0.0 || stalled {
                converged = true;
                break rms;
            }
            if iterations == params.max_iterations {
                break rms;
            }
            rms_previous = rms;
            let step = match params.metric {
                IcpMetric::PointToPoint => horn(&pairs),
                IcpMetric::PointToPlane => {
                    point_to_plane(&pairs, &mut normals).unwrap_or_else(|| horn(&pairs))
                }
            };
            current = step.compose(&current);
            iterations += 1;
        };
        self.normals = normals.cache;
        self.temporal_queries = search.bounded;
        self.temporal_scratch_bytes = search.slots.capacity() * std::mem::size_of::<u32>();
        Some(IcpResult {
            transform: current.around(&origin),
            rms_initial,
            rms_final: rms,
            iterations,
            converged,
        })
    }
}

/// Picked point pairs aligned by [`align_pairs`].
#[derive(Debug, Clone, PartialEq)]
pub struct PairAlignment {
    /// Maps the moving points onto the reference points.
    pub transform: Rigid,
    /// Distance left between each pair after the transform.
    pub residuals: Vec<f64>,
    pub rms: f64,
}

/// The rigid motion taking `moving[i]` onto `reference[i]` in the
/// least-squares sense. Returns `None` with fewer than three pairs, or when
/// either set of points is (nearly) collinear, which leaves the rotation
/// about that line undetermined.
pub fn align_pairs(moving: &[[f64; 3]], reference: &[[f64; 3]]) -> Option<PairAlignment> {
    if moving.len() != reference.len() || moving.len() < 3 {
        return None;
    }
    if [moving, reference].iter().any(|points| collinear(points)) {
        return None;
    }
    // Relative to the reference centroid, so georeferenced picks keep their precision.
    let origin = centroid(reference);
    let local = |p: &[f64; 3]| -> [f64; 3] { std::array::from_fn(|i| p[i] - origin[i]) };
    let pairs: Vec<Pair> = moving
        .iter()
        .zip(reference)
        .enumerate()
        .map(|(i, (m, r))| Pair {
            distance_sq: 0.0,
            moving: local(m),
            reference: local(r),
            reference_index: i,
        })
        .collect();
    let transform = horn(&pairs).around(&origin);
    let residuals: Vec<f64> = moving
        .iter()
        .zip(reference)
        .map(|(m, r)| {
            let p = transform.apply(m);
            (0..3).map(|a| (p[a] - r[a]).powi(2)).sum::<f64>().sqrt()
        })
        .collect();
    let rms = (residuals.iter().map(|d| d * d).sum::<f64>() / residuals.len() as f64).sqrt();
    Some(PairAlignment {
        transform,
        residuals,
        rms,
    })
}

/// Whether the points lie (almost) on a line: the spread across the main
/// direction is negligible next to the spread along it.
fn collinear(points: &[[f64; 3]]) -> bool {
    let c = centroid(points);
    let mut cov = [[0.0; 3]; 3];
    for p in points {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += (p[i] - c[i]) * (p[j] - c[j]);
            }
        }
    }
    let (mut values, _) = symmetric_eigen(cov);
    values.sort_by(|a, b| b.total_cmp(a));
    values[1].is_nan() || values[1] <= 1e-12 * values[0].max(f64::MIN_POSITIVE)
}

/// A moving point matched to a reference point.
struct Pair {
    distance_sq: f64,
    moving: [f64; 3],
    reference: [f64; 3],
    reference_index: usize,
}

/// Query state belongs to one fit and retains original sample indices.
struct CorrespondenceSearch<'a> {
    simd: bool,
    hints: Option<&'a NearestRadiusHints>,
    // Indexed by original sample order, independent of overlap partitioning.
    slots: Vec<u32>,
    bounded: usize,
}

impl CorrespondenceSearch<'_> {
    fn nearest(
        &mut self,
        tree: &KdTree,
        query: &[f64; 3],
        guess: Option<Nearest>,
        sample: usize,
    ) -> Nearest {
        if let Some(hints) = self.hints {
            let (hit, tighter) =
                tree.nearest_with_radius_hint(query, guess, self.slots[sample], hints, self.simd);
            self.slots[sample] = hit.tree_slot();
            self.bounded += usize::from(tighter);
            hit
        } else {
            tree.nearest_with_simd(query, guess, self.simd)
        }
    }
}

/// Pair every transformed sample with its nearest reference point, keep the
/// `keep` closest pairs, and return their RMS distance.
fn match_pairs(
    samples: &[[f64; 3]],
    transform: &Rigid,
    tree: &KdTree,
    reference: &[[f64; 3]],
    keep: usize,
    pairs: &mut Vec<Pair>,
    search: &mut CorrespondenceSearch,
) -> f64 {
    pairs.clear();
    let mut guess = None;
    if search.hints.is_none() {
        // Keep the historical hot loop when the experiment is disabled;
        // there is no per-query temporal-state dispatch or slot bookkeeping.
        let simd = search.simd;
        for s in samples {
            let p = transform.apply(s);
            let hit = tree.nearest_with_simd(&p, guess, simd);
            guess = Some(hit);
            pairs.push(Pair {
                distance_sq: hit.distance_sq,
                moving: p,
                reference: reference[hit.index],
                reference_index: hit.index,
            });
        }
    } else {
        for (i, s) in samples.iter().enumerate() {
            let p = transform.apply(s);
            let hit = search.nearest(tree, &p, guess, i);
            guess = Some(hit);
            pairs.push(Pair {
                distance_sq: hit.distance_sq,
                moving: p,
                reference: reference[hit.index],
                reference_index: hit.index,
            });
        }
    }
    if keep < pairs.len() {
        pairs.select_nth_unstable_by(keep - 1, |a, b| a.distance_sq.total_cmp(&b.distance_sq));
        pairs.truncate(keep);
    }
    (pairs.iter().map(|p| p.distance_sq).sum::<f64>() / pairs.len() as f64).sqrt()
}

/// Reference normals, estimated lazily by PCA over nearby points.
struct Normals<'a> {
    points: &'a [[f64; 3]],
    tree: &'a KdTree,
    /// `None` = not computed yet; `Some(None)` = degenerate neighbourhood.
    cache: Vec<Option<Option<[f64; 3]>>>,
    hits: Vec<(usize, f64)>,
    neighbours: Vec<[f64; 3]>,
    reuse: bool,
}

impl<'a> Normals<'a> {
    const NEIGHBOURS: usize = 12;

    fn new(points: &'a [[f64; 3]], tree: &'a KdTree, cache: Vec<Option<Option<[f64; 3]>>>) -> Self {
        Self {
            points,
            tree,
            cache,
            hits: Vec::new(),
            neighbours: Vec::new(),
            reuse: true,
        }
    }

    fn get(&mut self, index: usize) -> Option<[f64; 3]> {
        if let Some(normal) = self.cache[index] {
            return normal;
        }
        if self.reuse {
            self.tree
                .nearest_k_into(&self.points[index], Self::NEIGHBOURS, &mut self.hits);
            self.neighbours.clear();
            self.neighbours
                .extend(self.hits.iter().map(|&(i, _)| self.points[i]));
        } else {
            self.hits = self.tree.nearest_k(&self.points[index], Self::NEIGHBOURS);
            self.neighbours = self.hits.iter().map(|&(i, _)| self.points[i]).collect();
        }
        let normal = if self.hits.len() < 3 {
            None
        } else {
            let pts = &self.neighbours;
            let c = centroid(pts);
            let mut cov = [[0.0; 3]; 3];
            for p in pts {
                let d = std::array::from_fn(|i| p[i] - c[i]);
                crate::normals::accumulate_covariance(&mut cov, &d, false);
            }
            let (values, vectors) = jacobi(cov);
            let order = sorted_indices(&values);
            // A plane needs two clearly non-zero spreads.
            if values[order[1]] <= 1e-12 * values[order[2]].max(f64::MIN_POSITIVE) {
                None
            } else {
                let k = order[0];
                Some([vectors[0][k], vectors[1][k], vectors[2][k]])
            }
        };
        self.cache[index] = Some(normal);
        normal
    }
}

fn sorted_indices(values: &[f64; 3]) -> [usize; 3] {
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    order
}

// The triangular prototype is retained for comparison, but whole-RPC measured
// gains were inconsistent. Keep the original full matrix as the standard path.
const USE_TRIANGULAR_PLANE_SYSTEM: bool = false;

// Independent f64 columns retain the original accumulation order per cell.
// Whole-RPC measured gains vary by source, so keep pairing experimental.
const USE_SIMD_PLANE_SYSTEM: bool = false;

/// Finite products cannot create NaNs from a zero-initialized matrix. After
/// any unsafe scalar row, stay scalar for the rest of this system so existing
/// NaN/Inf entries retain historical propagation and operand order.
fn paired_plane_row_safe(row: &[f64; 6]) -> bool {
    row.iter().all(|v| v.is_finite() && v.abs() <= 1e150)
}

/// Linearised point-to-plane step: minimise `sum(((R p + t - q) . n)^2)`
/// for small rotations. `None` when too few pairs have normals.
fn point_to_plane(pairs: &[Pair], normals: &mut Normals) -> Option<Rigid> {
    let mut ata = [[0.0; 6]; 6];
    let mut atb = [0.0; 6];
    let mut symmetric = USE_TRIANGULAR_PLANE_SYSTEM;
    let mut paired = USE_SIMD_PLANE_SYSTEM
        && cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
        && !symmetric;
    let mut used = 0;
    for pair in pairs {
        let Some(n) = normals.get(pair.reference_index) else {
            continue;
        };
        let p = pair.moving;
        let c = [
            p[1] * n[2] - p[2] * n[1],
            p[2] * n[0] - p[0] * n[2],
            p[0] * n[1] - p[1] * n[0],
        ];
        let row = [c[0], c[1], c[2], n[0], n[1], n[2]];
        let residual: f64 = (0..3).map(|i| (pair.reference[i] - p[i]) * n[i]).sum();
        if paired && paired_plane_row_safe(&row) {
            accumulate_plane_row_paired(&mut ata, &mut atb, &row, residual);
        } else {
            paired = false;
            accumulate_plane_row(&mut ata, &mut atb, &row, residual, &mut symmetric);
        }
        used += 1;
    }
    if used < 6 {
        return None;
    }
    if symmetric {
        mirror_plane_system(&mut ata);
    }
    // A touch of damping keeps directions the scene cannot constrain (e.g.
    // sliding along a flat floor) near zero instead of blowing up.
    let trace: f64 = (0..6).map(|i| ata[i][i]).sum();
    for (i, row) in ata.iter_mut().enumerate() {
        row[i] += 1e-9 * trace / 6.0 + f64::MIN_POSITIVE;
    }
    let x = solve6(ata, atb)?;
    let (sa, ca) = x[0].sin_cos();
    let (sb, cb) = x[1].sin_cos();
    let (sg, cg) = x[2].sin_cos();
    // R = Rz(g) Ry(b) Rx(a)
    let rotation = [
        [cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa],
        [sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa],
        [-sb, cb * sa, cb * ca],
    ];
    Some(Rigid {
        rotation,
        translation: [x[3], x[4], x[5]],
    })
}

/// Finite multiplication is commutative bit-for-bit, so the two halves of
/// each outer product have identical accumulation histories. Keep only the
/// upper half until the solve. Nonfinite rows retain the original operand
/// order in both halves, including NaN propagation.
fn accumulate_plane_row(
    ata: &mut [[f64; 6]; 6],
    atb: &mut [f64; 6],
    row: &[f64; 6],
    residual: f64,
    symmetric: &mut bool,
) {
    if *symmetric && row.iter().any(|v| !v.is_finite()) {
        mirror_plane_system(ata);
        *symmetric = false;
    }
    if *symmetric {
        // Fixed row indices let the compiler simplify the short triangular
        // loops instead of selecting their first column on every row.
        accumulate_plane_upper::<0>(ata, atb, row, residual);
        accumulate_plane_upper::<1>(ata, atb, row, residual);
        accumulate_plane_upper::<2>(ata, atb, row, residual);
        accumulate_plane_upper::<3>(ata, atb, row, residual);
        accumulate_plane_upper::<4>(ata, atb, row, residual);
        accumulate_plane_upper::<5>(ata, atb, row, residual);
        return;
    }
    for i in 0..6 {
        atb[i] += row[i] * residual;
        for j in 0..6 {
            ata[i][j] += row[i] * row[j];
        }
    }
}

/// Caller keeps every row bounded and never enters here after a scalar row.
/// Products are finite; even accumulated overflow only adds finite terms to
/// infinity. RHS arithmetic stays scalar and preserves each original operand.
fn accumulate_plane_row_paired(
    ata: &mut [[f64; 6]; 6],
    atb: &mut [f64; 6],
    row: &[f64; 6],
    residual: f64,
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::*;
        // SAFETY: these three unaligned loads cover exactly six f64 values;
        // each matrix load/store covers a complete pair within its 6-f64 row.
        unsafe {
            let columns = [
                v128_load(row.as_ptr().cast()),
                v128_load(row.as_ptr().add(2).cast()),
                v128_load(row.as_ptr().add(4).cast()),
            ];
            for i in 0..6 {
                atb[i] += row[i] * residual;
                let factor = f64x2_splat(row[i]);
                for (j, columns) in columns.iter().enumerate() {
                    let target = ata[i].as_mut_ptr().add(j * 2).cast();
                    v128_store(
                        target,
                        f64x2_add(v128_load(target), f64x2_mul(factor, *columns)),
                    );
                }
            }
        }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    accumulate_plane_row(ata, atb, row, residual, &mut false);
}

fn accumulate_plane_upper<const I: usize>(
    ata: &mut [[f64; 6]; 6],
    atb: &mut [f64; 6],
    row: &[f64; 6],
    residual: f64,
) {
    atb[I] += row[I] * residual;
    for j in I..6 {
        ata[I][j] += row[I] * row[j];
    }
}

fn mirror_plane_system(ata: &mut [[f64; 6]; 6]) {
    for i in 1..6 {
        let (upper, lower) = ata.split_at_mut(i);
        for (value, row) in lower[0][..i].iter_mut().zip(upper) {
            *value = row[i];
        }
    }
}

/// Solve a 6x6 linear system by Gaussian elimination with partial pivoting.
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let pivot = (col..6).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let pivot_row = a[col];
        for row in col + 1..6 {
            let f = a[row][col] / pivot_row[col];
            for (value, &p) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *value -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 6];
    for row in (0..6).rev() {
        let s: f64 = (row + 1..6).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for p in points {
        for a in 0..3 {
            sum[a] += p[a];
        }
    }
    sum.map(|s| s / points.len() as f64)
}

/// Least-squares rigid motion taking each moving point onto its reference
/// point (Horn 1987).
fn horn(pairs: &[Pair]) -> Rigid {
    rigid_fit(pairs.iter().map(|pair| (pair.moving, pair.reference)))
}

/// Least-squares rigid motion taking each `p` onto its `q` in `(p, q)`
/// pairs, always a proper rotation (Horn's closed form, as [`horn`]).
pub(crate) fn rigid_fit<I>(pairs: I) -> Rigid
where
    I: Iterator<Item = ([f64; 3], [f64; 3])> + Clone,
{
    let n = pairs.clone().count() as f64;
    let mut mp = [0.0; 3];
    let mut mq = [0.0; 3];
    for (p, q) in pairs.clone() {
        for a in 0..3 {
            mp[a] += p[a] / n;
            mq[a] += q[a] / n;
        }
    }
    // Cross-covariance S[i][j] = sum (p_i - mp_i)(q_j - mq_j).
    let mut s = [[0.0; 3]; 3];
    for (p, q) in pairs {
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += (p[i] - mp[i]) * (q[j] - mq[j]);
            }
        }
    }
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = s;
    let m = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let (values, vectors) = jacobi(m);
    let best = (0..4)
        .max_by(|&i, &j| values[i].total_cmp(&values[j]))
        .unwrap();
    let q = [
        vectors[0][best],
        vectors[1][best],
        vectors[2][best],
        vectors[3][best],
    ];
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let [w, x, y, z] = q.map(|v| v / norm);
    let rotation = [
        [
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ];
    let r_mp = Rigid {
        rotation,
        translation: [0.0; 3],
    }
    .apply(&mp);
    Rigid {
        rotation,
        translation: std::array::from_fn(|i| mq[i] - r_mp[i]),
    }
}

/// Eigen-decomposition of a symmetric 3x3 matrix (see [`jacobi`]).
pub(crate) fn symmetric_eigen(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    jacobi(m)
}

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations:
/// eigenvalues and the matrix whose columns are the eigenvectors.
fn jacobi<const N: usize>(mut a: [[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut v = [[0.0; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..50 {
        let off: f64 = (0..N)
            .flat_map(|i| (0..N).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let scale: f64 = (0..N).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-30 * scale.max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..N - 1 {
            for q in p + 1..N {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (row_p, row_q) = (a[p], a[q]);
                for k in 0..N {
                    a[p][k] = c * row_p[k] - s * row_q[k];
                    a[q][k] = s * row_p[k] + c * row_q[k];
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    (std::array::from_fn(|i| a[i][i]), v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_plane_rows_preserve_full_system_and_guard_extreme_inputs() {
        let mut seed = 0x3910_449b_13e7_a91f_u64;
        for scale in [0.0, 1e-160, 1.0, 1e150] {
            let mut expected = [[0.0; 6]; 6];
            let mut rhs = [0.0; 6];
            let mut paired = expected;
            let mut paired_rhs = rhs;
            for step in 0..2048 {
                let mut next = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0) * scale
                };
                let row = std::array::from_fn(|_| next());
                let residual = if step % 17 == 0 { f64::NAN } else { next() };
                assert!(paired_plane_row_safe(&row));
                accumulate_plane_row(&mut expected, &mut rhs, &row, residual, &mut false);
                accumulate_plane_row_paired(&mut paired, &mut paired_rhs, &row, residual);
                assert_eq!(
                    expected.map(|r| r.map(f64::to_bits)),
                    paired.map(|r| r.map(f64::to_bits))
                );
                assert_eq!(rhs.map(f64::to_bits), paired_rhs.map(f64::to_bits));
            }
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e151, -1e151] {
            for axis in 0..6 {
                let mut row = [0.0; 6];
                row[axis] = bad;
                assert!(!paired_plane_row_safe(&row));
            }
        }
    }

    #[test]
    fn temporal_correspondences_keep_overlap_pair_order_and_complete_fits() {
        let same = |a: IcpResult, b: IcpResult| {
            assert_eq!(
                a.transform.to_matrix().map(f64::to_bits),
                b.transform.to_matrix().map(f64::to_bits)
            );
            assert_eq!(
                (
                    a.rms_initial.to_bits(),
                    a.rms_final.to_bits(),
                    a.iterations,
                    a.converged
                ),
                (
                    b.rms_initial.to_bits(),
                    b.rms_final.to_bits(),
                    b.iterations,
                    b.converged
                )
            );
        };
        for count in [17, 257, 1500] {
            let mut reference = surface(count, 71);
            reference[1] = reference[0];
            let tree = KdTree::new(&reference).unwrap();
            let hints = NearestRadiusHints::new(&tree);
            for keep in [3, count / 2, count] {
                let mut scalar = Vec::new();
                let mut bounded = Vec::new();
                let mut control = CorrespondenceSearch {
                    simd: false,
                    hints: None,
                    slots: Vec::new(),
                    bounded: 0,
                };
                let mut temporal = CorrespondenceSearch {
                    simd: false,
                    hints: Some(&hints),
                    slots: vec![u32::MAX; count],
                    bounded: 0,
                };
                for z in [0.0, 0.02, -0.01, 0.03] {
                    let transform = Rigid {
                        rotation: Rigid::IDENTITY.rotation,
                        translation: [0.0, 0.0, z],
                    };
                    let a = match_pairs(
                        &reference,
                        &transform,
                        &tree,
                        &reference,
                        keep,
                        &mut scalar,
                        &mut control,
                    );
                    let b = match_pairs(
                        &reference,
                        &transform,
                        &tree,
                        &reference,
                        keep,
                        &mut bounded,
                        &mut temporal,
                    );
                    assert_eq!(a.to_bits(), b.to_bits());
                    for (a, b) in scalar.iter().zip(&bounded) {
                        assert_eq!(
                            (
                                a.reference_index,
                                a.distance_sq.to_bits(),
                                a.moving.map(f64::to_bits)
                            ),
                            (
                                b.reference_index,
                                b.distance_sq.to_bits(),
                                b.moving.map(f64::to_bits)
                            )
                        );
                    }
                }
                assert!(temporal.bounded > 0);
            }
            let reference = cloud(reference);
            let mut prepared = IcpPrepared::new(&reference).unwrap();
            let bytes = prepared.retained_bytes();
            for (shift, sample, overlap, centroids) in [
                (0.02, 50_000, 0.8, false),
                (-0.03, 50, 1.0, true),
                (0.0, 3, 1.0, false),
            ] {
                let moving = cloud(
                    reference
                        .positions
                        .iter()
                        .map(|p| [p[0], p[1], p[2] + shift])
                        .collect(),
                );
                for metric in [IcpMetric::PointToPoint, IcpMetric::PointToPlane] {
                    let params = IcpParams {
                        metric,
                        sample,
                        overlap,
                        match_centroids: centroids,
                        max_iterations: 8,
                        ..Default::default()
                    };
                    let expected = IcpPrepared::new(&reference)
                        .unwrap()
                        .compute_with_correspondence(&moving, params, false, false)
                        .unwrap();
                    for simd in [false, true] {
                        same(
                            expected,
                            prepared
                                .compute_with_correspondence(&moving, params, simd, true)
                                .unwrap(),
                        );
                        assert_eq!(bytes, prepared.retained_bytes());
                        let samples = moving.len().div_ceil(moving.len().div_ceil(sample.max(3)));
                        assert_eq!(prepared.temporal_scratch_bytes(), samples * 4);
                        same(
                            expected,
                            prepared
                                .compute_with_correspondence(&moving, params, simd, false)
                                .unwrap(),
                        );
                        assert_eq!(prepared.temporal_queries(), 0);
                        assert_eq!(prepared.temporal_scratch_bytes(), 0);
                    }
                }
            }
            assert!(
                prepared
                    .compute(&PointCloud::default(), IcpParams::default())
                    .is_none()
            );
            assert_eq!(prepared.temporal_queries(), 0);
            assert_eq!(prepared.temporal_scratch_bytes(), 0);
        }
    }

    #[test]
    fn paired_correspondences_preserve_pairs_overlap_and_complete_fits() {
        let same = |a: IcpResult, b: IcpResult| {
            assert_eq!(
                a.transform.to_matrix().map(f64::to_bits),
                b.transform.to_matrix().map(f64::to_bits)
            );
            assert_eq!(
                (
                    a.rms_initial.to_bits(),
                    a.rms_final.to_bits(),
                    a.iterations,
                    a.converged
                ),
                (
                    b.rms_initial.to_bits(),
                    b.rms_final.to_bits(),
                    b.iterations,
                    b.converged
                )
            );
        };
        for count in [3, 17, 65, 257] {
            let mut reference = surface(count, 93);
            if count > 3 {
                reference[1] = reference[0];
                reference[2] = reference[0];
            }
            let tree = KdTree::new(&reference).unwrap();
            for shift in [0.0, 0.02, -0.03] {
                let moving: Vec<_> = reference
                    .iter()
                    .map(|p| [p[0], p[1], p[2] + shift])
                    .collect();
                for keep in [3, count] {
                    let mut scalar = Vec::new();
                    let mut paired = Vec::new();
                    let mut scalar_search = CorrespondenceSearch {
                        simd: false,
                        hints: None,
                        slots: Vec::new(),
                        bounded: 0,
                    };
                    let mut paired_search = CorrespondenceSearch {
                        simd: true,
                        hints: None,
                        slots: Vec::new(),
                        bounded: 0,
                    };
                    let a = match_pairs(
                        &moving,
                        &Rigid::IDENTITY,
                        &tree,
                        &reference,
                        keep,
                        &mut scalar,
                        &mut scalar_search,
                    );
                    let b = match_pairs(
                        &moving,
                        &Rigid::IDENTITY,
                        &tree,
                        &reference,
                        keep,
                        &mut paired,
                        &mut paired_search,
                    );
                    assert_eq!(a.to_bits(), b.to_bits());
                    for (a, b) in scalar.iter().zip(&paired) {
                        assert_eq!(a.reference_index, b.reference_index);
                        assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
                        assert_eq!(a.moving.map(f64::to_bits), b.moving.map(f64::to_bits));
                        assert_eq!(a.reference.map(f64::to_bits), b.reference.map(f64::to_bits));
                    }
                }
                let (moving, reference) = (cloud(moving), cloud(reference.clone()));
                let mut prepared = IcpPrepared::new(&reference).unwrap();
                let bytes = prepared.retained_bytes();
                for metric in [IcpMetric::PointToPoint, IcpMetric::PointToPlane] {
                    for (sample, overlap, match_centroids) in
                        [(3, 1.0, false), (50, 0.8, true), (50_000, 1.0, false)]
                    {
                        let params = IcpParams {
                            metric,
                            sample,
                            overlap,
                            match_centroids,
                            max_iterations: 8,
                            ..Default::default()
                        };
                        let Some(expected) = icp_with_simd(&moving, &reference, params, false)
                        else {
                            assert!(icp_with_simd(&moving, &reference, params, true).is_none());
                            assert!(prepared.compute_with_simd(&moving, params, true).is_none());
                            continue;
                        };
                        same(
                            expected,
                            icp_with_simd(&moving, &reference, params, true).unwrap(),
                        );
                        same(
                            expected,
                            prepared.compute_with_simd(&moving, params, true).unwrap(),
                        );
                        same(
                            expected,
                            prepared.compute_with_simd(&moving, params, false).unwrap(),
                        );
                        assert_eq!(bytes, prepared.retained_bytes());
                    }
                }
            }
        }
    }

    #[test]
    fn triangular_plane_system_preserves_full_matrix_and_rhs_bits() {
        let mut seed = 0x81de_b00b_573a_922d_u64;
        for mode in 0..6 {
            let mut full = [[0.0; 6]; 6];
            let mut rhs = [0.0; 6];
            let mut upper = full;
            let mut upper_rhs = rhs;
            let mut symmetric = true;
            for step in 0..2048 {
                let mut next = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let v = (seed >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0;
                    match mode {
                        1 => v * 1e-160,
                        2 => v * 1e155,
                        3 => {
                            if seed & 1 == 0 {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        _ => v * 1e5,
                    }
                };
                let mut row = std::array::from_fn(|_| next());
                let mut residual = next();
                if mode == 4 && step >= 1024 && step % 17 == 0 {
                    row[step % 6] = [
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::from_bits(0x7ff8_0000_0000_1234),
                        f64::from_bits(0xfff8_0000_0000_5678),
                    ][step % 4];
                }
                if mode == 5 && step % 13 == 0 {
                    residual = f64::from_bits(0x7ff8_0000_0000_9876);
                }
                for i in 0..6 {
                    rhs[i] += row[i] * residual;
                    for j in 0..6 {
                        full[i][j] += row[i] * row[j];
                    }
                }
                accumulate_plane_row(&mut upper, &mut upper_rhs, &row, residual, &mut symmetric);
                let mut expanded = upper;
                if symmetric {
                    mirror_plane_system(&mut expanded);
                }
                assert_eq!(
                    expanded.map(|r| r.map(f64::to_bits)),
                    full.map(|r| r.map(f64::to_bits)),
                    "mode={mode}, step={step}"
                );
                assert_eq!(
                    upper_rhs.map(f64::to_bits),
                    rhs.map(f64::to_bits),
                    "mode={mode}, step={step}"
                );
            }
        }
    }

    fn rotation(rx: f64, ry: f64, rz: f64) -> [[f64; 3]; 3] {
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        let x = Rigid {
            rotation: [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
            translation: [0.0; 3],
        };
        let y = Rigid {
            rotation: [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            translation: [0.0; 3],
        };
        let z = Rigid {
            rotation: [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        };
        z.compose(&y.compose(&x)).rotation
    }

    /// A bumpy 40 m x 30 m surface, offset like UTM data.
    fn surface(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (x, y) = (next() * 40.0, next() * 30.0);
                let z = 2.0 * (x / 5.0).sin() * (y / 4.0).cos() + 0.05 * x;
                [x + 368_000.0, y + 3_955_000.0, z + 40.0]
            })
            .collect()
    }

    fn cloud(positions: Vec<[f64; 3]>) -> PointCloud {
        PointCloud {
            positions,
            colors: None,
            attributes: Vec::new(),
        }
    }

    /// Largest disagreement between two transforms over `points`. Comparing
    /// raw translations would be meaningless for georeferenced data, where a
    /// micro-radian rotation about the far-away origin moves metres.
    fn displacement_error(a: &Rigid, b: &Rigid, points: &[[f64; 3]]) -> f64 {
        points
            .iter()
            .map(|p| {
                let (x, y) = (a.apply(p), b.apply(p));
                (0..3).map(|i| (x[i] - y[i]).powi(2)).sum::<f64>().sqrt()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn horn_recovers_an_exact_motion() {
        let truth = Rigid {
            rotation: rotation(0.3, -0.2, 1.1),
            translation: [1.0, -2.0, 0.5],
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ];
        let pairs: Vec<_> = points
            .iter()
            .map(|p| Pair {
                distance_sq: 0.0,
                moving: *p,
                reference: truth.apply(p),
                reference_index: 0,
            })
            .collect();
        let error = displacement_error(&horn(&pairs), &truth, &points);
        assert!(error < 1e-12, "{error}");
    }

    #[test]
    fn point_pairs_recover_a_georeferenced_motion() {
        let truth = Rigid {
            rotation: rotation(0.01, -0.02, 0.4),
            translation: [2.5, -1.0, 0.3],
        };
        // Four picks on a UTM-sized scene, one of them off by 1 cm.
        let reference: Vec<[f64; 3]> = surface(4, 7);
        let inverse = |p: &[f64; 3]| {
            // moving = truth⁻¹(reference)
            let r = &truth.rotation;
            let d: [f64; 3] = std::array::from_fn(|i| p[i] - truth.translation[i]);
            std::array::from_fn(|i| (0..3).map(|k| r[k][i] * d[k]).sum())
        };
        let mut moving: Vec<[f64; 3]> = reference.iter().map(inverse).collect();
        let exact = align_pairs(&moving, &reference).unwrap();
        let error = displacement_error(&exact.transform, &truth, &moving);
        assert!(error < 1e-6, "{error}");
        assert!(exact.rms < 1e-6, "{}", exact.rms);

        moving[2][0] += 0.01;
        let noisy = align_pairs(&moving, &reference).unwrap();
        assert!(noisy.rms > 1e-3 && noisy.rms < 0.01, "{}", noisy.rms);
        assert_eq!(noisy.residuals.len(), 4);
    }

    #[test]
    fn point_pairs_need_three_non_collinear_points() {
        let line = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let plane = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(align_pairs(&line, &line).is_none());
        assert!(align_pairs(&plane[..2], &plane[..2]).is_none());
        assert!(align_pairs(&plane, &plane).is_some());
    }

    #[test]
    fn matrix_roundtrip_and_composition() {
        let a = Rigid {
            rotation: rotation(0.1, 0.2, 0.3),
            translation: [1.0, 2.0, 3.0],
        };
        assert_eq!(Rigid::from_matrix(&a.to_matrix()), a);
        let p = [4.0, -5.0, 6.0];
        let b = Rigid {
            rotation: rotation(-0.4, 0.0, 0.9),
            translation: [0.0, 1.0, 0.0],
        };
        let composed = b.compose(&a).apply(&p);
        let sequential = b.apply(&a.apply(&p));
        for i in 0..3 {
            assert!((composed[i] - sequential[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn registers_a_displaced_scan() {
        let reference = cloud(surface(40_000, 1));
        // A different sampling of the same surface, displaced about its middle.
        let center = [368_020.0, 3_955_015.0, 40.0];
        let truth = Rigid {
            rotation: rotation(0.02, -0.015, 0.05),
            translation: [0.4, -0.3, 0.15],
        }
        .around(&center);
        let inverse_truth = invert(&truth);
        let moving = cloud(
            surface(20_000, 2)
                .iter()
                .map(|p| inverse_truth.apply(p))
                .collect(),
        );

        for metric in [IcpMetric::PointToPlane, IcpMetric::PointToPoint] {
            let params = IcpParams {
                metric,
                max_iterations: 200,
                ..IcpParams::default()
            };
            let result = icp(&moving, &reference, params).unwrap();
            let error = displacement_error(&result.transform, &truth, &moving.positions);
            assert!(
                result.rms_final < result.rms_initial,
                "{metric:?} {result:?}"
            );
            // Point-to-plane is exact up to sampling; point-to-point slides
            // slowly along the surface, so it only gets close.
            let limit = if metric == IcpMetric::PointToPlane {
                0.01
            } else {
                0.1
            };
            assert!(error < limit, "{metric:?} error={error} {result:?}");
            if metric == IcpMetric::PointToPlane {
                assert!(result.converged && result.iterations < 30, "{result:?}");
            }
        }
    }

    #[test]
    fn overlap_trimming_ignores_extra_points() {
        let reference = cloud(surface(30_000, 3));
        let shift = Rigid {
            rotation: rotation(0.0, 0.0, 0.0),
            translation: [0.3, 0.2, -0.1],
        };
        let mut moving: Vec<[f64; 3]> = surface(15_000, 4).iter().map(|p| shift.apply(p)).collect();
        // 20 % outliers floating well above the surface.
        let outliers: Vec<[f64; 3]> = moving
            .iter()
            .step_by(4)
            .map(|p| [p[0], p[1], p[2] + 8.0])
            .collect();
        moving.extend(outliers);
        let params = IcpParams {
            overlap: 0.75,
            ..IcpParams::default()
        };
        let moving = cloud(moving);
        let result = icp(&moving, &reference, params).unwrap();
        let truth = Rigid {
            rotation: rotation(0.0, 0.0, 0.0),
            translation: [-0.3, -0.2, 0.1],
        };
        let error = displacement_error(&result.transform, &truth, &moving.positions[..15_000]);
        assert!(error < 0.02, "error={error} {result:?}");
    }

    #[test]
    fn match_centroids_handles_large_offsets() {
        let reference = cloud(surface(20_000, 5));
        let moving = cloud(
            surface(20_000, 6)
                .iter()
                .map(|p| [p[0] + 500.0, p[1] - 200.0, p[2] + 30.0])
                .collect(),
        );
        let params = IcpParams {
            match_centroids: true,
            ..IcpParams::default()
        };
        let result = icp(&moving, &reference, params).unwrap();
        let truth = Rigid {
            rotation: rotation(0.0, 0.0, 0.0),
            translation: [-500.0, 200.0, -30.0],
        };
        let error = displacement_error(&result.transform, &truth, &moving.positions);
        assert!(error < 0.02, "error={error} {result:?}");
    }

    fn invert(t: &Rigid) -> Rigid {
        let r = t.rotation;
        let rotation = std::array::from_fn(|i| std::array::from_fn(|j| r[j][i]));
        let inv = Rigid {
            rotation,
            translation: [0.0; 3],
        };
        let rt = inv.apply(&t.translation);
        Rigid {
            rotation,
            translation: rt.map(|v| -v),
        }
    }
}
