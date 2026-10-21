//! Cloud-to-cloud (C2C) nearest-neighbour distances.

use crate::PointCloud;
use crate::kdtree::KdTree;
use crate::mesh::{MeshBvh, TriangleMesh, dot};

/// Summary statistics over a set of distances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub rms: f64,
    pub std_dev: f64,
    pub median: f64,
}

impl DistanceStats {
    pub fn from_distances(distances: &[f64]) -> Option<Self> {
        Self::from_distances_with_median_sort(distances, false)
    }

    /// Keep full sorting as an exact control for median selection benchmarks.
    /// All other reductions retain the input traversal and arithmetic order.
    pub fn from_distances_with_median_sort(distances: &[f64], sort_median: bool) -> Option<Self> {
        if distances.is_empty() {
            return None;
        }
        let n = distances.len() as f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for &d in distances {
            min = min.min(d);
            max = max.max(d);
            sum += d;
            sum_sq += d * d;
        }
        let mean = sum / n;
        let even = distances.len().is_multiple_of(2);
        let median = if sort_median {
            let mut values = distances.to_vec();
            values.sort_unstable_by(f64::total_cmp);
            let mid = values.len() / 2;
            if even {
                0.5 * (values[mid - 1] + values[mid])
            } else {
                values[mid]
            }
        } else if min == max && min != 0.0 && !sum.is_nan() {
            // Equal nonzero extrema and a non-NaN sum imply one identical
            // distance throughout. Exclude mixed signed zeros and ignored NaNs.
            // Retain the original overflowing/subnormal midpoint arithmetic.
            if even { 0.5 * (min + max) } else { min }
        } else if let Some(ordered) = ordered_median(distances, min, max) {
            ordered
        } else {
            let mut values = distances.to_vec();
            let mid = values.len() / 2;
            let (lower, middle, _) = values.select_nth_unstable_by(mid, f64::total_cmp);
            if even {
                // Select in total order, including signed zero and NaN payloads.
                // The lower partition contains exactly the preceding values.
                let low = lower.iter().copied().max_by(f64::total_cmp).unwrap();
                0.5 * (low + *middle)
            } else {
                *middle
            }
        };
        Some(Self {
            count: distances.len(),
            min,
            max,
            mean,
            rms: (sum_sq / n).sqrt(),
            std_dev: (sum_sq / n - mean * mean).max(0.0).sqrt(),
            median,
        })
    }
}

/// Only scan for order when the endpoints match the already-computed extrema.
/// The complete total-order check is required: matching endpoints alone cannot
/// establish that the interior is sorted. Nonempty input is guaranteed above.
fn ordered_median(distances: &[f64], min: f64, max: f64) -> Option<f64> {
    let first = distances[0].to_bits();
    let last = distances[distances.len() - 1].to_bits();
    let ascending = first == min.to_bits()
        && last == max.to_bits()
        && distances.windows(2).all(|p| p[0].total_cmp(&p[1]).is_le());
    let descending = !ascending
        && first == max.to_bits()
        && last == min.to_bits()
        && distances.windows(2).all(|p| p[0].total_cmp(&p[1]).is_ge());
    if !ascending && !descending {
        return None;
    }
    let mid = distances.len() / 2;
    let upper = if ascending {
        mid
    } else {
        distances.len() - 1 - mid
    };
    Some(if distances.len().is_multiple_of(2) {
        let lower = if ascending {
            mid - 1
        } else {
            distances.len() - mid
        };
        0.5 * (distances[lower] + distances[upper])
    } else {
        distances[upper]
    })
}

/// For every point of `points`, the distance to the closest point of `mesh`
/// (cloud-to-mesh, C2M). With `signed`, points on the back side of the
/// closest triangle (against its right-hand normal) get negative distances.
/// Returns `None` when the mesh has no triangles.
/// A query with no computable triangle hit returns positive infinity, even
/// when signed; it does not seed the next query's guessed triangle.
pub fn cloud_to_mesh(points: &[[f64; 3]], mesh: &TriangleMesh, signed: bool) -> Option<Vec<f64>> {
    let bvh = MeshBvh::new(mesh)?;
    Some(distances_with_mesh_bvh(points, mesh, &bvh, signed))
}

const USE_MESH_STACK_BOUNDS: bool = true;

pub(crate) fn distances_with_mesh_bvh(
    points: &[[f64; 3]],
    mesh: &TriangleMesh,
    bvh: &MeshBvh<'_>,
    signed: bool,
) -> Vec<f64> {
    let order = morton_order(points);
    distances_with_mesh_bvh_ordered(points, mesh, bvh, &order, signed)
}

pub(crate) fn distances_with_mesh_bvh_ordered(
    points: &[[f64; 3]],
    mesh: &TriangleMesh,
    bvh: &MeshBvh<'_>,
    order: &[usize],
    signed: bool,
) -> Vec<f64> {
    let mut distances = vec![0.0; points.len()];
    let mut guess = None;
    let mut stack = Vec::new();
    let mut bound_stack = Vec::new();
    for &i in order {
        let p = points[i];
        let hit = if USE_MESH_STACK_BOUNDS {
            bvh.nearest_with_cached_bounds(p, guess, &mut bound_stack)
        } else {
            bvh.nearest_with(p, guess, &mut stack)
        };
        if hit.triangle == usize::MAX {
            // Nonfinite coordinates or overflow can leave the initial hit
            // unchanged. Never index the mesh or seed a later query with it.
            guess = None;
            distances[i] = f64::INFINITY;
            continue;
        }
        guess = Some(hit.triangle);
        let mut d = hit.distance_sq.sqrt();
        if signed {
            let offset = [
                p[0] - hit.point[0],
                p[1] - hit.point[1],
                p[2] - hit.point[2],
            ];
            if dot(offset, mesh.normal(hit.triangle)) < 0.0 {
                d = -d;
            }
        }
        distances[i] = d;
    }
    distances
}

/// For every point of `compared`, the Euclidean distance to its nearest
/// neighbour in `reference`. Returns `None` when `reference` is empty.
pub fn cloud_to_cloud(compared: &PointCloud, reference: &PointCloud) -> Option<Vec<f64>> {
    cloud_to_cloud_with_simd(compared, reference, false)
}

/// Select paired f64 WASM leaf arithmetic without changing the tree, Morton
/// query order or guessed hits. Native and non-SIMD builds use scalar leaves.
pub fn cloud_to_cloud_with_simd(
    compared: &PointCloud,
    reference: &PointCloud,
    simd: bool,
) -> Option<Vec<f64>> {
    if reference.is_empty() {
        return None;
    }
    let tree = KdTree::new(&reference.positions)?;
    let order = morton_order(&compared.positions);
    Some(distances_with_tree(
        &compared.positions,
        &tree,
        &order,
        simd,
    ))
}

/// Immutable worker preparation for repeated C2C jobs on the same snapshots.
/// Queries, their Morton order and the reference tree are owned independently
/// of source clouds; each call starts with a fresh guessed hit and output.
pub struct PreparedC2c {
    queries: Vec<[f64; 3]>,
    tree: KdTree,
    order: Vec<usize>,
}

impl PreparedC2c {
    pub fn new(queries: Vec<[f64; 3]>, reference: &[[f64; 3]]) -> Option<Self> {
        let tree = KdTree::new(reference)?;
        let order = morton_order(&queries);
        Some(Self {
            queries,
            tree,
            order,
        })
    }

    pub fn distances(&self, simd: bool) -> Vec<f64> {
        distances_with_tree(&self.queries, &self.tree, &self.order, simd)
    }

    /// Owned tree, query and Morton-order vector capacity, excluding the allocator.
    pub fn retained_bytes(&self) -> usize {
        self.tree.storage_bytes()
            + self.queries.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.order.capacity() * std::mem::size_of::<usize>()
    }

    /// Share the immutable tree and cached query order across Rayon lanes.
    /// Each Morton chunk starts with its own guessed hit; output stays in query order.
    #[cfg(feature = "parallel")]
    pub fn distances_par(&self) -> Vec<f64> {
        let query = |p: &[f64; 3], guess: &mut Option<crate::kdtree::Nearest>| {
            let hit = self.tree.nearest(p, *guess);
            *guess = Some(hit);
            hit.distance_sq.sqrt()
        };
        let parts =
            par_morton_values::<Option<crate::kdtree::Nearest>>(&self.queries, &self.order, &query);
        let mut distances = vec![0.0; self.queries.len()];
        for (&index, distance) in self.order.iter().zip(parts.into_iter().flatten()) {
            distances[index] = distance;
        }
        distances
    }
}

/// Owned reference tree for repeated C2C queries, including changed query clouds.
/// Query order, guessed hits and output remain local to each call.
pub struct PreparedC2cReference {
    tree: KdTree,
}

impl PreparedC2cReference {
    /// Copy reference coordinates into the existing immutable k-d tree.
    pub fn new(reference: &[[f64; 3]]) -> Option<Self> {
        Some(Self {
            tree: KdTree::new(reference)?,
        })
    }

    /// Native vector payload capacity, excluding allocator overhead and queries.
    pub fn retained_bytes(&self) -> usize {
        self.tree.storage_bytes()
    }

    pub fn distances(&self, compared: &[[f64; 3]]) -> Vec<f64> {
        let order = morton_order(compared);
        distances_with_tree(compared, &self.tree, &order, false)
    }

    /// Same Morton chunks and scalar leaf policy as `cloud_to_cloud_par`.
    #[cfg(feature = "parallel")]
    pub fn distances_par(&self, compared: &[[f64; 3]]) -> Vec<f64> {
        distances_with_tree_par(compared, &self.tree)
    }
}

fn distances_with_tree(
    queries: &[[f64; 3]],
    tree: &KdTree,
    order: &[usize],
    simd: bool,
) -> Vec<f64> {
    let mut distances = vec![0.0; queries.len()];
    // Querying in spatial order keeps the tree's hot path in cache, and the
    // previous hit is an excellent starting guess for the next query.
    let mut guess = None;
    for &i in order {
        let hit = tree.nearest_with_simd(&queries[i], guess, simd);
        distances[i] = hit.distance_sq.sqrt();
        guess = Some(hit);
    }
    distances
}

/// One independent slice of a C2C job: `cloud_to_cloud` on these compared
/// points against these reference points gives exactly the same distances as
/// running it on the full clouds.
#[derive(Debug, Clone, PartialEq)]
pub struct C2cPart {
    /// Indices into the compared cloud.
    pub queries: Vec<u32>,
    /// Indices into the reference cloud that can be nearest to any query.
    pub reference: Vec<u32>,
}

/// Multi-threaded [`cloud_to_cloud`] over plain point slices (feature
/// `parallel`). Queries are taken in Morton order and split into chunks, each
/// seeded by its own previous hit, so results are identical to the serial
/// version.
#[cfg(feature = "parallel")]
pub fn cloud_to_cloud_par(compared: &[[f64; 3]], reference: &[[f64; 3]]) -> Option<Vec<f64>> {
    let tree = KdTree::new(reference)?;
    Some(distances_with_tree_par(compared, &tree))
}

#[cfg(feature = "parallel")]
fn distances_with_tree_par(compared: &[[f64; 3]], tree: &KdTree) -> Vec<f64> {
    par_in_morton_order(
        compared,
        false,
        false,
        false,
        |p, guess: &mut Option<crate::kdtree::Nearest>| {
            let hit = tree.nearest(p, *guess);
            *guess = Some(hit);
            hit.distance_sq.sqrt()
        },
    )
}

/// Multi-threaded [`cloud_to_mesh`] (feature `parallel`).
/// Each original 4096-point Morton chunk reuses its traversal stack; guessed
/// triangles and scratch reset at chunk boundaries and between calls.
#[cfg(feature = "parallel")]
pub fn cloud_to_mesh_par(
    points: &[[f64; 3]],
    mesh: &TriangleMesh,
    signed: bool,
) -> Option<Vec<f64>> {
    if USE_PAR_MESH_INDEXED_CORNERS && USE_PAR_MESH_STACK_BOUNDS {
        let bvh = crate::mesh::IndexedMeshBvh::new(mesh)?;
        return Some(par_mesh_queries(points, mesh, signed, |p, state| {
            bvh.nearest(*p, state.guess, &mut state.bounds)
        }));
    }
    let bvh = MeshBvh::new(mesh)?;
    Some(distances_with_mesh_bvh_par(points, mesh, &bvh, signed))
}

#[cfg(feature = "parallel")]
pub(crate) fn distances_with_mesh_bvh_par(
    points: &[[f64; 3]],
    mesh: &TriangleMesh,
    bvh: &MeshBvh<'_>,
    signed: bool,
) -> Vec<f64> {
    par_mesh_queries(points, mesh, signed, |p, state| {
        if USE_PAR_MESH_STACK_BOUNDS {
            if USE_PAR_MESH_TRIANGLE_SEED {
                bvh.nearest_with_triangle_seed(*p, &mut state.triangle_seed, &mut state.bounds)
            } else if USE_PAR_MESH_EAGER_PRUNE {
                bvh.nearest_with_cached_bounds_pruned(*p, state.guess, &mut state.bounds)
            } else if USE_PAR_MESH_SKIP_GUESS {
                bvh.nearest_with_cached_bounds_unique(*p, state.guess, &mut state.bounds)
            } else if USE_PAR_MESH_PAIRED_PROJECTION && USE_PAR_MESH_BOUND_PAIR_CLAMP {
                bvh.nearest_with_paired_projection(*p, state.guess, &mut state.bounds)
            } else if USE_PAR_MESH_BOUND_PAIR_CLAMP {
                bvh.nearest_with_cached_bounds_clamped(*p, state.guess, &mut state.bounds)
            } else if USE_PAR_MESH_BOUND_PAIR {
                bvh.nearest_with_cached_bounds_paired(*p, state.guess, &mut state.bounds)
            } else {
                bvh.nearest_with_cached_bounds(*p, state.guess, &mut state.bounds)
            }
        } else if USE_PAR_MESH_STACK_REUSE {
            bvh.nearest_with(*p, state.guess, &mut state.stack)
        } else {
            bvh.nearest(*p, state.guess)
        }
    })
}

#[cfg(feature = "parallel")]
fn par_mesh_queries(
    points: &[[f64; 3]],
    mesh: &TriangleMesh,
    signed: bool,
    nearest: impl Fn(&[f64; 3], &mut ParallelMeshState) -> crate::mesh::MeshHit + Sync,
) -> Vec<f64> {
    par_in_morton_order(
        points,
        USE_PAR_MESH_MORTON_PREPARATION,
        USE_PAR_MESH_COMPACT_RESULTS,
        USE_PAR_MESH_PAR_RESULTS,
        |p, state: &mut ParallelMeshState| {
            let hit = nearest(p, state);
            if hit.triangle == usize::MAX {
                state.guess = None;
                return f64::INFINITY;
            }
            state.guess = Some(hit.triangle);
            let d = hit.distance_sq.sqrt();
            let offset = [
                p[0] - hit.point[0],
                p[1] - hit.point[1],
                p[2] - hit.point[2],
            ];
            if signed && dot(offset, mesh.normal(hit.triangle)) < 0.0 {
                -d
            } else {
                d
            }
        },
    )
}

#[cfg(feature = "parallel")]
#[derive(Default)]
struct ParallelMeshState {
    guess: Option<usize>,
    stack: Vec<usize>,
    bounds: Vec<(usize, f64)>,
    triangle_seed: Option<crate::mesh::MeshTriangleSeed>,
}

// Read original immutable mesh vertices instead of copying every triangle.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_INDEXED_CORNERS: bool = false;

// Compute sibling lower bounds in two f64 SIMD lanes on native x86_64.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_BOUND_PAIR: bool = false;

// Clamp each difference before pairing to fold NaN handling into MAXPD.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_BOUND_PAIR_CLAMP: bool = true;

// Pair the three projection dot-product stages; retain adopted sibling bounds.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_PAIRED_PROJECTION: bool = false;

// Measure stack reuse separately from cached-bound traversal and scheduling.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_STACK_REUSE: bool = true;

// Reuse child bounds without changing the original parallel query chunks.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_STACK_BOUNDS: bool = true;

// Reuse one previous hit's corners/edges within each original query chunk.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_TRIANGLE_SEED: bool = false;

// Omit children whose cached lower bound already meets the best distance.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_EAGER_PRUNE: bool = false;

// Omit the seed triangle's second leaf test after its exact warm-start test.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_SKIP_GUESS: bool = false;

// Keep key-only unstable sorting and the original query chunks unchanged.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_MORTON_PREPARATION: bool = false;
#[cfg(feature = "parallel")]
const USE_PAR_MESH_COMPACT_RESULTS: bool = true;
// Gather into disjoint output slices after releasing the query-order vector.
#[cfg(feature = "parallel")]
const USE_PAR_MESH_PAR_RESULTS: bool = false;
#[cfg(feature = "parallel")]
const PAR_MORTON_MIN_POINTS: usize = 65_536;
#[cfg(feature = "parallel")]
const PAR_MORTON_CHUNK_POINTS: usize = 16_384;

/// Evaluate `query` for every point on the rayon pool, in Morton-ordered
/// chunks that each carry their own warm-start state.
#[cfg(feature = "parallel")]
fn par_in_morton_order<S: Default + Send>(
    points: &[[f64; 3]],
    parallel_order: bool,
    compact_results: bool,
    parallel_results: bool,
    query: impl Fn(&[f64; 3], &mut S) -> f64 + Sync,
) -> Vec<f64> {
    use rayon::prelude::*;
    let order = morton_order_with_parallel_preparation(points, parallel_order);
    if compact_results {
        let parts = par_morton_values::<S>(points, &order, &query);
        if parallel_results
            && points.len() >= PAR_MORTON_MIN_POINTS
            && points.len() <= u32::MAX as usize
        {
            return par_restore_morton_values(order, parts);
        }
        let mut out = vec![0.0; points.len()];
        for (chunk, values) in order.chunks(4096).zip(parts) {
            for (&i, d) in chunk.iter().zip(values) {
                out[i] = d;
            }
        }
        return out;
    }
    let parts: Vec<Vec<(usize, f64)>> = order
        .par_chunks(4096)
        .map(|chunk| {
            let mut state = S::default();
            chunk
                .iter()
                .map(|&i| (i, query(&points[i], &mut state)))
                .collect()
        })
        .collect();
    let mut out = vec![0.0; points.len()];
    for (i, d) in parts.into_iter().flatten() {
        out[i] = d;
    }
    out
}

/// Store each result once; the existing order carries its original index.
#[cfg(feature = "parallel")]
fn par_morton_values<S: Default + Send>(
    points: &[[f64; 3]],
    order: &[usize],
    query: impl Fn(&[f64; 3], &mut S) -> f64 + Sync,
) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    order
        .par_chunks(4096)
        .map(|chunk| {
            let mut state = S::default();
            chunk
                .iter()
                .map(|&i| query(&points[i], &mut state))
                .collect()
        })
        .collect()
}

/// The inverse map is temporary; queries keep their original chunks and state.
#[cfg(feature = "parallel")]
fn inverse_morton_order_u32(order: &[usize]) -> Vec<u32> {
    let mut inverse = vec![0; order.len()];
    for (position, &original) in order.iter().enumerate() {
        inverse[original] = position as u32;
    }
    inverse
}

#[cfg(feature = "parallel")]
fn par_restore_morton_values(order: Vec<usize>, parts: Vec<Vec<f64>>) -> Vec<f64> {
    use rayon::prelude::*;
    let inverse = inverse_morton_order_u32(&order);
    drop(order);
    let mut out = vec![0.0; inverse.len()];
    out.par_iter_mut()
        .zip(inverse.par_iter())
        .with_min_len(PAR_MORTON_CHUNK_POINTS)
        .for_each(|(value, &position)| {
            let position = position as usize;
            *value = parts[position / 4096][position % 4096];
        });
    out
}

/// Split a C2C job into up to `parts` spatially compact pieces for parallel
/// workers, so that each worker only builds a tree over nearby reference
/// points.
///
/// The reference subset for a group of queries is found without the full
/// tree: for a block of queries with bounding box `B` (center `c`,
/// half-diagonal `h`) and any reference point `s`, every query in `B` has its
/// nearest neighbour within `h + |c - s|`. Using the nearest point of a
/// reference subsample as `s` gives a tight, always-valid margin.
pub fn partition_c2c(
    compared: &PointCloud,
    reference: &PointCloud,
    parts: usize,
) -> Option<Vec<C2cPart>> {
    const SAMPLE: usize = 1 << 16;
    const PER_CELL: usize = 16;
    if reference.is_empty() || reference.len() > u32::MAX as usize {
        return None;
    }
    let stride = reference.len().div_ceil(SAMPLE);
    let sample: Vec<[f64; 3]> = reference
        .positions
        .iter()
        .step_by(stride)
        .copied()
        .collect();
    let sample_tree = KdTree::new(&sample)?;

    // Work on (point, index) pairs so the median splits stream through
    // contiguous memory instead of chasing indices.
    let mut items: Vec<([f64; 3], u32)> = compared
        .positions
        .iter()
        .zip(0..)
        .map(|(&p, i)| (p, i))
        .collect();
    let mut groups = Vec::new();
    split_ranges(&mut items, 0, parts.max(1), &mut groups);

    let mut result: Vec<C2cPart> = Vec::with_capacity(groups.len());
    let mut regions = Vec::with_capacity(groups.len());
    for group in groups {
        let slice = &items[group];
        let region = query_region(slice, &sample_tree, PER_CELL);
        regions.push(region);
        result.push(C2cPart {
            queries: slice.iter().map(|&(_, i)| i).collect(),
            reference: Vec::new(),
        });
    }
    for (i, p) in reference.positions.iter().enumerate() {
        for (part, (lo, hi)) in result.iter_mut().zip(&regions) {
            if (0..3).all(|a| p[a] >= lo[a] && p[a] <= hi[a]) {
                part.reference.push(i as u32);
            }
        }
    }
    Some(result)
}

/// Bounding box that contains the nearest reference point of every query in
/// `items`. Queries are binned into a uniform grid of roughly `per_cell`
/// points per cell; each occupied cell contributes its box expanded by
/// `half diagonal + distance(cell center, nearest sample point)`.
fn query_region(
    items: &[([f64; 3], u32)],
    sample_tree: &KdTree,
    per_cell: usize,
) -> ([f64; 3], [f64; 3]) {
    let (lo, hi) = bounds(items.iter().map(|(p, _)| p));
    let extent: [f64; 3] = std::array::from_fn(|a| hi[a] - lo[a]);
    // Cell edge: whichever of the 1D/2D/3D density estimates is coarsest, so
    // lines, surfaces and volumes all get about `per_cell` points per cell.
    let cells = (items.len() / per_cell).max(1) as f64;
    let mut sorted = extent;
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let edge = (sorted[0] / cells)
        .max((sorted[0] * sorted[1] / cells).sqrt())
        .max((sorted[0] * sorted[1] * sorted[2] / cells).cbrt());
    let edge = if edge > 0.0 { edge } else { 1.0 };
    let dims: [usize; 3] = std::array::from_fn(|a| (extent[a] / edge) as usize + 1);
    let cell_of = |p: &[f64; 3]| -> usize {
        let c: [usize; 3] =
            std::array::from_fn(|a| (((p[a] - lo[a]) / edge) as usize).min(dims[a] - 1));
        (c[2] * dims[1] + c[1]) * dims[0] + c[0]
    };
    let mut occupied = vec![false; dims[0] * dims[1] * dims[2]];
    for (p, _) in items {
        occupied[cell_of(p)] = true;
    }

    // Absorb rounding in the cell arithmetic, even for ECEF-sized coordinates.
    let magnitude = lo.iter().chain(&hi).fold(0.0f64, |m, v| m.max(v.abs()));
    let slack = magnitude * 1e-12 + 1e-9;
    let half_diagonal = 0.5 * edge * 3f64.sqrt();
    let mut region = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    let mut guess = None;
    for index in (0..occupied.len()).filter(|&i| occupied[i]) {
        let cell = [
            index % dims[0],
            index / dims[0] % dims[1],
            index / (dims[0] * dims[1]),
        ];
        let cell_lo: [f64; 3] = std::array::from_fn(|a| lo[a] + cell[a] as f64 * edge);
        let center: [f64; 3] = std::array::from_fn(|a| cell_lo[a] + 0.5 * edge);
        let hit = sample_tree.nearest(&center, guess);
        guess = Some(hit);
        let margin = half_diagonal + hit.distance_sq.sqrt() + slack;
        for (a, &l) in cell_lo.iter().enumerate() {
            region.0[a] = region.0[a].min(l - margin);
            region.1[a] = region.1[a].max(l + edge + margin);
        }
    }
    region
}

/// Recursively reorder `items` by median splits along the widest axis and
/// record `parts` near-equal, spatially compact ranges (offset by `offset`).
pub(crate) fn split_ranges(
    items: &mut [([f64; 3], u32)],
    offset: usize,
    parts: usize,
    out: &mut Vec<std::ops::Range<usize>>,
) {
    if parts <= 1 || items.len() < 2 {
        if !items.is_empty() {
            out.push(offset..offset + items.len());
        }
        return;
    }
    let (lo, hi) = bounds(items.iter().map(|(p, _)| p));
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap();
    let left_parts = parts / 2;
    let mid = items.len() * left_parts / parts;
    items.select_nth_unstable_by(mid, |a, b| a.0[axis].total_cmp(&b.0[axis]));
    let (left, right) = items.split_at_mut(mid);
    split_ranges(left, offset, left_parts, out);
    split_ranges(right, offset + mid, parts - left_parts, out);
}

/// Bounding box of `points`, ignoring NaN coordinates (infinite when empty).
///
/// The k-d tree build runs this once per level over all points. It compares
/// instead of calling `f64::min`/`max`, whose NaN rule costs extra
/// instructions around WASM's `f64.min`; with SIMD, x and y share one `f64x2`
/// register (`pmin(a, b)` is `b < a ? b : a`, so NaN is skipped the same way).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) fn bounds<'a>(points: impl Iterator<Item = &'a [f64; 3]>) -> ([f64; 3], [f64; 3]) {
    use core::arch::wasm32::*;
    let mut lo_xy = f64x2_splat(f64::INFINITY);
    let mut hi_xy = f64x2_splat(f64::NEG_INFINITY);
    let (mut lo_z, mut hi_z) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in points {
        // SAFETY: `p` holds three f64s; v128_load allows unaligned addresses.
        let xy = unsafe { v128_load(p.as_ptr().cast()) };
        lo_xy = f64x2_pmin(lo_xy, xy);
        hi_xy = f64x2_pmax(hi_xy, xy);
        if p[2] < lo_z {
            lo_z = p[2];
        }
        if p[2] > hi_z {
            hi_z = p[2];
        }
    }
    let xyz = |xy, z| [f64x2_extract_lane::<0>(xy), f64x2_extract_lane::<1>(xy), z];
    (xyz(lo_xy, lo_z), xyz(hi_xy, hi_z))
}

/// Bounding box of `points`, ignoring NaN coordinates (infinite when empty).
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub(crate) fn bounds<'a>(points: impl Iterator<Item = &'a [f64; 3]>) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for a in 0..3 {
            if p[a] < lo[a] {
                lo[a] = p[a];
            }
            if p[a] > hi[a] {
                hi[a] = p[a];
            }
        }
    }
    (lo, hi)
}

/// Indices of `points` sorted along a Z-order (Morton) curve.
pub(crate) fn morton_order(points: &[[f64; 3]]) -> Vec<usize> {
    morton_order_with_parallel_preparation(points, false)
}

fn morton_bounds_serial(points: &[[f64; 3]], first: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let (mut lo, mut hi) = (first, first);
    for p in points {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    (lo, hi)
}

#[cfg(feature = "parallel")]
fn morton_bounds_parallel(points: &[[f64; 3]], first: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    use rayon::prelude::*;
    let chunks: Vec<_> = points
        .par_chunks(PAR_MORTON_CHUNK_POINTS)
        .map(|chunk| {
            let (mut lo, mut hi) = (first, first);
            let mut merge = true;
            for p in chunk {
                for a in 0..3 {
                    merge &= p[a].is_finite() && p[a] != 0.0;
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
            (lo, hi, merge)
        })
        .collect();
    if chunks.iter().any(|c| !c.2) {
        // Preserve target-dependent signed-zero/NaN tie bits in source order.
        return morton_bounds_serial(points, first);
    }
    let (mut lo, mut hi) = (first, first);
    for (a, b, _) in chunks {
        for i in 0..3 {
            lo[i] = lo[i].min(a[i]);
            hi[i] = hi[i].max(b[i]);
        }
    }
    (lo, hi)
}

fn morton_order_with_parallel_preparation(points: &[[f64; 3]], _parallel: bool) -> Vec<usize> {
    let Some(first) = points.first() else {
        return Vec::new();
    };
    #[cfg(feature = "parallel")]
    let parallel = _parallel && points.len() >= PAR_MORTON_MIN_POINTS;
    #[cfg(feature = "parallel")]
    let (lo, hi) = if parallel {
        morton_bounds_parallel(points, *first)
    } else {
        morton_bounds_serial(points, *first)
    };
    #[cfg(not(feature = "parallel"))]
    let (lo, hi) = morton_bounds_serial(points, *first);
    const CELLS: f64 = ((1u32 << 21) - 1) as f64;
    let scale: [f64; 3] = std::array::from_fn(|i| {
        let extent = hi[i] - lo[i];
        if extent > 0.0 { CELLS / extent } else { 0.0 }
    });
    let key = |(idx, p): (usize, &[f64; 3])| {
        let cell = |i: usize| ((p[i] - lo[i]) * scale[i]) as u64;
        (
            spread_bits(cell(0)) | spread_bits(cell(1)) << 1 | spread_bits(cell(2)) << 2,
            idx,
        )
    };
    let serial_keys = || points.iter().enumerate().map(key).collect();
    #[cfg(feature = "parallel")]
    let mut keyed: Vec<(u64, usize)> = if parallel {
        use rayon::prelude::*;
        points
            .par_iter()
            .enumerate()
            .with_min_len(PAR_MORTON_CHUNK_POINTS)
            .map(key)
            .collect()
    } else {
        serial_keys()
    };
    #[cfg(not(feature = "parallel"))]
    let mut keyed: Vec<(u64, usize)> = serial_keys();
    keyed.sort_unstable_by_key(|&(key, _)| key);
    keyed.into_iter().map(|(_, idx)| idx).collect()
}

/// Insert two zero bits between each of the low 21 bits of `v`.
fn spread_bits(v: u64) -> u64 {
    let mut x = v & 0x1f_ffff;
    x = (x | x << 32) & 0x1f_0000_0000_ffff;
    x = (x | x << 16) & 0x1f_0000_ff00_00ff;
    x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
    x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
    x = (x | x << 2) & 0x1249_2492_4924_9249;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(points: &[[f64; 3]]) -> PointCloud {
        PointCloud {
            positions: points.to_vec(),
            colors: None,
            attributes: Vec::new(),
        }
    }

    #[test]
    fn distances_match_brute_force() {
        let reference = cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        let compared = cloud(&[[0.0, 0.0, 0.5], [1.0, 1.0, 0.0], [5.0, 0.0, 0.0]]);
        let d = cloud_to_cloud(&compared, &reference).unwrap();
        let expected = [0.5, 1.0, 4.0];
        for (got, want) in d.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "{got} != {want}");
        }
    }

    #[test]
    fn spread_bits_interleaves() {
        assert_eq!(spread_bits(0b1), 0b1);
        assert_eq!(spread_bits(0b11), 0b1001);
        assert_eq!(spread_bits(0x1f_ffff), 0x1249_2492_4924_9249);
    }

    #[test]
    fn bounds_skip_nan_like_f64_min() {
        let points = [
            [f64::NAN, 2.0, -1.0],
            [3.0, f64::NAN, 5.0],
            [-4.0, 7.5, f64::NAN],
            [0.5, -6.0, 2.0],
        ];
        let (lo, hi) = bounds(points.iter());
        for a in 0..3 {
            let v = points.iter().map(|p| p[a]);
            assert_eq!(lo[a], v.clone().fold(f64::INFINITY, f64::min));
            assert_eq!(hi[a], v.fold(f64::NEG_INFINITY, f64::max));
        }
        assert_eq!(bounds([].iter()).0, [f64::INFINITY; 3]);
    }

    #[test]
    fn morton_order_is_a_permutation() {
        let points: Vec<[f64; 3]> = (0..100)
            .map(|i| {
                let f = i as f64;
                [(f * 7.3) % 10.0, (f * 3.1) % 5.0, 1.0]
            })
            .collect();
        let mut order = morton_order(&points);
        order.sort_unstable();
        assert_eq!(order, (0..100).collect::<Vec<_>>());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn compact_and_parallel_results_preserve_stateful_bits_and_exact_payload_capacity() {
        let pools: Vec<_> = [1, 2, 4]
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
            })
            .into_iter()
            .collect();
        let query = |p: &[f64; 3], state: &mut (usize, u64)| {
            state.0 += 1;
            state.1 = state.1.rotate_left(7) ^ p[0].to_bits() ^ p[1].to_bits() ^ p[2].to_bits();
            match state.0 % 6 {
                0 => -0.0,
                1 => f64::from_bits(0x7ff8_0000_0000_0001 | (state.1 & 0x0000_0000_000f_ffff)),
                2 => f64::INFINITY,
                3 => f64::NEG_INFINITY,
                4 => f64::from_bits(0x3fe0_0000_0000_0000 | (state.1 & 0x000f_ffff_ffff_ffff)),
                _ => -(state.0 as f64),
            }
        };
        for n in [
            0, 1, 4095, 4096, 4097, 8193, 16_383, 16_384, 16_385, 65_535, 65_536, 65_537,
        ] {
            for mode in 0..4 {
                let mut points = pseudo_random(n, 89, [368000.0, 3955000.0, 40.0]).positions;
                match mode {
                    1 => points.fill([1.0, 2.0, 3.0]),
                    2 => {
                        if let Some(p) = points.first_mut() {
                            *p = [f64::NAN, -0.0, 0.0];
                        }
                        for i in (4095..n).step_by(4096) {
                            points[i] = [f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
                        }
                    }
                    3 => {
                        for (i, p) in points.iter_mut().enumerate() {
                            let x = if i % 2 == 0 { 1e308 } else { -1e308 };
                            *p = [x, -x, 1e300];
                        }
                    }
                    _ => {}
                }
                let order = morton_order(&points);
                for pool in &pools {
                    let old =
                        pool.install(|| par_in_morton_order(&points, false, false, false, query));
                    let new =
                        pool.install(|| par_in_morton_order(&points, false, true, false, query));
                    assert_eq!(
                        old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        new.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(old.capacity(), new.capacity());
                    let parts = pool.install(|| par_morton_values(&points, &order, query));
                    assert_eq!(parts.len(), n.div_ceil(4096));
                    assert_eq!(parts.iter().map(Vec::capacity).sum::<usize>(), n);
                    for (chunk, part) in order.chunks(4096).zip(&parts) {
                        assert_eq!(part.len(), chunk.len());
                        assert_eq!(part.capacity(), chunk.len());
                    }
                    let inverse = inverse_morton_order_u32(&order);
                    assert_eq!(inverse.capacity(), n);
                    for (position, &original) in order.iter().enumerate() {
                        assert_eq!(inverse[original] as usize, position);
                    }
                    let gathered = pool.install(|| par_restore_morton_values(order.clone(), parts));
                    let selected =
                        pool.install(|| par_in_morton_order(&points, false, true, true, query));
                    for actual in [gathered, selected] {
                        assert_eq!(actual.capacity(), old.capacity());
                        assert_eq!(
                            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            old.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_morton_preparation_preserves_frozen_order_bounds_and_capacity() {
        fn preceding(points: &[[f64; 3]]) -> (Vec<usize>, ([f64; 3], [f64; 3])) {
            let Some(first) = points.first() else {
                return (Vec::new(), ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]));
            };
            let (mut lo, mut hi) = (*first, *first);
            for p in points {
                for a in 0..3 {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
            let scale: [f64; 3] = std::array::from_fn(|a| {
                let extent = hi[a] - lo[a];
                if extent > 0.0 {
                    ((1u32 << 21) - 1) as f64 / extent
                } else {
                    0.0
                }
            });
            // Independent bit-by-bit encoding freezes the original policy.
            let spread = |v: u64| (0..21).fold(0, |key, b| key | ((v >> b) & 1) << (3 * b));
            let mut keyed: Vec<_> = points
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let cell = |a: usize| ((p[a] - lo[a]) * scale[a]) as u64;
                    (
                        spread(cell(0)) | spread(cell(1)) << 1 | spread(cell(2)) << 2,
                        i,
                    )
                })
                .collect();
            keyed.sort_unstable_by_key(|&(key, _)| key);
            (keyed.into_iter().map(|(_, i)| i).collect(), (lo, hi))
        }
        let pools: Vec<_> = [1, 2, 4]
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
            })
            .into_iter()
            .collect();
        for n in [
            0, 1, 4095, 4096, 4097, 16_383, 16_384, 16_385, 65_535, 65_536, 65_537,
        ] {
            for mode in 0..6 {
                let mut points = pseudo_random(n, 73, [368000.0, 3955000.0, 40.0]).positions;
                match mode {
                    1 => {
                        if let Some(p) = points.first_mut() {
                            *p = [-0.0, 0.0, -0.0];
                        }
                    }
                    2 => {
                        if let Some(p) = points.first_mut() {
                            *p = [f64::from_bits(0x7ff8_0000_0000_1234); 3];
                        }
                        for i in (4095..n).step_by(4096) {
                            points[i] = [f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
                        }
                    }
                    3 => points.fill([5.0, 6.0, 7.0]),
                    4 => {
                        for (i, p) in points.iter_mut().enumerate() {
                            let x = if i % 2 == 0 { 1e308 } else { -1e308 };
                            *p = [x, -x, 1e300];
                        }
                    }
                    5 => {
                        for (i, p) in points.iter_mut().enumerate() {
                            *p = [
                                (i % 17 + 1) as f64 * 1e-320,
                                (i % 13 + 1) as f64 * 1e-320,
                                (i % 11 + 1) as f64 * 1e-320,
                            ];
                        }
                    }
                    _ => {}
                }
                let (expected, (lo, hi)) = preceding(&points);
                for pool in &pools {
                    let actual =
                        pool.install(|| morton_order_with_parallel_preparation(&points, true));
                    assert_eq!(actual, expected, "n={n}, mode={mode}");
                    assert_eq!(actual.capacity(), expected.capacity());
                    if let Some(first) = points.first() {
                        let (a, b) = pool.install(|| morton_bounds_parallel(&points, *first));
                        assert_eq!(a.map(f64::to_bits), lo.map(f64::to_bits));
                        assert_eq!(b.map(f64::to_bits), hi.map(f64::to_bits));
                    }
                }
            }
        }
    }

    fn pseudo_random(n: usize, seed: u64, offset: [f64; 3]) -> PointCloud {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        PointCloud {
            positions: (0..n)
                .map(|_| {
                    let (x, y) = (next() * 50.0, next() * 20.0);
                    [x + offset[0], y + offset[1], (x / 5.0).sin() + offset[2]]
                })
                .collect(),
            colors: None,
            attributes: Vec::new(),
        }
    }

    #[test]
    fn prepared_c2c_preserves_queries_order_bits_and_repeatability() {
        for count in [0, 1, 2, 3, 15, 16, 17, 31, 33, 257] {
            for scale in [1e-160, 1.0, 1e150, 1e200] {
                let reference = cloud(
                    &(0..count + 3)
                        .map(|i| [(i % 7) as f64 * scale, (i % 3) as f64 * scale, -0.0])
                        .collect::<Vec<_>>(),
                );
                let compared = cloud(
                    &(0..count)
                        .map(|i| {
                            [
                                (i % 7) as f64 * scale + 0.25 * scale,
                                (i % 3) as f64 * scale,
                                scale,
                            ]
                        })
                        .collect::<Vec<_>>(),
                );
                let prepared =
                    PreparedC2c::new(compared.positions.clone(), &reference.positions).unwrap();
                assert!(prepared.retained_bytes() >= count * (24 + std::mem::size_of::<usize>()));
                for simd in [false, true, false] {
                    let actual = prepared.distances(simd);
                    let expected = cloud_to_cloud_with_simd(&compared, &reference, simd).unwrap();
                    assert_eq!(
                        actual.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|d| d.to_bits()).collect::<Vec<_>>()
                    );
                }
                #[cfg(feature = "parallel")]
                assert_eq!(
                    prepared
                        .distances_par()
                        .iter()
                        .map(|d| d.to_bits())
                        .collect::<Vec<_>>(),
                    prepared
                        .distances(false)
                        .iter()
                        .map(|d| d.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        assert!(PreparedC2c::new(vec![[0.0; 3]], &[]).is_none());
    }

    #[test]
    fn prepared_c2c_owns_sources_and_nonfinite_bits() {
        let mut reference = cloud(&[
            [0.0; 3],
            [1.0, 1.0, 1.0],
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 1.0, 1.0],
            [-0.0, 0.0, 0.0],
        ]);
        let mut compared = cloud(&[
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 1.0, 0.0],
            [0.0, -0.0, 0.0],
            [1e300; 3],
            [3.0, 4.0, 5.0],
        ]);
        let expected = cloud_to_cloud(&compared, &reference).unwrap();
        let prepared = PreparedC2c::new(compared.positions.clone(), &reference.positions).unwrap();
        reference.positions.fill([42.0; 3]);
        compared.positions.fill([99.0; 3]);
        drop(reference);
        drop(compared);
        for simd in [false, true, false] {
            assert_eq!(
                prepared
                    .distances(simd)
                    .iter()
                    .map(|d| d.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|d| d.to_bits()).collect::<Vec<_>>()
            );
        }
        #[cfg(feature = "parallel")]
        assert_eq!(
            prepared
                .distances_par()
                .iter()
                .map(|d| d.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|d| d.to_bits()).collect::<Vec<_>>()
        );
    }

    /// Run every part independently and scatter the results back.
    fn partitioned(compared: &PointCloud, reference: &PointCloud, parts: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; compared.len()];
        for part in partition_c2c(compared, reference, parts).unwrap() {
            let pick = |cloud: &PointCloud, idx: &[u32]| PointCloud {
                positions: idx.iter().map(|&i| cloud.positions[i as usize]).collect(),
                colors: None,
                attributes: Vec::new(),
            };
            let d = cloud_to_cloud(
                &pick(compared, &part.queries),
                &pick(reference, &part.reference),
            )
            .unwrap();
            for (&i, d) in part.queries.iter().zip(d) {
                assert!(out[i as usize].is_nan(), "query {i} in two parts");
                out[i as usize] = d;
            }
        }
        out
    }

    #[test]
    fn partitioned_c2c_is_exact() {
        let reference = pseudo_random(20_000, 1, [0.0; 3]);
        let compared = pseudo_random(15_000, 2, [0.0, 0.0, 0.05]);
        let full = cloud_to_cloud(&compared, &reference).unwrap();
        for parts in [1, 3, 8] {
            assert_eq!(
                partitioned(&compared, &reference, parts),
                full,
                "parts={parts}"
            );
        }
    }

    #[test]
    fn partitioned_c2c_is_exact_for_distant_clouds() {
        let reference = pseudo_random(5_000, 3, [0.0; 3]);
        let compared = pseudo_random(4_000, 4, [300.0, -40.0, 10.0]);
        let full = cloud_to_cloud(&compared, &reference).unwrap();
        assert_eq!(partitioned(&compared, &reference, 6), full);
    }

    #[test]
    fn partitions_keep_reference_subsets_small() {
        let reference = pseudo_random(40_000, 5, [0.0; 3]);
        let compared = pseudo_random(40_000, 6, [0.0; 3]);
        let parts = partition_c2c(&compared, &reference, 8).unwrap();
        let total: usize = parts.iter().map(|p| p.reference.len()).sum();
        assert!(total * 2 < 3 * reference.len(), "reference copies: {total}");
    }

    #[test]
    fn cloud_to_mesh_on_a_square() {
        // Two triangles covering [0, 10]^2 at z = 0, normal +z.
        let mesh = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let points = [
            [3.0, 4.0, 2.0],
            [7.0, 1.0, -0.5],
            [13.0, 14.0, 0.0],
            [5.0, 5.0, 0.0],
        ];
        let signed = cloud_to_mesh(&points, &mesh, true).unwrap();
        let unsigned = cloud_to_mesh(&points, &mesh, false).unwrap();
        let expected = [2.0, -0.5, 5.0, 0.0];
        for i in 0..points.len() {
            assert!(
                (signed[i] - expected[i]).abs() < 1e-12,
                "{i}: {}",
                signed[i]
            );
            assert!((unsigned[i] - expected[i].abs()).abs() < 1e-12);
        }
        assert!(cloud_to_mesh(&points, &TriangleMesh::default(), false).is_none());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_versions_match_serial() {
        let reference = pseudo_random(20_000, 1, [0.0; 3]);
        let compared = pseudo_random(15_000, 2, [0.0, 0.0, 0.05]);
        assert_eq!(
            cloud_to_cloud_par(&compared.positions, &reference.positions).unwrap(),
            cloud_to_cloud(&compared, &reference).unwrap()
        );
        let mesh = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
                [60.0, 30.0, 0.0],
                [0.0, 30.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        assert_eq!(
            cloud_to_mesh_par(&compared.positions, &mesh, true).unwrap(),
            cloud_to_mesh(&compared.positions, &mesh, true).unwrap()
        );
    }

    #[test]
    fn empty_reference_yields_none() {
        assert!(cloud_to_cloud(&cloud(&[[0.0; 3]]), &PointCloud::default()).is_none());
    }

    #[test]
    fn stats_are_correct() {
        let s = DistanceStats::from_distances(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(s.count, 4);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 4.0);
        assert_eq!(s.mean, 2.5);
        assert_eq!(s.median, 2.5);
        assert!((s.rms - 7.5f64.sqrt()).abs() < 1e-12);
        assert!((s.std_dev - 1.25f64.sqrt()).abs() < 1e-12);
    }

    fn assert_summary_matches_sort(distances: &[f64]) {
        let selected = DistanceStats::from_distances(distances);
        let sorted = DistanceStats::from_distances_with_median_sort(distances, true);
        let (Some(a), Some(b)) = (selected, sorted) else {
            assert!(selected.is_none() && sorted.is_none());
            return;
        };
        assert_eq!(a.count, b.count);
        for (x, y) in [
            (a.min, b.min),
            (a.max, b.max),
            (a.mean, b.mean),
            (a.rms, b.rms),
            (a.std_dev, b.std_dev),
            (a.median, b.median),
        ] {
            if x.is_nan() {
                assert!(y.is_nan()); // Arithmetic NaN payloads are not specified.
            } else {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
        if !distances.len().is_multiple_of(2) {
            let mut oracle = distances.to_vec();
            oracle.sort_unstable_by(f64::total_cmp);
            assert_eq!(a.median.to_bits(), oracle[oracle.len() / 2].to_bits());
        }
    }

    #[test]
    fn median_selection_matches_sort_for_ordered_duplicate_and_raw_bit_inputs() {
        assert_summary_matches_sort(&[]);
        for trap in [
            vec![1.0, 4.0, 2.0, 3.0, 5.0],
            vec![5.0, 3.0, 2.0, 4.0, 1.0],
            vec![1.0, 4.0, 2.0, 5.0],
            vec![5.0, 2.0, 4.0, 1.0],
        ] {
            assert_summary_matches_sort(&trap);
        }
        let mut state = 0x5eed_u64;
        for count in [1, 2, 3, 7, 16, 17, 31, 257, 1024, 4095, 16_384] {
            let raw: Vec<_> = (0..count)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    f64::from_bits(state)
                })
                .collect();
            assert_summary_matches_sort(&raw);
            let mut ordered = raw;
            ordered.sort_unstable_by(f64::total_cmp);
            assert_summary_matches_sort(&ordered);
            ordered.reverse();
            assert_summary_matches_sort(&ordered);
            assert_summary_matches_sort(&vec![9e-6; count]);
            assert_summary_matches_sort(
                &(0..count)
                    .map(|i| i.min(count - 1 - i) as f64)
                    .collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn median_selection_retains_total_order_and_original_midpoint_arithmetic() {
        let values = [
            f64::from_bits(0xfff8_0000_0000_0003),
            f64::NEG_INFINITY,
            -0.0,
            0.0,
            f64::MAX,
            f64::INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0002),
        ];
        for count in 1..=6 {
            for mut code in 0..values.len().pow(count) {
                let distances: Vec<_> = (0..count)
                    .map(|_| {
                        let value = values[code % values.len()];
                        code /= values.len();
                        value
                    })
                    .collect();
                assert_summary_matches_sort(&distances);
            }
        }
        assert_eq!(
            DistanceStats::from_distances(&[f64::MAX, f64::MAX])
                .unwrap()
                .median,
            f64::INFINITY,
        );
    }
}
