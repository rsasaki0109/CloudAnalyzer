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
        let mut sorted = distances.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        let median = if sorted.len().is_multiple_of(2) {
            0.5 * (sorted[mid - 1] + sorted[mid])
        } else {
            sorted[mid]
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

/// For every point of `points`, the distance to the closest point of `mesh`
/// (cloud-to-mesh, C2M). With `signed`, points on the back side of the
/// closest triangle (against its right-hand normal) get negative distances.
/// Returns `None` when the mesh has no triangles.
pub fn cloud_to_mesh(points: &[[f64; 3]], mesh: &TriangleMesh, signed: bool) -> Option<Vec<f64>> {
    let bvh = MeshBvh::new(mesh)?;
    let mut distances = vec![0.0; points.len()];
    let mut guess = None;
    let mut stack = Vec::new();
    for i in morton_order(points) {
        let p = points[i];
        let hit = bvh.nearest_with(p, guess, &mut stack);
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
    Some(distances)
}

/// For every point of `compared`, the Euclidean distance to its nearest
/// neighbour in `reference`. Returns `None` when `reference` is empty.
pub fn cloud_to_cloud(compared: &PointCloud, reference: &PointCloud) -> Option<Vec<f64>> {
    if reference.is_empty() {
        return None;
    }
    let tree = KdTree::new(&reference.positions)?;
    let mut distances = vec![0.0; compared.len()];
    // Querying in spatial order keeps the tree's hot path in cache, and the
    // previous hit is an excellent starting guess for the next query.
    let mut guess = None;
    for i in morton_order(&compared.positions) {
        let hit = tree.nearest(&compared.positions[i], guess);
        distances[i] = hit.distance_sq.sqrt();
        guess = Some(hit);
    }
    Some(distances)
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
    Some(par_in_morton_order(
        compared,
        |p, guess: &mut Option<crate::kdtree::Nearest>| {
            let hit = tree.nearest(p, *guess);
            *guess = Some(hit);
            hit.distance_sq.sqrt()
        },
    ))
}

/// Multi-threaded [`cloud_to_mesh`] (feature `parallel`).
#[cfg(feature = "parallel")]
pub fn cloud_to_mesh_par(
    points: &[[f64; 3]],
    mesh: &TriangleMesh,
    signed: bool,
) -> Option<Vec<f64>> {
    let bvh = MeshBvh::new(mesh)?;
    Some(par_in_morton_order(
        points,
        |p, guess: &mut Option<usize>| {
            let hit = bvh.nearest(*p, *guess);
            *guess = Some(hit.triangle);
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
    ))
}

/// Evaluate `query` for every point on the rayon pool, in Morton-ordered
/// chunks that each carry their own warm-start state.
#[cfg(feature = "parallel")]
fn par_in_morton_order<S: Default + Send>(
    points: &[[f64; 3]],
    query: impl Fn(&[f64; 3], &mut S) -> f64 + Sync,
) -> Vec<f64> {
    use rayon::prelude::*;
    let order = morton_order(points);
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

pub(crate) fn bounds<'a>(points: impl Iterator<Item = &'a [f64; 3]>) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    (lo, hi)
}

/// Indices of `points` sorted along a Z-order (Morton) curve.
pub(crate) fn morton_order(points: &[[f64; 3]]) -> Vec<usize> {
    let Some(first) = points.first() else {
        return Vec::new();
    };
    let (mut lo, mut hi) = (*first, *first);
    for p in points {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    const CELLS: f64 = ((1u32 << 21) - 1) as f64;
    let scale: [f64; 3] = std::array::from_fn(|i| {
        let extent = hi[i] - lo[i];
        if extent > 0.0 { CELLS / extent } else { 0.0 }
    });
    let mut keyed: Vec<(u64, usize)> = points
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let cell = |i: usize| ((p[i] - lo[i]) * scale[i]) as u64;
            (
                spread_bits(cell(0)) | spread_bits(cell(1)) << 1 | spread_bits(cell(2)) << 2,
                idx,
            )
        })
        .collect();
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
}
