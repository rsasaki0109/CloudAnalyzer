//! Point cloud filters: spatial and random subsampling, statistical outlier
//! removal, Gaussian-splat cleanup. Each returns the indices of the points to keep (in their original
//! order), so colors and attributes follow via [`PointCloud::select`].

use crate::kdtree::KdTree;
use std::collections::HashMap;

use crate::{AttributeValues, OPACITY, PointCloud, SPLAT_SIZE};

/// Keep one point per cubic voxel of edge `voxel` (the first point, in cloud
/// order, that falls into each occupied voxel). Original points are kept, not
/// averaged, so their coordinates, colors and attributes stay exact.
pub fn voxel_subsample(cloud: &PointCloud, voxel: f64) -> Vec<usize> {
    // Also rejects NaN.
    if cloud.is_empty() || voxel.is_nan() || voxel <= 0.0 {
        return (0..cloud.len()).collect();
    }
    let lo = cloud.bounds().map(|b| b.min).unwrap_or([0.0; 3]);
    let mut keyed: Vec<(u64, u32)> = cloud
        .positions
        .iter()
        .zip(0u32..)
        .map(|(p, i)| {
            let cell = |a: usize| (((p[a] - lo[a]) / voxel) as u64).min((1 << 21) - 1);
            (cell(0) | cell(1) << 21 | cell(2) << 42, i)
        })
        .collect();
    // Stable sort keeps the first point of each voxel first.
    keyed.sort_by_key(|&(key, _)| key);
    let mut keep: Vec<usize> = Vec::new();
    let mut previous = None;
    for &(key, i) in &keyed {
        if previous != Some(key) {
            keep.push(i as usize);
            previous = Some(key);
        }
    }
    keep.sort_unstable();
    keep
}

/// Keep points so that no two kept points are closer than `distance`
/// (CloudCompare's "space" subsampling). Points are visited in cloud order
/// and kept when no kept point lies within `distance`; kept points are found
/// through a hash grid with cells of that size, so only the 27 cells around
/// a point are searched.
pub fn spatial_subsample(cloud: &PointCloud, distance: f64) -> Vec<usize> {
    if cloud.is_empty() || distance.is_nan() || distance <= 0.0 {
        return (0..cloud.len()).collect();
    }
    let lo = cloud.bounds().map(|b| b.min).unwrap_or([0.0; 3]);
    let cell = |p: &[f64; 3]| -> [i64; 3] {
        std::array::from_fn(|a| ((p[a] - lo[a]) / distance).floor() as i64)
    };
    let d2 = distance * distance;
    let mut grid: HashMap<[i64; 3], Vec<u32>, BuildCellHasher> = HashMap::default();
    let mut keep = Vec::new();
    'points: for (i, p) in cloud.positions.iter().enumerate() {
        let c = cell(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(kept) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                        continue;
                    };
                    for &k in kept {
                        let q = &cloud.positions[k as usize];
                        if (0..3).map(|a| (p[a] - q[a]).powi(2)).sum::<f64>() < d2 {
                            continue 'points;
                        }
                    }
                }
            }
        }
        grid.entry(c).or_default().push(i as u32);
        keep.push(i);
    }
    keep
}

/// Keep the point nearest the centre of each occupied cell of octree
/// `level` (CloudCompare's "octree" subsampling): cells are the cloud's
/// bounding cube split `2^level` times along each axis.
pub fn octree_subsample(cloud: &PointCloud, level: u32) -> Vec<usize> {
    let Some(b) = cloud.bounds() else {
        return Vec::new();
    };
    let level = level.min(21);
    let size = (0..3).map(|a| b.max[a] - b.min[a]).fold(0.0, f64::max);
    if size <= 0.0 || level == 0 {
        return (0..cloud.len()).take(1).collect();
    }
    let cells = (1u64 << level) as f64;
    let edge = size / cells;
    let last = (1u64 << level) - 1;
    let mut keyed: Vec<(u64, f64, u32)> = cloud
        .positions
        .iter()
        .zip(0u32..)
        .map(|(p, i)| {
            let mut key = 0;
            let mut d2 = 0.0;
            for (a, (&v, &min)) in p.iter().zip(&b.min).enumerate() {
                let k = (((v - min) / edge) as u64).min(last);
                let center = min + (k as f64 + 0.5) * edge;
                d2 += (v - center).powi(2);
                key |= k << (21 * a);
            }
            (key, d2, i)
        })
        .collect();
    // Nearest to the centre first within each cell; ties keep cloud order.
    keyed.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)).then(x.2.cmp(&y.2)));
    let mut keep: Vec<usize> = Vec::new();
    let mut previous = None;
    for &(key, _, i) in &keyed {
        if previous != Some(key) {
            keep.push(i as usize);
            previous = Some(key);
        }
    }
    keep.sort_unstable();
    keep
}

/// A fast hasher for integer cell coordinates (SplitMix64 finaliser).
#[derive(Default, Clone, Copy)]
struct BuildCellHasher;

impl std::hash::BuildHasher for BuildCellHasher {
    type Hasher = CellHasher;
    fn build_hasher(&self) -> CellHasher {
        CellHasher(0)
    }
}

struct CellHasher(u64);

impl std::hash::Hasher for CellHasher {
    fn finish(&self) -> u64 {
        let mut z = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }
    fn write_i64(&mut self, v: i64) {
        self.0 = self.0.rotate_left(21) ^ (v as u64);
    }
}

/// Keep `count` points chosen uniformly at random (deterministic for a seed).
pub fn random_subsample(cloud: &PointCloud, count: usize, seed: u64) -> Vec<usize> {
    let n = cloud.len();
    if count >= n {
        return (0..n).collect();
    }
    // Partial Fisher-Yates over an index array with a xorshift generator.
    let mut indices: Vec<usize> = (0..n).collect();
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in 0..count {
        let j = i + (next() % (n - i) as u64) as usize;
        indices.swap(i, j);
    }
    indices.truncate(count);
    indices.sort_unstable();
    indices
}

/// Statistical outlier removal: drop points whose mean distance to their
/// `k` nearest neighbours exceeds the global mean by more than `ratio`
/// standard deviations (PCL's `StatisticalOutlierRemoval`).
pub fn statistical_outliers(cloud: &PointCloud, k: usize, ratio: f64) -> Vec<usize> {
    if cloud.len() <= k || k == 0 {
        return (0..cloud.len()).collect();
    }
    sor_keep(&knn_mean_distances(&cloud.positions, k), ratio)
}

/// Multi-threaded [`statistical_outliers`] (feature `parallel`); identical
/// results.
#[cfg(feature = "parallel")]
pub fn statistical_outliers_par(cloud: &PointCloud, k: usize, ratio: f64) -> Vec<usize> {
    use rayon::prelude::*;
    let n = cloud.len();
    if n <= k || k == 0 {
        return (0..n).collect();
    }
    let Some(tree) = KdTree::new(&cloud.positions) else {
        return Vec::new();
    };
    let order = crate::distance::morton_order(&cloud.positions);
    let parts: Vec<Vec<(usize, f64)>> = order
        .par_chunks(4096)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&i| (i, mean_of(&tree.nearest_k(&cloud.positions[i], k + 1))))
                .collect()
        })
        .collect();
    let mut means = vec![0.0; n];
    for (i, m) in parts.into_iter().flatten() {
        means[i] = m;
    }
    sor_keep(&means, ratio)
}

/// Gaussian-splat cleanup: keep the splats with opacity at least
/// `min_opacity` and size at most `max_size`. Raw 3DGS output is full of
/// nearly transparent and huge "floater" Gaussians that are not surfaces.
/// `None` when the cloud has no opacity / size attributes.
pub fn splat_cleanup(cloud: &PointCloud, min_opacity: f64, max_size: f64) -> Option<Vec<usize>> {
    let f32s = |name| match &cloud.attribute(name)?.values {
        AttributeValues::F32(v) => Some(v),
        AttributeValues::U8(_) => None,
    };
    let (opacity, size) = (f32s(OPACITY)?, f32s(SPLAT_SIZE)?);
    Some(
        (0..cloud.len())
            .filter(|&i| f64::from(opacity[i]) >= min_opacity && f64::from(size[i]) <= max_size)
            .collect(),
    )
}

/// Mean distance of every point to its `k` nearest other points (the SOR
/// statistic). Needs more than `k` points.
pub fn knn_mean_distances(points: &[[f64; 3]], k: usize) -> Vec<f64> {
    let Some(tree) = KdTree::new(points) else {
        return Vec::new();
    };
    // Query in Morton order so consecutive searches touch the same part of
    // the tree. k + 1 because the nearest neighbour of a point is itself.
    let mut means = vec![0.0; points.len()];
    for i in crate::distance::morton_order(points) {
        means[i] = mean_of(&tree.nearest_k(&points[i], k + 1));
    }
    means
}

/// Mean of the distances after the first (the point itself), from squared
/// distances in ascending order.
fn mean_of(hits: &[(usize, f64)]) -> f64 {
    let k = hits.len().saturating_sub(1).max(1);
    hits.iter().skip(1).map(|&(_, d2)| d2.sqrt()).sum::<f64>() / k as f64
}

/// Indices whose SOR statistic is at most `ratio` standard deviations above
/// the mean.
pub fn sor_keep(means: &[f64], ratio: f64) -> Vec<usize> {
    let n = means.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = means.iter().sum::<f64>() / n as f64;
    let var = means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / n as f64;
    let limit = mean + ratio * var.sqrt();
    (0..n).filter(|&i| means[i] <= limit).collect()
}

// ---------------------------------------------------------------- SOR on workers
//
// For workers that share no memory, the cloud is split into spatially
// compact parts. [`KnnPart::local`] finds each point's neighbours within its
// own part; the result is exact when the ball reaching its (k+1)-th local
// neighbour touches no other part's box. The other points go to
// [`KnnPart::within`] on each part their ball touches, and [`merge_knn`]
// combines the candidates. The statistic is bit-identical to [`knn_mean_distances`].

/// Axis-aligned box as `(min, max)`.
pub type Aabb = ([f64; 3], [f64; 3]);

/// Where each part's points lie, to tell which parts a ball can reach.
#[derive(Debug, Clone, PartialEq)]
pub enum Regions {
    /// One bounding box per part.
    Boxes(Vec<Aabb>),
    /// A `dims`³ lattice of cubic cells of edge `cell` from `lo`, each owned
    /// by at most one part (`u8::MAX` = none); every point of a part lies in
    /// a cell it owns.
    Grid {
        lo: [f64; 3],
        cell: f64,
        dims: usize,
        owner: Vec<u8>,
    },
}

impl Regions {
    /// Bit mask of the parts other than `own` with a region the ball of
    /// radius `r` around `p` reaches.
    pub fn reached(&self, p: &[f64; 3], r: f64, own: usize) -> u32 {
        let mut mask = 0u32;
        match self {
            Regions::Boxes(boxes) => {
                for (j, b) in boxes.iter().enumerate() {
                    let d2: f64 = (0..3)
                        .map(|a| (b.0[a] - p[a]).max(p[a] - b.1[a]).max(0.0).powi(2))
                        .sum();
                    if j != own && d2 <= r * r {
                        mask |= 1 << j;
                    }
                }
            }
            Regions::Grid {
                lo,
                cell,
                dims,
                owner,
            } => {
                // Cells overlapping the ball's bounding box (a superset of
                // those the ball reaches, so never misses a part), widened
                // a little for points right on a cell face.
                let slack = (lo.iter().fold(0.0f64, |m, v| m.max(v.abs())) + cell * *dims as f64)
                    * 1e-12
                    + cell * 1e-9;
                let range = |a: usize| {
                    let at = |v: f64| {
                        ((v - lo[a]) / cell).floor().clamp(0.0, (*dims - 1) as f64) as usize
                    };
                    at(p[a] - r - slack)..=at(p[a] + r + slack)
                };
                for z in range(2) {
                    for y in range(1) {
                        for x in range(0) {
                            let o = owner[(z * dims + y) * dims + x];
                            if o != u8::MAX && o as usize != own {
                                mask |= 1 << o;
                            }
                        }
                    }
                }
            }
        }
        mask
    }

    /// Flat form for passing to workers: `[0, boxes...]` (six numbers each)
    /// or `[1, loX, loY, loZ, cell, dims, owner...]`.
    pub fn to_flat(&self) -> Vec<f64> {
        match self {
            Regions::Boxes(boxes) => std::iter::once(0.0)
                .chain(
                    boxes
                        .iter()
                        .flat_map(|(lo, hi)| [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]),
                )
                .collect(),
            Regions::Grid {
                lo,
                cell,
                dims,
                owner,
            } => [1.0, lo[0], lo[1], lo[2], *cell, *dims as f64]
                .into_iter()
                .chain(owner.iter().map(|&o| o as f64))
                .collect(),
        }
    }

    pub fn from_flat(flat: &[f64]) -> Option<Self> {
        match flat.split_first()? {
            (&0.0, rest) if rest.len().is_multiple_of(6) => Some(Regions::Boxes(
                rest.as_chunks::<6>()
                    .0
                    .iter()
                    .map(|b| ([b[0], b[1], b[2]], [b[3], b[4], b[5]]))
                    .collect(),
            )),
            (&1.0, [x, y, z, cell, dims, owner @ ..]) => {
                let dims = *dims as usize;
                (dims > 0 && owner.len() == dims * dims * dims && *cell > 0.0).then(|| {
                    Regions::Grid {
                        lo: [*x, *y, *z],
                        cell: *cell,
                        dims,
                        owner: owner.iter().map(|&o| o as u8).collect(),
                    }
                })
            }
            _ => None,
        }
    }
}

/// Spatially compact parts of a cloud (at most 32) and where they lie.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnSplit {
    pub parts: Vec<Vec<u32>>,
    pub regions: Regions,
}

/// Split `points` into up to `parts` (at most 32) compact parts by median
/// cuts.
pub fn split_for_knn(points: &[[f64; 3]], parts: usize) -> KnnSplit {
    let mut items: Vec<([f64; 3], u32)> = points.iter().zip(0..).map(|(&p, i)| (p, i)).collect();
    let mut ranges = Vec::new();
    crate::distance::split_ranges(&mut items, 0, parts.clamp(1, 32), &mut ranges);
    let mut out = Vec::with_capacity(ranges.len());
    let mut boxes = Vec::with_capacity(ranges.len());
    for range in ranges {
        let slice = &items[range];
        boxes.push(crate::distance::bounds(slice.iter().map(|(p, _)| p)));
        out.push(slice.iter().map(|&(_, i)| i).collect());
    }
    KnnSplit {
        parts: out,
        regions: Regions::Boxes(boxes),
    }
}

/// Split a cloud in octree order (see [`crate::octree::Octree`]) into up to
/// `parts` (at most 32) compact parts without sorting: subtrees at `level`
/// (or leaves above it) are whole units, taken in depth-first order and
/// grouped to about equal size; the points kept by nodes above `level` join
/// the part owning their lattice cell.
pub fn split_octree_for_knn(
    positions: &[[f64; 3]],
    nodes: &[crate::octree::OctreeNode],
    parts: usize,
    level: u8,
) -> KnnSplit {
    use crate::octree::NO_CHILD;
    let parts = parts.clamp(1, 32);
    let Some(root) = nodes.first() else {
        return split_for_knn(positions, parts);
    };
    let dims = 1usize << level.min(6);
    let cell = root.size / dims as f64;
    let lo = root.min;
    let n = positions.len();
    // End of each node's subtree range (children come after their parent).
    let mut end = vec![0u32; nodes.len()];
    for id in (0..nodes.len()).rev() {
        let node = &nodes[id];
        end[id] = node
            .children
            .iter()
            .filter(|&&c| c != NO_CHILD)
            .map(|&c| end[c as usize])
            .fold(node.start + node.count, u32::max);
    }
    // Units in depth-first order; coarse nodes' own points are set aside.
    let mut units: Vec<usize> = Vec::new();
    let mut coarse: Vec<usize> = Vec::new();
    let mut stack = vec![0usize];
    while let Some(id) = stack.pop() {
        let node = &nodes[id];
        let leaf = node.children.iter().all(|&c| c == NO_CHILD);
        if node.level >= level || leaf {
            units.push(id);
        } else {
            coarse.push(id);
            stack.extend(
                node.children
                    .iter()
                    .rev()
                    .filter(|&&c| c != NO_CHILD)
                    .map(|&c| c as usize),
            );
        }
    }
    // Each unit goes to the part its middle point falls in, by count.
    let unit_points: usize = units
        .iter()
        .map(|&id| (end[id] - nodes[id].start) as usize)
        .sum();
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); parts];
    let mut owner = vec![u8::MAX; dims * dims * dims];
    let cell_of = |p: &[f64; 3]| -> [usize; 3] {
        std::array::from_fn(|a| {
            ((p[a] - lo[a]) / cell)
                .floor()
                .clamp(0.0, (dims - 1) as f64) as usize
        })
    };
    let index = |c: [usize; 3]| (c[2] * dims + c[1]) * dims + c[0];
    let mut before = 0usize;
    for &id in &units {
        let node = &nodes[id];
        let size = (end[id] - node.start) as usize;
        let j = ((before + size / 2) * parts / unit_points.max(1)).min(parts - 1);
        before += size;
        out[j].extend(node.start..end[id]);
        // Claim every cell of the unit's cube (one cell at `level`).
        let (c0, c1) = (
            cell_of(&node.min.map(|v| v + 0.25 * cell)),
            cell_of(&node.min.map(|v| v + node.size - 0.25 * cell)),
        );
        for z in c0[2]..=c1[2] {
            for y in c0[1]..=c1[1] {
                for x in c0[0]..=c1[0] {
                    owner[index([x, y, z])] = j as u8;
                }
            }
        }
    }
    // Coarse points join their cell's owner; an unowned cell goes to the
    // part of the first point that lands in it.
    for &id in &coarse {
        let node = &nodes[id];
        for i in node.start..node.start + node.count {
            let c = index(cell_of(&positions[i as usize]));
            if owner[c] == u8::MAX {
                let fewest = (0..parts).min_by_key(|&j| out[j].len()).unwrap_or(0);
                owner[c] = fewest as u8;
            }
            out[owner[c] as usize].push(i);
        }
    }
    out.retain(|p| !p.is_empty());
    // Renumber owners after dropping empty parts.
    let mut map = [u8::MAX; 32];
    let mut next = 0u8;
    for (j, slot) in map.iter_mut().enumerate().take(parts) {
        if owner.contains(&(j as u8)) {
            *slot = next;
            next += 1;
        }
    }
    let owner = owner
        .into_iter()
        .map(|o| if o == u8::MAX { o } else { map[o as usize] })
        .collect();
    debug_assert_eq!(out.iter().map(Vec::len).sum::<usize>(), n);
    KnnSplit {
        parts: out,
        regions: Regions::Grid {
            lo,
            cell,
            dims,
            owner,
        },
    }
}

/// Result of [`KnnPart::local`].
#[derive(Debug, Clone, PartialEq)]
pub struct LocalKnn {
    /// The statistic per point of the part, NaN where not yet exact.
    pub means: Vec<f64>,
    /// Part-local indices of the points still open.
    pub open: Vec<u32>,
    /// Their `k + 1` nearest squared distances within the part, ascending,
    /// padded with infinity.
    pub candidates: Vec<f64>,
    /// Per open point, the bit mask of the other parts to ask.
    pub reach: Vec<u32>,
}

/// One part of a split SOR job with its k-d tree, kept by a worker between
/// step 1 ([`KnnPart::local`]) and step 2 ([`KnnPart::within`]).
pub struct KnnPart {
    points: Vec<[f64; 3]>,
    tree: Option<KdTree>,
}

impl KnnPart {
    pub fn new(points: Vec<[f64; 3]>) -> Self {
        let tree = KdTree::new(&points);
        Self { points, tree }
    }

    /// Point `i` of the part.
    pub fn point(&self, i: usize) -> [f64; 3] {
        self.points[i]
    }

    /// Step 1 on part `own`, given where every part lies.
    pub fn local(&self, k: usize, own: usize, regions: &Regions) -> LocalKnn {
        let points = &self.points;
        let mut out = LocalKnn {
            means: vec![f64::NAN; points.len()],
            open: Vec::new(),
            candidates: Vec::new(),
            reach: Vec::new(),
        };
        let Some(tree) = &self.tree else {
            return out;
        };
        for i in crate::distance::morton_order(points) {
            let hits = tree.nearest_k(&points[i], k + 1);
            let radius = hits.get(k).map_or(f64::INFINITY, |h| h.1.sqrt());
            let reach = regions.reached(&points[i], radius, own);
            if reach == 0 {
                out.means[i] = mean_of(&hits);
            } else {
                out.open.push(i as u32);
                out.reach.push(reach);
                out.candidates.extend(hits.iter().map(|h| h.1));
                out.candidates
                    .extend(std::iter::repeat_n(f64::INFINITY, k + 1 - hits.len()));
            }
        }
        out
    }

    /// Step 2: for each query (an open point of another part), its `k + 1`
    /// nearest squared distances among this part's points, ascending,
    /// padded with infinity.
    pub fn within(&self, queries: &[[f64; 3]], k: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(queries.len() * (k + 1));
        for q in queries {
            let hits = self
                .tree
                .as_ref()
                .map(|t| t.nearest_k(q, k + 1))
                .unwrap_or_default();
            out.extend(hits.iter().map(|h| h.1));
            out.extend(std::iter::repeat_n(f64::INFINITY, k + 1 - hits.len()));
        }
        out
    }
}

/// The statistic of an open point from its candidate lists (`k + 1`
/// squared distances each, from its own part and every part it reached).
pub fn merge_knn<'a>(lists: impl IntoIterator<Item = &'a [f64]>, k: usize) -> f64 {
    let mut all: Vec<f64> = lists.into_iter().flatten().copied().collect();
    all.sort_by(f64::total_cmp);
    all.truncate(k + 1);
    let hits: Vec<(usize, f64)> = all.into_iter().map(|d| (0, d)).collect();
    mean_of(&hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Attribute, INTENSITY};

    fn cloud(positions: Vec<[f64; 3]>) -> PointCloud {
        PointCloud {
            positions,
            colors: None,
            attributes: Vec::new(),
        }
    }

    fn grid(n: usize, step: f64) -> Vec<[f64; 3]> {
        (0..n * n)
            .map(|i| [(i % n) as f64 * step, (i / n) as f64 * step, 0.0])
            .collect()
    }

    #[test]
    fn voxel_keeps_one_point_per_voxel() {
        // A 100 x 100 grid with 0.1 spacing, subsampled with 0.5 voxels.
        let c = cloud(grid(100, 0.1));
        let keep = voxel_subsample(&c, 0.5);
        assert_eq!(keep.len(), 20 * 20);
        assert!(keep.windows(2).all(|w| w[0] < w[1]));
        // The first point of each voxel is kept, e.g. the origin.
        assert_eq!(keep[0], 0);
        // A non-positive size keeps everything.
        assert_eq!(voxel_subsample(&c, 0.0).len(), c.len());
    }

    #[test]
    fn spatial_subsample_keeps_points_at_least_the_distance_apart() {
        // A 0.1 grid with a 0.25 minimum distance: roughly one point in six or more.
        let c = cloud(grid(30, 0.1));
        let keep = spatial_subsample(&c, 0.25);
        assert!((80..=160).contains(&keep.len()), "{}", keep.len());
        for (n, &i) in keep.iter().enumerate() {
            for &j in &keep[n + 1..] {
                let (p, q) = (c.positions[i], c.positions[j]);
                assert!((0..3).map(|a| (p[a] - q[a]).powi(2)).sum::<f64>() >= 0.25 * 0.25 - 1e-12);
            }
        }
        // Every dropped point has a kept one within the distance.
        let kept: std::collections::HashSet<usize> = keep.iter().copied().collect();
        for (i, p) in c
            .positions
            .iter()
            .enumerate()
            .filter(|(i, _)| !kept.contains(i))
        {
            assert!(
                keep.iter().any(|&k| {
                    let q = c.positions[k];
                    (0..3).map(|a| (p[a] - q[a]).powi(2)).sum::<f64>() < 0.25 * 0.25
                }),
                "point {i} has no kept neighbour"
            );
        }
        assert_eq!(spatial_subsample(&c, 0.0).len(), c.len());
    }

    #[test]
    fn octree_subsample_keeps_the_point_nearest_each_cell_centre() {
        // 64 x 64 grid over [0, 6.3]: level 3 cuts the 6.3 cube into 8 x 8 cells.
        let c = cloud(grid(64, 0.1));
        let keep = octree_subsample(&c, 3);
        assert_eq!(keep.len(), 64);
        // The first cell (0..0.7875) has its centre at 0.39375: the point at 0.4 wins.
        let first = c.positions[keep[0]];
        assert!(
            (first[0] - 0.4).abs() < 1e-9 && (first[1] - 0.4).abs() < 1e-9,
            "{first:?}"
        );
        assert_eq!(octree_subsample(&c, 21).len(), c.len());
    }

    #[test]
    fn random_subsample_is_exact_and_deterministic() {
        let c = cloud(grid(50, 1.0));
        let a = random_subsample(&c, 300, 7);
        let b = random_subsample(&c, 300, 7);
        assert_eq!(a, b);
        assert_eq!(a.len(), 300);
        let mut unique = a.clone();
        unique.dedup();
        assert_eq!(unique.len(), 300);
        assert_ne!(a, random_subsample(&c, 300, 8));
        assert_eq!(random_subsample(&c, 10_000, 1).len(), c.len());
    }

    #[test]
    fn splat_cleanup_drops_faint_and_huge_splats() {
        let mut c = cloud(grid(2, 1.0));
        assert_eq!(splat_cleanup(&c, 0.1, 1.0), None);
        for (name, values) in [
            (OPACITY, vec![0.9, 0.05, 0.5, 0.8]),
            (SPLAT_SIZE, vec![0.01, 0.01, 3.0, 0.2]),
        ] {
            c.attributes.push(Attribute {
                name: name.into(),
                values: AttributeValues::F32(values),
            });
        }
        assert_eq!(splat_cleanup(&c, 0.1, 1.0), Some(vec![0, 3]));
        assert_eq!(
            splat_cleanup(&c, 0.0, f64::INFINITY),
            Some(vec![0, 1, 2, 3])
        );
    }

    #[test]
    fn sor_removes_isolated_points() {
        let mut positions = grid(40, 0.1);
        let outliers = [[10.0, 10.0, 5.0], [-8.0, 3.0, 2.0], [2.0, 2.0, 9.0]];
        positions.extend(outliers);
        let c = cloud(positions);
        let keep = statistical_outliers(&c, 8, 1.0);
        assert_eq!(keep.len(), 40 * 40, "only the three far points go");
        assert!(keep.iter().all(|&i| i < 1600));
    }

    /// Run the worker protocol here and compare with the plain statistic.
    #[test]
    fn split_knn_matches_the_plain_statistic() {
        let mut positions = grid(90, 0.1);
        let mut s = 7u64;
        for p in positions.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            p[2] = (s % 1000) as f64 * 1e-4;
        }
        positions.extend([[20.0, 20.0, 3.0], [-5.0, 1.0, 0.0], [4.5, 4.5, 0.2]]);
        let k = 8;
        // In octree order, so the octree split applies too.
        let tree = crate::octree::Octree::build_in_place(
            &mut positions,
            crate::octree::OctreeParams {
                max_leaf: 300,
                grid: 8,
                max_depth: 20,
            },
        )
        .unwrap();
        let plain = knn_mean_distances(&positions, k);
        let splits = [1, 2, 5, 8].into_iter().flat_map(|parts| {
            [
                (format!("median {parts}"), split_for_knn(&positions, parts)),
                (
                    format!("octree {parts}"),
                    split_octree_for_knn(&positions, &tree.nodes, parts, 3),
                ),
            ]
        });
        for (name, split) in splits {
            let mut covered: Vec<u32> = split.parts.concat();
            covered.sort_unstable();
            assert_eq!(
                covered,
                (0..positions.len() as u32).collect::<Vec<_>>(),
                "{name}"
            );
            let mut means = vec![f64::NAN; positions.len()];
            let part_points: Vec<Vec<[f64; 3]>> = split
                .parts
                .iter()
                .map(|idx| idx.iter().map(|&i| positions[i as usize]).collect())
                .collect();
            for (own, idx) in split.parts.iter().enumerate() {
                let local = KnnPart::new(part_points[own].clone()).local(k, own, &split.regions);
                for (j, &m) in local.means.iter().enumerate() {
                    if !m.is_nan() {
                        means[idx[j] as usize] = m;
                    }
                }
                for (o, &j) in local.open.iter().enumerate() {
                    let p = part_points[own][j as usize];
                    let mut lists = vec![local.candidates[o * (k + 1)..(o + 1) * (k + 1)].to_vec()];
                    for (other, pts) in part_points.iter().enumerate() {
                        if local.reach[o] >> other & 1 == 1 {
                            lists.push(KnnPart::new(pts.clone()).within(&[p], k));
                        }
                    }
                    means[idx[j as usize] as usize] =
                        merge_knn(lists.iter().map(|l| l.as_slice()), k);
                }
            }
            assert_eq!(means, plain, "{name}");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_sor_matches_serial() {
        let mut positions = grid(120, 0.1);
        positions.extend([[20.0, 20.0, 3.0], [-5.0, 1.0, 0.0]]);
        let c = cloud(positions);
        assert_eq!(
            statistical_outliers_par(&c, 8, 1.0),
            statistical_outliers(&c, 8, 1.0)
        );
    }

    #[test]
    fn select_keeps_attributes() {
        let mut c = cloud(grid(10, 1.0));
        c.attributes.push(Attribute {
            name: INTENSITY.into(),
            values: AttributeValues::F32((0..100).map(|i| i as f32).collect()),
        });
        let kept = c.select(&voxel_subsample(&c, 2.0));
        assert_eq!(kept.len(), 25);
        let AttributeValues::F32(v) = &kept.attributes[0].values else {
            panic!()
        };
        assert_eq!(v[1], 2.0);
    }
}
