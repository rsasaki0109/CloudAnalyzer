//! Nested octree for level-of-detail rendering (Potree-style).
//!
//! Every point is stored in exactly one node. An inner node keeps a
//! spatially uniform subsample of its cube (at most one point per cell of a
//! `grid³` lattice) and passes the remaining points on to its eight children,
//! so rendering a node together with all of its ancestors shows the cloud at
//! that node's density. Points are reordered so each node's points are one
//! contiguous range, laid out in depth-first pre-order.

/// Marker for a missing child.
pub const NO_CHILD: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct OctreeNode {
    /// Range of this node's own points in [`Octree::order`].
    pub start: u32,
    pub count: u32,
    /// Minimum corner and edge length of the node's cube.
    pub min: [f64; 3],
    pub size: f64,
    pub level: u8,
    /// Child node indices by octant (`x | y << 1 | z << 2`), or [`NO_CHILD`].
    pub children: [u32; 8],
}

impl OctreeNode {
    /// Approximate distance between neighbouring points kept in this node.
    pub fn spacing(&self, grid: u32) -> f64 {
        self.size / grid as f64
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Octree {
    /// `order[i]` is the index of the original point stored at position `i`.
    pub order: Vec<u32>,
    /// Nodes in depth-first pre-order; the root is `nodes[0]`.
    pub nodes: Vec<OctreeNode>,
    /// Subsampling lattice resolution per node edge.
    pub grid: u32,
}

/// Parameters for [`Octree::build_in_place`].
#[derive(Debug, Clone, Copy)]
pub struct OctreeParams {
    /// Nodes with at most this many points become leaves.
    pub max_leaf: usize,
    /// Subsampling lattice resolution per node edge; a power of two.
    pub grid: u32,
    /// Hard depth limit, which also bounds recursion on duplicate points.
    /// Capped at [`MORTON_BITS`] levels.
    pub max_depth: u8,
}

impl Default for OctreeParams {
    fn default() -> Self {
        Self {
            max_leaf: 20_000,
            grid: 128,
            max_depth: 20,
        }
    }
}

/// Quantisation bits per axis of the Morton codes used during the build.
pub const MORTON_BITS: u32 = 21;

impl Octree {
    /// Build an octree over `points`, reordering them in place into octree
    /// order (`order[i]` is the original index of the point now at `i`).
    ///
    /// The build works on 64-bit Morton codes of the points (quantised to
    /// [`MORTON_BITS`] bits per axis within the root cube): octants and
    /// lattice cells are just bit fields of the code, so no floating point is
    /// needed per level. Points move together with their codes; the octant
    /// partition writes eight sequential streams, which stays cache-friendly
    /// even for randomly ordered input (a single final gather would not).
    ///
    /// Returns `None` for an empty input, more than `u32::MAX` points, or a
    /// `grid` that is not a power of two.
    pub fn build_in_place(points: &mut [[f64; 3]], params: OctreeParams) -> Option<Self> {
        Self::build(points, None, params)
    }

    /// Like [`Octree::build_in_place`], reordering the cloud's colors along
    /// with its positions.
    pub fn build_for_cloud(cloud: &mut crate::PointCloud, params: OctreeParams) -> Option<Self> {
        Self::build(&mut cloud.positions, cloud.colors.as_deref_mut(), params)
    }

    fn build(
        points: &mut [[f64; 3]],
        colors: Option<&mut [[u8; 3]]>,
        params: OctreeParams,
    ) -> Option<Self> {
        let (lo, hi) = bounds(points);
        let size = (0..3)
            .map(|a| hi[a] - lo[a])
            .fold(0.0f64, f64::max)
            .max(f64::MIN_POSITIVE);
        Self::build_cube(points, colors, lo, size, 0, None, params).map(|(tree, _)| tree)
    }

    /// Build only the levels above `split_level`, leaving every node at that
    /// level which would still be split as a [`PendingSubtree`]. Finish each
    /// with [`Octree::build_subtree`] (possibly in parallel, on the pending
    /// point range) and attach it with [`Octree::graft`]. Points and colors
    /// are reordered in place as far as the partial build goes.
    pub fn build_partial(
        cloud: &mut crate::PointCloud,
        params: OctreeParams,
        split_level: u8,
    ) -> Option<(Self, Vec<PendingSubtree>)> {
        let (lo, hi) = bounds(&cloud.positions);
        let size = (0..3)
            .map(|a| hi[a] - lo[a])
            .fold(0.0f64, f64::max)
            .max(f64::MIN_POSITIVE);
        Self::build_cube(
            &mut cloud.positions,
            cloud.colors.as_deref_mut(),
            lo,
            size,
            0,
            Some(split_level),
            params,
        )
    }

    /// Build the octree of one pending subtree over its own points (the
    /// slice `start..end` of the partially built cloud). Node ranges and
    /// `order` are relative to the slice.
    pub fn build_subtree(
        points: &mut [[f64; 3]],
        colors: Option<&mut [[u8; 3]]>,
        pending: &PendingSubtree,
        params: OctreeParams,
    ) -> Option<Self> {
        Self::build_cube(
            points,
            colors,
            pending.min,
            pending.size,
            pending.level,
            None,
            params,
        )
        .map(|(tree, _)| tree)
    }

    /// Replace the placeholder node of `pending` with `subtree` (built by
    /// [`Octree::build_subtree`]), rebasing its point ranges and node indices.
    /// The caller writes the subtree's reordered points back into
    /// `pending.start..pending.end`.
    pub fn graft(&mut self, pending: &PendingSubtree, subtree: Octree) {
        let base = self.nodes.len() as u32;
        let map = |k: u32| {
            if k == NO_CHILD {
                NO_CHILD
            } else if k == 0 {
                pending.node
            } else {
                base + k - 1
            }
        };
        let rebase = |mut n: OctreeNode| {
            n.start += pending.start;
            n.children = n.children.map(map);
            n
        };
        let mut nodes = subtree.nodes.into_iter();
        if let Some(root) = nodes.next() {
            self.nodes[pending.node as usize] = rebase(root);
        }
        self.nodes.extend(nodes.map(rebase));
        let range = pending.start as usize..pending.end as usize;
        let slice: Vec<u32> = self.order[range.clone()].to_vec();
        for (dst, &i) in self.order[range].iter_mut().zip(&subtree.order) {
            *dst = slice[i as usize];
        }
    }

    /// Build over a given cube starting at `level`, optionally stopping at
    /// `split_level` with pending subtrees.
    fn build_cube(
        points: &mut [[f64; 3]],
        colors: Option<&mut [[u8; 3]]>,
        lo: [f64; 3],
        size: f64,
        level: u8,
        split_level: Option<u8>,
        params: OctreeParams,
    ) -> Option<(Self, Vec<PendingSubtree>)> {
        if colors.as_ref().is_some_and(|c| c.len() != points.len()) {
            return None;
        }
        if points.is_empty() || points.len() > u32::MAX as usize || !params.grid.is_power_of_two() {
            return None;
        }
        let cells = (1u64 << MORTON_BITS) as f64;
        let scale = cells / size;
        let quantize = |v: f64, a: usize| -> u64 {
            (((v - lo[a]) * scale) as u64).min((1 << MORTON_BITS) - 1)
        };
        let codes: Vec<u64> = points
            .iter()
            .map(|p| {
                spread_bits(quantize(p[0], 0))
                    | spread_bits(quantize(p[1], 1)) << 1
                    | spread_bits(quantize(p[2], 2)) << 2
            })
            .collect();
        let grid_bits = params.grid.trailing_zeros();
        let order = (0..points.len() as u32).collect();
        let mut builder = Builder {
            points,
            colors,
            codes,
            order,
            params: OctreeParams {
                // The codes resolve MORTON_BITS levels below the build cube.
                max_depth: params
                    .max_depth
                    .min(level.saturating_add(MORTON_BITS as u8)),
                ..params
            },
            base_level: level,
            split_level,
            pending: Vec::new(),
            grid_bits,
            stamps: vec![0; 1usize << (3 * grid_bits)],
            stamp: 0,
            nodes: Vec::new(),
        };
        let len = builder.codes.len();
        builder.node(0, len, lo, size, level);
        Some((
            Self {
                order: builder.order,
                nodes: builder.nodes,
                grid: params.grid,
            },
            builder.pending,
        ))
    }
}

/// A node left unbuilt by [`Octree::build_partial`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PendingSubtree {
    /// Placeholder node to replace.
    pub node: u32,
    /// Point range of the whole subtree.
    pub start: u32,
    pub end: u32,
    pub min: [f64; 3],
    pub size: f64,
    pub level: u8,
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

struct Builder<'a> {
    points: &'a mut [[f64; 3]],
    colors: Option<&'a mut [[u8; 3]]>,
    /// Morton code of each point, permuted together with `points` and `order`.
    codes: Vec<u64>,
    order: Vec<u32>,
    params: OctreeParams,
    /// Level of the build cube; code bits are relative to it.
    base_level: u8,
    split_level: Option<u8>,
    pending: Vec<PendingSubtree>,
    grid_bits: u32,
    /// Per-cell "last seen" marker, reused across nodes to avoid clearing.
    stamps: Vec<u32>,
    stamp: u32,
    nodes: Vec<OctreeNode>,
}

impl Builder<'_> {
    fn swap(&mut self, a: usize, b: usize) {
        self.points.swap(a, b);
        if let Some(colors) = self.colors.as_deref_mut() {
            colors.swap(a, b);
        }
        self.codes.swap(a, b);
        self.order.swap(a, b);
    }

    /// Build the node for `codes[lo..hi]` and return its index.
    fn node(&mut self, lo: usize, hi: usize, min: [f64; 3], size: f64, level: u8) -> u32 {
        let id = self.nodes.len();
        self.nodes.push(OctreeNode {
            start: lo as u32,
            count: (hi - lo) as u32,
            min,
            size,
            level,
            children: [NO_CHILD; 8],
        });
        if hi - lo <= self.params.max_leaf || level >= self.params.max_depth {
            return id as u32;
        }
        if self.split_level == Some(level) {
            self.pending.push(PendingSubtree {
                node: id as u32,
                start: lo as u32,
                end: hi as u32,
                min,
                size,
                level,
            });
            return id as u32;
        }

        // Keep the first point of every occupied lattice cell here, moving
        // those points to the front of the range. The cell is the next
        // `grid_bits` octree levels below this node, i.e. a bit field of the
        // code (coarser near the bottom where fewer bits remain).
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.stamps.fill(0);
            self.stamp = 1;
        }
        let below = MORTON_BITS - u32::from(level - self.base_level);
        let cell_levels = self.grid_bits.min(below);
        let shift = 3 * (below - cell_levels);
        let mask = (1u64 << (3 * cell_levels)) - 1;
        let mut kept = lo;
        for i in lo..hi {
            let key = ((self.codes[i] >> shift) & mask) as usize;
            if self.stamps[key] != self.stamp {
                self.stamps[key] = self.stamp;
                self.swap(kept, i);
                kept += 1;
            }
        }
        self.nodes[id].count = (kept - lo) as u32;

        // In-place 8-way partition of the remaining points by octant
        // (American flag sort) on the next three code bits.
        let octant_shift = 3 * (below - 1);
        let octant = |code: u64| ((code >> octant_shift) & 7) as usize;
        let mut counts = [0usize; 8];
        for &code in &self.codes[kept..hi] {
            counts[octant(code)] += 1;
        }
        let mut starts = [0usize; 8];
        let mut acc = kept;
        for o in 0..8 {
            starts[o] = acc;
            acc += counts[o];
        }
        let mut next = starts;
        for o in 0..8 {
            let end = starts[o] + counts[o];
            while next[o] < end {
                let target = octant(self.codes[next[o]]);
                if target == o {
                    next[o] += 1;
                } else {
                    self.swap(next[o], next[target]);
                    next[target] += 1;
                }
            }
        }

        let half = 0.5 * size;
        for o in 0..8 {
            if counts[o] == 0 {
                continue;
            }
            let child_min: [f64; 3] =
                std::array::from_fn(|a| min[a] + if o >> a & 1 == 1 { half } else { 0.0 });
            let child = self.node(starts[o], starts[o] + counts[o], child_min, half, level + 1);
            self.nodes[id].children[o] = child;
        }
        id as u32
    }
}

fn bounds(points: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_points(n: usize) -> Vec<[f64; 3]> {
        (0..n * n)
            .map(|i| {
                [
                    (i % n) as f64 * 0.1,
                    (i / n) as f64 * 0.1,
                    ((i % 7) as f64).sin(),
                ]
            })
            .collect()
    }

    fn params(max_leaf: usize, grid: u32) -> OctreeParams {
        OctreeParams {
            max_leaf,
            grid,
            max_depth: 20,
        }
    }

    #[test]
    fn every_point_is_stored_once_inside_its_node() {
        let original = grid_points(200);
        let mut points = original.clone();
        let tree = Octree::build_in_place(&mut points, params(500, 16)).unwrap();
        for (i, &o) in tree.order.iter().enumerate() {
            assert_eq!(points[i], original[o as usize]);
        }
        let mut seen = tree.order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..points.len() as u32).collect::<Vec<_>>());
        let total: u32 = tree.nodes.iter().map(|n| n.count).sum();
        assert_eq!(total as usize, points.len());
        for node in &tree.nodes {
            for p in &points[node.start as usize..(node.start + node.count) as usize] {
                for (a, &v) in p.iter().enumerate() {
                    assert!(v >= node.min[a] - 1e-9 && v <= node.min[a] + node.size + 1e-9);
                }
            }
        }
    }

    #[test]
    fn subtrees_are_contiguous_in_preorder() {
        let mut points = grid_points(150);
        let tree = Octree::build_in_place(&mut points, params(300, 8)).unwrap();
        // A node's subtree spans from its own start to the end of its last descendant.
        fn end(tree: &Octree, id: usize) -> u32 {
            let node = &tree.nodes[id];
            node.children
                .iter()
                .filter(|&&c| c != NO_CHILD)
                .map(|&c| end(tree, c as usize))
                .max()
                .unwrap_or(node.start + node.count)
        }
        assert_eq!(end(&tree, 0) as usize, points.len());
        for (id, node) in tree.nodes.iter().enumerate() {
            let mut next = node.start + node.count;
            for &c in node.children.iter().filter(|&&c| c != NO_CHILD) {
                assert_eq!(tree.nodes[c as usize].start, next, "node {id}");
                assert!(c as usize > id);
                next = end(&tree, c as usize);
            }
        }
    }

    #[test]
    fn inner_nodes_keep_at_most_one_point_per_cell() {
        let mut points = grid_points(300);
        let tree = Octree::build_in_place(&mut points, params(1000, 32)).unwrap();
        let root = &tree.nodes[0];
        assert!(root.children.iter().any(|&c| c != NO_CHILD));
        assert!(root.count as usize <= 32 * 32 * 32);
        assert!(root.count > 0);
    }

    #[test]
    fn partial_build_plus_subtrees_is_a_valid_octree() {
        let original = grid_points(250);
        let mut cloud = crate::PointCloud {
            positions: original.clone(),
            colors: Some(
                (0..original.len())
                    .map(|i| [i as u8, (i >> 8) as u8, (i >> 16) as u8])
                    .collect(),
            ),
        };
        let p = params(300, 16);
        let (mut tree, pending) = Octree::build_partial(&mut cloud, p, 2).unwrap();
        assert!(pending.len() > 4, "{}", pending.len());
        for job in &pending {
            let range = job.start as usize..job.end as usize;
            let colors = cloud.colors.as_mut().unwrap();
            let sub = Octree::build_subtree(
                &mut cloud.positions[range.clone()],
                Some(&mut colors[range]),
                job,
                p,
            )
            .unwrap();
            tree.graft(job, sub);
        }
        // Every point exactly once, positions and colors consistent with order.
        let colors = cloud.colors.unwrap();
        let mut seen = tree.order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..original.len() as u32).collect::<Vec<_>>());
        for (i, &o) in tree.order.iter().enumerate() {
            assert_eq!(cloud.positions[i], original[o as usize]);
            assert_eq!(colors[i], [o as u8, (o >> 8) as u8, (o >> 16) as u8]);
        }
        let total: u32 = tree.nodes.iter().map(|n| n.count).sum();
        assert_eq!(total as usize, original.len());
        // Every node's points lie in its cube; every child lies in its parent.
        for node in &tree.nodes {
            for p in &cloud.positions[node.start as usize..(node.start + node.count) as usize] {
                for (a, &v) in p.iter().enumerate() {
                    assert!(v >= node.min[a] - 1e-9 && v <= node.min[a] + node.size + 1e-9);
                }
            }
            for &c in node.children.iter().filter(|&&c| c != NO_CHILD) {
                let child = &tree.nodes[c as usize];
                assert_eq!(child.level, node.level + 1);
                assert!((child.size - node.size / 2.0).abs() < 1e-12);
            }
        }
        // A plain build of the same cloud has a similar shape.
        let mut again = original.clone();
        let plain = Octree::build_in_place(&mut again, p).unwrap();
        assert_eq!(plain.nodes[0].count, tree.nodes[0].count);
    }

    #[test]
    fn colors_follow_their_points() {
        let original = grid_points(120);
        let mut cloud = crate::PointCloud {
            positions: original.clone(),
            colors: Some(
                (0..original.len())
                    .map(|i| [i as u8, (i >> 8) as u8, (i >> 16) as u8])
                    .collect(),
            ),
        };
        let tree = Octree::build_for_cloud(&mut cloud, params(400, 16)).unwrap();
        let colors = cloud.colors.unwrap();
        for (i, &o) in tree.order.iter().enumerate() {
            assert_eq!(cloud.positions[i], original[o as usize]);
            assert_eq!(colors[i], [o as u8, (o >> 8) as u8, (o >> 16) as u8]);
        }
    }

    #[test]
    fn rejects_non_power_of_two_grid() {
        let mut points = grid_points(10);
        assert!(Octree::build_in_place(&mut points, params(10, 100)).is_none());
    }

    #[test]
    fn duplicates_stop_at_max_depth() {
        let mut points = vec![[1.0, 1.0, 1.0]; 5000];
        let tree = Octree::build_in_place(&mut points, params(10, 4)).unwrap();
        assert!(tree.nodes.iter().all(|n| n.level <= 20));
        let total: u32 = tree.nodes.iter().map(|n| n.count).sum();
        assert_eq!(total, 5000);
    }
}
