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

/// Parameters for [`Octree::build`].
#[derive(Debug, Clone, Copy)]
pub struct OctreeParams {
    /// Nodes with at most this many points become leaves.
    pub max_leaf: usize,
    /// Subsampling lattice resolution per node edge.
    pub grid: u32,
    /// Hard depth limit, which also bounds recursion on duplicate points.
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

impl Octree {
    /// Build an octree over `points`, reordering them in place into octree
    /// order (`order[i]` is the original index of the point now at `i`).
    /// Working in place keeps every pass sequential in memory and needs no
    /// scratch copy of the points. Returns `None` for an empty input or more
    /// than `u32::MAX` points.
    pub fn build_in_place(points: &mut [[f64; 3]], params: OctreeParams) -> Option<Self> {
        if points.is_empty() || points.len() > u32::MAX as usize {
            return None;
        }
        let (lo, hi) = bounds(points);
        let size = (0..3)
            .map(|a| hi[a] - lo[a])
            .fold(0.0f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let len = points.len();
        let mut builder = Builder {
            order: (0..len as u32).collect(),
            points,
            params,
            stamps: vec![0; (params.grid as usize).pow(3)],
            stamp: 0,
            nodes: Vec::new(),
        };
        builder.node(0, len, lo, size, 0);
        Some(Self {
            order: builder.order,
            nodes: builder.nodes,
            grid: params.grid,
        })
    }
}

struct Builder<'a> {
    points: &'a mut [[f64; 3]],
    params: OctreeParams,
    order: Vec<u32>,
    /// Per-cell "last seen" marker, reused across nodes to avoid clearing.
    stamps: Vec<u32>,
    stamp: u32,
    nodes: Vec<OctreeNode>,
}

impl Builder<'_> {
    fn swap(&mut self, a: usize, b: usize) {
        self.points.swap(a, b);
        self.order.swap(a, b);
    }

    /// Build the node for `points[lo..hi]` and return its index.
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

        // Keep the first point of every occupied lattice cell here, moving
        // those points to the front of the range.
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.stamps.fill(0);
            self.stamp = 1;
        }
        let grid = self.params.grid as usize;
        let scale = grid as f64 / size;
        let mut kept = lo;
        for i in lo..hi {
            let p = self.points[i];
            let cell: [usize; 3] =
                std::array::from_fn(|a| (((p[a] - min[a]) * scale) as usize).min(grid - 1));
            let key = (cell[2] * grid + cell[1]) * grid + cell[0];
            if self.stamps[key] != self.stamp {
                self.stamps[key] = self.stamp;
                self.swap(kept, i);
                kept += 1;
            }
        }
        self.nodes[id].count = (kept - lo) as u32;

        // In-place 8-way partition of the remaining points by octant
        // (American flag sort).
        let half = 0.5 * size;
        let octant = |p: &[f64; 3]| -> usize {
            (0..3)
                .map(|a| usize::from(p[a] >= min[a] + half) << a)
                .sum()
        };
        let mut counts = [0usize; 8];
        for p in &self.points[kept..hi] {
            counts[octant(p)] += 1;
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
                let target = octant(&self.points[next[o]]);
                if target == o {
                    next[o] += 1;
                } else {
                    self.swap(next[o], next[target]);
                    next[target] += 1;
                }
            }
        }

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
    fn duplicates_stop_at_max_depth() {
        let mut points = vec![[1.0, 1.0, 1.0]; 5000];
        let tree = Octree::build_in_place(&mut points, params(10, 4)).unwrap();
        assert!(tree.nodes.iter().all(|n| n.level <= 20));
        let total: u32 = tree.nodes.iter().map(|n| n.count).sum();
        assert_eq!(total, 5000);
    }
}
