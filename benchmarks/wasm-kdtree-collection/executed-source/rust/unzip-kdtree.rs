//! A static 3D k-d tree specialised for nearest-neighbour queries.
//!
//! Points are copied into tree order so leaves are contiguous in memory.
//! Construction splits at the median of the widest axis, so it is
//! `O(n log n)` regardless of input order or duplicate coordinates.

const LEAF_SIZE: usize = 16;

#[cfg(test)]
mod gpu_hint_tests {
    use super::*;

    #[test]
    fn hinted_k_neighbours_preserve_exact_indices_distances_and_tie_order() {
        for mode in 0..5 {
            let points: Vec<_> = (0..257)
                .map(|i| {
                    let p = [(i % 7) as f64, ((i / 7) % 7) as f64, (i % 3) as f64];
                    match mode {
                        1 => p.map(|v| v + 3955000.0),
                        2 => [1.0, -0.0, 0.0],
                        3 => p.map(|v| v * 1e155),
                        4 if i == 0 => [f64::NAN, f64::INFINITY, 0.0],
                        _ => p,
                    }
                })
                .collect();
            let tree = KdTree::new(&points).unwrap();
            let hints = KnnRadiusHints::new(&tree);
            let queries = points.iter().step_by(17).copied().chain([
                [0.5; 3],
                [-0.0; 3],
                [f64::INFINITY; 3],
                [f64::NAN; 3],
            ]);
            for q in queries {
                for k in [0, 1, 3, 12, 24, 25, 32, 64, 500] {
                    let expected = tree
                        .nearest_k(&q, k)
                        .into_iter()
                        .map(|(i, d)| (i, d.to_bits()))
                        .collect::<Vec<_>>();
                    let nearest: Vec<_> = expected
                        .iter()
                        .map(|(index, _)| {
                            tree.indices
                                .iter()
                                .position(|v| *v as usize == *index)
                                .unwrap() as u32
                        })
                        .collect();
                    let arbitrary: Vec<_> = (0..k)
                        .map(|i| ((i * 19 + 113) % tree.len()) as u32)
                        .collect();
                    let mut reversed = nearest.clone();
                    reversed.reverse();
                    for slots in [
                        nearest,
                        reversed,
                        arbitrary,
                        vec![0; k],
                        vec![u32::MAX; k],
                        Vec::new(),
                    ] {
                        let mut found = Vec::new();
                        tree.nearest_k_into_candidates(&q, k, &slots, &hints, &mut found);
                        assert_eq!(
                            expected,
                            found
                                .into_iter()
                                .map(|(i, d)| (i, d.to_bits()))
                                .collect::<Vec<_>>()
                        );
                    }
                    for node in (0..tree.nodes.len() as u32).chain([u32::MAX]) {
                        let mut found = Vec::new();
                        tree.nearest_k_into_hinted(&q, k, node, &hints, &mut found);
                        assert_eq!(
                            expected,
                            found
                                .into_iter()
                                .map(|(i, d)| (i, d.to_bits()))
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Node {
    /// Range of `points` covered by this node.
    start: u32,
    end: u32,
    /// Split coordinate and axis; `axis == LEAF` marks a leaf.
    split: f64,
    axis: u8,
    /// Index of the right child; the left child is always `self + 1`.
    right: u32,
}

const LEAF: u8 = u8::MAX;

#[derive(Debug, Clone)]
pub struct KdTree {
    nodes: Vec<Node>,
    points: Vec<[f64; 3]>,
    /// Original index of each entry in `points`.
    indices: Vec<u32>,
}

/// Exact subtree boxes for optional GPU-selected k-neighbour block hints.
/// A hint only provides an upper radius; the original traversal still selects
/// and orders every final hit, including ties.
pub(crate) struct KnnRadiusHints {
    bounds: Vec<([f64; 3], [f64; 3])>,
    finite: bool,
}

impl KnnRadiusHints {
    pub(crate) fn new(tree: &KdTree) -> Self {
        let mut out = Self {
            bounds: vec![([0.0; 3], [0.0; 3]); tree.nodes.len()],
            finite: tree.points.iter().flatten().all(|v| v.is_finite()),
        };
        fn fill(tree: &KdTree, node: usize, bounds: &mut [([f64; 3], [f64; 3])]) {
            let n = tree.nodes[node];
            bounds[node] = if n.axis == LEAF {
                crate::distance::bounds(tree.points[n.start as usize..n.end as usize].iter())
            } else {
                fill(tree, node + 1, bounds);
                fill(tree, n.right as usize, bounds);
                (
                    std::array::from_fn(|a| {
                        bounds[node + 1].0[a].min(bounds[n.right as usize].0[a])
                    }),
                    std::array::from_fn(|a| {
                        bounds[node + 1].1[a].max(bounds[n.right as usize].1[a])
                    }),
                )
            };
        }
        fill(tree, 0, &mut out.bounds);
        out
    }

    pub(crate) fn storage_bytes(&self) -> usize {
        self.bounds.capacity() * std::mem::size_of::<([f64; 3], [f64; 3])>()
    }
}

/// A nearest-neighbour hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nearest {
    /// Index into the slice the tree was built from.
    pub index: usize,
    pub distance_sq: f64,
    /// Position of the hit in tree order, for use as a later `guess`.
    slot: u32,
}

/// Call-local state shared by all recursive nodes of one bounded query.
/// Keep the invariant radius beside the incumbent instead of passing another
/// f64 argument at every recursive call. No heap allocation is involved.
struct BoundedNearest {
    best: Nearest,
    upper: f64,
}

// Whole-RPC real-source comparisons found the shared-state prototype slower.
// Keep the original radius argument unless explicitly rebuilding for comparison.
const USE_SHARED_NEAREST_STATE: bool = false;

impl Nearest {
    pub(crate) fn tree_slot(self) -> u32 {
        self.slot
    }
}

/// Finiteness checked once for an immutable tree before bounded queries.
pub(crate) struct NearestRadiusHints {
    finite: bool,
}

impl NearestRadiusHints {
    pub(crate) fn new(tree: &KdTree) -> Self {
        Self {
            finite: tree.points.iter().flatten().all(|v| v.is_finite()),
        }
    }
}

/// Exact verification of untrusted tree-order candidates. A leaf certificate
/// can exclude every other leaf using original f64 sibling extrema; otherwise
/// verification searches either the complete tree or the leaf's ancestral
/// siblings with exact subtree boxes.
#[derive(Debug)]
pub struct CandidateKdTree {
    tree: KdTree,
    leaves: Vec<LeafCertificate>,
    leaf_for_slot: Vec<u32>,
    bounds: Vec<([f64; 3], [f64; 3])>,
    parents: Vec<u32>,
    finite: bool,
}

#[derive(Debug, Clone, Copy)]
struct LeafCertificate {
    node: u32,
    start: u32,
    end: u32,
    /// Every point outside this leaf belongs to an ancestral sibling. That
    /// sibling is separated along its split axis by one of these extrema.
    lower: [f64; 3],
    upper: [f64; 3],
    coincident: bool,
}

impl CandidateKdTree {
    pub fn new(points: &[[f64; 3]]) -> Option<Self> {
        let tree = KdTree::new(points)?;
        Some(Self::from_tree(tree))
    }

    fn from_tree(tree: KdTree) -> Self {
        Self::from_tree_with_bounds(tree, None)
    }

    fn from_tree_with_bounds(tree: KdTree, imported: Option<Vec<SnapshotBounds>>) -> Self {
        // With total-order extrema, any NaN or infinity occurs at an axis
        // minimum or maximum. Six root values certify finiteness without
        // another pass over every imported coordinate.
        let finite = imported.as_ref().map_or_else(
            || tree.points.iter().flatten().all(|v| v.is_finite()),
            |bounds| {
                bounds[0]
                    .0
                    .iter()
                    .chain(&bounds[0].1)
                    .all(|v| v.is_finite())
            },
        );
        let mut out = Self {
            finite,
            leaf_for_slot: vec![0; tree.len()],
            tree,
            leaves: Vec::new(),
            bounds: Vec::new(),
            parents: Vec::new(),
        };
        // Keep exact subtree boxes as well as the cheap axis certificates.
        // They can exclude diagonal/corner regions when one axis cannot.
        // Validation uses total ordering, whose finite extrema differ from
        // numeric min/max only in signed-zero bits. Comparisons and squared
        // gaps retain their values. Keep the old path for nonfinite data.
        let imported = imported.filter(|_| out.finite);
        let mut bounds = imported.unwrap_or_default();
        out.parents = vec![0; out.tree.nodes.len()];
        if bounds.is_empty() {
            bounds.resize(out.tree.nodes.len(), ([0.0; 3], [0.0; 3]));
            out.subtree_bounds(0, &mut bounds);
        }
        out.certificates(0, [f64::NEG_INFINITY; 3], [f64::INFINITY; 3], &bounds);
        out.bounds = bounds;
        out
    }

    /// Portable exact tree data; derived verification certificates are rebuilt on import.
    pub fn snapshot(&self) -> Vec<u8> {
        self.tree.snapshot()
    }

    pub fn from_snapshot(bytes: &[u8], max_points: usize) -> Result<Self, &'static str> {
        Self::from_snapshot_with_reused_bounds(bytes, max_points, true)
    }

    /// Retain validated finite subtree extrema instead of rescanning points.
    /// Snapshot bytes and every validation step remain unchanged.
    pub fn from_snapshot_with_reused_bounds(
        bytes: &[u8],
        max_points: usize,
        reuse_bounds: bool,
    ) -> Result<Self, &'static str> {
        if reuse_bounds {
            let (tree, bounds) = KdTree::read_snapshot::<true>(bytes, max_points)?;
            Ok(Self::from_tree_with_bounds(tree, Some(bounds)))
        } else {
            KdTree::from_snapshot(bytes, max_points).map(Self::from_tree)
        }
    }

    fn subtree_bounds(&self, node: usize, bounds: &mut [([f64; 3], [f64; 3])]) {
        let n = self.tree.nodes[node];
        bounds[node] = if n.axis == LEAF {
            crate::distance::bounds(self.tree.points[n.start as usize..n.end as usize].iter())
        } else {
            let left = node + 1;
            let right = n.right as usize;
            self.subtree_bounds(left, bounds);
            self.subtree_bounds(right, bounds);
            (
                std::array::from_fn(|a| bounds[left].0[a].min(bounds[right].0[a])),
                std::array::from_fn(|a| bounds[left].1[a].max(bounds[right].1[a])),
            )
        };
    }

    fn certificates(
        &mut self,
        node: usize,
        lower: [f64; 3],
        upper: [f64; 3],
        bounds: &[([f64; 3], [f64; 3])],
    ) {
        let n = self.tree.nodes[node];
        if n.axis == LEAF {
            self.leaf_for_slot[n.start as usize..n.end as usize].fill(self.leaves.len() as u32);
            self.leaves.push(LeafCertificate {
                node: node as u32,
                start: n.start,
                end: n.end,
                lower,
                upper,
                coincident: bounds[node].0 == bounds[node].1,
            });
        } else {
            self.parents[node + 1] = node as u32;
            self.parents[n.right as usize] = node as u32;
            let axis = n.axis as usize;
            let mut left_upper = upper;
            left_upper[axis] = left_upper[axis].min(bounds[n.right as usize].0[axis]);
            self.certificates(node + 1, lower, left_upper, bounds);
            let mut right_lower = lower;
            right_lower[axis] = right_lower[axis].max(bounds[node + 1].1[axis]);
            self.certificates(n.right as usize, right_lower, upper, bounds);
        }
    }

    /// Returns the exact hit and whether a leaf certificate sufficed. Invalid
    /// slots, nonfinite data and failed certificates use the complete search.
    pub fn nearest(&self, query: &[f64; 3], slot: u32) -> (Nearest, bool) {
        self.nearest_with_options(query, slot, false, false)
    }

    /// Scan the candidate leaf, then exclude or search its ancestral siblings
    /// with exact subtree boxes. Those disjoint subtrees cover every point
    /// outside the initial leaf, even for an incorrect candidate.
    pub fn nearest_bounded(&self, query: &[f64; 3], slot: u32) -> (Nearest, bool) {
        self.nearest_with_options(query, slot, true, true)
    }

    /// Select exact sibling bounds and pairwise WASM SIMD independently.
    /// SIMD keeps scalar arithmetic and hit order; other targets use scalar.
    pub fn nearest_with_options(
        &self,
        query: &[f64; 3],
        slot: u32,
        bounded: bool,
        simd: bool,
    ) -> (Nearest, bool) {
        if !self.finite || !query.iter().all(|v| v.is_finite()) || slot as usize >= self.tree.len()
        {
            return (self.tree.nearest_seeded(query, slot), false);
        }
        let leaf = self.leaves[self.leaf_for_slot[slot as usize] as usize];
        let mut best = Nearest {
            index: 0,
            distance_sq: distance_sq(&self.tree.points[slot as usize], query),
            slot,
        };
        if !leaf.coincident {
            scan_candidate_leaf(
                &self.tree.points,
                leaf.start,
                leaf.end,
                query,
                &mut best,
                simd,
            );
        }
        let mut certified = best.distance_sq.is_finite()
            && (0..3).all(|a| {
                let low = query[a] - leaf.lower[a];
                let high = leaf.upper[a] - query[a];
                low >= 0.0
                    && high >= 0.0
                    && low * low >= best.distance_sq
                    && high * high >= best.distance_sq
            });
        if !certified && bounded && best.distance_sq.is_finite() {
            let mut searched_leaf = false;
            let mut node = leaf.node as usize;
            while node != 0 {
                let parent = self.parents[node] as usize;
                let axis = self.tree.nodes[parent].axis as usize;
                let (sibling, gap) = if node == parent + 1 {
                    let sibling = self.tree.nodes[parent].right as usize;
                    (sibling, self.bounds[sibling].0[axis] - query[axis])
                } else {
                    let sibling = parent + 1;
                    (sibling, query[axis] - self.bounds[sibling].1[axis])
                };
                // Most ancestral siblings are far along their split axis.
                // Reject those with one exact component before loading and
                // projecting the full box.
                if !(gap >= 0.0 && gap * gap >= best.distance_sq) {
                    self.search_bounded(sibling, query, &mut best, &mut searched_leaf, simd);
                }
                node = parent;
            }
            certified = !searched_leaf;
        } else if !certified {
            self.tree.search::<false>(0, query, &mut best);
        }
        best.index = self.tree.indices[best.slot as usize] as usize;
        (best, certified)
    }

    fn search_bounded(
        &self,
        node: usize,
        q: &[f64; 3],
        best: &mut Nearest,
        searched_leaf: &mut bool,
        simd: bool,
    ) {
        let (lo, hi) = self.bounds[node];
        let delta: [f64; 3] = std::array::from_fn(|a| {
            if q[a] < lo[a] {
                lo[a] - q[a]
            } else if q[a] > hi[a] {
                hi[a] - q[a]
            } else {
                0.0
            }
        });
        // Same component order and arithmetic as point distance. Each
        // projected component is <= the corresponding component of every
        // point in the box; monotone nonnegative sums preserve the bound.
        let lower = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
        if lower >= best.distance_sq {
            return;
        }
        let n = self.tree.nodes[node];
        if n.axis == LEAF {
            *searched_leaf = true;
            let end = if lo == hi { n.start + 1 } else { n.end };
            scan_candidate_leaf(&self.tree.points, n.start, end, q, best, simd);
        } else {
            let diff = q[n.axis as usize] - n.split;
            let (near, far) = if diff < 0.0 {
                (node + 1, n.right as usize)
            } else {
                (n.right as usize, node + 1)
            };
            self.search_bounded(near, q, best, searched_leaf, simd);
            if diff * diff < best.distance_sq {
                self.search_bounded(far, q, best, searched_leaf, simd);
            }
        }
    }

    pub fn candidate_buffers(&self, origin: [f64; 3]) -> (Vec<f32>, Vec<u32>) {
        self.tree.candidate_buffers(origin)
    }

    /// Restore the input order so another WASM instance reconstructs identical
    /// tree slots. Never round these coordinates through the GPU buffers.
    pub fn original_points(&self) -> Vec<[f64; 3]> {
        let mut points = vec![[0.0; 3]; self.tree.len()];
        for (&point, &index) in self.tree.points.iter().zip(&self.tree.indices) {
            points[index as usize] = point;
        }
        points
    }

    pub fn original_point(&self, index: usize) -> Option<[f64; 3]> {
        self.tree
            .indices
            .iter()
            .position(|&i| i as usize == index)
            .map(|slot| self.tree.points[slot])
    }
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

type SnapshotBounds = ([f64; 3], [f64; 3]);

fn validate_snapshot_node<const KEEP_BOUNDS: bool>(
    tree: &KdTree,
    node: usize,
    start: u32,
    end: u32,
    bounds: &mut Vec<SnapshotBounds>,
) -> Option<(usize, SnapshotBounds)> {
    let n = *tree.nodes.get(node)?;
    if n.start != start || n.end != end || start >= end {
        return None;
    }
    if n.axis == LEAF {
        if n.right != 0 || n.split.to_bits() != 0 {
            return None;
        }
        let mut lo = tree.points[start as usize];
        let mut hi = lo;
        for p in &tree.points[start as usize + 1..end as usize] {
            for a in 0..3 {
                if p[a].total_cmp(&lo[a]).is_lt() {
                    lo[a] = p[a];
                }
                if p[a].total_cmp(&hi[a]).is_gt() {
                    hi[a] = p[a];
                }
            }
        }
        if KEEP_BOUNDS {
            bounds[node] = (lo, hi);
        }
        return Some((node + 1, (lo, hi)));
    }
    if end - start <= LEAF_SIZE as u32 {
        return None;
    }
    // Balanced splits limit recursion depth even for a malicious snapshot.
    let mid = start + (end - start) / 2;
    let (right, left_bounds) =
        validate_snapshot_node::<KEEP_BOUNDS>(tree, node + 1, start, mid, bounds)?;
    if n.right as usize != right {
        return None;
    }
    let (next, right_bounds) =
        validate_snapshot_node::<KEEP_BOUNDS>(tree, right, mid, end, bounds)?;
    let axis = n.axis as usize;
    if left_bounds.1[axis].total_cmp(&n.split).is_gt()
        || right_bounds.0[axis].total_cmp(&n.split).is_lt()
    {
        return None;
    }
    let lower = std::array::from_fn(|a| {
        if left_bounds.0[a].total_cmp(&right_bounds.0[a]).is_lt() {
            left_bounds.0[a]
        } else {
            right_bounds.0[a]
        }
    });
    let upper = std::array::from_fn(|a| {
        if left_bounds.1[a].total_cmp(&right_bounds.1[a]).is_gt() {
            left_bounds.1[a]
        } else {
            right_bounds.1[a]
        }
    });
    if KEEP_BOUNDS {
        bounds[node] = (lower, upper);
    }
    Some((next, (lower, upper)))
}

fn scan_candidate_leaf(
    points: &[[f64; 3]],
    start: u32,
    end: u32,
    q: &[f64; 3],
    best: &mut Nearest,
    simd: bool,
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if simd {
        use core::arch::wasm32::*;
        let (qx, qy, qz) = (f64x2_splat(q[0]), f64x2_splat(q[1]), f64x2_splat(q[2]));
        let (pairs, remainder) = points[start as usize..end as usize].as_chunks::<2>();
        let mut minimum = f64x2_splat(best.distance_sq);
        for pair in pairs {
            // SAFETY: each pair contains six contiguous f64s. The three
            // unaligned 16-byte loads exactly cover those 48 valid bytes.
            let (xy, zx, yz) = unsafe {
                let ptr = pair.as_ptr().cast::<f64>();
                (
                    v128_load(ptr.cast()),
                    v128_load(ptr.add(2).cast()),
                    v128_load(ptr.add(4).cast()),
                )
            };
            let dx = f64x2_sub(i64x2_shuffle::<0, 3>(xy, zx), qx);
            let dy = f64x2_sub(i64x2_shuffle::<1, 2>(xy, yz), qy);
            let dz = f64x2_sub(i64x2_shuffle::<0, 3>(zx, yz), qz);
            // Preserve (dx*dx + dy*dy) + dz*dz, with separate multiply/add.
            // No fused/relaxed arithmetic or reassociated sums.
            let d = f64x2_add(
                f64x2_add(f64x2_mul(dx, dx), f64x2_mul(dy, dy)),
                f64x2_mul(dz, dz),
            );
            // Candidates usually already have the nearest distance. Reduce
            // distances without extracting lanes/updating indices per pair.
            // Finite inputs imply squared distances are nonnegative (possibly
            // infinite), so lane-wise min cannot introduce NaN or signed zero.
            minimum = f64x2_pmin(minimum, d);
        }
        if f64x2_extract_lane::<0>(minimum) >= best.distance_sq
            && f64x2_extract_lane::<1>(minimum) >= best.distance_sq
        {
            if let Some(point) = remainder.first() {
                let d = distance_sq(point, q);
                if d < best.distance_sq {
                    best.distance_sq = d;
                    best.slot = end - 1;
                }
            }
            return;
        }
        // A closer point requires its slot. The scalar scan preserves the
        // original strict comparisons and tie order, including odd tails.
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    let _ = simd;
    for i in start..end {
        let d = distance_sq(&points[i as usize], q);
        if d < best.distance_sq {
            best.distance_sq = d;
            best.slot = i;
        }
    }
}

/// Ordinary nearest searches often find a better hit in a leaf. Retain two
/// lane minima and slots, then select the first strict improvement in tree
/// order without scalar comparisons/branches for every point.
fn scan_nearest_leaf(points: &[[f64; 3]], start: u32, end: u32, q: &[f64; 3], best: &mut Nearest) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::*;
        let qx = f64x2_splat(q[0]);
        let qy = f64x2_splat(q[1]);
        let qz = f64x2_splat(q[2]);
        let mut minimum = f64x2_splat(best.distance_sq);
        let mut hits = i64x2_splat(i64::from(best.slot));
        let mut slots = i64x2(i64::from(start), i64::from(start) + 1);
        let step = i64x2_splat(2);
        let (pairs, tail) = points[start as usize..end as usize].as_chunks::<2>();
        for pair in pairs {
            let ptr = pair.as_ptr().cast::<f64>();
            // Exactly six f64 coordinates; each load is within this pair.
            let (xy, zx, yz) = unsafe {
                (
                    v128_load(ptr.cast()),
                    v128_load(ptr.add(2).cast()),
                    v128_load(ptr.add(4).cast()),
                )
            };
            let dx = f64x2_sub(i64x2_shuffle::<0, 3>(xy, zx), qx);
            let dy = f64x2_sub(i64x2_shuffle::<1, 2>(xy, yz), qy);
            let dz = f64x2_sub(i64x2_shuffle::<0, 3>(zx, yz), qz);
            let d = f64x2_add(
                f64x2_add(f64x2_mul(dx, dx), f64x2_mul(dy, dy)),
                f64x2_mul(dz, dz),
            );
            let closer = f64x2_lt(d, minimum);
            minimum = v128_bitselect(d, minimum, closer);
            hits = v128_bitselect(slots, hits, closer);
            slots = i64x2_add(slots, step);
        }
        select_leaf_minimum(
            best,
            [
                f64x2_extract_lane::<0>(minimum),
                f64x2_extract_lane::<1>(minimum),
            ],
            [
                i64x2_extract_lane::<0>(hits) as u32,
                i64x2_extract_lane::<1>(hits) as u32,
            ],
        );
        if let Some(point) = tail.first() {
            let d = distance_sq(point, q);
            if d < best.distance_sq {
                best.distance_sq = d;
                best.slot = end - 1;
            }
        }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    scan_candidate_leaf(points, start, end, q, best, false);
}

#[cfg(any(test, all(target_arch = "wasm32", target_feature = "simd128")))]
fn select_leaf_minimum(best: &mut Nearest, distances: [f64; 2], slots: [u32; 2]) {
    let seed = best.distance_sq;
    // A tied seed must survive. Among strict improvements from different
    // parity lanes, the smallest tree slot is the scalar scan's first hit.
    for k in 0..2 {
        if distances[k] < seed
            && (distances[k] < best.distance_sq
                || (distances[k] == best.distance_sq && slots[k] < best.slot))
        {
            best.distance_sq = distances[k];
            best.slot = slots[k];
        }
    }
}

impl KdTree {
    /// Build a tree over `points`. Returns `None` when `points` is empty or
    /// holds more than `u32::MAX` entries.
    pub fn new(points: &[[f64; 3]]) -> Option<Self> {
        if points.is_empty() || points.len() > u32::MAX as usize {
            return None;
        }
        // Partition (point, index) pairs in place: no indirection, so the
        // median selections stream through contiguous memory.
        let mut items: Vec<([f64; 3], u32)> =
            points.iter().zip(0..).map(|(&p, i)| (p, i)).collect();
        let mut nodes = Vec::with_capacity(2 * points.len() / LEAF_SIZE + 1);
        build(&mut items, 0, &mut nodes);
        let (points, indices): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Some(Self { nodes, points, indices })
    }

    /// Portable, versioned exact tree data for independent verification workers.
    /// Coordinates and splits retain all f64 bits; no Rust struct layout or
    /// rounded GPU buffers are serialized. Derived certificates are rebuilt.
    pub fn snapshot(&self) -> Vec<u8> {
        let tree = self;
        let mut bytes = Vec::with_capacity(16 + tree.len() * 28 + tree.nodes.len() * 24);
        bytes.extend_from_slice(b"CAKD");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(tree.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(tree.nodes.len() as u32).to_le_bytes());
        for point in &tree.points {
            for v in point {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for index in &tree.indices {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        for node in &tree.nodes {
            for v in [node.start, node.end, node.right, u32::from(node.axis)] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.extend_from_slice(&node.split.to_le_bytes());
        }
        bytes
    }

    /// Import in O(n), validating sizes, index permutation, balanced preorder
    /// topology and split partitions before any traversal or certificate use.
    /// The explicit point budget bounds allocation even for untrusted headers.
    pub fn from_snapshot(bytes: &[u8], max_points: usize) -> Result<Self, &'static str> {
        Self::read_snapshot::<false>(bytes, max_points).map(|(tree, _)| tree)
    }

    fn read_snapshot<const KEEP_BOUNDS: bool>(
        bytes: &[u8],
        max_points: usize,
    ) -> Result<(Self, Vec<SnapshotBounds>), &'static str> {
        const INVALID: &str = "invalid exact k-d tree snapshot";
        if bytes.len() < 16 || &bytes[..4] != b"CAKD" || read_u32(&bytes[4..8]) != 1 {
            return Err(INVALID);
        }
        let count = read_u32(&bytes[8..12]) as usize;
        let nodes = read_u32(&bytes[12..16]) as usize;
        if count == 0 || count > max_points || nodes == 0 || nodes > count.saturating_mul(2) - 1 {
            return Err(INVALID);
        }
        let expected = count
            .checked_mul(28)
            .and_then(|n| nodes.checked_mul(24).and_then(|m| n.checked_add(m)))
            .and_then(|n| n.checked_add(16))
            .ok_or(INVALID)?;
        if bytes.len() != expected {
            return Err(INVALID);
        }
        // Balanced internal ranges contain > LEAF_SIZE points, so non-root
        // leaves contain at least LEAF_SIZE/2 points. Reject an impossible
        // node header before reserving the optional derived array.
        let max_valid_nodes = count.saturating_mul(2) / (LEAF_SIZE / 2) + 1;
        if KEEP_BOUNDS && nodes > max_valid_nodes {
            return Err(INVALID);
        }
        let point_end = 16 + count * 24;
        let index_end = point_end + count * 4;
        let points = bytes[16..point_end]
            .as_chunks::<24>()
            .0
            .iter()
            .map(|p| {
                std::array::from_fn(|a| f64::from_le_bytes(p[a * 8..a * 8 + 8].try_into().unwrap()))
            })
            .collect();
        let mut seen = vec![false; count];
        let mut indices = Vec::with_capacity(count);
        for index in bytes[point_end..index_end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| read_u32(b))
        {
            if index as usize >= count || std::mem::replace(&mut seen[index as usize], true) {
                return Err(INVALID);
            }
            indices.push(index);
        }
        let mut decoded = Vec::with_capacity(nodes);
        for node in bytes[index_end..].as_chunks::<24>().0 {
            let axis = read_u32(&node[12..16]);
            if axis > 2 && axis != u32::from(LEAF) {
                return Err(INVALID);
            }
            decoded.push(Node {
                start: read_u32(&node[..4]),
                end: read_u32(&node[4..8]),
                right: read_u32(&node[8..12]),
                axis: axis as u8,
                split: f64::from_le_bytes(node[16..24].try_into().unwrap()),
            });
        }
        let tree = KdTree {
            points,
            indices,
            nodes: decoded,
        };
        let mut bounds = if KEEP_BOUNDS {
            // One bulk initialization; validated nodes fill their own slot.
            vec![([0.0; 3], [0.0; 3]); nodes]
        } else {
            Vec::new()
        };
        let (next, _) =
            validate_snapshot_node::<KEEP_BOUNDS>(&tree, 0, 0, count as u32, &mut bounds)
                .ok_or(INVALID)?;
        if next != nodes {
            return Err(INVALID);
        }
        Ok((tree, bounds))
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Coordinates in traversal order, retaining the original f64 bits.
    pub fn tree_points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// Nearest point to `query`. `guess`, typically the hit of a spatially
    /// close previous query, seeds the search so most branches are pruned.
    pub fn nearest(&self, query: &[f64; 3], guess: Option<Nearest>) -> Nearest {
        self.nearest_with_simd(query, guess, false)
    }

    /// Optional paired f64 WASM leaf arithmetic. Traversal, guesses and strict
    /// tie comparisons retain scalar order; other targets remain scalar.
    pub fn nearest_with_simd(
        &self,
        query: &[f64; 3],
        guess: Option<Nearest>,
        simd: bool,
    ) -> Nearest {
        let slot = guess.map_or(0, |g| g.slot);
        self.nearest_from_slot(query, slot, simd)
    }

    /// A candidate supplies only a recomputed f64 upper radius. The original
    /// seed and strict improvement/tie order are retained. `hints` must have
    /// been constructed for this immutable tree. The boolean records a tighter
    /// radius, not adoption of the candidate or a guaranteed branch saving.
    pub(crate) fn nearest_with_radius_hint(
        &self,
        query: &[f64; 3],
        guess: Option<Nearest>,
        candidate: u32,
        hints: &NearestRadiusHints,
        simd: bool,
    ) -> (Nearest, bool) {
        self.nearest_with_radius_layout::<USE_SHARED_NEAREST_STATE>(
            query, guess, candidate, hints, simd,
        )
    }

    fn nearest_with_radius_layout<const SHARED: bool>(
        &self,
        query: &[f64; 3],
        guess: Option<Nearest>,
        candidate: u32,
        hints: &NearestRadiusHints,
        simd: bool,
    ) -> (Nearest, bool) {
        let upper = self.points.get(candidate as usize).and_then(|p| {
            if !hints.finite || !query.iter().all(|v| v.is_finite()) {
                return None;
            }
            let d = distance_sq(p, query);
            d.is_finite().then_some(d)
        });
        let Some(upper) = upper else {
            return (self.nearest_with_simd(query, guess, simd), false);
        };
        let slot = guess.map_or(0, |g| g.slot);
        let mut best = Nearest {
            index: 0,
            distance_sq: distance_sq(&self.points[slot as usize], query),
            slot,
        };
        let tighter = upper < best.distance_sq;
        if SHARED && tighter {
            let mut state = BoundedNearest { best, upper };
            if simd {
                self.search_hinted_context::<true>(0, query, &mut state);
            } else {
                self.search_hinted_context::<false>(0, query, &mut state);
            }
            best = state.best;
        } else if tighter {
            if simd {
                self.search_hinted::<true>(0, query, &mut best, upper);
            } else {
                self.search_hinted::<false>(0, query, &mut best, upper);
            }
        } else if simd {
            self.search::<true>(0, query, &mut best);
        } else {
            self.search::<false>(0, query, &mut best);
        }
        best.index = self.indices[best.slot as usize] as usize;
        (best, tighter)
    }

    /// An untrusted candidate in tree order only seeds the upper bound. The
    /// full f64 search still runs, so a rounded or incorrect GPU candidate
    /// cannot change the distance. Out-of-range candidates use slot zero.
    pub fn nearest_seeded(&self, query: &[f64; 3], slot: u32) -> Nearest {
        let slot = if (slot as usize) < self.points.len() {
            slot
        } else {
            0
        };
        self.nearest_from_slot(query, slot, false)
    }

    fn nearest_from_slot(&self, query: &[f64; 3], slot: u32, simd: bool) -> Nearest {
        let mut best = Nearest {
            index: 0,
            distance_sq: distance_sq(&self.points[slot as usize], query),
            slot,
        };
        if simd {
            self.search::<true>(0, query, &mut best);
        } else {
            self.search::<false>(0, query, &mut best);
        }
        best.index = self.indices[best.slot as usize] as usize;
        best
    }

    /// Packed WebGPU candidate-search data relative to `origin`. Points are
    /// vec4<f32>; each node is five u32 words: start, end, right, axis, and
    /// the shifted split's f32 bits. Axis 255 denotes a leaf. These rounded
    /// copies must never be used as final distance measurements.
    pub fn candidate_buffers(&self, origin: [f64; 3]) -> (Vec<f32>, Vec<u32>) {
        let points = self
            .points
            .iter()
            .flat_map(|p| {
                [
                    (p[0] - origin[0]) as f32,
                    (p[1] - origin[1]) as f32,
                    (p[2] - origin[2]) as f32,
                    0.0,
                ]
            })
            .collect();
        (points, self.candidate_nodes(origin))
    }

    /// Rounded split data without the unused tree-order point buffer.
    pub fn candidate_nodes(&self, origin: [f64; 3]) -> Vec<u32> {
        self.nodes
            .iter()
            .flat_map(|n| {
                [
                    n.start,
                    n.end,
                    n.right,
                    u32::from(n.axis),
                    if n.axis == LEAF {
                        0
                    } else {
                        ((n.split - origin[n.axis as usize]) as f32).to_bits()
                    },
                ]
            })
            .collect()
    }

    /// Indices of all points within `radius` of `query` (unordered).
    pub fn within(&self, query: &[f64; 3], radius: f64, out: &mut Vec<usize>) {
        self.within_with_bounds(query, radius, out, false);
    }

    /// Combine ancestral split-plane lower bounds without changing hit order.
    /// No subtree boxes or additional per-tree storage are needed.
    pub fn within_with_bounds(
        &self,
        query: &[f64; 3],
        radius: f64,
        out: &mut Vec<usize>,
        bounds: bool,
    ) {
        out.clear();
        if radius >= 0.0 {
            if bounds {
                self.search_within::<true, false>(0, query, radius * radius, [0.0; 3], out);
            } else {
                self.search_within::<false, false>(0, query, radius * radius, [0.0; 3], out);
            }
        }
    }

    /// Same unsorted hit order as `within_with_bounds`, expressed as tree slots.
    /// Consumers can use `tree_points` without retaining the original input.
    pub fn within_slots_with_bounds(
        &self,
        query: &[f64; 3],
        radius: f64,
        out: &mut Vec<usize>,
        bounds: bool,
    ) {
        out.clear();
        if radius >= 0.0 {
            if bounds {
                self.search_within::<true, true>(0, query, radius * radius, [0.0; 3], out);
            } else {
                self.search_within::<false, true>(0, query, radius * radius, [0.0; 3], out);
            }
        }
    }

    fn search_within<const BOUNDS: bool, const SLOTS: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        r2: f64,
        lower: [f64; 3],
        out: &mut Vec<usize>,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            for i in n.start as usize..n.end as usize {
                if distance_sq(&self.points[i], q) <= r2 {
                    out.push(if SLOTS { i } else { self.indices[i] as usize });
                }
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_within::<BOUNDS, SLOTS>(near, q, r2, lower, out);
        if diff * diff <= r2 {
            let mut far_lower = lower;
            if BOUNDS {
                // Every point in the far child is at least this far along
                // the split axis. Retain stronger bounds from older splits.
                // Sum in the same x+y+z order as distance_sq; each nonnegative
                // component is a lower bound on the corresponding leaf term.
                far_lower[n.axis as usize] = far_lower[n.axis as usize].max(diff * diff);
                if far_lower[0] + far_lower[1] + far_lower[2] > r2 {
                    return;
                }
            }
            self.search_within::<BOUNDS, SLOTS>(far, q, r2, far_lower, out);
        }
    }

    /// Visit radius hits in the same order as `within_slots_with_bounds`,
    /// reusing leaf deltas/distance for a consumer without a temporary hit list.
    pub fn visit_within_with_bounds(
        &self,
        query: &[f64; 3],
        radius: f64,
        bounds: bool,
        visit: &mut impl FnMut(usize, [f64; 3], f64),
    ) {
        if radius >= 0.0 {
            if bounds {
                self.visit_within::<true>(0, query, radius * radius, [0.0; 3], visit);
            } else {
                self.visit_within::<false>(0, query, radius * radius, [0.0; 3], visit);
            }
        }
    }

    fn visit_within<const BOUNDS: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        r2: f64,
        lower: [f64; 3],
        visit: &mut impl FnMut(usize, [f64; 3], f64),
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            for i in n.start as usize..n.end as usize {
                let p = self.points[i];
                let v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                let d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if d2 <= r2 {
                    visit(i, v, d2);
                }
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.visit_within::<BOUNDS>(near, q, r2, lower, visit);
        if diff * diff <= r2 {
            let mut far_lower = lower;
            if BOUNDS {
                far_lower[n.axis as usize] = far_lower[n.axis as usize].max(diff * diff);
                if far_lower[0] + far_lower[1] + far_lower[2] > r2 {
                    return;
                }
            }
            self.visit_within::<BOUNDS>(far, q, r2, far_lower, visit);
        }
    }

    /// Two independent radius queries in one traversal. Each output retains
    /// exactly its own query's unsorted slot order, including boundary hits.
    pub fn within_two_slots_with_bounds(
        &self,
        query: &[f64; 3],
        radii: [f64; 2],
        out: [&mut Vec<usize>; 2],
        bounds: bool,
    ) {
        let [first, second] = out;
        first.clear();
        second.clear();
        // A negative squared radius disables a query in a pruned subtree.
        let squared = radii.map(|r| if r >= 0.0 { r * r } else { -1.0 });
        if squared.iter().all(|&r| r < 0.0) {
            return;
        }
        if bounds {
            self.search_within_two::<true>(0, query, squared, [0.0; 3], first, second);
        } else {
            self.search_within_two::<false>(0, query, squared, [0.0; 3], first, second);
        }
    }

    fn search_within_two<const BOUNDS: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        radii: [f64; 2],
        lower: [f64; 3],
        first: &mut Vec<usize>,
        second: &mut Vec<usize>,
    ) {
        // Once one radius has pruned this subtree, retain the single-query
        // traversal for the survivor instead of carrying a disabled query.
        if radii[0] < 0.0 {
            self.search_within::<BOUNDS, true>(node, q, radii[1], lower, second);
            return;
        }
        if radii[1] < 0.0 {
            self.search_within::<BOUNDS, true>(node, q, radii[0], lower, first);
            return;
        }
        let n = self.nodes[node];
        if n.axis == LEAF {
            for i in n.start as usize..n.end as usize {
                let distance = distance_sq(&self.points[i], q);
                if distance <= radii[0] {
                    first.push(i);
                }
                if distance <= radii[1] {
                    second.push(i);
                }
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_within_two::<BOUNDS>(near, q, radii, lower, first, second);
        let mut far_lower = lower;
        if BOUNDS {
            far_lower[n.axis as usize] = far_lower[n.axis as usize].max(diff * diff);
        }
        let bound = far_lower[0] + far_lower[1] + far_lower[2];
        let far_radii = radii.map(|r| {
            if diff * diff <= r && !(BOUNDS && bound > r) {
                r
            } else {
                -1.0
            }
        });
        if far_radii.iter().any(|&r| r >= 0.0) {
            self.search_within_two::<BOUNDS>(far, q, far_radii, far_lower, first, second);
        }
    }

    /// The `k` nearest points to `query` as `(index, squared distance)`,
    /// closest first. Intended for small `k` (e.g. normal estimation).
    pub fn nearest_k(&self, query: &[f64; 3], k: usize) -> Vec<(usize, f64)> {
        let mut found: Vec<(usize, f64)> = Vec::with_capacity(k + 1);
        if k > 0 {
            self.search_k::<false, false>(0, query, k, [0.0; 3], &mut found);
        }
        found
            .into_iter()
            .map(|(slot, d)| (self.indices[slot] as usize, d))
            .collect()
    }

    /// Retained vector capacities, excluding allocator overhead and temporary queries.
    pub fn storage_bytes(&self) -> usize {
        self.points.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.indices.capacity() * std::mem::size_of::<u32>()
            + self.nodes.capacity() * std::mem::size_of::<Node>()
    }

    /// Same hits and tie order, using one retained caller-owned query buffer.
    pub fn nearest_k_into(&self, query: &[f64; 3], k: usize, found: &mut Vec<(usize, f64)>) {
        self.nearest_k_into_with_simd(query, k, found, false);
    }

    /// Optional paired f64 leaf arithmetic, retaining scalar insertion/tie order.
    pub fn nearest_k_into_with_simd(
        &self,
        query: &[f64; 3],
        k: usize,
        found: &mut Vec<(usize, f64)>,
        simd: bool,
    ) {
        self.nearest_k_into_with_options(query, k, found, simd, false);
    }

    /// Combine ancestral split-plane bounds in x/y/z distance order.
    pub fn nearest_k_into_with_options(
        &self,
        query: &[f64; 3],
        k: usize,
        found: &mut Vec<(usize, f64)>,
        simd: bool,
        bounds: bool,
    ) {
        self.nearest_k_slots_into_with_options(query, k, found, simd, bounds);
        for (slot, _) in found {
            *slot = self.indices[*slot] as usize;
        }
    }

    /// Same traversal/distances, leaving internal tree slots in the scratch.
    /// Distance-only consumers need not gather original point indices.
    #[inline]
    pub(crate) fn nearest_k_slots_into_with_options(
        &self,
        query: &[f64; 3],
        k: usize,
        found: &mut Vec<(usize, f64)>,
        simd: bool,
        bounds: bool,
    ) {
        found.clear();
        found.reserve(k.min(self.len()));
        if k > 0 {
            match (simd, bounds) {
                (false, false) => self.search_k::<false, false>(0, query, k, [0.0; 3], found),
                (false, true) => self.search_k::<false, true>(0, query, k, [0.0; 3], found),
                (true, false) => self.search_k::<true, false>(0, query, k, [0.0; 3], found),
                (true, true) => self.search_k::<true, true>(0, query, k, [0.0; 3], found),
            }
        }
    }

    /// An arbitrary valid block containing k distinct points supplies a safe
    /// f64 upper radius. Invalid, oversized or nonfinite hints use the original
    /// scalar search. No candidate is inserted ahead of the original traversal.
    pub(crate) fn nearest_k_into_hinted(
        &self,
        query: &[f64; 3],
        k: usize,
        node: u32,
        hints: &KnnRadiusHints,
        found: &mut Vec<(usize, f64)>,
    ) {
        let upper = self.nodes.get(node as usize).and_then(|n| {
            let count = (n.end - n.start) as usize;
            if !hints.finite
                || !query.iter().all(|v| v.is_finite())
                || k == 0
                || k > 32
                || count < k
                || count > 64
            {
                return None;
            }
            let mut distances = [0.0; 64];
            for (d, p) in distances[..count]
                .iter_mut()
                .zip(&self.points[n.start as usize..n.end as usize])
            {
                *d = distance_sq(p, query);
                if !d.is_finite() {
                    return None;
                }
            }
            let (_, kth, _) = distances[..count].select_nth_unstable_by(k - 1, f64::total_cmp);
            Some(*kth)
        });
        let Some(upper) = upper else {
            self.nearest_k_into(query, k, found);
            return;
        };
        self.nearest_k_with_radius(query, k, upper, hints, found);
    }

    /// GPU slots are untrusted tree-order indices, not final neighbours. Any
    /// k distinct finite points bound the true kth distance from above. Keep
    /// candidate insertion out of the original traversal to preserve ties.
    pub(crate) fn nearest_k_into_candidates(
        &self,
        query: &[f64; 3],
        k: usize,
        slots: &[u32],
        hints: &KnnRadiusHints,
        found: &mut Vec<(usize, f64)>,
    ) {
        let upper = (|| {
            if k == 0
                || k > 32
                || slots.len() != k
                || !hints.finite
                || !query.iter().all(|v| v.is_finite())
            {
                return None;
            }
            let mut seen = 0u64;
            let mut upper = 0.0f64;
            for (i, &slot) in slots.iter().enumerate() {
                let bit = 1u64 << (slot % 64);
                if seen & bit != 0 && slots[..i].contains(&slot) {
                    return None;
                }
                seen |= bit;
                let d = distance_sq(self.points.get(slot as usize)?, query);
                if !d.is_finite() {
                    return None;
                }
                upper = upper.max(d);
            }
            Some(upper)
        })();
        if let Some(upper) = upper {
            found.clear();
            found.reserve(k);
            self.search_k_radius(0, query, k, upper, found);
            for (slot, _) in found {
                *slot = self.indices[*slot] as usize;
            }
        } else {
            self.nearest_k_into(query, k, found);
        }
    }

    /// The known upper radius can exclude a far subtree even before the
    /// original traversal has filled its hit list. No per-node box work.
    fn search_k_radius(
        &self,
        node: usize,
        q: &[f64; 3],
        k: usize,
        upper: f64,
        found: &mut Vec<(usize, f64)>,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            for (i, p) in self.points[n.start as usize..n.end as usize]
                .iter()
                .enumerate()
            {
                insert_knn(found, k, n.start as usize + i, distance_sq(p, q));
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_k_radius(near, q, k, upper, found);
        let plane = diff * diff;
        // Equality stays eligible while the original list is incomplete;
        // retain the original strict split test once the list is full.
        if plane <= upper && (found.len() < k || plane < found[found.len() - 1].1) {
            self.search_k_radius(far, q, k, upper, found);
        }
    }

    fn nearest_k_with_radius(
        &self,
        query: &[f64; 3],
        k: usize,
        upper: f64,
        hints: &KnnRadiusHints,
        found: &mut Vec<(usize, f64)>,
    ) {
        found.clear();
        found.reserve(k);
        self.search_k_hinted(0, query, k, upper, hints, found);
        for (slot, _) in found {
            *slot = self.indices[*slot] as usize;
        }
    }

    /// CPU comparison for the same rounded block walk as the GPU shader.
    pub(crate) fn rounded_knn_block(&self, query: &[f64; 3], origin: [f64; 3], k: usize) -> u32 {
        let q: [f32; 3] = std::array::from_fn(|a| (query[a] - origin[a]) as f32);
        let mut node = 0usize;
        for _ in 0..32 {
            let n = self.nodes[node];
            if n.axis == LEAF {
                break;
            }
            let left = self.nodes[node + 1];
            let right = self.nodes[n.right as usize];
            if ((left.end - left.start).min(right.end - right.start) as usize) < k {
                break;
            }
            let split = (n.split - origin[n.axis as usize]) as f32;
            node = if q[n.axis as usize] < split {
                node + 1
            } else {
                n.right as usize
            };
        }
        node as u32
    }

    fn search_k_hinted(
        &self,
        node: usize,
        q: &[f64; 3],
        k: usize,
        upper: f64,
        hints: &KnnRadiusHints,
        found: &mut Vec<(usize, f64)>,
    ) {
        let (lo, hi) = hints.bounds[node];
        let gaps: [f64; 3] = std::array::from_fn(|a| {
            if q[a] < lo[a] {
                lo[a] - q[a]
            } else if q[a] > hi[a] {
                q[a] - hi[a]
            } else {
                0.0
            }
        });
        let lower = gaps[0] * gaps[0] + gaps[1] * gaps[1] + gaps[2] * gaps[2];
        let limit = if found.len() == k {
            upper.min(found[k - 1].1)
        } else {
            upper
        };
        // Strict > retains equal-distance nodes. Original near/far order and
        // full-list split-plane comparisons below preserve historical ties.
        if lower > limit {
            return;
        }
        let n = self.nodes[node];
        if n.axis == LEAF {
            for (i, p) in self.points[n.start as usize..n.end as usize]
                .iter()
                .enumerate()
            {
                insert_knn(found, k, n.start as usize + i, distance_sq(p, q));
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_k_hinted(near, q, k, upper, hints, found);
        if found.len() < k || diff * diff < found[found.len() - 1].1 {
            self.search_k_hinted(far, q, k, upper, hints, found);
        }
    }

    fn search_k<const SIMD: bool, const BOUNDS: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        k: usize,
        lower: [f64; 3],
        found: &mut Vec<(usize, f64)>,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            let range = &self.points[n.start as usize..n.end as usize];
            if SIMD {
                let (pairs, tail) = range.as_chunks::<2>();
                for (i, pair) in pairs.iter().enumerate() {
                    let d = pair_distances(pair, q);
                    insert_knn(found, k, n.start as usize + i * 2, d[0]);
                    insert_knn(found, k, n.start as usize + i * 2 + 1, d[1]);
                }
                if let Some(p) = tail.first() {
                    insert_knn(found, k, n.end as usize - 1, distance_sq(p, q));
                }
            } else {
                for (i, p) in range.iter().enumerate() {
                    insert_knn(found, k, n.start as usize + i, distance_sq(p, q));
                }
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_k::<SIMD, BOUNDS>(near, q, k, lower, found);
        if found.len() < k || diff * diff < found[found.len() - 1].1 {
            let mut far_lower = lower;
            if BOUNDS {
                let axis = n.axis as usize;
                far_lower[axis] = far_lower[axis].max(diff * diff);
                // Same positive-term addition order as distance_sq. Strict >
                // retains equality/NaN visits; an incomplete list never prunes.
                let bound = far_lower[0] + far_lower[1] + far_lower[2];
                if found.len() == k && bound > found[found.len() - 1].1 {
                    return;
                }
            }
            self.search_k::<SIMD, BOUNDS>(far, q, k, far_lower, found);
        }
    }

    fn search<const SIMD: bool>(&self, node: usize, q: &[f64; 3], best: &mut Nearest) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            if SIMD {
                scan_nearest_leaf(&self.points, n.start, n.end, q, best);
            } else {
                scan_candidate_leaf(&self.points, n.start, n.end, q, best, false);
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search::<SIMD>(near, q, best);
        if diff * diff < best.distance_sq {
            self.search::<SIMD>(far, q, best);
        }
    }

    fn search_hinted_context<const SIMD: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        state: &mut BoundedNearest,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            if SIMD {
                scan_nearest_leaf(&self.points, n.start, n.end, q, &mut state.best);
            } else {
                scan_candidate_leaf(&self.points, n.start, n.end, q, &mut state.best, false);
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_hinted_context::<SIMD>(near, q, state);
        // Retain equality visits at the hint radius. It never becomes the
        // incumbent, so the original seed and traversal choose all ties.
        if diff * diff < state.best.distance_sq && diff * diff <= state.upper {
            self.search_hinted_context::<SIMD>(far, q, state);
        }
    }

    fn search_hinted<const SIMD: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        best: &mut Nearest,
        upper: f64,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            if SIMD {
                scan_nearest_leaf(&self.points, n.start, n.end, q, best);
            } else {
                scan_candidate_leaf(&self.points, n.start, n.end, q, best, false);
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_hinted::<SIMD>(near, q, best, upper);
        if diff * diff < best.distance_sq && diff * diff <= upper {
            self.search_hinted::<SIMD>(far, q, best, upper);
        }
    }
}

#[inline]
fn insert_knn(found: &mut Vec<(usize, f64)>, k: usize, slot: usize, d: f64) {
    if found.len() < k || d < found[found.len() - 1].1 {
        // Sorted insert after equal distances; a full list drops
        // its last entry. Copy each displaced hit once, then write
        // the new hit once at its final position. Comparisons and
        // insertion after equal distances keep the historical order.
        if found.len() < k {
            found.push((slot, d));
        }
        let mut j = found.len() - 1;
        while j > 0 && found[j - 1].1 > d {
            found[j] = found[j - 1];
            j -= 1;
        }
        found[j] = (slot, d);
    }
}

#[inline]
fn pair_distances(pair: &[[f64; 3]; 2], q: &[f64; 3]) -> [f64; 2] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::*;
        let ptr = pair.as_ptr().cast::<f64>();
        // SAFETY: three unaligned loads cover exactly this pair's six f64 values.
        let (xy, zx, yz) = unsafe {
            (
                v128_load(ptr.cast()),
                v128_load(ptr.add(2).cast()),
                v128_load(ptr.add(4).cast()),
            )
        };
        let dx = f64x2_sub(i64x2_shuffle::<0, 3>(xy, zx), f64x2_splat(q[0]));
        let dy = f64x2_sub(i64x2_shuffle::<1, 2>(xy, yz), f64x2_splat(q[1]));
        let dz = f64x2_sub(i64x2_shuffle::<0, 3>(zx, yz), f64x2_splat(q[2]));
        let d = f64x2_add(
            f64x2_add(f64x2_mul(dx, dx), f64x2_mul(dy, dy)),
            f64x2_mul(dz, dz),
        );
        let result = [f64x2_extract_lane::<0>(d), f64x2_extract_lane::<1>(d)];
        // Keep scalar NaN propagation as well as neighbour ordering.
        if result.iter().all(|v| !v.is_nan()) {
            return result;
        }
    }
    [distance_sq(&pair[0], q), distance_sq(&pair[1], q)]
}

#[inline(always)]
fn distance_sq(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    let (dx, dy, dz) = (p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    dx * dx + dy * dy + dz * dz
}

/// Build the subtree for `items` (which starts at `offset` in the full
/// array) and return its node index.
fn build(items: &mut [([f64; 3], u32)], offset: usize, nodes: &mut Vec<Node>) -> usize {
    let id = nodes.len();
    nodes.push(Node {
        start: offset as u32,
        end: (offset + items.len()) as u32,
        split: 0.0,
        axis: LEAF,
        right: 0,
    });
    if items.len() <= LEAF_SIZE {
        return id;
    }
    let (lo, hi) = crate::distance::bounds(items.iter().map(|(p, _)| p));
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap();
    if hi[axis] - lo[axis] <= 0.0 {
        return id; // all points coincide
    }
    let mid = items.len() / 2;
    items.select_nth_unstable_by(mid, |a, b| a.0[axis].total_cmp(&b.0[axis]));
    let split = items[mid].0[axis];
    let (left, right) = items.split_at_mut(mid);
    build(left, offset, nodes);
    let right_id = build(right, offset + mid, nodes);
    let node = &mut nodes[id];
    node.axis = axis as u8;
    node.split = split;
    node.right = right_id as u32;
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_ancestral_bounds_keep_hit_bits_and_order() {
        for count in [1, 17, 257, 2049] {
            for mode in 0..7 {
                let mut points = pseudo_random(count, 73);
                for (i, p) in points.iter_mut().enumerate() {
                    *p = match mode {
                        1 => [p[0] + 368000.0, p[1] + 3955000.0, p[2] + 40.0],
                        2 => p.map(|v| v * 1e-160),
                        3 => p.map(|v| v * 1e155),
                        4 => [1.0, -0.0, 0.0],
                        5 => [(i % 13) as f64, (i / 13) as f64, 0.0],
                        6 if i % 17 == 0 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
                        _ => *p,
                    };
                }
                let tree = KdTree::new(&points).unwrap();
                let queries: Vec<_> = points
                    .iter()
                    .step_by((count / 17).max(1))
                    .copied()
                    .chain([
                        [0.0; 3],
                        [f64::NAN, 0.0, 0.0],
                        [f64::INFINITY, -0.0, 1.0],
                        [1e155; 3],
                    ])
                    .collect();
                for k in [0, 1, 3, 12, 24, 64, count + 2] {
                    for q in &queries {
                        for simd in [false, true] {
                            let (mut old, mut bounded) = (Vec::new(), Vec::new());
                            tree.nearest_k_into_with_options(q, k, &mut old, simd, false);
                            tree.nearest_k_into_with_options(q, k, &mut bounded, simd, true);
                            let bits = |v: &[(usize, f64)]| {
                                v.iter().map(|&(i, d)| (i, d.to_bits())).collect::<Vec<_>>()
                            };
                            assert_eq!(
                                bits(&old),
                                bits(&bounded),
                                "count={count}, mode={mode}, k={k}, q={q:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn knn_shift_preserves_swap_insert_bits_and_tie_order() {
        fn swapped(found: &mut Vec<(usize, f64)>, k: usize, slot: usize, d: f64) {
            if found.len() < k || d < found[found.len() - 1].1 {
                if found.len() < k {
                    found.push((slot, d));
                } else {
                    found[k - 1] = (slot, d);
                }
                let mut j = found.len() - 1;
                while j > 0 && found[j - 1].1 > d {
                    found.swap(j - 1, j);
                    j -= 1;
                }
            }
        }
        let specials = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff0_0000_0000_0123),
            f64::from_bits(0xfff8_0000_0000_0456),
            f64::MIN_POSITIVE,
            f64::MAX,
        ];
        for k in [1, 3, 12, 24, 64, 128] {
            for mode in 0..5 {
                let (mut expected, mut actual) = (Vec::new(), Vec::new());
                let mut random = 17u64;
                for i in 0..1024 {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    let d = match mode {
                        0 => i as f64,
                        1 => (1024 - i) as f64,
                        2 => (i % 7) as f64,
                        3 => (random % 1024) as f64 / 13.0,
                        _ => specials[(random as usize) % specials.len()],
                    };
                    swapped(&mut expected, k, i, d);
                    insert_knn(&mut actual, k, i, d);
                    let bits = |v: &[(usize, f64)]| {
                        v.iter()
                            .map(|&(slot, d)| (slot, d.to_bits()))
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "k={k}, mode={mode}, step={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn distance_only_queries_keep_tree_slot_order_and_original_distance_bits() {
        for count in [1, 17, 257] {
            for mode in 0..6 {
                let points: Vec<_> = (0..count)
                    .map(|i| {
                        let p = [(i % 11) as f64, ((i / 11) % 7) as f64, (i % 3) as f64];
                        match mode {
                            1 => p.map(|v| v + 3955000.0),
                            2 => [1.0, -0.0, 0.0],
                            3 => p.map(|v| v * 1e155),
                            4 => p.map(|v| v * 1e-160),
                            5 if i == 0 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
                            _ => p,
                        }
                    })
                    .collect();
                let tree = KdTree::new(&points).unwrap();
                let bytes = tree.storage_bytes();
                let mut hits = Vec::new();
                for q in points.iter().step_by(17).copied().chain([
                    [0.5; 3],
                    [-0.0; 3],
                    [f64::NAN; 3],
                    [f64::INFINITY; 3],
                ]) {
                    for k in [0, 1, 3, 12, 24, 64, count + 2] {
                        let expected: Vec<_> = tree
                            .nearest_k(&q, k)
                            .into_iter()
                            .map(|(i, d)| (i, d.to_bits()))
                            .collect();
                        for simd in [false, true] {
                            for bounds in [false, true] {
                                tree.nearest_k_slots_into_with_options(
                                    &q, k, &mut hits, simd, bounds,
                                );
                                let actual: Vec<_> = hits
                                    .iter()
                                    .map(|&(slot, d)| (tree.indices[slot] as usize, d.to_bits()))
                                    .collect();
                                assert_eq!(expected, actual);
                                assert_eq!(tree.storage_bytes(), bytes);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exact_snapshots_preserve_slots_bits_certificates_and_owned_data() {
        for count in [1, 15, 16, 17, 31, 33, 257, 4097] {
            for coincident in [false, true] {
                let points: Vec<_> = (0..count)
                    .map(|i| {
                        if coincident {
                            [6_378_137.000_001, -0.0, 1.0]
                        } else {
                            [
                                6_378_137.0 + (i % 47) as f64 * 0.000_001,
                                (i % 23) as f64 * 0.1,
                                if i % 2 == 0 { -0.0 } else { 0.0 },
                            ]
                        }
                    })
                    .collect();
                let tree = CandidateKdTree::new(&points).unwrap();
                let mut bytes = tree.snapshot();
                let restored = CandidateKdTree::from_snapshot(&bytes, count).unwrap();
                assert_eq!(restored.snapshot(), bytes);
                bytes.fill(0);
                for (p, q) in points.iter().zip(restored.original_points()) {
                    assert_eq!(p.map(f64::to_bits), q.map(f64::to_bits));
                }
                assert_eq!(
                    restored.original_point(0).unwrap().map(f64::to_bits),
                    points[0].map(f64::to_bits)
                );
                assert_eq!(restored.original_point(count), None);
                for i in 0..37 {
                    let q = [
                        points[i % count][0] + 0.000_003,
                        (i % 11) as f64 * 0.09,
                        0.01,
                    ];
                    for slot in [0, count as u32 - 1, count as u32 / 2, u32::MAX] {
                        for bounds in [false, true] {
                            for simd in [false, true] {
                                let (a, ac) = tree.nearest_with_options(&q, slot, bounds, simd);
                                let (b, bc) = restored.nearest_with_options(&q, slot, bounds, simd);
                                assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
                                assert_eq!((a.index, a.slot, ac), (b.index, b.slot, bc));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exact_snapshots_preserve_nonfinite_payloads_and_full_search_behavior() {
        let points: Vec<_> = (0..65)
            .map(|i| {
                [
                    i as f64,
                    match i % 5 {
                        0 => f64::from_bits(0x7ff8_0000_0000_0123),
                        1 => f64::NEG_INFINITY,
                        2 => f64::INFINITY,
                        3 => -0.0,
                        _ => 0.0,
                    },
                    (i % 3) as f64,
                ]
            })
            .collect();
        let tree = CandidateKdTree::new(&points).unwrap();
        let bytes = tree.snapshot();
        let restored = CandidateKdTree::from_snapshot(&bytes, 65).unwrap();
        assert_eq!(restored.snapshot(), bytes);
        for p in &points {
            for slot in [0, 64, u32::MAX] {
                let (a, ac) = tree.nearest_bounded(p, slot);
                let (b, bc) = restored.nearest_bounded(p, slot);
                assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
                assert_eq!((a.slot, ac), (b.slot, bc));
                assert!(!bc);
            }
        }
    }

    #[test]
    fn reused_validated_extrema_preserve_hits_certificates_and_snapshot_bits() {
        for count in [1, 16, 17, 65, 257, 2049] {
            for mode in 0..5 {
                let mut points = pseudo_random(count, 97);
                for (i, p) in points.iter_mut().enumerate() {
                    match mode {
                        0 => *p = [368_000.0 + p[0], 3_955_000.0 + p[1], p[2] * 1e-6],
                        1 => *p = [if i % 2 == 0 { -0.0 } else { 0.0 }; 3],
                        2 => *p = p.map(|v| v * f64::from_bits(1)),
                        3 => *p = p.map(|v| v * 1e150),
                        _ => p[2] = if i % 2 == 0 { -0.0 } else { 0.0 },
                    }
                }
                let tree = CandidateKdTree::new(&points).unwrap();
                let bytes = tree.snapshot();
                let old = CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, count, false)
                    .unwrap();
                let reused =
                    CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, count, true).unwrap();
                assert_eq!(reused.snapshot(), bytes);
                assert_eq!(reused.bounds.len(), reused.tree.nodes.len());
                for q in points.iter().step_by(17).copied().chain([
                    [0.0; 3],
                    [-0.0; 3],
                    [f64::from_bits(1); 3],
                    [f64::MAX; 3],
                    [f64::NAN; 3],
                    [f64::INFINITY; 3],
                ]) {
                    for slot in [0, count as u32 / 2, count as u32 - 1, u32::MAX] {
                        for bounds in [false, true] {
                            for simd in [false, true] {
                                let (a, ac) = old.nearest_with_options(&q, slot, bounds, simd);
                                let (b, bc) = reused.nearest_with_options(&q, slot, bounds, simd);
                                assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
                                assert_eq!((a.index, a.slot, ac), (b.index, b.slot, bc));
                            }
                        }
                    }
                }
            }
        }
        let points = [
            [f64::NAN, -0.0, 1.0],
            [f64::INFINITY; 3],
            [-0.0; 3],
            [f64::from_bits(0xfff8_0000_0000_0123), 1.0, 2.0],
            [f64::NEG_INFINITY; 3],
        ];
        let bytes = CandidateKdTree::new(&points).unwrap().snapshot();
        let old =
            CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, points.len(), false).unwrap();
        let reused =
            CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, points.len(), true).unwrap();
        assert!(!reused.finite);
        assert_eq!(reused.snapshot(), bytes);
        for q in points {
            let (a, ac) = old.nearest_bounded(&q, 0);
            let (b, bc) = reused.nearest_bounded(&q, 0);
            assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
            assert_eq!((a.slot, ac), (b.slot, bc));
        }
    }

    #[test]
    fn exact_snapshots_reject_invalid_headers_indices_topology_and_partitions() {
        for reuse_bounds in [false, true] {
            let tree =
                CandidateKdTree::new(&(0..65).map(|i| [i as f64, 0.0, 0.0]).collect::<Vec<_>>())
                    .unwrap();
            let bytes = tree.snapshot();
            let point_end = 16 + 65 * 24;
            let node_start = point_end + 65 * 4;
            assert!(
                CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, 64, reuse_bounds)
                    .is_err()
            );
            for len in [0, 3, 15, 16, point_end, bytes.len() - 1] {
                assert!(
                    CandidateKdTree::from_snapshot_with_reused_bounds(
                        &bytes[..len],
                        65,
                        reuse_bounds
                    )
                    .is_err()
                );
            }
            let mut padded = bytes.clone();
            let node_count = read_u32(&bytes[12..16]) as usize;
            padded.resize(bytes.len() + (129 - node_count) * 24, 0);
            padded[12..16].copy_from_slice(&129_u32.to_le_bytes());
            assert!(
                CandidateKdTree::from_snapshot_with_reused_bounds(&padded, 65, reuse_bounds)
                    .is_err()
            );
            let mut extra = bytes.clone();
            extra.push(0);
            assert!(
                CandidateKdTree::from_snapshot_with_reused_bounds(&extra, 65, reuse_bounds)
                    .is_err()
            );
            for (at, value) in [
                (0, 0),
                (4, 2),
                (8, 0),
                (8, u32::MAX),
                (12, 0),
                (12, u32::MAX),
                (point_end, 65),
                (point_end + 4, read_u32(&bytes[point_end..point_end + 4])),
                (node_start, 1),
                (node_start + 4, 64),
                (node_start + 8, 0),
                (node_start + 8, u32::MAX),
                (node_start + 12, 3),
                (node_start + 12, 256),
                (node_start + 24, 1),
                (node_start + 24 + 4, 65),
            ] {
                let mut corrupt = bytes.clone();
                corrupt[at..at + 4].copy_from_slice(&value.to_le_bytes());
                assert!(
                    CandidateKdTree::from_snapshot_with_reused_bounds(&corrupt, 65, reuse_bounds)
                        .is_err(),
                    "word at {at}: {value}"
                );
            }
            let mut split = bytes.clone();
            split[node_start + 16..node_start + 24].copy_from_slice(&(-1.0_f64).to_le_bytes());
            assert!(
                CandidateKdTree::from_snapshot_with_reused_bounds(&split, 65, reuse_bounds)
                    .is_err()
            );
            let mut partition = bytes.clone();
            partition[16..24].copy_from_slice(&1000.0_f64.to_le_bytes());
            assert!(
                CandidateKdTree::from_snapshot_with_reused_bounds(&partition, 65, reuse_bounds)
                    .is_err()
            );
            let leaf = tree.tree.nodes.iter().position(|n| n.axis == LEAF).unwrap();
            for (offset, value) in [(8, 1_u32), (16, 1)] {
                let mut corrupt = bytes.clone();
                let at = node_start + leaf * 24 + offset;
                corrupt[at..at + 4].copy_from_slice(&value.to_le_bytes());
                assert!(
                    CandidateKdTree::from_snapshot_with_reused_bounds(&corrupt, 65, reuse_bounds)
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn radius_slots_and_import_preserve_original_hit_coordinates_in_order() {
        for count in [1, 17, 65, 2057] {
            let mut points = pseudo_random(count, 731);
            points[0] = [-0.0, 0.0, f64::MIN_POSITIVE];
            if count > 17 {
                points[17] = [f64::from_bits(0x7ff8_0000_0000_0123), f64::INFINITY, 0.0];
            }
            let tree = KdTree::new(&points).unwrap();
            let restored = KdTree::from_snapshot(&tree.snapshot(), count).unwrap();
            for query in [[0.0; 3], [0.3, -0.1, 0.2], [f64::NAN; 3]] {
                for radius in [-1.0, 0.0, 0.5, f64::INFINITY, f64::NAN] {
                    for bounds in [false, true] {
                        let mut original = Vec::new();
                        tree.within_with_bounds(&query, radius, &mut original, bounds);
                        let expected: Vec<_> = original
                            .iter()
                            .map(|&i| points[i].map(f64::to_bits))
                            .collect();
                        for indexed in [&tree, &restored] {
                            let mut slots = Vec::new();
                            indexed.within_slots_with_bounds(&query, radius, &mut slots, bounds);
                            let actual: Vec<_> = slots
                                .iter()
                                .map(|&i| indexed.tree_points()[i].map(f64::to_bits))
                                .collect();
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
    }

    fn brute(points: &[[f64; 3]], q: &[f64; 3]) -> f64 {
        points
            .iter()
            .map(|p| (0..3).map(|a| (p[a] - q[a]).powi(2)).sum::<f64>())
            .fold(f64::INFINITY, f64::min)
    }

    fn pseudo_random(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| [next() * 10.0, next() * 10.0, (next() * 4.0).floor()])
            .collect()
    }

    #[test]
    fn matches_brute_force() {
        let points = pseudo_random(2000, 7);
        let tree = KdTree::new(&points).unwrap();
        for q in pseudo_random(300, 11) {
            let hit = tree.nearest(&q, None);
            assert_eq!(hit.distance_sq, brute(&points, &q));
            let p = points[hit.index];
            let d: f64 = (0..3).map(|a| (p[a] - q[a]).powi(2)).sum();
            assert_eq!(d, hit.distance_sq);
        }
    }

    #[test]
    fn any_guess_yields_the_nearest() {
        let points = pseudo_random(500, 3);
        let tree = KdTree::new(&points).unwrap();
        let far = tree.nearest(&[100.0, 100.0, 100.0], None);
        for q in pseudo_random(50, 5) {
            assert_eq!(tree.nearest(&q, Some(far)).distance_sq, brute(&points, &q));
        }
    }

    #[test]
    fn paired_nearest_keeps_scalar_hits_and_guess_order() {
        for count in [1, 2, 3, 15, 16, 17, 31, 33, 257, 2000] {
            let mut points = pseudo_random(count, 53);
            if count > 2 {
                points[1] = points[0];
                points[2] = points[0].map(|v| v + 1e-9);
            }
            let tree = KdTree::new(&points).unwrap();
            let mut scalar = None;
            let mut vector = None;
            for q in points.iter().copied().chain(pseudo_random(100, 57)) {
                let a = tree.nearest_with_simd(&q, scalar, false);
                let b = tree.nearest_with_simd(&q, vector, true);
                assert_eq!(a.index, b.index);
                assert_eq!(a.slot, b.slot);
                assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
                assert_eq!(b.distance_sq, brute(&points, &q));
                scalar = Some(a);
                vector = Some(b);
            }
        }
    }

    #[test]
    fn nearest_radius_hints_keep_seed_ties_distance_bits_and_fallbacks() {
        let mut tightened = 0;
        for count in [1, 17, 33, 257, 2000] {
            for scale in [1.0, 1e-160, 1e155] {
                let mut points = pseudo_random(count, 53);
                for p in &mut points {
                    *p = p.map(|v| v * scale);
                }
                if count > 2 {
                    points[1] = points[0];
                    points[2] = [0.0, -0.0, 0.0];
                }
                let tree = KdTree::new(&points).unwrap();
                let hints = NearestRadiusHints::new(&tree);
                let far = tree.nearest(&[100.0 * scale; 3], None);
                for q in points
                    .iter()
                    .take(19)
                    .copied()
                    .chain(
                        pseudo_random(19, 57)
                            .into_iter()
                            .map(|p| p.map(|v| v * scale)),
                    )
                    .chain([
                        [f64::NAN, 0.0, 0.0],
                        [f64::INFINITY, -0.0, 0.0],
                        [f64::MAX; 3],
                    ])
                {
                    for guess in [None, Some(far)] {
                        let expected = tree.nearest(&q, guess);
                        for candidate in [expected.slot, 0, count as u32 - 1, u32::MAX] {
                            for simd in [false, true] {
                                let original = tree.nearest_with_radius_layout::<false>(
                                    &q, guess, candidate, &hints, simd,
                                );
                                let shared = tree.nearest_with_radius_layout::<true>(
                                    &q, guess, candidate, &hints, simd,
                                );
                                assert_eq!(
                                    (
                                        original.0.index,
                                        original.0.slot,
                                        original.0.distance_sq.to_bits(),
                                        original.1
                                    ),
                                    (
                                        shared.0.index,
                                        shared.0.slot,
                                        shared.0.distance_sq.to_bits(),
                                        shared.1
                                    )
                                );
                                let (hit, tighter) = shared;
                                assert_eq!(
                                    (hit.index, hit.slot, hit.distance_sq.to_bits()),
                                    (
                                        expected.index,
                                        expected.slot,
                                        expected.distance_sq.to_bits()
                                    )
                                );
                                tightened += usize::from(tighter);
                            }
                        }
                    }
                }
            }
        }
        assert!(tightened > 0);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut points = pseudo_random(65, 11);
            points[2][0] = bad;
            let tree = KdTree::new(&points).unwrap();
            let hints = NearestRadiusHints::new(&tree);
            for q in pseudo_random(13, 17) {
                let expected = tree.nearest(&q, None);
                let (hit, tighter) =
                    tree.nearest_with_radius_hint(&q, None, expected.slot, &hints, false);
                assert!(!tighter);
                assert_eq!(
                    (hit.index, hit.slot, hit.distance_sq.to_bits()),
                    (
                        expected.index,
                        expected.slot,
                        expected.distance_sq.to_bits()
                    )
                );
            }
        }
    }

    #[test]
    fn paired_leaf_scan_preserves_ties_nonfinite_values_and_odd_tails() {
        let points = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 1.0, 0.0],
            [1e300, 0.0, 0.0],
            [0.0, 0.0, -0.0],
        ];
        for end in 1..=points.len() as u32 {
            for q in [
                [0.0; 3],
                [2.0; 3],
                [1e-160; 3],
                [1e300; 3],
                [f64::NAN; 3],
                [f64::INFINITY; 3],
            ] {
                for seed in [0.0, 1.0, f64::INFINITY, f64::NAN] {
                    let mut a = Nearest {
                        index: 0,
                        slot: u32::MAX,
                        distance_sq: seed,
                    };
                    let mut b = a;
                    scan_candidate_leaf(&points, 0, end, &q, &mut a, false);
                    scan_nearest_leaf(&points, 0, end, &q, &mut b);
                    assert_eq!(a.slot, b.slot);
                    assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
                }
            }
        }
    }

    #[test]
    fn lane_minima_select_first_improvement_and_retain_tied_or_nan_seeds() {
        for (seed, distances, slots, expected_distance, expected_slot) in [
            (2.0, [1.0, 1.0], [4, 3], 1.0, 3),
            (2.0, [1.0, 1.0], [2, 3], 1.0, 2),
            (1.0, [1.0, 1.0], [2, 3], 1.0, 99),
            (2.0, [1.0, 0.5], [2, 3], 0.5, 3),
            (f64::INFINITY, [1.0, f64::INFINITY], [2, 3], 1.0, 2),
            (f64::INFINITY, [f64::INFINITY; 2], [2, 3], f64::INFINITY, 99),
            (f64::NAN, [0.0, 1.0], [2, 3], f64::NAN, 99),
            (1.0, [f64::NAN, 0.5], [2, 3], 0.5, 3),
        ] {
            let mut best = Nearest {
                index: 0,
                slot: 99,
                distance_sq: seed,
            };
            select_leaf_minimum(&mut best, distances, slots);
            assert_eq!(best.slot, expected_slot);
            assert_eq!(best.distance_sq.to_bits(), expected_distance.to_bits());
        }
    }

    #[test]
    fn nearest_k_matches_brute_force() {
        let points = pseudo_random(1500, 13);
        let tree = KdTree::new(&points).unwrap();
        for q in pseudo_random(40, 17) {
            let got = tree.nearest_k(&q, 7);
            let mut all: Vec<f64> = points
                .iter()
                .map(|p| (0..3).map(|a| (p[a] - q[a]).powi(2)).sum())
                .collect();
            all.sort_by(f64::total_cmp);
            assert_eq!(
                got.iter().map(|g| g.1).collect::<Vec<_>>(),
                all[..7].to_vec()
            );
        }
        assert_eq!(tree.nearest_k(&[0.0; 3], 5000).len(), 1500);
    }

    #[test]
    fn within_matches_brute_force() {
        let points = pseudo_random(2000, 21);
        let tree = KdTree::new(&points).unwrap();
        let mut found = Vec::new();
        for q in pseudo_random(30, 23) {
            tree.within(&q, 1.3, &mut found);
            found.sort_unstable();
            let expected: Vec<usize> = (0..points.len())
                .filter(|&i| {
                    (0..3).map(|a| (points[i][a] - q[a]).powi(2)).sum::<f64>() <= 1.3 * 1.3
                })
                .collect();
            assert_eq!(found, expected);
        }
    }

    #[test]
    fn combined_radius_bounds_preserve_unsorted_hit_order_and_boundaries() {
        let mut samples = pseudo_random(2057, 31);
        samples.extend([[0.0; 3], [-0.0; 3], [3.0, 4.0, 0.0], [-3.0, -4.0, 0.0]]);
        for scale in [1e-150, 1.0, 1e150] {
            for offset in [0.0, 368_000.0] {
                let points: Vec<_> = samples
                    .iter()
                    .map(|p| p.map(|v| offset + v * scale))
                    .collect();
                let tree = KdTree::new(&points).unwrap();
                let mut original = vec![usize::MAX];
                let mut bounded = vec![usize::MAX];
                for q in points
                    .iter()
                    .step_by(23)
                    .chain(std::iter::once(&[offset; 3]))
                {
                    let boundary = distance_sq(&points[37], q).sqrt();
                    for radius in [
                        -1.0,
                        0.0,
                        scale * 0.01,
                        scale,
                        scale * 5.0,
                        boundary,
                        boundary.next_down(),
                        boundary.next_up(),
                        f64::INFINITY,
                        f64::NAN,
                    ] {
                        tree.within(q, radius, &mut original);
                        tree.within_with_bounds(q, radius, &mut bounded, true);
                        assert_eq!(
                            bounded, original,
                            "scale {scale}, offset {offset}, radius {radius}"
                        );
                    }
                }
            }
        }
        // Preserve the old path's behavior for nonfinite coordinates/queries,
        // squared-radius overflow and subnormal underflow as well.
        let points: Vec<_> = samples
            .into_iter()
            .chain([
                [f64::NAN, 1.0, 2.0],
                [f64::INFINITY, 0.0, 0.0],
                [f64::NEG_INFINITY, 0.0, 0.0],
                [f64::MAX; 3],
                [f64::from_bits(1); 3],
            ])
            .collect();
        let tree = KdTree::new(&points).unwrap();
        for q in points.iter().step_by(13).chain(points.iter().rev().take(5)) {
            for radius in [
                0.0,
                f64::from_bits(1),
                1.0,
                5.0,
                1e200,
                f64::INFINITY,
                f64::NAN,
            ] {
                let mut original = Vec::new();
                let mut bounded = Vec::new();
                tree.within(q, radius, &mut original);
                tree.within_with_bounds(q, radius, &mut bounded, true);
                assert_eq!(bounded, original);
            }
        }
    }

    #[test]
    fn handles_duplicates_and_tiny_inputs() {
        let points = vec![[1.0, 2.0, 3.0]; 100];
        let tree = KdTree::new(&points).unwrap();
        assert_eq!(tree.nearest(&[1.0, 2.0, 4.0], None).distance_sq, 1.0);
        let single = KdTree::new(&[[0.0; 3]]).unwrap();
        assert_eq!(single.nearest(&[3.0, 4.0, 0.0], None).distance_sq, 25.0);
        assert!(KdTree::new(&[]).is_none());
    }

    #[test]
    fn radius_visitor_preserves_hit_order_and_leaf_arithmetic() {
        for count in [1, 16, 17, 257, 2049] {
            for scale in [1e-150, 1.0, 1e150] {
                let mut points = pseudo_random(count, 83);
                for p in &mut points {
                    *p = p.map(|v| v * scale);
                }
                points.extend([
                    [0.0; 3],
                    [-0.0; 3],
                    [3.0 * scale, 4.0 * scale, 0.0],
                    [f64::NAN, 1.0, 2.0],
                    [f64::INFINITY, 0.0, 0.0],
                    [f64::MAX; 3],
                    [f64::from_bits(1); 3],
                ]);
                let tree = KdTree::new(&points).unwrap();
                for q in points.iter().step_by(71).chain(points.iter().rev().take(7)) {
                    let boundary = distance_sq(&points[0], q).sqrt();
                    for radius in [
                        -1.0,
                        -0.0,
                        f64::from_bits(1),
                        scale * 0.1,
                        scale,
                        boundary.next_down(),
                        boundary,
                        boundary.next_up(),
                        1e200,
                        f64::INFINITY,
                        f64::NAN,
                    ] {
                        for bounds in [false, true] {
                            let mut expected = Vec::new();
                            tree.within_slots_with_bounds(q, radius, &mut expected, bounds);
                            let mut actual = Vec::new();
                            tree.visit_within_with_bounds(q, radius, bounds, &mut |slot, v, d| {
                                let p = tree.tree_points()[slot];
                                assert_eq!(
                                    v.map(f64::to_bits),
                                    [p[0] - q[0], p[1] - q[1], p[2] - q[2]].map(f64::to_bits)
                                );
                                assert_eq!(d.to_bits(), distance_sq(&p, q).to_bits());
                                actual.push(slot);
                            });
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shared_radius_search_preserves_each_independent_slot_sequence() {
        for count in [1, 16, 17, 257, 2049] {
            for scale in [1e-150, 1.0, 1e150] {
                let mut points = pseudo_random(count, 83);
                for p in &mut points {
                    *p = p.map(|v| v * scale);
                }
                points.extend([
                    [0.0; 3],
                    [-0.0; 3],
                    [3.0 * scale, 4.0 * scale, 0.0],
                    [f64::NAN, 1.0, 2.0],
                    [f64::INFINITY, 0.0, 0.0],
                    [f64::NEG_INFINITY, 0.0, 0.0],
                    [f64::MAX; 3],
                    [f64::from_bits(1); 3],
                ]);
                let tree = KdTree::new(&points).unwrap();
                for q in points.iter().step_by(37).chain(points.iter().rev().take(8)) {
                    let boundary = distance_sq(&points[0], q).sqrt();
                    let radii = [
                        -1.0,
                        -0.0,
                        f64::from_bits(1),
                        scale * 0.1,
                        scale,
                        boundary.next_down(),
                        boundary,
                        boundary.next_up(),
                        1e200,
                        f64::INFINITY,
                        f64::NAN,
                    ];
                    for bounds in [false, true] {
                        for (i, &radius) in radii.iter().enumerate() {
                            for other in [radius, radii[(i + 3) % radii.len()]] {
                                let (mut a, mut b, mut combined_a, mut combined_b) =
                                    (Vec::new(), Vec::new(), vec![usize::MAX], vec![usize::MAX]);
                                tree.within_slots_with_bounds(q, radius, &mut a, bounds);
                                tree.within_slots_with_bounds(q, other, &mut b, bounds);
                                tree.within_two_slots_with_bounds(
                                    q,
                                    [radius, other],
                                    [&mut combined_a, &mut combined_b],
                                    bounds,
                                );
                                assert_eq!(combined_a, a);
                                assert_eq!(combined_b, b);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn untrusted_candidates_preserve_f64_distances() {
        let points: Vec<_> = pseudo_random(2000, 17)
            .into_iter()
            .map(|p| [p[0] + 368_000.0, p[1] + 3_955_000.0, p[2] * 1e-6])
            .collect();
        let tree = KdTree::new(&points).unwrap();
        for q in points.iter().step_by(13) {
            let q = [q[0] + 1e-4, q[1] - 2e-4, q[2] + 1e-7];
            let expected = tree.nearest(&q, None).distance_sq;
            for seed in [0, 19, 1999, 2000, u32::MAX] {
                assert_eq!(tree.nearest_seeded(&q, seed).distance_sq, expected);
            }
        }
        let (packed, nodes) = tree.candidate_buffers([368_000.0, 3_955_000.0, 0.0]);
        assert_eq!(packed.len(), 4 * points.len());
        assert_eq!(nodes.len(), 5 * tree.nodes.len());
        for (i, p) in tree.points.iter().enumerate() {
            assert_eq!(packed[4 * i], (p[0] - 368_000.0) as f32);
        }
    }

    #[test]
    fn certified_candidates_match_brute_force_with_any_seed() {
        let points = pseudo_random(2049, 73);
        let tree = CandidateKdTree::new(&points).unwrap();
        let mut certified = 0;
        let mut full = 0;
        for q in pseudo_random(300, 79) {
            let expected = brute(&points, &q);
            let correct = tree.tree.nearest(&q, None).slot;
            for seed in [0, 511, 2048, 2049, u32::MAX, correct] {
                let (hit, fast) = tree.nearest(&q, seed);
                assert_eq!(hit.distance_sq, expected);
                assert_eq!(distance_sq(&points[hit.index], &q), expected);
                let (bounded, _) = tree.nearest_bounded(&q, seed);
                assert_eq!(bounded.distance_sq, expected);
                assert_eq!(distance_sq(&points[bounded.index], &q), expected);
                certified += usize::from(fast);
                full += usize::from(!fast);
            }
        }
        assert!(certified > 0);
        assert!(full > 0);
    }

    #[test]
    fn certificates_preserve_boundaries_large_coordinates_and_duplicates() {
        let mut points: Vec<_> = (0..4096)
            .map(|i| {
                [
                    368_000.0 + (i % 64) as f64 * 0.1,
                    3_955_000.0 + (i / 64) as f64 * 0.1,
                    40.0,
                ]
            })
            .collect();
        points.extend_from_within(..17);
        let tree = CandidateKdTree::new(&points).unwrap();
        for leaf in &tree.leaves {
            // On either side of every certificate face, and exactly on it.
            for axis in 0..3 {
                for face in [leaf.lower[axis], leaf.upper[axis]] {
                    if !face.is_finite() {
                        continue;
                    }
                    for coordinate in [
                        face - 1e-6,
                        face.next_down(),
                        face,
                        face.next_up(),
                        face + 1e-6,
                    ] {
                        let mut q = tree.tree.points[leaf.start as usize];
                        q[axis] = coordinate;
                        q[2] += 9e-6;
                        assert_eq!(
                            tree.nearest(&q, leaf.start).0.distance_sq,
                            tree.tree.nearest(&q, None).distance_sq
                        );
                        assert_eq!(
                            tree.nearest_bounded(&q, leaf.start).0.distance_sq,
                            tree.tree.nearest(&q, None).distance_sq
                        );
                    }
                }
            }
        }
        // A distant/wrong seed must recover the nearer leaf via full search.
        let q = [368_000.001, 3_955_000.002, 40.000_009];
        let (hit, fast) = tree.nearest(&q, 4095);
        assert!(!fast);
        assert_eq!(hit.distance_sq, brute(&points, &q));

        let duplicates = CandidateKdTree::new(&vec![[6_378_137.000_01, 0.0, 0.0]; 20_000]).unwrap();
        let q = [6_378_137.000_009, 0.0, 0.0];
        let (hit, fast) = duplicates.nearest(&q, 19_999);
        assert!(fast);
        assert_eq!(hit.distance_sq, distance_sq(&duplicates.tree.points[0], &q));
        // Overflow cannot be used as a certificate.
        assert!(!duplicates.nearest(&[1e300; 3], 0).1);
        assert!(!duplicates.nearest_bounded(&[1e300; 3], 0).1);
    }

    #[test]
    fn sibling_boxes_certify_diagonal_distance_and_recover_wrong_leaves() {
        let points: Vec<_> = (0..32).map(|i| [i as f64, (i % 2) as f64, 0.0]).collect();
        let tree = CandidateKdTree::new(&points).unwrap();
        let query = [15.5, 10.0, 0.0];
        let slot = tree.tree.nearest(&query, None).slot;
        assert!(!tree.nearest(&query, slot).1);
        let (hit, certified) = tree.nearest_bounded(&query, slot);
        assert!(certified);
        assert_eq!(hit.distance_sq, brute(&points, &query));
        // Equal box bound is safe, as are its adjacent representable values.
        for x in [query[0].next_down(), query[0], query[0].next_up()] {
            for seed in 0..32 {
                let q = [x, 10.0, 0.0];
                assert_eq!(
                    tree.nearest_bounded(&q, seed).0.distance_sq,
                    brute(&points, &q)
                );
            }
        }
        // Exercise searches spanning multiple ancestor siblings, with gaps,
        // coincident subtrees, negative axes and near-overflow coordinates.
        for scale in [1e-150, 1.0, 1e150] {
            let positions: Vec<_> = (0..257)
                .map(|i| {
                    [
                        ((i % 17) as f64 - 8.0) * scale,
                        ((i / 17) as f64 - 7.0) * scale,
                        (i % 3) as f64 * scale,
                    ]
                })
                .collect();
            let tree = CandidateKdTree::new(&positions).unwrap();
            for q in [
                [-8.3 * scale, -7.7 * scale, -scale],
                [9.1 * scale, 7.8 * scale, 2.4 * scale],
                [0.1 * scale; 3],
            ] {
                let expected = brute(&positions, &q);
                for seed in [0, 17, 128, 256, u32::MAX] {
                    assert_eq!(tree.nearest_bounded(&q, seed).0.distance_sq, expected);
                }
            }
        }
    }
}
